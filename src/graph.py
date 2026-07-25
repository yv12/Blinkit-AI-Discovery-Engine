"""Stage 5 - Build a kNN similarity graph over unit embeddings.

See architecture.md §4 (Stage 5) and edgecases.md "Stage 5 - Similarity Graph"
for the full edge-case catalog (S5-xx IDs referenced below).

Design choices, made explicitly:

1. **Backend:** tries `faiss` first (not installed on this machine/pinned in
   requirements.txt - see requirements.txt's note on Windows wheel
   availability), falling back to `sklearn.neighbors.NearestNeighbors` with
   cosine metric otherwise (S5-05). Either way the computation is exact
   (brute-force / flat index), not approximate - at this corpus size
   (~90k units, 384-dim) an exact search is affordable and avoids ANN
   recall-loss as an extra variable in an already-approximate pipeline.
2. **Graph construction is "any-kNN" (union), not "mutual-kNN":** an edge
   between A and B is added if A is among B's top-k neighbors *or* B is
   among A's top-k neighbors (above the similarity threshold). This is the
   standard construction for kNN similarity graphs feeding community
   detection - requiring *mutual* top-k membership would silently prune
   asymmetric-density regions of the embedding space (S5-03 - top-k already
   bounds degree from each node's own perspective, so no separate hairball
   guard is needed beyond the threshold).
3. **Empty edge set (S5-02) is a hard error, not an auto-relax.** Silently
   loosening `graph.similarity_threshold` behind the user's back would
   contradict the "fail loudly on bad config" philosophy used everywhere
   else in this pipeline (edgecases.md X-08); instead we raise `GraphError`
   with the actual similarity distribution observed, so the user can pick an
   informed threshold.
4. **Isolated nodes (S5-04) are kept as singletons**, not dropped - Stage 6
   assigns them their own micro-community rather than losing the unit.
5. **Duplicate/near-duplicate embeddings (S5-06):** cosine similarity from
   two independently-normalized float32 vectors can drift fractionally above
   1.0 (see embed.py's verified norms of ~1.0000001); edge weights are
   clamped to `[0, 1]` before being stored.
6. **Serialization:** `networkx` >= 3.0 removed `write_gpickle`/
   `read_gpickle`; a plain `pickle` of the `Graph` object is used instead
   (this is networkx's own documented replacement), atomically written
   (temp file + `os.replace`, edgecases.md X-05) to `data/graph.gpickle`.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.config import Config, apply_global_seed, ensure_data_dir, load_config
from src.schema import Unit, read_jsonl, read_json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class GraphError(RuntimeError):
    """Raised for unrecoverable graph-construction failures."""


def _atomic_pickle_dump(path: Path, obj) -> None:
    """Write a pickle file atomically: temp file + os.replace (edgecases.md X-05)."""
    tmp_path = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _load_embeddings(config: Config) -> Tuple[np.ndarray, List[str]]:
    if not config.paths.embeddings.exists() or not config.paths.unit_index.exists():
        raise GraphError(
            f"Expected '{config.paths.embeddings}' and '{config.paths.unit_index}' but at "
            "least one is missing. Run `python -m src.embed` first (edgecases.md X-03)."
        )

    embeddings = np.load(config.paths.embeddings)
    index = read_json(config.paths.unit_index)
    unit_ids = index.get("unit_ids", [])

    if embeddings.shape[0] != len(unit_ids):
        # Defensive re-check of the S4-03 alignment contract - files could have
        # been regenerated independently/out of sync since Stage 4 ran.
        raise GraphError(
            f"Alignment mismatch: embeddings.npy has {embeddings.shape[0]} rows but "
            f"unit_index.json lists {len(unit_ids)} unit_ids. Re-run `python -m src.embed "
            "--refresh`."
        )
    if embeddings.shape[0] == 0:
        raise GraphError("embeddings.npy is empty - nothing to build a graph from (X-10).")

    return embeddings, unit_ids


def _knn_faiss(embeddings: np.ndarray, k: int):
    import faiss  # type: ignore

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product == cosine on L2-normalized vectors
    index.add(embeddings)
    similarities, indices = index.search(embeddings, k + 1)  # +1: includes self
    return similarities, indices


def _knn_sklearn(embeddings: np.ndarray, k: int):
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    similarities = 1.0 - distances  # cosine distance -> cosine similarity
    return similarities, indices


def _build_knn(embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, str]:
    n = embeddings.shape[0]
    effective_k = min(k, n - 1)
    if effective_k < k:
        logger.warning(
            "graph.knn_k=%d >= corpus size (%d); clamping to %d (edgecases.md S5-01).",
            k, n, effective_k,
        )
    if effective_k < 1:
        raise GraphError(f"Cannot build a kNN graph with only {n} unit(s).")

    try:
        similarities, indices = _knn_faiss(embeddings, effective_k)
        backend = "faiss"
    except ImportError:
        logger.info("faiss not installed; falling back to scikit-learn NearestNeighbors (S5-05).")
        similarities, indices = _knn_sklearn(embeddings, effective_k)
        backend = "sklearn"

    return similarities, indices, backend


def build_graph(config: Config, refresh: bool = False) -> None:
    ensure_data_dir(config)
    apply_global_seed(config.seed)  # no randomness in kNN itself, kept for convention

    if config.paths.graph.exists() and not refresh:
        logger.info(
            "'%s' already exists; skipping graph construction (pass --refresh to rebuild).",
            config.paths.graph,
        )
        return

    embeddings, unit_ids = _load_embeddings(config)
    n = embeddings.shape[0]

    unit_meta: Dict[str, Unit] = {
        u.unit_id: u for u in read_jsonl(config.paths.units, factory=Unit)
    }

    start = time.time()
    similarities, indices, backend = _build_knn(embeddings, config.graph.knn_k)
    logger.info(
        "kNN search complete via %s backend: %d nodes, k=%d, %.1fs.",
        backend, n, min(config.graph.knn_k, n - 1), time.time() - start,
    )

    import networkx as nx

    graph = nx.Graph()
    for i, unit_id in enumerate(unit_ids):
        meta = unit_meta.get(unit_id)
        graph.add_node(
            unit_id,
            review_id=meta.review_id if meta else None,
            rating=meta.rating if meta else None,
            lang=meta.lang if meta else "en",
        )

    threshold = config.graph.similarity_threshold
    edges_added = 0
    all_sims: List[float] = []  # for S5-02 diagnostics if the threshold turns out empty
    for i in range(n):
        src = unit_ids[i]
        for sim, j in zip(similarities[i], indices[i]):
            if j == i or j < 0:  # exclude self; faiss pads with -1 if fewer than k+1 found
                continue
            sim = float(min(max(sim, -1.0), 1.0))  # S5-06: clamp float drift into [-1, 1]
            all_sims.append(sim)
            if sim < threshold:
                continue
            dst = unit_ids[j]
            if graph.has_edge(src, dst):
                continue
            graph.add_edge(src, dst, weight=sim)
            edges_added += 1

    isolated = [node for node in graph.nodes if graph.degree(node) == 0]

    if edges_added == 0:
        arr = np.array(all_sims) if all_sims else np.array([0.0])
        raise GraphError(
            f"graph.similarity_threshold={threshold} excludes every candidate edge "
            f"(edgecases.md S5-02). Observed neighbor-similarity distribution: "
            f"min={arr.min():.3f}, p50={np.median(arr):.3f}, p90={np.percentile(arr, 90):.3f}, "
            f"max={arr.max():.3f}. Lower 'graph.similarity_threshold' in config.yaml and re-run "
            "with --refresh."
        )

    _atomic_pickle_dump(config.paths.graph, graph)
    _log_summary(graph, edges_added, isolated, backend, threshold)


def _log_summary(graph, edges_added: int, isolated: List[str], backend: str, threshold: float) -> None:
    degrees = [d for _, d in graph.degree()]
    logger.info(
        "Graph complete: %d nodes, %d edges (backend=%s, similarity_threshold=%.2f).",
        graph.number_of_nodes(), edges_added, backend, threshold,
    )
    logger.info(
        "  degree: min=%d, max=%d, mean=%.2f, median=%.1f",
        min(degrees), max(degrees), sum(degrees) / len(degrees), float(np.median(degrees)),
    )
    logger.info(
        "  isolated nodes (degree 0): %d (%.1f%% of corpus) - kept as singletons (S5-04)",
        len(isolated), 100.0 * len(isolated) / graph.number_of_nodes(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a kNN similarity graph over unit embeddings (Stage 5)."
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: project root)")
    parser.add_argument(
        "--refresh", action="store_true", help="Rebuild graph.gpickle even if it already exists"
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config) if args.config else load_config()
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    try:
        build_graph(config, refresh=args.refresh)
    except GraphError as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
