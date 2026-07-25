"""Stage 6 - Louvain community detection over the unit similarity graph.

See architecture.md §4 (Stage 6) and edgecases.md "Stage 6 - Community
Detection" for the full edge-case catalog (S6-xx IDs referenced below).

Design choices, made explicitly:

1. **No dedicated `Community` dataclass in schema.py.** Unlike `Theme`
   (Stage 7's LLM-labeled output), a community at this stage is just a
   grouping of unit ids with no label/sentiment/description yet - it would
   be a strict subset of `Theme`'s fields with nothing to validate beyond
   "non-empty list of known unit ids". `communities.json` is written as a
   plain JSON dict (same precedent as `embed.py`'s `unit_index.json`).
2. **`community_id` is re-indexed by descending size** (largest = 0), not
   whatever arbitrary integer `python-louvain` assigned. This is a pure
   presentation choice (stable, human-friendly ordering for Stage 7/UI) and
   does not change cluster membership; ties are broken by the community's
   smallest unit id for full determinism.
3. **Micro-communities are flagged, not deleted (S6-03 revision).** Every
   community `python-louvain` finds - including ones below
   `clustering.min_community_size` and true singletons from Stage 5's
   isolated nodes (S6-05) - is kept in `communities.json` with a
   `below_min_size` flag. Physically dropping them here would be
   irreversible data loss that Stage 9 (validation) might need; Stage 7
   (`summarize.py`) is the one that decides whether to skip/merge
   below-threshold communities when generating LLM-labeled themes.
4. **Reproducibility (S6-01):** `python-louvain`'s `best_partition` accepts
   `random_state`, seeded from `config.seed`. Louvain is still not
   perfectly deterministic across library/OS versions in general (its
   internal tie-breaking during modularity optimization can depend on
   dict/set iteration order), so minor partition variance across
   environments is expected and accepted, not eliminated - consistent with
   edgecases.md S4-08's treatment of the same class of issue for embeddings.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from collections import Counter
from statistics import mean, median
from typing import Dict, List

from src.config import Config, apply_global_seed, ensure_data_dir, load_config
from src.schema import Unit, read_jsonl, write_json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# S6-02: if the largest community holds more than this fraction of all nodes,
# warn that `clustering.louvain_resolution` is likely too low for useful
# granularity - purely advisory, does not change behavior.
_GIANT_COMMUNITY_WARN_FRACTION = 0.5


class ClusterError(RuntimeError):
    """Raised for unrecoverable clustering failures (e.g. missing/empty graph)."""


def _load_graph(config: Config):
    if not config.paths.graph.exists():
        raise ClusterError(
            f"Expected similarity graph at {config.paths.graph} but it does not exist. "
            "Run `python -m src.graph` first (edgecases.md X-03)."
        )
    with open(config.paths.graph, "rb") as f:
        graph = pickle.load(f)

    if graph.number_of_nodes() == 0:
        raise ClusterError("graph.gpickle has zero nodes - nothing to cluster (X-10).")
    if graph.number_of_edges() == 0:
        # Defensive re-check: Stage 5 already guards against this at construction
        # time, but the graph file could have been edited/regenerated independently.
        raise ClusterError(
            "graph.gpickle has zero edges (edgecases.md S6-06). Re-run `python -m src.graph "
            "--refresh` with a lower graph.similarity_threshold."
        )
    return graph


def detect_communities(config: Config, refresh: bool = False) -> None:
    ensure_data_dir(config)
    apply_global_seed(config.seed)

    if config.paths.communities.exists() and not refresh:
        logger.info(
            "'%s' already exists; skipping community detection (pass --refresh to rebuild).",
            config.paths.communities,
        )
        return

    graph = _load_graph(config)

    import networkx as nx

    try:
        import community as community_louvain  # python-louvain; imports as `community`
    except ImportError as exc:
        raise ClusterError(
            "python-louvain is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    num_components = nx.number_connected_components(graph)  # S6-04: informational only

    partition: Dict[str, int] = community_louvain.best_partition(
        graph,
        weight="weight",
        resolution=config.clustering.louvain_resolution,
        random_state=config.seed,  # S6-01
    )

    raw_groups: Dict[int, List[str]] = {}
    for unit_id, raw_cid in partition.items():
        raw_groups.setdefault(raw_cid, []).append(unit_id)

    # Re-index by descending size, ties broken by smallest unit id (deterministic - see
    # module docstring point 2). This is presentation only; membership is untouched.
    ordered = sorted(raw_groups.values(), key=lambda ids: (-len(ids), min(ids)))

    unit_meta: Dict[str, Unit] = {
        u.unit_id: u for u in read_jsonl(config.paths.units, factory=Unit)
    }

    min_size = config.clustering.min_community_size
    communities_payload = []
    sizes: List[int] = []
    num_below_min = 0
    num_singletons = 0

    for community_id, unit_ids in enumerate(ordered):
        size = len(unit_ids)
        sizes.append(size)
        below_min_size = size < min_size
        if below_min_size:
            num_below_min += 1
        if size == 1:
            num_singletons += 1

        ratings = [
            unit_meta[uid].rating
            for uid in unit_ids
            if uid in unit_meta and unit_meta[uid].rating is not None
        ]
        lang_counts = Counter(unit_meta[uid].lang for uid in unit_ids if uid in unit_meta)

        communities_payload.append(
            {
                "community_id": community_id,
                "size": size,
                "unit_ids": sorted(unit_ids),
                "avg_rating": round(mean(ratings), 3) if ratings else None,
                "lang_counts": dict(lang_counts),
                "below_min_size": below_min_size,
            }
        )

    largest_fraction = sizes[0] / graph.number_of_nodes() if sizes else 0.0
    if largest_fraction > _GIANT_COMMUNITY_WARN_FRACTION:
        logger.warning(
            "Largest community holds %.1f%% of all units (S6-02) - "
            "clustering.louvain_resolution=%.2f may be too low for useful granularity; "
            "consider increasing it and re-running with --refresh.",
            100.0 * largest_fraction,
            config.clustering.louvain_resolution,
        )

    payload = {
        "algorithm": "louvain",
        "resolution": config.clustering.louvain_resolution,
        "min_community_size": min_size,
        "seed": config.seed,
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "num_connected_components": num_components,
        "num_communities": len(communities_payload),
        "num_communities_below_min_size": num_below_min,
        "num_singleton_communities": num_singletons,
        "communities": communities_payload,
    }
    write_json(config.paths.communities, payload)
    _log_summary(payload, sizes)


def _log_summary(payload: dict, sizes: List[int]) -> None:
    logger.info(
        "Community detection complete: %d communities over %d nodes / %d edges "
        "(%d connected components, algorithm=louvain, resolution=%.2f).",
        payload["num_communities"],
        payload["num_nodes"],
        payload["num_edges"],
        payload["num_connected_components"],
        payload["resolution"],
    )
    logger.info(
        "  size: min=%d, max=%d, mean=%.2f, median=%.1f",
        min(sizes), max(sizes), mean(sizes), median(sizes),
    )
    logger.info(
        "  below min_community_size=%d: %d communities (flagged, not dropped - S6-03)",
        payload["min_community_size"], payload["num_communities_below_min_size"],
    )
    logger.info(
        "  singleton communities (size 1): %d (includes but not limited to S5-04 isolated nodes)",
        payload["num_singleton_communities"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Louvain community detection over the unit similarity graph (Stage 6)."
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: project root)")
    parser.add_argument(
        "--refresh", action="store_true", help="Rebuild communities.json even if it already exists"
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config) if args.config else load_config()
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    try:
        detect_communities(config, refresh=args.refresh)
    except ClusterError as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
