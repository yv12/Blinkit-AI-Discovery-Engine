"""Stage 4b - Filter units based on semantic relevance to research questions.

Reads `units_raw.jsonl` and `embeddings_raw.npy`, computes cosine similarity against
the research questions, and drops units that fall below `min_relevance_score` for all
questions. Outputs `units.jsonl` and `embeddings.npy` containing only the relevant subset.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import List

import numpy as np

from src.config import Config, apply_global_seed, ensure_data_dir, load_config
from src.schema import Unit, read_json, read_jsonl, write_json, write_jsonl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class FilterError(RuntimeError):
    """Raised for unrecoverable filtering failures."""


def _load_model(model_name: str):
    try:
        import torch  # noqa: F401
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise FilterError("sentence-transformers is not installed.") from exc
    return SentenceTransformer(model_name)


def filter_units(config: Config, refresh: bool = False) -> None:
    ensure_data_dir(config)
    apply_global_seed(config.seed)

    if not config.paths.units_raw.exists() or not config.paths.embeddings_raw.exists():
        raise FilterError(
            f"Expected raw artifacts at {config.paths.units_raw} and {config.paths.embeddings_raw}. "
            "Run `python -m src.embed` first."
        )

    if config.paths.units.exists() and config.paths.embeddings.exists() and not refresh:
        logger.info(
            "Filtered artifacts already exist; skipping filtering (pass --refresh to rebuild)."
        )
        return

    logger.info("Loading raw units and embeddings for filtering...")
    units_raw: List[Unit] = list(read_jsonl(config.paths.units_raw, factory=Unit))
    embeddings_raw = np.load(config.paths.embeddings_raw)
    unit_index_raw = read_json(config.paths.unit_index_raw)

    if len(units_raw) != embeddings_raw.shape[0]:
        raise FilterError("Mismatch between raw units and embeddings count.")

    logger.info("Loading embedding model '%s' for queries...", config.models.embedding_model)
    model = _load_model(config.models.embedding_model)

    queries = config.insights.question_queries
    logger.info("Embedding %d research questions...", len(queries))
    query_embs = model.encode(
        queries,
        batch_size=len(queries),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    start = time.time()
    logger.info("Calculating similarity and filtering against threshold %.2f...", config.filtering.min_relevance_score)
    
    # embeddings_raw is already L2-normalized, query_embs is L2-normalized.
    # dot product gives cosine similarity.
    # Shape: (num_units, dim) x (dim, num_queries) -> (num_units, num_queries)
    similarities = embeddings_raw @ query_embs.T
    
    # Max similarity across all questions for each unit
    max_sim = np.max(similarities, axis=1)
    
    # Boolean mask of units to keep
    keep_mask = max_sim >= config.filtering.min_relevance_score
    
    filtered_units = [u for i, u in enumerate(units_raw) if keep_mask[i]]
    filtered_embeddings = embeddings_raw[keep_mask]
    
    elapsed = time.time() - start

    if not filtered_units:
        raise FilterError(
            "Filtering dropped ALL units. Check min_relevance_score in config.yaml."
        )

    logger.info(
        "Filtering complete in %.1fs: kept %d / %d units (%.1f%%).",
        elapsed,
        len(filtered_units),
        len(units_raw),
        100.0 * len(filtered_units) / max(1, len(units_raw)),
    )

    # Save filtered artifacts
    write_jsonl(config.paths.units, filtered_units)
    
    from src.embed import _atomic_save_npy
    _atomic_save_npy(config.paths.embeddings, filtered_embeddings)
    
    index_payload = {
        "model": config.models.embedding_model,
        "dim": int(filtered_embeddings.shape[1]),
        "count": int(filtered_embeddings.shape[0]),
        "metric": "cosine",
        "seed": config.seed,
        "unit_ids": [u.unit_id for u in filtered_units],
    }
    write_json(config.paths.unit_index, index_payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter units based on semantic relevance to research questions."
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: project root)")
    parser.add_argument("--refresh", action="store_true", help="Rebuild filtered artifacts")
    args = parser.parse_args()

    try:
        config = load_config(args.config) if args.config else load_config()
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    try:
        filter_units(config, refresh=args.refresh)
    except FilterError as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
