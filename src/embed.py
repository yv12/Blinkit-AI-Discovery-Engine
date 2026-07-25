"""Stage 4 - Encode atomic units into a local sentence-transformer embedding space.

See architecture.md §4 (Stage 4) and edgecases.md "Stage 4 - Embed" for the
full edge-case catalog (S4-xx IDs referenced below).

Design choices, made explicitly:

1. Model loading failures (no cached weights + offline) raise a clear,
   actionable error rather than a raw stack trace (S4-01). Once downloaded,
   Hugging Face caches the weights locally, so subsequent runs are offline.
2. Vectors are L2-normalized at encode time (``normalize_embeddings=True``),
   so a plain dot product between rows is equivalent to cosine similarity
   downstream in Stage 5 (S4-04). This is recorded in ``unit_index.json``
   as ``"metric": "cosine"`` so Stage 5 doesn't have to guess.
3. Row/unit alignment is enforced by construction: embeddings are produced
   in the same order as ``unit_ids``, the row count is asserted against the
   unit count before saving, and ``unit_index.json`` persists the exact
   ``unit_ids`` list so Stage 5 can map row -> unit id without relying on
   file-read order alone (S4-03).
4. Long unit text is truncated by the tokenizer (sentence-transformers'
   default behavior) rather than chunked + mean-pooled. Unit texts are short
   review fragments (S3's splitter already caps fragment count/length), so
   this is expected to be rare; a word-count-based heuristic estimates how
   many units are likely affected and logs a warning rather than failing
   silently (S4-02).
5. Both ``embeddings.npy`` and ``unit_index.json`` are written atomically
   (temp file + ``os.replace``), matching the crash-resilience pattern used
   for every other pipeline artifact (edgecases.md X-05).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np

from src.config import Config, apply_global_seed, ensure_data_dir, load_config
from src.schema import Unit, read_jsonl, write_json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Rough tokens-per-word ratio used only to *estimate* how many units are likely
# to be truncated by the model's max sequence length (S4-02). This is a cheap
# heuristic, not an exact tokenizer count - it exists purely to surface a
# warning, not to change behavior.
_APPROX_TOKENS_PER_WORD = 1.3


class EmbedError(RuntimeError):
    """Raised for unrecoverable embedding failures (e.g. missing model, zero units)."""


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    """Write a .npy file atomically: write to a temp file, then ``os.replace`` (X-05)."""
    tmp_path = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        with open(tmp_path, "wb") as f:
            np.save(f, array)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _load_model(model_name: str):
    try:
        # Import torch directly *before* sentence-transformers/transformers pull it in
        # transitively. On Windows, letting transformers import torch as a side effect of
        # its own import chain can trip a native-DLL init conflict ("OSError: [WinError
        # 1114] A dynamic link library (DLL) initialization routine failed" on c10.dll)
        # even when a plain top-level `import torch` works fine on its own. Importing it
        # first, standalone, sidesteps the conflict (edgecases.md S4-01).
        import torch  # noqa: F401
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise EmbedError(
            "sentence-transformers is not installed. Run `pip install -r requirements.txt` "
            "(edgecases.md S4-01)."
        ) from exc

    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise EmbedError(
            f"Failed to load embedding model '{model_name}' (edgecases.md S4-01). "
            "If this is the first run, the model weights must be downloaded from Hugging "
            "Face - check your internet connection. Once downloaded, they are cached "
            "locally (~/.cache/huggingface) and later runs work fully offline."
        ) from exc


def embed_units(config: Config, refresh: bool = False) -> None:
    ensure_data_dir(config)
    apply_global_seed(config.seed)  # S4-08: same determinism convention as every other stage

    if not config.paths.units_raw.exists():
        raise EmbedError(
            f"Expected units corpus at {config.paths.units_raw} but it does not exist. "
            "Run `python -m src.units` first (edgecases.md X-03)."
        )

    if config.paths.embeddings_raw.exists() and config.paths.unit_index_raw.exists() and not refresh:
        logger.info(
            "'%s' and '%s' already exist; skipping embedding (pass --refresh to rebuild).",
            config.paths.embeddings_raw,
            config.paths.unit_index_raw,
        )
        return

    units: List[Unit] = list(read_jsonl(config.paths.units_raw, factory=Unit))
    if not units:
        raise EmbedError(
            "units.jsonl is empty - nothing to embed (edgecases.md S4-06/X-10). "
            "Run `python -m src.units` first."
        )

    model = _load_model(config.models.embedding_model)
    max_seq_length = getattr(model, "max_seq_length", None)

    lang_counts = Counter(u.lang for u in units)
    logger.info(
        "Embedding %d units with '%s' (per-lang breakdown, edgecases.md S4-07):",
        len(units),
        config.models.embedding_model,
    )
    for lang, count in lang_counts.most_common():
        logger.info("  lang=%s: %d units", lang, count)

    if max_seq_length:
        approx_long = sum(
            1
            for u in units
            if len(u.text.split()) * _APPROX_TOKENS_PER_WORD > max_seq_length
        )
        if approx_long:
            logger.warning(
                "~%d/%d units may exceed the model's max_seq_length=%d tokens (word-count "
                "estimate) and will be silently truncated by the tokenizer, not chunked + "
                "mean-pooled (edgecases.md S4-02 - truncation chosen for simplicity since "
                "unit texts are short review fragments and this is expected to be rare).",
                approx_long,
                len(units),
                max_seq_length,
            )

    texts = [u.text for u in units]
    unit_ids = [u.unit_id for u in units]

    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=config.models.embed_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # S4-04: L2-normalize so dot product == cosine similarity
    ).astype(np.float32)
    elapsed = time.time() - start

    if embeddings.shape[0] != len(units):
        # S4-03: row/unit alignment must never silently drift.
        raise EmbedError(
            f"Row count mismatch: got {embeddings.shape[0]} embedding rows for "
            f"{len(units)} units. Refusing to save a misaligned artifact."
        )

    _atomic_save_npy(config.paths.embeddings_raw, embeddings)

    index_payload = {
        "model": config.models.embedding_model,
        "dim": int(embeddings.shape[1]),
        "count": int(embeddings.shape[0]),
        "metric": "cosine",  # vectors are L2-normalized; a plain dot product == cosine similarity
        "seed": config.seed,
        # unit_ids[i] is the unit id for embeddings row i (S4-03 alignment contract).
        "unit_ids": unit_ids,
    }
    write_json(config.paths.unit_index_raw, index_payload)

    logger.info(
        "Embedding complete: %d units -> %s (dim=%d) in %.1fs (%.0f units/sec).",
        len(units),
        config.paths.embeddings_raw.name,
        embeddings.shape[1],
        elapsed,
        len(units) / max(elapsed, 1e-9),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode units into a local sentence-transformer embedding space (Stage 4)."
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: project root)")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Rebuild embeddings.npy/unit_index.json even if they already exist",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config) if args.config else load_config()
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    try:
        embed_units(config, refresh=args.refresh)
    except EmbedError as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
