"""Stage 3 - Extract atomic complaint/insight units from normalized reviews.

See architecture.md §4 (Stage 3) and edgecases.md "Stage 3 - Unit Extraction"
for the full edge-case catalog (S3-xx IDs referenced below).

Two filtering decisions, made explicitly (not left as defaults):

1. Reviews with fewer than ``units.min_words`` words (default 4, i.e. drop
   anything <= 3 words - "good", "nice", "super app", emoji-only, etc.) are
   **fully excluded**: no unit is produced, and they are not referenced in
   any downstream artifact or persisted stat (S3-09). This is a pure length
   cutoff, chosen over a generic-phrase-based filter for simplicity; it will
   also drop some genuinely short complaints (e.g. "no delivery") - a known,
   accepted trade-off.
2. Splitting is **rule-based, not LLM-based**, as the primary/only method.
   At real corpus scale (100k+ reviews) a per-review local LLM call is not
   practical in a zero-cost/local setup (edgecases.md S3-01) - the LLM is
   reserved for Stage 7, which only needs one call per *cluster*, not per
   review.

Each ``Unit`` also carries the parent ``Review``'s ``source`` (edgecases.md
S3-10, Docs/context.md §11 Phase 9) so provenance survives into every
downstream stage even after reviews from multiple sources are merged in
Stage 2.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import Counter
from typing import List

from src.config import Config, ensure_data_dir, load_config
from src.schema import Review, SchemaError, Unit, read_jsonl, write_jsonl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"[.!?;\n]+")
_CONJUNCTION_SPLIT_RE = re.compile(
    r"\s+(?:but|however|though|although|except|whereas)\s+", re.IGNORECASE
)
_PUNCT_STRIP_RE = re.compile(r"[^\w\s]")


class UnitsError(RuntimeError):
    """Raised for unrecoverable unit-extraction failures (e.g. zero units produced)."""


def _word_count(text: str) -> int:
    normalized = _PUNCT_STRIP_RE.sub("", text.lower()).strip()
    return len(normalized.split()) if normalized else 0


def _split_into_fragments(text: str, min_words_per_unit: int, max_units: int) -> List[str]:
    """Rule-based atomic-statement splitter (S3-03/S3-04/S3-08 handling).

    Splits on sentence boundaries, then on topic-shift conjunctions within
    each sentence. Fragments shorter than ``min_words_per_unit`` are dropped
    (not kept as trivial units). If nothing substantial survives, falls back
    to the whole review text as a single unit - a review that already passed
    the review-level word-count gate must never end up contributing zero
    units (S3-03). If more fragments remain than ``max_units``, the longest
    (most informative) fragments are kept (S3-08).
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]

    fragments: List[str] = []
    for sentence in sentences:
        parts = [p.strip() for p in _CONJUNCTION_SPLIT_RE.split(sentence) if p.strip()]
        fragments.extend(parts if parts else [sentence])

    seen = set()
    kept: List[str] = []
    for frag in fragments:
        key = frag.lower()
        if key in seen or _word_count(frag) < min_words_per_unit:
            continue
        seen.add(key)
        kept.append(frag)

    if not kept:
        return [text.strip()]

    if len(kept) > max_units:
        kept = sorted(kept, key=_word_count, reverse=True)[:max_units]

    return kept


def extract_units(config: Config, refresh: bool = False) -> None:
    ensure_data_dir(config)

    if not config.paths.reviews.exists():
        raise UnitsError(
            f"Expected normalized corpus at {config.paths.reviews} but it does not exist. "
            "Run `python -m src.normalize` first (edgecases.md X-03)."
        )

    if config.paths.units_raw.exists() and not refresh:
        logger.info(
            "'%s' already exists; skipping unit extraction (pass --refresh to rebuild).",
            config.paths.units_raw,
        )
        return

    if config.units.use_llm:
        logger.warning(
            "units.use_llm=true, but LLM-based per-review splitting is not implemented "
            "(impractical at real corpus scale - edgecases.md S3-01); using rule-based splitting."
        )

    stats: Counter = Counter()
    units: List[Unit] = []

    for review in read_jsonl(config.paths.reviews, factory=Review):
        stats["total_reviews"] += 1

        if _word_count(review.text) < config.units.min_words:
            stats["excluded_low_signal"] += 1  # console-only; not persisted anywhere (S3-09)
            continue

        fragments = _split_into_fragments(
            review.text, config.units.min_words_per_unit, config.units.max_units_per_review
        )
        produced_any = False
        for idx, fragment in enumerate(fragments):
            unit_id = f"{review.id}-{idx:02d}"
            try:
                unit = Unit(
                    unit_id=unit_id,
                    review_id=review.id,
                    text=fragment,
                    rating=review.rating,
                    date=review.date,
                    lang=review.metadata.lang,
                    source=review.source,
                )
            except SchemaError as exc:
                stats["schema_rejected"] += 1
                logger.warning("Rejected unit %s: %s", unit_id, exc)
                continue
            units.append(unit)
            produced_any = True

        if produced_any:
            stats["reviews_with_units"] += 1

    if not units:
        raise UnitsError(
            "Unit extraction produced zero units (edgecases.md X-10). Check units.min_words "
            f"({config.units.min_words}) against the normalized corpus for data quality issues."
        )

    write_jsonl(config.paths.units_raw, units)
    _log_summary(units, stats, config.units.min_words)


def _log_summary(units: List[Unit], stats: Counter, min_words: int) -> None:
    total = stats["total_reviews"]
    excluded = stats["excluded_low_signal"]
    logger.info(
        "Unit extraction complete: %d units from %d contributing reviews (of %d total; "
        "%d excluded as low-signal at <%d words - fully excluded, not counted downstream)",
        len(units), stats["reviews_with_units"], total, excluded, min_words,
    )
    if stats["schema_rejected"]:
        logger.info("  schema_rejected: %d", stats["schema_rejected"])

    rating_counts = Counter(u.rating for u in units)
    for rating in (1, 2, 3, 4, 5):
        logger.info("  rating=%d: %d units", rating, rating_counts.get(rating, 0))

    lang_counts = Counter(u.lang for u in units)
    for lang, count in lang_counts.most_common():
        logger.info("  lang=%s: %d units", lang, count)

    source_counts = Counter(u.source for u in units)
    for source, count in source_counts.most_common():
        logger.info("  source=%s: %d units", source, count)

    per_review: Counter = Counter()
    for u in units:
        per_review[u.review_id] += 1
    avg_units = len(units) / max(1, len(per_review))
    logger.info("  avg units per contributing review: %.2f", avg_units)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract atomic complaint/insight units from normalized reviews (Stage 3)."
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: project root)")
    parser.add_argument(
        "--refresh", action="store_true", help="Rebuild units.jsonl even if it already exists"
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config) if args.config else load_config()
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    try:
        extract_units(config, refresh=args.refresh)
    except UnitsError as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
