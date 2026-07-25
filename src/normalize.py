"""Stage 2 - Normalize raw scraped reviews into the canonical Review schema.

See architecture.md §4/§5 (Stage 2) and edgecases.md "Stage 2 - Normalize"
for the full edge-case catalog (S2-xx IDs referenced below).

Also normalizes and merges in the optional second source (Docs/context.md §11
Phase 9, edgecases.md S2-09): if Stage 1b (`src/scrape_mouthshut.py`) has
produced `raw_mouthshut.jsonl`, its rows are normalized via a dedicated
Mouthshut-specific path and appended to the same `reviews.jsonl` output,
tagged `source: "mouthshut"`. If that file doesn't exist, this stage behaves
exactly as it did before that addendum - Google Play only.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import logging
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from src.config import Config, ensure_data_dir, load_config
from src.schema import Review, ReviewMetadata, SchemaError, read_jsonl, write_jsonl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Strip non-printable control chars but keep newlines/tabs (S2-06). Preserves
# emoji and Devanagari, which live outside this range.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


class NormalizeError(RuntimeError):
    """Raised for unrecoverable normalization failures (e.g. empty corpus)."""


def _clean_text(raw_text: str) -> str:
    text = html.unescape(raw_text)
    text = _CONTROL_CHARS_RE.sub("", text)
    return text.replace("\r\n", "\n").strip()


def _detect_lang(text: str) -> str:
    """Lightweight script-based heuristic (S1-11/S2 lang tagging).

    Only distinguishes Devanagari-script Hindi from everything else; Hinglish
    written in Latin script cannot be reliably separated from English without
    a language-ID model, which is out of scope for this pass. Good enough to
    flag non-Latin-script reviews for later multilingual embedding handling
    (edgecases.md S4-07).
    """
    return "hi" if _DEVANAGARI_RE.search(text) else "en"


def _normalize_date(raw_value: Any) -> Tuple[str, bool]:
    """Parse to UTC ISO 8601 (S2-08). Returns (iso_string_or_empty, was_unparseable).

    An unparseable date does not drop the review (S2-01); the review is kept
    with an empty date string, and the caller counts the occurrence.
    """
    if not raw_value or not isinstance(raw_value, str):
        return "", True
    try:
        dt = datetime.fromisoformat(raw_value)
    except ValueError:
        return "", True
    # google-play-scraper returns naive datetimes; treated as UTC (S2-08).
    dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z"), False


def _normalize_rating(raw_value: Any) -> Optional[int]:
    """Coerce to int 1-5, else None (S2-02); never drop the review over this."""
    try:
        rating = int(raw_value)
    except (TypeError, ValueError):
        return None
    return rating if 1 <= rating <= 5 else None


def _normalize_int(raw_value: Any, default: int = 0) -> int:
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default


# Mouthshut relative-date phrasing observed in the real export (Docs/context.md
# Addendum): "N days ago" for anything within the last ~30 days, plus a couple of
# generic phrasings kept as a safety net even though not observed in the sample.
_MOUTHSHUT_RELATIVE_RE = re.compile(
    r"^(?:(a|an|\d+)\s+)?(hour|day|week|month|year)s?\s+ago$", re.IGNORECASE
)
_MOUTHSHUT_UNIT_DAYS = {"hour": 0, "day": 1, "week": 7, "month": 30, "year": 365}


def _normalize_mouthshut_date(raw_value: Any, reference_date: datetime) -> Tuple[str, bool]:
    """Parse Mouthshut's two observed date shapes (S2-01 convention: unparseable
    keeps the review, with an empty date string, rather than dropping it).

    1. Absolute: ``"Jun 20, 2026 05:15 PM"``.
    2. Relative: ``"N days ago"`` / ``"Today"`` / ``"Yesterday"`` - resolved against
       ``reference_date`` (the CSV's ingestion-time anchor, scrape_mouthshut.py).
    """
    if not raw_value or not isinstance(raw_value, str):
        return "", True
    text = raw_value.strip()
    if not text:
        return "", True

    if text.lower() == "today":
        dt = reference_date
    elif text.lower() == "yesterday":
        dt = reference_date - timedelta(days=1)
    else:
        m = _MOUTHSHUT_RELATIVE_RE.match(text)
        if m:
            count_str, unit = m.group(1), m.group(2).lower()
            count = 1 if count_str in (None, "a", "an") else int(count_str)
            dt = reference_date - timedelta(days=count * _MOUTHSHUT_UNIT_DAYS[unit])
        else:
            try:
                dt = datetime.strptime(text, "%b %d, %Y %I:%M %p")
            except ValueError:
                return "", True

    dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z"), False


def _normalize_one_mouthshut(raw: Dict[str, Any], stats: Counter, dup_hashes: Counter) -> Optional[Review]:
    review_id = raw.get("reviewId")
    if not review_id:
        stats["mouthshut_missing_id"] += 1
        return None

    title = (raw.get("title") or "").strip()
    body = (raw.get("content") or "").strip()
    # Mouthshut splits a short title from the review body; concatenated so unit
    # extraction (Stage 3) sees the whole complaint as one text, same as Play
    # Store's single `content` field.
    combined = f"{title}. {body}" if title and body else (title or body)
    cleaned_text = _clean_text(combined)
    if not cleaned_text:
        stats["mouthshut_dropped_empty_text"] += 1
        return None

    reference_date = datetime.fromisoformat(raw["reference_date"])
    iso_date, unparseable = _normalize_mouthshut_date(raw.get("review_date_raw"), reference_date)
    if unparseable:
        stats["mouthshut_unparseable_date"] += 1

    rating = _normalize_rating(raw.get("rating_raw"))
    if rating is None and raw.get("rating_raw") is not None:
        stats["mouthshut_invalid_rating"] += 1

    dup_hashes[hashlib.sha1(cleaned_text.lower().encode("utf-8")).hexdigest()] += 1

    metadata = ReviewMetadata(
        thumbs_up=0,  # not exposed by this export
        app_version=None,  # not applicable to a review-forum source
        developer_reply=None,  # not exposed by this export
        lang=_detect_lang(cleaned_text),
    )

    try:
        return Review(
            id=review_id,
            text=cleaned_text,
            date=iso_date,
            rating=rating,
            source="mouthshut",
            url=raw.get("url"),
            metadata=metadata,
        )
    except SchemaError as exc:
        stats["mouthshut_schema_rejected"] += 1
        logger.warning("Rejected Mouthshut review %s during normalization: %s", review_id, exc)
        return None


def _normalize_one(raw: Dict[str, Any], stats: Counter, dup_hashes: Counter) -> Optional[Review]:
    review_id = raw.get("reviewId")
    if not review_id:
        stats["missing_id"] += 1
        return None

    cleaned_text = _clean_text(raw.get("content") or "")
    if not cleaned_text:
        stats["dropped_empty_text"] += 1  # S2-03
        return None

    iso_date, unparseable = _normalize_date(raw.get("at"))
    if unparseable:
        stats["unparseable_date"] += 1  # S2-01

    rating = _normalize_rating(raw.get("score"))
    if rating is None and raw.get("score") is not None:
        stats["invalid_rating"] += 1  # S2-02

    dup_hashes[hashlib.sha1(cleaned_text.lower().encode("utf-8")).hexdigest()] += 1  # S2-05

    metadata = ReviewMetadata(
        thumbs_up=_normalize_int(raw.get("thumbsUpCount"), default=0),
        app_version=raw.get("appVersion") or raw.get("reviewCreatedVersion"),
        developer_reply=raw.get("replyContent"),  # S2-07
        lang=_detect_lang(cleaned_text),
    )

    try:
        return Review(
            id=review_id,
            text=cleaned_text,
            date=iso_date,
            rating=rating,
            source="google_play",
            url=None,
            metadata=metadata,
        )
    except SchemaError as exc:
        stats["schema_rejected"] += 1
        logger.warning("Rejected review %s during normalization: %s", review_id, exc)
        return None


def normalize_all(config: Config, refresh: bool = False) -> None:
    ensure_data_dir(config)

    if not config.paths.raw_reviews.exists():
        raise NormalizeError(
            f"Expected raw corpus at {config.paths.raw_reviews} but it does not exist. "
            "Run `python -m src.scrape` first (edgecases.md X-03)."
        )

    if config.paths.reviews.exists() and not refresh:
        logger.info(
            "'%s' already exists; skipping normalize (pass --refresh to rebuild).",
            config.paths.reviews,
        )
        return

    stats: Counter = Counter()
    dup_hashes: Counter = Counter()
    reviews = []

    for raw in read_jsonl(config.paths.raw_reviews, factory=None):
        stats["total_raw"] += 1
        review = _normalize_one(raw, stats, dup_hashes)
        if review is not None:
            reviews.append(review)

    # Second source (optional, Docs/context.md Addendum "Second data source -
    # Mouthshut"): only present if `python -m src.scrape_mouthshut` has already
    # run and found a CSV to ingest. Merged into the same reviews.jsonl, tagged
    # `source: "mouthshut"`, so every downstream stage sees one unified corpus.
    if config.paths.raw_mouthshut.exists():
        for raw in read_jsonl(config.paths.raw_mouthshut, factory=None):
            stats["total_raw_mouthshut"] += 1
            review = _normalize_one_mouthshut(raw, stats, dup_hashes)
            if review is not None:
                reviews.append(review)

    if not reviews:
        raise NormalizeError(
            "Normalization produced zero valid reviews (edgecases.md X-10). Inspect "
            f"{config.paths.raw_reviews} for data quality issues."
        )

    write_jsonl(config.paths.reviews, reviews)
    _log_summary(reviews, stats, dup_hashes)


def _log_summary(reviews: list, stats: Counter, dup_hashes: Counter) -> None:
    logger.info(
        "Normalize complete: %d reviews written (from %d Google Play raw + %d Mouthshut raw)",
        len(reviews), stats["total_raw"], stats["total_raw_mouthshut"],
    )
    for key in ("missing_id", "dropped_empty_text", "unparseable_date", "invalid_rating", "schema_rejected"):
        if stats[key]:
            logger.info("  google_play.%s: %d", key, stats[key])
    for key in (
        "mouthshut_missing_id", "mouthshut_dropped_empty_text", "mouthshut_unparseable_date",
        "mouthshut_invalid_rating", "mouthshut_schema_rejected",
    ):
        if stats[key]:
            logger.info("  %s: %d", key, stats[key])

    source_counts = Counter(r.source for r in reviews)
    for source, count in source_counts.most_common():
        logger.info("  source=%s: %d reviews", source, count)

    rating_counts = Counter(r.rating for r in reviews)
    for rating in (1, 2, 3, 4, 5):
        logger.info("  rating=%d: %d reviews", rating, rating_counts.get(rating, 0))
    logger.info("  rating=None (invalid/missing): %d reviews", rating_counts.get(None, 0))

    lang_counts = Counter(r.metadata.lang for r in reviews)
    for lang, count in lang_counts.most_common():
        logger.info("  lang=%s: %d reviews", lang, count)

    dates = [r.date for r in reviews if r.date]
    if dates:
        logger.info("  date range: %s to %s", min(dates), max(dates))

    duplicate_groups = sum(1 for count in dup_hashes.values() if count > 1)
    if duplicate_groups:
        logger.info("  near-duplicate text groups: %d (S2-05, not dropped)", duplicate_groups)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize raw scraped reviews into the canonical schema (Stage 2)."
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: project root)")
    parser.add_argument(
        "--refresh", action="store_true", help="Rebuild reviews.jsonl even if it already exists"
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config) if args.config else load_config()
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    try:
        normalize_all(config, refresh=args.refresh)
    except NormalizeError as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
