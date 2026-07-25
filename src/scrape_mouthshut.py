"""Stage 1b - Ingest a pre-scraped Mouthshut reviews CSV as the second data source.

See Docs/context.md Addendum ("Second data source - Mouthshut") for the full
rationale/decision record. This deliberately, explicitly relaxes the original
single-source (Google Play only) constraint in problemstatement.md §5 - a
decision made with the user, not a silent scope change (see schema.py
VALID_SOURCES).

Unlike src/scrape.py (Stage 1), this stage does not call any network API: the
CSV at ``paths.mouthshut_csv`` (default ``data/Mouthshut_reviews.csv``) is
expected to already exist, dropped in by hand or by a separate process. This
module's only job is to read it and emit a raw JSONL artifact in the same
"raw, unnormalized, one record per review" shape that Stage 2 (normalize)
already expects from Stage 1, tagged with ``source: "mouthshut"`` so
normalize.py can route each record to the right source-specific normalizer.

This second source is optional end-to-end: if the CSV is not present, this
stage logs and returns without writing anything, and the rest of the
pipeline runs exactly as it did with Google Play as the only source.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from src.config import Config, ensure_data_dir, load_config
from src.schema import write_jsonl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

REQUIRED_CSV_COLUMNS = {"review_url", "title", "body", "rating", "reviewer", "review_date"}


class MouthshutIngestError(RuntimeError):
    """Raised for unrecoverable ingestion failures (malformed CSV header)."""


def _reference_date(csv_path: Path) -> datetime:
    """Anchor for resolving relative dates like "6 days ago" (normalize.py).

    Mouthshut's own scrape date isn't recorded in the CSV, so the file's
    last-modified time is used as a documented best-effort proxy for "when
    this snapshot was taken" - cross-checked against the file's own data
    (Docs/context.md Addendum): the newest absolute date in the real file is
    2026-06-20 and its oldest relative date is "30 days ago", which is
    consistent with a reference date in the last few days of July 2026.
    """
    return datetime.fromtimestamp(csv_path.stat().st_mtime, tz=timezone.utc)


def _row_to_raw(row: Dict[str, Any], reference_date_iso: str) -> Dict[str, Any]:
    url = (row.get("review_url") or "").strip()
    # Deterministic id from the review's own URL (Mouthshut assigns no numeric
    # id in this export) - stable across re-ingests of the same CSV.
    review_id = "mouthshut-" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    return {
        "source": "mouthshut",
        "reviewId": review_id,
        "url": url or None,
        "title": (row.get("title") or "").strip(),
        "content": (row.get("body") or "").strip(),
        "rating_raw": row.get("rating"),
        "reviewer": (row.get("reviewer") or "").strip() or None,
        "review_date_raw": (row.get("review_date") or "").strip(),
        # Ingestion-time anchor normalize.py uses to resolve relative dates
        # ("6 days ago") - captured once per ingest run, not recomputed later,
        # so re-running normalize doesn't shift already-ingested rows' dates.
        "reference_date": reference_date_iso,
    }


def ingest_mouthshut(config: Config, refresh: bool = False) -> None:
    ensure_data_dir(config)

    csv_path = config.paths.mouthshut_csv
    if not csv_path.exists():
        logger.info(
            "'%s' not found; Mouthshut is an optional second source, skipping ingestion "
            "(the pipeline runs fine with Google Play as the only source).",
            csv_path,
        )
        return

    if config.paths.raw_mouthshut.exists() and not refresh:
        logger.info(
            "'%s' already exists; skipping Mouthshut ingestion (pass --refresh to re-ingest).",
            config.paths.raw_mouthshut,
        )
        return

    reference_date = _reference_date(csv_path)
    logger.info(
        "Ingesting Mouthshut CSV '%s' (relative-date reference: %s, file mtime-based)",
        csv_path, reference_date.date().isoformat(),
    )

    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not REQUIRED_CSV_COLUMNS.issubset(set(reader.fieldnames)):
            raise MouthshutIngestError(
                f"'{csv_path}' is missing expected columns {sorted(REQUIRED_CSV_COLUMNS)}. "
                f"Got: {reader.fieldnames}"
            )
        rows: List[Dict[str, Any]] = [
            _row_to_raw(row, reference_date.isoformat()) for row in reader
        ]

    if not rows:
        raise MouthshutIngestError(f"'{csv_path}' has a header but zero data rows.")

    write_jsonl(config.paths.raw_mouthshut, rows)
    dropped_no_url = sum(1 for r in rows if not r["url"])
    dropped_no_body = sum(1 for r in rows if not r["content"])
    logger.info(
        "Mouthshut ingestion complete: %d raw reviews written -> %s "
        "(%d missing url, %d empty body - not dropped here, normalize.py decides)",
        len(rows), config.paths.raw_mouthshut, dropped_no_url, dropped_no_body,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest the Mouthshut reviews CSV into a raw JSONL artifact (Stage 1b)."
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: project root)")
    parser.add_argument(
        "--refresh", action="store_true", help="Re-ingest even if raw_mouthshut.jsonl already exists"
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config) if args.config else load_config()
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    try:
        ingest_mouthshut(config, refresh=args.refresh)
    except MouthshutIngestError as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
