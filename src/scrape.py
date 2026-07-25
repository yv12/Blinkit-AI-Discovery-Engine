"""Stage 1 - Scrape Google Play Store reviews for the target app.

See architecture.md §4 (Stage 1) and edgecases.md "Stage 1 - Scrape" for the
full edge-case catalog. IDs referenced below (S1-xx, X-xx) trace back to
edgecases.md.

Collection target is a **rolling lookback window** (default: last N calendar
months, see config.yaml `scrape.lookback_months`), not a fixed review count.
Reviews are pulled per (sort, rating) bucket so the corpus has explicit
rating-band coverage (S1-10). For the "newest" sort (chronological), pagination
stops exactly at the window boundary once a page's reviews fall before the
cutoff - this is exact and complete. For non-chronological sorts (e.g.
"relevance"), coverage of the window is not guaranteed; see S1-12. Progress is
checkpointed after every bucket so an interrupted run can resume without
re-fetching completed buckets (S1-02, S1-06), and a completed corpus is not
re-scraped on the next run unless ``--refresh`` is passed (S1-09).
"""

from __future__ import annotations

import argparse
import calendar
import logging
import math
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from google_play_scraper import Sort
from google_play_scraper import app as gp_app
from google_play_scraper import reviews as gp_reviews

from src.config import Config, ensure_data_dir, load_config
from src.schema import read_json, read_jsonl, write_json, write_jsonl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

RATINGS = (1, 2, 3, 4, 5)
SORT_MODES = {"newest": Sort.NEWEST, "relevance": Sort.MOST_RELEVANT}
CHRONOLOGICAL_SORTS = {"newest"}  # only these support exact/complete date-window early-stop (S1-12)
MAX_PAGE_SIZE = 200  # Play Store's per-request cap.
REQUIRED_RAW_KEYS = {"reviewId", "content", "score", "at"}


class ScrapeError(RuntimeError):
    """Raised for unrecoverable scrape failures (app not found, zero results, ...)."""


def _months_ago(months: int, from_dt: datetime) -> datetime:
    """Exact calendar-month subtraction (e.g. "4 months ago"), no extra dependency needed."""
    total_month_index = from_dt.year * 12 + (from_dt.month - 1) - months
    year, month0 = divmod(total_month_index, 12)
    month = month0 + 1
    day = min(from_dt.day, calendar.monthrange(year, month)[1])
    return from_dt.replace(year=year, month=month, day=day, hour=0, minute=0, second=0, microsecond=0)


def _parse_iso_utc(value: Any) -> Optional[datetime]:
    """Coerce a datetime or ISO string to an aware UTC datetime (S2-08 convention: naive == UTC)."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value:
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _verify_app_exists(config: Config) -> None:
    """Fail before pulling anything if the app id is wrong/changed (S1-01)."""
    try:
        info = gp_app(config.app.app_id, lang=config.app.lang, country=config.app.country)
    except Exception as exc:
        raise ScrapeError(
            f"Could not verify Play Store app id '{config.app.app_id}' (edgecases.md S1-01): "
            f"{exc}. Confirm the id at "
            f"https://play.google.com/store/apps/details?id={config.app.app_id}"
        ) from exc
    if not info or not info.get("title"):
        raise ScrapeError(
            f"Play Store app id '{config.app.app_id}' returned no usable metadata "
            "(edgecases.md S1-01)."
        )
    logger.info("Verified app: %s (%s)", info.get("title"), config.app.app_id)


def _validate_review_schema(sample: Dict[str, Any]) -> None:
    """Detect a google-play-scraper API/field change early (S1-07)."""
    missing = REQUIRED_RAW_KEYS - sample.keys()
    if missing:
        raise ScrapeError(
            f"google-play-scraper response is missing expected keys {sorted(missing)} "
            f"(edgecases.md S1-07: library API may have changed). Got keys: {sorted(sample.keys())}"
        )


def _retry_with_backoff(fn, *, description: str, retries: int = 4, base_delay: float = 1.0, max_delay: float = 20.0):
    """Exponential backoff + jitter around a flaky network call (S1-02, S1-06)."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - library surfaces network errors generically
            last_exc = exc
            if attempt == retries:
                break
            delay = min(max_delay, base_delay * (2 ** (attempt - 1))) + random.uniform(0, 0.5)
            logger.warning(
                "%s failed (attempt %d/%d): %s - retrying in %.1fs",
                description, attempt, retries, exc, delay,
            )
            time.sleep(delay)
    raise ScrapeError(f"{description} failed after {retries} attempts: {last_exc}") from last_exc


def _serialize_raw(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Make a raw review dict JSON-safe without altering its values (datetimes -> ISO strings)."""
    serialized = dict(raw)
    for key in ("at", "repliedAt"):
        value = serialized.get(key)
        if isinstance(value, datetime):
            serialized[key] = value.isoformat()
    return serialized


def _fetch_bucket(
    app_id: str,
    lang: str,
    country: str,
    sort_name: str,
    sort_enum: Sort,
    rating: int,
    cutoff: datetime,
    max_count: int,
    max_pages: int,
    sleep_ms: int,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Fetch reviews for one (sort, rating) bucket, bounded by the lookback window.

    For chronological sorts ("newest"), pagination stops as soon as a page's
    reviews fall before ``cutoff`` - exact and complete. For non-chronological
    sorts, ordering can't be trusted to stop early, so we page up to
    ``max_pages``/``max_count`` and drop out-of-window reviews as we go;
    completeness is not guaranteed there (edgecases.md S1-12).

    Returns (reviews_within_window, hit_safety_cap).
    """
    chronological = sort_name in CHRONOLOGICAL_SORTS
    collected: List[Dict[str, Any]] = []
    token = None
    seen_tokens: set = set()
    hit_cap = False

    for page in range(max_pages):
        remaining = max_count - len(collected)
        if remaining <= 0:
            hit_cap = True
            break
        batch_size = min(MAX_PAGE_SIZE, remaining)

        def _do_fetch(_token=token, _batch_size=batch_size):
            return gp_reviews(
                app_id,
                lang=lang,
                country=country,
                sort=sort_enum,
                count=_batch_size,
                filter_score_with=rating,
                continuation_token=_token,
            )

        result, next_token = _retry_with_backoff(
            _do_fetch, description=f"reviews fetch (rating={rating}, sort={sort_name}, page={page})"
        )

        if not result:
            break
        if page == 0:
            _validate_review_schema(result[0])

        reached_cutoff = False
        for raw in result:
            review_dt = _parse_iso_utc(raw.get("at"))
            if review_dt is not None and review_dt < cutoff:
                if chronological:
                    reached_cutoff = True
                    break  # newest-sorted: everything after this is even older
                continue  # non-chronological: skip this one, keep paging
            collected.append(raw)

        if reached_cutoff:
            break
        if page == max_pages - 1:
            hit_cap = True

        token = next_token
        if token is None:
            break
        token_id = getattr(token, "token", None)
        if token_id is None:
            break
        if token_id in seen_tokens:
            logger.warning(
                "Continuation token repeated for rating=%d sort=%s; stopping pagination (S1-03)",
                rating, sort_name,
            )
            break
        seen_tokens.add(token_id)
        time.sleep(max(0, sleep_ms) / 1000.0 + random.uniform(0, 0.15))

    return collected, hit_cap


def _load_progress(config: Config, refresh: bool) -> Tuple[Dict[str, Dict[str, Any]], set, Optional[datetime]]:
    """Resume from a prior interrupted run unless --refresh is passed (S1-09).

    Reuses the *originally computed* cutoff stored alongside the progress
    file rather than recomputing "N months ago" from "now" on every resume -
    otherwise a multi-day interrupted resume would keep shifting the window.
    """
    if refresh:
        return {}, set(), None

    progress_path = config.paths.data_dir / "raw_reviews.progress.json"
    if not progress_path.exists():
        return {}, set(), None

    try:
        progress = read_json(progress_path)
        completed_buckets = set(progress.get("completed_buckets", []))
        stored_cutoff = _parse_iso_utc(progress.get("cutoff"))
    except Exception as exc:  # corrupt progress file: don't crash, just restart (X-04)
        logger.warning("Progress file unreadable (%s); restarting scrape from scratch.", exc)
        return {}, set(), None

    collected: Dict[str, Dict[str, Any]] = {}
    if config.paths.raw_reviews.exists():
        try:
            for raw in read_jsonl(config.paths.raw_reviews, factory=None):
                rid = raw.get("reviewId")
                if rid:
                    collected[rid] = raw
        except Exception as exc:  # corrupt raw file: don't crash, just restart (X-04)
            logger.warning("Raw reviews file unreadable (%s); restarting scrape from scratch.", exc)
            return {}, set(), None

    if completed_buckets:
        logger.info(
            "Resuming scrape (cutoff=%s): %d buckets already complete, %d reviews cached.",
            stored_cutoff.date().isoformat() if stored_cutoff else "unknown",
            len(completed_buckets), len(collected),
        )
    return collected, completed_buckets, stored_cutoff


def _checkpoint(
    config: Config, collected: Dict[str, Dict[str, Any]], completed_buckets: set, cutoff: datetime
) -> None:
    """Atomically persist progress after each bucket so a crash loses no work (X-05)."""
    write_jsonl(config.paths.raw_reviews, list(collected.values()))
    progress_path = config.paths.data_dir / "raw_reviews.progress.json"
    write_json(
        progress_path,
        {
            "completed_buckets": sorted(completed_buckets),
            "review_count": len(collected),
            "cutoff": cutoff.isoformat(),
        },
    )


def scrape_all(
    config: Config,
    refresh: bool = False,
    sleep_ms: int = 200,
    max_pages_override: Optional[int] = None,
) -> None:
    """Scrape every review within the lookback window, per (sort, rating) bucket, deduped by id."""
    ensure_data_dir(config)

    unknown_sorts = set(config.scrape.sorts) - set(SORT_MODES)
    if unknown_sorts:
        raise ScrapeError(f"Unknown sort mode(s) in config.yaml: {unknown_sorts}. Valid: {list(SORT_MODES)}")

    progress_path = config.paths.data_dir / "raw_reviews.progress.json"
    if config.paths.raw_reviews.exists() and not progress_path.exists() and not refresh:
        logger.info(
            "'%s' already exists and no progress file is pending; skipping scrape "
            "(pass --refresh to re-scrape). (S1-09)",
            config.paths.raw_reviews,
        )
        return

    _verify_app_exists(config)

    collected, completed_buckets, stored_cutoff = _load_progress(config, refresh)
    cutoff = stored_cutoff or _months_ago(config.scrape.lookback_months, datetime.now(timezone.utc))
    logger.info(
        "Collecting reviews on/after %s (last %d months)",
        cutoff.date().isoformat(), config.scrape.lookback_months,
    )

    non_chronological = set(config.scrape.sorts) - CHRONOLOGICAL_SORTS
    if non_chronological:
        logger.warning(
            "Sort mode(s) %s are not chronological; window coverage for those buckets is "
            "best-effort, not guaranteed complete (edgecases.md S1-12).",
            sorted(non_chronological),
        )

    buckets = [(sort_name, rating) for sort_name in config.scrape.sorts for rating in RATINGS]
    max_pages = max_pages_override or max(5, math.ceil(config.scrape.max_per_bucket / MAX_PAGE_SIZE) + 3)

    for sort_name, rating in buckets:
        bucket_key = f"{sort_name}:{rating}"
        if bucket_key in completed_buckets:
            logger.info("Skipping already-completed bucket %s", bucket_key)
            continue

        logger.info(
            "Fetching bucket sort=%s rating=%d (cutoff=%s, max_per_bucket=%d, max_pages=%d)",
            sort_name, rating, cutoff.date().isoformat(), config.scrape.max_per_bucket, max_pages,
        )
        try:
            batch, hit_cap = _fetch_bucket(
                config.app.app_id, config.app.lang, config.app.country,
                sort_name, SORT_MODES[sort_name], rating, cutoff,
                config.scrape.max_per_bucket, max_pages, sleep_ms,
            )
        except ScrapeError as exc:
            # One bucket failing shouldn't abort the whole corpus (S1-02/S1-06 degrade, not block).
            logger.error("Bucket %s failed, continuing with remaining buckets: %s", bucket_key, exc)
            batch, hit_cap = [], False

        if hit_cap:
            logger.warning(
                "Bucket %s hit its max_per_bucket/max_pages safety cap before exhausting the "
                "window; corpus for this rating may be incomplete. Raise scrape.max_per_bucket "
                "if this matters.",
                bucket_key,
            )
        if not batch:
            logger.warning(
                "Bucket %s returned 0 reviews within the window (edgecases.md S1-04/S1-10)",
                bucket_key,
            )

        new_count = 0
        for raw in batch:
            rid = raw.get("reviewId")
            if not rid:
                continue
            if rid not in collected:
                new_count += 1
            collected[rid] = _serialize_raw(raw)

        logger.info(
            "Bucket %s: %d within window (%d new, %d duplicates)",
            bucket_key, len(batch), new_count, len(batch) - new_count,
        )

        completed_buckets.add(bucket_key)
        _checkpoint(config, collected, completed_buckets, cutoff)

    if progress_path.exists():
        progress_path.unlink()

    _log_summary(collected, cutoff)


def _log_summary(collected: Dict[str, Dict[str, Any]], cutoff: datetime) -> None:
    total = len(collected)
    if total == 0:
        raise ScrapeError(
            "Scrape produced zero reviews within the lookback window (edgecases.md X-10). Check "
            "network access, the app id, and scrape.lookback_months before re-running."
        )

    rating_counts = Counter(r.get("score") for r in collected.values())
    logger.info(
        "Scrape complete: %d unique reviews collected (window: since %s)",
        total, cutoff.date().isoformat(),
    )
    for rating in RATINGS:
        count = rating_counts.get(rating, 0)
        flag = " <- EMPTY (edgecases.md S1-10)" if count == 0 else ""
        logger.info("  rating=%d: %d reviews%s", rating, count, flag)

    review_dates = [d for d in (_parse_iso_utc(r.get("at")) for r in collected.values()) if d is not None]
    if review_dates:
        logger.info(
            "  date range: %s to %s",
            min(review_dates).date().isoformat(), max(review_dates).date().isoformat(),
        )
        stale = sum(1 for d in review_dates if d < cutoff)
        if stale:
            logger.warning(
                "%d collected reviews are older than the lookback cutoff (expected only from "
                "non-chronological sort buckets; edgecases.md S1-12)", stale,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Google Play Store reviews for the target app within a lookback window (Stage 1)."
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: project root)")
    parser.add_argument(
        "--refresh", action="store_true",
        help="Ignore cached progress/output and re-scrape from scratch (edgecases.md S1-09)",
    )
    parser.add_argument(
        "--sleep-ms", type=int, default=200,
        help="Delay between page fetches in milliseconds (politeness / rate-limit avoidance)",
    )
    parser.add_argument(
        "--max-pages", type=int, default=None,
        help="Override max pages per (sort, rating) bucket (safety cap, edgecases.md S1-03)",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config) if args.config else load_config()
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    try:
        scrape_all(config, refresh=args.refresh, sleep_ms=args.sleep_ms, max_pages_override=args.max_pages)
    except ScrapeError as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
