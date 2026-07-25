"""Optional export stage - dump every review grouped by the theme it belongs to.

Why this exists: the pipeline artifacts (themes.json, communities.json) describe
themes at the *unit* level, which is hard to eyeball. A Growth PM often just wants
one flat, spreadsheet-friendly file: "show me the actual reviews that make up each
theme." This stage produces exactly that - a CSV (and optional JSON) that maps every
theme to its supporting reviews so the categorisation can be inspected by hand.

What it writes (into the data dir):
- theme_reviews.csv  : one row per (theme, review), grouped by theme, worst ratings
                       first within each theme. Opens directly in Excel / Sheets.
- theme_reviews.json : the same data as a nested {theme -> [reviews]} structure
                       (only when --json is passed).

Key correctness choices:
- A review is counted ONCE per theme even if several of its units landed in that
  community (deduped on review_id), matching how review_count is reported elsewhere.
- A review CAN appear under more than one theme - that is real (its sentences were
  clustered into different communities) and is preserved, not hidden.
- Themes are ranked by member_count (largest first). `theme_title` is a real,
  human-readable sentence, never keyword salad: the top-5 growth report's causal
  problem statement (src/theme_titles.py) where one exists, else the plain-English
  title from src/theme_labels.py, else the raw TF-IDF label (`theme_keywords`) as a
  last resort if neither optional LLM stage has been run.

Run with:  python -m src.export_theme_reviews
           python -m src.export_theme_reviews --json           (also write nested JSON)
           python -m src.export_theme_reviews --top-only       (only the top-5 report themes)
           python -m src.export_theme_reviews --max-per-theme 200
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from typing import Dict, List, Optional

from src.config import Config, load_config
from src.schema import Review, Unit, read_json, read_jsonl, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "theme_rank",
    "theme_id",
    "theme_title",
    "theme_keywords",
    "in_top5_report",
    "sentiment",
    "theme_review_count",
    "review_id",
    "rating",
    "date",
    "source",
    "review_text",
]


def _load_theme_titles(config: Config) -> Dict[str, dict]:
    """theme_id -> growth-report entry (title/severity/...), if the optional
    src/theme_titles.py stage (top-5 growth report) has been run. Empty otherwise."""
    path = config.paths.data_dir / "theme_titles.json"
    if not path.exists():
        return {}
    try:
        doc = read_json(path)
    except Exception:
        logger.warning("Found theme_titles.json but could not read it; continuing without titles.")
        return {}
    return {t["theme_id"]: t for t in doc.get("themes", []) if t.get("theme_id")}


def _load_theme_labels(config: Config) -> Dict[str, dict]:
    """theme_id -> plain-English title entry, if the optional src/theme_labels.py
    stage (titles for ALL themes) has been run. Empty otherwise."""
    path = config.paths.data_dir / "theme_labels.json"
    if not path.exists():
        return {}
    try:
        doc = read_json(path)
    except Exception:
        logger.warning("Found theme_labels.json but could not read it; continuing without it.")
        return {}
    return {t["theme_id"]: t for t in doc.get("themes", []) if t.get("theme_id")}


def _rating_sort_key(row: dict):
    # Worst ratings first (None ratings last), then oldest date, for stable output.
    rating = row["rating"]
    return (rating if rating is not None else 99, row["date"] or "")


def build_theme_reviews(
    config: Config, top_only: bool = False, max_per_theme: Optional[int] = None
) -> List[dict]:
    """Return a list of theme blocks: {theme meta, reviews: [...]}, ranked by size."""
    themes_doc = read_json(config.paths.themes)
    communities_by_id = {
        c["community_id"]: c for c in read_json(config.paths.communities)["communities"]
    }
    theme_titles = _load_theme_titles(config)  # top-5 growth report (causal problem statements)
    theme_labels = _load_theme_labels(config)  # plain-English title for every theme

    logger.info("Indexing units -> review ids...")
    unit_to_review: Dict[str, str] = {
        u.unit_id: u.review_id for u in read_jsonl(config.paths.units, factory=Unit)
    }

    logger.info("Indexing reviews...")
    review_by_id: Dict[str, Review] = {
        r.id: r for r in read_jsonl(config.paths.reviews, factory=Review)
    }

    themes = sorted(themes_doc["themes"], key=lambda t: -t["member_count"])

    blocks: List[dict] = []
    for rank, theme in enumerate(themes, start=1):
        tid = theme["theme_id"]
        if top_only and tid not in theme_titles:
            continue
        community = communities_by_id.get(theme["community_id"], {})
        unit_ids = community.get("unit_ids", [])

        seen: set = set()
        reviews: List[dict] = []
        for uid in unit_ids:
            rid = unit_to_review.get(uid)
            if rid is None or rid in seen:
                continue
            review = review_by_id.get(rid)
            if review is None:
                continue
            seen.add(rid)
            reviews.append(
                {
                    "review_id": rid,
                    "rating": review.rating,
                    "date": review.date,
                    "source": review.source,
                    "review_text": review.text.replace("\r", " ").replace("\n", " ").strip(),
                }
            )

        reviews.sort(key=_rating_sort_key)
        if max_per_theme is not None:
            reviews = reviews[:max_per_theme]

        # Prefer the top-5 growth report's causal problem statement; otherwise the
        # plain-English title from theme_labels.py; otherwise the raw TF-IDF label as
        # a last resort so a row is never left as pure keyword salad.
        growth_entry = theme_titles.get(tid, {})
        label_entry = theme_labels.get(tid, {})
        title = growth_entry.get("title") or label_entry.get("title") or theme["label"]
        sentiment = label_entry.get("sentiment") or theme.get("sentiment", "neutral")

        blocks.append(
            {
                "theme_rank": rank,
                "theme_id": tid,
                "theme_title": title,
                "theme_keywords": theme["label"],
                "in_top5_report": tid in theme_titles,
                "sentiment": sentiment,
                "theme_review_count": len(seen),
                "reviews": reviews,
            }
        )

    return blocks


def write_csv(blocks: List[dict], path) -> int:
    rows = 0
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for b in blocks:
            for r in b["reviews"]:
                writer.writerow(
                    {
                        "theme_rank": b["theme_rank"],
                        "theme_id": b["theme_id"],
                        "theme_title": b["theme_title"],
                        "theme_keywords": b["theme_keywords"],
                        "in_top5_report": b["in_top5_report"],
                        "sentiment": b["sentiment"],
                        "theme_review_count": b["theme_review_count"],
                        "review_id": r["review_id"],
                        "rating": r["rating"] if r["rating"] is not None else "",
                        "date": r["date"],
                        "source": r["source"],
                        "review_text": r["review_text"],
                    }
                )
                rows += 1
    return rows


def run_export(
    config: Config,
    also_json: bool = False,
    top_only: bool = False,
    max_per_theme: Optional[int] = None,
) -> None:
    for path in (config.paths.units, config.paths.themes, config.paths.communities, config.paths.reviews):
        if not path.exists():
            raise FileNotFoundError(
                f"Expected artifact at {path} but it does not exist. Run `python -m src.pipeline` first."
            )

    blocks = build_theme_reviews(config, top_only=top_only, max_per_theme=max_per_theme)

    csv_path = config.paths.data_dir / ("theme_reviews_top5.csv" if top_only else "theme_reviews.csv")
    rows = write_csv(blocks, csv_path)
    logger.info("Wrote %s: %d rows across %d themes.", csv_path, rows, len(blocks))

    if also_json:
        json_path = config.paths.data_dir / ("theme_reviews_top5.json" if top_only else "theme_reviews.json")
        write_json(json_path, {"themes": blocks})
        logger.info("Wrote %s (nested theme -> reviews).", json_path)

    for b in blocks[:10]:
        logger.info("  #%d [%s] \"%s\" - %d reviews", b["theme_rank"], b["theme_id"], b["theme_title"], b["theme_review_count"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export reviews grouped by theme to a browsable CSV/JSON.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--json", action="store_true", help="Also write a nested theme->reviews JSON file")
    parser.add_argument("--top-only", action="store_true", help="Only export the top-5 growth-report themes")
    parser.add_argument("--max-per-theme", type=int, default=None, help="Cap reviews exported per theme (default: all)")
    args = parser.parse_args()

    try:
        config = load_config(args.config) if args.config else load_config()
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    run_export(config, also_json=args.json, top_only=args.top_only, max_per_theme=args.max_per_theme)


if __name__ == "__main__":
    main()
