"""Optional presentation stage - plain-English titles for EVERY theme (not just
the top-5 growth report from src/theme_titles.py).

Why this exists: Stage 7 (`summarize.py`) labels each of the 40 themes with its
literal TF-IDF top terms (e.g. "app / good / best / nice", "blinkit / good / app /
service"). Fine for a machine, unreadable for a human skimming a spreadsheet. This
stage asks the LLM for a short, real-sentence title per theme - a problem statement
for complaint-heavy themes, a positive statement for praise-heavy ones - so every row
of `theme_reviews.csv` reads like something a person said, not a bag of keywords.

Difference from src/theme_titles.py: that stage FILTERS to the 5 most
growth-relevant themes and adds a causal "connection" sentence. This stage titles
ALL themes (positive included) with no filtering - it's a labeling convenience,
not a PM decision report.

Run with:  python -m src.theme_labels             (writes data/theme_labels.json)
           python -m src.theme_labels --refresh    (rebuild even if it exists)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from typing import Dict, List

from src.config import Config, load_config
from src.llm_synthesis import _RateLimiter, _call_groq, _load_api_key, LLMSynthesisError
from src.schema import read_json, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

MAX_QUOTES_PER_THEME = 3
MAX_QUOTE_CHARS = 140
BATCH_SIZE = 20  # keep each Groq call comfortably under the free-tier 6k TPM cap

_PROMPT_TEMPLATE = """You are labeling clusters of Blinkit (grocery delivery app) reviews for a \
spreadsheet a Product Manager will skim. Each cluster below has an ID, its average star rating, \
and a few real verbatim snippets.

For EACH cluster, write a `title`: a short, real, human-readable sentence (4-10 words) that \
says what these users are actually saying - NEVER bare topic words.
- If the cluster is mostly complaints/friction (low rating, negative language): phrase the title \
as a PROBLEM STATEMENT from the user's perspective.
  GOOD: "Customer support is unhelpful and slow to respond"; "Delivery charges feel too high"; \
"Orders arrive with missing or wrong items".
- If the cluster is mostly praise (high rating, positive language): phrase the title as what \
users like/appreciate.
  GOOD: "Users praise fast, reliable delivery"; "Users find the app simple and convenient".
- BAD (never output these - bare keyword salad): "Blinkit good app service"; "App good best nice"; \
"Delivery fast good time"; "Customer support service care".
- Also set `sentiment` to positive, negative, or neutral/mixed based on the snippets and rating.
- Base the title only on the snippets given; do not invent facts.

Respond with strict JSON only, no other text:
{{"clusters": [{{"cluster_id": "<id>", "title": "<4-10 word real sentence>", \
"sentiment": "positive|negative|neutral"}}, ...]}}

Clusters:
{clusters}"""


def _build_batches(themes: List[dict]) -> List[List[dict]]:
    return [themes[i : i + BATCH_SIZE] for i in range(0, len(themes), BATCH_SIZE)]


def _build_prompt(batch: List[dict]) -> str:
    blocks = []
    for t in batch:
        rating = t.get("avg_rating")
        lines = [
            f"[CLUSTER {t['theme_id']}] avg_rating={rating if rating is not None else 'n/a'}; "
            f"keywords: {', '.join(t['keywords'])}",
            "snippets:",
        ]
        lines += [f'- "{q[:MAX_QUOTE_CHARS]}"' for q in t["quotes"]]
        blocks.append("\n".join(lines))
    return _PROMPT_TEMPLATE.format(clusters="\n\n".join(blocks))


def _fallback_title(label: str) -> str:
    # Never leave a row with a raw keyword salad if the LLM fails on a batch.
    keywords = [k.strip() for k in label.split("/") if k.strip()]
    return " / ".join(keywords[:4]) if keywords else label


def run_theme_labels(config: Config, refresh: bool = False) -> None:
    out_path = config.paths.data_dir / "theme_labels.json"
    if out_path.exists() and not refresh:
        logger.info("'%s' already exists; skipping (pass --refresh to rebuild).", out_path)
        return

    if not config.paths.themes.exists() or not config.paths.communities.exists():
        raise LLMSynthesisError(
            "Expected themes.json/communities.json but they do not exist. Run `python -m src.pipeline` first."
        )

    api_key = _load_api_key()

    themes_doc = read_json(config.paths.themes)
    communities_by_id = {c["community_id"]: c for c in read_json(config.paths.communities)["communities"]}

    themes: List[dict] = []
    for t in themes_doc["themes"]:
        community = communities_by_id.get(t["community_id"], {})
        quotes = t.get("representative_quotes", [])[:MAX_QUOTES_PER_THEME]
        themes.append(
            {
                "theme_id": t["theme_id"],
                "label": t["label"],
                "keywords": [k.strip() for k in t["label"].split("/") if k.strip()],
                "quotes": quotes,
                "avg_rating": community.get("avg_rating"),
                "member_count": t["member_count"],
            }
        )

    limiter = _RateLimiter(config.llm_synthesis.requests_per_minute)
    decisions: Dict[str, dict] = {}
    batches = _build_batches(themes)
    logger.info("Labeling %d themes across %d Groq calls (%s)...", len(themes), len(batches), config.llm_synthesis.model)
    for i, batch in enumerate(batches, start=1):
        prompt = _build_prompt(batch)
        result = _call_groq(
            prompt, config.llm_synthesis.model, api_key, config.llm_synthesis.temperature,
            config.seed, config.llm_synthesis.max_retries, limiter,
        )
        if not result or not isinstance(result.get("clusters"), list):
            logger.warning("Batch %d/%d returned no usable titles; those themes will use a fallback label.", i, len(batches))
            continue
        for entry in result["clusters"]:
            try:
                cid = str(entry["cluster_id"]).strip()
                if cid:
                    decisions[cid] = entry
            except (KeyError, TypeError):
                continue

    out_themes = []
    fallback_count = 0
    for t in themes:
        decision = decisions.get(t["theme_id"], {})
        title = str(decision.get("title", "")).strip()
        if not (4 <= len(title.split()) <= 12):
            title = _fallback_title(t["label"])
            fallback_count += 1
        sentiment = str(decision.get("sentiment", "")).lower()
        if sentiment not in {"positive", "negative", "neutral"}:
            sentiment = "neutral"
        out_themes.append(
            {
                "theme_id": t["theme_id"],
                "title": title,
                "sentiment": sentiment,
                "avg_rating": t["avg_rating"],
                "member_count": t["member_count"],
            }
        )

    payload = {
        "method": "llm_groq",
        "model": config.llm_synthesis.model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "theme_count": len(out_themes),
        "fallback_count": fallback_count,
        "themes": out_themes,
    }
    write_json(out_path, payload)
    logger.info("Wrote %s: %d themes titled (%d fell back to a keyword label).", out_path, len(out_themes), fallback_count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Give every theme a plain-English title (Groq).")
    parser.add_argument("--config", default=None)
    parser.add_argument("--refresh", action="store_true", help="Rebuild even if data/theme_labels.json exists")
    args = parser.parse_args()

    try:
        config = load_config(args.config) if args.config else load_config()
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    try:
        run_theme_labels(config, refresh=args.refresh)
    except LLMSynthesisError as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
