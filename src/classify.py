"""Addressability classifier — classifies each unit into exactly one of:
{app_ux, operational, pricing_policy, praise_noise}

See addressability-spec.md for the full specification. Design choices:

1. **Keyword/heuristic pre-filter** handles the bulk of units cheaply:
   - `praise_noise`: very short (≤5 words), pure-sentiment, no specific issue.
   - `operational`: delivery/rider/refund/missing/damaged/late vocabulary.
   - `pricing_policy`: fees/charges/surge/expensive/threshold vocabulary.
   Units not caught by the pre-filter are sent to the LLM.

2. **LLM path** (Groq, same API as barrier_mapping.py/llm_synthesis.py) classifies
   uncertain units using the exact prompt from addressability-spec.md. Batched to
   minimize API calls. Falls back to `praise_noise` on failure (safest default —
   those get excluded anyway).

3. **Outputs**: `data/unit_labels.jsonl` (unit_id → label, reason, method) and
   `data/classification_spot_check.csv` (50 stratified samples for manual review).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

from src.config import Config, apply_global_seed, ensure_data_dir, load_config
from src.schema import Unit, read_jsonl, write_json, write_jsonl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

VALID_LABELS = frozenset({"app_ux", "operational", "pricing_policy", "praise_noise"})

# ─── Heuristic keyword patterns ──────────────────────────────────────────────

# Operational: delivery, rider, refund, missing items, wrong items, damaged, late
_OPERATIONAL_PATTERNS = re.compile(
    r"\b("
    r"deliver(?:y|ed|ing|s)?|"
    r"rider(?:s)?|delivery\s*(?:boy|partner|agent|man|person)|"
    r"refund(?:ed|s|ing)?|"
    r"missing\s*(?:item|product|order)?|"
    r"wrong\s*(?:item|product|order)|"
    r"damage(?:d|s)?|"
    r"cancel(?:led|ed|lation|s)?|"
    r"late\s*(?:delivery|order)?|"
    r"delay(?:ed|s|ing)?|"
    r"not\s*(?:deliver|received)|"
    r"lost\s*(?:order|item|package)|"
    r"out\s*of\s*stock|stock\s*out|"
    r"expired?\b|near[\s-]*expir(?:y|ed)|"
    r"rotten|stale|spoil(?:ed|t)|"
    r"broken|leak(?:ed|ing|s)?|"
    r"packag(?:e|ing)\s*(?:damage|torn|open)|"
    r"customer\s*(?:care|support|service)|"
    r"complaint|"
    r"replacement|"
    r"no\s*(?:refund|response|reply|support)"
    r")\b",
    re.IGNORECASE,
)

# Pricing/policy: fees, charges, surge, expensive, threshold, handling charge
_PRICING_PATTERNS = re.compile(
    r"\b("
    r"(?:delivery|handling|platform|surge|packing|service)\s*(?:fee|charge|cost)s?|"
    r"free\s*delivery|"
    r"minimum\s*(?:order|cart)|"
    r"threshold|"
    r"expensive|overpriced|costly|"
    r"price\s*(?:increase|hike|rise|high)|"
    r"(?:high|extra|hidden)\s*(?:charges?|fees?)|"
    r"mrp|markup|"
    r"discount(?:s)?|coupon(?:s)?|offer(?:s)?|"
    r"(?:small|low)\s*(?:cart|order)\s*(?:fee|charge|surcharge)"
    r")\b",
    re.IGNORECASE,
)

# App UX: search, browse, recommendation, notification, UI, bug, crash, cart, checkout
_APP_UX_PATTERNS = re.compile(
    r"\b("
    r"search(?:ing|ed|es)?|"
    r"(?:browse|browsing)|"
    r"recommendation(?:s)?|suggested\s*(?:product|item)|product\s*suggestion|personali[sz](?:ed|ation)|"
    r"(?:push\s*)?notification(?:s)?|spam(?:ming)?|"
    r"(?:app|ui|ux|interface)\s*(?:design|layout|bug|crash|issue|problem|error|glitch|freeze|hang)|"
    r"crash(?:ed|es|ing)?|"
    r"bug(?:s|gy)?|glitch(?:es|y)?|"
    r"(?:not\s*)?load(?:ing|s)?|"
    r"cart\b|checkout|"
    r"category\s*(?:page|section|browse|list)|"
    r"filter(?:s|ing)?|sort(?:ing)?|"
    r"product\s*(?:page|detail|info|image|photo|review)|"
    r"(?:expiry|expiration)\s*(?:date|info)|"
    r"(?:recently\s*viewed|repeat\s*order|past\s*order|reorder)|"
    r"banner(?:s)?|popup(?:s)?|ad(?:s|vert)?(?:\s*(?:in|on)\s*(?:app|search))?|"
    r"navigation|navigate|menu|sidebar|"
    r"update(?:d|s|ing)?\s*(?:app|version)|"
    r"(?:new|latest)\s*(?:version|update)|"
    r"(?:app|latest)\s*(?:version|update)\s*(?:issue|problem|bug)|"
    r"login|sign[\s-]*(?:in|up)|otp|"
    r"payment\s*(?:option|method|fail|issue|gateway)"
    r")\b",
    re.IGNORECASE,
)

# Praise/noise: very short generic sentiment
_PRAISE_NOISE_PHRASES = re.compile(
    r"^(?:"
    r"(?:very\s+)?(?:good|nice|great|best|worst|bad|poor|excellent|amazing|awesome|fantastic|"
    r"terrible|horrible|superb|outstanding|love|hate|wonderful|pathetic|useless|fraud|"
    r"bakwas|bekar|wahiyat|ghatiya|zabardast|accha|badhiya|mast)\s*"
    r"(?:app|service|experience|platform)?\s*"
    r"(?:ever|!+|\.+)?|"
    r"(?:love|hate)\s*(?:it|this|blinkit)?!?|"
    r"(?:super|ok|okay|fine|not\s+bad)\s*(?:app)?!?"
    r")$",
    re.IGNORECASE,
)

_PUNCT_STRIP_RE = re.compile(r"[^\w\s]")


def _word_count(text: str) -> int:
    normalized = _PUNCT_STRIP_RE.sub("", text.lower()).strip()
    return len(normalized.split()) if normalized else 0


def _heuristic_classify(text: str) -> Optional[Tuple[str, str]]:
    """Try to classify a unit via keyword heuristics. Returns (label, reason) or None."""
    stripped = text.strip()
    wc = _word_count(stripped)

    # Very short + no specific keywords → praise_noise
    if wc <= 5:
        if _PRAISE_NOISE_PHRASES.match(stripped):
            return ("praise_noise", "short generic sentiment phrase")
        # Very short but has a keyword — check below
        if not (_OPERATIONAL_PATTERNS.search(stripped) or
                _PRICING_PATTERNS.search(stripped) or
                _APP_UX_PATTERNS.search(stripped)):
            return ("praise_noise", "very short, no specific issue keywords")

    # Check for praise_noise even in longer texts
    if _PRAISE_NOISE_PHRASES.match(stripped):
        return ("praise_noise", "generic sentiment phrase")

    # Count keyword hits for each category
    op_hits = len(_OPERATIONAL_PATTERNS.findall(stripped))
    price_hits = len(_PRICING_PATTERNS.findall(stripped))
    ux_hits = len(_APP_UX_PATTERNS.findall(stripped))

    total_hits = op_hits + price_hits + ux_hits

    # No keywords at all — no specific friction mentioned
    if total_hits == 0:
        return ("praise_noise", "no specific friction keywords found")

    # Clear winner (dominant category has 2x or more hits, or is the only one)
    if op_hits > 0 and op_hits >= 2 * max(price_hits, ux_hits, 1):
        return ("operational", f"strong operational keywords ({op_hits} hits)")
    if price_hits > 0 and price_hits >= 2 * max(op_hits, ux_hits, 1):
        return ("pricing_policy", f"strong pricing keywords ({price_hits} hits)")
    if ux_hits > 0 and ux_hits >= 2 * max(op_hits, price_hits, 1):
        return ("app_ux", f"strong app UX keywords ({ux_hits} hits)")

    # Only one category matched
    if op_hits > 0 and price_hits == 0 and ux_hits == 0:
        return ("operational", f"operational keywords ({op_hits} hits)")
    if price_hits > 0 and op_hits == 0 and ux_hits == 0:
        return ("pricing_policy", f"pricing keywords ({price_hits} hits)")
    if ux_hits > 0 and op_hits == 0 and price_hits == 0:
        return ("app_ux", f"app UX keywords ({ux_hits} hits)")

    # Mixed keywords — uncertain, send to LLM
    return None


# ─── LLM classification (Groq) ──────────────────────────────────────────────

_CLASSIFY_BATCH_PROMPT = """Classify each of the following Blinkit app reviews into exactly one label:
- app_ux: friction with the app interface itself (search results, out-of-stock handling, recommendations and personalization — repetitive or irrelevant suggestions, recently-viewed items shown instead of new products, ads crowding results — category/browse navigation, missing product information like expiry or reviews, cart, checkout, bugs, crashes, notifications/spam)
- operational: delivery speed, delivery partners, stockouts, missing/wrong/damaged items, refunds, customer support
- pricing_policy: fees, handling/surge charges, thresholds, prices
- praise_noise: generic praise or abuse with no specific issue ("best app", "worst app ever", or generic unhelpful complaints like "worst experience")

Return ONLY a JSON dictionary where keys are the provided unit IDs, and values are objects with "label" and "reason" (a short phrase).

Reviews to classify (JSON format):
{reviews_json}"""


def _call_groq_classify_batch(
    units: List[Unit],
    model: str,
    api_key: str,
    seed: int,
    batch_size: int,
    max_retries: int = 3,
) -> Dict[str, Tuple[str, str]]:
    """Classify a list of units via Groq using batched prompts. Returns {unit_id: (label, reason)}."""
    from src.llm_synthesis import _call_groq, _load_api_key, _RateLimiter
    import json

    # Use a higher rate limit internally to allow faster bursts if the API allows it,
    # but the API itself might return 429 which _call_groq handles with backoff.
    limiter = _RateLimiter(requests_per_minute=25)
    results: Dict[str, Tuple[str, str]] = {}

    for i in range(0, len(units), batch_size):
        batch = units[i : i + batch_size]
        batch_dict = {u.unit_id: u.text[:500] for u in batch}  # Truncate to save tokens
        
        prompt = _CLASSIFY_BATCH_PROMPT.replace("{reviews_json}", json.dumps(batch_dict, indent=2))
        
        data = _call_groq(
            prompt=prompt,
            model=model,
            api_key=api_key,
            temperature=0.0,
            seed=seed,
            max_retries=max_retries,
            limiter=limiter,
        )
        
        if data and isinstance(data, dict):
            for unit in batch:
                uid = unit.unit_id
                if uid in data and isinstance(data[uid], dict) and data[uid].get("label") in VALID_LABELS:
                    results[uid] = (
                        data[uid]["label"],
                        data[uid].get("reason", "LLM classified"),
                    )
                else:
                    results[uid] = ("praise_noise", "LLM failed to classify correctly, defaulting")
        else:
            # Fallback for entire batch failure
            for unit in batch:
                results[unit.unit_id] = ("praise_noise", "LLM batch failed, defaulting to praise_noise")
            logger.warning("LLM batch classification failed for %d units.", len(batch))

        logger.info(
            "  LLM batch %d–%d / %d done (%d classified so far).",
            i, min(i + batch_size, len(units)), len(units), len(results),
        )

    return results


# ─── Spot-check CSV ──────────────────────────────────────────────────────────

def _write_spot_check_csv(
    labels: Dict[str, dict],
    unit_meta: Dict[str, Unit],
    sample_size: int,
    output_path,
    seed: int,
) -> None:
    """Write a stratified random sample across all 4 labels to CSV."""
    rng = random.Random(seed)
    by_label: Dict[str, List[str]] = {label: [] for label in VALID_LABELS}
    for uid, info in labels.items():
        by_label[info["label"]].append(uid)

    # Stratify: ~equal per label, fill up from larger groups
    per_label = max(1, sample_size // len(VALID_LABELS))
    sampled: List[dict] = []

    for label in sorted(VALID_LABELS):
        pool = by_label[label]
        n = min(per_label, len(pool))
        picked = rng.sample(pool, n)
        for uid in picked:
            unit = unit_meta.get(uid)
            if unit:
                sampled.append({
                    "unit_id": uid,
                    "review_text": unit.text,
                    "assigned_label": labels[uid]["label"],
                    "reason": labels[uid]["reason"],
                    "method": labels[uid]["method"],
                    "rating": unit.rating,
                })

    # Top up if we haven't reached sample_size
    remaining_ids = [uid for uid in labels if uid not in {s["unit_id"] for s in sampled}]
    if len(sampled) < sample_size and remaining_ids:
        extra = rng.sample(remaining_ids, min(sample_size - len(sampled), len(remaining_ids)))
        for uid in extra:
            unit = unit_meta.get(uid)
            if unit:
                sampled.append({
                    "unit_id": uid,
                    "review_text": unit.text,
                    "assigned_label": labels[uid]["label"],
                    "reason": labels[uid]["reason"],
                    "method": labels[uid]["method"],
                    "rating": unit.rating,
                })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["unit_id", "review_text", "assigned_label", "reason", "method", "rating"])
        writer.writeheader()
        writer.writerows(sampled)

    logger.info("Wrote %d spot-check samples to %s", len(sampled), output_path)


# ─── Main entry point ────────────────────────────────────────────────────────

def classify_units(config: Config, refresh: bool = False) -> None:
    """Classify every unit into {app_ux, operational, pricing_policy, praise_noise}."""
    ensure_data_dir(config)
    apply_global_seed(config.seed)

    if config.paths.unit_labels.exists() and not refresh:
        logger.info(
            "'%s' already exists; skipping classification (pass --refresh to rebuild).",
            config.paths.unit_labels,
        )
        return

    if not config.paths.units_raw.exists():
        raise RuntimeError(
            f"Expected units corpus at {config.paths.units_raw}. "
            "Run `python -m src.units` first."
        )

    logger.info("Loading units for addressability classification...")
    units: List[Unit] = list(read_jsonl(config.paths.units_raw, factory=Unit))
    unit_meta: Dict[str, Unit] = {u.unit_id: u for u in units}
    logger.info("Loaded %d units.", len(units))

    # Phase 1: Heuristic pre-filter
    labels: Dict[str, dict] = {}  # unit_id -> {label, reason, method}
    uncertain: List[Unit] = []

    for unit in units:
        result = _heuristic_classify(unit.text)
        if result is not None:
            labels[unit.unit_id] = {
                "label": result[0],
                "reason": result[1],
                "method": "heuristic",
            }
        else:
            uncertain.append(unit)

    heuristic_count = len(labels)
    logger.info(
        "Heuristic pre-filter: %d / %d classified (%.1f%%), %d uncertain → LLM.",
        heuristic_count, len(units),
        100.0 * heuristic_count / max(1, len(units)),
        len(uncertain),
    )

    # Phase 2: LLM classification for uncertain units
    if uncertain:
        try:
            from src.llm_synthesis import _load_api_key
            api_key = _load_api_key()
            if config.addressability.llm_batch_size <= 0:
                logger.info(
                    "Bypassing LLM classification (llm_batch_size <= 0); defaulting %d units to praise_noise.",
                    len(uncertain),
                )
                for unit in uncertain:
                    labels[unit.unit_id] = {
                        "label": "praise_noise",
                        "reason": "LLM bypassed",
                        "method": "heuristic_bypass",
                    }
            else:
                logger.info("Classifying %d uncertain units via Groq LLM...", len(uncertain))
                llm_results = _call_groq_classify_batch(
                    uncertain,
                    model=config.llm_synthesis.model,
                    api_key=api_key,
                    seed=config.seed,
                    batch_size=config.addressability.llm_batch_size,
                    max_retries=config.llm_synthesis.max_retries,
                )
                for uid, (label, reason) in llm_results.items():
                    labels[uid] = {"label": label, "reason": reason, "method": "llm"}
        except Exception as exc:
            logger.warning(
                "Groq LLM classification failed (%s); assigning remaining %d units as praise_noise.",
                exc, len(uncertain),
            )
            for unit in uncertain:
                if unit.unit_id not in labels:
                    labels[unit.unit_id] = {
                        "label": "praise_noise",
                        "reason": "LLM unavailable, defaulting to praise_noise",
                        "method": "fallback",
                    }

    # Ensure every unit is labeled
    for unit in units:
        if unit.unit_id not in labels:
            labels[unit.unit_id] = {
                "label": "praise_noise",
                "reason": "unclassified fallback",
                "method": "fallback",
            }

    # Write unit_labels.jsonl
    label_records = [
        {"unit_id": uid, **info}
        for uid, info in sorted(labels.items())
    ]
    write_jsonl(config.paths.unit_labels, label_records)

    # Write classification distribution stats as part of the artifact
    dist = Counter(info["label"] for info in labels.values())
    method_dist = Counter(info["method"] for info in labels.values())
    logger.info("Classification distribution:")
    for label in sorted(VALID_LABELS):
        count = dist.get(label, 0)
        pct = 100.0 * count / max(1, len(labels))
        logger.info("  %s: %d (%.1f%%)", label, count, pct)
    logger.info("Classification method distribution:")
    for method, count in method_dist.most_common():
        logger.info("  %s: %d", method, count)

    # Write spot-check CSV
    _write_spot_check_csv(
        labels, unit_meta,
        config.addressability.spot_check_sample_size,
        config.paths.classification_spot_check,
        config.seed,
    )

    logger.info(
        "Classification complete: %d units labeled → %s",
        len(label_records), config.paths.unit_labels,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify units into {app_ux, operational, pricing_policy, praise_noise} (addressability-spec.md)."
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: project root)")
    parser.add_argument("--refresh", action="store_true", help="Rebuild unit_labels.jsonl even if it exists")
    args = parser.parse_args()

    try:
        config = load_config(args.config) if args.config else load_config()
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    try:
        classify_units(config, refresh=args.refresh)
    except Exception as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
