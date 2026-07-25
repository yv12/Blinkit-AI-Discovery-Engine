"""Optional presentation stage - growth-relevant theme extraction for the headline
Growth PM view.

Why this exists: Stage 7 (`summarize.py`) labels each community with its literal
TF-IDF top terms (e.g. "blinkit / good / app / service"). Those are honest and
deterministic, but they read as topic-word salad, not as the actionable *user
problem* a Growth PM would act on. This stage screens the strongly-supported
clusters for a plausible link to category-exploration behaviour (full funnel:
direct discovery friction AND operational / pricing / quality / trust issues that
plausibly push users to stick to habitual categories), then ranks the survivors and
asks the LLM to phrase each as a short 4-8-word problem statement that names the
mechanism, plus one explicit sentence on how it connects to category exploration.

Design choices (same principles as src/llm_synthesis.py, Stage 11):
- **Remote Groq API, temperature 0 + fixed seed** for reproducible output. This is
  the same deliberate, opt-in deviation from "fully local" that Stage 11 documents
  (zero-cost preserved, offline not). Needs GROQ_API_KEY in `.env`.
- **The title / connection are model interpretation; every number and quote stays
  real.** `review_count` and `share` are computed from the artifacts; quote text is
  verbatim. Quote *selection* is claim-aware (see below), not centroid similarity.
- **A theme must clear a support bar** (`>= max(N% of total reviews, 15)`, N from
  the CLI / default 1.0) before relevance screening. The output is at most the top
  five eligible themes and never backfills clearly-irrelevant clusters to reach five.
- **Claim-supporting snippets, not centroid generics.** After the top 5 are chosen,
  each theme samples a wider pool of 30-50 cluster reviews (preferring longer,
  specific-incident text - never embedding-centroid rank), then the LLM picks 2-3
  quotes that name a concrete incident AND support the theme's stated mechanism.
  Generic repeated sentiment ("customer support is not good") is discarded; if the
  widened sample still has no specific-incident quote, fall back to one generic.
- **Inspectable exclusions.** Clusters excluded (below support, or judged
  irrelevant) are recorded in `near_misses` with the reason, so a zero/short result
  is never silent.

Run with:  python -m src.theme_titles            (writes data/theme_titles.json)
           python -m src.theme_titles --refresh  (rebuild even if it exists)
           python -m src.theme_titles --min-share 2.0   (require >= 2% support)
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import sys
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from src.config import Config, apply_global_seed, ensure_data_dir, load_config
from src.llm_synthesis import _RateLimiter, _call_groq, _load_api_key, LLMSynthesisError
from src.schema import Review, Unit, read_json, read_jsonl, write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TOP_N = 5  # never surface more than the top 5
MIN_REVIEW_FLOOR = 15  # absolute floor regardless of the % gate
MAX_SNIPPETS_PER_THEME = 5  # sample verbatims shown to the LLM per theme (Groq free tier is 6k TPM)
MAX_SNIPPET_CHARS = 140  # truncate each snippet in the prompt to stay under the TPM cap
MIN_SNIPPET_WORDS = 6
MAX_NEAR_MISSES = 10

# Claim-aware quote selection (replaces centroid-picked representative_quotes).
QUOTE_POOL_SIZE = 50  # widen first: score ~50 reviews per theme before picking
QUOTE_SHOW_SIZE = 35  # how many of that pool are shown to the LLM (TPM budget)
QUOTES_PER_THEME = 10  # enough for a scrollable supporting-reviews panel in the UI
MIN_QUOTE_WORDS = 12  # short generics like "support is not good" fail this floor
MAX_QUOTE_CHARS = 220  # keep incident detail; per-theme calls stay under free-tier TPM
_SPECIFICITY_RE = re.compile(
    r"\b(ordered|order(?:ed)?|bought|refund|missing|wrong|expired|cancel(?:led)?|"
    r"stock(?:out)?|never|won'?t|again|first|tried|try(?:ing)?|₹|rs\.?|rupees?|"
    r"\d+\s*(?:min|mins|minutes|hrs?|hours|rs|rupees?)|"
    r"milk|egg|fruit|vegetable|chicken|bread|oil|rice|dal|atta|"
    r"cheat|scam|fraud|trust|uninstall|never\s+again)\b",
    re.IGNORECASE,
)
_TRUST_BREAK_RE = re.compile(
    r"(never\s+again|won'?t\s+order|will\s+not\s+order|never\s+order|"
    r"don'?t\s+(?:want\s+to\s+)?order|lost\s+trust|no\s+trust|"
    r"cheat|scam|fraud|fool(?:ed)?|uninstall|"
    r"not\s+to\s+buy|only\s+when\s+i\s+have\s+no\s+other|"
    r"try(?:ied)?\s+\w+|first\s+order)",
    re.IGNORECASE,
)
_GENERIC_COMPLAINT_RE = re.compile(
    r"^(customer\s+)?support\s+(is\s+)?(not\s+good|worst|bad|poor)|"
    r"^delivery\s+(charges?\s+)?(are\s+)?(very\s+|a\s+bit\s+)?high|"
    r"^delivery\s+(is\s+|time\s+is\s+)?(very\s+)?(late|slow)|"
    r"^service\s+(is\s+)?(not\s+good|bad|poor)|"
    r"^good\s+app|^nice\s+app|^best\s+app",
    re.IGNORECASE,
)

RESEARCH_QUESTIONS = {
    1: "Why users repeat the same categories / stick to habits",
    2: "What stops users trying or discovering new categories",
    3: "How users currently find products (search vs browse vs habit)",
    4: "Habit's role in repeat behavior",
    5: "Trust / risk factors that make users hesitant to try something new",
    6: "Info gaps before trying a new category (price, quality, fit unknown)",
    7: "Which user segments explore more, and why",
    8: "Unmet needs that block category exploration",
}


_PROMPT_TEMPLATE = """You are a senior Growth PM analyst studying why Blinkit users keep buying \
the SAME product categories instead of trying NEW ones. Each cluster below has an ID, its \
community average rating, and real verbatim review snippets.

STEP 1 - RELEVANCE FILTER (full funnel, NOT discovery-only):
Mark a cluster eligible=true if it plausibly explains ANY of these, even indirectly:
1. Why users repeat the same categories / stick to habits
2. What stops users trying or discovering new categories
3. How users currently find products (search vs browse vs habit)
4. Habit's role in repeat behavior
5. Trust / risk factors that make users hesitant to try something new (stockouts, bad quality on \
a first try, refund friction)
6. Info gaps before trying a new category (price, quality, fit unknown)
7. Which user segments explore more, and why
8. Unmet needs that block category exploration

Operational, pricing, quality, refund, delivery, and trust complaints ARE eligible when they \
plausibly make a user stick to familiar categories or avoid trying new ones - INFER this \
connection; do NOT require the snippet to literally say "category" or "explore".

Mark eligible=false ONLY when there is no plausible link to category-exploration behavior at \
all: pure app-crash/bug reports, login/OTP issues, payment-gateway errors, or generic praise \
with no friction implied ("great app", "nice", "good"). When genuinely in doubt, mark \
eligible=true and state the link. Give ineligible clusters a short `reason` (<= 12 words).

STEP 2 - FOR ELIGIBLE CLUSTERS:
- title: a 4-8 word problem statement that names BOTH the friction AND its effect on trying / \
exploring new categories (or sticking to habits). The exploration consequence MUST be in the \
title, not only in the connection field. Map the operational cause to the exploration effect:
  "late delivery"          -> "Late deliveries push users back to staples"
  "high delivery fees"     -> "Delivery fees make users skip new-category trials"
  "missing / wrong items"  -> "Missing items kill trust in unfamiliar products"
  "poor customer support"  -> "Weak support after failures deters experimentation"
  "stockouts"              -> "One stockout stops users trying new categories"
  GOOD: "Users buy on habit, never open browse"; "Price uncertainty stops new-category trials".
  BAD (name ONLY the operational problem - never output these): "High delivery charges and fees", \
"Late delivery and inconsistent times", "Customer service is unhelpful", "Order issues and \
missing items", "Blinkit", "Good app".
- Each title MUST be DISTINCT; never reuse the same title for two clusters.
- connection: ONE full explanatory sentence, written by you, in the form \
"Because <this cluster's friction>, users <avoid trying new categories / fall back on familiar \
staples>." It must be your own words - NEVER a quote, snippet fragment, or copy of the title. \
Example: "Because refunds are hard to get after a bad item, users avoid risking money on \
unfamiliar products."
- question_ids: choose the 1-3 IDs the snippets MOST directly support. NEVER select more than 3, \
and never select all 8 - be selective.
- severity: high | medium | low, judged from BOTH the average rating and the language intensity.
- Do not invent facts beyond the snippets.

Respond with strict JSON only, no other text:
{{"clusters": [{{"cluster_id": "<id>", "eligible": true, "question_ids": [<1-8>], \
"title": "<4-8 word problem statement>", "connection": "<one explicit sentence>", \
"severity": "high|medium|low", "reason": ""}}, ...]}}

Clusters:
{clusters}"""


def _count_total_reviews(config: Config) -> int:
    total = 0
    with config.paths.reviews.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                total += 1
    return total


def _build_candidates(config: Config, threshold: int) -> Tuple[List[dict], List[dict]]:
    """Return (candidates, below_threshold).

    A cluster is a *growth-relevant candidate* purely on support here - the actual
    relevance judgement (full-funnel link to category exploration) is made by the
    LLM in STEP 1. review_count is the full distinct-review count of the cluster,
    since the whole cluster is what a PM would act on once it's judged relevant.
    below_threshold is kept only so short/zero results can name the near-misses.
    """
    themes_doc = read_json(config.paths.themes)
    communities_by_id = {c["community_id"]: c for c in read_json(config.paths.communities)["communities"]}
    unit_by_id: Dict[str, Unit] = {u.unit_id: u for u in read_jsonl(config.paths.units, factory=Unit)}

    candidates: List[dict] = []
    below_threshold: List[dict] = []
    for theme in themes_doc["themes"]:
        community = communities_by_id.get(theme["community_id"], {})
        unit_ids = community.get("unit_ids", [])

        review_ids = set()
        snippets: List[str] = []
        seen_reviews_for_snippet = set()
        for uid in unit_ids:
            u = unit_by_id.get(uid)
            if u is None:
                continue
            review_ids.add(u.review_id)
            if (
                len(snippets) < MAX_SNIPPETS_PER_THEME
                and len(u.text.split()) >= MIN_SNIPPET_WORDS
                and u.review_id not in seen_reviews_for_snippet
            ):
                seen_reviews_for_snippet.add(u.review_id)
                snippets.append(u.text.strip())

        review_count = len(review_ids)
        record = {
            "theme_id": theme["theme_id"],
            "label": theme["label"],
            "keywords": [k.strip() for k in theme["label"].split("/") if k.strip()],
            # Centroid quotes kept only as a last-resort fallback after claim-aware
            # selection fails; never used as the primary pick.
            "fallback_quotes": theme.get("representative_quotes", [])[:QUOTES_PER_THEME],
            "snippets": snippets[:MAX_SNIPPETS_PER_THEME] or theme.get("representative_quotes", [])[:MAX_SNIPPETS_PER_THEME],
            "review_count": review_count,
            "avg_rating": community.get("avg_rating"),
            "sentiment": theme.get("sentiment", "neutral"),
            "unit_ids": unit_ids,
            "review_ids": list(review_ids),
        }
        if review_count >= threshold:
            candidates.append(record)
        else:
            below_threshold.append(record)

    candidates.sort(key=lambda c: -c["review_count"])
    below_threshold.sort(key=lambda c: -c["review_count"])
    return candidates, below_threshold


def _specificity_score(text: str) -> float:
    """Prefer longer, concrete-incident reviews over short generic sentiment.

    Used only to RANK a widened sample pool - never as the final pick. The LLM
    (or the fallback) still decides which quotes actually support the claim.
    """
    words = text.split()
    n = len(words)
    if n < MIN_QUOTE_WORDS:
        return -1.0
    if _GENERIC_COMPLAINT_RE.match(text.strip()):
        return 0.1
    score = min(n, 80) / 10.0  # length up to ~80 words
    score += 2.0 * len(_SPECIFICITY_RE.findall(text))
    if any(ch.isdigit() for ch in text):
        score += 1.5
    # Prefer reviews that name a burn AND the trust/avoidance aftermath
    # (e.g. "tried fruit… rotten… never order again") — that is the claim evidence.
    if _TRUST_BREAK_RE.search(text):
        score += 4.0
    return score


def _sample_quote_pool(
    review_ids: List[str],
    review_by_id: Dict[str, Review],
    seed: int,
    pool_size: int = QUOTE_POOL_SIZE,
) -> List[str]:
    """Widen first: 30-50 distinct full reviews, ranked by specificity - NOT centroid."""
    texts: List[str] = []
    seen_norm = set()
    for rid in review_ids:
        r = review_by_id.get(rid)
        if r is None:
            continue
        text = r.text.replace("\r", " ").replace("\n", " ").strip()
        if len(text.split()) < MIN_QUOTE_WORDS:
            continue
        norm = re.sub(r"\s+", " ", text.lower())[:120]
        if norm in seen_norm:
            continue
        seen_norm.add(norm)
        texts.append(text)

    if not texts:
        return []

    # Prefer specific-incident text; break ties with a seeded shuffle so the pool
    # is reproducible but not identical to any centroid ranking.
    rng = random.Random(seed)
    scored = [( _specificity_score(t), rng.random(), t) for t in texts]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [t for _, _, t in scored[:pool_size]]


_QUOTE_PICK_PROMPT = """You are selecting evidence quotes for a Growth PM report.

Theme title: {title}
Claimed mechanism (how it blocks category exploration): {connection}

Below is a numbered pool of REAL review texts from this theme's cluster. Pick up to 10 quote \
numbers that:
1. Name a SPECIFIC incident (what was ordered / what went wrong / what happened next) - \
not a generic complaint like "customer support is not good" or "delivery is late".
2. Are NOT repeats of the same complaint in different words.
3. Genuinely support the CLAIM above - not just the topic. The best quotes show BOTH:
   (a) a concrete burn (ordered X, got Y / missing / rotten / cheated), AND
   (b) the trust aftermath (will never order again / won't try that category or new \
items / afraid of being cheated / stick to familiar only). Prefer these over incident-only \
reviews that never mention lost trust or future avoidance.
   Example for "order issues kill trust in unfamiliar products": a user who tried fruit, \
got rotten bananas, and now feels they should never order again - NOT a delivery-boy \
argument with no trust/exploration link.

If a quote is only topically related but does not support the claim, discard it.
If the pool has no specific-incident quote that supports the claim, fall back to ONE generic \
quote only (and leave the rest empty) - but prefer specific first.

Respond with strict JSON only:
{{"quote_indices": [<1-based numbers from the pool>]}}

Pool:
{pool}"""


def _pick_claim_quotes(
    themes: List[dict],
    candidate_by_id: Dict[str, dict],
    review_by_id: Dict[str, Review],
    config: Config,
    api_key: str,
    limiter: _RateLimiter,
) -> None:
    """Mutate each theme's representative_quotes in place with claim-supporting picks.

    One Groq call per theme (keeps each request under the free-tier TPM cap). The pool
    is widened and specificity-ranked first; the model only sees the top QUOTE_SHOW_SIZE.
    """
    for t in themes:
        c = candidate_by_id.get(t["theme_id"], {})
        full_pool = _sample_quote_pool(
            c.get("review_ids", []),
            review_by_id,
            config.seed + sum(ord(ch) for ch in t["theme_id"]),
        )
        if not full_pool:
            t["representative_quotes"] = list(c.get("fallback_quotes", []))[:QUOTES_PER_THEME]
            continue

        show = full_pool[:QUOTE_SHOW_SIZE]
        pool_lines = []
        for i, text in enumerate(show, start=1):
            clipped = text if len(text) <= MAX_QUOTE_CHARS else text[: MAX_QUOTE_CHARS - 1].rstrip() + "…"
            pool_lines.append(f'{i}. "{clipped}"')

        prompt = _QUOTE_PICK_PROMPT.format(
            title=t["title"],
            connection=t["connection"],
            pool="\n".join(pool_lines),
        )
        result = _call_groq(
            prompt, config.llm_synthesis.model, api_key, config.llm_synthesis.temperature,
            config.seed, config.llm_synthesis.max_retries, limiter,
        )

        idxs: List[int] = []
        if result and isinstance(result.get("quote_indices"), list):
            for i in result["quote_indices"]:
                try:
                    idxs.append(int(i))
                except (TypeError, ValueError):
                    continue

        chosen: List[str] = []
        seen_norm = set()
        for idx in idxs:
            if not (1 <= idx <= len(show)):
                continue
            text = show[idx - 1].strip()
            norm = re.sub(r"\s+", " ", text.lower())[:100]
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            chosen.append(text)
            if len(chosen) >= QUOTES_PER_THEME:
                break

        if not chosen:
            # Widen-first already happened; still nothing claim-usable -> one best
            # specific-looking quote from the full pool, else the old centroid fallback.
            specific = [
                p for p in full_pool
                if _specificity_score(p) >= 3.0 and not _GENERIC_COMPLAINT_RE.match(p)
            ]
            if specific:
                chosen = [specific[0]]
            else:
                chosen = list(c.get("fallback_quotes", full_pool[:1]))[:1]
            logger.warning("[%s] no claim-supporting quote from LLM; used fallback.", t["theme_id"])

        t["representative_quotes"] = chosen[:QUOTES_PER_THEME]
        logger.info(
            "  quotes[%s]: %d picked from widened pool of %d (showed %d)",
            t["theme_id"], len(t["representative_quotes"]), len(full_pool), len(show),
        )


def _build_prompt(candidates: List[dict]) -> str:
    blocks = []
    for c in candidates:
        rating = c["avg_rating"]
        lines = [
            f"[CLUSTER {c['theme_id']}] avg_rating={rating if rating is not None else 'n/a'}; "
            f"keywords: {', '.join(c['keywords'])}",
            "snippets:",
        ]
        lines += [f'- "{s[:MAX_SNIPPET_CHARS]}"' for s in c["snippets"]]
        blocks.append("\n".join(lines))
    return _PROMPT_TEMPLATE.format(clusters="\n\n".join(blocks))


def run_theme_titles(config: Config, min_share_pct: float = 1.0, refresh: bool = False) -> None:
    ensure_data_dir(config)
    apply_global_seed(config.seed)

    out_path = config.paths.data_dir / "theme_titles.json"
    if out_path.exists() and not refresh:
        logger.info("'%s' already exists; skipping (pass --refresh to rebuild).", out_path)
        return

    for path in (config.paths.units, config.paths.themes, config.paths.communities, config.paths.reviews):
        if not path.exists():
            raise LLMSynthesisError(
                f"Expected artifact at {path} but it does not exist. Run `python -m src.pipeline` first."
            )

    api_key = _load_api_key()

    total_reviews = _count_total_reviews(config)
    threshold = max(int(round((min_share_pct / 100.0) * total_reviews)), MIN_REVIEW_FLOOR)
    logger.info(
        "Support bar: >= %d reviews (max of %.1f%% of %d total, and the %d-review floor).",
        threshold, min_share_pct, total_reviews, MIN_REVIEW_FLOOR,
    )

    candidates, below_threshold = _build_candidates(config, threshold)

    def _share(rc: int) -> float:
        return round(100.0 * rc / total_reviews, 2)

    near_misses: List[dict] = []
    if not candidates:
        logger.warning("No theme clears the support bar - writing an empty result.")
        near_misses = [
            {"theme_id": c["theme_id"], "label": c["label"], "review_count": c["review_count"],
             "share_pct": _share(c["review_count"]), "reason": f"below support threshold ({c['review_count']} < {threshold})"}
            for c in below_threshold[:MAX_NEAR_MISSES]
        ]
        write_json(out_path, {
            "method": "llm_groq", "model": config.llm_synthesis.model,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_reviews": total_reviews, "min_share_pct": min_share_pct,
            "support_threshold_reviews": threshold, "support_qualified_cluster_count": 0,
            "relevance_rejected_cluster_count": 0, "themes": [], "near_misses": near_misses,
        })
        return

    logger.info(
        "%d themes clear the support bar; relevance-screening all of them via %s before ranking.",
        len(candidates), config.llm_synthesis.model,
    )

    prompt = _build_prompt(candidates)
    limiter = _RateLimiter(config.llm_synthesis.requests_per_minute)
    result = _call_groq(
        prompt, config.llm_synthesis.model, api_key, config.llm_synthesis.temperature,
        config.seed, config.llm_synthesis.max_retries, limiter,
    )
    if not result or not isinstance(result.get("clusters"), list):
        raise LLMSynthesisError("Groq returned no usable relevance decisions; try again (see logs above).")

    decision_by_id: Dict[str, dict] = {}
    for entry in result["clusters"]:
        try:
            cid = str(entry["cluster_id"]).strip()
            if cid:
                decision_by_id[cid] = entry
        except (KeyError, TypeError):
            continue

    eligible: List[dict] = []
    rejected: List[dict] = []
    for c in candidates:
        decision = decision_by_id.get(c["theme_id"], {})
        title = str(decision.get("title", "")).strip()
        connection = str(decision.get("connection", "")).strip()
        raw_qids = decision.get("question_ids", [])
        question_ids: List[int] = []  # keep the model's relevance order, cap at 3
        for q in raw_qids:
            if str(q).isdigit() and 1 <= int(q) <= 8 and int(q) not in question_ids:
                question_ids.append(int(q))
        question_ids = question_ids[:3]
        reason = str(decision.get("reason", "")).strip()

        valid_title = 4 <= len(title.split()) <= 8
        if not decision.get("eligible") or not valid_title or not connection or not question_ids:
            rejected.append(
                {
                    "theme_id": c["theme_id"], "label": c["label"], "review_count": c["review_count"],
                    "share_pct": _share(c["review_count"]),
                    "reason": reason or ("no category-exploration link" if not decision.get("eligible")
                                         else "incomplete/invalid title, connection, or question mapping"),
                }
            )
            continue

        severity = str(decision.get("severity", "")).lower()
        if severity not in {"high", "medium", "low"}:
            severity = "medium"
        eligible.append(
            {
                "theme_id": c["theme_id"],
                "title": title,
                "connection": connection,
                "review_count": c["review_count"],
                "share_pct": _share(c["review_count"]),
                "severity": severity,
                "avg_rating": c["avg_rating"],
                "sentiment": c["sentiment"],
                "question_ids": question_ids,
                "research_questions": [RESEARCH_QUESTIONS[q] for q in question_ids],
                # Filled after top-5 selection by claim-aware quote picker (not centroid).
                "representative_quotes": [],
            }
        )

    eligible.sort(key=lambda t: -t["review_count"])

    # Take the top 5 by review_count, but keep titles DISTINCT: an LLM sometimes gives
    # two clusters the same phrasing, and a PM report should not show duplicates.
    themes_out: List[dict] = []
    not_selected: List[dict] = []
    used_titles = set()
    for e in eligible:
        key = e["title"].strip().lower()
        if len(themes_out) >= TOP_N:
            not_selected.append({**e, "_reason": "eligible but outside the top 5 by review_count"})
        elif key in used_titles:
            not_selected.append({**e, "_reason": "eligible but a duplicate of a higher-ranked title"})
        else:
            used_titles.add(key)
            themes_out.append(e)

    # Claim-aware snippet selection for the final top 5: widen each cluster to a
    # 30-50 review pool (specificity-ranked, NOT embedding-centroid), then ask the
    # LLM to pick 2-3 quotes that name a specific incident AND support the claim.
    if themes_out:
        logger.info("Selecting claim-supporting quotes for %d themes (pool=%d each)...", len(themes_out), QUOTE_POOL_SIZE)
        candidate_by_id = {c["theme_id"]: c for c in candidates}
        # Only load the reviews we need for the final themes (keeps this step light).
        needed_rids = set()
        for t in themes_out:
            needed_rids.update(candidate_by_id.get(t["theme_id"], {}).get("review_ids", []))
        review_by_id: Dict[str, Review] = {}
        for r in read_jsonl(config.paths.reviews, factory=Review):
            if r.id in needed_rids:
                review_by_id[r.id] = r
        _pick_claim_quotes(themes_out, candidate_by_id, review_by_id, config, api_key, limiter)

    # Near-misses: everything that did NOT make the top 5, most-supported first, so a
    # short result is inspectable (STEP 5) rather than a silent drop.
    near_misses = rejected + [
        {"theme_id": e["theme_id"], "label": None, "review_count": e["review_count"],
         "share_pct": e["share_pct"], "reason": e["_reason"]}
        for e in not_selected
    ]
    if len(themes_out) < TOP_N:
        near_misses += [
            {"theme_id": c["theme_id"], "label": c["label"], "review_count": c["review_count"],
             "share_pct": _share(c["review_count"]), "reason": f"below support threshold ({c['review_count']} < {threshold})"}
            for c in below_threshold
        ]
    near_misses.sort(key=lambda m: -m["review_count"])
    near_misses = near_misses[:MAX_NEAR_MISSES]

    payload = {
        "method": "llm_groq",
        "model": config.llm_synthesis.model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_reviews": total_reviews,
        "min_share_pct": min_share_pct,
        "support_threshold_reviews": threshold,
        "support_qualified_cluster_count": len(candidates),
        "relevance_rejected_cluster_count": len(rejected),
        "themes": themes_out,
        "near_misses": near_misses,
    }
    write_json(out_path, payload)
    logger.info("Wrote %s: %d growth-relevant themes.", out_path, len(themes_out))
    for t in themes_out:
        logger.info(
            "  [%s] %s (%d reviews, %.1f%%, Q%s, %s)",
            t["theme_id"], t["title"], t["review_count"], t["share_pct"],
            ",".join(map(str, t["question_ids"])), t["severity"],
        )
    if len(themes_out) < TOP_N:
        logger.warning("Only %d themes qualified (< %d). Closest excluded themes:", len(themes_out), TOP_N)
        for m in near_misses:
            logger.warning("  [%s] %s reviews=%d - %s", m["theme_id"], m["label"] or "", m["review_count"], m["reason"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract the top growth-relevant themes as PM problem statements (Groq).")
    parser.add_argument("--config", default=None)
    parser.add_argument("--refresh", action="store_true", help="Rebuild even if data/theme_titles.json exists")
    parser.add_argument("--min-share", type=float, default=1.0, help="Min %% of total reviews a theme must cover (default 1.0)")
    args = parser.parse_args()

    try:
        config = load_config(args.config) if args.config else load_config()
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    try:
        run_theme_titles(config, min_share_pct=args.min_share, refresh=args.refresh)
    except LLMSynthesisError as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
