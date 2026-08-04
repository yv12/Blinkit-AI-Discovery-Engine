"""Live Ask-tab answer synthesis via Groq (same stack as Stage 11 / theme titles).

Builds an evidence-only context block from retrieved reviews (and optional
theme summaries) and asks the model for one flowing narrative paragraph.
Failures return None so the API can show ``SYNTHESIS_FALLBACK`` — never the
old stats-template answer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from src.config import load_config
from src.llm_synthesis import GROQ_CHAT_URL, GROQ_TIMEOUT_S, LLMSynthesisError, _load_api_key

logger = logging.getLogger(__name__)

# System prompt for Ask synthesis (Growth-PM behavioral lens).
SYSTEM_PROMPT = (
    "You are a product research analyst summarizing Blinkit reviews for a Growth PM. Use "
    "ONLY the evidence given. No outside knowledge, no invented facts. Write ONE flowing "
    "paragraph, 4-6 sentences, natural narrative prose. No bracket citations, no counts, "
    "no ratings, no corpus size in the answer. Refer to evidence like a person: \"one user "
    "complained...\", \"some users feel...\". Lead with the most dominant cause (judge from "
    "THEMES internally), transition through the rest, end with one summary sentence. If "
    "evidence shows symptoms but not root cause, say so plainly. "
    "Never use em dashes or en dashes; use commas or hyphens instead.\n\n"
    "GROUNDING RULES - apply to every answer:\n\n"
    "1. Distinguish explicitly between what reviewers STATE and what you INFER. "
    "Use framing like \"reviews directly mention X\" for stated content, and "
    "\"this plausibly contributes to Y, though reviewers do not state this connection "
    "themselves\" for inferences. Never present an inferred causal link as something "
    "users said.\n\n"
    "2. Only attach the category-exploration framing to a review theme when the reviews "
    "themselves mention exploration, trying new products, or category behavior. Otherwise "
    "report the theme neutrally and mark any link to exploration as an inference.\n\n"
    "3. If a question asks about behavior app reviews cannot fully reveal (habits, routines, "
    "why users repeat-purchase, positive drivers), open the answer by stating what the corpus "
    "CAN and CANNOT show, e.g.: \"Reviews rarely capture habitual behavior - satisfied routine "
    "purchases don't generate reviews. What the corpus does show is the deterrent side: ...\"\n\n"
    "4. Use a neutral, analytical tone. Report complaint themes without adopting reviewers' "
    "emotional framing (no \"frustrating experience\", \"deceptive practices\" as your own voice "
    "- attribute such characterizations to reviewers).\n\n"
    "5. Keep all existing citation behavior unchanged. Every stated claim still needs "
    "supporting review citations; inferences must be labeled as inferences and need no citation."
)

# One-shot example (user question → expected narrative answer).
EXAMPLE_QUESTION = "Why do users repeatedly buy from the same categories?"
EXAMPLE_ANSWER = (
    "Reviews rarely capture the positive, habitual side of repeat purchasing - satisfied "
    "routine buys don't generate reviews, so the corpus mainly shows deterrents rather than "
    "drivers. What reviews directly mention are product-quality failures: some users describe "
    "receiving expired or damaged items, and several report that resolving such issues through "
    "customer support is slow or unhelpful. Pricing complaints also appear, with users noting "
    "unexpected price increases on items they buy regularly. These themes plausibly contribute "
    "to users defaulting to categories where they already know what to expect, though reviewers "
    "do not state this connection themselves. Overall, the review evidence points to trust "
    "erosion from quality and support failures as a likely reinforcer of category stickiness, "
    "but the full picture of repeat-purchase behavior lies outside what app reviews can reveal."
)

_SOURCE_LABELS = {
    "google_play": "Google Play",
    "mouthshut": "Mouthshut",
}

SYNTHESIS_FALLBACK = "Couldn't generate a summary - here are the matching reviews"


def strip_em_dashes(text: str) -> str:
    """Replace em/en dashes so frontend copy never shows — or –."""
    if not text:
        return text
    return (
        text.replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("—", "-")
        .replace("–", "-")
    )


def _platform_label(source: Optional[str]) -> str:
    if not source:
        return "unknown"
    return _SOURCE_LABELS.get(source, source)


def _format_rating(rating: Any) -> str:
    if rating is None:
        return "n/a"
    try:
        return f"{float(rating):.1f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return str(rating)


def _theme_line(theme: dict) -> str:
    summary = (
        theme.get("theme_summary")
        or theme.get("description")
        or theme.get("label")
        or "Untitled theme"
    )
    count = theme.get("review_count", theme.get("member_count"))
    count_s = f"{int(count):,}" if count is not None else "?"
    avg = theme.get("avg_rating")
    avg_s = _format_rating(avg) if avg is not None else "n/a"
    return f"- {summary} - {count_s} reviews, avg {avg_s}★"


def _review_line(index: int, review: dict) -> str:
    text = (review.get("text") or "").replace("\n", " ").strip()
    platform = _platform_label(review.get("source") or review.get("platform"))
    date = review.get("date") or "unknown date"
    rating = _format_rating(review.get("rating"))
    return f'[{index}] "{text}" - {platform}, {date}, {rating}★'


def build_evidence_context(
    question: str,
    themes: List[dict],
    reviews: List[dict],
) -> str:
    """Fill the QUESTION / THEMES / REVIEWS evidence block for the user message.

    Omits the THEMES section entirely when ``themes`` is empty.
    """
    parts = [f"QUESTION: {question.strip()}", ""]

    if themes:
        parts.append("THEMES:")
        parts.extend(_theme_line(t) for t in themes)
        parts.append("")

    parts.append("REVIEWS:")
    if reviews:
        parts.extend(_review_line(i, r) for i, r in enumerate(reviews, start=1))
    else:
        parts.append("(none)")

    parts.append("")
    parts.append("Write the answer using only this evidence. Do not print citation numbers.")
    return "\n".join(parts)


def synthesize_answer(
    question: str,
    themes: List[dict],
    reviews: List[dict],
    *,
    model: Optional[str] = None,
    seed: Optional[int] = None,
    timeout_s: float = GROQ_TIMEOUT_S,
) -> Optional[str]:
    """Call Groq with the fixed system prompt, one-shot example, and evidence context.

    Returns the model's plain-text paragraph, or None on any failure/timeout so
    callers can show ``SYNTHESIS_FALLBACK`` and still display the reviews.
    """
    print(
        f"[ask_synthesis] synthesize_answer CALLED "
        f"question={question[:80]!r} themes={len(themes)} reviews={len(reviews)}",
        flush=True,
    )
    logger.info(
        "synthesize_answer called (themes=%d, reviews=%d, q=%r)",
        len(themes),
        len(reviews),
        question[:120],
    )

    try:
        config = load_config()
    except Exception as exc:
        print(f"[ask_synthesis] load_config failed: {exc!r}", flush=True)
        logger.exception("load_config failed during ask synthesis")
        config = None

    llm_model = model
    if not llm_model:
        llm_model = (
            config.llm_synthesis.model
            if config is not None
            else "llama-3.1-8b-instant"
        )

    call_seed = seed if seed is not None else (config.seed if config is not None else 42)

    try:
        api_key = _load_api_key()
    except LLMSynthesisError as exc:
        print(f"[ask_synthesis] API key error: {exc!r}", flush=True)
        logger.warning("Ask synthesis skipped: %s", exc)
        return None

    user_message = build_evidence_context(question, themes, reviews)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": llm_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"QUESTION: {EXAMPLE_QUESTION}"},
            {"role": "assistant", "content": EXAMPLE_ANSWER},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0,
        "seed": call_seed,
    }

    print(f"[ask_synthesis] calling Groq model={llm_model!r}", flush=True)
    import time
    # Hardcode a higher retry limit (8) for live Ask synthesis instead of using the
    # offline pipeline's config (which is tuned for batching). 8 retries gives an
    # exponential backoff of ~120 seconds, fully overcoming 1-minute rate limits.
    max_retries = 8

    resp = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=timeout_s)
        except requests.exceptions.Timeout as exc:
            if attempt == max_retries - 1:
                print(f"[ask_synthesis] TIMEOUT after {timeout_s}s: {exc!r}", flush=True)
                logger.warning("Groq ask synthesis timed out after %ss", timeout_s)
                return None
            time.sleep(2 ** attempt)
            continue
        except requests.exceptions.RequestException as exc:
            if attempt == max_retries - 1:
                print(f"[ask_synthesis] REQUEST EXCEPTION: {exc!r}", flush=True)
                logger.warning("Groq ask synthesis request failed: %s", exc, exc_info=True)
                return None
            time.sleep(2 ** attempt)
            continue
        except Exception as exc:
            print(f"[ask_synthesis] UNEXPECTED EXCEPTION: {exc!r}", flush=True)
            logger.exception("Unexpected error during Groq ask synthesis")
            return None

        if resp.status_code == 200:
            break

        if resp.status_code in {429, 500, 502, 503, 504}:
            if attempt == max_retries - 1:
                print(f"[ask_synthesis] HTTP {resp.status_code}: {resp.text[:500]}", flush=True)
                logger.warning("Groq ask synthesis HTTP %d: %s", resp.status_code, resp.text[:300])
                return None
            time.sleep(2 ** attempt)
            continue

        # Non-retryable error
        print(f"[ask_synthesis] HTTP {resp.status_code}: {resp.text[:500]}", flush=True)
        logger.warning("Groq ask synthesis HTTP %d: %s", resp.status_code, resp.text[:300])
        return None

    try:
        text = (resp.json()["choices"][0]["message"]["content"] or "").strip()
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        print(f"[ask_synthesis] parse error: {exc!r} body={resp.text[:500]!r}", flush=True)
        logger.warning("Groq ask synthesis response unusable: %s", exc)
        return None

    if not text:
        print("[ask_synthesis] empty content from Groq", flush=True)
        logger.warning("Groq ask synthesis returned empty content")
        return None

    print(f"[ask_synthesis] OK chars={len(text)} preview={text[:100]!r}", flush=True)
    logger.info("synthesize_answer succeeded (%d chars)", len(text))
    return strip_em_dashes(text)
