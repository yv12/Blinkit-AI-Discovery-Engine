"""Quick verification of grounding-rules fix in ask_synthesis.

Calls synthesize_answer for 3 test questions and prints the results.
Uses real theme data from theme_titles.json and representative sample
reviews so the model has realistic evidence to work with.
"""

import json
import sys
import textwrap
from pathlib import Path

# Make sure repo root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ask_synthesis import synthesize_answer


# ---------------------------------------------------------------------------
# Load real theme summaries from theme_titles.json
# ---------------------------------------------------------------------------
THEME_TITLES_PATH = Path("data/theme_titles.json")

def _load_themes(n: int = 3):
    """Return up to *n* theme dicts formatted for synthesis."""
    if not THEME_TITLES_PATH.exists():
        return []
    with THEME_TITLES_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    themes_raw = data.get("themes", data) if isinstance(data, dict) else data
    out = []
    for t in themes_raw[:n]:
        out.append({
            "theme_summary": t.get("title") or t.get("label") or "Untitled",
            "review_count": t.get("review_count", t.get("member_count", 0)),
            "avg_rating": t.get("avg_rating"),
        })
    return out


# ---------------------------------------------------------------------------
# Representative sample reviews (real phrasing, no invented quotes)
# ---------------------------------------------------------------------------
SAMPLE_REVIEWS = [
    {
        "text": "Ordered milk and it was expired. Had to call support multiple times to get refund.",
        "source": "google_play",
        "date": "2025-06-12",
        "rating": 1,
    },
    {
        "text": "Prices keep going up on items I buy every week. Very unhappy with the constant hikes.",
        "source": "google_play",
        "date": "2025-07-01",
        "rating": 2,
    },
    {
        "text": "Wrong item delivered. Customer care was no help at all, just kept saying sorry.",
        "source": "google_play",
        "date": "2025-05-20",
        "rating": 1,
    },
    {
        "text": "Quantity limits are annoying. Can't order more than 2 of the same item.",
        "source": "mouthshut",
        "date": "2025-04-15",
        "rating": 2,
    },
    {
        "text": "App crashes frequently during payment. Lost my order twice this month.",
        "source": "google_play",
        "date": "2025-06-28",
        "rating": 1,
    },
    {
        "text": "Delivery was quick and products were fresh. Happy with the service.",
        "source": "google_play",
        "date": "2025-07-10",
        "rating": 5,
    },
    {
        "text": "Support took 3 days to respond to my complaint about a missing item.",
        "source": "mouthshut",
        "date": "2025-03-22",
        "rating": 1,
    },
    {
        "text": "Good app for daily essentials but won't try anything new because returns are such a hassle.",
        "source": "google_play",
        "date": "2025-05-08",
        "rating": 3,
    },
]


# ---------------------------------------------------------------------------
# 3 test questions
# ---------------------------------------------------------------------------
QUESTIONS = [
    "Why do users repeatedly buy from the same categories?",
    "What role do habits play in shopping behavior?",
    "What frustrations emerge repeatedly?",
]

EXPECTED_BEHAVIORS = [
    "(should acknowledge reviews can't show habit, then report deterrents with stated-vs-inferred separation)",
    "(should lead with the corpus limitation)",
    "(should be largely unchanged - directly answerable from reviews; confirms fix doesn't over-hedge)",
]


def main():
    themes = _load_themes(3)
    print(f"Loaded {len(themes)} theme summaries for context.\n")

    for i, (q, expected) in enumerate(zip(QUESTIONS, EXPECTED_BEHAVIORS), 1):
        sep = "=" * 78
        print(f"\n{sep}")
        print(f"  TEST {i}: {q}")
        print(f"  EXPECTED: {expected}")
        print(sep)

        answer = synthesize_answer(q, themes, SAMPLE_REVIEWS)
        if answer is None:
            print("\n  *** SYNTHESIS RETURNED None (API failure) ***\n")
        else:
            wrapped = textwrap.fill(answer, width=78, initial_indent="  ", subsequent_indent="  ")
            print(f"\n{wrapped}\n")


if __name__ == "__main__":
    main()
