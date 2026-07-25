"""Stage 11 (optional) - Deep pattern synthesis via a remote LLM (Groq).

**This stage is intentionally NOT part of `python -m src.pipeline`'s S1-S9 and
is not required to satisfy the project's core Definition of Done** - it is an
explicit, user-directed addition on top of the already-complete pipeline.

Why this exists: Stage 7 (`summarize.py`) labels communities with literal
TF-IDF top terms, and Stage 8 (`insights.py`) maps themes to the 8 research
questions via embedding similarity between a theme's label text and a topic
description. Both methods can only ever surface *literal recurring
vocabulary* - they cannot infer an abstract behavioral explanation (e.g.
"users feel choice-overload when browsing unfamiliar categories") that is
never phrased that way in any single review but is implied across many
differently-worded ones. That is exactly the gap this stage targets: give a
capable LLM a curated sample of raw review excerpts and explicitly ask it to
name the *indirect/implicit* pattern behind each one, per research question.

Design choices, decided with the user rather than left as defaults:

1. **Remote API, not local Ollama** (`Groq`, OpenAI-compatible endpoint,
   `https://api.groq.com/openai/v1/chat/completions`). This is a deliberate,
   explicit deviation from the "fully local" principle in architecture.md
   §1 - zero-cost is preserved (Groq's free tier has no per-token charge,
   only rate limits: see README.md §12), but this stage requires network
   access and a `GROQ_API_KEY` in a project-root `.env` file (gitignored,
   never committed - loaded via `python-dotenv`).
2. **Bounded, prioritized sampling, not the full corpus.** Free-tier rate
   limits (~30 RPM, a per-model daily request cap) make running all 89K
   units impractical and unnecessary. The candidate pool prioritizes units
   already flagged "uncategorized" by Stage 8 (the ones the deterministic
   pipeline explicitly could not place against any of the 8 questions) and
   every emerging-signal member unit, then tops up with a seeded
   stratified-by-rating random sample up to `llm_synthesis.sample_size`
   (config.yaml). Sampling is deterministic given the same upstream
   artifacts + `config.seed`.
3. **One call per batch answers all 8 questions at once** (not one call per
   question) - far more token-efficient within the free-tier TPM budget for
   the same reasoning quality.
4. **Every inferred pattern must cite real verbatim quotes / unit ids.** The
   LLM is asked to reason about *why* an excerpt might relate to a question,
   never to invent or paraphrase away from the source text; output patterns
   are always traceable back to the exact review excerpts that produced them
   (same traceability principle as every other stage, edgecases.md
   throughout).
5. **Local aggregation, not LLM aggregation.** Individual per-batch "pattern"
   phrases from the LLM are merged into canonical patterns via local
   embedding similarity (`models.embedding_model`, same model as every other
   stage - no extra API calls, deterministic, free) rather than asking the
   LLM to deduplicate across thousands of excerpts itself.
6. **Resumable via a plain checkpoint file** (`data/llm_synthesis_checkpoint.jsonl`),
   matching `scrape.py`'s resumability philosophy - a long batch run can be
   safely interrupted and resumed without re-spending API quota on completed
   batches. `--refresh` discards the checkpoint and starts over.
7. **Every LLM/network failure degrades to "skip this batch, log a warning,
   keep going"** - never crashes the run. A batch that fails after
   `llm_synthesis.max_retries` retries is recorded as failed and excluded
   from aggregation; the run summary reports how many batches failed.
8. **`temperature: 0` + a fixed `seed`** on every request (README.md §8),
   the same reproducibility mitigation used for the optional Ollama path in
   `summarize.py` - eliminates sampling randomness as a source of run-to-run
   drift on the same model, though (like that path) this does not guarantee
   bit-for-bit identical output across different model/infra versions.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

from src.config import PROJECT_ROOT, Config, apply_global_seed, ensure_data_dir, load_config
from src.schema import Unit, read_json, read_jsonl, write_json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_TIMEOUT_S = 60

# The 8 research questions, phrased as short topic queries (same texts as
# config.yaml's insights.question_queries) - kept here only as a fallback if
# that config section is ever unavailable; the live values always come from
# `config.insights.question_queries` at call time.


class LLMSynthesisError(RuntimeError):
    """Raised for unrecoverable failures (missing API key, missing inputs)."""


def _load_api_key() -> str:
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise LLMSynthesisError(
            "python-dotenv is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    load_dotenv(PROJECT_ROOT / ".env")
    import os

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise LLMSynthesisError(
            "GROQ_API_KEY not found. Create a `.env` file at the project root "
            "(next to config.yaml) with a line: GROQ_API_KEY=gsk_... "
            "(see README.md \u00a712 for how to get a free Groq API key)."
        )
    return api_key


def _load_embed_model(model_name: str):
    """Local embedding model, used only for aggregating LLM-generated pattern
    phrases (design choice #5) - no extra API calls, same pattern as
    embed.py/insights.py."""
    import torch  # noqa: F401  (Windows DLL-order guard, edgecases.md S4-01/R-07)
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def _build_sample_pool(config: Config) -> List[Unit]:
    """Deterministic, prioritized sample of units to send to the LLM (design choice #2)."""
    unit_meta: Dict[str, Unit] = {u.unit_id: u for u in read_jsonl(config.paths.units, factory=Unit)}
    themes_doc = read_json(config.paths.themes)
    insights_doc = read_json(config.paths.insights)
    communities_by_id = {c["community_id"]: c for c in read_json(config.paths.communities)["communities"]}

    theme_community: Dict[str, int] = {t["theme_id"]: t["community_id"] for t in themes_doc["themes"]}
    signal_community: Dict[str, int] = {s["signal_id"]: s["community_id"] for s in themes_doc["emerging_signals"]}

    uncategorized = insights_doc.get("uncategorized", {"theme_ids": [], "signal_ids": []})
    rng = random.Random(config.seed)

    budget = config.llm_synthesis.sample_size
    selected: List[str] = []
    selected_set = set()

    def _add(unit_ids: List[str], cap: Optional[int] = None) -> None:
        pool = [uid for uid in unit_ids if uid in unit_meta and uid not in selected_set]
        if cap is not None and len(pool) > cap:
            pool = rng.sample(pool, cap)
        for uid in pool:
            if len(selected) >= budget:
                return
            selected.append(uid)
            selected_set.add(uid)

    # Priority 1: uncategorized themes - the deterministic pipeline's own explicit gap.
    # Capped per-theme so one giant generic theme (e.g. 8k+ "good app" praise) doesn't
    # consume the whole budget on low-value content.
    uncategorized_theme_ids = uncategorized.get("theme_ids", [])
    per_theme_cap = max(20, (budget // 2) // max(1, len(uncategorized_theme_ids)))
    for tid in uncategorized_theme_ids:
        community = communities_by_id.get(theme_community.get(tid, -1))
        if community:
            _add(community["unit_ids"], cap=per_theme_cap)

    # Priority 2: every emerging-signal member unit (naturally small in aggregate -
    # 819 signals below min_community_size, mostly singletons/pairs).
    for sid in signal_community:
        community = communities_by_id.get(signal_community[sid])
        if community:
            _add(community["unit_ids"])

    # Priority 3: fill remaining budget with a seeded, stratified-by-rating random
    # sample from the full corpus - catches units already matched to another
    # question that might *also* carry indirect signal Stage 8's single-best-match
    # mapping doesn't surface.
    remaining_budget = budget - len(selected)
    if remaining_budget > 0:
        by_rating: Dict[Optional[int], List[str]] = {}
        for uid, u in unit_meta.items():
            if uid in selected_set:
                continue
            by_rating.setdefault(u.rating, []).append(uid)
        ratings = sorted(by_rating.keys(), key=lambda r: (r is None, r))
        per_rating_budget = max(1, remaining_budget // max(1, len(ratings)))
        for r in ratings:
            _add(sorted(by_rating[r]), cap=per_rating_budget)

    logger.info(
        "Sampled %d/%d candidate units for LLM synthesis (budget=%d): "
        "%d from uncategorized themes/signals, remainder stratified by rating.",
        len(selected), len(unit_meta), budget, len(selected),
    )
    return [unit_meta[uid] for uid in selected]


# --------------------------------------------------------------------------- #
# Groq API
# --------------------------------------------------------------------------- #


class _RateLimiter:
    """Simple fixed-interval limiter honoring `llm_synthesis.requests_per_minute` (design choice #2)."""

    def __init__(self, requests_per_minute: int) -> None:
        self._min_interval = 60.0 / max(1, requests_per_minute)
        self._last_call = 0.0

    def wait(self) -> None:
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.time()


def _call_groq(
    prompt: str,
    model: str,
    api_key: str,
    temperature: float,
    seed: int,
    max_retries: int,
    limiter: _RateLimiter,
) -> Optional[dict]:
    """POST a chat completion to Groq; returns parsed JSON content or None on
    unrecoverable failure (design choice #7 - every failure degrades, never raises)."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "seed": seed,
        "response_format": {"type": "json_object"},
    }

    backoff = 2.0
    for attempt in range(max_retries + 1):
        limiter.wait()
        try:
            resp = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=GROQ_TIMEOUT_S)
        except requests.exceptions.RequestException as exc:
            logger.warning("Groq request failed (attempt %d/%d): %s", attempt + 1, max_retries + 1, exc)
            time.sleep(backoff)
            backoff *= 2
            continue

        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", backoff))
            logger.warning("Groq rate-limited (429); sleeping %.1fs (attempt %d/%d).", retry_after, attempt + 1, max_retries + 1)
            time.sleep(retry_after)
            backoff *= 2
            continue

        if resp.status_code >= 500:
            logger.warning("Groq server error %d (attempt %d/%d); retrying.", resp.status_code, attempt + 1, max_retries + 1)
            time.sleep(backoff)
            backoff *= 2
            continue

        if resp.status_code != 200:
            logger.warning("Groq returned HTTP %d: %s. Skipping batch.", resp.status_code, resp.text[:300])
            return None

        try:
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            logger.warning("Groq returned non-parseable response (%s); attempt %d/%d.", exc, attempt + 1, max_retries + 1)
            continue

    logger.warning("Batch failed after %d attempts; skipping (excluded from aggregation).", max_retries + 1)
    return None


_PROMPT_TEMPLATE = """You are a qualitative UX researcher analyzing short excerpts from Google \
Play Store reviews of the Blinkit grocery delivery app. For EACH numbered excerpt below, \
decide whether it relates - directly OR indirectly/implicitly (not just by literal keyword \
overlap) - to any of these 8 behavioral research questions about why users do or don't \
explore new product categories on the app:

1. Repeat purchases from the same familiar categories; brand loyalty; routine reordering habits
2. Barriers and reasons preventing exploration of new product categories
3. How users discover new products: browsing, search, recommendations, banners, notifications
4. Habitual, routine-driven shopping behavior and convenience
5. Information, details, or trust signals needed before trying a new product category
6. Recurring frustrations and complaints: delivery, quality, app experience, customer service
7. User segments more open to trying new things versus sticking to routine
8. Unmet needs, missing features, or product gaps mentioned repeatedly

For each excerpt that plausibly relates to one or more questions, output one entry per \
(excerpt, question) match with a short 3-8 word "pattern" phrase.

CRITICAL RULES for the "pattern" field:
- It must be YOUR OWN specific inference about what drives THIS excerpt, written in your own \
words - never copy or paraphrase the numbered question list above. If your pattern phrase \
resembles any of the 8 lines above, rewrite it to be more specific and concrete.
- It should name a concrete psychological/behavioral driver, not restate the excerpt's topic. \
For a complaint about slow delivery mapped to question 6, a BAD pattern is "delivery \
frustration" (too generic/topic-only); a GOOD pattern is "impatience with inaccurate ETA \
promises eroding trust".
- For question 2 or 5 especially, actively look for INDIRECT signals a keyword search would \
miss - e.g. a user who only ever mentions 2-3 product types might imply routine/no exploration \
even without saying so explicitly; infer this and name it (e.g. "narrow, unstated product \
vocabulary suggesting routine-only usage").
- Do NOT invent facts the excerpt's wording doesn't support.
- If an excerpt doesn't plausibly relate to any question, omit it entirely.

Example of correct output style (for illustration only, not from the real excerpts):
{{"hits": [
  {{"excerpt": 3, "question_id": 6, "pattern": "anger at repeated ETA promise-breaking"}},
  {{"excerpt": 5, "question_id": 2, "pattern": "sticks to known items to avoid delivery mistakes"}}
]}}

Respond with strict JSON only, no other text: \
{{"hits": [{{"excerpt": <int>, "question_id": <int 1-8>, "pattern": "<short specific phrase>"}}, ...]}}

Excerpts:
{excerpts}"""


def _build_prompt(texts: List[str]) -> str:
    excerpts = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    return _PROMPT_TEMPLATE.format(excerpts=excerpts)


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #


def _read_checkpoint(path: Path) -> Dict[int, dict]:
    if not path.exists():
        return {}
    done: Dict[int, dict] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                done[row["batch_index"]] = row
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate a truncated last line from an interrupted run
    return done


def _append_checkpoint(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _cluster_patterns(
    hits: List[dict],
    model,
    threshold: float,
    max_patterns: int,
    max_quotes: int,
    min_support: int,
) -> List[dict]:
    """Merge near-duplicate LLM-generated pattern phrases via local embedding
    similarity (design choice #5) into canonical patterns, ranked by support."""
    if not hits:
        return []

    from collections import Counter

    raw_patterns = [h["pattern"].strip().lower() for h in hits]
    counts = Counter(raw_patterns)
    unique_patterns = list(counts.keys())

    vecs = model.encode(unique_patterns, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    sims = vecs @ vecs.T

    order = sorted(range(len(unique_patterns)), key=lambda i: -counts[unique_patterns[i]])
    cluster_of: Dict[int, int] = {}
    canonical_indices: List[int] = []
    for i in order:
        if i in cluster_of:
            continue
        cluster_id = len(canonical_indices)
        canonical_indices.append(i)
        cluster_of[i] = cluster_id
        for j in order:
            if j in cluster_of:
                continue
            if sims[i, j] >= threshold:
                cluster_of[j] = cluster_id

    clusters: Dict[int, dict] = {}
    for i, pattern in enumerate(unique_patterns):
        cid = cluster_of[i]
        entry = clusters.setdefault(cid, {"support_count": 0, "quotes": [], "unit_ids": [], "labels": Counter()})
        entry["labels"][pattern] += counts[pattern]

    # Attach support/quotes/unit_ids from the original (non-deduped) hits so
    # every occurrence - not just unique phrasings - counts toward support.
    pattern_to_cluster = {unique_patterns[i]: cluster_of[i] for i in range(len(unique_patterns))}
    for h in hits:
        cid = pattern_to_cluster[h["pattern"].strip().lower()]
        entry = clusters[cid]
        entry["support_count"] += 1
        if h["quote"] not in entry["quotes"] and len(entry["quotes"]) < max_quotes:
            entry["quotes"].append(h["quote"])
        if h["unit_id"] not in entry["unit_ids"]:
            entry["unit_ids"].append(h["unit_id"])

    results = []
    for cid, entry in clusters.items():
        if entry["support_count"] < min_support:
            continue
        canonical_label = entry["labels"].most_common(1)[0][0]
        results.append(
            {
                "pattern": canonical_label,
                "support_count": entry["support_count"],
                "example_quotes": entry["quotes"][:max_quotes],
                "example_unit_ids": entry["unit_ids"][:max_quotes],
            }
        )
    results.sort(key=lambda r: -r["support_count"])
    return results[:max_patterns]


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def run_llm_synthesis(config: Config, refresh: bool = False) -> None:
    ensure_data_dir(config)
    apply_global_seed(config.seed)

    if config.paths.llm_insights.exists() and not refresh:
        logger.info(
            "'%s' already exists; skipping LLM synthesis (pass --refresh to rebuild).",
            config.paths.llm_insights,
        )
        return

    for path in (config.paths.units, config.paths.themes, config.paths.insights, config.paths.communities):
        if not path.exists():
            raise LLMSynthesisError(
                f"Expected artifact at {path} but it does not exist. Run `python -m src.pipeline` "
                "first - this stage builds on top of the full S1-S9 pipeline output."
            )

    api_key = _load_api_key()

    if refresh and config.paths.llm_synthesis_checkpoint.exists():
        config.paths.llm_synthesis_checkpoint.unlink()

    units = _build_sample_pool(config)
    if not units:
        raise LLMSynthesisError("Sample pool is empty - nothing to send to the LLM (X-10).")

    batch_size = config.llm_synthesis.batch_size
    batches: List[List[Unit]] = [units[i : i + batch_size] for i in range(0, len(units), batch_size)]

    done = _read_checkpoint(config.paths.llm_synthesis_checkpoint)
    limiter = _RateLimiter(config.llm_synthesis.requests_per_minute)

    logger.info(
        "Running LLM synthesis: %d units in %d batches (model=%s, %d/%d batches already checkpointed).",
        len(units), len(batches), config.llm_synthesis.model, len(done), len(batches),
    )

    num_failed = 0
    for batch_idx, batch in enumerate(batches):
        if batch_idx in done:
            continue
        texts = [u.text for u in batch]
        prompt = _build_prompt(texts)
        result = _call_groq(
            prompt,
            config.llm_synthesis.model,
            api_key,
            config.llm_synthesis.temperature,
            config.seed,
            config.llm_synthesis.max_retries,
            limiter,
        )
        row = {
            "batch_index": batch_idx,
            "unit_ids": [u.unit_id for u in batch],
            "hits": [],
            "failed": result is None,
        }
        if result and isinstance(result.get("hits"), list):
            for hit in result["hits"]:
                try:
                    excerpt_i = int(hit["excerpt"]) - 1
                    qid = int(hit["question_id"])
                    pattern = str(hit["pattern"]).strip()
                    if not (0 <= excerpt_i < len(batch)) or not (1 <= qid <= 8) or not pattern:
                        continue
                    row["hits"].append(
                        {
                            "unit_id": batch[excerpt_i].unit_id,
                            "quote": batch[excerpt_i].text,
                            "question_id": qid,
                            "pattern": pattern,
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue
        else:
            num_failed += 1
        _append_checkpoint(config.paths.llm_synthesis_checkpoint, row)

        if (batch_idx + 1) % 20 == 0 or batch_idx == len(batches) - 1:
            logger.info("Progress: %d/%d batches processed.", batch_idx + 1, len(batches))

    # Re-read the checkpoint fresh (covers both this run's writes and any prior resumed run).
    all_rows = _read_checkpoint(config.paths.llm_synthesis_checkpoint)
    raw_hits: List[dict] = [h for row in all_rows.values() for h in row.get("hits", [])]
    num_failed = sum(1 for row in all_rows.values() if row.get("failed"))

    # Real finding (README.md §12): very short/vague source excerpts (e.g. "sometimes they
    # does") cause the model to fall back to a generic catch-all inference rather than a real
    # specific one, which then falsely clusters into one large, low-signal "pattern". Dropping
    # low-word-count hits here (not at sampling time) keeps this a pure aggregation-time fix -
    # no API calls wasted, no change needed to the checkpoint itself.
    min_words = config.llm_synthesis.min_quote_words
    all_hits = [h for h in raw_hits if len(h["quote"].split()) >= min_words]

    logger.info(
        "LLM synthesis calls complete: %d/%d batches ok, %d failed, %d raw hits (%d after "
        "dropping excerpts under %d words).",
        len(all_rows) - num_failed, len(all_rows), num_failed, len(raw_hits), len(all_hits), min_words,
    )

    embed_model = _load_embed_model(config.models.embedding_model)
    questions_out = []
    for qid in range(1, 9):
        q_hits = [h for h in all_hits if h["question_id"] == qid]
        patterns = _cluster_patterns(
            q_hits,
            embed_model,
            config.llm_synthesis.pattern_cluster_threshold,
            config.llm_synthesis.max_patterns_per_question,
            config.llm_synthesis.max_quotes_per_pattern,
            config.llm_synthesis.min_pattern_support,
        )
        questions_out.append(
            {
                "question_id": qid,
                "raw_hit_count": len(q_hits),
                "patterns": patterns,
            }
        )

    payload = {
        "method": "llm_groq",
        "model": config.llm_synthesis.model,
        "sampled_unit_count": len(units),
        "batches_ok": len(all_rows) - num_failed,
        "batches_failed": num_failed,
        "questions": questions_out,
        "summary": {
            "total_patterns": sum(len(q["patterns"]) for q in questions_out),
            "total_raw_hits": len(all_hits),
        },
    }
    write_json(config.paths.llm_insights, payload)
    logger.info(
        "Wrote %s: %d inferred patterns across 8 questions.",
        config.paths.llm_insights, payload["summary"]["total_patterns"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 11 (optional) - infer indirect behavioral patterns via Groq."
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: project root)")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Discard any existing checkpoint/output and re-run from scratch",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config) if args.config else load_config()
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    try:
        run_llm_synthesis(config, refresh=args.refresh)
    except LLMSynthesisError as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
