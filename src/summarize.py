"""Stage 7 - Summarize communities into labeled themes (+ a long-tail signals pass).

See architecture.md §4 (Stage 7) and edgecases.md "Stage 7 - LLM Summarization"
for the full edge-case catalog (S7-xx IDs referenced below).

Design choices, made explicitly (decided with the user, not left as defaults):

1. **No-LLM extractive fallback is the primary path, not a secondary one**
   (`summarize.use_llm: false` by default). Ollama was verified unreachable
   on this machine (S7-01), so this is also what actually runs here, not
   just a theoretical fallback. Labels come from top TF-IDF terms per
   community; descriptions are templated from community stats; sentiment
   always comes from the members' rating distribution, never from the LLM
   (S7-06 - ratings are ground truth, an LLM sentiment guess would only ever
   be a redundant cross-check, so we skip asking for one at all). Setting
   `summarize.use_llm: true` with Ollama running locally additionally tries
   an LLM-generated label/description per theme, but degrades silently back
   to the extractive result per-community on any failure (S7-01/S7-03).
2. **Below-`min_community_size` communities never go through full theme
   summarization.** Running the same per-community pipeline (representative
   selection, TF-IDF, optional LLM call) on 800+ singleton/pair communities
   would be wasteful and produce noise, not signal. Instead they are
   processed in one dedicated, single pass (`_summarize_long_tail`) and
   surfaced as **`emerging_signals`** - plain dicts, deliberately *not*
   `Theme` records - each tagged with `support_count` (= community size) and
   a `confidence` tier (`"very_low"` for singletons, `"low"` for pairs,
   since `min_community_size` currently excludes anything size >= 3). They
   live in a separate top-level key in `themes.json`, not mixed into
   `themes`, and are not expected to flow through Stage 8's per-question
   mapping the same way (that stage may choose to treat them as
   supplementary evidence only).
3. **Representative selection (S7-04):** for each community, units are
   ranked by cosine similarity to the community's centroid embedding
   (mean vector) - a cheap O(n) proxy for medoid selection that avoids an
   O(n^2) pairwise computation on communities up to ~8,600 members. The top
   `summarize.max_representatives` form the candidate pool that TF-IDF terms
   and quotes are drawn from (`summarize.max_quotes`); an LLM prompt, if
   used, only ever sees this same pool - never the full community - keeping
   quotes and prompts bounded and every quote verbatim-real by construction
   (S7-05: we select existing member text, we never ask an LLM to *write* a
   quote).
4. **`Theme.questions` is left empty (`[]`) here.** Mapping themes to the 8
   research questions is Stage 8's (`insights.py`) job per architecture.md;
   Stage 7 only produces the theme itself.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

from src.config import Config, apply_global_seed, ensure_data_dir, load_config
from src.schema import SchemaError, Theme, Unit, read_json, read_jsonl, write_json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT_S = 60

_WORD_RE = re.compile(r"[a-zA-Z]{3,}")


class SummarizeError(RuntimeError):
    """Raised for unrecoverable summarization failures (e.g. missing inputs)."""


def _sentiment_from_rating(avg_rating: Optional[float]) -> str:
    """Rating-derived sentiment is authoritative; never overridden by an LLM (S7-06)."""
    if avg_rating is None:
        return "neutral"
    if avg_rating < 2.5:
        return "negative"
    if avg_rating > 3.5:
        return "positive"
    return "neutral"


def _truncate(text: str, max_chars: int = 300) -> str:
    text = text.strip()
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def _representative_unit_ids(
    unit_ids: List[str],
    id_to_row: Dict[str, int],
    embeddings: np.ndarray,
    top_n: int,
) -> List[str]:
    """Rank members by cosine similarity to the community centroid (S7-04)."""
    valid_ids = [uid for uid in unit_ids if uid in id_to_row]
    if not valid_ids:
        return []
    vectors = embeddings[[id_to_row[uid] for uid in valid_ids]]
    centroid = vectors.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 0:
        centroid = centroid / centroid_norm
    scores = vectors @ centroid
    order = np.argsort(-scores)
    ranked = [valid_ids[i] for i in order]
    return ranked[:top_n]


def _tfidf_top_terms(texts: List[str], max_terms: int) -> List[str]:
    """Top terms by aggregate TF-IDF score across a community (S7-01 fallback label)."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(stop_words="english", max_features=200, token_pattern=r"[a-zA-Z]{3,}")
        matrix = vectorizer.fit_transform(texts)
        scores = np.asarray(matrix.sum(axis=0)).ravel()
        terms = vectorizer.get_feature_names_out()
        order = np.argsort(-scores)
        return [terms[i] for i in order[:max_terms] if scores[i] > 0]
    except ValueError:
        # Degenerate corpus (e.g. only stopwords survive) - fall back to raw word frequency.
        counts = Counter()
        for text in texts:
            counts.update(w.lower() for w in _WORD_RE.findall(text))
        return [w for w, _ in counts.most_common(max_terms)]


def _call_ollama(prompt: str, model: str, seed: int, retries: int = 1) -> Optional[dict]:
    """POST to a local Ollama server; returns parsed JSON or None on any failure.

    Every failure mode (connection refused, timeout, non-JSON response) degrades
    to None rather than raising - callers fall back to the extractive result
    per-community (S7-01/S7-03). Retries once on a parse failure only, per S7-03.

    ``temperature: 0`` + a fixed ``seed`` (from `config.seed`, threaded through by
    every caller) are set on every request - the reproducibility mitigation for
    the pipeline's one genuinely non-deterministic step, per architecture.md §7
    ("LLM steps ... mitigated with low temperature + cached outputs"). This does
    not make Ollama outputs bit-for-bit guaranteed identical across model
    versions/hardware, but eliminates sampling randomness as a source of
    run-to-run drift on a fixed model + fixed machine (Phase 7 DoD).
    """
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0, "seed": seed},
                },
                timeout=OLLAMA_TIMEOUT_S,
            )
            resp.raise_for_status()
            return json.loads(resp.json()["response"])
        except requests.exceptions.RequestException as exc:
            logger.warning("Ollama request failed (%s); using extractive fallback.", exc)
            return None
        except (KeyError, json.JSONDecodeError) as exc:
            if attempt < retries:
                continue
            logger.warning("Ollama returned non-parseable JSON (%s); using extractive fallback.", exc)
            return None
    return None


def _theme_via_llm(model: str, rep_texts: List[str], label_hint: str, seed: int) -> Optional[Tuple[str, str]]:
    prompt = (
        "You are labeling a cluster of Google Play Store reviews for the Blinkit grocery app. "
        "Given ONLY the representative review excerpts below, respond with strict JSON of the form "
        '{"label": "a concise 3-8 word theme label", "description": "one sentence summarizing the '
        'shared complaint or feedback, based only on the excerpts"}. Do not invent details not '
        f"present in the excerpts. Candidate keywords (for context only): {label_hint}.\n\n"
        + "\n".join(f"- {t}" for t in rep_texts)
    )
    result = _call_ollama(prompt, model, seed)
    if not result or "label" not in result or "description" not in result:
        return None
    return str(result["label"]).strip(), str(result["description"]).strip()


def _build_theme(
    community: dict,
    unit_meta: Dict[str, Unit],
    id_to_row: Dict[str, int],
    embeddings: np.ndarray,
    config: Config,
) -> Theme:
    community_id = community["community_id"]
    unit_ids = community["unit_ids"]
    texts = [unit_meta[uid].text for uid in unit_ids if uid in unit_meta]

    rep_ids = _representative_unit_ids(unit_ids, id_to_row, embeddings, config.summarize.max_representatives)
    rep_texts = [unit_meta[uid].text for uid in rep_ids if uid in unit_meta]

    top_terms = _tfidf_top_terms(texts, config.summarize.max_tfidf_terms)
    label = " / ".join(top_terms) if top_terms else f"Community {community_id}"
    description = (
        f"{community['size']} reviews (avg rating "
        f"{community['avg_rating']:.1f})" if community["avg_rating"] is not None
        else f"{community['size']} reviews"
    ) + (f" recurring around: {', '.join(top_terms)}." if top_terms else ".")

    if config.summarize.use_llm:
        llm_result = _theme_via_llm(config.models.llm_model, rep_texts, ", ".join(top_terms), config.seed)
        if llm_result:
            label, description = llm_result

    quotes = [_truncate(t) for t in rep_texts[: config.summarize.max_quotes]]

    return Theme(
        theme_id=f"theme-{community_id:04d}",
        community_id=community_id,
        label=label,
        description=description,
        representative_quotes=quotes,
        member_count=community["size"],
        sentiment=_sentiment_from_rating(community["avg_rating"]),
        questions=[],  # Stage 8's responsibility (architecture.md §4)
    )


def _confidence_tier(support_count: int, min_community_size: int) -> str:
    if support_count <= 1:
        return "very_low"
    if support_count < min_community_size:
        return "low"
    return "medium"  # should not occur for long-tail input, kept for safety


def _summarize_long_tail(
    communities: List[dict],
    unit_meta: Dict[str, Unit],
    config: Config,
) -> List[dict]:
    """One dedicated, single pass over every below-min-size community (design choice #2)."""
    signals: List[dict] = []
    for community in communities:
        unit_ids = community["unit_ids"]
        texts = [unit_meta[uid].text for uid in unit_ids if uid in unit_meta]
        if not texts:
            continue
        longest = max(texts, key=len)
        signals.append(
            {
                "signal_id": f"signal-{community['community_id']:04d}",
                "community_id": community["community_id"],
                "label": _truncate(longest, max_chars=80),
                "description": (
                    f"{community['size']} review(s), avg rating "
                    f"{community['avg_rating']:.1f}" if community["avg_rating"] is not None
                    else f"{community['size']} review(s), rating unknown"
                ),
                "representative_quotes": [_truncate(t) for t in texts[:2]],
                "support_count": community["size"],
                "confidence": _confidence_tier(community["size"], config.clustering.min_community_size),
                "avg_rating": community["avg_rating"],
                "lang_counts": community["lang_counts"],
            }
        )

    if config.summarize.use_llm and signals:
        _label_long_tail_via_llm(signals, config)

    return signals


def _label_long_tail_via_llm(signals: List[dict], config: Config) -> None:
    """Optional: relabel long-tail signals via a *batched* LLM pass (design choice #2).

    One LLM call per batch of `summarize.long_tail_llm_batch_size` signals, not
    one call per signal - the whole point of batching the long tail. Any batch
    that fails to parse simply keeps its extractive labels (S7-01/S7-03).
    """
    batch_size = config.summarize.long_tail_llm_batch_size
    for start in range(0, len(signals), batch_size):
        batch = signals[start : start + batch_size]
        listing = "\n".join(f"{i + 1}. {s['label']}" for i, s in enumerate(batch))
        prompt = (
            "Each numbered line below is an excerpt from a single low-volume Google Play Store "
            "review of the Blinkit grocery app. Respond with strict JSON: a list of short 3-6 word "
            'labels, one per line, in order, as {"labels": ["...", ...]}. Do not add commentary.\n\n'
            + listing
        )
        result = _call_ollama(prompt, config.models.llm_model, config.seed)
        labels = result.get("labels") if result else None
        if not labels or len(labels) != len(batch):
            logger.warning(
                "Long-tail LLM batch [%d:%d] failed or returned wrong count; keeping extractive labels.",
                start, start + len(batch),
            )
            continue
        for signal, label in zip(batch, labels):
            signal["label"] = str(label).strip()


def summarize_themes(config: Config, refresh: bool = False) -> None:
    ensure_data_dir(config)
    apply_global_seed(config.seed)

    if config.paths.themes.exists() and not refresh:
        logger.info(
            "'%s' already exists; skipping summarization (pass --refresh to rebuild).",
            config.paths.themes,
        )
        return

    for path in (config.paths.communities, config.paths.units, config.paths.embeddings, config.paths.unit_index):
        if not path.exists():
            raise SummarizeError(
                f"Expected artifact at {path} but it does not exist. Run earlier pipeline stages "
                "first (edgecases.md X-03)."
            )

    communities_doc = read_json(config.paths.communities)
    communities: List[dict] = communities_doc["communities"]
    if not communities:
        raise SummarizeError("communities.json has zero communities - nothing to summarize (X-10).")

    unit_meta: Dict[str, Unit] = {u.unit_id: u for u in read_jsonl(config.paths.units, factory=Unit)}
    embeddings = np.load(config.paths.embeddings)
    index = read_json(config.paths.unit_index)
    id_to_row = {uid: i for i, uid in enumerate(index["unit_ids"])}

    if config.summarize.use_llm:
        logger.info("summarize.use_llm=true - will attempt Ollama ('%s') per theme, with fallback.", config.models.llm_model)
    else:
        logger.info("summarize.use_llm=false - using local TF-IDF/rating-derived summarization only.")

    qualifying = [c for c in communities if not c["below_min_size"]]
    long_tail = [c for c in communities if c["below_min_size"]]

    themes: List[Theme] = []
    schema_rejected = 0
    for community in qualifying:
        try:
            themes.append(_build_theme(community, unit_meta, id_to_row, embeddings, config))
        except SchemaError as exc:
            schema_rejected += 1
            logger.warning("Rejected theme for community %d: %s", community["community_id"], exc)

    if not themes:
        raise SummarizeError(
            "Zero themes produced from qualifying communities (X-10). Check clustering output."
        )

    emerging_signals = _summarize_long_tail(long_tail, unit_meta, config)

    payload = {
        "themes": [
            {
                "theme_id": t.theme_id,
                "community_id": t.community_id,
                "label": t.label,
                "description": t.description,
                "representative_quotes": t.representative_quotes,
                "member_count": t.member_count,
                "sentiment": t.sentiment,
                "questions": t.questions,
            }
            for t in themes
        ],
        "emerging_signals": emerging_signals,
        "summary": {
            "num_themes": len(themes),
            "num_emerging_signals": len(emerging_signals),
            "schema_rejected": schema_rejected,
            "use_llm": config.summarize.use_llm,
        },
    }
    write_json(config.paths.themes, payload)
    _log_summary(themes, emerging_signals, schema_rejected)


def _log_summary(themes: List[Theme], emerging_signals: List[dict], schema_rejected: int) -> None:
    logger.info(
        "Summarization complete: %d themes, %d emerging signals (long tail).",
        len(themes), len(emerging_signals),
    )
    if schema_rejected:
        logger.info("  schema_rejected: %d", schema_rejected)
    sentiment_counts = Counter(t.sentiment for t in themes)
    for sentiment, count in sentiment_counts.most_common():
        logger.info("  theme sentiment=%s: %d", sentiment, count)
    for t in sorted(themes, key=lambda t: -t.member_count)[:5]:
        logger.info(
            "  top theme: %s (%s) members=%d sentiment=%s",
            t.theme_id, t.label, t.member_count, t.sentiment,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize communities into labeled themes + long-tail signals (Stage 7)."
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: project root)")
    parser.add_argument(
        "--refresh", action="store_true", help="Rebuild themes.json even if it already exists"
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config) if args.config else load_config()
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    try:
        summarize_themes(config, refresh=args.refresh)
    except SummarizeError as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
