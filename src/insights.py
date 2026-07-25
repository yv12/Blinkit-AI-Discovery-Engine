"""Stage 8 - Map themes (and long-tail signals) to the 8 research questions.

See architecture.md §4 (Stage 8) and edgecases.md "Stage 8 - Insight Mapping"
for the full edge-case catalog (S8-xx IDs referenced below).

Design choices, decided with the user rather than left as defaults:

1. **Theme creation stays fully bottom-up.** The 8 research questions are
   never an input to clustering (Stage 6) or summarization (Stage 7) - this
   stage is a strictly post-hoc tagging pass over the already-formed
   `themes.json`, not the other way around.
2. **Mapping method is embedding similarity, not keywords or an LLM call.**
   Each theme's/signal's own `label + description` text is embedded with
   the same local model as Stage 4 (`models.embedding_model` - no new
   dependency, fully offline, deterministic) and compared via cosine
   similarity against a short topic description per research question
   (`insights.question_queries`, config.yaml). This keeps the zero-cost/
   no-LLM property that turned out to be the actual runtime path in Stages
   3 and 7 as well.
3. **Emerging signals (the long-tail from Stage 6/7) are mapped too, using
   the exact same embedding-similarity step**, run against Stage 7's
   already-batched long-tail summaries (the 819 `emerging_signals` entries
   themselves) - not re-run against the underlying raw units. Their
   evidence is tracked in a *separate* key per question
   (`signal_ids`/`signal_support_total`, tagged `signal_confidence: "low"`)
   so downstream reporting can distinguish strong theme-level evidence from
   weak long-tail signal, rather than blending them into one count.
4. **`coverage` reflects theme-level evidence only (S8-01).** A question is
   `"sufficient"` only if at least one real theme maps to it above
   threshold; emerging-signal-only support does not flip it to sufficient,
   consistent with point 3's strong/weak distinction.
5. **Off-topic themes/signals (S8-02) are bucketed, not dropped**: anything
   whose best similarity across all 8 questions is below
   `insights.similarity_threshold` is recorded under `"uncategorized"`.
6. **Headline `top_themes` (by `member_count`, `insights.top_themes_count`)
   is independent of question mapping** - just the biggest signals overall,
   for a quick "what matters most" view regardless of which question(s)
   they answer.
7. **`theme_segment_stats`** (rating-band + time-cohort distribution per
   theme) and **`category_graph`** (theme-to-theme similarity edges) are
   both computed here per architecture.md Stage 8's "quantify... rating-band
   spread, time-cohort spread" and problemstatement.md §7's "category-level
   graph" - the latter is a graph over our *own discovered themes*, not an
   external Blinkit product-category taxonomy (Google Play reviews carry no
   such label), built the same way as Stage 7's representative selection:
   each theme's centroid embedding compared to every other theme's.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from typing import Dict, List

import numpy as np

from src.config import Config, apply_global_seed, ensure_data_dir, load_config
from src.schema import QuestionInsight, SchemaError, Unit, read_json, read_jsonl, write_json
from src.summarize import _sentiment_from_rating

# Theme-to-theme "category graph" (problemstatement.md §7, architecture.md Stage 8): each theme
# is connected to its top-k most similar *other* themes by centroid-embedding similarity - a
# coarsened, theme-level analog of Stage 5's unit-level kNN graph, not an external product-category
# taxonomy (Google Play reviews carry no such label). Fixed at module scope since 40 themes makes
# this cheap regardless of corpus size; not worth a config knob.
_CATEGORY_GRAPH_TOP_K = 5

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class InsightsError(RuntimeError):
    """Raised for unrecoverable insight-mapping failures (e.g. missing themes.json)."""


def _load_model(model_name: str):
    try:
        # Same Windows DLL-conflict avoidance as embed.py: import torch directly before
        # sentence-transformers pulls it in transitively (edgecases.md S4-01/R-07).
        import torch  # noqa: F401
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise InsightsError(
            "sentence-transformers is not installed. Run `pip install -r requirements.txt`."
        ) from exc

    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise InsightsError(f"Failed to load embedding model '{model_name}': {exc}") from exc


def _embed(model, texts: List[str]) -> np.ndarray:
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)


def _theme_segment_stats(
    themes: List[dict],
    communities_by_id: Dict[int, dict],
    unit_meta: Dict[str, Unit],
) -> Dict[str, dict]:
    """Rating-band and time-cohort spread per theme (architecture.md Stage 8 quantification).

    Sourced from the theme's member units (via `communities.json`), not from the
    already-summarized `themes.json` alone - `avg_rating` there is only a mean, not a
    distribution, and this is exactly the kind of cross-segment spread context.md §5
    flags as needed for validation (do themes hold across rating bands / time periods?).
    """
    stats: Dict[str, dict] = {}
    for theme in themes:
        community = communities_by_id.get(theme["community_id"])
        if community is None:
            continue
        ratings: Counter = Counter()
        months: Counter = Counter()
        sources: Counter = Counter()
        for unit_id in community["unit_ids"]:
            unit = unit_meta.get(unit_id)
            if unit is None:
                continue
            if unit.rating is not None:
                ratings[unit.rating] += 1
            if unit.date:
                months[unit.date[:7]] += 1  # ISO 8601 -> "YYYY-MM"
            sources[unit.source] += 1
        stats[theme["theme_id"]] = {
            "rating_distribution": dict(sorted(ratings.items())),
            "time_cohort_distribution": dict(sorted(months.items())),
            # Informational only (Docs/context.md Addendum), same treatment as rating
            # bands (S9-05): with Mouthshut at ~4.4% of the merged unit corpus, a theme
            # showing overwhelming google_play dominance is expected, not evidence of
            # instability, so this is never folded into the cross-segment "stability"
            # judgment below.
            "source_distribution": dict(sorted(sources.items())),
        }
    return stats


def _category_graph_edges(
    themes: List[dict],
    communities_by_id: Dict[int, dict],
    id_to_row: Dict[str, int],
    embeddings: np.ndarray,
    top_k: int,
) -> List[dict]:
    """Theme-to-theme "category graph" (problemstatement.md §7): each theme's centroid embedding
    (cheap proxy for "averaged member-complaint similarities", same technique as Stage 7's
    representative selection and Stage 8's own RQ-matching) compared against every other theme's
    centroid; each theme keeps its top-k most similar *other* themes as edges.
    """
    centroids = np.zeros((len(themes), embeddings.shape[1]), dtype=np.float32)
    for i, theme in enumerate(themes):
        community = communities_by_id.get(theme["community_id"])
        rows = [id_to_row[uid] for uid in community["unit_ids"] if uid in id_to_row] if community else []
        if not rows:
            continue
        centroid = embeddings[rows].mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroids[i] = centroid / norm if norm > 0 else centroid

    sims = centroids @ centroids.T
    edges = []
    for i, theme in enumerate(themes):
        order = np.argsort(-sims[i])
        neighbors = [j for j in order if j != i][:top_k]
        for j in neighbors:
            edges.append(
                {
                    "theme_a": theme["theme_id"],
                    "theme_b": themes[j]["theme_id"],
                    "similarity": round(float(sims[i, j]), 3),
                }
            )
    return edges


def map_insights(config: Config, refresh: bool = False) -> None:
    ensure_data_dir(config)
    apply_global_seed(config.seed)

    if config.paths.insights.exists() and not refresh:
        logger.info(
            "'%s' already exists; skipping insight mapping (pass --refresh to rebuild).",
            config.paths.insights,
        )
        return

    for path in (config.paths.themes, config.paths.communities, config.paths.units, config.paths.embeddings, config.paths.unit_index):
        if not path.exists():
            raise InsightsError(
                f"Expected artifact at {path} but it does not exist. Run earlier pipeline stages "
                "first (edgecases.md X-03)."
            )

    themes_doc = read_json(config.paths.themes)
    themes: List[dict] = themes_doc.get("themes", [])
    signals: List[dict] = themes_doc.get("emerging_signals", [])
    if not themes:
        raise InsightsError("themes.json has zero themes - nothing to map (X-10).")

    communities_by_id = {c["community_id"]: c for c in read_json(config.paths.communities)["communities"]}
    unit_meta: Dict[str, Unit] = {u.unit_id: u for u in read_jsonl(config.paths.units, factory=Unit)}
    embeddings = np.load(config.paths.embeddings)
    unit_index = read_json(config.paths.unit_index)
    id_to_row = {uid: i for i, uid in enumerate(unit_index["unit_ids"])}

    model = _load_model(config.models.embedding_model)

    theme_texts = [f"{t['label']}. {t['description']}" for t in themes]
    signal_texts = [f"{s['label']}. {s['description']}" for s in signals]
    query_texts = config.insights.question_queries

    theme_vecs = _embed(model, theme_texts) if theme_texts else np.zeros((0, 1), dtype=np.float32)
    signal_vecs = _embed(model, signal_texts) if signal_texts else np.zeros((0, 1), dtype=np.float32)
    query_vecs = _embed(model, query_texts)

    theme_sims = theme_vecs @ query_vecs.T if len(theme_texts) else np.zeros((0, 8))
    signal_sims = signal_vecs @ query_vecs.T if len(signal_texts) else np.zeros((0, 8))

    threshold = config.insights.similarity_threshold
    max_verbatims = config.insights.max_verbatims_per_question
    required_sentiments = config.insights.question_required_sentiment
    signal_sentiments = [_sentiment_from_rating(s["avg_rating"]) for s in signals]

    questions: List[dict] = []
    for q_idx in range(8):
        question_id = q_idx + 1
        required_sentiment = required_sentiments[q_idx]

        theme_matches = [
            (themes[i], float(theme_sims[i, q_idx]))
            for i in range(len(themes))
            if theme_sims[i, q_idx] >= threshold
            and (required_sentiment is None or themes[i]["sentiment"] == required_sentiment)
        ]
        theme_matches.sort(key=lambda pair: -pair[1])
        signal_matches = [
            (signals[i], float(signal_sims[i, q_idx]))
            for i in range(len(signals))
            if signal_sims[i, q_idx] >= threshold
            and (required_sentiment is None or signal_sentiments[i] == required_sentiment)
        ]
        signal_matches.sort(key=lambda pair: -pair[1])

        coverage = "sufficient" if theme_matches else "insufficient"  # signals never flip this (design #4)

        top_verbatims: List[str] = []
        for theme, _ in theme_matches:
            for quote in theme["representative_quotes"]:
                if quote not in top_verbatims:
                    top_verbatims.append(quote)
                if len(top_verbatims) >= max_verbatims:
                    break
            if len(top_verbatims) >= max_verbatims:
                break

        try:
            insight = QuestionInsight(
                question_id=question_id,
                theme_ids=[t["theme_id"] for t, _ in theme_matches],
                total_count=sum(t["member_count"] for t, _ in theme_matches),
                top_verbatims=top_verbatims,
                coverage=coverage,
            )
        except SchemaError as exc:
            raise InsightsError(f"Invalid QuestionInsight for question {question_id}: {exc}") from exc

        questions.append(
            {
                "question_id": insight.question_id,
                "query": query_texts[q_idx],
                "coverage": insight.coverage,
                "theme_ids": insight.theme_ids,
                "theme_similarities": {t["theme_id"]: round(sim, 3) for t, sim in theme_matches},
                "total_count": insight.total_count,
                "top_verbatims": insight.top_verbatims,
                # Weak, long-tail-only evidence - kept separate per design choice #3/#4.
                "signal_ids": [s["signal_id"] for s, _ in signal_matches],
                "signal_support_total": sum(s["support_count"] for s, _ in signal_matches),
                "signal_confidence": "low" if signal_matches else None,
            }
        )

    matched_theme_ids = {tid for q in questions for tid in q["theme_ids"]}
    matched_signal_ids = {sid for q in questions for sid in q["signal_ids"]}
    uncategorized_theme_ids = [t["theme_id"] for t in themes if t["theme_id"] not in matched_theme_ids]
    uncategorized_signal_ids = [s["signal_id"] for s in signals if s["signal_id"] not in matched_signal_ids]

    top_themes = sorted(themes, key=lambda t: -t["member_count"])[: config.insights.top_themes_count]

    segment_stats = _theme_segment_stats(themes, communities_by_id, unit_meta)
    category_graph = _category_graph_edges(
        themes, communities_by_id, id_to_row, embeddings, _CATEGORY_GRAPH_TOP_K
    )

    payload = {
        "questions": questions,
        "uncategorized": {
            "theme_ids": uncategorized_theme_ids,
            "signal_ids": uncategorized_signal_ids,
        },
        "top_themes": [
            {"theme_id": t["theme_id"], "label": t["label"], "member_count": t["member_count"], "sentiment": t["sentiment"]}
            for t in top_themes
        ],
        "theme_segment_stats": segment_stats,
        "category_graph": category_graph,
        "summary": {
            "num_themes": len(themes),
            "num_emerging_signals": len(signals),
            "similarity_threshold": threshold,
            "num_questions_sufficient": sum(1 for q in questions if q["coverage"] == "sufficient"),
            "num_questions_insufficient": sum(1 for q in questions if q["coverage"] == "insufficient"),
            "num_uncategorized_themes": len(uncategorized_theme_ids),
            "num_uncategorized_signals": len(uncategorized_signal_ids),
            "num_category_graph_edges": len(category_graph),
        },
    }
    write_json(config.paths.insights, payload)
    _log_summary(payload)


def _log_summary(payload: dict) -> None:
    s = payload["summary"]
    logger.info(
        "Insight mapping complete: %d/%d questions sufficient, %d themes uncategorized (%d signals).",
        s["num_questions_sufficient"], 8, s["num_uncategorized_themes"], s["num_uncategorized_signals"],
    )
    for q in payload["questions"]:
        logger.info(
            "  Q%d [%s]: %d themes (count=%d), %d signals (support=%d)",
            q["question_id"], q["coverage"], len(q["theme_ids"]), q["total_count"],
            len(q["signal_ids"]), q["signal_support_total"],
        )
    logger.info(
        "  top %d themes by size: %s",
        len(payload["top_themes"]),
        ", ".join(f"{t['theme_id']}({t['member_count']})" for t in payload["top_themes"]),
    )
    logger.info("  category graph: %d edges across %d themes", s["num_category_graph_edges"], s["num_themes"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map themes and long-tail signals to the 8 research questions (Stage 8)."
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: project root)")
    parser.add_argument(
        "--refresh", action="store_true", help="Rebuild insights.json even if it already exists"
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config) if args.config else load_config()
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    try:
        map_insights(config, refresh=args.refresh)
    except InsightsError as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
