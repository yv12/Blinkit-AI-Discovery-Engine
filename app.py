"""Stage 10 - Streamlit UI. Browse themes, drill into verbatims, and see each of the
8 research questions answered with evidence (counts, sentiment, triangulation).

See architecture.md §4 (Stage 10) and Implementation-plan.md Phase 6. Reads only the
artifacts already produced by earlier stages (`insights.json`, `themes.json`,
`validation.json`, plus `reviews.jsonl`/`units.jsonl` for corpus-level counts) - it
never mutates or re-derives pipeline data, so it stays trivially re-runnable against
whatever `data/` currently holds.

Run with:  streamlit run app.py
Share on your LAN with:  streamlit run app.py --server.address 0.0.0.0
(then send teammates http://<your-ip>:8501 - see README.md for details).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import altair as alt
import networkx as nx
import pandas as pd
import streamlit as st

from src.config import Config, load_config
from src.schema import read_json

# The 8 literal research questions this whole pipeline exists to answer
# (problemstatement.md §3). `insights.json`'s `query` field is a short topic
# description used for *embedding matching*, not this literal text - both are
# shown in the UI so the connection between "what we asked" and "what we
# matched on" stays transparent.
RESEARCH_QUESTIONS: Dict[int, str] = {
    1: "Why do users repeatedly buy from the same categories?",
    2: "What prevents users from exploring new categories?",
    3: "How do users discover products today?",
    4: "What role do habits play in shopping behavior?",
    5: "What information do users need before trying a new category?",
    6: "What frustrations emerge repeatedly?",
    7: "Which user segments are more likely to experiment?",
    8: "What unmet needs emerge consistently across discussions?",
}

SENTIMENT_COLOR = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}
SENTIMENT_EMOJI = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}

STAGE_TABLE = [
    ("S1", "Scrape", "src/scrape.py", "raw_reviews.jsonl", "Rolling 4-month Play Store pull, all rating bands"),
    ("S1b", "Ingest Mouthshut", "src/scrape_mouthshut.py", "raw_mouthshut.jsonl", "Optional second source - CSV -> raw JSONL (Docs/context.md Addendum)"),
    ("S2", "Normalize", "src/normalize.py", "reviews.jsonl", "Canonical schema, ISO dates, dedup/lang tagging, merges both sources"),
    ("S3", "Unit extraction", "src/units.py", "units.jsonl", "Rule-based split into atomic complaint/insight statements"),
    ("S4", "Embed", "src/embed.py", "embeddings.npy", "Local all-MiniLM-L6-v2 sentence embeddings"),
    ("S5", "Similarity graph", "src/graph.py", "graph.gpickle", "kNN cosine graph over unit embeddings"),
    ("S6", "Community detection", "src/cluster.py", "communities.json", "Louvain clustering into theme communities"),
    ("S7", "Summarization", "src/summarize.py", "themes.json", "Label + description + verbatims per community"),
    ("S8", "Insight mapping", "src/insights.py", "insights.json", "Themes mapped to the 8 research questions"),
    ("S9", "Validation", "src/validate.py", "validation.json", "Coherence, triangulation, spot-check sample"),
    ("S10", "UI", "app.py", "(this app)", "Browse everything above, answer the 8 questions"),
]


# --------------------------------------------------------------------------- #
# Cached data loading
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner=False)
def get_config() -> Config:
    return load_config()


@st.cache_data(show_spinner=False)
def load_artifact(path_str: str) -> dict:
    return read_json(Path(path_str))


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


@st.cache_data(show_spinner="Computing corpus statistics (one-time, cached)...")
def load_corpus_stats(reviews_path_str: str, units_path_str: str) -> dict:
    reviews_path, units_path = Path(reviews_path_str), Path(units_path_str)
    rating_counts: Counter = Counter()
    lang_counts: Counter = Counter()
    source_counts: Counter = Counter()
    dates: List[str] = []
    num_reviews = 0
    for row in _iter_jsonl(reviews_path):
        num_reviews += 1
        if row.get("rating") is not None:
            rating_counts[int(row["rating"])] += 1
        lang = (row.get("metadata") or {}).get("lang", "en")
        lang_counts[lang] += 1
        source_counts[row.get("source", "google_play")] += 1
        if row.get("date"):
            dates.append(row["date"])
    num_units = sum(1 for _ in _iter_jsonl(units_path))
    return {
        "num_reviews": num_reviews,
        "num_units": num_units,
        "rating_distribution": dict(sorted(rating_counts.items())),
        "lang_distribution": dict(lang_counts),
        # Second source (Docs/context.md Addendum "Second data source - Mouthshut"):
        # google_play was the only value here before that addendum.
        "source_distribution": dict(source_counts),
        "date_min": min(dates) if dates else None,
        "date_max": max(dates) if dates else None,
    }


def build_theme_question_map(insights: dict) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    """Reverse `insights.json`'s question->theme_ids into theme_id->question_ids.

    `themes.json`'s own `questions` field is never populated by Stage 8 (mapping
    lives one-directionally in `insights.json`), so this reverse index is the
    only way to answer "which questions does this theme support?" per theme.
    """
    theme_to_q: Dict[str, List[int]] = {}
    signal_to_q: Dict[str, List[int]] = {}
    for q in insights["questions"]:
        for tid in q["theme_ids"]:
            theme_to_q.setdefault(tid, []).append(q["question_id"])
        for sid in q["signal_ids"]:
            signal_to_q.setdefault(sid, []).append(q["question_id"])
    return theme_to_q, signal_to_q


def build_validation_lookup(validation: dict) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    coherence_by_id = {t["theme_id"]: t for t in validation["coherence"]["themes"]}
    triangulation_by_id = {t["theme_id"]: t for t in validation["triangulation"]["themes"]}
    return coherence_by_id, triangulation_by_id


def coverage_badge(coverage: str) -> str:
    return "✅ Sufficient" if coverage == "sufficient" else "⚠️ Insufficient"


def sentiment_badge(sentiment: str) -> str:
    return f"{SENTIMENT_EMOJI.get(sentiment, '⚪')} {sentiment}"


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #


def render_overview(config: Config, themes_doc: dict, insights: dict, validation: dict) -> None:
    st.header("Discovery Engine — Overview")
    stats = load_corpus_stats(str(config.paths.reviews), str(config.paths.units))
    has_mouthshut = stats["source_distribution"].get("mouthshut", 0) > 0
    st.markdown(
        "An AI-powered analysis of **Google Play Store reviews for Blinkit** "
        "(`com.grofers.customerapp`)"
        + (
            " plus a **Mouthshut** review-forum sample (Docs/context.md Addendum — a "
            "deliberate, documented relaxation of the original single-source design)"
            if has_mouthshut
            else ""
        )
        + ", built to answer 8 research questions about category-exploration "
        "behavior — fully local/zero-cost: local embeddings, rule-based/local-LLM "
        "summarization, graph-based theme discovery."
    )

    s = insights["summary"]
    v = validation["summary"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Reviews scraped", f"{stats['num_reviews']:,}")
    c2.metric("Atomic units", f"{stats['num_units']:,}")
    c3.metric("Themes", s["num_themes"])
    c4.metric("Emerging signals", f"{s['num_emerging_signals']:,}")
    c5.metric("Questions answered", f"{s['num_questions_sufficient']}/8")

    c6, c7, c8, c9 = st.columns(4)
    c6.metric("Graph modularity", validation["coherence"]["modularity"])
    c7.metric("Cross-segment stable themes", f"{v['num_cross_segment_stable']}/{v['num_themes_validated']}")
    c8.metric("Uncategorized themes", s["num_uncategorized_themes"])
    c9.metric("Category-graph edges", s["num_category_graph_edges"])

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.subheader("Rating distribution (full corpus)")
        rating_df = pd.DataFrame(
            {"rating": list(stats["rating_distribution"].keys()), "count": list(stats["rating_distribution"].values())}
        )
        st.bar_chart(rating_df.set_index("rating"))
    with col_b:
        st.subheader("Language split")
        lang_df = pd.DataFrame(
            {"lang": list(stats["lang_distribution"].keys()), "count": list(stats["lang_distribution"].values())}
        )
        st.bar_chart(lang_df.set_index("lang"))
    with col_c:
        st.subheader("Source split")
        source_df = pd.DataFrame(
            {"source": list(stats["source_distribution"].keys()), "count": list(stats["source_distribution"].values())}
        )
        st.bar_chart(source_df.set_index("source"))
        if has_mouthshut:
            st.caption(
                "Mouthshut is a small, recently-added second source (Docs/context.md "
                "Addendum) — themes will read as overwhelmingly Google-Play-driven by "
                "volume; see a theme's `source_distribution` in Theme Explorer."
            )

    st.caption(
        f"Review window: {stats['date_min']} → {stats['date_max']}  •  "
        f"kNN similarity_threshold={config.graph.similarity_threshold}, k={config.graph.knn_k}  •  "
        f"Louvain resolution={config.clustering.louvain_resolution}"
    )

    st.subheader("Pipeline stages (S1 → S10)")
    st.dataframe(
        pd.DataFrame(STAGE_TABLE, columns=["Stage", "Name", "Module", "Artifact", "What it does"]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Top themes by size")
    top_df = pd.DataFrame(insights["top_themes"])
    top_df["sentiment"] = top_df["sentiment"].map(sentiment_badge)
    st.dataframe(top_df, use_container_width=True, hide_index=True)


def _render_llm_patterns(q_id: int, llm_insights: Optional[dict]) -> None:
    """Stage 11 (optional) - LLM-inferred indirect/implicit patterns for this question.

    Deliberately rendered as a visually distinct, clearly-labeled section: this evidence
    comes from a remote LLM (Groq) *inferring* an abstract behavioral driver from raw
    review text, not from the deterministic embedding-similarity mapping in insights.json
    (Stage 8). Every pattern still cites real verbatim quotes for traceability, but the
    *interpretation* itself is model-generated and should be read as a hypothesis to
    validate, not as extracted fact the way the rest of the app's evidence is.
    """
    st.markdown("### 🧠 LLM-Inferred Patterns *(optional, experimental — Groq)*")
    if llm_insights is None:
        st.caption(
            "Not run yet. `insights.json` above can only ever surface *literal* recurring "
            "vocabulary (TF-IDF terms / embedding similarity) - it can never infer an "
            "abstract, indirectly-implied driver (e.g. \"choice overload\", \"no trust in "
            "unfamiliar brands\") that no single review states outright. Run "
            "`python -m src.llm_synthesis` (requires a free `GROQ_API_KEY` in `.env` - "
            "see README.md \u00a712) to populate this section."
        )
        return

    q_llm = next((q for q in llm_insights["questions"] if q["question_id"] == q_id), None)
    st.caption(
        f"Model: `{llm_insights['model']}` (temperature=0, fixed seed) over "
        f"{llm_insights['sampled_unit_count']:,} sampled review excerpts (prioritized: units "
        "Stage 8 couldn't place + a stratified random top-up) - each pattern is the model's "
        "own inference, not a literal extraction; example quotes are always real verbatim "
        "text so every pattern stays traceable. **Known limitation** (observed on the real "
        "run, not hypothetical): the free-tier model occasionally defaults to a generic "
        "stock phrase for ambiguous/low-content excerpts, which can inflate one pattern's "
        "support count with genuinely weak examples - always skim the quotes below, don't "
        "trust `support_count` alone as a signal of importance."
    )
    if not q_llm or not q_llm["patterns"]:
        st.info(
            f"No inferred pattern cleared the minimum-support threshold for Q{q_id} "
            f"({q_llm['raw_hit_count'] if q_llm else 0} raw (excerpt, question) hits before "
            "clustering/filtering)."
        )
        return

    for p in q_llm["patterns"]:
        with st.expander(f"\u201c{p['pattern']}\u201d — {p['support_count']} supporting excerpts"):
            for quote in p["example_quotes"]:
                st.markdown(f"> {quote}")


def render_questions(themes_doc: dict, insights: dict, llm_insights: Optional[dict]) -> None:
    st.header("The 8 Research Questions")
    st.markdown(
        "Every theme is mapped to one or more of these questions via **embedding "
        "similarity** between the theme's label+description and a short topic "
        "description per question (`config.yaml: insights.question_queries`) — "
        "never keywords, never an LLM call. `coverage` is only ever `sufficient` "
        "when at least one real theme (not just a low-confidence emerging signal) "
        "clears the similarity threshold."
    )

    themes_by_id = {t["theme_id"]: t for t in themes_doc["themes"]}
    signals_by_id = {s["signal_id"]: s for s in themes_doc["emerging_signals"]}

    overview_rows = [
        {
            "Q": q["question_id"],
            "Question": RESEARCH_QUESTIONS[q["question_id"]],
            "Coverage": coverage_badge(q["coverage"]),
            "Themes": len(q["theme_ids"]),
            "Unit count": q["total_count"],
            "Signals (weak evidence)": len(q["signal_ids"]),
        }
        for q in insights["questions"]
    ]
    st.dataframe(pd.DataFrame(overview_rows), use_container_width=True, hide_index=True)

    q_id = st.selectbox(
        "Inspect a question",
        options=[q["question_id"] for q in insights["questions"]],
        format_func=lambda i: f"Q{i}: {RESEARCH_QUESTIONS[i]}",
    )
    q = next(q for q in insights["questions"] if q["question_id"] == q_id)

    st.markdown(f"### Q{q_id}: {RESEARCH_QUESTIONS[q_id]}")
    st.caption(f"Embedding query used for matching: \u201c{q['query']}\u201d")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Coverage", coverage_badge(q["coverage"]))
    m2.metric("Supporting themes", len(q["theme_ids"]))
    m3.metric("Unit count (themes only)", q["total_count"])
    m4.metric("Emerging-signal support", q["signal_support_total"])

    if q["coverage"] == "insufficient":
        st.warning(
            "No real theme clears the similarity threshold for this question yet — "
            "reported honestly rather than force-matched (edgecases.md S8-01). "
            f"{'There is some weak, long-tail signal support though (see below).' if q['signal_ids'] else ''}"
        )

    if q["theme_ids"]:
        st.markdown("**Supporting themes** (sorted by similarity to the question)")
        rows = []
        for tid in q["theme_ids"]:
            t = themes_by_id.get(tid)
            if not t:
                continue
            rows.append(
                {
                    "theme_id": tid,
                    "label": t["label"],
                    "similarity": q["theme_similarities"].get(tid),
                    "member_count": t["member_count"],
                    "sentiment": sentiment_badge(t["sentiment"]),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if q["top_verbatims"]:
        st.markdown("**Representative verbatims**")
        for v in q["top_verbatims"]:
            st.markdown(f"> {v}")

    if q["signal_ids"]:
        with st.expander(f"Weak long-tail signal support ({len(q['signal_ids'])} signals, low confidence)"):
            sig_rows = [
                {
                    "signal_id": sid,
                    "label": signals_by_id[sid]["label"],
                    "support_count": signals_by_id[sid]["support_count"],
                    "avg_rating": signals_by_id[sid]["avg_rating"],
                }
                for sid in q["signal_ids"]
                if sid in signals_by_id
            ]
            st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True)

    st.divider()
    _render_llm_patterns(q_id, llm_insights)

    st.divider()
    uncategorized = insights["uncategorized"]
    with st.expander(
        f"Uncategorized (below similarity threshold for all 8 questions): "
        f"{len(uncategorized['theme_ids'])} themes, {len(uncategorized['signal_ids'])} signals"
    ):
        st.write("Themes:", ", ".join(uncategorized["theme_ids"]) or "none")
        st.caption(
            "Kept, not dropped (edgecases.md S8-02) — visible in the Theme Explorer tab."
        )


def render_theme_explorer(
    themes_doc: dict,
    theme_to_q: Dict[str, List[int]],
    signal_to_q: Dict[str, List[int]],
    coherence_by_id: Dict[str, dict],
    triangulation_by_id: Dict[str, dict],
) -> None:
    st.header("Theme Explorer")
    tab_themes, tab_signals = st.tabs(
        [f"Themes ({len(themes_doc['themes'])})", f"Emerging signals ({len(themes_doc['emerging_signals'])})"]
    )

    with tab_themes:
        themes = themes_doc["themes"]
        col1, col2, col3 = st.columns(3)
        with col1:
            sentiments = col1.multiselect(
                "Sentiment", ["positive", "neutral", "negative"], default=["positive", "neutral", "negative"]
            )
        with col2:
            question_filter = col2.multiselect("Answers question", list(range(1, 9)))
        with col3:
            search = col3.text_input("Search label/description/quotes")

        filtered = []
        for t in themes:
            if t["sentiment"] not in sentiments:
                continue
            qids = theme_to_q.get(t["theme_id"], [])
            if question_filter and not set(question_filter) & set(qids):
                continue
            if search:
                haystack = " ".join([t["label"], t["description"], *t["representative_quotes"]]).lower()
                if search.lower() not in haystack:
                    continue
            filtered.append(t)

        rows = []
        for t in filtered:
            tri = triangulation_by_id.get(t["theme_id"], {})
            coh = coherence_by_id.get(t["theme_id"], {})
            rows.append(
                {
                    "theme_id": t["theme_id"],
                    "label": t["label"],
                    "member_count": t["member_count"],
                    "sentiment": sentiment_badge(t["sentiment"]),
                    "questions": ", ".join(f"Q{q}" for q in sorted(theme_to_q.get(t["theme_id"], []))) or "—",
                    "stability": tri.get("stability", "—"),
                    "silhouette": coh.get("silhouette_score"),
                }
            )
        rows.sort(key=lambda r: -r["member_count"])
        st.caption(f"{len(filtered)} / {len(themes)} themes match filters")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if not filtered:
            st.info("No themes match the current filters.")
            return

        sel_id = st.selectbox(
            "Inspect a theme",
            options=[t["theme_id"] for t in filtered],
            format_func=lambda tid: f"{tid} — {next(t['label'] for t in filtered if t['theme_id'] == tid)}",
        )
        theme = next(t for t in themes if t["theme_id"] == sel_id)
        _render_theme_detail(theme, theme_to_q, coherence_by_id, triangulation_by_id)

    with tab_signals:
        signals = themes_doc["emerging_signals"]
        search2 = st.text_input("Search signal label/quotes", key="signal_search")
        conf_filter = st.multiselect("Confidence", ["very_low", "low"], default=["very_low", "low"], key="signal_conf")
        filtered_signals = [
            s
            for s in signals
            if s["confidence"] in conf_filter
            and (not search2 or search2.lower() in " ".join([s["label"], *s["representative_quotes"]]).lower())
        ]
        st.caption(f"{len(filtered_signals)} / {len(signals)} signals match filters")
        sig_rows = [
            {
                "signal_id": s["signal_id"],
                "label": s["label"],
                "support_count": s["support_count"],
                "confidence": s["confidence"],
                "avg_rating": s["avg_rating"],
                "questions": ", ".join(f"Q{q}" for q in sorted(signal_to_q.get(s["signal_id"], []))) or "—",
            }
            for s in sorted(filtered_signals, key=lambda s: -s["support_count"])
        ]
        st.dataframe(pd.DataFrame(sig_rows), use_container_width=True, hide_index=True, height=400)


def _render_theme_detail(
    theme: dict,
    theme_to_q: Dict[str, List[int]],
    coherence_by_id: Dict[str, dict],
    triangulation_by_id: Dict[str, dict],
) -> None:
    st.markdown(f"#### {theme['theme_id']} — {theme['label']}")
    st.write(theme["description"])
    m1, m2, m3 = st.columns(3)
    m1.metric("Members", theme["member_count"])
    m2.metric("Sentiment", sentiment_badge(theme["sentiment"]))
    qids = sorted(theme_to_q.get(theme["theme_id"], []))
    m3.metric("Maps to questions", ", ".join(f"Q{q}" for q in qids) or "none (uncategorized)")

    st.markdown("**Representative verbatims**")
    for quote in theme["representative_quotes"]:
        st.markdown(f"> {quote}")

    tri = triangulation_by_id.get(theme["theme_id"])
    coh = coherence_by_id.get(theme["theme_id"])
    col_a, col_b, col_c, col_d = st.columns(4)
    if tri:
        with col_a:
            st.markdown("**Rating distribution**")
            st.bar_chart(pd.Series(tri["rating_distribution"], name="count"))
        with col_b:
            st.markdown("**Time-cohort distribution**")
            st.bar_chart(pd.Series(tri["time_cohort_distribution"], name="count"))
        with col_c:
            st.markdown("**Review-length distribution**")
            st.bar_chart(pd.Series(tri["length_bucket_distribution"], name="count"))
        with col_d:
            st.markdown("**Source distribution**")
            source_dist = tri.get("source_distribution") or {}
            if source_dist:
                st.bar_chart(pd.Series(source_dist, name="count"))
            else:
                st.caption("n/a (validation.json predates the second source)")
        stability_msg = (
            f"✅ Cross-segment stable" if tri["stability"] == "cross_segment"
            else f"⚠️ Segment-specific: {', '.join(tri['stability_reasons'])}"
        )
        st.caption(f"Cross-segment triangulation (Stage 9): {stability_msg}")
    if coh:
        st.caption(
            f"Coherence (Stage 9): intra-theme similarity={coh['intra_similarity']}, "
            f"nearest other theme={coh['nearest_theme_id']} (sim={coh['nearest_theme_similarity']}), "
            f"silhouette-style score={coh['silhouette_score']}"
        )


def render_category_graph(themes_doc: dict, insights: dict, config: Config) -> None:
    st.header("Category Graph")
    st.markdown(
        "A navigable map of theme-to-theme similarity (problemstatement.md §7) — each theme's "
        "centroid embedding compared against every other theme's; top-5 most similar neighbors "
        "kept as edges. Node size = member count, color = sentiment, edges = similarity."
    )

    themes = {t["theme_id"]: t for t in themes_doc["themes"]}
    edges = insights["category_graph"]

    graph = nx.Graph()
    for tid, t in themes.items():
        graph.add_node(tid, **t)
    for e in edges:
        if e["theme_a"] in themes and e["theme_b"] in themes:
            graph.add_edge(e["theme_a"], e["theme_b"], weight=e["similarity"])

    pos = nx.spring_layout(graph, seed=config.seed, weight="weight", k=0.9)

    nodes_df = pd.DataFrame(
        [
            {
                "theme_id": tid,
                "x": pos[tid][0],
                "y": pos[tid][1],
                "label": themes[tid]["label"],
                "member_count": themes[tid]["member_count"],
                "sentiment": themes[tid]["sentiment"],
            }
            for tid in graph.nodes
        ]
    )

    edge_rows = []
    for i, e in enumerate(edges):
        if e["theme_a"] in pos and e["theme_b"] in pos:
            xa, ya = pos[e["theme_a"]]
            xb, yb = pos[e["theme_b"]]
            edge_rows.append({"edge_id": i, "x": xa, "y": ya, "similarity": e["similarity"]})
            edge_rows.append({"edge_id": i, "x": xb, "y": yb, "similarity": e["similarity"]})
    edges_df = pd.DataFrame(edge_rows)

    line_layer = alt.Chart(edges_df).mark_line(opacity=0.2, color="gray").encode(
        x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None), detail="edge_id:N"
    )
    point_layer = alt.Chart(nodes_df).mark_circle().encode(
        x=alt.X("x:Q", axis=None),
        y=alt.Y("y:Q", axis=None),
        size=alt.Size("member_count:Q", scale=alt.Scale(range=[100, 3000]), legend=None),
        color=alt.Color(
            "sentiment:N",
            scale=alt.Scale(domain=list(SENTIMENT_COLOR.keys()), range=list(SENTIMENT_COLOR.values())),
        ),
        tooltip=["theme_id", "label", "member_count", "sentiment"],
    )
    text_layer = alt.Chart(nodes_df).mark_text(dy=-14, fontSize=10).encode(
        x=alt.X("x:Q", axis=None), y=alt.Y("y:Q", axis=None), text="theme_id:N"
    )

    chart = (line_layer + point_layer + text_layer).properties(width=900, height=650).interactive()
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Edges (raw)")
    st.dataframe(
        pd.DataFrame(edges).sort_values("similarity", ascending=False),
        use_container_width=True,
        hide_index=True,
        height=300,
    )


def render_validation(validation: dict, themes_doc: dict, config: Config) -> None:
    st.header("Validation & Methodology")
    st.markdown(
        "Three independent checks over the 40 qualifying themes (edgecases.md Stage 9): "
        "**coherence** (are the clusters actually tight?), **cross-segment triangulation** "
        "(does a theme hold up across rating bands / time / review length, or is it an artifact "
        "of one narrow slice?), and a **human spot-check sample**."
    )

    s = validation["summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Graph modularity", s["modularity"])
    c2.metric("Mean silhouette-style score", s["mean_silhouette_score"])
    c3.metric("Cross-segment stable", f"{s['num_cross_segment_stable']}/{s['num_themes_validated']}")
    c4.metric("Spot-check labels collected", s["spot_check_labeled_count"])

    themes_by_id = {t["theme_id"]: t for t in themes_doc["themes"]}

    st.subheader("Coherence — per theme (sorted least → most coherent)")
    st.caption(
        "High graph modularity + near-zero mean silhouette is not a contradiction: Louvain "
        "separates fine-grained unit-to-unit sub-topics (hence high modularity), while several "
        "themes share enough surface vocabulary that their coarse centroid vectors sit close "
        "together in embedding space."
    )
    coh_rows = []
    for t in validation["coherence"]["themes"]:
        coh_rows.append(
            {
                "theme_id": t["theme_id"],
                "label": themes_by_id.get(t["theme_id"], {}).get("label", ""),
                "size": t["size"],
                "intra_similarity": t["intra_similarity"],
                "nearest_theme": t["nearest_theme_id"],
                "nearest_similarity": t["nearest_theme_similarity"],
                "silhouette_score": t["silhouette_score"],
            }
        )
    coh_df = pd.DataFrame(coh_rows).sort_values("silhouette_score", na_position="last")
    st.dataframe(coh_df, use_container_width=True, hide_index=True)

    st.subheader("Cross-segment triangulation")
    tri = validation["triangulation"]
    stability_counts = Counter(t["stability"] for t in tri["themes"])
    st.bar_chart(pd.Series(stability_counts, name="themes"))

    specific = [t for t in tri["themes"] if t["stability"] == "segment_specific"]
    with st.expander(f"Segment-specific themes ({len(specific)})"):
        for t in specific:
            label = themes_by_id.get(t["theme_id"], {}).get("label", "")
            st.markdown(f"- **{t['theme_id']}** ({label}): {', '.join(t['stability_reasons'])}")

    st.subheader("Human spot-check sample")
    sc = validation["spot_check"]
    st.write(
        f"{sc['sample_size']} sampled units across all themes, {sc['labeled_count']} labeled so far"
        + (f", agreement rate = {sc['agreement_rate']}" if sc["agreement_rate"] is not None else ".")
    )
    st.info(sc["note"])

    sample_path = config.paths.spot_check_sample
    if sample_path.exists():
        with st.expander("Browse the spot-check sample"):
            sample = load_artifact(str(sample_path))
            st.dataframe(pd.DataFrame(sample), use_container_width=True, hide_index=True, height=400)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    st.set_page_config(page_title="Discovery Engine — Blinkit Reviews", layout="wide", page_icon="🔎")

    try:
        config = get_config()
    except Exception as exc:
        st.error(f"Failed to load config.yaml: {exc}")
        st.stop()

    missing = [
        p
        for p in (config.paths.insights, config.paths.themes, config.paths.validation)
        if not p.exists()
    ]
    if missing:
        st.error(
            "Required artifact(s) not found:\n\n"
            + "\n".join(f"- `{p}`" for p in missing)
            + "\n\nRun the pipeline first, e.g. `python -m src.pipeline` "
            "(see Docs/Implementation-plan.md for stage order)."
        )
        st.stop()

    try:
        insights = load_artifact(str(config.paths.insights))
        themes_doc = load_artifact(str(config.paths.themes))
        validation = load_artifact(str(config.paths.validation))
    except Exception as exc:
        st.error(f"Failed to load pipeline artifacts: {exc}")
        st.stop()

    # Stage 11 (optional) - only present if `python -m src.llm_synthesis` has been run.
    llm_insights: Optional[dict] = None
    if config.paths.llm_insights.exists():
        try:
            llm_insights = load_artifact(str(config.paths.llm_insights))
        except Exception as exc:
            st.sidebar.warning(f"Found llm_insights.json but failed to load it: {exc}")

    theme_to_q, signal_to_q = build_theme_question_map(insights)
    coherence_by_id, triangulation_by_id = build_validation_lookup(validation)

    st.sidebar.title("🔎 Discovery Engine")
    st.sidebar.caption("Blinkit review analysis — Google Play + Mouthshut")
    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Research Questions", "Theme Explorer", "Category Graph", "Validation & Methodology"],
    )
    st.sidebar.divider()
    st.sidebar.markdown(
        "**Share this app:**\n\n"
        "`streamlit run app.py --server.address 0.0.0.0`\n\n"
        "then share `http://<your-ip>:8501` on your network."
    )
    st.sidebar.divider()
    if llm_insights:
        st.sidebar.success(
            f"🧠 LLM synthesis (Stage 11): {llm_insights['summary']['total_patterns']} "
            "inferred patterns loaded — see Research Questions."
        )
    else:
        st.sidebar.caption(
            "🧠 Optional: run `python -m src.llm_synthesis` (needs a free Groq API key in "
            "`.env`) to add LLM-inferred indirect-pattern evidence to Research Questions."
        )

    if page == "Overview":
        render_overview(config, themes_doc, insights, validation)
    elif page == "Research Questions":
        render_questions(themes_doc, insights, llm_insights)
    elif page == "Theme Explorer":
        render_theme_explorer(themes_doc, theme_to_q, signal_to_q, coherence_by_id, triangulation_by_id)
    elif page == "Category Graph":
        render_category_graph(themes_doc, insights, config)
    elif page == "Validation & Methodology":
        render_validation(validation, themes_doc, config)


if __name__ == "__main__":
    main()
