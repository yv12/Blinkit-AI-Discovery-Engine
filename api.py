"""Stage 10 (current UI) - FastAPI backend serving the React interface in web/.

One `uvicorn api:app` command serves both the JSON API and the static frontend on a
single port. Reads the final artifacts (themes.json, insights.json, validation.json,
communities.json, units.jsonl, embeddings.npy) and never mutates pipeline data.

POST /api/ask runs *live* semantic search over the real unit embeddings (the same local
sentence-transformers/all-MiniLM-L6-v2 model Stage 4 already produced) to answer
free-form questions, not just the 8 canonical research questions. Matching reviews
are then synthesized into a short prose answer via Groq (``src.ask_synthesis``,
same ``GROQ_API_KEY`` / ``llm_synthesis.model`` as theme titles); if that call
fails, the UI still gets the review list.

Run with:  uvicorn api:app --host 0.0.0.0 --port 8000
Then open http://localhost:8000 (or http://<your-ip>:8000 on your LAN).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import os
import requests
import threading
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

HF_TOKEN = os.environ.get("HF_TOKEN")
EMBED_URL = (
    "https://router.huggingface.co/hf-inference/models/"
    "sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
)

def encode_query(text: str) -> np.ndarray:
    r = requests.post(
        EMBED_URL,
        headers={"Authorization": f"Bearer {HF_TOKEN}" if HF_TOKEN else {}},
        json={"inputs": [text]},
        timeout=20,
    )
    r.raise_for_status()
    vec = np.asarray(r.json(), dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

from src.ask_synthesis import SYNTHESIS_FALLBACK, strip_em_dashes, synthesize_answer
from src.config import load_config
from src.schema import read_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("api")

# The 8 literal research questions (problemstatement.md §3) - same text app.py uses.
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

# Maps the pipeline's rating-derived Theme.sentiment (negative/neutral/positive,
# never LLM-derived - src/summarize.py S7-06) onto the UI's visual vocabulary.
# "friction" / "pattern" are the two categories the original design shipped with;
# "positive" is a genuine third category real data surfaces that the mock data
# (all friction/pattern) never needed.
SENTIMENT_UI_MAP = {"negative": "friction", "neutral": "pattern", "positive": "positive"}

MIN_ASK_SIMILARITY = 0.15  # below this, a unit is not considered a match at all
ASK_WIDENED_SIMILARITY = 0.08  # used only when too few reviews clear the primary floor
ASK_MAX_REVIEWS = 20  # top reviews passed to synthesis + Supporting reviews list
ASK_MIN_REVIEWS_FOR_CONTEXT = 8  # widen threshold rather than synthesize on a thin set
ASK_MAX_THEMES = 3  # top matching theme clusters for the THEMES evidence block
ASK_REVIEWS_PER_THEME = 7  # diversify context across matched themes (not only global top-sim)
MIN_UNIT_WORDS = 4  # skip near-empty fragments as citations (same spirit as units.min_words)

app = FastAPI(title="Discovery Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    def _safe_load():
        try:
            _resolve_lfs_pointers()
            _load_state()
            logger.info("STATE LOADED OK")
        except Exception as e:
            app.state.startup_error = str(e)
            logger.error("Failed to load state", exc_info=True)
    threading.Thread(target=_safe_load, daemon=True).start()

def _resolve_lfs_pointers():
    """If the platform (like Railway) didn't pull Git LFS files, download them manually."""
    import os
    import requests
    data_dir = "data"
    if not os.path.exists(data_dir):
        return
    skip_files = {"embeddings_raw.npy", "raw_reviews.jsonl", "reviews.jsonl", "units_raw.jsonl"}
    for filename in os.listdir(data_dir):
        if filename in skip_files:
            continue
        path = os.path.join(data_dir, filename)
        if os.path.isfile(path) and os.path.getsize(path) < 1000:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(200)
            if "version https://git-lfs.github.com/spec" in content:
                logger.info(f"Downloading LFS file directly from GitHub: {filename}")
                url = f"https://media.githubusercontent.com/media/yv12/Blinkit-AI-Discovery-Engine/main/data/{filename}"
                r = requests.get(url, stream=True)
                r.raise_for_status()
                with open(path, "wb") as f_out:
                    for chunk in r.iter_content(chunk_size=8192):
                        f_out.write(chunk)
                logger.info(f"Downloaded {filename} successfully.")

_STATE: dict = {}


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _synthesize_objective(theme: dict) -> str:
    """A templated (not LLM-authored) growth-objective suggestion derived from the
    theme's own rating-derived sentiment + label. Deliberately generic/templated
    rather than fabricated specifics - keeps the project's traceability principle:
    this is a suggestion to validate, not a claim presented as extracted evidence.
    """
    label, count = theme["label"], theme["member_count"]
    if theme["sentiment"] == "negative":
        return (
            f"Address the friction behind \u201c{label}\u201d - a recurring pain point across "
            f"{count:,} reviews - since unresolved friction here likely discourages users from "
            "trying anything beyond what they already trust."
        )
    if theme["sentiment"] == "positive":
        return (
            f"Reinforce what\u2019s already working in \u201c{label}\u201d ({count:,} reviews) as a trust "
            "foundation, and use it as a springboard to nudge users toward one adjacent, unfamiliar "
            "category."
        )
    return (
        f"Investigate how \u201c{label}\u201d ({count:,} reviews) shapes everyday shopping behavior, and "
        "look for a low-effort nudge toward a new category that doesn\u2019t disrupt it."
    )


def _build_insight_text(theme: dict, theme_to_q: Dict[str, List[int]]) -> str:
    base = theme["description"]
    qids = sorted(theme_to_q.get(theme["theme_id"], []))
    if qids:
        base += " Maps to " + ", ".join(f"Q{q}" for q in qids) + " in the research-question framework."
    return base


READY = False

def _load_state() -> None:
    import os
    try:
        import resource
        def log_mem(step=""):
            logger.info(f"{step} peak RSS MB: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}")
    except ImportError:
        def log_mem(step=""):
            pass

    if os.path.exists("data"):
        logger.info(f"files: {[(f, os.path.getsize(os.path.join('data', f))) for f in os.listdir('data')]}")
    log_mem("Start")
    logger.info("Loading pipeline artifacts...")
    config = load_config()
    themes_doc = read_json(config.paths.themes)
    insights = read_json(config.paths.insights)
    validation = read_json(config.paths.validation)
    communities = read_json(config.paths.communities)

    llm_insights = None
    if config.paths.llm_insights.exists():
        try:
            llm_insights = read_json(config.paths.llm_insights)
        except Exception:
            logger.warning("Found llm_insights.json but failed to load it", exc_info=True)

    # Optional headline "Key Problems" view - top themes rewritten as problem
    # statements by src/theme_titles.py (Groq). Absent until that stage is run.
    theme_titles = None
    theme_titles_path = config.paths.data_dir / "theme_titles.json"
    if theme_titles_path.exists():
        try:
            theme_titles = read_json(theme_titles_path)
        except Exception:
            logger.warning("Found theme_titles.json but failed to load it", exc_info=True)

    theme_by_id = {t["theme_id"]: t for t in themes_doc["themes"]}

    theme_to_q: Dict[str, List[int]] = {}
    for q in insights["questions"]:
        for tid in q["theme_ids"]:
            theme_to_q.setdefault(tid, []).append(q["question_id"])

    community_to_theme = {t["community_id"]: t["theme_id"] for t in themes_doc["themes"]}
    community_unit_ids: Dict[int, List[str]] = {}
    community_by_id: Dict[int, dict] = {}
    unit_to_community: Dict[str, int] = {}
    for c in communities["communities"]:
        community_by_id[c["community_id"]] = c
        community_unit_ids[c["community_id"]] = c["unit_ids"]
        for uid in c["unit_ids"]:
            unit_to_community[uid] = c["community_id"]

    log_mem("After basic JSONs")
    logger.info("Indexing units.jsonl...")
    unit_by_id: Dict[str, dict] = {}
    for row in _iter_jsonl(config.paths.units):
        unit_by_id[row["unit_id"]] = {
            "text": row["text"],
            "rating": row.get("rating"),
            "date": row.get("date"),
            "source": row.get("source", "google_play"),
            "review_id": row.get("review_id"),
        }

    log_mem("After indexing units")
    logger.info("Loading embeddings + query encoder (%s)...", config.models.embedding_model)
    unit_index = read_json(config.paths.unit_index)
    row_unit_ids: List[str] = unit_index["unit_ids"]
    embeddings = np.load(config.paths.embeddings, mmap_mode="r")
    log_mem("After embeddings mmap")



    logger.info("Counting corpus stats...")
    source_counts: Dict[str, int] = {}
    date_min: Optional[str] = None
    date_max: Optional[str] = None
    unique_reviews = set()
    
    for u in unit_by_id.values():
        rid = u["review_id"]
        if rid not in unique_reviews:
            unique_reviews.add(rid)
            src = u.get("source", "google_play")
            source_counts[src] = source_counts.get(src, 0) + 1
            d = u.get("date")
            if d:
                if date_min is None or d < date_min:
                    date_min = d
                if date_max is None or d > date_max:
                    date_max = d
                    
    num_reviews = len(unique_reviews)

    # Share of units in themes mapped to ≥1 research question (corpus "relevance").
    mapped_theme_ids = {tid for q in insights["questions"] for tid in q.get("theme_ids", [])}
    mapped_units: Set[str] = set()
    for t in themes_doc["themes"]:
        if t["theme_id"] not in mapped_theme_ids:
            continue
        for uid in community_unit_ids.get(t["community_id"], []):
            mapped_units.add(uid)
    relevance_pct = round(100.0 * len(mapped_units) / max(1, len(unit_by_id)), 1)
    
    pipeline_stats = {}
    if config.paths.pipeline_stats.exists():
        try:
            pipeline_stats = read_json(config.paths.pipeline_stats)
        except Exception:
            logger.warning("Found pipeline_stats.json but failed to load it", exc_info=True)

    overview = {
        "num_reviews": num_reviews,
        "num_units": len(unit_by_id),
        "source_distribution": source_counts,
        "date_min": date_min,
        "date_max": date_max,
        "relevance_pct": relevance_pct,
        "pipeline_stats": pipeline_stats,
        **insights["summary"],
        "modularity": validation["summary"]["modularity"],
        "num_cross_segment_stable": validation["summary"]["num_cross_segment_stable"],
        "num_themes_validated": validation["summary"]["num_themes_validated"],
        "has_llm_synthesis": llm_insights is not None,
    }

    # Addressability (addressability-spec.md)
    appux_themes_doc = None
    if config.paths.themes_appux.exists():
        try:
            appux_themes_doc = read_json(config.paths.themes_appux)
        except Exception:
            logger.warning("Found themes_appux.json but failed to load it", exc_info=True)

    classifier_stats = None
    if config.paths.unit_labels.exists():
        from collections import Counter
        from src.schema import read_jsonl
        label_counts = Counter()
        method_counts = Counter()
        total_classified = 0
        try:
            for record in read_jsonl(config.paths.unit_labels):
                label_counts[record.get("label", "unknown")] += 1
                method_counts[record.get("method", "unknown")] += 1
                total_classified += 1
            classifier_stats = {
                "total": total_classified,
                "label_distribution": dict(sorted(label_counts.items())),
                "method_distribution": dict(sorted(method_counts.items())),
            }
        except Exception:
            logger.warning("Failed to load classifier stats", exc_info=True)

    _STATE.update(
        {
            "config": config,
            "themes_doc": themes_doc,
            "appux_themes_doc": appux_themes_doc,
            "classifier_stats": classifier_stats,
            "insights": insights,
            "validation": validation,
            "llm_insights": llm_insights,
            "theme_titles": theme_titles,
            "theme_by_id": theme_by_id,
            "theme_to_q": theme_to_q,
            "community_to_theme": community_to_theme,
            "community_unit_ids": community_unit_ids,
            "community_by_id": community_by_id,
            "unit_to_community": unit_to_community,
            "unit_by_id": unit_by_id,
            "row_unit_ids": row_unit_ids,
            "embeddings": embeddings,
            "overview": overview,
        }
    )
    logger.info(
        "Ready: %d themes, %d units, %d embeddings loaded.",
        len(theme_by_id),
        len(unit_by_id),
        embeddings.shape[0],
    )
    global READY
    READY = True


# --------------------------------------------------------------------------- #
# API routes
# --------------------------------------------------------------------------- #

@app.get("/health")
def get_health():
    if not READY:
        error = getattr(app.state, "startup_error", None)
        if error:
            return {"ready": False, "error": error}
        raise HTTPException(status_code=503, detail="Server is starting up (loading artifacts)...")
    return {"ready": True}

@app.get("/api/health")
def api_health():
    return {"ready": READY}


@app.get("/api/overview")
def get_overview():
    if not READY: raise HTTPException(status_code=503, detail="Still warming up...")
    return _STATE["overview"]


@app.get("/api/appux")
def get_appux():
    """App UX friction map data (addressability-spec.md)."""
    if not READY: raise HTTPException(status_code=503, detail="Still warming up...")
    return {
        "themes_doc": _STATE.get("appux_themes_doc"),
        "classifier_stats": _STATE.get("classifier_stats"),
    }


@app.get("/api/questions")
def get_questions():
    """The 8 canonical research questions with their deterministic pipeline answers
    (insights.json)."""
    if not READY: raise HTTPException(status_code=503, detail="Still warming up...")
    theme_by_id = _STATE["theme_by_id"]
    out = []
    for q in _STATE["insights"]["questions"]:
        out.append(
            {
                "question_id": q["question_id"],
                "text": RESEARCH_QUESTIONS[q["question_id"]],
                "coverage": q["coverage"],
                "total_count": q["total_count"],
                "theme_ids": q["theme_ids"],
                "themes": [
                    {"id": tid, "label": theme_by_id[tid]["label"]}
                    for tid in q["theme_ids"]
                    if tid in theme_by_id
                ],
                "top_verbatims": q["top_verbatims"],
            }
        )
    return out


@app.get("/api/top-themes")
def get_top_themes(limit: int = 5):
    """The top-5 problem-statement themes (src/theme_titles.py output), or an
    empty list if that optional stage hasn't been run."""
    if not READY: raise HTTPException(status_code=503, detail="Still warming up...")
    tt = _STATE.get("theme_titles")
    if not tt:
        return {"themes": [], "available": False}
    themes = []
    for row in tt.get("themes") or []:
        cleaned = dict(row)
        for key in ("title", "connection"):
            if key in cleaned and isinstance(cleaned[key], str):
                cleaned[key] = strip_em_dashes(cleaned[key])
        if cleaned.get("representative_quotes"):
            cleaned["representative_quotes"] = [
                strip_em_dashes(q) if isinstance(q, str) else q
                for q in cleaned["representative_quotes"]
            ]
        themes.append(cleaned)
    return {
        "themes": themes,
        "near_misses": tt.get("near_misses", []),
        "available": True,
        "total_reviews": tt.get("total_reviews"),
        "support_threshold_reviews": tt.get("support_threshold_reviews"),
        "min_share_pct": tt.get("min_share_pct"),
        "model": tt.get("model"),
    }


@app.get("/api/themes")
def get_themes():
    if not READY: raise HTTPException(status_code=503, detail="Still warming up...")
    themes = _STATE["themes_doc"]["themes"]
    theme_to_q = _STATE["theme_to_q"]
    out = []
    for t in sorted(themes, key=lambda x: -x["member_count"]):
        out.append(
            {
                "id": t["theme_id"],
                "label": strip_em_dashes(t["label"]),
                "sentiment": SENTIMENT_UI_MAP.get(t["sentiment"], "pattern"),
                "count": t["member_count"],
                "insight": strip_em_dashes(_build_insight_text(t, theme_to_q)),
                "objective": strip_em_dashes(_synthesize_objective(t)),
                "keywords": [k.strip() for k in t["label"].split("/") if k.strip()],
                "questions": sorted(theme_to_q.get(t["theme_id"], [])),
            }
        )
    return out


@app.get("/api/themes/{theme_id}/citations")
def get_theme_citations(theme_id: str, limit: int = 6):
    if not READY: raise HTTPException(status_code=503, detail="Still warming up...")
    theme = _STATE["theme_by_id"].get(theme_id)
    if not theme:
        raise HTTPException(404, f"Unknown theme_id: {theme_id}")
    unit_ids = _STATE["community_unit_ids"].get(theme["community_id"], [])
    out = []
    seen_reviews = set()
    for uid in unit_ids:
        u = _STATE["unit_by_id"].get(uid)
        if not u or len(u["text"].split()) < MIN_UNIT_WORDS:
            continue
        if u["review_id"] in seen_reviews:
            continue
        seen_reviews.add(u["review_id"])
        out.append(
            {"id": uid, "rating": u["rating"], "date": u["date"], "text": u["text"], "source": u["source"]}
        )
        if len(out) >= limit:
            break
    return out


class AskRequest(BaseModel):
    query: str


def _collect_ask_matches(
    top_idx: np.ndarray,
    sims: np.ndarray,
    *,
    min_sim: float,
) -> Tuple[Dict[str, float], List[dict]]:
    """Score themes and gather candidate units above ``min_sim`` (retrieval core)."""
    theme_by_id = _STATE["theme_by_id"]
    community_to_theme = _STATE["community_to_theme"]
    unit_to_community = _STATE["unit_to_community"]
    unit_by_id = _STATE["unit_by_id"]
    row_unit_ids = _STATE["row_unit_ids"]

    theme_scores: Dict[str, float] = {}
    candidates: List[dict] = []
    seen_reviews: Set[str] = set()
    for idx in top_idx:
        sim = float(sims[idx])
        if sim < min_sim:
            break
        uid = row_unit_ids[idx]
        community_id = unit_to_community.get(uid)
        tid = community_to_theme.get(community_id) if community_id is not None else None
        if tid is None:
            continue
        theme_scores[tid] = theme_scores.get(tid, 0.0) + sim
        u = unit_by_id.get(uid)
        if not u or len(u["text"].split()) < MIN_UNIT_WORDS:
            continue
        review_id = u["review_id"]
        if review_id in seen_reviews:
            continue
        seen_reviews.add(review_id)
        candidates.append(
            {
                "id": uid,
                "rating": u["rating"],
                "date": u["date"],
                "text": u["text"],
                "source": u["source"],
                "theme_id": tid,
                "theme_label": theme_by_id[tid]["label"],
                "similarity": round(sim, 3),
                "review_id": review_id,
            }
        )
    return theme_scores, candidates


def _select_citations_across_themes(
    candidates: List[dict],
    top_theme_ids: List[str],
    *,
    max_reviews: int,
    per_theme: int,
) -> List[dict]:
    """Take up to ``per_theme`` reviews from each top theme, then fill by similarity."""
    citations: List[dict] = []
    seen: Set[str] = set()
    per_counts: Dict[str, int] = {tid: 0 for tid in top_theme_ids}
    top_set = set(top_theme_ids)

    for c in candidates:
        if len(citations) >= max_reviews:
            break
        tid = c["theme_id"]
        if tid not in top_set or per_counts.get(tid, 0) >= per_theme:
            continue
        rid = c["review_id"]
        if rid in seen:
            continue
        seen.add(rid)
        per_counts[tid] = per_counts.get(tid, 0) + 1
        citations.append({k: v for k, v in c.items() if k != "review_id"})

    for c in candidates:
        if len(citations) >= max_reviews:
            break
        rid = c["review_id"]
        if rid in seen:
            continue
        seen.add(rid)
        citations.append({k: v for k, v in c.items() if k != "review_id"})

    return citations


@app.post("/api/ask")
def ask(req: AskRequest):
    if not READY: raise HTTPException(status_code=503, detail="Still warming up...")
    query = req.query.strip()
    if not query:
        return {
            "text": "Ask me something about the review dataset - a behavior, complaint, or category.",
            "citations": [],
        }

    embeddings: np.ndarray = _STATE["embeddings"]
    try:
        q_emb = encode_query(query).astype(embeddings.dtype)
    except Exception as e:
        logger.error(f"Failed to fetch embeddings: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Free-text search is unavailable (embedding API failed).")
    sims = embeddings @ q_emb  # both L2-normalized -> dot product == cosine similarity
    top_idx = np.argsort(-sims)[:80]

    theme_by_id = _STATE["theme_by_id"]

    theme_scores, candidates = _collect_ask_matches(top_idx, sims, min_sim=MIN_ASK_SIMILARITY)

    if not theme_scores:
        top_labels = ", ".join(
            t["label"] for t in sorted(theme_by_id.values(), key=lambda x: -x["member_count"])[:8]
        )
        return {
            "text": (
                "No reviews in the corpus closely match that question. Try asking about one of: "
                f"{top_labels}."
            ),
            "citations": [],
        }

    # Thin context: widen the similarity floor so synthesis still has enough evidence.
    if len(candidates) < ASK_MIN_REVIEWS_FOR_CONTEXT:
        _, candidates = _collect_ask_matches(top_idx, sims, min_sim=ASK_WIDENED_SIMILARITY)

    top_theme_ids = sorted(theme_scores, key=lambda k: -theme_scores[k])[:ASK_MAX_THEMES]
    citations = _select_citations_across_themes(
        candidates,
        top_theme_ids,
        max_reviews=ASK_MAX_REVIEWS,
        per_theme=ASK_REVIEWS_PER_THEME,
    )

    # Prefer theme_titles.json copy (same titles/connections as the Narrative panel)
    # when available; otherwise fall back to the Stage-7 TF-IDF description.
    community_by_id = _STATE["community_by_id"]
    title_by_id = {}
    tt = _STATE.get("theme_titles")
    if tt:
        for row in tt.get("themes") or []:
            title_by_id[row["theme_id"]] = row

    themes_for_synthesis: List[dict] = []
    for tid in top_theme_ids:
        t = theme_by_id[tid]
        titled = title_by_id.get(tid)
        community = community_by_id.get(t["community_id"], {})
        if titled:
            summary = titled.get("title") or ""
            connection = (titled.get("connection") or "").strip()
            if connection:
                summary = f"{summary} - {connection}" if summary else connection
            themes_for_synthesis.append(
                {
                    "theme_summary": summary.strip(),
                    "review_count": titled.get("review_count", t.get("member_count")),
                    "avg_rating": titled.get("avg_rating", community.get("avg_rating")),
                }
            )
        else:
            themes_for_synthesis.append(
                {
                    "theme_summary": (t.get("description") or t.get("label") or "").strip(),
                    "review_count": t.get("member_count"),
                    "avg_rating": community.get("avg_rating"),
                }
            )

    config = _STATE["config"]
    print(
        f"[api.ask] calling synthesize_answer for q={query[:80]!r} "
        f"themes={len(themes_for_synthesis)} citations={len(citations)}",
        flush=True,
    )
    answer = synthesize_answer(
        query,
        themes_for_synthesis,
        citations,
        model=config.llm_synthesis.model,
        seed=config.seed,
    )
    if not answer:
        print("[api.ask] synthesis returned None - using SYNTHESIS_FALLBACK", flush=True)
        answer = SYNTHESIS_FALLBACK
    else:
        print(f"[api.ask] synthesis OK ({len(answer)} chars)", flush=True)

    return {"text": strip_em_dashes(answer), "citations": citations}


# --------------------------------------------------------------------------- #
# Static frontend (web/) - mounted last so it never shadows the /api/* routes above
# --------------------------------------------------------------------------- #

from fastapi.responses import FileResponse

@app.get("/react")
def serve_react_app():
    return FileResponse(str(Path(__file__).parent / "web" / "index.html"))
