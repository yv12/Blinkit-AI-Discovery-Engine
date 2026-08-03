"""App UX subset re-clustering pipeline (addressability-spec.md §2, §3).

Orchestrates a parallel embed → graph → cluster → summarize pipeline on only
the `app_ux`-labeled units (from classify.py), producing separate output
artifacts that coexist with the full-corpus outputs. The existing pipeline
outputs are never modified or overwritten.

The summarize step adds a `journey_stage` field to each theme, classifying
where in the user journey the friction occurs (addressability-spec.md §3).

All code in this module calls into the existing stage modules' internal
functions where possible, but reads from / writes to the app_ux-specific paths.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import Config, apply_global_seed, ensure_data_dir, load_config
from src.schema import Unit, read_json, read_jsonl, write_json, write_jsonl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

VALID_JOURNEY_STAGES = frozenset({
    "search", "browse_discover", "recommendations",
    "product_page", "cart_checkout", "post_order",
})

# ─── Journey stage heuristic (keyword-based fallback) ────────────────────────

_JOURNEY_HEURISTICS = [
    ("search", [
        r"\bsearch", r"\bfind\b", r"\bfilter", r"\bsort\b",
        r"\bquery", r"\bresult", r"\bshow\b.*\bitem",
    ]),
    ("browse_discover", [
        r"\bbrowse", r"\bcategor", r"\bexplore", r"\bnavigate",
        r"\bdiscover", r"\bmenu\b", r"\bsection",
    ]),
    ("recommendations", [
        r"\brecommend", r"\bsuggestion", r"\bpersonali",
        r"\brecently\s*viewed", r"\breorder", r"\brepeat\s*order",
        r"\bad(?:s|vert)", r"\bbanner", r"\bpopup",
    ]),
    ("product_page", [
        r"\bproduct\s*(?:page|detail|info|image|photo)",
        r"\bexpiry", r"\bdescription", r"\breview\b.*\bproduct",
        r"\brating\b.*\bproduct", r"\bingredient",
    ]),
    ("cart_checkout", [
        r"\bcart\b", r"\bcheckout", r"\bpayment",
        r"\bcoupon", r"\bpromo\b", r"\bwallet\b", r"\botp\b",
    ]),
    ("post_order", [
        r"\bnotif", r"\bspam\b", r"\bpush\s*notification",
        r"\btrack", r"\border\s*status",
    ]),
]


def _heuristic_journey_stage(texts: List[str]) -> str:
    """Guess journey stage from keyword frequency across theme member texts."""
    import re
    combined = " ".join(texts).lower()
    scores: Dict[str, int] = {}
    for stage, patterns in _JOURNEY_HEURISTICS:
        count = sum(len(re.findall(p, combined, re.IGNORECASE)) for p in patterns)
        scores[stage] = count
    if max(scores.values(), default=0) == 0:
        return "browse_discover"  # safe default for app UX
    return max(scores, key=scores.get)


# ─── Step 1: Filter units to app_ux ─────────────────────────────────────────

def _filter_appux_units(config: Config) -> List[Unit]:
    """Read unit_labels.jsonl and units_raw.jsonl, return only app_ux units."""
    if not config.paths.unit_labels.exists():
        raise RuntimeError(
            f"unit_labels.jsonl not found at {config.paths.unit_labels}. "
            "Run `python -m src.classify` first."
        )

    # Build label lookup
    labels: Dict[str, str] = {}
    for record in read_jsonl(config.paths.unit_labels):
        labels[record["unit_id"]] = record["label"]

    # Filter units
    all_units = list(read_jsonl(config.paths.units_raw, factory=Unit))
    appux_units = [u for u in all_units if labels.get(u.unit_id) == "app_ux"]

    logger.info(
        "App UX filter: %d / %d units are app_ux (%.1f%%)",
        len(appux_units), len(all_units),
        100.0 * len(appux_units) / max(1, len(all_units)),
    )
    return appux_units


# ─── Step 2: Embed ──────────────────────────────────────────────────────────

def _embed_appux(units: List[Unit], config: Config) -> Tuple[np.ndarray, List[str]]:
    """Embed app_ux units using the same model as the main pipeline."""
    from src.embed import _load_model, _atomic_save_npy

    if (config.paths.embeddings_appux.exists() and
            config.paths.unit_index_appux.exists()):
        logger.info("App UX embeddings already exist; loading cached.")
        embeddings = np.load(config.paths.embeddings_appux)
        index = read_json(config.paths.unit_index_appux)
        return embeddings, index["unit_ids"]

    model = _load_model(config.models.embedding_model)
    texts = [u.text for u in units]
    unit_ids = [u.unit_id for u in units]

    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=config.models.embed_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    elapsed = time.time() - start

    _atomic_save_npy(config.paths.embeddings_appux, embeddings)
    index_payload = {
        "model": config.models.embedding_model,
        "dim": int(embeddings.shape[1]),
        "count": int(embeddings.shape[0]),
        "metric": "cosine",
        "seed": config.seed,
        "unit_ids": unit_ids,
    }
    write_json(config.paths.unit_index_appux, index_payload)

    logger.info(
        "App UX embedding complete: %d units → %s (dim=%d) in %.1fs.",
        len(units), config.paths.embeddings_appux.name,
        embeddings.shape[1], elapsed,
    )
    return embeddings, unit_ids


# ─── Step 3: Build graph ────────────────────────────────────────────────────

def _build_appux_graph(
    embeddings: np.ndarray,
    unit_ids: List[str],
    unit_meta: Dict[str, Unit],
    config: Config,
) -> None:
    """Build kNN similarity graph for app_ux units."""
    if config.paths.graph_appux.exists():
        logger.info("App UX graph already exists; skipping.")
        return

    import networkx as nx

    n = embeddings.shape[0]
    k = min(config.graph.knn_k, n - 1)
    if k < 1:
        raise RuntimeError(f"Cannot build graph with only {n} app_ux unit(s).")

    # kNN search
    try:
        from src.graph import _knn_faiss
        similarities, indices = _knn_faiss(embeddings, k)
    except ImportError:
        from src.graph import _knn_sklearn
        similarities, indices = _knn_sklearn(embeddings, k)

    threshold = config.graph.similarity_threshold
    graph = nx.Graph()
    for i, uid in enumerate(unit_ids):
        meta = unit_meta.get(uid)
        graph.add_node(
            uid,
            review_id=meta.review_id if meta else None,
            rating=meta.rating if meta else None,
        )

    edge_count = 0
    for i in range(n):
        for j_pos in range(similarities.shape[1]):
            j = int(indices[i, j_pos])
            if j == i:
                continue
            sim = float(min(similarities[i, j_pos], 1.0))
            if sim >= threshold:
                uid_a, uid_b = unit_ids[i], unit_ids[j]
                if graph.has_edge(uid_a, uid_b):
                    if sim > graph[uid_a][uid_b].get("weight", 0.0):
                        graph[uid_a][uid_b]["weight"] = sim
                else:
                    graph.add_edge(uid_a, uid_b, weight=sim)
                    edge_count += 1

    if edge_count == 0:
        raise RuntimeError(
            f"App UX graph has zero edges at threshold={threshold}. "
            "Lower graph.similarity_threshold or check app_ux unit count."
        )

    # Atomic pickle
    from src.graph import _atomic_pickle_dump
    _atomic_pickle_dump(config.paths.graph_appux, graph)
    logger.info("App UX graph: %d nodes, %d edges.", graph.number_of_nodes(), graph.number_of_edges())


# ─── Step 4: Cluster ────────────────────────────────────────────────────────

def _cluster_appux(unit_meta: Dict[str, Unit], config: Config) -> dict:
    """Run Louvain community detection on the app_ux graph."""
    if config.paths.communities_appux.exists():
        logger.info("App UX communities already exist; loading cached.")
        return read_json(config.paths.communities_appux)

    with open(config.paths.graph_appux, "rb") as f:
        graph = pickle.load(f)

    import community as community_louvain

    partition = community_louvain.best_partition(
        graph,
        resolution=config.clustering.louvain_resolution,
        random_state=config.seed,
        weight="weight",
    )

    # Group by community
    groups: Dict[int, List[str]] = {}
    for uid, cid in partition.items():
        groups.setdefault(cid, []).append(uid)

    # Sort by size descending
    sorted_groups = sorted(groups.items(), key=lambda x: (-len(x[1]), min(x[1])))

    communities = []
    for new_id, (_, members) in enumerate(sorted_groups):
        ratings = [unit_meta[uid].rating for uid in members if uid in unit_meta and unit_meta[uid].rating is not None]
        avg_rating = round(mean(ratings), 2) if ratings else None
        lang_counts = dict(Counter(unit_meta[uid].lang for uid in members if uid in unit_meta))

        communities.append({
            "community_id": new_id,
            "size": len(members),
            "unit_ids": sorted(members),
            "avg_rating": avg_rating,
            "lang_counts": lang_counts,
            "below_min_size": len(members) < config.clustering.min_community_size,
        })

    payload = {
        "num_communities": len(communities),
        "num_qualifying": sum(1 for c in communities if not c["below_min_size"]),
        "num_below_min": sum(1 for c in communities if c["below_min_size"]),
        "communities": communities,
    }
    write_json(config.paths.communities_appux, payload)
    logger.info(
        "App UX clustering: %d communities (%d qualifying, %d below min).",
        len(communities), payload["num_qualifying"], payload["num_below_min"],
    )
    return payload


# ─── Step 5: Summarize with journey_stage ───────────────────────────────────

_APPUX_SUMMARY_PROMPT = """You are analyzing clusters of Blinkit (Indian quick-commerce app) Play Store
reviews that are specifically about IN-APP UX FRICTION (not delivery/operations).
The goal is to identify what app-level issues block users from exploring new categories.

Given the representative review excerpts below, respond with strict JSON:
{{
  "label": "a concise 3-8 word theme label",
  "description": "one sentence summarizing the shared UX friction",
  "journey_stage": one of "search" | "browse_discover" | "recommendations" | "product_page" | "cart_checkout" | "post_order"
}}

Journey stage definitions:
- search: friction with search functionality (results quality, filters, sorting)
- browse_discover: friction browsing categories, navigating the app, discovering new products
- recommendations: repetitive/irrelevant suggestions, ads crowding results, recently-viewed flooding
- product_page: missing product info (expiry, reviews, ingredients), poor images/descriptions
- cart_checkout: cart issues, checkout bugs, payment problems, coupon issues
- post_order: notification spam, order tracking issues, post-purchase friction

Do not invent details not in the excerpts. Candidate keywords: {keywords}.

REVIEW EXCERPTS:
{excerpts}"""


def _summarize_appux_themes(
    communities_doc: dict,
    unit_meta: Dict[str, Unit],
    embeddings: np.ndarray,
    unit_ids: List[str],
    config: Config,
) -> None:
    """Summarize app_ux communities into themes with journey_stage."""
    if config.paths.themes_appux.exists():
        logger.info("App UX themes already exist; skipping.")
        return

    id_to_row = {uid: i for i, uid in enumerate(unit_ids)}
    communities = communities_doc["communities"]
    qualifying = [c for c in communities if not c["below_min_size"]]

    if not qualifying:
        logger.warning("No qualifying app_ux communities (all below min size).")
        qualifying = communities[:10]  # Take top 10 by size as fallback

    from src.summarize import (
        _representative_unit_ids, _tfidf_top_terms, _truncate,
        _sentiment_from_rating, _confidence_tier,
    )

    themes = []
    for community in qualifying:
        cid = community["community_id"]
        member_ids = community["unit_ids"]
        texts = [unit_meta[uid].text for uid in member_ids if uid in unit_meta]

        if not texts:
            continue

        # Representative selection
        rep_ids = _representative_unit_ids(member_ids, id_to_row, embeddings, config.summarize.max_representatives)
        rep_texts = [unit_meta[uid].text for uid in rep_ids if uid in unit_meta]

        # TF-IDF label
        top_terms = _tfidf_top_terms(texts, config.summarize.max_tfidf_terms)
        label = " / ".join(top_terms) if top_terms else f"App UX Community {cid}"
        description = (
            f"{community['size']} reviews (avg rating "
            f"{community['avg_rating']:.1f})" if community["avg_rating"] is not None
            else f"{community['size']} reviews"
        ) + (f" recurring around: {', '.join(top_terms)}." if top_terms else ".")

        quotes = [_truncate(t) for t in rep_texts[:config.summarize.max_quotes]]
        sentiment = _sentiment_from_rating(community["avg_rating"])

        # Journey stage via heuristic
        journey_stage = _heuristic_journey_stage(texts)

        # Try LLM for better label + journey_stage
        llm_label, llm_desc, llm_stage = _try_llm_summary(
            rep_texts, ", ".join(top_terms), config
        )
        if llm_label:
            label = llm_label
        if llm_desc:
            description = llm_desc
        if llm_stage and llm_stage in VALID_JOURNEY_STAGES:
            journey_stage = llm_stage

        themes.append({
            "theme_id": f"appux-{cid:04d}",
            "community_id": cid,
            "label": label,
            "description": description,
            "representative_quotes": quotes,
            "member_count": community["size"],
            "sentiment": sentiment,
            "journey_stage": journey_stage,
            "questions": [],
        })

    # Also process long-tail as emerging signals
    long_tail = [c for c in communities if c["below_min_size"]]
    emerging_signals = []
    for community in long_tail:
        member_ids = community["unit_ids"]
        texts = [unit_meta[uid].text for uid in member_ids if uid in unit_meta]
        if not texts:
            continue
        longest = max(texts, key=len)
        emerging_signals.append({
            "signal_id": f"appux-signal-{community['community_id']:04d}",
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
            "journey_stage": _heuristic_journey_stage(texts),
        })

    payload = {
        "themes": themes,
        "emerging_signals": emerging_signals,
        "summary": {
            "num_themes": len(themes),
            "num_emerging_signals": len(emerging_signals),
            "journey_stage_distribution": dict(Counter(t["journey_stage"] for t in themes)),
        },
    }
    write_json(config.paths.themes_appux, payload)
    logger.info(
        "App UX summarization: %d themes, %d emerging signals.",
        len(themes), len(emerging_signals),
    )
    for stage, count in sorted(Counter(t["journey_stage"] for t in themes).items()):
        logger.info("  journey_stage=%s: %d themes", stage, count)


def _try_llm_summary(
    rep_texts: List[str],
    keywords: str,
    config: Config,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Try Groq LLM for better theme label, description, and journey_stage."""
    try:
        from src.llm_synthesis import _call_groq, _load_api_key, _RateLimiter
        api_key = _load_api_key()
        limiter = _RateLimiter(requests_per_minute=25)

        excerpts = "\n".join(f"- {t[:300]}" for t in rep_texts)
        prompt = _APPUX_SUMMARY_PROMPT.replace("{keywords}", keywords).replace("{excerpts}", excerpts)

        data = _call_groq(
            prompt=prompt,
            model=config.llm_synthesis.model,
            api_key=api_key,
            temperature=0.0,
            seed=config.seed,
            max_retries=3,
            limiter=limiter,
        )
        if data and isinstance(data, dict):
            return (
                data.get("label"),
                data.get("description"),
                data.get("journey_stage"),
            )
    except Exception as exc:
        logger.warning("LLM summary failed (%s); using heuristic fallback.", exc)

    return None, None, None


# ─── Main orchestrator ──────────────────────────────────────────────────────

def run_appux_pipeline(config: Config, refresh: bool = False) -> None:
    """Run the full app_ux re-clustering pipeline."""
    ensure_data_dir(config)
    apply_global_seed(config.seed)

    if config.paths.themes_appux.exists() and not refresh:
        logger.info(
            "'%s' already exists; skipping app_ux pipeline (pass --refresh to rebuild).",
            config.paths.themes_appux,
        )
        return

    # Step 1: Filter to app_ux units
    appux_units = _filter_appux_units(config)
    if not appux_units:
        logger.warning("No app_ux units found. Skipping app_ux pipeline.")
        # Write empty output so pipeline doesn't re-run
        write_json(config.paths.themes_appux, {
            "themes": [], "emerging_signals": [],
            "summary": {"num_themes": 0, "num_emerging_signals": 0, "journey_stage_distribution": {}},
        })
        return

    # Write filtered units
    write_jsonl(config.paths.units_appux, appux_units)
    unit_meta = {u.unit_id: u for u in appux_units}

    # Step 2: Embed
    if refresh:
        # Delete cached artifacts on refresh
        for p in (config.paths.embeddings_appux, config.paths.unit_index_appux,
                  config.paths.graph_appux, config.paths.communities_appux,
                  config.paths.themes_appux):
            if p.exists():
                p.unlink()

    embeddings, unit_ids = _embed_appux(appux_units, config)

    # Step 3: Graph
    _build_appux_graph(embeddings, unit_ids, unit_meta, config)

    # Step 4: Cluster
    communities_doc = _cluster_appux(unit_meta, config)

    # Step 5: Summarize with journey_stage
    _summarize_appux_themes(communities_doc, unit_meta, embeddings, unit_ids, config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the app_ux subset re-clustering pipeline (addressability-spec.md)."
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--refresh", action="store_true", help="Rebuild all app_ux artifacts")
    args = parser.parse_args()

    try:
        config = load_config(args.config) if args.config else load_config()
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    try:
        run_appux_pipeline(config, refresh=args.refresh)
    except Exception as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
