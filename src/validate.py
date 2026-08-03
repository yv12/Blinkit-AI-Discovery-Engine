"""Stage 9 - Validate theme coherence, cross-segment stability, and a human spot-check sample.

See architecture.md §4 (Stage 9) and edgecases.md "Stage 9 - Validation" for
the full edge-case catalog (S9-xx IDs referenced below). Only the 5
qualifying `themes` are validated here, not the 819 `emerging_signals` -
those are already tagged low-confidence/supplementary evidence as of Stage
7/8 and aren't held to the same rigor.

Three independent checks, each addressing a different question:

1. **Coherence** ("are these clusters actually tight?"): a silhouette-style
   score per theme, computed on *centroid* embeddings (each theme's mean
   member vector), not literal per-point silhouette - infeasible at ~8,600
   members for the largest theme, and per S9-06 a plain silhouette score
   would be a metric mismatch for a graph-clustering method anyway. `a` =
   mean similarity of members to their own theme's centroid; `b` = the
   theme's highest similarity to any *other* theme's centroid (its nearest
   neighbor). `score = (a - b) / max(a, b)`, same shape as sklearn's
   silhouette, in `[-1, 1]`. Alongside it, **graph modularity** of the actual
   Louvain partition is reported as the primary, methodologically-correct
   global metric for a graph-based clustering method (S9-06).
2. **Cross-segment triangulation** ("does this theme hold up, or is it an
   artifact of one narrow slice of the data?"): each theme's member units are
   re-bucketed by rating, time cohort (month), and review length, all
   computed fresh from `communities.json` + `units.jsonl` (not read back from
   `insights.json`) so this stage is an independent check, not just a report
   of Stage 8's own numbers. A theme is tagged `"segment_specific"` if any
   single time-cohort or length-bucket holds >= `validation.dominant_share_threshold`
   of its members; **rating bands are deliberately excluded from this
   judgment** - for a sentiment-driven theme, rating concentration (e.g. a
   "customer service complaints" theme skewing 1-star) is an expected,
   healthy signal, not evidence of instability (S9-05).
3. **Spot-check sample** ("would a human agree with these groupings?"): a
   stratified random sample of member units per theme, written once to
   `data/spot_check_sample.json` with an empty `human_agrees` field for
   manual review. Never regenerated automatically once it exists, so manual
   labels are never silently overwritten (S9-04) - delete the file yourself
   to force a fresh sample. Agreement is computed over whatever fraction has
   actually been labeled, reported honestly as `null` if none has.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.config import Config, apply_global_seed, ensure_data_dir, load_config
from src.insights import _theme_segment_stats
from src.schema import Unit, read_json, read_jsonl, write_json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


class ValidateError(RuntimeError):
    """Raised for unrecoverable validation failures (e.g. missing inputs, empty themes)."""


def _atomic_write_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _length_bucket(word_count: int, config: Config) -> str:
    if word_count <= config.validation.short_unit_max_words:
        return "short"
    if word_count <= config.validation.medium_unit_max_words:
        return "medium"
    return "long"


def _theme_centroids(
    themes: List[dict],
    communities_by_id: Dict[int, dict],
    id_to_row: Dict[str, int],
    embeddings: np.ndarray,
) -> np.ndarray:
    centroids = np.zeros((len(themes), embeddings.shape[1]), dtype=np.float32)
    for i, theme in enumerate(themes):
        community = communities_by_id.get(theme["community_id"])
        rows = [id_to_row[uid] for uid in community["unit_ids"] if uid in id_to_row] if community else []
        if not rows:
            continue
        centroid = embeddings[rows].mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroids[i] = centroid / norm if norm > 0 else centroid
    return centroids


def _coherence(
    themes: List[dict],
    communities_by_id: Dict[int, dict],
    id_to_row: Dict[str, int],
    embeddings: np.ndarray,
    graph,
    communities_doc: dict,
) -> dict:
    centroids = _theme_centroids(themes, communities_by_id, id_to_row, embeddings)
    cross_sims = centroids @ centroids.T
    np.fill_diagonal(cross_sims, -np.inf)  # exclude self when finding nearest *other* theme

    theme_scores = []
    for i, theme in enumerate(themes):
        community = communities_by_id.get(theme["community_id"])
        rows = [id_to_row[uid] for uid in community["unit_ids"] if uid in id_to_row] if community else []
        if len(rows) < 2:
            # S9-03: coherence undefined for a singleton - skip, don't divide by zero.
            theme_scores.append(
                {"theme_id": theme["theme_id"], "size": len(rows), "intra_similarity": None,
                 "nearest_theme_id": None, "nearest_theme_similarity": None, "silhouette_score": None}
            )
            continue
        a = float((embeddings[rows] @ centroids[i]).mean())
        nearest_j = int(np.argmax(cross_sims[i]))
        b = float(cross_sims[i, nearest_j])
        score = (a - b) / max(abs(a), abs(b), 1e-9)
        theme_scores.append(
            {
                "theme_id": theme["theme_id"],
                "size": len(rows),
                "intra_similarity": round(a, 3),
                "nearest_theme_id": themes[nearest_j]["theme_id"],
                "nearest_theme_similarity": round(b, 3),
                "silhouette_score": round(score, 3),
            }
        )

    try:
        import community as community_louvain  # python-louvain

        partition = {
            uid: c["community_id"] for c in communities_doc["communities"] for uid in c["unit_ids"]
        }
        modularity = float(community_louvain.modularity(partition, graph, weight="weight"))
    except ImportError:
        modularity = None
        logger.warning("python-louvain not importable; skipping modularity computation.")

    scored = [t["silhouette_score"] for t in theme_scores if t["silhouette_score"] is not None]
    return {
        "modularity": round(modularity, 4) if modularity is not None else None,
        "mean_silhouette_score": round(float(np.mean(scored)), 3) if scored else None,
        "themes": theme_scores,
    }


def _triangulation(
    themes: List[dict],
    communities_by_id: Dict[int, dict],
    unit_meta: Dict[str, Unit],
    config: Config,
) -> dict:
    segment_stats = _theme_segment_stats(themes, communities_by_id, unit_meta)  # rating + time, S8's own helper

    corpus_rating_bands: Counter = Counter()
    corpus_time_cohorts: Counter = Counter()
    corpus_sources: Counter = Counter()
    threshold = config.validation.dominant_share_threshold

    theme_results = []
    for theme in themes:
        community = communities_by_id.get(theme["community_id"])
        stats = segment_stats.get(
            theme["theme_id"],
            {"rating_distribution": {}, "time_cohort_distribution": {}, "source_distribution": {}},
        )
        corpus_rating_bands.update(stats["rating_distribution"])
        corpus_time_cohorts.update(stats["time_cohort_distribution"])
        corpus_sources.update(stats.get("source_distribution", {}))

        length_buckets: Counter = Counter()
        total = 0
        for unit_id in (community["unit_ids"] if community else []):
            unit = unit_meta.get(unit_id)
            if unit is None:
                continue
            length_buckets[_length_bucket(len(unit.text.split()), config)] += 1
            total += 1

        def _dominant_share(counter: Counter) -> float:
            return (max(counter.values()) / total) if total and counter else 0.0

        time_dominant_share = _dominant_share(Counter(stats["time_cohort_distribution"]))
        length_dominant_share = _dominant_share(length_buckets)
        reasons = []
        if time_dominant_share >= threshold:
            reasons.append("concentrated in a single time cohort")
        if length_dominant_share >= threshold:
            reasons.append("concentrated in a single review-length bucket")
        stability = "segment_specific" if reasons else "cross_segment"

        theme_results.append(
            {
                "theme_id": theme["theme_id"],
                "rating_distribution": stats["rating_distribution"],
                "time_cohort_distribution": stats["time_cohort_distribution"],
                "length_bucket_distribution": dict(length_buckets),
                # Informational only, same rationale as rating bands (S9-05) - see
                # insights.py's _theme_segment_stats for why this never affects `stability`.
                "source_distribution": stats.get("source_distribution", {}),
                "time_dominant_share": round(time_dominant_share, 3),
                "length_dominant_share": round(length_dominant_share, 3),
                "stability": stability,
                "stability_reasons": reasons,
            }
        )

    return {
        "dominant_share_threshold": threshold,
        "corpus_rating_bands_observed": dict(sorted(corpus_rating_bands.items())),  # S9-01
        "corpus_time_cohorts_observed": dict(sorted(corpus_time_cohorts.items())),  # S9-02
        "corpus_sources_observed": dict(sorted(corpus_sources.items())),
        "themes": theme_results,
    }


def _load_or_create_spot_check_sample(
    themes: List[dict],
    communities_by_id: Dict[int, dict],
    unit_meta: Dict[str, Unit],
    config: Config,
) -> List[dict]:
    if config.paths.spot_check_sample.exists():
        logger.info(
            "'%s' already exists; reusing it (never auto-regenerated so manual labels aren't "
            "lost - delete it yourself to force a fresh sample).",
            config.paths.spot_check_sample,
        )
        return read_json(config.paths.spot_check_sample)

    rng = random.Random(config.seed)
    sample_size = config.validation.spot_check_sample_size_per_theme
    rows = []
    for theme in themes:
        community = communities_by_id.get(theme["community_id"])
        unit_ids = community["unit_ids"] if community else []
        picked = rng.sample(unit_ids, min(sample_size, len(unit_ids)))
        for unit_id in picked:
            unit = unit_meta.get(unit_id)
            if unit is None:
                continue
            rows.append(
                {
                    "theme_id": theme["theme_id"],
                    "theme_label": theme["label"],
                    "unit_id": unit.unit_id,
                    "review_id": unit.review_id,
                    "text": unit.text,
                    "rating": unit.rating,
                    "date": unit.date,
                    # For a human reviewer to fill in by hand (S9-04): does this unit genuinely
                    # belong to this theme? true / false / null (not yet reviewed).
                    "human_agrees": None,
                    "human_note": "",
                }
            )
    write_json(config.paths.spot_check_sample, rows)
    logger.info("Wrote a fresh spot-check sample: %d rows across %d themes.", len(rows), len(themes))
    return rows


def _spot_check_summary(sample: List[dict]) -> dict:
    labeled = [row for row in sample if row.get("human_agrees") is not None]
    agree = sum(1 for row in labeled if row["human_agrees"] is True)
    return {
        "sample_file": "spot_check_sample.json",
        "sample_size": len(sample),
        "labeled_count": len(labeled),
        "agreement_rate": round(agree / len(labeled), 3) if labeled else None,
        "note": (
            "No human labels present yet - open data/spot_check_sample.json, fill in "
            "`human_agrees` (true/false) per row, then re-run `python -m src.validate --refresh` "
            "to compute agreement (S9-04)."
            if not labeled
            else f"{len(labeled)}/{len(sample)} rows labeled so far."
        ),
    }


def validate_pipeline(config: Config, refresh: bool = False) -> None:
    ensure_data_dir(config)
    apply_global_seed(config.seed)

    if config.paths.validation.exists() and not refresh:
        logger.info(
            "'%s' already exists; skipping validation (pass --refresh to rebuild).",
            config.paths.validation,
        )
        return

    required = (
        config.paths.themes, config.paths.communities, config.paths.units,
        config.paths.embeddings, config.paths.unit_index, config.paths.graph,
    )
    for path in required:
        if not path.exists():
            raise ValidateError(
                f"Expected artifact at {path} but it does not exist. Run earlier pipeline stages "
                "first (edgecases.md X-03)."
            )

    themes_doc = read_json(config.paths.themes)
    themes: List[dict] = themes_doc.get("themes", [])
    if not themes:
        raise ValidateError("themes.json has zero themes - nothing to validate (S9-07/X-10).")

    communities_doc = read_json(config.paths.communities)
    communities_by_id = {c["community_id"]: c for c in communities_doc["communities"]}
    unit_meta: Dict[str, Unit] = {u.unit_id: u for u in read_jsonl(config.paths.units, factory=Unit)}
    embeddings = np.load(config.paths.embeddings)
    unit_index = read_json(config.paths.unit_index)
    id_to_row = {uid: i for i, uid in enumerate(unit_index["unit_ids"])}
    with open(config.paths.graph, "rb") as f:
        graph = pickle.load(f)

    coherence = _coherence(themes, communities_by_id, id_to_row, embeddings, graph, communities_doc)
    triangulation = _triangulation(themes, communities_by_id, unit_meta, config)
    spot_check_sample = _load_or_create_spot_check_sample(themes, communities_by_id, unit_meta, config)
    spot_check = _spot_check_summary(spot_check_sample)

    # Addressability classifier stats (addressability-spec.md §5) — optional,
    # only present if the classify stage has been run.
    classifier_stats = None
    if config.paths.unit_labels.exists():
        from collections import Counter as _Counter
        label_counts: _Counter = _Counter()
        method_counts: _Counter = _Counter()
        total_classified = 0
        for record in read_jsonl(config.paths.unit_labels):
            label_counts[record.get("label", "unknown")] += 1
            method_counts[record.get("method", "unknown")] += 1
            total_classified += 1
        classifier_stats = {
            "total_classified": total_classified,
            "label_distribution": dict(sorted(label_counts.items())),
            "method_distribution": dict(sorted(method_counts.items())),
            "classification_spot_check_file": (
                str(config.paths.classification_spot_check.name)
                if config.paths.classification_spot_check.exists()
                else None
            ),
        }
        logger.info(
            "Classifier stats: %d units, distribution: %s",
            total_classified, dict(label_counts),
        )

    num_cross_segment = sum(1 for t in triangulation["themes"] if t["stability"] == "cross_segment")
    num_segment_specific = len(triangulation["themes"]) - num_cross_segment

    payload = {
        "coherence": coherence,
        "triangulation": triangulation,
        "spot_check": spot_check,
        "classifier": classifier_stats,
        "summary": {
            "num_themes_validated": len(themes),
            "modularity": coherence["modularity"],
            "mean_silhouette_score": coherence["mean_silhouette_score"],
            "num_cross_segment_stable": num_cross_segment,
            "num_segment_specific": num_segment_specific,
            "spot_check_labeled_count": spot_check["labeled_count"],
            "spot_check_agreement_rate": spot_check["agreement_rate"],
        },
    }
    write_json(config.paths.validation, payload)
    _write_human_readable_summary(config, payload, themes)
    _log_summary(payload)


def _write_human_readable_summary(config: Config, payload: dict, themes: List[dict]) -> None:
    themes_by_id = {t["theme_id"]: t for t in themes}
    s = payload["summary"]
    lines = [
        "# Validation Summary (Stage 9)",
        "",
        f"Themes validated: {s['num_themes_validated']}",
        f"Graph modularity (Louvain partition quality): {s['modularity']}",
        f"Mean theme silhouette-style score (centroid-based, [-1, 1]): {s['mean_silhouette_score']}",
        f"Cross-segment stable themes: {s['num_cross_segment_stable']} / {s['num_themes_validated']}",
        f"Segment-specific themes: {s['num_segment_specific']} / {s['num_themes_validated']}",
        f"Spot-check labels collected so far: {s['spot_check_labeled_count']}",
        f"Spot-check agreement rate: {s['spot_check_agreement_rate']}",
        "",
        "## Coherence (sorted by silhouette-style score, ascending = least coherent first)",
        "",
        "| theme_id | label | size | intra_sim | nearest_theme | nearest_sim | silhouette |",
        "|---|---|---|---|---|---|---|",
    ]
    sorted_themes = sorted(
        payload["coherence"]["themes"],
        key=lambda t: (t["silhouette_score"] is None, t["silhouette_score"] or 0.0),
    )
    for t in sorted_themes:
        label = themes_by_id.get(t["theme_id"], {}).get("label", "")
        lines.append(
            f"| {t['theme_id']} | {label} | {t['size']} | {t['intra_similarity']} | "
            f"{t['nearest_theme_id']} | {t['nearest_theme_similarity']} | {t['silhouette_score']} |"
        )

    lines += [
        "",
        "## Segment-specific themes (concentrated in one time cohort or length bucket)",
        "",
    ]
    specific = [t for t in payload["triangulation"]["themes"] if t["stability"] == "segment_specific"]
    if specific:
        for t in specific:
            label = themes_by_id.get(t["theme_id"], {}).get("label", "")
            lines.append(f"- **{t['theme_id']}** ({label}): {', '.join(t['stability_reasons'])}")
    else:
        lines.append("- None - every theme spans multiple time cohorts and review-length buckets.")

    lines += ["", "## Spot-check", "", payload["spot_check"]["note"]]

    summary_path = config.paths.validation.with_name("validation_summary.md")
    _atomic_write_text(summary_path, "\n".join(lines) + "\n")


def _log_summary(payload: dict) -> None:
    s = payload["summary"]
    logger.info(
        "Validation complete: modularity=%s, mean_silhouette=%s, cross_segment=%d/%d, spot_check_labeled=%d.",
        s["modularity"], s["mean_silhouette_score"], s["num_cross_segment_stable"],
        s["num_themes_validated"], s["spot_check_labeled_count"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate theme coherence, cross-segment stability, and spot-check sample (Stage 9)."
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: project root)")
    parser.add_argument(
        "--refresh", action="store_true", help="Rebuild validation.json even if it already exists"
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config) if args.config else load_config()
    except Exception as exc:
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    try:
        validate_pipeline(config, refresh=args.refresh)
    except ValidateError as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
