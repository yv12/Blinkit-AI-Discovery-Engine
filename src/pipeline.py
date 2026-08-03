"""Pipeline orchestrator: wires stage entry points S1-S9 (+ optional S1b) in order.

Each stage is skippable if its output artifact already exists (pass
--force to rebuild). Stage modules are implemented incrementally per
Implementation-plan.md; until a stage module exists, running it raises
NotImplementedError so `python -m src.pipeline` fails loudly rather than
silently skipping work or producing empty downstream artifacts.

Stage 1b (`scrape_mouthshut`, Docs/context.md §11 Phase 9) is a no-op unless
`data/Mouthshut_reviews.csv` is present - it exists purely to merge in that
optional second source; every other stage's behavior is unaffected by its
presence or absence.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import json
from src import addressability, barrier_mapping, classify, cluster, embed, filter_relevant, graph, insights, normalize, scrape, scrape_mouthshut, summarize, units, validate
from src.config import Config, apply_global_seed, ensure_data_dir, load_config
from src.schema import read_json

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Stage:
    name: str
    output: Callable[[Config], Path]
    run: Callable[[Config, bool], None]  # (config, force) -> None


def _not_implemented(stage_name: str) -> Callable[[Config, bool], None]:
    def _run(_: Config, __: bool = False) -> None:
        raise NotImplementedError(
            f"Stage '{stage_name}' is not implemented yet. See Implementation-plan.md "
            f"for its phase and task list."
        )

    return _run


# Stage order mirrors architecture.md §3 (S1-S9). S10 (UI) is not part of the
# batch pipeline; it reads these artifacts separately. `force`
# is forwarded as each stage's own --refresh flag so a forced re-run actually
# redoes the work rather than relying only on the orchestrator's skip check.
STAGES: List[Stage] = [
    Stage("scrape", lambda c: c.paths.raw_reviews, lambda c, force: scrape.scrape_all(c, refresh=force)),
    # Optional second source (Docs/context.md Addendum "Second data source - Mouthshut"):
    # a no-op if paths.mouthshut_csv is absent, so the pipeline behaves identically to
    # before for anyone who never adds that file.
    Stage(
        "scrape_mouthshut",
        lambda c: c.paths.raw_mouthshut,
        lambda c, force: scrape_mouthshut.ingest_mouthshut(c, refresh=force),
    ),
    Stage("normalize", lambda c: c.paths.reviews, lambda c, force: normalize.normalize_all(c, refresh=force)),
    Stage("units", lambda c: c.paths.units_raw, lambda c, force: units.extract_units(c, refresh=force)),
    # Addressability classifier (addressability-spec.md): classifies each unit into
    # {app_ux, operational, pricing_policy, praise_noise} BEFORE embedding/clustering.
    Stage("classify", lambda c: c.paths.unit_labels, lambda c, force: classify.classify_units(c, refresh=force)),
    # App UX subset re-clustering pipeline: embed → graph → cluster → summarize on app_ux
    # units only, producing separate output artifacts (*_appux.*).
    Stage("addressability", lambda c: c.paths.themes_appux, lambda c, force: addressability.run_appux_pipeline(c, refresh=force)),
    Stage("embed", lambda c: c.paths.embeddings_raw, lambda c, force: embed.embed_units(c, refresh=force)),
    Stage("filter_relevant", lambda c: c.paths.embeddings, lambda c, force: filter_relevant.filter_units(c, refresh=force)),
    Stage("graph", lambda c: c.paths.graph, lambda c, force: graph.build_graph(c, refresh=force)),
    Stage("cluster", lambda c: c.paths.communities, lambda c, force: cluster.detect_communities(c, refresh=force)),
    Stage("summarize", lambda c: c.paths.themes, lambda c, force: summarize.summarize_themes(c, refresh=force)),
    Stage("barrier_mapping", lambda c: c.paths.data_dir / "barrier_mapping.json", lambda c, force: barrier_mapping.map_barriers(c, refresh=force)),
    Stage("insights", lambda c: c.paths.insights, lambda c, force: insights.map_insights(c, refresh=force)),
    Stage("validate", lambda c: c.paths.validation, lambda c, force: validate.validate_pipeline(c, refresh=force)),
]


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if line.strip())


def write_stats(config: Config) -> None:
    try:
        scraped = _count_jsonl(config.paths.raw_reviews) + _count_jsonl(config.paths.raw_mouthshut)
        cleaned = _count_jsonl(config.paths.units_raw)
        in_engine = _count_jsonl(config.paths.units)
        
        relevant = 0
        if config.paths.insights.exists() and config.paths.themes.exists() and config.paths.communities.exists():
            insights_data = read_json(config.paths.insights)
            themes_data = read_json(config.paths.themes)
            communities_data = read_json(config.paths.communities)
            
            mapped_theme_ids = {tid for q in insights_data.get("questions", []) for tid in q.get("theme_ids", [])}
            community_unit_ids = {c["community_id"]: c["unit_ids"] for c in communities_data.get("communities", [])}
            
            mapped_units = set()
            for t in themes_data.get("themes", []):
                if t["theme_id"] in mapped_theme_ids:
                    for uid in community_unit_ids.get(t["community_id"], []):
                        mapped_units.add(uid)
            relevant = len(mapped_units)
            
        stats = {
            "scraped": scraped,
            "cleaned": cleaned,
            "in_engine": in_engine,
            "relevant": relevant
        }
        with config.paths.pipeline_stats.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info("Wrote pipeline stats to %s", config.paths.pipeline_stats)
    except Exception as exc:
        logger.warning("Failed to write pipeline stats: %s", exc)


def run_pipeline(config: Config, force: bool = False, only: Optional[str] = None) -> None:
    """Run stages in order, skipping any whose output artifact already exists.

    edgecases.md X-10: an empty upstream artifact is each stage's own
    responsibility to detect (guard clause on read); the orchestrator only
    handles skip/force/select-stage control flow.
    """
    ensure_data_dir(config)
    apply_global_seed(config.seed)

    if only is not None and only not in {s.name for s in STAGES}:
        raise ValueError(f"Unknown stage '{only}'. Valid stages: {[s.name for s in STAGES]}")

    for stage in STAGES:
        if only and stage.name != only:
            continue
        output_path = stage.output(config)
        if output_path.exists() and not force:
            logger.info("Skipping stage '%s' (artifact exists: %s)", stage.name, output_path)
            continue
        logger.info("Running stage '%s' -> %s", stage.name, output_path)
        stage.run(config, force)

    if only is None:
        write_stats(config)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Discovery Engine pipeline (scrape -> validate)."
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml (default: project root)")
    parser.add_argument(
        "--force", action="store_true", help="Rebuild all selected stages, ignoring cached artifacts"
    )
    parser.add_argument(
        "--only", default=None, choices=[s.name for s in STAGES], help="Run a single stage only"
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config) if args.config else load_config()
    except Exception as exc:  # ConfigError or unexpected load failure
        logger.error("Failed to load config: %s", exc)
        sys.exit(1)

    try:
        run_pipeline(config, force=args.force, only=args.only)
    except NotImplementedError as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
