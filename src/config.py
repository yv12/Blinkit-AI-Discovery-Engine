"""Central configuration loader for the Discovery Engine pipeline.

Loads and validates ``config.yaml`` so a bad value fails loudly at startup
rather than corrupting a downstream stage (edgecases.md X-08). Paths are
resolved via ``pathlib`` for cross-platform correctness (edgecases.md R-03),
and the global random seed is applied to every RNG the pipeline touches
(edgecases.md R-06).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

SCHEMA_VERSION = 1


class ConfigError(ValueError):
    """Raised when config.yaml is missing, malformed, or has an invalid value."""


@dataclass(frozen=True)
class AppConfig:
    app_id: str
    country: str
    lang: str


@dataclass(frozen=True)
class ScrapeConfig:
    lookback_months: int
    max_per_bucket: int
    sorts: list


@dataclass(frozen=True)
class UnitsConfig:
    min_words: int
    min_words_per_unit: int
    max_units_per_review: int
    use_llm: bool


@dataclass(frozen=True)
class ModelsConfig:
    embedding_model: str
    llm_model: str
    embed_batch_size: int


@dataclass(frozen=True)
class GraphConfig:
    knn_k: int
    similarity_threshold: float


@dataclass(frozen=True)
class ClusteringConfig:
    louvain_resolution: float
    min_community_size: int


@dataclass(frozen=True)
class SummarizeConfig:
    use_llm: bool
    max_representatives: int
    max_quotes: int
    max_tfidf_terms: int
    long_tail_llm_batch_size: int


@dataclass(frozen=True)
class InsightsConfig:
    similarity_threshold: float
    top_themes_count: int
    max_verbatims_per_question: int
    question_queries: list
    question_required_sentiment: list


@dataclass(frozen=True)
class ValidationConfig:
    spot_check_sample_size_per_theme: int
    dominant_share_threshold: float
    short_unit_max_words: int
    medium_unit_max_words: int


@dataclass(frozen=True)
class LLMSynthesisConfig:
    """Stage 11 (optional, opt-in, not part of `python -m src.pipeline`) - see config.yaml
    for the full rationale. Requires GROQ_API_KEY in `.env`; never fatal to the core
    pipeline if this section is simply unused."""

    model: str
    temperature: float
    sample_size: int
    batch_size: int
    requests_per_minute: int
    max_retries: int
    pattern_cluster_threshold: float
    max_patterns_per_question: int
    max_quotes_per_pattern: int
    min_pattern_support: int
    min_quote_words: int


@dataclass(frozen=True)
class FilteringConfig:
    min_relevance_score: float
    min_theme_relevance_score: float
    max_theme_rating: int


@dataclass(frozen=True)
class AddressabilityConfig:
    """Addressability classifier (addressability-spec.md): classifies units into
    {app_ux, operational, pricing_policy, praise_noise} and re-clusters the
    app_ux subset separately to surface in-app UX friction themes."""

    llm_batch_size: int
    spot_check_sample_size: int


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path
    raw_reviews: Path
    mouthshut_csv: Path
    raw_mouthshut: Path
    reviews: Path
    units_raw: Path
    units: Path
    embeddings_raw: Path
    unit_index_raw: Path
    embeddings: Path
    unit_index: Path
    graph: Path
    communities: Path
    themes: Path
    insights: Path
    validation: Path
    spot_check_sample: Path
    llm_insights: Path
    llm_synthesis_checkpoint: Path
    pipeline_stats: Path
    # Addressability classifier outputs (addressability-spec.md)
    unit_labels: Path
    classification_spot_check: Path
    # App UX subset re-clustering pipeline outputs
    units_appux: Path
    embeddings_appux: Path
    unit_index_appux: Path
    graph_appux: Path
    communities_appux: Path
    themes_appux: Path


@dataclass(frozen=True)
class Config:
    seed: int
    app: AppConfig
    scrape: ScrapeConfig
    units: UnitsConfig
    models: ModelsConfig
    graph: GraphConfig
    clustering: ClusteringConfig
    summarize: SummarizeConfig
    insights: InsightsConfig
    filtering: FilteringConfig
    validation: ValidationConfig
    llm_synthesis: LLMSynthesisConfig
    addressability: AddressabilityConfig
    paths: PathsConfig


def _require(d: dict, key: str, section: str) -> Any:
    if not isinstance(d, dict) or key not in d or d[key] is None:
        raise ConfigError(f"Missing required config key: '{section}.{key}'")
    return d[key]


def _require_type(value: Any, expected: Union[type, tuple], dotted_path: str) -> Any:
    if not isinstance(value, expected):
        raise ConfigError(
            f"Config key '{dotted_path}' expected type {expected}, "
            f"got {type(value).__name__} ({value!r})"
        )
    return value


def _require_range(value: float, low: float, high: float, dotted_path: str) -> float:
    if not (low <= value <= high):
        raise ConfigError(f"Config key '{dotted_path}' must be in [{low}, {high}], got {value}")
    return value


def _build_paths(data_dir: Path) -> PathsConfig:
    return PathsConfig(
        data_dir=data_dir,
        raw_reviews=data_dir / "raw_reviews.jsonl",
        mouthshut_csv=data_dir / "Mouthshut_reviews.csv",
        raw_mouthshut=data_dir / "raw_mouthshut.jsonl",
        reviews=data_dir / "reviews.jsonl",
        units_raw=data_dir / "units_raw.jsonl",
        units=data_dir / "units.jsonl",
        embeddings_raw=data_dir / "embeddings_raw.npy",
        unit_index_raw=data_dir / "unit_index_raw.json",
        embeddings=data_dir / "embeddings.npy",
        unit_index=data_dir / "unit_index.json",
        graph=data_dir / "graph.gpickle",
        communities=data_dir / "communities.json",
        themes=data_dir / "themes.json",
        insights=data_dir / "insights.json",
        validation=data_dir / "validation.json",
        spot_check_sample=data_dir / "spot_check_sample.json",
        llm_insights=data_dir / "llm_insights.json",
        llm_synthesis_checkpoint=data_dir / "llm_synthesis_checkpoint.jsonl",
        pipeline_stats=data_dir / "pipeline_stats.json",
        # Addressability classifier outputs (addressability-spec.md)
        unit_labels=data_dir / "unit_labels.jsonl",
        classification_spot_check=data_dir / "classification_spot_check.csv",
        # App UX subset re-clustering pipeline outputs
        units_appux=data_dir / "units_appux.jsonl",
        embeddings_appux=data_dir / "embeddings_appux.npy",
        unit_index_appux=data_dir / "unit_index_appux.json",
        graph_appux=data_dir / "graph_appux.gpickle",
        communities_appux=data_dir / "communities_appux.json",
        themes_appux=data_dir / "themes_appux.json",
    )


def load_config(path: Union[Path, str] = DEFAULT_CONFIG_PATH) -> Config:
    """Load, validate, and return the pipeline configuration.

    Raises ``ConfigError`` on any missing key or out-of-range value
    (edgecases.md X-08) instead of letting an invalid value silently
    propagate into a downstream stage.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(
            f"Config file not found: {path}. Create config.yaml at the project "
            "root before running any pipeline stage."
        )

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"config.yaml is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError("config.yaml must define a top-level mapping.")

    seed = _require_type(_require(raw, "seed", "root"), int, "seed")

    app_raw = _require(raw, "app", "root")
    app = AppConfig(
        app_id=_require_type(_require(app_raw, "id", "app"), str, "app.id"),
        country=_require_type(_require(app_raw, "country", "app"), str, "app.country"),
        lang=_require_type(_require(app_raw, "lang", "app"), str, "app.lang"),
    )
    if app.app_id != "com.grofers.customerapp":
        # Not fatal (spec says "verify at build time") but flagged loudly so a
        # stale/incorrect app id doesn't silently scrape the wrong app.
        import warnings

        warnings.warn(
            f"app.id '{app.app_id}' differs from the expected Blinkit id "
            "'com.grofers.customerapp' (problemstatement.md §6, edgecases.md S1-01). "
            "Confirm this is intentional.",
            stacklevel=2,
        )

    scrape_raw = _require(raw, "scrape", "root")
    lookback_months = _require_type(
        _require(scrape_raw, "lookback_months", "scrape"), int, "scrape.lookback_months"
    )
    if not (1 <= lookback_months <= 60):
        raise ConfigError(
            f"'scrape.lookback_months' must be in [1, 60], got {lookback_months}"
        )
    max_per_bucket = _require_type(
        _require(scrape_raw, "max_per_bucket", "scrape"), int, "scrape.max_per_bucket"
    )
    if max_per_bucket <= 0:
        raise ConfigError(f"'scrape.max_per_bucket' must be positive, got {max_per_bucket}")
    sorts = _require_type(_require(scrape_raw, "sort", "scrape"), list, "scrape.sort")
    if not sorts:
        raise ConfigError("'scrape.sort' must list at least one sort mode")
    scrape = ScrapeConfig(
        lookback_months=lookback_months, max_per_bucket=max_per_bucket, sorts=list(sorts)
    )

    units_raw = _require(raw, "units", "root")
    min_words = _require_type(_require(units_raw, "min_words", "units"), int, "units.min_words")
    if min_words < 1:
        raise ConfigError(f"'units.min_words' must be >= 1, got {min_words}")
    min_words_per_unit = _require_type(
        _require(units_raw, "min_words_per_unit", "units"), int, "units.min_words_per_unit"
    )
    if min_words_per_unit < 1:
        raise ConfigError(f"'units.min_words_per_unit' must be >= 1, got {min_words_per_unit}")
    max_units_per_review = _require_type(
        _require(units_raw, "max_units_per_review", "units"), int, "units.max_units_per_review"
    )
    if max_units_per_review < 1:
        raise ConfigError(f"'units.max_units_per_review' must be >= 1, got {max_units_per_review}")
    use_llm = _require_type(_require(units_raw, "use_llm", "units"), bool, "units.use_llm")
    units = UnitsConfig(
        min_words=min_words,
        min_words_per_unit=min_words_per_unit,
        max_units_per_review=max_units_per_review,
        use_llm=use_llm,
    )

    models_raw = _require(raw, "models", "root")
    embed_batch_size = _require_type(
        _require(models_raw, "embed_batch_size", "models"), int, "models.embed_batch_size"
    )
    if embed_batch_size < 1:
        raise ConfigError(f"'models.embed_batch_size' must be >= 1, got {embed_batch_size}")
    models = ModelsConfig(
        embedding_model=_require_type(
            _require(models_raw, "embedding_model", "models"), str, "models.embedding_model"
        ),
        llm_model=_require_type(
            _require(models_raw, "llm_model", "models"), str, "models.llm_model"
        ),
        embed_batch_size=embed_batch_size,
    )

    graph_raw = _require(raw, "graph", "root")
    knn_k = _require_type(_require(graph_raw, "knn_k", "graph"), int, "graph.knn_k")
    if knn_k <= 0:
        raise ConfigError(f"'graph.knn_k' must be positive, got {knn_k}")
    similarity_threshold = _require_range(
        float(_require(graph_raw, "similarity_threshold", "graph")),
        0.0,
        1.0,
        "graph.similarity_threshold",
    )
    graph = GraphConfig(knn_k=knn_k, similarity_threshold=similarity_threshold)

    clustering_raw = _require(raw, "clustering", "root")
    louvain_resolution = float(_require(clustering_raw, "louvain_resolution", "clustering"))
    if louvain_resolution <= 0:
        raise ConfigError(
            f"'clustering.louvain_resolution' must be positive, got {louvain_resolution}"
        )
    min_community_size = _require_type(
        _require(clustering_raw, "min_community_size", "clustering"),
        int,
        "clustering.min_community_size",
    )
    if min_community_size < 1:
        raise ConfigError(
            f"'clustering.min_community_size' must be >= 1, got {min_community_size}"
        )
    clustering = ClusteringConfig(
        louvain_resolution=louvain_resolution, min_community_size=min_community_size
    )

    summarize_raw = _require(raw, "summarize", "root")
    summarize_use_llm = _require_type(
        _require(summarize_raw, "use_llm", "summarize"), bool, "summarize.use_llm"
    )
    max_representatives = _require_type(
        _require(summarize_raw, "max_representatives", "summarize"), int, "summarize.max_representatives"
    )
    if max_representatives < 1:
        raise ConfigError(f"'summarize.max_representatives' must be >= 1, got {max_representatives}")
    max_quotes = _require_type(
        _require(summarize_raw, "max_quotes", "summarize"), int, "summarize.max_quotes"
    )
    if max_quotes < 1:
        raise ConfigError(f"'summarize.max_quotes' must be >= 1, got {max_quotes}")
    max_tfidf_terms = _require_type(
        _require(summarize_raw, "max_tfidf_terms", "summarize"), int, "summarize.max_tfidf_terms"
    )
    if max_tfidf_terms < 1:
        raise ConfigError(f"'summarize.max_tfidf_terms' must be >= 1, got {max_tfidf_terms}")
    long_tail_llm_batch_size = _require_type(
        _require(summarize_raw, "long_tail_llm_batch_size", "summarize"),
        int,
        "summarize.long_tail_llm_batch_size",
    )
    if long_tail_llm_batch_size < 1:
        raise ConfigError(
            f"'summarize.long_tail_llm_batch_size' must be >= 1, got {long_tail_llm_batch_size}"
        )
    summarize = SummarizeConfig(
        use_llm=summarize_use_llm,
        max_representatives=max_representatives,
        max_quotes=max_quotes,
        max_tfidf_terms=max_tfidf_terms,
        long_tail_llm_batch_size=long_tail_llm_batch_size,
    )

    insights_raw = _require(raw, "insights", "root")
    insights_similarity_threshold = _require_range(
        float(_require(insights_raw, "similarity_threshold", "insights")),
        0.0,
        1.0,
        "insights.similarity_threshold",
    )
    top_themes_count = _require_type(
        _require(insights_raw, "top_themes_count", "insights"), int, "insights.top_themes_count"
    )
    if top_themes_count < 1:
        raise ConfigError(f"'insights.top_themes_count' must be >= 1, got {top_themes_count}")
    max_verbatims_per_question = _require_type(
        _require(insights_raw, "max_verbatims_per_question", "insights"),
        int,
        "insights.max_verbatims_per_question",
    )
    if max_verbatims_per_question < 1:
        raise ConfigError(
            f"'insights.max_verbatims_per_question' must be >= 1, got {max_verbatims_per_question}"
        )
    question_queries = _require_type(
        _require(insights_raw, "question_queries", "insights"), list, "insights.question_queries"
    )
    if len(question_queries) != 8:
        raise ConfigError(
            f"'insights.question_queries' must have exactly 8 entries (one per research question), "
            f"got {len(question_queries)}"
        )
    for i, q in enumerate(question_queries):
        if not isinstance(q, str) or not q.strip():
            raise ConfigError(f"'insights.question_queries[{i}]' must be a non-empty string")
    question_required_sentiment = _require_type(
        _require(insights_raw, "question_required_sentiment", "insights"),
        list,
        "insights.question_required_sentiment",
    )
    if len(question_required_sentiment) != 8:
        raise ConfigError(
            "'insights.question_required_sentiment' must have exactly 8 entries, "
            f"got {len(question_required_sentiment)}"
        )
    for i, s in enumerate(question_required_sentiment):
        if s is not None and s not in ("negative", "neutral", "positive"):
            raise ConfigError(
                f"'insights.question_required_sentiment[{i}]' must be null/negative/neutral/positive, "
                f"got {s!r}"
            )
    insights = InsightsConfig(
        similarity_threshold=insights_similarity_threshold,
        top_themes_count=top_themes_count,
        max_verbatims_per_question=max_verbatims_per_question,
        question_queries=list(question_queries),
        question_required_sentiment=list(question_required_sentiment),
    )

    filtering_raw = _require(raw, "filtering", "root")
    min_relevance_score = _require_range(
        float(_require(filtering_raw, "min_relevance_score", "filtering")),
        0.0,
        1.0,
        "filtering.min_relevance_score",
    )
    min_theme_relevance_score = _require_range(
        float(_require(filtering_raw, "min_theme_relevance_score", "filtering")),
        0.0,
        1.0,
        "filtering.min_theme_relevance_score",
    )
    max_theme_rating = _require_type(
        _require(filtering_raw, "max_theme_rating", "filtering"), int, "filtering.max_theme_rating"
    )
    if not (1 <= max_theme_rating <= 5):
        raise ConfigError(f"'filtering.max_theme_rating' must be 1-5, got {max_theme_rating}")
    filtering = FilteringConfig(
        min_relevance_score=min_relevance_score,
        min_theme_relevance_score=min_theme_relevance_score,
        max_theme_rating=max_theme_rating,
    )

    validation_raw = _require(raw, "validation", "root")
    spot_check_sample_size_per_theme = _require_type(
        _require(validation_raw, "spot_check_sample_size_per_theme", "validation"),
        int,
        "validation.spot_check_sample_size_per_theme",
    )
    if spot_check_sample_size_per_theme < 1:
        raise ConfigError(
            "'validation.spot_check_sample_size_per_theme' must be >= 1, got "
            f"{spot_check_sample_size_per_theme}"
        )
    dominant_share_threshold = _require_range(
        float(_require(validation_raw, "dominant_share_threshold", "validation")),
        0.0,
        1.0,
        "validation.dominant_share_threshold",
    )
    short_unit_max_words = _require_type(
        _require(validation_raw, "short_unit_max_words", "validation"), int, "validation.short_unit_max_words"
    )
    medium_unit_max_words = _require_type(
        _require(validation_raw, "medium_unit_max_words", "validation"),
        int,
        "validation.medium_unit_max_words",
    )
    if not (1 <= short_unit_max_words < medium_unit_max_words):
        raise ConfigError(
            "'validation.short_unit_max_words' must be >= 1 and < 'medium_unit_max_words', got "
            f"{short_unit_max_words} / {medium_unit_max_words}"
        )
    validation = ValidationConfig(
        spot_check_sample_size_per_theme=spot_check_sample_size_per_theme,
        dominant_share_threshold=dominant_share_threshold,
        short_unit_max_words=short_unit_max_words,
        medium_unit_max_words=medium_unit_max_words,
    )

    llm_synthesis_raw = _require(raw, "llm_synthesis", "root")
    llm_sample_size = _require_type(
        _require(llm_synthesis_raw, "sample_size", "llm_synthesis"), int, "llm_synthesis.sample_size"
    )
    if llm_sample_size < 1:
        raise ConfigError(f"'llm_synthesis.sample_size' must be >= 1, got {llm_sample_size}")
    llm_batch_size = _require_type(
        _require(llm_synthesis_raw, "batch_size", "llm_synthesis"), int, "llm_synthesis.batch_size"
    )
    if llm_batch_size < 1:
        raise ConfigError(f"'llm_synthesis.batch_size' must be >= 1, got {llm_batch_size}")
    llm_rpm = _require_type(
        _require(llm_synthesis_raw, "requests_per_minute", "llm_synthesis"), int, "llm_synthesis.requests_per_minute"
    )
    if llm_rpm < 1:
        raise ConfigError(f"'llm_synthesis.requests_per_minute' must be >= 1, got {llm_rpm}")
    llm_max_retries = _require_type(
        _require(llm_synthesis_raw, "max_retries", "llm_synthesis"), int, "llm_synthesis.max_retries"
    )
    if llm_max_retries < 0:
        raise ConfigError(f"'llm_synthesis.max_retries' must be >= 0, got {llm_max_retries}")
    llm_cluster_threshold = _require_range(
        float(_require(llm_synthesis_raw, "pattern_cluster_threshold", "llm_synthesis")),
        0.0,
        1.0,
        "llm_synthesis.pattern_cluster_threshold",
    )
    llm_max_patterns = _require_type(
        _require(llm_synthesis_raw, "max_patterns_per_question", "llm_synthesis"),
        int,
        "llm_synthesis.max_patterns_per_question",
    )
    if llm_max_patterns < 1:
        raise ConfigError(
            f"'llm_synthesis.max_patterns_per_question' must be >= 1, got {llm_max_patterns}"
        )
    llm_max_quotes = _require_type(
        _require(llm_synthesis_raw, "max_quotes_per_pattern", "llm_synthesis"),
        int,
        "llm_synthesis.max_quotes_per_pattern",
    )
    if llm_max_quotes < 1:
        raise ConfigError(f"'llm_synthesis.max_quotes_per_pattern' must be >= 1, got {llm_max_quotes}")
    llm_min_support = _require_type(
        _require(llm_synthesis_raw, "min_pattern_support", "llm_synthesis"),
        int,
        "llm_synthesis.min_pattern_support",
    )
    if llm_min_support < 1:
        raise ConfigError(f"'llm_synthesis.min_pattern_support' must be >= 1, got {llm_min_support}")
    llm_min_quote_words = _require_type(
        _require(llm_synthesis_raw, "min_quote_words", "llm_synthesis"),
        int,
        "llm_synthesis.min_quote_words",
    )
    if llm_min_quote_words < 1:
        raise ConfigError(f"'llm_synthesis.min_quote_words' must be >= 1, got {llm_min_quote_words}")
    llm_synthesis = LLMSynthesisConfig(
        model=_require_type(_require(llm_synthesis_raw, "model", "llm_synthesis"), str, "llm_synthesis.model"),
        temperature=float(_require(llm_synthesis_raw, "temperature", "llm_synthesis")),
        sample_size=llm_sample_size,
        batch_size=llm_batch_size,
        requests_per_minute=llm_rpm,
        max_retries=llm_max_retries,
        pattern_cluster_threshold=llm_cluster_threshold,
        max_patterns_per_question=llm_max_patterns,
        max_quotes_per_pattern=llm_max_quotes,
        min_pattern_support=llm_min_support,
        min_quote_words=llm_min_quote_words,
    )

    addressability_raw = _require(raw, "addressability", "root")
    addr_llm_batch_size = _require_type(
        _require(addressability_raw, "llm_batch_size", "addressability"),
        int,
        "addressability.llm_batch_size",
    )
    if addr_llm_batch_size < 0:
        raise ConfigError(
            f"'addressability.llm_batch_size' must be >= 0, got {addr_llm_batch_size}"
        )
    addr_spot_check = _require_type(
        _require(addressability_raw, "spot_check_sample_size", "addressability"),
        int,
        "addressability.spot_check_sample_size",
    )
    if addr_spot_check < 1:
        raise ConfigError(
            f"'addressability.spot_check_sample_size' must be >= 1, got {addr_spot_check}"
        )
    addressability = AddressabilityConfig(
        llm_batch_size=addr_llm_batch_size,
        spot_check_sample_size=addr_spot_check,
    )

    paths_raw = _require(raw, "paths", "root")
    data_dir_str = _require_type(
        _require(paths_raw, "data_dir", "paths"), str, "paths.data_dir"
    )
    data_dir = (PROJECT_ROOT / data_dir_str).resolve()
    paths = _build_paths(data_dir)

    return Config(
        seed=seed,
        app=app,
        scrape=scrape,
        units=units,
        models=models,
        graph=graph,
        clustering=clustering,
        summarize=summarize,
        insights=insights,
        filtering=filtering,
        validation=validation,
        llm_synthesis=llm_synthesis,
        addressability=addressability,
        paths=paths,
    )


def apply_global_seed(seed: int) -> None:
    """Seed every RNG the pipeline touches directly (edgecases.md R-06).

    Per-stage code that uses additional libraries (e.g. Louvain's own RNG)
    must also be passed this seed explicitly at call time.
    """
    random.seed(seed)
    np.random.seed(seed)


def ensure_data_dir(config: Config) -> None:
    config.paths.data_dir.mkdir(parents=True, exist_ok=True)
