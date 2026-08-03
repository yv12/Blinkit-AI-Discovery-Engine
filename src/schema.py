"""Canonical data schemas and JSON/JSONL I/O helpers.

Every artifact the pipeline produces is validated against these schemas on
write and on read (edgecases.md X-06: schema drift), written atomically so a
crash never leaves a half-written file behind (edgecases.md X-04, X-05), and
read/written strictly as UTF-8 (edgecases.md X-09).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Type

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

# Fraction of malformed lines tolerated before a JSONL read is treated as a
# corrupt artifact rather than a partially-usable one (edgecases.md X-04).
MAX_BAD_LINE_RATIO = 0.05

# Sources the pipeline accepts (Docs/context.md Addendum, post-Phase-7 §2:
# "Second data source - Mouthshut"). Originally a single-source ("google_play"
# only) constraint per problemstatement.md §5; deliberately relaxed, with the
# user, to also accept Mouthshut review-forum data as a second, clearly-tagged
# source - never silently, and never any other source beyond these two.
VALID_SOURCES = frozenset({"google_play", "mouthshut"})


class SchemaError(ValueError):
    """Raised when a record is missing required fields or has an invalid value."""


class CorruptArtifactError(RuntimeError):
    """Raised when an artifact has too many malformed lines/bytes to trust (X-04)."""


class _Record:
    """Mixin providing a uniform ``from_dict`` for schema dataclasses.

    Subclasses with nested non-primitive fields (e.g. ``Review.metadata``)
    override this to reconstruct the nested type; the default assumes all
    fields are JSON primitives.
    """

    @classmethod
    def from_dict(cls, data: dict) -> "_Record":
        return cls(**data)


@dataclass(frozen=True)
class ReviewMetadata:
    thumbs_up: int = 0
    app_version: Optional[str] = None
    developer_reply: Optional[str] = None
    lang: str = "en"


@dataclass(frozen=True)
class Review(_Record):
    """One normalized review from a source in VALID_SOURCES (originally Google Play
    Store only; Mouthshut added as a second source, Docs/context.md §11 Phase 9).
    See architecture.md §5/§7/§12.
    """

    id: str
    text: str
    date: str  # ISO 8601
    rating: Optional[int] = None
    source: str = "google_play"
    url: Optional[str] = None
    metadata: ReviewMetadata = field(default_factory=ReviewMetadata)

    def __post_init__(self) -> None:
        if not self.id:
            raise SchemaError("Review.id must be non-empty")
        if not self.text or not self.text.strip():
            raise SchemaError(f"Review.text must be non-empty (id={self.id})")
        if self.source not in VALID_SOURCES:
            # Multi-source constraint (edgecases.md X-02; relaxed from the original
            # single-source google_play-only rule - see VALID_SOURCES above).
            raise SchemaError(
                f"Review.source must be one of {sorted(VALID_SOURCES)}, got {self.source!r} "
                f"(id={self.id})"
            )
        if self.rating is not None and not (1 <= self.rating <= 5):
            raise SchemaError(f"Review.rating must be 1-5 or None, got {self.rating} (id={self.id})")

    @classmethod
    def from_dict(cls, data: dict) -> "Review":
        data = dict(data)
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            data["metadata"] = ReviewMetadata(**metadata)
        elif metadata is None:
            data["metadata"] = ReviewMetadata()
        return cls(**data)


@dataclass(frozen=True)
class Unit(_Record):
    """One atomic complaint/insight statement extracted from a Review."""

    unit_id: str
    review_id: str
    text: str
    rating: Optional[int] = None
    date: Optional[str] = None
    lang: str = "en"
    relevance_score: float = 0.0
    # Propagated from the parent Review (default kept for backward-compat with
    # units.jsonl written before the second source was added - those are all
    # implicitly google_play). See VALID_SOURCES above.
    source: str = "google_play"

    def __post_init__(self) -> None:
        if not self.unit_id:
            raise SchemaError("Unit.unit_id must be non-empty")
        if not self.review_id:
            raise SchemaError(f"Unit.review_id must be non-empty (unit_id={self.unit_id})")
        if not self.text or not self.text.strip():
            raise SchemaError(f"Unit.text must be non-empty (unit_id={self.unit_id})")
        if self.source not in VALID_SOURCES:
            raise SchemaError(
                f"Unit.source must be one of {sorted(VALID_SOURCES)}, got {self.source!r} "
                f"(unit_id={self.unit_id})"
            )


@dataclass(frozen=True)
class Theme(_Record):
    """One LLM-summarized community of units. See architecture.md §5."""

    theme_id: str
    community_id: int
    label: str
    description: str = ""
    representative_quotes: List[str] = field(default_factory=list)
    member_count: int = 0
    sentiment: str = "neutral"
    questions: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.theme_id:
            raise SchemaError("Theme.theme_id must be non-empty")
        if not self.label or not self.label.strip():
            raise SchemaError(f"Theme.label must be non-empty (theme_id={self.theme_id})")
        if self.sentiment not in ("negative", "neutral", "positive"):
            raise SchemaError(
                f"Theme.sentiment must be negative/neutral/positive, got {self.sentiment!r} "
                f"(theme_id={self.theme_id})"
            )
        for q in self.questions:
            if not (1 <= q <= 8):
                raise SchemaError(
                    f"Theme question id must be 1-8, got {q} (theme_id={self.theme_id})"
                )


@dataclass(frozen=True)
class QuestionInsight(_Record):
    """Evidence-backed support for one of the 8 research questions."""

    question_id: int
    theme_ids: List[str] = field(default_factory=list)
    total_count: int = 0
    top_verbatims: List[str] = field(default_factory=list)
    # "insufficient" when no theme maps to this question (edgecases.md S8-01).
    coverage: str = "sufficient"

    def __post_init__(self) -> None:
        if not (1 <= self.question_id <= 8):
            raise SchemaError(f"QuestionInsight.question_id must be 1-8, got {self.question_id}")
        if self.coverage not in ("sufficient", "insufficient"):
            raise SchemaError(
                f"QuestionInsight.coverage must be sufficient/insufficient, got {self.coverage!r}"
            )


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via temp file + os.replace so a crash/disk-full never leaves a
    half-written artifact behind (edgecases.md X-05)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


def write_jsonl(path: Path, records: Iterable[Any]) -> int:
    """Serialize dataclass instances (or plain dicts) to a JSONL file atomically."""
    lines = []
    for record in records:
        payload = asdict(record) if hasattr(record, "__dataclass_fields__") else record
        lines.append(json.dumps(payload, ensure_ascii=False))
    body = "\n".join(lines) + ("\n" if lines else "")
    _atomic_write_text(path, body)
    return len(lines)


def read_jsonl(
    path: Path,
    factory: Optional[Type[_Record]] = None,
    max_bad_line_ratio: float = MAX_BAD_LINE_RATIO,
) -> Iterator[Any]:
    """Read a JSONL artifact, validating each line against ``factory`` if given.

    Malformed or schema-invalid lines are skipped and counted rather than
    crashing the whole read (edgecases.md X-04). If the bad-line ratio
    exceeds ``max_bad_line_ratio``, raises ``CorruptArtifactError`` instead
    of silently returning a partial/untrustworthy corpus.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Expected artifact not found: {path}. Run the stage that produces it first "
            "(see Implementation-plan.md for stage order)."
        )

    total = 0
    bad = 0
    records: List[Any] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                payload = json.loads(line)
                record = factory.from_dict(payload) if factory is not None else payload
            except (json.JSONDecodeError, TypeError, SchemaError) as exc:
                bad += 1
                logger.warning("Skipping malformed line %d in %s: %s", line_no, path, exc)
                continue
            records.append(record)

    if total > 0 and bad / total > max_bad_line_ratio:
        raise CorruptArtifactError(
            f"{path} has {bad}/{total} malformed lines (> {max_bad_line_ratio:.0%} threshold); "
            "treating artifact as corrupt. Re-run the stage that produces it."
        )
    if bad:
        logger.warning("%s: skipped %d/%d malformed lines", path, bad, total)

    return iter(records)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    """Serialize a JSON-able object atomically (edgecases.md X-05)."""
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
    _atomic_write_text(path, text)


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected artifact not found: {path}. Run the stage that produces it first."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise CorruptArtifactError(f"{path} is not valid JSON: {exc}") from exc
