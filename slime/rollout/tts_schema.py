"""Strict Pydantic models for canonical rows and sample metadata."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator, model_validator

_SPECIAL_TOKEN_PATTERN = re.compile(r"<\|[^<>]*\|>")
ReferenceAudioUse = Literal["timbre", "accent", "prosody", "emotion"]
REFERENCE_AUDIO_USE_ORDER: tuple[ReferenceAudioUse, ...] = (
    "timbre",
    "accent",
    "prosody",
    "emotion",
)


def _plain_text(value: str, field_name: str) -> str:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    if _SPECIAL_TOKEN_PATTERN.search(value) is not None:
        raise ValueError(f"{field_name} contains reserved special-token-shaped text")
    return value


class TTSScriptItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    text: str
    local_instruction: str | None = None

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return _plain_text(value, "script item text")

    @field_validator("local_instruction")
    @classmethod
    def validate_local_instruction(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _plain_text(value, "local_instruction")


class TTSRubricItem(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, strict=True)
    __pydantic_extra__: dict[str, JsonValue]

    dimension: str
    statement: str

    @field_validator("dimension")
    @classmethod
    def validate_dimension(cls, value: str) -> str:
        value = _plain_text(value, "rubric dimension")
        if len(value.strip()) > 128:
            raise ValueError("rubric dimension must contain at most 128 Unicode characters")
        return value

    @field_validator("statement")
    @classmethod
    def validate_statement(cls, value: str) -> str:
        value = _plain_text(value, "rubric statement")
        if len(value) > 512:
            raise ValueError("rubric statement must contain at most 512 Unicode characters")
        return value


class TTSReferenceAudio(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: str
    path: Path
    uses: tuple[ReferenceAudioUse, ...]

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("reference audio path must be absolute")
        return value

    @field_validator("uses")
    @classmethod
    def validate_uses(cls, value: tuple[ReferenceAudioUse, ...]) -> tuple[ReferenceAudioUse, ...]:
        if not value:
            raise ValueError("reference audio uses must be non-empty")
        if len(value) != len(set(value)):
            raise ValueError("reference audio uses must not contain duplicates")
        return value


def _validate_reference_assignments(
    references: tuple[TTSReferenceAudio, ...],
) -> tuple[TTSReferenceAudio, ...]:
    ids = [reference.id for reference in references]
    if len(ids) != len(set(ids)):
        raise ValueError("reference audio ids must be unique within a row")
    assigned_uses: set[ReferenceAudioUse] = set()
    for reference in references:
        duplicate_uses = assigned_uses.intersection(reference.uses)
        if duplicate_uses:
            rendered = ", ".join(use for use in REFERENCE_AUDIO_USE_ORDER if use in duplicate_uses)
            raise ValueError(f"reference audio uses must be assigned at most once per row: {rendered}")
        assigned_uses.update(reference.uses)
    return references


class TTSJsonRow(BaseModel):
    """Canonical JSONL row used by both training and evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: str | None = None
    text: str | None = None
    script: str | tuple[TTSScriptItem, ...] | None = None
    global_instruction: str | None = None
    instructions: str | None = None
    ref_audio: Path | None = None
    ref_text: str | None = None
    reference_audios: tuple[TTSReferenceAudio, ...] = ()
    target_text: str | None = None
    rubric: tuple[TTSRubricItem, ...] = ()
    language: str | None = None
    domain: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("id", "text", "target_text", "instructions", "ref_text")
    @classmethod
    def validate_required_text(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _plain_text(value, info.field_name)

    @field_validator("script")
    @classmethod
    def validate_script(cls, value: str | tuple[TTSScriptItem, ...]) -> str | tuple[TTSScriptItem, ...]:
        if value is None:
            return None
        if isinstance(value, str):
            return _plain_text(value, "script")
        if not value:
            raise ValueError("script list must be non-empty")
        return value

    @field_validator("global_instruction", mode="before")
    @classmethod
    def normalize_global_instruction(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("global_instruction")
    @classmethod
    def validate_global_instruction(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _plain_text(value, "global_instruction")

    @field_validator("language", "domain")
    @classmethod
    def validate_optional_text(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        value = _plain_text(value, info.field_name)
        normalized = value.strip()
        return normalized.casefold() if info.field_name == "language" else normalized

    @model_validator(mode="after")
    def validate_reference_assignments(self) -> TTSJsonRow:
        if (self.text is None) == (self.script is None):
            raise ValueError("Provide exactly one of text or script")
        if self.script is not None and (self.id is None or self.target_text is None):
            raise ValueError("Canonical script rows require id and target_text")
        if self.instructions is not None and self.global_instruction is not None:
            raise ValueError("Provide instructions or global_instruction, not both")
        _validate_reference_assignments(self.reference_audios)
        return self


@dataclass(slots=True)
class ReferenceAudioMetadataError(ValueError):
    detail: str

    def __str__(self) -> str:
        return self.detail


def parse_reference_audios_metadata(value: JsonValue) -> tuple[TTSReferenceAudio, ...]:
    """Re-validate reference metadata when a Sample crosses a process boundary."""
    if not isinstance(value, list):
        raise ReferenceAudioMetadataError("sample.metadata['reference_audios'] must be an object array")
    references: list[TTSReferenceAudio] = []
    for index, item in enumerate(value):
        normalized = item
        if isinstance(item, dict):
            normalized = dict(item)
            raw_path = normalized.get("path")
            raw_uses = normalized.get("uses")
            if isinstance(raw_path, str):
                normalized["path"] = Path(raw_path)
            if isinstance(raw_uses, list):
                normalized["uses"] = tuple(raw_uses)
        try:
            references.append(TTSReferenceAudio.model_validate(normalized))
        except ValidationError as error:
            details = []
            for issue in error.errors(include_url=False, include_input=False):
                location = ".".join(str(part) for part in issue["loc"]) or "item"
                details.append(f"{location}: {issue['msg']}")
            raise ReferenceAudioMetadataError(
                f"sample.metadata['reference_audios'][{index}] is invalid: {'; '.join(details)}"
            ) from error
    parsed = tuple(references)
    try:
        return _validate_reference_assignments(parsed)
    except ValueError as error:
        raise ReferenceAudioMetadataError(str(error)) from error


@dataclass(slots=True)
class ManifestRowError(ValueError):
    line_number: int
    detail: str

    def __str__(self) -> str:
        return f"line {self.line_number} is not a canonical MOSS-TTS JSON row: {self.detail}"


def parse_manifest_row_json(raw: str, *, line_number: int) -> TTSJsonRow:
    """Parse one JSONL line while preserving a stable line-numbered error."""
    try:
        return TTSJsonRow.model_validate_json(raw)
    except ValidationError as error:
        details = []
        for item in error.errors(include_url=False, include_input=False):
            location = ".".join(str(part) for part in item["loc"]) or "row"
            details.append(f"{location}: {item['msg']}")
        raise ManifestRowError(line_number=line_number, detail="; ".join(details)) from error
