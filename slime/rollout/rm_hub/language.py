"""Language contracts shared by MOSS-TTS generation and WER scoring."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageSpec:
    canonical: str
    mosstts: str
    qwen_asr: str
    token_mode: str


_CHARACTER_LANGUAGES = frozenset({"cantonese", "chinese", "japanese", "korean", "thai"})
_LANGUAGE_NAMES = (
    "arabic",
    "cantonese",
    "chinese",
    "czech",
    "danish",
    "dutch",
    "english",
    "finnish",
    "french",
    "german",
    "greek",
    "hebrew",
    "hindi",
    "hungarian",
    "italian",
    "japanese",
    "korean",
    "macedonian",
    "malay",
    "persian",
    "polish",
    "portuguese",
    "romanian",
    "russian",
    "spanish",
    "swahili",
    "swedish",
    "tagalog",
    "thai",
    "turkish",
    "vietnamese",
)

LANGUAGE_SPECS = {
    name: LanguageSpec(
        canonical=name,
        mosstts=name,
        qwen_asr=name.title(),
        token_mode="character" if name in _CHARACTER_LANGUAGES else "word",
    )
    for name in _LANGUAGE_NAMES
}


def resolve_language(value: object) -> LanguageSpec:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("MOSS-TTS requires a non-empty canonical language string.")
    canonical = value.strip().casefold()
    canonical = {"en": "english", "zh": "chinese"}.get(canonical, canonical)
    try:
        return LANGUAGE_SPECS[canonical]
    except KeyError as error:
        raise ValueError(f"Unsupported MOSS-TTS language {value!r}. Expected one of {sorted(LANGUAGE_SPECS)}.") from error
