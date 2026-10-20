"""Audio reward component math, independent of Prism samples and HTTP."""

import copy
import importlib
import inspect
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence

from .protocol import ScoringError, finite

NORMALIZATION = "nfkc_casefold_no_punctuation_cjk_char_other_word_v1"
CHAR_LANGUAGES = {"zh", "zho", "chinese", "中文", "ja", "jpn", "japanese", "日语", "ko", "kor", "korean", "韩语"}


def normalized_tokens(text: str, language: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = "".join(ch for ch in text if not unicodedata.category(ch).startswith(("P", "S")))
    if language.strip().lower().replace("_", "-").split("-")[0] in CHAR_LANGUAGES:
        return [ch for ch in text if not ch.isspace()]
    return text.split()


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, actual in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (expected != actual)))
        previous = current
    return previous[-1]


def component_result(raw, transformed, body=None, **details):
    return {
        "raw": raw,
        "transformed": transformed,
        "status": "success",
        "identity": body.get("identity") if body else None,
        "details": details,
    }


def normalized_vector(result, dimension):
    values = result.get("embedding")
    if (
        result.get("normalized") is not True
        or result.get("dimension") != dimension
        or not isinstance(values, list)
        or len(values) != dimension
    ):
        raise ScoringError("SIM embedding dimension/normalization contract differs")
    vector = tuple(finite(value, "SIM embedding") for value in values)
    if abs(math.fsum(value * value for value in vector) - 1) > 1e-3:
        raise ScoringError("SIM embedding is not normalized")
    return vector


def transcription_score(reference, hypothesis, language):
    expected = normalized_tokens(reference, language)
    if not expected:
        raise ValueError("WER transcript_target has no tokens after normalization")
    actual = normalized_tokens(hypothesis, language)
    errors = edit_distance(expected, actual)
    raw = errors / len(expected)
    return component_result(
        raw,
        1 / (1 + raw),
        normalization=NORMALIZATION,
        language=language,
        asr_language_mode="auto",
        reward_transform="inverse_error",
        errors=errors,
        reference_tokens=len(expected),
        hypothesis_tokens=len(actual),
        transcript=hypothesis,
    )


def speaker_similarity(candidate, reference):
    return max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(candidate, reference, strict=True))))


def judge_probabilities(audio, texts, scale, bias):
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32)
    text = np.asarray(texts, dtype=np.float32)
    if not np.isfinite(audio).all() or not np.isfinite(text).all():
        raise ScoringError("embedding is not representable in FP32")
    margin = audio[None, :] @ text.T - np.square(text).sum(-1)[None, :]
    logits = margin[0] * np.float32(scale) + np.float32(bias)
    if not np.isfinite(logits).all():
        raise ScoringError("JudgeCLAP logits overflowed")
    exp = np.exp(-np.abs(logits))
    scores = np.where(logits >= 0, 1 / (1 + exp), exp / (1 + exp)).tolist()
    if any(not math.isfinite(x) for x in scores):
        raise ScoringError("nonfinite JudgeCLAP score")
    return scores


def default_wer(*, reference, hypothesis, language, identity):
    """Default text scoring; identity is available to replacement scorers."""
    result = transcription_score(reference, hypothesis, language)
    return {key: result[key] for key in ("raw", "transformed", "details")}


def default_sim(*, candidate, reference, identity):
    """Cosine of validated unit speaker vectors, clipped at zero for reward."""
    raw = speaker_similarity(candidate, reference)
    return {"raw": raw, "transformed": max(0, raw)}


def default_judge_clap(*, audio, texts, queries, identity):
    """Checkpoint-calibrated attribute probabilities, averaged into one score."""
    scoring = identity.get("scoring", {})
    if not isinstance(scoring, dict) or scoring.get("type") != "siglip_margin_v1":
        raise ScoringError("unsupported JudgeCLAP scoring rule")
    scale, bias = finite(scoring.get("scale"), "scale"), finite(scoring.get("bias"), "bias")
    if scale <= 0:
        raise ScoringError("JudgeCLAP scale must be positive")
    scores = judge_probabilities(audio, texts, scale, bias)
    mean = math.fsum(scores) / len(scores)
    return {
        "raw": mean,
        "transformed": mean,
        "details": {"attribute_scores": scores, "attribute_count": len(scores)},
    }


def default_prosody(*, distance, groups, identity):
    """Keep the benchmark distance and map it monotonically to a [0, 1] reward."""
    return {
        "raw": distance,
        "transformed": 1 / (1 + distance),
        "details": {"groups": groups, "reward_transform": "inverse_distance"},
    }


_DEFAULT_SCORERS = {
    "wer": default_wer,
    "sim": default_sim,
    "judge_clap": default_judge_clap,
    "prosody": default_prosody,
}


class ComponentScorers:
    """Resolve optional dotted-path scorers once and validate their score records."""

    def __init__(self, config=None):
        if config is None:
            config = {}
        if not isinstance(config, Mapping) or set(config) - _DEFAULT_SCORERS.keys():
            raise ValueError("scorers must map wer, sim, judge_clap, and prosody to path/kwargs objects")
        self._custom = {}
        for name, options in config.items():
            if not isinstance(options, Mapping) or "path" not in options or set(options) - {"path", "kwargs"}:
                raise ValueError(f"{name} scorer requires path and optional kwargs")
            path, kwargs = options["path"], options.get("kwargs", {})
            if not isinstance(path, str) or not path.strip() or "." not in path:
                raise ValueError(f"{name} scorer path must be a dotted function path")
            inputs = inspect.signature(_DEFAULT_SCORERS[name]).parameters
            if (
                not isinstance(kwargs, Mapping)
                or any(not isinstance(key, str) for key in kwargs)
                or set(kwargs) & inputs.keys()
            ):
                raise ValueError(f"{name} scorer kwargs must be a mapping without reserved input names")
            try:
                # Snapshot serializable configuration; callers and callbacks cannot mutate it.
                kwargs = json.loads(json.dumps(dict(kwargs), allow_nan=False))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} scorer kwargs must contain finite JSON values") from error
            module, _, attribute = path.rpartition(".")
            try:
                function = getattr(importlib.import_module(module), attribute)
            except (ImportError, AttributeError) as error:
                raise ValueError(f"cannot load {name} scorer: {path}") from error
            if not callable(function):
                raise ValueError(f"{name} scorer is not callable: {path}")
            self._custom[name] = (function, path, kwargs)

    async def score(self, name, *, identity, **inputs):
        custom = self._custom.get(name)
        if custom is None:
            result = _DEFAULT_SCORERS[name](identity=identity, **inputs)
        else:
            function, path, kwargs = custom
            # In particular, JudgeCLAP text arrays and identity belong to shared caches.
            result = function(**copy.deepcopy({"identity": identity, **inputs, **kwargs}))
            if inspect.isawaitable(result):
                result = await result
        if (
            not isinstance(result, Mapping)
            or not {"raw", "transformed"} <= result.keys()
            or set(result) - {"raw", "transformed", "details"}
        ):
            raise ScoringError(f"{name} scorer must return raw, transformed, and optional details")
        raw = finite(result["raw"], f"{name} scorer raw")
        transformed = finite(result["transformed"], f"{name} scorer transformed")
        details = result.get("details", {})
        if not isinstance(details, Mapping) or any(not isinstance(key, str) for key in details):
            raise ScoringError(f"{name} scorer details must be a JSON object")
        try:
            details = json.loads(json.dumps(dict(details), allow_nan=False))
        except (TypeError, ValueError) as error:
            raise ScoringError(f"{name} scorer details must contain finite JSON values") from error
        if custom is not None:
            details["scorer"] = {"path": path, "kwargs": copy.deepcopy(kwargs)}
        record = component_result(raw, transformed, {"identity": identity})
        record["details"] = details
        return record
