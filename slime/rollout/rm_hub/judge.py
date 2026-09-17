"""OpenAI-compatible AnyAudio-Judge targeted-logprob reward."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from functools import lru_cache
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, TypedDict

from slime.utils.types import Sample

from .audio import wav_data_url, wav_file_uri
from .config import ComponentReward
from .runtime import RewardHttpRuntime, RewardServiceError

MAX_RUBRICS_PER_SAMPLE: Final = 32
MAX_RUBRIC_DIMENSION_CHARS: Final = 128
MAX_RUBRIC_STATEMENT_CHARS: Final = 512
MIN_JUDGE_AUDIO_SECONDS: Final = 0.1
DEFAULT_JUDGE_TOKENIZER_PATH: Final = "/inspire/qb-ilm2/project/cq-scientific-cooperation-zone/public/kyu/models/AnyAudio-Judge-30B"
_YES_NO_VARIANTS: Final[dict[str, tuple[str, ...]]] = {
    "yes": ("yes", " yes", "Yes", " Yes", "YES", " YES"),
    "no": ("no", " no", "No", " No", "NO", " NO"),
}
JUDGE_LOGITS_SYSTEM_PROMPT: Final = """你是一位专业的音频感知评估专家。你的任务是仔细聆听提供的音频片段，并对给出的判断题作答（yes 或 no）。

## 核心规则
1. 只听不推断：仅根据音频中实际可感知的内容作答，不依赖背景知识或常识推断
2. 方向已统一：所有题目已保证“yes = 该特征符合描述”，请直接按此方向作答
3. 模糊从严：如果该特征在音频中难以明确感知，回答 no

请只回答 yes 或 no。"""


@dataclass(frozen=True, slots=True)
class RubricItem:
    dimension: str
    statement: str


class RubricScoreMetadata(TypedDict):
    index: int
    dimension: str
    p_yes: float


@dataclass(frozen=True, slots=True)
class RubricScore:
    index: int
    dimension: str
    p_yes: float

    def to_metadata(self) -> RubricScoreMetadata:
        return {
            "index": self.index,
            "dimension": self.dimension,
            "p_yes": self.p_yes,
        }


@dataclass(frozen=True, slots=True)
class JudgeLabelTokenIds:
    yes: tuple[int, ...]
    no: tuple[int, ...]

    @property
    def all(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys((*self.yes, *self.no)))


def yes_probability(logprob_yes: float, logprob_no: float) -> float:
    """Convert two targeted logprobs to a numerically stable yes probability."""
    if not math.isfinite(logprob_yes) or not math.isfinite(logprob_no):
        raise ValueError("Judge yes/no logprobs must be finite.")
    maximum = max(logprob_yes, logprob_no)
    yes = math.exp(logprob_yes - maximum)
    no = math.exp(logprob_no - maximum)
    return yes / (yes + no)


def _parse_token_id(value: object) -> int | None:
    if not isinstance(value, str) or not value.startswith("token_id:"):
        return None
    raw_token_id = value.removeprefix("token_id:")
    try:
        token_id = int(raw_token_id)
    except ValueError:
        return None
    return token_id if token_id >= 0 else None


def parse_yes_no_logprobs(
    response: Any,
    *,
    yes_token_ids: Sequence[int],
    no_token_ids: Sequence[int],
) -> tuple[float, float]:
    try:
        entries = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError) as error:
        raise RewardServiceError("judge", "protocol", "missing targeted OpenAI logprobs") from error
    if not isinstance(entries, list):
        raise RewardServiceError("judge", "protocol", "top_logprobs must be a list")
    yes_ids = set(yes_token_ids)
    no_ids = set(no_token_ids)
    class_values: dict[str, list[float]] = {"yes": [], "no": []}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        token_id = _parse_token_id(entry.get("token"))
        value = entry.get("logprob")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if token_id in yes_ids:
            class_values["yes"].append(float(value))
        elif token_id in no_ids:
            class_values["no"].append(float(value))
    if not class_values["yes"] or not class_values["no"]:
        raise RewardServiceError(
            "judge",
            "protocol",
            "both yes and no targeted token logprobs are required",
        )
    return max(class_values["yes"]), max(class_values["no"])


def parse_yes_no_logits(
    response: Any,
    *,
    yes_token_ids: Sequence[int],
    no_token_ids: Sequence[int],
) -> tuple[float, float]:
    """Backward-compatible name for the targeted-logprob parser."""
    return parse_yes_no_logprobs(
        response,
        yes_token_ids=yes_token_ids,
        no_token_ids=no_token_ids,
    )


def _encode_label_token_ids(tokenizer: Any, label: str) -> tuple[int, ...]:
    token_ids: set[int] = set()
    for variant in _YES_NO_VARIANTS[label]:
        encoded = tokenizer.encode(variant, add_special_tokens=False)
        for token_id in encoded:
            decoded = tokenizer.decode([token_id]).strip().casefold()
            if decoded == label:
                token_ids.add(int(token_id))
    if not token_ids:
        token_ids.update(int(token_id) for token_id in tokenizer.encode(f" {label}", add_special_tokens=False))
    return tuple(sorted(token_ids))


@lru_cache(maxsize=4)
def _load_judge_label_token_ids(tokenizer_path: str) -> JudgeLabelTokenIds:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    yes_token_ids = _encode_label_token_ids(tokenizer, "yes")
    no_token_ids = _encode_label_token_ids(tokenizer, "no")
    if not yes_token_ids or not no_token_ids:
        raise ValueError("Judge tokenizer must provide at least one yes and one no token ID.")
    return JudgeLabelTokenIds(yes=yes_token_ids, no=no_token_ids)


def judge_label_token_ids(tokenizer_path: str | None) -> JudgeLabelTokenIds:
    return _load_judge_label_token_ids(tokenizer_path or DEFAULT_JUDGE_TOKENIZER_PATH)


def _generated_audio(sample: Sample) -> Path:
    value = sample.audio_path
    if not isinstance(value, str) or not value or not Path(value).is_file():
        raise ValueError("Judge reward requires readable generated audio metadata.")
    return Path(value)


def _audio_is_too_short_for_judge(path: Path) -> bool:
    try:
        with wave.open(str(path), "rb") as stream:
            frame_count = stream.getnframes()
            sample_rate_hz = stream.getframerate()
    except (EOFError, wave.Error) as error:
        raise ValueError("Judge generated audio must be a readable PCM WAV file.") from error
    if frame_count <= 0 or sample_rate_hz <= 0:
        raise ValueError("Judge generated audio must contain valid WAV frames.")
    minimum_frames = math.ceil(sample_rate_hz * MIN_JUDGE_AUDIO_SECONDS)
    return frame_count < minimum_frames


def parse_sample_rubrics(sample: Sample) -> tuple[RubricItem, ...]:
    metadata = sample.metadata
    if not isinstance(metadata, dict):
        raise TypeError("Judge reward requires sample.metadata to be an object.")

    instruction = metadata.get("instructions")
    if instruction is None or isinstance(instruction, str) and not instruction.strip():
        raise ValueError("Judge reward requires non-empty string metadata.instructions.")
    if not isinstance(instruction, str):
        raise TypeError("Judge reward requires metadata.instructions to be a string.")

    raw_rubrics = metadata.get("rubric")
    if raw_rubrics is None:
        raise ValueError("Judge reward requires metadata.rubric.")
    if not isinstance(raw_rubrics, list):
        raise TypeError("metadata.rubric must be an array for Judge reward.")
    if not 1 <= len(raw_rubrics) <= MAX_RUBRICS_PER_SAMPLE:
        raise ValueError(f"Judge reward requires between 1 and {MAX_RUBRICS_PER_SAMPLE} metadata.rubric items.")

    rubrics: list[RubricItem] = []
    for index, raw_item in enumerate(raw_rubrics):
        item_name = f"metadata.rubric[{index}]"
        if not isinstance(raw_item, Mapping):
            raise TypeError(f"{item_name} must be an object for Judge reward.")
        dimension = raw_item.get("dimension")
        if not isinstance(dimension, str):
            raise TypeError(f"Judge reward requires {item_name}.dimension to be a string.")
        normalized_dimension = dimension.strip()
        if not normalized_dimension:
            raise ValueError(f"{item_name}.dimension must be non-empty for Judge reward.")
        if len(normalized_dimension) > MAX_RUBRIC_DIMENSION_CHARS:
            raise ValueError(f"Judge reward requires {item_name}.dimension to contain at most {MAX_RUBRIC_DIMENSION_CHARS} Unicode characters.")
        statement = raw_item.get("statement")
        if not isinstance(statement, str):
            raise TypeError(f"Judge reward requires {item_name}.statement to be a string.")
        normalized_statement = statement.strip()
        if not normalized_statement:
            raise ValueError(f"{item_name}.statement must be non-empty for Judge reward.")
        if len(normalized_statement) > MAX_RUBRIC_STATEMENT_CHARS:
            raise ValueError(f"Judge reward requires {item_name}.statement to contain at most {MAX_RUBRIC_STATEMENT_CHARS} Unicode characters.")
        rubrics.append(RubricItem(dimension=normalized_dimension, statement=normalized_statement))
    return tuple(rubrics)


async def score(sample: Sample, runtime: RewardHttpRuntime) -> ComponentReward:
    """Score each rubric item from the Judge's targeted yes/no logprobs."""
    if isinstance(sample.metadata, dict):
        sample.metadata.pop("judge_rubric_scores", None)
        sample.metadata.pop("judge_skipped_reason", None)
    rubrics = parse_sample_rubrics(sample)
    config = runtime.service("judge")
    if config.model is None:
        raise RewardServiceError("judge", "configuration", "service model name is required")
    if sample.audio_path is None and sample.trajectory is not None and sample.trajectory.num_frames == 0:
        sample.metadata["judge_skipped_reason"] = "empty_generation"
        return ComponentReward("judge", 0.0, 0.0)
    audio_path = _generated_audio(sample)
    if _audio_is_too_short_for_judge(audio_path):
        sample.metadata["judge_skipped_reason"] = "audio_too_short"
        return ComponentReward("judge", 0.0, 0.0)

    tokenizer_path = config.tokenizer_path
    try:
        label_token_ids = judge_label_token_ids(tokenizer_path)
    except (ImportError, KeyError, OSError, ValueError) as error:
        raise RewardServiceError(
            "judge",
            "configuration",
            f"unable to load Judge tokenizer from {tokenizer_path or DEFAULT_JUDGE_TOKENIZER_PATH}",
        ) from error
    audio_url = wav_file_uri(audio_path) if config.protocol == "openai_chat_path" else wav_data_url(audio_path)

    async def score_rubric(index: int, rubric: RubricItem) -> RubricScore:
        payload = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": JUDGE_LOGITS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": audio_url}},
                        {"type": "text", "text": f"请判断：【{rubric.dimension}】{rubric.statement}"},
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": 1,
            "logprobs": True,
            "logprob_token_ids": list(label_token_ids.all),
            "return_tokens_as_token_ids": True,
        }
        response = await runtime.post_json("judge", payload)
        logprob_yes, logprob_no = parse_yes_no_logprobs(
            response,
            yes_token_ids=label_token_ids.yes,
            no_token_ids=label_token_ids.no,
        )
        p_yes = yes_probability(logprob_yes, logprob_no)
        return RubricScore(index=index, dimension=rubric.dimension, p_yes=p_yes)

    # Each rubric is an independent request, but a protocol/configuration error
    # invalidates the complete Judge component rather than a partial average.
    results = await asyncio.gather(
        *(score_rubric(index, rubric) for index, rubric in enumerate(rubrics)),
        return_exceptions=True,
    )
    scores: list[RubricScore] = []
    for result in results:
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, Exception):
            raise result
        scores.append(result)
    sample.metadata["judge_rubric_scores"] = [rubric_score.to_metadata() for rubric_score in scores]
    reward = sum(rubric_score.p_yes for rubric_score in scores) / len(scores)
    return ComponentReward("judge", reward, reward)
