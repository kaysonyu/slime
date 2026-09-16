"""Public sample and transport types; model-native trajectories keep their geometry."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import torch


class SpeechTrajectory(Protocol):
    weight_version: str
    finish_reason: str

    @property
    def num_actions(self) -> int: ...

    @property
    def num_frames(self) -> int: ...

    @property
    def sequence_length(self) -> int: ...

    def training_tensors(self, config): ...


@dataclass
class Sample:
    group_index: int | None = None
    index: int | None = None
    rollout_id: int | None = None
    prompt: str = ""
    label: str | None = None
    reward: float | dict | None = None
    trajectory: SpeechTrajectory | None = None
    audio_path: str | None = None
    teacher_scores: torch.Tensor | None = None
    advantage: float | None = None
    metadata: dict = field(default_factory=dict)

    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"
        FAILED = "failed"

    status: Status = Status.PENDING

    def get_reward_value(self, args):
        key = getattr(args, "reward_key", None)
        return self.reward[key] if key else self.reward

    @property
    def effective_response_length(self):
        return self.trajectory.num_actions if self.trajectory is not None else 0

    def to_dict(self):
        return {**self.__dict__, "status": self.status.value}

    @classmethod
    def from_dict(cls, data):
        fields = cls.__dataclass_fields__
        values = {key: value for key, value in data.items() if key in fields}
        values["status"] = cls.Status(values.get("status", "pending"))
        return cls(**values)


RolloutBatch = dict[str, Any]
