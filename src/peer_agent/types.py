from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum, StrEnum, auto

import pydantic


class Verdict(Enum):
    NO_EFFECT = auto()
    INVALID = auto()
    SHIP = auto()
    EXTEND = auto()


class RandomizationUnit(StrEnum):
    """
    String-valued so a model writing structured output produces a
    self-describing value ("by_user") instead of an opaque int it has to
    guess the meaning of — and can't accidentally emit as the wrong type.
    """

    BY_USER = "by_user"
    BY_SESSION = "by_session"


BY_USER = RandomizationUnit.BY_USER
BY_SESSION = RandomizationUnit.BY_SESSION


@dataclass(frozen=True)
class SRMResult:
    mismatch: bool
    p_value: float
    n_control: int
    n_treatment: int


@dataclass(frozen=True)
class AnalyzeResult:
    p_value: float
    lift: float
    ci_low: float
    ci_high: float
    width: float


@pydantic.dataclasses.dataclass(frozen=True)
class DesignSpec:
    baseline: float
    mde: float
    power: float
    days: int
    n_per_arm: int
    unit: RandomizationUnit
    metric: str
    guardrails: tuple[str, ...] = ()
    if_flat: str | None = None

    @pydantic.model_validator(mode="before")
    @classmethod
    def _parse_if_stringified(cls, data: object) -> object:
        """Some models emit a nested object arg as a JSON string; decode it here."""
        return json.loads(data) if isinstance(data, str) else data


@dataclass(frozen=True)
class SequentialResult:
    reject: bool
    p_value: float
    alpha: float


@dataclass(frozen=True)
class SegmentScanResult:
    winners: tuple[str, ...]
    reversal: bool


@dataclass(frozen=True)
class NoveltyResult:
    decaying: bool
    early_lift: float
    late_lift: float


@dataclass(frozen=True)
class GuardrailResult:
    breached: tuple[str, ...]


@dataclass(frozen=True)
class PowerResult:
    n_per_arm: int


@dataclass(frozen=True)
class SimulationResult:
    power: float
    promised: float
    warnings: tuple[str, ...]
    verdict: Verdict
    mean_estimate: float
