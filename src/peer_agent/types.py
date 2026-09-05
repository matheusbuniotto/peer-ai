from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class Verdict(Enum):
    NO_EFFECT = auto()
    INVALID = auto()
    SHIP = auto()
    EXTEND = auto()


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
