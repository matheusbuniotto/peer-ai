from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from typing import Literal

import pydantic


class Verdict(Enum):
    NO_EFFECT = auto()  # the interval excludes the MDE: a real null, not a shrug
    INCONCLUSIVE = auto()  # the interval covers both zero and the MDE: not enough data
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
    counts: tuple[tuple[str, int], ...] = ()
    # Per-stratum verdicts when `by` was given. A split can balance overall and
    # still tilt inside a segment, which is how Simpson's paradox arrives.
    strata: tuple[tuple[str, bool], ...] = ()


@dataclass(frozen=True)
class AnalyzeResult:
    p_value: float
    lift: float
    ci_low: float
    ci_high: float
    width: float
    # None when no MDE was pre-registered, in which case "not significant" is the
    # most that can be said and NO_EFFECT is not available as a verdict.
    equivalent_to_null: bool | None = None
    mde: float | None = None
    # "row" means independence was assumed rather than enforced.
    unit_of_analysis: str = "row"
    rows_per_unit: float = 1.0


@pydantic.dataclasses.dataclass(frozen=True)
class DesignSpec:
    baseline: float
    mde: float
    power: float
    days: int
    n_per_arm: int
    unit: RandomizationUnit
    metric: str
    guardrails: tuple[Guardrail, ...] = ()
    if_flat: str | None = None
    # The analysis choices, declared before anyone looks. Defaults reproduce what
    # stats.py did when they were hardcoded, so an unspecified spec changes nothing.
    alpha: float = 0.05
    control_value: bool | int | str = False
    covariate: str | None = None
    # The column holding the randomisation unit, when rows are visits not people.
    unit_column: str | None = None
    looks: int = 1
    ratio: float = 1.0
    daily_traffic: int | None = None

    @pydantic.model_validator(mode="before")
    @classmethod
    def _parse_if_stringified(cls, data: object) -> object:
        """Some models emit a nested object arg as a JSON string; decode it here."""
        return json.loads(data) if isinstance(data, str) else data

    @pydantic.field_validator("guardrails", mode="before")
    @classmethod
    def _lift_bare_names(cls, value: object) -> object:
        """A guardrail named as a plain string gets the default margin."""
        if isinstance(value, str | bytes) or not isinstance(value, Iterable):
            return value
        return tuple({"name": item} if isinstance(item, str) else item for item in value)


@dataclass(frozen=True)
class SequentialResult:
    reject: bool
    p_value: float
    alpha: float


@dataclass(frozen=True)
class SegmentScanResult:
    winners: tuple[str, ...]
    reversal: bool
    # Does one effect explain every segment? Low means no.
    heterogeneity_p: float = float("nan")
    # Per-segment split checks: a reversal in the outcome is often a skew in the counts.
    composition_srm: tuple[tuple[str, bool], ...] = ()


@dataclass(frozen=True)
class NoveltyResult:
    decaying: bool
    slope: float  # change in lift per day
    slope_ci: tuple[float, float]
    daily_lifts: tuple[tuple[int, float], ...] = ()
    # False when only a calendar day was available. Novelty decays on days since
    # first exposure, and the two answer different questions.
    cohort_day: bool = False


class GuardrailStatus(StrEnum):
    CLEAN = "clean"  # harm worse than the margin is ruled out
    BREACHED = "breached"  # harm worse than the margin is established
    INCONCLUSIVE = "inconclusive"  # too wide to tell — blocks, exactly like a breach


@pydantic.dataclasses.dataclass(frozen=True)
class Guardrail:
    """
    A metric that must not move against us by more than `margin` (relative).
    Naming one without a margin is naming a wish, so the margin has a default
    rather than being optional.
    """

    name: str
    margin: float = 0.03
    direction: Literal["down_is_bad", "up_is_bad"] = "down_is_bad"


@dataclass(frozen=True)
class GuardrailResult:
    statuses: tuple[tuple[str, GuardrailStatus], ...] = ()
    # The bound on the harmful side of each guardrail's interval — the number a
    # reader needs to judge the call for themselves.
    bounds: tuple[tuple[str, float], ...] = ()
    blocks_ship: bool = False


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
