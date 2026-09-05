from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from peer_agent.types import Verdict

Flaw = Callable[[pd.DataFrame, np.random.Generator], pd.DataFrame]

BASELINE = 0.12  # was 0.10 in phase 1 — power_analysis's numbers assume this


def SRM(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    drop = df.arm.to_numpy() & (rng.random(len(df)) < 0.15)
    return df.loc[~drop].reset_index(drop=True)


def NOVELTY(
    df: pd.DataFrame,
    rng: np.random.Generator,
    *,
    half_life_days: float = 3.0,
    initial_boost: float = 0.6,
) -> pd.DataFrame:
    """A real early lift that fades. Needs make_case(..., days=...)."""
    if "day" not in df.columns:
        raise ValueError("NOVELTY needs make_case(..., days=...)")
    decay = np.exp(-df.day.to_numpy() / half_life_days)
    boost = initial_boost * decay
    arm = df.arm.to_numpy()
    propensity = df.pre_period_metric.to_numpy()
    rate = np.clip(np.where(arm, propensity * (1 + boost), propensity), 0, 1)
    return df.assign(converted=rng.random(len(df)) < rate)


def SIMPSON(
    df: pd.DataFrame,
    rng: np.random.Generator,
    *,
    within_segment_penalty: float = 0.92,
    skew: float = 0.55,
) -> pd.DataFrame:
    """
    Two hidden segments where treatment is slightly *worse* in every segment,
    but a composition skew between arms (treatment overrepresents the
    naturally-higher-converting segment) reverses the sign in aggregate.
    """
    n = len(df)
    arm = df.arm.to_numpy()
    segment = rng.random(n) < 0.5
    base = np.where(segment, 0.30, 0.05)
    rate = np.where(arm, base * within_segment_penalty, base)
    out = df.assign(segment=segment, converted=rng.random(n) < rate)
    drop_control_power = (~arm) & segment & (rng.random(n) < skew)
    drop_treat_casual = arm & (~segment) & (rng.random(n) < skew)
    return out.loc[~(drop_control_power | drop_treat_casual)].reset_index(drop=True)


def PEEKING(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Peeking isn't a property of the data, it's a property of how it's read:
    checked every day, stopped the moment it looks good. by_day() creates
    the repeated looks; this flaw is a named pass-through so a peeking case
    can be built and labelled like any other.
    """
    del rng
    return df


def OUTLIERS(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    A handful of extreme values skew pre_period_metric's distribution. No stats.py
    tool measures skewness — answering "is this skewed?" needs the agent to write
    its own code instead of reaching for a calibrated tool that doesn't exist.
    """
    n_outliers = max(1, len(df) // 200)
    idx = rng.choice(len(df), size=n_outliers, replace=False)
    out = df.copy()
    out.loc[idx, "pre_period_metric"] *= rng.uniform(20, 50, size=n_outliers)
    return out


@dataclass(frozen=True)
class Case:
    df: pd.DataFrame
    truth: Verdict
    _path: str | None = field(default=None, repr=False, compare=False)

    def by_day(self) -> list[pd.DataFrame]:
        if "day" not in self.df.columns:
            raise ValueError("by_day() needs make_case(..., days=...)")
        return [
            self.df.loc[self.df.day <= d].reset_index(drop=True)
            for d in range(int(self.df.day.max()) + 1)
        ]

    @property
    def path(self) -> str:
        """Lazily write df to a temp parquet so out-of-process tools (MCP) can load it."""
        path = self._path
        if path is None:
            fd, tmp = tempfile.mkstemp(suffix=".parquet")
            os.close(fd)
            self.df.to_parquet(tmp)
            object.__setattr__(self, "_path", tmp)
            path = tmp
        return path


def make_case(  # noqa: PLR0913 (a simulator needs independent knobs)
    n: int,
    lift: float = 0.0,
    *,
    seed: int = 0,
    flaws: list[Flaw] | None = None,
    days: int | None = None,
    segments: int | None = None,
) -> Case:
    rng = np.random.default_rng(seed)
    arm = rng.random(n) < 0.5
    # Beta shape with mean BASELINE; low enough (0.7) to keep the CUPED covariate
    # correlated with the outcome without swamping the fixed-seed lift estimate.
    shape_a = 0.7
    shape_b = shape_a * (1 - BASELINE) / BASELINE
    propensity = rng.beta(shape_a, shape_b, size=n)  # per-unit pre-experiment propensity
    rate = np.clip(np.where(arm, propensity * (1 + lift), propensity), 0, 1)
    converted = rng.random(n) < rate
    data: dict[str, np.ndarray] = {
        "arm": arm,
        "converted": converted,
        "pre_period_metric": propensity,
    }
    if days:
        data["day"] = rng.integers(0, days, size=n)
    if segments:
        data["segment"] = rng.integers(0, segments, size=n)
    df = pd.DataFrame(data)
    for flaw in flaws or []:
        df = flaw(df, rng)
    truth = Verdict.NO_EFFECT if lift == 0.0 else Verdict.SHIP
    return Case(df=df, truth=truth)
