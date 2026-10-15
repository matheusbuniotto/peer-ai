from __future__ import annotations

import math
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
    """Stop at the first day d >= 3 whose cumulative test has p < 0.05.

    The rows are generated under the true lift (the suite uses 0). Truncating
    at an opportunistic look makes the final p-value an optional-stopping
    artifact, so the case's truth is INVALID. `make_case` walks seeds until
    this happens and records the seed that worked on `df.attrs`.
    """
    del rng
    if "day" not in df.columns:
        raise ValueError("PEEKING needs make_case(..., days=...)")
    from peer_agent import stats

    last = int(df.day.max())
    for day in range(3, last + 1):
        snap = df.loc[df.day <= day]
        if _two_prop_p(snap) >= 0.05 or stats.analyze(snap).p_value >= 0.05:
            continue
        out = snap.reset_index(drop=True)
        out.attrs["stopped_day"] = int(day)
        out.attrs["looks"] = int(day) + 1
        out.attrs["planned_days"] = int(df.attrs.get("planned_days", last + 1))
        return out
    return df


def _two_prop_p(df: pd.DataFrame) -> float:
    """Fast screen. `stats.analyze` confirms a hit before the frame is kept."""
    arm = df.arm.to_numpy().astype(bool)
    converted = df.converted.to_numpy().astype(float)
    n_t = int(arm.sum())
    n_c = int((~arm).sum())
    if n_t == 0 or n_c == 0:
        return 1.0
    pooled = float(converted.mean())
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_t + 1 / n_c))
    if se == 0:
        return 1.0
    z = (float(converted[arm].mean()) - float(converted[~arm].mean())) / se
    return math.erfc(abs(z) / math.sqrt(2))


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


def REPEATS(
    df: pd.DataFrame,
    rng: np.random.Generator,
    *,
    returning_share: float = 0.3,
    max_extra_visits: float = 4,
) -> pd.DataFrame:
    """
    Some users come back, and every visit is logged as its own row. Nothing
    about the effect changes — the same people, the same arms, the same
    outcomes — only the row count does.

    Analysed per row, those extra rows read as extra evidence and shrink every
    p-value. Analysed per user, they say nothing new. This is the most common
    shape of real experiment data and the one stats.py assumed away.
    """
    out = df.assign(user_id=np.arange(len(df)))
    returning = rng.random(len(df)) < returning_share
    extra = rng.integers(1, int(max_extra_visits) + 1, size=len(df)) * returning
    return out.loc[out.index.repeat(1 + extra)].reset_index(drop=True)


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
    planted = flaws or []
    if PEEKING in planted:
        df = _peek_until_it_stops(n, lift, seed, days, segments, planted)
        return Case(df=df, truth=Verdict.INVALID)
    rng = np.random.default_rng(seed)
    df = _populate(n, lift, rng, days, segments)
    for flaw in planted:
        df = flaw(df, rng)
    truth = Verdict.NO_EFFECT if lift == 0.0 else Verdict.SHIP
    return Case(df=df, truth=truth)


def _populate(
    n: int,
    lift: float,
    rng: np.random.Generator,
    days: int | None,
    segments: int | None,
) -> pd.DataFrame:
    arm = rng.random(n) < 0.5
    # Beta shape with mean BASELINE; low enough (0.7) to keep the CUPED covariate
    # correlated with the outcome without swamping the fixed-seed lift estimate.
    shape_a = 0.7
    shape_b = shape_a * (1 - BASELINE) / BASELINE
    propensity = rng.beta(shape_a, shape_b, size=n)
    rate = np.clip(np.where(arm, propensity * (1 + lift), propensity), 0, 1)
    data: dict[str, np.ndarray] = {
        "arm": arm,
        "converted": rng.random(n) < rate,
        "pre_period_metric": propensity,
    }
    if days:
        data["day"] = rng.integers(0, days, size=n)
    if segments:
        data["segment"] = rng.integers(0, segments, size=n)
    return pd.DataFrame(data)


def _peek_until_it_stops(  # noqa: PLR0913, PLR0917 (the same knobs make_case already takes)
    n: int,
    lift: float,
    seed: int,
    days: int | None,
    segments: int | None,
    planted: list[Flaw],
) -> pd.DataFrame:
    if not days:
        raise ValueError("PEEKING needs make_case(..., days=...)")
    others = [flaw for flaw in planted if flaw is not PEEKING]
    for attempt in range(50):
        trial = seed + attempt * 1000
        rng = np.random.default_rng(trial)
        df = _populate(n, lift, rng, days, segments)
        df.attrs["planned_days"] = days
        for flaw in others:
            df = flaw(df, rng)
        peeked = PEEKING(df, rng)
        if "stopped_day" in peeked.attrs:
            peeked.attrs["peeking_seed"] = trial
            return peeked
    raise RuntimeError(f"PEEKING found no optional stop within 50 seeds of {seed}")
