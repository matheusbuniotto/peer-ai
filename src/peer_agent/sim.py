from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from peer_agent.types import Verdict

Flaw = Callable[[pd.DataFrame, np.random.Generator], pd.DataFrame]


@dataclass(frozen=True)
class Case:
    df: pd.DataFrame
    truth: Verdict


def SRM(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    drop = df.arm.to_numpy() & (rng.random(len(df)) < 0.15)
    return df.loc[~drop].reset_index(drop=True)


def NOVELTY(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    raise NotImplementedError("phase 3")


def SIMPSON(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    raise NotImplementedError("phase 3")


def make_case(
    n: int,
    lift: float = 0.0,
    seed: int = 0,
    flaws: list[Flaw] | None = None,
    days: int | None = None,  # noqa: ARG001 (reserved for phase 3's NOVELTY flaw)
) -> Case:
    rng = np.random.default_rng(seed)
    arm = rng.random(n) < 0.5
    baseline = 0.10
    rate = np.where(arm, baseline * (1 + lift), baseline)
    converted = rng.random(n) < rate
    df = pd.DataFrame({"arm": arm, "converted": converted})
    for flaw in flaws or []:
        df = flaw(df, rng)
    truth = Verdict.NO_EFFECT if lift == 0.0 else Verdict.SHIP
    return Case(df=df, truth=truth)
