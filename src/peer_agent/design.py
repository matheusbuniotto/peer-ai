from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as sps

from peer_agent import stats
from peer_agent.types import BY_SESSION, DesignSpec, PowerResult, SimulationResult, Verdict

_ALPHA = 0.05
_MAX_PLAUSIBLE_MDE = 0.25  # policy constant, not statistical — see phase-2 plan
_SESSION_CONTAMINATION = (
    0.5  # fraction of a per-user effect that survives session-level split
)


def power_analysis(
    baseline: float, mde: float, power: float = 0.8, alpha: float = _ALPHA
) -> PowerResult:
    """Textbook two-proportion z-test sample size."""
    p1, p2 = baseline, baseline * (1 + mde)
    pooled = (p1 + p2) / 2
    z_alpha = sps.norm.ppf(1 - alpha / 2)
    z_power = sps.norm.ppf(power)
    numerator = z_alpha * math.sqrt(2 * pooled * (1 - pooled)) + z_power * math.sqrt(
        p1 * (1 - p1) + p2 * (1 - p2)
    )
    return PowerResult(n_per_arm=math.ceil((numerator / (p2 - p1)) ** 2))


def simulate_design(
    spec: DesignSpec, lift: float, runs: int, seed: int
) -> SimulationResult:
    """
    Plants `lift` on spec.baseline and re-simulates the design `runs` times,
    measuring what actually happens instead of trusting the arithmetic.
    """
    rng = np.random.default_rng(seed)
    effective_lift = lift * _SESSION_CONTAMINATION if spec.unit is BY_SESSION else lift
    p1, p2 = spec.baseline, spec.baseline * (1 + effective_lift)

    rejections = 0
    estimates: list[float] = []
    for _ in range(runs):
        arm = rng.random(spec.n_per_arm * 2) < 0.5
        rate = np.where(arm, p2, p1)
        converted = rng.random(len(arm)) < rate
        result = stats.analyze(
            pd.DataFrame({"arm": arm, "converted": converted}), alpha=spec.alpha
        )
        rejections += result.p_value < spec.alpha
        estimates.append(result.lift)

    return SimulationResult(
        power=rejections / runs,
        promised=spec.power,
        warnings=tuple(_warnings(spec)),
        verdict=Verdict.INVALID if spec.mde > _MAX_PLAUSIBLE_MDE else Verdict.SHIP,
        mean_estimate=float(np.mean(estimates)),
    )


def _warnings(spec: DesignSpec) -> list[str]:
    warnings = []
    if spec.unit is BY_SESSION:
        warnings.append(
            "randomizing by session lets one user land in both arms, "
            "contaminating a per-user effect"
        )
    if spec.days % 7 != 0:
        warnings.append(f"{spec.days} days is not a whole number of weeks")
    if not spec.guardrails:
        warnings.append("no guardrail metrics named")
    if not spec.if_flat:
        warnings.append("no decision attached to a flat result")
    return warnings
