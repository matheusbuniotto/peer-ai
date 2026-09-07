from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pyarrow as pa
import tea_tasting as tt
from scipy import stats as sps
from tea_tasting.aggr import Aggregates

from peer_agent.types import BY_SESSION, DesignSpec, PowerResult, SimulationResult, Verdict

_ALPHA = 0.05
_OUTCOME = "converted"
_MAX_PLAUSIBLE_MDE = 0.25  # policy constant, not statistical — see phase-2 plan
_SESSION_CONTAMINATION = (
    0.5  # fraction of a per-user effect that survives session-level split
)


def power_analysis(
    baseline: float, mde: float, power: float = 0.8, alpha: float = _ALPHA
) -> PowerResult:
    """Sample size for a relative `mde` on a `baseline` conversion rate."""
    prior = Aggregates(
        count_=1,
        mean_={_OUTCOME: baseline},
        var_={_OUTCOME: baseline * (1 - baseline)},
    )
    metric = tt.Mean(_OUTCOME, alpha=alpha, power=power, rel_effect_size=mde)
    n_total = metric.solve_power_from_aggregates(prior, "n_obs")[0].n_obs
    return PowerResult(n_per_arm=math.ceil(n_total / 2), n_total=math.ceil(n_total))


def simulate_design(
    spec: DesignSpec, lift: float, runs: int, seed: int
) -> SimulationResult:
    """
    Plants `lift` on spec.baseline and re-randomizes the design `runs` times,
    measuring what actually happens instead of trusting the arithmetic.
    """
    rng = np.random.default_rng(seed)
    effective = lift * _SESSION_CONTAMINATION if spec.unit is BY_SESSION else lift
    pool = pd.DataFrame(
        {_OUTCOME: (rng.random(spec.n_per_arm * 2) < spec.baseline).astype(int)}
    )

    experiment = tt.Experiment(
        {"conversion": tt.Mean(_OUTCOME, confidence_level=1 - spec.alpha)},
        variant="variant",
    )
    results = experiment.simulate(
        pool, runs, rng=rng, treat=_treatment(spec.baseline, effective, rng)
    ).to_pandas()

    rejections = int((results.pvalue < spec.alpha).sum())
    low, high = sps.binomtest(rejections, runs).proportion_ci()
    return SimulationResult(
        power=rejections / runs,
        promised=spec.power,
        warnings=tuple(_warnings(spec)),
        verdict=Verdict.INVALID if spec.mde > _MAX_PLAUSIBLE_MDE else Verdict.SHIP,
        mean_estimate=float(results.rel_effect_size.mean()),
        runs=runs,
        power_ci=(low, high),
        shortfall=high < spec.power,
        bias=float(results.rel_effect_size.mean()) - effective,
    )


def _treatment(baseline: float, lift: float, rng: np.random.Generator):
    """Converts a share of the treated arm's non-converters, planting `lift` exactly."""
    extra = lift * baseline / (1 - baseline)

    def treat(table: pa.Table) -> pa.Table:
        outcome = table.column(_OUTCOME).to_numpy()
        converted = np.where(outcome == 1, 1, rng.random(len(outcome)) < extra)
        index = table.schema.get_field_index(_OUTCOME)
        return table.set_column(index, _OUTCOME, pa.array(converted.astype(int)))

    return treat


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
    if spec.daily_traffic and spec.n_per_arm * 2 > spec.daily_traffic * spec.days:
        needed = math.ceil(spec.n_per_arm * 2 / spec.daily_traffic)
        warnings.append(
            f"{spec.n_per_arm * 2:,} units at {spec.daily_traffic:,}/day needs "
            f"{needed} days, not {spec.days}"
        )
    return warnings
