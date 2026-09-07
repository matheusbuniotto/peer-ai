"""
Phase 4 — calibration.

Phases 1-3 ask whether the agent reaches for the right tool. This file asks
whether the tool's answer is right: given a known truth, repeated across seeds,
does each check fire at the rate it claims to? Properties, not fixed outputs —
a single seed can't tell a calibrated test from a lucky one.

The xfails are the defects docs/plans/phase-4-plan.md exists to fix, written as
executable evidence instead of prose. They are strict, so the day a fix lands
the marker itself fails the build until someone deletes it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import pytest

from peer_agent import stats
from peer_agent.sim import BASELINE, NOVELTY, SIMPSON, make_case
from peer_agent.types import Guardrail, GuardrailStatus

SEEDS = range(200, 240)


def null_case(seed: int, n: int = 8_000, **kw) -> pd.DataFrame:
    """An experiment where nothing happened, so every rejection is a false one."""
    return make_case(n=n, lift=0.0, seed=seed, **kw).df


def rate(fires: Callable[[int], bool], seeds: Sequence[int] | range = SEEDS) -> float:
    """How often a check fires across independent datasets."""
    return sum(bool(fires(seed)) for seed in seeds) / len(seeds)


# --------------------------------------------------------------------------- #
# What already holds
# --------------------------------------------------------------------------- #


def test_the_primary_test_holds_its_nominal_false_positive_rate():
    alpha = 0.05

    def fires(seed: int) -> bool:
        return stats.analyze(null_case(seed), alpha=alpha).p_value < alpha

    assert rate(fires) <= 0.15  # 40 seeds at alpha=0.05: 6 is already generous


def test_check_srm_does_not_cry_wolf_on_a_fair_split():
    assert rate(lambda seed: stats.check_srm(null_case(seed)).mismatch) == 0.0


def test_a_deliberate_holdout_is_not_a_broken_split():
    """A 90/10 holdout is a design, not a defect. The old check pinned ratio=1."""
    rng = np.random.default_rng(0)
    n = 60_000
    holdout = pd.DataFrame({"arm": rng.random(n) < 0.10, "converted": rng.random(n) < 0.12})

    assert stats.check_srm(holdout).mismatch  # judged against the wrong ratio
    assert not stats.check_srm(holdout, ratio=1 / 9).mismatch


def test_a_split_can_balance_overall_and_still_be_broken_inside_a_segment():
    """
    Differential dropout by arm *within* segment is what SIMPSON plants, and a
    global count check cannot see it — the arms come out even in aggregate.
    This is the composition skew, caught directly instead of inferred from a
    sign flip in scan_segments.
    """
    simpson = make_case(n=60_000, lift=0.0, seed=5, flaws=[SIMPSON]).df

    assert not stats.check_srm(simpson).mismatch
    assert stats.check_srm(simpson, by=("segment",)).mismatch


def test_naming_the_control_arm_does_not_change_the_answer():
    """The default control arm is implicit; making it explicit must be a no-op."""
    df = null_case(11)
    assert stats.analyze(df).lift == stats.analyze(df, control=False).lift


# --------------------------------------------------------------------------- #
# What doesn't — one xfail per plan item
# --------------------------------------------------------------------------- #


@pytest.mark.xfail(strict=True, reason="A3: no clustering — see plan ticket 08")
def test_duplicating_every_row_must_not_shrink_the_p_value():
    """
    The same experiment with each unit counted twice is not twice the evidence,
    but nothing in stats.py knows that: on this seed the duplicate rows alone
    carry a null result from p=0.082 to p=0.014 and manufacture a win. Any
    dataset with repeat visitors is already this dataset.
    """
    df = null_case(9, n=6_000)
    doubled = pd.concat([df, df], ignore_index=True)
    assert stats.analyze(doubled).p_value >= stats.analyze(df).p_value


def test_novelty_does_not_fire_on_a_steady_effect():
    """A lift that never decays must not be reported as decaying."""

    def fires(seed: int) -> bool:
        steady = make_case(n=20_000, lift=0.06, seed=seed, days=14).df
        return stats.check_novelty(steady).decaying

    assert rate(fires) <= 0.10


def test_a_real_decay_is_still_caught_every_time():
    """Quiet on nulls is only worth having if it stays loud on the real thing."""

    def fires(seed: int) -> bool:
        fading = make_case(n=60_000, seed=seed, days=14, flaws=[NOVELTY]).df
        return stats.check_novelty(fading).decaying

    assert rate(fires, range(1, 11)) == 1.0


def test_the_decay_verdict_carries_its_own_uncertainty():
    """A slope with no interval around it is the old magic threshold again."""
    result = stats.check_novelty(make_case(n=60_000, seed=1, days=14, flaws=[NOVELTY]).df)
    low, high = result.slope_ci

    assert low < result.slope < high < 0
    assert len(result.daily_lifts) > 3
    assert result.cohort_day is False  # only a calendar day was available


def test_a_segment_scan_does_not_invent_a_reversal():
    """Under a true null every segment's sign is a coin flip, not a finding."""

    def fires(seed: int) -> bool:
        return stats.scan_segments(null_case(seed, n=20_000, segments=3)).reversal

    assert rate(fires) <= 0.10


def test_a_real_reversal_still_registers():
    """The fix must not have bought its quiet by going blind."""
    simpson = make_case(n=60_000, lift=0.0, seed=5, flaws=[SIMPSON]).df

    assert stats.scan_segments(simpson).reversal


def test_a_composition_skew_is_reported_as_counts_not_as_heterogeneity():
    """
    SIMPSON penalises treatment equally in both segments, so one effect really
    does explain them — Q is right not to fire. What's wrong is the mix, and
    the per-segment split check is the thing that says so.
    """
    result = stats.scan_segments(make_case(n=60_000, lift=0.0, seed=5, flaws=[SIMPSON]).df)

    assert result.heterogeneity_p > 0.05
    assert all(broken for _, broken in result.composition_srm)


def _guardrail_case(treatment_rate: float, n: int = 20_000, seed: int = 0):
    """
    A frame whose `revenue` guardrail moves from 0.120 to `treatment_rate`.
    The draw order matters — revenue takes the second stream, which is what
    makes seed 0 the worked example from the plan.
    """
    rng = np.random.default_rng(seed)
    arm = rng.random(n) < 0.5
    revenue = rng.random(n) < np.where(arm, treatment_rate, 0.120)
    return pd.DataFrame(
        {
            "arm": arm,
            "converted": np.random.default_rng(seed + 1).random(n) < BASELINE,
            "revenue": revenue.astype(float),
        }
    )


def test_a_guardrail_too_wide_to_clear_is_not_reported_clean():
    """
    The worked example: revenue down ~7%, one-sided bound reaching past -12%,
    two-sided p=0.058. A superiority test can't reject at 0.01, calls it clean
    and ships the regression. Non-inferiority asks whether harm past the 3%
    margin is ruled out, and here it plainly isn't.
    """
    result = stats.check_guardrails(_guardrail_case(0.118), ("revenue",))

    assert dict(result.statuses)["revenue"] is GuardrailStatus.INCONCLUSIVE
    assert result.blocks_ship
    assert dict(result.bounds)["revenue"] < -0.03


def test_an_unmeasured_guardrail_does_not_pass_by_default():
    """Naming a guardrail with no column behind it used to read as clean."""
    result = stats.check_guardrails(_guardrail_case(0.120), ("latency_p95",))

    assert dict(result.statuses)["latency_p95"] is GuardrailStatus.INCONCLUSIVE
    assert result.blocks_ship


def test_a_guardrail_that_really_is_flat_clears():
    """Scepticism that blocks everything is as useless as none at all."""
    result = stats.check_guardrails(
        _guardrail_case(0.120, n=400_000), (Guardrail("revenue", margin=0.05),)
    )

    assert dict(result.statuses)["revenue"] is GuardrailStatus.CLEAN
    assert not result.blocks_ship


def test_a_real_regression_past_the_margin_breaches():
    result = stats.check_guardrails(
        _guardrail_case(0.090, n=200_000), (Guardrail("revenue", margin=0.05),)
    )

    assert dict(result.statuses)["revenue"] is GuardrailStatus.BREACHED
    assert result.blocks_ship


# --------------------------------------------------------------------------- #
# A null is not the same as not enough data
# --------------------------------------------------------------------------- #


def test_a_flat_result_is_only_a_null_when_the_mde_is_excluded():
    """
    Same data, same p-value, two different answers. Without an MDE all you can
    say is "not significant"; with one you can say whether the experiment was
    big enough for that to mean anything.
    """
    df = null_case(3, n=60_000)

    assert stats.analyze(df).equivalent_to_null is None
    assert stats.analyze(df, mde=0.20).equivalent_to_null is True
    assert stats.analyze(df, mde=0.01).equivalent_to_null is False
