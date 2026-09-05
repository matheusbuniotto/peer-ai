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
from peer_agent.sim import BASELINE, make_case

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


@pytest.mark.xfail(strict=True, reason="A4: threshold, not a test — see plan ticket 09")
def test_novelty_does_not_fire_on_a_steady_effect():
    """A lift that never decays must not be reported as decaying."""

    def fires(seed: int) -> bool:
        steady = make_case(n=20_000, lift=0.06, seed=seed, days=14).df
        return stats.check_novelty(steady).decaying

    assert rate(fires) <= 0.10


@pytest.mark.xfail(strict=True, reason="A7: sign flip, not a test — see plan ticket 11")
def test_a_segment_scan_does_not_invent_a_reversal():
    """Under a true null every segment's sign is a coin flip, not a finding."""

    def fires(seed: int) -> bool:
        return stats.scan_segments(null_case(seed, n=20_000, segments=3)).reversal

    assert rate(fires) <= 0.10


@pytest.mark.xfail(strict=True, reason="A1: fails open — see plan ticket 04")
def test_a_guardrail_too_wide_to_clear_is_not_reported_clean():
    """
    The worked example from the plan: revenue down ~7%, interval reaching -13%,
    two-sided p=0.058. A superiority test calls that clean and ships it. The
    question a guardrail asks is whether harm past the margin can be ruled out,
    and here it plainly can't.

    Ticket 04 replaces `.breached` with a three-state status; this assertion
    gets rewritten there to expect INCONCLUSIVE rather than CLEAN.
    """
    rng = np.random.default_rng(0)
    arm = rng.random(20_000) < 0.5
    df = pd.DataFrame(
        {
            "arm": arm,
            "converted": rng.random(20_000) < BASELINE,
            "revenue": (rng.random(20_000) < np.where(arm, 0.118, 0.120)).astype(float),
        }
    )
    assert stats.check_guardrails(df, ("revenue",)).breached
