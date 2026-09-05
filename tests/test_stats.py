"""
Calibration for stats.py.

These are not example tests. Checking that analyze() returns 0.031 on a fixture
only proves it still does what it did yesterday. These check the estimators
have the properties they claim: right error rates over many repeats, unbiased,
monotonic, order-independent.

Everything in stats.py is a function that takes a dataframe and returns a
frozen dataclass, so each test here reads as arrange, call, assert.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from peer_agent.design import power_analysis
from peer_agent.sim import make_case
from peer_agent.types import Verdict
from scipy import stats as sps

from peer_agent import stats

ALPHA = 0.05
REPS = 1000


def rate_ci(hits, n):
    """Wide interval, so these fail on bugs rather than on noise."""
    return sps.binomtest(hits, n).proportion_ci(confidence_level=0.999)


# --------------------------------------------------------------------------- #
# The false positive rate is the whole ballgame
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_false_positives_land_on_alpha():
    hits = sum(
        stats.analyze(make_case(n=20_000, lift=0.0, seed=s).df).p_value < ALPHA
        for s in range(REPS)
    )
    lo, hi = rate_ci(hits, REPS)
    assert lo <= ALPHA <= hi, f"rejecting {hits / REPS:.3f} of true nulls"


@pytest.mark.slow
def test_null_p_values_are_uniform():
    """Stronger than the rate alone: the whole distribution is right."""
    ps = [
        stats.analyze(make_case(n=20_000, lift=0.0, seed=s).df).p_value for s in range(500)
    ]
    assert sps.kstest(ps, "uniform").pvalue > 0.01


@pytest.mark.slow
def test_the_estimate_is_unbiased():
    est = [
        stats.analyze(make_case(n=40_000, lift=0.05, seed=s).df).lift for s in range(300)
    ]
    mean = float(np.mean(est))
    se = float(np.std(est, ddof=1)) / np.sqrt(len(est))
    assert abs(mean - 0.05) < 3 * se, f"biased: {mean:.4f}"


@pytest.mark.slow
def test_intervals_cover_the_truth_95_percent_of_the_time():
    hits = sum(
        stats.analyze(make_case(n=30_000, lift=0.04, seed=s).df).covers(0.04)
        for s in range(REPS)
    )
    lo, hi = rate_ci(hits, REPS)
    assert lo <= 0.95 <= hi, f"coverage {hits / REPS:.3f}"


# --------------------------------------------------------------------------- #
# Variance reduction
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_cuped_narrows_the_interval_without_moving_the_estimate():
    plain, adjusted = [], []
    for s in range(200):
        df = make_case(n=30_000, lift=0.05, seed=s).df
        plain.append(stats.analyze(df))
        adjusted.append(stats.analyze(df, cuped=True))

    assert np.mean([r.width for r in adjusted]) < np.mean([r.width for r in plain]) * 0.95
    assert abs(np.mean([r.lift for r in adjusted]) - 0.05) < 0.006


def test_cuped_rejects_a_covariate_from_after_assignment(clean):
    """Adjusting on a post-treatment column biases the answer. Refuse loudly."""
    with pytest.raises(ValueError, match="after assignment"):
        stats.analyze(clean.df, cuped=True, covariate="post_sessions")


# --------------------------------------------------------------------------- #
# Sample ratio mismatch
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_srm_does_not_cry_wolf():
    fired = sum(
        stats.check_srm(make_case(n=20_000, seed=s).df).mismatch for s in range(REPS)
    )
    _, hi = rate_ci(fired, REPS)
    assert hi < 0.02, f"false alarms at {fired / REPS:.4f}"


def test_srm_catches_a_realistic_skew(srm):
    """A point and a half of imbalance. Small, common, fatal."""
    assert stats.check_srm(srm.df).mismatch


@given(n=st.integers(min_value=200, max_value=5_000))
@settings(max_examples=25, deadline=None)
def test_srm_scales_with_sample_size(n):
    """Use a test statistic, not a hardcoded ratio, or small tests all fail."""
    assert not stats.check_srm(make_case(n=n, seed=n).df).mismatch


# --------------------------------------------------------------------------- #
# Peeking
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_sequential_holds_alpha_under_daily_peeking():
    """First check the naive test really does break, then that this one doesn't."""
    naive = seq = 0
    for s in range(400):
        looks = make_case(n=28_000, lift=0.0, days=14, seed=s).by_day()
        naive += any(stats.analyze(d).p_value < ALPHA for d in looks)
        seq += any(stats.sequential(d, looks=14).reject for d in looks)

    assert naive / 400 > 0.15, "the peeking problem is not being reproduced"
    _, hi = rate_ci(seq, 400)
    assert hi < 0.08, f"leaking alpha at {seq / 400:.3f}"


@pytest.mark.slow
def test_sequential_still_finds_a_real_effect():
    """Alpha control is trivial if you never reject."""
    hits = sum(
        any(
            stats.sequential(d, looks=14).reject
            for d in make_case(n=60_000, lift=0.08, days=14, seed=s).by_day()
        )
        for s in range(200)
    )
    assert hits / 200 > 0.6


# --------------------------------------------------------------------------- #
# Segments
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_scanning_twenty_segments_does_not_manufacture_a_winner():
    found = sum(
        bool(
            stats.scan_segments(
                make_case(n=40_000, lift=0.0, segments=20, seed=s).df
            ).winners
        )
        for s in range(300)
    )
    _, hi = rate_ci(found, 300)
    assert hi < 0.12, f"phantom winners {found / 300:.3f} of the time"


def test_a_scan_after_a_null_primary_is_exploratory(clean):
    scan = stats.scan_segments(clean.df, primary_null=True)
    assert scan.exploratory
    assert not scan.supports_shipping


def test_simpsons_reversal_is_caught(simpson):
    assert stats.scan_segments(simpson.df).reversal


# --------------------------------------------------------------------------- #
# Novelty
# --------------------------------------------------------------------------- #


def test_a_decaying_lift_is_not_a_durable_one(novelty):
    durable = make_case(n=60_000, lift=0.06, days=14, seed=32)
    assert stats.check_novelty(novelty.df).decaying
    assert not stats.check_novelty(durable.df).decaying


def test_decay_means_extend_not_kill(novelty):
    """A fading lift is not proof of no effect. Don't conflate them."""
    assert stats.check_novelty(novelty.df).verdict is Verdict.EXTEND


# --------------------------------------------------------------------------- #
# Power
# --------------------------------------------------------------------------- #


@given(base=st.floats(0.01, 0.5), mde=st.floats(0.02, 0.3))
@settings(max_examples=50, deadline=None)
def test_smaller_effects_need_more_traffic(base, mde):
    assert power_analysis(base, mde / 2).n_per_arm > power_analysis(base, mde).n_per_arm


@given(power=st.floats(0.5, 0.95))
@settings(max_examples=30, deadline=None)
def test_more_power_costs_more_traffic(power):
    lo = power_analysis(0.1, 0.05, power=power)
    hi = power_analysis(0.1, 0.05, power=min(power + 0.04, 0.99))
    assert hi.n_per_arm >= lo.n_per_arm


@pytest.mark.slow
def test_the_power_i_promise_is_the_power_you_get():
    """
    The claim the whole project rests on. If power_analysis says 80%, running
    that design 400 times has to detect the effect about 80% of the time.
    """
    n = power_analysis(0.12, 0.08, power=0.8).n_per_arm
    hits = sum(
        stats.analyze(make_case(n=n * 2, lift=0.08, seed=s).df).p_value < ALPHA
        for s in range(400)
    )
    lo, hi = rate_ci(hits, 400)
    assert lo <= 0.80 <= hi, f"realized {hits / 400:.3f}, promised 0.80"


# --------------------------------------------------------------------------- #
# Invariance
# --------------------------------------------------------------------------- #


def test_row_order_does_not_matter(real_win):
    a = stats.analyze(real_win.df)
    b = stats.analyze(real_win.df.sample(frac=1, random_state=9))
    assert a.p_value == pytest.approx(b.p_value, rel=1e-9)


def test_swapping_the_arms_flips_the_sign_and_nothing_else(real_win):
    a = stats.analyze(real_win.df)
    b = stats.analyze(real_win.df.assign(arm=~real_win.df.arm))
    assert b.lift == pytest.approx(-a.lift, rel=1e-6)
    assert b.p_value == pytest.approx(a.p_value, rel=1e-9)


def test_an_empty_arm_raises(real_win):
    with pytest.raises(ValueError, match="no observations"):
        stats.analyze(real_win.df[real_win.df.arm])
