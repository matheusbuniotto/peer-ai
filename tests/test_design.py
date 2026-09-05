"""
Grading a design.

There is no single correct sample size, which is why most people give up on
testing design. But there is ground truth: plant a known effect, run the
proposed design a few hundred times, and measure what happens. A design that
promises 80% power and delivers 42% is wrong, provably.

simulate_design takes a DesignSpec and returns realized power, the false
positive rate, and a list of warnings. Nothing about the grade is opinion.
"""

from __future__ import annotations

import pytest
from peer_agent.design import simulate_design
from peer_agent.types import BY_SESSION, BY_USER, DesignSpec, Verdict

pytestmark = pytest.mark.slow


def spec(**overrides):
    base = {
        "baseline": 0.12,
        "mde": 0.08,
        "power": 0.80,
        "days": 14,
        "n_per_arm": 18_000,
        "unit": BY_USER,
        "metric": "purchase_rate",
        "guardrails": ("latency_p95", "refund_rate"),
        "if_flat": "keep the current flow",
    }
    return DesignSpec(**{**base, **overrides})


# --------------------------------------------------------------------------- #
# The core loop
# --------------------------------------------------------------------------- #


def test_a_sound_design_delivers_what_it_promises():
    result = simulate_design(spec(), lift=0.08, runs=500, seed=0)
    assert 0.75 <= result.power <= 0.87
    assert not result.warnings


def test_power_is_measured_not_echoed_back():
    result = simulate_design(spec(n_per_arm=2_000), lift=0.08, runs=500, seed=0)
    assert result.promised == 0.80
    assert result.power < 0.4


def test_a_design_with_no_effect_to_find_rejects_at_alpha():
    assert (
        0.02 <= simulate_design(spec(), lift=0.0, runs=800, seed=1).false_positives <= 0.09
    )


# --------------------------------------------------------------------------- #
# Randomization unit, the mistake that silently halves power
# --------------------------------------------------------------------------- #


def test_randomizing_by_session_costs_power():
    """Same traffic, same effect, same days. Only the unit changes."""
    user = simulate_design(spec(unit=BY_USER), lift=0.08, runs=500, seed=2)
    session = simulate_design(spec(unit=BY_SESSION), lift=0.08, runs=500, seed=2)
    assert session.power < user.power - 0.15


def test_randomizing_by_session_inflates_the_error_rate_too():
    """Correlated sessions don't just widen intervals, they break the test."""
    assert (
        simulate_design(spec(unit=BY_SESSION), lift=0.0, runs=800, seed=3).false_positives
        > 0.09
    )


def test_the_unit_problem_is_named_in_the_warning():
    """A low number is no use to a PM without the reason attached."""
    result = simulate_design(spec(unit=BY_SESSION), lift=0.08, runs=300, seed=4)
    assert "randomi" in " ".join(result.warnings).lower()


# --------------------------------------------------------------------------- #
# Duration
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("days", [3, 5, 10, 11])
def test_partial_weeks_are_flagged(days):
    result = simulate_design(spec(days=days), lift=0.05, runs=200, seed=5)
    assert any("week" in w for w in result.warnings)


@pytest.mark.parametrize("days", [7, 14, 21])
def test_whole_weeks_are_not(days):
    result = simulate_design(spec(days=days), lift=0.05, runs=200, seed=6)
    assert not any("week" in w for w in result.warnings)


def test_a_test_shorter_than_the_novelty_window_is_flagged():
    result = simulate_design(spec(days=7), lift=0.0, runs=200, seed=7, novelty_days=9)
    assert any("novelty" in w.lower() for w in result.warnings)


# --------------------------------------------------------------------------- #
# Tests that cannot answer their own question
# --------------------------------------------------------------------------- #


def test_an_mde_larger_than_any_plausible_effect_is_rejected():
    """A test that can only see a 40% lift on mature checkout is a foregone conclusion."""
    result = simulate_design(spec(mde=0.40), lift=0.05, runs=300, seed=8)
    assert result.verdict is Verdict.INVALID


def test_a_test_with_no_decision_attached_is_rejected():
    """If a flat result changes nothing, don't run it."""
    result = simulate_design(spec(if_flat=None), lift=0.05, runs=100, seed=9)
    assert any("decision" in w.lower() for w in result.warnings)


def test_missing_guardrails_are_flagged():
    result = simulate_design(spec(guardrails=()), lift=0.05, runs=100, seed=10)
    assert any("guardrail" in w.lower() for w in result.warnings)


# --------------------------------------------------------------------------- #
# Interference
# --------------------------------------------------------------------------- #


def test_leakage_between_arms_pulls_the_estimate_toward_zero():
    """Marketplace and social features break the no-interference assumption."""
    clean = simulate_design(spec(), lift=0.10, runs=400, seed=11)
    leaky = simulate_design(spec(), lift=0.10, runs=400, seed=11, interference=0.4)
    assert leaky.mean_estimate < clean.mean_estimate * 0.8


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #


def test_same_seed_same_answer():
    a = simulate_design(spec(), lift=0.06, runs=200, seed=42)
    b = simulate_design(spec(), lift=0.06, runs=200, seed=42)
    assert a == b


def test_different_seed_different_answer():
    a = simulate_design(spec(), lift=0.06, runs=200, seed=1)
    b = simulate_design(spec(), lift=0.06, runs=200, seed=2)
    assert a.mean_estimate != b.mean_estimate


# --------------------------------------------------------------------------- #
# The comparison the report page publishes
# --------------------------------------------------------------------------- #


def test_the_considered_design_beats_the_naive_one():
    naive = simulate_design(
        spec(days=10, n_per_arm=9_000, unit=BY_SESSION), lift=0.08, runs=500, seed=7
    )
    considered = simulate_design(
        spec(days=14, n_per_arm=18_000, unit=BY_USER, cuped=True),
        lift=0.08,
        runs=500,
        seed=7,
    )
    assert naive.power < 0.45
    assert considered.power > 0.85
    assert considered.days > naive.days, "don't claim the better design is also faster"
