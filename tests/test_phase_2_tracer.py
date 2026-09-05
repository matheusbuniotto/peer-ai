"""
Phase 2 — make it true.

Phase 1 proved the wiring. Nothing in it proved a single number was right. This
phase turns the thread into something a data scientist would trust: calibrated
statistics, the remaining flaws, the design simulator, and the first live tests
where a real model has to show judgement.

Definition of done:

    pytest tests/test_phase_2_correct.py -m "not live"    # green, minutes
    pytest tests/test_stats.py -m "not slow"              # green
    pytest tests/test_design.py                           # green
    pytest tests/test_phase_2_correct.py -m live          # green, costs money

This file is the checkpoint. test_stats.py and test_design.py are the permanent
grade and go deeper; if they disagree with anything here, they win.

What I have to build for this:

    sim.py        NOVELTY, SIMPSON, PEEKING, OUTLIERS. by_day() for repeated looks.
    stats.py      sequential, scan_segments, check_novelty, check_guardrails, CUPED
    design.py     power_analysis, simulate_design, randomization units
    prompt.md     the protocol the agent follows
    agent.py      design() alongside review(), and the ask tool

Still not here: the sandbox, MCP, the eval harness, the report page. All phase 3.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from conftest import args_for, called_before, names, quotes_number
from scipy import stats as sps

from peer_agent import stats
from peer_agent.design import power_analysis, simulate_design
from peer_agent.sim import NOVELTY, PEEKING, SIMPSON, make_case
from peer_agent.types import BY_SESSION, BY_USER, DesignSpec, Verdict

# --------------------------------------------------------------------------- #
# The statistics are calibrated, not just plausible
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_the_false_positive_rate_is_five_percent():
    """
    The gate for this whole phase. Until this passes, every number the agent
    produces is decoration.
    """
    hits = sum(
        stats.analyze(make_case(n=20_000, lift=0.0, seed=s).df).p_value < 0.05
        for s in range(600)
    )
    lo, hi = sps.binomtest(hits, 600).proportion_ci(confidence_level=0.999)
    assert lo <= 0.05 <= hi, f"rejecting {hits / 600:.3f} of true nulls"


@pytest.mark.slow
def test_the_estimate_is_centred_on_the_truth():
    est = [
        stats.analyze(make_case(n=40_000, lift=0.05, seed=s).df).lift for s in range(200)
    ]
    se = float(np.std(est, ddof=1)) / np.sqrt(len(est))
    assert abs(float(np.mean(est)) - 0.05) < 3 * se


@pytest.mark.slow
def test_the_promised_power_is_the_delivered_power():
    """power_analysis is the thing the design phase leans on. It has to be honest."""
    n = power_analysis(0.12, 0.08, power=0.8).n_per_arm
    hits = sum(
        stats.analyze(make_case(n=n * 2, lift=0.08, seed=s).df).p_value < 0.05
        for s in range(300)
    )
    lo, hi = sps.binomtest(hits, 300).proportion_ci(confidence_level=0.999)
    assert lo <= 0.80 <= hi


# --------------------------------------------------------------------------- #
# The remaining flaws exist and are detectable
# --------------------------------------------------------------------------- #


def test_every_flaw_produces_data_that_looks_wrong():
    """
    A flaw that doesn't change the data grades nothing. Each one has to be
    visible to the check built for it.
    """
    assert stats.check_novelty(
        make_case(n=60_000, flaws=[NOVELTY], days=14, seed=1).df
    ).decaying
    assert stats.scan_segments(make_case(n=60_000, flaws=[SIMPSON], seed=2).df).reversal

    peeked = make_case(n=28_000, lift=0.0, flaws=[PEEKING], days=14, seed=3)
    assert any(stats.analyze(d).p_value < 0.05 for d in peeked.by_day())


@pytest.mark.slow
def test_the_sequential_test_holds_where_the_naive_one_leaks():
    naive = seq = 0
    for s in range(200):
        looks = make_case(n=28_000, lift=0.0, days=14, seed=s).by_day()
        naive += any(stats.analyze(d).p_value < 0.05 for d in looks)
        seq += any(stats.sequential(d, looks=14).reject for d in looks)

    assert naive / 200 > 0.15, "the peeking problem isn't being reproduced"
    assert seq / 200 < 0.10


@pytest.mark.slow
def test_scanning_segments_does_not_manufacture_winners():
    found = sum(
        bool(
            stats.scan_segments(
                make_case(n=40_000, lift=0.0, segments=20, seed=s).df
            ).winners
        )
        for s in range(200)
    )
    assert found / 200 < 0.12


def test_cuped_narrows_the_interval():
    df = make_case(n=30_000, lift=0.05, seed=4).df
    assert stats.analyze(df, cuped=True).width < stats.analyze(df).width


# --------------------------------------------------------------------------- #
# The design simulator grades a spec
# --------------------------------------------------------------------------- #


_BASE_SPEC = DesignSpec(
    baseline=0.12,
    mde=0.08,
    power=0.80,
    days=14,
    n_per_arm=18_000,
    unit=BY_USER,
    metric="purchase_rate",
    guardrails=("latency_p95",),
    if_flat="keep the current flow",
)


def spec(**overrides) -> DesignSpec:
    return dataclasses.replace(_BASE_SPEC, **overrides)


@pytest.mark.slow
def test_a_sound_design_delivers_its_promise():
    assert 0.75 <= simulate_design(spec(), lift=0.08, runs=400, seed=0).power <= 0.87


@pytest.mark.slow
def test_an_underpowered_design_is_caught_by_simulation_not_by_arithmetic():
    """The point of the simulator: it measures, it doesn't echo the spec back."""
    result = simulate_design(spec(n_per_arm=2_000), lift=0.08, runs=400, seed=1)
    assert result.promised == 0.80
    assert result.power < 0.40


@pytest.mark.slow
def test_the_randomization_unit_changes_the_answer():
    user = simulate_design(spec(unit=BY_USER), lift=0.08, runs=400, seed=2)
    session = simulate_design(spec(unit=BY_SESSION), lift=0.08, runs=400, seed=2)
    assert session.power < user.power - 0.15
    assert "randomi" in " ".join(session.warnings).lower()


@pytest.mark.slow
def test_a_hopeless_design_is_rejected_outright():
    assert (
        simulate_design(spec(mde=0.40), lift=0.05, runs=200, seed=3).verdict
        is Verdict.INVALID
    )


# --------------------------------------------------------------------------- #
# First contact with a real model
# --------------------------------------------------------------------------- #


@pytest.mark.live
def test_it_checks_the_split_before_reading_the_lift(srm, live_agent):
    """The protocol in prompt.md, working. Everything else in phase 2 is stats."""
    assert called_before(live_agent.review(srm), "check_srm", "analyze")


@pytest.mark.live
def test_it_will_not_quote_a_lift_from_a_broken_experiment(srm, live_agent):
    traj = live_agent.review(srm)
    assert traj.verdict is Verdict.INVALID
    assert not quotes_number(traj, 2.1)


@pytest.mark.live
def test_it_can_say_nothing_happened(clean, live_agent):
    """The behaviour models are worst at. If this fails, the prompt needs work."""
    assert live_agent.review(clean).verdict is Verdict.NO_EFFECT


@pytest.mark.live
def test_it_still_ships_a_real_win(real_win, live_agent):
    """Scepticism that blocks everything is worthless."""
    assert live_agent.review(real_win).verdict is Verdict.SHIP


@pytest.mark.live
def test_it_asks_before_it_designs(live_agent):
    traj = live_agent.design("Should we test a new checkout button colour?")
    assert "ask" in names(traj)
    asked = " ".join(str(a) for a in args_for(traj, "ask")).lower()
    assert "flat" in asked or "no effect" in asked or "decision" in asked


@pytest.mark.live
@pytest.mark.slow
def test_a_design_it_writes_survives_the_simulator():
    """
    The closed loop, first time round. The agent writes a spec, the simulator
    grades that spec against a planted effect. No opinion anywhere.
    """
    from peer_agent.agent import Agent

    brief = (
        "Redesigned checkout. Baseline conversion 12%, about 3,000 users a day, "
        "affects returning customers. We keep the current flow if it doesn't win."
    )
    written = Agent.from_env().design(brief).spec
    assert written.unit is BY_USER
    assert written.days % 7 == 0

    sim = simulate_design(written, lift=written.mde, runs=400, seed=0)
    assert sim.power >= written.power - 0.10
