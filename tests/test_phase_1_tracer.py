"""
Phase 1 — the tracer bullet.

One thin thread through every layer, end to end, with everything faked that
can be faked. No real model, no container, no calibration, one flaw, one
statistic. The point is that data flows from the generator through the stats
through the loop and out the other side as a verdict I can print.

Definition of done: this file is green offline in under a second.

    pytest tests/test_phase_1_tracer.py

Deliberately not here yet: whether the statistics are correct (phase 2),
whether the agent has judgement (phase 2), the sandbox, MCP, the eval harness
(all phase 3). A tracer bullet is allowed to be wrong. It is not allowed to be
disconnected.

What I have to build for this:

    types.py      Verdict, and one frozen result dataclass
    sim.py        make_case with a lift and the SRM flaw. Nothing else.
    stats.py      check_srm and analyze. Textbook formulas, no CUPED.
    tools.py      two tools, hand-written schemas
    agent.py      the loop: call model, dispatch tool, feed result back, stop
    cli.py        peer review <file>
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from conftest import FakeLLM, names
from peer_agent.sim import SRM, make_case
from peer_agent.tools import default_tools
from peer_agent.types import Verdict

from peer_agent import stats

# --------------------------------------------------------------------------- #
# The generator produces something with a known answer
# --------------------------------------------------------------------------- #


def test_a_case_carries_its_own_truth():
    """Without this, nothing downstream can ever be graded."""
    case = make_case(n=10_000, lift=0.0, seed=1)
    assert case.truth is Verdict.NO_EFFECT
    assert len(case.df) == 10_000
    assert set(case.df.columns) >= {"arm", "converted"}


def test_the_same_seed_gives_the_same_data():
    a = make_case(n=5_000, lift=0.03, seed=7)
    b = make_case(n=5_000, lift=0.03, seed=7)
    assert a.df.equals(b.df)


def test_a_planted_lift_shows_up_in_the_raw_rates():
    """Crude, but it catches a generator that ignores its own argument."""
    df = make_case(n=200_000, lift=0.20, seed=2).df
    rates = df.groupby("arm").converted.mean()
    assert rates[True] > rates[False]


def test_the_srm_flaw_actually_breaks_the_split():
    df = make_case(n=50_000, lift=0.0, flaws=[SRM], seed=3).df
    share = df.arm.mean()
    assert not 0.495 < share < 0.505


# --------------------------------------------------------------------------- #
# Two statistics, roughly right
# --------------------------------------------------------------------------- #


def test_analyze_returns_the_numbers_i_need():
    result = stats.analyze(make_case(n=40_000, lift=0.05, seed=4).df)
    assert 0 <= result.p_value <= 1
    assert result.lift == pytest.approx(0.05, abs=0.03)
    assert result.ci_low < result.lift < result.ci_high


def test_analyze_finds_a_large_effect_and_ignores_a_missing_one():
    assert stats.analyze(make_case(n=80_000, lift=0.15, seed=5).df).p_value < 0.01
    assert stats.analyze(make_case(n=80_000, lift=0.0, seed=6).df).p_value > 0.01


def test_check_srm_fires_on_a_broken_split_and_not_on_a_clean_one():
    assert stats.check_srm(make_case(n=50_000, flaws=[SRM], seed=7).df).mismatch
    assert not stats.check_srm(make_case(n=50_000, seed=8).df).mismatch


# --------------------------------------------------------------------------- #
# The loop turns tool calls into a trajectory
# --------------------------------------------------------------------------- #


def test_the_loop_calls_a_tool_and_finishes(sandbox_free_agent, clean):
    llm = FakeLLM(turns=[[("check_srm", {})], "No mismatch. Nothing to report."])
    traj = sandbox_free_agent(llm).review(clean)

    assert traj.done
    assert names(traj) == ["check_srm"]
    assert "mismatch" in traj.answer.lower()


def test_the_tool_result_reaches_the_model(sandbox_free_agent, srm):
    """If this fails the agent is guessing, not reading."""
    llm = FakeLLM(turns=[[("check_srm", {})], "Invalid."])
    sandbox_free_agent(llm).review(srm)
    assert "mismatch" in str(llm.seen[-1]).lower()


def test_two_tools_in_sequence(sandbox_free_agent, clean):
    llm = FakeLLM(turns=[[("check_srm", {})], [("analyze", {})], "No effect."])
    traj = sandbox_free_agent(llm).review(clean)
    assert names(traj) == ["check_srm", "analyze"]


def test_the_loop_stops_instead_of_spinning(sandbox_free_agent, clean):
    """A runaway loop is the first thing that goes wrong. Cap it on day one."""
    from peer_agent.agent import OverBudget

    llm = FakeLLM(react=lambda _: [("check_srm", {})])
    with pytest.raises(OverBudget):
        sandbox_free_agent(llm, max_turns=5).review(clean)


def test_the_verdict_comes_out_as_an_enum(sandbox_free_agent, srm):
    """Free text is unparseable and ungradeable. Parse it now, not later."""
    llm = FakeLLM(turns=[[("check_srm", {})], "VERDICT: INVALID. The split is broken."])
    assert sandbox_free_agent(llm).review(srm).verdict is Verdict.INVALID


# --------------------------------------------------------------------------- #
# Both tools are visible to the model
# --------------------------------------------------------------------------- #


def test_the_registry_offers_exactly_what_exists_so_far():
    assert {t.name for t in default_tools()} == {"check_srm", "analyze"}


def test_each_tool_has_a_name_a_description_and_a_schema():
    for tool in default_tools():
        assert tool.name and tool.description.strip()
        assert tool.schema["type"] == "object"


# --------------------------------------------------------------------------- #
# It runs from a terminal
# --------------------------------------------------------------------------- #


def test_the_cli_prints_a_verdict(tmp_path, srm):
    """The thread has to reach a human, or it isn't end to end."""
    path = tmp_path / "exp.parquet"
    srm.df.to_parquet(path)

    result = subprocess.run(
        [sys.executable, "-m", "peer_agent", "review", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PEER_FAKE_LLM": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert "invalid" in result.stdout.lower()
