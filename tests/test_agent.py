"""
What the agent does, not what it says.

This is the file that makes it a specialist rather than a chatbot holding a
stats library. Every assertion is about something observable: a call in the
trajectory, or the verdict at the end.

The offline tests drive FakeLLM and run in milliseconds. The live ones need a
real model, because judgement is the thing being tested.
"""

from __future__ import annotations

import pytest
from conftest import FakeLLM, args_for, called_before, names, quotes_number
from peer_agent.agent import OverBudget, Trajectory
from peer_agent.sim import OUTLIERS, make_case
from peer_agent.tools import default_tools
from peer_agent.types import Verdict

BLOCKING = {Verdict.INVALID, Verdict.EXTEND, Verdict.NO_EFFECT}


# --------------------------------------------------------------------------- #
# Order of operations
# --------------------------------------------------------------------------- #


@pytest.mark.live
def test_validity_comes_before_the_headline_number(srm, live_agent):
    """
    Validate, then analyse. Reading the lift first and checking the split
    afterwards is how broken tests get shipped, even when both checks run.
    """
    assert called_before(live_agent.review(srm), "check_srm", "analyze")


@pytest.mark.live
def test_a_broken_split_stops_the_analysis(srm, live_agent):
    traj = live_agent.review(srm)
    assert traj.verdict is Verdict.INVALID
    assert not quotes_number(traj, 2.1), "quoted a lift from an invalid experiment"


@pytest.mark.live
def test_guardrails_are_checked_before_shipping(real_win, live_agent):
    traj = live_agent.review(real_win)
    if traj.verdict is Verdict.SHIP:
        assert "check_guardrails" in names(traj)


# --------------------------------------------------------------------------- #
# Saying no, and saying yes
# --------------------------------------------------------------------------- #


@pytest.mark.live
def test_a_null_result_is_reported_as_null(clean, live_agent):
    """The hardest behaviour: saying nothing happened, without trawling."""
    assert live_agent.review(clean).verdict in BLOCKING


@pytest.mark.live
def test_no_segment_hunting_after_a_null_primary(clean, live_agent):
    traj = live_agent.review(clean)
    for call in args_for(traj, "scan_segments"):
        assert call.get("primary_null") is True


@pytest.mark.live
def test_a_decaying_lift_is_not_a_win(novelty, live_agent):
    assert live_agent.review(novelty).verdict is Verdict.EXTEND


@pytest.mark.live
def test_a_real_win_actually_ships(real_win, live_agent):
    """Scepticism is worthless if it blocks everything."""
    assert live_agent.review(real_win).verdict is Verdict.SHIP


@pytest.mark.live
def test_it_asks_what_happens_if_the_result_is_flat(live_agent):
    """The question teams skip, and the one that kills pointless tests."""
    traj = live_agent.design("Should we test a new checkout button colour?")
    asked = " ".join(str(a) for a in args_for(traj, "ask")).lower()
    assert "flat" in asked or "no effect" in asked or "decision" in asked


# --------------------------------------------------------------------------- #
# Tool choice
# --------------------------------------------------------------------------- #


@pytest.mark.live
def test_it_reaches_for_the_built_tool_before_the_sandbox(srm, live_agent):
    """Hand-rolling a chi-square skips the calibrated threshold in stats.py."""
    assert "check_srm" in names(live_agent.review(srm))


@pytest.mark.live
def test_it_reaches_for_the_sandbox_when_nothing_fits(live_agent):
    case = make_case(n=40_000, flaws=[OUTLIERS], seed=77)
    traj = live_agent.review(case, question="Is the metric distribution skewed?")
    assert "run_python" in names(traj)


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #


def test_tool_results_come_back_to_the_model(agent, srm):
    llm = FakeLLM(turns=[[("check_srm", {})], "Invalid experiment."])
    agent(llm).review(srm)
    assert any("check_srm" in str(m) for m in llm.seen[-1])


def test_a_failing_tool_becomes_a_message_not_a_crash(agent, clean):
    """One bad call shouldn't end the run. Hand the error back and let it retry."""
    llm = FakeLLM(
        turns=[[("analyze", {"alpha": "banana"})], [("analyze", {})], "No effect."]
    )
    traj = agent(llm).review(clean)
    assert traj.done
    assert any(c.error for c in traj.calls)


def test_an_unknown_tool_is_an_error_message(agent, clean):
    llm = FakeLLM(turns=[[("summon_p_value", {})], "done"])
    traj = agent(llm).review(clean)
    assert "unknown tool" in traj.calls[0].error.lower()


def test_the_turn_budget_is_enforced(agent, clean):
    llm = FakeLLM(react=lambda _: [("check_srm", {})])
    with pytest.raises(OverBudget):
        agent(llm, max_turns=5).review(clean)
    assert len(llm.seen) <= 6


def test_the_cost_budget_is_enforced(agent, clean):
    llm = FakeLLM(react=lambda _: [("check_srm", {})])
    with pytest.raises(OverBudget, match="cost"):
        agent(llm, max_cost=0.01).review(clean)


def test_the_trajectory_records_the_run(agent, clean):
    llm = FakeLLM(turns=[[("check_srm", {})], [("analyze", {})], "No effect."])
    traj = agent(llm).review(clean)
    assert len(traj.turns) == 3
    assert traj.tokens > 0
    assert traj.seconds >= 0


def test_a_trajectory_survives_a_round_trip(agent, clean, tmp_path):
    """The report page replays these, so the format is part of the contract."""
    traj = agent(FakeLLM(turns=["No effect."])).review(clean)
    path = tmp_path / "run.json"
    path.write_text(traj.to_json())
    assert Trajectory.from_json(path.read_text()) == traj


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


def test_every_tool_has_a_usable_schema(sandbox):
    import jsonschema

    for tool in default_tools(sandbox):
        jsonschema.Draft202012Validator.check_schema(tool.schema)
        assert tool.description.strip()


def test_the_tool_names_are_the_ones_the_prompt_refers_to(sandbox):
    """Renaming one invalidates every stored trajectory. Do it on purpose."""
    assert {t.name for t in default_tools(sandbox)} == {
        "check_srm",
        "analyze",
        "sequential",
        "scan_segments",
        "check_novelty",
        "check_guardrails",
        "power_analysis",
        "simulate_design",
        "run_python",
        "ask",
    }


def test_bad_arguments_are_rejected_before_the_stats_run(sandbox, clean):
    tools = default_tools(sandbox)
    result = tools.call("power_analysis", {"baseline": 2.0}, data=clean.df)
    assert result.error and "baseline" in result.error
