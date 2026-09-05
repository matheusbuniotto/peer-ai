"""
End to end. The tests that decide whether this is finished.

Everything else checks a piece. These run the real thing: a real model, a real
sandbox, fifty simulated experiments with known answers, and the results.json
the report page is built from.

    pytest -m "not live and not docker"   # every commit, offline, seconds
    pytest -m live                        # the grade, costs money, gates release
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
from peer_agent.design import simulate_design
from peer_agent.evals import load_suite, run_suite
from peer_agent.types import BY_USER, Verdict

REPLAY = "fixtures/recorded.jsonl"


# --------------------------------------------------------------------------- #
# The grade
# --------------------------------------------------------------------------- #


@pytest.mark.live
@pytest.mark.slow
def test_the_suite_hits_the_accuracy_i_publish(tmp_path):
    result = run_suite(load_suite(), model="claude-opus-5", out=tmp_path)

    assert result.n == 50
    assert result.accuracy >= 0.90, f"page says 94%, measured {result.accuracy:.0%}"

    # Accuracy alone hides a bias in either direction, so bound both.
    assert result.false_ships <= 0.04, "shipped something invalid"
    assert result.false_blocks <= 0.15, "blocked real wins; scepticism miscalibrated"


@pytest.mark.live
@pytest.mark.slow
@pytest.mark.parametrize(
    "flaw,floor",
    [
        ("srm", 0.9),
        ("peeking", 0.9),
        ("clean", 0.9),
        ("novelty", 0.8),
        ("simpson", 0.7),
    ],
)
def test_no_category_is_carried_by_the_others(flaw, floor, tmp_path):
    """90% overall is compatible with total failure on one column of the page."""
    result = run_suite(load_suite(flaw=flaw), model="claude-opus-5", out=tmp_path)
    assert result.accuracy >= floor, f"{flaw}: {result.accuracy:.0%} under {floor:.0%}"


@pytest.mark.live
@pytest.mark.slow
def test_the_tools_are_worth_building():
    """
    The premise. If the model scores nearly as well with no tools, this is a
    prompt in a trenchcoat and I should rethink the project. Written to fail
    loudly rather than let me find out later.
    """
    suite = load_suite()
    with_tools = run_suite(suite, model="claude-opus-5")
    bare = run_suite(suite, model="claude-opus-5", tools=[])
    assert with_tools.accuracy - bare.accuracy > 0.35


@pytest.mark.live
@pytest.mark.slow
def test_a_smaller_model_still_clears_a_useful_bar():
    """If it only works on the biggest model it's a demo, not a tool."""
    assert run_suite(load_suite(), model="claude-sonnet-5").accuracy >= 0.80


# --------------------------------------------------------------------------- #
# Design, closed loop
# --------------------------------------------------------------------------- #


@pytest.mark.live
def test_a_design_it_writes_survives_my_own_simulator():
    """
    The loop that makes design gradeable. It writes a spec, the simulator runs
    that spec against a planted effect, and the realized power has to match
    what the spec promised.
    """
    brief = (
        "We want to test a redesigned checkout. Baseline conversion is 12%. "
        "We ship if it wins and keep the current flow if it doesn't. "
        "Around 3,000 users a day. The change affects returning customers."
    )
    from peer_agent.agent import Agent

    spec = Agent.from_env().design(brief).spec

    assert spec.unit is BY_USER, "session randomization on a returning-user feature"
    assert spec.days % 7 == 0
    assert spec.guardrails

    sim = simulate_design(spec, lift=spec.mde, runs=500, seed=0)
    assert sim.power >= spec.power - 0.07, (
        f"promised {spec.power:.0%}, gives {sim.power:.0%}"
    )


@pytest.mark.live
def test_it_refuses_a_test_that_cannot_answer_its_question():
    from peer_agent.agent import Agent

    brief = (
        "Test a new homepage headline. Baseline conversion 0.4%. About 200 "
        "visitors a day and we need an answer by Friday."
    )
    traj = Agent.from_env().design(brief)
    assert traj.verdict is Verdict.INVALID
    assert "power" in traj.answer.lower() or "traffic" in traj.answer.lower()


# --------------------------------------------------------------------------- #
# Reproducible, and guarded against regression
# --------------------------------------------------------------------------- #


def test_a_replayed_run_is_byte_identical(tmp_path):
    a = run_suite(load_suite(), replay=REPLAY, out=tmp_path / "a")
    b = run_suite(load_suite(), replay=REPLAY, out=tmp_path / "b")
    assert a.to_json() == b.to_json()


def test_accuracy_has_not_slipped_since_the_last_commit(tmp_path):
    """
    The merge gate. A prompt tweak that drops SRM detection from 94% to 71%
    fails the build instead of quietly shipping.
    """
    baseline = json.loads(Path("evals/baseline.json").read_text())
    current = run_suite(load_suite(), replay=REPLAY, out=tmp_path)

    assert current.accuracy >= baseline["accuracy"] - 0.02
    for flaw, score in baseline["by_flaw"].items():
        assert current.by_flaw[flaw] >= score - 0.10, f"regression in {flaw}"


def test_every_case_ships_with_its_answer():
    """A case with no known truth can't be graded, so it shouldn't exist."""
    for case in load_suite():
        assert case.truth in Verdict
        assert case.seed is not None


def test_the_model_does_not_mark_its_own_homework():
    suite = load_suite()
    for case in suite:
        assert case.grading == "rubric" or case.judge != suite.model


# --------------------------------------------------------------------------- #
# The file the page reads
# --------------------------------------------------------------------------- #


def test_results_json_matches_what_the_page_expects(tmp_path):
    run_suite(load_suite(), replay=REPLAY, out=tmp_path)
    report = json.loads((tmp_path / "results.json").read_text())

    assert {"accuracy", "cases", "by_flaw", "models", "generated_at"} <= report.keys()
    assert len(report["cases"]) == 50
    for case in report["cases"]:
        assert {
            "id",
            "flaw",
            "correct",
            "truth",
            "verdict",
            "trajectory",
        } <= case.keys()
    assert any("no tools" in m["name"] for m in report["models"]), (
        "the page publishes a no-tools baseline, so the report has to contain one"
    )


def test_the_failures_i_publish_are_still_failing(tmp_path):
    """The page has a 'where it fails' section. Don't perform humility."""
    run_suite(load_suite(), replay=REPLAY, out=tmp_path)
    report = json.loads((tmp_path / "results.json").read_text())
    published = {f["case_id"] for f in report["published_failures"]}
    failing = {c["id"] for c in report["cases"] if not c["correct"]}
    assert published <= failing, f"claimed as broken but now passing: {published - failing}"


# --------------------------------------------------------------------------- #
# Layering and cold start
# --------------------------------------------------------------------------- #


def test_stats_knows_nothing_about_the_agent():
    """
    stats.py and design.py are functions over dataframes. They must not import
    the loop, the tool registry, or an SDK. This is the layering, checked
    rather than described in the README.
    """
    banned = (
        "peer_agent.agent",
        "peer_agent.tools",
        "peer_agent.cli",
        "anthropic",
        "click",
    )
    for path in [Path("src/peer_agent/stats.py"), Path("src/peer_agent/design.py")]:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(banned), f"{path}: {node.module}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(banned), f"{path}: {alias.name}"


@pytest.mark.slow
def test_the_first_command_in_the_readme_works():
    """Nobody will debug my install. The quickstart has to run."""
    result = subprocess.run(
        [sys.executable, "-m", "peer_agent", "review", "fixtures/sample.parquet"],
        capture_output=True,
        text=True,
        timeout=300,
        env={"ABCODE_REPLAY": REPLAY},
    )
    assert result.returncode == 0, result.stderr
    assert "verdict" in result.stdout.lower()


@pytest.mark.docker
def test_the_mcp_server_answers_over_stdio():
    """How an engineer looking at my work will actually try it."""
    from peer_agent.mcp import stdio_roundtrip

    assert any(t["name"] == "check_srm" for t in stdio_roundtrip("tools/list"))
