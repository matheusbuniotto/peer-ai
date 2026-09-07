"""
Phase 3 — make it a project.

Phases 1 and 2 built something correct that I can run. This phase makes it
something other people can run, and makes its claims checkable: the sandbox so
the agent can write its own code, the eval harness so fifty cases produce one
number, MCP so an engineer can try it in thirty seconds, and the results.json
the report page is built from.

Definition of done — the whole thing, and the page is wired to real output:

    pytest -m "not live and not docker"     # every commit
    pytest -m docker                        # needs a container runtime
    pytest -m live                          # the grade, gates release

What I have to build for this:

    sandbox.py    container, read-only data, no network, timeout, memory cap
    tools.py      run_python and the rest of the registry
    evals.py      load_suite, run_suite, results.json, recorded replay
    mcp.py        the same tools over stdio, plus the protocol as a resource
    evals/        suite.yaml with 50 cases, baseline.json as the merge gate
    web/          the page, reading results.json

test_sandbox.py, test_agent.py and test_e2e.py are the permanent versions and
go deeper. This file is the checkpoint that says the phase is finished.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from conftest import names

from peer_agent.evals import load_suite, run_suite
from peer_agent.tools import default_tools

REPLAY = "fixtures/recorded.jsonl"


# --------------------------------------------------------------------------- #
# The sandbox: the agent can write code, and it can't escape
# --------------------------------------------------------------------------- #


@pytest.mark.docker
def test_it_runs_analysis_code_against_the_experiment(sandbox, clean):
    result = sandbox.run(
        "import pandas as pd; print(pd.read_parquet('/data/exp.parquet').arm.mean())",
        clean.df,
    )
    assert result.ok
    assert 0.4 < float(result.stdout) < 0.6


@pytest.mark.docker
def test_it_cannot_reach_the_network_or_write_to_the_data(sandbox, clean):
    net = sandbox.run(
        "import urllib.request; urllib.request.urlopen('https://example.com', timeout=5)",
        clean.df,
    )
    write = sandbox.run("open('/data/exp.parquet', 'w').write('x')", clean.df)
    assert not net.ok and not write.ok


@pytest.mark.docker
def test_it_survives_code_that_would_hang_or_eat_the_machine(sandbox, clean):
    assert not sandbox.run("while True: pass", clean.df).ok
    assert not sandbox.run("x = bytearray(4 * 1024**3)", clean.df).ok


@pytest.mark.docker
@pytest.mark.live
def test_the_agent_writes_its_own_code_when_no_tool_fits(live_agent):
    from peer_agent.sim import OUTLIERS, make_case

    case = make_case(n=40_000, flaws=[OUTLIERS], seed=9)
    traj = live_agent.review(case, question="Is the metric distribution skewed?")
    assert "run_python" in names(traj)


@pytest.mark.docker
@pytest.mark.live
def test_it_still_prefers_the_built_tool_when_one_exists(srm, live_agent):
    """Hand-rolling a chi-square skips the calibrated threshold in stats.py."""
    assert "check_srm" in names(live_agent.review(srm))


def test_the_registry_is_complete(sandbox):
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


# --------------------------------------------------------------------------- #
# The harness: fifty cases, one number
# --------------------------------------------------------------------------- #


def test_every_case_in_the_suite_has_a_known_answer():
    from peer_agent.types import Verdict

    suite = load_suite()
    assert len(suite) == 50
    for case in suite:
        assert case.truth in Verdict
        assert case.seed is not None


def test_the_suite_covers_all_five_flaws():
    """A grade averaged over four categories isn't the grade I publish."""
    flaws = {case.flaw for case in load_suite()}
    assert flaws == {"srm", "novelty", "simpson", "peeking", "clean"}


def test_a_replayed_run_is_byte_identical(tmp_path):
    """Recorded responses in, same report out. Without this, CI is theatre."""
    a = run_suite(load_suite(), replay=REPLAY, out=tmp_path / "a")
    b = run_suite(load_suite(), replay=REPLAY, out=tmp_path / "b")
    assert a.to_json() == b.to_json()


def test_the_baseline_gate_catches_a_regression(tmp_path):
    with open("evals/baseline.json") as f:
        baseline = json.loads(f.read())
    current = run_suite(load_suite(), replay=REPLAY, out=tmp_path)

    assert current.accuracy >= baseline["accuracy"] - 0.02
    for flaw, score in baseline["by_flaw"].items():
        assert current.by_flaw[flaw] >= score - 0.10, f"regression in {flaw}"


@pytest.mark.live
@pytest.mark.slow
def test_the_real_grade(tmp_path):
    """A real model, fifty cases, the number that goes on the page."""
    result = run_suite(load_suite(), model="claude-opus-5", out=tmp_path)
    assert result.accuracy >= 0.90
    assert result.false_ships <= 0.04
    assert result.false_blocks <= 0.15


@pytest.mark.live
@pytest.mark.slow
def test_the_toolbox_earns_its_existence():
    """
    The premise of the project. If a bare model scores nearly as well, this is
    a prompt in a trenchcoat and I need to know before I write the README.
    """
    suite = load_suite()
    assert (
        run_suite(suite, model="claude-opus-5").accuracy
        - run_suite(suite, model="claude-opus-5", tools=[]).accuracy
        > 0.35
    )


# --------------------------------------------------------------------------- #
# MCP: how someone else tries it
# --------------------------------------------------------------------------- #


@pytest.mark.docker
def test_the_server_lists_the_tools_over_stdio():
    from peer_agent.mcp import stdio_roundtrip

    tools = stdio_roundtrip("tools/list")
    assert {"check_srm", "analyze", "simulate_design"} <= {t["name"] for t in tools}


@pytest.mark.docker
def test_the_server_gives_the_same_answer_as_the_library(srm):
    from peer_agent import stats
    from peer_agent.mcp import stdio_roundtrip

    reply = stdio_roundtrip("tools/call", name="check_srm", arguments={"path": srm.path})
    assert reply["mismatch"] == stats.check_srm(srm.df).mismatch


@pytest.mark.docker
def test_the_protocol_is_published_as_a_resource():
    """So the checklist travels with the tools instead of living in my prompt."""
    from peer_agent.mcp import stdio_roundtrip

    text = stdio_roundtrip("resources/read", uri="peer://protocol")
    assert "validate" in text.lower()


def test_a_tool_takes_a_path_or_inline_rows_but_needs_exactly_one(srm):
    """A caller with no shared filesystem — another MCP server's JSON reply in
    hand, nowhere local to stage it — passes `data` instead of `path`."""
    from peer_agent.mcp import _load_df

    by_path = _load_df(path=srm.path)
    by_data = _load_df(data=srm.df.to_json(orient="records"))
    pd.testing.assert_frame_equal(by_path.reset_index(drop=True), by_data)

    for kwargs in ({}, {"path": srm.path, "data": "[]"}):
        with pytest.raises(ValueError, match="exactly one"):
            _load_df(**kwargs)


@pytest.mark.docker
def test_the_server_accepts_rows_handed_over_from_another_mcp_server(srm):
    """The Databricks-in-Claude case: a query tool already returned JSON rows,
    and check_srm has to work from those directly, not a file the client wrote."""
    from peer_agent import stats
    from peer_agent.mcp import stdio_roundtrip

    reply = stdio_roundtrip(
        "tools/call",
        name="check_srm",
        arguments={"data": srm.df.to_json(orient="records")},
    )
    assert reply["mismatch"] == stats.check_srm(srm.df).mismatch


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #


def test_results_json_is_the_shape_the_page_reads(tmp_path):
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


def test_the_page_can_replay_a_trajectory(tmp_path):
    """The step-through is the best thing on the page. It needs real turns."""
    run_suite(load_suite(), replay=REPLAY, out=tmp_path)
    report = json.loads((tmp_path / "results.json").read_text())
    turns = report["cases"][0]["trajectory"]
    assert len(turns) >= 2
    assert all({"turn", "act", "body"} <= t.keys() for t in turns)


def test_the_no_tools_baseline_is_in_the_report(tmp_path):
    run_suite(load_suite(), replay=REPLAY, out=tmp_path)
    report = json.loads((tmp_path / "results.json").read_text())
    assert any("no tools" in m["name"] for m in report["models"])


def test_the_failures_i_publish_are_still_failing(tmp_path):
    """Don't perform humility about bugs I've since fixed."""
    run_suite(load_suite(), replay=REPLAY, out=tmp_path)
    report = json.loads((tmp_path / "results.json").read_text())
    published = {f["case_id"] for f in report["published_failures"]}
    failing = {c["id"] for c in report["cases"] if not c["correct"]}
    assert published <= failing, f"claimed as broken but now passing: {published - failing}"


def test_the_page_reads_nothing_but_results_json():
    """
    The page is static and has no backend. If it imports anything from src, it
    will break the moment it's deployed.
    """
    import pathlib
    import re

    for path in pathlib.Path("web").rglob("*.jsx"):
        assert not re.search(r"from ['\"].*(src|peer_agent)", path.read_text()), path
