"""Eval variant wiring: python-only control vs full tools."""

from __future__ import annotations

from peer_agent.evals import _tools_for_kind, run_suite
from peer_agent.sandbox import Sandbox
from peer_agent.tools import python_only_tools


def test_python_only_registry_is_just_run_python(sandbox):
    names = {tool.name for tool in python_only_tools(sandbox)}
    assert names == {"run_python"}


def test_tools_for_kind_python_only_uses_the_sandbox():
    tools = _tools_for_kind("python-only", None, Sandbox())
    assert len(tools) == 1
    assert tools[0].name == "run_python"


def test_default_live_variants_include_python_only(tmp_path):
    from peer_agent.evals import load_suite

    suite = load_suite()[:1]

    def llm_for(_ec, _trial):
        return lambda _messages, _tools: "VERDICT: NO_EFFECT — no signal."

    report = run_suite(suite, model="fake", llm_for=llm_for, out=tmp_path)
    labels = {variant["label"] for variant in report.variants}
    assert any(label.endswith(" (tools)") for label in labels)
    assert any(label.endswith(" (python-only)") for label in labels)
