"""Repeated trials change pass^k even when the data seed does not."""

from __future__ import annotations

from conftest import FakeLLM

from peer_agent.evals import EvalCase, run_suite
from peer_agent.types import Verdict


def test_pass_hat_k_needs_every_trial(tmp_path):
    suite = [
        EvalCase(id="a", flaw="clean", seed=1, n=200, lift=0.0, truth=Verdict.NO_EFFECT),
        EvalCase(id="b", flaw="clean", seed=2, n=200, lift=0.0, truth=Verdict.NO_EFFECT),
    ]

    def llm_for(case: EvalCase, trial: int) -> FakeLLM:
        if case.id == "b" and trial:
            return FakeLLM(["VERDICT: SHIP. Looks like a win."])
        return FakeLLM(["VERDICT: NO_EFFECT. Flat."])

    report = run_suite(
        suite, model="fake", llm_for=llm_for, variants=("no-tools",), trials=3, out=tmp_path
    )
    variant = report.variants[0]
    assert len(report.cases) == 6
    assert variant["pass_at_k"] == 1.0
    assert variant["pass_hat_k"] == 0.5
    assert variant["accuracy_by_trial"][0] == 1.0
    assert variant["accuracy_by_trial"][1] == 0.5
