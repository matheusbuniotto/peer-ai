"""One eval case emits the GenAI usage attributes on its span."""

from __future__ import annotations

from conftest import FakeLLM

from peer_agent.evals import EvalCase, run_suite
from peer_agent.types import Verdict


def test_the_case_span_carries_usage_and_the_score(capfire):
    suite = [
        EvalCase(
            id="span-01", flaw="clean", seed=1, n=200, lift=0.0, truth=Verdict.NO_EFFECT
        )
    ]

    def llm_for(case: EvalCase, trial: int) -> FakeLLM:
        del case, trial
        return FakeLLM(["VERDICT: NO_EFFECT. Flat."])

    run_suite(suite, model="fake", llm_for=llm_for, variants=("no-tools",))
    spans = [
        span
        for span in capfire.exporter.exported_spans
        if span.name == "eval.case"
    ]
    assert spans
    attributes = dict(spans[-1].attributes or {})
    assert attributes["peer.correct"] is True
    assert attributes["peer.error_category"] == ""
    assert "gen_ai.usage.input_tokens" in attributes
    assert "gen_ai.usage.output_tokens" in attributes
    assert "peer.cost_usd" in attributes
