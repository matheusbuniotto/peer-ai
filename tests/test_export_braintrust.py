"""Braintrust export maps rows and refuses to run without a key."""

from __future__ import annotations

import json
import sys
import types

from peer_agent.export_braintrust import export_results


def test_missing_key_exits_2(monkeypatch, capsys):
    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
    assert export_results("results.json", "peer-ai") == 2
    assert "BRAINTRUST_API_KEY" in capsys.readouterr().out


def test_the_payload_is_one_row_per_case(monkeypatch, tmp_path):
    monkeypatch.setenv("BRAINTRUST_API_KEY", "test-key")
    logged: list[dict] = []

    class _Experiment:
        def log(self, **kwargs: object) -> None:
            logged.append(kwargs)

    sdk = types.SimpleNamespace(init=lambda project: _Experiment())
    monkeypatch.setitem(sys.modules, "braintrust", sdk)

    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "srm-01",
                        "flaw": "srm",
                        "verdict": "INVALID",
                        "truth": "INVALID",
                        "correct": True,
                        "srm_checked_before_lift": True,
                        "error": None,
                        "cost_usd": None,
                        "latency_s": 0.1,
                        "variant": "fake (tools)",
                        "trial": 0,
                    }
                ]
            }
        )
    )
    assert export_results(path, "peer-ai") == 0
    row = logged[0]
    assert row["input"] == {"case_id": "srm-01", "flaw": "srm"}
    assert row["output"] == "INVALID"
    assert row["expected"] == "INVALID"
    assert row["scores"]["correct"] == 1.0
    assert row["metadata"]["cost"] is None
