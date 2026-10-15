"""Schema v2: tokens stay null on a replay, and the error counts add up."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from peer_agent.evals import load_suite, run_suite

REPLAY = "fixtures/recorded.jsonl"


def test_a_replay_validates_and_does_not_invent_a_cost(tmp_path):
    suite = [case for case in load_suite() if case.id == "srm-01"]
    run_suite(suite, replay=REPLAY, out=tmp_path)
    payload = json.loads((tmp_path / "results.json").read_text())
    schema = json.loads(Path("evals/results.schema.json").read_text())
    jsonschema.validate(payload, schema)

    assert payload["mode"] == "replay"
    assert payload["schema_version"] == 2
    for case in payload["cases"]:
        assert case["input_tokens"] is None
        assert case["output_tokens"] is None
        assert case["cost_usd"] is None
        assert case["latency_s"] >= 0
    for variant in payload["variants"]:
        assert variant["tokens"]["input"] is None
        assert variant["cost_usd"]["total"] is None
        label = variant["label"]
        rows = [row for row in payload["cases"] if row["variant"] == label]
        misses = sum(not row["correct"] for row in rows)
        assert sum(variant["errors"].values()) == misses
