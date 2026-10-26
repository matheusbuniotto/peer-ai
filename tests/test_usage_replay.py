"""Recorded usage round-trips through replay; missing usage stays unlabeled cost."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from peer_agent.evals import load_suite, run_suite

REPLAY = "fixtures/recorded.jsonl"


def test_fixture_replay_marks_usage_missing(tmp_path):
    suite = [case for case in load_suite() if case.id == "srm-01"]
    run_suite(suite, replay=REPLAY, out=tmp_path)
    payload = json.loads((tmp_path / "results.json").read_text())
    case = payload["cases"][0]
    assert case["usage_source"] == "missing"
    assert case["input_tokens"] is None
    assert case["cost_usd"] is None
    variant = payload["variants"][0]
    assert variant["cost_usd"]["usage_source"] == "missing"
    assert variant["tokens"]["usage_source"] == "missing"
    assert variant["cost_usd"]["total"] is None


def test_recorded_usage_round_trips_through_replay(tmp_path):
    fixture_line = next(
        line
        for line in Path(REPLAY).read_text().splitlines()
        if line.startswith('{"case_id": "srm-01"')
    )
    row = json.loads(fixture_line)
    row["usage"] = [
        {
            "input_tokens": 1_000,
            "output_tokens": 100,
            "cache_read_tokens": 50,
            "cache_write_tokens": 0,
        },
        {
            "input_tokens": 200,
            "output_tokens": 50,
            "cache_read_tokens": 0,
            "cache_write_tokens": 10,
        },
    ]
    path = tmp_path / "with_usage.jsonl"
    no_tools = {**row, "model": "openai/gpt-6-luna (no tools)"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(no_tools) + "\n")

    suite = [case for case in load_suite() if case.id == "srm-01"]
    run_suite(suite, replay=path, out=tmp_path / "out")
    payload = json.loads((tmp_path / "out" / "results.json").read_text())
    schema = json.loads(Path("evals/results.schema.json").read_text())
    jsonschema.validate(payload, schema)

    case = payload["cases"][0]
    assert case["usage_source"] == "recorded"
    assert case["input_tokens"] == 1200
    assert case["output_tokens"] == 150
    assert case["cost_usd"] is not None
    variant = payload["variants"][0]
    assert variant["cost_usd"]["usage_source"] == "recorded"
    assert variant["tokens"]["usage_source"] == "recorded"
