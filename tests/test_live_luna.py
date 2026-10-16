"""The committed score is a live Luna run, and the demo shows that one score."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from peer_agent.agent import Trajectory, _log_trajectory
from peer_agent.cli import main
from peer_agent.evals import load_suite, promote_baseline, run_suite

_PRECHANGE = "84342b0"
_TOOLS = "openai/gpt-6-luna (tools)"
_NO_TOOLS = "openai/gpt-6-luna (no tools)"


def _latest_payload() -> dict:
    run_id = Path("results/LATEST").read_text().strip()
    return json.loads((Path("results") / run_id / "results.json").read_text())


def test_a_blank_log_path_writes_the_default_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PEER_LOG_PATH", "")
    _log_trajectory("review", Trajectory(answer="ok", done=True))
    assert (tmp_path / "logs" / "runs.jsonl").is_file()


def test_latest_is_a_three_trial_luna_run():
    payload = _latest_payload()
    primary = next(variant for variant in payload["variants"] if variant["tools"])
    assert payload["mode"] == "live"
    assert payload["trials"] == 3
    assert primary["model"] == "openai/gpt-6-luna"


def test_recorded_turns_are_the_luna_run():
    previous = subprocess.check_output(
        ["git", "show", f"{_PRECHANGE}:fixtures/recorded.jsonl"], text=True
    )

    def _role(label: str) -> str:
        if label.endswith(" (no tools)"):
            return "no-tools"
        return "tools"

    old = {}
    for line in previous.splitlines():
        row = json.loads(line)
        old[(row["case_id"], _role(row["model"]))] = row["turns"]
    recorded = {}
    for line in Path("fixtures/recorded.jsonl").read_text().splitlines():
        row = json.loads(line)
        recorded.setdefault((row["case_id"], row["model"]), set()).add(int(row["trial"]))
        if int(row["trial"]) == 0:
            assert row["turns"] != old[(row["case_id"], _role(row["model"]))]
    case_ids = {case.id for case in load_suite()}
    assert len(case_ids) == 55
    for case_id in case_ids:
        assert recorded[(case_id, _TOOLS)] == {0, 1, 2}
        assert recorded[(case_id, _NO_TOOLS)] == {0, 1, 2}


def test_readme_table_quotes_the_live_run():
    payload = _latest_payload()
    primary = next(variant for variant in payload["variants"] if variant["tools"])
    text = Path("README.md").read_text()
    block = text.split("<!-- results:start -->", 1)[1].split("<!-- results:end -->", 1)[0]
    assert payload["run_id"] in block
    assert f"{primary['accuracy']:.1%}" in block


def test_readme_calls_the_table_a_live_luna_run():
    text = Path("README.md").read_text()
    results = text.split("## Results", 1)[1].split("## ", 1)[0]
    limits = text.split("## Limitations", 1)[1].split("## ", 1)[0]
    for section in (results, limits):
        assert "live" in section
        assert "openai/gpt-6-luna" in section
        assert "trials 3" in section


def test_a_live_record_with_no_tokens_keeps_the_fixture(tmp_path, monkeypatch):
    dest = tmp_path / "recorded.jsonl"
    original = b'{"kept": true}\n'
    dest.write_bytes(original)

    def _rejected(_model: str) -> object:
        def _call(_messages: object, _tools: object) -> object:
            raise RuntimeError("provider rejected the key")

        return _call

    monkeypatch.setattr("peer_agent.evals._live_model", _rejected)
    code = main(
        [
            "eval",
            "--model",
            "openai/gpt-6-luna",
            "--trials",
            "1",
            "--variants",
            "tools",
            "--record",
            str(dest),
        ]
    )
    assert code == 1
    assert dest.read_bytes() == original
    assert not Path(f"{dest}.partial").exists()


def test_a_live_promote_keeps_the_one_trial_replay_score(tmp_path):
    published = Path("evals/published_failures.json").read_bytes()
    live = {
        "mode": "live",
        "run_id": "live-run",
        "trials": 3,
        "published_failures": [],
        "variants": [
            {
                "label": _TOOLS,
                "model": "openai/gpt-6-luna",
                "tools": True,
                "accuracy": 0.5,
                "accuracy_ci": [0.4, 0.6],
                "pass_hat_k": 0.5,
                "false_ships": 0.1,
                "by_flaw": {"peeking": {"acc": 0.2}},
            }
        ],
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(live))
    baseline = promote_baseline(path, tmp_path / "baseline.json")
    harness = run_suite(load_suite(), replay="fixtures/recorded.jsonl", trials=1)
    assert baseline["accuracy"] == 0.5
    assert baseline["false_ships"] == 0.1
    assert baseline["by_flaw"] == {"peeking": 0.2}
    assert baseline["replay_expected"]["accuracy"] == harness.accuracy
    assert baseline["replay_expected"]["by_flaw"] == harness.by_flaw
    assert json.loads((tmp_path / "published_failures.json").read_text()) == []
    assert Path("evals/published_failures.json").read_bytes() == published


def test_demo_prints_one_results_table(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert out.count("| Variant |") == 1


def test_demo_prints_no_logfire_line(capsys):
    assert main(["demo"]) == 0
    captured = capsys.readouterr()
    combined = f"{captured.out}\n{captured.err}"
    assert "Logfire" not in combined


def test_sample_parquet_has_a_segment_column():
    frame = pd.read_parquet("fixtures/sample.parquet")
    assert {"segment", "arm", "converted", "pre_period_metric"} <= set(frame.columns)


@pytest.mark.live
def test_the_dispatched_live_job_ran_the_gate():
    listed = subprocess.check_output(
        [
            "gh",
            "run",
            "list",
            "--workflow",
            "live-eval.yml",
            "--limit",
            "1",
            "--json",
            "databaseId,conclusion,headSha",
        ],
        text=True,
    )
    run = json.loads(listed)[0]
    log = subprocess.check_output(
        ["gh", "run", "view", str(run["databaseId"]), "--log"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    assert "regression gate: OK" in log or "REGRESSION GATE FAILED" in log
