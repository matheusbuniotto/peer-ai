"""The README table and the error note are rendered from the committed run."""

from __future__ import annotations

import json
from pathlib import Path

from peer_agent.report import render_errors, render_table


def _committed() -> tuple[Path, dict]:
    run_id = Path("results/LATEST").read_text().strip()
    path = Path("results") / run_id / "results.json"
    return path, json.loads(path.read_text())


def test_readme_table_is_fresh():
    path, payload = _committed()
    text = Path("README.md").read_text()
    block = text.split("<!-- results:start -->", 1)[1].split("<!-- results:end -->", 1)[0]
    primary = next(v for v in payload["variants"] if v.get("tools"))
    k = int(payload["trials"])
    assert payload["run_id"] in block
    assert f"{primary['accuracy']:.1%}" in block
    assert f"pass^{k}" in block
    assert "python-only" in block
    assert "pending" in block.lower()


def test_pending_variant_rows_are_not_scored():
    block = render_table(
        {
            "trials": 3,
            "variants": [
                {
                    "label": "m (tools)",
                    "model": "m",
                    "tools": True,
                    "accuracy": 0.5,
                    "accuracy_ci": [0.3, 0.7],
                    "pass_hat_k": 0.4,
                    "false_ships": 0.1,
                    "false_blocks": 0.0,
                    "cost_usd": {"per_case_mean": None},
                    "latency_s": {"p50": 1.0},
                    "by_flaw": {"clean": {"n": 1}},
                },
                {
                    "label": "m (python-only)",
                    "model": "m",
                    "pending": True,
                },
            ],
            "comparisons": [
                {"a": "m (tools)", "b": "m (python-only)", "pending": True},
            ],
            "run_id": "x",
            "generated_at": "2026-01-01T00:00:00+00:00",
            "git_sha": "abc",
        }
    )
    assert "_not recorded_" in block
    assert "pending live run" in block


def test_error_report_matches_the_replay_and_adds_up():
    path, payload = _committed()
    text = render_errors(payload)
    assert text == (path.parent / "errors.md").read_text()
    misses = sum(not case["correct"] for case in payload["cases"])
    assert f"Matrix sum: {misses}." in text


def test_analysis_sql_runs_cleanly():
    import duckdb

    sql = Path("analysis/eval_queries.sql").read_text()
    conn = duckdb.connect()
    conn.execute(sql)


def test_web_results_matches_latest():
    path, payload = _committed()
    web_payload = json.loads(Path("web/results.json").read_text())
    assert web_payload["run_id"] == payload["run_id"]
