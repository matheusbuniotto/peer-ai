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
    block = render_table(payload, source=str(path))
    assert block in Path("README.md").read_text()
    k = int(payload["trials"])
    assert f"pass^{k}" in block
    if k != 3:
        assert "pass^3" not in block


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
