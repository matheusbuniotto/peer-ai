"""One command that shows the harness, the toolbox, and the committed score."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from peer_agent import stats
from peer_agent.agent import Agent
from peer_agent.evals import check_regression, load_suite, run_suite
from peer_agent.report import render_table

_REPLAY = "fixtures/recorded.jsonl"
_SAMPLE = "fixtures/sample.parquet"
_BASELINE = Path("evals/baseline.json")


def run_demo(*, live: bool = False) -> int:
    _srm_contrast()
    _toolbox()
    code = _gate()
    _latest()
    if live:
        _live_review()
    print("open http://127.0.0.1:8000 with peer web")
    return code


def _srm_contrast() -> None:
    print("## Replay: SRM")
    case = next(item for item in load_suite() if item.id == "srm-01")
    report = run_suite([case], replay=_REPLAY)
    for row in report.cases:
        calls = " → ".join(turn["act"] for turn in row["trajectory"])
        print(f"{row['variant']}: {calls}")
        print(f"  verdict {row['verdict']} (truth {row['truth']})")


def _toolbox() -> None:
    print("\n## Toolbox on the sample")
    frame = pd.read_parquet(_SAMPLE)
    print(f"check_srm: {stats.check_srm(frame)}")
    print(f"analyze: {stats.analyze(frame)}")
    try:
        print(f"scan_segments: {stats.scan_segments(frame)}")
    except ValueError as exc:
        print(f"scan_segments: {exc}")


def _gate() -> int:
    print("\n## Harness gate")
    report = run_suite(load_suite(), replay=_REPLAY)
    baseline = json.loads(_BASELINE.read_text())
    violations = check_regression(report, baseline, mode="replay")
    print(render_table(report.to_json()))
    if violations:
        print("regression gate: FAILED")
        for item in violations:
            print(f"  - {item}")
        return 1
    print("regression gate: OK")
    return 0


def _latest() -> None:
    print("\n## Latest results")
    pointer = Path("results/LATEST")
    if not pointer.exists():
        print("results/LATEST is missing. No committed run to show.")
        return
    run_id = pointer.read_text().strip()
    path = Path("results") / run_id / "results.json"
    payload = json.loads(path.read_text())
    print(path)
    print(render_table(payload, source=str(path)))
    if payload.get("mode") != "live":
        print("This committed file is a harness replay. It is not a live model score.")


def _live_review() -> None:
    print("\n## Live review")
    if not os.environ.get("LLM_API_KEY"):
        print("LLM_API_KEY is not set. Skipping the live review.")
        return
    frame = pd.read_parquet(_SAMPLE)
    from peer_agent.sim import Case
    from peer_agent.types import Verdict

    traj = Agent.from_env().review(Case(df=frame, truth=Verdict.NO_EFFECT))
    print(traj.answer)
    link = os.environ.get("LOGFIRE_TRACE_URL")
    if link:
        print(f"logfire: {link}")
