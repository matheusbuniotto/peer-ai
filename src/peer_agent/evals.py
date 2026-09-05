from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_evals import Case as EvalsCase
from pydantic_evals import Dataset

from peer_agent.agent import Agent, Trajectory, json_default
from peer_agent.sim import NOVELTY, PEEKING, SIMPSON, SRM, Case, Flaw, make_case
from peer_agent.tools import Tool, analysis_tools
from peer_agent.types import Verdict

_SUITE_PATH = Path("evals/suite.yaml")
_PUBLISHED_FAILURES_PATH = Path("evals/published_failures.json")
_TOOLS_SUFFIX = " (tools)"
_NO_TOOLS_SUFFIX = " (no tools)"

_FLAW_FNS: dict[str, Flaw] = {
    "srm": SRM,
    "novelty": NOVELTY,
    "simpson": SIMPSON,
    "peeking": PEEKING,
}


@dataclass(frozen=True)
class EvalCase:
    id: str
    flaw: str
    seed: int
    truth: Verdict
    n: int = 40_000
    lift: float = 0.0
    days: int | None = None

    def build(self) -> Case:
        flaws = [_FLAW_FNS[self.flaw]] if self.flaw in _FLAW_FNS else None
        return make_case(
            n=self.n, lift=self.lift, seed=self.seed, flaws=flaws, days=self.days
        )


def load_suite(path: str | Path = _SUITE_PATH) -> list[EvalCase]:
    data = yaml.safe_load(Path(path).read_text())
    return [
        EvalCase(
            id=row["id"],
            flaw=row["flaw"],
            seed=row["seed"],
            truth=Verdict[row["truth"]],
            n=row.get("n", 40_000),
            lift=row.get("lift", 0.0),
            days=row.get("days"),
        )
        for row in data["cases"]
    ]


class _ReplayLLM:
    """Same turn shape as conftest.py's FakeLLM — a tool-call list, or a final string."""

    def __init__(self, turns: list[Any]):
        self._turns = list(turns)

    def __call__(self, messages: Any, tools: Any) -> Any:
        del messages, tools
        turn = self._turns.pop(0)
        if isinstance(turn, str):
            return turn
        return [(name, args) for name, args in turn]


def _load_recorded(path: str | Path) -> dict[tuple[str, str], list[Any]]:
    recorded: dict[tuple[str, str], list[Any]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        recorded[(row["case_id"], row["model"])] = row["turns"]
    return recorded


def _live_model(model: str) -> OpenAIChatModel:
    provider = OpenAIProvider(
        base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"]
    )
    settings = OpenAIChatModelSettings(openai_store=False)
    return OpenAIChatModel(model, provider=provider, settings=settings)


def _turns(traj: Trajectory) -> list[dict[str, Any]]:
    turns = [
        {"turn": i, "act": c.name, "body": {"args": c.args, "result": c.result}}
        for i, c in enumerate(traj.calls)
    ]
    turns.append(
        {
            "turn": len(turns),
            "act": "verdict" if traj.verdict is not None else "answer",
            "body": traj.answer,
        }
    )
    return turns


@dataclass
class _Variant:
    label: str
    accuracy: float
    false_ships: float
    false_blocks: float
    by_flaw: dict[str, float]
    cases: list[dict[str, Any]]


def _run_variant(
    suite: list[EvalCase],
    *,
    label: str,
    tools: list[Tool],
    llm_for: Callable[[EvalCase], Any],
) -> _Variant:
    dataset = Dataset(
        name="peer-eval", cases=[EvalsCase(name=ec.id, inputs=ec) for ec in suite]
    )

    def task(ec: EvalCase) -> Trajectory:
        return Agent(llm_for(ec), tools=tools).review(ec.build())

    report = dataset.evaluate_sync(task, progress=False)
    by_id = {rc.name: rc for rc in report.cases}

    cases: list[dict[str, Any]] = []
    correct = false_ships = false_blocks = 0
    flaw_correct: dict[str, int] = {}
    flaw_total: dict[str, int] = {}
    for ec in suite:
        traj: Trajectory = by_id[ec.id].output
        is_correct = traj.verdict is ec.truth
        correct += is_correct
        if traj.verdict is Verdict.SHIP and ec.truth is not Verdict.SHIP:
            false_ships += 1
        if traj.verdict is not Verdict.SHIP and ec.truth is Verdict.SHIP:
            false_blocks += 1
        flaw_total[ec.flaw] = flaw_total.get(ec.flaw, 0) + 1
        flaw_correct[ec.flaw] = flaw_correct.get(ec.flaw, 0) + is_correct
        cases.append(
            {
                "id": ec.id,
                "flaw": ec.flaw,
                "correct": is_correct,
                "truth": ec.truth.name,
                "verdict": traj.verdict.name if traj.verdict else None,
                "trajectory": _turns(traj),
            }
        )
    n = len(suite)
    return _Variant(
        label=label,
        accuracy=correct / n,
        false_ships=false_ships / n,
        false_blocks=false_blocks / n,
        by_flaw={flaw: flaw_correct[flaw] / flaw_total[flaw] for flaw in flaw_total},
        cases=cases,
    )


def _load_published_failures() -> list[dict[str, Any]]:
    if not _PUBLISHED_FAILURES_PATH.exists():
        return []
    data: list[dict[str, Any]] = json.loads(_PUBLISHED_FAILURES_PATH.read_text())
    return data


@dataclass
class Report:
    accuracy: float
    false_ships: float
    false_blocks: float
    by_flaw: dict[str, float]
    cases: list[dict[str, Any]]
    models: list[dict[str, Any]]
    published_failures: list[dict[str, Any]]
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_json(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "false_ships": self.false_ships,
            "false_blocks": self.false_blocks,
            "by_flaw": self.by_flaw,
            "cases": self.cases,
            "models": self.models,
            "published_failures": self.published_failures,
        }

    def write(self, out: str | Path) -> None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        payload = self.to_json() | {"generated_at": self.generated_at}
        (out / "results.json").write_text(
            json.dumps(payload, default=json_default, indent=2)
        )


def run_suite(
    suite: list[EvalCase],
    *,
    replay: str | Path | None = None,
    model: str | None = None,
    tools: list[Tool] | None = None,
    out: str | Path | None = None,
) -> Report:
    if replay is not None:
        recorded = _load_recorded(replay)
        model_names = {
            full.removesuffix(_TOOLS_SUFFIX)
            for _, full in recorded
            if full.endswith(_TOOLS_SUFFIX)
        }
        (model_name,) = model_names
        tools_label = f"{model_name}{_TOOLS_SUFFIX}"
        no_tools_label = f"{model_name}{_NO_TOOLS_SUFFIX}"
        primary = _run_variant(
            suite,
            label=tools_label,
            tools=analysis_tools(),
            llm_for=lambda ec: _ReplayLLM(recorded[(ec.id, tools_label)]),
        )
        baseline = _run_variant(
            suite,
            label=no_tools_label,
            tools=[],
            llm_for=lambda ec: _ReplayLLM(recorded[(ec.id, no_tools_label)]),
        )
        models = [
            {"name": primary.label, "accuracy": primary.accuracy},
            {"name": baseline.label, "accuracy": baseline.accuracy},
        ]
    else:
        if model is None:
            raise ValueError("run_suite needs either replay= or model=")
        use_tools = analysis_tools() if tools is None else tools
        label = f"{model}{_TOOLS_SUFFIX if tools is None else _NO_TOOLS_SUFFIX}"
        primary = _run_variant(
            suite, label=label, tools=use_tools, llm_for=lambda _ec: _live_model(model)
        )
        models = [{"name": primary.label, "accuracy": primary.accuracy}]

    report = Report(
        accuracy=primary.accuracy,
        false_ships=primary.false_ships,
        false_blocks=primary.false_blocks,
        by_flaw=primary.by_flaw,
        cases=primary.cases,
        models=models,
        published_failures=_load_published_failures(),
    )
    if out is not None:
        report.write(out)
    return report


_ACCURACY_TOLERANCE = 0.02
_FLAW_TOLERANCE = 0.10


def check_regression(report: Report, baseline: dict[str, Any]) -> list[str]:
    """Mirrors test_the_baseline_gate_catches_a_regression's tolerances exactly."""
    violations = []
    if report.accuracy < baseline["accuracy"] - _ACCURACY_TOLERANCE:
        floor = baseline["accuracy"] - _ACCURACY_TOLERANCE
        violations.append(f"accuracy {report.accuracy:.3f} < baseline floor {floor:.3f}")
    for flaw, score in baseline["by_flaw"].items():
        current = report.by_flaw.get(flaw, 0.0)
        if current < score - _FLAW_TOLERANCE:
            violations.append(
                f"{flaw}: {current:.3f} < baseline {score:.3f} - {_FLAW_TOLERANCE}"
            )
    return violations
