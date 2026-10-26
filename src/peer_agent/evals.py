"""Run the suite, score it, and decide whether a run may merge.

Replay is a harness gate: the recorded turns are fixed, so a change in parsing,
scoring, or the schema moves the number and an unchanged model cannot. A live
run is the model and prompt gate. The two are not the same comparison.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import logfire
import numpy as np
import pandas as pd
import yaml
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_evals import Case as EvalsCase
from pydantic_evals import Dataset

from peer_agent.agent import ROWS_ARE_LOADED, Agent, Trajectory, chat_settings, json_default
from peer_agent.sandbox import Sandbox
from peer_agent.scoring import (
    classify_error,
    cluster_rate_regression,
    false_ship_row,
    format_fail_detail,
    mcnemar,
    pass_at_k,
    pass_hat_k,
    run_failure_row,
    srm_checked_before_lift,
    wilson,
)
from peer_agent.sim import (
    NOVELTY,
    OUTLIERS,
    PEEKING,
    SIMPSON,
    SRM,
    Case,
    Flaw,
    make_case,
    make_sequential_case,
)
from peer_agent.tools import Tool, analysis_tools, python_only_tools
from peer_agent.types import BY_USER, DesignSpec, Verdict

_SUITE_PATH = Path("evals/suite.yaml")
_PRICES_PATH = Path("evals/pricing.yaml")
_TOOLS_SUFFIX = " (tools)"
_NO_TOOLS_SUFFIX = " (no tools)"
_PYTHON_ONLY_SUFFIX = " (python-only)"
_VARIANT_KEYS = frozenset({"tools", "python-only", "no-tools"})
_PROMPT_PATH = "prompt.md"

_FLAW_FNS: dict[str, Flaw] = {
    "srm": SRM,
    "novelty": NOVELTY,
    "simpson": SIMPSON,
    "peeking": PEEKING,
    "outliers": OUTLIERS,
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
        if self.flaw == "sequential":
            return make_sequential_case(
                self.n, self.lift, seed=self.seed, days=self.days or 14
            )
        flaws = [_FLAW_FNS[self.flaw]] if self.flaw in _FLAW_FNS else None
        return make_case(
            n=self.n, lift=self.lift, seed=self.seed, flaws=flaws, days=self.days
        )  # noqa: E501


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


def load_prices(path: str | Path = _PRICES_PATH) -> dict[str, dict[str, float]]:
    file = Path(path)
    if not file.exists():
        return {}
    data = yaml.safe_load(file.read_text()) or {}
    models = data.get("models", data)
    return models if isinstance(models, dict) else {}


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


def _load_recorded(path: str | Path) -> dict[tuple[str, str, int], list[Any]]:
    recorded: dict[tuple[str, str, int], list[Any]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        recorded[(row["case_id"], row["model"], int(row.get("trial", 0)))] = row["turns"]
    return recorded


def _live_model(model: str) -> OpenAIChatModel:
    from peer_agent.agent import openai_provider

    base_url = os.environ["LLM_BASE_URL"]
    provider = openai_provider(base_url, os.environ["LLM_API_KEY"])
    return OpenAIChatModel(model, provider=provider, settings=chat_settings(base_url))


def _prereg_spec(ec: EvalCase) -> DesignSpec:
    """Same pre-registration for every case — analysis defaults, not a flaw hint."""
    days = ec.days or 14
    # 5% MDE: below the 6% planted lifts on clean-06..10 so real wins stay SHIP,
    # while null clean cases can still reach NO_EFFECT when powered.
    looks = days if ec.flaw == "sequential" else 1
    # Per-case traffic hint (deterministic from id) so prereg text is not identical
    # across flaws that share the same truth — without using sample size as a label.
    digest = hashlib.sha256(ec.id.encode()).hexdigest()
    traffic = 10_000 + int(digest[:8], 16) % 8_000
    return DesignSpec(
        baseline=0.12,
        mde=0.05,
        power=0.8,
        days=days,
        looks=looks,
        n_per_arm=max(ec.n // 2, 1_000),
        unit=BY_USER,
        metric="converted",
        daily_traffic=traffic,
    )


def format_prereg_for_prompt(spec: DesignSpec) -> str:
    """Pre-reg block shown to the model — omits n_per_arm (visible in the data)."""
    parts = [
        f"baseline={spec.baseline}",
        f"mde={spec.mde}",
        f"power={spec.power}",
        f"days={spec.days}",
        f"looks={spec.looks}",
        f"unit={spec.unit.value}",
        f"metric={spec.metric}",
        f"alpha={spec.alpha}",
    ]
    if spec.daily_traffic is not None:
        parts.append(f"daily_traffic={spec.daily_traffic}")
    if spec.covariate:
        parts.append(f"covariate={spec.covariate}")
    if spec.if_flat:
        parts.append(f"if_flat={spec.if_flat!r}")
    return "Pre-registered analysis defaults:\n" + "\n".join(parts)


def _review_question() -> str:
    return f"Review this experiment.\n\n{ROWS_ARE_LOADED}"


def review_prompt_text(ec: EvalCase) -> str:
    """Full user message for a suite case (preamble + prereg + question)."""
    from peer_agent.agent import _PREREG_PREAMBLE

    spec = _prereg_spec(ec)
    return f"{_PREREG_PREAMBLE}\n\n{format_prereg_for_prompt(spec)}\n\n{_review_question()}"


def _turns(traj: Trajectory) -> list[dict[str, Any]]:
    turns = [
        {"turn": i, "act": call.name, "body": {"args": call.args, "result": call.result}}
        for i, call in enumerate(traj.calls)
    ]
    turns.append(
        {
            "turn": len(turns),
            "act": "verdict" if traj.verdict is not None else "answer",
            "body": traj.answer,
        }
    )
    return turns


def _replay_turns(traj: Trajectory) -> list[Any]:
    turns: list[Any] = [[[call.name, call.args]] for call in traj.calls]
    turns.append(traj.answer)
    return turns


def _cost(
    model: str, row: dict[str, Any], prices: dict[str, dict[str, float]]
) -> float | None:  # noqa: E501
    card = prices.get(model)
    if card is None or row["input_tokens"] is None or row["output_tokens"] is None:
        return None
    million = 1_000_000
    return (
        row["input_tokens"] * card["input_per_mtok"]
        + row["output_tokens"] * card["output_per_mtok"]  # noqa: E501
    ) / million


def _usage_tokens(traj: Trajectory, mode: str) -> tuple[int | None, int | None]:
    if mode == "replay" or traj.usage is None:
        return None, None
    return traj.usage.input_tokens, traj.usage.output_tokens


def _case_row(  # noqa: PLR0913 (one field group for a case row)
    ec: EvalCase,
    traj: Trajectory,
    *,
    label: str,
    model: str,
    trial: int,
    mode: str,
    prices: dict[str, dict[str, float]],
) -> dict[str, Any]:
    error = classify_error(ec, traj)
    input_tokens, output_tokens = _usage_tokens(traj, mode)
    failure_detail = format_fail_detail(traj) if error is not None else None
    row: dict[str, Any] = {
        "id": ec.id,
        "flaw": ec.flaw,
        "truth": ec.truth.name,
        "variant": label,
        "model": model,
        "trial": trial,
        "verdict": traj.verdict.name if traj.verdict else None,
        "correct": error is None,
        "error": None if error is None else error.value,
        "failure_detail": failure_detail,
        "srm_checked_before_lift": srm_checked_before_lift(traj),
        "latency_s": traj.latency_s,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "trajectory": _turns(traj),
    }
    row["cost_usd"] = _cost(model, row, prices)
    return row


def _annotate(span: Any, row: dict[str, Any]) -> None:
    setter = getattr(span, "set_attribute", None)
    if setter is None:
        return
    setter("gen_ai.usage.input_tokens", row["input_tokens"] or 0)
    setter("gen_ai.usage.output_tokens", row["output_tokens"] or 0)
    setter("peer.cost_usd", 0 if row["cost_usd"] is None else row["cost_usd"])
    setter("peer.error_category", row["error"] or "")
    setter("peer.correct", bool(row["correct"]))


@dataclass(frozen=True)
class _Ctx:
    label: str
    model: str
    mode: str
    trial: int
    tools: list[Tool]
    prices: dict[str, dict[str, float]]
    recorder: Callable[[dict[str, Any]], None] | None


def _one_case(ec: EvalCase, llm: Any, ctx: _Ctx) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        built = ec.build()
        with logfire.span(
            "eval.case", case_id=ec.id, flaw=ec.flaw, trial=ctx.trial, variant=ctx.label
        ) as span:
            traj = Agent(llm, tools=ctx.tools).review(
                built,
                spec=_prereg_spec(ec),
                question=_review_question(),
            )  # noqa: E501
            row = _case_row(
                ec,
                traj,
                label=ctx.label,
                model=ctx.model,
                trial=ctx.trial,
                mode=ctx.mode,
                prices=ctx.prices,  # noqa: E501
            )
            _annotate(span, row)
    except Exception as exc:
        traj = Trajectory(
            done=False,
            error=f"{type(exc).__name__}: {exc}",
            latency_s=time.perf_counter() - started,
        )
        row = _case_row(
            ec,
            traj,
            label=ctx.label,
            model=ctx.model,
            trial=ctx.trial,
            mode=ctx.mode,
            prices=ctx.prices,  # noqa: E501
        )
    if ctx.recorder is not None:
        ctx.recorder(
            {
                "case_id": ec.id,
                "model": ctx.label,
                "trial": ctx.trial,
                "turns": _replay_turns(traj),
            }  # noqa: E501
        )
    return row


def _run_once(
    suite: list[EvalCase],
    factory: Callable[[EvalCase, int], Any],
    ctx: _Ctx,
    concurrency: int,
) -> list[dict[str, Any]]:  # noqa: E501
    dataset = Dataset(
        name="peer-eval", cases=[EvalsCase(name=ec.id, inputs=ec) for ec in suite]
    )  # noqa: E501

    def task(ec: EvalCase) -> dict[str, Any]:
        return _one_case(ec, factory(ec, ctx.trial), ctx)

    report = dataset.evaluate_sync(task, progress=False, max_concurrency=concurrency)
    order = {ec.id: index for index, ec in enumerate(suite)}
    rows = [case.output for case in report.cases]
    rows.sort(key=lambda row: order[row["id"]])
    return rows


def _vote(flags: list[bool]) -> bool:
    if len(flags) == 1:
        return flags[0]
    votes = sum(flags)
    if votes * 2 == len(flags):
        return flags[0]
    return votes * 2 > len(flags)


def _matrix(
    rows: list[dict[str, Any]], case_ids: list[str], trials: int
) -> list[list[bool]]:  # noqa: E501
    found = {(row["id"], row["trial"]): bool(row["correct"]) for row in rows}
    return [[found[(case_id, trial)] for trial in range(trials)] for case_id in case_ids]


def _mean(values: list[float]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return float(sum(values)) / len(values)


def _by_flaw(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Trial 0 only, so `n` is the number of cases and the Wilson interval matches it."""
    trial0 = [row for row in rows if row["trial"] == 0]
    flaws = sorted({row["flaw"] for row in trial0})
    summary = {}
    for flaw in flaws:
        subset = [row for row in trial0 if row["flaw"] == flaw]
        hits = sum(bool(row["correct"]) for row in subset)
        low, high = wilson(hits, len(subset))
        summary[flaw] = {"acc": hits / len(subset), "ci": [low, high], "n": len(subset)}
    return summary


def _summarize(  # noqa: PLR0913, PLR0917
    rows: list[dict[str, Any]],
    case_ids: list[str],
    trials: int,
    label: str,
    model: str,
    tools: bool,
) -> dict[str, Any]:
    by_trial = []
    for trial in range(trials):
        subset = [row for row in rows if row["trial"] == trial]
        by_trial.append(sum(bool(row["correct"]) for row in subset) / len(subset))
    trial0 = [row for row in rows if row["trial"] == 0]
    hits = sum(bool(row["correct"]) for row in trial0)
    latency = np.asarray([row["latency_s"] for row in rows], dtype=float)
    tokens_in = [row["input_tokens"] for row in rows if row["input_tokens"] is not None]
    tokens_out = [row["output_tokens"] for row in rows if row["output_tokens"] is not None]
    costs = [row["cost_usd"] for row in rows if row["cost_usd"] is not None]
    missing_usage = sum(
        1 for row in rows if row["input_tokens"] is None or row["output_tokens"] is None
    )
    total_cost = float(sum(costs)) if costs else None
    matrix = _matrix(rows, case_ids, trials)
    return {
        "label": label,
        "model": model,
        "tools": tools,
        "accuracy": float(np.mean(by_trial)),
        "accuracy_ci": list(wilson(hits, len(trial0))),
        "accuracy_by_trial": by_trial,
        "pass_at_k": pass_at_k(matrix),
        "pass_hat_k": pass_hat_k(matrix),
        "false_ships": _rate(rows, "SHIP", wrong=True),
        "false_blocks": _block_rate(rows),
        "by_flaw": _by_flaw(rows),
        "errors": dict(Counter(row["error"] for row in rows if row["error"])),
        "process": {
            "srm_checked_before_lift": float(
                np.mean([row["srm_checked_before_lift"] for row in rows])
            )  # noqa: E501
        },
        "cost_usd": {
            "total": total_cost,
            "per_case_mean": None if not costs else total_cost / len(costs),
            "missing_rows": missing_usage,
        },  # noqa: E501
        "latency_s": {
            "p50": float(np.percentile(latency, 50)),
            "p95": float(np.percentile(latency, 95)),
        },  # noqa: E501
        "tokens": {
            "input": int(sum(tokens_in)) if tokens_in else None,
            "output": int(sum(tokens_out)) if tokens_out else None,
            "missing_rows": missing_usage,
        },
    }


def _rate(rows: list[dict[str, Any]], verdict: str, *, wrong: bool) -> float:
    del wrong
    flagged = sum(row["verdict"] == verdict and row["truth"] != verdict for row in rows)
    return flagged / len(rows)


def _block_rate(rows: list[dict[str, Any]]) -> float:
    flagged = sum(row["truth"] == "SHIP" and row["verdict"] != "SHIP" for row in rows)
    return flagged / len(rows)


def _compare(  # noqa: PLR0913, PLR0917
    a: dict[str, Any],
    b: dict[str, Any],
    a_rows: list[dict[str, Any]],
    b_rows: list[dict[str, Any]],
    case_ids: list[str],
    trials: int,
) -> dict[str, Any]:
    result = mcnemar(_voted(a_rows, case_ids, trials), _voted(b_rows, case_ids, trials))
    return {
        "a": a["label"],
        "b": b["label"],
        "b_only_a_right": result.b,
        "c_only_b_right": result.c,
        "p_value": result.p_value,
        "method": result.method,
    }


def _voted(rows: list[dict[str, Any]], case_ids: list[str], trials: int) -> list[bool]:
    return [_vote(flags) for flags in _matrix(rows, case_ids, trials)]


@dataclass
class Report:
    accuracy: float
    false_ships: float
    false_blocks: float
    by_flaw: dict[str, float]
    cases: list[dict[str, Any]]
    models: list[dict[str, Any]]
    published_failures: list[dict[str, Any]]
    accuracy_ci: tuple[float, float] = (0.0, 1.0)
    variants: list[dict[str, Any]] = field(default_factory=list)
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    schema_version: int = 2
    run_id: str = ""
    git_sha: str = ""
    suite_sha256: str = ""
    prompt_sha256: str = ""
    mode: str = "replay"
    trials: int = 1
    temperature: float | None = None
    generated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "git_sha": self.git_sha,
            "suite_sha256": self.suite_sha256,
            "prompt_sha256": self.prompt_sha256,
            "mode": self.mode,
            "trials": self.trials,
            "temperature": self.temperature,
            "generated_at": self.generated_at,
            "accuracy": self.accuracy,
            "accuracy_ci": list(self.accuracy_ci),
            "false_ships": self.false_ships,
            "false_blocks": self.false_blocks,
            "by_flaw": self.by_flaw,
            "models": self.models,
            "variants": self.variants,
            "comparisons": self.comparisons,
            "cases": self.cases,
            "published_failures": self.published_failures,
        }

    def write(self, out: str | Path, *, nest: bool = False) -> Path:
        dest = Path(out)
        if nest:
            dest = dest / self.run_id
            Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out) / "LATEST").write_text(self.run_id + "\n")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "results.json").write_text(
            json.dumps(self.to_json(), default=json_default, indent=2) + "\n"
        )  # noqa: E501
        _write_cases(dest / "cases.parquet", self.cases)
        return dest


def _write_cases(path: Path, cases: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(
        [{key: case.get(key) for key in _PARQUET_COLUMNS} for case in cases]
    )
    if frame.empty:
        frame = pd.DataFrame(columns=_PARQUET_COLUMNS)
    frame["correct"] = frame["correct"].astype("int64")
    frame["srm_checked_before_lift"] = frame["srm_checked_before_lift"].astype("int64")
    frame.to_parquet(path, index=False)


_PARQUET_COLUMNS = (
    "id",
    "flaw",
    "truth",
    "variant",
    "model",
    "trial",
    "verdict",
    "correct",
    "error",
    "srm_checked_before_lift",
    "latency_s",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "failure_detail",
)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError, FileNotFoundError:
        return "unknown"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prompt_sha() -> str:
    from importlib import resources

    blob = resources.files("peer_agent").joinpath(_PROMPT_PATH).read_bytes()
    return hashlib.sha256(blob).hexdigest()


def _run_id(model: str, when: datetime) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in model).strip("-")
    slug = "-".join(part for part in slug.split("-") if part) or "run"
    stamp = when.strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{stamp}_{slug}_{_git_sha()}"


def _model_of(label: str) -> str:
    return label.removesuffix(_TOOLS_SUFFIX).removesuffix(_NO_TOOLS_SUFFIX)


def _failures(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "case_id": row["id"],
            "variant": row["variant"],
            "trial": row["trial"],
            "note": row["error"],
        }  # noqa: E501
        for row in cases
        if not row["correct"]
    ]


def _primary(variants: list[dict[str, Any]]) -> dict[str, Any]:
    for variant in variants:
        if variant["tools"]:
            return variant
    return variants[0]


def _variant_label(model: str, key: str) -> str:
    if key == "tools":
        return f"{model}{_TOOLS_SUFFIX}"
    if key == "python-only":
        return f"{model}{_PYTHON_ONLY_SUFFIX}"
    if key == "no-tools":
        return f"{model}{_NO_TOOLS_SUFFIX}"
    raise ValueError(f"unknown variant {key!r}")


def _live_variant_keys(
    tools: list[Tool] | None, variants: tuple[str, ...] | None
) -> tuple[str, ...]:
    if variants is not None:
        return variants
    if tools == []:
        return ("no-tools",)
    if tools is None:
        return ("tools", "python-only")
    return ("tools",)


def _tools_for_kind(key: str, tools: list[Tool] | None, sandbox: Sandbox) -> list[Tool]:
    if key == "no-tools":
        return []
    if key == "python-only":
        return python_only_tools(sandbox)
    if tools == []:
        return []
    return analysis_tools() if tools is None else tools


def _recorded_label(recorded: dict[tuple[str, str, int], list[Any]], label: str) -> bool:
    return any(name == label for _, name, _ in recorded)


def _replay_variant_specs(
    recorded: dict[tuple[str, str, int], list[Any]],
    base: str,
    variants: tuple[str, ...] | None,
) -> list[tuple[str, str, Callable[[EvalCase, int], Any] | None]]:
    specs: list[tuple[str, str, Callable[[EvalCase, int], Any] | None]] = []
    for key in _live_variant_keys(None, variants):
        if key == "tools":
            label = f"{base}{_TOOLS_SUFFIX}"
            specs.append((label, key, _replay_factory(recorded, label)))
        elif key == "python-only":
            label = f"{base}{_PYTHON_ONLY_SUFFIX}"
            if _recorded_label(recorded, label):
                specs.append((label, key, _replay_factory(recorded, label)))
            else:
                specs.append((label, key, None))
        elif key == "no-tools":
            label = f"{base}{_NO_TOOLS_SUFFIX}"
            specs.append((label, key, _replay_factory(recorded, label)))
    return specs


def _variant_specs(
    replay: str | Path | None,
    model: str | None,
    tools: list[Tool] | None,
    variants: tuple[str, ...] | None,
    llm_for: Callable[[EvalCase, int], Any] | None,
) -> list[tuple[str, str, Callable[[EvalCase, int], Any] | None]]:
    if replay is not None:
        recorded = _load_recorded(replay)
        names = sorted({name for _, name, _ in recorded if name.endswith(_TOOLS_SUFFIX)})
        if len(names) != 1:
            raise ValueError(f"replay needs one tools variant, found {names}")
        base = names[0].removesuffix(_TOOLS_SUFFIX)
        return _replay_variant_specs(recorded, base, variants)
    if llm_for is not None:
        return _live_specs(model or "fake", _live_variant_keys(tools, variants), llm_for)
    if model is None:
        raise ValueError("run_suite needs either replay= or model=")
    return _live_specs(
        model, _live_variant_keys(tools, variants), lambda _ec, _trial: _live_model(model)
    )


def _pending_variant_summary(label: str, model: str) -> dict[str, Any]:
    return {
        "label": label,
        "model": model,
        "tools": False,
        "pending": True,
        "pending_reason": (
            "No recorded turns for this variant; run a live eval to score it."
        ),
        "accuracy": None,
        "accuracy_ci": None,
        "accuracy_by_trial": [],
        "pass_at_k": None,
        "pass_hat_k": None,
        "false_ships": None,
        "false_blocks": None,
        "by_flaw": {},
        "errors": {},
        "process": {"srm_checked_before_lift": None},
        "cost_usd": {"total": None, "per_case_mean": None, "missing_rows": 0},
        "latency_s": {"p50": None, "p95": None},
        "tokens": {"input": None, "output": None, "missing_rows": 0},
    }


def _replay_factory(
    recorded: dict[tuple[str, str, int], list[Any]], label: str
) -> Callable[[EvalCase, int], Any]:  # noqa: E501
    def factory(ec: EvalCase, trial: int) -> _ReplayLLM:
        try:
            turns = recorded[(ec.id, label, trial)]
        except KeyError as exc:
            raise KeyError(
                f"replay has no turns for {ec.id} trial {trial} ({label})"
            ) from exc  # noqa: E501
        return _ReplayLLM(turns)

    return factory


def _live_specs(
    model: str,
    chosen: tuple[str, ...],
    llm_for: Callable[[EvalCase, int], Any],
) -> list[tuple[str, str, Callable[[EvalCase, int], Any]]]:
    unknown = set(chosen) - _VARIANT_KEYS
    if unknown:
        raise ValueError(f"unknown variants {sorted(unknown)}")
    return [(_variant_label(model, name), name, llm_for) for name in chosen]


def run_suite(  # noqa: PLR0913 (the CLI flags are the arguments)
    suite: list[EvalCase],
    *,
    replay: str | Path | None = None,
    model: str | None = None,
    tools: list[Tool] | None = None,
    out: str | Path | None = None,
    trials: int = 1,
    variants: tuple[str, ...] | None = None,
    concurrency: int = 1,
    record: str | Path | None = None,
    llm_for: Callable[[EvalCase, int], Any] | None = None,
    nest: bool = False,
    suite_path: str | Path | None = None,
) -> Report:
    if trials < 1:
        raise ValueError("trials must be >= 1")
    specs = _variant_specs(replay, model, tools, variants, llm_for)
    prices = {} if replay is not None else load_prices()
    mode = "replay" if replay is not None else "live"
    when = datetime.now(UTC)
    model_name = _model_of(specs[0][0])
    recorder, record_file = _recorder(record)
    case_ids = [ec.id for ec in suite]
    variant_rows: list[list[dict[str, Any]]] = []
    summaries: list[dict[str, Any]] = []
    sandbox = Sandbox()
    try:
        for label, kind, factory in specs:
            if factory is None:
                variant_rows.append([])
                summaries.append(_pending_variant_summary(label, model_name))
                continue
            rows: list[dict[str, Any]] = []
            tool_list = _tools_for_kind(kind, tools, sandbox)
            for trial in range(trials):
                ctx = _Ctx(label, model_name, mode, trial, tool_list, prices, recorder)
                rows.extend(_run_once(suite, factory, ctx, concurrency))
            variant_rows.append(rows)
            summaries.append(
                _summarize(rows, case_ids, trials, label, model_name, kind == "tools")
            )  # noqa: E501
    finally:
        if record_file is not None:
            record_file.close()
    return _finish(
        summaries,
        variant_rows,
        case_ids,
        trials,
        mode,
        model_name,
        when,
        out,
        nest,
        suite_path,  # noqa: E501
    )


def _recorder(
    record: str | Path | None,
) -> tuple[Callable[[dict[str, Any]], None] | None, Any]:  # noqa: E501
    if record is None:
        return None, None
    path = Path(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w")
    lock = threading.Lock()

    def write(row: dict[str, Any]) -> None:
        with lock:
            handle.write(json.dumps(row, default=json_default) + "\n")

    return write, handle


def _finish(  # noqa: PLR0913, PLR0917 (assembles one report)
    summaries: list[dict[str, Any]],
    variant_rows: list[list[dict[str, Any]]],
    case_ids: list[str],
    trials: int,
    mode: str,
    model_name: str,
    when: datetime,
    out: str | Path | None,
    nest: bool,
    suite_path: str | Path | None,
) -> Report:
    primary = _primary(summaries)
    cases = [row for rows in variant_rows for row in rows]
    comparisons = []
    if len(summaries) >= 2:
        if summaries[1].get("pending"):
            comparisons.append(
                {
                    "a": summaries[0]["label"],
                    "b": summaries[1]["label"],
                    "pending": True,
                }
            )
        else:
            comparisons.append(
                _compare(
                    summaries[0],
                    summaries[1],
                    variant_rows[0],
                    variant_rows[1],
                    case_ids,
                    trials,
                )
            )  # noqa: E501
    digest = Path(suite_path or _SUITE_PATH)
    report = Report(
        accuracy=primary["accuracy"],
        accuracy_ci=tuple(primary["accuracy_ci"]),
        false_ships=primary["false_ships"],
        false_blocks=primary["false_blocks"],
        by_flaw={flaw: stats["acc"] for flaw, stats in primary["by_flaw"].items()},
        cases=cases,
        models=[
            {"name": item["label"], "accuracy": item["accuracy"]} for item in summaries
        ],  # noqa: E501
        published_failures=_failures(cases),
        variants=summaries,
        comparisons=comparisons,
        run_id=_run_id(model_name, when),
        git_sha=_git_sha(),
        suite_sha256=_sha256(digest) if digest.exists() else "unknown",
        prompt_sha256=_prompt_sha(),
        mode=mode,
        trials=trials,
        temperature=None,
        generated_at=when.isoformat(),
    )
    if out is not None:
        report.write(out, nest=nest)
    return report


def check_regression(
    report: Report, baseline: dict[str, Any], mode: str = "replay"
) -> list[str]:  # noqa: E501
    """Replay must match exactly. Live fails only when the interval excludes the baseline."""  # noqa: E501
    if mode == "live":
        return _live_gate(report, baseline)
    return _replay_gate(report, baseline)


def _replay_gate(report: Report, baseline: dict[str, Any]) -> list[str]:
    expected = baseline.get("replay_expected", baseline)
    violations = []
    if not math.isclose(report.accuracy, float(expected["accuracy"]), abs_tol=1e-9):
        violations.append(
            f"accuracy {report.accuracy:.6f} != replay_expected {expected['accuracy']}"
        )  # noqa: E501
    for flaw, score in expected["by_flaw"].items():
        target = score["acc"] if isinstance(score, dict) else float(score)
        current = float(report.by_flaw.get(flaw, 0.0))
        if not math.isclose(current, float(target), abs_tol=1e-9):
            violations.append(f"{flaw}: {current:.6f} != replay_expected {target}")
    return violations


def _live_gate(report: Report, baseline: dict[str, Any]) -> list[str]:
    """
    Live regression gate (tools variant, all trials, case-clustered):

    - Accuracy: Wilson upper bound on trial-0 accuracy vs baseline point.
    - False ships / run failures: case-bootstrap lower bound vs baseline rate.
      Crashes and format failures count as regressions — the gate must not
      reward a broken harness for fewer false ships.
    """
    violations = []
    high = float(report.accuracy_ci[1])
    if high < float(baseline["accuracy"]):
        violations.append(
            f"accuracy CI high {high:.3f} < baseline {float(baseline['accuracy']):.3f}"
        )
    variant = baseline.get("primary_variant") or _tools_label(report.cases)
    rows = [row for row in report.cases if variant is None or row.get("variant") == variant]
    ship_reject, ship_rate, ship_lower, _ship_upper = cluster_rate_regression(
        rows, float(baseline["false_ships"]), false_ship_row, seed=42
    )
    if ship_reject:
        base = float(baseline["false_ships"])
        violations.append(
            f"false_ships {ship_rate:.3f} (bootstrap lower {ship_lower:.3f}) "
            f"> baseline {base:.3f}"
        )
    failure_base = float(baseline.get("run_failure_rate", 0.048484848484848485))
    fail_reject, fail_rate, fail_lower, _fail_upper = cluster_rate_regression(
        rows, failure_base, run_failure_row, seed=43
    )
    if fail_reject:
        violations.append(
            f"run_failures {fail_rate:.3f} (bootstrap lower {fail_lower:.3f}) "
            f"> baseline {failure_base:.3f}"
        )
    if violations:
        violations.extend(_paired_note(report, baseline))
    return violations


def _paired_note(report: Report, baseline: dict[str, Any]) -> list[str]:
    variant = baseline.get("primary_variant") or _tools_label(report.cases)
    left = _align(report.cases, variant)
    right = _align(baseline.get("cases", []), variant)
    shared = sorted(set(left) & set(right))
    if len(shared) < 2:
        return []
    result = mcnemar(
        [left[case_id] for case_id in shared], [right[case_id] for case_id in shared]
    )  # noqa: E501
    if result.p_value < 0.05 and report.accuracy < float(baseline["accuracy"]):
        return [
            f"McNemar p={result.p_value:.3g} ({result.method}), b={result.b}, c={result.c}"
        ]  # noqa: E501
    return []


def _tools_label(cases: list[dict[str, Any]]) -> str | None:
    for case in cases:
        label = case.get("variant")
        if isinstance(label, str) and label.endswith(_TOOLS_SUFFIX):
            return label
    return None


def compare_cases(
    a_cases: list[dict[str, Any]], b_cases: list[dict[str, Any]]
) -> dict[str, Any]:
    """McNemar plus a Newcombe interval on trial-0 rows paired by case id.

    The gate fails when the exact or continuity-corrected test is below 0.05.
    """
    from peer_agent.scoring import newcombe_paired

    left = _align(a_cases, _tools_label(a_cases))
    right = _align(b_cases, _tools_label(b_cases))
    shared = sorted(set(left) & set(right))
    a_flags = [left[case_id] for case_id in shared]
    b_flags = [right[case_id] for case_id in shared]
    result = mcnemar(a_flags, b_flags)
    low, high = newcombe_paired(a_flags, b_flags) if shared else (0.0, 0.0)
    return {
        "n": len(shared),
        "b_only_a_right": result.b,
        "c_only_b_right": result.c,
        "p_value": result.p_value,
        "method": result.method,
        "delta": (sum(a_flags) - sum(b_flags)) / len(shared) if shared else 0.0,
        "delta_ci": [low, high],
        "passed": result.p_value >= 0.05,
    }


def _align(cases: list[dict[str, Any]], variant: str | None) -> dict[str, bool]:
    aligned = {}
    for case in cases:
        if int(case.get("trial", 0)) != 0:
            continue
        if variant is not None and case.get("variant") not in (None, variant):
            continue
        aligned[case["id"]] = bool(case["correct"])
    return aligned


def promote_baseline(
    results: str | Path, dest: str | Path = "evals/baseline.json"
) -> dict[str, Any]:  # noqa: E501
    """Rewrite baseline.json from a results.json. replay_expected is the harness score."""
    payload = json.loads(Path(results).read_text())
    primary = _primary(payload["variants"])
    by_flaw = {flaw: stats["acc"] for flaw, stats in primary["by_flaw"].items()}
    if payload.get("mode") == "live":
        harness = run_suite(load_suite(), replay="fixtures/recorded.jsonl", trials=1)
        replay_expected = {"accuracy": harness.accuracy, "by_flaw": harness.by_flaw}
    else:
        replay_expected = {"accuracy": primary["accuracy"], "by_flaw": by_flaw}
    primary_rows = [
        row for row in payload.get("cases", []) if row.get("variant") == primary["label"]
    ]
    run_failures = sum(
        1 for row in primary_rows if row.get("error") in ("format_fail", "crashed")
    )
    baseline = {
        "schema_version": 2,
        "source_run": payload["run_id"],
        "model": primary["model"],
        "trials": payload["trials"],
        "accuracy": primary["accuracy"],
        "accuracy_ci": primary["accuracy_ci"],
        "pass_hat_k": primary["pass_hat_k"],
        "false_ships": primary["false_ships"],
        "run_failure_rate": run_failures / max(len(primary_rows), 1),
        "by_flaw": by_flaw,
        "replay_expected": replay_expected,
        "primary_variant": primary["label"],
    }
    Path(dest).write_text(json.dumps(baseline, indent=2) + "\n")
    failures = Path(dest).with_name("published_failures.json")
    failures.write_text(json.dumps(payload["published_failures"], indent=2) + "\n")
    return baseline
