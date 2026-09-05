from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from peer_agent.agent import Agent, Trajectory, configure_logfire
from peer_agent.sim import Case
from peer_agent.tools import default_tools
from peer_agent.types import Verdict

_DEFAULT_SAMPLE = "fixtures/sample.parquet"


class _ScriptedFakeLLM:
    """
    An offline stand-in for CI: `PEER_FAKE_LLM=1` runs `peer review` without a
    network or model, so `test_the_cli_prints_a_verdict` can exercise the real
    CLI subprocess end to end. Real runs never set this and go through
    Agent.from_env() instead.
    """

    def __init__(self) -> None:
        self._called = False

    def __call__(self, messages: Any, tools: Any) -> Any:
        del messages, tools
        if not self._called:
            self._called = True
            return [("check_srm", {})]
        return "VERDICT: INVALID. The traffic split doesn't match the assigned ratio."


def _print_trajectory(traj: Trajectory) -> None:
    for call in traj.calls:
        result = repr(call.result)
        if len(result) > 200:
            result = result[:200] + "…"
        print(f"[tool] {call.name}({call.args}) -> {result}")
    if traj.verdict is not None:
        print(f"\nVERDICT: {traj.verdict.name}")
    elif traj.spec is not None:
        print(f"\nSPEC: {traj.spec}")
    print(traj.answer)


def _trajectory_markdown(title: str, traj: Trajectory) -> str:
    lines = [f"# {title}", ""]
    if traj.verdict is not None:
        lines += [f"**Verdict: {traj.verdict.name}**", ""]
    elif traj.spec is not None:
        lines += ["**Spec:**", "", "```", str(traj.spec), "```", ""]
    lines += [traj.answer, ""]
    if traj.calls:
        lines += ["## Trajectory", ""]
        lines += [
            f"{i}. `{c.name}({c.args})` → `{c.result!r}`"
            for i, c in enumerate(traj.calls, 1)
        ]
        lines.append("")
    return "\n".join(lines)


def _write_trajectory(out: str | None, title: str, traj: Trajectory) -> None:
    if out is None:
        return
    path = Path(out)
    path.write_text(_trajectory_markdown(title, traj))
    print(f"\nwrote {path}")


def _review(args: argparse.Namespace) -> int:
    df = pd.read_parquet(args.path)
    case = Case(df=df, truth=Verdict.NO_EFFECT)
    if os.environ.get("PEER_FAKE_LLM") == "1":
        agent = Agent(_ScriptedFakeLLM(), tools=default_tools())
    else:
        agent = Agent.from_env()
    traj = agent.review(case)
    _print_trajectory(traj)
    _write_trajectory(args.out, f"Review — {args.path}", traj)
    return 0


def _design(args: argparse.Namespace) -> int:
    traj = Agent.from_env().design(args.brief)
    _print_trajectory(traj)
    _write_trajectory(args.out, f"Design — {args.brief}", traj)
    return 0


def _chat(args: argparse.Namespace) -> int:
    df = pd.read_parquet(args.path)
    Agent.from_env().chat_agent(df).to_cli_sync(prog_name="peer")
    return 0


def _web(args: argparse.Namespace) -> int:
    import uvicorn

    df = pd.read_parquet(args.path)
    app = Agent.from_env().chat_agent(df).to_web()
    print(f"peer web: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _eval(args: argparse.Namespace) -> int:
    from peer_agent.evals import check_regression, load_suite, run_suite

    suite = load_suite()
    report = (
        run_suite(suite, model=args.model, out=args.out)
        if args.model
        else run_suite(suite, replay=args.replay, out=args.out)
    )

    print(f"accuracy: {report.accuracy:.1%}")
    for flaw, score in sorted(report.by_flaw.items()):
        print(f"  {flaw:10s} {score:.1%}")
    print(f"false_ships: {report.false_ships:.1%}  false_blocks: {report.false_blocks:.1%}")
    for m in report.models:
        print(f"  {m['name']}: {m['accuracy']:.1%}")

    if not args.gate:
        return 0
    baseline = json.loads(Path("evals/baseline.json").read_text())
    violations = check_regression(report, baseline)
    if violations:
        print("\nREGRESSION GATE FAILED:")
        for v in violations:
            print(f"  - {v}")
        return 1
    print("\nregression gate: OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="peer")
    sub = parser.add_subparsers(dest="command", required=True)

    review = sub.add_parser("review", help="Review a finished experiment.")
    review.add_argument("path")
    review.add_argument(
        "-o", "--out", help="Write the verdict + trajectory to this markdown file."
    )
    review.set_defaults(fn=_review)

    design = sub.add_parser("design", help="Design a new experiment from a brief.")
    design.add_argument("brief")
    design.add_argument(
        "-o", "--out", help="Write the spec + trajectory to this markdown file."
    )
    design.set_defaults(fn=_design)

    chat = sub.add_parser("chat", help="Interactive terminal chat with the agent.")
    chat.add_argument("path", nargs="?", default=_DEFAULT_SAMPLE)
    chat.set_defaults(fn=_chat)

    web = sub.add_parser("web", help="Interactive browser chat with the agent.")
    web.add_argument("path", nargs="?", default=_DEFAULT_SAMPLE)
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8000)
    web.set_defaults(fn=_web)

    eval_p = sub.add_parser(
        "eval", help="Run the eval suite and print an accuracy summary."
    )
    eval_p.add_argument("--replay", default="fixtures/recorded.jsonl")
    eval_p.add_argument("--model", help="Run against a real model instead of replaying.")
    eval_p.add_argument("--out", help="Directory to write results.json into.")
    eval_p.add_argument(
        "--gate",
        action="store_true",
        help="Exit nonzero if accuracy regresses past evals/baseline.json's tolerance.",
    )
    eval_p.set_defaults(fn=_eval)

    args = parser.parse_args(argv)
    configure_logfire()
    return int(args.fn(args))
