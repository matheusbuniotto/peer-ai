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
    report = run_suite(
        suite,
        replay=None if args.model else args.replay,
        model=args.model,
        out=args.out,
        trials=args.trials,
        variants=_variants(args.variants),
        concurrency=args.concurrency,
        record=args.record,
        nest=bool(args.out),
    )
    _print_report(report)
    if not args.gate:
        return 0
    baseline = _baseline_for_gate()
    violations = check_regression(report, baseline, mode=args.gate)
    if violations:
        print("\nREGRESSION GATE FAILED:")
        for item in violations:
            print(f"  - {item}")
        return 1
    print("\nregression gate: OK")
    return 0


def _variants(raw: str | None) -> tuple[str, ...] | None:
    if not raw:
        return None
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _print_report(report: Any) -> None:
    print(f"accuracy: {report.accuracy:.1%}")
    for flaw, score in sorted(report.by_flaw.items()):
        print(f"  {flaw:10s} {score:.1%}")
    print(f"false_ships: {report.false_ships:.1%}  false_blocks: {report.false_blocks:.1%}")
    for model in report.models:
        print(f"  {model['name']}: {model['accuracy']:.1%}")


def _baseline_for_gate() -> dict[str, Any]:
    baseline = json.loads(Path("evals/baseline.json").read_text())
    if baseline.get("cases") or not baseline.get("source_run"):
        return baseline
    path = Path("results") / baseline["source_run"] / "results.json"
    if not path.exists():
        return baseline
    payload = json.loads(path.read_text())
    tools = next((item["label"] for item in payload["variants"] if item.get("tools")), None)
    return {**baseline, "cases": payload["cases"], "primary_variant": tools}


def _promote(args: argparse.Namespace) -> int:
    from peer_agent.evals import promote_baseline

    baseline = promote_baseline(args.results, args.dest)
    print(f"promoted {baseline['source_run']} -> {args.dest or 'evals/baseline.json'}")
    return 0


def _report(args: argparse.Namespace) -> int:
    from peer_agent.report import (
        END,
        START,
        load_results,
        render_errors,
        render_table,
        splice_readme,
    )

    path = _results_path(args.results)
    payload = load_results(path)
    if args.errors:
        text = render_errors(payload)
        print(text)
        dest = path.parent / "errors.md"
        dest.write_text(text)
        print(f"wrote {dest}")
        return 0
    table = render_table(payload, source=str(path))
    print(table)
    if args.write_readme:
        readme = Path("README.md")
        readme.write_text(splice_readme(readme.read_text(), table))
        print(f"updated {readme} between {START} and {END}")
    return 0


def _results_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    pointer = Path("results/LATEST")
    if not pointer.exists():
        raise SystemExit("pass --results, or create results/LATEST")
    return Path("results") / pointer.read_text().strip() / "results.json"


def _demo(args: argparse.Namespace) -> int:
    from peer_agent.demo import run_demo

    return run_demo(live=args.live)


def _compare(args: argparse.Namespace) -> int:
    from peer_agent.evals import compare_cases
    from peer_agent.report import load_results

    left = load_results(_resolve_run(args.run_a))
    right = load_results(_resolve_run(args.run_b))
    result = compare_cases(left["cases"], right["cases"])
    low, high = result["delta_ci"]
    print(
        f"McNemar b={result['b_only_a_right']} c={result['c_only_b_right']} "
        f"p={result['p_value']:.3g} ({result['method']})"
    )
    print(f"delta {result['delta']:+.3f} (Newcombe 95% CI {low:+.3f}, {high:+.3f})")
    return 0 if result["passed"] else 1


def _resolve_run(name: str) -> Path:
    path = Path(name)
    if path.name == "LATEST" and path.is_file():
        return path.parent / path.read_text().strip() / "results.json"
    if path.is_dir():
        return path / "results.json"
    if path.suffix == ".json":
        return path
    return Path("results") / name / "results.json"


def _export(args: argparse.Namespace) -> int:
    from peer_agent.export_braintrust import export_results

    return export_results(args.results, args.project)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="peer")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_run_commands(sub)
    _add_eval_commands(sub)
    return parser


def _add_run_commands(sub: Any) -> None:
    review = sub.add_parser("review", help="Review a finished experiment.")
    review.add_argument("path")
    review.add_argument("-o", "--out", help="Write the verdict and trajectory as markdown.")
    review.set_defaults(fn=_review)

    design = sub.add_parser("design", help="Design a new experiment from a brief.")
    design.add_argument("brief")
    design.add_argument("-o", "--out", help="Write the spec and trajectory as markdown.")
    design.set_defaults(fn=_design)

    chat = sub.add_parser("chat", help="Interactive terminal chat with the agent.")
    chat.add_argument("path", nargs="?", default=_DEFAULT_SAMPLE)
    chat.set_defaults(fn=_chat)

    web = sub.add_parser("web", help="Interactive browser chat with the agent.")
    web.add_argument("path", nargs="?", default=_DEFAULT_SAMPLE)
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8000)
    web.set_defaults(fn=_web)

    demo = sub.add_parser("demo", help="Show a replay, the toolbox, and the harness gate.")
    demo.add_argument("--live", action="store_true")
    demo.set_defaults(fn=_demo)


def _add_eval_commands(sub: Any) -> None:
    eval_p = sub.add_parser("eval", help="Run the suite and print accuracy.")
    eval_p.add_argument("--replay", default="fixtures/recorded.jsonl")
    eval_p.add_argument("--model", help="Run against a real model instead of replaying.")
    eval_p.add_argument("--trials", type=int, default=1)
    eval_p.add_argument("--variants", help="Comma-separated: tools,no-tools.")
    eval_p.add_argument("--concurrency", type=int, default=1)
    eval_p.add_argument("--record", help="Write live turns as a replay JSONL.")
    eval_p.add_argument("--out", help="Directory for results.json. Nested by run id.")
    eval_p.add_argument(
        "--gate",
        nargs="?",
        const="replay",
        choices=("replay", "live"),
        help="replay: exact harness match. live: CI must exclude the baseline to fail.",
    )
    eval_p.set_defaults(fn=_eval)
    _add_report_commands(sub)


def _add_report_commands(sub: Any) -> None:
    baseline = sub.add_parser("baseline", help="Promote a run into the committed baseline.")
    baseline_sub = baseline.add_subparsers(dest="baseline_command", required=True)
    promote = baseline_sub.add_parser("promote")
    promote.add_argument("results")
    promote.add_argument("--dest", default="evals/baseline.json")
    promote.set_defaults(fn=_promote)

    report = sub.add_parser("report", help="Render the committed results table.")
    report.add_argument("--results")
    report.add_argument("--write-readme", action="store_true")
    report.add_argument("--errors", action="store_true")
    report.set_defaults(fn=_report)

    compare = sub.add_parser("compare", help="Paired McNemar test of two runs.")
    compare.add_argument("run_a")
    compare.add_argument("run_b")
    compare.set_defaults(fn=_compare)

    export = sub.add_parser("export", help="Send a run somewhere else.")
    export_sub = export.add_subparsers(dest="export_command", required=True)
    braintrust = export_sub.add_parser("braintrust")
    braintrust.add_argument("results")
    braintrust.add_argument("--project", default="peer-ai")
    braintrust.set_defaults(fn=_export)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    configure_logfire()
    return int(args.fn(args))
