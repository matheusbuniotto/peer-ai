from __future__ import annotations

import argparse
import os
from typing import Any

import pandas as pd

from peer_agent.agent import Agent
from peer_agent.sim import Case
from peer_agent.tools import default_tools
from peer_agent.types import Verdict


class _ScriptedFakeLLM:
    def __init__(self) -> None:
        self._called = False

    def __call__(self, messages: Any, tools: Any) -> Any:
        del messages, tools
        if not self._called:
            self._called = True
            return [("check_srm", {})]
        return "VERDICT: INVALID. The traffic split doesn't match the assigned ratio."


def _build_llm() -> Any:
    if os.environ.get("PEER_FAKE_LLM") == "1":
        return _ScriptedFakeLLM()
    raise NotImplementedError("phase 2: real model wiring")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="peer")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("review").add_argument("path")
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.path)
    case = Case(df=df, truth=Verdict.NO_EFFECT)
    traj = Agent(_build_llm(), tools=default_tools()).review(case)

    verdict = traj.verdict.name if traj.verdict else "UNKNOWN"
    print(f"VERDICT: {verdict}\n{traj.answer}")
    return 0
