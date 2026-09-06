from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from types import FunctionType
from typing import Any, cast

import pandas as pd
from pydantic_ai import Tool as PaiTool

from peer_agent import design, stats

_DUMMY_DF = pd.DataFrame()


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    schema: dict[str, object]
    fn: Callable[..., object]


def _from_function(fn: FunctionType, *, bind_df: bool = False) -> Tool:
    """Derive a Tool's name/description/schema from fn's own type hints and docstring."""
    target: Callable[..., object] = fn
    if bind_df:
        bound: Any = functools.partial(fn, _DUMMY_DF)
        bound.__name__ = fn.__name__
        bound.__qualname__ = fn.__qualname__
        bound.__doc__ = fn.__doc__
        target = bound
    tool_def = PaiTool(target, takes_ctx=False).tool_def
    return Tool(
        name=tool_def.name,
        description=tool_def.description or "",
        schema=tool_def.parameters_json_schema,
        fn=fn,
    )


def _stats_tools() -> list[Tool]:
    return [
        _from_function(stats.check_srm, bind_df=True),
        _from_function(stats.analyze, bind_df=True),
    ]


def default_tools(sandbox: object | None = None) -> list[Tool]:
    """
    With no sandbox, the phase-1 registry (check_srm, analyze) — kept exactly as-is
    since that contract is frozen. Given a sandbox, the full 10-tool union (analysis
    + design), the shape the MCP server lists and the tool-registry test checks.
    """
    if sandbox is None:
        return _stats_tools()
    return analysis_tools(sandbox) + design_tools()


def analysis_tools(sandbox: object | None = None) -> list[Tool]:
    """default_tools() plus the calibrated phase-2 checks, for a real review loop."""
    tools = _stats_tools() + [
        _from_function(stats.sequential, bind_df=True),
        _from_function(stats.scan_segments, bind_df=True),
        _from_function(stats.check_novelty, bind_df=True),
        _from_function(stats.check_guardrails, bind_df=True),
    ]
    if sandbox is not None:
        tools.append(_from_function(_make_run_python(sandbox), bind_df=True))
    return tools


def design_tools(answerer: Callable[[str], str] | None = None) -> list[Tool]:
    """Given an `answerer`, `ask` reaches a real person; without one it says so."""
    return [
        _from_function(design.power_analysis),
        _from_function(design.simulate_design),
        _from_function(_make_ask(answerer)),
    ]


UNANSWERED = (
    "Nobody is available to answer that. Do not invent an answer and do not "
    "assume one: put the question in your write-up as an open decision, and "
    "mark anything that depends on it as provisional."
)


def _make_ask(answerer: Callable[[str], str] | None) -> FunctionType:
    """
    The old canned reply told the model to 'proceed on your best judgement',
    which is the opposite of what asking is for — it taught the model to guess
    the answer to the question it had just been told to escalate.
    """

    def ask(question: str) -> str:
        """Ask the person requesting the design a clarifying question."""
        return answerer(question) if answerer else UNANSWERED

    return cast(FunctionType, ask)


def _make_run_python(sandbox: Any) -> FunctionType:
    """
    A real closure (not functools.partial) so it has genuine __name__/__annotations__ —
    agent.py's _bind() wraps every Tool.fn in one more partial to bind df/case data,
    and pydantic-ai's schema introspection only ever unwraps a single partial layer.
    A partial-of-a-partial would break that; a closure-of-a-partial doesn't.
    """

    def run_python(df: pd.DataFrame, code: str) -> Any:
        """Run Python analysis code in an isolated container when no built-in tool fits."""
        return sandbox.run(code, df)

    return cast(FunctionType, run_python)
