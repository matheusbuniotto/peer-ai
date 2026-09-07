from __future__ import annotations

import asyncio
import dataclasses
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib import resources
from typing import Any

import pandas as pd
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.mcpserver import MCPServer
from mcp.types import TextContent, TextResourceContents

from peer_agent import design, stats
from peer_agent.sandbox import Sandbox
from peer_agent.tools import UNANSWERED
from peer_agent.types import DesignSpec, Guardrail

_PROTOCOL_PATH = resources.files("peer_agent") / "skills" / "review-protocol" / "SKILL.md"

server = MCPServer("peer-agent")


def _asdict(result: Any) -> Any:
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return dataclasses.asdict(result)
    return result


@server.tool()
def check_srm(path: str, ratio: float = 1.0, by: tuple[str, ...] = ()) -> dict[str, Any]:
    """Check the traffic split against the assigned ratio, overall and per stratum."""
    return _asdict(stats.check_srm(pd.read_parquet(path), ratio=ratio, by=by))


@server.tool()
def analyze(
    path: str,
    cuped: bool = False,
    covariate: str | None = None,
    mde: float | None = None,
    unit: str | None = None,
) -> dict[str, Any]:
    """Run the primary conversion-rate test, treatment vs. control."""
    df = pd.read_parquet(path)
    return _asdict(stats.analyze(df, cuped=cuped, covariate=covariate, mde=mde, unit=unit))


@server.tool()
def sequential(path: str, looks: int) -> dict[str, Any]:
    """Bonferroni-corrected look: conservative, transparent, easy to defend."""
    return _asdict(stats.sequential(pd.read_parquet(path), looks))


@server.tool()
def scan_segments(path: str) -> dict[str, Any]:
    """Test every segment under one family-wise error budget, plus heterogeneity."""
    return _asdict(stats.scan_segments(pd.read_parquet(path)))


@server.tool()
def check_novelty(path: str) -> dict[str, Any]:
    """Fit the daily effect against time and test whether the trend is decaying."""
    return _asdict(stats.check_novelty(pd.read_parquet(path)))


@server.tool()
def check_guardrails(
    path: str, guardrails: tuple[Guardrail | str, ...] = ()
) -> dict[str, Any]:
    """Check whether any guardrail could still be moving past its margin."""
    return _asdict(stats.check_guardrails(pd.read_parquet(path), guardrails))


@server.tool()
def run_python(path: str, code: str) -> dict[str, Any]:
    """Run Python analysis code in an isolated container when no built-in tool fits."""
    with Sandbox() as sandbox:
        result = sandbox.run(code, pd.read_parquet(path))
    return _asdict(result)


@server.tool()
def power_analysis(
    baseline: float, mde: float, power: float = 0.8, alpha: float = 0.05
) -> dict[str, Any]:
    """Sample size for a relative `mde` on a `baseline` conversion rate."""
    return _asdict(design.power_analysis(baseline, mde, power=power, alpha=alpha))


@server.tool()
def simulate_design(
    spec: dict[str, Any], lift: float, runs: int, seed: int
) -> dict[str, Any]:
    """Plants `lift` on spec.baseline and re-randomizes the design `runs` times."""
    return _asdict(design.simulate_design(DesignSpec(**spec), lift, runs, seed))


@server.tool()
def ask(question: str) -> str:
    """Ask the person requesting the design a clarifying question."""
    del question
    # Over stdio there is nobody on the other end of this, and telling the model
    # to use its judgement is how a brief with no data becomes a confident spec.
    return UNANSWERED


@server.resource("peer://protocol")
def protocol() -> str:
    """The review-protocol skill's checklist, so it travels with the tools."""
    return _PROTOCOL_PATH.read_text()


def main() -> None:
    """Entry point for the `peer-mcp` script and `peer mcp`. Speaks stdio."""
    server.run()


# --------------------------------------------------------------------------- #
# A real client, for tests that insist the server works over a process boundary
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def _session() -> AsyncIterator[ClientSession]:
    params = StdioServerParameters(command=sys.executable, args=["-m", "peer_agent.mcp"])
    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


async def _list_tools(session: ClientSession, _: dict[str, Any]) -> Any:
    result = await session.list_tools()
    return [tool.model_dump() for tool in result.tools]


async def _call_tool(session: ClientSession, kwargs: dict[str, Any]) -> Any:
    result = await session.call_tool(kwargs["name"], kwargs.get("arguments") or {})
    if result.structured_content is not None:
        return result.structured_content
    content = result.content[0]
    assert isinstance(content, TextContent)
    return content.text


async def _read_resource(session: ClientSession, kwargs: dict[str, Any]) -> Any:
    result = await session.read_resource(kwargs["uri"])
    contents = result.contents[0]
    assert isinstance(contents, TextResourceContents)
    return contents.text


_METHODS: dict[str, Callable[[ClientSession, dict[str, Any]], Awaitable[Any]]] = {
    "tools/list": _list_tools,
    "tools/call": _call_tool,
    "resources/read": _read_resource,
}


def stdio_roundtrip(method: str, **kwargs: Any) -> Any:
    """Spawns this module as a subprocess MCP server and talks to it over stdio."""
    if method not in _METHODS:
        raise ValueError(f"unknown method: {method}")

    async def run() -> Any:
        async with _session() as session:
            return await _METHODS[method](session, kwargs)

    return asyncio.run(run())


if __name__ == "__main__":
    main()
