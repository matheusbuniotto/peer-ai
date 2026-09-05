from __future__ import annotations

import asyncio
import dataclasses
import sys
from importlib import resources
from typing import Any

import pandas as pd
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.mcpserver import MCPServer
from mcp.types import TextContent, TextResourceContents

from peer_agent import design, stats
from peer_agent.sandbox import Sandbox
from peer_agent.types import DesignSpec

_PROTOCOL_PATH = resources.files("peer_agent") / "skills" / "review-protocol" / "SKILL.md"

server = MCPServer("peer-agent")


def _asdict(result: Any) -> Any:
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return dataclasses.asdict(result)
    return result


@server.tool()
def check_srm(path: str) -> dict[str, Any]:
    """Check whether the traffic split matches the assigned ratio."""
    return _asdict(stats.check_srm(pd.read_parquet(path)))


@server.tool()
def analyze(path: str, cuped: bool = False, covariate: str | None = None) -> dict[str, Any]:
    """Run the primary conversion-rate test, treatment vs. control."""
    df = pd.read_parquet(path)
    return _asdict(stats.analyze(df, cuped=cuped, covariate=covariate))


@server.tool()
def sequential(path: str, looks: int) -> dict[str, Any]:
    """Bonferroni-corrected look: conservative, transparent, easy to defend."""
    return _asdict(stats.sequential(pd.read_parquet(path), looks))


@server.tool()
def scan_segments(path: str) -> dict[str, Any]:
    """Check each segment for a significant effect or a sign reversal."""
    return _asdict(stats.scan_segments(pd.read_parquet(path)))


@server.tool()
def check_novelty(path: str) -> dict[str, Any]:
    """Check whether an early effect is decaying over time."""
    return _asdict(stats.check_novelty(pd.read_parquet(path)))


@server.tool()
def check_guardrails(path: str, guardrails: tuple[str, ...] = ()) -> dict[str, Any]:
    """Check whether any named guardrail metric regressed."""
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
    """Textbook two-proportion z-test sample size."""
    return _asdict(design.power_analysis(baseline, mde, power=power, alpha=alpha))


@server.tool()
def simulate_design(
    spec: dict[str, Any], lift: float, runs: int, seed: int
) -> dict[str, Any]:
    """Plants `lift` on spec.baseline and re-simulates the design `runs` times."""
    return _asdict(design.simulate_design(DesignSpec(**spec), lift, runs, seed))


@server.tool()
def ask(question: str) -> str:
    """Ask the person requesting the design a clarifying question."""
    del question
    return "No answer available yet; proceed on your best judgement."


@server.resource("peer://protocol")
def protocol() -> str:
    """The review-protocol skill's checklist, so it travels with the tools."""
    return _PROTOCOL_PATH.read_text()


async def _roundtrip(method: str, **kwargs: Any) -> Any:
    params = StdioServerParameters(command=sys.executable, args=["-m", "peer_agent.mcp"])
    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        if method == "tools/list":
            result = await session.list_tools()
            return [t.model_dump() for t in result.tools]
        if method == "tools/call":
            result = await session.call_tool(kwargs["name"], kwargs.get("arguments") or {})
            if result.structured_content is not None:
                return result.structured_content
            content = result.content[0]
            assert isinstance(content, TextContent)
            return content.text
        if method == "resources/read":
            result = await session.read_resource(kwargs["uri"])
            contents = result.contents[0]
            assert isinstance(contents, TextResourceContents)
            return contents.text
        raise ValueError(f"unknown method: {method}")


def stdio_roundtrip(method: str, **kwargs: Any) -> Any:
    """Spawns this module as a subprocess MCP server and talks to it over stdio."""
    return asyncio.run(_roundtrip(method, **kwargs))


if __name__ == "__main__":
    server.run()
