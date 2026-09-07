from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
from importlib import resources
from pathlib import Path
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


_READERS = {".parquet": pd.read_parquet, ".csv": pd.read_csv, ".json": pd.read_json}


def _load_df(path: str | None = None, data: str | None = None) -> pd.DataFrame:
    """
    A local `path` (parquet/csv/json, by suffix), or `data` — JSON rows another
    tool already handed the caller, with no shared filesystem to stage a file on.
    Exactly one is expected.
    """
    if path is not None and data is not None:
        raise ValueError("give exactly one of `path` or `data`")
    if data is not None:
        return pd.DataFrame(json.loads(data))
    if path is None:
        raise ValueError("give exactly one of `path` or `data`")
    suffix = Path(path).suffix
    try:
        reader = _READERS[suffix]
    except KeyError:
        raise ValueError(f"unsupported file type: {suffix or path!r}") from None
    return reader(path)


@server.tool()
def check_srm(
    path: str | None = None,
    data: str | None = None,
    ratio: float = 1.0,
    by: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Check the split against the ratio the design asked for, overall and by stratum."""
    return _asdict(stats.check_srm(_load_df(path, data), ratio=ratio, by=by))


@server.tool()
def analyze(
    path: str | None = None,
    data: str | None = None,
    cuped: bool = False,
    covariate: str | None = None,
    unit: str | None = None,
) -> dict[str, Any]:
    """Run the primary conversion-rate test, treatment vs. control."""
    df = _load_df(path, data)
    return _asdict(stats.analyze(df, cuped=cuped, covariate=covariate, unit=unit))


@server.tool()
def sequential(
    path: str | None = None, data: str | None = None, *, looks: int
) -> dict[str, Any]:
    """Bonferroni-corrected look: conservative, transparent, easy to defend."""
    return _asdict(stats.sequential(_load_df(path, data), looks))


@server.tool()
def scan_segments(path: str | None = None, data: str | None = None) -> dict[str, Any]:
    """Check each segment for a significant effect or a sign reversal."""
    return _asdict(stats.scan_segments(_load_df(path, data)))


@server.tool()
def check_novelty(path: str | None = None, data: str | None = None) -> dict[str, Any]:
    """Check whether an early effect is decaying over time."""
    return _asdict(stats.check_novelty(_load_df(path, data)))


@server.tool()
def check_guardrails(
    path: str | None = None,
    data: str | None = None,
    guardrails: tuple[Guardrail | str, ...] = (),
) -> dict[str, Any]:
    """Check whether any guardrail could still be moving past its margin."""
    return _asdict(stats.check_guardrails(_load_df(path, data), guardrails))


@server.tool()
def run_python(
    path: str | None = None, data: str | None = None, *, code: str
) -> dict[str, Any]:
    """Run Python analysis code in an isolated container when no built-in tool fits."""
    with Sandbox() as sandbox:
        result = sandbox.run(code, _load_df(path, data))
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
    # Over stdio there is nobody on the other end of this, and telling the model
    # to use its judgement is how a brief with no data becomes a confident spec.
    return UNANSWERED


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


def main() -> None:
    """Entry point for the `peer-mcp` script. Speaks stdio."""
    server.run()


if __name__ == "__main__":
    main()
