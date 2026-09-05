# 09 — MCP server via FastMCP

Type: AFK
Source: docs/plans/phase-3-plan.md (phase 3)

## What to build

"How someone else tries it in thirty seconds." Build `mcp.py` on
`mcp.server.fastmcp.FastMCP` (confirmed available — `mcp`/`fastmcp` come
in transitively via `pydantic-ai-slim[mcp]`) instead of hand-rolling
JSON-RPC parsing:

```python
from mcp.server.fastmcp import FastMCP
server = FastMCP("peer-agent")

@server.tool()
def check_srm(path: str) -> dict: ...  # loads df from path, calls stats.check_srm

if __name__ == "__main__":
    server.run()  # stdio by default
```

Register one `@server.tool()` per entry in `default_tools(sandbox)`
(issue 04's 10-tool union). Unlike the in-process agent tools, MCP tool
calls cross a process boundary — stats tools take `{"path": ...}` and
load the dataframe themselves (`pd.read_parquet(path)`) rather than
receiving a bound `df`, which is why `Case.path` (issue 04) exists.
Register `peer://protocol` as a resource serving the review-protocol
skill's markdown (the "validate first" checklist).

`stdio_roundtrip(method, **kwargs)` (the test helper the tracer file
imports from `peer_agent.mcp`) spawns the server as a subprocess and
talks to it using a real MCP client — `fastmcp.Client` or the `mcp` SDK's
client, not a hand-rolled stdio reader/writer.

## Acceptance criteria

- [ ] `uv run pytest tests/test_phase_3_tracer.py -m docker -k "server_lists\|same_answer\|protocol_is_published"`
      fully green
- [ ] `tools/list` returns at least `check_srm`, `analyze`,
      `simulate_design`
- [ ] `tools/call` for `check_srm` with `{"path": ...}` matches
      `stats.check_srm(df).mismatch` exactly
- [ ] `resources/read` for `peer://protocol` returns text containing
      "validate"
- [ ] `uv run prek` and `uv run ty check` clean

## Blocked by

- 08 — Docker sandbox
- 04 — Tool-registry reconciliation
