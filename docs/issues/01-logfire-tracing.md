# 01 — Logfire tracing wired into the agent loop

Type: AFK
Source: docs/plans/phase-3-plan.md (phase 3)

## What to build

Every `Agent.review()` / `Agent.design()` run (and the interactive chat UI
from issue 02) should emit a trace — every tool call, its args, its
result, timing, and any retries — visible in a real trace viewer instead
of only in `Trajectory.calls` after the fact.

Use Logfire, already transitively installed via
`pydantic-ai-slim[logfire]` (confirmed in `uv.lock`) — do not build a
custom tracing/logging layer. Two things:

1. `logfire.configure(send_to_logfire='if-token-present')` once, at
   process start (`cli.py`'s `main()`, and wherever `Agent.from_env()` is
   constructed). This documented option means the app runs with zero
   account/token needed locally, and automatically starts sending to a
   hosted Logfire project the moment `LOGFIRE_TOKEN` is set in `.env` —
   no code branch needed for "local vs. hosted."
2. Instrument the underlying `pydantic_ai.Agent` instances built inside
   `review()`/`design()`/chat (`PydanticAgent(..., instrument=True)`, or
   `logfire.instrument_pydantic_ai()` — confirm which is current for the
   installed pydantic-ai version) so tool calls and model turns are
   captured as spans automatically.

Add `LOGFIRE_TOKEN=` (empty, optional) to `.env.example` next to the
existing `LLM_*` vars, with a comment that it's optional.

Also write a local JSONL log, independent of Logfire entirely — grep-able
even with no network, no token, and no browser open. One line per
completed `review()`/`design()` call: `{"ts", "kind": "review"|"design",
"case_id"?, "calls": [{"name","args","result"}...], "verdict"?|"spec"?,
"answer"}` — essentially `Trajectory` serialized, appended to
`logs/runs.jsonl` (create the dir if missing). This is deliberately
separate from Logfire's spans: Logfire is for watching a run live /
timing/retries, the JSONL file is for "what did the last N runs actually
decide," diffable and scriptable without any UI. Path configurable via
`PEER_LOG_PATH` env var (default `logs/runs.jsonl`); add `logs/` to
`.gitignore`.

## Acceptance criteria

- [ ] `logfire.configure(send_to_logfire='if-token-present')` called once
      at process start, no `LOGFIRE_TOKEN` required to run locally
- [ ] Running `uv run python -m peer_agent review fixtures/sample.parquet`
      (once issue 03 exists) or any existing `live_agent.review(case)`
      call produces a local trace (console span output at minimum;
      Logfire's local dev UI if reachable without a token)
- [ ] Setting `LOGFIRE_TOKEN` in `.env` sends the same trace to a hosted
      Logfire project with no other code change
- [ ] `.env.example` documents `LOGFIRE_TOKEN` as optional
- [ ] Every `review()`/`design()` call appends one JSON line to
      `logs/runs.jsonl` (path overridable via `PEER_LOG_PATH`) containing
      the full tool-call trajectory and final verdict/spec/answer —
      verify by running twice and `wc -l`/`jq` on the file
- [ ] `logs/` is gitignored
- [ ] `uv run pytest tests/test_phase_1_tracer.py tests/test_phase_2_tracer.py -m "not live and not docker"`
      still fully green (instrumentation must not change `Trajectory` or
      `ToolCall` behavior)
- [ ] `uv run prek` and `uv run ty check` clean

## Blocked by

None - can start immediately
