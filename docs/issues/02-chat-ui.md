# 02 — `peer chat` + `peer web`: interactive agent via built-in UIs

Type: AFK
Source: docs/plans/phase-3-plan.md (phase 3)

## What to build

The fastest path to "I can run this and poke at it like a PM," using
pydantic-ai's built-in interfaces instead of hand-building a chat frontend
(confirmed via docs: `Agent.to_cli_sync()` — terminal chat, renders tool
calls natively; `Agent.to_web()` — browser chat, ASGI app, explicitly
documented as built for local development/debugging).

Add two `cli.py` subcommands:

- `peer chat` → builds a `pydantic_ai.Agent` from `Agent.from_env()`'s
  model + `analysis_tools()` + `_PROMPT` + the review-protocol skill (the
  same construction `review()` does internally, minus the
  `Trajectory`-capturing wrapper), then calls `.to_cli_sync()`.
- `peer web` → same agent construction, `.to_web()`, served via
  `uvicorn` on a local port (print the URL on start).

This is a raw chat surface, not `review()`/`design()`'s structured
verdict/spec output — a PM types "review this experiment" or pastes a
brief and talks to it turn by turn, same as talking to any other
pydantic-ai agent. Tool calls show up in-line because that's how these
built-in UIs already render them — no custom trajectory printer needed
here (that's issue 03, for the scripted/non-interactive path).

## Acceptance criteria

- [ ] `uv run python -m peer_agent chat` opens an interactive terminal
      session against a real model (needs `.env` populated); asking it to
      review a sample experiment triggers visible tool calls
- [ ] `uv run python -m peer_agent web` starts a local ASGI server and
      printed URL opens a working chat UI in a browser
- [ ] Both share the same tool registry (`analysis_tools()`) and system
      prompt/skills as `review()`, so behavior is representative of the
      "real" agent, not a stripped-down demo
- [ ] No new custom UI code (HTML/JS) — verify by diff: this issue adds
      only `cli.py` wiring, no new frontend files
- [ ] `uv run prek` and `uv run ty check` clean

## Blocked by

None - can start immediately
