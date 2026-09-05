# 04 — Tool-registry reconciliation (`analysis_tools`/`design_tools`/`default_tools`, `Case.path`)

Type: AFK
Source: docs/plans/phase-3-plan.md (phase 3)

## What to build

Small, foundational — issues 06 and 08 depend on it. Fixes a real binding
hazard: `agent.py`'s `_bind()` always prepends `case.df` as the first
positional arg to every tool in `review()`'s tool list.
`power_analysis`/`simulate_design`/`ask` don't take a `df` — so they must
never appear in that list, which becomes a real risk once
`test_the_registry_is_complete` wants a single 10-tool union.

Contract:

- `analysis_tools(sandbox=None)` → the 6 stats tools (`bind_df=True`),
  plus `run_python` **iff** `sandbox` is given (`run_python`'s raw
  signature is `_run_python(df, code, *, sandbox)`, built via
  `functools.partial(_run_python, sandbox=sandbox)` at tool-construction
  time — `sandbox` closes over there, `df` stays the first positional
  param so `agent.py`'s existing `_bind(tool, calls, case.df)` keeps
  working unmodified). This is what `Agent.from_env()` uses for `review()`.
- `design_tools()` → unchanged (`power_analysis`, `simulate_design`,
  `ask` — no `df`, only ever used inside `design()`, which never binds
  `case.df`).
- `default_tools(sandbox)` → `analysis_tools(sandbox) + design_tools()`,
  the full 10-tool union `test_the_registry_is_complete` checks against
  and what issue 09's MCP server lists — **not** fed into `Agent()`
  directly.

Also add `sim.py`'s `Case.path`: a lazy-write temp-parquet property
(`object.__setattr__` on the frozen dataclass) so a `Case` can hand a file
path to out-of-process consumers (needed by issue 09's MCP roundtrip
test, `srm.path`).

## Acceptance criteria

- [ ] `default_tools(sandbox)` returns exactly the 10 names
      `test_the_registry_is_complete` expects
      (`check_srm, analyze, sequential, scan_segments, check_novelty,
      check_guardrails, power_analysis, simulate_design, run_python, ask`)
- [ ] `analysis_tools(sandbox)` and `design_tools()` remain independently
      usable by `review()`/`design()` with no `df`-binding mismatch
- [ ] `Case.path` lazily writes to a temp parquet and returns a stable
      path on repeat access; `>=` column-set checks in phase-1 tests
      still hold (i.e. `Case`'s public shape is otherwise unchanged)
- [ ] `uv run pytest tests/test_phase_1_tracer.py tests/test_phase_2_tracer.py -m "not live and not docker"`
      still fully green
- [ ] `uv run prek` and `uv run ty check` clean

## Blocked by

None - can start immediately
