# 07 — Static `web/` report page

Type: AFK
Source: docs/plans/phase-3-plan.md (phase 3)

## What to build

A static page reading `results.json` — this is the eval-report artifact
(distinct from issue 02's `peer web` interactive chat, which is for
poking at the live agent, not viewing the graded suite). Plain HTML +
vanilla JS (`web/index.html`, `web/app.js`), **no framework, no build
step** — this is a Python project; a JS toolchain would be pure overhead
for one static page, and the tracer file is explicit: "the page is
static and has no backend... if it imports anything from src, it will
break the moment it's deployed."

Layout: summary header (accuracy, by-flaw bars, models compared
including the "no tools" baseline), a case table (id, flaw, ✓/✗, truth
vs. verdict), click a row to expand its `trajectory` turn-by-turn (tool
name, args, result) — this expand-to-inspect view is the debuggable part
a PM would actually use to understand *why* a case passed or failed.

## Acceptance criteria

- [ ] `uv run python -m http.server -d web 8000` serves a working page
      against a real `results.json` (from issue 05/06) — summary
      numbers, case table, and per-case trajectory expansion all render
- [ ] `uv run pytest tests/test_phase_3_tracer.py -k page_reads_nothing_but_results_json -m "not live and not docker"`
      green
- [ ] No `.jsx` files, no imports from `src`/`peer_agent` anywhere under
      `web/`
- [ ] `uv run prek` clean (note: `ty check` doesn't apply to `web/`, it's
      not Python)

## Blocked by

- 05 — Eval harness on pydantic_evals
