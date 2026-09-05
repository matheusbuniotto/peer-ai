# 05 — Eval suite + replay fixtures → `results.json`, on `pydantic_evals`

Type: AFK
Source: docs/plans/phase-3-plan.md (phase 3)

## What to build

`evals/`, `fixtures/` are currently empty. Build the harness on top of
`pydantic_evals` (already installed via `pydantic-ai-slim[evals]` — do
not hand-roll a scoring/report loop). `pydantic_evals` gives you
`Dataset`/`Case`/`Evaluator`/`evaluate_sync` and a pretty-printed report
table for free:

```python
from pydantic_evals import Case, Dataset
dataset = Dataset(name="peer-eval", cases=[...])
report = dataset.evaluate_sync(task_fn)
```

Write `evals.py` as a thin adapter: `load_suite()` reads
`evals/suite.yaml` into your own `EvalCase` records (id, flaw, seed,
truth — needed because `test_every_case_in_the_suite_has_a_known_answer`
checks `case.truth`/`case.seed` directly, and
`test_the_suite_covers_all_five_flaws` checks `case.flaw`), each building
a `pydantic_evals.Case` whose task function drives
`Agent(fake_llm, tools=analysis_tools()).review(case.build())`.
`run_suite(...)` calls `dataset.evaluate_sync(...)` and then reshapes
`EvaluationReport` (`.cases`, `.failures`) into your own `Report`
dataclass matching the tracer file's exact `results.json` schema —
`accuracy`, `false_ships`, `false_blocks`, `by_flaw`, `cases`, `models`,
`published_failures`, `generated_at`.

**Key finding from reading the tracer file closely:** every offline
harness test (`test_a_replayed_run_is_byte_identical`,
`test_the_baseline_gate_catches_a_regression`, all `results.json` shape
tests) only ever calls `run_suite(..., replay=REPLAY)`. Only
`test_the_real_grade`/`test_the_toolbox_earns_its_existence` are
`@pytest.mark.live @pytest.mark.slow`. So `fixtures/recorded.jsonl` (JSON
Lines, one record per `(case_id, model_name)`, scripted turns in the same
shape `conftest.py`'s `FakeLLM.turns` already uses) and
`evals/baseline.json` are authored, not captured from a real model — this
issue's author controls accuracy, `by_flaw` scores, and which case IDs
are "published failures." Script two model rows per case —
`"<model> (tools)"` mostly-correct, `"<model> (no tools)"` deliberately
worse, straight to a verdict — satisfying
`test_the_no_tools_baseline_is_in_the_report`. Deliberately script 2–3
case IDs to end wrong and record those same IDs in
`evals/published_failures.json`, so
`test_the_failures_i_publish_are_still_failing`'s subset check holds by
construction. `evals/baseline.json`'s numbers are computed by running the
finished replay suite once and committing the actual output — not
guessed.

`suite.yaml`: 50 entries, 10 each across `srm`, `novelty`, `simpson`,
`peeking`, `clean`. `clean` splits between `lift: 0.0` → `NO_EFFECT` and a
real lift → `SHIP`. `peeking` cases use `PEEKING`'s identity pass-through
(`lift: 0.0`, `days: 14`, truth `NO_EFFECT`) — flagged as a judgement call
in the source plan (an alternative is a real early-then-fading lift
testing whether the agent resists a premature look); starting with the
simpler form.

`Report.to_json()` excludes `generated_at` (canonical form for the
byte-identical test); `Report.write(out)` writes `results.json` =
`to_json()`'s dict + `generated_at`.

## Acceptance criteria

- [ ] `uv run pytest tests/test_phase_3_tracer.py -k "suite or replay or baseline or results_json or trajectory or no_tools or failures" -m "not live and not docker"`
      fully green
- [ ] `evals/suite.yaml` has 50 cases covering all 5 flaws
      (`test_the_suite_covers_all_five_flaws`)
- [ ] `fixtures/recorded.jsonl` and `evals/baseline.json` committed, with
      baseline numbers computed from an actual replay run, not guessed
- [ ] `evals/published_failures.json` case IDs are a subset of what the
      current replay run reports as failing
- [ ] `results.json` shape matches exactly:
      `accuracy, cases, by_flaw, models, generated_at` at minimum, each
      case has `id, flaw, correct, truth, verdict, trajectory`, each
      trajectory turn has `turn, act, body`
- [ ] `uv run prek` and `uv run ty check` clean

## Blocked by

- 04 — Tool-registry reconciliation
