# 06 — `peer eval` CLI + baseline regression gate

Type: AFK
Source: docs/plans/phase-3-plan.md (phase 3)

## What to build

Add `peer eval [--replay path] [--out dir]` to `cli.py` (same
`argparse` subcommand pattern as issue 03), calling
`run_suite(load_suite(), replay=args.replay, out=Path(args.out))` and
printing a summary (accuracy, by-flaw breakdown — `pydantic_evals`'s own
`report.print(...)` table output is a reasonable model for the format, or
call it directly if the adapter in issue 05 keeps a handle to the
underlying `EvaluationReport`).

Wire the regression gate into `uv run prek` (the pre-commit entry point
this project already runs at the end of every task, per CLAUDE.md) so a
prompt or tool-registry change that drops accuracy below
`evals/baseline.json - 0.02` (or a flaw score by more than `0.10`) fails
the build — mirrors `test_the_baseline_gate_catches_a_regression`'s
tolerance exactly, so the gate and the test agree.

## Acceptance criteria

- [ ] `uv run python -m peer_agent eval` runs the full 50-case suite in
      replay mode by default and prints an accuracy/by-flaw summary to
      the terminal
- [ ] `uv run python -m peer_agent eval --replay fixtures/recorded.jsonl --out evals/out`
      writes `evals/out/results.json`
- [ ] A pre-commit (or equivalent `prek`) hook runs the eval gate and
      fails when accuracy regresses past `baseline.json`'s tolerance —
      verify by temporarily corrupting a scripted trajectory in
      `recorded.jsonl` and confirming the hook fails, then reverting
- [ ] `uv run pytest tests/test_phase_3_tracer.py -k baseline_gate -m "not live and not docker"`
      green
- [ ] `uv run prek` and `uv run ty check` clean

## Blocked by

- 05 — Eval harness on pydantic_evals
- 03 — Scripted CLI (`peer review`/`peer design` pattern to follow)
