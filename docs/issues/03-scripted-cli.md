# 03 — `peer review <path>` / `peer design "<brief>"`: scripted, trajectory printed

Type: AFK
Source: docs/plans/phase-3-plan.md (phase 3)

## What to build

`cli.py` currently has a phase-1 leftover: `_build_llm()` raises
`NotImplementedError` for anything but `_ScriptedFakeLLM`
(`PEER_FAKE_LLM=1`), even though `Agent.from_env()` has worked since
phase 2. Rewire it for real, non-interactive use (batch review of a
parquet file, or scripting/CI) — the complement to issue 02's interactive
chat.

```python
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="peer")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("review").add_argument("path")
    d = sub.add_parser("design"); d.add_argument("brief")
    args = parser.parse_args(argv)

    if args.command == "review":
        traj = Agent.from_env().review(Case(df=pd.read_parquet(args.path), truth=Verdict.NO_EFFECT))
    elif args.command == "design":
        traj = Agent.from_env().design(args.brief)
    _print_trajectory(traj)
    return 0
```

(Sketch from the phase-3 plan doc — the exact arg/flag shape is free to
adjust.) `_print_trajectory` prints every `ToolCall` in order (name, args,
truncated result repr), then the final verdict/spec and answer — nothing
about what the agent did is hidden after a run. Drop
`_ScriptedFakeLLM`/`PEER_FAKE_LLM` entirely; it was a phase-1 crutch fully
superseded by `Agent.from_env()`.

## Acceptance criteria

- [ ] `peer review <parquet path>` runs a real model end-to-end and
      prints every tool call plus the final verdict/answer
- [ ] `peer design "<brief>"` runs a real model end-to-end and prints
      every tool call plus the final spec or clarifying question
- [ ] `_ScriptedFakeLLM`, `PEER_FAKE_LLM`, and the `NotImplementedError`
      branch are removed from `cli.py`
- [ ] `fixtures/sample.parquet` exists (a small real-shaped experiment
      dataframe) so the review command has something to point at out of
      the box
- [ ] `uv run pytest tests/test_phase_1_tracer.py tests/test_phase_2_tracer.py -m "not live and not docker"`
      still fully green
- [ ] `uv run prek` and `uv run ty check` clean

## Blocked by

None - can start immediately
