# Phase 3 — make it a project: implementation plan

## Context

`tests/test_phase_3_tracer.py` is the fixed source of truth. Its own
docstring orders the work as sandbox → harness → MCP → page. This plan
**reorders that** on explicit request: the goal right now is something
usable at the CLI/web end, with every output visible and debuggable, even
before the docker sandbox and MCP server are hardened. Nothing below drops
scope — it resequences it so a real deliverable exists early and the
harder infrastructure (containers, stdio protocol) comes after.

What's already standing from phases 1–2 (verified green,
`pytest tests/test_phase_1_tracer.py tests/test_phase_2_tracer.py -m "not live and not docker"` →
26 passed): `types.py`, `sim.py`, `stats.py`, `design.py`, `agent.py`
(`review()`, `design()`, `Agent.from_env()` all real), `tools.py`
(`default_tools`, `analysis_tools`, `design_tools`), and
`.agents/skills/{review,design}-protocol/SKILL.md`. `cli.py` exists but is
a phase-1 leftover — `_build_llm()` raises `NotImplementedError` for
anything but a scripted fake, despite `Agent.from_env()` already working.
`evals/`, `fixtures/`, `web/` are empty directories. `sandbox.py` is a
context-manager stub with no container logic.

A key finding that de-risks the harness: **the offline harness tests
(`test_a_replayed_run_is_byte_identical`, `test_the_baseline_gate_catches_a_regression`,
all the `results.json` shape tests) only ever call `run_suite(..., replay=REPLAY)`.**
Only `test_the_real_grade` and `test_the_toolbox_earns_its_existence` are
`@pytest.mark.live @pytest.mark.slow`. That means `fixtures/recorded.jsonl`
and `evals/baseline.json` are both **authored by me**, not captured from a
real model — I write the scripted trajectories, so I control accuracy,
`by_flaw` scores, and which case IDs are the "published failures." This
makes the whole harness buildable and testable without spending API money,
using the same `FakeLLM`-style scripted-turns pattern already in
`conftest.py`.

## Reprioritized build order

1. **`sim.py` / `types.py` small extensions** — `Case.path` (lazy temp
   parquet), an `EvalCase` shape. Foundation for everything else.
2. **`evals.py` + `evals/suite.yaml` + `fixtures/recorded.jsonl` +
   `evals/baseline.json`** — the harness, replay-only. Produces
   `results.json`. This is the first point real output exists to look at.
3. **`web/`** — a static page reading `results.json`. No build step. This
   is the "UI."
4. **`cli.py` rewrite** — `peer review`, `peer design`, `peer eval`, wired
   to `Agent.from_env()` for real, with verbose, debuggable trajectory
   printing. This is the "CLI."
5. **`sandbox.py` + `Dockerfile` + `run_python` tool** — real container
   sandbox, wired into `analysis_tools()` so `Agent.from_env()` can write
   its own code.
6. **`mcp.py`** — stdio server over the unified tool registry, last,
   because it's "how someone else tries it," not what you're using today.

Steps 5–6 are exactly as specified in the tracer file; nothing about them
changes, they just move later in the build order.

## Module contracts

### `src/peer_agent/sim.py` — `Case.path`

```python
import tempfile

@dataclass(frozen=True)
class Case:
    df: pd.DataFrame
    truth: Verdict
    _path: str | None = field(default=None, compare=False, repr=False)

    @property
    def path(self) -> str:
        if self._path is None:
            fd, path = tempfile.mkstemp(suffix=".parquet")
            self.df.to_parquet(path)
            object.__setattr__(self, "_path", path)
        return self._path
```

Frozen dataclass + lazy-write via `object.__setattr__` keeps `Case`
otherwise unchanged (`>=` column-set checks in phase-1 tests still hold).
Needed only by the `docker`-marked MCP test (`srm.path`) — low priority,
included here for definition-of-done completeness, not blocking steps 1–4.

### `evals/suite.yaml` + `src/peer_agent/evals.py`

`suite.yaml`: 50 entries, 10 each across the five required flaws
(`srm`, `novelty`, `simpson`, `peeking`, `clean`), each a flat record:

```yaml
- id: srm-01
  flaw: srm
  seed: 101
  n: 50000
  lift: 0.0
  days: null
  truth: INVALID
```

`clean` cases split between `lift: 0.0` → `NO_EFFECT` and `lift: 0.06` →
`SHIP`, so the suite exercises both non-flawed outcomes. `peeking` cases:
`lift: 0.0`, `days: 14`, no flaw applied (`PEEKING` is the identity
pass-through per `sim.py`) — truth `NO_EFFECT`. **Open judgement call,
flagged rather than guessed:** whether "peeking" cases should instead
carry a real lift that's only significant on an early day-slice (testing
whether the agent's own `sequential`/day-slicing resists it) is genuinely
ambiguous from the tracer file alone; starting with the simpler
identity-flaw form and revisiting once the replay fixture is authored.

```python
@dataclass(frozen=True)
class EvalCase:
    id: str
    flaw: str
    seed: int
    truth: Verdict

    def build(self) -> Case:
        ...  # sim.make_case(...) using stored n/lift/days/flaw-fn lookup


def load_suite(path: str = "evals/suite.yaml") -> list[EvalCase]: ...


@dataclass(frozen=True)
class Report:
    accuracy: float
    false_ships: float   # truth in {NO_EFFECT, INVALID} but verdict == SHIP
    false_blocks: float  # truth == SHIP but verdict != SHIP
    by_flaw: dict[str, float]
    cases: list[dict]        # id, flaw, correct, truth, verdict, trajectory
    models: list[dict]       # [{"name": ..., "accuracy": ...}, ...] incl. "no tools" baseline
    published_failures: list[dict]
    generated_at: str

    def to_json(self) -> str:
        """Canonical form for equality checks — excludes generated_at."""

    def write(self, out: Path) -> None:
        """Writes results.json = to_json()'s dict + generated_at."""


def run_suite(
    suite: list[EvalCase],
    *,
    model: str | None = None,
    tools: list | None = None,
    replay: str | None = None,
    out: Path | None = None,
) -> Report: ...
```

Replay mechanics: `fixtures/recorded.jsonl` is JSON-lines, one record per
`(case_id, model_name)`, each a list of scripted turns in the same shape
`FakeLLM.turns` already uses (`[("check_srm", {}), ..., "VERDICT: ..."]`).
`run_suite(replay=...)` groups records by case, builds a `FakeLLM(turns=...)`
per case, drives `Agent(fake_llm, tools=analysis_tools()).review(case)`,
and scores against `EvalCase.truth`. Two model rows get recorded per case —
`"claude-opus-5"` (scripted to use tools, mostly correct) and
`"claude-opus-5 (no tools)"` (scripted to skip straight to a verdict,
deliberately worse) — satisfying `test_the_no_tools_baseline_is_in_the_report`.
I deliberately script 2–3 case IDs to end in the wrong verdict and record
those same IDs in a small hand-curated `evals/published_failures.json`, so
`test_the_failures_i_publish_are_still_failing`'s subset check holds by
construction. `evals/baseline.json`'s `accuracy`/`by_flaw` are computed by
running the finished replay suite once and committing the actual numbers —
not guessed — so the regression gate is real from day one.

`generated_at` exclusion from `to_json()` is what makes
`test_a_replayed_run_is_byte_identical` possible without freezing the
clock.

### `web/` — the page

Plain HTML + vanilla JS (`index.html`, `app.js`), **no framework, no build
step** — deliberate, since this is a Python project and a JS toolchain
would be pure overhead for a single static page. Reads `results.json` via
`fetch()`. Layout: summary header (accuracy, by-flaw bars, models
compared), a case table (id, flaw, ✓/✗, truth vs verdict), click a row to
expand its `trajectory` turn-by-turn (tool name, args, result — the
"debuggable" part). `test_the_page_reads_nothing_but_results_json` scans
`web/**/*.jsx`; using plain `.js` means that glob matches nothing and the
test passes trivially, but the real point — no import from `src`/`peer_agent` —
is honored by construction since it's a fetch-only static page.

### `src/peer_agent/cli.py` — rewrite

```python
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="peer")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("review").add_argument("path")
    d = sub.add_parser("design"); d.add_argument("brief")
    e = sub.add_parser("eval"); e.add_argument("--replay"); e.add_argument("--out", default="evals/out")
    args = parser.parse_args(argv)

    if args.command == "review":
        traj = Agent.from_env().review(Case(df=pd.read_parquet(args.path), truth=Verdict.NO_EFFECT))
        _print_trajectory(traj)
    elif args.command == "design":
        traj = Agent.from_env().design(args.brief)
        _print_trajectory(traj)
    elif args.command == "eval":
        report = run_suite(load_suite(), replay=args.replay, out=Path(args.out))
        _print_report(report)
    return 0
```

`_print_trajectory` prints every `ToolCall` (name, args, truncated repr of
result) in order, then the final verdict/spec and answer — this is the
"visible and debuggable" requirement: nothing about what the agent did is
hidden after a run. Drops `_ScriptedFakeLLM`/`PEER_FAKE_LLM` entirely —
that was a phase-1 crutch now superseded by `Agent.from_env()`.

### `src/peer_agent/tools.py` + `sandbox.py` + `agent.py` — the sandbox tools

Registry reconciliation (resolves a real binding hazard): `agent.py`'s
`_bind()` always prepends `case.df` as the first positional arg to every
tool in `review()`'s list. `power_analysis`/`simulate_design`/`ask` don't
take a `df` — so they must never appear in the list passed to
`Agent(..., tools=...)` for `review()`. Contract:

- `analysis_tools(sandbox=None)` → the 6 stats tools (`bind_df=True`), plus
  `run_python` **iff** `sandbox` is given. This is what `Agent.from_env()`
  and `review()`-time tests use.
- `design_tools()` → unchanged (`power_analysis`, `simulate_design`, `ask`
  — no `df`, only used inside `design()`, which never binds `case.df`).
- `default_tools(sandbox)` → `analysis_tools(sandbox) + design_tools()`,
  the full 10-tool union `test_the_registry_is_complete` checks against
  and what `mcp.py` lists — **not** fed into `Agent()` directly.

`run_python`'s raw signature is `_run_python(df, code, *, sandbox)`, built
via `functools.partial(_run_python, sandbox=sandbox)` at tool-construction
time (sandbox is closed over then, `df` stays the first positional param
so agent.py's existing `_bind(tool, calls, case.df)` still works
unmodified).

`sandbox.py`: `docker run --rm -i --network none --memory {mb}m --read-only
-v {tmpdir}:/data:ro <image> python -c -`, code piped via stdin (avoids
arg-length/shell-injection concerns), `subprocess.run(timeout=self.timeout)`.
`Sandbox.run(code, df) -> RunResult(ok, stdout, stderr)`: writes `df` to
`{tmpdir}/exp.parquet` first. Timeout → `TimeoutExpired` caught, `ok=False`,
container killed via `docker kill <name>` (named, not anonymous, so it's
killable after the client subprocess dies). Memory-eat test relies on the
container's OOM kill (exit 137) → `ok=False`. A small `Dockerfile` (new,
repo root) pins `numpy`/`pandas`/`pyarrow` versions matching `uv.lock` on a
`python:3.14-slim` base — the image `sandbox.py` runs.

### `src/peer_agent/mcp.py` — last

`stdio_roundtrip(method, **kwargs)` test helper implies a real stdio
JSON-RPC loop over `default_tools(sandbox)`; `tools/call` for a stats tool
takes `{"path": ...}` and loads the dataframe itself (`pd.read_parquet(path)`)
rather than receiving a bound `df` — a different calling contract from the
in-process agent tools, by necessity of crossing a process boundary.
`resources/read` for `peer://protocol` serves the review-protocol skill's
markdown (the "validate" checklist) as a resource. Deferred to last per the
reprioritization above; not needed for the CLI/web deliverable.

## pyproject.toml / dependency changes

- `pyyaml` (or `ruamel.yaml`) for `evals/suite.yaml` — not currently a
  dependency, needs `uv add`.
- No new dependency for the web page (plain JS) or the sandbox (`docker`
  CLI is a subprocess dependency, not a Python package).

## Verification

```bash
# steps 1-2: harness, offline, no docker, no network
uv run pytest tests/test_phase_3_tracer.py -m "not live and not docker" -q

# step 3/4 manual check — no automated browser test in scope
uv run python -m http.server -d web 8000   # then eyeball it against a real results.json
uv run python -m peer_agent eval && uv run python -m peer_agent review fixtures/sample.parquet

# step 5, once Docker Desktop / a container runtime is available
docker build -t peerai-sandbox .
uv run pytest tests/test_phase_3_tracer.py -m docker -q

# step 6
uv run pytest tests/test_phase_3_tracer.py -m docker -k mcp -q

# phases 1-2 still green throughout
uv run pytest tests/test_phase_1_tracer.py tests/test_phase_2_tracer.py -m "not live and not docker" -q

# costs money, run once the harness above is solid
uv run pytest tests/test_phase_3_tracer.py -m live -q

uv run prek
uv run ty check
```

## Open/risky items — flagged rather than guessed

- **`peeking` eval-case design** (identity-flaw vs. a real early-then-fading
  significance) is a genuine judgement call — see suite.yaml section above.
- **`evals/baseline.json` numbers are only as real as the scripted
  `recorded.jsonl` trajectories I author** — they prove the harness/gate
  machinery works, not that a real model scores that well. `test_the_real_grade`
  (`-m live`) is the actual grade; don't read the committed baseline as a
  claim about model quality.
- **Docker sandbox networking/memory-limit flags are host-dependent**
  (`--network none`, `--memory` behavior can differ Docker Desktop vs. Linux
  daemon) — the three `docker`-marked sandbox tests are the real
  verification; don't consider `sandbox.py` done from reading the code
  alone.
- **`Case.path`'s lazy temp-file never gets cleaned up** — acceptable for
  test/CLI runs (OS temp dir, short-lived process) but worth a `tempfile.TemporaryDirectory`-owning
  wrapper if `Case` objects start living long in a long-running process
  (e.g. inside `mcp.py`'s server loop).
- **No live-model cost/provider decision needed here** — `Agent.from_env()`
  and `.env` are already wired from phase 2; step 4's `cli.py` just calls
  it.

### Critical files for implementation

- `src/peer_agent/evals.py` (new)
- `evals/suite.yaml`, `evals/baseline.json`, `evals/published_failures.json` (new)
- `fixtures/recorded.jsonl` (new)
- `web/index.html`, `web/app.js` (new)
- `src/peer_agent/cli.py` (rewrite)
- `src/peer_agent/sandbox.py`, `Dockerfile` (new)
- `src/peer_agent/tools.py`, `src/peer_agent/agent.py` (registry reconciliation)
- `src/peer_agent/mcp.py` (new, last)
- `tests/test_phase_3_tracer.py` (read-only source of truth)
