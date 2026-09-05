# Phase 1 — tracer bullet: vertical-slice issues

Source: `docs/plans/phase-1-plan.md`. `tests/test_phase_1_tracer.py` and
`tests/conftest.py` are fixed; `conftest.py` imports every `peer_agent.*`
module eagerly at collection time, so these slices are strictly sequential
(each one is blocked by the last) rather than independently parallel — but
each still lands a real, independently-verifiable increment of green tests.

No issue tracker is configured for this repo (no git remote). These are
written as issue-ready bodies to paste into a tracker once one exists.

---

## 1. Packaging bootstrap

**Type:** AFK
**Blocked by:** None — can start immediately

### What to build

Make `peer_agent` an installable, importable package under `src/` layout,
with the dependencies Phase 1 needs, before any module code exists.

- Add `[build-system]` (hatchling) + `[tool.hatch.build.targets.wheel]
  packages = ["src/peer_agent"]` to `pyproject.toml`.
- Add `numpy`, `pandas`, `tea-tasting`, `pyarrow` to `[project.dependencies]`
  (`pydantic-ai[skill]` and `pydantic-harness` are already present).
- Add `[dependency-groups] dev = ["pytest"]`.
- Add an empty `src/peer_agent/__init__.py`.
- Do **not** add `tests/__init__.py` — pytest's default import mode needs its
  absence for `conftest.py`'s bare `from conftest import FakeLLM, names` to
  resolve.

### Acceptance criteria

- [ ] `uv sync` completes without error
- [ ] `uv run python -c "import peer_agent"` succeeds
- [ ] `tests/__init__.py` does not exist

### Blocked by

None - can start immediately

---

## 2. Module skeleton

**Type:** AFK
**Blocked by:** #1 (packaging bootstrap)

### What to build

Stub every module `tests/conftest.py` imports at collection time, with just
enough shape (classes, function names, signatures) to be import-safe. No
real logic yet — bodies can `raise NotImplementedError` or return dummy
values. Modules: `types.py`, `sim.py`, `stats.py`, `tools.py`, `sandbox.py`,
`agent.py`, `cli.py`, `__main__.py`.

Specifically must exist and be importable: `peer_agent.agent.Agent`,
`peer_agent.sandbox.Sandbox`, `peer_agent.sim.{NOVELTY,SIMPSON,SRM,make_case}`,
`peer_agent.tools.default_tools` — these are imported unconditionally by
`conftest.py`, so pytest cannot even collect `test_phase_1_tracer.py` until
all of them resolve.

### Acceptance criteria

- [ ] `uv run pytest --collect-only tests/test_phase_1_tracer.py` collects
      all 15 tests with zero collection errors
- [ ] No test passes yet (stubs are non-functional by design)

### Blocked by

#1 (packaging bootstrap)

---

## 3. Generator produces graded cases

**Type:** AFK
**Blocked by:** #2 (module skeleton)

### What to build

Real `types.py` and `sim.py`. The generator has to produce data with a known
answer (`Case.truth`), be deterministic under a fixed seed, actually respond
to the `lift` argument in the raw conversion rates, and support one flaw
(`SRM`) that visibly breaks the ~50/50 traffic split.

```python
class Verdict(Enum):
    NO_EFFECT = auto()
    INVALID = auto()
    SHIP = auto()
    EXTEND = auto()

@dataclass(frozen=True)
class Case:
    df: pd.DataFrame
    truth: Verdict
```

`make_case(n, lift=0.0, seed=0, flaws=None, days=None)` uses
`np.random.default_rng(seed)`, plants the lift on `arm == True`, and sets
`truth = Verdict.NO_EFFECT if lift == 0.0 else Verdict.SHIP`. `NOVELTY` and
`SIMPSON` stay as `NotImplementedError` stubs (phase 3) — they only need to
exist for `conftest.py`'s fixtures to construct without erroring at
collection time.

### Acceptance criteria

- [ ] `test_a_case_carries_its_own_truth` passes
- [ ] `test_the_same_seed_gives_the_same_data` passes
- [ ] `test_a_planted_lift_shows_up_in_the_raw_rates` passes
- [ ] `test_the_srm_flaw_actually_breaks_the_split` passes

### Blocked by

#2 (module skeleton)

---

## 4. Stats layer on tea-tasting

**Type:** AFK
**Blocked by:** #3 (generator produces graded cases)

### What to build

Real `stats.py`, backed by `tea_tasting.Experiment` rather than hand-rolled
formulas (per `CLAUDE.md`'s suggested stack, adopted starting Phase 1).

```python
def check_srm(df) -> SRMResult:
    result = tt.Experiment({"sample_ratio": tt.SampleRatio()}, variant="arm").analyze(df)["sample_ratio"]
    ...

def analyze(df) -> AnalyzeResult:
    result = tt.Experiment({"conversion": tt.Mean("converted")}, variant="arm").analyze(df)["conversion"]
    ...  # rel_effect_size -> lift, rel_effect_size_ci_lower/upper -> ci_low/ci_high
```

Two things to verify empirically while implementing, since they're unresolved
in the design:

- Whether tea-tasting accepts a boolean `arm` column directly, or needs
  casting to int.
- Whether the `_SRM_ALPHA` threshold (design assumes `0.001`) and the `SRM`
  flaw's drop fraction (design assumes ~15% of `arm==True` rows) actually
  produce a mismatch on seed=3/7 and a clean result on seed=8 in the fixed
  test data — tune either if a specific seed misbehaves.

### Acceptance criteria

- [ ] `test_analyze_returns_the_numbers_i_need` passes
- [ ] `test_analyze_finds_a_large_effect_and_ignores_a_missing_one` passes
- [ ] `test_check_srm_fires_on_a_broken_split_and_not_on_a_clean_one` passes

### Blocked by

#3 (generator produces graded cases)

---

## 5. Tool registry

**Type:** AFK
**Blocked by:** #4 (stats layer on tea-tasting)

### What to build

Real `tools.py`: a project-owned `Tool` dataclass (name, description,
hand-written JSON schema, function) wrapping the two `stats.py` functions,
plus `default_tools()` returning both.

```python
@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    schema: dict
    fn: Callable[[pd.DataFrame], object]

def default_tools(sandbox=None) -> list[Tool]:
    return [
        Tool("check_srm", "...", {"type": "object", "properties": {}}, stats.check_srm),
        Tool("analyze", "...", {"type": "object", "properties": {}}, stats.analyze),
    ]
```

`Tool` lives here, not in `types.py` — it's registry plumbing, not a domain
type. `sandbox=None` stays optional only because `conftest.py`'s (Phase-1
unused) `agent` fixture calls `default_tools(sandbox)`.

### Acceptance criteria

- [ ] `test_the_registry_offers_exactly_what_exists_so_far` passes
- [ ] `test_each_tool_has_a_name_a_description_and_a_schema` passes

### Blocked by

#4 (stats layer on tea-tasting)

---

## 6. Agent loop on pydantic-ai FunctionModel

**Type:** AFK
**Blocked by:** #5 (tool registry)

### What to build

Real `agent.py`: `Agent.review()` drives a real `pydantic_ai.Agent` whose
model is a `FunctionModel` adapter forwarding pydantic-ai's native
`messages` list straight into the fixed `FakeLLM(messages, tools) -> turns`
protocol from `conftest.py`, translating the return value into a
`ModelResponse`. Tool calls dispatch through `Tool.from_schema(...)` built
from the `tools.py` registry. `UsageLimits(request_limit=max_turns)` on
`run_sync` gives budget enforcement; catch `UsageLimitExceeded` and re-raise
as `OverBudget`. Parse the final verdict out of free text via
`VERDICT:\s*([A-Z_]+)`.

```python
class OverBudget(Exception): ...

@dataclass
class Trajectory:
    calls: list[ToolCall] = field(default_factory=list)
    answer: str = ""
    done: bool = False
    verdict: Verdict | None = None
```

Highest integration risk in the whole plan: `test_the_tool_result_reaches_the_model`
asserts `"mismatch" in str(llm.seen[-1]).lower()`, which depends on the
`repr()` of a pydantic-ai `ModelRequest`/`ToolReturnPart` wrapping an
`SRMResult` dataclass including the literal field name `mismatch=...`.
Verify this specific assertion first — if native pydantic-ai formatting
doesn't surface the field name as expected, the adapter needs to shape the
tool-return content differently before this slice can be called done.

### Acceptance criteria

- [ ] `test_the_loop_calls_a_tool_and_finishes` passes
- [ ] `test_the_tool_result_reaches_the_model` passes
- [ ] `test_two_tools_in_sequence` passes
- [ ] `test_the_loop_stops_instead_of_spinning` passes (raises `OverBudget`)
- [ ] `test_the_verdict_comes_out_as_an_enum` passes

### Blocked by

#5 (tool registry)

---

## 7. CLI entrypoint

**Type:** AFK
**Blocked by:** #6 (agent loop on pydantic-ai FunctionModel)

### What to build

`cli.py` + `__main__.py` so the tracer bullet reaches a human from a
terminal: `python -m peer_agent review <parquet-file>` prints a verdict.

- `_ScriptedFakeLLM`: a self-contained, package-local fake (does not import
  `tests/conftest.py`) mirroring `FakeLLM`'s turn protocol, gated behind
  `PEER_FAKE_LLM=1`.
- `main(argv=None)`: `argparse` with a `review` subcommand taking a `path`,
  reads the parquet into a `Case`, runs `Agent(_build_llm(),
  tools=default_tools()).review(case)`, prints `VERDICT: {verdict}\n{answer}`.

### Acceptance criteria

- [ ] `test_the_cli_prints_a_verdict` passes (subprocess exits 0, stdout
      contains "invalid")
- [ ] Full file green: `uv run pytest tests/test_phase_1_tracer.py -q` → 15/15
      passed

### Blocked by

#6 (agent loop on pydantic-ai FunctionModel)
