# Phase 1 — the tracer bullet: implementation plan

## Context

`tests/test_phase_1_tracer.py` is the fixed source of truth for Phase 1 (scope
is this file only — the five other test files in git history are later
phases and are out of scope for now). `tests/conftest.py` is also fixed and
imports `peer_agent.agent.Agent`, `peer_agent.sandbox.Sandbox`,
`peer_agent.sim.{NOVELTY,SIMPSON,SRM,make_case}`, `peer_agent.tools.default_tools`
unconditionally at module scope — every one of these names must exist and be
importable before pytest can even collect the file. `src/peer_agent/` is
currently empty (only a `.git_keep`); this is a from-scratch build.

The project's `CLAUDE.md` names a suggested stack (Pydantic AI 2.0 as the
agent harness, `tea-tasting` for A/B stats) that we're adopting starting now,
even though the Phase 1 test/README language ("textbook formulas", "imports
numpy, pandas, scipy, nothing else") was written assuming a hand-rolled loop
and hand-rolled stats. Confirmed via docs that both libraries fit without
forking the test contract:

- `pydantic_ai.models.function.FunctionModel` lets a plain Python callable
  stand in for the model — the test's `FakeLLM(messages, tools) -> turns`
  object works underneath it, we just need an adapter.
- `pydantic_ai.Tool.from_schema(function, name, description, json_schema,
  takes_ctx=False)` builds a tool from a hand-written JSON schema instead of
  inferring one from a function signature — matches `tools.py`'s "hand-written
  schemas" requirement exactly.
- `UsageLimits(request_limit=n)` on `agent.run_sync(...)` raises
  `UsageLimitExceeded` once the model-call budget is spent — this is what
  `max_turns` / `OverBudget` wraps.
- `tea_tasting.Experiment({"name": tt.SampleRatio()}, variant="arm")` and
  `tt.Experiment({"name": tt.Mean("converted")}, variant="arm")` give p-value,
  and (for `Mean`) `rel_effect_size` / `rel_effect_size_ci_lower` /
  `rel_effect_size_ci_upper` directly off a boolean `arm` column (control =
  lowest value = `False`, treatment = `True`, matching how `sim.py` plants the
  lift on `arm == True`).

## Module contracts

**`src/peer_agent/types.py`**
```python
class Verdict(Enum):
    NO_EFFECT = auto()
    INVALID = auto()
    SHIP = auto()
    EXTEND = auto()

@dataclass(frozen=True)
class SRMResult:
    mismatch: bool
    p_value: float
    n_control: int
    n_treatment: int

@dataclass(frozen=True)
class AnalyzeResult:
    p_value: float
    lift: float
    ci_low: float
    ci_high: float
```
(All four `Verdict` members included now for forward compat with the deleted
later-phase tests; only `NO_EFFECT`/`INVALID` are exercised in Phase 1.)

**`src/peer_agent/sim.py`**
```python
@dataclass(frozen=True)
class Case:
    df: pd.DataFrame
    truth: Verdict

def SRM(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    # drop ~15% of arm==True rows so the split is no longer ~50/50
    ...

def NOVELTY(df, rng):
    raise NotImplementedError("phase 3")

def SIMPSON(df, rng):
    raise NotImplementedError("phase 3")

def make_case(n, lift=0.0, seed=0, flaws=None, days=None) -> Case:
    rng = np.random.default_rng(seed)
    arm = rng.random(n) < 0.5
    baseline = 0.10
    rate = np.where(arm, baseline * (1 + lift), baseline)
    converted = rng.random(n) < rate
    df = pd.DataFrame({"arm": arm, "converted": converted})
    for flaw in flaws or []:
        df = flaw(df, rng)
    truth = Verdict.NO_EFFECT if lift == 0.0 else Verdict.SHIP
    return Case(df=df, truth=truth)
```
`NOVELTY`/`SIMPSON` and the `days` kwarg exist only so `conftest.py`'s
`novelty`/`simpson` fixtures stay importable and constructible; they are not
exercised by any Phase 1 test and deliberately raise if actually called.

**`src/peer_agent/stats.py`** (tea-tasting backed)
```python
import tea_tasting as tt

_SRM_ALPHA = 0.001

def check_srm(df) -> SRMResult:
    result = tt.Experiment({"sample_ratio": tt.SampleRatio()}, variant="arm").analyze(df)["sample_ratio"]
    return SRMResult(
        mismatch=result.pvalue < _SRM_ALPHA,
        p_value=result.pvalue,
        n_control=int((~df.arm).sum()),
        n_treatment=int(df.arm.sum()),
    )

def analyze(df) -> AnalyzeResult:
    result = tt.Experiment({"conversion": tt.Mean("converted")}, variant="arm").analyze(df)["conversion"]
    return AnalyzeResult(
        p_value=result.pvalue,
        lift=result.rel_effect_size,
        ci_low=result.rel_effect_size_ci_lower,
        ci_high=result.rel_effect_size_ci_upper,
    )
```
Verify empirically that tea-tasting accepts a boolean `variant` column
directly; fall back to `df.assign(arm=df.arm.astype(int))` if not. Also
verify the `_SRM_ALPHA = 0.001` threshold against the fixed seeds in the test
(n=50,000, clean vs SRM-broken) once implemented — adjust the threshold or
the SRM drop fraction (currently ~15% of treatment rows) if a specific seed
misbehaves.

**`src/peer_agent/tools.py`**
```python
@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    schema: dict
    fn: Callable[[pd.DataFrame], object]

def default_tools(sandbox=None) -> list[Tool]:
    return [
        Tool("check_srm", "Check whether the traffic split matches the assigned ratio.",
             {"type": "object", "properties": {}}, stats.check_srm),
        Tool("analyze", "Run the primary conversion-rate test comparing treatment to control.",
             {"type": "object", "properties": {}}, stats.analyze),
    ]
```
`Tool` lives in `tools.py` (not `types.py`) — it's registry plumbing, not a
domain type. `sandbox=None` optional param kept only because `conftest.py`'s
(currently unused-by-Phase-1) `agent` fixture calls `default_tools(sandbox)`.

**`src/peer_agent/sandbox.py`** — stub only, import-safe, unexercised by Phase 1:
```python
class Sandbox:
    def __init__(self, timeout: int = 30, memory_mb: int = 512):
        self.timeout, self.memory_mb = timeout, memory_mb
    def __enter__(self): return self
    def __exit__(self, *exc): return None
```

**`src/peer_agent/agent.py`** — the loop, built on `pydantic_ai.Agent` + `FunctionModel`:
```python
class OverBudget(Exception): ...

@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict
    result: object

@dataclass
class Trajectory:
    calls: list[ToolCall] = field(default_factory=list)
    answer: str = ""
    done: bool = False
    verdict: Verdict | None = None

_VERDICT_RE = re.compile(r"VERDICT:\s*([A-Z_]+)")

class Agent:
    def __init__(self, llm, tools, max_turns=8):
        self.llm, self.tools, self.max_turns = llm, tools, max_turns

    @classmethod
    def from_env(cls):
        raise NotImplementedError("phase 2: real model wiring")

    def review(self, case) -> Trajectory:
        calls: list[ToolCall] = []

        def bind(tool):
            def bound(**kwargs):
                result = tool.fn(case.df)
                calls.append(ToolCall(tool.name, kwargs, result))
                return result
            return bound

        pai_tools = [
            PaiTool.from_schema(function=bind(t), name=t.name, description=t.description,
                                 json_schema=t.schema, takes_ctx=False)
            for t in self.tools
        ]

        def adapter(messages, info) -> ModelResponse:
            turn = self.llm(messages, self.tools)
            if isinstance(turn, str):
                return ModelResponse(parts=[TextPart(turn)])
            return ModelResponse(parts=[ToolCallPart(tool_name=n, args=a) for n, a in turn])

        pai_agent = PydanticAgent(FunctionModel(adapter), tools=pai_tools)
        try:
            result = pai_agent.run_sync("Review this experiment.",
                                         usage_limits=UsageLimits(request_limit=self.max_turns))
        except UsageLimitExceeded as exc:
            raise OverBudget(str(exc)) from exc

        answer = result.output
        m = _VERDICT_RE.search(answer.upper())
        verdict = Verdict[m.group(1)] if m and m.group(1) in Verdict.__members__ else None
        return Trajectory(calls=calls, answer=answer, done=True, verdict=verdict)
```
Key point: `self.llm` is called with pydantic-ai's *native* `messages` list —
no reformatting needed. `str(llm.seen[-1])` (the test's check) will naturally
contain `"mismatch"` because it's the `repr()` of a `ModelRequest` holding a
`ToolReturnPart(content=SRMResult(mismatch=True, ...))`, and dataclass `repr`
includes field names. Verify this specific assertion first since it's the
one place native pydantic-ai formatting is load-bearing.

**`src/peer_agent/cli.py` + `src/peer_agent/__main__.py`**
```python
# __main__.py
from .cli import main
if __name__ == "__main__":
    raise SystemExit(main())
```
```python
# cli.py
class _ScriptedFakeLLM:
    def __init__(self): self._called = False
    def __call__(self, messages, tools):
        if not self._called:
            self._called = True
            return [("check_srm", {})]
        return "VERDICT: INVALID. The traffic split doesn't match the assigned ratio."

def _build_llm():
    if os.environ.get("PEER_FAKE_LLM") == "1":
        return _ScriptedFakeLLM()
    raise NotImplementedError("phase 2: real model wiring")

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="peer")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("review").add_argument("path")
    args = parser.parse_args(argv)

    df = pd.read_parquet(args.path)
    case = Case(df=df, truth=Verdict.NO_EFFECT)  # truth unknown/unused by review()
    traj = Agent(_build_llm(), tools=default_tools()).review(case)

    verdict = traj.verdict.name if traj.verdict else "UNKNOWN"
    print(f"VERDICT: {verdict}\n{traj.answer}")
    return 0
```
`_ScriptedFakeLLM` is a self-contained package-local fake (not importing
`tests/conftest.py`), scripted identically to the test's `FakeLLM` turn
protocol, so it drives the same `Agent.review()` code path.

## pyproject.toml changes

- Add to `[project.dependencies]`: `numpy`, `pandas`, `tea-tasting`, `pyarrow`
  (needed for `pd.read_parquet`/`to_parquet`). `pydantic-ai[skill]` and
  `pydantic-harness` are already present and cover `Agent`, `Tool`,
  `FunctionModel`, `UsageLimits`. No direct `scipy` dependency needed —
  tea-tasting pulls it in transitively.
- Add a `[dependency-groups]` `dev = ["pytest"]` (uv-idiomatic; pytest isn't
  currently declared anywhere despite tests existing).
- Add `[build-system]` (`hatchling`) + `[tool.hatch.build.targets.wheel]
  packages = ["src/peer_agent"]` so `import peer_agent` and `python -m
  peer_agent` work at all — currently there is no build-system table and the
  package is not installable/importable.
- Do **not** add a `tests/__init__.py` — pytest's default import mode needs
  its absence for `from conftest import FakeLLM, names` (a bare, non-package
  import) to resolve.

## Build order (TDD, red -> green)

Because `conftest.py` imports every module eagerly at collection time, no
test in this file can even be collected until every module exists at least
as an import-safe stub. Order:

0. Stub every module (`types.py`, `sim.py`, `stats.py`, `tools.py`,
   `sandbox.py`, `agent.py`, `cli.py`, `__main__.py`) enough that
   `pytest --collect-only tests/test_phase_1_tracer.py` succeeds.
1. `types.py` for real.
2. `sim.py` for real -> green: the 4 "generator" tests.
3. `stats.py` for real (tea-tasting) -> green: the 3 "statistics" tests.
4. `tools.py` for real -> green: the 2 "registry" tests.
5. `agent.py` for real (pydantic-ai loop) -> green: the 6 "loop" tests.
6. `cli.py` + `__main__.py` -> green: the CLI subprocess test.
7. `pyproject.toml` deps/build-system + `uv sync` -> full file green.

## Verification

```
uv sync
uv run pytest tests/test_phase_1_tracer.py -q
```
All 15 tests green. Note: the docstring's "under a second" is aspirational —
the CLI test spawns a subprocess that imports pandas/tea-tasting/pydantic-ai
fresh, which may push past 1s; that's an acceptable, not a blocking, gap.
Sanity-check the CLI manually:
```
uv run python -c "
from peer_agent.sim import make_case, SRM
make_case(n=50_000, seed=3, flaws=[SRM]).df.to_parquet('/tmp/exp.parquet')
"
PEER_FAKE_LLM=1 uv run python -m peer_agent review /tmp/exp.parquet
```
Expect stdout containing `VERDICT: INVALID`.
