# Phase 4 — statistical safety, then real data: implementation plan

## Context

Phases 1–3 built the machine: a tool registry, an agent loop that can't
skip the protocol, a sandbox, an eval suite with a merge gate, a CLI and
a web report. What phase 3 did *not* do is audit the statistics those
tools compute. The calibration was chosen to make five simulated flaw
classes detectable, not to survive contact with a real experiment.

This plan closes that gap. It has two tracks, in this order:

- **Track A — make the stats defensible.** Twelve defects in
  `stats.py` / `design.py`, ranked by how likely each is to produce a
  confidently wrong verdict, plus two systemic changes (pre-registration
  binding, no-invented-numbers) that matter more than any individual fix.
- **Track B — read real data.** A source-resolver layer so the same tools
  run against a Databricks warehouse instead of a local parquet, using
  SQL pushdown rather than row extraction.

**A before B is deliberate.** Connecting a production warehouse to an
analysis layer that fails open on guardrails is worse than not connecting
it at all — it converts a demo's wrong answer into a shipped regression.

Track A also breaks the phase-3 baseline. Items A2 (verdict taxonomy) and
A1 (guardrail direction) change what a correct verdict *is* for some
cases, so `evals/suite.yaml`, `fixtures/recorded.jsonl` and
`evals/baseline.json` must be re-authored as part of the work, not after
it. Doing this now, while the suite is 50 hand-authored cases, is far
cheaper than doing it later.

## Build order

### Track A — statistical safety

1. **A1** guardrails as non-inferiority (fails open today)
2. **A2** `INCONCLUSIVE` verdict + equivalence testing (`NO_EFFECT` is a lie today)
3. **A3** clustering / randomization-unit enforcement
4. **A4** `check_novelty` — uncertainty instead of a magic threshold
5. **A5** `sequential` — actually sequential (alpha spending or confidence sequences)
6. **A6** `check_srm` — expected ratio, multi-arm, per-segment and per-day
7. **A7** `scan_segments` — heterogeneity test, not sign comparison
8. **A8** explicit `control=` (one-line sign-inversion guard)
9. **A9** CUPED validity + auditable variance reduction
10. **A10** `simulate_design` simulates the *actual* analysis plan + Monte Carlo error
11. **A11** traffic feasibility; demote `_MAX_PLAUSIBLE_MDE` from verdict to flag
12. **A12** ratio metrics, bootstrap, winsorization, a real `alpha` parameter
13. **A13** *(systemic)* bind `review()` to a `DesignSpec` — pre-registration
14. **A14** *(systemic)* no-invented-numbers checker over the trajectory

A1, A2 and A8 are the cheapest-to-highest-value cluster and should land
first as one changeset with the re-authored baseline. A3 and A13 are the
two structural ones and will touch the most files.

### Track B — Databricks source layer

15. **B1** `sources.py` — resolve `parquet://`, `databricks://`, `sql:`
16. **B2** aggregate pushdown via narwhals/Ibis (`tt.aggr.read_aggregates`)
17. **B3** auth, read-only enforcement, statement timeout, row cap
18. **B4** Unity Catalog discovery as an MCP *resource*
19. **B5** packaging: run as a Databricks App with UC-scoped permissions

---

## Track A — module contracts

### A1 — Guardrails are backwards (`stats.py:97-109`)

Today `_regressed` is `p_value < 0.01 and rel_effect_size < 0`: a
*superiority* test used as a *safety* check. It fails open — an
underpowered guardrail reports clean.

Demonstrated against the real library (n=20k, true rate 0.120 → 0.118):

```
two-sided:  rel_effect = -6.95%,  p = 0.0580,  CI [-13.63%, +0.25%]
            → _regressed() == False → "no guardrail breached" → SHIP
```

A −7% point estimate on a guardrail, with the interval reaching −13.6%,
currently blesses a ship because a *superiority* test didn't clear 0.01.

The correct question is non-inferiority: *can we rule out harm worse than
the margin?* tea-tasting gives the one-sided readout directly:

```
tt.Mean("revenue", alternative="greater")
            → rel_effect_size_ci_lower = -12.59%,  upper = inf
            → -12.59% < -3% margin → cannot rule out breach
```

Contract:

```python
@dataclass(frozen=True)
class Guardrail:
    name: str
    margin: float           # e.g. 0.03 → tolerate at most a 3% relative move
    direction: Literal["down_is_bad", "up_is_bad"] = "down_is_bad"


class GuardrailStatus(StrEnum):
    CLEAN = "clean"           # CI rules out harm worse than margin
    BREACHED = "breached"     # CI excludes the margin on the harmful side
    INCONCLUSIVE = "inconclusive"  # too wide to say — BLOCKS, does not pass


@dataclass(frozen=True)
class GuardrailResult:
    statuses: tuple[tuple[str, GuardrailStatus], ...]
    bounds: tuple[tuple[str, float], ...]   # the one-sided bound per guardrail

    @property
    def blocks_ship(self) -> bool:
        return any(s is not GuardrailStatus.CLEAN for _, s in self.statuses)
```

`direction` selects `alternative="greater"` vs `"less"` and which CI bound
is compared against `∓margin`. **`INCONCLUSIVE` must block** — that is the
entire point of the fix. The current `tuple[str, ...]` guardrail argument
becomes `tuple[Guardrail, ...]`, which changes the tool schema the model
sees, so `review-protocol/SKILL.md` step 2 needs updating in the same
change: the model must supply a margin, and "inconclusive" is not a pass.

### A2 — `NO_EFFECT` conflates null with underpowered (`types.py:10`)

`p > 0.05 → NO_EFFECT` is the classic sin. There is no verdict for "we
don't know." Add one, and make `NO_EFFECT` earn itself:

```python
class Verdict(Enum):
    NO_EFFECT = auto()      # CI excludes the MDE — a real null result
    INCONCLUSIVE = auto()   # CI includes both 0 and the MDE — underpowered
    INVALID = auto()
    SHIP = auto()
    EXTEND = auto()
```

`analyze()` gains an equivalence readout (TOST against ±MDE, which comes
from the spec — see A13):

```python
@dataclass(frozen=True)
class AnalyzeResult:
    p_value: float
    lift: float
    ci_low: float
    ci_high: float
    width: float
    equivalent_to_null: bool | None   # None when no MDE was supplied
    mde: float | None
```

Verdict rule: significant → SHIP/EXTEND as today; not significant *and*
CI excludes ±MDE → NO_EFFECT; otherwise → INCONCLUSIVE.

**This is the change that most affects the eval suite.** Several `clean`
cases with `lift: 0.0` will now be INCONCLUSIVE rather than NO_EFFECT
unless `n` is large enough for the CI to exclude the MDE. Either raise `n`
in those cases or relabel them — decide per case, deliberately, and record
the reasoning in `suite.yaml` comments. `Report.false_ships` /
`false_blocks` also need a third bucket for "correctly refused to answer."

### A3 — Everything assumes iid rows

No clustering anywhere. If a unit appears in more than one row (sessions,
repeat visits), standard errors are understated and every p-value in the
system is optimistic. This is the largest single source of false positives
in real A/B data, and it becomes live the moment Track B lands.

`design.py:14`'s `_SESSION_CONTAMINATION = 0.5` acknowledges the problem
with an invented constant instead of solving it.

Contract — make the randomization unit explicit and enforce it:

```python
def analyze(
    df: pd.DataFrame,
    *,
    unit: str | None = None,      # column identifying the randomization unit
    cuped: bool = False,
    covariate: str | None = None,
    alpha: float = 0.05,
) -> AnalyzeResult:
    """
    When `unit` is given and has duplicates, rows are aggregated to one row
    per unit before testing (tt.aggr) — analyzing them raw understates
    variance. When `unit` is None and the frame has no obvious unit column,
    rows are assumed iid and the result says so.
    """
```

`AnalyzeResult` grows `unit_of_analysis: str | None` and
`rows_per_unit: float` so a reader can see whether iid was assumed or
enforced. If `rows_per_unit > 1` and no `unit` was passed, **raise**
rather than silently return an optimistic p-value — a `ModelRetry` the
agent can recover from by naming the unit.

Replace `_SESSION_CONTAMINATION` with a value derived from observed
sessions-per-user where data exists; where it doesn't, keep a constant but
surface it as a stated assumption in `SimulationResult.warnings` with a
sensitivity range, not a silent multiplier.

### A4 — `check_novelty` (`stats.py:83-94`)

Three defects: `early.lift > late.lift + 0.02` is a magic threshold on two
point estimates with no uncertainty (two noisy halves cross it routinely);
`midpoint = df.day.max() // 2` splits by day *value* rather than equal
exposure, so unequal daily volume gives unbalanced halves; and it uses
calendar day where novelty decays on *days since first exposure*.

Contract: fit the effect across time and test the trend.

```python
@dataclass(frozen=True)
class NoveltyResult:
    decaying: bool          # slope significantly negative, not a threshold
    slope: float            # change in lift per day
    slope_ci: tuple[float, float]
    daily_lifts: tuple[tuple[int, float], ...]   # for the report page
    cohort_day: bool        # True if measured on days-since-exposure
```

Implementation: per-day `analyze()`, then a weighted least-squares fit of
lift on day (weights = inverse variance), reporting the slope CI.
`decaying` is `slope_ci[1] < 0`. Prefer a `cohort_day` column when one
exists; when only calendar `day` exists, set `cohort_day=False` and say so
in the result — the interpretation genuinely differs and the reader needs
to know which one they got. `sim.py:139` assigns `day` uniformly at
random, so simulated cases are calendar-day; that's fine for the suite but
must be documented as the assumption it is.

### A5 — `sequential` is not sequential (`stats.py:49-55`)

Today: Bonferroni over a `looks` count the caller asserts, applied once to
the final frame. Nothing verifies the same alpha was spent at earlier
looks, and power collapses at many looks. If peeking is a first-class flaw
class in the simulator, the defense should be genuinely always-valid.

Contract — two named methods, chosen explicitly, never silently:

```python
class SequentialMethod(StrEnum):
    BONFERRONI = "bonferroni"        # conservative, transparent, kept as fallback
    OBRIEN_FLEMING = "obrien_fleming"  # Lan-DeMets alpha spending
    ALWAYS_VALID = "always_valid"      # mSPRT / empirical-Bernstein confidence sequence


@dataclass(frozen=True)
class SequentialResult:
    reject: bool
    method: SequentialMethod
    p_value: float
    alpha_spent: float
    information_fraction: float   # n_observed / n_planned — required for spending
    looks_taken: int
```

`information_fraction` requires the planned sample size, i.e. the spec
(A13). Alpha spending without it is not well defined, which is exactly why
the current version had to fake it with a caller-asserted `looks`.
`ALWAYS_VALID` is the right default for an agent that reads data whenever
it's asked, because it needs no look schedule at all.

### A6 — `check_srm` (`stats.py:20-30`)

`tt.SampleRatio()` defaults to `ratio=1`, hardcoding a 50/50 split — any
90/10 holdout or unequal three-arm test is reported INVALID forever. It
also assumes exactly two arms (`result.control` / `result.treatment`).

```python
def check_srm(
    df: pd.DataFrame,
    *,
    ratio: float | dict[Hashable, float] = 1.0,   # from the spec
    alpha: float = 0.001,
    by: tuple[str, ...] = (),   # ("segment",), ("day",) — where SRM actually hides
) -> SRMResult
```

`SRMResult.counts` becomes a mapping over arms rather than two fields, and
gains `by_group: tuple[tuple[str, bool], ...]` for per-stratum results.

The `by` argument is the substantive addition: a global SRM pass with a
per-segment SRM failure is the actual signature of the `SIMPSON` flaw in
`sim.py:41-61`, where the effect is created by dropping units
differentially by arm *within* segment. The current global-only check
cannot see it, and `scan_segments` (A7) tries to infer it from sign flips
instead — which is the wrong instrument for it.

### A7 — `scan_segments` reversal is noise (`stats.py:74-79`)

Sign comparison on raw point estimates: a segment at −0.001 against an
overall +0.001 is flagged as a reversal. Also `np.sign(...)` in a boolean
context is a truthiness test on a float, which reads as a guard against
zero but isn't a clear one.

Contract:

```python
@dataclass(frozen=True)
class SegmentScanResult:
    winners: tuple[str, ...]       # FWER-controlled, as today
    reversal: bool                 # segment CI excludes 0 on the opposite side
    heterogeneity_p: float         # Cochran's Q across segments
    composition_srm: tuple[tuple[str, bool], ...]   # per-segment SRM (A6)
```

A reversal now requires the segment's CI to exclude zero in the direction
opposite the overall effect — not merely a differing sign. Cochran's Q
answers the real question ("do these segments share one effect?") in a
single test rather than by eyeballing a scan. Keep Bonferroni/Šidák for
`winners` (`tt.adjust_fwer`), but the wider family of secondary results
should be FDR-controlled with `tt.adjust_fdr` — both ship with
tea-tasting and neither is currently used anywhere in the codebase.

### A8 — Control arm is inferred, never declared

`tt.Experiment.analyze(data, control=None)` is called without `control` in
all four call sites (`stats.py:22, 39, 108`). tea-tasting therefore picks
the control arm by ordering. This is correct for the current boolean
`arm` column and silently inverts the sign of every reported lift for a
frame encoding arms as `"A"/"B"` or with the treatment sorting first.

Pass `control=` explicitly everywhere; take the value from the spec, and
default to `False` to preserve today's behaviour. One line per call site,
and it removes an entire class of silent sign inversion that no existing
test would catch.

### A9 — CUPED validity is unverified (`stats.py:33-46`)

Nothing enforces that the covariate is pre-treatment. A post-treatment
covariate biases the estimate toward zero while *tightening* the interval
— it looks like a better result. Require the covariate to be declared
pre-period (spec field, A13) and refuse an undeclared one. Report
`theta`, `covariate_correlation` and `variance_reduction` on
`AnalyzeResult` so the adjustment is auditable rather than magic.

### A10 — `simulate_design` validates the wrong design (`design.py:33-60`)

The simulator hardcodes `_ALPHA = 0.05`, ignores CUPED, ignores looks,
ignores guardrails and ignores clustering. If the real analysis is
Bonferroni over 14 looks, realized power is nowhere near what this reports
— the design gate passes a design that will not be run.

Thread the analysis plan through: `simulate_design(spec, ...)` should use
`spec.alpha`, apply `spec.cuped`, and apply the sequential method at the
planned look schedule.

Second defect: `power = rejections / runs` with `runs=500` has a binomial
SE of ≈0.018. "Realized 0.78 against promised 0.80" is Monte Carlo noise,
so a gate built on it is itself unsound.

```python
@dataclass(frozen=True)
class SimulationResult:
    power: float
    power_ci: tuple[float, float]   # binomial CI on the estimate
    runs: int
    promised: float
    shortfall: bool     # power_ci[1] < promised — a real miss, not noise
    warnings: tuple[str, ...]
    mean_estimate: float
    bias: float         # mean_estimate - planted lift
```

`shortfall` (not the raw comparison) is what any gate should read.

### A11 — Traffic feasibility (`design.py:63-76`)

`_warnings()` checks weeks, guardrails and a flat-result decision, but
never reconciles `n_per_arm`, `days` and available traffic. "You need 340k
users per arm and the site gets 6k a week" is the most common real design
failure and is currently undetectable. Add `daily_traffic` to the spec
(or `ask` for it) and warn when `n_per_arm * 2 > daily_traffic * days`,
reporting the days actually required.

Also `_MAX_PLAUSIBLE_MDE = 0.25 → Verdict.INVALID` (`design.py:58`) is a
policy heuristic emitting a statistical verdict. Demote it to a
`implausible_mde` flag on the result; let the protocol decide what to do
with it.

### A12 — Missing metric machinery

- **Ratio metrics.** Revenue-per-visitor and friends need the delta
  method — `tt.RatioOfMeans(numer, denom)`. Currently every metric goes
  through `tt.Mean("converted")`, so any ratio metric is unavailable or,
  worse, approximated wrongly by hand in the sandbox.
- **Heavy tails.** A t-test on revenue with modest `n` is unreliable;
  `tt.Bootstrap(cols, statistic, method="bca")` is already available.
- **Outliers.** `OUTLIERS` is a first-class flaw class in `sim.py:75` but
  no tool implements a winsorization or trimming policy — the agent is
  expected to write it in the sandbox each time. Add an explicit,
  parameterised policy so the decision is recorded rather than improvised.
- **`alpha` is hardcoded in three places**: `_SRM_ALPHA = 0.001`
  (`stats.py:16`), `0.05 / looks` (`stats.py:53`), `alpha: float = 0.01`
  (`stats.py:107`), `_ALPHA = 0.05` (`design.py:12`). All become
  parameters sourced from the spec.

### A13 — Bind the review to a pre-registration *(systemic)*

`review()` takes a `Case` and nothing else (`agent.py:198`). Nothing
forces the analysis to be the one declared before anyone looked at the
data, which is the thing that actually separates a defensible readout from
a well-tooled fishing trip. It is also the prerequisite for A2 (needs the
MDE), A5 (needs planned `n`), A6 (needs the ratio), A9 (needs the declared
covariate) and A12 (needs alpha).

```python
def review(
    self,
    case: Any,
    spec: DesignSpec | None = None,
    question: str = "Review this experiment.",
) -> Trajectory
```

`DesignSpec` grows `alpha`, `covariate`, `looks`, `sequential_method`,
`control_value`, `daily_traffic`, and `guardrails: tuple[Guardrail, ...]`.

Add a `check_preregistration(df, spec) -> PreregResult` tool that compares
observed data against the declared plan — realized `n` against planned,
arms present against declared, metric column present, days run against
planned — and reports each deviation. A deviation is a first-class
finding, not a warning buried in prose; stopping early is exactly how a
null becomes a "win."

When `spec is None`, the review runs in an explicitly degraded mode that
must be stated in the output: no equivalence testing, no alpha spending,
default alpha. It should not silently look like a pre-registered analysis.

### A14 — No-invented-numbers checker *(systemic)*

The trajectory already records every tool call and result
(`agent.py:139-152`). That makes a mechanical hallucination guard
available almost for free: every numeric literal in `traj.answer` should
appear in some tool result, within a rounding tolerance.

```python
def unsupported_numbers(traj: Trajectory) -> tuple[str, ...]:
    """Numeric literals in the answer that no tool call produced."""
```

Wire it beside the existing SRM redaction (`agent.py:233-237`), which is
already an enforcement point of exactly this kind. On a violation, prefer
`ModelRetry` over silent redaction so the model gets a chance to fix its
own write-up; redact only if it fails twice.

This is the direct answer to the phase-3 open question — *"how is data
used/generated to run, given that I didn't give anything and the analysis
was done?"* Two contributors, both fixable here:

1. `ask()` (`tools.py:82-85`, and identically `mcp.py:97-101`) returns
   *"No answer available yet; proceed on your best judgement."* That
   string trains the model to invent an answer to a question it was told
   to ask a human. In interactive mode (`peer chat` / `peer web`) it must
   route to the actual user and block; only the eval harness should get a
   canned reply, and that reply should be a refusal to proceed.
2. `design()` will happily emit a fully-specified pre-registration from a
   brief containing no data and no baseline. Require the baseline to come
   either from a user-supplied number or from a tool computed over real
   data, and label any spec built on an assumed baseline as provisional in
   `DesignSpec` itself.

Related and cheap: `_VERDICT_RE` (`agent.py:32`) parses the verdict out of
free text. Replace with a pydantic `output_type=ReviewResult` binding
verdict, findings and cited tool-call evidence into one validated object.

---

## Track B — Databricks source layer

`mcp.py`'s tools already take a `path` and the stats functions are already
pure over frames, so most of the work is a resolver, not a rewrite.

### B1 — `sources.py`

```python
def resolve(uri: str) -> nw.LazyFrame:
    """
    parquet://path/to/file.parquet
    databricks://catalog.schema.table
    sql:SELECT ...            (read-only, validated)
    """
```

Every MCP tool's `path: str` becomes `source: str` and calls `resolve()`.
`stats.py` and `design.py` are untouched by this track — they keep taking
frames, which is what makes the layering worth having.

### B2 — Pushdown, not extraction

The important design decision: **do not pull rows into pandas.**
`tea_tasting.aggr.read_aggregates(data, group_col, *, has_count,
mean_cols, var_cols, cov_cols)` accepts any narwhals-compatible frame and
computes from sufficient statistics — counts, means, variances and
covariances per arm. Those aggregate in SQL. `tt.Experiment.analyze()`
accepts `dict[Hashable, Aggregates]` directly in place of a frame.

So the pipeline is: Ibis expression against the Databricks SQL warehouse →
narwhals → `read_aggregates` → the existing tests running locally on a few
dozen numbers. A 200M-row experiment costs the same as a 200k-row one, and
this is the only version that works against production tables at all.

Constraint to verify early: metrics that need row-level data
(`tt.Bootstrap`, `tt.Quantile`, `tt.MannWhitneyU`, and the A12
winsorization policy) are `MetricBaseGranular` — they cannot run from
aggregates. Those need either a sampled extract or an in-warehouse
approximation, and the tool should say which it used rather than quietly
downsampling.

### B3 — Access safety

Credentials from environment only (PAT or OAuth M2M), never as a tool
argument — a model must not be able to pass a token, and tokens must not
land in `logs/runs.jsonl`, which records every tool call's arguments
verbatim (`agent.py:88-101`). Read-only warehouse, statement timeout, row
cap, and rejection of anything that isn't a single `SELECT`. The existing
`Sandbox` already runs `--network none`, so sandboxed code cannot reach
the warehouse — keep that property; the source layer is the only path to
data.

### B4 — Discovery as a resource

`mcp.py` already serves `peer://protocol` as a resource
(`mcp.py:104-107`). Add `peer://catalog` listing Unity Catalog tables the
warehouse can see, so the model can discover experiment tables without
burning a tool call, and add `peer://schema/{table}` for column types.

### B5 — Packaging

Run as a Databricks App so it executes next to the data under UC-scoped
permissions, rather than shipping warehouse credentials to a laptop.

### Scope warning — worth deciding before building B

A generic "Databricks connector MCP" is commodity: Databricks ships
managed MCP servers for UC functions and Genie spaces. Building that is
building someone else's product. The defensible thing here is the
**experiment-review MCP that happens to read Databricks** — the value is
Track A encoded as tools a model cannot skip, and the connection is a
detail. Keep `check_srm` / `analyze` / guardrails as the public surface
and resist growing a general-purpose SQL tool, which would also hand the
model a way around every protocol in Track A.

---

## Dependency changes

- `ibis-framework[databricks]` **or** `databricks-sql-connector` (B1/B2).
  `narwhals` 2.25 is already present transitively via tea-tasting and
  supports lazy backends; `ibis` is not installed. Prefer Ibis for the
  aggregate pushdown — it composes with narwhals rather than requiring
  hand-written SQL per tool.
- `databricks-sdk` for Unity Catalog listing (B4) and OAuth (B3).
- No new dependency for Track A: `scipy` covers Cochran's Q, WLS and
  alpha spending, and tea-tasting already ships `adjust_fdr`,
  `adjust_fwer`, `RatioOfMeans`, `Bootstrap`, `Quantile`, `Proportion`
  and `aggr` — none of which the codebase currently uses.

## Verification

```bash
# per-item unit tests, offline
uv run pytest tests/ -m "not live and not docker" -q

# the calibration claims — new, and the point of track A
uv run pytest tests/test_phase_4_calibration.py -q

# re-author the baseline after A1/A2 land, then confirm the gate is real
uv run python -m peer_agent eval --out evals/out
uv run python -m peer_agent eval --gate

# track B, needs credentials — mark them live
uv run pytest tests/test_phase_4_sources.py -m live -q

uv run prek
uv run ty check
```

`tests/test_phase_4_calibration.py` is the new artifact that makes Track A
checkable rather than merely argued. It should assert the properties, not
the outputs:

- **False-positive rate.** Under a true null, across many seeds, each
  check rejects at approximately its nominal alpha. This is the test that
  would have caught A4's magic threshold and A7's sign comparison.
- **Guardrail non-inferiority.** A −7% guardrail move with a wide CI is
  `INCONCLUSIVE` and blocks — the A1 worked example above, frozen as a
  regression test.
- **Clustering.** Duplicating every row must not shrink the p-value
  (A3). One assertion, and it fails loudly today.
- **Sequential.** Peeking daily under a true null rejects at ≤ alpha
  across the whole look sequence, not per look (A5).
- **Power calibration.** `simulate_design`'s realized power covers the
  promise within Monte Carlo error (A10).

## Open/risky items — flagged rather than guessed

- **A2 changes ground truth in `evals/suite.yaml`.** Some `lift: 0.0`
  cases become INCONCLUSIVE rather than NO_EFFECT unless `n` rises enough
  for the CI to exclude the MDE. Whether to raise `n` or relabel is a
  per-case judgement about what the suite is testing, and should be
  decided case by case with the reasoning recorded — not resolved by a
  global find-and-replace.
- **A5's always-valid method is a real statistical choice**, not a
  refactor. mSPRT needs a prior on the effect size; empirical-Bernstein
  confidence sequences need a variance bound. Both are defensible, they
  trade power differently, and picking one without benchmarking against
  the peeking cases would be guessing.
- **A3 may have no unit column to enforce against.** `sim.py` generates
  one row per unit, so simulated data is iid by construction and the
  clustering fix is untestable against the current suite. It needs a new
  simulator flaw (repeated users) to be verifiable at all — otherwise A3
  ships as untested code that only matters once Track B lands, which is
  the worst possible ordering.
- **A13 is a breaking API change** to `review()`, `mcp.py`'s tool
  schemas, both SKILL.md protocols, and every eval trajectory in
  `fixtures/recorded.jsonl`. Land it as one changeset, not incrementally.
- **B2's aggregate pushdown is unverified against a real warehouse.**
  narwhals' Databricks/Ibis support is the load-bearing assumption of the
  whole track; confirm `read_aggregates` works against a live SQL
  warehouse *before* building B3–B5 on top of it.
- **Track B's cost model is unknown.** Every tool call becomes a
  warehouse query, and an agent that retries is an agent that bills. Needs
  a query budget per review, and probably a cache keyed on
  (source, filters), before it points at anything expensive.

### Critical files

Track A:
- `src/peer_agent/stats.py` (A1, A4–A9, A12 — the bulk of the work)
- `src/peer_agent/types.py` (A1, A2, A13 — new verdict, `Guardrail`, spec fields)
- `src/peer_agent/design.py` (A10, A11)
- `src/peer_agent/agent.py` (A13, A14 — `review()` signature, structured output)
- `src/peer_agent/tools.py`, `src/peer_agent/mcp.py` (`ask()`, schema churn)
- `.agents/skills/{review,design}-protocol/SKILL.md` (the protocol must
  change with the tools, or the model follows the old one)
- `evals/suite.yaml`, `fixtures/recorded.jsonl`, `evals/baseline.json` (re-authored)
- `tests/test_phase_4_calibration.py` (new)

Track B:
- `src/peer_agent/sources.py` (new)
- `src/peer_agent/mcp.py` (`path` → `source`, catalog resources)
- `tests/test_phase_4_sources.py` (new, `-m live`)
- `pyproject.toml` (ibis / databricks-sdk)
