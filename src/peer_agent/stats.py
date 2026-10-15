from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
import pandas as pd
import tea_tasting as tt
from scipy import stats as sps
from tea_tasting.metrics.mean import MeanResult
from tea_tasting.metrics.proportion import SampleRatioResult

from peer_agent.types import (
    AnalyzeResult,
    Guardrail,
    GuardrailResult,
    GuardrailStatus,
    NoveltyResult,
    SegmentScanResult,
    SequentialResult,
    SRMResult,
)

SRM_ALPHA = 0.001
ALPHA = 0.05
GUARDRAIL_ALPHA = 0.01
CONTROL: bool | int | str = False  # the `arm` value meaning control, absent a spec
_CUPED_COVARIATE = "pre_period_metric"
# Columns that name a randomisation unit if the frame happens to carry one.
_UNIT_COLUMNS = ("unit_id", "user_id", "visitor_id")


def _measure(
    metric: tt.Mean, df: pd.DataFrame, control: bool | int | str, key: str
) -> MeanResult:
    """
    tea-tasting types Experiment.analyze()[...] as the MetricResult protocol,
    which carries none of the fields every caller here reads. One cast, in one
    place, instead of the same unchecked attribute access in six.
    """
    return cast(
        MeanResult, tt.Experiment({key: metric}, variant="arm").analyze(df, control)[key]
    )


def check_srm(  # noqa: PLR0913 (each argument is one pre-registered choice)
    df: pd.DataFrame,
    *,
    ratio: float = 1.0,
    control: bool | int | str = CONTROL,
    alpha: float = SRM_ALPHA,
    by: tuple[str, ...] = (),
) -> SRMResult:
    """
    Check the split against the ratio the design actually asked for.

    `ratio` is treatment per unit of control. It used to be pinned at 1, which
    reported every 90/10 holdout as broken forever.

    `by` re-runs the check inside each level of a column. A split that balances
    overall while tilting inside a segment is how Simpson's paradox arrives, and
    the global check is blind to it by construction.
    """
    overall = _split_p_value(df, ratio, control)
    per_stratum = [
        (f"{column}={level}", _split_p_value(part, ratio, control))
        for column in by
        for level, part in df.groupby(column, observed=True)
        if part.arm.nunique() > 1
    ]
    # Bonferroni over the strata: twenty segments each judged at 0.001 is a 2%
    # chance of condemning a healthy split.
    stratum_alpha = alpha / max(len(per_stratum), 1)
    strata = tuple((name, p < stratum_alpha) for name, p in per_stratum)
    return SRMResult(
        mismatch=overall < alpha or any(broken for _, broken in strata),
        p_value=overall,
        counts=tuple((str(arm), int(n)) for arm, n in df.arm.value_counts().items()),
        strata=strata,
    )


def _split_p_value(df: pd.DataFrame, ratio: float, control: bool | int | str) -> float:
    """Bonferroni-adjusted across arms, so a four-arm test isn't three chances to fail."""
    experiment = tt.Experiment({"sample_ratio": tt.SampleRatio(ratio)}, variant="arm")
    if df.arm.nunique() <= 2:
        result = cast(SampleRatioResult, experiment.analyze(df, control)["sample_ratio"])
        return result.pvalue
    against_control = experiment.analyze(df, control, all_variants=True)
    p_values = [
        cast(SampleRatioResult, r["sample_ratio"]).pvalue for r in against_control.values()
    ]
    return min(min(p_values) * len(p_values), 1.0)


def analyze(  # noqa: PLR0913 (each argument is one pre-registered choice)
    df: pd.DataFrame,
    *,
    cuped: bool = False,
    covariate: str | None = None,
    control: bool | int | str = CONTROL,
    alpha: float = ALPHA,
    mde: float | None = None,
    unit: str | None = None,
) -> AnalyzeResult:
    """
    Run the primary conversion-rate test, treatment vs. control.

    `unit` names the column the experiment randomised on. Rows are not evidence;
    units are. If a user shows up on four rows, testing the rows understates the
    variance and shrinks the p-value for free, so the rows are collapsed to one
    row per unit first. When the frame plainly has repeat units and nobody said
    which column identifies them, this refuses rather than quietly answering a
    question about rows.

    Given the pre-registered `mde`, also answer the question a p-value can't:
    is the effect small enough to call this a real null? That's two one-sided
    tests, so the interval it needs is the 1-2*alpha one, not the reported one.
    """
    df, rows_per_unit = _one_row_per_unit(df, unit)
    covariate = covariate or (_CUPED_COVARIATE if cuped else None)
    # tea-tasting's own `alpha` is power-analysis only; the interval is set by
    # confidence_level, so the pre-registered alpha has to arrive that way.
    metric = tt.Mean("converted", covariate, confidence_level=1 - alpha)
    result = _measure(metric, df, control, "conversion")
    return AnalyzeResult(
        p_value=result.pvalue,
        lift=result.rel_effect_size,
        ci_low=result.rel_effect_size_ci_lower,
        ci_high=result.rel_effect_size_ci_upper,
        width=result.rel_effect_size_ci_upper - result.rel_effect_size_ci_lower,
        equivalent_to_null=_equivalent(df, covariate, control, alpha, mde),
        mde=mde,
        unit_of_analysis=unit or "row",
        rows_per_unit=rows_per_unit,
    )


def _one_row_per_unit(df: pd.DataFrame, unit: str | None) -> tuple[pd.DataFrame, float]:
    """Collapse repeat rows onto the unit the experiment actually randomised."""
    if unit is None:
        for candidate in _UNIT_COLUMNS:
            if candidate in df.columns and df[candidate].nunique() < len(df):
                raise ValueError(
                    f"'{candidate}' repeats across rows, so these rows are not "
                    f"independent — call analyze with unit='{candidate}'"
                )
        return df, 1.0

    units = df[unit].nunique()
    grouped = df.groupby(unit, observed=True)
    if (grouped.arm.nunique() > 1).any():
        raise ValueError(f"some '{unit}' values appear in both arms — the split leaks")
    how: dict[str, str] = {
        column: "mean" if pd.api.types.is_numeric_dtype(dtype) else "first"
        for column, dtype in df.dtypes.items()
        if column != unit
    }
    how["arm"] = "first"
    return grouped.agg(how).reset_index(), len(df) / units


def _equivalent(
    df: pd.DataFrame,
    covariate: str | None,
    control: bool | int | str,
    alpha: float,
    mde: float | None,
) -> bool | None:
    """TOST: the 1-2*alpha interval sits entirely inside the indifference zone."""
    if mde is None:
        return None
    metric = tt.Mean("converted", covariate, confidence_level=1 - 2 * alpha)
    tost = _measure(metric, df, control, "conversion")
    bound = abs(mde)
    return bool(
        tost.rel_effect_size_ci_lower > -bound and tost.rel_effect_size_ci_upper < bound
    )


def obf_boundaries(info_fractions: Sequence[float], alpha: float = ALPHA) -> np.ndarray:
    """Two-sided Lan-DeMets O'Brien-Fleming boundaries on the z scale.

    The spending function is the one ldbounds uses for a symmetric two-sided
    test: ``4 (1 − Φ(z_{(α/2)/2} / √t))``. It spends almost nothing at the first
    look and reaches exactly ``alpha`` at ``t = 1``. Bounds are the multivariate
    normal exit thresholds with correlation ``√(t_i / t_j)``, which is what
    ``ldBounds(seq(0.2, 1, length=5))`` prints for α = 0.05:
    4.877, 3.357, 2.680, 2.290, 2.031.
    """
    fractions = _fractions(info_fractions)
    return np.asarray(_obf_cached(fractions, float(alpha)), dtype=float)


def sequential(  # noqa: PLR0913 (each knob is one pre-registered choice)
    df: pd.DataFrame,
    looks: int | None = None,
    *,
    info_fractions: Sequence[float] | None = None,
    control: bool | int | str = CONTROL,
    alpha: float = ALPHA,
    method: Literal["obf_lan_demets", "bonferroni"] = "obf_lan_demets",
) -> SequentialResult:
    """Group-sequential test. Default is Lan-DeMets OBF, not alpha/k.

    ``method="bonferroni"`` keeps the old rule: compare the final p-value with
    ``alpha / looks``. That is a correction, not an alpha-spending design.
    """
    if method == "bonferroni":
        return _bonferroni(df, looks or 1, control=control, alpha=alpha)
    if _observed_peek(df, looks, info_fractions):
        return _sequential_observed(df, control=control, alpha=alpha)
    fractions, frames = _look_frames(df, looks, info_fractions)
    bounds = obf_boundaries(fractions, alpha)
    zs = tuple(_z_stat(frame, control) for frame in frames)
    stopped = _first_cross(zs, bounds)
    z_at = zs[(stopped or len(zs)) - 1]
    return SequentialResult(
        reject=stopped is not None,
        p_value=float(2 * sps.norm.sf(abs(z_at))),
        alpha=alpha,
        boundaries=tuple(float(bound) for bound in bounds),
        z_by_look=tuple(float(z) for z in zs),
        stopped_at=stopped,
        method="obf_lan_demets",
    )


def _first_cross(zs: tuple[float, ...], bounds: np.ndarray) -> int | None:
    for index, (z, bound) in enumerate(zip(zs, bounds, strict=True)):
        if abs(z) >= bound:
            return index + 1
    return None


def _bonferroni(
    df: pd.DataFrame, looks: int, *, control: bool | int | str, alpha: float
) -> SequentialResult:
    result = analyze(df, control=control, alpha=alpha)
    spent = alpha / looks
    return SequentialResult(
        reject=result.p_value < spent,
        p_value=result.p_value,
        alpha=spent,
        method="bonferroni",
    )


def _fractions(info_fractions: Sequence[float]) -> tuple[float, ...]:
    fractions = tuple(float(t) for t in info_fractions)
    if not fractions or any(t <= 0 or t > 1 for t in fractions):
        raise ValueError("info fractions must lie in (0, 1]")
    ordered = all(
        later > earlier for earlier, later in zip(fractions, fractions[1:], strict=False)
    )  # noqa: E501
    if not ordered or abs(fractions[-1] - 1) > 1e-9:
        raise ValueError("info fractions must increase and end at 1")
    return fractions


@functools.lru_cache(maxsize=32)
def _obf_cached(fractions: tuple[float, ...], alpha: float) -> tuple[float, ...]:
    times = np.asarray(fractions, dtype=float)
    bounds: list[float] = []
    for k in range(1, len(times) + 1):
        target = 1 - _obf_spend(float(times[k - 1]), alpha)
        bounds.append(_bound_for(bounds, times[:k], target))
    return tuple(bounds)


def _obf_spend(t: float, alpha: float) -> float:
    """Cumulative type I error at information fraction ``t`` (ldbounds, two-sided)."""
    z = float(sps.norm.ppf(1 - (alpha / 2) / 2))
    return float(4 * (1 - sps.norm.cdf(z / np.sqrt(t))))


def _bound_for(earlier: list[float], times: np.ndarray, target_inside: float) -> float:
    lo, hi = 0.0, 20.0
    for _ in range(30):
        mid = (lo + hi) / 2
        inside = _inside(earlier + [mid], times)
        if inside < target_inside:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _inside(bounds: list[float], times: np.ndarray) -> float:
    upper = np.asarray(bounds, dtype=float)
    ratio = np.minimum(times[:, None], times[None, :]) / np.maximum(
        times[:, None], times[None, :]
    )  # noqa: E501
    corr = np.sqrt(ratio)
    return float(
        sps.multivariate_normal.cdf(
            upper,
            mean=np.zeros(len(upper)),
            cov=corr,
            lower_limit=-upper,
            allow_singular=True,  # noqa: E501
        )
    )


def _observed_peek(
    df: pd.DataFrame, looks: int | None, info_fractions: Sequence[float] | None
) -> bool:
    return (
        looks is None
        and info_fractions is None
        and "planned_days" in df.attrs
        and "day" in df.columns
    )


def _sequential_observed(
    df: pd.DataFrame, *, control: bool | int | str, alpha: float
) -> SequentialResult:
    """Compare each day that was actually seen with the boundary for that day
    on the plan, not with a boundary recomputed on the truncated sample."""
    planned = int(df.attrs["planned_days"])
    days = sorted(int(day) for day in df["day"].unique())
    # A boundary at day d depends only on the looks up to d, so the grid stops
    # at the last observed day instead of the full plan.
    fractions = tuple((day + 1) / planned for day in days)
    bounds = np.asarray(_obf_cached(fractions, float(alpha)))
    zs = tuple(_z_stat(df.loc[df["day"] <= day], control) for day in days)
    stopped = _first_cross(zs, bounds)
    z_at = zs[(stopped or len(zs)) - 1]
    return SequentialResult(
        reject=stopped is not None,
        p_value=float(2 * sps.norm.sf(abs(z_at))),
        alpha=alpha,
        boundaries=tuple(float(bound) for bound in bounds),
        z_by_look=tuple(float(z) for z in zs),
        stopped_at=stopped,
        method="obf_lan_demets",
    )


def _look_frames(
    df: pd.DataFrame, looks: int | None, info_fractions: Sequence[float] | None
) -> tuple[tuple[float, ...], list[pd.DataFrame]]:
    if info_fractions is None:
        k = looks or (int(df["day"].nunique()) if "day" in df.columns else 1)
        info_fractions = tuple(i / k for i in range(1, k + 1))
    fractions = _fractions(info_fractions)
    ordered = df.sort_values("day", kind="mergesort") if "day" in df.columns else df
    n = len(ordered)
    frames = [
        ordered.iloc[: max(1, min(n, int(np.ceil(t * n - 1e-12))))] for t in fractions
    ]  # noqa: E501
    return fractions, frames


def _z_stat(df: pd.DataFrame, control: bool | int | str) -> float:
    """Pooled two-proportion z. Nested prefixes then correlate as √(n_i / n_j)."""
    arm = df["arm"].to_numpy()
    converted = df["converted"].to_numpy().astype(float)
    control_rows = arm == control
    n_c = int(control_rows.sum())
    n_t = int((~control_rows).sum())
    if n_c == 0 or n_t == 0:
        return 0.0
    pooled = float(converted.mean())
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_c + 1 / n_t))
    if se == 0:
        return 0.0
    effect = float(converted[~control_rows].mean() - converted[control_rows].mean())
    return effect / se


def scan_segments(
    df: pd.DataFrame, *, control: bool | int | str = CONTROL, alpha: float = ALPHA
) -> SegmentScanResult:
    """
    Look for a segment that behaves differently from the whole.

    A reversal now needs the segment's interval to exclude zero on the far side
    of the overall effect, not merely a differing sign: under a true null every
    segment's sign is a coin flip, and comparing signs reported a reversal on
    80% of null datasets. Cochran's Q asks the real question — does one effect
    explain every segment? — in a single test instead of by eyeballing a scan.
    """
    if "segment" not in df.columns:
        raise ValueError("scan_segments needs a 'segment' column")
    overall = analyze(df, control=control, alpha=alpha)
    parts = {
        str(segment): analyze(subset, control=control, alpha=alpha)
        for segment, subset in df.groupby("segment", observed=True)
        if subset.arm.nunique() > 1
    }
    per_segment = alpha / max(len(parts), 1)  # Bonferroni over the family
    return SegmentScanResult(
        winners=tuple(name for name, r in parts.items() if r.p_value < per_segment),
        reversal=any(_opposes(r, overall.lift) for r in parts.values()),
        heterogeneity_p=_cochran_q(list(parts.values()), alpha),
        composition_srm=check_srm(df, control=control, by=("segment",)).strata,
    )


def _opposes(segment: AnalyzeResult, overall: float) -> bool:
    """A reversal is a segment whose interval clears zero on the far side."""
    return (segment.ci_high < 0 < overall) or (overall < 0 < segment.ci_low)


def _cochran_q(results: list[AnalyzeResult], alpha: float) -> float:
    """
    One effect behind every segment, or several? Q is the inverse-variance
    weighted spread of the segment effects, read against chi-square on k-1 df.
    Each segment's standard error is recovered from the interval it reported.
    """
    if len(results) < 2:
        return float("nan")
    z = sps.norm.ppf(1 - alpha / 2)
    errors = np.array([(r.ci_high - r.ci_low) / (2 * z) for r in results])
    if not np.all(np.isfinite(errors)) or np.any(errors <= 0):
        return float("nan")
    weights = 1 / errors**2
    lifts = np.array([r.lift for r in results])
    q = float(np.sum(weights * (lifts - np.average(lifts, weights=weights)) ** 2))
    return float(sps.chi2.sf(q, len(results) - 1))


def check_novelty(
    df: pd.DataFrame,
    *,
    control: bool | int | str = CONTROL,
    alpha: float = ALPHA,
) -> NoveltyResult:
    """
    Check whether the effect is fading as the experiment runs.

    Splitting the run in half and comparing two point estimates against a fixed
    0.02 gap called a perfectly steady +6% lift "decaying" on 42% of seeds — two
    noisy halves cross any fixed threshold routinely. This fits the daily lifts
    against time, weighted by how well each day is measured, and asks whether
    the slope is negative by more than its own uncertainty.

    Measured on `cohort_day` where the data has one, since novelty decays on
    days since a user's first exposure rather than on the calendar.
    """
    column = "cohort_day" if "cohort_day" in df.columns else "day"
    if column not in df.columns:
        raise ValueError("check_novelty needs a 'day' or 'cohort_day' column")

    daily = [
        (_day_index(day), analyze(part, control=control, alpha=alpha))
        for day, part in df.groupby(column, observed=True)
        if part.arm.nunique() > 1
    ]
    slope, half_width = _weighted_trend(daily, alpha)
    return NoveltyResult(
        decaying=slope + half_width < 0,
        slope=slope,
        slope_ci=(slope - half_width, slope + half_width),
        daily_lifts=tuple((day, result.lift) for day, result in daily),
        cohort_day=column == "cohort_day",
    )


def _day_index(day: object) -> int:
    if isinstance(day, int):
        return day
    return int(str(day))


def _weighted_trend(
    daily: list[tuple[int, AnalyzeResult]], alpha: float
) -> tuple[float, float]:
    """
    Inverse-variance weighted least squares of lift on day. Returns the slope
    and the half-width of its interval; a day measured badly pulls on the line
    less than a day measured well.
    """
    z = sps.norm.ppf(1 - alpha / 2)
    days = np.array([day for day, _ in daily], dtype=float)
    lifts = np.array([r.lift for _, r in daily])
    errors = np.array([(r.ci_high - r.ci_low) / (2 * z) for _, r in daily])

    usable = np.isfinite(errors) & (errors > 0) & np.isfinite(lifts)
    if usable.sum() < 3:
        return float("nan"), float("nan")

    days, lifts, weights = days[usable], lifts[usable], 1 / errors[usable] ** 2
    centred = days - np.average(days, weights=weights)
    spread = float(np.sum(weights * centred**2))
    if spread <= 0:
        return float("nan"), float("nan")
    slope = float(np.sum(weights * centred * lifts) / spread)
    return slope, float(z / np.sqrt(spread))


def check_guardrails(
    df: pd.DataFrame,
    guardrails: tuple[Guardrail | str, ...] = (),
    *,
    control: bool | int | str = CONTROL,
    alpha: float = GUARDRAIL_ALPHA,
) -> GuardrailResult:
    """
    Check whether any guardrail can still be moving against us by more than its
    margin.

    This is a non-inferiority question, not a superiority one. Asking "did it
    drop significantly?" fails open: a guardrail down 7% with an interval
    reaching -13% answers no, because an underpowered test can't reject, and
    ships the regression. So a guardrail is CLEAN only when harm past the margin
    is ruled out, BREACHED when harm past the margin is established, and
    INCONCLUSIVE otherwise — which blocks, exactly like a breach does.
    """
    statuses, bounds = [], []
    for guardrail in (_as_guardrail(g) for g in guardrails):
        status, bound = _judge(df, guardrail, control=control, alpha=alpha)
        statuses.append((guardrail.name, status))
        bounds.append((guardrail.name, bound))
    return GuardrailResult(
        statuses=tuple(statuses),
        bounds=tuple(bounds),
        blocks_ship=any(s is not GuardrailStatus.CLEAN for _, s in statuses),
    )


def _as_guardrail(guardrail: Guardrail | str) -> Guardrail:
    return Guardrail(name=guardrail) if isinstance(guardrail, str) else guardrail


def _judge(
    df: pd.DataFrame,
    guardrail: Guardrail,
    *,
    control: bool | int | str,
    alpha: float,
) -> tuple[GuardrailStatus, float]:
    """The status and the bound on the harmful side of the interval."""
    if guardrail.name not in df.columns:
        # A guardrail nobody measured is not a guardrail that passed.
        return GuardrailStatus.INCONCLUSIVE, float("nan")

    low, high = _one_sided_bounds(df, guardrail.name, control=control, alpha=alpha)
    down_is_bad = guardrail.direction == "down_is_bad"
    # Read every guardrail as though a fall were the harm, so one rule covers both.
    harmful, opposite = (low, high) if down_is_bad else (-high, -low)
    tolerated = -guardrail.margin

    if harmful > tolerated:
        status = GuardrailStatus.CLEAN
    elif opposite < tolerated:
        status = GuardrailStatus.BREACHED
    else:
        status = GuardrailStatus.INCONCLUSIVE
    return status, (low if down_is_bad else high)


def _one_sided_bounds(
    df: pd.DataFrame, column: str, *, control: bool | int | str, alpha: float
) -> tuple[float, float]:
    """Both one-sided bounds on the relative move, each at level `alpha`."""

    def side(alternative: Literal["greater", "less"]) -> MeanResult:
        metric = tt.Mean(column, alternative=alternative, confidence_level=1 - alpha)
        return _measure(metric, df, control, column)

    return (
        side("greater").rel_effect_size_ci_lower,
        side("less").rel_effect_size_ci_upper,
    )
