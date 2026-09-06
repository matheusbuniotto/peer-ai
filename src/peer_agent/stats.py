from __future__ import annotations

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
) -> AnalyzeResult:
    """
    Run the primary conversion-rate test, treatment vs. control.

    Given the pre-registered `mde`, also answer the question a p-value can't:
    is the effect small enough to call this a real null? That's two one-sided
    tests, so the interval it needs is the 1-2*alpha one, not the reported one.
    """
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
    )


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


def sequential(
    df: pd.DataFrame,
    looks: int,
    *,
    control: bool | int | str = CONTROL,
    alpha: float = ALPHA,
) -> SequentialResult:
    """Bonferroni-corrected look: conservative, transparent, easy to defend."""
    result = analyze(df, control=control, alpha=alpha)
    spent = alpha / looks
    return SequentialResult(
        reject=result.p_value < spent, p_value=result.p_value, alpha=spent
    )


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
        (int(day), analyze(part, control=control, alpha=alpha))
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
