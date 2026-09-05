from __future__ import annotations

import numpy as np
import pandas as pd
import tea_tasting as tt

from peer_agent.types import (
    AnalyzeResult,
    GuardrailResult,
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


def check_srm(
    df: pd.DataFrame, *, control: bool | int | str = CONTROL, alpha: float = SRM_ALPHA
) -> SRMResult:
    """Check whether the traffic split matches the assigned ratio."""
    result = tt.Experiment({"sample_ratio": tt.SampleRatio()}, variant="arm").analyze(
        df, control
    )["sample_ratio"]
    return SRMResult(
        mismatch=result.pvalue < alpha,
        p_value=result.pvalue,
        n_control=result.control,
        n_treatment=result.treatment,
    )


def analyze(  # noqa: PLR0913 (each argument is one pre-registered choice)
    df: pd.DataFrame,
    *,
    cuped: bool = False,
    covariate: str | None = None,
    control: bool | int | str = CONTROL,
    alpha: float = ALPHA,
) -> AnalyzeResult:
    """Run the primary conversion-rate test, treatment vs. control."""
    covariate = covariate or (_CUPED_COVARIATE if cuped else None)
    # tea-tasting's own `alpha` is power-analysis only; the interval is set by
    # confidence_level, so the pre-registered alpha has to arrive that way.
    metric = tt.Mean("converted", covariate, confidence_level=1 - alpha)
    result = tt.Experiment({"conversion": metric}, variant="arm").analyze(df, control)[
        "conversion"
    ]
    return AnalyzeResult(
        p_value=result.pvalue,
        lift=result.rel_effect_size,
        ci_low=result.rel_effect_size_ci_lower,
        ci_high=result.rel_effect_size_ci_upper,
        width=result.rel_effect_size_ci_upper - result.rel_effect_size_ci_lower,
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
    """Check each segment for a significant effect or a sign reversal."""
    if "segment" not in df.columns:
        raise ValueError("scan_segments needs a 'segment' column")
    overall = analyze(df, control=control, alpha=alpha)
    segments = sorted(df.segment.unique())
    per_segment = alpha / len(segments)  # Bonferroni over the family, not per-segment FDR
    winners: list[str] = []
    reversal = False
    for segment in segments:
        subset = df.loc[df.segment == segment]
        if subset.arm.nunique() < 2:
            continue
        result = analyze(subset, control=control, alpha=alpha)
        if result.p_value < per_segment:
            winners.append(str(segment))
        if (
            np.sign(result.lift)
            and np.sign(overall.lift)
            and np.sign(result.lift) != np.sign(overall.lift)
        ):
            reversal = True
    return SegmentScanResult(winners=tuple(winners), reversal=reversal)


def check_novelty(
    df: pd.DataFrame, *, control: bool | int | str = CONTROL
) -> NoveltyResult:
    """Check whether an early effect is decaying over time."""
    if "day" not in df.columns:
        raise ValueError("check_novelty needs a 'day' column")
    midpoint = df.day.max() // 2
    early = analyze(df.loc[df.day <= midpoint], control=control)
    late = analyze(df.loc[df.day > midpoint], control=control)
    return NoveltyResult(
        decaying=early.lift > late.lift + 0.02,
        early_lift=early.lift,
        late_lift=late.lift,
    )


def check_guardrails(
    df: pd.DataFrame,
    guardrails: tuple[str, ...] = (),
    *,
    control: bool | int | str = CONTROL,
    alpha: float = GUARDRAIL_ALPHA,
) -> GuardrailResult:
    """Check whether any named guardrail metric regressed."""
    # Provisional: not exercised by any in-scope test. A guardrail "breaches" if
    # it has its own column and treatment moves it down significantly.
    breached = tuple(
        name
        for name in guardrails
        if name in df.columns and _regressed(df, name, control=control, alpha=alpha)
    )
    return GuardrailResult(breached=breached)


def _regressed(
    df: pd.DataFrame,
    column: str,
    *,
    control: bool | int | str = CONTROL,
    alpha: float = GUARDRAIL_ALPHA,
) -> bool:
    result = tt.Experiment({column: tt.Mean(column)}, variant="arm").analyze(df, control)[
        column
    ]
    return result.pvalue < alpha and result.rel_effect_size < 0
