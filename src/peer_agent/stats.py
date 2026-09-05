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

_SRM_ALPHA = 0.001
_CUPED_COVARIATE = "pre_period_metric"


def check_srm(df: pd.DataFrame) -> SRMResult:
    """Check whether the traffic split matches the assigned ratio."""
    result = tt.Experiment({"sample_ratio": tt.SampleRatio()}, variant="arm").analyze(df)[
        "sample_ratio"
    ]
    return SRMResult(
        mismatch=result.pvalue < _SRM_ALPHA,
        p_value=result.pvalue,
        n_control=result.control,
        n_treatment=result.treatment,
    )


def analyze(
    df: pd.DataFrame, *, cuped: bool = False, covariate: str | None = None
) -> AnalyzeResult:
    """Run the primary conversion-rate test, treatment vs. control."""
    covariate = covariate or (_CUPED_COVARIATE if cuped else None)
    metric = tt.Mean("converted", covariate) if covariate else tt.Mean("converted")
    result = tt.Experiment({"conversion": metric}, variant="arm").analyze(df)["conversion"]
    return AnalyzeResult(
        p_value=result.pvalue,
        lift=result.rel_effect_size,
        ci_low=result.rel_effect_size_ci_lower,
        ci_high=result.rel_effect_size_ci_upper,
        width=result.rel_effect_size_ci_upper - result.rel_effect_size_ci_lower,
    )


def sequential(df: pd.DataFrame, looks: int) -> SequentialResult:
    """Bonferroni-corrected look: conservative, transparent, easy to defend."""
    result = analyze(df)
    alpha = 0.05 / looks
    return SequentialResult(
        reject=result.p_value < alpha, p_value=result.p_value, alpha=alpha
    )


def scan_segments(df: pd.DataFrame) -> SegmentScanResult:
    """Check each segment for a significant effect or a sign reversal."""
    if "segment" not in df.columns:
        raise ValueError("scan_segments needs a 'segment' column")
    overall = analyze(df)
    segments = sorted(df.segment.unique())
    alpha = 0.05 / len(segments)  # Bonferroni over the family, not per-segment FDR
    winners: list[str] = []
    reversal = False
    for segment in segments:
        subset = df.loc[df.segment == segment]
        if subset.arm.nunique() < 2:
            continue
        result = analyze(subset)
        if result.p_value < alpha:
            winners.append(str(segment))
        if (
            np.sign(result.lift)
            and np.sign(overall.lift)
            and np.sign(result.lift) != np.sign(overall.lift)
        ):
            reversal = True
    return SegmentScanResult(winners=tuple(winners), reversal=reversal)


def check_novelty(df: pd.DataFrame) -> NoveltyResult:
    """Check whether an early effect is decaying over time."""
    if "day" not in df.columns:
        raise ValueError("check_novelty needs a 'day' column")
    midpoint = df.day.max() // 2
    early = analyze(df.loc[df.day <= midpoint])
    late = analyze(df.loc[df.day > midpoint])
    return NoveltyResult(
        decaying=early.lift > late.lift + 0.02,
        early_lift=early.lift,
        late_lift=late.lift,
    )


def check_guardrails(df: pd.DataFrame, guardrails: tuple[str, ...] = ()) -> GuardrailResult:
    """Check whether any named guardrail metric regressed."""
    # Provisional: not exercised by any in-scope test. A guardrail "breaches" if
    # it has its own column and treatment moves it down significantly.
    breached = tuple(
        name for name in guardrails if name in df.columns and _regressed(df, name)
    )
    return GuardrailResult(breached=breached)


def _regressed(df: pd.DataFrame, column: str, alpha: float = 0.01) -> bool:
    result = tt.Experiment({column: tt.Mean(column)}, variant="arm").analyze(df)[column]
    return result.pvalue < alpha and result.rel_effect_size < 0
