from __future__ import annotations

import pandas as pd
import tea_tasting as tt

from peer_agent.types import AnalyzeResult, SRMResult

_SRM_ALPHA = 0.001


def check_srm(df: pd.DataFrame) -> SRMResult:
    result = tt.Experiment({"sample_ratio": tt.SampleRatio()}, variant="arm").analyze(df)[
        "sample_ratio"
    ]
    return SRMResult(
        mismatch=result.pvalue < _SRM_ALPHA,
        p_value=result.pvalue,
        n_control=result.control,
        n_treatment=result.treatment,
    )


def analyze(df: pd.DataFrame) -> AnalyzeResult:
    result = tt.Experiment({"conversion": tt.Mean("converted")}, variant="arm").analyze(df)[
        "conversion"
    ]
    return AnalyzeResult(
        p_value=result.pvalue,
        lift=result.rel_effect_size,
        ci_low=result.rel_effect_size_ci_lower,
        ci_high=result.rel_effect_size_ci_upper,
    )
