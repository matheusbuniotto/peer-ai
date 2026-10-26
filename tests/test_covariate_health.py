"""Covariate health screen for CUPED / outlier flaws."""

from __future__ import annotations

from peer_agent import stats
from peer_agent.sim import OUTLIERS, make_case


def test_outliers_flaw_flags_the_covariate():
    df = make_case(n=20_000, lift=0.0, seed=1, flaws=[OUTLIERS]).df
    result = stats.check_covariate_health(df)
    assert result.suspicious
    assert result.cuped_unsafe
    assert result.n_extreme >= 1


def test_clean_covariate_is_not_suspicious():
    df = make_case(n=10_000, lift=0.0, seed=2).df
    result = stats.check_covariate_health(df)
    assert not result.suspicious
    assert not result.cuped_unsafe
