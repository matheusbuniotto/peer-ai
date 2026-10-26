"""Bootstrap, Bayesian, and Type M/S tools — ported from ab-test-analytics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps
from statsmodels.stats.proportion import proportions_ztest

from peer_agent import stats
from peer_agent.sim import make_case


def test_bootstrap_p_value_tracks_the_z_test_on_a_large_sample():
    df = make_case(n=80_000, lift=0.0, seed=11).df
    boot = stats.bootstrap_analyze(df, n_bootstrap=20_000, seed=11)
    freq = stats.analyze(df)
    assert abs(boot.p_value - freq.p_value) < 0.02
    assert abs(boot.lift - freq.lift) < 1e-9


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    df = make_case(n=20_000, lift=0.05, seed=3).df
    a = stats.bootstrap_analyze(df, seed=99)
    b = stats.bootstrap_analyze(df, seed=99)
    assert a.p_value == b.p_value
    assert a.ci_low == b.ci_low
    assert a.ci_high == b.ci_high


def test_bootstrap_ci_brackets_the_observed_relative_lift():
    df = make_case(n=30_000, lift=0.08, seed=7).df
    result = stats.bootstrap_analyze(df, seed=1)
    assert result.ci_low <= result.lift <= result.ci_high


def test_bayesian_favors_treatment_when_the_effect_is_large():
    df = make_case(n=40_000, lift=0.12, seed=2).df
    result = stats.bayesian_analyze(df, seed=0)
    assert result.prob_treatment_better > 0.99


def test_bayesian_is_agnostic_near_equal_arms():
    df = make_case(n=50_000, lift=0.0, seed=4).df
    result = stats.bayesian_analyze(df, seed=0)
    assert 0.35 < result.prob_treatment_better < 0.65


def test_bayesian_posterior_means_match_beta_moments():
    df = make_case(n=10_000, lift=0.05, seed=8).df
    n_c, n_t, k_c, k_t, p_c, p_t = stats._binary_arms(df)  # noqa: SLF001
    post_c = sps.beta(1 + k_c, 1 + n_c - k_c)
    post_t = sps.beta(1 + k_t, 1 + n_t - k_t)
    result = stats.bayesian_analyze(df, n_samples=500_000, seed=3)
    assert abs(result.control_rate - p_c) < 1e-12
    assert abs(result.treatment_rate - p_t) < 1e-12
    analytic_rel = (post_t.mean() - post_c.mean()) / post_c.mean()
    assert abs(result.expected_rel_lift - analytic_rel) < 0.01


@pytest.mark.parametrize(
    ("true_rel", "power", "type_s", "type_m"),
    [
        (0.005, 0.078, 0.0898, 4.84),
        (0.0284, 0.802, 0.0, 1.12),
    ],
)
def test_type_m_s_matches_the_kaggle_scale_reference(true_rel, power, type_s, type_m):
    """Reference rows from ab-test-analytics (SE_rel ≈ 1.01%, α = 0.05)."""
    se_rel = 0.0101
    z_crit = float(sps.norm.ppf(0.975))
    got_power, got_type_s, got_type_m = stats._retrodesign_type_m_s(  # noqa: SLF001
        true_rel, se_rel, z_crit
    )
    assert got_power == pytest.approx(power, rel=0.05, abs=0.01)
    assert got_type_s == pytest.approx(type_s, rel=0.05, abs=0.01)
    assert got_type_m == pytest.approx(type_m, rel=0.05, abs=0.08)


def test_audit_type_m_s_uses_the_sample_in_the_dataframe():
    df = make_case(n=120_000, lift=0.0, seed=5).df
    audit = stats.audit_type_m_s(df)
    assert audit.baseline_rate > 0
    assert audit.se_rel > 0
    assert len(audit.entries) == len(stats._DEFAULT_TYPE_M_GRID)  # noqa: SLF001
    at_mde = next(e for e in audit.entries if e.true_rel_lift == 0.0284)
    assert 0 < at_mde.power < 1
    assert at_mde.type_m >= 1.0


def test_z_test_from_statsmodels_aligns_with_analyze():
    df = make_case(n=25_000, lift=-0.02, seed=6).df
    n_c, n_t, k_c, k_t, _, _ = stats._binary_arms(df)  # noqa: SLF001
    count = np.array([k_c, k_t])
    nobs = np.array([n_c, n_t])
    z_stat, p_sm = proportions_ztest(count, nobs)
    freq = stats.analyze(df)
    assert abs(p_sm - freq.p_value) < 0.001
    assert abs(z_stat) == pytest.approx(
        abs(
            (k_t / n_t - k_c / n_c)
            / np.sqrt(
                ((k_c + k_t) / (n_c + n_t))
                * (1 - (k_c + k_t) / (n_c + n_t))
                * (1 / n_c + 1 / n_t)
            )
        ),
        rel=1e-6,
    )
