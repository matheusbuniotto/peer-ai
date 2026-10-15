"""Lan-DeMets O'Brien-Fleming boundaries, checked against the ldBounds vignette."""

from __future__ import annotations

import pytest

from peer_agent import stats
from peer_agent.sim import make_case

_VIGNETTE = [4.877, 3.357, 2.680, 2.290, 2.031]


def test_five_equal_looks_match_ldbounds():
    bounds = stats.obf_boundaries([0.2, 0.4, 0.6, 0.8, 1.0])
    assert list(bounds) == pytest.approx(_VIGNETTE, abs=0.01)


def test_bonferroni_is_still_alpha_over_looks():
    frame = make_case(n=2_000, lift=0.0, seed=1).df
    result = stats.sequential(frame, looks=5, method="bonferroni")
    assert result.method == "bonferroni"
    assert result.alpha == pytest.approx(0.01)
    assert result.boundaries == ()


@pytest.mark.slow
def test_obf_type_i_error_stays_near_five_percent():
    rejects = sum(
        stats.sequential(make_case(n=4_000, lift=0.0, seed=seed).df, looks=5).reject
        for seed in range(2_000)
    )
    rate = rejects / 2_000
    assert 0.035 <= rate <= 0.065, rate
