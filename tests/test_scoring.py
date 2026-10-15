"""Eval statistics: Wilson intervals, pass@k, McNemar, and the error taxonomy.

The expected intervals are checked two ways. The hand values are the acceptance
targets. statsmodels is the independent implementation and is not used in
production.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from statsmodels.stats.proportion import proportion_confint

from peer_agent.agent import ToolCall, Trajectory
from peer_agent.scoring import (
    ErrorCategory,
    classify_error,
    mcnemar,
    newcombe_paired,
    pass_at_k,
    pass_hat_k,
    srm_checked_before_lift,
    wilson,
)
from peer_agent.types import Verdict


def test_wilson_matches_the_hand_values_and_statsmodels():
    low, high = wilson(47, 50)
    assert low == pytest.approx(0.8378, abs=1e-3)
    assert high == pytest.approx(0.9794, abs=1e-3)
    ref_low, ref_high = proportion_confint(47, 50, method="wilson")
    assert (low, high) == pytest.approx((ref_low, ref_high), abs=1e-12)

    empty = wilson(0, 10)
    full = wilson(10, 10)
    assert empty[0] == 0.0
    assert empty[1] == pytest.approx(0.2775, abs=1e-3)
    assert full[0] == pytest.approx(0.7225, abs=1e-3)
    assert full[1] == 1.0
    assert empty == pytest.approx(proportion_confint(0, 10, method="wilson"), abs=1e-12)
    assert full == pytest.approx(proportion_confint(10, 10, method="wilson"), abs=1e-12)


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson(1, 0)
    with pytest.raises(ValueError):
        wilson(3, 2)


def test_pass_at_k_rejects_an_empty_matrix():
    with pytest.raises(ValueError):
        pass_at_k([])


def test_pass_at_k_and_pass_hat_k():
    assert pass_at_k([[1, 0, 0], [0, 0, 0]]) == 0.5
    assert pass_hat_k([[1, 1, 1], [1, 0, 1]]) == 0.5
    assert pass_at_k(np.array([True, False])) == 0.5


def test_mcnemar_uses_the_exact_test_under_25_discordant_pairs():
    only_a = [True] * 17 + [False] * 3
    only_b = [False] * 17 + [False] * 3
    result = mcnemar(only_a, only_b)
    assert result.b == 17
    assert result.c == 0
    assert result.method == "exact"
    assert result.p_value == pytest.approx(1.52587890625e-5, rel=1e-6)

    tied = mcnemar([True] * 5 + [False] * 5, [False] * 5 + [True] * 5)
    assert (tied.b, tied.c) == (5, 5)
    assert tied.p_value == 1.0
    assert tied.method == "exact"


def test_mcnemar_uses_chi_square_with_continuity_correction_at_25():
    a = [True] * 20 + [False] * 10 + [True] * 5
    b = [False] * 20 + [True] * 10 + [True] * 5
    result = mcnemar(a, b)
    assert (result.b, result.c) == (20, 10)
    assert result.method == "chi2_cc"
    assert 0.0 < result.p_value < 1.0


def test_mcnemar_with_no_discordant_pairs_is_one():
    result = mcnemar([True, False], [True, False])
    assert (result.b, result.c, result.p_value, result.method) == (0, 0, 1.0, "exact")


def test_mcnemar_requires_paired_sequences():
    with pytest.raises(ValueError):
        mcnemar([True], [True, False])


def test_newcombe_interval_covers_a_paired_difference():
    low, high = newcombe_paired([True] * 8 + [False] * 2, [True] * 5 + [False] * 5)
    assert low < 0.3 < high
    assert low < high
    tied_low, tied_high = newcombe_paired([True, True], [True, True])
    assert tied_low <= 0 <= tied_high


def test_newcombe_requires_paired_sequences():
    with pytest.raises(ValueError):
        newcombe_paired([True], [True, False])


def _case(truth: Verdict, flaw: str = "clean"):
    return SimpleNamespace(truth=truth, flaw=flaw)


def _traj(verdict: Verdict | None, calls: list[str] | None = None, **kwargs) -> Trajectory:
    return Trajectory(
        calls=[ToolCall(name, {}, None) for name in calls or []],
        verdict=verdict,
        done=kwargs.get("done", True),
        answer=kwargs.get("answer", "VERDICT: X"),
        error=kwargs.get("error"),
    )


@pytest.mark.parametrize(
    ("case", "traj", "expected"),
    [
        (_case(Verdict.INVALID, "srm"), _traj(None), ErrorCategory.FORMAT_FAIL),
        (
            _case(Verdict.NO_EFFECT),
            _traj(Verdict.NO_EFFECT, done=False),
            ErrorCategory.CRASHED,
        ),
        (
            _case(Verdict.INVALID, "srm"),
            _traj(Verdict.SHIP, ["analyze"]),
            ErrorCategory.SKIPPED_SRM_CHECK,
        ),
        (
            _case(Verdict.NO_EFFECT),
            _traj(Verdict.SHIP, ["check_srm", "analyze"]),
            ErrorCategory.FALSE_SHIP,
        ),
        (
            _case(Verdict.SHIP),
            _traj(Verdict.NO_EFFECT, ["check_srm", "analyze"]),
            ErrorCategory.FALSE_BLOCK,
        ),
        (
            _case(Verdict.NO_EFFECT),
            _traj(Verdict.INVALID, ["check_srm"]),
            ErrorCategory.WRONG_INVALID_REASON,
        ),
        (
            _case(Verdict.EXTEND, "novelty"),
            _traj(Verdict.NO_EFFECT, ["check_srm", "analyze"]),
            ErrorCategory.MISSED_FLAW,
        ),
        (
            _case(Verdict.NO_EFFECT),
            _traj(Verdict.EXTEND, ["check_srm", "analyze"]),
            ErrorCategory.OTHER_WRONG,
        ),
        (
            _case(Verdict.INVALID, "srm"),
            _traj(Verdict.INVALID, ["check_srm"]),
            None,
        ),
    ],
)
def test_classify_error_picks_the_first_matching_rule(case, traj, expected):
    assert classify_error(case, traj) is expected


def test_srm_checked_before_lift_requires_the_tool_first():
    good = _traj(Verdict.INVALID, ["check_srm", "analyze"], answer="The split is broken.")
    late = _traj(Verdict.INVALID, ["analyze", "check_srm"], answer="lift 5%")
    missing = _traj(Verdict.SHIP, ["analyze"], answer="lift 5%")
    assert srm_checked_before_lift(good)
    assert not srm_checked_before_lift(late)
    assert not srm_checked_before_lift(missing)
