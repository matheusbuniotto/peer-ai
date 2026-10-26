"""Pure statistics for eval scores. numpy and scipy only, no I/O.

Accuracy for one trial is the mean of `correct`. The number reported for a
multi-trial run is the mean of those trial means. The Wilson interval uses
trial 0 only (one independent Bernoulli trial per case). Pooling trials would
treat correlated repeats as extra cases and make the interval too narrow.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import numpy as np
from scipy import stats

from peer_agent.types import Verdict

_LIFT_TOOLS = frozenset({"analyze", "scan_segments"})


class ErrorCategory(StrEnum):
    FORMAT_FAIL = "format_fail"
    CRASHED = "crashed"
    SKIPPED_SRM_CHECK = "skipped_srm_check"
    FALSE_SHIP = "false_ship"
    FALSE_BLOCK = "false_block"
    WRONG_INVALID_REASON = "wrong_invalid_reason"
    MISSED_FLAW = "missed_flaw"
    OTHER_WRONG = "other_wrong"


@dataclass(frozen=True)
class McNemarResult:
    b: int  # A right, B wrong
    c: int  # A wrong, B right
    p_value: float
    method: Literal["exact", "chi2_cc"]


def false_ship_rate(rows: Sequence[dict[str, Any]], *, trial: int = 0) -> tuple[int, int]:
    """Count false ships among rows for one trial (verdict SHIP, truth not SHIP)."""
    subset = [row for row in rows if int(row["trial"]) == trial]
    if not subset:
        return 0, 0
    k = sum(
        1 for row in subset if row.get("verdict") == "SHIP" and row.get("truth") != "SHIP"
    )
    return k, len(subset)


def false_ship_regression(
    k: int, n: int, baseline_rate: float, alpha: float = 0.05
) -> tuple[bool, float]:
    """
    One-sided binomial test: is the false-ship count above the baseline rate?

    Returns (reject, p_value). `reject` is True when the gate should fail — the
    observed false-ship rate is significantly higher than the baseline at `alpha`.
    """
    if n <= 0:
        return False, 1.0
    p0 = min(max(float(baseline_rate), 0.0), 1.0)
    if k == 0:
        return False, 1.0
    result = stats.binomtest(k, n, p0, alternative="greater")
    p_value = float(result.pvalue)
    return p_value < alpha, p_value


def wilson(successes: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns (low, high)."""
    if n <= 0 or not 0 <= successes <= n:
        raise ValueError(f"need 0 <= successes <= n, got successes={successes} n={n}")
    z = float(stats.norm.ppf(1 - (1 - conf) / 2))
    p = successes / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    low = 0.0 if successes == 0 else float(max(0.0, center - margin))
    high = 1.0 if successes == n else float(min(1.0, center + margin))
    return low, high


def _as_matrix(correct_matrix: np.ndarray | Sequence[Sequence[bool]]) -> np.ndarray:
    arr = np.asarray(correct_matrix, dtype=bool)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("correct_matrix must be a non-empty cases by trials array")
    return arr


def pass_at_k(correct_matrix: np.ndarray | Sequence[Sequence[bool]]) -> float:
    """Fraction of cases with at least one correct trial. 'pass@k'."""
    arr = _as_matrix(correct_matrix)
    return float(np.mean(np.any(arr, axis=1)))


def pass_hat_k(correct_matrix: np.ndarray | Sequence[Sequence[bool]]) -> float:
    """Fraction of cases correct on every trial. 'pass^k'."""
    arr = _as_matrix(correct_matrix)
    return float(np.mean(np.all(arr, axis=1)))


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> McNemarResult:
    """Paired comparison of two variants on the same cases.

    `b` counts cases only A got right. `c` counts cases only B got right.
    Exact two-sided binomial test when b + c < 25, otherwise chi-square with
    Yates's continuity correction.
    """
    a = np.asarray(list(a_correct), dtype=bool)
    b = np.asarray(list(b_correct), dtype=bool)
    if a.shape != b.shape:
        raise ValueError("mcnemar needs two sequences of the same length")
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    discordant = only_a + only_b
    if discordant < 25:
        p_value = _exact_p(only_a, discordant)
        return McNemarResult(only_a, only_b, p_value, "exact")
    statistic = (abs(only_a - only_b) - 1) ** 2 / discordant
    return McNemarResult(only_a, only_b, float(stats.chi2.sf(statistic, 1)), "chi2_cc")


def newcombe_paired(
    a_correct: Sequence[bool], b_correct: Sequence[bool], conf: float = 0.95
) -> tuple[float, float]:
    """Newcombe's interval for the difference of two paired proportions, p_a - p_b."""
    a = np.asarray(list(a_correct), dtype=bool)
    b = np.asarray(list(b_correct), dtype=bool)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("newcombe_paired needs two non-empty paired sequences")
    both = int(np.sum(a & b))
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    neither = int(np.sum(~a & ~b))
    n = both + only_a + only_b + neither
    p_a = (both + only_a) / n
    p_b = (both + only_b) / n
    low_a, high_a = wilson(both + only_a, n, conf)
    low_b, high_b = wilson(both + only_b, n, conf)
    phi = _phi(both, only_a, only_b, neither)
    lower = p_a - p_b - _newcombe_term(p_a - low_a, high_b - p_b, phi)
    upper = p_a - p_b + _newcombe_term(high_a - p_a, p_b - low_b, phi)
    return float(lower), float(upper)


def _exact_p(only_a: int, discordant: int) -> float:
    if discordant == 0:
        return 1.0
    return float(stats.binomtest(only_a, discordant, 0.5).pvalue)


def _phi(both: int, only_a: int, only_b: int, neither: int) -> float:
    left = (both + only_a) * (only_b + neither) * (both + only_b) * (only_a + neither)
    if left <= 0:
        return 0.0
    return (both * neither - only_a * only_b) / np.sqrt(left)


def _newcombe_term(width_a: float, width_b: float, phi: float) -> float:
    radicand = width_a**2 + width_b**2 - 2 * phi * width_a * width_b
    return float(np.sqrt(max(0.0, radicand)))


def _called(traj: Any, name: str) -> bool:
    return any(call.name == name for call in traj.calls)


def classify_error(case: Any, traj: Any) -> ErrorCategory | None:
    """First matching failure mode, or None when the verdict is the truth.

    A right verdict is an ace. Tool order is scored separately by
    `srm_checked_before_lift`, so a lucky INVALID still counts as correct.
    """
    failure = _run_failure(traj)
    if failure is not None:
        return failure
    if traj.verdict is case.truth:
        return None
    return _wrong_verdict(case, traj)


def format_fail_detail(traj: Any) -> str:
    """Human-readable reason a trajectory failed to produce a scored verdict."""
    detail = getattr(traj, "error", None)
    if detail:
        return str(detail)
    answer = getattr(traj, "answer", "") or ""
    if not str(answer).strip():
        return "empty_model_response"
    return "missing_or_invalid_verdict_line"


def _run_failure(traj: Any) -> ErrorCategory | None:
    if traj.verdict is None:
        return ErrorCategory.FORMAT_FAIL
    if getattr(traj, "error", None) or not traj.done:
        return ErrorCategory.CRASHED
    return None


def _skipped_srm(case: Any, traj: Any) -> bool:
    return (
        case.truth is Verdict.INVALID
        and getattr(case, "flaw", None) == "srm"
        and not _called(traj, "check_srm")
    )


def _missed_flaw(truth: Verdict, verdict: Verdict) -> bool:
    return truth in {Verdict.INVALID, Verdict.EXTEND} and verdict in {
        Verdict.NO_EFFECT,
        Verdict.INCONCLUSIVE,
    }


def _wrong_verdict(case: Any, traj: Any) -> ErrorCategory:
    verdict, truth = traj.verdict, case.truth
    rules = (
        (_skipped_srm(case, traj), ErrorCategory.SKIPPED_SRM_CHECK),
        (verdict is Verdict.SHIP, ErrorCategory.FALSE_SHIP),
        (truth is Verdict.SHIP, ErrorCategory.FALSE_BLOCK),
        (verdict is Verdict.INVALID, ErrorCategory.WRONG_INVALID_REASON),
        (_missed_flaw(truth, verdict), ErrorCategory.MISSED_FLAW),
    )
    for matched, category in rules:
        if matched:
            return category
    return ErrorCategory.OTHER_WRONG


def srm_checked_before_lift(traj: Any) -> bool:
    """True when check_srm runs before analyze or scan_segments.

    Tool calls precede the answer, so a lift cited in the write-up is only
    allowed once check_srm has already been called.
    """
    names = [call.name for call in traj.calls]
    if "check_srm" not in names:
        return False
    srm_at = names.index("check_srm")
    return all(index > srm_at for index, name in enumerate(names) if name in _LIFT_TOOLS)
