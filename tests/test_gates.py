"""Live gate uses Wilson (accuracy) and case-clustered bootstrap rates."""

from __future__ import annotations

import numpy as np

from peer_agent.evals import Report, check_regression, compare_cases
from peer_agent.scoring import cluster_rate_regression, false_ship_row


def _report(accuracy: float, high: float, false_ships: float, cases: list[dict]) -> Report:
    return Report(
        accuracy=accuracy,
        accuracy_ci=(0.0, high),
        false_ships=false_ships,
        false_blocks=0.0,
        by_flaw={},
        cases=cases,
        models=[],
        published_failures=[],
    )


def _false_ship_rows(n_cases: int, trials: int, rate: float, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    label = "openai/gpt-6-luna (tools)"
    rows = []
    for case in range(n_cases):
        for trial in range(trials):
            ship = bool(rng.random() < rate)
            rows.append(
                {
                    "id": f"c{case:02d}",
                    "trial": trial,
                    "variant": label,
                    "verdict": "SHIP" if ship else "NO_EFFECT",
                    "truth": "NO_EFFECT",
                    "error": None,
                    "correct": not ship,
                }
            )
    return rows


def _failure_rows(n_cases: int, trials: int, rate: float, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    label = "openai/gpt-6-luna (tools)"
    rows = []
    for case in range(n_cases):
        for trial in range(trials):
            failed = bool(rng.random() < rate)
            rows.append(
                {
                    "id": f"c{case:02d}",
                    "trial": trial,
                    "variant": label,
                    "verdict": None,
                    "truth": "NO_EFFECT",
                    "error": "format_fail" if failed else None,
                    "correct": not failed,
                }
            )
    return rows


def _false_ship_cases(n_false: int, n_total: int = 55) -> list[dict]:
    label = "openai/gpt-6-luna (tools)"
    rows = []
    for i in range(n_total):
        ship = i < n_false
        rows.append(
            {
                "id": f"c{i:02d}",
                "trial": 0,
                "variant": label,
                "verdict": "SHIP" if ship else "NO_EFFECT",
                "truth": "NO_EFFECT",
                "correct": not ship,
            }
        )
    return rows


def test_the_live_gate_uses_accuracy_not_the_replay_figure():
    report = _report(0.70, 0.80, 0.0, [])
    baseline = {
        "accuracy": 0.50,
        "false_ships": 0.0,
        "run_failure_rate": 0.0,
        "replay_expected": {"accuracy": 0.764, "by_flaw": {}},
    }
    violations = check_regression(report, baseline, mode="live")
    assert not any("accuracy" in item for item in violations)


def test_a_drop_inside_the_interval_passes():
    report = _report(0.70, 0.85, 0.0, [])
    baseline = {"accuracy": 0.80, "false_ships": 0.0, "run_failure_rate": 0.0}
    assert check_regression(report, baseline, mode="live") == []


def test_a_drop_the_interval_excludes_fails():
    cases = [{"id": f"c{i}", "trial": 0, "correct": i < 4} for i in range(10)]
    baseline_cases = [{"id": f"c{i}", "trial": 0, "correct": True} for i in range(10)]
    report = _report(0.40, 0.70, 0.0, cases)
    violations = check_regression(
        report,
        {
            "accuracy": 0.90,
            "false_ships": 0.0,
            "run_failure_rate": 0.0,
            "cases": baseline_cases,
        },
        mode="live",
    )
    assert any("CI high" in item for item in violations)


def test_false_ships_near_baseline_noise_passes_cluster_gate():
    rows = _false_ship_rows(55, 3, 0.24, seed=1)
    rate = sum(false_ship_row(r) for r in rows) / len(rows)
    report = _report(0.40, 0.55, rate, rows)
    baseline = {
        "accuracy": 0.418,
        "false_ships": 0.24242424242424243,
        "run_failure_rate": 0.05,
        "primary_variant": "openai/gpt-6-luna (tools)",
    }
    violations = check_regression(report, baseline, mode="live")
    assert not any("false_ships" in item for item in violations)


def test_false_ships_plus_ten_points_fails_cluster_gate():
    rows = _false_ship_rows(55, 3, 0.34, seed=2)
    rate = sum(false_ship_row(r) for r in rows) / len(rows)
    reject, _, _, _ = cluster_rate_regression(
        rows, 0.24242424242424243, false_ship_row, seed=42
    )
    assert reject
    report = _report(0.40, 0.55, rate, rows)
    baseline = {
        "accuracy": 0.418,
        "false_ships": 0.24242424242424243,
        "run_failure_rate": 0.05,
        "primary_variant": "openai/gpt-6-luna (tools)",
    }
    violations = check_regression(report, baseline, mode="live")
    assert any("false_ships" in item for item in violations)


def test_run_failure_spike_fails_the_gate():
    rows = _failure_rows(55, 3, 0.22, seed=3)
    report = _report(0.40, 0.55, 0.0, rows)
    baseline = {
        "accuracy": 0.418,
        "false_ships": 0.24242424242424243,
        "run_failure_rate": 0.048484848484848485,
        "primary_variant": "openai/gpt-6-luna (tools)",
    }
    violations = check_regression(report, baseline, mode="live")
    assert any("run_failures" in item for item in violations)


def test_cluster_gate_false_alarm_rate_at_baseline_is_about_five_percent():
    """With no true lift above baseline, reject should fire ~5% of the time (alpha)."""
    base = 0.24242424242424243
    n_cases, trials = 55, 3
    rejects = 0
    n_seeds = 40
    for seed in range(n_seeds):
        rows = _false_ship_rows(n_cases, trials, base, seed=seed)
        reject, _, _, _ = cluster_rate_regression(rows, base, false_ship_row, seed=99)
        rejects += int(reject)
    rate = rejects / n_seeds
    assert rate <= 0.12, f"false-alarm rate {rate:.2%} too high across {n_seeds} seeds"


def test_cluster_gate_power_at_five_and_ten_point_lift():
    base = 0.24242424242424243
    n_cases, trials = 55, 3
    seeds = range(30)

    def power(shift: float) -> float:
        hits = 0
        for seed in seeds:
            rows = _false_ship_rows(n_cases, trials, base + shift, seed=seed)
            reject, _, _, _ = cluster_rate_regression(rows, base, false_ship_row, seed=99)
            hits += int(reject)
        return hits / len(seeds)

    p5 = power(0.05)
    p10 = power(0.10)
    # Documented on PR #17 (lower-bound gate, 55x3, 30 seeds): ~37% at +5pp, ~97% at +10pp.
    assert p5 >= 0.25, f"power at +5pp unexpectedly low: {p5:.2%}"
    assert p10 >= 0.80, f"power at +10pp unexpectedly low: {p10:.2%}"


def test_compare_fails_when_one_arm_takes_every_discordant_case():
    only_a = [{"id": f"c{i}", "trial": 0, "correct": True} for i in range(10)]
    only_b = [{"id": f"c{i}", "trial": 0, "correct": False} for i in range(10)]
    assert compare_cases(only_a, only_b)["passed"] is False
