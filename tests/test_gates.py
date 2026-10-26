"""Live gate uses Wilson (accuracy) and a one-sided binomial test (false ships)."""

from __future__ import annotations

from peer_agent.evals import Report, check_regression, compare_cases
from peer_agent.scoring import false_ship_regression


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
        "replay_expected": {"accuracy": 0.764, "by_flaw": {}},
    }
    violations = check_regression(report, baseline, mode="live")
    assert not any("accuracy" in item for item in violations)
    tighter = check_regression(_report(0.55, 0.60, 0.0, []), baseline, mode="live")
    assert not any("accuracy" in item for item in tighter)


def test_a_drop_inside_the_interval_passes():
    report = _report(0.70, 0.85, 0.0, [])
    baseline = {"accuracy": 0.80, "false_ships": 0.0}
    assert check_regression(report, baseline, mode="live") == []


def test_a_drop_the_interval_excludes_fails():
    cases = [{"id": f"c{i}", "trial": 0, "correct": i < 4} for i in range(10)]
    baseline_cases = [{"id": f"c{i}", "trial": 0, "correct": True} for i in range(10)]
    report = _report(0.40, 0.70, 0.0, cases)
    violations = check_regression(
        report,
        {"accuracy": 0.90, "false_ships": 0.0, "cases": baseline_cases},
        mode="live",
    )
    assert any("CI high" in item for item in violations)
    assert any("McNemar" in item for item in violations)


def test_false_ships_at_baseline_noise_does_not_fail_the_live_gate():
    """16/55 ≈ 29% vs baseline 24% — the old +0.02 rule tripped here by chance."""
    cases = _false_ship_cases(16)
    report = _report(0.40, 0.55, 16 / 55, cases)
    baseline = {
        "accuracy": 0.418,
        "false_ships": 0.24242424242424243,
        "primary_variant": "openai/gpt-6-luna (tools)",
    }
    violations = check_regression(report, baseline, mode="live")
    assert not any("false_ships" in item for item in violations)


def test_false_ships_clearly_above_baseline_fails_the_live_gate():
    cases = _false_ship_cases(30)
    report = _report(0.40, 0.55, 30 / 55, cases)
    baseline = {
        "accuracy": 0.418,
        "false_ships": 0.24242424242424243,
        "primary_variant": "openai/gpt-6-luna (tools)",
    }
    violations = check_regression(report, baseline, mode="live")
    assert any("false_ships" in item for item in violations)
    k, n = 30, 55
    reject, _ = false_ship_regression(k, n, baseline["false_ships"])
    assert reject


def test_compare_fails_when_one_arm_takes_every_discordant_case():
    only_a = [{"id": f"c{i}", "trial": 0, "correct": True} for i in range(10)]
    only_b = [{"id": f"c{i}", "trial": 0, "correct": False} for i in range(10)]
    assert compare_cases(only_a, only_b)["passed"] is False


def test_compare_passes_a_three_three_split():
    left, right = [], []
    for i in range(3):
        left.append({"id": f"a{i}", "trial": 0, "correct": True})
        right.append({"id": f"a{i}", "trial": 0, "correct": False})
        left.append({"id": f"b{i}", "trial": 0, "correct": False})
        right.append({"id": f"b{i}", "trial": 0, "correct": True})
    result = compare_cases(left, right)
    assert result["b_only_a_right"] == 3
    assert result["c_only_b_right"] == 3
    assert result["passed"] is True
    assert result["delta_ci"][0] < 0 < result["delta_ci"][1]
