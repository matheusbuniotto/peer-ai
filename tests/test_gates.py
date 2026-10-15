"""The live gate fails only when the interval excludes the baseline."""

from __future__ import annotations

from peer_agent.evals import Report, check_regression, compare_cases


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


def test_false_ships_past_two_points_fail():
    report = _report(0.90, 0.99, 0.05, [])
    baseline = {"accuracy": 0.80, "false_ships": 0.0}
    violations = check_regression(report, baseline, mode="live")
    assert any("false_ships" in item for item in violations)


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
