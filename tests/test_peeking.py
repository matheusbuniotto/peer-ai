"""Peeking and pre-registered sequential cases in the suite."""

from __future__ import annotations

import pytest

from peer_agent import stats
from peer_agent.evals import _review_question, load_suite
from peer_agent.sim import make_sequential_case, sequential_review_truth


def test_unplanned_peeking_cases_are_invalid_and_naively_significant():
    peeked = [case for case in load_suite() if case.flaw == "peeking"]
    assert len(peeked) == 5
    for case in peeked:
        built = case.build()
        assert built.truth.name == "INVALID"
        assert stats.analyze(built.df).p_value < 0.05
        assert "stopped_day" in built.df.attrs
        question = _review_question(built)
        assert "stopped on day" not in question.lower()
        assert "0 through" not in question


def test_sequential_cases_labels_need_the_sequential_tool():
    seq_cases = [case for case in load_suite() if case.flaw == "sequential"]
    assert len(seq_cases) == 5
    divergences = 0
    for case in seq_cases:
        built = case.build()
        primary = stats.analyze(built.df, mde=0.05)
        seq = stats.sequential(built.df, looks=14)
        naive_ship = primary.p_value < 0.05 and primary.lift > 0
        truth = sequential_review_truth(built.df, case.lift, looks=14, mde=0.05)
        assert truth == built.truth
        assert truth.name == case.truth.name
        if naive_ship != seq.reject:
            divergences += 1
    assert divergences >= 2


@pytest.mark.slow
def test_sequential_rejects_at_most_one_planted_peek():
    rejects = 0
    for case in load_suite():
        if case.flaw != "peeking":
            continue
        rejects += int(stats.sequential(case.build().df).reject)
    assert rejects <= 1


def test_sequential_review_truth_matches_deterministic_helper():
    for seed, lift in ((4, 0.0), (8, 0.06), (36, 0.06), (56, 0.06), (2, 0.0)):
        built = make_sequential_case(40_000, lift, seed=seed, days=14)
        again = sequential_review_truth(built.df, lift, looks=14, mde=0.05)
        assert again == built.truth
