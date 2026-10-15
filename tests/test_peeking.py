"""Planted peeking is a real optional stop, and the suite says so."""

from __future__ import annotations

import pytest

from peer_agent import stats
from peer_agent.evals import _review_question, load_suite


def test_every_peeking_case_is_significant_under_a_naive_read():
    peeked = [case for case in load_suite() if case.flaw == "peeking"]
    assert len(peeked) == 10
    for case in peeked:
        built = case.build()
        assert built.truth.name == "INVALID"
        assert stats.analyze(built.df).p_value < 0.05
        question = _review_question(built)
        assert "checked daily and stopped on day" in question
        assert str(built.df.attrs["stopped_day"]) in question


@pytest.mark.slow
def test_sequential_rejects_at_most_one_planted_peek():
    rejects = 0
    for case in load_suite():
        if case.flaw != "peeking":
            continue
        rejects += int(stats.sequential(case.build().df).reject)
    assert rejects <= 1
