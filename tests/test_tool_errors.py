"""Tool calls with missing columns must not crash the agent loop."""

from __future__ import annotations

import pytest
from pydantic_ai.exceptions import ModelRetry

from peer_agent.agent import _RecordingCall
from peer_agent.sim import make_case
from peer_agent.stats import check_srm


def test_check_srm_lists_available_columns_when_by_is_missing():
    df = make_case(n=5_000, lift=0.0, seed=1, days=14, flaws=[]).df
    with pytest.raises(ValueError, match="available columns"):
        check_srm(df, by=("segment",))


def test_recording_call_turns_keyerror_into_model_retry():
    df = make_case(n=2_000, lift=0.0, seed=2, days=14).df
    calls: list = []
    bound = _RecordingCall(check_srm, df)
    bound._calls = calls  # noqa: SLF001
    bound._tool_name = "check_srm"
    with pytest.raises(ModelRetry, match="segment"):
        bound(by=["segment"])
    assert calls == []
