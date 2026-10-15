"""OpenRouter is the documented live endpoint. Its price card is a published rate."""

from __future__ import annotations

import pytest

from peer_agent.agent import chat_settings
from peer_agent.evals import _cost, load_prices


def test_gpt_6_luna_uses_the_openrouter_list_price():
    prices = load_prices()
    card = prices["openai/gpt-6-luna"]
    assert card["input_per_mtok"] == pytest.approx(0.10)
    assert card["output_per_mtok"] == pytest.approx(0.50)
    row = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    assert _cost("openai/gpt-6-luna", row, prices) == pytest.approx(0.60)
    assert _cost("missing-model", row, prices) is None


def test_openrouter_omits_the_openai_store_field():
    settings = chat_settings("https://openrouter.ai/api/v1")
    assert "openai_store" not in settings
    assert chat_settings("https://example.test/v1")["openai_store"] is False
