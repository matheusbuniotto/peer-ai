"""
Phase 4 — the answer has to trace back to the toolbox.

The trajectory already records every call and every result, which makes one
class of hallucination mechanically checkable: a figure in the write-up that no
tool ever produced. These tests pin the tolerance rule, because a checker that
cries wolf on "14 days" or an ordered-list marker would be turned off by lunch.
"""

from __future__ import annotations

import pytest
from conftest import FakeLLM

from peer_agent.agent import ToolCall, Trajectory, unsupported_numbers
from peer_agent.types import AnalyzeResult, SRMResult, Verdict

_ANALYZE = AnalyzeResult(
    p_value=0.058,
    lift=-0.069504,
    ci_low=-0.13631,
    ci_high=0.00248,
    width=0.13879,
)
_SRM = SRMResult(mismatch=False, p_value=0.42, counts=(("False", 18432), ("True", 18291)))


def answer(text: str) -> Trajectory:
    calls = [
        ToolCall("check_srm", {}, _SRM),
        ToolCall("analyze", {"alpha": 0.05}, _ANALYZE),
    ]
    return Trajectory(calls=calls, answer=text)


def test_a_faithful_write_up_is_clean():
    traj = answer(
        "VERDICT: NO_EFFECT. Conversion moved -6.95% (p = 0.058, 95% CI "
        "[-13.63%, 0.25%]). Control 18432 users against treatment 18291."
    )
    assert unsupported_numbers(traj) == ()


def test_rounding_the_way_a_human_would_is_still_supported():
    """-0.069504 is fairly written as 7%, and 0.058 as 0.06."""
    assert unsupported_numbers(answer("Conversion fell about 7% (p = 0.06).")) == ()


def test_a_figure_from_nowhere_is_caught():
    traj = answer("Revenue per user rose 4.2% and retention improved by 11.8%.")
    assert unsupported_numbers(traj) == ("4.2%", "11.8%")


def test_a_grouped_number_is_one_figure_not_three():
    """
    A sample size is exactly the figure a reader trusts on sight, and models
    write it with separators. Split on the commas, "50,000" reads as a 50 too
    small to check beside a 000, and "1,234,567" gets reported as 234 and 567.
    """
    assert unsupported_numbers(answer("We enrolled 50,000 users.")) == ("50,000",)
    assert unsupported_numbers(answer("Across 1,234,567 sessions.")) == ("1,234,567",)


def test_prose_integers_are_not_statistics():
    """Ordered lists, arm counts and run lengths are writing, not findings."""
    traj = answer(
        "1. Validate the split\n"
        "2. Check the guardrails\n"
        "3. Judge the primary metric\n"
        "Ran 14 days across 2 arms."
    )
    assert unsupported_numbers(traj) == ()


def test_dates_and_years_are_not_statistics():
    traj = answer("Ran 2026-09-05 to 2026-09-19; written up in 2026.")
    assert unsupported_numbers(traj) == ()


@pytest.mark.parametrize("conventional", ["95% CI", "alpha = 0.05", "a 50/50 split"])
def test_protocol_constants_are_not_inventions(conventional):
    """These belong to the method, not to this dataset, and appear in any write-up."""
    assert unsupported_numbers(answer(f"We used {conventional}.")) == ()


def test_an_empty_trajectory_supports_nothing():
    """The phase-3 complaint: a full analysis written with no tool calls behind it."""
    traj = Trajectory(calls=[], answer="Conversion rose 3.4% (p = 0.021).")
    assert unsupported_numbers(traj) == ("3.4%", "0.021")


# --------------------------------------------------------------------------- #
# The check has to reach the loop, not just sit in a module
# --------------------------------------------------------------------------- #

_INVENTED = "VERDICT: SHIP. Conversion rose 9999.9%."


def test_the_model_is_sent_back_to_fix_its_own_write_up(clean, sandbox_free_agent):
    llm = FakeLLM(
        turns=[
            [("analyze", {})],
            _INVENTED,
            "VERDICT: SHIP. The effect is real and holds up.",
        ]
    )
    traj = sandbox_free_agent(llm).review(clean)
    assert "9999.9" not in traj.answer
    assert traj.verdict is Verdict.SHIP


def test_a_second_invention_is_struck_out(clean, sandbox_free_agent):
    """It had its chance; the reader still doesn't get an unbacked figure."""
    llm = FakeLLM(turns=[[("analyze", {})], _INVENTED, _INVENTED])
    traj = sandbox_free_agent(llm).review(clean)
    assert "[unsupported]" in traj.answer
    assert "9999.9" not in traj.answer
