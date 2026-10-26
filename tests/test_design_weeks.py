from peer_agent.design import round_to_whole_weeks


def test_round_to_whole_weeks_snaps_to_calendar_weeks():
    assert round_to_whole_weeks(32) == 35
    assert round_to_whole_weeks(14) == 14
