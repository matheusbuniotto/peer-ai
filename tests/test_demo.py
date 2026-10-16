"""peer demo runs offline and prints the sections a reviewer can follow."""

from __future__ import annotations

from peer_agent.cli import main


def test_demo_prints_the_harness_sections(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "check_srm" in out
    assert "truth INVALID" in out
    assert "scan_segments: SegmentScanResult" in out
    assert out.count("| Variant |") == 1
    assert "regression gate: OK" in out
    assert "open http://127.0.0.1:8000 with peer web" in out
