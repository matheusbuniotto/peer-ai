"""The README command check fails when its extractor fails."""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path("scripts/check_readme_commands.sh")


def test_the_script_uses_uv_and_set_e():
    text = SCRIPT.read_text()
    assert "set -euo pipefail" in text
    assert "uv run python" in text


def test_a_failing_extractor_fails_the_script(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    uv = bindir / "uv"
    uv.write_text("#!/bin/sh\nexit 1\n")
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    env = os.environ.copy()
    env["PATH"] = f"{bindir}{os.pathsep}{env['PATH']}"
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


@pytest.mark.live
def test_the_readme_script_exits_0():
    result = subprocess.run(["bash", str(SCRIPT)], cwd=Path.cwd())
    assert result.returncode == 0
