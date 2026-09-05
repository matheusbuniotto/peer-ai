from __future__ import annotations

import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_IMAGE = "peerai-sandbox"


@dataclass(frozen=True)
class RunResult:
    ok: bool
    stdout: str
    stderr: str


class Sandbox:
    def __init__(self, timeout: int = 30, memory_mb: int = 512):
        self.timeout = timeout
        self.memory_mb = memory_mb

    def __enter__(self) -> Sandbox:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def run(self, code: str, df: pd.DataFrame) -> RunResult:
        """
        Runs `code` (piped over stdin, not a shell arg) inside a throwaway,
        network-disabled, read-only container with `df` mounted read-only at
        /data/exp.parquet — no network, no writes, no way out.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            df.to_parquet(Path(tmpdir) / "exp.parquet")
            name = f"peer-sandbox-{uuid.uuid4().hex[:12]}"
            cmd = [
                "docker",
                "run",
                "--rm",
                "-i",
                "--name",
                name,
                "--network",
                "none",
                f"--memory={self.memory_mb}m",
                "--read-only",
                "-v",
                f"{tmpdir}:/data:ro",
                _IMAGE,
                "python",
                "-",
            ]
            try:
                proc = subprocess.run(
                    cmd, input=code, capture_output=True, text=True, timeout=self.timeout
                )
            except subprocess.TimeoutExpired:
                # The client subprocess dying on timeout doesn't kill the container;
                # it's named so we can reach it directly.
                subprocess.run(["docker", "kill", name], capture_output=True)
                return RunResult(ok=False, stdout="", stderr="timed out")
            return RunResult(
                ok=proc.returncode == 0, stdout=proc.stdout, stderr=proc.stderr
            )
