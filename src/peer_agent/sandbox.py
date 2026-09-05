from __future__ import annotations


class Sandbox:
    def __init__(self, timeout: int = 30, memory_mb: int = 512):
        self.timeout = timeout
        self.memory_mb = memory_mb

    def __enter__(self) -> Sandbox:
        return self

    def __exit__(self, *exc: object) -> None:
        return None
