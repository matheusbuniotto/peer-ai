from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from peer_agent import stats


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    schema: dict[str, object]
    fn: Callable[[pd.DataFrame], object]


def default_tools(sandbox: object | None = None) -> list[Tool]:
    del sandbox
    return [
        Tool(
            name="check_srm",
            description="Check whether the traffic split matches the assigned ratio.",
            schema={"type": "object", "properties": {}},
            fn=stats.check_srm,
        ),
        Tool(
            name="analyze",
            description="Run the primary conversion-rate test, treatment vs. control.",
            schema={"type": "object", "properties": {}},
            fn=stats.analyze,
        ),
    ]
