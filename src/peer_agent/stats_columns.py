"""Shared column validation for stats tools."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def require_columns(df: pd.DataFrame, columns: Sequence[str], *, tool: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        available = ", ".join(sorted(df.columns))
        raise ValueError(
            f"{tool}: missing column(s) {missing}; available columns: {available}"
        )
