"""Optional Braintrust export. Nothing here runs unless BRAINTRUST_API_KEY is set."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast


def export_results(path: str | Path, project: str) -> int:
    """Log one experiment row per case. Exit 2 when the key or the SDK is missing."""
    if not os.environ.get("BRAINTRUST_API_KEY"):
        print("BRAINTRUST_API_KEY is not set. Not exporting.")
        return 2
    try:
        sdk = cast(Any, __import__("braintrust"))
    except ImportError:
        print("braintrust is not installed. Install it with: uv sync --extra braintrust")
        return 2
    payload = json.loads(Path(path).read_text())
    experiment = sdk.init(project=project)
    for case in payload["cases"]:
        experiment.log(**_row(case))
    print(f"exported {len(payload['cases'])} rows to Braintrust project {project}")
    return 0


def _row(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "input": {"case_id": case["id"], "flaw": case["flaw"]},
        "output": case["verdict"],
        "expected": case["truth"],
        "scores": {
            "correct": float(bool(case["correct"])),
            "srm_checked_before_lift": float(bool(case["srm_checked_before_lift"])),
            "fabricated": float(bool(case.get("fabricated", False))),
        },
        "metadata": {
            "error": case["error"],
            "cost": case["cost_usd"],
            "latency": case["latency_s"],
            "variant": case["variant"],
            "trial": case["trial"],
        },
    }
