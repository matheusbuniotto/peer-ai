"""Markdown for a committed results.json. The README table is this, not a hand edit."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

START = "<!-- results:start -->"
END = "<!-- results:end -->"


def load_results(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def render_table(payload: dict[str, Any], *, source: str | None = None) -> str:
    """One row per variant, then the paired comparison and the run footer."""
    k = int(payload["trials"])
    header = (
        f"| Variant | Accuracy (95% Wilson CI) | pass^{k} | False ship | "
        "False block | Cost/case | p50 latency | n |"
    )
    lines = [
        header,
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in payload["variants"]:
        lines.append(_variant_row(variant))
    lines.append("")
    if payload.get("comparisons"):
        comparison = payload["comparisons"][0]
        if comparison.get("pending"):
            lines.append(
                f"McNemar `{comparison['a']}` vs `{comparison['b']}`: pending live run."
            )
        else:
            lines.append(_comparison_line(comparison))
        lines.append("")
    if source:
        lines.append(f"Source: [`{source}`]({source}).")
        lines.append("")
    primary = next(
        (v for v in payload["variants"] if v.get("tools")), payload["variants"][0]
    )  # noqa: E501
    lines.append(
        f"Model `{primary['model']}`, run `{payload['run_id']}`, "
        f"{payload['generated_at'][:10]}, git `{payload['git_sha']}`."
    )
    return "\n".join(lines)


def splice_readme(text: str, block: str) -> str:
    start = text.find(START)
    end = text.find(END)
    if start < 0 or end < 0 or end < start:
        raise ValueError(f"README is missing {START} / {END}")
    return text[: start + len(START)] + "\n\n" + block.strip() + "\n\n" + text[end:]


def render_errors(payload: dict[str, Any]) -> str:
    """Flaw by error-category counts, process rates, and the five shortest misses."""
    cases = payload["cases"]
    misses = [case for case in cases if not case["correct"]]
    categories = sorted({case["error"] for case in misses if case["error"]})
    flaws = sorted({case["flaw"] for case in cases})
    lines = ["# Error taxonomy", "", _matrix(flaws, categories, misses), ""]
    total = _matrix_sum(flaws, categories, misses)
    lines.append(f"Errors: {len(misses)}. Matrix sum: {total}.")
    lines.append("")
    lines.append("## Process")
    lines.append("")
    for label, rows in _by_variant(cases).items():
        rate = sum(bool(row["srm_checked_before_lift"]) for row in rows) / len(rows)
        lines.append(f"- `{label}`: srm_checked_before_lift {rate:.1%} (n={len(rows)})")
    lines += ["", "## Shortest misses", ""]
    shortest = sorted(misses, key=_shortness)[:5]
    if not shortest:
        lines.append("No misses.")
    for case in shortest:
        turns = len(case.get("trajectory") or [])
        lines.append(
            f"- `{case['id']}` ({case['flaw']}, {case['variant']}, trial {case['trial']}): "
            f"{case['error']} in {turns} turns. "
            f"Truth {case['truth']}, verdict {case['verdict']}."
        )
    return "\n".join(lines) + "\n"


def _variant_row(variant: dict[str, Any]) -> str:
    low, high = variant["accuracy_ci"]
    n = sum(int(stats["n"]) for stats in variant["by_flaw"].values())
    cost = variant["cost_usd"]["per_case_mean"]
    cost_cell = "—" if cost is None else f"${cost:.4f}"
    p50 = variant["latency_s"]["p50"]
    return (
        f"| {variant['label']} | {_pct(variant['accuracy'])} ({_pct(low)}–{_pct(high)}) "
        f"| {_pct(variant['pass_hat_k'])} | {_pct(variant['false_ships'])} "
        f"| {_pct(variant['false_blocks'])} | {cost_cell} | {p50:.2f}s | {n} |"
    )


def _comparison_line(comparison: dict[str, Any]) -> str:
    return (
        f"McNemar `{comparison['a']}` vs `{comparison['b']}`: "
        f"b={comparison['b_only_a_right']} (only A right), "
        f"c={comparison['c_only_b_right']} (only B right), "
        f"p={comparison['p_value']:.3g} ({comparison['method']})."
    )


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _by_variant(cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(case["variant"], []).append(case)
    return grouped


def _matrix(flaws: list[str], categories: list[str], misses: list[dict[str, Any]]) -> str:
    if not categories:
        return "_No errors._"
    header = "| flaw | " + " | ".join(categories) + " |"
    rule = "| --- | " + " | ".join("---:" for _ in categories) + " |"
    rows = [header, rule]
    for flaw in flaws:
        counts = Counter(case["error"] for case in misses if case["flaw"] == flaw)
        cells = " | ".join(str(counts[category]) for category in categories)
        rows.append(f"| {flaw} | {cells} |")
    return "\n".join(rows)


def _shortness(case: dict[str, Any]) -> tuple[int, str]:
    return (len(case.get("trajectory") or []), case["id"])


def _matrix_sum(
    flaws: list[str], categories: list[str], misses: list[dict[str, Any]]
) -> int:
    counts = Counter((case["flaw"], case["error"]) for case in misses)
    return sum(counts[(flaw, category)] for flaw in flaws for category in categories)
