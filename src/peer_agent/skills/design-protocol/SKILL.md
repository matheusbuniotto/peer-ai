---
name: design-protocol
description: The protocol for designing a new A/B test from a brief - size it, check it survives simulation, and ask before guessing on an undecided flat-result policy. Load this before calling power_analysis/simulate_design/ask.
---

1. Work out baseline, mde, and the OEC (primary metric) from the brief. If the
   brief omits an MDE or flat-result policy, `ask` — do not invent numbers or
   return a `DesignSpec` with guessed effect sizes.
2. If the brief doesn't say what happens on a flat result, `ask` before
   writing anything — a test with no decision attached to it shouldn't run.
3. Use `power_analysis` to size it, then `simulate_design` to check the
   promise actually holds before handing back a design. Name guardrail metrics
   when the brief mentions latency, errors, or revenue risk.
4. Runtime must be a whole number of weeks — call `round_to_whole_weeks` on any
   day count before returning a `DesignSpec`.
