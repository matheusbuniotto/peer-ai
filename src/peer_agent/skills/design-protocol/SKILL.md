---
name: design-protocol
description: The protocol for designing a new A/B test from a brief - size it, check it survives simulation, and ask before guessing on an undecided flat-result policy. Load this before calling power_analysis/simulate_design/ask.
---

1. Work out baseline, mde, and the OEC (primary metric) from the brief when they are
   stated. When the brief omits an MDE or a flat-result policy, call `ask` first.
   If `ask` reports that nobody is available, you may use only the **conservative
   defaults** below — never silent invention. Record each default as an
   **unconfirmed assumption** in the written design and in `if_flat` (or your
   rationale): it is not a stakeholder answer.
   - **MDE (only when baseline/OEC are known but MDE is missing):** 5% relative
     uplift on the primary metric.
   - **Flat result (only when the brief is silent):** retain the current
     experience / do not launch the variant (`if_flat` must say this is
     unconfirmed).
   Any other number must come from the brief or from a tool (`power_analysis`,
   `simulate_design`), not from guesswork.
2. Use `power_analysis` to size it, then `simulate_design` to check the
   promise actually holds before handing back a design. Name guardrail metrics
   when the brief mentions latency, errors, or revenue risk.
3. Runtime must be a whole number of weeks — call `round_to_whole_weeks` on any
   day count before returning a `DesignSpec`.
