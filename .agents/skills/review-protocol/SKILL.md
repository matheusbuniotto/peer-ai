---
name: review-protocol
description: The protocol for reviewing a finished A/B test - validate the split, check guardrails, then judge the primary metric. Load this before calling check_srm/analyze/sequential/scan_segments/check_novelty/check_guardrails.
---

Never skip steps. Never invent a number you haven't computed with a tool.

1. **Validate first.** Call `check_srm`. If the split is broken, stop:
   verdict is INVALID. Do not call `analyze`. State the verdict and *why*
   the split is broken in plain words only. Do not write down a single
   digit — no percentages, counts, ratios, multipliers ("roughly Nx"), or
   any other figure describing the imbalance or anything derived from it.
   Nothing computed from a broken split is trustworthy enough to put in
   front of a reader, including numbers about the brokenness itself. If
   you feel the urge to quantify how broken it is, that's the tell to stop
   and say only that it's broken.
2. **Guardrails before the headline.** Before shipping anything, call
   `check_guardrails`. A breached guardrail blocks SHIP even if the primary
   metric looks good.
3. **The primary metric.** Call `analyze` (and `sequential` if the
   experiment has been running long enough to have been peeked at).
   - No significant effect → say so plainly. Verdict is NO_EFFECT. Do not
     go looking for a segment where it "worked" — a null primary result is
     not a license to fish.
   - A significant early effect that `check_novelty` flags as decaying is
     not a win — verdict is EXTEND, not SHIP.
   - A real, durable, guardrail-clean effect ships. Verdict is SHIP.
     Scepticism that blocks everything is as useless as none at all.
4. State the verdict as `VERDICT: <NO_EFFECT|INVALID|SHIP|EXTEND>` followed
   by your reasoning.
