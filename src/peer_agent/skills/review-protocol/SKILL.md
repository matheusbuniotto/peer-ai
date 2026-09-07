---
name: review-protocol
description: The protocol for reviewing a finished A/B test - validate the split, check guardrails, then judge the primary metric. Load this before calling check_srm/analyze/sequential/scan_segments/check_novelty/check_guardrails.
---

Never skip steps. Never invent a number you haven't computed with a tool.

1. **Validate first.** Call `check_srm`, passing the pre-registered `ratio`
   (a 90/10 holdout is a design, not a defect) and `by` for any segment or
   day column the data has — a split can balance overall while tilting
   inside a segment, and that is what a composition skew looks like. If the
   split is broken, stop:
   verdict is INVALID. Do not call `analyze`. State the verdict and *why*
   the split is broken in plain words only. Do not write down a single
   digit — no percentages, counts, ratios, multipliers ("roughly Nx"), or
   any other figure describing the imbalance or anything derived from it.
   Nothing computed from a broken split is trustworthy enough to put in
   front of a reader, including numbers about the brokenness itself. If
   you feel the urge to quantify how broken it is, that's the tell to stop
   and say only that it's broken.
2. **Guardrails before the headline.** Before shipping anything, call
   `check_guardrails` with every guardrail the pre-registration names, each
   with its margin. It answers `clean`, `breached`, or `inconclusive` per
   guardrail. Only `clean` lets a ship through. `inconclusive` means the
   data can't rule out harm past the margin, and it blocks exactly like a
   breach — an underpowered safety check is not a passed one, and
   `blocks_ship` on the result is the single flag to read. Report the bound,
   not just the status.
3. **The primary metric.** Call `analyze` (and `sequential` if the
   experiment has been running long enough to have been peeked at).
   - Significant, durable, guardrails clean → SHIP. Scepticism that blocks
     everything is as useless as none at all.
   - A significant early effect that `check_novelty` flags as decaying is
     not a win — verdict is EXTEND, not SHIP.
   - Not significant, and `equivalent_to_null` is true → the experiment was
     big enough to rule out an effect worth having. Verdict is NO_EFFECT.
   - Not significant, and `equivalent_to_null` is false or null → you cannot
     tell an absent effect from a test too small to find one. Verdict is
     INCONCLUSIVE. Say what it would take to answer the question. Do not
     dress this up as NO_EFFECT, and do not go looking for a segment where
     it "worked" — an unresolved primary is not a license to fish.
4. State the verdict as
   `VERDICT: <NO_EFFECT|INCONCLUSIVE|INVALID|SHIP|EXTEND>` followed by your
   reasoning.
