---
name: review-protocol
description: The protocol for reviewing a finished A/B test - validate the split, check guardrails, then judge the primary metric. Load this before calling check_srm/analyze/sequential/scan_segments/check_novelty/check_guardrails/check_covariate_health/bootstrap_analyze/bayesian_analyze/audit_type_m_s.
---

Never skip steps. Never invent a number you haven't computed with a tool.

Think in Kohavi's OEC: one primary metric decides ship/no-ship; guardrails
protect the business; diagnostics explain mechanism. Do not optimize a
secondary metric at the expense of the OEC.

1. **Validate first (trust).** Call `check_srm`, passing the pre-registered `ratio`
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
3. **Covariate and segments before the headline lift.**
   - If `pre_period_metric` exists, call `check_covariate_health` before any
     CUPED or `analyze(cuped=True)`. When `cuped_unsafe` is true, run
     `analyze` without CUPED and say CUPED is not defensible on this frame.
   - If a `segment` column exists, call `scan_segments` after the primary
     `analyze`. A `reversal` or composition SRM inside segments means the
     aggregate lift is not a fair OEC read — verdict INVALID or INCONCLUSIVE,
     not SHIP on the headline alone.
4. **Optional stopping and peeking.** When the brief says the team checked
   daily, stopped early, or the data carries a stopping day, call
   `sequential` (Lan-DeMets OBF by default) before treating a significant
   `analyze` p-value as valid. A significant result after unplanned looks
   without a sequential pass is INVALID (optional stopping), not SHIP.
5. **The primary metric.** Call `analyze` with the pre-registered `mde` when
   one was fixed before the run (needed for `equivalent_to_null`). Pass
   `unit` when rows are visits rather than people. Repeat rows are not repeat
   evidence, and `analyze` will refuse rather than answer a question about rows.
   - Significant, durable, guardrails clean → SHIP. Scepticism that blocks
     everything is as useless as none at all.
   - A significant early effect that `check_novelty` flags as decaying is
     not a win — verdict is EXTEND, not SHIP.
   - Not significant, and `equivalent_to_null` is true → adequately powered
     null. Verdict is NO_EFFECT.
   - Not significant, and `equivalent_to_null` is false or null → underpowered
     or no MDE on file. Verdict is INCONCLUSIVE. Say what it would take to
     answer the question. Do not dress this up as NO_EFFECT, and do not go
     looking for a segment where it "worked" — an unresolved primary is not a
     license to fish.
   - **Twyman's law:** if the relative lift looks surprisingly large (roughly
     above 25%), call `audit_type_m_s` before celebrating — underpowered
     winners exaggerate truth (Type M). Optional corroboration: `bootstrap_analyze`
     and `bayesian_analyze` on the same `unit` rule as `analyze`.
6. State the verdict as
   `VERDICT: <NO_EFFECT|INCONCLUSIVE|INVALID|SHIP|EXTEND>` followed by your
   reasoning.
