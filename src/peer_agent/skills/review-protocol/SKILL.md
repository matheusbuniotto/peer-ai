---
name: review-protocol
description: The protocol for reviewing a finished A/B test - validate the split, check guardrails, then judge the primary metric. Load this before calling check_srm/analyze/sequential/scan_segments/check_novelty/check_guardrails/check_covariate_health/bootstrap_analyze/bayesian_analyze/audit_type_m_s.
---

Never skip steps. Never invent a number you haven't computed with a tool.

Think in Kohavi's OEC: one primary metric decides ship/no-ship; guardrails
protect the business; diagnostics explain mechanism. Do not optimize a
secondary metric at the expense of the OEC.

1. **Validate first (trust).** Call `check_srm`, passing the pre-registered `ratio`
   and `by` only for columns that exist in the data (if a column is missing, the
   tool will list what is available — do not guess). A 90/10 holdout is a design,
   not a defect. If the split is broken, stop: verdict is INVALID. Do not call
   `analyze`. State the verdict and *why* the split is broken in plain words only.
   Do not write down digits describing the imbalance.
2. **Guardrails before the headline.** Call `check_guardrails` with every named
   guardrail and margin. Only `clean` lets a ship through; `inconclusive` blocks
   like a breach (`blocks_ship` is the flag to read).
3. **Covariate and segments before trusting the headline.**
   - If `pre_period_metric` exists, call `check_covariate_health` before CUPED.
     When `cuped_unsafe` is true, run `analyze` without CUPED.
   - If `segment` exists, call `scan_segments`. A `reversal` or segment composition
     SRM means the aggregate OEC is misleading — do not SHIP on the headline alone;
     use INCONCLUSIVE (or NO_EFFECT if the primary is null and adequately powered)
     and explain the segment story. A reversal is not automatic INVALIDity unless
     SRM or optional stopping already broke trust.
4. **Multiple looks / optional stopping.** When a `day` or `cohort_day` column
   exists, call `sequential` with the pre-registered `looks` and `alpha` from the
   spec (do not pass `looks: null` when a multi-look plan was pre-registered).
   - If `looks` is 1 but the data cover fewer calendar days than the pre-registered
     `days`, the run stopped early without a sequential plan → INVALID (optional
     stopping), even when `sequential` or `analyze` look significant.
   - If `looks` > 1, a pre-registered group-sequential plan applies: trust a
     positive primary only when `sequential` reports `reject: true` and guardrails
     are clean; if `reject: false` on a null experiment, proceed to step 5 for
     equivalence / INCONCLUSIVE — do not call INVALID merely because monitoring
     was daily.
5. **Primary metric.** Call `analyze` with the pre-registered `mde` when one was
   supplied. Pass `unit` when rows are visits, not people.
   - Significant positive lift (`p` below `alpha`), guardrails clean, and step 4
     did not invalidate the run → SHIP (check novelty / Type M if indicated).
   - Decaying novelty → EXTEND.
   - Not significant and `equivalent_to_null` → NO_EFFECT (only after ruling out
     a significant win in the previous bullet).
   - Not significant and not equivalent → INCONCLUSIVE (underpowered or no MDE).
   - Surprisingly large lift → `audit_type_m_s` (Twyman / Type M) before trusting it.
6. State `VERDICT: <NO_EFFECT|INCONCLUSIVE|INVALID|SHIP|EXTEND>` and reasoning.
