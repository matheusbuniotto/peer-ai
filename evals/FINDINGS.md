# Eval findings (open)

Recorded replay: `results/2026-10-16T01-06-57Z_openai-gpt-6-luna_1264920`, primary variant `openai/gpt-6-luna (tools)`, trial 0. Baseline `evals/baseline.json` matches these flaw accuracies.

## F-001 — `peeking` (0/10 correct)

- **Symptom:** Optional-stopping cases are never judged correctly; the agent often ships on noise or fails to emit a scored verdict.
- **Evidence:** Trial 0 on all ten cases `peeking-01` … `peeking-10` is incorrect (`evals/published_failures.json`). Error mix on trial 0: five `false_ship`, three `format_fail`, one `missed_flaw` (`peeking-02`), one `other_wrong` (`peeking-06`). Aggregate accuracy 0.0 with Wilson 95% CI 0.0%–27.8% (`n=10`).
- **Hypothesis:** The model treats planted interim significance as a standard fixed-horizon win and does not call `sequential` / infer optional stopping from the daily path, so verdicts skew to `SHIP` or unscored output.
- **Fix idea:** Force an explicit sequential or peeking audit in the review protocol before any lift-based ship recommendation; separate from prompt tuning on this suite until reporting is stable.
- **Status:** open

## F-002 — `simpson` (0/10 correct)

- **Symptom:** Segment-reversal / composition cases are always wrong; many false ships on aggregate lift.
- **Evidence:** Trial 0 misses on `simpson-01` … `simpson-10` (`evals/published_failures.json`). Six `false_ship`, four `other_wrong` on trial 0. Aggregate accuracy 0.0 with Wilson 95% CI 0.0%–27.8% (`n=10`).
- **Hypothesis:** `scan_segments` is skipped or its heterogeneity signal is ignored, so the agent cites the pooled effect and ships.
- **Fix idea:** Require stratum-level SRM or segment scan before trusting a global lift; document expected tool order in the review skill.
- **Status:** open

## F-003 — `outliers` (0/5 correct)

- **Symptom:** CUPED covariate distortion cases are never caught; verdicts disagree with `NO_EFFECT` truth.
- **Evidence:** Trial 0 misses on `outliers-01` … `outliers-05` (`evals/published_failures.json`), all `other_wrong` on trial 0. Aggregate accuracy 0.0 with Wilson 95% CI 0.0%–43.4% (`n=5`).
- **Hypothesis:** `check_covariate_health` is not invoked (or not weighted) before CUPED-backed lifts, so inflated covariate rows drive a ship call.
- **Fix idea:** Gate CUPED reads on covariate health and treat `cuped_unsafe` as blocking evidence for ship.
- **Status:** open
