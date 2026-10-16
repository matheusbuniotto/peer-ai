# Live Luna score

## Problem

The number a reviewer reads is the harness replay of a hand-written script. `fixtures/recorded.jsonl` has 110 lines. The 100 case keys that already existed in commit `8525fca` still have those turn lists; this file adds a `trial` field and the ten outlier rows. The string `scripted` occurs in zero lines. `evals/baseline.json` records `accuracy` `0.7636363636363637` and `by_flaw.peeking` `0.0`. Outside peeking the tools variant scores 42/45 = 93.3%. The ten peeking misses pull the headline to 42/55 = 76.4%. README and `peer demo` present that 76.4% under two different run ids.

The live gate compares a new run's Wilson upper bound with `baseline["accuracy"]`. That field is the replay figure above. `.github/workflows/live-eval.yml` checks out whatever commit it runs on and applies that comparison.

`scripts/check_readme_commands.sh` already starts with `set -euo pipefail`. The README extractor is bare `python` inside a process substitution. On this machine that prints `python: command not found` and the script still exits 0, because bash does not apply `set -e` to a process substitution.

`uv run peer demo` on 2026-10-15 printed `scan_segments: scan_segments needs a 'segment' column'`. `fixtures/sample.parquet` has columns `arm`, `converted`, `pre_period_metric`. The same command printed two markdown tables, run `2026-10-15T21-24-20Z_claude-opus-5_84342b0` and run `2026-10-15T19-40-53Z_claude-opus-5_71c0af1`, with the same 76.4% and 45.5% rows. Stderr was empty. `configure_logfire` already passes `console=False`, which also turns off the project-link summary.

`GET https://openrouter.ai/api/v1/models` on 2026-10-15 lists `openai/gpt-6-luna` with `pricing.prompt` `0.0000001` and `pricing.completion` `0.0000005`. That is $0.10 and $0.50 per million tokens. `evals/pricing.yaml` already stores those two rates. A prompt of 272000 tokens or more uses a higher override on the catalog; this card is the standard rate.

A reviewer who runs the demo and reads the README currently sees a script score, a segment error, and two runs that share one result. After this ships, the committed headline is a 3-trial live run of `openai/gpt-6-luna`, the live gate's threshold is that run, the demo shows one table and a segment scan, and the README command check fails when its extractor fails.

## Flow

This reuses `run_suite`, `promote_baseline`, `splice_readme`, and `configure_logfire`. It does not add a second scorer.

1. The catalog rates stay in `evals/pricing.yaml` (exists). No fetch is wired into the process.
2. `cli._eval` (exists) calls `evals.run_suite` (exists) as `uv run peer eval --model openai/gpt-6-luna --trials 3 --record fixtures/recorded.jsonl --out results`. The CLI treats `--record` as a path, ignores `--replay` when `--model` is set, and with `--out` nests the run and writes `results/LATEST`. Default variants are `tools` and `no-tools`: 55 cases × 2 × 3 = 330 calls. The fixture file is replaced only after `run_suite` returns and every tools-variant row has an integer `input_tokens`.
3. `evals.promote_baseline` (exists, door 1) copies the live primary `accuracy`, `false_ships`, and `by_flaw` from that results file, and sets `replay_expected` from a 1-trial replay of `fixtures/recorded.jsonl`. The pre-commit hook is `uv run peer eval --gate`, which is replay mode at `trials` 1.
4. `report.splice_readme` (exists) rewrites the table between the results markers from `results/LATEST`. The paragraphs under `## Results` and `## Limitations` that describe that table are updated in the same change.
5. `.github/workflows/live-eval.yml` (exists) is dispatched once, on the commit that contains the promoted baseline. The workflow file's command stays `peer eval --model openai/gpt-6-luna --trials 3 --gate live`.
6. `scripts/check_readme_commands.sh` (exists) keeps `set -euo pipefail`. The extractor runs as `uv run python` where that `set -e` observes its exit code, then the loop runs the commands.
7. `demo.run_demo` (exists) scans `fixtures/sample.parquet` (exists) and prints the harness verdict plus the single table from `results/LATEST`. `configure_logfire` (exists) stays at `console=False`.

## Impact

| Front | What changes |
| --- | --- |
| domain | `baseline.accuracy` meant the replay headline `0.7636363636363637`. It becomes the primary accuracy of the promoted live run. `_live_gate` in `evals.py`, `live-eval.yml`, and `prompt-gate.yml` all read that field. |
| domain | `replay_expected` stays the harness score. `_replay_gate` and the pre-commit hook `uv run peer eval --gate` read it. A live promote must leave it equal to a 1-trial replay, which is what that hook executes. |
| stored data | `fixtures/recorded.jsonl` is replaced by the live recording. Git history keeps the script. `results/LATEST` moves to the live run. `fixtures/sample.parquet` gains a `segment` column and keeps `arm`, `converted`, and `pre_period_metric`. Nothing is backfilled. |

## Relations

None - no stored-data shape change. The baseline file keeps its current keys. Door 1 changes which run fills `accuracy` and which run fills `replay_expected`.

## Surface

None - nothing consumed outside. `peer eval`, `peer baseline promote`, `peer report`, `peer demo`, and `workflow_dispatch` keep their current flags and inputs.

## Landing

| One-way door | Literal shape | Alternative rejected |
| --- | --- | --- |
| Live promote fills two different scores | `evals/baseline.json` keeps `accuracy` / `false_ships` / `by_flaw` from the live primary variant and `replay_expected.accuracy` plus `replay_expected.by_flaw` from `peer eval --replay fixtures/recorded.jsonl` at `trials` 1 | Copying the 3-trial live accuracy into `replay_expected`. The pre-commit hook replays one trial, so that copy fails the hook on every commit. |

- Nothing else in this change is hard to reverse. The sample column, the README sentences, and the extractor are ordinary edits.

## Criteria

### S1: The committed score is a live Luna run (P1)

The README table and `results/LATEST` come from one 3-trial call to `openai/gpt-6-luna`, and the fixture turns are that call's recording.

**Acceptance Criteria**

1. The system SHALL keep `evals/pricing.yaml` at `input_per_mtok: 0.10` and `output_per_mtok: 0.50` for `openai/gpt-6-luna` while `GET https://openrouter.ai/api/v1/models` lists that id with `pricing.prompt` `0.0000001` and `pricing.completion` `0.0000005`.
2. WHEN `uv run peer eval --model openai/gpt-6-luna --trials 3 --record fixtures/recorded.jsonl --out results` exits 0 THEN `results/LATEST` SHALL name a `results.json` whose `mode` is `live`, whose `trials` is `3`, and whose primary variant model is `openai/gpt-6-luna`.
3. The committed `fixtures/recorded.jsonl` SHALL contain trials `0`, `1`, and `2` for all 55 suite cases on both `openai/gpt-6-luna (tools)` and `openai/gpt-6-luna (no tools)`, and each trial-0 turn list SHALL differ from the turn list stored for that case id and variant in the pre-change `fixtures/recorded.jsonl`.
4. WHEN `uv run peer report --write-readme` has run on that live file THEN the README block between `<!-- results:start -->` and `<!-- results:end -->` SHALL contain that run's `run_id` and the tools-variant accuracy printed from that file.
5. The README paragraphs under `## Results` and `## Limitations` SHALL call that table a live run of `openai/gpt-6-luna` with `trials` 3.
6. IF `run_suite` returns with any tools-variant row whose `input_tokens` is null THEN `peer eval` SHALL exit non-zero and `fixtures/recorded.jsonl` SHALL stay byte-for-byte the file from process start.

**Independent test:** fetch the catalog id and the two rates, then run the eval, promote, and report commands and read `results/LATEST`, the fixture keys, and the README markers.

### S2: The live gate uses the live baseline (P1)

Promoting the live run moves the threshold the weekly workflow compares against, and a 1-trial replay still matches `replay_expected`.

**Acceptance Criteria**

7. WHEN `promote_baseline` reads a results file whose `mode` is `live` THEN `evals/baseline.json` `accuracy`, `false_ships`, and `by_flaw` SHALL equal that file's primary variant, and `replay_expected` SHALL equal the accuracy and `by_flaw` of a 1-trial replay of `fixtures/recorded.jsonl`.
8. WHEN `check_regression` is called with `mode` `live` on a report whose Wilson high is `0.80` and a baseline whose `accuracy` is `0.50` and whose `replay_expected.accuracy` is `0.764` THEN the returned violations SHALL contain no accuracy line.
9. WHEN `live-eval.yml` is dispatched on the commit that contains the promoted baseline THEN the `live` job log SHALL contain `regression gate: OK` or `REGRESSION GATE FAILED`.

**Independent test:** promote a fixture live results file and assert both baseline fields, then dispatch the workflow and read the job log.

### S3: The README command check fails when the extractor fails (P1)

A missing interpreter is a failed check.

**Acceptance Criteria**

10. The script `scripts/check_readme_commands.sh` SHALL contain `set -euo pipefail` and SHALL invoke `uv run python` to read `README.md`.
11. IF that extractor exits 1 THEN the script SHALL exit non-zero.
12. WHEN every command the extractor prints exits 0 THEN the script SHALL exit 0.

**Independent test:** run the script once against this README, and once with the extractor forced to exit 1.

### S4: The demo shows one scan and one table (P1)

An interviewer running `peer demo` sees the toolbox succeed and one score.

**Acceptance Criteria**

13. WHEN `uv run peer demo` runs THEN stdout SHALL contain a `scan_segments: SegmentScanResult` line.
14. WHEN `uv run peer demo` runs THEN stdout SHALL contain one line `| Variant |`.
15. WHEN `uv run peer demo` runs THEN stdout and stderr together SHALL contain zero lines matching `Logfire`.
16. The file `fixtures/sample.parquet` SHALL include the column `segment` and SHALL keep the columns `arm`, `converted`, and `pre_period_metric`.

**Independent test:** `uv run peer demo` and a column read of `fixtures/sample.parquet`.

## Out of scope

| Excluded | Why |
| --- | --- |
| Cache-read, cache-write, and the 272000-token override on the Luna card | The catalog's standard `prompt` and `completion` already match the card. The suite prompts sit under that override. |
| A target accuracy or a target peeking rate for Luna | One run measures the model. The obligation is that the published number is that run. |
| Editing `prompt.md` or the skills so peeking scores higher | The misleading figure is the script, and the live run replaces the figure. |
| Changing the Wilson rule or the 2-point false-ship margin | `live-eval.yml` already implements that rule. The threshold value is what moves. |
| A second results table on the web page | The duplicate tables are the two blocks `peer demo` prints. |

## Assumptions

| Assumption | Chosen default | Rationale | Confirmed? |
| --- | --- | --- | --- |
| Which variants the live command scores | `tools` and `no-tools`, the CLI default when `--model` is set and `--variants` is omitted | The README table has both rows. Passing `--variants` would drop one. | n |
| A finished Luna response carries input tokens | A tools-variant row with null `input_tokens` fails the command and keeps the previous fixture | That is how a rejected key or a missing model shows up in `run_suite` today: the case is stored with no usage, and the process would still exit 0. | n |
| Dispatch ref | The commit that contains the promoted baseline, and only after an explicit push approval | `workflow_dispatch` checks out the ref. Dispatching current `main` would gate against `0.7636363636363637` again. | n |

**Open questions:**

| # | Kind | Question | Until answered |
| --- | --- | --- | --- |
| 2 | blocks | Can the commit that contains the promoted baseline be pushed before the dispatch? | AC 9. The local `.env` is set and the Actions secrets `LLM_API_KEY` and `LLM_BASE_URL` exist. Without the push the workflow reads the replay baseline. |

## Observable

| Surface | Decision | Landing |
| --- | --- | --- |
| command `peer eval` | flags and defaults | AC 2, assumption on variants |
| command `peer eval` | exit code when the provider returns no usage | AC 6 |
| command `peer eval` | output format | existing - accuracy, per-flaw lines, and `results/<run_id>/results.json` |
| command `peer baseline promote` | output and the file it rewrites | AC 7 |
| command `peer baseline promote` | destructive overwrite confirms | n/a - the command is the confirmation the request already names |
| command `peer report --write-readme` | what it rewrites | AC 4. The paragraphs outside the markers are AC 5, because the command does not edit them |
| command `peer demo` | error state of the toolbox | AC 13 |
| command `peer demo` | how many score tables | AC 14 |
| command `peer demo` | empty `results/LATEST` | existing - prints `results/LATEST is missing. No committed run to show.` |
| command `peer demo` | exit code when the replay gate passes | existing - exit 0 and `regression gate: OK` |
| script `check_readme_commands.sh` | extractor and exit codes | AC 10, AC 11, AC 12 |
| script `check_readme_commands.sh` | a README command fails halfway | existing - `set -euo pipefail` stops the loop on the first non-zero command |
| document README | which run the table claims to be | AC 4, AC 5 |
| document README | tone and the next command | existing - the command blocks stay, including the `# needs LLM_API_KEY` skip |
| workflow `live-eval.yml` | model input default `openai/gpt-6-luna` | existing - `workflow_dispatch` input |
| workflow `live-eval.yml` | missing secrets | existing - warning and exit 0. AC 9 is a dispatch where the secrets are present |
| collection `fixtures/recorded.jsonl` | one row per case, variant, and trial | AC 3 |
| collection `fixtures/recorded.jsonl` | duplicate keys | existing - `_load_recorded` keeps the last row for a key |

## Sources

- `GET https://openrouter.ai/api/v1/models` on 2026-10-15 — `openai/gpt-6-luna` is listed; `prompt` `0.0000001`, `completion` `0.0000005`; a 272000-token override exists and is out of scope
- `evals/baseline.json` and `fixtures/recorded.jsonl` against commit `8525fca` — the 76.4% figure and the reused turn lists
