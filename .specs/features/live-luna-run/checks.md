# Live Luna score checks

Profile: light
Plan: `.specs/features/live-luna-run/plan.md`

16 checks in 4 slices · 1 one-way door · 1 open, of which 1 blocks

## Checks

### S1 - Committed live score · 6 files · 58 KB · ~15k

**C1** - `openai/gpt-6-luna` is listed with prompt `0.0000001` and completion `0.0000005`, and the card stays at `0.10` / `0.50` (AC 1)
Proof: `uv run pytest tests/test_openrouter.py -k test_openrouter_lists_luna_at_the_card_rate`

**C2** - `results/LATEST` points at `mode` `live`, `trials` `3`, model `openai/gpt-6-luna` (AC 2)
Proof: `uv run pytest tests/test_live_luna.py -k test_latest_is_a_three_trial_luna_run`

**C3** - The fixture has trials `0`, `1`, and `2` for all 55 cases on both Luna variants, and each trial-0 turn list differs from commit `84342b0` (AC 3)
Proof: `uv run pytest tests/test_live_luna.py -k test_recorded_turns_are_the_luna_run`

**C4** - The README results block contains that run id and the tools-variant accuracy printed to one decimal percent (AC 4)
Proof: `uv run pytest tests/test_live_luna.py -k test_readme_table_quotes_the_live_run`

**C5** - The `## Results` and `## Limitations` paragraphs call the table a live `openai/gpt-6-luna` run with `trials 3` (AC 5)
Proof: `uv run pytest tests/test_live_luna.py -k test_readme_calls_the_table_a_live_luna_run`

**C6** - A live record whose tools rows have null `input_tokens` exits `1` and leaves the fixture bytes unchanged (AC 6)
Proof: `uv run pytest tests/test_live_luna.py -k test_a_live_record_with_no_tokens_keeps_the_fixture`

### S2 - Live baseline · 2 files · 30 KB · ~8k

**C7** - A live promote copies `accuracy`, `false_ships`, and `by_flaw` from the live primary, and `replay_expected` from a 1-trial replay (AC 7)
Proof: `uv run pytest tests/test_live_luna.py -k test_a_live_promote_keeps_the_one_trial_replay_score`

**C8** - A live gate with Wilson high `0.80` against baseline accuracy `0.50` and replay accuracy `0.764` returns no accuracy violation (AC 8)
Proof: `uv run pytest tests/test_gates.py -k test_the_live_gate_uses_accuracy_not_the_replay_figure`

**C9** - The dispatched `live-eval.yml` job log contains `regression gate: OK` or `REGRESSION GATE FAILED` (AC 9)
Proof: `uv run pytest tests/test_live_luna.py -k test_the_dispatched_live_job_ran_the_gate -m live`

### S3 - README command check · 1 file · 1 KB · ~1k

**C10** - `scripts/check_readme_commands.sh` contains `set -euo pipefail` and `uv run python` (AC 10)
Proof: `uv run pytest tests/test_readme_check.py -k test_the_script_uses_uv_and_set_e`

**C11** - An extractor that exits `1` makes the script exit non-zero (AC 11)
Proof: `uv run pytest tests/test_readme_check.py -k test_a_failing_extractor_fails_the_script`

**C12** - The script exits `0` when every extracted command exits `0` (AC 12)
Proof: `uv run pytest tests/test_readme_check.py -k test_the_readme_script_exits_0 -m live`

### S4 - Demo · 3 files · 55 KB · ~14k

**C13** - `peer demo` prints `scan_segments: SegmentScanResult` (AC 13)
Proof: `uv run pytest tests/test_demo.py -k test_demo_prints_the_harness_sections`

**C14** - `peer demo` prints one `| Variant |` line (AC 14)
Proof: `uv run pytest tests/test_live_luna.py -k test_demo_prints_one_results_table`

**C15** - `peer demo` stdout and stderr contain zero lines matching `Logfire` (AC 15)
Proof: `uv run pytest tests/test_live_luna.py -k test_demo_prints_no_logfire_line`

**C16** - `fixtures/sample.parquet` has `segment`, `arm`, `converted`, and `pre_period_metric` (AC 16)
Proof: `uv run pytest tests/test_live_luna.py -k test_sample_parquet_has_a_segment_column`

## Coverage

| Set (size) | Member -> proof | Unproven |
| --- | --- | --- |
| baseline scores (2) | `accuracy` C7 · `replay_expected` C7 | - |
| recorded variants (2) | `openai/gpt-6-luna (tools)` C3 · `openai/gpt-6-luna (no tools)` C3 | - |
| recorded trials (3) | `0` C3 · `1` C3 · `2` C3 | - |
| sample columns (4) | `segment` C16 · `arm` C16 · `converted` C16 · `pre_period_metric` C16 | - |
| README sections that name the table (2) | `## Results` C5 · `## Limitations` C5 | - |
| extractor outcomes (2) | exit `1` C11 · exit `0` C12 | - |
| live promote door (1) | two scores C7 | - |

- Claims naming an exit code: C6, C11, C12
- No other check claims more than the single case its proof exercises
- No route rows. The plan's Surface is none.

## Swept

- validation: C6
- failure modes: C6, C11
- idempotency: n/a - a second promote overwrites the baseline from the run you pass
- authorization: n/a - the key stays in the caller environment; this feature adds no permission check
- concurrency: existing - `_recorder` holds a lock and the live command uses concurrency `1`
- data lifecycle: C3, C6
- dependency failure: C1, C6
- state transitions: C7, C8
- observability: C15

## Handoff

- S1 = 15k, all in the eval command and the committed score; S2 enters the baseline at 23k; S3 enters the shell script at 24k; S4 enters the demo at 38k. `wc -c` on the files these slices touch is 133582 bytes, about 33k tokens, under the 150k budget. One builder.

- **Boundary:** C1–C8, C10–C11, and C13–C16 are proved at `b27ee03`. The live run's `errors.md` is at `5fba6f1`. The freshness test requires `pass^{trials}`: a 3-trial run must say `pass^3`, and any other trial count must not. The 3.6% fixture stays. C12 stays red because `test_the_page_can_replay_a_trajectory` still needs two turns on the first replayed case, and that case has one. C9 is not dispatched.
- **Settled mid-build:** The local `.env` and the Actions secrets were in place before the run. `uv run --env-file .env` loaded the key. A blank `PEER_LOG_PATH` was treated as the default log path, because an empty value made every review raise `IsADirectoryError` and discard the model's answer. The live command used `--concurrency 8`.
- **Abandoned:** The first live attempt exited 1 and left `fixtures/recorded.jsonl` untouched. Its results directory was deleted and `results/LATEST` was restored before the successful run.
