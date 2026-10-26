# peer-ai

An agent that reviews and designs A/B tests with a deterministic statistics toolbox, and publishes the score of that agent with an interval.

Requires Python 3.14+ and `uv`.

```bash
uv sync
uv run peer eval --gate replay
uv run peer demo
uv run pytest -m "not live and not docker and not slow"
```

## Live model

The example endpoint is OpenRouter. Copy `.env.example` to `.env` and set `LLM_API_KEY` to an OpenRouter key.

| Variable | Value |
| --- | --- |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` |
| `LLM_MODEL` | `openai/gpt-6-luna` |

OpenRouter's list price on 2026-10-15 is $0.10 per million input tokens and $0.50 per million output tokens. That card is `evals/pricing.yaml`. A model that is not on the card gets `cost_usd: null`.

```bash
uv run peer review fixtures/sample.parquet  # needs LLM_API_KEY
uv run peer eval --model openai/gpt-6-luna --trials 3 --gate live --out results  # needs LLM_API_KEY
```

The weekly workflow (`.github/workflows/live-eval.yml`) uses the same model. It skips when `LLM_API_KEY` or `LLM_BASE_URL` is missing. After a live run, `uv run peer baseline promote results/<run_id>/results.json` and `uv run peer report --write-readme` replace the table below.

`make demo` is the same offline demo. `make sql` runs `analysis/eval_queries.sql` against `results/*/cases.parquet`.

## What the gates catch

The replay gate (`uv run peer eval --gate replay`, and the pre-commit hook `uv run peer eval --gate`) compares the recorded fixture with `evals/baseline.json` → `replay_expected`. The turns are fixed, so this catches a harness, parser, scoring, or schema regression. It does not catch a model or prompt regression.

The live gate (`.github/workflows/live-eval.yml`, weekly and manual) runs the suite against a model. It fails when the Wilson upper bound of the new accuracy sits below the baseline accuracy, or when the false-ship rate rises by more than 2 points. A pull request that edits `src/peer_agent/prompt.md` or `src/peer_agent/skills/**` runs one live trial when the secrets exist, and posts a warning otherwise. It does not block a PR that has no key.

## Results

This table is the live run of `openai/gpt-6-luna` with trials 3 in `results/LATEST`. `uv run peer report --write-readme` renders it. Do not edit the numbers by hand.

<!-- results:start -->

| Variant | Accuracy (95% Wilson CI) | pass^3 | False ship | False block | Cost/case | p50 latency | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| openai/gpt-6-luna (tools) | 41.8% (28.1%–53.2%) | 38.2% | 24.2% | 2.4% | — | 19.73s | 55 |
| openai/gpt-6-luna (no tools) | 0.0% (0.0%–6.5%) | 0.0% | 0.0% | 9.1% | — | 19.06s | 55 |

McNemar `openai/gpt-6-luna (tools)` vs `openai/gpt-6-luna (no tools)`: b=24 (only A right), c=0 (only B right), p=1.19e-07 (exact).

Source: [`results/2026-10-16T01-06-57Z_openai-gpt-6-luna_1264920/results.json`](results/2026-10-16T01-06-57Z_openai-gpt-6-luna_1264920/results.json).

Model `openai/gpt-6-luna`, run `2026-10-16T01-06-57Z_openai-gpt-6-luna_1264920`, 2026-10-16, git `1264920`.

<!-- results:end -->

## How the review works

The agent calls tools instead of inventing a p-value. SRM (binomial sample-ratio test) and CUPED are computed by tea-tasting; `stats.py` adds unit-of-analysis collapse, TOST, segment heterogeneity (Cochran Q), novelty trend and sequential boundaries, plus seeded bootstrap and Beta–Binomial Bayesian checks and a Gelman–Carlin Type M/S audit (from the owner's [`ab-test-analytics`](https://github.com/matheusbuniotto/ab-test-analytics) repo).

`sequential` defaults to Lan-DeMets O'Brien-Fleming spending. `method="bonferroni"` is the older rule: compare the final p-value with `alpha / looks`. That is a correction, not an alpha-spending design.

Peeking cases in `evals/suite.yaml` are planted. The simulator draws a true lift of 0, looks at each cumulative day, and keeps the first day `d >= 3` with `p < 0.05`. The truth label is `INVALID`. The review prompt says the team checked daily and stopped on that day. Five `outliers` cases have lift 0; the truth is `NO_EFFECT`, and a `SHIP` on a distorted CUPED covariate is a false ship.

Verdicts are `NO_EFFECT`, `INCONCLUSIVE`, `INVALID`, `SHIP`, and `EXTEND`.

## Layout

```text
src/peer_agent/
  agent.py        tool loop, usage, latency
  evals.py        suite runner, results schema v2, replay and live gates
  scoring.py      Wilson, pass^k, McNemar, error taxonomy
  stats.py        SRM, CUPED, TOST, segments, novelty, sequential, bootstrap, Bayes, Type M/S
  sim.py          planted flaws, including the peeking stopping rule
  report.py       README table and error report
  demo.py         one-command demo
  export_braintrust.py
  cli.py          review, design, chat, web, eval, report, demo, compare, export
  web/            static page that reads results.json
  skills/         review protocol loaded by the agent
evals/            suite.yaml, baseline.json, results.schema.json, pricing.yaml
results/          committed run (results.json, cases.parquet, LATEST)
analysis/         SQL over cases.parquet
fixtures/         sample.parquet and the recorded replay
```

## Commands

```bash
uv run peer review fixtures/sample.parquet -o /tmp/review.md  # needs LLM_API_KEY
uv run peer design "ship a 5% conversion lift in 14 days"  # needs LLM_API_KEY
uv run peer eval --replay fixtures/recorded.jsonl --gate replay
uv run peer report --errors
uv run peer compare results/LATEST results/LATEST
```

`peer baseline promote results/<run_id>/results.json` rewrites `evals/baseline.json` from a run. `peer export braintrust <results.json>` needs `BRAINTRUST_API_KEY` and the optional `braintrust` extra; without the key it exits 2.

Set `OTEL_EXPORTER_OTLP_ENDPOINT` to export spans over OTLP/HTTP in addition to Logfire. Each eval case is a span `eval.case` with `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `peer.cost_usd`, `peer.error_category`, and `peer.correct`.

## Limitations

The numbers in the table are a live run of `openai/gpt-6-luna` with trials 3. `fixtures/recorded.jsonl` stores that run's turns, and the replay gate re-scores them.

Replay rows have null tokens and a null cost. Latency is the time to replay the recorded turns and run the tools. The Luna price card applies only when a live run reports token counts.

The replay gate cannot see a prompt change, because the recorded answers do not change when the prompt does. That is the live gate's job.

Peeking truth is `INVALID` because the final p-value is an optional-stopping artifact. A reader who would rather call that `EXTEND` will disagree with ten cases.

There is no human-labelled LLM judge. Scoring is the match between the parsed verdict and the planted truth, plus the process check that `check_srm` ran before a lift was read.

The suite is 55 simulated cases (10 each of SRM, novelty, Simpson, peeking, and clean, plus 5 outliers). It is not production traffic.
