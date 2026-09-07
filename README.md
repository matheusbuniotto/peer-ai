# peer-ai

An AI agent that reviews and designs A/B tests the way a skeptical senior data scientist would.

---

I spent years looking at experimentation dashboards where teams celebrated double-digit lifts on broken tests.

The failure modes in A/B testing are almost always the same: someone stopped the test early the moment p dipped below 0.05, traffic allocation broke (sample ratio mismatch) and went unnoticed, or someone sliced the metrics across enough segments until one showed a green number purely by chance.

When you feed those same broken datasets into a standard LLM, it behaves like an eager junior intern: it nods along, writes a confident story about why the variant won, and completely misses that the underlying data is mathematically compromised.

I built `peer-ai` to see if an agent could be forced to be skeptical.

---

## How it works

The agent handles two core responsibilities: **reviewing** completed experiments and **designing** new ones.

It does not generate free-form statistical opinions out of thin air. Instead, it operates over a deterministic toolbox of statistical primitives:
- `check_srm`: Tests for sample ratio mismatch before touching variance or effect sizes.
- `sequential`: Adjusts for continuous monitoring and optional stopping.
- `variance_reduction`: Applies CUPED using historical pre-experiment covariates.
- `scan_segments`: Controls for multiple testing corrections across subgroups.
- `sandbox`: An isolated container environment for custom data manipulations when standard schemas don't fit.

If the traffic split is invalid, it refuses to report a lift. If the primary metric is flat, it resists inventing secondary findings. And before it designs a test, it asks what decision will actually be taken if the result comes back neutral.

---

## Evals: Ground truth instead of vibes

The hardest part of building domain-specific agents is knowing if they are actually reliable or just sounding articulate.

To keep this measurable, `peer-ai` is evaluated against 50 simulated experiment cases where the ground truth is mathematically known in advance:
- Planted Sample Ratio Mismatches (SRM)
- Novelty effects that decay over time
- Simpson's paradox across hidden user cohorts
- Peeking-induced false positives
- Extreme heavy-tailed outliers

Each case is run through the agent and scored against ground truth. Designs are evaluated similarly: the agent writes a specification, a simulator runs 500 Monte Carlo draws against a planted true effect, and realized statistical power is compared against promised power.

The eval suite runs in CI as a merge gate. If a prompt tweak or tool adjustment drops flaw-detection accuracy, the build fails.

---

## Architecture

```text
peer-agent/
├── src/peer_agent/
│   ├── types.py        Verdict, DesignSpec, frozen dataclasses
│   ├── sim.py          make_case() with planted flaws (SRM, Simpson, peeking)
│   ├── stats.py        SRM checks, sequential testing, CUPED, segment scans
│   ├── design.py       Power analysis (tea-tasting's Mean.solve_power) and
│   │                   Monte Carlo design simulation (Experiment.simulate)
│   ├── sandbox.py      Containerized code execution environment
│   ├── tools.py        JSON schemas exposing stats and sandbox to the model
│   ├── agent.py        The core reasoning and tool loop
│   ├── mcp.py          MCP server over stdio — `peer mcp` / `peer-mcp`
│   └── cli.py          CLI commands: review, design, and mcp
├── evals/
│   ├── suite.yaml      50 benchmark cases with known ground truth
│   └── baseline.json   Committed scores used as CI merge gate
└── tests/              Unit and integration test suite
```

---

## Quick Start

### Installation

Requires Python 3.11+ and `uv`.

```bash
git clone https://github.com/matheusbuniotto/peer-ai.git
cd peer-ai
uv sync
```

### Review an Experiment

```bash
uv run peer-ai review --data path/to/experiment.parquet --metric conversion_rate
```

### Run Evals & Tests

```bash
uv run pytest tests/
uv run python -m peer_agent.evals --suite evals/suite.yaml
```

---

## Design Choices

- Math transparency: `stats.py` relies strictly on `numpy`, `pandas`, and `scipy`. Any data scientist can open the file and verify the equations directly.
- Model Context Protocol: Includes an MCP server implementation over stdio (`peer mcp` / `peer-mcp`), allowing `peer-ai` to be plugged into Claude Code, Cursor, or external agent workbenches.

---

## Using it as an MCP server

The same toolbox any MCP client can call, over stdio:

```bash
uv run peer mcp          # from a checkout
uvx --from peerai peer-mcp   # from a wheel, nothing to install
```

Claude Code:

```bash
claude mcp add peer-agent -- uv run --directory /path/to/peerai peer mcp
```

Or in an `mcpServers` config block:

```json
{ "peer-agent": { "command": "uvx", "args": ["--from", "peerai", "peer-mcp"] } }
```

Tools take a `path` to a parquet file and load it themselves. The
review-protocol checklist is published as the `peer://protocol` resource, so a
client that reads resources gets the "validate first" rules along with the
tools.
