## About the project
**peer-agent** — an AI agent (data scientist) that design and reviews A/B tests the way a skeptical senior data scientist would.

It reviews finished experiments and designs new ones, calling a small toolbox of calibrated statistics — sample ratio mismatch, sequential testing, CUPED, segment scans — and writing its own analysis code in a sandbox when nothing fits. It refuses to report a lift from a broken split, resists inventing a finding when nothing happened, and asks a PM what they'll do if the result comes back flat before it will write a spec.

Every experiment it's graded on is simulated, so the true answer is known in advance. That makes the agent measurable rather than merely impressive: 50 cases across five ways a test lies to you, scored against ground truth, with a no-tools baseline showing what the toolbox is actually worth. Designs are graded the same way — the agent writes a spec, the simulator runs it 500 times against a planted effect, and realized power is compared to promised power.

The eval suite runs in CI as a merge gate. A prompt change that drops detection accuracy fails the build.


## Proposed initial architecture (can be improved, not written in stone)
peer-agent/
├── src/peer_agent/
│   ├── types.py        Verdict, DesignSpec, frozen result dataclasses
│   ├── sim.py          make_case() — experiments with known truth. The flaws
│   │                   (SRM, NOVELTY, SIMPSON, PEEKING, OUTLIERS) are functions
│   │                   that perturb a dataframe, applied in a list.
│   ├── stats.py        check_srm, analyze, sequential, scan_segments,
│   │                   check_novelty, check_guardrails. Functions over a
│   │                   dataframe, returning frozen dataclasses. Imports numpy,
│   │                   pandas, scipy. Nothing else. A data scientist should be
│   │                   able to open this file and check the math.
│   ├── design.py       power_analysis and simulate_design. Same rules as stats.
│   ├── sandbox.py      Sandbox — context manager over a container.
│   ├── tools.py        default_tools() — JSON schemas wrapping stats/design/
│   │                   sandbox. The only place the model's vocabulary lives.
│   ├── agent.py        Agent, Trajectory, OverBudget. The loop.
│   ├── prompt.md       The protocol: validate, guardrails, primary, segments.
│   ├── evals.py        load_suite, run_suite. Writes results.json.
│   ├── mcp.py          Same tools over stdio.
│   └── cli.py          `abcode review`, `abcode design`.
├── evals/
│   ├── suite.yaml      50 cases: seed, flaws, true verdict.
│   └── baseline.json   Committed scores. The merge gate compares to this.
├── fixtures/
│   ├── recorded.jsonl  Model responses for offline replay.
│   └── sample.parquet
├── tests/              conftest, stats, design, agent, sandbox, e2e.
└── web/                The report page. Reads results.json, nothing else.
