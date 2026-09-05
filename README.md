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
