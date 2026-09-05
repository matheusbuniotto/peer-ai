# 08 — Docker sandbox + `run_python` tool

Type: AFK
Source: docs/plans/phase-3-plan.md (phase 3)

## What to build

`sandbox.py` is currently a no-op context manager. Build the real thing:
read-only data mount, no network, timeout, memory cap — per the tracer
file's own docstring and its three `@pytest.mark.docker` tests.

**Considered and rejected:** `pydantic_ai_harness.modal_sandbox`, a
built-in cloud-sandboxing capability (checked because "use what already
exists" applies here too). Rejected because the tracer file's own
`docker` marker ("needs a container runtime") and its
"cannot reach the network"/"cannot write to `/data/exp.parquet`" tests
fix the contract to a locally-run, network-disabled container — a cloud
sandbox is a different trust/network model and would need its own
verification story the test file doesn't describe. Local Docker stays as
originally planned.

`docker run --rm -i --network none --memory {mb}m --read-only -v
{tmpdir}:/data:ro <image> python -c -`, code piped via stdin (avoids
arg-length/shell-injection concerns), `subprocess.run(timeout=self.timeout)`.
`Sandbox.run(code, df) -> RunResult(ok, stdout, stderr)` writes `df` to
`{tmpdir}/exp.parquet` first. On `TimeoutExpired`, kill the (named, not
anonymous) container via `docker kill <name>` — the client subprocess
dying doesn't kill the container otherwise. The memory-eat test relies on
the container's own OOM kill (exit 137) → `ok=False`.

Add a `Dockerfile` (repo root) on a `python:3.14-slim` base, pinning
`numpy`/`pandas`/`pyarrow` to the same versions as `uv.lock`.

Wire `run_python` into `analysis_tools(sandbox)` per issue 04's contract
so `Agent.from_env()` (and therefore `peer chat`/`peer web`/`peer review`)
can actually reach it.

## Acceptance criteria

- [ ] `docker build -t peerai-sandbox .` succeeds
- [ ] `uv run pytest tests/test_phase_3_tracer.py -m docker -k "sandbox or agent_writes_its_own_code or still_prefers_the_built_tool"`
      fully green against a real container runtime
- [ ] `sandbox.run("import pandas as pd; print(pd.read_parquet('/data/exp.parquet').arm.mean())", df)`
      returns `ok=True` with the expected stdout
- [ ] Network access and writes to `/data` both fail (`ok=False`)
- [ ] An infinite loop and a 4GB allocation both fail (`ok=False`,
      timeout/OOM respectively) without hanging the test run
- [ ] `peer chat`/`peer review` (issues 02/03) can trigger `run_python`
      when asked something no built stats tool covers
- [ ] `uv run prek` and `uv run ty check` clean

## Blocked by

- 04 — Tool-registry reconciliation
