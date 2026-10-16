# LESSONS - auto-maintained by scripts/lessons.py

> Machine-owned. Do NOT hand-edit. Changes are overwritten on the next `lessons.py` write.
> Canonical state lives in `.specs/lessons.json`. Edit lessons only via the script.
> promote_threshold=2 distinct features · window_days=45 · quarantine_threshold=2

## Confirmed (load these at Plan/Checks)

Corroborated across multiple features. Safe to apply as guidance.

_none_

## Candidates (under observation - do NOT load as guidance yet)

Seen once or not yet corroborated. Tracked, not trusted.

### L-001 - Assert every literal the check names, including words such as live, not only the identifiers beside them.
- signal: `ac_gap` · recurrence: 1 feature(s) · harmful: 0
- features: live-luna-run
- evidence: C5
- last seen: 2026-10-16T00:35:28Z

### L-002 - Do not require a README command runner to exit 0 while the suite it executes still fails a trajectory-length assertion.
- signal: `gate_fail` · recurrence: 1 feature(s) · harmful: 0
- features: live-luna-run
- evidence: C12
- last seen: 2026-10-16T00:35:28Z

## Quarantined (failed when applied - ignore)

A confirmed lesson that recurred alongside failure. Kept for the maintainer to review.

_none_
