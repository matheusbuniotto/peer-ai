# Project state

## Decisions

No active decisions yet. The baseline split in `live-luna-run` becomes AD-001 if the plan is approved.

## Handoff

**Feature**: live-luna-run
**Where**: C1-C8, C10-C11, and C13-C16 committed at `b27ee03`. The live run's `errors.md` is at `5fba6f1`. The freshness test now requires `pass^{trials}`.
**In progress**: none
**Next step**: prek, then a push of `ft-eval-honesty` only after an explicit yes, then dispatch `live-eval.yml`, then the Verifier
**Blockers**: `test_the_page_can_replay_a_trajectory` stays red on the kept 3.6% fixture (first replayed trajectory has one turn). C12 fails with it. C9 needs the promoted baseline on the ref Actions checks out.
**Uncommitted**: none
**Branch**: ft-eval-honesty
