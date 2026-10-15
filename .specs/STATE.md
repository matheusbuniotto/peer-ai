# Project state

## Decisions

No active decisions yet. The baseline split in `live-luna-run` becomes AD-001 if the plan is approved.

## Handoff

**Feature**: live-luna-run
**Where**: C1-C8, C10-C11, and C13-C16 committed at `b27ee03`. The live run's `errors.md` is written. C12 is red on two pre-existing tests. C9 is not dispatched.
**In progress**: none
**Next step**: decide the `pass^3` assertion and the one-turn first trajectory, then prek, push `ft-eval-honesty`, dispatch `live-eval.yml`, then the Verifier
**Blockers**: `test_readme_table_is_fresh` forbids the trials-3 column name. `test_the_page_can_replay_a_trajectory` needs two turns on the first replayed case. C9 needs the promoted baseline on the ref Actions checks out.
**Uncommitted**: none after the error-report commit
**Branch**: ft-eval-honesty
