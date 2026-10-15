# Error taxonomy

| flaw | false_ship | missed_flaw | skipped_srm_check |
| --- | ---: | ---: | ---: |
| clean | 0 | 0 | 0 |
| novelty | 1 | 10 | 0 |
| outliers | 0 | 0 | 0 |
| peeking | 0 | 20 | 0 |
| simpson | 1 | 0 | 0 |
| srm | 0 | 1 | 10 |

Errors: 43. Matrix sum: 43.

## Process

- `claude-opus-5 (tools)`: srm_checked_before_lift 100.0% (n=55)
- `claude-opus-5 (no tools)`: srm_checked_before_lift 0.0% (n=55)

## Shortest misses

- `novelty-01` (novelty, claude-opus-5 (no tools), trial 0): missed_flaw in 1 turns. Truth EXTEND, verdict NO_EFFECT.
- `novelty-02` (novelty, claude-opus-5 (no tools), trial 0): missed_flaw in 1 turns. Truth EXTEND, verdict NO_EFFECT.
- `novelty-03` (novelty, claude-opus-5 (no tools), trial 0): missed_flaw in 1 turns. Truth EXTEND, verdict NO_EFFECT.
- `novelty-04` (novelty, claude-opus-5 (no tools), trial 0): missed_flaw in 1 turns. Truth EXTEND, verdict NO_EFFECT.
- `novelty-05` (novelty, claude-opus-5 (no tools), trial 0): missed_flaw in 1 turns. Truth EXTEND, verdict NO_EFFECT.
