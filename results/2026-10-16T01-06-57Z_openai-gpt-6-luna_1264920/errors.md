# Error taxonomy

| flaw | false_block | false_ship | format_fail | missed_flaw | other_wrong | skipped_srm_check |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 14 | 3 | 13 | 0 | 19 | 0 |
| novelty | 0 | 1 | 14 | 17 | 0 | 0 |
| outliers | 0 | 1 | 9 | 0 | 20 | 0 |
| peeking | 0 | 17 | 25 | 15 | 3 | 0 |
| simpson | 0 | 18 | 13 | 0 | 29 | 0 |
| srm | 0 | 0 | 19 | 0 | 0 | 11 |

Errors: 261. Matrix sum: 261.

## Process

- `openai/gpt-6-luna (tools)`: srm_checked_before_lift 98.2% (n=165)
- `openai/gpt-6-luna (no tools)`: srm_checked_before_lift 0.0% (n=165)

## Shortest misses

- `clean-01` (clean, openai/gpt-6-luna (no tools), trial 0): format_fail in 1 turns. Truth NO_EFFECT, verdict None.
- `clean-01` (clean, openai/gpt-6-luna (no tools), trial 1): other_wrong in 1 turns. Truth NO_EFFECT, verdict INCONCLUSIVE.
- `clean-01` (clean, openai/gpt-6-luna (no tools), trial 2): format_fail in 1 turns. Truth NO_EFFECT, verdict None.
- `clean-02` (clean, openai/gpt-6-luna (no tools), trial 0): other_wrong in 1 turns. Truth NO_EFFECT, verdict INCONCLUSIVE.
- `clean-02` (clean, openai/gpt-6-luna (no tools), trial 1): format_fail in 1 turns. Truth NO_EFFECT, verdict None.
