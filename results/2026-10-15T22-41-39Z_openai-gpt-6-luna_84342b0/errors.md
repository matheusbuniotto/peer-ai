# Error taxonomy

| flaw | false_block | false_ship | format_fail | missed_flaw | other_wrong | skipped_srm_check |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 1 | 0 | 57 | 0 | 0 | 0 |
| novelty | 0 | 0 | 57 | 1 | 0 | 0 |
| outliers | 0 | 0 | 30 | 0 | 0 | 0 |
| peeking | 0 | 5 | 52 | 3 | 0 | 0 |
| simpson | 0 | 0 | 56 | 0 | 4 | 0 |
| srm | 0 | 0 | 57 | 0 | 0 | 1 |

Errors: 324. Matrix sum: 324.

## Process

- `openai/gpt-6-luna (tools)`: srm_checked_before_lift 12.1% (n=165)
- `openai/gpt-6-luna (no tools)`: srm_checked_before_lift 0.0% (n=165)

## Shortest misses

- `clean-01` (clean, openai/gpt-6-luna (tools), trial 0): format_fail in 1 turns. Truth NO_EFFECT, verdict None.
- `clean-01` (clean, openai/gpt-6-luna (tools), trial 1): format_fail in 1 turns. Truth NO_EFFECT, verdict None.
- `clean-01` (clean, openai/gpt-6-luna (tools), trial 2): format_fail in 1 turns. Truth NO_EFFECT, verdict None.
- `clean-01` (clean, openai/gpt-6-luna (no tools), trial 0): format_fail in 1 turns. Truth NO_EFFECT, verdict None.
- `clean-01` (clean, openai/gpt-6-luna (no tools), trial 1): format_fail in 1 turns. Truth NO_EFFECT, verdict None.
