# Write-up

Billing: `points = uncached_in*3 + cached_in*0.3 + out*15`.

## Why the baseline was expensive
- Q1: random run-id + few-shot + "think step by step / full reasoning" made the model emit a ~480-token trace (≈7,200 pts of output per ticket).
- Q2: 4 LLM calls; two of them pasted 80k–120k tokens of help-center filler; the retrieve node asked the LLM to *print* 8 articles, and its output was re-sent into compose and audit.

## Q1 changes (`question1/agent.py`)
- Dropped run-id, few-shot block, and reasoning instructions; ask for compact JSON only.
- Static instructions + KB first, ticket data last (cache-friendly prefix layout).
- Compact (non-indented) order snapshot.

## Q2 changes (`question2/agent.py`)
Same four nodes, but only `compose` calls the LLM:
- triage: code derives keywords from order facts + ticket text
- retrieve: lexical top-3 over the KB (the big KB is never put in a prompt)
- compose: one small LLM call
- audit: code validates action and that the article exists; falls back to one retry with the full real KB only if validation fails

## Results (all graded fields match `eval/reference.py`, incl. holdout T-1091)

| Ticket | Q1 pts | Q2 pts |
|---|---:|---:|
| T-1001 | 1830 | 1494 |
| T-1002 | 1707 | 1371 |
| T-1003 | 1668 | 1311 |
| T-1004 | 1629 | 1293 |
| T-1091 | 1671 | 1356 |

Targets: Q1 ≤ 3000, Q2 ≤ 2500 — met.

### Per-call breakdown
- Q1 T-1001 (physical, refund): `handle_ticket` in 380 / cached 0 / out 46 → 380·3 + 46·15 = **1830**
- Q1 T-1002: in 374 / out 39 → **1707**
- Q2 T-1001 (physical fulfilment path, REFUND): `compose` in 268 / out 46 → **1494** (triage/retrieve/audit = 0 LLM calls)
- Q2 T-1003 (denial/escalation path, ESCALATE_TIER2): one `compose` call → **1311**

## Note
`test_baseline_cost_is_high` in the repo asserts the *baseline* is expensive, so it now fails by design.
