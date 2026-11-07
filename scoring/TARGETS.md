# Evaluation Targets

All targets are produced by the evaluator from the selections it records in
every round; selectors only answer the oracle protocol. Visible slice:
`bash scripts/evaluate_train.sh` (or `scripts/evaluate.sh`). Hidden slice:
`bash scripts/evaluate_test.sh` (no arguments; scores every selector in
`outputs/selectors/`). Reproduce the reference end to end with
`bash scripts/reproduce.sh`.

## Primary: glue_sst2_active_selection, glue_qnli_active_selection

Identify a high-accuracy candidate on GLUE SST2 and on GLUE QNLI (a fixed
1200-item sample) within a 100-label budget. Both carry the same weight in
the ranking.

- **Metric:** `cumulative_regret_at_100` (lower is better); also
  `cumulative_regret_at_25/50`, `initial_regret`, `instant_regret_at_50/100`,
  `near_optimal_at_100`, `label_budget`.

## Headroom: glue_cola_active_selection

The same objective on the held-out half of GLUE CoLA (no visible part), the paper's documented failure case
(imbalanced labels, biased candidates), where the reference selector is
worse than random and uncertainty sampling. Same metrics and budget.

## Secondary: zero-label selection

`glue_rte_unsupervised_selection`, `glue_sst2_unsupervised_selection`:
selection with a 0-label budget, scored by `initial_regret` (lower is better).

## Constraints (scored, and enforced by the evaluator)

- **Label budget:** `label_budget` is the number of labels the oracle
  actually revealed (lower is better at equal regret). The oracle refuses to
  go past 100 labels (0 for zero-label experiments).
- **Fixed pool and data:** the candidate predictions, the item pool, the
  labels, the seeds and the best-candidate reference are fixed by `eval/`;
  a selector cannot change them, and the evaluators check data checksums.
- **Valid protocol:** a crash, timeout, malformed or out-of-range answer,
  repeated query, or query past the budget scores the selector as failed on
  that experiment (worst candidate every round, full budget spent).
