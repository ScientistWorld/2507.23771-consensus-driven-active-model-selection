# Evaluation Targets

## Primary: glue_rte_active_selection

Tests whether an active model selector can identify a high-accuracy candidate classifier on GLUE RTE using a fixed 100-label budget.

- **Metric:** `cumulative_regret_at_100` (lower is better)
- **Run:** `bash scripts/reproduce.sh glue_rte 100 5`

## Additional: glue_sst2_active_selection

Tests the same active model selection objective on GLUE SST2, which has a larger item set and a different candidate pool.

- **Metric:** `cumulative_regret_at_100` (lower is better)
- **Run:** `MERGE_EXISTING=1 bash scripts/method.sh glue_sst2 100 2 64`

## Secondary: zero_label_selection

Tests whether a selector can choose a strong model before querying any labels, matching the paper's Table 5 setup.

- **Experiments:** `glue_rte_unsupervised_selection`, `glue_sst2_unsupervised_selection`
- **Metric:** `initial_regret` (lower is better)
- **Runs:** `MERGE_EXISTING=1 bash scripts/unsupervised.sh glue_rte 128` and `MERGE_EXISTING=1 bash scripts/unsupervised.sh glue_sst2 64`

## Constraint: fixed_budget_and_task_size

Ensures improvements are obtained under the same annotation budget and benchmark scale rather than by querying more labels or changing the candidate pool.

- **Metrics:** `label_budget` (lower is better when performance is comparable), `num_models` and `num_items` (must match the benchmark)
- **Run:** `bash scripts/evaluate.sh outputs/coda glue_rte glue_rte_active_selection`
