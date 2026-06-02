# Train/Test Evaluation Design Notes

## Checks Performed

- Read `scripts/evaluate_train.sh`, `scripts/evaluate_test.sh`, `eval/train/evaluate.py`, `eval/test/evaluate.py`, and `scoring/reference.json` for train/test copy-paste errors, hardcoded method defaults, method detachment, and train/test leakage.
- Confirmed the split evaluators read method-produced `seed_*.json` traces and benchmark prediction data, rather than importing or running `method/`.
- Grepped for cross-imports between train and test evaluators and for forbidden paper-agreement or hardware runtime metrics.
- Checked `bash -n` on both train/test shell wrappers.
- Confirmed no `scoring/scores_train.json` or `scoring/scores_test.json` stubs exist.

## Changes Made

- Expanded both train and test evaluators so no-argument runs enumerate every active/unsupervised experiment in `scoring/reference.json` instead of scoring only `glue_rte_active_selection`.
- Kept single-task overrides available: passing `task` and optional `experiment` still evaluates just that slice.
- Added `UNSUPERVISED_RESULTS_DIR` support in both wrappers so zero-label experiments can be scored from `outputs/coda_unsupervised` without mixing them with active 100-label traces.
- Added top-level `slice: "train"` and `slice: "test"` markers to the generated train/test score files.
- Added active-selection triangulation metrics computed from the same trace interface: `cumulative_regret_at_25`, `cumulative_regret_at_50`, `initial_regret`, `instant_regret_at_50`, and `near_optimal_at_100`.
- Added required `description` fields for all nonzero metrics in `scoring/reference.json`.
- Kept all scoring metrics on the underlying scientific quantities: regret, near-optimal selection, label budget, candidate count, and item count. No metric rewards matching published paper values.

## Runtime Concerns For Next Iteration

- The train/test split is a post-hoc item partition over the benchmark labels. The method traces are generated once on the full pool, then rescored on train and test partitions. This is lightweight and anti-shortcut useful, but the interaction itself is not rerun separately per partition.
- Existing local outputs may have active and zero-label traces mixed under `outputs/coda` from prior runs. The updated wrappers default zero-label scoring to `outputs/coda_unsupervised`; the runner should verify those traces exist for both unsupervised experiments before running the full split.
- The active curve metrics assume traces contain at least the requested budget steps. If a future method emits fewer than 101 steps for a 100-label experiment, the evaluator will use the last available step for instantaneous regret; the runner should flag any unexpectedly short traces.
- `near_optimal_at_100` uses the paper's one-percentage-point regret threshold. It may saturate on easy slices, which is why it is weighted as a secondary triangulation metric rather than the primary objective.
