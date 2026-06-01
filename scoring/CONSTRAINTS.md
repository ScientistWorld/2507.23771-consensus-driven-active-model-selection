# Scientist Constraints

## Data

Use the provided GLUE RTE prediction benchmark in `data/downloads/modelselector/glue_rte.npz` or regenerate it with `data/prepare_modelselector.py`. The candidate model predictions, labels, number of items, and number of candidate models must remain fixed. Do not add external labeled examples or replace the benchmark with a different task when reporting `glue_rte_active_selection`.

## Label Budget

The active selector must query at most 100 ground-truth labels for the primary score. Querying more labels and reporting the same experiment would bypass the label-efficiency question.

## Output Contract

Methods must write traces to `outputs/coda/<task>/<method>/seed_*.json` with selected model and regret per step, then be scored by `scripts/evaluate.sh`. Evaluation must not import the submitted method implementation.

## Compliance Checklist

- [ ] Uses the fixed `glue_rte` prediction and label arrays.
- [ ] Reports the primary result at 100 queried labels or fewer.
- [ ] Does not train or add new candidate classifiers for this experiment.
- [ ] Writes method outputs in the trace format consumed by `eval/evaluate_results.py`.
