# Scientist Constraints

## Data

Use the provided GLUE prediction benchmarks in `data/downloads/modelselector/glue_rte.npz`, `data/downloads/modelselector/glue_sst2.npz`, and `data/downloads/modelselector/glue_mrpc.npz`, or regenerate them with `data/prepare_modelselector.py`. The candidate model predictions, labels, number of items, and number of candidate models must remain fixed for each experiment. Do not add external labeled examples or replace the benchmark with a different task when reporting an active-selection experiment.

## Label Budget

The active selector must query at most 100 ground-truth labels for the primary score. Querying more labels and reporting the same experiment would bypass the label-efficiency question.

The unsupervised selection experiments must query exactly zero labels. Reporting `glue_rte_unsupervised_selection` or `glue_sst2_unsupervised_selection` after observing labels changes the experiment into active selection and invalidates the Table 5 comparison.

## Output Contract

Methods must write traces to `outputs/coda/<task>/<method>/seed_*.json` with the selected model at each step, then be scored by `scripts/evaluate.sh`. Evaluation recomputes regret from the fixed benchmark labels and must not import the submitted method implementation.

## Compliance Checklist

- [ ] Uses the fixed task prediction and label arrays.
- [ ] Reports the primary result at 100 queried labels or fewer.
- [ ] Reports unsupervised results with exactly zero queried labels.
- [ ] Does not train or add new candidate classifiers for this experiment.
- [ ] Writes method outputs in the trace format consumed by `eval/evaluate_results.py`.
