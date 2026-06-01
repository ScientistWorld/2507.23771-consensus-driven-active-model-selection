# Evaluation

## Metrics

The primary metric is cumulative regret at 100 label queries. At each step, regret is the difference between the empirical 0-1 loss of the selected model and the empirical 0-1 loss of the true best model on the full benchmark labels. Lower is better.

Secondary metrics include instantaneous regret at step 100 and fixed-resource checks: the label budget, number of candidate models, and number of evaluated items.

## Baselines and Targets

The core reproduced benchmark uses GLUE RTE from the paper's ModelSelector-derived task suite. Paper-reported targets are taken from the active model selection tables:

- Random sampling: cumulative regret 375.7 at step 100; instantaneous regret 1.2 at step 100.
- Uncertainty sampling: cumulative regret 390.3 at step 100; instantaneous regret 0.0 at step 100.
- Proposed method: cumulative regret 283.8 at step 100; instantaneous regret 0.0 at step 100.

## Evaluation Protocol

Evaluation reads method-produced JSON traces from `outputs/coda/<task>/<method>/seed_*.json`. It does not call implementation code from `method/`. Each trace records the selected model at each active labeling step, and the evaluator aggregates metrics across seeds into `scoring/scores.json`.
