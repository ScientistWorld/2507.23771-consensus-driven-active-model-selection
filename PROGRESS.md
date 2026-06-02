# Progress

## What Works

The paper's CODA algorithm is implemented in `method/coda_reproduction.py`. It initializes consensus-informed Dirichlet confusion priors, computes probability of best model via row-wise Beta integration, selects labels by expected information gain, and updates beliefs with queried labels.

The reusable data path uses public ModelSelector raw prediction files converted to portable `.npz` arrays in `data/downloads/modelselector/`. Evaluation in `eval/evaluate_results.py` reads output traces and does not import from `method/`.

## Results

A one-step smoke test on GLUE RTE ran successfully for CODA, random sampling, and uncertainty sampling.

The first full 100-label, 5-seed GLUE RTE job completed, but did not support the core claim. It produced cumulative regret values of 432.42 for random sampling, 383.75 for uncertainty sampling, and 477.26 for CODA when including the step-0 prior regret. After matching the paper's post-query cumulative regret convention, those traces correspond to 420.51 for random sampling, 371.84 for uncertainty sampling, and 462.45 for CODA. Since CODA was worse than the baselines, the milestone remains `method_runs`.

The retry fixes a reproduction divergence found against the released implementation: CODA's EIG acquisition evaluates hypothetical observations with a unit pseudo-count, while the actual posterior update uses the smaller online learning rate. The previous implementation used the learning rate for both, making EIG too flat on RTE.

Paper reference for GLUE RTE at step 100:

- Random sampling cumulative regret: 375.7
- Uncertainty cumulative regret: 390.3
- CODA cumulative regret: 283.8

## Remaining

Run the corrected CPU job, inspect `scoring/scores.json`, compare with `scoring/reference.json`, and iterate if CODA does not outperform the baselines on the core RTE setting.

## Issues

The login-node PyTorch install path produced a broken GPU wheel, so the runnable reproduction avoids PyTorch by using verified ModelSelector `.npy` files converted to `.npz`. The authors' Google Drive CODA archive was still downloaded and lightly inspected.

## Deviations from Paper

The core job uses the paper's GLUE RTE black-box/class-ID prediction setting through the original ModelSelector raw files instead of the authors' `.pt` release tensors. This is a data-format substitution only: the candidate predictions, labels, and task match the paper's ModelSelector-derived RTE benchmark, and CODA's algorithmic procedure is unchanged.
