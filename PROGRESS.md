# Progress

## What Works

The paper's CODA algorithm is implemented in `method/coda_reproduction.py`. It initializes consensus-informed Dirichlet confusion priors, computes probability of best model via row-wise Beta integration, selects labels by expected information gain, and updates beliefs with queried labels.

The reusable data path uses public ModelSelector raw prediction files converted to portable `.npz` arrays in `data/downloads/modelselector/`. Evaluation in `eval/evaluate_results.py` reads output traces and does not import from `method/`.

## Results

A one-step smoke test on GLUE RTE ran successfully for CODA, random sampling, and uncertainty sampling. Full 100-label results are not available yet; `scripts/run.sh` is prepared to run the 100-step, 5-seed RTE experiment and write `scoring/scores.json`.

Paper reference for GLUE RTE at step 100:

- Random sampling cumulative regret: 375.7
- Uncertainty cumulative regret: 390.3
- CODA cumulative regret: 283.8

## Remaining

Run the submitted CPU job, inspect `scoring/scores.json`, compare with `scoring/reference.json`, and iterate if CODA does not outperform the baselines on the core RTE setting.

## Issues

The login-node PyTorch install path produced a broken GPU wheel, so the runnable reproduction avoids PyTorch by using verified ModelSelector `.npy` files converted to `.npz`. The authors' Google Drive CODA archive was still downloaded and lightly inspected.

## Deviations from Paper

The core job uses the paper's GLUE RTE black-box/class-ID prediction setting through the original ModelSelector raw files instead of the authors' `.pt` release tensors. This is a data-format substitution only: the candidate predictions, labels, and task match the paper's ModelSelector-derived RTE benchmark, and CODA's algorithmic procedure is unchanged.
