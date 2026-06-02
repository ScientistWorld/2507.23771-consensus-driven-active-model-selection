# Progress

## What Works

The paper's CODA algorithm is implemented in `method/coda_reproduction.py`. It initializes consensus-informed Dirichlet confusion priors, computes probability of best model via row-wise Beta integration, selects labels by expected information gain, and updates beliefs with queried labels.

The reusable data path uses public ModelSelector raw prediction files converted to portable `.npz` arrays in `data/downloads/modelselector/`. Evaluation in `eval/evaluate_results.py` reads output traces and does not import from `method/`.

## Results

A one-step smoke test on GLUE RTE ran successfully for CODA, random sampling, and uncertainty sampling.

The first full 100-label, 5-seed GLUE RTE job completed, but did not support the core claim. It produced cumulative regret values of 432.42 for random sampling, 383.75 for uncertainty sampling, and 477.26 for CODA when including the step-0 prior regret. After matching the paper's post-query cumulative regret convention, those traces correspond to 420.51 for random sampling, 371.84 for uncertainty sampling, and 462.45 for CODA. Since CODA was worse than the baselines, the milestone remains `method_runs`.

The retry fixes a reproduction divergence found against the released implementation: CODA's EIG acquisition evaluates hypothetical observations with a unit pseudo-count, while the actual posterior update uses the smaller online learning rate. The previous implementation used the learning rate for both, making EIG too flat on RTE.

The corrected 100-label, 5-seed GLUE RTE job supports the core claim:

- Random sampling cumulative regret: 420.51
- Uncertainty cumulative regret: 371.84
- CODA cumulative regret: 290.04

CODA is close to the paper's reported RTE result of 283.8 and substantially improves over the reproduced baselines under the same label budget.

Paper reference for GLUE RTE at step 100:

- Random sampling cumulative regret: 375.7
- Uncertainty cumulative regret: 390.3
- CODA cumulative regret: 283.8

## Expansion Result

An additional GLUE SST2 setting was run to broaden beyond the core RTE reproduction. The SST2 run uses the same algorithm and 100-label budget, with two seeds and a smaller numerical integration grid to fit the CPU scheduler timeout.

The SST2 expansion also supports the paper's claim:

- Random sampling cumulative regret: 345.47
- Uncertainty cumulative regret: 100.92
- CODA cumulative regret: 51.72

CODA essentially matches the paper's SST2 result of 51.7 and outperforms both reproduced baselines. This brings the workspace to `core_claim_plus`.

## Secondary Claim

The completed CPU job `f500fff1-d02` added the paper's zero-label unsupervised selection setting from Table 5. These runs evaluate the selected model before any labels are queried, while preserving the same candidate pools and full benchmark labels used for regret measurement.

- GLUE RTE EnsV initial regret: 14.801
- GLUE RTE CODA initial regret: 14.801
- GLUE SST2 EnsV initial regret: 3.555
- GLUE SST2 CODA initial regret: 3.555

These match the paper's reported Table 5 scale: 14.8 for GLUE RTE and 3.6 for GLUE SST2. `python validate.py --compare` passes with all four scored experiments.

## Continuing Audit

The continuing-agent audit restored the prior `secondary_claims` milestone. The method code contains CODA's consensus labels, consensus-confusion Dirichlet prior, probability-of-best integration, expected-information-gain acquisition, and online posterior updates. The evaluator reads method output traces and does not execute or import method code.

The audit found one scoring-completeness gap: `scoring/TARGETS.md` and `scoring/reference.json` now include `glue_mrpc_active_selection`, but `scoring/scores.json` does not yet contain that experiment. `scripts/run.sh` is prepared to run MRPC conversion, random and uncertainty baselines, CODA, score merging, and `python validate.py --compare`; that CPU job is being submitted.

## MRPC Continuation

Job `8ad02e06-c3a` timed out at the scheduler's 3600-second CPU limit. It was useful partial progress rather than a failed method run: MRPC data conversion completed, all five random and uncertainty baseline traces were written, and CODA wrote complete traces for seeds 0, 1, and 2 before the kill.

The runner now supports `--start-seed` and `--skip-existing` so continuation jobs can preserve the paper's five-seed protocol without recomputing completed traces. `scripts/run.sh` now resumes MRPC CODA at seed 3, evaluates random, uncertainty, and CODA together, and runs `python validate.py --compare`.

## What Remains

The next queued expansion is GLUE MRPC active selection at the paper's 100-label budget. Paper targets are 318.2 cumulative regret for random sampling, 301.1 for uncertainty sampling, and 49.0 for CODA. The raw MRPC prediction and label files have been fetched into `data/downloads/modelselector_raw/glue_mrpc/`. A continuation job is prepared to finish CODA seeds 3 and 4, merge the five-seed MRPC result into `scoring/scores.json`, and validate.

Further expansion after MRPC would mean reproducing additional rows of the 26-task main table or adding the paper's ablations on priors and acquisition functions. The packaged gym already includes the method, two active GLUE experiments, two zero-label secondary experiments, baselines, constraints, download script, and reusable evaluator.

## Issues

The login-node PyTorch install path produced a broken GPU wheel, so the runnable reproduction avoids PyTorch by using verified ModelSelector `.npy` files converted to `.npz`. The authors' Google Drive CODA archive was still downloaded and lightly inspected.

## Deviations from Paper

The GLUE jobs use the paper's black-box/class-ID prediction setting through the original ModelSelector raw files instead of the authors' `.pt` release tensors. This is a data-format substitution only: the candidate predictions, labels, and tasks match the paper's ModelSelector-derived GLUE benchmarks, and CODA's algorithmic procedure is unchanged.
