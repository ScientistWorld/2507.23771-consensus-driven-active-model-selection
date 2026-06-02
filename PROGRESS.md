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

The continuation job `6d949629-9d9` completed the remaining CODA seeds. Its first validation pass failed only because active-selection scores included `initial_regret`, which is not a declared metric in `scoring/reference.json`. The evaluator now filters emitted metrics to the experiment's reference metric set and uses only the Python standard library.

The final MRPC active-selection result matches the paper's proposed-method scale:

- Random sampling cumulative regret: 349.02
- Uncertainty cumulative regret: 256.13
- CODA cumulative regret: 49.02

The paper reports MRPC cumulative regret values of 318.2 for random sampling, 301.1 for uncertainty sampling, and 49.0 for CODA. `python validate.py --compare` now passes with all five declared experiments in `scoring/scores.json`.

## What Remains

Further expansion would mean reproducing additional rows of the 26-task main table or adding the paper's ablations on priors and acquisition functions. The packaged gym already includes the method, three active GLUE experiments, two zero-label secondary experiments, baselines, constraints, download script, and reusable evaluator.

## Corrective Rerun

A final audit found two issues to fix before treating the existing scores as final gym evidence. First, the main evaluator trusted regret and benchmark-size fields written by method traces and only scored one experiment by default. It now recomputes regret, model count, and item count from the fixed benchmark arrays in `data/downloads/modelselector/`, and `scripts/evaluate.sh` evaluates every experiment in `scoring/reference.json` unless a specific task or experiment is requested.

Second, the CODA implementation scored EIG over every unlabeled item, while the released implementation first filters candidates to points where at least one model disagrees with the majority prediction. The method now uses that disagreement candidate filter and removes the previous beta-PDF clipping approximation. Because this changes the query sequence, stale CODA traces were removed from scoring and a CPU rerun is being submitted to regenerate CODA active traces for RTE, SST2, and MRPC, rerun zero-label checks, and validate the refreshed scores.

The first corrective CPU rerun timed out at the scheduler's 3600-second CPU limit. It still produced useful corrected artifacts: all five GLUE RTE CODA traces and GLUE SST2 CODA seed 0 completed. The next resume job skips completed RTE work, finishes GLUE SST2 seed 1, regenerates GLUE MRPC CODA traces with a 64-point quadrature grid, and then reruns the full evaluator. Until that job completes and validation passes, the supported milestone remains `method_runs`.

## Issues

The login-node PyTorch install path produced a broken GPU wheel, so the runnable reproduction avoids PyTorch by using verified ModelSelector `.npy` files converted to `.npz`. The authors' Google Drive CODA archive was still downloaded and lightly inspected.

## Deviations from Paper

The GLUE jobs use the paper's black-box/class-ID prediction setting through the original ModelSelector raw files instead of the authors' `.pt` release tensors. This is a data-format substitution only: the candidate predictions, labels, and tasks match the paper's ModelSelector-derived GLUE benchmarks, and CODA's algorithmic procedure is unchanged.
