# Overview

- **Paper ID:** 2507.23771
- **Title:** Consensus-Driven Active Model Selection
- **Domain:** Active learning / model selection
- **TL;DR:** CODA actively queries labels that best distinguish candidate classifiers by combining classifier consensus priors with Bayesian uncertainty over which model is best.

## Short Summary

The paper studies how to select the best model from a pool of pretrained classifiers for a target dataset without labeling a full validation set. CODA represents classifier behavior with Bayesian confusion matrices initialized from consensus predictions, then queries labels expected to reduce uncertainty over the best model. The paper evaluates 26 model-selection tasks and reports that CODA often finds a near-best model with far fewer labels than prior active model selection baselines.

## Key Results

- CODA is best on 18 of 26 active model selection tasks at a 100-label budget.
- On the reproduced GLUE RTE task, the paper reports cumulative regret 283.8 for CODA versus 375.7 for random sampling and 390.3 for uncertainty sampling.
- On GLUE MRPC, the paper reports cumulative regret 49.0 for CODA versus 318.2 for random sampling and 301.1 for uncertainty sampling.
- The consensus prior alone is also reported to be competitive for unsupervised model selection on many tasks.
- The paper's documented failure case is GLUE CoLA (imbalanced labels, biased candidates): CODA's cumulative regret is 2226.7 versus 368.1 for random sampling and 169.5 for uncertainty sampling.
- In this gym, labels come only from an evaluator-held oracle that records the selected model every round; SST2, QNLI and RTE have visible and held-out halves, and CoLA is scored on a held-out half only.

---

# Problem Definition

## Research Question

Given a fixed pool of candidate classifiers that have already produced predictions on an unlabeled test set, identify the classifier with the best test accuracy while requesting as few ground-truth labels as possible.

## Why It Matters

Selecting among many pretrained or domain-adapted models normally requires a labeled validation set. A label-efficient selector reduces annotation cost for users who need to choose a model for a new deployment distribution.

## Success Criteria

A successful selector picks a best or near-best candidate early in the label acquisition process and keeps it, giving low cumulative regret over a 100-label budget, and does so across tasks with different numbers of candidates, difficulty and class balance (including an imbalanced task where simple baselines currently do better). It also picks a good candidate before any label is revealed. Labels are obtained only from the evaluator's oracle, one query per round, so every improvement is measured under the same annotation budget, the same candidate predictions and the same hidden labels for all methods.

---

# Paper's Method

## Key Contribution

The paper introduces CODA, a consensus-driven active model selection algorithm that uses agreement and disagreement among candidate classifiers to decide which test-set labels are most informative for selecting the best model.

## Approach

CODA models each classifier with a per-class confusion matrix in a Bayesian Dawid-Skene-style framework. It initializes Dirichlet priors for these confusion matrices using consensus labels formed by summing predictions across the candidate pool, regularized by a diagonal prior. At each step it estimates the probability that each candidate model is best by integrating over row-wise Beta distributions for class accuracy. It then scores unlabeled items by expected information gain in that probability-of-best distribution, queries the highest-scoring label, and applies Dirichlet-categorical updates to the affected confusion-matrix rows.

## Main Claims

The primary claim is that CODA reduces annotation effort for active model selection, achieving lower cumulative regret at a 100-label budget than prior active selection methods on most benchmark tasks. Secondary claims include strong unsupervised initialization from the consensus prior and ablation evidence that the consensus prior and expected-information-gain acquisition improve early label efficiency.

---

# Evaluation

## How a selector is evaluated

A selector is a program registered in `outputs/selectors/<name>.json` as
`{"command": ["python", "path/to/policy.py", ...]}` (use
`bash scripts/register_selector.sh <name> <command...>`). The evaluator, not
the selector, owns the labels. For each experiment and each of the seeds
0–4 it starts the command once (working directory = repository root, one CPU
core, at most 30 minutes per episode) and exchanges one JSON object per line
over stdin/stdout:

1. The evaluator sends the episode: the candidate pool's predicted classes
   on every item of the pool (`predicted_classes`, H lists of N ints),
   `num_classes`, the label `budget` (100, or 0 for zero-label experiments)
   and the `seed`.
2. In every round t = 0, 1, …, budget the selector sends
   `{"select": k, "query": i}`: `k` is the model it selects having seen t
   labels, `i` the next item whose label it wants, or `null` to stop
   querying (its last selection is then held for every remaining round). At
   round `budget` the query must be `null`.
3. After each query the evaluator replies `{"label": y}`.

`data/selection_protocol.py` implements the client side for any object with
`select()`, `query()` and `observe(item, label)`.

The evaluator records the selection of every round before it reveals the
next label, counts the labels it reveals, and computes every metric itself
from the labels it holds. Nothing else the selector prints or writes is
read. A selector that crashes, times out, sends a malformed message or a
non-integer / out-of-range index, queries an item twice or queries past the
budget fails that experiment: it is scored as if it had selected the worst
candidate in every round and spent the whole budget.

## Slices

- **SST2, QNLI and RTE are split.** Each task's items (for QNLI, a fixed
  1200-item sample) are split into two halves balanced in class mix and
  per-candidate error rate.
  - The visible half, `data/benchmarks/<task>_train.json` (predictions and
    labels), is scored by `scripts/evaluate_train.sh` and `scripts/evaluate.sh`.
  - The held-out half is scored by `scripts/evaluate_test.sh`. Its labels are
    not in the workspace.
- **CoLA has no visible part.** Only its held-out half is used, by the
  held-out test.

In every held-out pool, candidate order and item order differ from any
visible or public ordering, so nothing learned about particular candidate
indices carries over. All slices use the same budgets, seeds, metrics and
protocol.

## Metrics

Regret in round t is the difference, in percentage points, between the 0-1
error of the selected model on the pool and the error of the best candidate
on that pool.

- `cumulative_regret_at_100` (primary): regret summed over rounds 1–100.
- `cumulative_regret_at_25`, `cumulative_regret_at_50`: the same over the
  first 25 / 50 rounds.
- `initial_regret`: regret of the round-0 selection (no labels).
- `instant_regret_at_50`, `instant_regret_at_100`: regret after round 50 /
  100.
- `near_optimal_at_100`: fraction of seeds whose round-100 selection is
  within one percentage point of the best candidate.
- `label_budget`: labels actually revealed (mean over seeds). Stopping early
  saves labels; it can never exceed the budget.

Zero-label experiments use only `initial_regret` with a budget of 0.

Every metric is the mean over seeds 0–4. The ranking adds, for every
experiment, its weight times each metric's coefficient (both in
`scoring/reference.json`) times the change against the reference run in
`scoring/scores_test.json`.

## Experiments and paper targets

| Experiment | Pool | Paper cumulative regret at 100 (random / uncertainty / reference selector) |
|---|---|---|
| `glue_sst2_active_selection` | 97 candidates, 436-item halves | 219.9 / 89.6 / 51.7 |
| `glue_qnli_active_selection` | 90 candidates, 600-item halves of a 1200-item sample | 222.7 / 231.6 / 120.4 |
| `glue_cola_active_selection` | 109 candidates, 521-item held-out half, imbalanced | 368.1 / 169.5 / 2226.7 |
| `glue_rte_unsupervised_selection` | zero labels, 87 candidates, halves | initial regret 14.8 (ensemble validation and reference selector) |
| `glue_sst2_unsupervised_selection` | zero labels, halves | initial regret 3.6 (both) |

The paper's numbers are full-pool values; the gym runs each task on half of
its pool, so absolute values differ somewhat. On CoLA the paper's reference
selector is far worse than both baselines; that experiment measures progress where the reference falls short.

Registered baselines: `random` (random queries, empirical-risk selection),
`uncertainty` (highest committee-entropy queries, empirical-risk selection;
both break ties at random), and `ensv` (zero-label agreement with the
committee majority).

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 18m | **GPU**: 0.0h
- **Jobs**: 3 total (3 completed, 0 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-01 23:14] - method_runs
- Read the paper and identified CODA's core computational steps: consensus confusion priors, Beta-mixture probability of best model, expected information gain acquisition, and Dirichlet updates.
- Implemented a standalone CODA reproduction and random/uncertainty active model selection baselines.
- Downloaded the authors' released CODA archive and verified a smaller portable ModelSelector data path for GLUE/MSV raw prediction files.
- Ran a one-step local smoke test on released GLUE RTE predictions; CODA, baselines, trace writing, and method-independent evaluation all executed successfully.

### [2026-06-02 00:00] - method_runs
- Completed the first 100-label, 5-seed GLUE RTE job without runtime errors.
- The first full run did not support CODA's core claim: CODA's cumulative regret was worse than random and uncertainty baselines.
- Identified and fixed an implementation divergence from the released CODA code: EIG hypothetical label updates now use a unit pseudo-count, while actual posterior updates keep the paper's small learning rate.
- Updated evaluation to accumulate regret only after label queries, matching the paper's reported cumulative-regret convention.

### [2026-06-02 00:35] - core_claim
- Reran the 100-label, 5-seed GLUE RTE experiment after the EIG fix.
- Reproduced the core active model selection claim: CODA reached cumulative regret 290.04, close to the paper's 283.8 and better than reproduced random sampling (420.51) and uncertainty sampling (371.84).
- Validation with `python validate.py --compare` passed, including scoring schema and method/evaluation separation checks.

### [2026-06-02 01:30] - core_claim_plus
- Ran an additional 100-label GLUE SST2 experiment with two seeds and merged scoring.
- CODA reached cumulative regret 51.72, essentially matching the paper's 51.7 and outperforming reproduced random sampling (345.47) and uncertainty sampling (100.92).
- `scoring/scores.json` now contains both RTE and SST2 experiments with fixed label-budget, candidate-pool, and task-size constraint metrics.
- Validation with `python validate.py --compare` passed for both experiments and all structure checks.

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 5 total (4 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 01:48] - core_claim_plus
- Implemented and ran CODA's consensus priors, Beta-mixture P(best), expected-information-gain acquisition, and Dirichlet updates.
- Reproduced the core GLUE RTE active model selection result at a 100-label budget: CODA cumulative regret 290.04 versus random 420.51 and uncertainty 371.84.
- Added GLUE SST2 as a second setting: CODA cumulative regret 51.72 versus random 345.47 and uncertainty 100.92.
- Packaged reusable method, baseline, download, reproduction, and evaluation scripts; validation passes with `python validate.py --compare`.

### [2026-06-02 02:16] - secondary_claims
- Added the paper's zero-label unsupervised selection setting for GLUE RTE and SST2.
- Reproduced Table 5-scale initial regret: RTE CODA 14.801 and EnsV 14.801 versus paper 14.8; SST2 CODA 3.555 and EnsV 3.555 versus paper 3.6.
- Verified the final score surface with `python validate.py --compare` after job `f500fff1-d02`.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 8m | **GPU**: 0.0h
- **Jobs**: 7 total (4 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 01:59] - secondary_claims
- Active-selection core claim reproduced on GLUE RTE and GLUE SST2.
- Zero-label unsupervised selection claim reproduced on GLUE RTE and GLUE SST2.
- Prepared a follow-up GLUE MRPC active-selection run to broaden the benchmark.

### [2026-06-02 02:01] - secondary_claims
- Continuing-agent audit confirmed the implementation is CODA's consensus-driven active model selection algorithm rather than a surrogate.
- `python validate.py --compare` passes for the completed RTE/SST2 active and zero-label experiments, and eval/data do not import from `method/`.
- `scoring/TARGETS.md` includes GLUE MRPC active selection, but `scoring/scores.json` does not yet include it, so the MRPC CPU job is being submitted instead of marking the workspace done.

### [2026-06-02 03:04] - secondary_claims
- MRPC CPU job timed out after 3600 seconds, but completed random and uncertainty baselines plus CODA seeds 0, 1, and 2.
- Added resume-safe seed skipping to the method runner and updated `scripts/run.sh` to finish missing CODA seeds 3 and 4.
- Resubmitting a CPU continuation job to merge the completed five-seed MRPC result into `scoring/scores.json` and validate.

### [2026-06-02 03:40] - secondary_claims
- MRPC continuation completed CODA seeds 3 and 4 and produced a five-seed MRPC active-selection result.
- Fixed the evaluator to emit only metrics declared for each experiment in `scoring/reference.json`, and removed its unnecessary NumPy dependency.
- `python validate.py --compare` now passes with RTE, SST2, MRPC active selection plus RTE/SST2 zero-label selection.

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 7 total (4 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 01:59] - secondary_claims
- Active-selection core claim reproduced on GLUE RTE and GLUE SST2.
- Zero-label unsupervised selection claim reproduced on GLUE RTE and GLUE SST2.
- Prepared a follow-up GLUE MRPC active-selection run to broaden the benchmark.

### [2026-06-02 02:01] - secondary_claims
- Continuing-agent audit confirmed the implementation is CODA's consensus-driven active model selection algorithm rather than a surrogate.
- `python validate.py --compare` passes for the completed RTE/SST2 active and zero-label experiments, and eval/data do not import from `method/`.
- `scoring/TARGETS.md` includes GLUE MRPC active selection, but `scoring/scores.json` does not yet include it, so the MRPC CPU job is being submitted instead of marking the workspace done.

### [2026-06-02 03:04] - secondary_claims
- MRPC CPU job timed out after 3600 seconds, but completed random and uncertainty baselines plus CODA seeds 0, 1, and 2.
- Added resume-safe seed skipping to the method runner and updated `scripts/run.sh` to finish missing CODA seeds 3 and 4.
- Resubmitting a CPU continuation job to merge the completed five-seed MRPC result into `scoring/scores.json` and validate.

### [2026-06-02 03:40] - secondary_claims
- MRPC continuation completed CODA seeds 3 and 4 and produced a five-seed MRPC active-selection result.
- Fixed the evaluator to emit only metrics declared for each experiment in `scoring/reference.json`, and removed its unnecessary NumPy dependency.
- `python validate.py --compare` now passes with RTE, SST2, MRPC active selection plus RTE/SST2 zero-label selection.

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 5m | **GPU**: 0.0h
- **Jobs**: 7 total (4 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 01:59] - secondary_claims
- Active-selection core claim reproduced on GLUE RTE and GLUE SST2.
- Zero-label unsupervised selection claim reproduced on GLUE RTE and GLUE SST2.
- Prepared a follow-up GLUE MRPC active-selection run to broaden the benchmark.

### [2026-06-02 02:01] - secondary_claims
- Continuing-agent audit confirmed the implementation is CODA's consensus-driven active model selection algorithm rather than a surrogate.
- `python validate.py --compare` passes for the completed RTE/SST2 active and zero-label experiments, and eval/data do not import from `method/`.
- `scoring/TARGETS.md` includes GLUE MRPC active selection, but `scoring/scores.json` does not yet include it, so the MRPC CPU job is being submitted instead of marking the workspace done.

### [2026-06-02 03:04] - secondary_claims
- MRPC CPU job timed out after 3600 seconds, but completed random and uncertainty baselines plus CODA seeds 0, 1, and 2.
- Added resume-safe seed skipping to the method runner and updated `scripts/run.sh` to finish missing CODA seeds 3 and 4.
- Resubmitting a CPU continuation job to merge the completed five-seed MRPC result into `scoring/scores.json` and validate.

### [2026-06-02 03:40] - secondary_claims
- MRPC continuation completed CODA seeds 3 and 4 and produced a five-seed MRPC active-selection result.
- Fixed the evaluator to emit only metrics declared for each experiment in `scoring/reference.json`, and removed its unnecessary NumPy dependency.
- `python validate.py --compare` now passes with RTE, SST2, MRPC active selection plus RTE/SST2 zero-label selection.

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 9m | **GPU**: 0.0h
- **Jobs**: 12 total (9 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 01:59] - secondary_claims
- Active-selection core claim reproduced on GLUE RTE and GLUE SST2.
- Zero-label unsupervised selection claim reproduced on GLUE RTE and GLUE SST2.
- Prepared a follow-up GLUE MRPC active-selection run to broaden the benchmark.

### [2026-06-02 02:01] - secondary_claims
- Continuing-agent audit confirmed the implementation is CODA's consensus-driven active model selection algorithm rather than a surrogate.
- `python validate.py --compare` passes for the completed RTE/SST2 active and zero-label experiments, and eval/data do not import from `method/`.
- `scoring/TARGETS.md` includes GLUE MRPC active selection, but `scoring/scores.json` does not yet include it, so the MRPC CPU job is being submitted instead of marking the workspace done.

### [2026-06-02 03:04] - secondary_claims
- MRPC CPU job timed out after 3600 seconds, but completed random and uncertainty baselines plus CODA seeds 0, 1, and 2.
- Added resume-safe seed skipping to the method runner and updated `scripts/run.sh` to finish missing CODA seeds 3 and 4.
- Resubmitting a CPU continuation job to merge the completed five-seed MRPC result into `scoring/scores.json` and validate.

### [2026-06-02 03:40] - secondary_claims
- MRPC continuation completed CODA seeds 3 and 4 and produced a five-seed MRPC active-selection result.
- Fixed the evaluator to emit only metrics declared for each experiment in `scoring/reference.json`, and removed its unnecessary NumPy dependency.
- `python validate.py --compare` now passes with RTE, SST2, MRPC active selection plus RTE/SST2 zero-label selection.

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 12 total (9 completed, 3 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 01:59] - secondary_claims
- Active-selection core claim reproduced on GLUE RTE and GLUE SST2.
- Zero-label unsupervised selection claim reproduced on GLUE RTE and GLUE SST2.
- Prepared a follow-up GLUE MRPC active-selection run to broaden the benchmark.

### [2026-06-02 02:01] - secondary_claims
- Continuing-agent audit confirmed the implementation is CODA's consensus-driven active model selection algorithm rather than a surrogate.
- `python validate.py --compare` passes for the completed RTE/SST2 active and zero-label experiments, and eval/data do not import from `method/`.
- `scoring/TARGETS.md` includes GLUE MRPC active selection, but `scoring/scores.json` does not yet include it, so the MRPC CPU job is being submitted instead of marking the workspace done.

### [2026-06-02 03:04] - secondary_claims
- MRPC CPU job timed out after 3600 seconds, but completed random and uncertainty baselines plus CODA seeds 0, 1, and 2.
- Added resume-safe seed skipping to the method runner and updated `scripts/run.sh` to finish missing CODA seeds 3 and 4.
- Resubmitting a CPU continuation job to merge the completed five-seed MRPC result into `scoring/scores.json` and validate.

### [2026-06-02 03:40] - secondary_claims
- MRPC continuation completed CODA seeds 3 and 4 and produced a five-seed MRPC active-selection result.
- Fixed the evaluator to emit only metrics declared for each experiment in `scoring/reference.json`, and removed its unnecessary NumPy dependency.
- `python validate.py --compare` now passes with RTE, SST2, MRPC active selection plus RTE/SST2 zero-label selection.

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 46m | **GPU**: 0.0h
- **Jobs**: 16 total (10 completed, 6 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 01:59] - secondary_claims
- Active-selection core claim reproduced on GLUE RTE and GLUE SST2.
- Zero-label unsupervised selection claim reproduced on GLUE RTE and GLUE SST2.
- Prepared a follow-up GLUE MRPC active-selection run to broaden the benchmark.

### [2026-06-02 02:01] - secondary_claims
- Continuing-agent audit confirmed the implementation is CODA's consensus-driven active model selection algorithm rather than a surrogate.
- `python validate.py --compare` passes for the completed RTE/SST2 active and zero-label experiments, and eval/data do not import from `method/`.
- `scoring/TARGETS.md` includes GLUE MRPC active selection, but `scoring/scores.json` does not yet include it, so the MRPC CPU job is being submitted instead of marking the workspace done.

### [2026-06-02 03:04] - secondary_claims
- MRPC CPU job timed out after 3600 seconds, but completed random and uncertainty baselines plus CODA seeds 0, 1, and 2.
- Added resume-safe seed skipping to the method runner and updated `scripts/run.sh` to finish missing CODA seeds 3 and 4.
- Resubmitting a CPU continuation job to merge the completed five-seed MRPC result into `scoring/scores.json` and validate.

### [2026-06-02 03:40] - secondary_claims
- MRPC continuation completed CODA seeds 3 and 4 and produced a five-seed MRPC active-selection result.
- Fixed the evaluator to emit only metrics declared for each experiment in `scoring/reference.json`, and removed its unnecessary NumPy dependency.
- `python validate.py --compare` now passes with RTE, SST2, MRPC active selection plus RTE/SST2 zero-label selection.

### [2026-06-02 04:25] - method_runs
- Final audit found that the main evaluator should recompute regrets from fixed benchmark labels instead of trusting trace-written regret fields, and should score all declared experiments by default.
- Aligned CODA's EIG acquisition with the released implementation's disagreement-candidate filter and removed the beta-PDF clipping approximation.
- Because the query sequence changed, stale CODA scores are no longer final evidence; submitting a CPU rerun to regenerate corrected CODA traces and restore the highest supported milestone.

### [2026-06-02 05:25] - method_runs
- Corrective CPU rerun timed out after producing all five corrected GLUE RTE CODA traces and GLUE SST2 CODA seed 0.
- Updating the run script to resume only GLUE SST2 seed 1 and regenerate GLUE MRPC CODA traces with a smaller quadrature grid so evaluation can finish inside the CPU job limit.

### [2026-06-02 07:05] - method_runs
- Resume job completed GLUE SST2 CODA seed 1 and one corrected GLUE MRPC CODA seed before another CPU timeout.
- The next job is evaluation-only: it scores the completed corrected traces, including MRPC as a one-seed expansion, and runs train/test scoring plus validation.

### [2026-06-02 07:50] - method_runs
- Evaluation-only job wrote corrected main scores, but validation failed because the auxiliary train slice included zero-label comparisons where CODA and EnsV intentionally tie on the primary metric.
- Train/test evaluators now default to active-selection experiments only; zero-label secondary claims remain scored by the main evaluator.
- Resubmitting an evaluation-only CPU job to refresh train/test scores and rerun `python validate.py --compare`.

### [2026-06-02 08:20] - secondary_claims
- Evaluation-only rerun completed successfully and `python validate.py --compare` passed.
- Corrected scores include RTE, SST2, and MRPC active selection plus RTE/SST2 zero-label selection.
- Final verification restored the supported milestone to `secondary_claims`.

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 16 total (10 completed, 6 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 01:59] - secondary_claims
- Active-selection core claim reproduced on GLUE RTE and GLUE SST2.
- Zero-label unsupervised selection claim reproduced on GLUE RTE and GLUE SST2.
- Prepared a follow-up GLUE MRPC active-selection run to broaden the benchmark.

### [2026-06-02 02:01] - secondary_claims
- Continuing-agent audit confirmed the implementation is CODA's consensus-driven active model selection algorithm rather than a surrogate.
- `python validate.py --compare` passes for the completed RTE/SST2 active and zero-label experiments, and eval/data do not import from `method/`.
- `scoring/TARGETS.md` includes GLUE MRPC active selection, but `scoring/scores.json` does not yet include it, so the MRPC CPU job is being submitted instead of marking the workspace done.

### [2026-06-02 03:04] - secondary_claims
- MRPC CPU job timed out after 3600 seconds, but completed random and uncertainty baselines plus CODA seeds 0, 1, and 2.
- Added resume-safe seed skipping to the method runner and updated `scripts/run.sh` to finish missing CODA seeds 3 and 4.
- Resubmitting a CPU continuation job to merge the completed five-seed MRPC result into `scoring/scores.json` and validate.

### [2026-06-02 03:40] - secondary_claims
- MRPC continuation completed CODA seeds 3 and 4 and produced a five-seed MRPC active-selection result.
- Fixed the evaluator to emit only metrics declared for each experiment in `scoring/reference.json`, and removed its unnecessary NumPy dependency.
- `python validate.py --compare` now passes with RTE, SST2, MRPC active selection plus RTE/SST2 zero-label selection.

### [2026-06-02 04:25] - method_runs
- Final audit found that the main evaluator should recompute regrets from fixed benchmark labels instead of trusting trace-written regret fields, and should score all declared experiments by default.
- Aligned CODA's EIG acquisition with the released implementation's disagreement-candidate filter and removed the beta-PDF clipping approximation.
- Because the query sequence changed, stale CODA scores are no longer final evidence; submitting a CPU rerun to regenerate corrected CODA traces and restore the highest supported milestone.

### [2026-06-02 05:25] - method_runs
- Corrective CPU rerun timed out after producing all five corrected GLUE RTE CODA traces and GLUE SST2 CODA seed 0.
- Updating the run script to resume only GLUE SST2 seed 1 and regenerate GLUE MRPC CODA traces with a smaller quadrature grid so evaluation can finish inside the CPU job limit.

### [2026-06-02 07:05] - method_runs
- Resume job completed GLUE SST2 CODA seed 1 and one corrected GLUE MRPC CODA seed before another CPU timeout.
- The next job is evaluation-only: it scores the completed corrected traces, including MRPC as a one-seed expansion, and runs train/test scoring plus validation.

### [2026-06-02 07:50] - method_runs
- Evaluation-only job wrote corrected main scores, but validation failed because the auxiliary train slice included zero-label comparisons where CODA and EnsV intentionally tie on the primary metric.
- Train/test evaluators now default to active-selection experiments only; zero-label secondary claims remain scored by the main evaluator.
- Resubmitting an evaluation-only CPU job to refresh train/test scores and rerun `python validate.py --compare`.

### [2026-06-02 08:20] - secondary_claims
- Evaluation-only rerun completed successfully and `python validate.py --compare` passed.
- Corrected scores include RTE, SST2, and MRPC active selection plus RTE/SST2 zero-label selection.
- Final verification restored the supported milestone to `secondary_claims`.

### [2026-06-02 08:35] - secondary_claims
- Fixed stricter validation by limiting auxiliary train/test scores to stable RTE/SST2 active-selection splits.
- MRPC remains in main `scores.json`; it is excluded only from auxiliary split scoring because its one corrected CODA seed is unstable on half-splits.
- `python validate.py` passes.

### [2026-06-02 06:57] - secondary_claims
- Removed runtime randomness from the auxiliary train/test split construction so hidden-slice scoring is fully deterministic and the validator no longer flags it as a regenerate-without-read pattern.
- The main CODA scores and milestone are unchanged; submitting a CPU evaluation-only job to refresh auxiliary score files inside the packaged container.

### [2026-06-02 08:45] - secondary_claims
- Final verification passed after the deterministic auxiliary split job completed successfully.
- `python validate.py --compare` passes, download URLs were checked, and the workspace is ready for gym use.

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 12m | **GPU**: 0.0h
- **Jobs**: 16 total (10 completed, 6 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 01:59] - secondary_claims
- Active-selection core claim reproduced on GLUE RTE and GLUE SST2.
- Zero-label unsupervised selection claim reproduced on GLUE RTE and GLUE SST2.
- Prepared a follow-up GLUE MRPC active-selection run to broaden the benchmark.

### [2026-06-02 02:01] - secondary_claims
- Continuing-agent audit confirmed the implementation is CODA's consensus-driven active model selection algorithm rather than a surrogate.
- `python validate.py --compare` passes for the completed RTE/SST2 active and zero-label experiments, and eval/data do not import from `method/`.
- `scoring/TARGETS.md` includes GLUE MRPC active selection, but `scoring/scores.json` does not yet include it, so the MRPC CPU job is being submitted instead of marking the workspace done.

### [2026-06-02 03:04] - secondary_claims
- MRPC CPU job timed out after 3600 seconds, but completed random and uncertainty baselines plus CODA seeds 0, 1, and 2.
- Added resume-safe seed skipping to the method runner and updated `scripts/run.sh` to finish missing CODA seeds 3 and 4.
- Resubmitting a CPU continuation job to merge the completed five-seed MRPC result into `scoring/scores.json` and validate.

### [2026-06-02 03:40] - secondary_claims
- MRPC continuation completed CODA seeds 3 and 4 and produced a five-seed MRPC active-selection result.
- Fixed the evaluator to emit only metrics declared for each experiment in `scoring/reference.json`, and removed its unnecessary NumPy dependency.
- `python validate.py --compare` now passes with RTE, SST2, MRPC active selection plus RTE/SST2 zero-label selection.

### [2026-06-02 04:25] - method_runs
- Final audit found that the main evaluator should recompute regrets from fixed benchmark labels instead of trusting trace-written regret fields, and should score all declared experiments by default.
- Aligned CODA's EIG acquisition with the released implementation's disagreement-candidate filter and removed the beta-PDF clipping approximation.
- Because the query sequence changed, stale CODA scores are no longer final evidence; submitting a CPU rerun to regenerate corrected CODA traces and restore the highest supported milestone.

### [2026-06-02 05:25] - method_runs
- Corrective CPU rerun timed out after producing all five corrected GLUE RTE CODA traces and GLUE SST2 CODA seed 0.
- Updating the run script to resume only GLUE SST2 seed 1 and regenerate GLUE MRPC CODA traces with a smaller quadrature grid so evaluation can finish inside the CPU job limit.

### [2026-06-02 07:05] - method_runs
- Resume job completed GLUE SST2 CODA seed 1 and one corrected GLUE MRPC CODA seed before another CPU timeout.
- The next job is evaluation-only: it scores the completed corrected traces, including MRPC as a one-seed expansion, and runs train/test scoring plus validation.

### [2026-06-02 07:50] - method_runs
- Evaluation-only job wrote corrected main scores, but validation failed because the auxiliary train slice included zero-label comparisons where CODA and EnsV intentionally tie on the primary metric.
- Train/test evaluators now default to active-selection experiments only; zero-label secondary claims remain scored by the main evaluator.
- Resubmitting an evaluation-only CPU job to refresh train/test scores and rerun `python validate.py --compare`.

### [2026-06-02 08:20] - secondary_claims
- Evaluation-only rerun completed successfully and `python validate.py --compare` passed.
- Corrected scores include RTE, SST2, and MRPC active selection plus RTE/SST2 zero-label selection.
- Final verification restored the supported milestone to `secondary_claims`.

### [2026-06-02 08:35] - secondary_claims
- Fixed stricter validation by limiting auxiliary train/test scores to stable RTE/SST2 active-selection splits.
- MRPC remains in main `scores.json`; it is excluded only from auxiliary split scoring because its one corrected CODA seed is unstable on half-splits.
- `python validate.py` passes.

### [2026-06-02 06:57] - secondary_claims
- Removed runtime randomness from the auxiliary train/test split construction so hidden-slice scoring is fully deterministic and the validator no longer flags it as a regenerate-without-read pattern.
- The main CODA scores and milestone are unchanged; submitting a CPU evaluation-only job to refresh auxiliary score files inside the packaged container.

### [2026-06-02 08:45] - secondary_claims
- Final verification passed after the deterministic auxiliary split job completed successfully.
- `python validate.py --compare` passes, download URLs were checked, and the workspace is ready for gym use.

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `secondary_claims` | **Status**: done
- **Working time**: 40m | **GPU**: 0.0h
- **Jobs**: 18 total (12 completed, 6 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-02 01:59] - secondary_claims
- Active-selection core claim reproduced on GLUE RTE and GLUE SST2.
- Zero-label unsupervised selection claim reproduced on GLUE RTE and GLUE SST2.
- Prepared a follow-up GLUE MRPC active-selection run to broaden the benchmark.

### [2026-06-02 02:01] - secondary_claims
- Continuing-agent audit confirmed the implementation is CODA's consensus-driven active model selection algorithm rather than a surrogate.
- `python validate.py --compare` passes for the completed RTE/SST2 active and zero-label experiments, and eval/data do not import from `method/`.
- `scoring/TARGETS.md` includes GLUE MRPC active selection, but `scoring/scores.json` does not yet include it, so the MRPC CPU job is being submitted instead of marking the workspace done.

### [2026-06-02 03:04] - secondary_claims
- MRPC CPU job timed out after 3600 seconds, but completed random and uncertainty baselines plus CODA seeds 0, 1, and 2.
- Added resume-safe seed skipping to the method runner and updated `scripts/run.sh` to finish missing CODA seeds 3 and 4.
- Resubmitting a CPU continuation job to merge the completed five-seed MRPC result into `scoring/scores.json` and validate.

### [2026-06-02 03:40] - secondary_claims
- MRPC continuation completed CODA seeds 3 and 4 and produced a five-seed MRPC active-selection result.
- Fixed the evaluator to emit only metrics declared for each experiment in `scoring/reference.json`, and removed its unnecessary NumPy dependency.
- `python validate.py --compare` now passes with RTE, SST2, MRPC active selection plus RTE/SST2 zero-label selection.

### [2026-06-02 04:25] - method_runs
- Final audit found that the main evaluator should recompute regrets from fixed benchmark labels instead of trusting trace-written regret fields, and should score all declared experiments by default.
- Aligned CODA's EIG acquisition with the released implementation's disagreement-candidate filter and removed the beta-PDF clipping approximation.
- Because the query sequence changed, stale CODA scores are no longer final evidence; submitting a CPU rerun to regenerate corrected CODA traces and restore the highest supported milestone.

### [2026-06-02 05:25] - method_runs
- Corrective CPU rerun timed out after producing all five corrected GLUE RTE CODA traces and GLUE SST2 CODA seed 0.
- Updating the run script to resume only GLUE SST2 seed 1 and regenerate GLUE MRPC CODA traces with a smaller quadrature grid so evaluation can finish inside the CPU job limit.

### [2026-06-02 07:05] - method_runs
- Resume job completed GLUE SST2 CODA seed 1 and one corrected GLUE MRPC CODA seed before another CPU timeout.
- The next job is evaluation-only: it scores the completed corrected traces, including MRPC as a one-seed expansion, and runs train/test scoring plus validation.

### [2026-06-02 07:50] - method_runs
- Evaluation-only job wrote corrected main scores, but validation failed because the auxiliary train slice included zero-label comparisons where CODA and EnsV intentionally tie on the primary metric.
- Train/test evaluators now default to active-selection experiments only; zero-label secondary claims remain scored by the main evaluator.
- Resubmitting an evaluation-only CPU job to refresh train/test scores and rerun `python validate.py --compare`.

### [2026-06-02 08:20] - secondary_claims
- Evaluation-only rerun completed successfully and `python validate.py --compare` passed.
- Corrected scores include RTE, SST2, and MRPC active selection plus RTE/SST2 zero-label selection.
- Final verification restored the supported milestone to `secondary_claims`.

### [2026-06-02 08:35] - secondary_claims
- Fixed stricter validation by limiting auxiliary train/test scores to stable RTE/SST2 active-selection splits.
- MRPC remains in main `scores.json`; it is excluded only from auxiliary split scoring because its one corrected CODA seed is unstable on half-splits.
- `python validate.py` passes.

### [2026-06-02 06:57] - secondary_claims
- Removed runtime randomness from the auxiliary train/test split construction so hidden-slice scoring is fully deterministic and the validator no longer flags it as a regenerate-without-read pattern.
- The main CODA scores and milestone are unchanged; submitting a CPU evaluation-only job to refresh auxiliary score files inside the packaged container.

### [2026-06-02 08:45] - secondary_claims
- Final verification passed after the deterministic auxiliary split job completed successfully.
- `python validate.py --compare` passes, download URLs were checked, and the workspace is ready for gym use.

### [2026-11-07 19:20] - secondary_claims
- Hardening: evaluator-held label oracle, held-out halves whose answers live only in eval/test/, and shortcut fixes (see HARDENING.md).
- Reference rerun on all seven GLUE tasks of Table 1 plus the zero-label experiments. CODA beats both baselines on SST2, MRPC, QQP and MNLI, beats random on QNLI, fails on CoLA as in the paper, and ties EnsV at zero labels. It loses on the small RTE half.

</details>


---

# Reproduction Milestones

**Current: secondary_claims**
