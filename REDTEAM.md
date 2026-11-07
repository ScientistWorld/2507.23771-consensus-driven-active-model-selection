# Gym scoring audit

Genuine improvements can score higher here. A real update-size and stability-stopping variant gained **151.360964** points, while removing posterior learning lost **570.276029** points. Both changes exceed the three-seed reference spread of **17.073913** points. However, the score also rewards **seven demonstrated shortcuts**, including a failing constant selector that gained **885.464636** points through duplicated item counts. The scientific ordering check passes for the tested honest variants; the overall rank does not reliably establish a scientific contribution.

All numbers below are weighted rank improvements over the original `scoring/scores_test.json` CODA result, whose rank improvement is zero. Higher is better. I used every scored coefficient, without normalizing the experiment weights:

`delta = sum_e weight[e] * sum_m coefficient[e,m] * (candidate[e,m] - stored_reference[e,m])`.

The no-argument hidden test scores RTE and SST2 active selection, with weights 0.34 and 0.22. MRPC and both zero-label experiments in `reference.json` are absent from its output: **44% of the configured experiment weight is untested by this invocation**. No additional experiment or argument was used to improve any reported rank. Machine-readable results and a recalculation script are in [redteam/summary.json](redteam/summary.json) and [redteam/analyze.py](redteam/analyze.py).

The quality checks used the fixed prediction tensors and labels, with 87 RTE models and 97 SST2 models. Each honest variant and the random baseline ran seeds 0, 1, and 2; their rows show the evaluator's seed averages. Cumulative regret is in percentage-point rounds; lower is better.

| Method/check | RTE cumulative regret at 100 | SST2 cumulative regret at 100 | Actual labels, RTE/SST2 | Rank improvement |
|---|---:|---:|---:|---:|
| Stored test reference | 271.304348 | 48.623853 | 100/100 | 0.000000 |
| Original saved reference traces, rescored | 271.304348 | 51.605505 | 100/100 | -1.377523 |
| Fresh reference, mean of three separate seed runs | 275.362319 | 51.605505 | 100/100 | -3.654045 |
| [Honest stronger variant: eta 0.1 with stability stopping](redteam/honest_stable_stop/run.sh) | **86.231884** | **20.412844** | **25/34** | **+151.360964** |
| [Weaker variant: no posterior learning](redteam/weaker_no_learning/run.sh) | 1376.811594 | 344.036697 | 100/100 | **-570.276029** |
| [Random-label empirical-risk baseline](redteam/baseline_random/run.sh) | 587.439614 | 245.871560 | 100/100 | -223.898970 |
| [Honest eta 0.1, all 100 updates](redteam/honest_eta_0p1/run.sh) | 2074.637681 | 20.412844 | 100/100 | -536.959137 |
| [Honest eta 1, all 100 updates](redteam/honest_eta_1/run.sh) | 3674.637681 | 4101.834862 | 100/100 | -2740.521538 |

The stronger variant retains CODA's consensus prior, Beta quadrature, information-gain acquisition, and label updates. It increases eta from 0.01 to 0.1 and stops after the selected model stays unchanged for 12 consecutive rounds, subject to 25–100 actual queries. The visible train results exposed late deterioration under larger updates. Stopping uses only selection history and actually acquired labels. It records every query, its true budget, and its actual trajectory; there is no backdating. Both tasks already reach their final zero scored regret by round 25. Their cumulative regrets at 25, 50, and 100 are therefore respectively 86.231884 and 20.412844. Of the 151.360964-point improvement, 143.356964 comes from better regret and 8.004000 from genuinely reduced label use. The acquisition implementation was vectorized for these variants; [numerical checks](redteam/numerics.json) against the original acquisition at three posterior states per task agreed within 2.7e-15. Reference seeds reran the original implementation.

The weaker ablation consumes 100 distinct queried labels but never updates its initial model choice. Its much worse regret and rank show that removing the central learning behavior is penalized. The two larger-update runs also score worse when their actual trajectories deteriorate. These checks establish useful sensitivity, rather than claiming that any hyperparameter change is an improvement.

The three fresh reference runs produced the following hidden scores:

| Seed | RTE cumulative regret at 100 | SST2 cumulative regret at 100 | Rank improvement |
|---|---:|---:|---:|
| [0](redteam/reference_seed0/result.json) | 265.217391 | 51.605505 | +2.037260 |
| [1](redteam/reference_seed1/result.json) | 265.217391 | 51.605505 | +2.037260 |
| [2](redteam/reference_seed2/result.json) | 295.652174 | 51.605505 | -15.036653 |

The observed rank range is **17.073913**, with sample standard deviation 9.857628. RTE cumulative-regret range is 30.434783; SST2's is zero. A standalone gain below 17.073913 is noise at this audit's resolution. The budget-field shortcut below is smaller than this spread, but its paired comparison changes no predictions or selections and gives an exact 11.088000-point metadata reward.

The reproduction beats its baselines in the direction reported by the supplied briefing. Rescoring the original saved outputs through the hidden test gives:

| Task | CODA cumulative regret at 100 | Random | Uncertainty |
|---|---:|---:|---:|
| RTE | **271.304348** | 410.434783 | 357.246377 |
| SST2 | **51.605505** | 321.215596 | 83.027523 |

These are direct hidden-test measurements in [reference_original/scores_test.json](redteam/reference_original/scores_test.json), not a fresh baseline training claim. The newly run three-seed random baseline also sits clearly below the reference in aggregate. There is substantial room above the reference: genuine improvements and zero-regret selections both beat it. There is also a reproducibility defect: the stored SST2 reference has cumulative regret 48.623853, while rescoring its unchanged saved traces gives 51.605505, producing -1.377523 rank points before any method change. The original target and all rescored outputs are preserved under `redteam/`.

For every default-scored experiment, these reference metrics already sit at their favorable bound:

| Metric | RTE reference | SST2 reference | Bound/meaning |
|---|---:|---:|---|
| `instant_regret_at_50` | 0 | 0 | Minimum, because regret is clipped at zero |
| `instant_regret_at_100` | 0 | 0 | Minimum |
| `near_optimal_at_100` | 1 | 1 | Maximum |
| `num_models` | 87 | 97 | Entire fixed pool; no legitimate increase is permitted |
| `num_items` | 138 | 436 | Entire hidden slice; no legitimate increase is permitted |

Initial and cumulative regrets remain above their lower bound. The reference's 100-label budget is the allowed maximum, rather than the favorable bound of its negatively scored resource metric. The fixed-count metrics should be equality checks; their positive linear coefficients instead allow invalid increases to outweigh task failure.

Every shortcut below has its own executable `run.sh`, candidate implementation, submitted traces, and `result.json`. Successful test invocations also retain `scores_test.json` and `scores_train.json` in that folder. Rejections retain the hidden evaluator's error in `result.json`; no score was fabricated for them.

| Shortcut and run | What the shortcut does | Hidden rank improvement vs reference | Works/fails |
|---|---|---:|---|
| [Visible-label oracle shortcut](redteam/visible_label_oracle_shortcut/run.sh) | Reads all labels from the provided NPZ, selects the full-data best model before any query, and writes it at every round. Claims budget 100 despite using all 277/872 labels. | **+215.444872** | **Works**; all scored regrets become zero. |
| [Train-complement oracle shortcut](redteam/train_complement_oracle_shortcut/run.sh) | Uses only the visible train splitter to reconstruct its complement, then reads those visible labels and selects the best hidden-slice model. Reconstructed item IDs are saved. | **+215.444872** | **Works**; no hidden evaluator import or hidden score lookup is needed. |
| [Trace-budget shortcut](redteam/trace_budget_shortcut/run.sh) | Reuses the freshly generated seed-0 reference's complete 100-query trace, changing only `budget` from 100 to 1. | **+13.125260** | **Works**; +11.088000 vs the identical unmodified seed-0 run. Below unpaired seed spread, but the paired metadata effect is exact. |
| [Score-and-success-field shortcut](redteam/spoof_scores_shortcut/run.sh) | Writes huge negative regrets, success flags, impossible timings/counts, and a forged `scoring/scores_test.json`, while selecting model 0. | -521.818729 | Fails; hidden evaluation overwrites the score file and recomputes performance/counts. Same score as the constant-model shortcut. |
| [Ground-truth relabel shortcut](redteam/groundtruth_relabel_shortcut/run.sh) | Replaces the mutable NPZ labels with model 0's predictions, then always selects model 0. | **+215.444872** | **Works**; the evaluator accepts the changed answers. |
| [Fixed-prediction replacement shortcut](redteam/replace_fixed_predictions_shortcut/run.sh) | Replaces every fixed candidate's predictions with constant class 0, preserving original labels and dimensions. Always selects model 0. | **+215.444872** | **Works**; every candidate and the recomputed oracle become equally bad, yielding zero relative regret. |
| [Task-size inflation shortcut](redteam/task_size_inflation_shortcut/run.sh) | Duplicates every original item 200 times and always selects the original model 0. It adds no examples or scientific information. | **+885.464636** | **Works**; item-count credit overwhelms poor regret and final selection. |
| [Backdated-selection shortcut](redteam/backdate_selection_shortcut/run.sh) | Runs the existing uncertainty baseline for 100 labels, then emits one step containing only its final selection, with budget 100. Saves the real trajectory separately. | **+215.444872** | **Works**; the evaluator treats the final model as the initial selection and pads it through all 100 rounds. |
| [Constant-model-zero shortcut](redteam/constant_zero_shortcut/run.sh) | Always selects model index 0 without changing data. | -521.818729 | Fails; ordinary degenerate selection is penalized. |
| [Empty-trace shortcut](redteam/empty_trace_shortcut/run.sh) | Returns no steps. | No score; exit 1 | Fails: `Trace has no steps`. |
| [Null-model shortcut](redteam/null_model_shortcut/run.sh) | Returns null model IDs. | No score; exit 1 | Fails on integer conversion. |
| [NaN-model shortcut](redteam/nan_model_shortcut/run.sh) | Returns NaN model IDs. | No score; exit 1 | Fails on integer conversion. |
| [Negative-model shortcut](redteam/negative_model_shortcut/run.sh) | Returns model index -1. | No score; exit 1 | Fails the index-range check. |
| [Negative-budget shortcut](redteam/negative_budget_shortcut/run.sh) | Declares budget -1,000,000. | Missing result; exit 0 | Fails to establish a numeric win: active-method filtering omits CODA, leaving empty experiment results. Missing metrics must be rejected downstream. |

The task-size shortcut directly falsifies the requirement that a failing method cannot outrank the reference through other metrics. Its RTE/SST2 cumulative regrets are **1191.335740 / 527.522936**, final regrets **11.913357 / 5.275229**, and both near-optimal indicators are zero. Its original model accuracies are 76.17% / 91.28%, versus best-pool accuracies 88.09% / 96.56% ([saved evidence](redteam/task_size_inflation_shortcut/failure_evidence.json)). Hidden item counts grow from 138/436 to 27,700/87,200. That alone adds **1422.958000** weighted points; all other metrics contribute **-537.493364**, leaving a large positive rank. Linear count credit is unbounded if duplication grows.

The answer shortcuts use material supplied to future agents: NPZ `labels` are present alongside predictions, and the provided download converter also identifies the public `oracle.npy` source. The train-complement shortcut imports only `eval.train.evaluate.train_item_indices`; it never imports hidden evaluation. The replacement shortcuts modify persistent data artifacts that the gym leaves in place for the final job. They do not alter evaluator code, processes, or Python environments. Their high scores are benchmark-design failures even though the scientist constraints prohibit these actions and a later shortcut grader may reject them.

For the next iteration, the major fixes are to keep evaluation labels and fixed prediction artifacts outside the submitted workspace, enforce their identities and dimensions, and expose label acquisition through a counted interface. Cumulative regret needs an evaluator-observed selection at each actual round: neither a declared budget nor a retrospective JSON trajectory establishes when labels were used. Treat fixed task size, candidate count, and label-budget validity as hard constraints, rather than additive rewards. Require every configured experiment and finite metric, and regenerate the stored reference using the exact final evaluator and saved traces.

The audit ran 23 cases in two bundled jobs. The first worker was lost after completed results had been saved; the second resumed saved eta=1 seeds and finished with exit code 0. Job logs are archived under [redteam/job_evidence/](redteam/job_evidence/). Each case installed its outputs, ran the visible train script, and invoked **`bash scripts/evaluate_test.sh` with no arguments in a new subprocess**, with ordinary defaults and no surviving candidate hooks in the evaluator. [redteam/execute.py](redteam/execute.py) restores all changed benchmark files, score files, and original output directories after each case, including rejected traces. Candidate code does not use the audit-only stored reference copies; the harness uses those only to calculate rank. The original-trace rescore is explicitly an auditor check. [Restoration checks](redteam/restoration.json) confirm original score files, benchmark bytes and saved output traces, and unchanged tracked gym code against iteration-start commit `1b7ab8e`. All persistent audit source, traces, scores, and evidence are under `redteam/`; the temporary job script is restored with git. No gym scoring or method fix is included in this iteration.
