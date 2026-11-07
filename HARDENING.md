# Hardening report

**Verdict.** Better results now score higher, and every shortcut in REDTEAM.md now scores far below the reference.
- The paper's own acquisition ablation (CODA posterior with committee-entropy queries, Fig. 6) is genuinely better on the held-out data. It ranks **+113.0** over the reference.
- Removing posterior learning ranks **−173.6**.
- Random sampling ranks −60.0, uncertainty sampling −91.6 and EnsV −154.2.
- The seed-to-seed rank range of the reference is 0.61.
- All 7 shortcuts that paid before (+13 to +885) now score between **0.0** (forged bookkeeping, ignored) and **−1 602**.
- Malformed selectors are scored as failed runs (**−8 243**).

The cost: CODA's advantage does not reproduce on the 138-item held-out half of RTE (see "Not fixed").

Ranks below use the task's formula: `sum_e weight_e · sum_m coefficient_{e,m} · (candidate − reference)`.
- **After** numbers come from `scoring/reference.json` and the `coda` entry of the regenerated `scoring/scores_test.json`. Every row was produced by the unmodified `bash scripts/evaluate_test.sh` in job `eb5cef80-a44`.
- Evidence for each case is in `redteam/hardening/<case>/` (`result.json`, `scores_test.json`, `selector.json`), with the totals in `redteam/hardening/summary.json`.
- **Before** numbers are REDTEAM.md's, which used the old reference on the old two-experiment test.

## 1. What was wrong and what changed

REDTEAM.md's shortcuts had two root causes:
1. Every selector could read every label (`data/downloads/modelselector/*.npz`).
2. The hidden test scored a JSON trace that the method wrote itself.

That trace held the claimed budget, a backdatable trajectory, and item and model counts that earned credit. No per-shortcut patch fixes that, so the test was rebuilt around an oracle the evaluator owns.

- **Evaluator-held label oracle** (`eval/test/evaluate.py`, mirrored independently in `eval/train/evaluate.py`; client in `data/selection_protocol.py`).
  - Selectors are registered as commands in `outputs/selectors/<name>.json`.
  - For each experiment and seeds 0–4, the evaluator starts the command and sends it the pool's predicted classes.
  - In every round, the selector must commit `select` before it receives the label of its `query`.
  - The evaluator records the selections, counts the labels it reveals and computes every metric from labels only it holds. Nothing else the selector writes is read.
  - A crash, timeout (30 min per episode), malformed or non-integer message, out-of-range index, repeated query, or query past the budget fails the experiment. The failure is scored as if the selector chose the worst candidate in every round and spent the full budget.
- **Held-out answers live only in `eval/test/`.**
  - Each task's items are split into two halves balanced in class mix and per-candidate error.
  - Visible half: `data/benchmarks/<task>_train.json`, with predictions and labels.
  - Held-out half: `eval/test/benchmarks/<task>_test.json`. Its labels are stored as keyed codes, and its candidate order and item order are shuffled with fixed secret seeds. Knowledge of candidate indices from the visible half therefore does not transfer.
  - Both evaluators check SHA-256 checksums of their data.
  - Provenance script: `eval/test/build_benchmarks.py`. It fetches ModelSelector into a temporary directory that is deleted afterwards.
- **Removed from the workspace:**
  - the full-pool downloads and the 3.25 GB release zip, which contained every label;
  - the old traces, which recorded full-pool per-model accuracies;
  - all `redteam/*/traces`, `reconstructed_answers.json` and the backdate trajectories;
  - the red-team `_state` copies.
  - `scripts/download.sh` no longer fetches labelled full-pool arrays; it verifies the tracked visible files.
  - The hidden evaluator's per-episode log omits per-round selections, which would reveal the best hidden index.
- **Scoring (`scoring/reference.json`):**
  - `num_models` and `num_items` are removed. The pool is fixed by `eval/`, so they could only reward invalid inflation.
  - `label_budget` is now the evaluator's own count; it cannot be claimed.
  - Every experiment in `reference.json` is in the no-argument hidden test, including MRPC and both zero-label experiments, which were previously untested.
- **Baseline fidelity.** The released random and uncertainty baselines break risk and entropy ties at random, including the all-tied round 0. The gym's versions picked index 0. They now follow the released code (`baseline/selectors.py`). EnsV is registered as a selector.
- **Reference speed.** CODA (`method/coda_reproduction.py`) keeps the paper's steps:
  - consensus Dirichlet prior with α=0.1, T=0.5;
  - Beta-quadrature P_best;
  - EIG over disagreement items, with a unit hypothetical update;
  - η=0.01 posterior updates.

  The EIG is vectorized. A hypothetical label changes one row, and each model has only two possible Beta updates in that row. `redteam/hardening/check_fast_coda.py` compares it with the original per-model loop at six posterior states: maximum differences are 2.8e-17 in P_best and 3.6e-15 in EIG, with identical argmax sets. A 100-label episode now takes 10–60 s instead of about 10 minutes.

## 2. Do better results score higher? (variants before → after)

| Variant | What it is | Before (REDTEAM) | After |
|---|---|---:|---:|
| CODA + uncertainty acquisition | Paper's Fig. 6 ablation: CODA's posterior selects, queries go to the highest committee-entropy item | — | **+113.0** |
| Reference CODA | Paper defaults | 0 | 0 (seed range 0.61) |
| Random sampling | Registered baseline | −223.9 | −60.0 |
| Uncertainty sampling | Registered baseline | — | −91.6 |
| No posterior learning | η = 0: CODA prior and EIG queries, beliefs never update | −570.3 | −173.6 |
| EnsV | Zero-label ensemble validation | — | −154.2 |
| "Honest stronger variant": η 0.1 + stop after 12 unchanged rounds (≥25 labels) | REDTEAM's better variant | +151.4 | **−755.9** |

Cumulative regret at 100 on the held-out halves (zero-label rows: initial regret):

| Task | CODA | + uncertainty acquisition | η 0.1 + stability stop | No learning | Random | Uncertainty | EnsV |
|---|---:|---:|---:|---:|---:|---:|---:|
| RTE | 702.2 | **383.8** | 3039.9 | 1376.8 | 349.7 | 278.6 | 1376.8 |
| SST2 | **27.5** | 51.4 | 33.7 | 367.0 | 208.6 | 77.5 | 367.0 |
| MRPC | 99.5 | **86.9** | 178.9 | 98.0 | 230.0 | 424.7 | 98.0 |
| QNLI | 219.7 | 120.6 | **68.7** | 216.7 | 244.3 | 176.5 | 216.7 |
| QQP | 21.2 | **17.8** | 2239.2 | 33.3 | 110.0 | 65.5 | 33.3 |
| MNLI | 23.8 | **19.8** | 101.2 | 450.0 | 466.9 | 512.3 | 450.0 |
| CoLA | 3068.9 | 1550.1 | 4967.6 | 595.0 | 282.7 | **139.6** | 479.8 |
| RTE zero-label | 13.8 | 13.8 | 13.8 | 13.8 | 15.1 | 15.1 | 13.8 |
| SST2 zero-label | 3.7 | 3.7 | 3.7 | 3.7 | 6.1 | 6.1 | 3.7 |

**The ablation is better on both slices.** On the visible slice it beats CODA on RTE (293 vs 470), QNLI (65 vs 178), QQP (14.6 vs 31.0) and CoLA (999 vs 1046). The gym ranks it above the reference on the held-out data, which is the "better results score higher" check.

**REDTEAM's "better" variant was better only where it had been tuned.**
- On the visible halves it looks excellent: RTE 48.9, SST2 14.2 and MRPC 1.0, stopping after 25 labels.
- On the held-out RTE half its selection never stays unchanged for 12 rounds, so it never stops. With η=0.1 it then drifts to a 40.6-point-regret model (3039.9).
- It also collapses on QQP (2239, on both slices) and CoLA.
- Its old +151 came from two full-pool tasks with eta tuned on the same items.
- On held-out data it is genuinely worse, so ranking it lower is correct. This is the gym catching a method tuned to the visible data.

**Effect sizes vs. noise.**
- Seeds 0–4 are fixed by `eval/`, so a rerun reproduces the reference exactly. CODA is deterministic on 6 of 7 tasks; on QNLI its seeds range from 216.2 to 220.8. The spread of single-seed ranks around the 5-seed mean is 0.61.
- Every effect above is at least 60 points.
- The stochastic baselines' seed variability is averaged over five seeds.

## 3. Shortcuts: fix and score before → after

| Shortcut (REDTEAM) | Before | Fix | After |
|---|---:|---|---:|
| Visible-label oracle (read the NPZ labels, pick the best model, claim budget 100) | +215.4 | Held-out labels exist only in `eval/test/`. The visible labels belong to other items, and candidate order differs on the held-out half. The evaluator counts 0 labels used. | **−1455.8** |
| Train-complement oracle (rebuild the hidden split from `eval/train`, read its labels) | +215.4 | No visible code or data contains held-out membership or labels. The train evaluator only knows its own file. The case cannot be expressed; its nearest analog is the row above. | closed |
| Trace budget (reference trace, `budget` 1) | +13.1 | Budget is the oracle's count of revealed labels. Reference CODA with forged `budget`, `label_budget`, `num_items`, `num_models`, regret and success fields in every message. | **0.0** (identical to reference) |
| Ground-truth relabel (overwrite NPZ labels) | +215.4 | Hidden test reads its own checksummed copy. Visible labels overwritten with model 0's predictions; selects model 0. | **−1602.3** |
| Fixed-prediction replacement | +215.4 | Same as relabel; visible predictions set to class 0. | **−1602.3** |
| Task-size inflation (duplicate items ×200) | +885.5 | `num_items`/`num_models` removed as metrics. Pool is fixed by `eval/`. Visible items duplicated ×200. | **−1602.3** |
| Backdated selection (one-step trace with the final pick) | +215.4 | Selection is recorded every round before the next label is revealed. Commits model 0 for 100 queried rounds, then announces the best with a forged full `trajectory`. | **−1600.6** |
| Forged score files | −521.8 | Evaluator writes its output after all selector processes have ended. | −1602.3 |
| Constant model 0 | −521.8 | — | −1602.3 |
| Empty trace / null / NaN / negative model id | error, no score | Scored as failed run (worst candidate each round, full budget) | **−8242.6** each |
| Negative budget declared | missing result | Declared budget is ignored; covered by the trace-budget row | — |
| New: float id, duplicate query, query past budget, non-zero exit after answering | — | Protocol violations → failed run | **−8242.6** each |

Every case restored the visible data and `scoring/` files byte for byte (`restored: true` in each `result.json`).

## 4. Shortcuts left open, and why

- **Reading `eval/test/` at test time.** The selector runs in the same container as the hidden files, so it could open `eval/test/benchmarks/*.json`.
  - The labels there are keyed codes, so a blind scan for labels finds nothing usable. Decoding them requires reading the hidden evaluator's code.
  - The task says not to build defenses against the process environment. Any method that opens `eval/test/` is a clear violation for the separate grader.
- **Matching candidates across slices.** The ModelSelector arrays, including all labels, are public on GitHub. Git history also holds earlier traces with full-pool accuracies (original candidate order) and the old item-ID file.
  - Exploiting either requires fetching external data or mining history, then recovering the secret candidate permutation of the held-out pool by statistical matching.
  - The constraints forbid both, and both are visible in code.
- **Per-task tuning.** A selector could detect a task (for example by pool size) and use settings tuned on that task's visible half.
  - The halves are balanced, so such tuning partly transfers.
  - `scoring/CONSTRAINTS.md` requires one configuration for all tasks; the grader must check it.
- **Leftover or daemonised processes** writing files after the evaluator exits: out of scope by instruction.

## 5. Reference score before and after

**Before** (old `scores_test.json`: full-pool traces rescored on the old half; only RTE and SST2 scored):
- RTE: cumulative regret 271.3, near-optimal 1.0.
- SST2: cumulative regret 48.6, near-optimal 1.0.
- 44% of the configured weight (MRPC and both zero-label experiments) was untested.

**After** (evaluator-driven, held-out halves, seeds 0–4):

| Experiment | Cuml. regret @100 | @25 | @50 | Initial | Instant @100 | Near-opt @100 | Labels | Paper CODA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RTE | 702.2 | 344.2 | 661.6 | 13.8 | 0.0 | 1.0 | 100 | 283.8 |
| SST2 | 27.5 | 24.1 | 27.5 | 3.7 | 0.0 | 1.0 | 100 | 51.7 |
| MRPC | 99.5 | 32.8 | 70.1 | 1.0 | 1.0 | 1.0 | 100 | 49.0 |
| QNLI | 219.7 | 76.3 | 159.7 | 2.2 | 0.03 | 1.0 | 100 | 120.4 |
| QQP | 21.2 | 19.2 | 20.5 | 0.3 | 0.0 | 1.0 | 100 | 4.8 |
| MNLI | 23.8 | 13.8 | 21.0 | 4.5 | 0.0 | 1.0 | 100 | 23.5 |
| CoLA | 3068.9 | 90.0 | 321.5 | 6.0 | 56.0 | 0.0 | 100 | 2226.7 (instant 56.0) |
| RTE zero-label | — | — | — | 13.8 | — | — | 0 | 14.8 |
| SST2 zero-label | — | — | — | 3.7 | — | — | 0 | 3.6 |

**The paper's direction holds on the held-out data, task by task:**
- CODA beats both baselines on SST2, MRPC, QQP and MNLI. MNLI matches the paper (23.8 vs 23.5).
- On QNLI it beats random (219.7 vs 244.3) but not uncertainty sampling (176.5).
- On CoLA it fails as the paper reports, ending on the biased model with 56.0 regret, exactly the paper's Table 4 value.
- Zero-label CODA ties EnsV, as in Table 5.
- On RTE it loses (see section 7).
- Overall, both simple baselines rank below the reference.

`scoring/scores.json` and `scores_train.json` hold the same evaluation on the visible halves (reproduced via `scripts/evaluate.sh`).

## 6. Headroom check

**Metrics at their favorable bound for the reference on the held-out data:**
- `instant_regret_at_100` = 0 on RTE, SST2, QQP and MNLI (0.03 on QNLI).
- `near_optimal_at_100` = 1 on all six regular tasks.
- `instant_regret_at_50` = 0 on several tasks, including SST2, MRPC and MNLI.

**Metrics far from their bound:**
- `cumulative_regret_at_25/50/100` on every task: RTE 702, CoLA 3069, QNLI 220, MRPC 99.5 against a bound of 0.
- `label_budget`: 100 against a bound of 0.
- `initial_regret`: 13.8 on RTE.

**What I added.** I added GLUE CoLA, the paper's own documented failure (Sec. 7, Fig. 4), as a scored experiment where the reference falls short. Both baselines are 10–20× better there. I also added QNLI, QQP and MNLI, so the hidden test covers all seven GLUE rows of Table 1.

**Headroom is demonstrated.** The ablation variant gains +113, and on QNLI the η=0.1 variant reaches 68.7.

## 7. Weights (and why they changed)

- **Six regular Table-1 tasks** (RTE, SST2, MRPC, QNLI, QQP, MNLI): **0.13 each**. The paper's main claim is per task ("best on 18 of 26"), so the tasks are equal.
- **CoLA: 0.02.** Its regret scale is about 10× the others' (reference 3069). At 0.13 it would decide the ranking on its own: a selector that only fixed CoLA would outrank any improvement on the main claim. At 0.02 its maximum swing (about 60) is comparable to another task's.
- **Zero-label experiments: 0.10 each** (Table 5 secondary claim; previously 0.14 and 0.10).
- **Weights were fixed before the final run.** In the first pass (RTE/SST2/MRPC/CoLA only; `redteam/hardening/first_pass/`), RTE's flip plus CoLA at 0.06 let random (+257) and uncertainty (+245) outrank the reference. That is why the remaining Table-1 GLUE tasks were added.
- **Coefficients are unchanged.**

## 8. Train/test agreement

**Both slices use the same experiments, budgets, seeds, metrics and protocol.** `eval/train` and `eval/test` share no code and neither imports `method/`.

**Honest methods mostly order alike on both.** The ablation beats CODA on RTE, QNLI, QQP and CoLA on both slices. Uncertainty sampling beats random on all 7 tasks of the visible slice and 5 of 7 of the held-out slice (not MRPC or MNLI).

**Absolute values differ, as expected on 200–600-item pools:**
- CODA's RTE score is 470 on the visible slice and 702 on the held-out slice.
- MNLI: 65 vs 24.
- CoLA: 1046 vs 3069.

**A method tuned to the visible data loses on held-out data.** The η 0.1 + stopping variant shows this clearly (RTE 48.9 visible, 3039.9 held-out).

## 9. Not fixed, stated plainly

- **RTE reverses the paper's direction on half pools.** On the 138-item held-out half, CODA needs about 48 labels to leave its 13.8-regret starting choice (about 80 on the visible half), against about 20 on the full 277-item pool. It loses to uncertainty (278.6) and random (349.7).
  - Halving cannot be avoided without placing held-out labels next to visible ones.
  - CODA is deterministic here, so this is a property of the smaller pool, not noise. Fewer items separate the top candidates.
  - RTE therefore carries the same weight as the other tasks, not more, and its shortfall is headroom.
- **`python validate.py --post-design` flags 13 primary values more than 2× from the paper.**
  - CODA: RTE (held-out), MRPC (held-out, 99.5 vs 49.0), QQP (21–31 vs 4.8), MNLI (visible, 65 vs 23.5) and CoLA (visible, 1046 vs 2226.7).
  - Uncertainty: QQP (31–65 vs 388.9, much better than the paper) and MNLI (held-out, 512 vs 80.8).
  - All come from the scaled setup: half pools, and 1200-item samples for QNLI/QQP/MNLI. The paper's numbers are kept unchanged.
  - Plain `python validate.py` passes.
- **Old answers remain reachable through git history.** Previous commits contain full-pool accuracies (original candidate order) and the earlier item-ID file. History cannot be rewritten from here; exploiting it also needs the hidden candidate permutation (section 4).
- **Hidden-test cost.** With four registered selectors the hidden test runs about 25 CPU-minutes, almost all of it CODA (5 seeds × 9 experiments). A slower submitted selector is bounded by the 30-minute-per-episode limit.
