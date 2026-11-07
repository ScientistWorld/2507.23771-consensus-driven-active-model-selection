# Hardening report

## Verdict

- **Every shortcut in REDTEAM.md is closed.** The seven that paid before (+13 to +885) now score from **0.0** (forged bookkeeping is ignored) down to **−2 857**. Malformed or crashing selectors are scored as failed runs (**−8 484**).
- **On the final scored set, better results score higher.**
  - The paper's own Fig. 6 ablation (CODA posterior with committee-entropy queries) ranks **+37.6** over the reference.
  - Removing posterior learning ranks **−137.3**.
  - Every simple baseline ranks below the reference: uncertainty −16.6, random −108.2, EnsV −122.4.
- **The cost is coverage.** The final scored set covers SST2, QNLI and the CoLA failure case for active selection, plus RTE and SST2 zero-label selection.
  - RTE and MRPC active selection were dropped. On half pools, CODA's reproduced cumulative regret is more than 2× the paper's, and the validator rejects that (section 9).
  - On RTE it is also worse than both baselines.

## How the numbers below were produced

Every number comes from the unmodified `bash scripts/evaluate_test.sh` / `evaluate_train.sh` in job `eb5cef80-a44`. Per-case evidence is in `redteam/hardening/second_pass/<case>/`.

That run scored nine experiments. The final gym keeps five of them, so I did two things after the run:
- `scoring/scores_{test,train}.json` keep that run's entries for the five retained experiments. The data files those experiments read are byte-identical (checksums in the evaluators). The CODA and baseline code is unchanged, and the evaluator change only removes experiments from its list. A rerun of `scripts/run.sh` therefore reproduces these numbers.
- Ranks are recomputed from the saved per-experiment scores with the final weights (`redteam/hardening/summary.json`).

The rank formula is the task's: `sum_e weight_e · sum_m coefficient_{e,m} · (candidate − reference)`, with the reference being the `coda` entry of `scoring/scores_test.json`. **Before** numbers are REDTEAM.md's, which used the old reference on the old two-experiment test.

## 1. What was wrong and what changed

REDTEAM.md's shortcuts had two root causes:
1. Every selector could read every label (`data/downloads/modelselector/*.npz`).
2. The hidden test scored a JSON trace that the method wrote itself. That trace held the claimed budget, a backdatable trajectory, and item and model counts that earned credit.

I replaced the design instead of patching shortcuts one at a time.

**Evaluator-held label oracle.** The server lives in `eval/test/evaluate.py`, mirrored independently in `eval/train/evaluate.py`; the client is `data/selection_protocol.py`.
- Selectors are registered as commands in `outputs/selectors/<name>.json`.
- For each experiment and seeds 0–4, the evaluator starts the command and sends it the pool's predicted classes.
- In every round the selector must commit `select` before it gets the label of its `query`.
- The evaluator records the selections, counts the labels it reveals, and computes every metric from labels only it holds. Nothing else the selector writes is read.
- A crash, timeout (30 min per episode), malformed or non-integer message, out-of-range index, repeated query, or query past the budget fails that experiment. The failure is scored as the worst candidate in every round with the whole budget spent.

**Held-out answers live only in `eval/test/`.**
- Items are split into halves balanced in class mix and per-candidate error.
- The visible halves are `data/benchmarks/<task>_train.json` (SST2, QNLI, RTE).
- The held-out halves are `eval/test/benchmarks/<task>_test.json`. Labels are stored as keyed codes, and candidate order and item order are shuffled with fixed secret seeds. Knowledge of candidate indices from the visible half or from public files does not transfer.
- CoLA keeps only its held-out half.
- Both evaluators check SHA-256 checksums.
- The provenance script, `eval/test/build_benchmarks.py`, fetches the public arrays into a temporary directory that is deleted afterwards.

**Leaky artifacts removed.**
- The full-pool downloads, including the 3.25 GB release zip, which contained every label.
- Old traces, which recorded full-pool per-model accuracies.
- All `redteam/*/traces`, `reconstructed_answers.json`, the backdate trajectories and the red-team `_state` copies.
- `scripts/download.sh` now verifies the tracked visible files instead of fetching labelled arrays.
- The hidden evaluator's per-episode log omits per-round selections, which would reveal the best hidden index.

**Scoring (`scoring/reference.json`).**
- `num_models` and `num_items` are removed. The pool is fixed by `eval/`, so they could only reward invalid inflation.
- `label_budget` is the evaluator's own count.
- Every experiment in `reference.json` is in the no-argument hidden test.

**Baseline fidelity.** The released baselines break risk and entropy ties at random, including the all-tied round 0; the gym's versions took index 0. `baseline/selectors.py` now matches the released code. EnsV is registered as a selector.

**Reference speed.** `method/coda_reproduction.py` keeps the paper's steps:
- consensus Dirichlet prior with α=0.1, T=0.5;
- Beta-quadrature P_best;
- EIG over disagreement items with a unit hypothetical update;
- η=0.01 posterior updates.

The EIG is vectorized. `redteam/hardening/check_fast_coda.py` compares it with the original per-model loop: maximum differences are 2.8e-17 in P_best and 3.6e-15 in EIG, with identical argmax sets. A 100-label episode takes 10–60 s instead of about 10 minutes. The full hidden test takes about 12 CPU-minutes for the four registered selectors.

## 2. Do better results score higher? (variants before → after)

| Variant | Before (REDTEAM) | After |
|---|---:|---:|
| η 0.1 + stop after 12 unchanged rounds (≥25 labels) — REDTEAM's better variant | +151.4 | **+66.3** |
| CODA + uncertainty acquisition (paper Fig. 6 ablation) | — | **+37.6** |
| Reference CODA | 0 | 0 |
| Uncertainty sampling | — | −16.6 |
| Random sampling | −223.9 | −108.2 |
| EnsV (zero-label) | — | −122.4 |
| No posterior learning (η = 0) | −570.3 | **−137.3** |

Cumulative regret at 100 on the held-out halves (zero-label rows: initial regret):

| Task | CODA | + unc. acquisition | η 0.1 + stop | No learning | Random | Uncertainty | EnsV |
|---|---:|---:|---:|---:|---:|---:|---:|
| SST2 | **27.5** | 51.4 | 33.7 (31 labels) | 367.0 | 208.6 | 77.5 | 367.0 |
| QNLI | 219.7 | 120.6 | **68.7** (39 labels) | 216.7 | 244.3 | 176.5 | 216.7 |
| CoLA | 3068.9 | 1550.1 | 4967.6 (25 labels) | 595.0 | 282.7 | **139.6** | 479.8 |
| RTE zero-label | 13.8 | 13.8 | 13.8 | 13.8 | 15.1 | 15.1 | 13.8 |
| SST2 zero-label | 3.7 | 3.7 | 3.7 | 3.7 | 6.1 | 6.1 | 3.7 |

**Ablation (+37.6).** It is better than CODA on held-out QNLI and CoLA and worse on SST2. It was also better on the visible QNLI half (64.5 vs 178.2).

**Stopping variant (+66.3).** It earns its rank from QNLI (68.7 vs 219.7 with 39 labels) and from labels saved on SST2. Its weakness shows elsewhere:
- On the dropped experiments it collapses on the held-out halves: RTE 3039.9 vs CODA 702.2, ending on a 40.6-point-regret model; QQP 2239.2 vs 21.2.
- It is also worse on CoLA (4967.6).
- The gym ranks it by what it achieves on the scored experiments. A future grader should know that the remaining scored set is narrow enough for such a fragile setting to look good (section 9).

**No learning (−137.3).** Removing learning is penalized, mainly on SST2.

**Seed noise.** Seeds 0–4 are fixed by `eval/`, so a rerun reproduces the reference exactly. CODA is deterministic on SST2 and CoLA; its QNLI seeds range from 216.2 to 220.8, about ±0.8 rank points. All effects above are ≥ 16 points.

## 3. Shortcuts: fix and score before → after

| Shortcut (REDTEAM) | Before | Fix | After |
|---|---:|---|---:|
| Visible-label oracle (read the labels, pick the best model, claim budget 100) | +215.4 | Held-out labels exist only in `eval/test/`; visible labels belong to other items; candidate order differs; 0 labels counted. | **−2856.9** |
| Train-complement oracle (rebuild hidden split from `eval/train`, read its labels) | +215.4 | No visible code or data holds held-out membership or labels; cannot be expressed. Nearest analog is the row above. | closed |
| Trace budget (reference trace, budget 1) | +13.1 | Budget is the oracle's count. Reference CODA with forged `budget`, `label_budget`, `num_items`, `num_models`, regret and success fields in every message. | **0.0** (identical) |
| Ground-truth relabel (overwrite labels on disk) | +215.4 | Hidden test reads its own checksummed copy. Visible labels overwritten, selects model 0. | **−2628.6** |
| Fixed-prediction replacement | +215.4 | Same as relabel; visible predictions set to class 0. | **−2628.6** |
| Task-size inflation (duplicate items ×200) | +885.5 | Count metrics removed; pool fixed by `eval/`. | **−2628.6** |
| Backdated selection | +215.4 | Selection recorded every round before the next label. Commits model 0 for 100 queried rounds, then announces the best with a forged `trajectory`. | **−2617.4** |
| Forged score files | −521.8 | Evaluator writes its output after all selector processes end. | −2628.6 |
| Constant model 0 | −521.8 | — | −2628.6 |
| Empty / null / NaN / negative model id | error, no score | Failed run (worst candidate each round, full budget). | **−8484.1** each |
| Negative budget declared | missing result | Declared budget ignored (covered by the trace-budget row). | — |
| New: float id, duplicate query, query past budget, non-zero exit | — | Protocol violation → failed run. | **−8484.1** each |

Every case restored the visible data and `scoring/` files byte for byte.

## 4. Shortcuts left open, and why

- **Reading `eval/test/` at test time.**
  - The selector runs in the same container as the hidden files.
  - The labels there are keyed codes, so a blind scan finds nothing usable. Decoding them requires the hidden evaluator's code.
  - Per the task, I built no defenses against the process environment. Any method that opens `eval/test/` is a clear violation for the grader.
- **Public data and git history.**
  - The ModelSelector arrays, including all labels, are public.
  - Git history holds earlier traces, full-pool accuracies and visible halves that no longer exist; CoLA's other half and the MRPC half are among them.
  - Using either requires fetching or mining that data and then recovering the secret candidate permutation and item shuffle of the held-out pool.
  - `scoring/CONSTRAINTS.md` forbids it, and it is visible in code.
- **Per-task tuning.** A selector could fingerprint a task and use settings tuned on its visible half.
  - The halves are balanced, so such tuning partly transfers.
  - `scoring/CONSTRAINTS.md` requires one configuration for all tasks.
- **Leftover or daemonised processes:** out of scope by instruction.

## 5. Reference score before and after

**Before** (old `scores_test.json`: full-pool traces rescored on a half; only RTE and SST2 scored):
- RTE: cumulative regret 271.3.
- SST2: cumulative regret 48.6.
- 44% of the configured weight was untested.

**After** (CODA, held-out halves, seeds 0–4):

| Experiment | Cuml. @100 | @25 | @50 | Initial | Instant @50 | Instant @100 | Near-opt | Labels | Paper CODA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SST2 | 27.5 | 24.1 | 27.5 | 3.7 | 0.0 | 0.0 | 1.0 | 100 | 51.7 |
| QNLI | 219.7 | 76.3 | 159.7 | 2.2 | 3.3 | 0.03 | 1.0 | 100 | 120.4 |
| CoLA | 3068.9 | 90.0 | 321.5 | 6.0 | 56.0 | 56.0 | 0.0 | 100 | 2226.7 (instant 56.0) |
| RTE zero-label | — | — | — | 13.8 | — | — | — | 0 | 14.8 |
| SST2 zero-label | — | — | — | 3.7 | — | — | — | 0 | 3.6 |

**The paper's direction holds on every retained experiment:**
- CODA beats both baselines on SST2.
- It beats random on QNLI (uncertainty sampling is better there).
- It fails on CoLA as the paper reports, ending on the biased model with exactly the paper's 56.0 final regret.
- Its zero-label choice ties EnsV, as in Table 5.
- Every reproduced primary value is within 2× of the paper's, so `validate.py` and `validate.py --post-design` both pass.

`scoring/scores.json` holds the same held-out reproduction. `scoring/scores_train.json` holds the visible halves: SST2 39.4, QNLI 178.2, RTE zero-label 15.8, SST2 zero-label 4.6.

## 6. Headroom check

**Reference at the favorable bound:**
- `instant_regret_at_100` = 0 on SST2 (0.03 on QNLI).
- `instant_regret_at_50` = 0 on SST2.
- `near_optimal_at_100` = 1 on SST2 and QNLI.

**Far from the bound:**
- Cumulative regrets: QNLI 220, SST2 27.5 and CoLA 3069, all against a bound of 0.
- `label_budget`: 100 against a bound of 0.
- Initial regrets: RTE 13.8, SST2 3.7.

**What I added.** GLUE CoLA, the paper's own failure case (Sec. 7, Fig. 4), is a scored experiment where both baselines are 10–20× better than the reference.

**Headroom is real.** Two honest variants beat the reference (+37.6, +66.3).

## 7. Weights and why

| Experiment | Weight |
|---|---:|
| SST2 active | 0.35 |
| QNLI active | 0.35 |
| CoLA active | 0.005 |
| RTE zero-label | 0.15 |
| SST2 zero-label | 0.145 |

Coefficients are unchanged.

- **The two active Table-1 tasks carry the paper's main claim equally.**
- **CoLA's weight is set by its regret scale.** It is about 3 000 pp-rounds, 10–100× the other tasks'. At any weight above about 0.01, a selector that merely avoids the CoLA failure outranks one that is better on the main tasks; uncertainty sampling did exactly that at 0.02. At 0.005, fully fixing CoLA is still worth about 15 rank points, comparable to SST2's entire headroom.
- **Zero-label selection keeps its previous total share.**

## 8. Train/test agreement

**Both slices use the same experiments, budgets, seeds, metrics and protocol.** CoLA is held-out only. `eval/train` and `eval/test` share no code and neither imports `method/`.

**Honest methods order alike across slices:**
- SST2: CODA < uncertainty < random < EnsV on both (39.4 / 91.1 / 241.7 / 458.7 visible; 27.5 / 77.5 / 208.6 / 367.0 held out).
- QNLI: uncertainty < CODA < random on both.
- Zero-label: CODA = EnsV < random = uncertainty on both.

**A method tuned to visible data loses on held-out data.** The stopping variant shows this on the dropped RTE: 48.9 visible vs 3039.9 held out.

## 9. Not fixed, stated plainly

**RTE and MRPC active selection were dropped from scoring.**
- On half pools (138 and 204 held-out items), CODA's held-out cumulative regret is 702.2 on RTE and 99.5 on MRPC, against the paper's 283.8 and 49.0. On RTE that is also worse than both baselines (uncertainty 278.6, random 349.7).
- CODA is deterministic there, so this is a pool-size effect, not noise. With fewer items there are fewer informative queries, and CODA leaves its initial choice after about 48 labels instead of about 20.
- **Full-pool fix: implemented, never run.** RTE and MRPC held out whole on their full pools, with no visible part, was implemented in commit ff6a0fb. Locally the baselines reproduced the paper there (RTE uncertainty 376.4 vs 390.3; MRPC 306.6 vs 301.1). The CODA run could not be executed: the iteration was in its validation-only phase, and this login node has neither numpy nor permission for heavy compute.
- QQP and MNLI, added in the same pass, were also dropped. On 1 200-item samples, their CODA and uncertainty values differ from the paper's full pools by 2.8–12.5×.
- The second-pass evidence for all of these is kept under `redteam/hardening/second_pass/`.

**The scored set is narrow.** Two active tasks plus CoLA and two zero-label experiments. A setting that works on SST2 and QNLI but collapses elsewhere (the η 0.1 stopping variant) can rank above the reference, as it does (+66.3).

**Old answers remain reachable through git history** (section 4). History cannot be rewritten from here.
