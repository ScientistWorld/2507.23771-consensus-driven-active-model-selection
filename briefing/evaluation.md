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

- **Held out whole: RTE, MRPC and CoLA.** These run on their full item pools,
  the paper's protocol. Their labels are not in the workspace at all, and
  there is no visible slice for them: they also test that a selector
  generalizes to tasks it was not developed on.
- **Split: SST2 and QNLI (a fixed 1200-item sample of QNLI).** The items are
  split into two halves balanced in class mix and per-candidate error rate.
  - The visible half, `data/benchmarks/<task>_train.json` (predictions and
    labels), is scored by `scripts/evaluate_train.sh` and `scripts/evaluate.sh`.
  - The held-out half is scored by `scripts/evaluate_test.sh`.

In every held-out pool, candidate order and item order differ from any
public or visible ordering, so nothing learned about particular candidate
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
| `glue_rte_active_selection` | 87 candidates, 277 items, held out whole | 375.7 / 390.3 / 283.8 |
| `glue_sst2_active_selection` | 97 candidates, 436-item halves | 219.9 / 89.6 / 51.7 |
| `glue_mrpc_active_selection` | 95 candidates, 408 items, held out whole | 318.2 / 301.1 / 49.0 |
| `glue_qnli_active_selection` | 90 candidates, 600-item halves of a 1200-item sample | 222.7 / 231.6 / 120.4 |
| `glue_cola_active_selection` | 109 candidates, 1043 items, imbalanced, held out whole | 368.1 / 169.5 / 2226.7 |
| `glue_rte_unsupervised_selection` | zero labels, held out whole | initial regret 14.8 (ensemble validation and reference selector) |
| `glue_sst2_unsupervised_selection` | zero labels, halves | initial regret 3.6 (both) |

The paper's numbers are full-pool values. RTE, MRPC and CoLA are run exactly
at that scale; SST2 and QNLI run on half pools, so their absolute values differ
somewhat. On CoLA the paper's reference selector is far worse than both
baselines; that experiment measures progress where the reference falls short.

Registered baselines: `random` (random queries, empirical-risk selection),
`uncertainty` (highest committee-entropy queries, empirical-risk selection;
both break ties at random), and `ensv` (zero-label agreement with the
committee majority).
