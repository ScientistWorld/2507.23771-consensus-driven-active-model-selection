# Scientist Constraints

## Data

Use the candidate predictions the evaluator sends in each episode. The
visible benchmark files `data/benchmarks/<task>_train.json` (SST2 and QNLI
halves) are for development; RTE, MRPC and CoLA have no visible data; do not modify them (the evaluators verify their checksums).
Do not add external labeled data, other candidate models, or the upstream
full-pool label files, and do not read anything under `eval/test/` or any
other file holding held-out labels or candidate orders.

## Labels

Labels may only be obtained through the oracle protocol: one query per round,
at most 100 per active episode and none in zero-label experiments. A selector
must not use visible-slice labels at evaluation time (for example as a prior
over candidate indices): the held-out pool has a different candidate order and
different items.

## Settings

Use one fixed configuration for all tasks, as the paper does. Choosing
settings per task (by task name, pool size or any other fingerprint of a
particular benchmark) tunes to the benchmark rather than solving the problem.

## Output Contract

Register a selector with `bash scripts/register_selector.sh <name> <command...>`.
It must speak the protocol in `data/selection_protocol.py`: write only protocol
messages to stdout, select an integer model index every round, query distinct
item indices or `null`, exit with status 0, and finish each episode within 30
minutes on one CPU core. Evaluation never imports your code; it runs your
command and scores the selections it records.

## Compliance Checklist

- [ ] Uses only the predictions sent in the episode and the labels returned by the oracle.
- [ ] Queries at most the episode's budget, each item at most once.
- [ ] Same configuration on every task; no task-specific or index-specific knowledge.
- [ ] Does not read `eval/test/` or modify `data/benchmarks/`.
- [ ] Registered under `outputs/selectors/` and scored by `scripts/evaluate_*.sh`.
