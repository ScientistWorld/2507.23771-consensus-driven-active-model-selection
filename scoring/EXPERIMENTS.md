# Experiment Overview

All experiments run every registered selector through the evaluator's label
oracle (see `briefing/evaluation.md`) on one half of each task's items, with
the full candidate pool, seeds 0–4, and the selection recorded by the
evaluator in every round. The visible half is scored by
`scripts/evaluate_train.sh`; the held-out half, whose labels and candidate
order are not in the workspace, by `scripts/evaluate_test.sh`.

## glue_rte_active_selection

Active model selection on a small, hard natural-language inference task with
many candidate predictors and a 100-label budget. It establishes whether a
selector can turn a few queried labels into a choice whose accuracy on the
whole pool is close to the best candidate's, and how quickly it gets there.

## glue_sst2_active_selection

The same question on a larger sentiment task where most candidates are
accurate and differ only on a minority of items. It checks that label-efficient
selection transfers to a different pool and item distribution.

## glue_mrpc_active_selection

The same question on a paraphrase task with a skewed label distribution,
where a reasonable zero-label choice already exists. It measures whether
queried labels improve on the starting choice rather than degrade it.

## glue_qnli_active_selection

The same question on a question-answering entailment task (a fixed sample
of its items), with a mid-sized candidate pool.

## glue_qqp_active_selection

The same question on a paraphrase task (a fixed sample of its items) where
candidates are close in accuracy, so a selector must separate near-ties
without wasting labels.

## glue_mnli_active_selection

The same question on a three-class inference task (a fixed sample of its
items), checking that label-efficient selection is not limited to binary
classification.

## glue_cola_active_selection

The same question on an acceptability task with imbalanced labels and some
strongly biased candidates. Here simple label-driven baselines beat the
paper's reference selector by a wide margin, so this experiment measures
progress where the reference falls short.

## glue_rte_unsupervised_selection

Zero-label selection on RTE: the selector gets a budget of 0 and is scored on
its single choice. It measures how much can be inferred about candidate
quality from the unlabeled prediction matrix alone.

## glue_sst2_unsupervised_selection

Zero-label selection on SST2, checking the same capability on a larger,
easier pool.
