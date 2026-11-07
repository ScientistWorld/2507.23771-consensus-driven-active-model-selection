# Experiment Overview

All experiments run every registered selector through the evaluator's label
oracle (see `briefing/evaluation.md`) with the full candidate pool, seeds 0–4,
and the selection recorded by the evaluator in every round. SST2, QNLI and
RTE have a visible half (`scripts/evaluate_train.sh`) and a held-out half
(`scripts/evaluate_test.sh`). CoLA is scored on its held-out half only.

## glue_sst2_active_selection

Active model selection on a sentiment task where most candidates are accurate
and differ only on a minority of items, with a 100-label budget. It
establishes whether a selector can turn a few queried labels into a choice
whose accuracy on the whole pool is close to the best candidate's, and how
quickly it gets there.

## glue_qnli_active_selection

The same question on a question-answering entailment task (a fixed sample
of its items), where the starting choice is already good and early labels
easily lead a selector astray.

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
