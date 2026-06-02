# Experiment Overview

## glue_rte_active_selection

This experiment tests active model selection on a small natural-language classification benchmark with many candidate predictors and a limited label budget. It measures whether the selector can use a small number of queried labels to choose a model whose full-test-set accuracy is close to the true best model in the candidate pool.

## glue_sst2_active_selection

This experiment repeats the active model selection test on a larger sentiment-classification benchmark. It checks whether the same label-efficient selection behavior transfers to a different GLUE task while keeping the candidate pool and label budget fixed.

## glue_mrpc_active_selection

This experiment adds a paraphrase-classification benchmark from the same prediction-suite family. It broadens the active-selection score beyond the two completed GLUE settings while preserving the fixed prediction pool and fixed annotation budget.

## glue_rte_unsupervised_selection

This experiment tests model selection on GLUE RTE before any labels are queried. It measures whether the selected candidate is close to the best model in the pool under a strict zero-label budget.

## glue_sst2_unsupervised_selection

This experiment repeats the zero-label model selection test on GLUE SST2. It checks whether no-query selection behavior transfers to a larger sentiment-classification benchmark while keeping the same fixed candidate pool and evaluation labels.
