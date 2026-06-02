# Experiment Overview

## glue_rte_active_selection

This experiment tests active model selection on a small natural-language classification benchmark with many candidate predictors and a limited label budget. It measures whether the selector can use a small number of queried labels to choose a model whose full-test-set accuracy is close to the true best model in the candidate pool.

## glue_sst2_active_selection

This experiment repeats the active model selection test on a larger sentiment-classification benchmark. It checks whether the same label-efficient selection behavior transfers to a different GLUE task while keeping the candidate pool and label budget fixed.
