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
