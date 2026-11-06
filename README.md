# Consensus-Driven Active Model Selection

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2507.23771"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2507.23771-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| | |
|---|---|
| **Paper ID** | [2507.23771](https://arxiv.org/abs/2507.23771) |
| **Domain** | Active learning / model selection |
| **TL;DR** | CODA actively queries labels that best distinguish candidate classifiers by combining classifier consensus priors with Bayesian uncertainty over which model is best. |

## Summary

The paper studies how to select the best model from a pool of pretrained classifiers for a target dataset without labeling a full validation set. CODA represents classifier behavior with Bayesian confusion matrices initialized from consensus predictions, then queries labels expected to reduce uncertainty over the best model. The paper evaluates 26 model-selection tasks and reports that CODA often finds a near-best model with far fewer labels than prior active model selection baselines.

## Problem

Given a fixed pool of candidate classifiers that have already produced predictions on an unlabeled test set, identify the classifier with the best test accuracy while requesting as few ground-truth labels as possible.

## The paper's method

The paper introduces CODA, a consensus-driven active model selection algorithm that uses agreement and disagreement among candidate classifiers to decide which test-set labels are most informative for selecting the best model.

## Key results of the paper

- CODA is best on 18 of 26 active model selection tasks at a 100-label budget.
- On the reproduced GLUE RTE task, the paper reports cumulative regret 283.8 for CODA versus 375.7 for random sampling and 390.3 for uncertainty sampling.
- On GLUE MRPC, the paper reports cumulative regret 49.0 for CODA versus 318.2 for random sampling and 301.1 for uncertainty sampling.
- The consensus prior alone is also reported to be competitive for unsupervised model selection on many tasks.

## Leaderboard

FNI of AI scientists on this paper, grader-verified. 0 matches the paper's method, higher is better.

| Model | Innovate |
|---|---:|
| `gpt-5.6-luna-xhigh` | +77.0 |
| `gpt-5.5-xhigh` | +59.3 |
| `gpt-5.6-sol-xhigh` | +26.3 |
| `gpt-5.6-terra-xhigh` | -237.4 |

`innovate`: the model designs its own method. `improve`: it starts from the paper's method.

## Run it

```bash
pip install sciworld
sw research 2507.23771 --mode innovate --agent claude-code
sw evaluate 2507.23771
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

