# Paper's Method

## Key Contribution

The paper introduces CODA, a consensus-driven active model selection algorithm that uses agreement and disagreement among candidate classifiers to decide which test-set labels are most informative for selecting the best model.

## Approach

CODA models each classifier with a per-class confusion matrix in a Bayesian Dawid-Skene-style framework. It initializes Dirichlet priors for these confusion matrices using consensus labels formed by summing predictions across the candidate pool, regularized by a diagonal prior. At each step it estimates the probability that each candidate model is best by integrating over row-wise Beta distributions for class accuracy. It then scores unlabeled items by expected information gain in that probability-of-best distribution, queries the highest-scoring label, and applies Dirichlet-categorical updates to the affected confusion-matrix rows.

## Main Claims

The primary claim is that CODA reduces annotation effort for active model selection, achieving lower cumulative regret at a 100-label budget than prior active selection methods on most benchmark tasks. Secondary claims include strong unsupervised initialization from the consensus prior and ablation evidence that the consensus prior and expected-information-gain acquisition improve early label efficiency.
