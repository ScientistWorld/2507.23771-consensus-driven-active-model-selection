# Problem Definition

## Research Question

Given a fixed pool of candidate classifiers that have already produced predictions on an unlabeled test set, identify the classifier with the best test accuracy while requesting as few ground-truth labels as possible.

## Why It Matters

Selecting among many pretrained or domain-adapted models normally requires a labeled validation set. A label-efficient selector reduces annotation cost for users who need to choose a model for a new deployment distribution.

## Success Criteria

A successful solution should choose a near-best or best candidate model early in the label acquisition process. It should achieve low instantaneous and cumulative regret under a fixed label budget, on the same prediction tensors and hidden labels used by all compared methods.
