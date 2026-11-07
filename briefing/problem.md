# Problem Definition

## Research Question

Given a fixed pool of candidate classifiers that have already produced predictions on an unlabeled test set, identify the classifier with the best test accuracy while requesting as few ground-truth labels as possible.

## Why It Matters

Selecting among many pretrained or domain-adapted models normally requires a labeled validation set. A label-efficient selector reduces annotation cost for users who need to choose a model for a new deployment distribution.

## Success Criteria

A successful selector picks a best or near-best candidate early in the label acquisition process and keeps it, giving low cumulative regret over a 100-label budget, and does so across tasks with different numbers of candidates, difficulty and class balance (including an imbalanced task where simple baselines currently do better). It also picks a good candidate before any label is revealed. Labels are obtained only from the evaluator's oracle, one query per round, so every improvement is measured under the same annotation budget, the same candidate predictions and the same hidden labels for all methods.
