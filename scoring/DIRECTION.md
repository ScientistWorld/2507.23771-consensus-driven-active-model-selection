# Research Direction

## Research Problem

This benchmark concerns active model selection: choosing the best classifier from a fixed candidate pool for a target dataset while minimizing label queries.

## Core Contribution Scope

Improvements should focus on better use of the unlabeled prediction matrix and adaptively acquired labels to decide both which item to label next and which candidate model to select.

## Approach Scope

In-bounds approaches may redesign the probabilistic model over candidate performance, improve the acquisition function, change uncertainty estimation, or add principled priors derived from the same prediction matrix. They must remain active model selection methods operating on fixed candidate predictions.

## Out of Bounds

Replacing the task with supervised training of a new classifier is out of bounds. Adding new candidate models, using external labeled validation data, or querying beyond the stated budget is also out of bounds for the primary experiment because these changes sidestep the paper's label-efficiency claim.
