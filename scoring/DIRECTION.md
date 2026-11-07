# Research Direction

## Research Problem

This benchmark concerns active model selection: choosing the best classifier
from a fixed candidate pool for a target dataset while minimizing label
queries.

## Core Contribution Scope

Improvements should make better use of the unlabeled prediction matrix and
of adaptively acquired labels, both to decide which item to label next and
which candidate to select, and when further labels are no longer worth
acquiring. Known weak spots of the reference are worth attention: early
labels that pull it away from a good starting choice (QNLI), and imbalanced
tasks with biased candidates (CoLA), where it is beaten by simple baselines.

## Approach Scope

In-bounds approaches may redesign the probabilistic model of candidate
performance, the acquisition function, the uncertainty estimates, the
selection rule, stopping rules, or priors derived from the same prediction
matrix. They must remain active model selection methods operating on the
fixed candidate predictions through the label oracle, with one configuration
for all tasks.

## Out of Bounds

Training new classifiers, adding candidate models, using external or visible
labels as a stand-in for queried ones, reading held-out evaluation files,
per-task tuning, and exceeding the label budget are out of bounds: they
sidestep the label-efficiency question the benchmark measures.
