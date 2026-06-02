# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-06-01 23:14] - method_runs
- Read the paper and identified CODA's core computational steps: consensus confusion priors, Beta-mixture probability of best model, expected information gain acquisition, and Dirichlet updates.
- Implemented a standalone CODA reproduction and random/uncertainty active model selection baselines.
- Downloaded the authors' released CODA archive and verified a smaller portable ModelSelector data path for GLUE/MSV raw prediction files.
- Ran a one-step local smoke test on released GLUE RTE predictions; CODA, baselines, trace writing, and method-independent evaluation all executed successfully.

### [2026-06-02 00:00] - method_runs
- Completed the first 100-label, 5-seed GLUE RTE job without runtime errors.
- The first full run did not support CODA's core claim: CODA's cumulative regret was worse than random and uncertainty baselines.
- Identified and fixed an implementation divergence from the released CODA code: EIG hypothetical label updates now use a unit pseudo-count, while actual posterior updates keep the paper's small learning rate.
- Updated evaluation to accumulate regret only after label queries, matching the paper's reported cumulative-regret convention.

## Stop Justification
- Not stopped. A corrected CPU job is being submitted to rerun the 100-label GLUE RTE core experiment.
