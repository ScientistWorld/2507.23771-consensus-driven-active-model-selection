# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-06-01 23:14] - method_runs
- Read the paper and identified CODA's core computational steps: consensus confusion priors, Beta-mixture probability of best model, expected information gain acquisition, and Dirichlet updates.
- Implemented a standalone CODA reproduction and random/uncertainty active model selection baselines.
- Downloaded the authors' released CODA archive and verified a smaller portable ModelSelector data path for GLUE/MSV raw prediction files.
- Ran a one-step local smoke test on released GLUE RTE predictions; CODA, baselines, trace writing, and method-independent evaluation all executed successfully.

## Stop Justification
- Not stopped. A CPU job is being submitted to run the 100-label GLUE RTE core experiment.
