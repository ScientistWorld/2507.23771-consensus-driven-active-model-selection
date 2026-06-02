# Reproduction Milestones

**Current: core_claim**

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

### [2026-06-02 00:35] - core_claim
- Reran the 100-label, 5-seed GLUE RTE experiment after the EIG fix.
- Reproduced the core active model selection claim: CODA reached cumulative regret 290.04, close to the paper's 283.8 and better than reproduced random sampling (420.51) and uncertainty sampling (371.84).
- Validation with `python validate.py --compare` passed, including scoring schema and method/evaluation separation checks.

## Stop Justification
- Not stopped. A CPU job is being submitted for an additional GLUE SST2 setting to attempt `core_claim_plus`.
