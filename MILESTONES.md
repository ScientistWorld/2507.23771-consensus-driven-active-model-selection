# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-02 01:48] - core_claim_plus
- Implemented and ran CODA's consensus priors, Beta-mixture P(best), expected-information-gain acquisition, and Dirichlet updates.
- Reproduced the core GLUE RTE active model selection result at a 100-label budget: CODA cumulative regret 290.04 versus random 420.51 and uncertainty 371.84.
- Added GLUE SST2 as a second setting: CODA cumulative regret 51.72 versus random 345.47 and uncertainty 100.92.
- Packaged reusable method, baseline, download, reproduction, and evaluation scripts; validation passes with `python validate.py --compare`.

## Stop Justification
- Completed at milestone `core_claim_plus`.
