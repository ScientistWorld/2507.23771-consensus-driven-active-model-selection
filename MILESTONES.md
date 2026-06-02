# Reproduction Milestones

**Current: secondary_claims**

## Progress Log

### [2026-06-02 01:48] - core_claim_plus
- Implemented and ran CODA's consensus priors, Beta-mixture P(best), expected-information-gain acquisition, and Dirichlet updates.
- Reproduced the core GLUE RTE active model selection result at a 100-label budget: CODA cumulative regret 290.04 versus random 420.51 and uncertainty 371.84.
- Added GLUE SST2 as a second setting: CODA cumulative regret 51.72 versus random 345.47 and uncertainty 100.92.
- Packaged reusable method, baseline, download, reproduction, and evaluation scripts; validation passes with `python validate.py --compare`.

### [2026-06-02 02:16] - secondary_claims
- Added the paper's zero-label unsupervised selection setting for GLUE RTE and SST2.
- Reproduced Table 5-scale initial regret: RTE CODA 14.801 and EnsV 14.801 versus paper 14.8; SST2 CODA 3.555 and EnsV 3.555 versus paper 3.6.
- Verified the final score surface with `python validate.py --compare` after job `f500fff1-d02`.

## Stop Justification
- Completed at milestone `secondary_claims`.
