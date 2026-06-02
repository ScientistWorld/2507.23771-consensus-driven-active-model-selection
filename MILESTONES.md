# Reproduction Milestones

**Current: secondary_claims**

## Progress Log

### [2026-06-02 01:59] - secondary_claims
- Active-selection core claim reproduced on GLUE RTE and GLUE SST2.
- Zero-label unsupervised selection claim reproduced on GLUE RTE and GLUE SST2.
- Prepared a follow-up GLUE MRPC active-selection run to broaden the benchmark.

### [2026-06-02 02:01] - secondary_claims
- Continuing-agent audit confirmed the implementation is CODA's consensus-driven active model selection algorithm rather than a surrogate.
- `python validate.py --compare` passes for the completed RTE/SST2 active and zero-label experiments, and eval/data do not import from `method/`.
- `scoring/TARGETS.md` includes GLUE MRPC active selection, but `scoring/scores.json` does not yet include it, so the MRPC CPU job is being submitted instead of marking the workspace done.

### [2026-06-02 03:04] - secondary_claims
- MRPC CPU job timed out after 3600 seconds, but completed random and uncertainty baselines plus CODA seeds 0, 1, and 2.
- Added resume-safe seed skipping to the method runner and updated `scripts/run.sh` to finish missing CODA seeds 3 and 4.
- Resubmitting a CPU continuation job to merge the completed five-seed MRPC result into `scoring/scores.json` and validate.

### [2026-06-02 03:40] - secondary_claims
- MRPC continuation completed CODA seeds 3 and 4 and produced a five-seed MRPC active-selection result.
- Fixed the evaluator to emit only metrics declared for each experiment in `scoring/reference.json`, and removed its unnecessary NumPy dependency.
- `python validate.py --compare` now passes with RTE, SST2, MRPC active selection plus RTE/SST2 zero-label selection.
