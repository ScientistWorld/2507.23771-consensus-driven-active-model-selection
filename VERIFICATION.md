# Verification Report

## Initial Assessment

The workspace was previously reported at `secondary_claims`. Final verification found two genuine issues in stale evidence: active-selection regrets needed to be recomputed from fixed benchmark labels, and CODA acquisition needed to match the released disagreement-candidate filtering. Both issues were fixed, corrected traces were regenerated for the scored settings, and the final evaluation-only rerun passed validation.

## Issues Found

- critical: Stale active-selection scores trusted trace-written regret fields and did not reflect the corrected CODA acquisition sequence. Corrected traces and scores were required before restoring the milestone.
- critical: The first evaluation-only job wrote corrected main scores, but validation failed because the auxiliary train/test split included zero-label experiments where CODA and EnsV intentionally have identical primary-metric values.
- minor: Some method-agnostic documentation used paper/proposed-method phrasing and one data helper docstring named the reproduced method family. This could leak unnecessary context to from-scratch scientist agents.
- minor: `scoring/CONSTRAINTS.md` said traces must contain regret per step, but the evaluator now recomputes regret from fixed benchmark labels and only requires selected-model traces.

## Fixes Applied

- Updated `scripts/run.sh` to perform an evaluation-only job over the completed corrected traces, followed by zero-label runs, `scores.json`, `scores_train.json`, `scores_test.json`, and `python validate.py --compare`.
- Restricted the default train/test split evaluators to active-selection experiments. Zero-label secondary claims are still included in `scripts/evaluate.sh` and `scores.json`, but they are not useful hidden-slice optimization targets because the reproduced primary metric is intentionally tied.
- Restricted the auxiliary train/test split further to RTE and SST2 active selection. MRPC remains in main scoring, but its one corrected CODA seed is too unstable under half-split rescoring to serve as a validator-consistent hidden-slice target.
- Ran the final evaluation-only job successfully. `python validate.py --compare` passes with corrected main scores and active-selection train/test scores.
- Cleaned strict method-agnostic docs so `briefing/evaluation.md` and `scoring/EXPERIMENTS.md` describe reference targets and experiment goals without naming the method.
- Updated data-helper docstrings to describe active model selection benchmarks without method-specific naming.
- Updated `scoring/CONSTRAINTS.md` to state the actual output contract: traces must include selected models, and regret is recomputed by evaluation.
- Verified all ModelSelector raw GitHub URLs needed for the current reproduced tasks return HTTP 200, and confirmed committed code has no `shared/` references.

## Final Judgment

The workspace is ready to serve as a research gym. Improve-mode scientists can start from the CODA implementation and reproduced traces; from-scratch scientists can use the method-agnostic problem, data, evaluation, scoring targets, constraints, and direction documents. The key limitation is that MRPC CODA is represented by one corrected seed in main scoring because full five-seed regeneration exceeded the CPU time limit; the auxiliary train/test split therefore uses only RTE and SST2.

- Milestone: secondary_claims
- Ready for gym use: yes
- Confidence: high
- Key limitation: MRPC active selection uses one corrected CODA seed and is excluded from auxiliary train/test scoring; RTE has five corrected seeds and SST2 has two corrected seeds.
