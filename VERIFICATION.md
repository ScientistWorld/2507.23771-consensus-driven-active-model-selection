# Verification Report

## Initial Assessment

The workspace was previously reported at `secondary_claims`, and the earlier results did reproduce active selection on GLUE RTE/SST2 plus zero-label selection. During final verification, a genuine issue was already identified and partly fixed before this report: active-selection regrets are now recomputed from fixed benchmark labels, and CODA acquisition was aligned with the released disagreement-candidate filtering. Those fixes invalidate stale CODA traces for the active RTE/SST2 experiments.

Because the corrected traces have not yet been regenerated, the currently supported milestone is temporarily `method_runs`, not `secondary_claims`.

## Issues Found

- critical: `scoring/scores.json` currently lacks corrected `coda` reproduced rows for `glue_rte_active_selection` and `glue_sst2_active_selection`. The file still validates structurally, but it is incomplete evidence for the paper's active-selection core claim after the evaluator/method correction.
- minor: Some method-agnostic documentation used paper/proposed-method phrasing and one data helper docstring named the reproduced method family. This could leak unnecessary context to from-scratch scientist agents.
- minor: `scoring/CONSTRAINTS.md` said traces must contain regret per step, but the evaluator now recomputes regret from fixed benchmark labels and only requires selected-model traces.

## Fixes Applied

- Kept `scripts/run.sh` and `action.yaml` set to submit a CPU rerun that regenerates CODA traces for GLUE RTE, GLUE SST2, and GLUE MRPC, reruns zero-label experiments, writes `scores.json`, `scores_train.json`, and `scores_test.json`, then runs `python validate.py --compare`.
- Cleaned strict method-agnostic docs so `briefing/evaluation.md` and `scoring/EXPERIMENTS.md` describe reference targets and experiment goals without naming the method.
- Updated data-helper docstrings to describe active model selection benchmarks without method-specific naming.
- Updated `scoring/CONSTRAINTS.md` to state the actual output contract: traces must include selected models, and regret is recomputed by evaluation.
- Verified all ModelSelector raw GitHub URLs needed for the current reproduced tasks return HTTP 200, and confirmed committed code has no `shared/` references.

## Final Judgment

The workspace is not ready to finalize until the corrective CPU rerun completes and restores corrected CODA rows for the active-selection experiments. The environment structure, scoring design, data separation, and evaluator independence are now in good shape; the blocker is regenerated evidence, not architecture.

- Milestone: method_runs
- Ready for gym use: partial
- Confidence: medium
- Key limitation: corrected active-selection CODA scores are pending regeneration.
