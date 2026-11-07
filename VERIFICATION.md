# Verification Report

## Initial Assessment

The workspace started this final pass at `secondary_claims`. The claimed milestone is justified: the active model selection method has corrected scored results on RTE, SST2, and MRPC, and the zero-label secondary setting is scored on RTE and SST2. The latest evaluation-only job completed successfully and `python validate.py --compare` passed with main, train-slice, and test-slice scores.

## Issues Found

- minor: `action.yaml` still requested a CPU test submission from the previous iteration even though that job had completed successfully.
- minor: `VERIFICATION.md` still read like an iteration log instead of a final handoff judgment.

No critical issues were found in the final pass. The method-agnostic files remain clean, `eval/` does not import from `method/`, committed code does not reference `shared/`, scoring contains no wall-clock or paper-deviation metrics, and the scored experiments match the documented targets.

## Fixes Applied

- Rechecked the self-contained download path. The Google Drive CODA archive endpoint and all ModelSelector raw GitHub URLs used by the data preparation script returned HTTP 200 after redirects.
- Re-ran `python validate.py --compare`; validation passed.
- Rewrote this report as a final verification record.
- Added a final milestone log entry and changed `action.yaml` to `done`.

## Final Judgment

The workspace is ready to serve as a research gym. Improve-mode agents can build from the reproduced method and traces, while from-scratch agents can use the problem definition, data loaders, evaluation scripts, scoring rubric, constraints, and research direction without seeing method details. The remaining limitation is documented and does not invalidate the milestone: MRPC has one corrected CODA seed in main scoring, while the auxiliary train/test split is limited to the more stable RTE and SST2 active-selection experiments.

- Milestone: secondary_claims
- Ready for gym use: yes
- Confidence: high
- Key limitation: MRPC active selection uses one corrected CODA seed and is excluded from auxiliary train/test scoring.

## Hardening addendum (iteration 11)

The evaluation design described above was replaced; see HARDENING.md and DESIGN_NOTES.md. `python validate.py` and `python validate.py --post-design` both pass on the final five-experiment set.
