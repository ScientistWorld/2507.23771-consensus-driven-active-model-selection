# Train/Test Evaluation Design Notes

Current design (hardening iteration; details and evidence in HARDENING.md).

- **Protocol.** Selectors are commands registered in `outputs/selectors/<name>.json`. The evaluators run each command once per episode and act as the label oracle over stdin/stdout. In every round the selector commits a model before it receives the label it queried. The evaluator records those selections, counts revealed labels and computes all metrics itself. The client side is `data/selection_protocol.py`.
- **Slices.**
  - RTE, MRPC and CoLA are held out whole: the full pools follow the paper's protocol, nothing about them is visible, and they also test generalization to unseen tasks.
  - SST2 and QNLI (a fixed 1200-item sample) are split into balanced halves: the visible half is `data/benchmarks/<task>_train.json`, scored by `eval/train/evaluate.py` through `scripts/evaluate_train.sh` and `scripts/evaluate.sh`.
  - Every held-out pool (`eval/test/benchmarks/<task>_test.json`) has keyed labels and shuffled candidate and item order, and is scored by `eval/test/evaluate.py` through `scripts/evaluate_test.sh`.
  - The two evaluators share no code. Both check data checksums.
- **Experiments.** Active selection with a 100-label budget on GLUE RTE, SST2, MRPC, QNLI and CoLA, plus zero-label selection on RTE and SST2. Seeds 0–4 throughout.
- **Failures.** Protocol violations, crashes and timeouts (30 min per episode) score as worst-candidate-every-round with the full budget spent.
- **Provenance.** `eval/test/build_benchmarks.py` regenerates both halves from the public ModelSelector arrays. It is hidden, because it reconstructs the held-out labels.
