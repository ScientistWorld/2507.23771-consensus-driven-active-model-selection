#!/bin/bash
set -euo pipefail
cd /home/user
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python redteam/check_numerics.py
for name in reference_original reference_seed0 reference_seed1 reference_seed2 \
            honest_eta_0p1 honest_eta_1 weaker_no_learning baseline_random \
            visible_label_oracle_shortcut train_complement_oracle_shortcut \
            trace_budget_shortcut spoof_scores_shortcut groundtruth_relabel_shortcut \
            replace_fixed_predictions_shortcut task_size_inflation_shortcut \
            backdate_selection_shortcut constant_zero_shortcut empty_trace_shortcut \
            null_model_shortcut nan_model_shortcut negative_model_shortcut negative_budget_shortcut; do
    if [ -e "redteam/$name/result.json" ]; then
        echo "Completed already: $name"
        continue
    fi
    echo "Running audit case: $name"
    bash "redteam/$name/run.sh"
done
