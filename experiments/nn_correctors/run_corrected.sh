#!/usr/bin/env bash
# Regenerate the corrected Claim-3 (nn_correctors) artifacts on the rig.
# Runs the full pipeline in the pinned `simon` env: rebuild datasets with the corrected A2
# sign + meta, retrain A1/A2, then re-evaluate and regenerate results + plots.
#
# Usage (from anywhere):   bash experiments/nn_correctors/run_corrected.sh
# Override the interpreter: PY=/path/to/python bash experiments/nn_correctors/run_corrected.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$HOME/miniforge3/envs/simon/bin/python}"
LOG="$HERE/run_corrected.log"

echo "[run] python: $PY"
"$PY" -c "import numpy, torch, rebound; print('[run] env OK — torch', torch.__version__, '| rebound', rebound.__version__)"

{
  echo "==================== $(date) ===================="
  echo "### STAGE 1/3  gen_data.py  (rebuild datasets.npz + eval_set.json + gen_meta.json) ###"
  "$PY" "$HERE/gen_data.py"
  echo "### STAGE 2/3  train.py     (retrain A1 + A2 -> a1_weights.npz / a2_weights.npz) ###"
  "$PY" "$HERE/train.py"
  echo "### STAGE 3/3  evaluate.py  (regenerate results.json / results_table.txt / *.png) ###"
  "$PY" "$HERE/evaluate.py"
  echo "### DONE — verdict line: ###"
  grep -E "VERDICT" "$HERE/results_table.txt" || true
} 2>&1 | tee "$LOG"

echo
echo "[run] complete. Review: $HERE/results_table.txt  (and gen_meta.json for config/skip counts)"
echo "[run] then update the Finding-3 table in guide/DATA_GUIDE.md and the outline from results_table.txt."
