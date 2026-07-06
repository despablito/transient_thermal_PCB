#!/bin/zsh
# Full reproduction pipeline: data gen -> training -> honest benchmarks -> plots
set -e
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=./venv/bin/python

echo "=== [1/6] Data generation (1000 FDM sequences) — start $(date '+%H:%M:%S') ==="
$PY data_gen.py

# Keep only the full cumulative file — train.py loads every checkpoint_*.npz
# into RAM before picking the largest, and the partial ones waste ~4 GB.
rm -f checkpoint_0*.npz
ls -lh checkpoint_1000.npz transient_data.npz

echo "=== [2/6] Training 30 epochs, seeded split — start $(date '+%H:%M:%S') ==="
$PY train.py --fresh

echo "=== [3/6] Hold-out benchmark (fresh FDM sequences, full horizon) ==="
$PY benchmark_eval.py --fresh-data --n 50 --out benchmark_holdout_results.json

echo "=== [4/6] Split-based benchmark (now valid: seeded split) ==="
$PY benchmark_eval.py --n 50 --out benchmark_split_results.json

echo "=== [5/6] Physics validation suite ==="
$PY validate_physics.py

echo "=== [6/6] Diagnostic visualization ==="
$PY test_vis.py --checkpoint checkpoints/best_model.pth

echo "=== PIPELINE DONE $(date '+%H:%M:%S') ==="
