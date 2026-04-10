#!/usr/bin/env bash
# scripts/quick_test.sh
# ----------------------
# Smoke-test: 2 rounds, breast_cancer only, TabICLv2 only, no ablations.
# Runs in ~2 minutes on a T4 GPU.

set -euo pipefail

echo "=== D-ICL Quick Smoke Test ==="
echo "Rounds  : 2"
echo "Dataset : breast_cancer"
echo "Backbone: tabicl"
echo ""

python main.py \
  --clf-datasets breast_cancer \
  --reg-datasets diabetes_reg \
  --backbones tabicl \
  --rounds 2 \
  --no-ablations \
  --output ./dicl_smoke_test.json

echo ""
echo "✅ Smoke test passed."
echo "   Results : ./dicl_smoke_test.json"
echo "   Figures : ./figures/"
