#!/usr/bin/env bash
# scripts/quick_test.sh
# ----------------------
# Smoke-test: 2 rounds, breast_cancer only, TabICLv2 only, no ablations.


set -euo pipefail

echo "=== D-ICL Quick Smoke Test ==="
echo "Rounds  : 2"
echo "Dataset : vehicle"
echo "Backbone: tabicl"
echo ""

python main.py \
  --clf-datasets vehicle \
  --reg-datasets bike \
  --backbones tabicl \
  --rounds 2 \
  --no-ablations \
  --output ./dicl_smoke_test.json

echo ""
echo "✅ Smoke test passed."
echo "   Results : ./dicl_smoke_test.json"
echo "   Figures : ./figures/"
