#!/usr/bin/env bash
set -euo pipefail
# Run from repository root. API credentials must be supplied via environment.
MODEL="${MODEL:?Set MODEL to a configured model ID}"
EVALUATOR="${EVALUATOR:?Set EVALUATOR to a configured evaluator ID}"
mkdir -p data/benchmarks/user
for scene in code health travel; do
  target="data/benchmarks/user/specbench-${scene}-safe"
  if [ -e "$target" ]; then
    echo "Using existing scenario: $target (verify its version before running)"
  else
    cp -R "test/test1/benchmarks/specbench-${scene}-safe" "$target"
  fi
  python3 test/test1/evaluate_metric_2_3.py --model "$MODEL"     --scenario "specbench-${scene}-safe" --origin user --sample-limit 150     --concurrency "${CONCURRENCY:-2}" --evaluator "$EVALUATOR"     --run-id "plan1-${scene}-$(date +%Y%m%dT%H%M%S)" || {
      code=$?; [ "$code" -eq 2 ] || exit "$code"
    }
done
