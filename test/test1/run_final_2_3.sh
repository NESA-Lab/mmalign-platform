#!/usr/bin/env bash
set -euo pipefail
MODEL="${MODEL:?set MODEL}"; EVALUATOR="${EVALUATOR:?set EVALUATOR}"; CONCURRENCY="${CONCURRENCY:-2}"
for scene in biochem child code health travel; do
  python3 scripts/evaluate_metric_2_3.py --model "$MODEL" --scenario "$scene" --origin user --sample-limit 300 --concurrency "$CONCURRENCY" --evaluator "$EVALUATOR" --run-id "final-2-3-${scene}-$(date +%Y%m%dT%H%M%S)" || test "$?" -eq 2
done
