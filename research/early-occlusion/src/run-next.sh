#!/usr/bin/env bash
# Run from any directory. Outputs and downloads stay in ignored work/ by default.
set -euo pipefail
experiment_root="$(cd "$(dirname "$0")/.." && pwd)"
repository_root="$(cd "$experiment_root/../.." && pwd)"
experiment_python="${EXPERIMENT_PYTHON:-$repository_root/.venv/bin/python}"
experiment_work="${EXPERIMENT_WORK:-$repository_root/work/early-occlusion-reproduction}"
mkdir -p "$experiment_work/reports" "$experiment_work/logs"
candidate="$experiment_root/checkpoints/v21-presence.pt"
if [[ "${1:-}" == "--retrain" ]]; then
  "$experiment_python" "$experiment_root/src/audit-fragments.py" \
    --output "$experiment_work/reports/assignment.json" --features "$experiment_work/features.pt" \
    > "$experiment_work/logs/assignment.log" 2>&1
  "$experiment_python" "$experiment_root/src/train-presence-head.py" \
    --features "$experiment_work/features.pt" --target visible --regularization .01 \
    --output "$experiment_work/v20-presence.pt" > "$experiment_work/logs/train-v20.log" 2>&1
  "$experiment_python" "$experiment_root/src/compare-presence.py" --candidate "$experiment_work/v20-presence.pt" \
    --benchmark previous --cache "$experiment_work/v20-development.pt" \
    --output "$experiment_work/reports/v20-development.json" > "$experiment_work/logs/v20-development.log" 2>&1
  candidate="$experiment_work/v21-presence.pt"
  "$experiment_python" "$experiment_root/src/train-presence-head.py" \
    --features "$experiment_work/features.pt" --target tal --regularization .1 \
    --output "$candidate" > "$experiment_work/logs/train-v21.log" 2>&1
elif [[ $# != 0 ]]; then
  echo "Usage: bash src/run-next.sh [--retrain]" >&2
  exit 2
fi
"$experiment_python" "$experiment_root/src/verify.py" > "$experiment_work/logs/verify.log" 2>&1
"$experiment_python" "$experiment_root/src/replay.py" --checkpoint v18 --motion --device cpu \
  --output "$experiment_work/reports/v18-replay-cpu.json" > "$experiment_work/logs/replay.log" 2>&1
"$experiment_python" "$experiment_root/src/verify.py" --replay "$experiment_work/reports/v18-replay-cpu.json" \
  >> "$experiment_work/logs/verify.log" 2>&1
"$experiment_python" "$experiment_root/src/compare-presence.py" --candidate "$candidate" --benchmark previous \
  --cache "$experiment_work/development.pt" --output "$experiment_work/reports/development.json" \
  > "$experiment_work/logs/development.log" 2>&1
"$experiment_python" "$experiment_root/src/collect-new-matches.py" \
  --manifest "$experiment_root/data/next/sources.json" --destination "$experiment_work/media" \
  > "$experiment_work/logs/download.log" 2>&1
comparison=("$experiment_python" "$experiment_root/src/compare-presence.py" --candidate "$candidate" \
  --benchmark new --annotations "$experiment_root/data/next/benchmark.json" \
  --sources "$experiment_root/data/next/sources.json" --media "$experiment_work/media" \
  --cache "$experiment_work/predictions.pt" --calibration-report "$experiment_work/reports/development.json")
"${comparison[@]}" --output "$experiment_work/reports/new20-comparison.json" \
  > "$experiment_work/logs/comparison.log" 2>&1
for iou in .2 .5; do
  "${comparison[@]}" --iou "$iou" --output "$experiment_work/reports/sensitivity-iou${iou}.json" \
    > "$experiment_work/logs/sensitivity-iou${iou}.log" 2>&1
done
"${comparison[@]}" --gt-padding 2 --output "$experiment_work/reports/sensitivity-pad2.json" \
  > "$experiment_work/logs/sensitivity-pad2.log" 2>&1
"$experiment_python" "$experiment_root/src/benchmark-presence.py" --candidate "$candidate" \
  --media "$experiment_work/media" --calibration-report "$experiment_work/reports/development.json" \
  --output "$experiment_work/reports/latency-cpu.json" > "$experiment_work/logs/latency.log" 2>&1
"$experiment_python" "$experiment_root/src/audit-first-pixels.py" --candidate "$candidate" \
  --media "$experiment_work/media" --output "$experiment_work/reports/raw-firsts-unfiltered.json" \
  > "$experiment_work/logs/raw-firsts.log" 2>&1
"$experiment_python" "$experiment_root/src/analyze-presence.py" --cache "$experiment_work/predictions.pt" \
  --report "$experiment_work/reports/new20-comparison.json" --output "$experiment_work/reports/new20-analysis.json"
"$experiment_python" "$experiment_root/src/render-next.py" --media "$experiment_work/media" \
  --audit "$experiment_work/reports/raw-firsts-unfiltered.json" --output "$experiment_work/reports/first-fragments.png"
"$experiment_python" "$experiment_root/src/test-presence.py" > "$experiment_work/logs/tests.log" 2>&1
echo "Reports: $experiment_work/reports"
