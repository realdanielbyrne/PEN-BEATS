#!/bin/bash
# Dual-GPU experiment runner for ae_dataset_comparison
# Runs WikiText-2 config on GPU 0 and FineWeb config on GPU 1 in parallel

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "Starting dual-GPU experiment run"
echo "=========================================="
echo ""
echo "GPU 0: WikiText-2 config (60 runs)"
echo "GPU 1: FineWeb config (60 runs)"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Launch WikiText-2 config on GPU 0 in background
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting WikiText-2 config on GPU 0..."
PYTORCH_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python scripts/run_from_yaml.py \
    scripts/experiments/ae_dataset_comparison.yaml \
    --config wikitext2 \
    --gpu-id 0 \
  > scripts/experiments/results/ae_dataset_comparison/logs/wikitext2_gpu0.log 2>&1 &
PID_GPU0=$!

# Launch FineWeb config on GPU 1 in background
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting FineWeb config on GPU 1..."
PYTORCH_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python scripts/run_from_yaml.py \
    scripts/experiments/ae_dataset_comparison.yaml \
    --config fineweb \
    --gpu-id 1 \
  > scripts/experiments/results/ae_dataset_comparison/logs/fineweb_gpu1.log 2>&1 &
PID_GPU1=$!

echo ""
echo "Both experiments launched:"
echo "  WikiText-2 (GPU 0): PID $PID_GPU0"
echo "  FineWeb (GPU 1):    PID $PID_GPU1"
echo ""
echo "Monitor progress with:"
echo "  tail -f scripts/experiments/results/ae_dataset_comparison/logs/wikitext2_gpu0.log"
echo "  tail -f scripts/experiments/results/ae_dataset_comparison/logs/fineweb_gpu1.log"
echo ""

# Wait for both to complete
wait $PID_GPU0
STATUS_GPU0=$?

wait $PID_GPU1
STATUS_GPU1=$?

echo ""
echo "=========================================="
echo "Experiment run complete"
echo "=========================================="
echo "WikiText-2 (GPU 0): $([ $STATUS_GPU0 -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo "FineWeb (GPU 1):    $([ $STATUS_GPU1 -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo ""

exit $(( $STATUS_GPU0 + $STATUS_GPU1 ))

