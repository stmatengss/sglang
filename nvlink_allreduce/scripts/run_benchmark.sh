#!/usr/bin/env bash
# Run the all-reduce benchmark.
#
# Usage:  source /opt/sft.sh && bash scripts/run_benchmark.sh [WORLD_SIZE]
#
# Sweeps:  world sizes (2/4/8), dtypes (bf16/fp16/fp32), algos (auto/1stage/2stage)
# Writes:  results/bench_ws<N>_<dtype>_<algo>.json + Markdown to stdout.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

WS="${1:-8}"
PORT_BASE="${MASTER_PORT:-29600}"
OUT="${OUT_DIR:-$ROOT/results}"
mkdir -p "$OUT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

i=0
for DTYPE in bf16 fp16 fp32; do
  for ALGO in auto 1stage 2stage; do
    PORT=$((PORT_BASE + i))
    i=$((i + 1))
    echo "[bench] world_size=$WS dtype=$DTYPE algo=$ALGO port=$PORT"
    torchrun --nproc-per-node="$WS" --master-port="$PORT" \
        tests/bench_allreduce.py \
        --dtype "$DTYPE" --algo "$ALGO" \
        --warmup 20 --iters 200 \
        --output-dir "$OUT" || true
    echo
  done
done
echo "[bench] all runs complete; JSON in $OUT"
