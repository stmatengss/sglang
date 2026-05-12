#!/usr/bin/env bash
# Run the correctness tests under torchrun.
#
# Usage:  source /opt/sft.sh && bash scripts/run_correctness.sh [WORLD_SIZE]
#
# Default WORLD_SIZE = 8 (full H200 island).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

WS="${1:-8}"
PORT="${MASTER_PORT:-29501}"
# Honour in-place build if pip-editable wasn't used.
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "[correctness] world_size=$WS port=$PORT"
torchrun --nproc-per-node="$WS" --master-port="$PORT" \
    tests/test_correctness.py
