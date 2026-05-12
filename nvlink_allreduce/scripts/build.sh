#!/usr/bin/env bash
# Build the NVLink AllReduce PyTorch extension.
#
# Usage:  source /opt/sft.sh && bash scripts/build.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc not found. Did you run 'source /opt/sft.sh' first?" >&2
    exit 1
fi
if ! python -c "import torch" 2>/dev/null; then
    echo "ERROR: torch not importable. Did you run 'source /opt/sft.sh' first?" >&2
    exit 1
fi

# Default to H100/H200 (sm_90) if not set.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

echo "Building nvlink_allreduce (TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST) ..."
# Primary path: editable install.  Falls back to in-place build if pip can't
# do an editable install in the active environment (e.g. PEP 668 managed
# envs without --break-system-packages).
if ! python -m pip install --no-build-isolation -e . ; then
    echo "(falling back to in-place build_ext)"
    python setup.py build_ext --inplace
    # Make the package importable from the source tree.
    export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
    echo "  PYTHONPATH=$PYTHONPATH"
fi

echo
echo "Build complete. Quick sanity check:"
PYTHONPATH="$ROOT:${PYTHONPATH:-}" python -c "import nvlink_allreduce as nar; print('  meta_size =', nar._C.meta_size(), 'bytes')"
