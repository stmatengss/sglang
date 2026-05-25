# SPDX-License-Identifier: Apache-2.0
"""Build script for the NVLink AllReduce PyTorch extension.

This builds a single CUDA extension module ``nvlink_allreduce._C`` that
exposes the kernels and IPC machinery declared in ``csrc/``.  No NCCL
linkage is required — we only link against the standard CUDA driver/runtime.

Tested with PyTorch 2.1+, CUDA 12.x, GCC 11+.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent.resolve()
CSRC = ROOT / "csrc"


def _detect_arch_flags() -> list[str]:
    """Return the right ``-gencode`` flags for the host's CUDA arches.

    On an H200 (sm_90a) we want to compile sm_90 specifically.  Allow the
    user to override via TORCH_CUDA_ARCH_LIST.
    """
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return []  # Torch will read the env var itself.
    arches: set[str] = set()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            arches.add(f"{major}{minor}")
    if not arches:
        # Sensible default for modern data-center GPUs: H100/H200/B200.
        arches = {"90"}
    flags = []
    for a in sorted(arches):
        flags += [
            "-gencode",
            f"arch=compute_{a},code=sm_{a}",
        ]
    return flags


nvcc_flags = [
    "-O3",
    "-std=c++17",
    "--use_fast_math",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--threads",
    "4",
] + _detect_arch_flags()

cxx_flags = ["-O3", "-std=c++17", "-Wno-unused-function", "-Wno-unused-variable"]


ext = CUDAExtension(
    name="nvlink_allreduce._C",
    sources=[str(CSRC / "nvlink_allreduce.cu")],
    include_dirs=[str(CSRC)],
    extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
)


setup(
    name="nvlink_allreduce",
    version="0.1.0",
    description=(
        "NCCL-free PyTorch all-reduce backend using NVLink P2P + CUDA IPC."
    ),
    author="cursor agent",
    packages=find_packages(exclude=("tests*",)),
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
    install_requires=["torch>=2.1"],
    zip_safe=False,
)
