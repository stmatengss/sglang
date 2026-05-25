# SPDX-License-Identifier: Apache-2.0
"""Side-by-side benchmark of our NVLink backend vs. PyTorch's NCCL backend.

This script is provided **only** to validate the performance of our custom
backend.  It is NOT part of the project runtime — the user's constraint is
that the project architecture must not depend on NCCL, which our backend
honours (see ``nvlink_allreduce/backend.py``).  Here we briefly import the
``nccl`` backend just to print reference numbers for the perf report.

Usage
-----
torchrun --nproc-per-node=8 tests/bench_compare.py
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import nvlink_allreduce as nar  # noqa: E402

DEFAULT_SIZES_BYTES = [
    4 * 1024,
    64 * 1024,
    256 * 1024,
    1 * 1024 * 1024,
    4 * 1024 * 1024,
    16 * 1024 * 1024,
    64 * 1024 * 1024,
    128 * 1024 * 1024,
    256 * 1024 * 1024,
]


def fmt_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    f = float(n)
    for u in units:
        if f < 1024:
            return f"{f:>7.2f} {u}"
        f /= 1024
    return f"{f:.2f} TiB"


def time_op(op, t, warmup, iters, device):
    for _ in range(warmup):
        op(t)
    torch.cuda.synchronize(device)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        op(t)
    e.record()
    e.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters  # us


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--sizes", nargs="*", type=int, default=None)
    ap.add_argument("--skip-nccl", action="store_true")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    sizes = args.sizes if args.sizes else DEFAULT_SIZES_BYTES
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    itemsize = torch.empty(1, dtype=dtype).element_size()
    vec_elems = 16 // itemsize

    # ------------------------------------------------------------------ #
    # 1. Our backend.
    # ------------------------------------------------------------------ #
    pg = nar.init_process_group(max_buf_bytes=512 * 1024 * 1024)
    nvlink_rows = []
    for nb in sizes:
        n = (nb // itemsize // vec_elems) * vec_elems
        if n == 0:
            continue
        t = torch.randn(n, dtype=torch.float32, device=device).to(dtype)
        us = time_op(lambda x: nar.all_reduce(x), t, args.warmup, args.iters, device)
        algo_bw = (n * itemsize) * 2 * (world_size - 1) / world_size / (us * 1e-6) / 1e9
        nvlink_rows.append((nb, us, algo_bw))
    nar.destroy_process_group()

    # ------------------------------------------------------------------ #
    # 2. NCCL reference (optional; only used for the report).
    # ------------------------------------------------------------------ #
    nccl_rows = []
    nccl_available = (not args.skip_nccl) and dist.is_nccl_available()
    if nccl_available:
        # Use a different port so we don't collide with our store.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = str(int(os.environ.get("MASTER_PORT", "29500")) + 1)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        for nb in sizes:
            n = (nb // itemsize // vec_elems) * vec_elems
            if n == 0:
                continue
            t = torch.randn(n, dtype=torch.float32, device=device).to(dtype)
            us = time_op(lambda x: dist.all_reduce(x, op=dist.ReduceOp.SUM),
                         t, args.warmup, args.iters, device)
            algo_bw = (n * itemsize) * 2 * (world_size - 1) / world_size / (us * 1e-6) / 1e9
            nccl_rows.append((nb, us, algo_bw))
        dist.destroy_process_group()

    if rank == 0:
        print()
        print(f"# AllReduce comparison: NVLink-AR (ours) vs. NCCL (reference)")
        print(f"# world_size={world_size}  dtype={args.dtype}  iters={args.iters}")
        print()
        if nccl_available:
            print("| size       | ours (us) | ours (GB/s) | nccl (us) | nccl (GB/s) | speedup |")
            print("|------------|-----------|-------------|-----------|-------------|---------|")
            for (nb, ours_us, ours_bw), (_, nccl_us, nccl_bw) in zip(nvlink_rows, nccl_rows):
                speedup = nccl_us / ours_us if ours_us > 0 else float("inf")
                print(f"| {fmt_bytes(nb):>10s} "
                      f"| {ours_us:>9.2f} | {ours_bw:>11.2f} "
                      f"| {nccl_us:>9.2f} | {nccl_bw:>11.2f} "
                      f"| {speedup:>6.2f}x |")
        else:
            print("# (NCCL unavailable — printing our results only)")
            print("| size       | ours (us) | ours (GB/s) |")
            print("|------------|-----------|-------------|")
            for nb, us, bw in nvlink_rows:
                print(f"| {fmt_bytes(nb):>10s} | {us:>9.2f} | {bw:>11.2f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
