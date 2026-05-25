# SPDX-License-Identifier: Apache-2.0
"""Performance benchmark for the NVLink AllReduce backend.

Measures per-call latency (us) and effective algorithmic bandwidth (GB/s)
for a sweep of payload sizes.  Output is a Markdown table on rank 0 plus a
JSON dump under ``--output-dir`` for archival / plotting.

The "algorithmic bandwidth" of an all-reduce of ``B`` bytes across ``N``
GPUs is defined (following the NCCL conventions) as
``B * 2 * (N - 1) / (N * latency)`` for the ring algorithm.  We report the
same metric so numbers can be compared apples-to-apples with NCCL.

Usage
-----
torchrun --nproc-per-node=8 tests/bench_allreduce.py \
    --warmup 10 --iters 100 --output-dir results/
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import nvlink_allreduce as nar  # noqa: E402


# Payload sweep — covers latency-bound (small) and bandwidth-bound (large)
# regimes that map directly onto the one-shot / two-shot regions of the
# kernel selector.
DEFAULT_SIZES_BYTES = [
    4 * 1024,                  # 4 KiB
    16 * 1024,                 # 16 KiB
    64 * 1024,                 # 64 KiB
    256 * 1024,                # 256 KiB
    1 * 1024 * 1024,           # 1 MiB
    4 * 1024 * 1024,           # 4 MiB
    16 * 1024 * 1024,          # 16 MiB
    64 * 1024 * 1024,          # 64 MiB
    128 * 1024 * 1024,         # 128 MiB
    256 * 1024 * 1024,         # 256 MiB
]


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:>7.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


def bench_one(pg: nar.NVLinkProcessGroup, n_bytes: int, dtype: torch.dtype,
              warmup: int, iters: int, algo: str | None) -> dict:
    if algo:
        os.environ["NVLINK_AR_ALGO"] = algo
    else:
        os.environ.pop("NVLINK_AR_ALGO", None)

    itemsize = torch.empty(1, dtype=dtype).element_size()
    vec_elems = 16 // itemsize
    n = (n_bytes // itemsize // vec_elems) * vec_elems
    if n == 0:
        return {"skipped": True}

    device = pg.device
    t = torch.randn(n, dtype=torch.float32, device=device).to(dtype)

    # Warm-up.
    for _ in range(warmup):
        pg.all_reduce(t)
    torch.cuda.synchronize(device)
    # Cross-rank barrier so all ranks start timing together.
    pg.barrier()

    # CUDA events for in-stream timing.
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(iters):
        pg.all_reduce(t)
    end_evt.record()
    end_evt.synchronize()
    total_ms = start_evt.elapsed_time(end_evt)
    avg_us = total_ms * 1000.0 / iters

    # NCCL-style algorithmic bandwidth.
    N = pg.world_size
    payload = n * itemsize
    algo_bw = payload * 2 * (N - 1) / N / (avg_us * 1e-6) / 1e9  # GB/s
    bus_bw = algo_bw  # same definition for ring-equivalent collectives

    return {
        "skipped": False,
        "dtype": str(dtype),
        "n_bytes": payload,
        "iters": iters,
        "avg_us": avg_us,
        "algo_bw_GBps": algo_bw,
        "bus_bw_GBps": bus_bw,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    ap.add_argument("--algo", choices=["auto", "1stage", "2stage"], default="auto")
    ap.add_argument("--sizes", nargs="*", type=int, default=None,
                    help="Override the size sweep (in bytes).")
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--max-buf-bytes", type=int, default=512 * 1024 * 1024)
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    pg = nar.init_process_group(max_buf_bytes=args.max_buf_bytes)

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    sizes = args.sizes if args.sizes else DEFAULT_SIZES_BYTES
    algo = None if args.algo == "auto" else args.algo

    rows = []
    for n_bytes in sizes:
        if n_bytes > args.max_buf_bytes:
            if rank == 0:
                print(f"# skipping {fmt_bytes(n_bytes)} (> max_buf_bytes)")
            continue
        r = bench_one(pg, n_bytes, dtype, args.warmup, args.iters, algo)
        rows.append(r)

    if rank == 0:
        print()
        print(f"# NVLink AllReduce — world_size={world_size}, dtype={args.dtype}, "
              f"algo={args.algo}, iters={args.iters}, warmup={args.warmup}")
        print()
        print("| size       | latency (us) | alg. BW (GB/s) | bus BW (GB/s) |")
        print("|------------|--------------|----------------|---------------|")
        for r in rows:
            if r.get("skipped"):
                continue
            print(f"| {fmt_bytes(r['n_bytes']):>10s} "
                  f"| {r['avg_us']:>12.2f} "
                  f"| {r['algo_bw_GBps']:>14.2f} "
                  f"| {r['bus_bw_GBps']:>13.2f} |")
        print()

        if args.output_dir:
            outdir = Path(args.output_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            payload = {
                "world_size": world_size,
                "dtype": args.dtype,
                "algo": args.algo,
                "iters": args.iters,
                "warmup": args.warmup,
                "rows": rows,
            }
            fn = outdir / f"bench_ws{world_size}_{args.dtype}_{args.algo}.json"
            with open(fn, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"# Saved JSON to {fn}")

    nar.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
