# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the NVLink AllReduce backend.

Run under ``torchrun --nproc-per-node=<n> tests/test_correctness.py``.

For every combination of (dtype, size, algorithm) we

1.  Seed a deterministic per-rank tensor.
2.  Run the NVLink all-reduce.
3.  Compute the reference sum locally by all-gathering the inputs via the
    same TCPStore-backed mechanism the backend uses (no NCCL here).
4.  Assert max-abs-error is within a dtype-appropriate tolerance.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys

import torch
import torch.distributed as dist

# Add parent directory to path so we can import the local package without
# requiring it to be installed system-wide during quick iteration.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import nvlink_allreduce as nar  # noqa: E402


def _all_gather_tensor_via_store(t: torch.Tensor, store, rank: int,
                                 world_size: int, key: str) -> torch.Tensor:
    """Gather ``t`` to every rank via TCPStore (NCCL-free).

    We host-stage the tensor as a uint8 byte view for the exchange so that
    dtypes that numpy can't natively represent (e.g. bf16) still go through
    cleanly.  This path is only used in tests to compute the reference sum.
    """
    cpu = t.detach().cpu().contiguous()
    payload = bytes(cpu.view(torch.uint8).numpy())
    keys = [f"{key}:{r}" for r in range(world_size)]
    store.set(keys[rank], payload)
    store.wait(keys, datetime.timedelta(seconds=600))
    out = torch.zeros(world_size, *t.shape, dtype=t.dtype)
    for r in range(world_size):
        buf = bytes(store.get(keys[r]))
        flat_u8 = torch.frombuffer(bytearray(buf), dtype=torch.uint8)
        out[r] = flat_u8.view(t.dtype).view(*t.shape)
    return out


def _tol(dtype: torch.dtype, world_size: int) -> float:
    base = {
        torch.float32: 1e-5,
        torch.float16: 5e-3,
        torch.bfloat16: 1e-2,
    }[dtype]
    # Reduction over ``world_size`` summands accumulates ~sqrt(N) error.
    return base * (world_size ** 0.5)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-elements", type=int, default=4 * 1024 * 1024,
                        help="largest tensor to test (default 4M elements)")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    pg = nar.init_process_group()
    # Also keep a handle to the store for gathering reference data.
    store = pg._store  # noqa: SLF001

    sizes = [
        64,                  # < 1 KiB
        4096,                # 16 KiB (fp32)
        64 * 1024,           # 256 KiB
        512 * 1024,          # 2 MiB  (crosses small/large threshold)
        2 * 1024 * 1024,     # 8 MiB
        args.max_elements,
    ]
    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    algos = [None, "1stage", "2stage"]

    failures = 0
    total = 0

    for algo in algos:
        if algo is None:
            os.environ.pop("NVLINK_AR_ALGO", None)
            label = "auto"
        else:
            os.environ["NVLINK_AR_ALGO"] = algo
            label = algo
        for dtype in dtypes:
            for n in sizes:
                total += 1
                # Determinism: same seed everywhere; per-rank offset added.
                torch.manual_seed(0)
                t = (torch.randn(n, dtype=torch.float32, device=device)
                     + 0.1 * (rank + 1)).to(dtype)
                ref_local = t.detach().clone()

                # Reference: sum the inputs from all ranks (CPU side).
                gathered = _all_gather_tensor_via_store(
                    ref_local, store, rank, world_size,
                    key=f"ref::{algo}::{dtype}::{n}")
                expected = gathered.sum(dim=0).to(dtype).to(device)

                nar.all_reduce(t)
                torch.cuda.synchronize(device)

                err = (t.float() - expected.float()).abs().max().item()
                tol = _tol(dtype, world_size)
                ok = err <= tol
                if rank == 0:
                    flag = "OK " if ok else "FAIL"
                    print(f"[{flag}] algo={label:6s} dtype={str(dtype):17s} "
                          f"n={n:>8d}  max_abs_err={err:.3e} (tol={tol:.1e})",
                          flush=True)
                if not ok:
                    failures += 1

    nar.destroy_process_group()
    if rank == 0:
        print(f"\nResults: {total - failures}/{total} configurations passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
