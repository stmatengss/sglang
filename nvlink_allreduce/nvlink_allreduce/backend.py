# SPDX-License-Identifier: Apache-2.0
"""PyTorch-side glue for the NVLink AllReduce backend.

Architecture
------------
* **Control plane (no NCCL)**: a ``torch.distributed.TCPStore`` is used as a
  rendezvous mechanism to exchange ``cudaIpcMemHandle_t`` byte strings and to
  synchronise initialisation.  TCPStore is a pure-CPU TCP server distributed
  with PyTorch and is independent of NCCL.
* **Data plane (NVLink only)**: every rank owns a ``Signal`` region and a
  large *staging buffer*; their IPC handles are exchanged once at init.  An
  ``all_reduce(tensor)`` copies ``tensor`` into the local staging buffer, the
  kernel does ``ngpus``-way reduce-scatter / all-gather directly over
  NVLink, and the result is copied back.
* **Sub-groups**: a process group is built with ``init_process_group``; the
  module also exposes a class :class:`NVLinkProcessGroup` that may be
  instantiated multiple times for non-default groups.

The staging buffer pre-allocates ``max_buf_bytes`` (default 512 MiB) so all
calls to ``all_reduce`` are zero-allocation in steady state.  Calls with
tensors larger than ``max_buf_bytes`` are chunked transparently.
"""

from __future__ import annotations

import datetime
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.distributed as dist

from . import _C  # type: ignore[attr-defined]

log = logging.getLogger("nvlink_allreduce")

_DEFAULT_MAX_BUF_BYTES = 512 * 1024 * 1024  # 512 MiB per rank
_RANK_DATA_SLOTS = 8                        # we only register one buffer
# Per-rank slack on the two-shot tmp region.  The kernel's `largest_part`
# can be up to `world_size - 1` packed elements past `part`; 4 KiB is plenty.
_TMP_SLACK_BYTES = 4096

_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

# Map torch.dtype -> the dtype-name string accepted by _C.tensor_from_ptr.
_DTYPE_NAME = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}


@dataclass
class _State:
    pg: "NVLinkProcessGroup"


_GLOBAL: Optional[_State] = None


def _all_gather_bytes_via_store(store: dist.Store, key_prefix: str,
                                rank: int, world_size: int,
                                payload: bytes) -> List[bytes]:
    """All-gather byte payloads via a TCPStore.

    Each rank writes ``<key_prefix>:<rank>`` and then reads every other key.
    """
    store.set(f"{key_prefix}:{rank}", payload)
    out: List[bytes] = []
    keys = [f"{key_prefix}:{r}" for r in range(world_size)]
    store.wait(keys, datetime.timedelta(seconds=600))
    for r in range(world_size):
        out.append(store.get(keys[r]))
    return out


class NVLinkProcessGroup:
    """Process group offering NCCL-free SUM all-reduce over NVLink.

    Parameters
    ----------
    rank, world_size : int
        Standard rank info.
    store : torch.distributed.Store
        Pre-built rendezvous store (e.g. TCPStore).  Used only at init time.
    device : torch.device, optional
        CUDA device to bind to.  Defaults to ``cuda:rank``.
    max_buf_bytes : int, optional
        Size of the pre-allocated staging buffer per rank.
    full_nvlink : bool, optional
        Set to False to fall back to one-shot only (e.g. PCIe).
    name : str
        Used to namespace TCPStore keys so multiple groups can coexist.
    """

    def __init__(self, rank: int, world_size: int, store: dist.Store,
                 device: Optional[torch.device] = None,
                 max_buf_bytes: int = _DEFAULT_MAX_BUF_BYTES,
                 full_nvlink: bool = True, name: str = "default"):
        if world_size not in (2, 4, 6, 8):
            raise ValueError(
                f"NVLinkProcessGroup supports world_size in (2,4,6,8); got {world_size}"
            )
        self.rank = rank
        self.world_size = world_size
        self.device = device or torch.device(f"cuda:{rank}")
        self.max_buf_bytes = max_buf_bytes
        self.name = name
        self._store = store
        torch.cuda.set_device(self.device)

        # 1. Enable NVLink P2P from this device to every other device on the
        #    same node.  cudaDeviceCanAccessPeer returns False when there is
        #    no NVLink/PCIe P2P path — in that case we cannot proceed.
        dev_idx = self.device.index
        peer_devs = [d for d in range(world_size) if d != dev_idx]
        for peer in peer_devs:
            ok = _C.enable_p2p_access(dev_idx, peer)
            if not ok:
                raise RuntimeError(
                    f"GPU {dev_idx} cannot peer-access GPU {peer}. "
                    "NVLink AllReduce requires a fully P2P-capable topology."
                )

        # 2. Allocate the Signal region.  The kernel stores barrier counters
        #    in the first sizeof(Signal) bytes; the rest is a two-shot
        #    reduce-scatter staging area.  Per-rank slot must fit
        #    ``max_buf_bytes / world_size`` (the worst case slice size).
        meta = _C.meta_size()
        self._tmp_bytes = (max_buf_bytes // world_size) + _TMP_SLACK_BYTES
        self._signal_bytes = meta + self._tmp_bytes
        self._signal_ptr = _C.cuda_malloc(self._signal_bytes)
        self._signal_handle = _C.get_ipc_mem_handle(self._signal_ptr)

        # 3. Allocate the data staging buffer.
        self._buf_ptr = _C.cuda_malloc(self.max_buf_bytes)
        self._buf_handle = _C.get_ipc_mem_handle(self._buf_ptr)

        # 4. Allocate the per-rank scratch region for kernel-arg RankData.
        #    A RankData is 8 ptrs * 8 bytes = 64 bytes; we register only one
        #    buffer so a small region with 256B alignment is plenty.
        self._rank_data_bytes = max(_RANK_DATA_SLOTS * 64, 256)
        self._rank_data_ptr = _C.cuda_malloc(self._rank_data_bytes)

        # 5. Exchange IPC handles via TCPStore.
        sig_handles = _all_gather_bytes_via_store(
            store, f"nvlink_ar::{name}::sig", rank, world_size,
            self._signal_handle)
        buf_handles = _all_gather_bytes_via_store(
            store, f"nvlink_ar::{name}::buf", rank, world_size,
            self._buf_handle)

        # 6. Build the C++ handle (opens peer Signal regions).
        self._handle = _C.init_handle(
            sig_handles, self._rank_data_ptr, self._rank_data_bytes,
            self._signal_ptr, rank, world_size, full_nvlink)

        # 7. Open the peer staging-buffer handles and register the buffer.
        peer_buf_ptrs: List[int] = []
        for r in range(world_size):
            if r == rank:
                peer_buf_ptrs.append(self._buf_ptr)
            else:
                peer_buf_ptrs.append(_C.open_ipc_handle(buf_handles[r]))
        self._peer_buf_ptrs = peer_buf_ptrs
        _C.register_buffer(self._handle, peer_buf_ptrs)

        # 8. Synchronise — make sure every rank finished registration before
        #    any rank attempts an allreduce.
        self._store_barrier(f"nvlink_ar::{name}::init_done")

        log.info(
            "[rank %d/%d] NVLinkProcessGroup '%s' ready on %s "
            "(staging=%.1f MiB, signal=%.1f MiB)",
            rank, world_size, name, self.device,
            self.max_buf_bytes / 2**20, self._signal_bytes / 2**20,
        )

    # ---------- collective API ---------- #

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """In-place SUM all-reduce.  Returns the tensor for chaining."""
        if not tensor.is_cuda:
            raise ValueError("tensor must reside on CUDA")
        if tensor.device != self.device:
            raise ValueError(
                f"tensor device {tensor.device} != backend device {self.device}"
            )
        if tensor.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"unsupported dtype: {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError("tensor must be contiguous")

        itemsize = tensor.element_size()
        max_elems = self.max_buf_bytes // itemsize
        # 16-byte vector alignment requirement of the kernel:
        vec_elems = 16 // itemsize
        # Stay a multiple of vec_elems.
        max_elems = (max_elems // vec_elems) * vec_elems
        flat = tensor.view(-1)
        n = flat.numel()
        # Buffer view tensor; reused per chunk.
        for start in range(0, n, max_elems):
            chunk_n = min(max_elems, n - start)
            if chunk_n % vec_elems != 0:
                # Pad to vec_elems by reducing one element at a time isn't
                # great, so we fall back to a small staging copy with pad.
                self._reduce_unaligned(flat, start, chunk_n)
            else:
                self._reduce_chunk(flat, start, chunk_n)
        return tensor

    def barrier(self) -> None:
        """Cheap barrier: do a 1-element all-reduce."""
        t = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.all_reduce(t)
        torch.cuda.synchronize(self.device)

    # ---------- internals ---------- #

    def _buf_view(self, dtype: torch.dtype, numel: int) -> torch.Tensor:
        return _C.tensor_from_ptr(
            self._buf_ptr, [numel], _DTYPE_NAME[dtype], self.device.index)

    def _reduce_chunk(self, flat: torch.Tensor, start: int, n: int) -> None:
        buf = self._buf_view(flat.dtype, n)
        src = flat.narrow(0, start, n)
        buf.copy_(src, non_blocking=True)
        _C.all_reduce(self._handle, buf, buf)
        src.copy_(buf, non_blocking=True)

    def _reduce_unaligned(self, flat: torch.Tensor, start: int, n: int) -> None:
        """Slow path for the tail when n is not a multiple of 16 bytes."""
        itemsize = flat.element_size()
        vec_elems = 16 // itemsize
        pad_n = ((n + vec_elems - 1) // vec_elems) * vec_elems
        buf = self._buf_view(flat.dtype, pad_n)
        buf.zero_()
        src = flat.narrow(0, start, n)
        buf.narrow(0, 0, n).copy_(src, non_blocking=True)
        _C.all_reduce(self._handle, buf, buf)
        src.copy_(buf.narrow(0, 0, n), non_blocking=True)

    def _store_barrier(self, key: str) -> None:
        # Each rank announces "done", every rank waits for everyone.
        keys = [f"{key}:{r}" for r in range(self.world_size)]
        self._store.set(keys[self.rank], b"1")
        self._store.wait(keys, datetime.timedelta(seconds=600))

    # ---------- lifecycle ---------- #

    def destroy(self) -> None:
        if getattr(self, "_handle", None) is not None:
            _C.dispose(self._handle)
            self._handle = None
        for attr in ("_buf_ptr", "_signal_ptr", "_rank_data_ptr"):
            ptr = getattr(self, attr, 0)
            if ptr:
                _C.cuda_free(ptr)
                setattr(self, attr, 0)

    def __del__(self):  # pragma: no cover - best-effort
        try:
            self.destroy()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# torch.distributed-style functional API.
# ---------------------------------------------------------------------------

def init_process_group(rank: Optional[int] = None,
                       world_size: Optional[int] = None,
                       master_addr: Optional[str] = None,
                       master_port: Optional[int] = None,
                       device: Optional[torch.device] = None,
                       max_buf_bytes: int = _DEFAULT_MAX_BUF_BYTES,
                       full_nvlink: bool = True) -> NVLinkProcessGroup:
    """Initialise the global NVLink process group.

    Reads ``RANK``/``WORLD_SIZE``/``MASTER_ADDR``/``MASTER_PORT`` from the
    environment if the corresponding arguments are not provided, so the
    function can be used under ``torchrun`` unchanged.
    """
    global _GLOBAL
    if _GLOBAL is not None:
        raise RuntimeError("NVLink process group already initialised")

    rank = int(rank if rank is not None else os.environ["RANK"])
    world_size = int(world_size if world_size is not None else os.environ["WORLD_SIZE"])
    master_addr = master_addr or os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = int(master_port if master_port is not None else os.environ.get("MASTER_PORT", "29500"))

    store = dist.TCPStore(master_addr, master_port,
                          world_size, rank == 0,
                          timeout=datetime.timedelta(seconds=900))

    pg = NVLinkProcessGroup(rank, world_size, store, device=device,
                            max_buf_bytes=max_buf_bytes, full_nvlink=full_nvlink)
    _GLOBAL = _State(pg=pg)
    return pg


def is_initialized() -> bool:
    return _GLOBAL is not None


def get_rank() -> int:
    if _GLOBAL is None:
        raise RuntimeError("process group not initialised")
    return _GLOBAL.pg.rank


def get_world_size() -> int:
    if _GLOBAL is None:
        raise RuntimeError("process group not initialised")
    return _GLOBAL.pg.world_size


def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """In-place SUM all-reduce on the global NVLink process group."""
    if _GLOBAL is None:
        raise RuntimeError("call init_process_group() first")
    return _GLOBAL.pg.all_reduce(tensor)


def barrier() -> None:
    if _GLOBAL is None:
        raise RuntimeError("call init_process_group() first")
    _GLOBAL.pg.barrier()


def destroy_process_group() -> None:
    global _GLOBAL
    if _GLOBAL is None:
        return
    _GLOBAL.pg.destroy()
    _GLOBAL = None
