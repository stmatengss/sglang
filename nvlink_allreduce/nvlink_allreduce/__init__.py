# SPDX-License-Identifier: Apache-2.0
"""NVLink AllReduce — NCCL-free PyTorch all-reduce backend.

Public API mirrors ``torch.distributed`` so user code can swap NCCL out:

    import nvlink_allreduce as nar

    nar.init_process_group(rank, world_size, master_addr, master_port)
    nar.all_reduce(tensor)        # in-place SUM, like dist.all_reduce
    nar.destroy_process_group()

Internally we exchange CUDA IPC handles over a ``torch.distributed.TCPStore``
and use a custom CUDA kernel (see ``csrc/``) to reduce buffers directly over
NVLink P2P.  No NCCL is linked or invoked.
"""

from .backend import (
    NVLinkProcessGroup,
    all_reduce,
    barrier,
    destroy_process_group,
    get_rank,
    get_world_size,
    init_process_group,
    is_initialized,
)

__version__ = "0.1.0"

__all__ = [
    "NVLinkProcessGroup",
    "all_reduce",
    "barrier",
    "destroy_process_group",
    "get_rank",
    "get_world_size",
    "init_process_group",
    "is_initialized",
]
