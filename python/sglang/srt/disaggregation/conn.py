from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import numpy as np
import numpy.typing as npt
# import mooncake_vllm_adaptor as mva

logger = logging.getLogger(__name__)


class KVArgs:
    engine_rank: int
    kv_data_ptrs: list[int]
    kv_data_lens: list[int]
    kv_item_lens: list[int]
    aux_data_ptrs: list[int]
    aux_data_lens: list[int]
    aux_item_lens: list[int]
    ib_device: str


class KVManager:
    def __init__(self, args: KVArgs):
        # self.engine = mva.mooncake_vllm_adaptor()
        # self.local_rank = local_rank

        # self.config = MooncakeTransferEngineConfig.load_from_env()
        # self.engine.initialize(local_hostname, metadata_server, protocol, device_name)
        pass


class KVPoll:
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4


class KVSender:
    def __init__(self, mgr: KVManager, bootstrap_addr: str, bootstrap_room: int):
        self.has_sent = False

    def init(self, num_kv_indices: int, aux_index: Optional[int] = None): ...

    def send(self, kv_indices: npt.NDArray[np.int32]):
        # data = [kv_data_ptr[i] for i in kv_indices]
        # buffer_src = allocateManagedBuffer(len)
        # writeBytestoBuffer(buffer_src, data)
        # transferSync(buffer_src, ???) # TODO Maybe it should be postBatchSend
        # zmq.send(finished)
        self.has_sent = True

    def poll(self) -> KVPoll:
        # pollCq?
        if self.has_sent is False:
            # Assume handshake completed instantly
            return KVPoll.WaitingForInput
        else:
            # Assume transfer completed instantly
            return KVPoll.Success

    def failure_exception(self):
        raise Exception("Fake KVSender Exception")


class KVReceiver:
    def __init__(
        self, mgr: KVManager, bootstrap_addr: str, bootstrap_room: Optional[int] = None
    ):
        self.has_init = False

    def init(self, kv_indices: npt.NDArray[np.int32], aux_index: Optional[int] = None):
        # if (zmq.recv() is finished):
        #     data = [kv_data_ptr[i] for i in kv_indices]
        #     readBytesFromBuffer(buffer_dst, data)
        #     self.has_init = True
        self.has_init = True

    def poll(self) -> KVPoll:
        if self.has_init is False:
            # Assume handshake completed instantly
            return KVPoll.WaitingForInput
        else:
            # Assume transfer completed instantly
            return KVPoll.Success

    def failure_exception(self):
        raise Exception("Fake KVReceiver Exception")


class KVBootstrapServer:
    def __init__(self, port: int): ...

    def poll(self) -> KVPoll: ...
