from __future__ import annotations

import logging
import json
import os
import zmq
from enum import Enum
from typing import Optional

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
# import mooncake_vllm_adaptor as mva

logger = logging.getLogger(__name__)

# 1. The sender and receiver are created when the request arrives
# 2. They handshake and bootstrap
# 3. The bootstrap room is the unique identifier paired with the request. The bootstrap server assigns a queue pair based on the unique identifier.
# 4. We host the bootstrap server on prefill's tokenizer manager

class KVArgs:
    engine_rank: int
    kv_data_ptrs: list[int]
    kv_data_lens: list[int]
    kv_item_lens: list[int]
    aux_data_ptrs: list[int]
    aux_data_lens: list[int]
    aux_item_lens: list[int]
    ib_device: str

@dataclass
class MooncakeTransferEngineConfig:
    localhost_name: str
    metadata_backend: Union[str, None]
    metadata_server: str
    protocol: str
    device_name: str

    @staticmethod
    def from_file(file_path: str) -> 'MooncakeTransferEngineConfig':
        """Load the config from a JSON file."""
        with open(file_path) as fin:
            config = json.load(fin)
        return MooncakeTransferEngineConfig(
            localhost_name=config.get("localhost_name", None),
            metadata_backend=config.get("metadata_backend", None),
            metadata_server=config.get("metadata_server"),
            protocol=config.get("protocol", "rdma"),
            device_name=config.get("device_name", ""),
        )

    @staticmethod
    def load_from_env() -> 'MooncakeTransferEngineConfig':
        """Load config from a file specified in the environment variable."""
        config_file_path = os.getenv('MOONCAKE_CONFIG_PATH')
        if config_file_path is None:
            raise ValueError(
                "The environment variable 'MOONCAKE_CONFIG_PATH' is not set.")
        return MooncakeTransferEngineConfig.from_file(config_file_path)

class KVManager:

    pub_socket: zmq.Context.socket
    sub_socket: zmq.Context.socket

    def __init__(self, args: KVArgs):
        self.args = args

        try:
            import mooncake_vllm_adaptor as mva
        except ImportError as e:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "  # noqa: E501
                "to run vLLM with MooncakeConnector.") from e
        
        self.engine = mva.mooncake_vllm_adaptor()
        try:
            self.config = MooncakeTransferEngineConfig.load_from_env()
            logger.info("Mooncake Configuration loaded successfully.")
        except ValueError as e:
            logger.error(e)
            raise
        except Exception as exc:
            logger.error(
                "An error occurred while loading the configuration: %s", exc)
            raise

        self.config = MooncakeTransferEngineConfig.load_from_env()
        self.initialize(self.config.localhost_name,
                        self.config.metadata_server, self.config.protocol,
                        self.config.device_name, self.config.metadata_backend)
    
    def prefill_setup_metadata_sockets(self) -> None:
        self.context = zmq.Context()
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://*:{7788 + self.args.engine_rank}")
    
    def pub_data(self, bootstrap_room: int, data: str) -> None:
        self.pub_socket.send_string(f"{bootstrap_room} {data}")
    
    def decode_setup_metadata_sockets(self, bootstrap_room: int) -> None:
        self.context = zmq.Context()
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://*:{7788 + self.args.engine_rank}")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "{bootstrap_room}")
    
    def sub_data(self) -> str:
        message = str("")
        while True:
            full_msg = self.sub_socket.recv_string()
            _, message = full_msg.split(' ', 1)
        return message

    def initialize(self, local_hostname: str, metadata_server: str,
                   protocol: str, device_name: str,
                   metadata_backend: Union[str, None]) -> None:
        """Initialize the mooncake instance."""
        if metadata_backend is None:
            self.engine.initialize(local_hostname, metadata_server, protocol,
                                   device_name)
        else:
            supported_backend = ["etcd", "redis"]
            metadata_backend = metadata_backend.lower()
            if metadata_backend not in supported_backend:
                raise ValueError(
                    "Mooncake Configuration error. `metadata_backend`"
                    f" should be one of {supported_backend}.")

            self.engine.initializeExt(local_hostname, metadata_server,
                                      protocol, device_name, metadata_backend)

    def register(self):
        for kv_data_ptr, kv_data_len in zip(self.args.kv_data_ptrs, self.args.kv_data_lens):
            self.engine.expRegisterMemory(kv_data_ptr, kv_data_len)
    
    def deregister(self):
        for kv_data_ptr in self.args.kv_data_ptrs:
            self.engine.expUnregisterMemory(kv_data_ptr)
    
    def close(self) -> None:
        """Cleanup logic when closing the pipe."""
        if self.pub_socket:
            self.pub_socket.close()
        if self.sub_socket:
            self.sub_socket.close()

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
        # mgr.register
        # for kv_indices in kv_indices:
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
        self.mgr = mgr
        self.has_init = False

    def init(self, kv_indices: npt.NDArray[np.int32], aux_index: Optional[int] = None):
        # if (zmq.recv() is finished):
        #     data = [kv_data_ptr[i] for i in kv_indices]
        #     readBytesFromBuffer(buffer_dst, data)
        # send_args, send_kv_indices = zmq.recv()
        # for send_index, index in zip(send_kv_indices, kv_indices):
        #     self.engine.transfer(
        #         arg.kv_data_ptrs[index]+, send_arg.kv_data_ptrs[send_index], arg.kv_length_ptrs[index])
        # self.mgr.decode_setup_metadata_sockets()
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
