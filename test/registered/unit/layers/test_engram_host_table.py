"""Host-backed DeepSeek-V4.1 Engram tables stay on CPU and match the checkpoint."""

import os
import socket
import unittest
from unittest.mock import patch

import torch

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.environ import envs
from sglang.srt.layers.engram import _HostTable, engram_host_table_enabled
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

ROWS = 2048
DIM = 256
BLK = 32


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _init() -> None:
    port = _free_port()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="gloo",
    )
    initialize_model_parallel(tensor_model_parallel_size=1, backend="gloo")


def _teardown() -> None:
    destroy_model_parallel()
    destroy_distributed_environment()


def _checkpoint_table(seed: int = 0, rows: int = ROWS):
    g = torch.Generator().manual_seed(seed)
    weight = (torch.randn(rows, DIM, generator=g) * 3).to(torch.float8_e4m3fn)
    scale = torch.randint(
        100, 140, (rows, DIM // BLK), dtype=torch.uint8, generator=g
    ).view(torch.float8_e8m0fnu)
    weight.view(torch.uint8)[0, :BLK] = (
        torch.tensor(1.0).to(torch.float8_e4m3fn).view(torch.uint8)
    )
    scale.view(torch.uint8)[0, 0] = 0
    return weight, scale


def _reference(weight, scale, ids):
    ids = ids.cpu()
    rows = weight[ids].float().unflatten(-1, (-1, BLK))
    return (rows * scale[ids].float().unsqueeze(-1)).flatten(-2).to(torch.bfloat16)


def _build(rows: int, layout: str):
    from sglang.srt.layers.engram import EngramEmbedding

    with (
        envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.override(True),
        envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.override(layout),
        envs.SGLANG_ENABLE_DSV41_ENGRAM_DROP_PAGE_CACHE.override(False),
        torch.device("cpu"),
    ):
        return EngramEmbedding(rows, DIM, layer_id=1)


class TestEngramHostTable(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("needs CUDA")
        torch.cuda.set_device(0)
        _init()
        self.addCleanup(_teardown)

    def _run_layout(self, layout: str) -> None:
        weight, scale = _checkpoint_table()
        with (
            get_parallel().override(tp_size=1, tp_rank=0),
            patch("sglang.srt.layers.engram.get_attention_dp_size", return_value=1),
        ):
            embed = _build(ROWS, layout)
            embed.weight.weight_loader(embed.weight, weight)
            embed.scale.weight_loader(embed.scale, scale)
            embed.finish_load()
            self.assertEqual(embed.weight.device.type, "cpu")
            self.assertEqual(embed.scale.device.type, "cpu")
            embed.cuda()
            self.assertEqual(embed.weight.device.type, "cpu")
            ids = torch.arange(ROWS, device="cuda", dtype=torch.int64).view(-1, 1)
            out = embed(ids)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out.cpu(), _reference(weight, scale, ids)))
        self.assertEqual(out[0, 0, 0].item(), 2**-127)

    def test_private_host_lookup_stays_on_cpu(self):
        self._run_layout("private")

    def test_shared_host_lookup_stays_on_cpu(self):
        self._run_layout("shared")


class TestEngramHostTableLayout(CustomTestCase):
    def test_choose_layout_rejects_unknown(self):
        with self.assertRaisesRegex(ValueError, "must be auto, shared, or private"):
            _HostTable.choose_layout("hbm")

    def test_choose_layout_passthrough(self):
        self.assertEqual(_HostTable.choose_layout("shared"), "shared")
        self.assertEqual(_HostTable.choose_layout("private"), "private")

    def test_enabled_follows_env_override(self):
        with envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.override(False):
            self.assertFalse(engram_host_table_enabled())
        with envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.override(True):
            self.assertTrue(engram_host_table_enabled())


if __name__ == "__main__":
    unittest.main()
