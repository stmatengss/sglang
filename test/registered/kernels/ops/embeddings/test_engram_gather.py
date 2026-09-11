"""Triton engram gather reads device tables and pinned host tables the same way."""

import unittest

import torch

from sglang.kernels.ops.embeddings.engram_gather import engram_gather
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")

ROWS = 128
DIM = 256
BLK = 32


def _table(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    weight = (torch.randn(ROWS, DIM, generator=g) * 3).to(torch.float8_e4m3fn)
    scale = torch.randint(
        100, 140, (ROWS, DIM // BLK), dtype=torch.uint8, generator=g
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


class TestEngramGather(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("needs CUDA")

    def test_device_and_host_gather_match_reference(self):
        weight, scale = _table()
        ids = torch.tensor(
            [0, 1, ROWS - 1, 7, 0], device="cuda", dtype=torch.int64
        )
        expected = _reference(weight, scale, ids)

        out_dev = torch.empty(ids.numel(), DIM, dtype=torch.bfloat16, device="cuda")
        engram_gather(
            weight.cuda().data_ptr(),
            scale.cuda().data_ptr(),
            ids,
            out_dev,
            DIM,
            BLK,
        )
        torch.testing.assert_close(out_dev.cpu(), expected, atol=0, rtol=0)

        w_host = torch.empty_like(weight, pin_memory=True)
        s_host = torch.empty_like(scale, pin_memory=True)
        w_host.copy_(weight)
        s_host.copy_(scale)
        out_host = torch.empty(ids.numel(), DIM, dtype=torch.bfloat16, device="cuda")
        engram_gather(
            w_host.data_ptr(),
            s_host.data_ptr(),
            ids,
            out_host,
            DIM,
            BLK,
        )
        torch.testing.assert_close(out_host.cpu(), expected, atol=0, rtol=0)
        self.assertEqual(out_host[0, 0].item(), 2**-127)

    def test_unowned_rows_are_zero(self):
        weight, scale = _table()
        ids = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int64)
        out = torch.empty(ids.numel(), DIM, dtype=torch.bfloat16, device="cuda")
        engram_gather(
            weight.cuda().data_ptr(),
            scale.cuda().data_ptr(),
            ids,
            out,
            DIM,
            BLK,
            row_lo=1,
            row_hi=2,
        )
        self.assertTrue(torch.equal(out[0], torch.zeros_like(out[0])))
        self.assertFalse(torch.equal(out[1], torch.zeros_like(out[1])))
        self.assertTrue(torch.equal(out[2], torch.zeros_like(out[2])))


if __name__ == "__main__":
    unittest.main()
