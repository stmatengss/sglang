"""GPU E2E coverage for the direct Mooncake radix-cache backend."""

import os

from test_hicache_storage_mooncake_backend import (
    HiCacheStorageMooncakeBackendBaseMixin,
)

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import DEFAULT_MLA_MODEL_NAME_FOR_TEST, CustomTestCase

register_cuda_ci(est_time=300, stage="base-b", runner_config="2-gpu-large")


class MooncakeRadixCacheE2EMixin(HiCacheStorageMooncakeBackendBaseMixin):
    @classmethod
    def _get_base_server_args(cls):
        args = {
            "--tp-size": 2,
            "--radix-cache-backend": "mooncake",
            "--page-size": 64,
            "--mem-fraction-static": 0.6,
            "--enable-cache-report": True,
            "--enable-deterministic-inference": True,
        }
        if backend := os.getenv("SGLANG_MOONCAKE_RADIX_ATTENTION_BACKEND"):
            args["--attention-backend"] = backend
        return args

    @classmethod
    def _get_additional_server_args_and_env(cls):
        return {}, {
            "MOONCAKE_MASTER": f"127.0.0.1:{cls.mooncake_master_port}",
            "MOONCAKE_PROTOCOL": "tcp",
            "MC_MS_AUTO_DISC": "0",
            "MOONCAKE_DEVICE": "",
            "MOONCAKE_TE_META_DATA_SERVER": (
                f"http://127.0.0.1:{cls.mooncake_metadata_port}/metadata"
            ),
            "MOONCAKE_GLOBAL_SEGMENT_SIZE": "268435456",
            "SGLANG_ENABLE_DETERMINISTIC_INFERENCE": "1",
        }

    @staticmethod
    def _response_text(response):
        return response.get("text", response.get("output", ""))

    def _assert_remote_round_trip(self, prompt, min_cached_tokens):
        first = self.send_request(prompt, max_tokens=32)
        # flush_cache waits for the radix backend's async stores, clears GPU
        # radix state, and intentionally leaves Mooncake objects intact.
        self.flush_cache()
        second = self.send_request(prompt, max_tokens=32)
        self.assertGreaterEqual(self.get_cached_tokens(second), min_cached_tokens)
        self.assertTrue(self._response_text(first).strip())
        self.assertEqual(self._response_text(second), self._response_text(first))
        return first

    def test_chat_input_output_via_direct_l2_mooncake(self):
        context = "You are a helpful assistant. " * 100
        prompt = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": context},
                {
                    "role": "user",
                    "content": "Reply with exactly this word: ORANGE",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        first = self._assert_remote_round_trip(prompt, 64)
        self.assertIn("ORANGE", self._response_text(first).upper())

    def test_direct_l2_mooncake_write_and_loadback(self):
        prompt_a = self.gen_prompt(768)
        self._assert_remote_round_trip(prompt_a, 640)


class TestMooncakeRadixCacheMHA(MooncakeRadixCacheE2EMixin, CustomTestCase):
    @classmethod
    def _get_model_name(cls):
        return (
            os.getenv("SGLANG_MOONCAKE_RADIX_MHA_MODEL")
            or super()._get_model_name()
        )


class TestMooncakeRadixCacheMLA(MooncakeRadixCacheE2EMixin, CustomTestCase):
    @classmethod
    def _get_model_name(cls):
        return os.getenv(
            "SGLANG_MOONCAKE_RADIX_MLA_MODEL", DEFAULT_MLA_MODEL_NAME_FOR_TEST
        )
