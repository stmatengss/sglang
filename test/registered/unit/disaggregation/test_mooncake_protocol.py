"""Unit tests for Mooncake PD transfer protocol options."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.arg_groups.pd_disaggregation_hook import handle_pd_disaggregation
from sglang.srt.disaggregation.mooncake.protocol import (
    MOONCAKE_TRANSFER_PROTOCOL_CHOICES,
    apply_mooncake_protocol,
    is_mooncake_backend,
    parse_mooncake_backend_alias,
    resolve_path_transport_hint,
    validate_mooncake_protocol,
)
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
    _call_with_transport_hint,
)
from sglang.srt.environ import envs
from sglang.srt.server_args import DISAGG_TRANSFER_BACKEND_CHOICES, ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_OWNED_FORCE_ENV = (
    "MC_FORCE_TCP",
    "MC_FORCE_MNNVL",
    "MC_INTRANODE_NVLINK",
    "MC_FORCE_MUSA",
)


class TestMooncakeProtocolCatalog(CustomTestCase):
    def test_catalog_covers_mooncake_transports(self):
        for protocol in (
            "rdma",
            "tcp",
            "efa",
            "nvlink",
            "nvlink_intra",
            "hip",
            "barex",
            "ascend",
            "musa",
            "cxl",
            "nvmeof",
            "tpu",
            "mpcomm",
        ):
            self.assertIn(protocol, MOONCAKE_TRANSFER_PROTOCOL_CHOICES)

    def test_backend_aliases_are_registered(self):
        self.assertIn("mooncake", DISAGG_TRANSFER_BACKEND_CHOICES)
        for protocol in MOONCAKE_TRANSFER_PROTOCOL_CHOICES:
            self.assertIn(f"mooncake_{protocol}", DISAGG_TRANSFER_BACKEND_CHOICES)

    def test_parse_alias(self):
        self.assertEqual(parse_mooncake_backend_alias("mooncake_tcp"), "tcp")
        self.assertEqual(parse_mooncake_backend_alias("mooncake_efa"), "efa")
        self.assertEqual(parse_mooncake_backend_alias("mooncake_nvlink"), "nvlink")
        self.assertIsNone(parse_mooncake_backend_alias("mooncake"))
        self.assertIsNone(parse_mooncake_backend_alias("nixl"))
        self.assertIsNone(parse_mooncake_backend_alias("mooncake_unknown"))

    def test_is_mooncake_backend(self):
        self.assertTrue(is_mooncake_backend("mooncake"))
        self.assertTrue(is_mooncake_backend("mooncake_tcp"))
        self.assertTrue(is_mooncake_backend("mooncake_nvlink_intra"))
        self.assertFalse(is_mooncake_backend("nixl"))
        self.assertFalse(is_mooncake_backend("mooncake_unknown"))

    def test_validate_rejects_unknown(self):
        with self.assertRaises(ValueError):
            validate_mooncake_protocol("infiniband", "--disaggregation-mooncake-protocol")
        validate_mooncake_protocol(None, "--disaggregation-mooncake-protocol")
        validate_mooncake_protocol("rdma", "--disaggregation-mooncake-protocol")


class TestMooncakePathHints(CustomTestCase):
    def test_per_path_overrides_global(self):
        args = SimpleNamespace(
            disaggregation_mooncake_protocol="rdma",
            disaggregation_mooncake_kv_protocol="nvlink",
            disaggregation_mooncake_aux_protocol="tcp",
            disaggregation_mooncake_state_protocol=None,
            disaggregation_mooncake_staging_protocol=None,
        )
        self.assertEqual(resolve_path_transport_hint(args, "kv"), "nvlink")
        self.assertEqual(resolve_path_transport_hint(args, "aux"), "tcp")
        self.assertEqual(resolve_path_transport_hint(args, "state"), "rdma")
        self.assertEqual(resolve_path_transport_hint(args, "staging"), "rdma")

    def test_empty_when_unset(self):
        args = SimpleNamespace(
            disaggregation_mooncake_protocol=None,
            disaggregation_mooncake_kv_protocol=None,
            disaggregation_mooncake_aux_protocol=None,
            disaggregation_mooncake_state_protocol=None,
            disaggregation_mooncake_staging_protocol=None,
        )
        self.assertEqual(resolve_path_transport_hint(args, "kv"), "")

    def test_unknown_path_raises(self):
        with self.assertRaises(ValueError):
            resolve_path_transport_hint(SimpleNamespace(), "weights")


class TestApplyMooncakeProtocol(CustomTestCase):
    def setUp(self):
        self._saved_env = {key: os.environ.get(key) for key in _OWNED_FORCE_ENV}
        self._mem_pool_was_set = envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.is_set()
        self._mem_pool_value = os.environ.get(envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.name)
        self._protocol_token = envs.MOONCAKE_PROTOCOL.override(
            envs.MOONCAKE_PROTOCOL.default
        )
        self._protocol_token.__enter__()
        self._ascend_token = envs.ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE.override(False)
        self._ascend_token.__enter__()
        for key in _OWNED_FORCE_ENV:
            os.environ.pop(key, None)
        envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.clear()

    def tearDown(self):
        self._ascend_token.__exit__(None, None, None)
        self._protocol_token.__exit__(None, None, None)
        if self._mem_pool_was_set:
            os.environ[envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.name] = self._mem_pool_value
        else:
            envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.clear()
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_tcp_sets_force_flag_and_protocol(self):
        spec = apply_mooncake_protocol("tcp")
        self.assertTrue(spec.clear_ib_device)
        self.assertEqual(envs.MOONCAKE_PROTOCOL.get(), "tcp")
        self.assertEqual(os.environ.get("MC_FORCE_TCP"), "1")

    def test_nvlink_sets_mem_pool_and_mnnvl(self):
        spec = apply_mooncake_protocol("nvlink")
        self.assertTrue(spec.clear_ib_device)
        self.assertEqual(os.environ.get("MC_FORCE_MNNVL"), "1")
        self.assertEqual(envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get(), "NVLINK")
        self.assertIsNone(os.environ.get("MC_FORCE_TCP"))

    def test_switching_protocol_clears_previous_force_flag(self):
        apply_mooncake_protocol("tcp")
        apply_mooncake_protocol("rdma")
        self.assertEqual(envs.MOONCAKE_PROTOCOL.get(), "rdma")
        self.assertIsNone(os.environ.get("MC_FORCE_TCP"))

    def test_efa_keeps_ib_device(self):
        spec = apply_mooncake_protocol("efa")
        self.assertFalse(spec.clear_ib_device)
        self.assertEqual(envs.MOONCAKE_PROTOCOL.get(), "efa")


class TestMooncakePdHook(CustomTestCase):
    def setUp(self):
        self._saved_env = {key: os.environ.get(key) for key in _OWNED_FORCE_ENV}
        self._protocol_token = envs.MOONCAKE_PROTOCOL.override(
            envs.MOONCAKE_PROTOCOL.default
        )
        self._protocol_token.__enter__()
        for key in _OWNED_FORCE_ENV:
            os.environ.pop(key, None)

    def tearDown(self):
        self._protocol_token.__exit__(None, None, None)
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_mooncake_tcp_alias_sets_tcp_protocol(self):
        server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="mooncake_tcp",
            disaggregation_ib_device="mlx5_0",
        )
        handle_pd_disaggregation(server_args)
        self.assertEqual(server_args.disaggregation_transfer_backend, "mooncake")
        self.assertEqual(server_args.disaggregation_mooncake_protocol, "tcp")
        self.assertIsNone(server_args.disaggregation_ib_device)
        self.assertEqual(os.environ.get("MC_FORCE_TCP"), "1")

    def test_mooncake_efa_alias(self):
        server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            disaggregation_transfer_backend="mooncake_efa",
        )
        handle_pd_disaggregation(server_args)
        self.assertEqual(server_args.disaggregation_transfer_backend, "mooncake")
        self.assertEqual(server_args.disaggregation_mooncake_protocol, "efa")
        self.assertEqual(envs.MOONCAKE_PROTOCOL.get(), "efa")

    def test_explicit_protocol_wins_over_alias(self):
        server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="mooncake_tcp",
            disaggregation_mooncake_protocol="rdma",
        )
        handle_pd_disaggregation(server_args)
        self.assertEqual(server_args.disaggregation_transfer_backend, "mooncake")
        self.assertEqual(server_args.disaggregation_mooncake_protocol, "rdma")
        self.assertEqual(envs.MOONCAKE_PROTOCOL.get(), "rdma")
        self.assertIsNone(os.environ.get("MC_FORCE_TCP"))


class TestMooncakeTransferHintCall(CustomTestCase):
    def test_passes_transport_hint_kwarg(self):
        fn = MagicMock(return_value=0)
        ret = _call_with_transport_hint(
            fn, "session", [1], [2], [3], transport_hint="tcp"
        )
        self.assertEqual(ret, 0)
        fn.assert_called_once_with("session", [1], [2], [3], transport_hint="tcp")

    def test_omits_empty_hint(self):
        fn = MagicMock(return_value=0)
        _call_with_transport_hint(fn, "session", [1], [2], [3], transport_hint="")
        fn.assert_called_once_with("session", [1], [2], [3])

    def test_falls_back_when_kwarg_unsupported(self):
        fn = MagicMock(side_effect=[TypeError("unexpected kwarg"), 0])
        ret = _call_with_transport_hint(
            fn, "session", [1], [2], [3], transport_hint="nvlink"
        )
        self.assertEqual(ret, 0)
        self.assertEqual(fn.call_count, 2)
        self.assertEqual(fn.call_args_list[1].kwargs, {})

    def test_batch_transfer_sync_forwards_hint(self):
        engine = MooncakeTransferEngine.__new__(MooncakeTransferEngine)
        engine.engine = SimpleNamespace(
            batch_transfer_sync_write=MagicMock(return_value=0)
        )
        ret = engine.batch_transfer_sync("sid", [1], [2], [3], transport_hint="efa")
        self.assertEqual(ret, 0)
        engine.engine.batch_transfer_sync_write.assert_called_once_with(
            "sid", [1], [2], [3], transport_hint="efa"
        )


class TestMooncakeAuxZmqFallback(CustomTestCase):
    def test_aux_tcp_without_tent_uses_zmq(self):
        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.server_args = SimpleNamespace(
            disaggregation_mooncake_protocol="rdma",
            disaggregation_mooncake_kv_protocol=None,
            disaggregation_mooncake_aux_protocol="tcp",
            disaggregation_mooncake_state_protocol=None,
            disaggregation_mooncake_staging_protocol=None,
        )
        manager.enable_custom_mem_pool = False
        manager.custom_mem_pool_type = None
        with envs.SGLANG_MOONCAKE_SEND_AUX_TCP.override(False):
            saved = os.environ.get("MC_USE_TENT")
            os.environ.pop("MC_USE_TENT", None)
            try:
                self.assertTrue(manager._should_send_aux_via_zmq())
                os.environ["MC_USE_TENT"] = "1"
                self.assertFalse(manager._should_send_aux_via_zmq())
                os.environ.pop("MC_USE_TENT", None)
                manager.server_args.disaggregation_mooncake_protocol = "tcp"
                manager.server_args.disaggregation_mooncake_aux_protocol = None
                self.assertFalse(manager._should_send_aux_via_zmq())
            finally:
                if saved is None:
                    os.environ.pop("MC_USE_TENT", None)
                else:
                    os.environ["MC_USE_TENT"] = saved
