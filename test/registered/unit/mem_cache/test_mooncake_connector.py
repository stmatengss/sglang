"""Unit tests for MooncakeConnector keys and GPU page metadata."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestMooncakePageKeys(CustomTestCase):
    def test_keys_differ_by_extra_key_and_tp_rank(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_connector import (
            build_page_keys,
        )

        key_a = RadixKey(array("q", [1, 2, 3, 4]), extra_key=None)
        key_b = RadixKey(array("q", [1, 2, 3, 4]), extra_key="lora-1")
        keys_a = build_page_keys(
            key_a, page_size=2, device_tokens=0, tp_rank=0, namespace="ns"
        )
        keys_b = build_page_keys(
            key_b, page_size=2, device_tokens=0, tp_rank=0, namespace="ns"
        )
        keys_tp1 = build_page_keys(
            key_a, page_size=2, device_tokens=0, tp_rank=1, namespace="ns"
        )
        self.assertNotEqual(keys_a, keys_b)
        self.assertNotEqual(keys_a, keys_tp1)

    def test_missing_tail_keeps_full_prefix_hash_chain(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_connector import (
            build_page_keys,
        )

        key = RadixKey(array("q", [1, 2, 3, 4, 5, 6]))
        all_keys = build_page_keys(
            key, page_size=2, device_tokens=0, tp_rank=0, namespace="ns"
        )
        tail_keys = build_page_keys(
            key, page_size=2, device_tokens=2, tp_rank=0, namespace="ns"
        )
        self.assertEqual(tail_keys, all_keys[1:])


class TestMooncakeGpuPageMeta(CustomTestCase):
    def test_mha_page_meta_has_k_and_v_per_layer(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_connector import (
            get_gpu_page_buffer_meta,
        )

        page_size, layers, heads, dim = 2, 2, 2, 4
        kvcache = MagicMock()
        kvcache.page_size = page_size
        kvcache.layer_num = layers
        kvcache.k_buffer = [
            torch.zeros(16, heads, dim) for _ in range(layers)
        ]
        kvcache.v_buffer = [
            torch.zeros(16, heads, dim) for _ in range(layers)
        ]
        slots = torch.tensor([2, 3], dtype=torch.int64)
        ptrs, sizes = get_gpu_page_buffer_meta(kvcache, slots)
        self.assertEqual(len(ptrs), 4)
        self.assertEqual(len(sizes), 4)
        self.assertTrue(all(size == page_size * heads * dim * 4 for size in sizes))

    def test_mla_page_meta_has_one_span_per_layer(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_connector import (
            get_gpu_page_buffer_meta,
        )

        kvcache = MagicMock()
        kvcache.page_size = 2
        kvcache.layer_num = 2
        kvcache.kv_buffer = [torch.zeros(16, 1, 8) for _ in range(2)]
        ptrs, sizes = get_gpu_page_buffer_meta(
            kvcache, torch.tensor([4, 5], dtype=torch.int64)
        )
        self.assertEqual(len(ptrs), 2)
        self.assertEqual(sizes, [2 * 1 * 8 * 4, 2 * 1 * 8 * 4])

    def test_rejects_non_contiguous_slots(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_connector import (
            get_gpu_page_buffer_meta,
        )

        kvcache = MagicMock()
        kvcache.page_size = 2
        kvcache.layer_num = 1
        kvcache.k_buffer = [torch.zeros(16, 2, 4)]
        kvcache.v_buffer = [torch.zeros(16, 2, 4)]
        with self.assertRaises(ValueError):
            get_gpu_page_buffer_meta(kvcache, torch.tensor([2, 5]))

    def test_rejects_partial_page(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_connector import (
            get_gpu_page_buffer_meta,
        )

        kvcache = MagicMock()
        kvcache.page_size = 2
        kvcache.layer_num = 1
        kvcache.k_buffer = [torch.zeros(16, 2, 4)]
        kvcache.v_buffer = [torch.zeros(16, 2, 4)]
        with self.assertRaises(ValueError):
            get_gpu_page_buffer_meta(kvcache, torch.tensor([2]))


class _FakeMooncakeStore:
    def __init__(self):
        self.registered = []
        self.exists = []
        self.get_results = []
        self.put_results = []
        self.get_calls = []
        self.put_calls = []
        self.descriptors = []

    def register_buffer(self, ptr, size):
        self.registered.append((ptr, size))
        return 0

    def batch_is_exist(self, keys):
        return self.exists or [0] * len(keys)

    def batch_get_into_multi_buffers(self, keys, ptrs, sizes):
        self.get_calls.append((keys, ptrs, sizes))
        return self.get_results or [0] * len(keys)

    def batch_put_from_multi_buffers(self, keys, ptrs, sizes):
        self.put_calls.append((keys, ptrs, sizes))
        return self.put_results or [0] * len(keys)

    def batch_get_replica_desc(self, keys):
        return self.descriptors

    def remove_all(self):
        return 0


class _MinTPGroup:
    def __init__(self, value):
        self.value = value

    def all_reduce(self, tensor, op=None):
        tensor.fill_(self.value)


class _ReplicaDescriptor:
    def __init__(self, tier):
        self.tier = tier

    def is_memory_replica(self):
        return self.tier == "memory"

    def is_disk_replica(self):
        return self.tier == "disk"


class TestMooncakeConnectorIO(CustomTestCase):
    def _make_connector(self, *, store=None, tp_size=1, tp_group=None):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_connector import (
            MooncakeConnector,
        )

        kvcache = SimpleNamespace(
            page_size=2,
            layer_num=1,
            k_buffer=[torch.zeros(16, 2, 4)],
            v_buffer=[torch.zeros(16, 2, 4)],
            kv_cache_layout="nhd",
        )
        server_args = SimpleNamespace(
            model_path="test-model",
            revision="main",
            kv_cache_dtype="float32",
            mooncake_store_workers=1,
            mooncake_max_pending_stores=2,
        )
        return MooncakeConnector(
            kvcache=kvcache,
            model_config=SimpleNamespace(hf_config=None),
            server_args=server_args,
            tp_rank=0,
            tp_size=tp_size,
            tp_group=tp_group,
            _store=store or _FakeMooncakeStore(),
            _skip_setup=True,
        )

    def test_registers_each_gpu_allocation_once(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_connector import (
            MooncakeConnector,
        )

        allocation = torch.zeros(32, 2, 4)
        kvcache = SimpleNamespace(
            page_size=2,
            layer_num=1,
            k_buffer=[allocation[:16]],
            v_buffer=[allocation[16:]],
            kv_cache_layout="nhd",
        )
        store = _FakeMooncakeStore()
        connector = MooncakeConnector(
            kvcache=kvcache,
            model_config=SimpleNamespace(hf_config=None),
            server_args=SimpleNamespace(
                model_path="model",
                revision=None,
                kv_cache_dtype="float32",
                mooncake_store_workers=1,
                mooncake_max_pending_stores=2,
            ),
            tp_rank=0,
            tp_size=1,
            tp_group=None,
            _store=store,
            _skip_setup=True,
        )
        self.addCleanup(connector.close)
        self.assertEqual(len(store.registered), 1)

    def test_lookup_reports_consecutive_tp_minimum(self):
        store = _FakeMooncakeStore()
        store.exists = [1, 1, 0]
        connector = self._make_connector(
            store=store, tp_size=2, tp_group=_MinTPGroup(1)
        )
        self.addCleanup(connector.close)
        key = RadixKey(array("q", [1, 2, 3, 4, 5, 6]))

        page_keys, hit_tokens = connector.lookup(key, device_tokens=0)

        self.assertEqual(len(page_keys), 1)
        self.assertEqual(hit_tokens, 2)

    def test_load_and_store_use_matching_multi_buffer_shapes(self):
        store = _FakeMooncakeStore()
        connector = self._make_connector(store=store)
        self.addCleanup(connector.close)
        key = RadixKey(array("q", [1, 2, 3, 4]))
        page_keys, _ = connector.lookup(key, device_tokens=0)
        # lookup misses by default; derive the two logical keys explicitly.
        store.exists = [1, 1]
        page_keys, _ = connector.lookup(key, device_tokens=0)
        slots = torch.tensor([2, 3, 4, 5])

        self.assertEqual(connector.load(page_keys, slots), 4)
        self.assertTrue(connector.store_async("rid", key, slots))
        connector.wait_for_all_stores()

        get_keys, get_ptrs, get_sizes = store.get_calls[0]
        put_keys, put_ptrs, put_sizes = store.put_calls[0]
        self.assertEqual(get_keys, put_keys)
        self.assertEqual([len(x) for x in get_ptrs], [2, 2])
        self.assertEqual([len(x) for x in put_ptrs], [2, 2])
        self.assertEqual(get_sizes, put_sizes)

    def test_l2_store_uses_mooncake_batch_put_directly(self):
        store = _FakeMooncakeStore()
        connector = self._make_connector(store=store)
        self.addCleanup(connector.close)
        key = RadixKey(array("q", [1, 2, 3, 4]))
        slots = torch.tensor([2, 3, 4, 5])

        self.assertTrue(connector.store_async("direct-l2", key, slots))
        connector.wait_for_all_stores()

        self.assertEqual(len(store.put_calls), 1)
        page_keys, page_ptrs, page_sizes = store.put_calls[0]
        self.assertEqual(len(page_keys), 2)
        self.assertEqual([len(ptrs) for ptrs in page_ptrs], [2, 2])
        self.assertEqual([len(sizes) for sizes in page_sizes], [2, 2])

    def test_partial_load_does_not_claim_later_pages(self):
        store = _FakeMooncakeStore()
        store.exists = [1, 1]
        store.get_results = [0, -1]
        connector = self._make_connector(store=store)
        self.addCleanup(connector.close)
        key = RadixKey(array("q", [1, 2, 3, 4]))
        page_keys, _ = connector.lookup(key, device_tokens=0)
        loaded = connector.load(page_keys, torch.tensor([2, 3, 4, 5]))
        self.assertEqual(loaded, 2)

    def test_replica_tiers_use_descriptors(self):
        store = _FakeMooncakeStore()
        store.descriptors = {
            "a": [_ReplicaDescriptor("memory")],
            "b": [_ReplicaDescriptor("disk")],
            "c": [],
        }
        connector = self._make_connector(store=store)
        self.addCleanup(connector.close)
        self.assertEqual(
            connector.replica_tiers(["a", "b", "c"]),
            {"a": "memory", "b": "disk", "c": "missing"},
        )


if __name__ == "__main__":
    unittest.main()
