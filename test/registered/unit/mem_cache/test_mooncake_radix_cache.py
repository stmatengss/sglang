"""CPU unit tests for the direct Mooncake radix-cache contract."""

from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InitLoadBackParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_radix_cache import (
    MooncakeRadixCache,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Allocator:
    device = "cpu"

    def __init__(self):
        self.next_slot = 100
        self.freed = []

    def available_size(self):
        return 1024

    def alloc(self, count):
        result = torch.arange(self.next_slot, self.next_slot + count)
        self.next_slot += count
        return result

    def free(self, slots):
        self.freed.append(torch.as_tensor(slots).clone())


class _Connector:
    def __init__(self):
        self.lookup_result = ([], 0)
        self.loaded = 0
        self.lookup_calls = []
        self.load_calls = []
        self.store_calls = []
        self.completed = []
        self.wait_count = 0
        self.released = []

    def lookup(self, key, device_tokens):
        self.lookup_calls.append((key, device_tokens))
        return self.lookup_result

    def load(self, keys, slots):
        self.load_calls.append((keys, slots.clone()))
        return self.loaded

    def store_async(self, rid, key, slots):
        self.store_calls.append((rid, key, slots.clone()))
        return True

    def completed_store_rids(self):
        result, self.completed = self.completed, []
        return result

    def wait_for_all_stores(self):
        self.wait_count += 1

    def release(self, rid):
        self.released.append(rid)

    def reset(self):
        pass

    def close(self):
        pass


def _make_cache(page_size=2):
    allocator = _Allocator()
    base = RadixCache.create_simulated(
        mock_allocator=allocator, page_size=page_size
    )
    cache = MooncakeRadixCache.__new__(MooncakeRadixCache)
    cache.__dict__.update(base.__dict__)
    cache.mooncake_connector = _Connector()
    cache._load_markers = {}
    cache._inflight_store_nodes = {}
    import threading

    cache._node_lock = threading.Lock()
    return cache, allocator


class TestMooncakeRadixLoadBack(CustomTestCase):
    def test_l1_only_hit_bypasses_mooncake(self):
        cache, _ = _make_cache()
        key = RadixKey(array("q", [1, 2, 3, 4]))
        cache.insert(InsertParams(key=key, value=torch.tensor([10, 11, 12, 13])))
        result = cache.match_prefix(
            MatchPrefixParams(key=key, req=SimpleNamespace(rid="r"))
        )
        self.assertEqual(result.device_indices.tolist(), [10, 11, 12, 13])
        self.assertEqual(cache.mooncake_connector.lookup_calls, [])

    def test_external_hit_snapshots_key_and_reports_missing_tail(self):
        cache, _ = _make_cache()
        ids = array("q", [1, 2, 3, 4])
        key = RadixKey(ids)
        cache.mooncake_connector.lookup_result = (["p0", "p1"], 4)
        result = cache.match_prefix(
            MatchPrefixParams(key=key, req=SimpleNamespace(rid="r"))
        )
        ids.extend([5, 6])
        self.assertEqual(result.host_hit_length, 4)
        self.assertEqual(len(cache._load_markers["r"].key), 4)

    def test_init_load_back_allocates_loads_and_inserts_node(self):
        cache, _ = _make_cache()
        cache.mooncake_connector.lookup_result = (["p0", "p1"], 4)
        cache.mooncake_connector.loaded = 4
        req = SimpleNamespace(rid="r")
        match = cache.match_prefix(
            MatchPrefixParams(
                key=RadixKey(array("q", [1, 2, 3, 4])), req=req
            )
        )
        slots, node = cache.init_load_back(
            InitLoadBackParams(
                best_match_node=match.best_match_node,
                host_hit_length=match.host_hit_length,
                req=req,
            )
        )
        self.assertEqual(slots.tolist(), [100, 101, 102, 103])
        self.assertEqual(node.key.token_ids.tolist(), [1, 2, 3, 4])
        self.assertNotIn("r", cache._load_markers)

    def test_failed_load_frees_slots_and_does_not_insert(self):
        cache, allocator = _make_cache()
        cache.mooncake_connector.lookup_result = (["p0"], 2)
        req = SimpleNamespace(rid="r")
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [1, 2])), req=req)
        )
        slots, node = cache.init_load_back(
            InitLoadBackParams(
                best_match_node=match.best_match_node,
                host_hit_length=2,
                req=req,
            )
        )
        self.assertEqual(slots.numel(), 0)
        self.assertIs(node, cache.root_node)
        self.assertEqual(len(cache.root_node.children), 0)
        self.assertEqual(allocator.freed[0].tolist(), [100, 101])


class TestMooncakeRadixStoreLifecycle(CustomTestCase):
    def test_finished_request_writes_directly_to_l2_mooncake(self):
        cache, _ = _make_cache()
        cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([[10, 11, 12, 13]])
        )
        req = SimpleNamespace(
            rid="direct-l2",
            kv_committed_len=4,
            origin_input_ids=[1, 2, 3, 4],
            output_ids=[],
            extra_key=None,
            req_pool_idx=0,
            cache_protected_len=0,
            last_node=cache.root_node,
            priority=0,
            pop_committed_kv_cache=MagicMock(return_value=4),
        )

        cache.cache_finished_req(req, kv_len_to_handle=4)

        self.assertFalse(hasattr(cache, "cache_controller"))
        self.assertEqual(len(cache.mooncake_connector.store_calls), 1)
        rid, key, slots = cache.mooncake_connector.store_calls[0]
        self.assertEqual(rid, "direct-l2")
        self.assertEqual(key.raw_token_ids(), [1, 2, 3, 4])
        self.assertEqual(slots.tolist(), [10, 11, 12, 13])

    def test_store_completion_unlocks_source_once(self):
        cache, _ = _make_cache()
        cache.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([[10, 11, 12, 13]])
        )
        req = SimpleNamespace(
            rid="r",
            kv_committed_len=4,
            origin_input_ids=[1, 2, 3, 4],
            output_ids=[],
            extra_key=None,
            req_pool_idx=0,
            cache_protected_len=0,
            last_node=cache.root_node,
            priority=0,
            pop_committed_kv_cache=MagicMock(return_value=4),
        )
        cache.cache_finished_req(req, kv_len_to_handle=4)
        node = cache._inflight_store_nodes["r"]
        self.assertEqual(node.lock_ref, 1)
        cache.mooncake_connector.completed = ["r"]
        cache.check_hicache_events()
        self.assertEqual(node.lock_ref, 0)
        cache.check_hicache_events()
        self.assertEqual(node.lock_ref, 0)

    def test_evict_waits_for_stores_before_freeing(self):
        cache, _ = _make_cache()
        cache.evict(EvictParams(num_tokens=1))
        self.assertEqual(cache.mooncake_connector.wait_count, 1)

    def test_abort_releases_marker_and_connector_state(self):
        cache, _ = _make_cache()
        cache._load_markers["r"] = MagicMock()
        cache.release_aborted_request("r")
        self.assertNotIn("r", cache._load_markers)
        self.assertEqual(cache.mooncake_connector.released, ["r"])
