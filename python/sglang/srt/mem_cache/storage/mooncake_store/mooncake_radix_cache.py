from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    EvictResult,
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_connector import (
    MooncakeConnector,
)

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass
class _LoadBackMarker:
    key: RadixKey
    value_numel: int
    page_keys: list[str]


class MooncakeRadixCache(RadixCache):
    """Direct GPU radix cache whose external page placement is owned by Mooncake."""

    def __init__(
        self,
        params: CacheInitParams,
        model_config: Optional[ModelConfig],
        server_args: ServerArgs,
        tp_rank: int,
        tp_size: int,
        tp_group=None,
    ) -> None:
        super().__init__(params)
        self.mooncake_connector = MooncakeConnector(
            kvcache=self.token_to_kv_pool_allocator.get_kvcache(),
            model_config=model_config,
            server_args=server_args,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
        )
        self._load_markers: dict[str, _LoadBackMarker] = {}
        self._inflight_store_nodes: dict[str, TreeNode] = {}
        self._node_lock = threading.Lock()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        key = params.key
        if self.disable or len(key) == 0:
            return super().match_prefix(params)
        key = key.page_aligned(self.page_size)
        base_result = super().match_prefix(params)
        if len(key) == 0 or params.req is None:
            return base_result

        device_tokens = int(base_result.device_indices.numel())
        if device_tokens >= len(key):
            return base_result

        page_keys, hit_tokens = self.mooncake_connector.lookup(key, device_tokens)
        if hit_tokens <= 0:
            return base_result

        token_ids = key.raw_token_ids()
        token_snapshot = token_ids[:] if token_ids is key.token_ids else token_ids
        self._load_markers[params.req.rid] = _LoadBackMarker(
            key=RadixKey(
                token_snapshot,
                key.extra_key,
                is_bigram=key.is_bigram,
            ),
            value_numel=device_tokens,
            page_keys=page_keys,
        )
        last_node = base_result.last_device_node
        return MatchResult(
            device_indices=base_result.device_indices,
            last_device_node=last_node,
            last_host_node=last_node,
            best_match_node=last_node,
            host_hit_length=hit_tokens,
        )

    def init_load_back(
        self, params: InitLoadBackParams
    ) -> Tuple[torch.Tensor, Optional[TreeNode]]:
        req = params.req
        last_node = params.best_match_node
        if req is None:
            return self._empty_slots(), last_node
        marker = self._load_markers.pop(req.rid, None)
        if marker is None:
            self.mooncake_connector.release(req.rid)
            return self._empty_slots(), last_node

        result = self._allocate_and_load(
            marker=marker,
            uncached_len=params.host_hit_length,
            last_node=last_node,
        )
        if result is None:
            self.mooncake_connector.release(req.rid)
            return self._empty_slots(), last_node
        return result

    def _empty_slots(self) -> torch.Tensor:
        return torch.empty((0,), dtype=torch.int64, device=self.device)

    def _allocate_and_load(
        self,
        *,
        marker: _LoadBackMarker,
        uncached_len: int,
        last_node: TreeNode,
    ) -> Optional[Tuple[torch.Tensor, TreeNode]]:
        if uncached_len <= 0:
            return None
        if self.token_to_kv_pool_allocator.available_size() < uncached_len:
            self.evict(EvictParams(num_tokens=uncached_len))
        token_slots = self.token_to_kv_pool_allocator.alloc(uncached_len)
        if token_slots is None:
            return None

        loaded_tokens = self.mooncake_connector.load(
            marker.page_keys, token_slots.to(torch.int64)
        )
        loaded_tokens = min(int(loaded_tokens), uncached_len)
        loaded_tokens = loaded_tokens // self.page_size * self.page_size
        if loaded_tokens <= 0:
            self.token_to_kv_pool_allocator.free(token_slots)
            return None

        if loaded_tokens < uncached_len:
            self.token_to_kv_pool_allocator.free(token_slots[loaded_tokens:])
        fetched_slots = token_slots[:loaded_tokens]

        new_node = TreeNode(priority=last_node.priority)
        start = marker.value_numel
        new_node.key = marker.key[start : start + loaded_tokens]
        new_node.value = fetched_slots
        new_node.parent = last_node
        last_node.children[new_node.key.child_key(self.page_size)] = new_node
        self.evictable_size_ += loaded_tokens
        self._update_leaf_status(last_node)
        self._update_leaf_status(new_node)
        self._record_store_event(new_node)
        return fetched_slots, new_node

    def cache_finished_req(self, req: Req, is_insert: bool = True) -> None:
        if not is_insert:
            super().cache_finished_req(req, is_insert=False)
            self._load_markers.pop(req.rid, None)
            self.mooncake_connector.release(req.rid)
            return

        committed_len = int(req.kv_committed_len)
        token_ids = (req.origin_input_ids + req.output_ids)[:committed_len]
        radix_key = RadixKey(
            token_ids, req.extra_key, is_bigram=self.is_eagle
        ).page_aligned(self.page_size)
        slots = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(radix_key)
        ].to(dtype=torch.int64, copy=True)

        super().cache_finished_req(req, is_insert=True)
        self._load_markers.pop(req.rid, None)
        if len(radix_key) == 0:
            return

        match_result = super().match_prefix(MatchPrefixParams(key=radix_key))
        source_node = match_result.last_device_node
        if source_node is None:
            return
        self.inc_lock_ref(source_node)
        try:
            accepted = self.mooncake_connector.store_async(
                req.rid, radix_key, slots
            )
        except Exception:
            self.dec_lock_ref(source_node)
            raise
        if not accepted:
            self.dec_lock_ref(source_node)
            return
        with self._node_lock:
            self._inflight_store_nodes[req.rid] = source_node

    def _drain_completed_stores(self) -> None:
        completed = self.mooncake_connector.completed_store_rids()
        if not completed:
            return
        with self._node_lock:
            for rid in completed:
                node = self._inflight_store_nodes.pop(rid, None)
                if node is not None:
                    self.dec_lock_ref(node)

    def check_hicache_events(self) -> None:
        self._drain_completed_stores()

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()
        self.mooncake_connector.wait_for_all_stores()
        self._drain_completed_stores()
        return super().evict(params)

    def release_aborted_request(self, rid: str) -> None:
        self._load_markers.pop(rid, None)
        with self._node_lock:
            node = self._inflight_store_nodes.pop(rid, None)
        if node is not None:
            self.dec_lock_ref(node)
        self.mooncake_connector.release(rid)

    def reset(self) -> None:
        connector = getattr(self, "mooncake_connector", None)
        if connector is not None:
            connector.wait_for_all_stores()
            self._drain_completed_stores()
        if hasattr(self, "_load_markers"):
            self._load_markers.clear()
        super().reset()
        if connector is not None:
            connector.reset()

    def shutdown(self) -> None:
        connector = getattr(self, "mooncake_connector", None)
        if connector is None:
            return
        connector.wait_for_all_stores()
        self._drain_completed_stores()
        connector.close()

    @property
    def hicache_storage_pass_prefix_keys(self) -> bool:
        return False
