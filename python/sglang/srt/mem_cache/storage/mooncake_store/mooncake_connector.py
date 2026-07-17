from __future__ import annotations

import hashlib
import json
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import RadixKey

logger = logging.getLogger(__name__)


def _namespace_with_extra_key(namespace: str, extra_key: object) -> str:
    """Return a stable namespace digest that also isolates adapter/cache salts."""
    encoded_extra = json.dumps(
        extra_key,
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    )
    return hashlib.sha256(f"{namespace}\0{encoded_extra}".encode()).hexdigest()


def build_page_keys(
    key: RadixKey,
    *,
    page_size: int,
    device_tokens: int,
    tp_rank: int,
    namespace: str,
) -> list[str]:
    """Build position-aware Mooncake keys for the page-aligned missing tail."""
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if device_tokens < 0 or device_tokens % page_size != 0:
        raise ValueError(
            f"device_tokens must be non-negative and page-aligned, got {device_tokens}"
        )

    aligned_tokens = len(key) // page_size * page_size
    if device_tokens > aligned_tokens:
        raise ValueError(
            f"device_tokens={device_tokens} exceeds aligned key length={aligned_tokens}"
        )

    object_namespace = _namespace_with_extra_key(namespace, key.extra_key)
    prior_hash = None
    page_keys: list[str] = []
    for start in range(0, aligned_tokens, page_size):
        page_hash = key.hash_page(start, start + page_size, prior_hash)
        prior_hash = page_hash
        if start >= device_tokens:
            page_keys.append(f"{object_namespace}@tp{tp_rank}@{page_hash}")
    return page_keys


def _page_spans_for_buffer(
    buffer: torch.Tensor, first_slot: int, page_size: int
) -> tuple[int, int]:
    page = buffer[first_slot : first_slot + page_size]
    if page.shape[0] != page_size:
        raise ValueError(
            f"page starting at slot {first_slot} exceeds buffer shape {tuple(buffer.shape)}"
        )
    if not page.is_contiguous():
        raise ValueError("Mooncake direct I/O requires contiguous NHD page spans")
    return page.data_ptr(), page.numel() * page.element_size()


def get_gpu_page_buffer_meta(
    kvcache, slots: torch.Tensor
) -> tuple[list[int], list[int]]:
    """Return flattened per-page multi-buffer pointers and sizes.

    The order is page-major. Within each MHA page it is K0, V0, K1, V1,
    while MLA contributes one combined KV span per layer.
    """
    page_size = int(kvcache.page_size)
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")

    flat_slots = [int(slot) for slot in slots.detach().cpu().reshape(-1).tolist()]
    if len(flat_slots) % page_size != 0:
        raise ValueError(
            f"slot count {len(flat_slots)} is not page-aligned to {page_size}"
        )

    for offset in range(0, len(flat_slots), page_size):
        first = flat_slots[offset]
        if flat_slots[offset : offset + page_size] != list(
            range(first, first + page_size)
        ):
            raise ValueError("Mooncake direct I/O requires contiguous slots per page")

    layer_num = int(kvcache.layer_num)
    kv_buffers = getattr(kvcache, "kv_buffer", None)
    is_mla = isinstance(kv_buffers, (list, tuple))
    if is_mla:
        if len(kv_buffers) != layer_num:
            raise ValueError(
                f"MLA layer_num={layer_num} does not match {len(kv_buffers)} buffers"
            )
    else:
        k_buffers = getattr(kvcache, "k_buffer", None)
        v_buffers = getattr(kvcache, "v_buffer", None)
        if not isinstance(k_buffers, (list, tuple)) or not isinstance(
            v_buffers, (list, tuple)
        ):
            raise TypeError("Unsupported KV cache: expected MHA K/V or MLA KV buffers")
        if len(k_buffers) != layer_num or len(v_buffers) != layer_num:
            raise ValueError("MHA layer count does not match K/V buffer counts")
        layout = getattr(kvcache, "kv_cache_layout", "nhd")
        if isinstance(layout, str) and layout.lower() != "nhd":
            raise ValueError(
                f"Mooncake direct I/O currently requires NHD layout, got {layout!r}"
            )

    ptrs: list[int] = []
    sizes: list[int] = []
    for offset in range(0, len(flat_slots), page_size):
        first_slot = flat_slots[offset]
        if is_mla:
            for buffer in kv_buffers:
                ptr, size = _page_spans_for_buffer(buffer, first_slot, page_size)
                ptrs.append(ptr)
                sizes.append(size)
        else:
            for k_buffer, v_buffer in zip(k_buffers, v_buffers):
                k_ptr, k_size = _page_spans_for_buffer(
                    k_buffer, first_slot, page_size
                )
                v_ptr, v_size = _page_spans_for_buffer(
                    v_buffer, first_slot, page_size
                )
                ptrs.extend((k_ptr, v_ptr))
                sizes.extend((k_size, v_size))
    return ptrs, sizes


def _stable_namespace(model_config, server_args, kvcache, tp_size: int) -> str:
    hf_config = getattr(model_config, "hf_config", None)
    config_identity = None
    if hf_config is not None:
        to_diff_dict = getattr(hf_config, "to_diff_dict", None)
        if callable(to_diff_dict):
            try:
                config_identity = to_diff_dict()
            except Exception:
                config_identity = None
    payload = {
        "model_path": getattr(server_args, "model_path", None),
        "revision": getattr(server_args, "revision", None),
        "model_config": config_identity,
        "kv_cache_dtype": str(getattr(server_args, "kv_cache_dtype", None)),
        "kv_cache_layout": str(getattr(kvcache, "kv_cache_layout", "mla")),
        "page_size": int(kvcache.page_size),
        "tp_size": int(tp_size),
        "buffer_shapes": [
            (tuple(buffer.shape), str(buffer.dtype))
            for buffer in _kv_buffers(kvcache)
        ],
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def _kv_buffers(kvcache) -> list[torch.Tensor]:
    kv_buffers = getattr(kvcache, "kv_buffer", None)
    if isinstance(kv_buffers, (list, tuple)):
        return list(kv_buffers)
    k_buffers = getattr(kvcache, "k_buffer", None)
    v_buffers = getattr(kvcache, "v_buffer", None)
    if not isinstance(k_buffers, (list, tuple)) or not isinstance(
        v_buffers, (list, tuple)
    ):
        raise TypeError("Unsupported KV cache: expected MHA K/V or MLA KV buffers")
    return list(k_buffers) + list(v_buffers)


class MooncakeConnector:
    def __init__(
        self,
        *,
        kvcache,
        model_config,
        server_args,
        tp_rank,
        tp_size,
        tp_group,
        _store=None,
        _skip_setup: bool = False,
    ):
        # Imported lazily so key/meta helpers remain usable without Mooncake.
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeBaseStore,
        )

        self.kvcache = kvcache
        self.page_size = int(kvcache.page_size)
        self.tp_rank = int(tp_rank)
        self.tp_size = int(tp_size)
        self.tp_group = tp_group
        self.namespace = _stable_namespace(
            model_config, server_args, kvcache, self.tp_size
        )
        self._base_store = MooncakeBaseStore()
        if _store is None:
            self.store = self._base_store._import_mooncake_store()()
        else:
            self.store = _store
        self._base_store.store = self.store

        buffers = _kv_buffers(kvcache)
        if not _skip_setup:
            self._base_store.config = self._base_store._load_config(None)
            self._base_store._setup_distributed_store(
                storage_config=SimpleNamespace(
                    tp_size=self.tp_size,
                    tp_rank=self.tp_rank,
                ),
                standalone_required_bytes=sum(
                    buffer.untyped_storage().nbytes() for buffer in buffers
                ),
            )

        self._registered_allocations: list[tuple[int, int]] = []
        seen_allocations: set[tuple[int, int]] = set()
        for buffer in buffers:
            storage = buffer.untyped_storage()
            allocation = (int(storage.data_ptr()), int(storage.nbytes()))
            if allocation in seen_allocations:
                continue
            seen_allocations.add(allocation)
            ret_code = self.store.register_buffer(*allocation)
            if ret_code != 0:
                raise RuntimeError(
                    "Failed to register GPU KV allocation with Mooncake, "
                    f"error code: {ret_code}"
                )
            self._registered_allocations.append(allocation)

        max_workers = int(
            getattr(server_args, "mooncake_store_workers", 4) or 4
        )
        max_pending = int(
            getattr(server_args, "mooncake_max_pending_stores", max_workers * 2)
            or max_workers * 2
        )
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="mooncake-radix-store"
        )
        self._store_slots = threading.BoundedSemaphore(max_pending)
        self._store_futures: dict[str, Future] = {}
        self._future_lock = threading.Lock()
        self._closed = False

    def _buffers_per_page(self) -> int:
        return int(self.kvcache.layer_num) * (
            1
            if isinstance(getattr(self.kvcache, "kv_buffer", None), (list, tuple))
            else 2
        )

    def _split_page_meta(
        self, slots: torch.Tensor
    ) -> tuple[list[list[int]], list[list[int]]]:
        ptrs, sizes = get_gpu_page_buffer_meta(self.kvcache, slots)
        width = self._buffers_per_page()
        return (
            [ptrs[i : i + width] for i in range(0, len(ptrs), width)],
            [sizes[i : i + width] for i in range(0, len(sizes), width)],
        )

    @staticmethod
    def _consecutive_successes(results, expected: int) -> int:
        if results is None:
            return 0
        count = 0
        for result in list(results)[:expected]:
            if isinstance(result, bool):
                success = result
            else:
                success = int(result) >= 0
            if not success:
                break
            count += 1
        return count

    def _tp_min(self, value: int) -> int:
        if self.tp_size <= 1 or self.tp_group is None:
            return value
        buffers = _kv_buffers(self.kvcache)
        device = buffers[0].device if buffers else torch.device("cpu")
        count = torch.tensor([value], dtype=torch.int64, device=device)
        try:
            import torch.distributed as dist

            reduced = self.tp_group.all_reduce(count, op=dist.ReduceOp.MIN)
        except TypeError:
            reduced = self.tp_group.all_reduce(count)
        if isinstance(reduced, torch.Tensor):
            count = reduced
        agreed = int(count.item())
        if agreed != value:
            logger.warning(
                "Mooncake TP ranks disagreed on consecutive page count: local=%d min=%d",
                value,
                agreed,
            )
        return agreed

    def lookup(self, key: RadixKey, device_tokens: int) -> tuple[list[str], int]:
        page_keys = build_page_keys(
            key,
            page_size=self.page_size,
            device_tokens=device_tokens,
            tp_rank=self.tp_rank,
            namespace=self.namespace,
        )
        if not page_keys:
            return [], 0
        exists = self.store.batch_is_exist(page_keys)
        local_pages = 0
        for present in list(exists)[: len(page_keys)]:
            if int(present) != 1:
                break
            local_pages += 1
        hit_pages = self._tp_min(local_pages)
        return page_keys[:hit_pages], hit_pages * self.page_size

    def load(self, page_keys: list[str], slots: torch.Tensor) -> int:
        if not page_keys:
            return 0
        page_ptrs, page_sizes = self._split_page_meta(slots)
        count = min(len(page_keys), len(page_ptrs))
        if count == 0:
            return 0
        results = self.store.batch_get_into_multi_buffers(
            page_keys[:count], page_ptrs[:count], page_sizes[:count]
        )
        loaded_pages = self._tp_min(self._consecutive_successes(results, count))
        return loaded_pages * self.page_size

    def _store_pages(self, page_keys, page_ptrs, page_sizes, event) -> bool:
        if event is not None:
            event.synchronize()
        results = self.store.batch_put_from_multi_buffers(
            page_keys, page_ptrs, page_sizes
        )
        return self._consecutive_successes(results, len(page_keys)) == len(page_keys)

    def store_async(self, rid: str, key: RadixKey, slots: torch.Tensor) -> bool:
        if self._closed or not self._store_slots.acquire(blocking=False):
            return False
        try:
            page_keys = build_page_keys(
                key,
                page_size=self.page_size,
                device_tokens=0,
                tp_rank=self.tp_rank,
                namespace=self.namespace,
            )
            page_ptrs, page_sizes = self._split_page_meta(slots)
            if not page_keys or len(page_keys) != len(page_ptrs):
                return False
            with self._future_lock:
                if rid in self._store_futures:
                    return False
                event = None
                if slots.device.type == "cuda":
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(slots.device))
                future = self._executor.submit(
                    self._store_pages, page_keys, page_ptrs, page_sizes, event
                )
                future.add_done_callback(lambda _future: self._store_slots.release())
                self._store_futures[rid] = future
            return True
        finally:
            # Accepted futures release the slot in their completion callback.
            with self._future_lock:
                accepted = rid in self._store_futures
            if not accepted:
                self._store_slots.release()

    def completed_store_rids(self) -> list[str]:
        completed = []
        with self._future_lock:
            for rid, future in list(self._store_futures.items()):
                if not future.done():
                    continue
                try:
                    if not future.result():
                        logger.warning("Mooncake async store failed for rid=%s", rid)
                except Exception:
                    logger.exception("Mooncake async store raised for rid=%s", rid)
                completed.append(rid)
                del self._store_futures[rid]
        return completed

    def release(self, rid: str) -> None:
        with self._future_lock:
            future = self._store_futures.get(rid)
            if future is not None and future.cancel():
                del self._store_futures[rid]

    def wait_for_all_stores(self) -> None:
        while True:
            with self._future_lock:
                futures = list(self._store_futures.values())
            if not futures:
                return
            for future in futures:
                try:
                    future.result()
                except Exception:
                    logger.exception("Mooncake async store raised while draining")
            self.completed_store_rids()

    def replica_tiers(self, page_keys: list[str]) -> dict[str, str]:
        descriptors_by_key = self.store.batch_get_replica_desc(page_keys)
        tiers = {}
        for index, key in enumerate(page_keys):
            if isinstance(descriptors_by_key, dict):
                descriptors = descriptors_by_key.get(key, [])
            else:
                # Compatibility with older bindings and simple fake stores.
                descriptors = (
                    descriptors_by_key[index]
                    if index < len(descriptors_by_key)
                    else []
                )
            if descriptors is None:
                descriptors = []
            if not isinstance(descriptors, (list, tuple)):
                descriptors = [descriptors]

            if any(
                callable(method := getattr(descriptor, "is_memory_replica", None))
                and method()
                for descriptor in descriptors
            ):
                tiers[key] = "memory"
            elif any(
                callable(method := getattr(descriptor, "is_disk_replica", None))
                and method()
                for descriptor in descriptors
            ):
                tiers[key] = "disk"
            else:
                tiers[key] = "missing"
        return tiers

    def reset(self) -> None:
        self.wait_for_all_stores()

    def close(self) -> None:
        if self._closed:
            return
        self.wait_for_all_stores()
        self._closed = True
        self._executor.shutdown(wait=True)
        unregister = getattr(self.store, "unregister_buffer", None)
        if callable(unregister):
            for ptr, _size in self._registered_allocations:
                try:
                    unregister(ptr)
                except TypeError:
                    unregister(ptr, _size)
