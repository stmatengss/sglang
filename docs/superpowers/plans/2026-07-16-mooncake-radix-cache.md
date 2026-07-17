# Mooncake Radix Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace HiCache’s SGLang-owned L2 path for this mode with a dedicated Mooncake radix backend that directly stores and retrieves GPU KV pages while Mooncake manages DRAM and SSD placement.

**Architecture:** `MooncakeRadixCache(RadixCache)` follows the existing FlexKV two-phase lookup/loadback contract but delegates page lookup, direct GPU multi-buffer I/O, TP consistency, and async-store lifecycle to `MooncakeConnector`. The backend is selected only by `--radix-cache-backend mooncake`; existing `--enable-hierarchical-cache --hicache-storage-backend mooncake` behavior remains unchanged.

**Tech Stack:** Python, PyTorch/CUDA, `MooncakeDistributedStore`, SGLang radix-cache registry, `unittest`/`pytest`, Mooncake TCP services and SSD offload.

---

## Scope and file ownership

| File | Role |
|------|------|
| Create: `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_connector.py` | Store setup, GPU registration, page keys, TP sync, replica-tier inspection, batch I/O |
| Create: `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_radix_cache.py` | Scheduler-facing match, loadback, store, eviction, reset, abort, drain |
| Modify: `python/sglang/srt/mem_cache/storage/mooncake_store/__init__.py` | Register `_mooncake_factory` |
| Modify: `python/sglang/srt/mem_cache/registry.py` | Lazy-load built-in `mooncake` backend |
| Modify: `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py` | Extract shared config/setup only; keep `MooncakeStore(HiCacheStorage)` intact |
| Modify: `python/sglang/srt/server_args.py` | `_validate_mooncake_radix_cache()` |
| Modify: `python/sglang/srt/managers/scheduler.py` | Periodic drain + abort cleanup for mooncake backend |
| Create: `test/registered/unit/mem_cache/test_mooncake_connector.py` | CPU unit tests for connector |
| Create: `test/registered/unit/mem_cache/test_mooncake_radix_cache.py` | CPU unit tests for radix cache |
| Create: `test/registered/hicache/test_mooncake_radix_cache_e2e.py` | GPU E2E DRAM + SSD |
| Modify: `test/registered/unit/mem_cache/test_registry.py` | Lazy registration coverage |
| Modify: `test/registered/unit/server_args/test_server_args.py` | Validation coverage |

**Out of scope (deferred):** hybrid SWA/SSM, speculative draft KV, PP/CP/PD, runtime attach/detach, layerwise overlap, multi-node TP, convenience `--enable-mooncake-*` flag.

---

### Task 1: Built-in registration and startup validation

**Files:**
- Modify: `test/registered/unit/mem_cache/test_registry.py`
- Modify: `test/registered/unit/server_args/test_server_args.py`
- Modify: `python/sglang/srt/mem_cache/registry.py`
- Modify: `python/sglang/srt/mem_cache/storage/mooncake_store/__init__.py`
- Modify: `python/sglang/srt/server_args.py`

- [ ] **Step 1: Write the failing registry test**

Add to `test/registered/unit/mem_cache/test_registry.py`:

```python
class TestMooncakeBuiltinRegistration(_RegistryIsolationMixin, CustomTestCase):
    def test_create_tree_cache_lazy_loads_mooncake_backend(self):
        # Ensure mooncake is not already registered (isolation mixin cleared it).
        self.assertIsNone(get_radix_cache_factory("mooncake"))

        fake_cache = MagicMock()
        fake_cache.supports_streaming_session.return_value = True
        fake_module = MagicMock()

        def _side_effect_register(name):
            # Simulate package import calling register_radix_cache_backend.
            if name == "sglang.srt.mem_cache.storage.mooncake_store":
                register_radix_cache_backend(
                    "mooncake", MagicMock(return_value=fake_cache)
                )
            return fake_module

        with patch(
            "importlib.import_module", side_effect=_side_effect_register
        ) as import_mod:
            # Force create_tree_cache to attempt builtin load. If registry
            # has no builtin map, this raises ValueError("not registered").
            # Patch get_radix_cache_factory path by clearing registry first.
            result = create_tree_cache(_make_ctx(backend="mooncake"))

        self.assertIs(result, fake_cache)
        import_mod.assert_any_call("sglang.srt.mem_cache.storage.mooncake_store")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python3 -m pytest test/registered/unit/mem_cache/test_registry.py::TestMooncakeBuiltinRegistration -v
```

Expected: FAIL with `ValueError` / unregistered backend (no builtin lazy import yet).

- [ ] **Step 3: Implement lazy builtin loading in registry**

In `python/sglang/srt/mem_cache/registry.py`, add:

```python
import importlib

_BUILTIN_RADIX_CACHE_MODULES = {
    "mooncake": "sglang.srt.mem_cache.storage.mooncake_store",
}


def _load_builtin_radix_cache_backend(name: str) -> None:
    module = _BUILTIN_RADIX_CACHE_MODULES.get(name)
    if module is not None and get_radix_cache_factory(name) is None:
        importlib.import_module(module)
```

In `create_tree_cache`, before looking up the factory:

```python
if name:
    _load_builtin_radix_cache_backend(name)
    factory = get_radix_cache_factory(name)
    ...
```

- [ ] **Step 4: Register factory in mooncake_store package**

In `python/sglang/srt/mem_cache/storage/mooncake_store/__init__.py`:

```python
from __future__ import annotations

import logging

from sglang.srt.mem_cache.registry import register_radix_cache_backend

logger = logging.getLogger(__name__)


def _mooncake_factory(ctx):
    from sglang.srt.mem_cache.storage.mooncake_store.mooncake_radix_cache import (
        MooncakeRadixCache,
    )

    return MooncakeRadixCache(
        params=ctx.params,
        model_config=ctx.model_config,
        server_args=ctx.server_args,
        tp_rank=ctx.tp_rank,
        tp_size=ctx.tp_size,
        tp_group=ctx.tp_group,
    )


try:
    register_radix_cache_backend("mooncake", _mooncake_factory)
except ValueError as exc:
    logger.debug("mooncake backend already registered: %s", exc)
```

Note: `MooncakeRadixCache` can be a stub class until Task 4; factory import will fail until that file exists. For Task 1 only, temporarily stub:

```python
# python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_radix_cache.py
class MooncakeRadixCache:
    def __init__(self, **kwargs):
        raise NotImplementedError("MooncakeRadixCache not yet implemented")
```

- [ ] **Step 5: Write failing ServerArgs validation tests**

In `test/registered/unit/server_args/test_server_args.py`, add cases that expect `ValueError` when `radix_cache_backend="mooncake"` is combined with:

- `enable_hierarchical_cache=True`
- `enable_flexkv=True`
- `enable_lmcache=True`
- `pp_size=2`
- `attn_cp_size=2` (or equivalent CP flag)
- `disaggregation_mode="prefill"`
- speculative algorithm set
- `disable_radix_cache=True`

Allow `tp_size` in `{1, 2}` with standard MHA/MLA.

- [ ] **Step 6: Implement `_validate_mooncake_radix_cache()`**

Call from `ServerArgs.__post_init__` when `radix_cache_backend == "mooncake"`. Reject the combinations listed above with clear error messages. Do not require `--enable-hierarchical-cache`.

- [ ] **Step 7: Re-run unit tests**

```bash
python3 -m pytest test/registered/unit/mem_cache/test_registry.py -v
python3 -m pytest test/registered/unit/server_args/test_server_args.py -k mooncake -v
```

Expected: PASS.

- [ ] **Step 8: Commit** (only if user explicitly asks)

```bash
git add python/sglang/srt/mem_cache/registry.py \
  python/sglang/srt/mem_cache/storage/mooncake_store/__init__.py \
  python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_radix_cache.py \
  python/sglang/srt/server_args.py \
  test/registered/unit/mem_cache/test_registry.py \
  test/registered/unit/server_args/test_server_args.py
git commit -m "$(cat <<'EOF'
feat(mooncake): register radix-cache-backend and validate startup flags

EOF
)"
```

---

### Task 2: Shared Mooncake setup and stable page metadata

**Files:**
- Create: `test/registered/unit/mem_cache/test_mooncake_connector.py`
- Modify: `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py`
- Create/Modify helpers used by `mooncake_connector.py`

- [ ] **Step 1: Write failing key + page-meta tests**

Create `test/registered/unit/mem_cache/test_mooncake_connector.py`:

```python
"""Unit tests for MooncakeConnector keys and GPU page metadata."""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestMooncakePageKeys(CustomTestCase):
    def test_keys_differ_by_extra_key_and_tp_rank(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_connector import (
            build_page_keys,
        )
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from array import array

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
        # contiguous slots for one page starting at token index 2
        slots = torch.tensor([2, 3], dtype=torch.int64)
        ptrs, sizes = get_gpu_page_buffer_meta(kvcache, slots)
        # 2 layers * (K + V) = 4 spans for one page
        self.assertEqual(len(ptrs), 4)
        self.assertEqual(len(sizes), 4)

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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python3 -m pytest test/registered/unit/mem_cache/test_mooncake_connector.py -v
```

Expected: FAIL with `ImportError` / missing helpers.

- [ ] **Step 3: Extract shared Mooncake config/setup**

Move `MooncakeStoreConfig`, `_parse_global_segment_size`, and the real/dummy `setup`/`setup_dummy` branching into a reusable helper in the same package (either keep on `MooncakeBaseStore` or a small `mooncake_config.py`). Both `MooncakeStore` and the new connector must call the same loader. Do not change HiCache `MooncakeStore` public behavior.

- [ ] **Step 4: Implement `build_page_keys` and `get_gpu_page_buffer_meta`**

Key format:

```
{namespace}@tp{tp_rank}@{page_hash}
```

Where `namespace` is a stable digest of model identity, revision/config, KV dtype/layout, page size, and TP size; `page_hash` comes from SGLang’s chained `RadixKey.hash_page` / `get_hash_str` chain; `extra_key` is folded into the namespace.

Page meta for MHA NHD layout: for each page, for each layer, append `(k_buffer[layer][first_slot].data_ptr(), page_size * heads * dim * itemsize)` and the matching V span. For MLA: one span per layer from `kv_buffer`. Reject non-page-aligned length or non-contiguous slots.

- [ ] **Step 5: Run connector + regression unit tests**

```bash
python3 -m pytest test/registered/unit/mem_cache/test_mooncake_connector.py -v
python3 -m pytest test/registered/unit/mem_cache/test_mooncake_group_semantics.py \
  test/registered/unit/mem_cache/test_mooncake_standalone_dummy_mamba.py -v
```

Expected: PASS.

---

### Task 3: Direct GPU Mooncake connector

**Files:**
- Modify: `test/registered/unit/mem_cache/test_mooncake_connector.py`
- Create: `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_connector.py`

- [ ] **Step 1: Write failing fake-store I/O tests**

Extend `test_mooncake_connector.py` with a fake `MooncakeDistributedStore` that records `register_buffer`, `batch_is_exist`, `batch_put_from_multi_buffers`, `batch_get_into_multi_buffers`, and `batch_get_replica_desc`. Assert:

1. Each distinct GPU allocation is registered once.
2. Lookup returns only the consecutive page prefix present on every TP rank.
3. Hit counts are MIN-reduced across `tp_group` before reporting.
4. Store/load pass matching logical keys + multi-buffer ptr/size lists.
5. `batch_get_replica_desc` classifies memory vs disk replicas (no latency heuristic).
6. Partial load/store failures do not claim cached tokens.

- [ ] **Step 2: Run tests — expect missing-method failures**

```bash
python3 -m pytest test/registered/unit/mem_cache/test_mooncake_connector.py -v
```

- [ ] **Step 3: Implement `MooncakeConnector`**

Public contract:

```python
class MooncakeConnector:
    def __init__(self, *, kvcache, model_config, server_args, tp_rank, tp_size, tp_group):
        ...

    def lookup(self, key: RadixKey, device_tokens: int) -> tuple[list[str], int]:
        """Return (page_keys_for_missing_tail, hit_token_count)."""

    def load(self, page_keys: list[str], slots: torch.Tensor) -> int:
        """Load pages into GPU slots; return tokens successfully loaded."""

    def store_async(self, rid: str, key: RadixKey, slots: torch.Tensor) -> bool:
        """Queue async store after recording a CUDA event; return False if rejected."""

    def completed_store_rids(self) -> list[str]:
        """Non-blocking drain of finished store rids."""

    def release(self, rid: str) -> None:
        """Drop pending lookup/store state for an aborted request."""

    def wait_for_all_stores(self) -> None:
        """Block until in-flight stores finish (reset/shutdown/evict)."""

    def replica_tiers(self, page_keys: list[str]) -> dict[str, str]:
        """Map key -> 'memory' | 'disk' | 'missing' via batch_get_replica_desc."""

    def reset(self) -> None: ...
    def close(self) -> None: ...
```

Implementation notes:

- Register all GPU KV buffers in `__init__` via `store.register_buffer`.
- One logical Mooncake object per page; object payload is multi-buffer (all layers K/V or MLA).
- Per-TP-rank keys remain separate (`tp_rank` in key).
- Async stores: bounded executor; worker waits on CUDA event then calls Mooncake synchronously.
- MIN-reduce lookup/load page counts on `tp_group`; on disagreement prefer the MIN (safe undercount) and log a warning (or raise in debug).

- [ ] **Step 4: Re-run connector tests**

```bash
python3 -m pytest test/registered/unit/mem_cache/test_mooncake_connector.py -v
```

Expected: PASS with no real Mooncake service and no GPU requirement (fake store + CPU tensors).

---

### Task 4: Radix-cache two-phase loadback and async-store locking

**Files:**
- Create: `test/registered/unit/mem_cache/test_mooncake_radix_cache.py`
- Create/Modify: `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_radix_cache.py`
- Reference: `python/sglang/srt/mem_cache/storage/flexkv/flexkv_radix_cache.py`

- [ ] **Step 1: Write failing radix-cache unit tests**

Cover with a fake connector + `RadixCache.create_simulated` / mock allocator:

1. L1-only hit bypasses Mooncake (`lookup` not called).
2. External consecutive hit sets only the missing-tail `host_hit_length` and snapshots the mutable `RadixKey`.
3. `init_load_back()` allocates slots, retrieves pages, inserts a child node, consumes the marker.
4. Allocation/load failure releases pending state and never inserts a false hit.
5. Async store increments source node lock; completion drain decrements exactly once.
6. `evict()` drains/synchronizes stores before freeing slots.
7. Abort and reset release markers/locks; reset waits for stores before clearing the tree.

Register: `register_cpu_ci(est_time=10, suite="base-a-test-cpu")`.

- [ ] **Step 2: Run — expect FAIL (stub NotImplementedError / missing methods)**

```bash
python3 -m pytest test/registered/unit/mem_cache/test_mooncake_radix_cache.py -v
```

- [ ] **Step 3: Implement `MooncakeRadixCache(RadixCache)`**

Port FlexKV MP-mode pieces only:

- `_LoadBackMarker`
- `match_prefix` / `_mp_match_prefix`
- `init_load_back` / `_allocate_and_load`
- `cache_finished_req` (super + `inc_lock_ref` + `store_async`)
- `check_hicache_events` → `completed_store_rids` + `dec_lock_ref`
- `evict` → drain + `wait_for_all_stores` then `super().evict`
- `release_aborted_request`
- `reset` / `shutdown`

Invariants:

- Page-aligned hits
- `host_hit_length` is only the missing tail
- Snapshot `RadixKey` before `req.fill_ids` grows
- Source nodes locked until store completion
- Partial Mooncake I/O never becomes a radix hit

- [ ] **Step 4: Run radix unit tests**

```bash
python3 -m pytest test/registered/unit/mem_cache/test_mooncake_radix_cache.py -v
```

Expected: PASS.

---

### Task 5: Scheduler lifecycle integration

**Files:**
- Modify: `python/sglang/srt/managers/scheduler.py`
- Add/extend a focused unit test under `test/registered/unit/managers/` (or mem_cache) that mocks scheduler flags

- [ ] **Step 1: Write failing scheduler-gate test**

Prove that when `radix_cache_backend == "mooncake"` and both `enable_hierarchical_cache` and `enable_flexkv` are false, the scheduler still calls `tree_cache.check_hicache_events()` on the event tick, and abort paths call `release_aborted_request` if present.

- [ ] **Step 2: Run — expect FAIL under current OR-gates**

- [ ] **Step 3: Wire scheduler**

During scheduler init:

```python
self.enable_external_radix_cache = (
    self.server_args.radix_cache_backend == "mooncake"
)
```

Extend the periodic drain condition (~line 2796):

```python
if (
    self.enable_hierarchical_cache
    or self.server_args.enable_flexkv
    or self.enable_external_radix_cache
):
    self.tree_cache.check_hicache_events()
```

Mirror the same OR for abort cleanup that currently gates on `enable_hicache_storage` / flexkv as needed so Mooncake markers and locks are released.

Ensure shutdown/reset waits for connector stores before freeing device slots / unregistering buffers.

- [ ] **Step 4: Run focused unit suite**

```bash
python3 -m pytest \
  test/registered/unit/mem_cache/test_registry.py \
  test/registered/unit/mem_cache/test_mooncake_connector.py \
  test/registered/unit/mem_cache/test_mooncake_radix_cache.py \
  -v
```

Expected: PASS (plus the new scheduler unit test).

---

### Task 6: End-to-end DRAM and SSD loadback

**Files:**
- Modify: `test/registered/hicache/test_hicache_storage_mooncake_backend.py` (fixture extensibility)
- Create: `test/registered/hicache/test_mooncake_radix_cache_e2e.py`

- [ ] **Step 1: Extend Mooncake service mixin**

Allow class-level master/client extra args (e.g. `--enable_offload=true`) without changing existing subclasses’ defaults.

- [ ] **Step 2: Write E2E test file (RED first)**

```python
register_cuda_ci(est_time=300, stage="base-b", runner_config="2-gpu-large")
```

Server args for the direct backend (no hierarchical cache):

```python
server_args = {
    "--tp-size": 2,
    "--radix-cache-backend": "mooncake",
    "--page-size": 64,
    "--enable-cache-report": True,
}
env_vars = {
    "MOONCAKE_MASTER": f"127.0.0.1:{cls.mooncake_master_port}",
    "MOONCAKE_PROTOCOL": "tcp",
    "MC_MS_AUTO_DISC": "0",
    "MOONCAKE_DEVICE": "",
    "MOONCAKE_TE_META_DATA_SERVER": f"http://127.0.0.1:{cls.mooncake_metadata_port}/metadata",
    "MOONCAKE_GLOBAL_SEGMENT_SIZE": "67108864",  # 64 MiB — force pressure for SSD case
    "MOONCAKE_ENABLE_SSD_OFFLOAD": "1",
    "MOONCAKE_OFFLOAD_FILE_STORAGE_PATH": cls.ssd_dir,
    "SGLANG_ENABLE_DETERMINISTIC_INFERENCE": "1",
}
```

Start `mooncake_master` with `--enable_offload=true`. Use `CustomTestCase` and defensive `tearDownClass`.

Deterministic tier proof:

1. Prompt A → wait for async stores → flush GPU radix only → poll `batch_get_replica_desc` until A’s keys report memory replica.
2. Reissue A → assert high `cached_tokens` + deterministic output match.
3. Pressure prompts until A’s keys report `is_disk_replica()` (no timing heuristic).
4. Flush GPU → reissue A → assert same cached-prefix/output when pre-load descriptor is disk-backed.

Include:

- MHA TP2 DRAM+SSD class (primary)
- Smaller MLA TP2 direct-loadback class using `DEFAULT_MLA_MODEL_NAME_FOR_TEST`

- [ ] **Step 3: Run E2E before backend is complete — confirm RED**

```bash
python3 -m pytest test/registered/hicache/test_mooncake_radix_cache_e2e.py -v
```

Expected: FAIL at startup / NotImplemented until Tasks 3–5 land. After implementation, expect PASS.

- [ ] **Step 4: Run E2E + HiCache Mooncake regression**

```bash
python3 -m pytest test/registered/hicache/test_mooncake_radix_cache_e2e.py -v
python3 -m pytest test/registered/hicache/test_hicache_storage_mooncake_backend.py::TestMooncakeBackendMLAModel -v
```

Expected: both PASS. Existing HiCache Mooncake path must remain unchanged.

---

### Task 7: Documentation and final verification

**Files:**
- Modify: `python/sglang/srt/mem_cache/storage/mooncake_store/README.md`
- Modify: `docs_new/docs/advanced_features/hicache_design.mdx` (if present; otherwise the closest hicache design doc)

- [ ] **Step 1: Document two modes**

Clearly separate:

1. Legacy HiCache L3: `--enable-hierarchical-cache --hicache-storage-backend mooncake` (SGLang host pool + Mooncake L3)
2. Direct radix backend: `--radix-cache-backend mooncake` (GPU ↔ Mooncake; Mooncake owns DRAM/SSD)

List supported (MHA/MLA, single-node TP1/TP2) and unsupported combinations.

- [ ] **Step 2: Final verification checklist**

```bash
# CPU
python3 -m pytest \
  test/registered/unit/mem_cache/test_registry.py \
  test/registered/unit/mem_cache/test_mooncake_connector.py \
  test/registered/unit/mem_cache/test_mooncake_radix_cache.py \
  test/registered/unit/mem_cache/test_mooncake_group_semantics.py \
  -v

# GPU E2E (when hardware available)
python3 -m pytest test/registered/hicache/test_mooncake_radix_cache_e2e.py -v
python3 -m pytest test/registered/hicache/test_hicache_storage_mooncake_backend.py::TestMooncakeBackendMLAModel -v
```

Self-review:

- [ ] No `HostKVCache` / `HiCacheController` construction in direct mode
- [ ] Mooncake owns DRAM/SSD placement
- [ ] Old HiCache Mooncake path still works
- [ ] TP ranks agree on hits (MIN-reduce)
- [ ] Both tiers proven via `ReplicaDescriptor`, not latency

- [ ] **Step 3: Commit** only when the user explicitly asks; prefer one commit per completed task boundary.

---

## Spec coverage / self-review

| Requirement | Task |
|-------------|------|
| Dedicated `--radix-cache-backend mooncake` (not HiCache) | 1, 4 |
| Mooncake manages DRAM + SSD | 3, 6 |
| Bypass SGLang HostKVCache / HiCacheController | 4 |
| MHA + MLA, single-node TP | 2, 3, 6 |
| Reject hybrid/spec/PD/PP/CP | 1 |
| E2E proves both DRAM and SSD hits | 6 |
| Scheduler drain (PR #29701 pattern) | 5 |
| Preserve existing HiCache Mooncake | 2, 6, 7 |

Placeholder scan: no TBD/TODO left in task steps. Connector method signatures and key invariants are defined before Task 4 consumes them. Type names (`MooncakeConnector`, `MooncakeRadixCache`, `lookup`/`load`/`store_async`) are consistent across tasks.
