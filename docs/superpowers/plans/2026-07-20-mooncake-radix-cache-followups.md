# Mooncake Radix Cache Follow-up Implementation Plan

**Goal:** Evolve the direct Mooncake radix-cache backend from a validated MVP into a production-ready backend with HiCache-grade lifecycle safety, asynchronous scheduling, observability, and incremental compatibility, while keeping the direct GPU-to-Mooncake data path independent from `HostKVCache` and `HiCacheController`.

**Baseline:** Start from `feat/mooncake-radix-cache` after merge commit `d72454301003651a52ded31205794203757ceada`. The baseline supports direct GPU Mooncake store/load for MHA and MLA, single-node TP1/TP2, page-aligned keys, TP-min read agreement, and asynchronous stores.

**Non-goals for the first three phases:** Do not replace the legacy HiCache Mooncake backend, do not route the direct backend through host memory, and do not enable PP, CP, PD disaggregation, speculative decoding, SWA, or Mamba until the lifecycle and performance foundations are complete.

## Design principles

1. Keep changes scoped to the registered Mooncake radix backend and generic capability hooks that are necessary for external radix caches.
2. Preserve existing HiCache, FlexKV, LMCache, and built-in radix-cache behavior.
3. Never release or reuse GPU slots while a Mooncake operation can still access their pointers.
4. Make operation admission and visible results consistent across all participating attention ranks.
5. Keep scheduler hot paths non-blocking except where correctness requires an explicit bounded wait.
6. Add metrics before tuning concurrency so performance changes are measurable.
7. Land each phase as an independently testable change; avoid combining compatibility expansion with core lifecycle refactors.

## Success metrics

### Correctness

- No stale-pointer access during abort, eviction, reset, or shutdown.
- All TP ranks make the same store/load admission decision.
- Partial and failed operations never advertise an invalid prefix.
- Cache namespace changes whenever stored KV becomes incompatible.
- Mooncake outage degrades to local radix-cache behavior instead of hanging serving.

### Performance

- No global wait for unrelated stores during ordinary eviction.
- Remote-hit lookup and load overlap waiting-queue time where possible.
- Store write amplification is measurable and reduced by missing-page writes.
- P95 remote-hit TTFT is lower than the synchronous baseline for long prompts.
- Miss-path throughput remains within 3% of the direct-backend-disabled baseline.

### Operations

- Metrics expose hit/miss tokens, latency, bytes, queue depth, rejection, timeout, and failure counts.
- SSD and DRAM placement are verified by replica descriptors rather than timing.
- Cross-instance cache reuse is covered by an automated E2E test.

---

## Phase 0: Establish baselines and fault-injection fixtures

**Purpose:** Make later safety and performance changes measurable without changing production behavior.

**Files:**

- Modify: `test/registered/unit/mem_cache/test_mooncake_connector.py`
- Modify: `test/registered/unit/mem_cache/test_mooncake_radix_cache.py`
- Create: `test/registered/unit/mem_cache/mooncake_test_utils.py`
- Create: `benchmark/mooncake/bench_mooncake_radix_cache.py`

### Steps

1. Extract a thread-safe fake Mooncake store with controllable gates for lookup, get, and put.
2. Support injected delay, timeout, exception, partial result, and per-rank result divergence.
3. Record operation start/end order, page keys, pointers, and released slots.
4. Add a small benchmark for:
   - lookup latency versus prefix length;
   - pointer metadata construction versus page/layer count;
   - store queue saturation;
   - repeated shared-prefix write amplification.
5. Record baseline results for 1K, 8K, 32K, and 128K token prompts where hardware permits.

### Acceptance

- Fixtures can deterministically hold an asynchronous put while abort and eviction execute.
- Benchmark emits JSON or CSV suitable for before/after comparison.
- No production code changes in this phase.

---

## Phase 1: Make asynchronous store lifecycle safe

**Priority:** P0

**Purpose:** Ensure node locks and GPU slots outlive every operation that may dereference them.

**Files:**

- Modify: `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_connector.py`
- Modify: `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_radix_cache.py`
- Modify: `test/registered/unit/mem_cache/test_mooncake_connector.py`
- Modify: `test/registered/unit/mem_cache/test_mooncake_radix_cache.py`

### Design

Introduce a `MooncakeStoreOperation` record with:

- operation ID independent of request ID;
- request ID;
- page keys and source node;
- future and CUDA readiness event;
- state: `reserved`, `running`, `succeeded`, `failed`, `cancel_requested`, `cancelled`;
- one-shot completion/finalization guard;
- timestamps and error category.

The radix cache owns the node lock until operation finalization. An abort requests cancellation but does not directly unlock a running operation. The completion drain is the only normal owner of lock release.

### Steps

1. Write failing tests for abort before worker start and abort after worker start.
2. Write a failing test that attempts eviction while the fake store still reads source pointers.
3. Replace `_store_futures: dict[rid, Future]` with operation records keyed by operation ID.
4. Make completion finalization idempotent.
5. Change `release_aborted_request()` to:
   - remove pending load markers;
   - request cancellation;
   - retain the node lock when cancellation cannot complete immediately;
   - let the completion drain release the lock.
6. Ensure reset and shutdown drain or boundedly abandon operations without double-unlocking.
7. Add assertions for negative `lock_ref`, duplicate completion, and leaked operations in tests.

### Acceptance

- Abort + running store + eviction stress passes under repeated execution.
- Every accepted operation releases its node lock exactly once.
- A cancelled-before-start operation never calls Mooncake put.
- A running operation never observes slots that have been freed or reused.

---

## Phase 2: Coordinate TP admission and add bounded failure handling

**Priority:** P0

**Purpose:** Prevent per-rank queue divergence and serving hangs during Mooncake failures.

**Files:**

- Modify: `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_connector.py`
- Modify: `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_radix_cache.py`
- Modify: `python/sglang/srt/server_args.py`
- Modify: `test/registered/unit/mem_cache/test_mooncake_connector.py`
- Modify: `test/registered/unit/server_args/test_server_args.py`

### Steps

1. Split store admission into local reservation and collective commit.
2. Use the attention/TP group MIN result so all ranks accept or reject together.
3. Release local reservations on every rank when collective admission fails.
4. Add bounded configuration values for lookup, load, store, drain, and shutdown timeout.
5. Add a small retry policy for transient Mooncake errors with exponential backoff and a hard total deadline.
6. Add a circuit breaker:
   - closed: normal operation;
   - open: bypass Mooncake and use local radix cache;
   - half-open: probe with a limited operation.
7. Never retry a store after its source slots are no longer protected.
8. Categorize errors as timeout, unavailable, partial, invalid response, and local validation.

### Acceptance

- A full queue on one TP rank causes all ranks to reject the operation.
- Mooncake service loss does not block generation, eviction, flush, or shutdown indefinitely.
- Local L1 radix hits continue while the circuit breaker is open.
- Retry count and deadlines are covered with deterministic tests.

---

## Phase 3: Add direct-backend observability

**Priority:** P0/P1

**Purpose:** Provide the measurements needed to validate subsequent performance work.

**Files:**

- Modify: `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_connector.py`
- Modify: `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_radix_cache.py`
- Modify: the existing cache/storage metrics collector modules selected by the current upstream API
- Modify: unit tests for metric registration and updates

### Metrics

- lookup/load/store operation totals by result;
- lookup/load/store latency histograms;
- hit, miss, loaded, and stored token/page counts;
- submitted and newly-written bytes;
- pending/running operation gauges;
- backpressure rejection and collective-admission rejection;
- timeout, retry, circuit-open, and partial-result counts;
- DRAM, SSD, and missing replica counts when descriptors are available;
- time spent blocked in scheduler, eviction, flush, reset, and shutdown;
- write amplification ratio.

### Steps

1. Reuse existing collector registration patterns to avoid duplicate Prometheus collectors.
2. Attach rank and backend labels consistent with HiCache storage metrics.
3. Avoid high-cardinality labels such as request ID, page key, or model path.
4. Add structured debug logs carrying operation ID and summarized page counts.
5. Add tests proving metrics update on success, partial result, timeout, cancellation, and rejection.

### Acceptance

- Metrics are disabled with negligible overhead when serving metrics are disabled.
- No request/page identifiers appear as metric labels.
- A single E2E round trip exposes remote-hit tokens and nonzero Mooncake transfer bytes.

---

## Phase 4: Move lookup and load out of the scheduler critical path

**Priority:** P1

**Purpose:** Overlap remote cache work with waiting-queue time and reduce cache-hit TTFT.

**Files:**

- Modify: `python/sglang/srt/managers/scheduler.py`
- Modify: `python/sglang/srt/mem_cache/base_prefix_cache.py` only if a generic capability hook is required
- Modify: `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_radix_cache.py`
- Modify: `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_connector.py`
- Add focused scheduler and connector tests

### Design

Follow the useful parts of HiCache's two-stage prefetch contract without constructing host-cache state:

1. Request enters waiting queue.
2. Mooncake lookup begins asynchronously.
3. If useful and within quota, destination GPU pages are reserved and load begins.
4. Scheduler polls readiness without blocking.
5. Completed pages are inserted into the GPU radix tree immediately before request admission.
6. Timeout, preemption, or abort cancels/revokes the operation and frees reserved pages safely.

### Steps

1. Define capability methods such as `supports_external_prefetch`, `start_prefetch`, `check_prefetch`, and `terminate_prefetch`; avoid hard-coding the backend name in new scheduler paths.
2. Add configurable minimum remote-hit threshold and maximum tokens reserved for prefetch.
3. Add `best_effort`, `wait_complete`, and `timeout` stop policies.
4. Ensure partial consecutive page loads can be committed safely.
5. Keep synchronous `init_load_back` as a temporary fallback behind the same operation abstraction.
6. Compare P50/P95 TTFT against the Phase 0 baseline.

### Acceptance

- Scheduler does not block on ordinary Mooncake network I/O.
- Waiting requests with unfinished prefetch are skipped without starving unrelated requests.
- Reserved GPU pages are always released after timeout, preemption, or abort.
- Long-prompt remote-hit P95 TTFT improves measurably over the synchronous baseline.

---

## Phase 5: Remove global store waits from eviction

**Priority:** P1

**Purpose:** Eliminate head-of-line blocking caused by waiting for every in-flight store before freeing any cache node.

**Files:**

- Modify: `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_radix_cache.py`
- Modify: `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_connector.py`
- Modify: related radix-cache unit tests

### Steps

1. Track the exact nodes/pages protected by each store operation.
2. Let ordinary radix eviction skip protected leaves and continue through other candidates.
3. When memory remains insufficient, wait only for the earliest relevant operation and only up to a bounded deadline.
4. Re-run eviction after each completion instead of draining all operations.
5. Preserve full drain behavior only for reset/shutdown paths that explicitly require it.
6. Record eviction wait time and the number of skipped protected nodes.

### Acceptance

- A delayed store for one prefix does not prevent eviction of unrelated prefixes.
- No operation reads an evicted source page.
- Under store saturation, scheduler stall is bounded by configured policy.

---

## Phase 6: Reduce write amplification and support incremental persistence

**Priority:** P1

**Purpose:** Avoid rewriting shared prefixes and make long/chunked requests useful before final completion.

**Files:**

- Modify: `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_connector.py`
- Modify: `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_radix_cache.py`
- Modify: `python/sglang/srt/server_args.py`
- Add unit and E2E tests for repeated prefixes and chunked requests

### Steps

1. Calculate page keys incrementally and cache hash-chain state on radix nodes where practical.
2. Before store, identify the consecutive missing tail and submit only missing pages.
3. Coalesce concurrent writes for identical page keys with a singleflight table.
4. Add write policies:
   - `always` for current behavior;
   - `missing_only` as the intended default after validation;
   - `selective` based on prefix hit count or minimum token length.
5. Hook page-aligned additions from `cache_unfinished_req` so long/chunked prefills can persist incrementally.
6. Ensure the final request store does not duplicate pages already persisted incrementally.
7. Bound metadata and I/O batch sizes for very long contexts.

### Acceptance

- Repeated shared prompts do not rewrite already-present pages under `missing_only`.
- Concurrent identical requests perform one logical write per page and all waiters complete correctly.
- Chunked requests expose completed page-aligned prefixes after safe commit.
- Write amplification is substantially lower for shared system-prompt and multi-turn workloads.

---

## Phase 7: Optimize page metadata and allocation compatibility

**Priority:** P1/P2

**Purpose:** Remove avoidable CPU synchronization and handle realistic allocator output safely.

**Files:**

- Modify: `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_connector.py`
- Modify: allocator/cache-pool interfaces only through a narrowly scoped optional capability
- Add GPU tests for fragmented allocation and large layer/page counts

### Steps

1. Cache immutable buffer base pointers, strides, page sizes, and per-layer widths at connector construction.
2. Replace repeated `slots.detach().cpu().tolist()` metadata construction where possible.
3. Add a page-run representation generated by the allocator or radix cache.
4. Support multiple contiguous runs in one logical operation.
5. If Mooncake bindings cannot express fragmented pages directly, allocate page-aligned contiguous runs or use a bounded staging strategy explicitly rather than failing late.
6. Chunk multi-buffer metadata to cap Python object count and Mooncake RPC size.

### Acceptance

- Fragmented allocator output either works or is rejected before reserving unsafe resources.
- Metadata-construction time scales with page runs rather than tokens where possible.
- MHA and MLA pointer/size ordering remains unchanged.

---

## Phase 8: Namespace integrity, retention, and runtime control

**Priority:** P2

**Purpose:** Prevent stale KV reuse and make external storage manageable in long-running deployments.

**Files:**

- Modify: `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_connector.py`
- Modify: `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_radix_cache.py`
- Modify: `python/sglang/srt/server_args.py`
- Modify: scheduler/admin request handling only through generic external-cache capabilities
- Update Mooncake README

### Steps

1. Add explicit `--mooncake-cache-namespace` and `--mooncake-cache-version` overrides.
2. Prefer immutable model revision or weight-manifest fingerprint over mutable path/revision strings.
3. Include all KV-affecting parameters in namespace identity: model weights, KV dtype/layout, page size, parallel partitioning, relevant model config, and adapter/cache salt.
4. Add optional namespace TTL and quota configuration.
5. Implement capability-based clear, attach, detach, and health/status operations where supported.
6. Define behavior when old Mooncake bindings lack remove, TTL, or descriptor APIs.
7. Document rolling upgrade and namespace migration procedures.

### Acceptance

- Replacing model weights without changing the serving path cannot silently reuse old KV.
- Two compatible serving instances can intentionally share a namespace.
- Operators can inspect and clear the direct backend without restarting the server where Mooncake supports it.

---

## Phase 9: Strengthen E2E and chaos coverage

**Priority:** P1/P2

**Files:**

- Modify: `test/registered/hicache/test_mooncake_radix_cache_e2e.py`
- Add helper processes for multi-instance tests
- Reuse: `test/registered/hicache/test_hicache_storage_mooncake_backend.py` fixtures without altering legacy behavior

### Required scenarios

1. **MHA TP2 DRAM:** write, flush local radix, remote load, identical output.
2. **MLA TP2 DRAM:** same contract with an MLA model.
3. **SSD:** enable Mooncake SSD offload, pressure DRAM, poll replica descriptors until pages report disk, flush local radix, then load and verify output.
4. **Cross-instance:** server A writes; server B with compatible namespace loads without warming locally.
5. **Namespace isolation:** different model revision, adapter salt, KV dtype, or explicit version does not hit.
6. **Abort race:** abort while lookup, load, and store are independently blocked.
7. **Memory pressure:** eviction proceeds around protected store nodes.
8. **Partial operation:** only the consecutive successful prefix is reported.
9. **Service interruption:** stop/restart metadata or master service while generation continues locally.
10. **Concurrency:** many requests share prefixes while pending queues saturate.

### Acceptance

- Tier tests use `ReplicaDescriptor`, never latency heuristics.
- Input and output correctness are checked in addition to cached-token counts.
- Failure tests have bounded timeouts and deterministic cleanup.
- Legacy HiCache Mooncake E2E remains green.

---

## Phase 10: Expand compatibility in isolated increments

Do not combine these into one patch. Each item requires an explicit topology/key design, startup validation update, and dedicated E2E coverage.

### Recommended order

1. Independent single-node serving instances sharing Mooncake.
2. Chunked prefill and request retraction.
3. Multi-node TP.
4. Pipeline parallelism.
5. Context parallelism and DP attention groups.
6. Prefill/decode disaggregation.
7. Speculative decoding target/draft cache separation.
8. FP8/FP4 and other KV layouts.
9. SWA, Mamba, and hybrid cache families.

### Gate for each compatibility item

- Define rank-aware page-key ownership.
- Define collective group and failure agreement semantics.
- Define namespace changes.
- Define abort/reset/shutdown behavior.
- Add CPU contract tests and the smallest representative GPU E2E.
- Remove only the corresponding startup rejection after all tests pass.

---

## Recommended pull request sequence

1. **PR 1 — lifecycle safety:** Phases 0 and 1.
2. **PR 2 — TP reliability:** Phase 2.
3. **PR 3 — metrics:** Phase 3.
4. **PR 4 — asynchronous remote prefetch/load:** Phase 4.
5. **PR 5 — non-blocking eviction:** Phase 5.
6. **PR 6 — missing-page and incremental stores:** Phase 6.
7. **PR 7 — metadata/allocation optimization:** Phase 7.
8. **PR 8 — namespace and runtime operations:** Phase 8.
9. **PR 9 — SSD, cross-instance, and chaos E2E:** Phase 9.
10. **Later PRs — one compatibility feature each:** Phase 10.

## Verification commands

Run the smallest relevant group after every PR and the full matrix before changing defaults:

```bash
# Direct Mooncake unit contracts
python -m pytest \
  test/registered/unit/mem_cache/test_mooncake_connector.py \
  test/registered/unit/mem_cache/test_mooncake_radix_cache.py \
  test/registered/unit/mem_cache/test_registry.py \
  test/registered/unit/server_args/test_server_args.py -k mooncake -v

# Legacy HiCache/Mooncake and nearby cache regressions
python -m pytest \
  test/registered/unit/mem_cache/test_hiradix_cache_unit.py \
  test/registered/unit/mem_cache/test_hiradix_pp_sync_drain.py \
  test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py \
  test/registered/unit/mem_cache/test_mooncake_group_semantics.py -v

# GPU serving E2E
python -m pytest \
  test/registered/hicache/test_mooncake_radix_cache_e2e.py -v -s

# Legacy Mooncake storage E2E
python -m pytest \
  test/registered/hicache/test_hicache_storage_mooncake_backend.py -v -s
```

Before merging each PR:

- run `git diff --check`;
- compile modified Python files;
- confirm direct mode still constructs no `HostKVCache` or `HiCacheController`;
- confirm legacy HiCache Mooncake files have no unintended behavioral diff;
- compare benchmark results against the saved Phase 0 baseline;
- document any unsupported Mooncake binding version or feature fallback.

## First implementation milestone

The first milestone is complete only after Phases 0–3 land. At that point the direct backend should be safe under abort and eviction, consistent across TP ranks, bounded under service failure, and observable enough to evaluate asynchronous scheduling work. Compatibility flags should remain unchanged until that milestone is stable.
