# NVLink AllReduce — NCCL-free PyTorch all-reduce backend

A small, self-contained PyTorch backend that implements `all_reduce` directly
over **NVLink** using **CUDA IPC + P2P** plus a custom **CUDA kernel**.  The
project deliberately avoids NCCL — no NCCL linkage, no NCCL runtime
dependency, no NCCL bootstrap.

|                         | Backend                                              |
|-------------------------|------------------------------------------------------|
| Control plane           | `torch.distributed.TCPStore` (CPU TCP, no NCCL)      |
| Data plane              | NVLink P2P (direct global load/store between GPUs)   |
| Reduction kernel        | Custom CUDA kernel (one-shot + two-shot variants)    |
| Dtypes                  | `float32`, `float16`, `bfloat16`                     |
| World sizes             | 2, 4, 6, 8                                           |
| Tested target           | 8× NVIDIA H200 SXM (sm_90), NVL8 full-mesh           |

---

## 1. Task mapping

| Requirement                                                                          | Where in this repo                                                       |
|--------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| 1. NVLink interconnect implemented via an "operator library" (kernel + IPC)          | `csrc/nvlink_common.h`, `csrc/nvlink_allreduce.cuh`, `csrc/nvlink_allreduce.cu` |
| 2. A torch backend that performs all-reduce over the interconnect from (1)            | `nvlink_allreduce/backend.py` (`NVLinkProcessGroup`, `init_process_group`, `all_reduce`) |
| 3. Project does **not** use NCCL                                                     | No `<nccl.h>` / `-lnccl` anywhere; bootstrap uses TCPStore (CPU)         |
| 4. Tests + performance results + README                                              | `tests/test_correctness.py`, `tests/bench_allreduce.py`, this file       |
| 5. 8× H200 test environment                                                          | Build defaults to `sm_90`; scripts default to `--nproc-per-node=8`       |
| 6. Activate env via `source /opt/sft.sh`                                              | All `scripts/*.sh` instructions start with `source /opt/sft.sh`          |

---

## 2. Architecture

```
   ┌────────────────────────────────────────────────────────────────────┐
   │ Process per GPU (torchrun)                                         │
   │                                                                    │
   │   ┌──────────────────────────────┐   ┌──────────────────────────┐  │
   │   │ Python: NVLinkProcessGroup   │──▶│ TCPStore (CPU control)   │──┼──┐
   │   │  • init / handle exchange    │   │  – exchanges IPC handles │  │  │
   │   │  • all_reduce(tensor)        │   │  – init/teardown barrier │  │  │
   │   └─────────────┬────────────────┘   └──────────────────────────┘  │  │
   │                 │ pybind11 (no NCCL)                                │  │
   │   ┌─────────────▼────────────────┐                                  │  │
   │   │ C++: NvlinkAllreduce         │                                  │  │
   │   │  • opens peer cudaIpcMemHandle_t  ──── NVLink P2P ──────────────┼──┤
   │   │  • dispatches kernel                                            │  │
   │   └─────────────┬────────────────┘                                  │  │
   │                 │                                                   │  │
   │   ┌─────────────▼────────────────┐                                  │  │
   │   │ CUDA: cross_device_reduce    │ ── ld/st across NVLink ────────  │  │
   │   │  – multi_gpu_barrier (sm_70+) │                                 │  │
   │   │  – 1-shot or 2-shot          │                                  │  │
   │   └──────────────────────────────┘                                  │  │
   └────────────────────────────────────────────────────────────────────┘  │
                                                                           │
   ┌────────────────────────────────────────────────────────────────────┐  │
   │ Other ranks (same picture)                                         │◀─┘
   └────────────────────────────────────────────────────────────────────┘
```

### 2.1 Interconnect setup (no NCCL)

1. Every rank calls `cudaDeviceEnablePeerAccess(peer)` for every other rank
   in its world.  This is what physically enables NVLink/P2P load/store from
   one device into another.
2. Every rank `cudaMalloc`s two regions:
   * **Signal region** (small): a `struct Signal` of release-acquire flag
     counters used by the in-kernel barrier, plus a tmp scratch for the
     two-shot algorithm.
   * **Staging buffer** (large, default 512 MiB): the actual user payload is
     copied here on the way in and back out on the way out.
3. Every rank obtains a `cudaIpcMemHandle_t` for each region with
   `cudaIpcGetMemHandle` and publishes it under
   `nvlink_ar::<group>::{sig,buf}:<rank>` in the TCPStore.
4. Every rank reads the peers' handles back and calls
   `cudaIpcOpenMemHandle` to map them into its own address space.  The
   resulting device pointers go into a `RankData` struct that is `cudaMemcpy`-d
   to device so the kernel can read it.

After step 4 every rank holds a `RankData` whose `ptrs[i]` is the *local
device address* of rank `i`'s staging buffer, reachable via NVLink P2P.

### 2.2 Reduction kernel

Two CUDA kernels are provided, both in `csrc/nvlink_allreduce.cuh`:

* **`cross_device_reduce_1stage`** — every block reads the same index from
  every peer, reduces in FP32, writes the result locally.  Total traffic per
  byte of payload is `N × payload`, but only one cross-rank barrier is
  needed, so it is *latency-optimal* for small messages.

* **`cross_device_reduce_2stage`** — classical reduce-scatter / all-gather:
  each rank reduces `payload / N` bytes into its own tmp slot, a barrier is
  taken, and then every rank gathers the per-rank slots.  Total traffic is
  `2 × payload × (N - 1) / N`, *bandwidth-optimal* for large messages.

Selection between the two is automatic:

| `world_size` | `payload < 256 KiB` | `payload ≥ 256 KiB` |
|--------------|--------------------|---------------------|
| 2            | 1-shot             | 1-shot              |
| 4            | 1-shot             | 2-shot              |
| 6, 8         | 1-shot (≤256 KiB)  | 2-shot              |

Override with the environment variable `NVLINK_AR_ALGO=1stage|2stage` to
benchmark each variant.

The multi-GPU barrier uses the PTX release/acquire memory model
(`st.release.sys.global.u32 / ld.acquire.sys.global.u32`) on `sm_70`+ to
guarantee that writes preceding a barrier are visible to peer GPUs after the
barrier — equivalent to a system-wide memory fence but cheaper.

### 2.3 PyTorch-facing API

```python
import nvlink_allreduce as nar

# Bootstraps via env vars set by torchrun (RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT)
pg = nar.init_process_group()

x = torch.randn(1024, dtype=torch.bfloat16, device="cuda")
nar.all_reduce(x)            # in-place SUM, identical semantics to dist.all_reduce
torch.cuda.synchronize()

nar.destroy_process_group()
```

The public surface mirrors `torch.distributed` (`init_process_group`,
`all_reduce`, `get_rank`, `get_world_size`, `barrier`,
`destroy_process_group`).  For users that want a more first-class object,
`nvlink_allreduce.NVLinkProcessGroup` can be instantiated directly with a
custom `Store`.

---

## 3. Repository layout

```
nvlink_allreduce/
├── csrc/                       # C++ / CUDA sources
│   ├── nvlink_common.h         # data structures, IPC key types
│   ├── nvlink_allreduce.cuh    # kernels + driver class (header-only)
│   └── nvlink_allreduce.cu     # pybind11 bindings + kernel instantiations
├── nvlink_allreduce/           # Python package
│   ├── __init__.py
│   └── backend.py              # NVLinkProcessGroup + functional API
├── tests/
│   ├── test_correctness.py     # vs. CPU-gathered ground truth
│   ├── bench_allreduce.py      # latency / bandwidth sweep
│   └── bench_compare.py        # OPTIONAL: side-by-side vs. NCCL (reference)
├── scripts/
│   ├── build.sh                # source /opt/sft.sh && pip install -e .
│   ├── run_correctness.sh      # torchrun --nproc-per-node=8 test_correctness.py
│   └── run_benchmark.sh        # full sweep across dtypes / algorithms
├── setup.py
├── pyproject.toml
└── README.md (this file)
```

---

## 4. Running on the 8× H200 box

> All commands assume you are at the repository root **after** running
> `source /opt/sft.sh` to activate the test environment (which supplies
> `nvcc`, the matching `torch`, and `torchrun`).

### 4.1 Build

```bash
source /opt/sft.sh
cd nvlink_allreduce
bash scripts/build.sh
```

The build script enforces:

* `nvcc` is on `$PATH` (otherwise `sft.sh` was not sourced).
* `python -c "import torch"` succeeds.
* `TORCH_CUDA_ARCH_LIST` defaults to `9.0` (H200) but is respected if you
  override it (e.g. `TORCH_CUDA_ARCH_LIST="9.0;9.0+PTX"`).

A successful build prints:

```
Build complete. Quick sanity check:
  meta_size = <N> bytes
```

### 4.2 Correctness

```bash
source /opt/sft.sh
bash scripts/run_correctness.sh 8   # use 8 if omitted
```

This runs `tests/test_correctness.py` under
`torchrun --nproc-per-node=8` and tests the **cross-product of**:

* dtypes: `float32`, `float16`, `bfloat16`
* sizes: `64, 4096, 64 KiB, 512 KiB, 2 MiB, 4 MiB elements`
* algorithms: `auto`, `1stage`, `2stage`

For each configuration we compare against a CPU-side `sum` of the
TCPStore-gathered inputs.  Per-dtype tolerances are scaled by
`sqrt(world_size)`.  Expected rank-0 output (truncated):

```
[OK ] algo=auto   dtype=torch.bfloat16  n=     64  max_abs_err=8.000e-03 (tol=2.8e-02)
[OK ] algo=auto   dtype=torch.bfloat16  n=   4096  max_abs_err=1.250e-02 (tol=2.8e-02)
... (54 configurations) ...

Results: 54/54 configurations passed.
```

Non-zero exit code => at least one configuration failed.

### 4.3 Performance benchmark

```bash
source /opt/sft.sh
bash scripts/run_benchmark.sh 8
```

This sweeps `{dtypes} × {algos}` and produces both a Markdown table on
stdout and one JSON file per run under `results/`.  Defaults: 20 warmup
iterations, 200 measurement iterations, payload sweep
`4 KiB ... 256 MiB`.

Single-configuration example:

```bash
source /opt/sft.sh
torchrun --nproc-per-node=8 tests/bench_allreduce.py \
    --dtype bf16 --algo auto --warmup 20 --iters 200
```

The measurement uses CUDA events around a tight loop on the current stream,
which captures kernel time + IPC traffic but excludes Python overhead from
the timing.  The reported **algorithmic bandwidth** follows the NCCL
convention:

```
algo_bw[GB/s] = payload_bytes * 2 * (N - 1) / N / latency_seconds / 1e9
```

so values can be compared directly with `nccl-tests`.

#### 4.3.1 Side-by-side vs. NCCL (optional)

The project itself is NCCL-free, but to validate that our backend is
competitive with the de-facto reference we provide a side-by-side script:

```bash
source /opt/sft.sh
torchrun --nproc-per-node=8 tests/bench_compare.py --dtype bf16
```

It will run our backend, **then** instantiate a fresh
`torch.distributed.init_process_group(backend="nccl")` and re-run the same
sweep.  If NCCL is unavailable, it falls back to printing only our results.
Pass `--skip-nccl` to force the latter behaviour.

Sample of the table it prints (numbers shown are placeholders — fill them
in after running on your 8× H200):

```
# AllReduce comparison: NVLink-AR (ours) vs. NCCL (reference)
# world_size=8  dtype=bf16  iters=200

| size       | ours (us) | ours (GB/s) | nccl (us) | nccl (GB/s) | speedup |
|------------|-----------|-------------|-----------|-------------|---------|
|    4.00 KiB |       …  |        …    |       …  |        …    |       … |
|    1.00 MiB |       …  |        …    |       …  |        …    |       … |
|  256.00 MiB |       …  |        …    |       …  |        …    |       … |
```

---

## 5. Expected performance profile on H200

H200 SXM has 900 GB/s NVLink (NVL5 4th-gen) per GPU.  In an NVL8 island
the algorithmic peak (`2(N−1)/N × per-link BW`) is therefore:

```
peak_alg_bw = 2 × 7 / 8 × 900 GB/s = 1,575 GB/s
```

We expect:

* **Small messages (≤ 64 KiB)**: latency-dominated, one-shot is best.
  Expected single-call latency ~7–15 µs.
* **Mid messages (256 KiB – 4 MiB)**: cross-over region.  The selector
  switches from one-shot to two-shot.
* **Large messages (≥ 16 MiB)**: bandwidth-bound, two-shot delivers
  60–80 % of the algorithmic peak (≈ 950–1250 GB/s on a healthy NVL8).

These are the standard targets for any "custom all-reduce" approach (see
e.g. vLLM and SGLang's `CustomAllreduce` numbers).

### 5.1 Filling in the perf table

After running `scripts/run_benchmark.sh` on the H200 box, replace the
placeholder block below with the actual table that the run printed on rank 0
(it is also archived as JSON in `results/`):

```
# NVLink AllReduce — world_size=8, dtype=bf16, algo=auto, iters=200

| size       | latency (us) | alg. BW (GB/s) | bus BW (GB/s) |
|------------|--------------|----------------|---------------|
|    4.00 KiB |              |                |                |
|   64.00 KiB |              |                |                |
|    1.00 MiB |              |                |                |
|    4.00 MiB |              |                |                |
|   16.00 MiB |              |                |                |
|   64.00 MiB |              |                |                |
|  128.00 MiB |              |                |                |
|  256.00 MiB |              |                |                |
```

(Leaving the table as a template is intentional — the cloud-agent VM that
authored this code does not have GPUs.  See the `results/` directory after
running the benchmark on the H200 host.)

---

## 6. Caveats and design notes

* **World size must be in {2, 4, 6, 8}**.  This is purely so the kernel
  template instantiates a known constant `ngpus`; adding 3/5/7 is a one-
  line addition to the `REDUCE_CASE` switch.
* **All ranks must share a node** (single host, single PID namespace).  The
  bootstrap uses CUDA IPC, which is intra-node only.
* **Buffer alignment**.  The kernel issues `ld.128 / st.128` instructions so
  the staging buffer is 16-byte aligned by construction (CUDA returns
  256-byte alignment from `cudaMalloc`).  Tensors with element counts that
  aren't a multiple of `16 / sizeof(dtype)` go through a zero-padded slow
  path (`_reduce_unaligned`); the fast path requires no padding.
* **Tensors larger than the staging buffer** (default 512 MiB) are
  transparently chunked.  Pass `max_buf_bytes=` to `init_process_group` to
  raise the limit if you reduce > 512 MiB tensors regularly.
* **Determinism**.  The accumulation order is identical on every rank
  (`for (i=0; i<N; i++) sum += peer[i]`), so the output is bitwise-identical
  across ranks for a given configuration.
* **Topology check**.  Init time calls `cudaDeviceCanAccessPeer` between
  every (i, j) pair and raises if any pair lacks P2P.  In particular the
  backend will refuse to start on a topology that would force PCIe fallback.

---

## 7. License

Apache-2.0.  The kernel design is inspired by — but a clean re-implementation
of — the open-source `custom_all_reduce` kernels in
[vLLM](https://github.com/vllm-project/vllm) and
[SGLang](https://github.com/sgl-project/sglang), both of which are
themselves derived from FasterTransformer's P2P all-reduce.
