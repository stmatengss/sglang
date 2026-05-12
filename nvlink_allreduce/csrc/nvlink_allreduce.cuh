// SPDX-License-Identifier: Apache-2.0
// NVLink AllReduce — CUDA kernels and host-side driver class.
//
// Two algorithms:
//   * one-shot: each block reads all-rank inputs over NVLink, reduces, writes
//     local output.  Latency-optimal for small messages.
//   * two-shot: reduce-scatter into a per-rank temp buffer, barrier, then
//     all-gather from peers.  Bandwidth-optimal for large messages.
//
// Communication primitives are raw CUDA + NVLink P2P only; no NCCL.

#pragma once

#include "nvlink_common.h"

namespace nvlink_ar {

// ---------------------------------------------------------------------------
// Vectorisation helpers.
// ---------------------------------------------------------------------------
template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

template <typename T>
struct packed_t {
  // 16-byte packed load/store type — emits ld.128 / st.128.
  using P = array_t<T, 16 / sizeof(T)>;
  // Accumulator type (always FP32 — bitwise-deterministic across ranks).
  using A = array_t<float, 16 / sizeof(T)>;
};

#define DINLINE __device__ __forceinline__

DINLINE float upcast_s(half v) { return __half2float(v); }
DINLINE float upcast_s(float v) { return v; }
template <typename T>
DINLINE T downcast_s(float val);
template <>
DINLINE half downcast_s(float val) { return __float2half(val); }
template <>
DINLINE float downcast_s(float val) { return val; }

DINLINE half& assign_add(half& a, half b) { a = __hadd(a, b); return a; }
DINLINE float& assign_add(float& a, float b) { return a += b; }

#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
DINLINE float upcast_s(nv_bfloat16 v) { return __bfloat162float(v); }
template <>
DINLINE nv_bfloat16 downcast_s(float val) { return __float2bfloat16(val); }
DINLINE nv_bfloat16& assign_add(nv_bfloat16& a, nv_bfloat16 b) {
  a = __hadd(a, b);
  return a;
}
#endif

template <typename T, int N>
DINLINE array_t<T, N>& packed_assign_add(array_t<T, N>& a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; i++) assign_add(a.data[i], b.data[i]);
  return a;
}

template <typename T, int N>
DINLINE array_t<float, N> upcast(array_t<T, N> v) {
  if constexpr (std::is_same<T, float>::value) {
    return v;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; i++) out.data[i] = upcast_s(v.data[i]);
    return out;
  }
}

template <typename O>
DINLINE O downcast(array_t<float, O::size> v) {
  if constexpr (std::is_same<typename O::type, float>::value) {
    return v;
  } else {
    O out;
#pragma unroll
    for (int i = 0; i < O::size; i++)
      out.data[i] = downcast_s<typename O::type>(v.data[i]);
    return out;
  }
}

// ---------------------------------------------------------------------------
// Cross-GPU release/acquire flag I/O.  Required for sm_70+ memory model.
// ---------------------------------------------------------------------------
static DINLINE void st_flag_release(FlagType* p, FlagType v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(v), "l"(p));
#else
  asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::"r"(v), "l"(p));
#endif
}

static DINLINE FlagType ld_flag_acquire(FlagType* p) {
  FlagType v;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
#else
  asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;" : "=r"(v) : "l"(p));
#endif
  return v;
}

static DINLINE void st_flag_volatile(FlagType* p, FlagType v) {
  asm volatile("st.volatile.global.u32 [%1], %0;" ::"r"(v), "l"(p));
}
static DINLINE FlagType ld_flag_volatile(FlagType* p) {
  FlagType v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

// Multi-GPU barrier.  Each block has its own pair of counters per peer to
// allow back-to-back barriers without false matches (counter%2 toggles).
template <int ngpus, bool is_start, bool need_fence = false>
DINLINE void multi_gpu_barrier(const RankSignals& sg, Signal* self_sg, int rank) {
  if constexpr (!is_start) __syncthreads();
  static_assert(!(is_start && need_fence));
  if (threadIdx.x < ngpus) {
    auto val = self_sg->self_counter[blockIdx.x][threadIdx.x] += 1;
    auto peer_counter_ptr =
        &sg.signals[threadIdx.x]->peer_counter[val % 2][blockIdx.x][rank];
    auto self_counter_ptr =
        &self_sg->peer_counter[val % 2][blockIdx.x][threadIdx.x];
    if constexpr (need_fence) {
      st_flag_release(peer_counter_ptr, val);
      while (ld_flag_acquire(self_counter_ptr) != val)
        ;
    } else {
      st_flag_volatile(peer_counter_ptr, val);
      while (ld_flag_volatile(self_counter_ptr) != val)
        ;
    }
  }
  if constexpr (is_start || need_fence) __syncthreads();
}

template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) packed_assign_add(tmp, upcast(ptrs[i][idx]));
  return downcast<P>(tmp);
}

// One-shot all-reduce: each block fully reads every peer.  Best for small
// payloads where the barrier latency dominates.
template <typename T, int ngpus>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
    cross_device_reduce_1stage(
        RankData* _dp, RankSignals sg, Signal* self_sg, T* __restrict__ result,
        int rank, int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  auto dp = *_dp;
  multi_gpu_barrier<ngpus, true>(sg, self_sg, rank);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] = packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
  }
  multi_gpu_barrier<ngpus, false>(sg, self_sg, rank);
}

template <typename P>
DINLINE P* get_tmp_buf(Signal* sg) {
  return (P*)(((Signal*)sg) + 1);
}

// Two-shot all-reduce: reduce-scatter then all-gather using a per-rank tmp
// buffer in the same IPC region as Signal.  Best for large payloads.
template <typename T, int ngpus>
__global__ void __launch_bounds__(kMaxThreadsPerBlock, 1)
    cross_device_reduce_2stage(
        RankData* _dp, RankSignals sg, Signal* self_sg, T* __restrict__ result,
        int rank, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  int part = size / ngpus;
  int start = rank * part;
  int end = rank == ngpus - 1 ? size : start + part;
  int largest_part = part + size % ngpus;
  const P* ptrs[ngpus];
  P* tmps[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = (const P*)_dp->ptrs[target];
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];
  multi_gpu_barrier<ngpus, true>(sg, self_sg, rank);
  // stage 1: reduce-scatter into local tmp
  for (int idx = start + tid; idx < end; idx += stride) {
    tmp_out[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
  }
  multi_gpu_barrier<ngpus, false, true>(sg, self_sg, rank);
  // stage 2: all-gather across peers; thread-tid matched to stage 1 so the
  // values it wrote are guaranteed visible.
  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int gather_from_rank = ((rank + i) % ngpus);
      if (gather_from_rank == ngpus - 1 || idx < part) {
        int dst_idx = gather_from_rank * part + idx;
        ((P*)result)[dst_idx] = tmps[i][idx];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Host-side driver.  Owns IPC handle bookkeeping and dispatches the right
// kernel for the message size.  No CUDA memory is owned by this class — the
// caller supplies the (already-shared) Signal buffer and a small rank-data
// scratch region.
// ---------------------------------------------------------------------------
class NvlinkAllreduce {
 public:
  int rank_;
  int world_size_;
  bool full_nvlink_;

  RankSignals sg_;
  std::unordered_map<void*, RankData*> buffers_;
  Signal* self_sg_;

  RankData *d_rank_data_base_, *d_rank_data_end_;
  std::map<IpcKey, char*> ipc_handles_;

  NvlinkAllreduce(Signal** signals, void* rank_data, size_t rank_data_sz,
                  int rank, int world_size, bool full_nvlink = true)
      : rank_(rank),
        world_size_(world_size),
        full_nvlink_(full_nvlink),
        self_sg_(signals[rank]),
        d_rank_data_base_(reinterpret_cast<RankData*>(rank_data)),
        d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData)) {
    for (int i = 0; i < world_size_; i++) sg_.signals[i] = signals[i];
  }

  char* open_ipc_handle(const void* ipc_handle) {
    auto [it, fresh] = ipc_handles_.insert({*((IpcKey*)ipc_handle), nullptr});
    if (fresh) {
      char* ipc_ptr;
      NVLINK_CUDA_CHECK(cudaIpcOpenMemHandle(
          (void**)&ipc_ptr, *((const cudaIpcMemHandle_t*)ipc_handle),
          cudaIpcMemLazyEnablePeerAccess));
      it->second = ipc_ptr;
    }
    return it->second;
  }

  void check_rank_data_capacity(size_t num = 1) {
    if (d_rank_data_base_ + num > d_rank_data_end_)
      throw std::runtime_error("rank data buffer overflow");
  }

  // Register a buffer whose peer device pointers are already known on the
  // host (e.g. via IPC handle exchange done in Python).
  void register_buffer(void** ptrs) {
    check_rank_data_capacity();
    RankData data;
    for (int i = 0; i < world_size_; i++) data.ptrs[i] = ptrs[i];
    auto d_data = d_rank_data_base_++;
    NVLINK_CUDA_CHECK(cudaMemcpy(d_data, &data, sizeof(RankData),
                                 cudaMemcpyHostToDevice));
    buffers_[ptrs[rank_]] = d_data;
  }

  template <typename T>
  void allreduce(cudaStream_t stream, T* input, T* output, int size,
                 int threads = kDefaultThreads,
                 int block_limit = kDefaultBlockLimit) {
    auto d = packed_t<T>::P::size;
    if (size % d != 0)
      throw std::runtime_error(
          "input length must be a multiple of " + std::to_string(d));
    if (block_limit > kMaxBlocks)
      throw std::runtime_error("block_limit > kMaxBlocks");

    auto it = buffers_.find(input);
    if (it == buffers_.end())
      throw std::runtime_error("buffer not registered");
    RankData* ptrs = it->second;

    size /= d;
    auto bytes = size * sizeof(typename packed_t<T>::P);
    int blocks = std::min(block_limit, (size + threads - 1) / threads);

    const char* env_algo = std::getenv("NVLINK_AR_ALGO");
    bool force_1stage = false, force_2stage = false;
    if (env_algo) {
      std::string s(env_algo);
      if (s == "1stage" || s == "oneshot") force_1stage = true;
      else if (s == "2stage" || s == "twoshot") force_2stage = true;
      else
        throw std::runtime_error("NVLINK_AR_ALGO must be 1stage|2stage");
    }

#define KL(ngpus, name) \
  name<T, ngpus><<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, output, rank_, size);

#define REDUCE_CASE(ngpus)                                                     \
  case ngpus: {                                                                \
    if (force_1stage) { KL(ngpus, cross_device_reduce_1stage); }               \
    else if (force_2stage) { KL(ngpus, cross_device_reduce_2stage); }          \
    else if (world_size_ == 2) { KL(ngpus, cross_device_reduce_1stage); }      \
    else if (full_nvlink_) {                                                   \
      if ((world_size_ <= kAllReduceGPUSmall &&                                \
           bytes < kAllReduceSmallThreshold) ||                                \
          (world_size_ <= kAllReduceGPULarge &&                                \
           bytes < kAllReduceLargeThreshold)) {                                \
        KL(ngpus, cross_device_reduce_1stage);                                 \
      } else {                                                                 \
        KL(ngpus, cross_device_reduce_2stage);                                 \
      }                                                                        \
    }                                                                          \
    break;                                                                     \
  }
    switch (world_size_) {
      REDUCE_CASE(2)
      REDUCE_CASE(4)
      REDUCE_CASE(6)
      REDUCE_CASE(8)
      default:
        throw std::runtime_error(
            "world_size must be in {2,4,6,8}; got " +
            std::to_string(world_size_));
    }
#undef REDUCE_CASE
#undef KL
  }

  ~NvlinkAllreduce() {
    for (auto& [_, ptr] : ipc_handles_) {
      cudaIpcCloseMemHandle(ptr);
    }
  }
};

}  // namespace nvlink_ar
