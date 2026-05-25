// SPDX-License-Identifier: Apache-2.0
// NVLink AllReduce — common header.
//
// This file defines the data structures used by the NVLink interconnect
// library and the all-reduce kernels.  The library is intentionally
// self-contained and does NOT depend on NCCL or any other collective library.
// All cross-GPU communication is performed via:
//   * cudaIpc{Get,Open}MemHandle  — to share device memory across processes
//   * cudaDeviceEnablePeerAccess  — to enable direct NVLink P2P load/store
//   * a custom CUDA kernel        — to perform the reduce + barrier
//
// The kernel design follows the well-known one-shot / two-shot algorithms
// used by vLLM's custom_all_reduce and SGLang's allreduce, both of which
// in turn are derivatives of the FasterTransformer "p2p allreduce".
//
// The barrier uses release/acquire memory ordering (sm_70+) which lets us
// avoid expensive system-wide fences in the steady state.

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace nvlink_ar {

// ---------------------------------------------------------------------------
// Tuning knobs.  These have been picked to match A100/H100 behaviour
// reported by vLLM/SGLang and apply equally well to H200 since the NVLink5
// fabric on H200 retains the same SM architecture as H100.
// ---------------------------------------------------------------------------
constexpr int kMaxBlocks = 36;
constexpr int kDefaultThreads = 512;
constexpr int kDefaultBlockLimit = 36;
constexpr int kMaxThreadsPerBlock = 512;
constexpr int kMaxRanks = 8;

// Algorithm-selection thresholds (bytes).  Small reductions live entirely on
// NVLink with the one-shot algorithm; larger reductions need the two-shot
// algorithm to avoid hammering the fabric with O(N^2) traffic.
constexpr int kAllReduceGPUSmall = 4;
constexpr int kAllReduceGPULarge = 8;
constexpr size_t kAllReduceSmallThreshold = 512 * 1024;  // 512 KiB
constexpr size_t kAllReduceLargeThreshold = 256 * 1024;  // 256 KiB

using FlagType = uint32_t;

// Per-rank "signal" buffer.  We allocate one of these in CUDA-IPC-shareable
// memory on every rank.  Threads in the all-reduce kernel use it to implement
// a multi-GPU barrier.
struct Signal {
  alignas(128) FlagType self_counter[kMaxBlocks][kMaxRanks];
  // Two sets of peer counters are needed for two consecutive barriers — see
  // multi_gpu_barrier() for the reasoning.
  alignas(128) FlagType peer_counter[2][kMaxBlocks][kMaxRanks];
};

struct __align__(16) RankData {
  const void* __restrict__ ptrs[kMaxRanks];
};

struct __align__(16) RankSignals {
  Signal* signals[kMaxRanks];
};

// CUDA error helper.
#define NVLINK_CUDA_CHECK(stmt)                                                 \
  do {                                                                          \
    cudaError_t err__ = (stmt);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      throw std::runtime_error(                                                 \
          std::string("CUDA error: ") + cudaGetErrorString(err__) + " at " +    \
          __FILE__ + ":" + std::to_string(__LINE__));                           \
    }                                                                           \
  } while (0)

using IpcKey = std::array<uint8_t, sizeof(cudaIpcMemHandle_t)>;
static_assert(sizeof(IpcKey) == sizeof(cudaIpcMemHandle_t));
static_assert(alignof(IpcKey) == alignof(cudaIpcMemHandle_t));

}  // namespace nvlink_ar
