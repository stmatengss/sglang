// SPDX-License-Identifier: Apache-2.0
// NVLink AllReduce — PyTorch bindings.
//
// This translation unit instantiates the kernel templates for fp32/fp16/bf16
// and exposes a small C++ API to Python:
//   * meta_size()                          — sizeof(Signal) for the host
//   * init(...)                            — build a NvlinkAllreduce handle
//   * register_buffer(...)                 — register peer pointers
//   * all_reduce(...)                      — dispatch the kernel
//   * get_ipc_mem_handle(tensor)           — to be shared via TCPStore
//   * open_ipc_handle(handle_bytes)        — open a peer handle locally
//   * dispose(handle)                      — tear down
// The actual handle exchange between ranks is done in pure Python over
// TCPStore so we do not depend on NCCL.

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cstring>
#include <vector>

#include "nvlink_allreduce.cuh"

using namespace nvlink_ar;

namespace {

NvlinkAllreduce* as_handle(int64_t h) {
  return reinterpret_cast<NvlinkAllreduce*>(static_cast<uintptr_t>(h));
}

}  // namespace

int64_t meta_size() { return static_cast<int64_t>(sizeof(Signal)); }

// Build a NvlinkAllreduce.
//   signal_ipc_handles : list of `world_size` byte-strings, each is a
//                        cudaIpcMemHandle_t pointing at the Signal+tmpbuf
//                        region owned by that rank.  signal_ipc_handles[rank]
//                        is the *local* rank's own handle (so we can also
//                        compute its device address consistently).
//   rank_data_ptr      : opaque device address (int64) of a per-rank scratch
//                        region used to store kernel-arg RankData entries.
//   rank_data_size     : size of that region in bytes.
//   self_signal_ptr    : device address (int64) of the local Signal region.
//   rank, world_size   : standard.
//   full_nvlink        : True on a fully-connected NVLink island (NVL8).
int64_t init_handle(const std::vector<std::string>& signal_ipc_handles,
                    int64_t rank_data_ptr, int64_t rank_data_size,
                    int64_t self_signal_ptr, int64_t rank, int64_t world_size,
                    bool full_nvlink) {
  TORCH_CHECK(static_cast<int64_t>(signal_ipc_handles.size()) == world_size,
              "expected ", world_size, " signal handles");
  TORCH_CHECK(world_size >= 2 && world_size <= kMaxRanks,
              "world_size must be in [2, ", kMaxRanks, "]");
  TORCH_CHECK(rank >= 0 && rank < world_size, "bad rank");

  Signal* sigs[kMaxRanks] = {nullptr};
  sigs[rank] = reinterpret_cast<Signal*>(static_cast<uintptr_t>(self_signal_ptr));
  for (int i = 0; i < world_size; i++) {
    if (i == rank) continue;
    TORCH_CHECK(signal_ipc_handles[i].size() == sizeof(cudaIpcMemHandle_t),
                "ipc handle size mismatch for rank ", i);
    cudaIpcMemHandle_t h;
    std::memcpy(&h, signal_ipc_handles[i].data(), sizeof(h));
    void* ptr = nullptr;
    NVLINK_CUDA_CHECK(
        cudaIpcOpenMemHandle(&ptr, h, cudaIpcMemLazyEnablePeerAccess));
    sigs[i] = reinterpret_cast<Signal*>(ptr);
  }
  auto* obj = new NvlinkAllreduce(
      sigs,
      reinterpret_cast<void*>(static_cast<uintptr_t>(rank_data_ptr)),
      static_cast<size_t>(rank_data_size), static_cast<int>(rank),
      static_cast<int>(world_size), full_nvlink);
  return static_cast<int64_t>(reinterpret_cast<uintptr_t>(obj));
}

void dispose(int64_t h) { delete as_handle(h); }

std::string get_ipc_mem_handle(int64_t device_ptr) {
  cudaIpcMemHandle_t handle;
  NVLINK_CUDA_CHECK(cudaIpcGetMemHandle(
      &handle, reinterpret_cast<void*>(static_cast<uintptr_t>(device_ptr))));
  return std::string(reinterpret_cast<const char*>(&handle), sizeof(handle));
}

// Register a user buffer.  peer_ptrs is a list of size world_size of int64
// device pointers (the local rank's entry may also be present and is used
// directly; remote entries must already be peer-opened by the caller via
// open_ipc_handle).
void register_buffer(int64_t h, const std::vector<int64_t>& peer_ptrs) {
  auto* obj = as_handle(h);
  TORCH_CHECK(static_cast<int>(peer_ptrs.size()) == obj->world_size_,
              "peer_ptrs size mismatch");
  void* ptrs[kMaxRanks] = {nullptr};
  for (int i = 0; i < obj->world_size_; i++) {
    ptrs[i] = reinterpret_cast<void*>(static_cast<uintptr_t>(peer_ptrs[i]));
  }
  obj->register_buffer(ptrs);
}

// Allocate a CUDA region via cudaMalloc so it has a stable base address that
// can be shared with cudaIpcGetMemHandle.  We do NOT use torch's caching
// allocator here because IPC handles are only valid for whole base regions.
int64_t cuda_malloc(int64_t bytes) {
  void* ptr = nullptr;
  NVLINK_CUDA_CHECK(cudaMalloc(&ptr, static_cast<size_t>(bytes)));
  NVLINK_CUDA_CHECK(cudaMemset(ptr, 0, static_cast<size_t>(bytes)));
  return static_cast<int64_t>(reinterpret_cast<uintptr_t>(ptr));
}

void cuda_free(int64_t ptr) {
  if (ptr) cudaFree(reinterpret_cast<void*>(static_cast<uintptr_t>(ptr)));
}

// Wrap a raw device pointer in a torch::Tensor without copying or owning it.
// The Tensor is a view into the caller-managed allocation.  Used so Python
// can treat our cudaMalloc'd region as a normal tensor for compute.
// dtype_name is one of {"float32", "float16", "bfloat16"}.  We accept a
// string rather than a Python ``torch.dtype`` so we don't depend on any
// PyTorch private API to round-trip the type.
torch::Tensor tensor_from_ptr(int64_t ptr, std::vector<int64_t> sizes,
                              const std::string& dtype_name,
                              int64_t device_idx) {
  at::ScalarType dtype;
  if (dtype_name == "float32" || dtype_name == "float") dtype = at::kFloat;
  else if (dtype_name == "float16" || dtype_name == "half") dtype = at::kHalf;
  else if (dtype_name == "bfloat16") dtype = at::kBFloat16;
  else
    TORCH_CHECK(false, "unsupported dtype name: ", dtype_name);
  auto options = torch::TensorOptions()
                     .dtype(dtype)
                     .device(torch::Device(torch::kCUDA, device_idx));
  return torch::from_blob(reinterpret_cast<void*>(static_cast<uintptr_t>(ptr)),
                          sizes, options);
}

int64_t open_ipc_handle(const std::string& handle_bytes) {
  TORCH_CHECK(handle_bytes.size() == sizeof(cudaIpcMemHandle_t),
              "bad ipc handle size");
  cudaIpcMemHandle_t h;
  std::memcpy(&h, handle_bytes.data(), sizeof(h));
  void* ptr = nullptr;
  NVLINK_CUDA_CHECK(cudaIpcOpenMemHandle(&ptr, h, cudaIpcMemLazyEnablePeerAccess));
  return static_cast<int64_t>(reinterpret_cast<uintptr_t>(ptr));
}

void all_reduce(int64_t h, torch::Tensor& inp, torch::Tensor& out) {
  auto* obj = as_handle(h);
  TORCH_CHECK(inp.is_cuda() && out.is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(inp.is_contiguous() && out.is_contiguous(),
              "tensors must be contiguous");
  TORCH_CHECK(inp.scalar_type() == out.scalar_type(), "dtype mismatch");
  TORCH_CHECK(inp.numel() == out.numel(), "numel mismatch");

  c10::cuda::CUDAGuard guard(inp.device());
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  int size = static_cast<int>(inp.numel());

  switch (inp.scalar_type()) {
    case at::ScalarType::Float:
      obj->allreduce<float>(stream, inp.data_ptr<float>(),
                            out.data_ptr<float>(), size);
      break;
    case at::ScalarType::Half:
      obj->allreduce<half>(stream, reinterpret_cast<half*>(inp.data_ptr()),
                           reinterpret_cast<half*>(out.data_ptr()), size);
      break;
    case at::ScalarType::BFloat16:
      obj->allreduce<nv_bfloat16>(
          stream, reinterpret_cast<nv_bfloat16*>(inp.data_ptr()),
          reinterpret_cast<nv_bfloat16*>(out.data_ptr()), size);
      break;
    default:
      TORCH_CHECK(false, "unsupported dtype: ", inp.scalar_type());
  }
}

// Try to fully populate the NVLink P2P mesh from one device to another.
// Returns true if NVLink P2P is supported and was enabled, false otherwise.
// Called once at init.
bool enable_p2p_access(int64_t from_dev, int64_t to_dev) {
  if (from_dev == to_dev) return true;
  int can_access = 0;
  NVLINK_CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, from_dev, to_dev));
  if (!can_access) return false;
  cudaSetDevice(from_dev);
  cudaError_t err = cudaDeviceEnablePeerAccess(to_dev, 0);
  // Already-enabled is fine.
  if (err == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
  else NVLINK_CUDA_CHECK(err);
  return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "NVLink AllReduce — NCCL-free PyTorch all-reduce backend.";
  m.def("meta_size", &meta_size,
        "Return sizeof(Signal) — the minimum bytes a Signal region needs.");
  m.def("init_handle", &init_handle,
        "Initialise an NVLink AllReduce instance, returning an opaque int64 handle.");
  m.def("dispose", &dispose, "Free an instance previously returned by init_handle.");
  m.def("get_ipc_mem_handle", &get_ipc_mem_handle,
        "Return cudaIpcMemHandle_t bytes for a device pointer (int64).");
  m.def("open_ipc_handle", &open_ipc_handle,
        "Open a remote cudaIpcMemHandle_t and return its local device address.");
  m.def("register_buffer", &register_buffer,
        "Register an already-shared buffer (list of peer device pointers).");
  m.def("all_reduce", &all_reduce, "In-place sum all-reduce.");
  m.def("enable_p2p_access", &enable_p2p_access,
        "cudaDeviceEnablePeerAccess from -> to. Returns False if not P2P-capable.");
  m.def("cuda_malloc", &cuda_malloc, "cudaMalloc(bytes); returns device ptr (int64).");
  m.def("cuda_free", &cuda_free, "cudaFree(ptr).");
  m.def("tensor_from_ptr", &tensor_from_ptr,
        "Construct a non-owning CUDA tensor view over a raw device pointer.");
}
