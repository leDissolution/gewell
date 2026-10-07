#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

namespace gewell::cuda_detail {
[[noreturn]] inline void fail(std::string_view operation, std::string_view detail) {
  throw std::runtime_error(std::string(operation) + ": " +
                           std::string(detail));
}

inline void check_cuda(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess) {
    fail(operation, cudaGetErrorString(status));
  }
}

class DeviceAllocation {
 public:
  explicit DeviceAllocation(std::size_t bytes) : bytes_(bytes) {
    if (bytes == 0) {
      fail("cudaMalloc", "zero-sized allocation");
    }
    check_cuda(cudaMalloc(&pointer_, bytes), "cudaMalloc");
  }
  ~DeviceAllocation() {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
  }
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  [[nodiscard]] void* data() const { return pointer_; }
  [[nodiscard]] std::size_t size() const { return bytes_; }

 private:
  void* pointer_{nullptr};
  std::size_t bytes_{};
};

// The cold KV tier is one fixed pinned allocation. Individual CPU-pool
// allocations are offsets inside it, just like GPU-pool allocations are
// offsets inside DeviceAllocation. Keeping it pinned makes the initial
// synchronous D2H/H2D path correct without a separate unaccounted staging
// buffer.
class PinnedHostAllocation {
 public:
  explicit PinnedHostAllocation(std::size_t bytes) : bytes_(bytes) {
    if (bytes == 0) {
      fail("cudaMallocHost", "zero-sized allocation");
    }
    check_cuda(cudaMallocHost(&pointer_, bytes), "cudaMallocHost cold KV pool");
  }
  ~PinnedHostAllocation() {
    if (pointer_ != nullptr) {
      cudaFreeHost(pointer_);
    }
  }
  PinnedHostAllocation(const PinnedHostAllocation&) = delete;
  PinnedHostAllocation& operator=(const PinnedHostAllocation&) = delete;
  [[nodiscard]] void* data() const { return pointer_; }
  [[nodiscard]] std::size_t size() const { return bytes_; }

 private:
  void* pointer_{nullptr};
  std::size_t bytes_{};
};

}  // namespace gewell::cuda_detail
