#pragma once

#include "gewell/vision_engine.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

namespace gewell::vision_executor {

using BFloat16 = __nv_bfloat16;

// Separate component order, with each tensor padded to a 4 KiB boundary.
std::size_t vision_weight_slice_bytes(vision_engine::Model model);

struct VisionWeightSlice {
  // base must be 256-byte aligned.
  const BFloat16* base{};
  std::size_t bytes{};
};

enum class CaptureDType : std::uint8_t {
  bf16,
  f32,
  i32,
};

struct CaptureTensor {
  std::string_view name;
  const void* device_data{};
  CaptureDType dtype{};
  std::uint32_t rows{};
  std::uint32_t columns{};
};

// Optional diagnostic hook. capture() is called while device_data is live.
// Implementations must copy it, or enqueue a copy on stream, before returning.
// They must not retain device_data or name.
class CaptureSink {
 public:
  virtual ~CaptureSink() = default;
  virtual void capture(const CaptureTensor& tensor, cudaStream_t stream) = 0;
};

// Batch-one, prefill-only Gemma 4 vision executor. The object
// owns CUDA library descriptors but no activation storage. Weights, scratch,
// request buffers, stream, and their lifetimes remain caller-owned.
class VisionExecutor {
 public:
  VisionExecutor(vision_engine::Model model, VisionWeightSlice weights,
                 std::uint32_t soft_token_count);
  ~VisionExecutor();

  VisionExecutor(const VisionExecutor&) = delete;
  VisionExecutor& operator=(const VisionExecutor&) = delete;
  VisionExecutor(VisionExecutor&&) noexcept;
  VisionExecutor& operator=(VisionExecutor&&) noexcept;

  [[nodiscard]] std::uint32_t soft_token_count() const;
  [[nodiscard]] std::uint32_t patch_rows() const;
  [[nodiscard]] std::size_t scratch_bytes() const;

  // Update row-dependent layouts without rebuilding handles or weight bindings.
  // Already enqueued calls retain their launch parameters. Subsequent run()
  // calls must provide scratch_bytes() bytes for the new shape.
  void set_soft_token_count(std::uint32_t soft_token_count);

  // Enqueues the complete tower and bridge on stream. scratch_device must be
  // 256-byte aligned, provide scratch_bytes() bytes, and not overlap inputs,
  // output, or the weight slice. The executor performs no allocation or
  // synchronization; host/device copies occur only through CaptureSink.
  void run(const vision_engine::PrefillRequest& request,
           void* scratch_device, std::size_t scratch_capacity_bytes,
           cudaStream_t stream, CaptureSink* captures = nullptr) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gewell::vision_executor
