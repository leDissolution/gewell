#pragma once
#include "../weights.cuh"
#include "gewell/vision_engine.h"
#include "gewell/vision_executor.h"

namespace gewell::gemma4_31b::sm120 {
// One reusable encoder and device workspace for the GPU owner. Buffers grow
// to the largest image seen; subsequent images reuse them on the same stream.
class PreparedImage {
 public:
  explicit PreparedImage(const WeightArena& weights);
  // Enqueues upload and encoding. Input bytes must remain live until stream
  // completion. Consumers must use this stream or synchronize before reading.
  void prepare(const std::vector<std::uint8_t>& pixels,
               const std::vector<std::uint8_t>& positions, std::uint32_t padded_patch_rows,
               std::uint32_t soft_token_count, cudaStream_t stream = nullptr);
  const BFloat16* data() const { return static_cast<const BFloat16*>(features_->data()); }
  std::size_t size() const { return features_->size(); }
  std::size_t scratch_bytes() const { return scratch_->size(); }
  // Diagnostic-only synchronization; serving uses the scheduler's completion event.
  float gpu_milliseconds() const;
 private:
  vision_executor::VisionExecutor tower_;
  std::unique_ptr<DeviceAllocation> features_, pixels_, positions_, scratch_;
  CudaEvent begin_, end_;
};
}
