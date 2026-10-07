#pragma once
#include "cuda_memory.cuh"
#include "projection.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace gewell::gemma4_26b_a4b::sm120 {
struct ExpertWeights {
  Projection gate, up, down;
};
class GroupedNvfp4;

// One workspace shared by all layers. Dispatch never downloads routing data,
// allocates storage, or creates plans. Each expert can receive every input row.
// Format flags come from the immutable layer bindings. fp4 selects W4A4;
// otherwise NVFP4 weights are decoded only into the active WMMA shared tile.
class MoeWorkspace {
 public:
  explicit MoeWorkspace(unsigned max_rows);
  ~MoeWorkspace();
  void run(const __nv_bfloat16* input, const __nv_bfloat16* scores,
           const __nv_bfloat16* expert_scales, const ExpertWeights* weights,
           __nv_bfloat16* output, unsigned rows, bool has_bf16, bool has_fp8, bool has_nvfp4, bool fp4,
           cudaStream_t stream = nullptr);
  void dispatch(const __nv_bfloat16* scores, const __nv_bfloat16* expert_scales,
                unsigned rows, cudaStream_t stream = nullptr);
  std::size_t bytes() const;
  std::size_t routing_bytes() const;
  std::size_t projection_bytes() const { return gate_up_.size() + activated_.size() + quantized_.size(); }
  std::size_t reduction_bytes() const { return expanded_.size(); }
  const int* selected_experts() const;
  const float* routing_weights() const;
  const int* offsets() const;
  const int* sorted_assignments() const;
  const __nv_bfloat16* gate_up_capture() const { return static_cast<const __nv_bfloat16*>(gate_up_.data()); }
  const __nv_bfloat16* activated_capture() const { return static_cast<const __nv_bfloat16*>(activated_.data()); }
  const __nv_bfloat16* down_capture() const { return static_cast<const __nv_bfloat16*>(expanded_.data()); }
 private:
  unsigned max_rows_;
  cuda_detail::DeviceAllocation metadata_, expanded_, gate_up_, activated_, quantized_, grouped_;
  unsigned grouped_blocks_;
  std::unique_ptr<cuda_detail::DeviceAllocation> sort_scratch_;
  std::unique_ptr<GroupedNvfp4> grouped_fp4_;
  std::size_t sort_bytes_{};
};
}
