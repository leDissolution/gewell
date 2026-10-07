#pragma once
#include "cuda_memory.cuh"
#include "gewell/kv_view.h"
#include <cublas_v2.h>
#include "gewell/attention_compute.h"
#include "gewell/mtp_attention.h"
#include <memory>
#include <vector>

namespace gewell::gemma4_26b_a4b::sm120 {
struct AttentionInput {
  const __nv_bfloat16 *query{}, *key{}, *value{};
  kv_cache::DeviceView cache{};
  unsigned position{}, rows{};
  __nv_bfloat16* output{};
  bool image{};
};
// Bounded BF16 or native FP8 compute over independently selected stored KV. One workspace
// is reused across segments and layers; current Q/K/V are token-major.
// The backend serializes workspace/handle reuse on its execution stream.
class AttentionWorkspace {
 public:
  AttentionWorkspace(unsigned max_rows = 4096,
      attention::Compute local = attention::Compute::bf16,
      attention::Compute global = attention::Compute::bf16);
  void run(cublasHandle_t handle, const __nv_bfloat16* query,
           const __nv_bfloat16* key, const __nv_bfloat16* value,
           const __nv_bfloat16* key_norm, kv_cache::DeviceView cache,
           bool global, unsigned position, unsigned rows,
           __nv_bfloat16* output, cudaStream_t stream = nullptr,
           bool image = false);
  // Independent inputs; cached history remains read-only. Single-row BF16
  // queries use the shared 31B fused decode arithmetic and common launches.
  void run_batch(cublasHandle_t handle, const std::vector<AttentionInput>& inputs,
                 const __nv_bfloat16* key_norm, bool global, cudaStream_t stream);
  [[nodiscard]] std::size_t bytes() const {
    return (storage_ ? storage_->size() : 0) + (fp8_ ? fp8_->size() : 0) + (packed_ ? packed_->size() : 0);
  }
 private:
  const unsigned max_rows_;
  const attention::Compute local_, global_;
  std::unique_ptr<cuda_detail::DeviceAllocation> storage_, fp8_, packed_;
  std::vector<mtp_attention::BatchInput> decode_inputs_;
};
}  // namespace gewell::gemma4_26b_a4b::sm120
