#pragma once
#include "gewell/models/gemma4/31b/model.h"

#include "gewell/compact_global_cache.h"
#include "gewell/attention_compute.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace gewell::mtp_assistant {

using BFloat16 = __nv_bfloat16;

enum class Model { gemma4_31b, gemma4_26b_a4b };

// All weights are borrowed device pointers, ordered within the component:
// embedding, four eleven-tensor layers, final norm,
// pre-projection, post-projection. The assistant embedding is its output head.
struct Weights {
  const BFloat16* target_embedding{};
  std::array<const BFloat16*, 48> assistant{};
  const BFloat16* target_global_k_norm{};
};

// Target layers 58/59 (31B) or 28/29 (26B) supply local/global KV.
// Separate local K/V and exactly one compact global representation are present.
// The cache is read-only throughout a complete draft cycle. Local visibility
// is [max(0,L-1024),L), global visibility is [0,L), including no pending KV.
struct FrozenCache {
  const BFloat16* local_key{};
  const BFloat16* local_value{};
  compact_global_cache::PagedView<BFloat16> global{};
  const BFloat16* global_compact{};
  std::uint32_t global_capacity{};
  std::uint32_t processed_tokens{};
  kv_cache::Format local_format{kv_cache::Format::bf16};
  kv_cache::Format global_format{kv_cache::Format::bf16};
};

// Optional caller-owned rows for the independent assistant oracle. Hidden
// rows are [1024]; the query/context rows use each layer's actual head width.
struct Trace {
  BFloat16* pre_projection{};
  std::array<BFloat16*, 4> layers{};
  BFloat16* final_norm{};
  std::array<BFloat16*, 4> query_norm{};  // Each [model_query_heads,layer_head_width].
  std::array<BFloat16*, 4> query_rope{};
  std::array<BFloat16*, 4> attention{};
};

// The frozen-prefix primitive is exposed for independent cache-layout and
// numerical tests. Query/context are [32,D]. The supplied scratch must have
// attention_scratch_bytes(processed_tokens) bytes; calls allocate nothing.
// Scores, online softmax, partial contexts and final reduction are FP32;
// context rounds to BF16. Unrotated compact K reconstructs with a BF16 multiply.
std::size_t attention_scratch_bytes(std::uint32_t context_capacity);
void attend_frozen_prefix(
    const BFloat16* query, const FrozenCache& cache,
    const BFloat16* target_global_k_norm, gemma4_31b::AttentionKind kind,
    void* scratch, BFloat16* context, cudaStream_t stream = nullptr);

struct Input {
  const std::uint32_t* token{};
  const BFloat16* hidden{};
  FrozenCache cache{};
  BFloat16* logits{};
  BFloat16* feedback{};
  // Optional device row [1024], captured after final normalization.
  BFloat16* capture_hidden{};
};

class Executor final {
 public:
  explicit Executor(Model model, cublasLtHandle_t handle, const Weights& weights,
                    std::uint32_t context_capacity = 262'144,
                    std::uint32_t batch_capacity = 1,
                    attention::Compute local_compute = attention::Compute::bf16,
                    attention::Compute global_compute = attention::Compute::bf16);
  ~Executor();
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;

  [[nodiscard]] std::size_t scratch_bytes() const;

  // input_hidden is the target's final-normalized terminal hidden on the
  // first step, then the preceding assistant feedback. Token is device-side.
  // Its value must be a valid target-vocabulary ID.
  // Outputs are BF16 logits[262144] without a softcap and feedback[target_hidden_size].
  // Feedback may alias input_hidden. Other outputs/scratch must not overlap.
  // Every step uses the frozen processed_tokens as its RoPE position.
  void forward(const std::uint32_t* token, const BFloat16* input_hidden,
               const FrozenCache& cache, BFloat16* logits,
               BFloat16* feedback, cudaStream_t stream = nullptr,
               const Trace* trace = nullptr);

  // Independent requests share each projection GEMM. Each cache remains
  // frozen at its own position; feedback may alias that request's hidden.
  void forward_batch(const std::vector<Input>& inputs,
                     cudaStream_t stream = nullptr);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gewell::mtp_assistant
