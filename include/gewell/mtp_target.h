#pragma once

#include "gewell/kv_view.h"
#include "gewell/attention_compute.h"

#include "gewell/models/gemma4/31b/model.h"
#include "gewell/models/gemma4/31b/sm120/fp8_projections.h"
#include "gewell/models/gemma4/31b/sm120/nvfp4_projections.h"
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <cublasLt.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace gewell::mtp_target {
using BFloat16 = __nv_bfloat16;
inline constexpr std::uint32_t kMaxDepth = 1279;
using Weights = std::array<const BFloat16*, gemma4_31b::kLogicalTensorCount>;

using Caches = std::array<kv_cache::DeviceView, gemma4_31b::kLayerCount>;

// Tokens occupy consecutive rows within each request. Every request supplies
// private KV staging, whose physical slot stride may exceed its current rows.
struct LayerCapture {
  std::uint32_t completed_layers{};
  BFloat16* output{};  // Device [request rows, hidden width], raw post-layer residual.
};
struct BatchInput {
  std::uint32_t base_position{};
  std::uint32_t rows{};
  Caches caches{};
  void* staging{};
  std::size_t staging_size{};
  std::uint32_t staging_capacity_rows{};
  std::vector<LayerCapture> captures;
};

// Copy an accepted prefix from full-layout speculative KV into committed
// caches. The physical staging slots use capacity_rows; head-major K inside
// each slot uses source_rows. Neither stride is the accepted commit_rows.
void commit_staged_rows(const Caches& caches, std::uint32_t base_position,
                        std::uint32_t source_rows,
                        std::uint32_t capacity_rows, const void* staging,
                        std::size_t staging_size, std::uint32_t commit_rows,
                        cudaStream_t stream);

struct CommitInput {
  const Caches* caches{};
  std::uint32_t base_position{};
  std::uint32_t source_rows{};
  std::uint32_t capacity_rows{};
  const void* staging{};
  std::size_t staging_size{};
  std::uint32_t commit_rows{};
};

// Commit independent accepted prefixes with one launch per model layer and
// bounded groups of 32 requests. Validation completes before any cache write.
void commit_staged_rows_batch(const std::vector<CommitInput>& inputs,
                              cudaStream_t stream);

// Target execution for supplied consecutive tokens. All per-layer current KV
// lives in caller-provided staging; run() never writes committed cache contents.
// This class owns only non-KV scratch and cuBLAS descriptors. No adaptive policy.
class Verifier {
 public:
  Verifier(cublasLtHandle_t handle, const Weights& weights,
           std::uint32_t capacity_rows,
           std::uint32_t context_capacity = 262'144,
           const nvfp4::Weights* native_weights = nullptr,
           nvfp4::ActivationPolicy activation_policy = nvfp4::ActivationPolicy::always,
           const fp8::Weights* fp8_weights = nullptr,
           attention::Compute local_compute = attention::Compute::bf16,
           attention::Compute global_compute = attention::Compute::bf16);
  ~Verifier();
  Verifier(const Verifier&) = delete;
  Verifier& operator=(const Verifier&) = delete;
  static std::size_t staging_bytes(std::uint32_t capacity_rows);
  std::size_t scratch_bytes() const;
  void prepare(std::uint32_t rows);
  void run(const std::uint32_t* device_tokens, std::uint32_t base_position,
           std::uint32_t rows, const Caches& caches, void* staging,
           std::size_t staging_size, cudaStream_t stream);
  // Dense projections share the flattened rows; attention remains isolated
  // within each input. hidden()/logits() follow input order, then token order.
  // Commit each request with commit_staged_rows(); commit() is invalid here.
  void run_batch(const std::uint32_t* flattened_device_tokens,
                 const std::vector<BatchInput>& inputs, cudaStream_t stream);
  void commit(const Caches& caches, std::uint32_t base_position,
              std::uint32_t commit_rows, cudaStream_t stream);
  const BFloat16* logits() const;
  const BFloat16* hidden() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gewell::mtp_target
