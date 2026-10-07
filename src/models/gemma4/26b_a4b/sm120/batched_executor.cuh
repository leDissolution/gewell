#pragma once
#include "gewell/models/gemma4/26b_a4b/executor.h"
#include "gewell/kv_view.h"
#include "gewell/models/gemma4/26b_a4b/artifact.h"
#include "gewell/models/gemma4/26b_a4b/weight_qdq.h"
#include <cuda_runtime.h>
#include "gewell/nvfp4_policy.h"
#include "gewell/attention_compute.h"
#include <array>
#include "mtp_staging.cuh"

namespace gewell::gemma4_26b_a4b::sm120 {
struct LayerCapture {
  unsigned completed_layers{};
  __nv_bfloat16* output{};  // [segment rows,2816], raw residual after this layer.
};
struct Segment {
  unsigned position{}, rows{};
  std::array<kv_cache::DeviceView,30> cache;
  MtpStaging staging{};  // Non-null data selects read-only verification.
  std::vector<LayerCapture> captures;
  // One complete image, already projected to [rows,2816] BF16. No text
  // embedding scale is applied. Null selects ordinary token embeddings.
  const __nv_bfloat16* image_features{};
};

// One resident weight set and bounded scratch. The backend serializes calls
// on the stream bound at construction; that stream must outlive the executor.
// Ordinary segments own distinct writable cache views. Verifier segments
// instead own private staging; their committed cache views are read-only.
// the shared ledger reserves pages and resolves copy-on-write before dispatch.
class BatchedExecutor {
 public:
  BatchedExecutor(ArtifactFile weights, unsigned max_rows, cudaStream_t stream,
                  nvfp4::ActivationPolicy policy,
                  attention::Compute local = attention::Compute::bf16,
                  attention::Compute global = attention::Compute::bf16);
  ~BatchedExecutor();
  // Startup-only overlay; inference must not be in flight.
  weight_qdq::ApplySummary apply_qdq(const QdqMask& mask);
  // Device token IDs have already been range-checked by the runtime. Segments
  // concatenate in input order. Returned final-normalized hidden rows remain
  // valid until the next forward; the caller saves requested terminal rows.
  // Staged verification uses decode activation policy, <=1280 total rows,
  // and exactly the same complete MoE blocks as ordinary execution.
  const __nv_bfloat16* forward(const std::uint32_t* tokens,
      const std::vector<Segment>& segments, nvfp4::Phase phase,
      const Capture& capture = {});
  // Only selected decision rows need vocabulary projections. Output storage
  // belongs to the backend, not to the per-layer scratch allocation.
  void head(const __nv_bfloat16* hidden, unsigned rows, __nv_bfloat16* logits);
  const __nv_bfloat16* target_embedding() const;
  const __nv_bfloat16* shared_global_key_norm() const;
  struct MemoryUsage {
    std::size_t weights, expert_tables, shared_projection, positions,
        routing, expert_projection, expert_reduction, attention, quantized_projection;
  };
  MemoryUsage memory_usage() const;
  std::size_t weight_bytes() const;
  std::size_t scratch_bytes() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gewell::gemma4_26b_a4b::sm120
