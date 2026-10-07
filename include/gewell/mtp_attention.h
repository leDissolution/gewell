#pragma once

#include "gewell/mtp_target.h"

namespace gewell::mtp_attention {

inline constexpr std::uint32_t kQueryTileRows = 8;
inline constexpr std::uint32_t kKeysPerTile = 32;
inline constexpr std::uint32_t kMaxSplits = 256;

// Non-KV scratch is bounded independently of the draft depth and context.
// Consecutive groups of at most eight query rows reuse the same allocation.
std::size_t scratch_bytes(std::uint32_t capacity_rows,
                          std::uint32_t context_capacity = 262144);

// Read-only attention over committed KV followed by the causal staged prefix.
// Query and staged K are [heads, rows, D]; staged V and output are token-major.
// Global compact K reconstruction rounds V * k_norm to BF16 before the dot
// product, including staged keys. Scores and all partial reductions use FP32;
// P*V probabilities round once to BF16.
void run(const mtp_target::BFloat16* query,
         const mtp_target::BFloat16* staged_key,
         const mtp_target::BFloat16* staged_value,
         const kv_cache::DeviceView& cache,
         const mtp_target::BFloat16* global_k_norm,
         std::uint32_t base_position, std::uint32_t rows,
         gemma4_31b::AttentionKind kind, mtp_target::BFloat16* context,
         void* scratch, std::size_t scratch_size, cudaStream_t stream);

struct BatchInput {
  const mtp_target::BFloat16* query{};
  const mtp_target::BFloat16* staged_key{};
  const mtp_target::BFloat16* staged_value{};
  kv_cache::DeviceView cache{};
  std::uint32_t base_position{}, rows{};
  mtp_target::BFloat16* context{};
  // FP8 26B local prefill only: this entire input is one complete image.
  // Image queries see future image rows, retaining the 1024 past cutoff.
  bool local_image{};
};

// Native E4M3 QK and PV with FP32 softmax/accumulation. Q/K/V scales are
// per vector; V scales are folded into P before per-query/tile P quantization.
// Uses the same bounded scratch as BF16 and never modifies cached/staged KV.
// frozen=true accepts one query per input over [0,base_position), for ordinary
// decode after commit and assistant attention; otherwise current rows are causal.
// Explicit geometry: 16-head 26B or 32-head 31B, with KV heads /2 local and /8 global.
std::size_t fp8_scratch_bytes(unsigned query_heads, std::uint32_t capacity_rows,
                              std::uint32_t context_capacity);
void run_fp8_batch(unsigned query_heads, const std::vector<BatchInput>& inputs,
                   const mtp_target::BFloat16* global_k_norm,
                   gemma4_31b::AttentionKind kind, void* scratch,
                   std::size_t scratch_size, cudaStream_t stream,
                   bool frozen = false);

// Coalesce independent request tiles into common launches, with disjoint
// partial results in the caller's existing scratch. Splits balance visible KV
// work within each launch, subject to a block target and scratch capacity.
// Scores, softmax metadata and reductions stay FP32; P*V probabilities round
// once to BF16. Changing batch geometry can change reduction order.
// Single-request calls retain the serial split count.
// Large batches use bounded groups; no allocation, upload or synchronization.
void run_batch(const std::vector<BatchInput>& inputs,
               const mtp_target::BFloat16* global_k_norm,
               gemma4_31b::AttentionKind kind, void* scratch,
               std::size_t scratch_size, cudaStream_t stream);

// Single causal decode rows for 16-head 26B or 32-head 31B. Uses the same
// fused BF16 arithmetic as run_batch. Local attention keeps 32-key splits;
// global attention shares its adaptive split planner. Current K/V stay BF16
// even when committed cache storage is FP8. Inputs must have rows=1.
void run_decode_batch(unsigned query_heads, const std::vector<BatchInput>& inputs,
                      const mtp_target::BFloat16* global_k_norm,
                      gemma4_31b::AttentionKind kind, void* scratch,
                      std::size_t scratch_size, cudaStream_t stream);

// Coalesce independent assistant queries over committed global KV. Each input
// has rows=1, base_position=processed_tokens, and null staged K/V. Cache and
// probability arithmetic match run_frozen_global(); batching only repartitions
// the FP32 split reduction to expose enough concurrent work.
void run_frozen_global_batch(const std::vector<BatchInput>& inputs,
                             const mtp_target::BFloat16* global_k_norm,
                             void* scratch, std::size_t scratch_size,
                             cudaStream_t stream);

// Single frozen query for the 16-head 26B or 32-head 31B assistant. Local
// visibility is [max(0,L-1024),L), global is [0,L); pending KV is never read.
// Local/global KV head counts are query_heads/2 and query_heads/8. Query and
// output are [query_heads,D]. Arithmetic matches the BF16 run() policy above;
// both BF16 and FP8 cache storage are supported. Cache is never written.
std::size_t frozen_scratch_bytes(unsigned query_heads,
                                std::uint32_t context_capacity);
void run_frozen_prefix(const mtp_target::BFloat16* query,
                       const kv_cache::DeviceView& cache,
                       const mtp_target::BFloat16* global_k_norm,
                       std::uint32_t processed_tokens, unsigned query_heads,
                       gemma4_31b::AttentionKind kind,
                       mtp_target::BFloat16* context, void* scratch,
                       std::size_t scratch_size, cudaStream_t stream);

// Batch the same frozen-prefix arithmetic without changing per-request split
// counts. Groups of up to 32 requests reuse the supplied scratch; no allocation
// or upload. Results match run_frozen_prefix at any batch geometry.
void run_frozen_prefix_batch(unsigned query_heads, const std::vector<BatchInput>& inputs,
                             const mtp_target::BFloat16* global_k_norm,
                             gemma4_31b::AttentionKind kind, void* scratch,
                             std::size_t scratch_size, cudaStream_t stream);

// Assistant global attention uses the same key reconstruction and arithmetic,
// over committed [0,processed_tokens) only. Query/output are [32,512]; scratch
// is scratch_bytes(1, processed_tokens). No cache or staged KV is written.
void run_frozen_global(const mtp_target::BFloat16* query,
                       const kv_cache::DeviceView& cache,
                       const mtp_target::BFloat16* global_k_norm,
                       std::uint32_t processed_tokens,
                       mtp_target::BFloat16* context, void* scratch,
                       std::size_t scratch_size, cudaStream_t stream);

}  // namespace gewell::mtp_attention
