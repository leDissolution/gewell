#pragma once

#include "gewell/compact_global_cache.h"
#include "gewell/fp8_attention.h"
#include "gewell/models/gemma4/31b/model.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace gewell::prefill_primitives {

using BFloat16 = __nv_bfloat16;

inline constexpr std::uint32_t kTokenCount = 1'024;
// Runtime text chunks may be wider than the fixed M=1024 operations below.
// This capacity does not change the model's local window or image prompt cap.
inline constexpr std::uint32_t kMaxChunkTokenCount = 4'096;
inline constexpr std::uint32_t kGlobalCacheMinimumCapacity = 1'026;
inline constexpr std::uint32_t kGlobalCompactRotatedKeyElements =
    compact_global_cache::kRotatedKeyElements;
inline constexpr std::uint32_t kGlobalCompactCacheRowElements =
    compact_global_cache::kRowElements;

// A persistent compact-global cache uses one fixed-size physical page for all
// global layers. `page_offsets` contains BF16-element offsets from `page_pool`
// for the page covering each consecutive 256-token range. The table is
// device-resident and may point at shared immutable pages.
using CompactGlobalPagedCache = compact_global_cache::PagedView<BFloat16>;
inline constexpr std::size_t kAttentionMatrixElements =
    static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * kTokenCount *
    kTokenCount;
inline constexpr std::size_t kAttentionScoreScratchBytes =
    kAttentionMatrixElements * sizeof(BFloat16);
inline constexpr std::size_t kAttentionProbabilityBytes =
    kAttentionMatrixElements * sizeof(BFloat16);

// Tensor attention keeps a fixed key tile while query capacity is independent.
// K/V staging uses the largest KV shape (local: 16 heads x 1024 x 256), while
// the numerator uses the largest head width (global: 512). The component
// constants below describe the default 1024-query capacity.
inline constexpr std::uint32_t kTensorAttentionTileTokens = 1'024;
inline constexpr std::uint32_t kTensorAttentionMaximumQueryRows =
    kMaxChunkTokenCount;
inline constexpr std::uint32_t kTensorAttentionMaximumKvHeads =
    gemma4_31b::kLocalKvHeadCount;
inline constexpr std::uint32_t kTensorAttentionMaximumHeadSize =
    gemma4_31b::kGlobalHeadSize;
inline constexpr std::size_t kTensorAttentionStagedElements =
    static_cast<std::size_t>(kTensorAttentionMaximumKvHeads) *
    kTensorAttentionTileTokens * gemma4_31b::kLocalHeadSize;
inline constexpr std::size_t kTensorAttentionStagedBytes =
    kTensorAttentionStagedElements * sizeof(BFloat16);
inline constexpr std::size_t kTensorAttentionScoreBytes =
    static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * kTokenCount *
    kTensorAttentionTileTokens * sizeof(float);
inline constexpr std::size_t kTensorAttentionProbabilityBytes =
    static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * kTokenCount *
    kTensorAttentionTileTokens * sizeof(BFloat16);
inline constexpr std::size_t kTensorAttentionNumeratorBytes =
    static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * kTokenCount *
    kTensorAttentionMaximumHeadSize * sizeof(float);
inline constexpr std::size_t kTensorAttentionStateBytes =
    static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * kTokenCount *
    sizeof(float);
inline constexpr std::size_t kTensorAttentionScratchBytes =
    2 * kTensorAttentionStagedBytes + kTensorAttentionScoreBytes +
    kTensorAttentionProbabilityBytes + kTensorAttentionNumeratorBytes +
    2 * kTensorAttentionStateBytes;

// query_capacity must be in [1,kTensorAttentionMaximumQueryRows]. Allocating
// for a capacity covers every smaller actual row count. Every internal region
// starts on a 256-byte boundary, including the state for odd row counts.
constexpr std::size_t tensor_attention_scratch_bytes(
    std::uint32_t query_capacity) {
  const std::size_t state_bytes =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * query_capacity *
      sizeof(float);
  const std::size_t aligned_state_bytes = (state_bytes + 255) / 256 * 256;
  return 2 * kTensorAttentionStagedBytes +
         (kTensorAttentionScoreBytes + kTensorAttentionProbabilityBytes +
          kTensorAttentionNumeratorBytes) / kTokenCount * query_capacity +
         2 * aligned_state_bytes;
}

static_assert(kTensorAttentionStagedBytes == 8'388'608);
static_assert(kTensorAttentionScoreBytes == 134'217'728);
static_assert(kTensorAttentionProbabilityBytes == 67'108'864);
static_assert(kTensorAttentionNumeratorBytes == 67'108'864);
static_assert(kTensorAttentionStateBytes == 131'072);
static_assert(kTensorAttentionScratchBytes == 285'474'816);
static_assert(tensor_attention_scratch_bytes(kTokenCount) ==
              kTensorAttentionScratchBytes);

// All pointers are device pointers. Calls allocate no device or host storage,
// and input/output regions must not overlap. The fixed APIs in this first
// section intentionally implement only batch-one prefill at positions 0..1023.

// Generate token-major RoPE tables. Local outputs are [1024,256] with
// theta=10,000 and full rotation. Global outputs are [1024,512] with
// theta=1,000,000 and only the first 64 frequencies in each split half
// rotating; every other global factor is the exact BF16 identity.
void generate_rope_factors_m1024(BFloat16* local_cos, BFloat16* local_sin,
                                 BFloat16* global_cos, BFloat16* global_sin,
                                 cudaStream_t stream = nullptr);

// Apply split-half RoPE and transpose [1024,heads,D] into
// [heads,1024,D]. Each multiply and the following add round independently to
// BF16. `heads` must be 32 or the selected kind's KV-head count.
void apply_rope_transpose_m1024(
    const BFloat16* input_token_major, const BFloat16* cosine,
    const BFloat16* sine, BFloat16* output_head_major, std::uint32_t heads,
    gemma4_31b::AttentionKind kind, cudaStream_t stream = nullptr);

// Copy post-RoPE K [KV,1024,D] and normalized V [1024,KV,D] into head-major
// caches [KV,capacity,D]. Local capacity must equal 1024. Global capacity must
// be at least 1026; slots 1024 and later are left untouched.
void write_kv_cache_m1024(
    const BFloat16* key_head_major, const BFloat16* value_token_major,
    BFloat16* key_cache, BFloat16* value_cache, std::uint32_t capacity,
    gemma4_31b::AttentionKind kind, cudaStream_t stream = nullptr);

// Fixed causal GQA. Q/K are [heads,1024,D], V is [1024,KV,D], score_scratch
// and probabilities are [32,1024,1024], and context is [1024,32,D]. Dot
// products use the cached-M1 256-thread FP32 reduction order and round to BF16
// in score_scratch. Softmax is FP32 and rounds probabilities to BF16 before
// the FP32-accumulating P*V product; context rounds once to BF16.
void causal_gqa_attention_m1024(
    const BFloat16* query_head_major, const BFloat16* key_head_major,
    const BFloat16* value_token_major, BFloat16* score_scratch,
    BFloat16* probabilities, BFloat16* context_token_major,
    gemma4_31b::AttentionKind kind, cudaStream_t stream = nullptr);

// Runtime chunk-prefill operations. `token_count` must be in [1,4096], and
// `base_position` is the absolute position of chunk row zero. These calls use
// no attention-matrix scratch.

// Generate token-major RoPE tables [token_count,D] for absolute positions
// base_position..base_position+token_count-1. The local/global rotation rules
// are identical to generate_rope_factors_m1024.
void generate_rope_factors_chunk(
    BFloat16* local_cos, BFloat16* local_sin, BFloat16* global_cos,
    BFloat16* global_sin, std::uint32_t base_position,
    std::uint32_t token_count, cudaStream_t stream = nullptr);

// Apply split-half RoPE and transpose token-major [token_count,heads,D] into
// head-major [heads,token_count,D]. `heads` must be 32 or the selected kind's
// KV-head count.
void apply_rope_transpose_chunk(
    const BFloat16* input_token_major, const BFloat16* cosine,
    const BFloat16* sine, BFloat16* output_head_major, std::uint32_t heads,
    std::uint32_t token_count, gemma4_31b::AttentionKind kind,
    cudaStream_t stream = nullptr);

// Scratch-free causal GQA over cached history plus the uncommitted current
// chunk. Q is [32,token_count,D], current K is [KV,token_count,D], current V is
// [token_count,KV,D], K/V caches are [KV,cache_capacity,D], and context is
// [token_count,32,D]. Scores, online softmax state, and context accumulation
// remain FP32 until the final BF16 context write.
//
// Local cache capacity must equal 1024 and represents an absolute-position
// ring (`slot = position % 1024`). Global capacity must cover the exclusive
// chunk end. The current chunk is read from current K/V rather than cache, so
// this operation must complete before write_kv_cache_chunk when local ring
// slots could overlap still-visible history.
void causal_gqa_attention_cached_chunk(
    const BFloat16* query_head_major,
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major, const BFloat16* key_cache,
    const BFloat16* value_cache, std::uint32_t base_position,
    std::uint32_t token_count, std::uint32_t cache_capacity,
    BFloat16* context_token_major, gemma4_31b::AttentionKind kind,
    cudaStream_t stream = nullptr,
    kv_cache::Format format = kv_cache::Format::bf16);

// Gemma 4 global layers remain causal for image tokens. Local layers use
// (causal OR same-image-block) AND (key_position > query_position - 1024):
// image lookahead is permitted, but earlier image tokens still expire from
// the local window. The complete image block must be in the current chunk.
// Ordinary M=1 cached decoding resumes after multimodal prefill.
void image_block_gqa_attention_cached_chunk(
    const BFloat16* query_head_major,
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major, const BFloat16* key_cache,
    const BFloat16* value_cache, std::uint32_t base_position,
    std::uint32_t token_count, std::uint32_t cache_capacity,
    std::uint32_t image_begin, std::uint32_t image_end,
    BFloat16* context_token_major, gemma4_31b::AttentionKind kind,
    cudaStream_t stream = nullptr,
    kv_cache::Format format = kv_cache::Format::bf16);

// Tensor-core runtime attention for 1..4096 actual query rows. In the modular
// backend, visible absolute keys are gathered in <=1024-key macroblocks from
// cache or the uncommitted current K/V tensors. QK and P*V use BF16 tensor operands with FP32 output and
// accumulation. Softmax probabilities are unnormalized BF16 weights; the
// persistent denominator sums those exact rounded values. Tensor QK reduction
// order and rounded weights intentionally permit small drift from the warp
// path. The handle must use host pointer mode; this call assigns it to
// `stream`. `scratch` must provide at least
// tensor_attention_scratch_bytes(token_count) bytes. Cache commit still happens
// separately, after this call. Query and current K head strides use token_count; key macroblocks
// remain at most 1024 rows. Local chunks with at least 768 rows use 256-query
// slices over each slice's visible key union. Packed Q and intermediate state
// reuse the same scratch allocation; current K/V keep their full-chunk strides.
// Supplying fp8 selects native E4M3 QK/PV operands, including current rows.
// Softmax/state/output accumulation stay FP32. The FP8 context owns additional
// packing scratch and prepares/caches host matmul plans for new query shapes.
// A nonempty image_begin/image_end span adds same-image lookahead in local
// layers, retaining the 1024-token past cutoff. Global layers stay causal.
// Query slices still read the complete staged image; cache writes follow all slices.
void causal_gqa_attention_cached_chunk_tensor(
    cublasHandle_t handle, const BFloat16* query_head_major,
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major, const BFloat16* key_cache,
    const BFloat16* value_cache, std::uint32_t base_position,
    std::uint32_t token_count, std::uint32_t cache_capacity, void* scratch,
    BFloat16* context_token_major, gemma4_31b::AttentionKind kind,
    cudaStream_t stream = nullptr,
    kv_cache::Format format = kv_cache::Format::bf16,
    Fp8Attention* fp8 = nullptr,
    std::uint32_t image_begin = 0, std::uint32_t image_end = 0);

// Commit current K [KV,token_count,D] and V [token_count,KV,D] after attention.
// Local writes use the 1024-slot ring; global writes use absolute slots.
void write_kv_cache_chunk(
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major, BFloat16* key_cache,
    BFloat16* value_cache, std::uint32_t base_position,
    std::uint32_t token_count, std::uint32_t cache_capacity,
    gemma4_31b::AttentionKind kind, cudaStream_t stream = nullptr,
    kv_cache::Format format = kv_cache::Format::bf16);

// Explicit global-only compact BF16 cache variant. The cache is
// [4,cache_capacity,640]. Each row stores the 128 position-dependent key
// dimensions (source dimensions 0..63 followed by 256..319), then all 512
// value dimensions. Unrotated key dimensions are reconstructed as a BF16
// multiply of the BF16 value and BF16 K-normalization scale. The same
// reconstruction is used for uncommitted current rows.
void causal_gqa_attention_cached_chunk_global_compact(
    const BFloat16* query_head_major,
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major,
    const BFloat16* compact_kv_cache,
    const BFloat16* global_k_norm_scale, std::uint32_t base_position,
    std::uint32_t token_count, std::uint32_t cache_capacity,
    BFloat16* context_token_major, cudaStream_t stream = nullptr,
    kv_cache::Format format = kv_cache::Format::bf16);

void image_block_gqa_attention_cached_chunk_global_compact(
    const BFloat16* query_head_major,
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major,
    const BFloat16* compact_kv_cache,
    const BFloat16* global_k_norm_scale, std::uint32_t base_position,
    std::uint32_t token_count, std::uint32_t cache_capacity,
    std::uint32_t image_begin, std::uint32_t image_end,
    BFloat16* context_token_major, cudaStream_t stream = nullptr,
    kv_cache::Format format = kv_cache::Format::bf16);

void causal_gqa_attention_cached_chunk_tensor_global_compact(
    cublasHandle_t handle, const BFloat16* query_head_major,
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major,
    const BFloat16* compact_kv_cache,
    const BFloat16* global_k_norm_scale, std::uint32_t base_position,
    std::uint32_t token_count, std::uint32_t cache_capacity, void* scratch,
    BFloat16* context_token_major, cudaStream_t stream = nullptr,
    kv_cache::Format format = kv_cache::Format::bf16,
    Fp8Attention* fp8 = nullptr);

void write_kv_cache_chunk_global_compact(
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major, BFloat16* compact_kv_cache,
    std::uint32_t base_position, std::uint32_t token_count,
    std::uint32_t cache_capacity, cudaStream_t stream = nullptr,
    kv_cache::Format format = kv_cache::Format::bf16);

// Paged compact-global variants use the same [Krot128,V512] BF16 rows without
// rebuilding a contiguous copy of the retained prefix.
void causal_gqa_attention_cached_chunk_global_compact_paged(
    const BFloat16* query_head_major,
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major,
    const CompactGlobalPagedCache& cache,
    const BFloat16* global_k_norm_scale, std::uint32_t base_position,
    std::uint32_t token_count, BFloat16* context_token_major,
    cudaStream_t stream = nullptr);

// Paged image prefill validates the complete image span in the current chunk,
// then applies ordinary causal global attention through the fixed page table.
void image_block_gqa_attention_cached_chunk_global_compact_paged(
    const BFloat16* query_head_major,
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major,
    const CompactGlobalPagedCache& cache,
    const BFloat16* global_k_norm_scale, std::uint32_t base_position,
    std::uint32_t token_count, std::uint32_t image_begin,
    std::uint32_t image_end, BFloat16* context_token_major,
    cudaStream_t stream = nullptr);

void causal_gqa_attention_cached_chunk_tensor_global_compact_paged(
    cublasHandle_t handle, const BFloat16* query_head_major,
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major,
    const CompactGlobalPagedCache& cache,
    const BFloat16* global_k_norm_scale, std::uint32_t base_position,
    std::uint32_t token_count, void* scratch, BFloat16* context_token_major,
    cudaStream_t stream = nullptr, Fp8Attention* fp8 = nullptr);

void write_kv_cache_chunk_global_compact_paged(
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major,
    const CompactGlobalPagedCache& cache, std::uint32_t base_position,
    std::uint32_t token_count, cudaStream_t stream = nullptr);

}  // namespace gewell::prefill_primitives
