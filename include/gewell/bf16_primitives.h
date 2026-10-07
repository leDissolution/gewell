#pragma once

#include "gewell/compact_global_cache.h"
#include "gewell/models/gemma4/31b/model.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gewell::bf16_primitives {

using BFloat16 = __nv_bfloat16;

// Pointers are device pointers unless documented otherwise. The caller owns
// storage, the cuBLASLt handle, and stream lifetime. These operations cover the fixed BF16 proof
// paths and row-batched decode operations needed by Gemma 4 31B.
// Input and output regions must not overlap unless an operation explicitly
// documents in-place behavior. trained_scalar and the branch_state argument
// of the two fused residual/norm chains below are in-place.

// output[0, :] = input[0, :] @ weight[:, :]^T. Matrices are row-major,
// accumulation and alpha/beta are FP32, and output is rounded to BF16.
void linear_m1(cublasLtHandle_t handle, const BFloat16* input,
               const BFloat16* weight, BFloat16* output,
               std::uint32_t input_width, std::uint32_t output_width,
               cudaStream_t stream = nullptr);

// Fetch one [5376] embedding row and multiply it by BF16(sqrt(5376)) = 73.5,
// matching Gemma4TextScaledWordEmbedding's BF16 multiply.
void embedding_lookup(const BFloat16* table, std::uint32_t token_id,
                      BFloat16* output, cudaStream_t stream = nullptr);

// Prefill variant: token_ids is a host array, copied into bounded kernel
// parameters before returning. Output is [rows,5376]; IDs must be valid.
void embedding_lookup_host_tokens(const BFloat16* table,
                                  const std::uint32_t* token_ids,
                                  BFloat16* output, std::uint32_t rows,
                                  cudaStream_t stream = nullptr);

// CUDA-graph decode variant. `token_id` is one device-side uint32 so a replay
// can consume the previous replay's result without a host update. The token
// must be in the Gemma 4 vocabulary. This fixed graph surface always requires
// an explicit stream and performs no allocation, synchronization, or copy.
void embedding_lookup_device_token(const BFloat16* table,
                                   const std::uint32_t* token_id,
                                   BFloat16* output, cudaStream_t stream);

struct DeviceTokenEmbeddingInput {
  const std::uint32_t* token_id{};
  BFloat16* output{};
};

// Pointer-batched form for independent device-side token IDs and contiguous
// output rows. Bounded groups of 128 share one launch.
void embedding_lookup_device_token_batch(
    const BFloat16* table,
    const std::vector<DeviceTokenEmbeddingInput>& inputs,
    cudaStream_t stream);

// Batched equivalent: token_ids is [rows] and output is [rows,5376]. All
// token IDs must be valid vocabulary indices. Each row matches the M1 lookup.
// Row counts for the batched decode APIs must fit a positive signed int.
void embedding_lookup_device_tokens(const BFloat16* table,
                                    const std::uint32_t* token_ids,
                                    BFloat16* output, std::uint32_t rows,
                                    cudaStream_t stream);

// Pinned Gemma4RMSNorm semantics: each row has an independent FP32
// sum-of-squares reduction and the learned checkpoint weight is used directly.
// This is not older Gemma's unit-offset (1 + weight) convention. Width must be
// one of the text or vision head/hidden widths; rows covers all independent
// head or residual-stream rows in one launch.
void rms_norm(const BFloat16* input, const BFloat16* weight,
              BFloat16* output, std::uint32_t rows, std::uint32_t width,
              float epsilon = 1.0e-6F, cudaStream_t stream = nullptr);

// Gemma 4 value-head normalization has with_scale=False and therefore has no
// checkpoint weight. Local V uses [16,256]; global shared K=V uses [4,512].
void rms_norm_unscaled(const BFloat16* input, BFloat16* output,
                       std::uint32_t rows, std::uint32_t width,
                       float epsilon = 1.0e-6F,
                       cudaStream_t stream = nullptr);

// Split row-major fused [Q,K,V] (local) or [Q,K] (global) projections while
// normalizing each head. Outputs are separate token-major matrices. Global
// V normalizes the raw K slice without a learned scale. Reduction order and
// rounding match the three separate RMS norms above.
void qkv_rms_norm(const BFloat16* qkv, const BFloat16* q_weight,
                  const BFloat16* k_weight, BFloat16* query, BFloat16* key,
                  BFloat16* value, std::uint32_t rows,
                  gemma4_31b::AttentionKind kind, cudaStream_t stream = nullptr);

struct QkvRopeInput {
  const BFloat16* qkv{};
  const BFloat16* cosine{};
  const BFloat16* sine{};
  BFloat16* query{};
  BFloat16* key{};
  BFloat16* value{};
  std::uint32_t rows{};
};

// Normalize fused projections, apply RoPE, and transpose Q/K into each
// request's private [heads,rows,D] region. V remains token-major. All RMS and
// RoPE BF16 boundaries match qkv_rms_norm followed by separate RoPE kernels.
void qkv_rms_rope_batch(const std::vector<QkvRopeInput>& inputs,
    const BFloat16* q_weight, const BFloat16* k_weight,
    gemma4_31b::AttentionKind kind, cudaStream_t stream);

// Fixed batch-one decode normalization for the 5,376-wide residual stream.
// These operations use the Gemma 4 epsilon and an sm_120-oriented reduction;
// their FP32 reduction order intentionally need not match rms_norm above.
void rms_norm_hidden_m1(const BFloat16* input, const BFloat16* weight,
                        BFloat16* output,
                        cudaStream_t stream = nullptr);

// Normalize the raw attention branch in place, round it to BF16, add the BF16
// residual into branch_state, then normalize that rounded state for the MLP.
void post_attention_residual_pre_feedforward_norm_m1(
    BFloat16* branch_state, const BFloat16* post_attention_weight,
    const BFloat16* residual, const BFloat16* pre_feedforward_weight,
    BFloat16* normalized, cudaStream_t stream = nullptr);

// Normalize the raw MLP branch in place, round every Gemma 4 boundary while
// adding the residual and scalar, then prepare the next layer's input norm.
// next_norm_weight is the final norm weight after layer 59.
void post_feedforward_residual_scalar_next_norm_m1(
    BFloat16* branch_state, const BFloat16* post_feedforward_weight,
    const BFloat16* residual, const BFloat16* scalar,
    const BFloat16* next_norm_weight, BFloat16* next_normalized,
    cudaStream_t stream = nullptr);

// Batched decode norms over contiguous [rows,5376] activations. Each row uses
// the corresponding M1 operation's reduction order and BF16 boundaries.
// Weights [5376] and the one-element layer scalar are shared by all rows.
// The fused chains update branch_state in place; branch_state, residual, and
// normalized output regions must be disjoint, as in the M1 operations.
void rms_norm_hidden_rows(const BFloat16* input, const BFloat16* weight,
                          BFloat16* output, std::uint32_t rows,
                          cudaStream_t stream = nullptr);
void post_attention_residual_pre_feedforward_norm_rows(
    BFloat16* branch_state, const BFloat16* post_attention_weight,
    const BFloat16* residual, const BFloat16* pre_feedforward_weight,
    BFloat16* normalized, std::uint32_t rows,
    cudaStream_t stream = nullptr);
void post_feedforward_residual_scalar_next_norm_rows(
    BFloat16* branch_state, const BFloat16* post_feedforward_weight,
    const BFloat16* residual, const BFloat16* scalar,
    const BFloat16* next_norm_weight, BFloat16* next_normalized,
    std::uint32_t rows, cudaStream_t stream = nullptr);

// BF16 residual addition and the checkpoint's learned per-layer scalar are
// separate operations because each boundary rounds to BF16 in Transformers.
void residual_add(const BFloat16* residual, const BFloat16* branch,
                  BFloat16* output, std::size_t elements,
                  cudaStream_t stream = nullptr);
void trained_scalar(BFloat16* values, const BFloat16* scalar,
                    std::size_t elements, cudaStream_t stream = nullptr);

// Expand one position's local [16,256] or global [4,512] value heads into the
// 32 query-head layout expected by the attention value product.
void expand_value_heads(const BFloat16* compact, BFloat16* expanded,
                        gemma4_31b::AttentionKind kind,
                        cudaStream_t stream = nullptr);

// Generate the exact two-position RoPE factors used by Gemma 4 31B. Outputs
// are token-major [2, D]: local uses D=256, theta=10,000 and rotates every
// dimension; global uses D=512, theta=1,000,000 and rotates only the first 128
// dimensions (64 frequencies in each split half). All four outputs are device
// pointers. Position zero is emitted as the exact BF16 identity.
void generate_rope_factors_m2(BFloat16* local_cos, BFloat16* local_sin,
                              BFloat16* global_cos, BFloat16* global_sin,
                              cudaStream_t stream = nullptr);

// Generate the same Gemma 4 factors for the fixed short-decode positions
// 0..23. Outputs are token-major [24,D] and retain exact BF16 identity values
// in every unrotated dimension.
void generate_rope_factors_24(BFloat16* local_cos, BFloat16* local_sin,
                              BFloat16* global_cos, BFloat16* global_sin,
                              cudaStream_t stream = nullptr);

inline constexpr std::uint32_t kCachedAttentionM1BoundaryPositionCount = 1'026;
inline constexpr std::uint32_t kMaxContextTokenCount = 262'144;

// Generate one exact Gemma 4 RoPE row for an absolute position in the model's
// supported 0..262143 context range.
// Outputs are [256] for local and [512] for global; unlike the fixed-table
// generators, no token-major position dimension is present.
void generate_rope_factors_m1(BFloat16* local_cos, BFloat16* local_sin,
                              BFloat16* global_cos, BFloat16* global_sin,
                              std::uint32_t absolute_position,
                              cudaStream_t stream = nullptr);

struct RopeFactorsM1Input {
  BFloat16* local_cos{};
  BFloat16* local_sin{};
  BFloat16* global_cos{};
  BFloat16* global_sin{};
  std::uint32_t absolute_position{};
};

// Generate independent factor rows in two launches (local and global).
void generate_rope_factors_m1_batch(
    const std::vector<RopeFactorsM1Input>& inputs,
    cudaStream_t stream = nullptr);

// Prefill counterparts keep seven values per thread in registers and use
// a 768-thread FP32 reduction with rsqrtf. Reduction order differs from
// rms_norm; all BF16 boundaries and the disjoint buffer contract are retained.
void post_attention_residual_pre_feedforward_norm_prefill(
    BFloat16* branch_state, const BFloat16* post_attention_weight,
    const BFloat16* residual, const BFloat16* pre_feedforward_weight,
    BFloat16* normalized, std::uint32_t rows, cudaStream_t stream = nullptr);
// A null next_norm_weight omits the terminal norm and leaves next_normalized
// unused; non-final prefill chunks only need the updated residual state.
void post_feedforward_residual_scalar_next_norm_prefill(
    BFloat16* branch_state, const BFloat16* post_feedforward_weight,
    const BFloat16* residual, const BFloat16* scalar,
    const BFloat16* next_norm_weight, BFloat16* next_normalized,
    std::uint32_t rows, cudaStream_t stream = nullptr);

inline constexpr std::uint32_t kGraphPromptTokenCount = 1'024;
inline constexpr std::uint32_t kGraphOutputTokenCount = 512;
inline constexpr std::uint32_t kGraphAttentionPositionCount = 1'536;
inline constexpr std::uint32_t kGraphDecodeFirstPosition =
    kGraphPromptTokenCount;
inline constexpr std::uint32_t kGraphDecodeLastPosition =
    kGraphPromptTokenCount + kGraphOutputTokenCount - 2;
inline constexpr std::uint32_t kGraphDecodeFinalPosition =
    kGraphDecodeLastPosition + 1;

// Fixed CUDA-graph decode variant for positions 1024..1534. Position is one
// device-side uint32 and all other shapes match generate_rope_factors_m1.
void generate_rope_factors_m1_device_position(
    BFloat16* local_cos, BFloat16* local_sin, BFloat16* global_cos,
    BFloat16* global_sin, const std::uint32_t* absolute_position,
    cudaStream_t stream);

// Apply Gemma's split-half RoPE to token-major [2, heads, D] Q or K states and
// transpose into attention's head-major [heads, 2, D] layout. Each multiply
// and the following add round independently to BF16, matching the eager
// reference expression. `heads` must be 32 or the selected kind's KV count.
void apply_rope_transpose_m2(const BFloat16* input, const BFloat16* cos,
                             const BFloat16* sin, BFloat16* output,
                             std::uint32_t heads,
                             gemma4_31b::AttentionKind kind,
                             cudaStream_t stream = nullptr);

// Fixed two-token causal GQA. Q and K are head-major [heads,2,D], while V is
// deliberately consumed directly from RMSNorm's token-major [2,kv_heads,D]
// layout. Probabilities are [32,2,2] and context is token-major [2,32*D]. Dot
// products accumulate in FP32 and round to BF16 before the FP32 softmax;
// probabilities round to BF16 before the FP32-accumulating P*V product.
void causal_gqa_attention_m2(const BFloat16* query, const BFloat16* key,
                             const BFloat16* value_token_major,
                             BFloat16* probabilities,
                             BFloat16* context_token_major,
                             gemma4_31b::AttentionKind kind,
                             cudaStream_t stream = nullptr);

// Apply one position's split-half RoPE without changing the head-major
// [heads,D] layout. The multiply/multiply/add BF16 boundaries are identical to
// apply_rope_transpose_m2. `heads` supports 16/32 query heads or the selected
// kind's 31B KV count. Both models share these RoPE factors and boundaries.
void apply_rope_m1(const BFloat16* input, const BFloat16* cos,
                   const BFloat16* sin, BFloat16* output,
                   std::uint32_t heads, gemma4_31b::AttentionKind kind,
                   cudaStream_t stream = nullptr);

// Row-batched M1 RoPE. Input/output are [rows,heads,D], while cos/sin are
// [rows,D]. Every row preserves apply_rope_m1's BF16 boundaries.
void apply_rope_m1_batch(const BFloat16* input, const BFloat16* cos,
                         const BFloat16* sin, BFloat16* output,
                         std::uint32_t rows, std::uint32_t heads,
                         gemma4_31b::AttentionKind kind,
                         cudaStream_t stream = nullptr);

// Write one head-major [KV,D] K/V pair into separate head-major ring buffers
// [KV,capacity,D]. `absolute_position % capacity` selects the cache slot.
void write_kv_cache_m1(const BFloat16* key, const BFloat16* value,
                       BFloat16* key_cache, BFloat16* value_cache,
                       std::uint32_t absolute_position,
                       std::uint32_t capacity,
                       gemma4_31b::AttentionKind kind,
                       cudaStream_t stream = nullptr,
    kv_cache::Format format = kv_cache::Format::bf16);

inline constexpr std::uint32_t kGlobalCompactKeySize =
    compact_global_cache::kRotatedKeyElements;
inline constexpr std::uint32_t kGlobalCompactKvSize =
    compact_global_cache::kRowElements;

// Global-only compact cache write. Each [640] row stores rotated K dimensions
// [0..63,256..319] followed by all 512 normalized V dimensions. Unrotated K
// is represented implicitly: attention folds the learned KNorm scale into Q
// once per unrotated query dimension, then dots that FP32 value with BF16 V.
void write_kv_cache_m1_global_compact(
    const BFloat16* key, const BFloat16* value,
    BFloat16* compact_kv_cache, std::uint32_t absolute_position,
    std::uint32_t capacity, cudaStream_t stream = nullptr,
    kv_cache::Format format = kv_cache::Format::bf16);

inline constexpr std::uint32_t kGraphAttentionLocalCapacity =
    gemma4_31b::kLocalWindowSize;
inline constexpr std::uint32_t kGraphAttentionGlobalCapacity =
    kGraphAttentionPositionCount;

// Fixed CUDA-graph decode cache write. Position is device-side; local layers
// use a [16,1024,256] ring and global layers use [4,1536,512].
void write_kv_cache_m1_device_position(
    const BFloat16* key, const BFloat16* value, BFloat16* key_cache,
    BFloat16* value_cache, const std::uint32_t* absolute_position,
    gemma4_31b::AttentionKind kind, cudaStream_t stream);

// Fixed cached one-query causal GQA for absolute position 0 or 1. Query is
// [32,D], caches are [KV,capacity,D], probabilities are always [32,2], and
// context is [32,D]. Position zero emits [1,0] without reading slot one;
// position one attends slots zero and one. Scores, probabilities, and P*V
// preserve the same BF16 boundaries as causal_gqa_attention_m2.
void causal_gqa_attention_cached_m1(
    const BFloat16* query, const BFloat16* key_cache,
    const BFloat16* value_cache, std::uint32_t absolute_position,
    std::uint32_t capacity, BFloat16* probabilities, BFloat16* context,
    gemma4_31b::AttentionKind kind, cudaStream_t stream = nullptr);

inline constexpr std::uint32_t kCachedAttentionM1ShortCapacity = 24;

// Model-specific short-decode GQA for one query at absolute positions 0..23.
// Query is [32,D], caches are head-major [KV,capacity,D], probabilities are
// fixed [32,24], and context is [32,D]. Every probability after the visible
// causal prefix is written as exact BF16 +0. Cache positions are addressed
// modulo the caller-provided capacity, which must retain the visible prefix.
void causal_gqa_attention_cached_m1_24(
    const BFloat16* query, const BFloat16* key_cache,
    const BFloat16* value_cache, std::uint32_t absolute_position,
    std::uint32_t capacity, BFloat16* probabilities, BFloat16* context,
    gemma4_31b::AttentionKind kind, cudaStream_t stream = nullptr);

inline constexpr std::uint32_t kCachedAttentionM1BoundaryLocalCapacity =
    gemma4_31b::kLocalWindowSize;
inline constexpr std::uint32_t kCachedAttentionM1BoundaryGlobalCapacity =
    kCachedAttentionM1BoundaryPositionCount;
inline constexpr std::size_t kCachedAttentionM1BoundaryScoreScratchBytes =
    static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) *
    kCachedAttentionM1BoundaryPositionCount * sizeof(BFloat16);

// Fixed boundary-proof GQA for one query at absolute positions 0..1025.
// Local caches are exactly [16,1024,256] and expose
// [max(0, position-1023), position] through modulo slots. Global caches are
// exactly [4,1026,512] and expose [0,position]. Probabilities are absolute-
// indexed [32,1026], with exact BF16 +0 before/after the visible interval;
// context is [32,D]. `score_scratch` is caller-owned BF16 storage for
// [32,1026], may be overwritten, and must not alias another argument.
void causal_gqa_attention_cached_m1_boundary(
    const BFloat16* query, const BFloat16* key_cache,
    const BFloat16* value_cache, std::uint32_t absolute_position,
    BFloat16* score_scratch, BFloat16* probabilities, BFloat16* context,
    gemma4_31b::AttentionKind kind, cudaStream_t stream = nullptr);

inline constexpr std::size_t kGraphAttentionScoreScratchBytes =
    static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) *
    kGraphAttentionPositionCount * sizeof(BFloat16);

// Fixed CUDA-graph decode GQA for device-side positions 1024..1534. Local
// visibility is [position-1023,position] in a 1024-slot ring; global visibility
// is [0,position] in 1536 contiguous slots. Scores and probabilities are
// [32,1536]. Only visible score entries are defined; probabilities are exact
// BF16 +0 outside the visible interval. `score_scratch` is caller-owned and
// may be overwritten.
void causal_gqa_attention_cached_m1_device_position(
    const BFloat16* query, const BFloat16* key_cache,
    const BFloat16* value_cache,
    const std::uint32_t* absolute_position, BFloat16* score_scratch,
    BFloat16* probabilities, BFloat16* context,
    gemma4_31b::AttentionKind kind, cudaStream_t stream);

inline constexpr std::uint32_t kGraphAttentionFusedLocalSplitCount = 32;
inline constexpr std::uint32_t kGraphAttentionFusedGlobalSplitCount = 48;
inline constexpr std::size_t kGraphAttentionFusedScratchBytes =
    static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) *
    kGraphAttentionFusedGlobalSplitCount *
    (gemma4_31b::kGlobalHeadSize + 2) * sizeof(float);

// Graph-only split-GQA decode attention. K and V retain their independent
// [kv_heads,capacity,D] caches. QK, split softmax, and partial P*V are fused;
// `scratch` holds FP32 partial contexts and log-sum-exp metadata and must have
// kGraphAttentionFusedScratchBytes available. The split reduction order and
// FP32 softmax intentionally need not reproduce the reference BF16 score and
// probability boundaries.
void causal_gqa_attention_cached_m1_device_position_fused(
    const BFloat16* query, const BFloat16* key_cache,
    const BFloat16* value_cache,
    const std::uint32_t* absolute_position, void* scratch,
    BFloat16* context, gemma4_31b::AttentionKind kind, cudaStream_t stream);

// Caller-owned mixed-precision storage needed by the arbitrary-context fused
// attention below. Global attention stores 32-token partial context numerators
// in FP16 and its log-sum-exp metadata in FP32; long compact-global attention
// also reserves FP32 storage for a 64-way hierarchical reduction. Local
// attention retains FP32 for both.
// Local attention requires the exact 1024-slot ring capacity. Global attention
// accepts any request capacity in 1..262144. The function validates the
// contract and throws on invalid input or a size_t overflow.
std::size_t causal_gqa_attention_cached_m1_fused_scratch_bytes(
    std::uint32_t capacity, gemma4_31b::AttentionKind kind);

// Arbitrary-context, host-position split-GQA decode attention. Local K/V are
// separate [16,1024,256] rings and visibility saturates to
// [max(0,position-1023),position]. Global K/V are separate
// [4,capacity,512] arrays and visibility is [0,position], so capacity must be
// at least position+1. QK, split softmax, and partial P*V accumulate in FP32.
// Global partial numerators round once to FP16 before the FP32 final reduction;
// local partial numerators remain FP32. Context rounds once to BF16. The caller
// must provide the exact scratch size returned above for the same capacity and
// kind.
void causal_gqa_attention_cached_m1_fused(
    const BFloat16* query, const BFloat16* key_cache,
    const BFloat16* value_cache, std::uint32_t absolute_position,
    std::uint32_t capacity, void* scratch, BFloat16* context,
    gemma4_31b::AttentionKind kind, cudaStream_t stream = nullptr,
    kv_cache::Format format = kv_cache::Format::bf16);

struct FrozenLocalAttentionInput {
  const BFloat16* query{};
  const BFloat16* key_cache{};
  const BFloat16* value_cache{};
  std::uint32_t absolute_position{};
  BFloat16* context{};
  kv_cache::Format format{kv_cache::Format::bf16};
};

// Read-only local attention over independent frozen 1,024-token rings. Each
// request retains the single-request split boundaries and FP32 partials above.
// scratch_size must fit at least one local request; additional whole-request
// regions permit up to 32 requests to share each pair of kernel launches.
void causal_gqa_attention_cached_m1_fused_local_batch(
    const std::vector<FrozenLocalAttentionInput>& inputs, void* scratch,
    std::size_t scratch_size, cudaStream_t stream = nullptr);

// Global-only arbitrary-context attention over [4,capacity,640] compact rows.
// k_norm_scale is the learned BF16 [512] global KNorm scale. Its unrotated
// dimensions are folded into FP32 Q once per query; rotated Q/K dimensions are
// unchanged. Each 32-token compact tile is loaded once and retained for both
// QK and P*V. From 1,024 visible splits, BF16 Tensor Cores evaluate QK and
// P*V while the persistent compact cache and softmax/reduction metadata retain
// their FP32 storage; shorter contexts use the scalar FP32 path.
void causal_gqa_attention_cached_m1_fused_global_compact(
    const BFloat16* query, const BFloat16* compact_kv_cache,
    const BFloat16* k_norm_scale, std::uint32_t absolute_position,
    std::uint32_t capacity, void* scratch, BFloat16* context,
    cudaStream_t stream = nullptr,
    kv_cache::Format format = kv_cache::Format::bf16);

// The same split-K decode path over non-contiguous 256-token compact-global
// pages. The page table contains BF16-element offsets from page_pool and must
// cover absolute_position.
void causal_gqa_attention_cached_m1_fused_global_compact_paged(
    const BFloat16* query,
    const compact_global_cache::PagedView<BFloat16>& cache,
    const BFloat16* k_norm_scale, std::uint32_t absolute_position,
    void* scratch, BFloat16* context, cudaStream_t stream = nullptr);

// One normalized Q/K/V row and its private cache. Local caches have exactly
// 1024 slots; global caches use the paged compact layout above. rotated_query
// is caller-owned [32,D] scratch. All input/output regions are disjoint.
struct DecodeAttentionInput {
  const BFloat16* query{};
  const BFloat16* key{};
  const BFloat16* value{};
  const BFloat16* cosine{};
  const BFloat16* sine{};
  BFloat16* rotated_query{};
  BFloat16* local_key{};
  BFloat16* local_value{};
  compact_global_cache::PagedView<BFloat16> global_cache{};
  std::uint32_t position{};
  BFloat16* context{};
  kv_cache::Format format{kv_cache::Format::bf16};
};

// Apply RoPE, write each current KV row, and attend across independent
// requests. Bounded groups partition the supplied scratch without allocating,
// copying descriptors to the device, or synchronizing. scratch_size must fit
// the largest individual request; larger storage permits concurrent requests.
// Split boundaries, partial precision, and reduction order match M1 above.
void decode_attention_batch(const std::vector<DecodeAttentionInput>& inputs,
    const BFloat16* k_norm_scale, void* scratch, std::size_t scratch_size,
    gemma4_31b::AttentionKind kind, cudaStream_t stream);

// Seed and advance the deliberately fixed prompt-1024/output-512 greedy
// decode state. All pointers are device pointers. Seed stores first_token in
// current_token and outputs[0], and sets absolute_position to 1024. Commit
// consumes next_token at position p, stores outputs[p-1023], updates
// current_token, and increments p. Commit is a no-op outside 1024..1534 so an
// accidental extra graph replay cannot write beyond outputs[511].
void seed_graph_decode_state(const std::uint32_t* first_token,
                             std::uint32_t* current_token,
                             std::uint32_t* absolute_position,
                             std::uint32_t* outputs, cudaStream_t stream);
void commit_graph_decode_state(const std::uint32_t* next_token,
                               std::uint32_t* current_token,
                               std::uint32_t* absolute_position,
                               std::uint32_t* outputs,
                               cudaStream_t stream);

// output = BF16(BF16(gelu_tanh(gate)) * up), preserving the activation's BF16
// boundary before the gated product.
void gelu_tanh_multiply(const BFloat16* gate, const BFloat16* up,
                        BFloat16* output, std::size_t elements,
                        cudaStream_t stream = nullptr);

// Same BF16 activation boundaries, reading a fused [rows,2*width] GEMM.
void gelu_tanh_multiply_interleaved(const BFloat16* gate_up, BFloat16* output,
                                   std::uint32_t rows, std::uint32_t width,
                                   cudaStream_t stream = nullptr);

// Apply Gemma's three BF16 softcap boundaries (divide, tanh, multiply), then
// expose all capped elements without a token selection.
void softcap_logits(const BFloat16* logits, BFloat16* capped,
                    std::uint32_t elements, float cap = 30.0F,
                    cudaStream_t stream = nullptr);

// Apply the same softcap and
// choose the largest result. Equal values deterministically choose the lowest
// token id. argmax is one device-side uint32.
void softcap_and_argmax(const BFloat16* logits, BFloat16* capped,
                        std::uint32_t* argmax, std::uint32_t elements,
                        float cap = 30.0F, cudaStream_t stream = nullptr);

// The same softcap and lowest-token-id argmax independently on contiguous
// [rows,elements] logits. Capped logits retain every row for later sampling.
void softcap_and_argmax_rows(const BFloat16* logits, BFloat16* capped,
    std::uint32_t* argmax, std::uint32_t rows, std::uint32_t elements,
    float cap, cudaStream_t stream);

// Reusable storage and execution contract for device-side temperature,
// top-k, and top-p sampling. Logits are already post-processed by the model
// (for example, soft-capped). `uniform` must be finite and in [0, 1). The
// selected token remains on device so autoregressive decode does not need a
// host round trip. Equal logits are ranked by ascending token ID.
std::size_t sampling_scratch_bytes(std::uint32_t elements);
void sample_top_k_top_p(const BFloat16* logits, std::uint32_t elements,
                        float temperature, float top_p, std::uint32_t top_k,
                        float uniform, void* scratch,
                        std::size_t scratch_bytes, std::uint32_t* selected,
                        cudaStream_t stream = nullptr);

}  // namespace gewell::bf16_primitives
