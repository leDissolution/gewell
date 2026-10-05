#pragma once

#include "gewell/models/gemma4/31b/model.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace gewell::vision_primitives {

using BFloat16 = __nv_bfloat16;

inline constexpr std::uint32_t kMaxPatchRows =
    gemma4_31b::kVisionMaxSoftTokenCount *
    gemma4_31b::kVisionPoolSize * gemma4_31b::kVisionPoolSize;

// All pointers name device memory and all tensors are contiguous row-major.
// patch_rows is the compact valid prefix: padding rows are never evaluated.
// Input and output regions must not overlap except where explicitly allowed.

// Apply Gemma 4's model-side pixel transform and cast the result to BF16:
// output = BF16(2 * (input - 0.5)). Tensors are [patch_rows,768].
void normalize_patch_values(const float* input, BFloat16* output,
                            std::uint32_t patch_rows,
                            cudaStream_t stream = nullptr);

// Add the learned x/y embeddings to projected patches. projected is
// [patch_rows,1152], position_ids is [patch_rows,2] in (x,y) order, and the
// table is [2,10240,1152]. Both additions preserve their BF16 boundaries.
// projected and output may be the same pointer.
void add_patch_position_embeddings(
    const BFloat16* projected, const std::int32_t* position_ids,
    const BFloat16* position_table, BFloat16* output,
    std::uint32_t patch_rows, cudaStream_t stream = nullptr);

// Apply the pinned theta=100 two-dimensional RoPE to token-major
// [patch_rows,16,72], then transpose to head-major [16,patch_rows,72].
// Each spatial coordinate rotates its own 36-channel half.
void apply_2d_rope_transpose(const BFloat16* input,
                             const std::int32_t* position_ids,
                             BFloat16* output, std::uint32_t patch_rows,
                             cudaStream_t stream = nullptr);

// Layout-only transposes used for V before attention and context afterward.
void token_heads_to_head_tokens(const BFloat16* input, BFloat16* output,
                                std::uint32_t patch_rows,
                                cudaStream_t stream = nullptr);
void head_tokens_to_token_heads(const BFloat16* input, BFloat16* output,
                                std::uint32_t patch_rows,
                                cudaStream_t stream = nullptr);

// Full bidirectional attention, scale=1. Q/K/V are [16,patch_rows,72];
// output is [patch_rows,16,72]. BF16 inputs/output, FP32 online softmax and
// accumulation, with no materialized score/probability matrix or padding keys.
void full_attention(const BFloat16* query, const BFloat16* key,
                    const BFloat16* value, BFloat16* output,
                    std::uint32_t patch_rows, cudaStream_t stream = nullptr);

// Row-wise FP32 softmax with BF16 input and output. This is the boundary used
// after the BF16 QK score matrix. columns may be at most kMaxPatchRows;
// scores and probabilities may be the same pointer.
void softmax_rows(const BFloat16* scores, BFloat16* probabilities,
                  std::uint32_t rows, std::uint32_t columns,
                  cudaStream_t stream = nullptr);

// Position-aware 3x3 average pooling. input is BF16 [patch_rows,1152],
// position_ids contains a complete rectangular x-fastest grid. The device
// kernel derives its width from the final valid position. The average rounds
// to BF16 before sqrt(1152) scaling, producing FP32 [patch_rows/9,1152].
void pool_3x3_scaled(const BFloat16* input,
                     const std::int32_t* position_ids, float* output,
                     std::uint32_t patch_rows,
                     cudaStream_t stream = nullptr);

// output = BF16((input - bias) * scale), with FP32 arithmetic. input/output
// are [rows,1152], while bias and scale are BF16 [1152].
void standardize(const float* input, const BFloat16* bias,
                 const BFloat16* scale, BFloat16* output,
                 std::uint32_t rows, cudaStream_t stream = nullptr);

}  // namespace gewell::vision_primitives
