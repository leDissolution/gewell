#include "prefill_common.cuh"

namespace gewell::prefill_primitives {
namespace {

using namespace detail;

struct TensorAttentionScratchLayout {
  static constexpr std::size_t kStagedKey = 0;
  static constexpr std::size_t kStagedValue =
      kStagedKey + kTensorAttentionStagedBytes;
  static constexpr std::size_t kScores =
      kStagedValue + kTensorAttentionStagedBytes;
  std::size_t probabilities;
  std::size_t numerator;
  std::size_t maximum;
  std::size_t denominator;
  std::size_t bytes;

  explicit constexpr TensorAttentionScratchLayout(std::uint32_t rows)
      : probabilities(kScores + kTensorAttentionScoreBytes / kTokenCount * rows),
        numerator(probabilities +
                  kTensorAttentionProbabilityBytes / kTokenCount * rows),
        maximum(numerator +
                kTensorAttentionNumeratorBytes / kTokenCount * rows),
        denominator((maximum + kTensorAttentionStateBytes / kTokenCount * rows +
                     255) / 256 * 256),
        bytes((denominator + kTensorAttentionStateBytes / kTokenCount * rows +
               255) / 256 * 256) {}
};

static_assert(TensorAttentionScratchLayout::kStagedValue >
              TensorAttentionScratchLayout::kStagedKey);
static_assert(TensorAttentionScratchLayout::kScores >
              TensorAttentionScratchLayout::kStagedValue);
constexpr bool tensor_scratch_layout_is_valid() {
  for (std::uint32_t rows = 1; rows <= kTensorAttentionMaximumQueryRows; ++rows) {
    const TensorAttentionScratchLayout layout(rows);
    if (layout.probabilities <= layout.kScores ||
        layout.numerator <= layout.probabilities ||
        layout.maximum <= layout.numerator ||
        layout.denominator <= layout.maximum ||
        layout.bytes != tensor_attention_scratch_bytes(rows) ||
        layout.kStagedKey % 256 || layout.kStagedValue % 256 ||
        layout.kScores % 256 || layout.probabilities % 256 ||
        layout.numerator % 256 || layout.maximum % 256 ||
        layout.denominator % 256 || layout.bytes % 256) {
      return false;
    }
  }
  return true;
}

static_assert(tensor_scratch_layout_is_valid());

template <std::uint32_t HeadSize, std::uint32_t KvHeads, bool Local>
__global__ void gather_tensor_attention_tile_kernel(
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major, const BFloat16* key_cache,
    const BFloat16* value_cache, std::uint32_t base_position,
    std::uint32_t token_count, std::uint32_t cache_capacity,
    std::uint32_t tile_start,
    std::uint32_t tile_count, std::uint32_t staged_count, BFloat16* staged_key,
    BFloat16* staged_value, std::size_t elements, kv_cache::Format format) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }

  const std::uint32_t dimension =
      static_cast<std::uint32_t>(index % HeadSize);
  const std::size_t row = index / HeadSize;
  const std::uint32_t tile_position =
      static_cast<std::uint32_t>(row % staged_count);
  const std::uint32_t kv_head =
      static_cast<std::uint32_t>(row / staged_count);
  const std::size_t staged_index =
      (static_cast<std::size_t>(kv_head) * kTensorAttentionTileTokens +
       tile_position) * HeadSize + dimension;
  // Padded keys are not history: do not read current K/V or a wrapped ring
  // slot for them. Both operands must be finite even with zero probability.
  if (tile_position >= tile_count) {
    staged_key[staged_index] = __float2bfloat16_rn(0.0F);
    staged_value[staged_index] = __float2bfloat16_rn(0.0F);
    return;
  }
  const std::uint32_t absolute_position = tile_start + tile_position;
  const bool current = absolute_position >= base_position;
  const std::uint32_t current_position =
      current ? absolute_position - base_position : 0;
  const std::uint32_t cache_position =
      Local ? absolute_position % gemma4_31b::kLocalWindowSize
            : absolute_position;

  const std::size_t current_key_index =
      (static_cast<std::size_t>(kv_head) * token_count + current_position) *
          HeadSize +
      dimension;
  const std::size_t current_value_index =
      (static_cast<std::size_t>(current_position) * KvHeads + kv_head) *
          HeadSize +
      dimension;
  const std::size_t cache_index =
      (static_cast<std::size_t>(kv_head) * cache_capacity + cache_position) *
          HeadSize +
      dimension;
  staged_key[staged_index] =
      current ? current_key_head_major[current_key_index]
              : kv_storage::load(kv_storage::row(key_cache, cache_index / HeadSize, HeadSize, format), dimension, HeadSize, format);
  staged_value[staged_index] =
      current ? current_value_token_major[current_value_index]
              : kv_storage::load(kv_storage::row(value_cache, cache_index / HeadSize, HeadSize, format), dimension, HeadSize, format);
}

__global__ void gather_tensor_attention_tile_global_compact_kernel(
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major,
    const BFloat16* compact_kv_cache,
    const BFloat16* global_k_norm_scale,
    std::uint32_t base_position, std::uint32_t token_count,
    std::uint32_t cache_capacity, std::uint32_t tile_start,
    std::uint32_t tile_count,
    BFloat16* staged_key, BFloat16* staged_value,
    std::size_t elements, kv_cache::Format format) {
  constexpr std::uint32_t kHeadSize = gemma4_31b::kGlobalHeadSize;
  constexpr std::uint32_t kKvHeads = gemma4_31b::kGlobalKvHeadCount;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }

  const std::uint32_t dimension =
      static_cast<std::uint32_t>(index % kHeadSize);
  const std::size_t row = index / kHeadSize;
  const std::uint32_t tile_position =
      static_cast<std::uint32_t>(row % tile_count);
  const std::uint32_t kv_head =
      static_cast<std::uint32_t>(row / tile_count);
  const std::uint32_t absolute_position = tile_start + tile_position;
  const bool current = absolute_position >= base_position;
  const std::uint32_t current_position =
      current ? absolute_position - base_position : 0;
  const std::size_t current_key_index =
      (static_cast<std::size_t>(kv_head) * token_count + current_position) *
          kHeadSize +
      dimension;
  const std::size_t current_value_index =
      (static_cast<std::size_t>(current_position) * kKvHeads + kv_head) *
          kHeadSize +
      dimension;
  const BFloat16* cache_row = current ? nullptr : kv_storage::row(
      compact_kv_cache, std::size_t(kv_head) * cache_capacity + absolute_position,
      640, format, 2);
  const std::size_t staged_index =
      (static_cast<std::size_t>(kv_head) *
           kTensorAttentionTileTokens +
       tile_position) *
          kHeadSize +
      dimension;
  const BFloat16 value =
      current
          ? current_value_token_major[current_value_index]
          : kv_storage::load(cache_row, 128 + dimension, 640, format, 128);
  const BFloat16 key =
      is_compact_global_key_dimension(dimension)
          ? (current
                 ? current_key_head_major[current_key_index]
                 : kv_storage::load(cache_row, compact_global_key_index(dimension), 640, format, 128))
          : compact_global_reconstructed_key(
                value, global_k_norm_scale[dimension]);
  staged_key[staged_index] = key;
  staged_value[staged_index] = value;
}

__global__ void gather_tensor_attention_tile_global_compact_paged_kernel(
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major,
    const BFloat16* page_pool, const std::uint64_t* page_offsets,
    const BFloat16* global_k_norm_scale, std::uint32_t page_tokens,
    std::size_t layer_offset_elements, std::uint32_t base_position,
    std::uint32_t token_count, std::uint32_t tile_start,
    std::uint32_t tile_count,
    BFloat16* staged_key, BFloat16* staged_value,
    std::size_t elements, kv_cache::Format format) {
  constexpr std::uint32_t kHeadSize = gemma4_31b::kGlobalHeadSize;
  constexpr std::uint32_t kKvHeads = gemma4_31b::kGlobalKvHeadCount;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }

  const std::uint32_t dimension =
      static_cast<std::uint32_t>(index % kHeadSize);
  const std::size_t row = index / kHeadSize;
  const std::uint32_t tile_position =
      static_cast<std::uint32_t>(row % tile_count);
  const std::uint32_t kv_head =
      static_cast<std::uint32_t>(row / tile_count);
  const std::uint32_t absolute_position = tile_start + tile_position;
  const bool current = absolute_position >= base_position;
  const std::uint32_t current_position =
      current ? absolute_position - base_position : 0;
  const std::size_t current_key_index =
      (static_cast<std::size_t>(kv_head) * token_count + current_position) *
          kHeadSize +
      dimension;
  const std::size_t current_value_index =
      (static_cast<std::size_t>(current_position) * kKvHeads + kv_head) *
          kHeadSize +
      dimension;
  const BFloat16* const cache_row =
      current ? nullptr
              : compact_global_cache_row<true>(
                    page_pool, page_offsets, page_tokens,
                    layer_offset_elements, kv_head, absolute_position, 0, format);
  const std::size_t staged_index =
      (static_cast<std::size_t>(kv_head) * kTensorAttentionTileTokens +
       tile_position) *
          kHeadSize +
      dimension;
  const BFloat16 value =
      current
          ? current_value_token_major[current_value_index]
          : kv_storage::load(cache_row, 128 + dimension, 640, format, 128);
  const BFloat16 key =
      is_compact_global_key_dimension(dimension)
          ? (current
                 ? current_key_head_major[current_key_index]
                 : kv_storage::load(cache_row, compact_global_key_index(dimension), 640, format, 128))
          : compact_global_reconstructed_key(
                value, global_k_norm_scale[dimension]);
  staged_key[staged_index] = key;
  staged_value[staged_index] = value;
}

template <std::uint32_t HeadSize, bool Local, bool Fp8 = false, bool Image = false>
__global__ void tensor_attention_softmax_update_kernel(
    const float* scores, BFloat16* probabilities, float* numerator,
    float* row_maximum, float* row_denominator,
    std::uint32_t base_position, std::uint32_t token_count,
    std::uint32_t tile_start, std::uint32_t tile_count, bool first_tile,
    const float* query_scales, const float* key_scales,
    std::uint32_t image_begin, std::uint32_t image_end,
    const float* value_token_scales = nullptr, const float* value_output_scales = nullptr) {
  static_assert(HeadSize <= kTensorAttentionMaximumHeadSize);
  // One warp owns a row: eight rows share a CTA, and all reductions and
  // scores stay in registers. No communication between warps is needed.
  const unsigned lane = threadIdx.x % kWarpSize;
  const std::uint32_t query_head = blockIdx.x;
  const std::uint32_t query_position =
      blockIdx.y * kWarpsPerBlock + threadIdx.x / kWarpSize;
  if (query_position >= token_count) return;
  const std::uint32_t absolute_query_position = base_position + query_position;
  const std::size_t state_index =
      static_cast<std::size_t>(query_head) * token_count + query_position;
  const std::size_t tile_row = state_index * kTensorAttentionTileTokens;
  const std::size_t numerator_row = state_index * kTensorAttentionMaximumHeadSize;
  const float negative_infinity = -CUDART_INF_F;
  constexpr unsigned kScoresPerLane = kTensorAttentionTileTokens / kWarpSize;
  static_assert(kTensorAttentionTileTokens == 1024);
  float values[kScoresPerLane];
  float maximum = negative_infinity;
#pragma unroll
  for (unsigned i = 0; i < kScoresPerLane; ++i) {
    const unsigned position = lane + i * kWarpSize;
    const unsigned absolute_key = tile_start + position;
    const bool same_image = Image && absolute_query_position >= image_begin &&
        absolute_query_position < image_end && absolute_key >= image_begin && absolute_key < image_end;
    const bool visible = position < tile_count &&
        (absolute_key <= absolute_query_position || same_image) &&
        (!Local || (Image && absolute_key >= absolute_query_position) || absolute_query_position - absolute_key <
                       gemma4_31b::kLocalWindowSize);
    float score = visible ? scores[tile_row + position] : 0.0F;
    if constexpr (Fp8) {
      constexpr unsigned KvHeads = Local ? 16 : 4;
      const unsigned kv_head = query_head / (32 / KvHeads);
      score *= query_scales[state_index] * key_scales[kv_head * kTensorAttentionTileTokens + position];
    }
    values[i] = visible ? score : negative_infinity;
    maximum = fmaxf(maximum, values[i]);
  }
#pragma unroll
  for (unsigned delta = kWarpSize / 2; delta; delta /= 2)
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, delta));
  maximum = __shfl_sync(0xffffffffU, maximum, 0);

  float previous_maximum = negative_infinity, previous_denominator = 0.0F;
  if (lane == 0 && !first_tile) {
    previous_maximum = row_maximum[state_index];
    previous_denominator = row_denominator[state_index];
  }
  previous_maximum = __shfl_sync(0xffffffffU, previous_maximum, 0);
  const float next_maximum = fmaxf(previous_maximum, maximum);
  const float old_scale = maximum == negative_infinity
      ? (first_tile ? 0.0F : 1.0F)
      : (previous_maximum == negative_infinity ? 0.0F
                                              : expf(previous_maximum - next_maximum));
  if (lane == 0) row_maximum[state_index] = next_maximum;

  // Most later key tiles do not raise the running maximum. Multiplication by
  // one leaves the FP32 numerator unchanged, so avoid that entire memory pass.
  // The first P*V GEMM overwrites the numerator with beta=0; it needs no clear.
  if (!first_tile && old_scale != 1.0F) {
    for (unsigned dimension = lane; dimension < HeadSize; dimension += kWarpSize)
      numerator[numerator_row + dimension] *= old_scale;
  }

  // Preserve the former 256-thread denominator tree: each lane reproduces
  // eight old threads, each summing four keys spaced 256 apart. Merge those
  // partials at strides 128, 64, 32 before the final warp reduction.
  float sums[kWarpsPerBlock] = {};
#pragma unroll
  for (unsigned i = 0; i < kScoresPerLane; ++i) {
    const unsigned position = lane + i * kWarpSize;
    const float exponential = values[i] == negative_infinity ? 0.0F : expf(values[i] - next_maximum);
    if constexpr (Fp8) {
      constexpr unsigned KvHeads = Local ? 16 : 4;
      const unsigned kv_head = query_head / (32 / KvHeads);
      const float scaled = value_token_scales
          ? exponential * value_token_scales[kv_head * kTensorAttentionTileTokens + position] /
                value_output_scales[kv_head * HeadSize]
          : exponential * 256.0F;
      const auto weight = __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
      if (value_token_scales) sums[i % kWarpsPerBlock] += exponential;
      else {
        const auto half = __nv_cvt_fp8_to_halfraw(weight, __NV_E4M3);
        sums[i % kWarpsPerBlock] += __half2float(static_cast<__half>(half)) / 256.0F;
      }
      // Native P*V contracts the full padded tile. Masked/future entries must
      // be explicitly zero even when tile_count is not a multiple of 16.
      reinterpret_cast<unsigned char*>(probabilities)[tile_row + position] = weight;
    } else {
      const auto weight = __float2bfloat16_rn(exponential);
      sums[i % kWarpsPerBlock] += __bfloat162float(weight);
      // Local BF16 GEMMs round the key width to eight. Visibility and the
      // denominator still use the real tile_count; padded probabilities are 0.
      const unsigned stored_columns = Local ? (tile_count + 7U) / 8U * 8U : tile_count;
      if (position < stored_columns) probabilities[tile_row + position] = weight;
    }
  }
#pragma unroll
  for (unsigned stride = kWarpsPerBlock / 2; stride; stride /= 2) {
#pragma unroll
    for (unsigned i = 0; i < stride; ++i) sums[i] += sums[i + stride];
  }
  float sum = sums[0];
#pragma unroll
  for (unsigned delta = kWarpSize / 2; delta; delta /= 2)
    sum += __shfl_down_sync(0xffffffffU, sum, delta);
  if (lane == 0) {
    // Keep the denominator's scale/add rounding separate, as in the original
    // shared-memory implementation (which stored the scaled value first).
    row_denominator[state_index] = __fmul_rn(previous_denominator, old_scale) + sum;
  }
}

template <std::uint32_t HeadSize>
__global__ void tensor_attention_finalize_kernel(
    const float* numerator, const float* row_denominator,
    BFloat16* context_token_major, std::uint32_t token_count,
    std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  const std::uint32_t dimension =
      static_cast<std::uint32_t>(index % HeadSize);
  const std::size_t row = index / HeadSize;
  const std::uint32_t query_head =
      static_cast<std::uint32_t>(row % gemma4_31b::kQueryHeadCount);
  const std::uint32_t query_position = static_cast<std::uint32_t>(
      row / gemma4_31b::kQueryHeadCount);
  const std::size_t state_index =
      static_cast<std::size_t>(query_head) * token_count + query_position;
  const std::size_t numerator_index =
      state_index * kTensorAttentionMaximumHeadSize + dimension;
  context_token_major[index] = __float2bfloat16_rn(
      numerator[numerator_index] / row_denominator[state_index]);
}

// Pack one query slice so its GQA repeats remain contiguous for cuBLAS.
// Current K/V retain their full-chunk strides until attention finishes.
template <std::uint32_t HeadSize>
__global__ void gather_tensor_attention_query_kernel(
    const BFloat16* query, BFloat16* staged_query, std::uint32_t token_count,
    std::uint32_t query_start, std::uint32_t query_count) {
  constexpr unsigned kHeadSize = HeadSize;
  const unsigned dimension = threadIdx.x;
  const unsigned position = blockIdx.x;
  const unsigned head = blockIdx.y;
  staged_query[(std::size_t(head) * query_count + position) * kHeadSize + dimension] =
      query[(std::size_t(head) * token_count + query_start + position) *
                kHeadSize + dimension];
}

enum class TensorCacheLayout {
  separate,
  compact_global,
  paged_compact_global,
};

template <std::uint32_t HeadSize, std::uint32_t KvHeads, bool Local,
          TensorCacheLayout CacheLayout, bool Image = false>
void causal_gqa_attention_cached_chunk_tensor_impl(
    cublasHandle_t handle, const BFloat16* query_head_major,
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major, const BFloat16* key_cache,
    const BFloat16* value_cache, const BFloat16* global_k_norm_scale,
    const CompactGlobalPagedCache* paged_cache,
    std::uint32_t base_position, std::uint32_t token_count,
    std::uint32_t cache_capacity, void* scratch,
    BFloat16* context_token_major, cudaStream_t stream, kv_cache::Format format,
    Fp8Attention* fp8, std::uint32_t image_begin = 0, std::uint32_t image_end = 0) {
  static_assert(gemma4_31b::kQueryHeadCount % KvHeads == 0);
  static_assert((CacheLayout != TensorCacheLayout::compact_global &&
                 CacheLayout != TensorCacheLayout::paged_compact_global) ||
                (HeadSize == gemma4_31b::kGlobalHeadSize &&
                 KvHeads == gemma4_31b::kGlobalKvHeadCount && !Local));
  constexpr std::uint32_t kRepeats =
      gemma4_31b::kQueryHeadCount / KvHeads;
  // FP8 contracts padded 1024-key tiles; wider queries amortize packing
  // without increasing its local padded matmul work.
  const std::uint32_t query_tile_rows = fp8 ? 512 : 256;
  // Local slices shrink the sliding-window union. Global slices bound the
  // score/probability working set, including ragged BF16 chunks. FP8 retains
  // its existing complete-tile gate to amortize the extra packing passes.
  constexpr std::uint32_t kSlicedMinimumRows = 768;
  const bool slice_queries = token_count >= kSlicedMinimumRows &&
      (Local || !fp8 || token_count % 256 == 0);
  const std::uint32_t query_capacity =
      slice_queries ? query_tile_rows : token_count;
  const TensorAttentionScratchLayout scratch_layout(query_capacity);
  // A slice and its packed Q tile fit within the existing full-chunk allocation.
  static_assert(TensorAttentionScratchLayout(512).bytes +
                    std::size_t(gemma4_31b::kQueryHeadCount) * 512 *
                        HeadSize * sizeof(BFloat16) <=
                tensor_attention_scratch_bytes(kSlicedMinimumRows));
  auto* const scratch_bytes = static_cast<std::uint8_t*>(scratch);
  auto* const staged_key = reinterpret_cast<BFloat16*>(
      scratch_bytes + TensorAttentionScratchLayout::kStagedKey);
  auto* const staged_value = reinterpret_cast<BFloat16*>(
      scratch_bytes + TensorAttentionScratchLayout::kStagedValue);
  auto* const scores = reinterpret_cast<float*>(
      scratch_bytes + TensorAttentionScratchLayout::kScores);
  auto* const probabilities = reinterpret_cast<BFloat16*>(
      scratch_bytes + scratch_layout.probabilities);
  auto* const numerator = reinterpret_cast<float*>(
      scratch_bytes + scratch_layout.numerator);
  auto* const row_maximum = reinterpret_cast<float*>(
      scratch_bytes + scratch_layout.maximum);
  auto* const row_denominator = reinterpret_cast<float*>(
      scratch_bytes + scratch_layout.denominator);

  check_cublas(cublasSetStream(handle, stream),
               "set tensor attention cuBLAS stream");
  for (std::uint32_t query_start = 0; query_start < token_count;
       query_start += query_capacity) {
    const std::uint32_t query_count =
        std::min(query_capacity, token_count - query_start);
    const std::uint32_t query_base = base_position + query_start;
    const BFloat16* query = query_head_major;
    if (slice_queries) {
      auto* staged_query = reinterpret_cast<BFloat16*>(scratch_bytes + scratch_layout.bytes);
      gather_tensor_attention_query_kernel<HeadSize>
          <<<dim3(query_count, gemma4_31b::kQueryHeadCount),
             HeadSize, 0, stream>>>(
              query_head_major, staged_query, token_count, query_start, query_count);
      check_cuda(cudaGetLastError(), "tensor attention Q gather kernel launch");
      query = staged_query;
    }
    if (fp8) fp8->prepare(query, query_count, HeadSize, KvHeads, stream);
    const std::uint32_t visible_begin =
        Local && query_base >= gemma4_31b::kLocalWindowSize - 1
            ? query_base - (gemma4_31b::kLocalWindowSize - 1)
            : 0;
    const std::uint32_t query_end = query_base + query_count;
    const std::uint32_t visible_end = Image && query_base < image_end && query_end > image_begin
        ? std::max(query_end, image_end) : query_end;
    bool first_tile = true;
    for (std::uint32_t tile_start = visible_begin; tile_start < visible_end;) {
      const std::uint32_t tile_count =
          std::min(kTensorAttentionTileTokens, visible_end - tile_start);
      const std::uint32_t staged_count =
          Local && !fp8 ? (tile_count + 7U) / 8U * 8U : tile_count;
      const std::size_t staged_elements =
          static_cast<std::size_t>(KvHeads) * staged_count * HeadSize;
      const bool direct_cache = fp8 && format == kv_cache::Format::fp8 &&
          CacheLayout != TensorCacheLayout::separate;
      if (!direct_cache) {
      if constexpr (CacheLayout == TensorCacheLayout::separate) {
        gather_tensor_attention_tile_kernel<HeadSize, KvHeads, Local>
            <<<blocks_for(staged_elements), kThreads, 0, stream>>>(
                current_key_head_major, current_value_token_major, key_cache,
                value_cache, base_position, token_count, cache_capacity,
                tile_start, tile_count, staged_count, staged_key, staged_value,
                staged_elements, format);
      } else if constexpr (CacheLayout == TensorCacheLayout::compact_global) {
        gather_tensor_attention_tile_global_compact_kernel
            <<<blocks_for(staged_elements), kThreads, 0, stream>>>(
                current_key_head_major, current_value_token_major, key_cache,
                global_k_norm_scale, base_position, token_count,
                cache_capacity, tile_start, tile_count, staged_key,
                staged_value, staged_elements, format);
      } else {
        gather_tensor_attention_tile_global_compact_paged_kernel
            <<<blocks_for(staged_elements), kThreads, 0, stream>>>(
                current_key_head_major, current_value_token_major,
                paged_cache->page_pool, paged_cache->page_offsets,
                global_k_norm_scale, paged_cache->page_tokens,
                paged_cache->layer_offset_elements, base_position,
                token_count, tile_start, tile_count, staged_key, staged_value,
                staged_elements, format);
      }
      check_cuda(cudaGetLastError(),
                 "tensor attention K/V gather kernel launch");
      }

      constexpr int kInner = static_cast<int>(HeadSize);
      const int grouped_rows = static_cast<int>(kRepeats * query_count);
      constexpr int kNumeratorLeadingDimension =
          static_cast<int>(kTensorAttentionMaximumHeadSize);
      constexpr long long kStagedHeadStride =
          static_cast<long long>(kTensorAttentionTileTokens) * HeadSize;
      const long long query_group_stride =
          static_cast<long long>(kRepeats) * query_count * HeadSize;
      const long long tile_matrix_group_stride =
          static_cast<long long>(kRepeats) * query_count *
          kTensorAttentionTileTokens;
      const long long numerator_group_stride =
          static_cast<long long>(kRepeats) * query_count *
          kTensorAttentionMaximumHeadSize;
      const int tile_columns = static_cast<int>(staged_count);
      const float alpha = 1.0F;
      const float score_beta = 0.0F;
      if (fp8) {
        if (direct_cache) {
          const CompactGlobalPagedCache contiguous{const_cast<BFloat16*>(key_cache),
              nullptr, cache_capacity, 0, 0, 0, format};
          fp8->qk_compact(current_key_head_major, current_value_token_major,
              paged_cache ? *paged_cache : contiguous, global_k_norm_scale,
              base_position, token_count, tile_start, tile_count, scores, stream);
        } else fp8->qk(staged_key, staged_value, tile_count, scores, stream);
      } else check_cublas(
          cublasGemmStridedBatchedEx(
              handle, CUBLAS_OP_T, CUBLAS_OP_N, tile_columns, grouped_rows,
              kInner, &alpha, staged_key, CUDA_R_16BF, kInner,
              kStagedHeadStride, query, CUDA_R_16BF, kInner,
              query_group_stride, &score_beta, scores, CUDA_R_32F,
              static_cast<int>(kTensorAttentionTileTokens),
              tile_matrix_group_stride, static_cast<int>(KvHeads),
              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "tensor attention QK cublasGemmStridedBatchedEx");

      const dim3 state_grid(gemma4_31b::kQueryHeadCount,
                            (query_count + kWarpsPerBlock - 1) / kWarpsPerBlock);
      if (fp8) {
        tensor_attention_softmax_update_kernel<HeadSize, Local, true, Image>
            <<<state_grid, kThreads, 0, stream>>>(
                scores, probabilities, numerator, row_maximum, row_denominator,
                query_base, query_count, tile_start, tile_count, first_tile,
                fp8->query_scales(), fp8->key_scales(), image_begin, image_end, fp8->value_token_scales(),
                fp8->value_output_scales());
      } else tensor_attention_softmax_update_kernel<HeadSize, Local, false, Image>
          <<<state_grid, kThreads, 0, stream>>>(
              scores, probabilities, numerator, row_maximum, row_denominator,
              query_base, query_count, tile_start, tile_count, first_tile,
              nullptr, nullptr, image_begin, image_end);
      check_cuda(cudaGetLastError(),
                 "tensor attention softmax update kernel launch");

      const float numerator_beta = first_tile ? 0.0F : 1.0F;
      if (fp8) {
        fp8->pv(probabilities, numerator, first_tile, stream);
      } else check_cublas(
          cublasGemmStridedBatchedEx(
              handle, CUBLAS_OP_N, CUBLAS_OP_N, kInner, grouped_rows,
              tile_columns, &alpha, staged_value, CUDA_R_16BF, kInner,
              kStagedHeadStride, probabilities, CUDA_R_16BF,
              static_cast<int>(kTensorAttentionTileTokens),
              tile_matrix_group_stride, &numerator_beta, numerator, CUDA_R_32F,
              kNumeratorLeadingDimension, numerator_group_stride,
              static_cast<int>(KvHeads), CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "tensor attention PV cublasGemmStridedBatchedEx");
      first_tile = false;
      tile_start += tile_count;
    }

    const std::size_t context_elements =
        static_cast<std::size_t>(query_count) *
        gemma4_31b::kQueryHeadCount * HeadSize;
    tensor_attention_finalize_kernel<HeadSize>
        <<<blocks_for(context_elements), kThreads, 0, stream>>>(
            numerator, row_denominator,
            context_token_major + std::size_t(query_start) *
                gemma4_31b::kQueryHeadCount * HeadSize,
            query_count, context_elements);
    check_cuda(cudaGetLastError(),
               "tensor attention finalize kernel launch");
  }
}

}  // namespace

void causal_gqa_attention_cached_chunk_tensor(
    cublasHandle_t handle, const BFloat16* query_head_major,
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major, const BFloat16* key_cache,
    const BFloat16* value_cache, std::uint32_t base_position,
    std::uint32_t token_count, std::uint32_t cache_capacity, void* scratch,
    BFloat16* context_token_major, gemma4_31b::AttentionKind kind,
    cudaStream_t stream,
    kv_cache::Format format, Fp8Attention* fp8,
    std::uint32_t image_begin, std::uint32_t image_end) {
  check_tensor_chunk_range(base_position, token_count,
                           "causal_gqa_attention_cached_chunk_tensor");
  if (handle == nullptr) {
    fail("causal_gqa_attention_cached_chunk_tensor",
         "null cuBLAS handle");
  }
  check_pointer(query_head_major,
                "tensor cached chunk attention query");
  check_pointer(current_key_head_major,
                "tensor cached chunk attention current key");
  check_pointer(current_value_token_major,
                "tensor cached chunk attention current value");
  check_pointer(key_cache, "tensor cached chunk attention key cache");
  check_pointer(value_cache,
                "tensor cached chunk attention value cache");
  check_pointer(scratch, "tensor cached chunk attention scratch");
  check_pointer(context_token_major,
                "tensor cached chunk attention context");
  check_kind(kind, "causal_gqa_attention_cached_chunk_tensor");
  check_chunk_cache(base_position, token_count, cache_capacity, kind,
                    "causal_gqa_attention_cached_chunk_tensor");
  if (image_begin || image_end)
    check_image_block(base_position, token_count, image_begin, image_end,
                      "causal_gqa_attention_cached_chunk_tensor");

  if (kind == gemma4_31b::AttentionKind::global) {
    causal_gqa_attention_cached_chunk_tensor_impl<
        gemma4_31b::kGlobalHeadSize, gemma4_31b::kGlobalKvHeadCount, false,
        TensorCacheLayout::separate>(
        handle, query_head_major, current_key_head_major,
        current_value_token_major, key_cache, value_cache, nullptr, nullptr,
        base_position, token_count, cache_capacity, scratch,
        context_token_major, stream, format, fp8);
  } else if (image_end) {
    causal_gqa_attention_cached_chunk_tensor_impl<
        gemma4_31b::kLocalHeadSize, gemma4_31b::kLocalKvHeadCount, true,
        TensorCacheLayout::separate, true>(
        handle, query_head_major, current_key_head_major,
        current_value_token_major, key_cache, value_cache, nullptr, nullptr,
        base_position, token_count, cache_capacity, scratch,
        context_token_major, stream, format, fp8, image_begin, image_end);
  } else {
    // Compile out image predicates on text: they otherwise increase register
    // pressure, particularly in the FP8 softmax specialization.
    causal_gqa_attention_cached_chunk_tensor_impl<
        gemma4_31b::kLocalHeadSize, gemma4_31b::kLocalKvHeadCount, true,
        TensorCacheLayout::separate>(
        handle, query_head_major, current_key_head_major,
        current_value_token_major, key_cache, value_cache, nullptr, nullptr,
        base_position, token_count, cache_capacity, scratch,
        context_token_major, stream, format, fp8);
  }
}

void causal_gqa_attention_cached_chunk_tensor_global_compact(
    cublasHandle_t handle, const BFloat16* query_head_major,
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major,
    const BFloat16* compact_kv_cache,
    const BFloat16* global_k_norm_scale,
    std::uint32_t base_position, std::uint32_t token_count,
    std::uint32_t cache_capacity, void* scratch,
    BFloat16* context_token_major, cudaStream_t stream,
    kv_cache::Format format, Fp8Attention* fp8) {
  check_tensor_chunk_range(
      base_position, token_count,
      "causal_gqa_attention_cached_chunk_tensor_global_compact");
  if (handle == nullptr) {
    fail("causal_gqa_attention_cached_chunk_tensor_global_compact",
         "null cuBLAS handle");
  }
  check_pointer(query_head_major,
                "tensor compact-global attention query");
  check_pointer(current_key_head_major,
                "tensor compact-global attention current key");
  check_pointer(current_value_token_major,
                "tensor compact-global attention current value");
  check_pointer(compact_kv_cache,
                "tensor compact-global attention cache");
  check_pointer(global_k_norm_scale,
                "tensor compact-global attention K scale");
  check_pointer(scratch, "tensor compact-global attention scratch");
  check_pointer(context_token_major,
                "tensor compact-global attention context");
  check_chunk_cache(
      base_position, token_count, cache_capacity,
      gemma4_31b::AttentionKind::global,
      "causal_gqa_attention_cached_chunk_tensor_global_compact");

  causal_gqa_attention_cached_chunk_tensor_impl<
      gemma4_31b::kGlobalHeadSize, gemma4_31b::kGlobalKvHeadCount, false,
      TensorCacheLayout::compact_global>(
      handle, query_head_major, current_key_head_major,
      current_value_token_major, compact_kv_cache, nullptr,
      global_k_norm_scale, nullptr, base_position, token_count,
      cache_capacity, scratch, context_token_major, stream, format, fp8);
}

void causal_gqa_attention_cached_chunk_tensor_global_compact_paged(
    cublasHandle_t handle, const BFloat16* query_head_major,
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major,
    const CompactGlobalPagedCache& cache,
    const BFloat16* global_k_norm_scale, std::uint32_t base_position,
    std::uint32_t token_count, void* scratch, BFloat16* context_token_major,
    cudaStream_t stream, Fp8Attention* fp8) {
  constexpr std::string_view kOperation =
      "causal_gqa_attention_cached_chunk_tensor_global_compact_paged";
  check_tensor_chunk_range(base_position, token_count, kOperation);
  if (handle == nullptr) {
    fail(kOperation, "null cuBLAS handle");
  }
  check_pointer(query_head_major,
                "tensor paged compact-global attention query");
  check_pointer(current_key_head_major,
                "tensor paged compact-global attention current key");
  check_pointer(current_value_token_major,
                "tensor paged compact-global attention current value");
  check_pointer(global_k_norm_scale,
                "tensor paged compact-global attention K scale");
  check_pointer(scratch, "tensor paged compact-global attention scratch");
  check_pointer(context_token_major,
                "tensor paged compact-global attention context");
  check_paged_compact_global_cache(cache, base_position, token_count,
                                   kOperation);

  causal_gqa_attention_cached_chunk_tensor_impl<
      gemma4_31b::kGlobalHeadSize, gemma4_31b::kGlobalKvHeadCount, false,
      TensorCacheLayout::paged_compact_global>(
      handle, query_head_major, current_key_head_major,
      current_value_token_major, cache.page_pool, nullptr,
      global_k_norm_scale, &cache, base_position, token_count, 0, scratch,
      context_token_major, stream, cache.format, fp8);
}

}  // namespace gewell::prefill_primitives
