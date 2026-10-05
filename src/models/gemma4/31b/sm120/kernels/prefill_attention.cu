#include "prefill_common.cuh"

namespace gewell::prefill_primitives {
namespace {

using namespace detail;

// Short cached suffixes used to serialize both QK and PV in one warp. Compute
// scores across eight warps, then retain the scalar path's exact online-softmax
// and per-dimension accumulation order. In particular, probabilities stay FP32.
template <unsigned HeadSize, unsigned KvHeads, bool Local,
          bool Compact = false, bool Paged = false>
__global__ void short_cached_attention_kernel(
    const BFloat16* query, const BFloat16* current_key,
    const BFloat16* current_value, const BFloat16* key_cache,
    const BFloat16* value_cache, const std::uint64_t* page_offsets,
    const BFloat16* norm, unsigned page_tokens, std::size_t layer_offset,
    unsigned base, unsigned rows, unsigned capacity, BFloat16* output,
    kv_cache::Format format) {
  constexpr unsigned kMaximumKeys = gemma4_31b::kLocalWindowSize + 32;
  __shared__ float scores[kMaximumKeys];
  __shared__ float old_weights[kMaximumKeys];
  __shared__ float new_weights[kMaximumKeys];
  __shared__ float denominator;
  const unsigned head = blockIdx.x, row = blockIdx.y;
  const unsigned kv_head = head / (gemma4_31b::kQueryHeadCount / KvHeads);
  const unsigned lane = threadIdx.x % kWarpSize;
  const unsigned warp = threadIdx.x / kWarpSize;
  const unsigned last = base + row;
  const unsigned first = Local && last >= gemma4_31b::kLocalWindowSize - 1
      ? last - (gemma4_31b::kLocalWindowSize - 1) : 0;
  const unsigned count = last - first + 1;
  const auto load = [&](unsigned position, unsigned d, bool key) {
    const bool current = position >= base;
    const auto current_k = (std::size_t(kv_head) * rows + position - base) * HeadSize + d;
    const auto current_v = (std::size_t(position - base) * KvHeads + kv_head) * HeadSize + d;
    if constexpr (Compact) {
      const auto* cached = current ? nullptr : compact_global_cache_row<Paged>(
          key_cache, page_offsets, page_tokens, layer_offset, kv_head, position, capacity, format);
      if (key && is_compact_global_key_dimension(d))
        return current ? current_key[current_k]
            : kv_storage::load(cached, compact_global_key_index(d), 640, format, 128);
      const auto v = current ? current_value[current_v]
          : kv_storage::load(cached, 128 + d, 640, format, 128);
      return key ? compact_global_reconstructed_key(v, norm[d]) : v;
    } else {
      if (current) return key ? current_key[current_k] : current_value[current_v];
      const unsigned slot = Local ? position % gemma4_31b::kLocalWindowSize : position;
      const auto* cached = kv_storage::row(key ? key_cache : value_cache,
          std::size_t(kv_head) * capacity + slot, HeadSize, format);
      return kv_storage::load(cached, d, HeadSize, format);
    }
  };
  float q[HeadSize / kWarpSize];
#pragma unroll
  for (unsigned i = 0; i < HeadSize / kWarpSize; ++i)
    q[i] = __bfloat162float(query[(std::size_t(head) * rows + row) * HeadSize + lane + i * kWarpSize]);
  for (unsigned key = warp; key < count; key += kWarpsPerBlock) {
    float score = 0.0F;
#pragma unroll
    for (unsigned i = 0; i < HeadSize / kWarpSize; ++i)
      score = fmaf(q[i], __bfloat162float(load(first + key, lane + i * kWarpSize, true)), score);
#pragma unroll
    for (unsigned offset = kWarpSize / 2; offset; offset /= 2)
      score += __shfl_down_sync(0xffffffffU, score, offset);
    if (!lane) scores[key] = score;
  }
  __syncthreads();
  if (!threadIdx.x) {
    float maximum = -CUDART_INF_F, sum = 0.0F;
    for (unsigned key = 0; key < count; ++key) {
      float old_weight, new_weight;
      if (scores[key] <= maximum) {
        old_weight = 1.0F;
        new_weight = expf(scores[key] - maximum);
        sum += new_weight;
      } else {
        old_weight = expf(maximum - scores[key]);
        new_weight = 1.0F;
        sum = fmaf(sum, old_weight, new_weight);
        maximum = scores[key];
      }
      old_weights[key] = old_weight;
      new_weights[key] = new_weight;
    }
    denominator = sum;
  }
  __syncthreads();
  for (unsigned d = threadIdx.x; d < HeadSize; d += kThreads) {
    float numerator = 0.0F;
    for (unsigned key = 0; key < count; ++key)
      numerator = fmaf(new_weights[key], __bfloat162float(load(first + key, d, false)),
                       numerator * old_weights[key]);
    output[(std::size_t(row) * gemma4_31b::kQueryHeadCount + head) * HeadSize + d] =
        __float2bfloat16_rn(numerator / denominator);
  }
}

bool short_cached_chunk(unsigned base, unsigned rows) {
  return base >= 16 && base < 1024 && rows < 32;
}

template <std::uint32_t HeadSize, std::uint32_t KvHeads, bool Local>
__global__ void causal_gqa_attention_cached_chunk_kernel(
    const BFloat16* query_head_major,
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major, const BFloat16* key_cache,
    const BFloat16* value_cache, std::uint32_t base_position,
    std::uint32_t token_count, std::uint32_t cache_capacity,
    std::uint32_t image_begin, std::uint32_t image_end,
    BFloat16* context_token_major, kv_cache::Format format) {
  static_assert(gemma4_31b::kQueryHeadCount % KvHeads == 0);
  static_assert(HeadSize % kWarpSize == 0);
  constexpr unsigned kRepeats = gemma4_31b::kQueryHeadCount / KvHeads;
  constexpr unsigned kValuesPerLane = HeadSize / kWarpSize;
  static_assert(kRepeats == 2 || kRepeats == 8);

  const std::uint32_t kv_head = blockIdx.x;
  const std::uint32_t chunk_query_position = blockIdx.y;
  const std::uint32_t absolute_query_position =
      base_position + chunk_query_position;
  const unsigned lane = threadIdx.x % kWarpSize;
  const unsigned warp = threadIdx.x / kWarpSize;
  const std::uint32_t query_head = kv_head * kRepeats + warp;
  const std::size_t query_row =
      (static_cast<std::size_t>(query_head) * token_count +
       chunk_query_position) *
      HeadSize;

  float query_values[kValuesPerLane];
  float context_numerator[kValuesPerLane];
#pragma unroll
  for (unsigned value_index = 0; value_index < kValuesPerLane;
       ++value_index) {
    const unsigned dimension = lane + value_index * kWarpSize;
    query_values[value_index] =
        __bfloat162float(query_head_major[query_row + dimension]);
    context_numerator[value_index] = 0.0F;
  }
  float running_maximum = -__int_as_float(0x7f800000);
  float denominator = 0.0F;

  const std::uint32_t first_key_position =
      Local && absolute_query_position >= gemma4_31b::kLocalWindowSize - 1
          ? absolute_query_position - (gemma4_31b::kLocalWindowSize - 1)
          : 0;
  const bool image_query = Local && absolute_query_position >= image_begin &&
                           absolute_query_position < image_end;
  // Gemma 4 uses causal global attention. Local attention adds same-image
  // lookahead, but its past-window cutoff still applies inside long images.
  const std::uint32_t last_key_position =
      image_query ? image_end - 1 : absolute_query_position;
  std::uint32_t absolute_key_position = first_key_position;
  while (true) {
    const bool current = absolute_key_position >= base_position;
    const std::uint32_t chunk_key_position =
        current ? absolute_key_position - base_position : 0;
    const std::uint32_t cache_key_position =
        Local ? absolute_key_position % gemma4_31b::kLocalWindowSize
              : absolute_key_position;
    const BFloat16* key =
        current
            ? current_key_head_major +
                  (static_cast<std::size_t>(kv_head) * token_count +
                   chunk_key_position) *
                      HeadSize
            : kv_storage::row(key_cache,
                  std::size_t(kv_head) * cache_capacity + cache_key_position,
                  HeadSize, format);
    const BFloat16* value =
        current
            ? current_value_token_major +
                  (static_cast<std::size_t>(chunk_key_position) * KvHeads +
                   kv_head) *
                      HeadSize
            : kv_storage::row(value_cache,
                  std::size_t(kv_head) * cache_capacity + cache_key_position,
                  HeadSize, format);

    float score = 0.0F;
#pragma unroll
    for (unsigned value_index = 0; value_index < kValuesPerLane;
         ++value_index) {
      const unsigned dimension = lane + value_index * kWarpSize;
      score = fmaf(query_values[value_index],
                   __bfloat162float(kv_storage::load(key, dimension, HeadSize, current ? kv_cache::Format::bf16 : format)), score);
    }
#pragma unroll
    for (unsigned offset = kWarpSize / 2; offset != 0; offset /= 2) {
      score += __shfl_down_sync(0xffffffffU, score, offset);
    }
    const float complete_score = __shfl_sync(0xffffffffU, score, 0);

    float old_weight = 0.0F;
    float new_weight = 0.0F;
    if (lane == 0) {
      if (complete_score <= running_maximum) {
        old_weight = 1.0F;
        new_weight = expf(complete_score - running_maximum);
        denominator += new_weight;
      } else {
        old_weight = expf(running_maximum - complete_score);
        new_weight = 1.0F;
        denominator = fmaf(denominator, old_weight, new_weight);
        running_maximum = complete_score;
      }
    }
    old_weight = __shfl_sync(0xffffffffU, old_weight, 0);
    new_weight = __shfl_sync(0xffffffffU, new_weight, 0);
#pragma unroll
    for (unsigned value_index = 0; value_index < kValuesPerLane;
         ++value_index) {
      const unsigned dimension = lane + value_index * kWarpSize;
      context_numerator[value_index] = fmaf(
          new_weight, __bfloat162float(kv_storage::load(value, dimension, HeadSize, current ? kv_cache::Format::bf16 : format)),
          context_numerator[value_index] * old_weight);
    }

    if (absolute_key_position == last_key_position) {
      break;
    }
    ++absolute_key_position;
  }

  const std::size_t context_row =
      (static_cast<std::size_t>(chunk_query_position) *
           gemma4_31b::kQueryHeadCount +
       query_head) *
      HeadSize;
  const float complete_denominator =
      __shfl_sync(0xffffffffU, denominator, 0);
#pragma unroll
  for (unsigned value_index = 0; value_index < kValuesPerLane;
       ++value_index) {
    const unsigned dimension = lane + value_index * kWarpSize;
    context_token_major[context_row + dimension] =
        __float2bfloat16_rn(context_numerator[value_index] /
                           complete_denominator);
  }
}

template <bool Paged>
__global__ void causal_gqa_attention_cached_chunk_global_compact_kernel(
    const BFloat16* query_head_major,
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major,
    const BFloat16* compact_kv_cache, const std::uint64_t* page_offsets,
    const BFloat16* global_k_norm_scale,
    std::uint32_t page_tokens, std::size_t layer_offset_elements,
    std::uint32_t base_position, std::uint32_t token_count,
    std::uint32_t cache_capacity, BFloat16* context_token_major,
    kv_cache::Format format) {
  constexpr std::uint32_t kHeadSize = gemma4_31b::kGlobalHeadSize;
  constexpr std::uint32_t kKvHeads = gemma4_31b::kGlobalKvHeadCount;
  constexpr unsigned kRepeats =
      gemma4_31b::kQueryHeadCount / kKvHeads;
  constexpr unsigned kValuesPerLane = kHeadSize / kWarpSize;
  static_assert(kRepeats == 8);

  const std::uint32_t kv_head = blockIdx.x;
  const std::uint32_t chunk_query_position = blockIdx.y;
  const std::uint32_t absolute_query_position =
      base_position + chunk_query_position;
  const unsigned lane = threadIdx.x % kWarpSize;
  const unsigned warp = threadIdx.x / kWarpSize;
  const std::uint32_t query_head = kv_head * kRepeats + warp;
  const std::size_t query_row =
      (static_cast<std::size_t>(query_head) * token_count +
       chunk_query_position) *
      kHeadSize;

  float query_values[kValuesPerLane];
  float context_numerator[kValuesPerLane];
#pragma unroll
  for (unsigned value_index = 0; value_index < kValuesPerLane;
       ++value_index) {
    const unsigned dimension = lane + value_index * kWarpSize;
    query_values[value_index] =
        __bfloat162float(query_head_major[query_row + dimension]);
    context_numerator[value_index] = 0.0F;
  }
  float running_maximum = -__int_as_float(0x7f800000);
  float denominator = 0.0F;

  const std::uint32_t last_key_position = absolute_query_position;
  std::uint32_t absolute_key_position = 0;
  while (true) {
    const bool current = absolute_key_position >= base_position;
    const std::uint32_t chunk_key_position =
        current ? absolute_key_position - base_position : 0;
    const BFloat16* current_key =
        current
            ? current_key_head_major +
                  (static_cast<std::size_t>(kv_head) * token_count +
                   chunk_key_position) *
                      kHeadSize
            : nullptr;
    const BFloat16* compact_row =
        current
            ? nullptr
            : compact_global_cache_row<Paged>(
                  compact_kv_cache, page_offsets, page_tokens,
                  layer_offset_elements, kv_head, absolute_key_position,
                  cache_capacity, format);
    const BFloat16* value =
        current
            ? current_value_token_major +
                  (static_cast<std::size_t>(chunk_key_position) * kKvHeads +
                   kv_head) *
                      kHeadSize
            : compact_row + kGlobalCompactRotatedKeyElements;

    float score = 0.0F;
#pragma unroll
    for (unsigned value_index = 0; value_index < kValuesPerLane;
         ++value_index) {
      const unsigned dimension = lane + value_index * kWarpSize;
      const BFloat16 key_value =
          is_compact_global_key_dimension(dimension)
              ? (current
                     ? current_key[dimension]
                     : kv_storage::load(compact_row, compact_global_key_index(dimension), 640, format, 128))
              : compact_global_reconstructed_key(
                    (current ? value[dimension] : kv_storage::load(compact_row, 128 + dimension, 640, format, 128)), global_k_norm_scale[dimension]);
      score = fmaf(query_values[value_index],
                   __bfloat162float(key_value), score);
    }
#pragma unroll
    for (unsigned offset = kWarpSize / 2; offset != 0; offset /= 2) {
      score += __shfl_down_sync(0xffffffffU, score, offset);
    }
    const float complete_score = __shfl_sync(0xffffffffU, score, 0);

    float old_weight = 0.0F;
    float new_weight = 0.0F;
    if (lane == 0) {
      if (complete_score <= running_maximum) {
        old_weight = 1.0F;
        new_weight = expf(complete_score - running_maximum);
        denominator += new_weight;
      } else {
        old_weight = expf(running_maximum - complete_score);
        new_weight = 1.0F;
        denominator = fmaf(denominator, old_weight, new_weight);
        running_maximum = complete_score;
      }
    }
    old_weight = __shfl_sync(0xffffffffU, old_weight, 0);
    new_weight = __shfl_sync(0xffffffffU, new_weight, 0);
#pragma unroll
    for (unsigned value_index = 0; value_index < kValuesPerLane;
         ++value_index) {
      const unsigned dimension = lane + value_index * kWarpSize;
      context_numerator[value_index] = fmaf(
          new_weight, __bfloat162float((current ? value[dimension] : kv_storage::load(compact_row, 128 + dimension, 640, format, 128))),
          context_numerator[value_index] * old_weight);
    }

    if (absolute_key_position == last_key_position) {
      break;
    }
    ++absolute_key_position;
  }

  const std::size_t context_row =
      (static_cast<std::size_t>(chunk_query_position) *
           gemma4_31b::kQueryHeadCount +
       query_head) *
      kHeadSize;
  const float complete_denominator =
      __shfl_sync(0xffffffffU, denominator, 0);
#pragma unroll
  for (unsigned value_index = 0; value_index < kValuesPerLane;
       ++value_index) {
    const unsigned dimension = lane + value_index * kWarpSize;
    context_token_major[context_row + dimension] =
        __float2bfloat16_rn(context_numerator[value_index] /
                           complete_denominator);
  }
}

template <std::uint32_t HeadSize>
__device__ float cached_m1_dot(const BFloat16* query,
                              const BFloat16* key,
                              unsigned lane) {
  float partial[8];
#pragma unroll
  for (unsigned group = 0; group < 8; ++group) {
    const unsigned original_thread = lane + group * kWarpSize;
    float sum = 0.0F;
#pragma unroll
    for (unsigned dimension = original_thread; dimension < HeadSize;
         dimension += kThreads) {
      sum = fmaf(__bfloat162float(query[dimension]),
                 __bfloat162float(key[dimension]), sum);
    }
    partial[group] = sum;
  }
#pragma unroll
  for (unsigned group = 0; group < 4; ++group) {
    partial[group] += partial[group + 4];
  }
#pragma unroll
  for (unsigned group = 0; group < 2; ++group) {
    partial[group] += partial[group + 2];
  }
  float sum = partial[0] + partial[1];
#pragma unroll
  for (unsigned offset = kWarpSize / 2; offset != 0; offset /= 2) {
    const float other = __shfl_down_sync(0xffffffffU, sum, offset);
    if (lane < offset) {
      sum += other;
    }
  }
  return sum;
}

template <std::uint32_t HeadSize, std::uint32_t KvHeads>
__global__ void attention_scores_kernel(const BFloat16* query_head_major,
                                        const BFloat16* key_head_major,
                                        BFloat16* score_scratch) {
  static_assert(gemma4_31b::kQueryHeadCount % KvHeads == 0);
  const std::uint32_t query_head = blockIdx.x;
  const std::uint32_t query_position = blockIdx.y;
  const unsigned warp = threadIdx.x / kWarpSize;
  const unsigned lane = threadIdx.x % kWarpSize;
  const std::uint32_t repeats = gemma4_31b::kQueryHeadCount / KvHeads;
  const std::uint32_t kv_head = query_head / repeats;
  const BFloat16* query =
      query_head_major +
      (static_cast<std::size_t>(query_head) * kTokenCount + query_position) *
          HeadSize;
  const std::size_t score_row =
      (static_cast<std::size_t>(query_head) * kTokenCount + query_position) *
      kTokenCount;

  for (std::uint32_t key_position = warp;
       key_position <= query_position; key_position += kWarpsPerBlock) {
    const BFloat16* key =
        key_head_major +
        (static_cast<std::size_t>(kv_head) * kTokenCount + key_position) *
            HeadSize;
    const float score = cached_m1_dot<HeadSize>(query, key, lane);
    if (lane == 0) {
      score_scratch[score_row + key_position] =
          __float2bfloat16_rn(score);
    }
  }
}

__global__ void attention_softmax_kernel(const BFloat16* score_scratch,
                                         BFloat16* probabilities) {
  const std::uint32_t query_head = blockIdx.x;
  const std::uint32_t query_position = blockIdx.y;
  const std::size_t row =
      (static_cast<std::size_t>(query_head) * kTokenCount + query_position) *
      kTokenCount;

  if (threadIdx.x == 0) {
    float maximum = -__int_as_float(0x7f800000);
    for (std::uint32_t key_position = 0; key_position <= query_position;
         ++key_position) {
      maximum = fmaxf(
          maximum, __bfloat162float(score_scratch[row + key_position]));
    }

    float denominator = 0.0F;
    for (std::uint32_t key_position = 0; key_position <= query_position;
         ++key_position) {
      denominator += expf(
          __bfloat162float(score_scratch[row + key_position]) - maximum);
    }
    for (std::uint32_t key_position = 0; key_position <= query_position;
         ++key_position) {
      const float exponential = expf(
          __bfloat162float(score_scratch[row + key_position]) - maximum);
      probabilities[row + key_position] =
          __float2bfloat16_rn(exponential / denominator);
    }
  }

  for (std::uint32_t key_position = query_position + 1 + threadIdx.x;
       key_position < kTokenCount; key_position += blockDim.x) {
    probabilities[row + key_position] = __float2bfloat16_rn(0.0F);
  }
}

template <std::uint32_t HeadSize, std::uint32_t KvHeads>
__global__ void attention_context_kernel(
    const BFloat16* value_token_major, const BFloat16* probabilities,
    BFloat16* context_token_major) {
  static_assert(gemma4_31b::kQueryHeadCount % KvHeads == 0);
  const std::uint32_t query_head = blockIdx.x;
  const std::uint32_t query_position = blockIdx.y;
  const std::uint32_t repeats = gemma4_31b::kQueryHeadCount / KvHeads;
  const std::uint32_t kv_head = query_head / repeats;
  const std::size_t probability_row =
      (static_cast<std::size_t>(query_head) * kTokenCount + query_position) *
      kTokenCount;
  const std::size_t context_row =
      (static_cast<std::size_t>(query_position) *
           gemma4_31b::kQueryHeadCount +
       query_head) *
      HeadSize;

  for (std::uint32_t dimension = threadIdx.x; dimension < HeadSize;
       dimension += blockDim.x) {
    const std::size_t value_zero =
        static_cast<std::size_t>(kv_head) * HeadSize + dimension;
    float sum = __bfloat162float(probabilities[probability_row]) *
                __bfloat162float(value_token_major[value_zero]);
    for (std::uint32_t key_position = 1; key_position <= query_position;
         ++key_position) {
      const std::size_t value_index =
          (static_cast<std::size_t>(key_position) * KvHeads + kv_head) *
              HeadSize +
          dimension;
      sum = fmaf(
          __bfloat162float(probabilities[probability_row + key_position]),
          __bfloat162float(value_token_major[value_index]), sum);
    }
    context_token_major[context_row + dimension] =
        __float2bfloat16_rn(sum);
  }
}

}  // namespace

void causal_gqa_attention_m1024(
    const BFloat16* query_head_major, const BFloat16* key_head_major,
    const BFloat16* value_token_major, BFloat16* score_scratch,
    BFloat16* probabilities, BFloat16* context_token_major,
    gemma4_31b::AttentionKind kind, cudaStream_t stream) {
  check_pointer(query_head_major,
                "causal_gqa_attention_m1024 query head major");
  check_pointer(key_head_major,
                "causal_gqa_attention_m1024 key head major");
  check_pointer(value_token_major,
                "causal_gqa_attention_m1024 value token major");
  check_pointer(score_scratch, "causal_gqa_attention_m1024 score scratch");
  check_pointer(probabilities, "causal_gqa_attention_m1024 probabilities");
  check_pointer(context_token_major,
                "causal_gqa_attention_m1024 context token major");
  check_kind(kind, "causal_gqa_attention_m1024");

  const dim3 grid(gemma4_31b::kQueryHeadCount, kTokenCount);
  if (kind == gemma4_31b::AttentionKind::global) {
    attention_scores_kernel<gemma4_31b::kGlobalHeadSize,
                            gemma4_31b::kGlobalKvHeadCount>
        <<<grid, kThreads, 0, stream>>>(query_head_major, key_head_major,
                                        score_scratch);
  } else {
    attention_scores_kernel<gemma4_31b::kLocalHeadSize,
                            gemma4_31b::kLocalKvHeadCount>
        <<<grid, kThreads, 0, stream>>>(query_head_major, key_head_major,
                                        score_scratch);
  }
  check_cuda(cudaGetLastError(), "M=1024 attention score kernel launch");

  attention_softmax_kernel<<<grid, kThreads, 0, stream>>>(score_scratch,
                                                          probabilities);
  check_cuda(cudaGetLastError(), "M=1024 attention softmax kernel launch");

  if (kind == gemma4_31b::AttentionKind::global) {
    attention_context_kernel<gemma4_31b::kGlobalHeadSize,
                             gemma4_31b::kGlobalKvHeadCount>
        <<<grid, kThreads, 0, stream>>>(value_token_major, probabilities,
                                        context_token_major);
  } else {
    attention_context_kernel<gemma4_31b::kLocalHeadSize,
                             gemma4_31b::kLocalKvHeadCount>
        <<<grid, kThreads, 0, stream>>>(value_token_major, probabilities,
                                        context_token_major);
  }
  check_cuda(cudaGetLastError(), "M=1024 attention context kernel launch");
}

void causal_gqa_attention_cached_chunk(
    const BFloat16* query_head_major,
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major, const BFloat16* key_cache,
    const BFloat16* value_cache, std::uint32_t base_position,
    std::uint32_t token_count, std::uint32_t cache_capacity,
    BFloat16* context_token_major, gemma4_31b::AttentionKind kind,
    cudaStream_t stream,
    kv_cache::Format format) {
  check_pointer(query_head_major,
                "causal_gqa_attention_cached_chunk query");
  check_pointer(current_key_head_major,
                "causal_gqa_attention_cached_chunk current key");
  check_pointer(current_value_token_major,
                "causal_gqa_attention_cached_chunk current value");
  check_pointer(key_cache, "causal_gqa_attention_cached_chunk key cache");
  check_pointer(value_cache,
                "causal_gqa_attention_cached_chunk value cache");
  check_pointer(context_token_major,
                "causal_gqa_attention_cached_chunk context");
  check_kind(kind, "causal_gqa_attention_cached_chunk");
  check_chunk_cache(base_position, token_count, cache_capacity, kind,
                    "causal_gqa_attention_cached_chunk");

  if (short_cached_chunk(base_position, token_count)) {
    const dim3 grid(gemma4_31b::kQueryHeadCount, token_count);
    if (kind == gemma4_31b::AttentionKind::global)
      short_cached_attention_kernel<512, 4, false><<<grid, kThreads, 0, stream>>>(
          query_head_major, current_key_head_major, current_value_token_major,
          key_cache, value_cache, nullptr, nullptr, 0, 0, base_position,
          token_count, cache_capacity, context_token_major, format);
    else
      short_cached_attention_kernel<256, 16, true><<<grid, kThreads, 0, stream>>>(
          query_head_major, current_key_head_major, current_value_token_major,
          key_cache, value_cache, nullptr, nullptr, 0, 0, base_position,
          token_count, cache_capacity, context_token_major, format);
    check_cuda(cudaGetLastError(), "short cached attention kernel launch");
    return;
  }

  if (kind == gemma4_31b::AttentionKind::global) {
    constexpr unsigned kGlobalRepeats =
        gemma4_31b::kQueryHeadCount / gemma4_31b::kGlobalKvHeadCount;
    const dim3 grid(gemma4_31b::kGlobalKvHeadCount, token_count);
    causal_gqa_attention_cached_chunk_kernel<
        gemma4_31b::kGlobalHeadSize, gemma4_31b::kGlobalKvHeadCount, false>
        <<<grid, kGlobalRepeats * kWarpSize, 0, stream>>>(
            query_head_major, current_key_head_major,
            current_value_token_major, key_cache, value_cache, base_position,
            token_count, cache_capacity, 0, 0, context_token_major, format);
  } else {
    constexpr unsigned kLocalRepeats =
        gemma4_31b::kQueryHeadCount / gemma4_31b::kLocalKvHeadCount;
    const dim3 grid(gemma4_31b::kLocalKvHeadCount, token_count);
    causal_gqa_attention_cached_chunk_kernel<
        gemma4_31b::kLocalHeadSize, gemma4_31b::kLocalKvHeadCount, true>
        <<<grid, kLocalRepeats * kWarpSize, 0, stream>>>(
            query_head_major, current_key_head_major,
            current_value_token_major, key_cache, value_cache, base_position,
            token_count, cache_capacity, 0, 0, context_token_major, format);
  }
  check_cuda(cudaGetLastError(), "cached chunk attention kernel launch");
}

void image_block_gqa_attention_cached_chunk(
    const BFloat16* query_head_major,
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major, const BFloat16* key_cache,
    const BFloat16* value_cache, std::uint32_t base_position,
    std::uint32_t token_count, std::uint32_t cache_capacity,
    std::uint32_t image_begin, std::uint32_t image_end,
    BFloat16* context_token_major, gemma4_31b::AttentionKind kind,
    cudaStream_t stream,
    kv_cache::Format format) {
  check_pointer(query_head_major, "image-block chunk attention query");
  check_pointer(current_key_head_major,
                "image-block chunk attention current key");
  check_pointer(current_value_token_major,
                "image-block chunk attention current value");
  check_pointer(key_cache, "image-block chunk attention key cache");
  check_pointer(value_cache, "image-block chunk attention value cache");
  check_pointer(context_token_major, "image-block chunk attention context");
  check_kind(kind, "image_block_gqa_attention_cached_chunk");
  check_chunk_cache(base_position, token_count, cache_capacity, kind,
                    "image_block_gqa_attention_cached_chunk");
  check_image_block(base_position, token_count, image_begin, image_end,
                    "image_block_gqa_attention_cached_chunk");

  if (kind == gemma4_31b::AttentionKind::global) {
    constexpr unsigned kRepeats = gemma4_31b::kQueryHeadCount /
                                  gemma4_31b::kGlobalKvHeadCount;
    const dim3 grid(gemma4_31b::kGlobalKvHeadCount, token_count);
    causal_gqa_attention_cached_chunk_kernel<
        gemma4_31b::kGlobalHeadSize, gemma4_31b::kGlobalKvHeadCount, false>
        <<<grid, kRepeats * kWarpSize, 0, stream>>>(
            query_head_major, current_key_head_major,
            current_value_token_major, key_cache, value_cache, base_position,
            token_count, cache_capacity, image_begin, image_end,
            context_token_major, format);
  } else {
    constexpr unsigned kRepeats = gemma4_31b::kQueryHeadCount /
                                  gemma4_31b::kLocalKvHeadCount;
    const dim3 grid(gemma4_31b::kLocalKvHeadCount, token_count);
    causal_gqa_attention_cached_chunk_kernel<
        gemma4_31b::kLocalHeadSize, gemma4_31b::kLocalKvHeadCount, true>
        <<<grid, kRepeats * kWarpSize, 0, stream>>>(
            query_head_major, current_key_head_major,
            current_value_token_major, key_cache, value_cache, base_position,
            token_count, cache_capacity, image_begin, image_end,
            context_token_major, format);
  }
  check_cuda(cudaGetLastError(),
             "image-block cached chunk attention kernel launch");
}

void causal_gqa_attention_cached_chunk_global_compact(
    const BFloat16* query_head_major,
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major,
    const BFloat16* compact_kv_cache,
    const BFloat16* global_k_norm_scale,
    std::uint32_t base_position, std::uint32_t token_count,
    std::uint32_t cache_capacity, BFloat16* context_token_major,
    cudaStream_t stream,
    kv_cache::Format format) {
  check_pointer(query_head_major,
                "compact-global cached chunk attention query");
  check_pointer(current_key_head_major,
                "compact-global cached chunk attention current key");
  check_pointer(current_value_token_major,
                "compact-global cached chunk attention current value");
  check_pointer(compact_kv_cache,
                "compact-global cached chunk attention cache");
  check_pointer(global_k_norm_scale,
                "compact-global cached chunk attention K scale");
  check_pointer(context_token_major,
                "compact-global cached chunk attention context");
  check_chunk_cache(base_position, token_count, cache_capacity,
                    gemma4_31b::AttentionKind::global,
                    "causal_gqa_attention_cached_chunk_global_compact");

  if (short_cached_chunk(base_position, token_count)) {
    const dim3 grid(gemma4_31b::kQueryHeadCount, token_count);
    short_cached_attention_kernel<512, 4, false, true><<<grid, kThreads, 0, stream>>>(
        query_head_major, current_key_head_major, current_value_token_major,
        compact_kv_cache, nullptr, nullptr, global_k_norm_scale, 0, 0,
        base_position, token_count, cache_capacity, context_token_major, format);
    check_cuda(cudaGetLastError(), "short compact-global attention kernel launch");
    return;
  }

  constexpr unsigned kGlobalRepeats =
      gemma4_31b::kQueryHeadCount /
      gemma4_31b::kGlobalKvHeadCount;
  const dim3 grid(gemma4_31b::kGlobalKvHeadCount, token_count);
  causal_gqa_attention_cached_chunk_global_compact_kernel<false>
      <<<grid, kGlobalRepeats * kWarpSize, 0, stream>>>(
          query_head_major, current_key_head_major,
          current_value_token_major, compact_kv_cache, nullptr,
          global_k_norm_scale, 0, 0, base_position, token_count,
          cache_capacity, context_token_major, format);
  check_cuda(cudaGetLastError(),
             "compact-global cached chunk attention kernel launch");
}

void image_block_gqa_attention_cached_chunk_global_compact(
    const BFloat16* query_head_major,
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major,
    const BFloat16* compact_kv_cache,
    const BFloat16* global_k_norm_scale, std::uint32_t base_position,
    std::uint32_t token_count, std::uint32_t cache_capacity,
    std::uint32_t image_begin, std::uint32_t image_end,
    BFloat16* context_token_major, cudaStream_t stream,
    kv_cache::Format format) {
  check_pointer(query_head_major,
                "image-block compact-global attention query");
  check_pointer(current_key_head_major,
                "image-block compact-global attention current key");
  check_pointer(current_value_token_major,
                "image-block compact-global attention current value");
  check_pointer(compact_kv_cache,
                "image-block compact-global attention cache");
  check_pointer(global_k_norm_scale,
                "image-block compact-global attention K scale");
  check_pointer(context_token_major,
                "image-block compact-global attention context");
  check_chunk_cache(base_position, token_count, cache_capacity,
                    gemma4_31b::AttentionKind::global,
                    "image_block_gqa_attention_cached_chunk_global_compact");
  check_image_block(base_position, token_count, image_begin, image_end,
                    "image_block_gqa_attention_cached_chunk_global_compact");

  constexpr unsigned kRepeats = gemma4_31b::kQueryHeadCount /
                                gemma4_31b::kGlobalKvHeadCount;
  const dim3 grid(gemma4_31b::kGlobalKvHeadCount, token_count);
  causal_gqa_attention_cached_chunk_global_compact_kernel<false>
      <<<grid, kRepeats * kWarpSize, 0, stream>>>(
          query_head_major, current_key_head_major,
          current_value_token_major, compact_kv_cache, nullptr,
          global_k_norm_scale, 0, 0, base_position, token_count,
          cache_capacity, context_token_major, format);
  check_cuda(cudaGetLastError(),
             "image-block compact-global attention kernel launch");
}

void causal_gqa_attention_cached_chunk_global_compact_paged(
    const BFloat16* query_head_major,
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major,
    const CompactGlobalPagedCache& cache,
    const BFloat16* global_k_norm_scale, std::uint32_t base_position,
    std::uint32_t token_count, BFloat16* context_token_major,
    cudaStream_t stream) {
  check_pointer(query_head_major,
                "paged compact-global cached chunk attention query");
  check_pointer(current_key_head_major,
                "paged compact-global cached chunk attention current key");
  check_pointer(
      current_value_token_major,
      "paged compact-global cached chunk attention current value");
  check_pointer(context_token_major,
                "paged compact-global cached chunk attention context");
  check_pointer(global_k_norm_scale,
                "paged compact-global cached chunk attention K scale");
  check_paged_compact_global_cache(
      cache, base_position, token_count,
      "causal_gqa_attention_cached_chunk_global_compact_paged", true);

  if (short_cached_chunk(base_position, token_count)) {
    const dim3 grid(gemma4_31b::kQueryHeadCount, token_count);
    short_cached_attention_kernel<512, 4, false, true, true><<<grid, kThreads, 0, stream>>>(
        query_head_major, current_key_head_major, current_value_token_major,
        cache.page_pool, nullptr, cache.page_offsets, global_k_norm_scale,
        cache.page_tokens, cache.layer_offset_elements, base_position, token_count,
        0, context_token_major, cache.format);
    check_cuda(cudaGetLastError(), "short paged compact-global attention kernel launch");
    return;
  }

  constexpr unsigned kRepeats = gemma4_31b::kQueryHeadCount /
                                gemma4_31b::kGlobalKvHeadCount;
  const dim3 grid(gemma4_31b::kGlobalKvHeadCount, token_count);
  causal_gqa_attention_cached_chunk_global_compact_kernel<true>
      <<<grid, kRepeats * kWarpSize, 0, stream>>>(
          query_head_major, current_key_head_major, current_value_token_major,
          cache.page_pool, cache.page_offsets, global_k_norm_scale,
          cache.page_tokens, cache.layer_offset_elements, base_position,
          token_count, 0, context_token_major, cache.format);
  check_cuda(cudaGetLastError(),
             "paged compact-global cached chunk attention kernel launch");
}

void image_block_gqa_attention_cached_chunk_global_compact_paged(
    const BFloat16* query_head_major,
    const BFloat16* current_key_head_major,
    const BFloat16* current_value_token_major,
    const CompactGlobalPagedCache& cache,
    const BFloat16* global_k_norm_scale, std::uint32_t base_position,
    std::uint32_t token_count, std::uint32_t image_begin,
    std::uint32_t image_end, BFloat16* context_token_major,
    cudaStream_t stream) {
  constexpr std::string_view kOperation =
      "image_block_gqa_attention_cached_chunk_global_compact_paged";
  check_pointer(query_head_major, "paged image-block attention query");
  check_pointer(current_key_head_major,
                "paged image-block attention current key");
  check_pointer(current_value_token_major,
                "paged image-block attention current value");
  check_pointer(global_k_norm_scale, "paged image-block attention K scale");
  check_pointer(context_token_major, "paged image-block attention context");
  check_paged_compact_global_cache(cache, base_position, token_count,
                                   kOperation);
  check_image_block(base_position, token_count, image_begin, image_end,
                    kOperation);

  constexpr unsigned kRepeats = gemma4_31b::kQueryHeadCount /
                                gemma4_31b::kGlobalKvHeadCount;
  const dim3 grid(gemma4_31b::kGlobalKvHeadCount, token_count);
  causal_gqa_attention_cached_chunk_global_compact_kernel<true>
      <<<grid, kRepeats * kWarpSize, 0, stream>>>(
          query_head_major, current_key_head_major, current_value_token_major,
          cache.page_pool, cache.page_offsets, global_k_norm_scale,
          cache.page_tokens, cache.layer_offset_elements, base_position,
          token_count, 0, context_token_major, cache.format);
  check_cuda(cudaGetLastError(),
             "paged image-block compact-global attention kernel launch");
}

}  // namespace gewell::prefill_primitives
