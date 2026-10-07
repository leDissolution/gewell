#include "bf16_common.cuh"
#include "gewell/models/gemma4/26b_a4b/model.h"
#include "rope_inverse_frequency.cuh"
#include <algorithm>
#include <cmath>

namespace gewell::bf16_primitives {
namespace {

using namespace detail;

template <std::uint32_t HeadSize, std::uint32_t RotatedFrequencies,
          std::uint32_t Theta>
__global__ void generate_rope_factors_m2_kernel(BFloat16* cosine,
                                                BFloat16* sine) {
  static_assert(HeadSize % 2 == 0);
  static_assert(RotatedFrequencies <= HeadSize / 2);
  static_assert((HeadSize == gemma4_31b::kLocalHeadSize &&
                 RotatedFrequencies == gemma4_31b::kLocalHeadSize / 2 &&
                 Theta == 10'000) ||
                (HeadSize == gemma4_31b::kGlobalHeadSize &&
                 RotatedFrequencies == 64 && Theta == 1'000'000));
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= 2 * HeadSize) {
    return;
  }

  const std::uint32_t token = index / HeadSize;
  const std::uint32_t dimension = index % HeadSize;
  if (token == 0) {
    cosine[index] = __float2bfloat16_rn(1.0F);
    sine[index] = __float2bfloat16_rn(0.0F);
    return;
  }

  const std::uint32_t frequency = dimension % (HeadSize / 2);
  float angle = 0.0F;
  if (frequency < RotatedFrequencies) {
    angle = HeadSize == gemma4_31b::kGlobalHeadSize
                ? rope_inverse_frequency::get<true>(frequency)
                : rope_inverse_frequency::get<false>(frequency);
  }
  cosine[index] = __float2bfloat16_rn(cosf(angle));
  sine[index] = __float2bfloat16_rn(sinf(angle));
}

template <std::uint32_t HeadSize, std::uint32_t RotatedFrequencies,
          std::uint32_t Theta>
__global__ void generate_rope_factors_24_kernel(BFloat16* cosine,
                                                BFloat16* sine) {
  static_assert(HeadSize % 2 == 0);
  static_assert(RotatedFrequencies <= HeadSize / 2);
  static_assert((HeadSize == gemma4_31b::kLocalHeadSize &&
                 RotatedFrequencies == gemma4_31b::kLocalHeadSize / 2 &&
                 Theta == 10'000) ||
                (HeadSize == gemma4_31b::kGlobalHeadSize &&
                 RotatedFrequencies == 64 && Theta == 1'000'000));
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= kCachedAttentionM1ShortCapacity * HeadSize) {
    return;
  }

  const std::uint32_t token = index / HeadSize;
  const std::uint32_t dimension = index % HeadSize;
  if (token == 0) {
    cosine[index] = __float2bfloat16_rn(1.0F);
    sine[index] = __float2bfloat16_rn(0.0F);
    return;
  }

  const std::uint32_t frequency = dimension % (HeadSize / 2);
  float angle = 0.0F;
  if (frequency < RotatedFrequencies) {
    const float inverse_frequency =
        HeadSize == gemma4_31b::kGlobalHeadSize
            ? rope_inverse_frequency::get<true>(frequency)
            : rope_inverse_frequency::get<false>(frequency);
    angle = static_cast<float>(token) * inverse_frequency;
  }
  cosine[index] = __float2bfloat16_rn(cosf(angle));
  sine[index] = __float2bfloat16_rn(sinf(angle));
}

template <std::uint32_t HeadSize, std::uint32_t RotatedFrequencies,
          std::uint32_t Theta>
__global__ void generate_rope_factors_m1_kernel(
    BFloat16* cosine, BFloat16* sine, std::uint32_t absolute_position) {
  static_assert(HeadSize % 2 == 0);
  static_assert(RotatedFrequencies <= HeadSize / 2);
  static_assert((HeadSize == gemma4_31b::kLocalHeadSize &&
                 RotatedFrequencies == gemma4_31b::kLocalHeadSize / 2 &&
                 Theta == 10'000) ||
                (HeadSize == gemma4_31b::kGlobalHeadSize &&
                 RotatedFrequencies == 64 && Theta == 1'000'000));
  const std::uint32_t dimension = blockIdx.x * blockDim.x + threadIdx.x;
  if (dimension >= HeadSize) {
    return;
  }
  if (absolute_position == 0) {
    cosine[dimension] = __float2bfloat16_rn(1.0F);
    sine[dimension] = __float2bfloat16_rn(0.0F);
    return;
  }

  const std::uint32_t frequency = dimension % (HeadSize / 2);
  float angle = 0.0F;
  if (frequency < RotatedFrequencies) {
    const float inverse_frequency =
        HeadSize == gemma4_31b::kGlobalHeadSize
            ? rope_inverse_frequency::get<true>(frequency)
            : rope_inverse_frequency::get<false>(frequency);
    angle = static_cast<float>(absolute_position) * inverse_frequency;
  }
  cosine[dimension] = __float2bfloat16_rn(cosf(angle));
  sine[dimension] = __float2bfloat16_rn(sinf(angle));
}

constexpr unsigned kRopeFactorsBatchEntries = 128;
struct RopeFactorsBatch {
  RopeFactorsM1Input inputs[kRopeFactorsBatchEntries];
};

template <std::uint32_t HeadSize, std::uint32_t RotatedFrequencies,
          bool Global>
__global__ void generate_rope_factors_m1_batch_kernel(
    const __grid_constant__ RopeFactorsBatch batch) {
  const auto& input = batch.inputs[blockIdx.y];
  auto* cosine = Global ? input.global_cos : input.local_cos;
  auto* sine = Global ? input.global_sin : input.local_sin;
  const std::uint32_t dimension = blockIdx.x * blockDim.x + threadIdx.x;
  if (dimension >= HeadSize) return;
  if (input.absolute_position == 0) {
    cosine[dimension] = __float2bfloat16_rn(1.0F);
    sine[dimension] = __float2bfloat16_rn(0.0F);
    return;
  }
  const std::uint32_t frequency = dimension % (HeadSize / 2);
  float angle = 0.0F;
  if (frequency < RotatedFrequencies) {
    const float inverse_frequency =
        rope_inverse_frequency::get<Global>(frequency);
    angle = static_cast<float>(input.absolute_position) * inverse_frequency;
  }
  cosine[dimension] = __float2bfloat16_rn(cosf(angle));
  sine[dimension] = __float2bfloat16_rn(sinf(angle));
}

template <std::uint32_t HeadSize, std::uint32_t RotatedFrequencies,
          std::uint32_t Theta>
__global__ void generate_rope_factors_m1_device_position_kernel(
    BFloat16* cosine, BFloat16* sine,
    const std::uint32_t* absolute_position) {
  static_assert(HeadSize % 2 == 0);
  static_assert(RotatedFrequencies <= HeadSize / 2);
  static_assert((HeadSize == gemma4_31b::kLocalHeadSize &&
                 RotatedFrequencies == gemma4_31b::kLocalHeadSize / 2 &&
                 Theta == 10'000) ||
                (HeadSize == gemma4_31b::kGlobalHeadSize &&
                 RotatedFrequencies == 64 && Theta == 1'000'000));
  const std::uint32_t position = absolute_position[0];
  if (position < kGraphDecodeFirstPosition ||
      position > kGraphDecodeLastPosition) {
    return;
  }
  const std::uint32_t dimension = blockIdx.x * blockDim.x + threadIdx.x;
  if (dimension >= HeadSize) {
    return;
  }

  const std::uint32_t frequency = dimension % (HeadSize / 2);
  float angle = 0.0F;
  if (frequency < RotatedFrequencies) {
    const float inverse_frequency =
        HeadSize == gemma4_31b::kGlobalHeadSize
            ? rope_inverse_frequency::get<true>(frequency)
            : rope_inverse_frequency::get<false>(frequency);
    angle = static_cast<float>(position) * inverse_frequency;
  }
  cosine[dimension] = __float2bfloat16_rn(cosf(angle));
  sine[dimension] = __float2bfloat16_rn(sinf(angle));
}

template <std::uint32_t HeadSize>
__global__ void apply_rope_transpose_m2_kernel(const BFloat16* input,
                                               const BFloat16* cosine,
                                               const BFloat16* sine,
                                               BFloat16* output,
                                               std::uint32_t heads,
                                               std::size_t elements) {
  static_assert(HeadSize % 2 == 0);
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }

  const std::uint32_t dimension =
      static_cast<std::uint32_t>(index % HeadSize);
  const std::size_t row = index / HeadSize;
  const std::uint32_t head = static_cast<std::uint32_t>(row % heads);
  const std::uint32_t token = static_cast<std::uint32_t>(row / heads);
  const std::uint32_t paired_dimension =
      dimension < HeadSize / 2 ? dimension + HeadSize / 2
                              : dimension - HeadSize / 2;
  const std::size_t input_row =
      (static_cast<std::size_t>(token) * heads + head) * HeadSize;
  const float value = __bfloat162float(input[input_row + dimension]);
  float rotated = __bfloat162float(input[input_row + paired_dimension]);
  if (dimension < HeadSize / 2) {
    rotated = -rotated;
  }

  const std::size_t factor_index =
      static_cast<std::size_t>(token) * HeadSize + dimension;
  // These named BF16 temporaries are intentional: eager Transformers rounds
  // both products before it rounds their sum.
  const BFloat16 direct_product = __float2bfloat16_rn(
      value * __bfloat162float(cosine[factor_index]));
  const BFloat16 rotated_product = __float2bfloat16_rn(
      rotated * __bfloat162float(sine[factor_index]));
  const BFloat16 result = __float2bfloat16_rn(
      __bfloat162float(direct_product) + __bfloat162float(rotated_product));
  const std::size_t output_index =
      (static_cast<std::size_t>(head) * 2 + token) * HeadSize + dimension;
  output[output_index] = result;
}

template <std::uint32_t HeadSize>
__global__ void apply_rope_m1_kernel(const BFloat16* input,
                                     const BFloat16* cosine,
                                     const BFloat16* sine,
                                     BFloat16* output,
                                     std::size_t elements) {
  static_assert(HeadSize % 2 == 0);
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }

  const std::uint32_t dimension =
      static_cast<std::uint32_t>(index % HeadSize);
  const std::size_t row_offset = index - dimension;
  const std::uint32_t paired_dimension =
      dimension < HeadSize / 2 ? dimension + HeadSize / 2
                              : dimension - HeadSize / 2;
  const float value = __bfloat162float(input[index]);
  float rotated = __bfloat162float(input[row_offset + paired_dimension]);
  if (dimension < HeadSize / 2) {
    rotated = -rotated;
  }

  const BFloat16 direct_product = __float2bfloat16_rn(
      value * __bfloat162float(cosine[dimension]));
  const BFloat16 rotated_product = __float2bfloat16_rn(
      rotated * __bfloat162float(sine[dimension]));
  output[index] = __float2bfloat16_rn(
      __bfloat162float(direct_product) +
      __bfloat162float(rotated_product));
}

template <std::uint32_t HeadSize>
__global__ void apply_rope_m1_batch_kernel(
    const BFloat16* input, const BFloat16* cosine, const BFloat16* sine,
    BFloat16* output, std::size_t row_elements) {
  const std::size_t row = blockIdx.y;
  input += row * row_elements;
  output += row * row_elements;
  cosine += row * HeadSize;
  sine += row * HeadSize;
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= row_elements) return;
  const std::uint32_t dimension =
      static_cast<std::uint32_t>(index % HeadSize);
  const std::size_t row_offset = index - dimension;
  const std::uint32_t paired_dimension =
      dimension < HeadSize / 2 ? dimension + HeadSize / 2
                               : dimension - HeadSize / 2;
  const float value = __bfloat162float(input[index]);
  float rotated = __bfloat162float(input[row_offset + paired_dimension]);
  if (dimension < HeadSize / 2) rotated = -rotated;
  const BFloat16 direct_product = __float2bfloat16_rn(
      value * __bfloat162float(cosine[dimension]));
  const BFloat16 rotated_product = __float2bfloat16_rn(
      rotated * __bfloat162float(sine[dimension]));
  output[index] = __float2bfloat16_rn(
      __bfloat162float(direct_product) +
      __bfloat162float(rotated_product));
}

}  // namespace

void generate_rope_factors_m2(BFloat16* local_cos, BFloat16* local_sin,
                              BFloat16* global_cos, BFloat16* global_sin,
                              cudaStream_t stream) {
  check_pointer(local_cos, "generate_rope_factors_m2 local cosine");
  check_pointer(local_sin, "generate_rope_factors_m2 local sine");
  check_pointer(global_cos, "generate_rope_factors_m2 global cosine");
  check_pointer(global_sin, "generate_rope_factors_m2 global sine");

  constexpr std::size_t kLocalElements =
      2 * gemma4_31b::kLocalHeadSize;
  constexpr std::size_t kGlobalElements =
      2 * gemma4_31b::kGlobalHeadSize;
  generate_rope_factors_m2_kernel<gemma4_31b::kLocalHeadSize,
                                  gemma4_31b::kLocalHeadSize / 2, 10'000>
      <<<blocks_for(kLocalElements), kThreads, 0, stream>>>(local_cos,
                                                            local_sin);
  check_cuda(cudaGetLastError(),
             "generate local RoPE factors M=2 kernel launch");
  generate_rope_factors_m2_kernel<gemma4_31b::kGlobalHeadSize, 64,
                                  1'000'000>
      <<<blocks_for(kGlobalElements), kThreads, 0, stream>>>(global_cos,
                                                             global_sin);
  check_cuda(cudaGetLastError(),
             "generate global RoPE factors M=2 kernel launch");
}

void generate_rope_factors_24(BFloat16* local_cos, BFloat16* local_sin,
                              BFloat16* global_cos, BFloat16* global_sin,
                              cudaStream_t stream) {
  check_pointer(local_cos, "generate_rope_factors_24 local cosine");
  check_pointer(local_sin, "generate_rope_factors_24 local sine");
  check_pointer(global_cos, "generate_rope_factors_24 global cosine");
  check_pointer(global_sin, "generate_rope_factors_24 global sine");

  constexpr std::size_t kLocalElements =
      kCachedAttentionM1ShortCapacity * gemma4_31b::kLocalHeadSize;
  constexpr std::size_t kGlobalElements =
      kCachedAttentionM1ShortCapacity * gemma4_31b::kGlobalHeadSize;
  generate_rope_factors_24_kernel<gemma4_31b::kLocalHeadSize,
                                  gemma4_31b::kLocalHeadSize / 2, 10'000>
      <<<blocks_for(kLocalElements), kThreads, 0, stream>>>(local_cos,
                                                            local_sin);
  check_cuda(cudaGetLastError(),
             "generate local RoPE factors 24 kernel launch");
  generate_rope_factors_24_kernel<gemma4_31b::kGlobalHeadSize, 64,
                                  1'000'000>
      <<<blocks_for(kGlobalElements), kThreads, 0, stream>>>(global_cos,
                                                             global_sin);
  check_cuda(cudaGetLastError(),
             "generate global RoPE factors 24 kernel launch");
}

void generate_rope_factors_m1(BFloat16* local_cos, BFloat16* local_sin,
                              BFloat16* global_cos, BFloat16* global_sin,
                              std::uint32_t absolute_position,
                              cudaStream_t stream) {
  check_pointer(local_cos, "generate_rope_factors_m1 local cosine");
  check_pointer(local_sin, "generate_rope_factors_m1 local sine");
  check_pointer(global_cos, "generate_rope_factors_m1 global cosine");
  check_pointer(global_sin, "generate_rope_factors_m1 global sine");
  if (absolute_position >= kMaxContextTokenCount) {
    fail("generate_rope_factors_m1",
         "absolute position must be in 0..262143");
  }

  generate_rope_factors_m1_kernel<gemma4_31b::kLocalHeadSize,
                                  gemma4_31b::kLocalHeadSize / 2, 10'000>
      <<<blocks_for(gemma4_31b::kLocalHeadSize), kThreads, 0, stream>>>(
          local_cos, local_sin, absolute_position);
  check_cuda(cudaGetLastError(),
             "generate local RoPE factors M=1 kernel launch");
  generate_rope_factors_m1_kernel<gemma4_31b::kGlobalHeadSize, 64,
                                  1'000'000>
      <<<blocks_for(gemma4_31b::kGlobalHeadSize), kThreads, 0, stream>>>(
          global_cos, global_sin, absolute_position);
  check_cuda(cudaGetLastError(),
             "generate global RoPE factors M=1 kernel launch");
}

void generate_rope_factors_m1_batch(
    const std::vector<RopeFactorsM1Input>& inputs, cudaStream_t stream) {
  if (inputs.empty()) fail("generate_rope_factors_m1_batch", "empty batch");
  for (const auto& input : inputs) {
    check_pointer(input.local_cos, "batched M1 RoPE local cosine");
    check_pointer(input.local_sin, "batched M1 RoPE local sine");
    check_pointer(input.global_cos, "batched M1 RoPE global cosine");
    check_pointer(input.global_sin, "batched M1 RoPE global sine");
    if (input.absolute_position >= kMaxContextTokenCount)
      fail("generate_rope_factors_m1_batch",
           "absolute position must be in 0..262143");
  }
  RopeFactorsBatch batch{};
  for (std::size_t first = 0; first < inputs.size();
       first += kRopeFactorsBatchEntries) {
    const auto count = static_cast<unsigned>(std::min<std::size_t>(
        kRopeFactorsBatchEntries, inputs.size() - first));
    std::copy_n(inputs.begin() + first, count, batch.inputs);
    generate_rope_factors_m1_batch_kernel<256, 128, false>
        <<<dim3(blocks_for(256), count), kThreads, 0, stream>>>(batch);
    check_cuda(cudaGetLastError(), "generate batched local M1 RoPE factors");
    generate_rope_factors_m1_batch_kernel<512, 64, true>
        <<<dim3(blocks_for(512), count), kThreads, 0, stream>>>(batch);
    check_cuda(cudaGetLastError(), "generate batched global M1 RoPE factors");
  }
}

void generate_rope_factors_m1_device_position(
    BFloat16* local_cos, BFloat16* local_sin, BFloat16* global_cos,
    BFloat16* global_sin, const std::uint32_t* absolute_position,
    cudaStream_t stream) {
  check_pointer(local_cos,
                "generate_rope_factors_m1_device_position local cosine");
  check_pointer(local_sin,
                "generate_rope_factors_m1_device_position local sine");
  check_pointer(global_cos,
                "generate_rope_factors_m1_device_position global cosine");
  check_pointer(global_sin,
                "generate_rope_factors_m1_device_position global sine");
  check_pointer(absolute_position,
                "generate_rope_factors_m1_device_position position");

  generate_rope_factors_m1_device_position_kernel<
      gemma4_31b::kLocalHeadSize, gemma4_31b::kLocalHeadSize / 2, 10'000>
      <<<blocks_for(gemma4_31b::kLocalHeadSize), kThreads, 0, stream>>>(
          local_cos, local_sin, absolute_position);
  check_cuda(
      cudaGetLastError(),
      "generate local RoPE factors device-position M=1 kernel launch");
  generate_rope_factors_m1_device_position_kernel<
      gemma4_31b::kGlobalHeadSize, 64, 1'000'000>
      <<<blocks_for(gemma4_31b::kGlobalHeadSize), kThreads, 0, stream>>>(
          global_cos, global_sin, absolute_position);
  check_cuda(
      cudaGetLastError(),
      "generate global RoPE factors device-position M=1 kernel launch");
}

void apply_rope_transpose_m2(const BFloat16* input, const BFloat16* cos,
                             const BFloat16* sin, BFloat16* output,
                             std::uint32_t heads,
                             gemma4_31b::AttentionKind kind,
                             cudaStream_t stream) {
  check_pointer(input, "apply_rope_transpose_m2 input");
  check_pointer(cos, "apply_rope_transpose_m2 cosine");
  check_pointer(sin, "apply_rope_transpose_m2 sine");
  check_pointer(output, "apply_rope_transpose_m2 output");
  if (kind != gemma4_31b::AttentionKind::local &&
      kind != gemma4_31b::AttentionKind::global) {
    fail("apply_rope_transpose_m2", "invalid Gemma 4 attention kind");
  }
  const bool global = kind == gemma4_31b::AttentionKind::global;
  const std::uint32_t kv_heads = global ? gemma4_31b::kGlobalKvHeadCount
                                        : gemma4_31b::kLocalKvHeadCount;
  if (heads != gemma4_31b::kQueryHeadCount && heads != kv_heads) {
    fail("apply_rope_transpose_m2",
         "head count is neither Q nor the selected kind's KV count");
  }
  const std::uint32_t head_size = global ? gemma4_31b::kGlobalHeadSize
                                         : gemma4_31b::kLocalHeadSize;
  const std::size_t elements =
      static_cast<std::size_t>(2) * heads * head_size;
  if (global) {
    apply_rope_transpose_m2_kernel<gemma4_31b::kGlobalHeadSize>
        <<<blocks_for(elements), kThreads, 0, stream>>>(input, cos, sin,
                                                        output, heads,
                                                        elements);
  } else {
    apply_rope_transpose_m2_kernel<gemma4_31b::kLocalHeadSize>
        <<<blocks_for(elements), kThreads, 0, stream>>>(input, cos, sin,
                                                        output, heads,
                                                        elements);
  }
  check_cuda(cudaGetLastError(),
             "apply_rope_transpose_m2 kernel launch");
}

void apply_rope_m1(const BFloat16* input, const BFloat16* cos,
                   const BFloat16* sin, BFloat16* output,
                   std::uint32_t heads, gemma4_31b::AttentionKind kind,
                   cudaStream_t stream) {
  check_pointer(input, "apply_rope_m1 input");
  check_pointer(cos, "apply_rope_m1 cosine");
  check_pointer(sin, "apply_rope_m1 sine");
  check_pointer(output, "apply_rope_m1 output");
  if (kind != gemma4_31b::AttentionKind::local &&
      kind != gemma4_31b::AttentionKind::global) {
    fail("apply_rope_m1", "invalid Gemma 4 attention kind");
  }
  const bool global = kind == gemma4_31b::AttentionKind::global;
  const std::uint32_t kv_heads = global ? gemma4_31b::kGlobalKvHeadCount
                                        : gemma4_31b::kLocalKvHeadCount;
  if (heads != gemma4_31b::kQueryHeadCount &&
      heads != gemma4_26b_a4b::kQueryHeadCount && heads != kv_heads) {
    fail("apply_rope_m1",
         "head count is neither 26B/31B Q nor the selected 31B KV count");
  }
  const std::uint32_t head_size = global ? gemma4_31b::kGlobalHeadSize
                                         : gemma4_31b::kLocalHeadSize;
  const std::size_t elements = static_cast<std::size_t>(heads) * head_size;
  if (global) {
    apply_rope_m1_kernel<gemma4_31b::kGlobalHeadSize>
        <<<blocks_for(elements), kThreads, 0, stream>>>(input, cos, sin,
                                                        output, elements);
  } else {
    apply_rope_m1_kernel<gemma4_31b::kLocalHeadSize>
        <<<blocks_for(elements), kThreads, 0, stream>>>(input, cos, sin,
                                                        output, elements);
  }
  check_cuda(cudaGetLastError(), "apply_rope_m1 kernel launch");
}

void apply_rope_m1_batch(const BFloat16* input, const BFloat16* cos,
                         const BFloat16* sin, BFloat16* output,
                         std::uint32_t rows, std::uint32_t heads,
                         gemma4_31b::AttentionKind kind,
                         cudaStream_t stream) {
  check_pointer(input, "apply_rope_m1_batch input");
  check_pointer(cos, "apply_rope_m1_batch cosine");
  check_pointer(sin, "apply_rope_m1_batch sine");
  check_pointer(output, "apply_rope_m1_batch output");
  check_decode_rows(rows, "apply_rope_m1_batch");
  if (kind != gemma4_31b::AttentionKind::local &&
      kind != gemma4_31b::AttentionKind::global)
    fail("apply_rope_m1_batch", "invalid Gemma 4 attention kind");
  const bool global = kind == gemma4_31b::AttentionKind::global;
  const auto kv_heads = global ? gemma4_31b::kGlobalKvHeadCount
                               : gemma4_31b::kLocalKvHeadCount;
  if (heads != gemma4_31b::kQueryHeadCount &&
      heads != gemma4_26b_a4b::kQueryHeadCount && heads != kv_heads)
    fail("apply_rope_m1_batch",
         "head count is neither 26B/31B Q nor the selected 31B KV count");
  const std::size_t row_elements = std::size_t(heads) * (global ? 512 : 256);
  const dim3 grid{blocks_for(row_elements), rows};
  if (global)
    apply_rope_m1_batch_kernel<512><<<grid, kThreads, 0, stream>>>(
        input, cos, sin, output, row_elements);
  else
    apply_rope_m1_batch_kernel<256><<<grid, kThreads, 0, stream>>>(
        input, cos, sin, output, row_elements);
  check_cuda(cudaGetLastError(), "apply_rope_m1_batch kernel launch");
}

}  // namespace gewell::bf16_primitives
