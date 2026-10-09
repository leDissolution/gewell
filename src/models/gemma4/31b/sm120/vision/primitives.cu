#include "gewell/vision_primitives.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace gewell::vision_primitives {
namespace {

constexpr unsigned kThreads = 256;
constexpr std::uint32_t kSpatialHeadSize =
    gemma4_31b::kVisionHeadSize / 2;
constexpr std::uint32_t kSpatialHalfSize = kSpatialHeadSize / 2;

// Gemma4VisionRotaryEmbedding constructs these FP32 values on CPU, then moves
// them to the GPU. Keep the exact resulting bits instead of recomputing them
// with device powf, which differs by one ULP for one frequency.
__device__ __constant__ float kVisionInverseFrequencies[kSpatialHalfSize] = {
    0x1.000000p+0F, 0x1.8c6c4ap-1F, 0x1.32ef98p-1F,
    0x1.db4c78p-2F, 0x1.7001acp-2F, 0x1.1cef1ep-2F,
    0x1.b93a6ap-3F, 0x1.55a082p-3F, 0x1.088266p-3F,
    0x1.99999ap-4F, 0x1.3d236cp-4F, 0x1.eb18f6p-5F,
    0x1.7c3d2ap-5F, 0x1.2667bep-5F, 0x1.c7e500p-6F,
    0x1.60fb8cp-6F, 0x1.114d36p-6F, 0x1.a7370cp-7F,
};

static_assert(gemma4_31b::kVisionHeadSize == 72);
static_assert(kSpatialHeadSize == 36);
static_assert(kSpatialHalfSize == 18);

[[noreturn]] void fail(std::string_view operation, std::string_view detail) {
  throw std::runtime_error(std::string(operation) + ": " +
                           std::string(detail));
}

void check_cuda(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess) {
    fail(operation, cudaGetErrorString(status));
  }
}

void check_pointer(const void* pointer, std::string_view name) {
  if (pointer == nullptr) {
    fail(name, "null device pointer");
  }
}

void check_patch_rows(std::uint32_t patch_rows, std::string_view operation) {
  if (patch_rows == 0 || patch_rows > kMaxPatchRows) {
    fail(operation, "patch_rows must be in 1..10080");
  }
}

unsigned blocks_for(std::size_t elements) {
  const std::size_t blocks = (elements + kThreads - 1) / kThreads;
  if (blocks > std::numeric_limits<unsigned>::max()) {
    fail("vision kernel launch", "element count exceeds the CUDA grid");
  }
  return static_cast<unsigned>(blocks);
}

__global__ void normalize_patch_values_kernel(const std::uint8_t* input,
                                               BFloat16* output,
                                               std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const float value = __fmul_rn(input[index], 1.0F / 255.0F);
    output[index] = __float2bfloat16_rn(2.0F * (value - 0.5F));
  }
}

__global__ void add_patch_position_embeddings_kernel(
    const BFloat16* projected, const std::int32_t* position_ids,
    const BFloat16* position_table, BFloat16* output,
    std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  constexpr std::size_t kHidden = gemma4_31b::kVisionHiddenSize;
  constexpr std::size_t kPositionPlane =
      static_cast<std::size_t>(gemma4_31b::kVisionPositionCount) * kHidden;
  const std::size_t row = index / kHidden;
  const std::size_t dimension = index % kHidden;
  const std::int32_t x = position_ids[row * 2];
  const std::int32_t y = position_ids[row * 2 + 1];
  if (x < 0 || y < 0 ||
      x >= static_cast<std::int32_t>(gemma4_31b::kVisionPositionCount) ||
      y >= static_cast<std::int32_t>(gemma4_31b::kVisionPositionCount)) {
    output[index] = __float2bfloat16_rn(NAN);
    return;
  }
  const BFloat16 position = __float2bfloat16_rn(
      __bfloat162float(position_table[static_cast<std::size_t>(x) * kHidden +
                                      dimension]) +
      __bfloat162float(position_table[kPositionPlane +
                                      static_cast<std::size_t>(y) * kHidden +
                                      dimension]));
  output[index] = __float2bfloat16_rn(__bfloat162float(projected[index]) +
                                      __bfloat162float(position));
}

__device__ __forceinline__ float vision_inverse_frequency(
    std::uint32_t frequency) {
  return kVisionInverseFrequencies[frequency];
}

__global__ void apply_2d_rope_transpose_kernel(
    const BFloat16* input, const std::int32_t* position_ids,
    BFloat16* output, std::uint32_t patch_rows) {
  constexpr std::uint32_t kHeads = gemma4_31b::kVisionHeadCount;
  constexpr std::uint32_t kHeadSize = gemma4_31b::kVisionHeadSize;
  __shared__ BFloat16 cosine[kHeadSize];
  __shared__ BFloat16 sine[kHeadSize];
  const std::uint32_t token = blockIdx.x;
  if (threadIdx.x < kHeadSize) {
    const std::uint32_t spatial_axis = threadIdx.x / kSpatialHeadSize;
    const std::uint32_t spatial_dimension = threadIdx.x % kSpatialHeadSize;
    const std::uint32_t frequency = spatial_dimension % kSpatialHalfSize;
    const std::int32_t position = position_ids[token * 2 + spatial_axis];
    if (position < 0) {
      cosine[threadIdx.x] = __float2bfloat16_rn(NAN);
      sine[threadIdx.x] = __float2bfloat16_rn(NAN);
    } else {
      const float angle =
          static_cast<float>(position) * vision_inverse_frequency(frequency);
      cosine[threadIdx.x] = __float2bfloat16_rn(cosf(angle));
      sine[threadIdx.x] = __float2bfloat16_rn(sinf(angle));
    }
  }
  __syncthreads();

  constexpr std::uint32_t kTokenElements = kHeads * kHeadSize;
  for (std::uint32_t token_index = threadIdx.x; token_index < kTokenElements;
       token_index += blockDim.x) {
    const std::uint32_t head = token_index / kHeadSize;
    const std::uint32_t dimension = token_index % kHeadSize;
    const std::uint32_t spatial_axis = dimension / kSpatialHeadSize;
    const std::uint32_t spatial_dimension = dimension % kSpatialHeadSize;
    const std::uint32_t paired_spatial_dimension =
        spatial_dimension < kSpatialHalfSize
            ? spatial_dimension + kSpatialHalfSize
            : spatial_dimension - kSpatialHalfSize;
    const std::uint32_t paired_dimension =
        spatial_axis * kSpatialHeadSize + paired_spatial_dimension;
    const std::size_t input_row =
        (static_cast<std::size_t>(token) * kHeads + head) * kHeadSize;
    float rotated = __bfloat162float(input[input_row + paired_dimension]);
    if (spatial_dimension < kSpatialHalfSize) {
      rotated = -rotated;
    }
    const BFloat16 direct_product = __float2bfloat16_rn(
        __bfloat162float(input[input_row + dimension]) *
        __bfloat162float(cosine[dimension]));
    const BFloat16 rotated_product = __float2bfloat16_rn(
        rotated * __bfloat162float(sine[dimension]));
    output[(static_cast<std::size_t>(head) * patch_rows + token) * kHeadSize +
           dimension] = __float2bfloat16_rn(
        __bfloat162float(direct_product) + __bfloat162float(rotated_product));
  }
}

__global__ void token_heads_to_head_tokens_kernel(
    const BFloat16* input, BFloat16* output, std::uint32_t patch_rows,
    std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  constexpr std::uint32_t kHeads = gemma4_31b::kVisionHeadCount;
  constexpr std::uint32_t kHeadSize = gemma4_31b::kVisionHeadSize;
  const std::uint32_t dimension = index % kHeadSize;
  const std::size_t row = index / kHeadSize;
  const std::uint32_t head = row % kHeads;
  const std::uint32_t token = row / kHeads;
  output[(static_cast<std::size_t>(head) * patch_rows + token) * kHeadSize +
         dimension] = input[index];
}

__global__ void head_tokens_to_token_heads_kernel(
    const BFloat16* input, BFloat16* output, std::uint32_t patch_rows,
    std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  constexpr std::uint32_t kHeadSize = gemma4_31b::kVisionHeadSize;
  const std::uint32_t dimension = index % kHeadSize;
  const std::size_t row = index / kHeadSize;
  const std::uint32_t token = row % patch_rows;
  const std::uint32_t head = row / patch_rows;
  output[(static_cast<std::size_t>(token) *
              gemma4_31b::kVisionHeadCount +
          head) *
             kHeadSize +
         dimension] = input[index];
}

__global__ void softmax_rows_kernel(const BFloat16* scores,
                                    BFloat16* probabilities,
                                    std::uint32_t columns) {
  __shared__ float reduction[kThreads];
  const std::size_t row_offset =
      static_cast<std::size_t>(blockIdx.x) * columns;
  float maximum = -INFINITY;
  for (std::uint32_t column = threadIdx.x; column < columns;
       column += blockDim.x) {
    maximum = fmaxf(maximum, __bfloat162float(scores[row_offset + column]));
  }
  reduction[threadIdx.x] = maximum;
  __syncthreads();
  for (unsigned offset = kThreads / 2; offset != 0; offset /= 2) {
    if (threadIdx.x < offset) {
      reduction[threadIdx.x] =
          fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  maximum = reduction[0];

  float sum = 0.0F;
  for (std::uint32_t column = threadIdx.x; column < columns;
       column += blockDim.x) {
    sum += expf(__bfloat162float(scores[row_offset + column]) - maximum);
  }
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (unsigned offset = kThreads / 2; offset != 0; offset /= 2) {
    if (threadIdx.x < offset) {
      reduction[threadIdx.x] += reduction[threadIdx.x + offset];
    }
    __syncthreads();
  }
  const float inverse_sum = 1.0F / reduction[0];
  for (std::uint32_t column = threadIdx.x; column < columns;
       column += blockDim.x) {
    probabilities[row_offset + column] = __float2bfloat16_rn(
        expf(__bfloat162float(scores[row_offset + column]) - maximum) *
        inverse_sum);
  }
}

__global__ void pool_3x3_scaled_kernel(
    const BFloat16* input, const std::int32_t* position_ids, float* output,
    std::uint32_t patch_rows) {
  __shared__ std::int32_t source_rows[9];
  const std::int32_t derived_width = position_ids[(patch_rows - 1) * 2] + 1;
  if (derived_width <= 0 || derived_width % 3 != 0 ||
      patch_rows % static_cast<std::uint32_t>(derived_width) != 0) {
    for (std::uint32_t dimension = threadIdx.x;
         dimension < gemma4_31b::kVisionHiddenSize;
         dimension += blockDim.x) {
      output[static_cast<std::size_t>(blockIdx.x) *
                 gemma4_31b::kVisionHiddenSize +
             dimension] = NAN;
    }
    return;
  }
  const std::uint32_t patch_grid_width =
      static_cast<std::uint32_t>(derived_width);
  const std::uint32_t output_grid_width = patch_grid_width / 3;
  const std::uint32_t output_x = blockIdx.x % output_grid_width;
  const std::uint32_t output_y = blockIdx.x / output_grid_width;
  if (threadIdx.x < 9) {
    source_rows[threadIdx.x] = -1;
  }
  __syncthreads();
  for (std::uint32_t row = threadIdx.x; row < patch_rows;
       row += blockDim.x) {
    const std::int32_t x = position_ids[row * 2];
    const std::int32_t y = position_ids[row * 2 + 1];
    if (x >= 0 && y >= 0 && static_cast<std::uint32_t>(x / 3) == output_x &&
        static_cast<std::uint32_t>(y / 3) == output_y) {
      const std::uint32_t slot =
          static_cast<std::uint32_t>(y % 3) * 3 +
          static_cast<std::uint32_t>(x % 3);
      source_rows[slot] = static_cast<std::int32_t>(row);
    }
  }
  __syncthreads();

  constexpr float kAverageWeight = 1.0F / 9.0F;
  constexpr float kRootHidden = 33.941125496954281F;
  for (std::uint32_t dimension = threadIdx.x;
       dimension < gemma4_31b::kVisionHiddenSize;
       dimension += blockDim.x) {
    float average = 0.0F;
    bool complete = true;
    for (std::uint32_t slot = 0; slot < 9; ++slot) {
      const std::int32_t source_row = source_rows[slot];
      if (source_row < 0) {
        complete = false;
        break;
      }
      average = fmaf(
          kAverageWeight,
          __bfloat162float(
              input[static_cast<std::size_t>(source_row) *
                        gemma4_31b::kVisionHiddenSize + dimension]),
          average);
    }
    const BFloat16 average_bf16 =
        complete ? __float2bfloat16_rn(average) : __float2bfloat16_rn(NAN);
    output[static_cast<std::size_t>(blockIdx.x) *
               gemma4_31b::kVisionHiddenSize +
           dimension] = __bfloat162float(average_bf16) * kRootHidden;
  }
}

__global__ void standardize_kernel(const float* input, const BFloat16* bias,
                                   const BFloat16* scale, BFloat16* output,
                                   std::size_t elements) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    const std::size_t dimension =
        index % gemma4_31b::kVisionHiddenSize;
    output[index] = __float2bfloat16_rn(
        (input[index] - __bfloat162float(bias[dimension])) *
        __bfloat162float(scale[dimension]));
  }
}

}  // namespace

void normalize_patch_values(const std::uint8_t* input, BFloat16* output,
                            std::uint32_t patch_rows, cudaStream_t stream) {
  check_pointer(input, "normalize_patch_values input");
  check_pointer(output, "normalize_patch_values output");
  check_patch_rows(patch_rows, "normalize_patch_values");
  const std::size_t elements =
      static_cast<std::size_t>(patch_rows) * gemma4_31b::kVisionPatchWidth;
  normalize_patch_values_kernel<<<blocks_for(elements), kThreads, 0, stream>>>(
      input, output, elements);
  check_cuda(cudaGetLastError(), "normalize_patch_values kernel launch");
}

void add_patch_position_embeddings(
    const BFloat16* projected, const std::int32_t* position_ids,
    const BFloat16* position_table, BFloat16* output,
    std::uint32_t patch_rows, cudaStream_t stream) {
  check_pointer(projected, "add_patch_position_embeddings projected");
  check_pointer(position_ids, "add_patch_position_embeddings positions");
  check_pointer(position_table, "add_patch_position_embeddings table");
  check_pointer(output, "add_patch_position_embeddings output");
  check_patch_rows(patch_rows, "add_patch_position_embeddings");
  const std::size_t elements =
      static_cast<std::size_t>(patch_rows) * gemma4_31b::kVisionHiddenSize;
  add_patch_position_embeddings_kernel
      <<<blocks_for(elements), kThreads, 0, stream>>>(
          projected, position_ids, position_table, output, elements);
  check_cuda(cudaGetLastError(),
             "add_patch_position_embeddings kernel launch");
}

void apply_2d_rope_transpose(const BFloat16* input,
                             const std::int32_t* position_ids,
                             BFloat16* output, std::uint32_t patch_rows,
                             cudaStream_t stream) {
  check_pointer(input, "apply_2d_rope_transpose input");
  check_pointer(position_ids, "apply_2d_rope_transpose positions");
  check_pointer(output, "apply_2d_rope_transpose output");
  check_patch_rows(patch_rows, "apply_2d_rope_transpose");
  apply_2d_rope_transpose_kernel<<<patch_rows, kThreads, 0, stream>>>(
      input, position_ids, output, patch_rows);
  check_cuda(cudaGetLastError(), "apply_2d_rope_transpose kernel launch");
}

void token_heads_to_head_tokens(const BFloat16* input, BFloat16* output,
                                std::uint32_t patch_rows,
                                cudaStream_t stream) {
  check_pointer(input, "token_heads_to_head_tokens input");
  check_pointer(output, "token_heads_to_head_tokens output");
  check_patch_rows(patch_rows, "token_heads_to_head_tokens");
  const std::size_t elements =
      static_cast<std::size_t>(patch_rows) *
      gemma4_31b::kVisionHeadCount * gemma4_31b::kVisionHeadSize;
  token_heads_to_head_tokens_kernel
      <<<blocks_for(elements), kThreads, 0, stream>>>(input, output,
                                                      patch_rows, elements);
  check_cuda(cudaGetLastError(),
             "token_heads_to_head_tokens kernel launch");
}

void head_tokens_to_token_heads(const BFloat16* input, BFloat16* output,
                                std::uint32_t patch_rows,
                                cudaStream_t stream) {
  check_pointer(input, "head_tokens_to_token_heads input");
  check_pointer(output, "head_tokens_to_token_heads output");
  check_patch_rows(patch_rows, "head_tokens_to_token_heads");
  const std::size_t elements =
      static_cast<std::size_t>(patch_rows) *
      gemma4_31b::kVisionHeadCount * gemma4_31b::kVisionHeadSize;
  head_tokens_to_token_heads_kernel
      <<<blocks_for(elements), kThreads, 0, stream>>>(input, output,
                                                      patch_rows, elements);
  check_cuda(cudaGetLastError(),
             "head_tokens_to_token_heads kernel launch");
}

void softmax_rows(const BFloat16* scores, BFloat16* probabilities,
                  std::uint32_t rows, std::uint32_t columns,
                  cudaStream_t stream) {
  check_pointer(scores, "softmax_rows scores");
  check_pointer(probabilities, "softmax_rows probabilities");
  if (rows == 0) {
    fail("softmax_rows", "rows must be positive");
  }
  if (columns == 0 || columns > kMaxPatchRows) {
    fail("softmax_rows", "columns must be in 1..10080");
  }
  softmax_rows_kernel<<<rows, kThreads, 0, stream>>>(scores, probabilities,
                                                     columns);
  check_cuda(cudaGetLastError(), "softmax_rows kernel launch");
}

void pool_3x3_scaled(const BFloat16* input,
                     const std::int32_t* position_ids, float* output,
                     std::uint32_t patch_rows, cudaStream_t stream) {
  check_pointer(input, "pool_3x3_scaled input");
  check_pointer(position_ids, "pool_3x3_scaled positions");
  check_pointer(output, "pool_3x3_scaled output");
  check_patch_rows(patch_rows, "pool_3x3_scaled");
  constexpr std::uint32_t kPoolArea =
      gemma4_31b::kVisionPoolSize * gemma4_31b::kVisionPoolSize;
  if (patch_rows % kPoolArea != 0) {
    fail("pool_3x3_scaled", "patch_rows must be divisible by 9");
  }
  const std::uint32_t output_rows = patch_rows / kPoolArea;
  pool_3x3_scaled_kernel<<<output_rows, kThreads, 0, stream>>>(
      input, position_ids, output, patch_rows);
  check_cuda(cudaGetLastError(), "pool_3x3_scaled kernel launch");
}

void standardize(const float* input, const BFloat16* bias,
                 const BFloat16* scale, BFloat16* output,
                 std::uint32_t rows, cudaStream_t stream) {
  check_pointer(input, "standardize input");
  check_pointer(bias, "standardize bias");
  check_pointer(scale, "standardize scale");
  check_pointer(output, "standardize output");
  if (rows == 0 || rows > gemma4_31b::kVisionMaxSoftTokenCount) {
    fail("standardize", "rows must be in 1..1120");
  }
  const std::size_t elements =
      static_cast<std::size_t>(rows) * gemma4_31b::kVisionHiddenSize;
  standardize_kernel<<<blocks_for(elements), kThreads, 0, stream>>>(
      input, bias, scale, output, elements);
  check_cuda(cudaGetLastError(), "standardize kernel launch");
}

}  // namespace gewell::vision_primitives
