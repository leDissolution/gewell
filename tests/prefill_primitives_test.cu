#include "gewell/prefill_primitives.h"
#ifdef GEWELL_TEST_FUSED_PREFILL
#include "fused_prefill.h"
#endif

#include "gewell/bf16_primitives.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace gewell::prefill_primitives {
namespace {

constexpr unsigned kReferenceThreads = 256;
constexpr unsigned kRuntimeAttentionThreads = 32;
constexpr std::array<std::uint32_t, 4> kSelectedPositions{0, 1, 1'022,
                                                         1'023};

static_assert(kAttentionMatrixElements == 33'554'432);
static_assert(kAttentionScoreScratchBytes == 67'108'864);
static_assert(kAttentionProbabilityBytes == 67'108'864);
static_assert(kTensorAttentionStagedBytes == 8'388'608);
static_assert(kTensorAttentionScoreBytes == 134'217'728);
static_assert(kTensorAttentionProbabilityBytes == 67'108'864);
static_assert(kTensorAttentionNumeratorBytes == 67'108'864);
static_assert(kTensorAttentionStateBytes == 131'072);
static_assert(kTensorAttentionScratchBytes == 285'474'816);

[[noreturn]] void fail(std::string_view operation, std::string_view detail) {
  throw std::runtime_error(std::string(operation) + ": " +
                           std::string(detail));
}

void check_cuda(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess) {
    fail(operation, cudaGetErrorString(status));
  }
}

void check_cublas(cublasStatus_t status, std::string_view operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    fail(operation, "cuBLAS status " + std::to_string(status));
  }
}

class CublasHandle {
 public:
  CublasHandle() {
    check_cublas(cublasCreate(&handle_), "prefill test cublasCreate");
    check_cublas(cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH),
                 "prefill test tensor-op math mode");
  }
  ~CublasHandle() {
    if (handle_ != nullptr) {
      cublasDestroy(handle_);
    }
  }
  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;
  cublasHandle_t get() const { return handle_; }

 private:
  cublasHandle_t handle_{nullptr};
};

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t elements) : elements_(elements) {
    if (elements == 0) {
      fail("DeviceBuffer", "zero-sized allocation");
    }
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer_),
                          elements * sizeof(T)),
               "prefill primitive test cudaMalloc");
  }

  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* get() { return pointer_; }
  const T* get() const { return pointer_; }

  void copy_from(const std::vector<T>& values) {
    if (values.size() != elements_) {
      fail("DeviceBuffer::copy_from", "host/device size mismatch");
    }
    check_cuda(cudaMemcpy(pointer_, values.data(), elements_ * sizeof(T),
                          cudaMemcpyHostToDevice),
               "prefill primitive test host-to-device copy");
  }

  void fill_byte(unsigned char value) {
    check_cuda(cudaMemset(pointer_, value, elements_ * sizeof(T)),
               "prefill primitive test device fill");
  }

  std::vector<T> copy_to_host() const {
    return copy_slice(0, elements_);
  }

  std::vector<T> copy_slice(std::size_t offset, std::size_t count) const {
    if (offset > elements_ || count > elements_ - offset || count == 0) {
      fail("DeviceBuffer::copy_slice", "invalid slice");
    }
    std::vector<T> values(count);
    check_cuda(cudaMemcpy(values.data(), pointer_ + offset, count * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "prefill primitive test device-to-host copy");
    return values;
  }

 private:
  T* pointer_{nullptr};
  std::size_t elements_{};
};

BFloat16 to_bf16(float value) { return __float2bfloat16_rn(value); }

float to_float(BFloat16 value) { return __bfloat162float(value); }

std::uint16_t bf16_bits(BFloat16 value) {
  std::uint16_t result = 0;
  static_assert(sizeof(result) == sizeof(value));
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

class GuardedContext {
 public:
  explicit GuardedContext(std::size_t elements)
      : elements_(elements), storage_(elements + 2 * kGuardElements) {
    storage_.fill_byte(0x5a);
    fill_byte(0xff);
  }

  BFloat16* get() { return storage_.get() + kGuardElements; }
  void fill_byte(unsigned char value) {
    check_cuda(cudaMemset(get(), value, elements_ * sizeof(BFloat16)),
               "guarded context fill");
  }
  std::vector<BFloat16> copy_to_host() const {
    for (std::size_t offset : {std::size_t{0}, kGuardElements + elements_}) {
      for (BFloat16 value : storage_.copy_slice(offset, kGuardElements)) {
        if (bf16_bits(value) != 0x5a5a) {
          fail("guarded context", "write outside active token rows");
        }
      }
    }
    return storage_.copy_slice(kGuardElements, elements_);
  }

 private:
  static constexpr std::size_t kGuardElements = 64;
  std::size_t elements_;
  DeviceBuffer<BFloat16> storage_;
};

class GuardedTensorScratch {
 public:
  explicit GuardedTensorScratch(std::uint32_t rows)
      : bytes_(tensor_attention_scratch_bytes(rows)),
        storage_(bytes_ + 2 * kGuardBytes) {
    storage_.fill_byte(0x5a);
    check_cuda(cudaMemset(get(), 0xff, bytes_), "poison tensor scratch");
  }

  std::uint8_t* get() { return storage_.get() + kGuardBytes; }
  void check_guards() const {
    for (std::size_t offset : {std::size_t{0}, kGuardBytes + bytes_}) {
      for (std::uint8_t value : storage_.copy_slice(offset, kGuardBytes)) {
        if (value != 0x5a) {
          fail("guarded tensor scratch", "write outside declared scratch bytes");
        }
      }
    }
  }

 private:
  static constexpr std::size_t kGuardBytes = 256;
  std::size_t bytes_;
  DeviceBuffer<std::uint8_t> storage_;
};

void require_equal(const std::vector<BFloat16>& actual,
                   const std::vector<BFloat16>& expected,
                   std::string_view label) {
  if (actual.size() != expected.size()) {
    fail(label, "size mismatch");
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (bf16_bits(actual[index]) != bf16_bits(expected[index])) {
      fail(label, "BF16 mismatch at element " + std::to_string(index));
    }
  }
}

float require_close(const std::vector<BFloat16>& actual,
                    const std::vector<BFloat16>& expected, float tolerance,
                    std::string_view label) {
  if (actual.size() != expected.size()) {
    fail(label, "size mismatch");
  }
  float maximum_error = 0.0F;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const float error =
        std::abs(to_float(actual[index]) - to_float(expected[index]));
    maximum_error = std::max(maximum_error, error);
    if (!std::isfinite(to_float(actual[index])) ||
        !std::isfinite(to_float(expected[index])) || error > tolerance) {
      fail(label, "difference " + std::to_string(error) +
                      " exceeds tolerance at element " +
                      std::to_string(index));
    }
  }
  return maximum_error;
}

template <typename Function>
void require_failure(std::string_view label, Function&& function) {
  try {
    function();
  } catch (const std::runtime_error&) {
    return;
  }
  fail(label, "invalid call did not fail");
}

std::vector<BFloat16> token_major_pattern(std::uint32_t heads,
                                         std::uint32_t head_size,
                                         std::uint32_t seed,
                                         float scale) {
  std::vector<BFloat16> result(
      static_cast<std::size_t>(kTokenCount) * heads * head_size);
  for (std::uint32_t position = 0; position < kTokenCount; ++position) {
    for (std::uint32_t head = 0; head < heads; ++head) {
      const std::size_t row =
          (static_cast<std::size_t>(position) * heads + head) * head_size;
      for (std::uint32_t dimension = 0; dimension < head_size; ++dimension) {
        const std::uint32_t mixed =
            position * 17 + head * 13 + dimension * 7 + seed;
        const int integer = static_cast<int>(mixed % 29) - 14;
        result[row + dimension] =
            to_bf16(static_cast<float>(integer) * scale);
      }
    }
  }
  return result;
}

BFloat16 runtime_pattern_value(std::uint32_t absolute_position,
                               std::uint32_t head,
                               std::uint32_t dimension, std::uint32_t seed,
                               float scale) {
  const std::uint32_t mixed =
      absolute_position * 17 + head * 13 + dimension * 7 + seed;
  const int integer = static_cast<int>(mixed % 29) - 14;
  return to_bf16(static_cast<float>(integer) * scale);
}

std::vector<BFloat16> runtime_token_major_pattern(
    std::uint32_t token_count, std::uint32_t heads, std::uint32_t head_size,
    std::uint32_t base_position, std::uint32_t seed, float scale) {
  std::vector<BFloat16> result(
      static_cast<std::size_t>(token_count) * heads * head_size);
  for (std::uint32_t position = 0; position < token_count; ++position) {
    for (std::uint32_t head = 0; head < heads; ++head) {
      const std::size_t row =
          (static_cast<std::size_t>(position) * heads + head) * head_size;
      for (std::uint32_t dimension = 0; dimension < head_size; ++dimension) {
        result[row + dimension] = runtime_pattern_value(
            base_position + position, head, dimension, seed, scale);
      }
    }
  }
  return result;
}

std::vector<BFloat16> runtime_head_major_pattern(
    std::uint32_t heads, std::uint32_t token_count, std::uint32_t head_size,
    std::uint32_t base_position, std::uint32_t seed, float scale) {
  std::vector<BFloat16> result(
      static_cast<std::size_t>(heads) * token_count * head_size);
  for (std::uint32_t head = 0; head < heads; ++head) {
    for (std::uint32_t position = 0; position < token_count; ++position) {
      const std::size_t row =
          (static_cast<std::size_t>(head) * token_count + position) *
          head_size;
      for (std::uint32_t dimension = 0; dimension < head_size; ++dimension) {
        result[row + dimension] = runtime_pattern_value(
            base_position + position, head, dimension, seed, scale);
      }
    }
  }
  return result;
}

bool is_compact_global_key_dimension_host(std::uint32_t dimension) {
  return dimension < 64 || (dimension >= 256 && dimension < 320);
}

std::uint32_t compact_global_key_index_host(std::uint32_t dimension) {
  if (!is_compact_global_key_dimension_host(dimension)) {
    fail("compact_global_key_index_host", "dimension is not rotated");
  }
  return dimension < 64 ? dimension : 64 + (dimension - 256);
}

BFloat16 compact_global_reconstructed_key_host(BFloat16 value,
                                                BFloat16 scale) {
  return to_bf16(to_float(value) * to_float(scale));
}

std::vector<BFloat16> compact_global_cache(
    const std::vector<BFloat16>& key_cache,
    const std::vector<BFloat16>& value_cache, std::uint32_t capacity) {
  constexpr std::uint32_t kKvHeads = gemma4_31b::kGlobalKvHeadCount;
  constexpr std::uint32_t kHeadSize = gemma4_31b::kGlobalHeadSize;
  const std::size_t separate_elements =
      static_cast<std::size_t>(kKvHeads) * capacity * kHeadSize;
  if (key_cache.size() != separate_elements ||
      value_cache.size() != separate_elements) {
    fail("compact_global_cache", "separate cache size mismatch");
  }
  std::vector<BFloat16> result(
      static_cast<std::size_t>(kKvHeads) * capacity *
      kGlobalCompactCacheRowElements);
  for (std::uint32_t kv_head = 0; kv_head < kKvHeads; ++kv_head) {
    for (std::uint32_t position = 0; position < capacity; ++position) {
      const std::size_t separate_row =
          (static_cast<std::size_t>(kv_head) * capacity + position) *
          kHeadSize;
      const std::size_t compact_row =
          (static_cast<std::size_t>(kv_head) * capacity + position) *
          kGlobalCompactCacheRowElements;
      for (std::uint32_t compact_dimension = 0;
           compact_dimension < kGlobalCompactRotatedKeyElements;
           ++compact_dimension) {
        const std::uint32_t source_dimension =
            compact_dimension < 64
                ? compact_dimension
                : 256 + (compact_dimension - 64);
        result[compact_row + compact_dimension] =
            key_cache[separate_row + source_dimension];
      }
      std::copy_n(value_cache.begin() + separate_row, kHeadSize,
                  result.begin() + compact_row +
                      kGlobalCompactRotatedKeyElements);
    }
  }
  return result;
}

void reconstruct_unrotated_current_global_keys(
    std::vector<BFloat16>* key_head_major,
    const std::vector<BFloat16>& value_token_major,
    const std::vector<BFloat16>& scale, std::uint32_t token_count) {
  constexpr std::uint32_t kKvHeads = gemma4_31b::kGlobalKvHeadCount;
  constexpr std::uint32_t kHeadSize = gemma4_31b::kGlobalHeadSize;
  for (std::uint32_t kv_head = 0; kv_head < kKvHeads; ++kv_head) {
    for (std::uint32_t position = 0; position < token_count; ++position) {
      const std::size_t key_row =
          (static_cast<std::size_t>(kv_head) * token_count + position) *
          kHeadSize;
      const std::size_t value_row =
          (static_cast<std::size_t>(position) * kKvHeads + kv_head) *
          kHeadSize;
      for (std::uint32_t dimension = 0; dimension < kHeadSize; ++dimension) {
        if (!is_compact_global_key_dimension_host(dimension)) {
          (*key_head_major)[key_row + dimension] =
              compact_global_reconstructed_key_host(
                  value_token_major[value_row + dimension],
                  scale[dimension]);
        }
      }
    }
  }
}

void reconstruct_unrotated_cached_global_keys(
    std::vector<BFloat16>* key_cache,
    const std::vector<BFloat16>& value_cache,
    const std::vector<BFloat16>& scale, std::uint32_t capacity) {
  constexpr std::uint32_t kKvHeads = gemma4_31b::kGlobalKvHeadCount;
  constexpr std::uint32_t kHeadSize = gemma4_31b::kGlobalHeadSize;
  for (std::uint32_t kv_head = 0; kv_head < kKvHeads; ++kv_head) {
    for (std::uint32_t position = 0; position < capacity; ++position) {
      const std::size_t row =
          (static_cast<std::size_t>(kv_head) * capacity + position) *
          kHeadSize;
      for (std::uint32_t dimension = 0; dimension < kHeadSize; ++dimension) {
        if (!is_compact_global_key_dimension_host(dimension)) {
          (*key_cache)[row + dimension] =
              compact_global_reconstructed_key_host(
                  value_cache[row + dimension], scale[dimension]);
        }
      }
    }
  }
}

BFloat16 rope_factor_reference(std::uint32_t head_size,
                               std::uint32_t rotated_frequencies, float theta,
                               std::uint32_t position,
                               std::uint32_t dimension, bool cosine) {
  const std::uint32_t frequency = dimension % (head_size / 2);
  if (position == 0 || frequency >= rotated_frequencies) {
    return to_bf16(cosine ? 1.0F : 0.0F);
  }
  const float exponent = static_cast<float>(2 * frequency) /
                         static_cast<float>(head_size);
  const float inverse_frequency = 1.0F / std::pow(theta, exponent);
  const float angle = static_cast<float>(position) * inverse_frequency;
  return to_bf16(cosine ? std::cos(angle) : std::sin(angle));
}

void test_rope_factors(std::ostream& report) {
  constexpr std::size_t kLocalElements =
      static_cast<std::size_t>(kTokenCount) * gemma4_31b::kLocalHeadSize;
  constexpr std::size_t kGlobalElements =
      static_cast<std::size_t>(kTokenCount) * gemma4_31b::kGlobalHeadSize;
  DeviceBuffer<BFloat16> device_local_cos(kLocalElements);
  DeviceBuffer<BFloat16> device_local_sin(kLocalElements);
  DeviceBuffer<BFloat16> device_global_cos(kGlobalElements);
  DeviceBuffer<BFloat16> device_global_sin(kGlobalElements);
  generate_rope_factors_m1024(
      device_local_cos.get(), device_local_sin.get(), device_global_cos.get(),
      device_global_sin.get());

  const std::vector<BFloat16> local_cos = device_local_cos.copy_to_host();
  const std::vector<BFloat16> local_sin = device_local_sin.copy_to_host();
  const std::vector<BFloat16> global_cos = device_global_cos.copy_to_host();
  const std::vector<BFloat16> global_sin = device_global_sin.copy_to_host();
  for (const std::uint32_t position : kSelectedPositions) {
    for (std::uint32_t dimension = 0;
         dimension < gemma4_31b::kLocalHeadSize; ++dimension) {
      const std::size_t index =
          static_cast<std::size_t>(position) *
              gemma4_31b::kLocalHeadSize +
          dimension;
      const BFloat16 expected_cos = rope_factor_reference(
          gemma4_31b::kLocalHeadSize,
          gemma4_31b::kLocalHeadSize / 2, 10'000.0F, position, dimension,
          true);
      const BFloat16 expected_sin = rope_factor_reference(
          gemma4_31b::kLocalHeadSize,
          gemma4_31b::kLocalHeadSize / 2, 10'000.0F, position, dimension,
          false);
      if (bf16_bits(local_cos[index]) != bf16_bits(expected_cos) ||
          bf16_bits(local_sin[index]) != bf16_bits(expected_sin)) {
        fail("rope_factors_m1024_local",
             "factor mismatch at position " + std::to_string(position) +
                 " dimension " + std::to_string(dimension));
      }
    }
    for (std::uint32_t dimension = 0;
         dimension < gemma4_31b::kGlobalHeadSize; ++dimension) {
      const std::size_t index =
          static_cast<std::size_t>(position) *
              gemma4_31b::kGlobalHeadSize +
          dimension;
      const BFloat16 expected_cos = rope_factor_reference(
          gemma4_31b::kGlobalHeadSize, 64, 1'000'000.0F, position, dimension,
          true);
      const BFloat16 expected_sin = rope_factor_reference(
          gemma4_31b::kGlobalHeadSize, 64, 1'000'000.0F, position, dimension,
          false);
      if (bf16_bits(global_cos[index]) != bf16_bits(expected_cos) ||
          bf16_bits(global_sin[index]) != bf16_bits(expected_sin)) {
        fail("rope_factors_m1024_global",
             "factor mismatch at position " + std::to_string(position) +
                 " dimension " + std::to_string(dimension));
      }
    }
  }
  report << "prefill primitive rope_factors_m1024: positions=0,1,1022,1023 "
            "local/global exact=1\n";
}

BFloat16 rope_value_reference(const std::vector<BFloat16>& input,
                              const std::vector<BFloat16>& cosine,
                              const std::vector<BFloat16>& sine,
                              std::uint32_t heads, std::uint32_t head_size,
                              std::uint32_t position, std::uint32_t head,
                              std::uint32_t dimension) {
  const std::uint32_t half = head_size / 2;
  const std::uint32_t paired =
      dimension < half ? dimension + half : dimension - half;
  const std::size_t row =
      (static_cast<std::size_t>(position) * heads + head) * head_size;
  float rotated = to_float(input[row + paired]);
  if (dimension < half) {
    rotated = -rotated;
  }
  const std::size_t factor =
      static_cast<std::size_t>(position) * head_size + dimension;
  const BFloat16 direct =
      to_bf16(to_float(input[row + dimension]) * to_float(cosine[factor]));
  const BFloat16 crossed =
      to_bf16(rotated * to_float(sine[factor]));
  return to_bf16(to_float(direct) + to_float(crossed));
}

std::array<std::uint32_t, 4> selected_query_heads(std::uint32_t kv_heads) {
  const std::uint32_t repeats = gemma4_31b::kQueryHeadCount / kv_heads;
  return {0, repeats - 1, repeats, gemma4_31b::kQueryHeadCount - 1};
}

void check_rope_rows(const std::vector<BFloat16>& actual_head_major,
                     const std::vector<BFloat16>& input_token_major,
                     const std::vector<BFloat16>& cosine,
                     const std::vector<BFloat16>& sine,
                     std::uint32_t heads, std::uint32_t head_size,
                     const std::vector<std::uint32_t>& selected_heads,
                     std::string_view label) {
  for (const std::uint32_t position : kSelectedPositions) {
    for (const std::uint32_t head : selected_heads) {
      for (std::uint32_t dimension = 0; dimension < head_size; ++dimension) {
        const std::size_t actual_index =
            (static_cast<std::size_t>(head) * kTokenCount + position) *
                head_size +
            dimension;
        const BFloat16 expected = rope_value_reference(
            input_token_major, cosine, sine, heads, head_size, position, head,
            dimension);
        if (bf16_bits(actual_head_major[actual_index]) !=
            bf16_bits(expected)) {
          fail(label, "transpose/RoPE mismatch at position " +
                          std::to_string(position) + " head " +
                          std::to_string(head) + " dimension " +
                          std::to_string(dimension));
        }
      }
    }
  }
}

float attention_score_reference(const std::vector<BFloat16>& query,
                                const std::vector<BFloat16>& key,
                                std::uint32_t query_head,
                                std::uint32_t kv_head,
                                std::uint32_t query_position,
                                std::uint32_t key_position,
                                std::uint32_t head_size) {
  std::array<float, kReferenceThreads> partial{};
  const std::size_t query_row =
      (static_cast<std::size_t>(query_head) * kTokenCount + query_position) *
      head_size;
  const std::size_t key_row =
      (static_cast<std::size_t>(kv_head) * kTokenCount + key_position) *
      head_size;
  for (unsigned thread = 0; thread < kReferenceThreads; ++thread) {
    for (std::uint32_t dimension = thread; dimension < head_size;
         dimension += kReferenceThreads) {
      partial[thread] =
          std::fma(to_float(query[query_row + dimension]),
                   to_float(key[key_row + dimension]), partial[thread]);
    }
  }
  for (unsigned offset = kReferenceThreads / 2; offset != 0; offset /= 2) {
    for (unsigned thread = 0; thread < offset; ++thread) {
      partial[thread] += partial[thread + offset];
    }
  }
  return to_float(to_bf16(partial[0]));
}

struct AttentionRowReference {
  std::vector<BFloat16> scores;
  std::vector<BFloat16> probabilities;
  std::vector<BFloat16> context;
};

AttentionRowReference attention_row_reference(
    const std::vector<BFloat16>& query,
    const std::vector<BFloat16>& key,
    const std::vector<BFloat16>& value, std::uint32_t query_head,
    std::uint32_t query_position, std::uint32_t kv_heads,
    std::uint32_t head_size) {
  const std::uint32_t repeats = gemma4_31b::kQueryHeadCount / kv_heads;
  const std::uint32_t kv_head = query_head / repeats;
  AttentionRowReference result{
      std::vector<BFloat16>(query_position + 1),
      std::vector<BFloat16>(query_position + 1),
      std::vector<BFloat16>(head_size)};
  float maximum = -std::numeric_limits<float>::infinity();
  for (std::uint32_t key_position = 0; key_position <= query_position;
       ++key_position) {
    const float score = attention_score_reference(
        query, key, query_head, kv_head, query_position, key_position,
        head_size);
    result.scores[key_position] = to_bf16(score);
    maximum = std::max(maximum, score);
  }
  float denominator = 0.0F;
  for (std::uint32_t key_position = 0; key_position <= query_position;
       ++key_position) {
    denominator +=
        std::exp(to_float(result.scores[key_position]) - maximum);
  }
  for (std::uint32_t key_position = 0; key_position <= query_position;
       ++key_position) {
    const float exponential =
        std::exp(to_float(result.scores[key_position]) - maximum);
    result.probabilities[key_position] =
        to_bf16(exponential / denominator);
  }
  for (std::uint32_t dimension = 0; dimension < head_size; ++dimension) {
    const std::size_t value_zero =
        static_cast<std::size_t>(kv_head) * head_size + dimension;
    float sum = to_float(result.probabilities[0]) *
                to_float(value[value_zero]);
    for (std::uint32_t key_position = 1; key_position <= query_position;
         ++key_position) {
      const std::size_t value_index =
          (static_cast<std::size_t>(key_position) * kv_heads + kv_head) *
              head_size +
          dimension;
      sum = std::fma(to_float(result.probabilities[key_position]),
                     to_float(value[value_index]), sum);
    }
    result.context[dimension] = to_bf16(sum);
  }
  return result;
}

void check_cache(const std::vector<BFloat16>& key_cache,
                 const std::vector<BFloat16>& value_cache,
                 const std::vector<BFloat16>& key_head_major,
                 const std::vector<BFloat16>& value_token_major,
                 std::uint32_t capacity, std::uint32_t kv_heads,
                 std::uint32_t head_size, BFloat16 sentinel,
                 std::string_view label) {
  for (std::uint32_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
    for (std::uint32_t position = 0; position < capacity; ++position) {
      for (std::uint32_t dimension = 0; dimension < head_size; ++dimension) {
        const std::size_t cache_index =
            (static_cast<std::size_t>(kv_head) * capacity + position) *
                head_size +
            dimension;
        BFloat16 expected = sentinel;
        if (position < kTokenCount) {
          const std::size_t key_index =
              (static_cast<std::size_t>(kv_head) * kTokenCount + position) *
                  head_size +
              dimension;
          const std::size_t value_index =
              (static_cast<std::size_t>(position) * kv_heads + kv_head) *
                  head_size +
              dimension;
          if (bf16_bits(key_cache[cache_index]) !=
              bf16_bits(key_head_major[key_index])) {
            fail(label, "key cache mismatch at element " +
                            std::to_string(cache_index));
          }
          expected = value_token_major[value_index];
        } else if (bf16_bits(key_cache[cache_index]) !=
                   bf16_bits(sentinel)) {
          fail(label, "global key tail was overwritten");
        }
        if (bf16_bits(value_cache[cache_index]) != bf16_bits(expected)) {
          fail(label, "value cache mismatch at element " +
                          std::to_string(cache_index));
        }
      }
    }
  }
}

void check_boundary_m1_equivalence(
    const std::vector<BFloat16>& query_head_major,
    const DeviceBuffer<BFloat16>& device_key_cache,
    const DeviceBuffer<BFloat16>& device_value_cache,
    const DeviceBuffer<BFloat16>& device_prefill_scores,
    const DeviceBuffer<BFloat16>& device_prefill_probabilities,
    const DeviceBuffer<BFloat16>& device_prefill_context,
    std::uint32_t head_size, gemma4_31b::AttentionKind kind,
    std::string_view label) {
  constexpr std::size_t kBoundaryMatrixElements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) *
      bf16_primitives::kCachedAttentionM1BoundaryPositionCount;
  const std::size_t query_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * head_size;
  DeviceBuffer<BFloat16> device_query(query_elements);
  DeviceBuffer<BFloat16> device_scores(kBoundaryMatrixElements);
  DeviceBuffer<BFloat16> device_probabilities(kBoundaryMatrixElements);
  DeviceBuffer<BFloat16> device_context(query_elements);

  for (const std::uint32_t query_position : kSelectedPositions) {
    std::vector<BFloat16> query(query_elements);
    for (std::uint32_t query_head = 0;
         query_head < gemma4_31b::kQueryHeadCount; ++query_head) {
      const std::size_t source =
          (static_cast<std::size_t>(query_head) * kTokenCount +
           query_position) *
          head_size;
      const std::size_t destination =
          static_cast<std::size_t>(query_head) * head_size;
      std::copy_n(query_head_major.begin() + source, head_size,
                  query.begin() + destination);
    }
    device_query.copy_from(query);
    device_scores.fill_byte(0xff);
    device_probabilities.fill_byte(0xff);
    device_context.fill_byte(0xff);
    bf16_primitives::causal_gqa_attention_cached_m1_boundary(
        device_query.get(), device_key_cache.get(), device_value_cache.get(),
        query_position, device_scores.get(), device_probabilities.get(),
        device_context.get(), kind);

    const std::vector<BFloat16> boundary_scores =
        device_scores.copy_to_host();
    const std::vector<BFloat16> boundary_probabilities =
        device_probabilities.copy_to_host();
    const std::vector<BFloat16> boundary_context =
        device_context.copy_to_host();
    for (std::uint32_t query_head = 0;
         query_head < gemma4_31b::kQueryHeadCount; ++query_head) {
      const std::size_t prefill_matrix_row =
          (static_cast<std::size_t>(query_head) * kTokenCount +
           query_position) *
          kTokenCount;
      const std::size_t boundary_matrix_row =
          static_cast<std::size_t>(query_head) *
          bf16_primitives::kCachedAttentionM1BoundaryPositionCount;
      const std::vector<BFloat16> prefill_scores =
          device_prefill_scores.copy_slice(prefill_matrix_row, kTokenCount);
      const std::vector<BFloat16> prefill_probabilities =
          device_prefill_probabilities.copy_slice(prefill_matrix_row,
                                                   kTokenCount);
      for (std::uint32_t key_position = 0;
           key_position <= query_position; ++key_position) {
        if (bf16_bits(prefill_scores[key_position]) !=
            bf16_bits(boundary_scores[boundary_matrix_row + key_position])) {
          fail(label, "boundary-M1 score mismatch at query " +
                          std::to_string(query_position) + " head " +
                          std::to_string(query_head) + " key " +
                          std::to_string(key_position));
        }
      }
      for (std::uint32_t key_position = 0; key_position < kTokenCount;
           ++key_position) {
        if (bf16_bits(prefill_probabilities[key_position]) !=
            bf16_bits(
                boundary_probabilities[boundary_matrix_row + key_position])) {
          fail(label, "boundary-M1 probability mismatch at query " +
                          std::to_string(query_position) + " head " +
                          std::to_string(query_head) + " key " +
                          std::to_string(key_position));
        }
      }
      const std::size_t prefill_context_row =
          (static_cast<std::size_t>(query_position) *
               gemma4_31b::kQueryHeadCount +
           query_head) *
          head_size;
      const std::size_t boundary_context_row =
          static_cast<std::size_t>(query_head) * head_size;
      const std::vector<BFloat16> prefill_context =
          device_prefill_context.copy_slice(prefill_context_row, head_size);
      require_equal(
          prefill_context,
          std::vector<BFloat16>(
              boundary_context.begin() + boundary_context_row,
              boundary_context.begin() + boundary_context_row + head_size),
          label);
    }
  }
}

void test_kind(std::ostream& report, gemma4_31b::AttentionKind kind) {
  const bool global = kind == gemma4_31b::AttentionKind::global;
  const std::uint32_t head_size = global ? gemma4_31b::kGlobalHeadSize
                                         : gemma4_31b::kLocalHeadSize;
  const std::uint32_t kv_heads = global ? gemma4_31b::kGlobalKvHeadCount
                                        : gemma4_31b::kLocalKvHeadCount;
  const std::uint32_t capacity =
      global ? kGlobalCacheMinimumCapacity : kTokenCount;
  const std::string label = global ? "global" : "local";
  const std::size_t local_factor_elements =
      static_cast<std::size_t>(kTokenCount) * gemma4_31b::kLocalHeadSize;
  const std::size_t global_factor_elements =
      static_cast<std::size_t>(kTokenCount) * gemma4_31b::kGlobalHeadSize;
  DeviceBuffer<BFloat16> device_local_cos(local_factor_elements);
  DeviceBuffer<BFloat16> device_local_sin(local_factor_elements);
  DeviceBuffer<BFloat16> device_global_cos(global_factor_elements);
  DeviceBuffer<BFloat16> device_global_sin(global_factor_elements);
  generate_rope_factors_m1024(
      device_local_cos.get(), device_local_sin.get(), device_global_cos.get(),
      device_global_sin.get());
  const std::vector<BFloat16> cosine =
      global ? device_global_cos.copy_to_host()
             : device_local_cos.copy_to_host();
  const std::vector<BFloat16> sine =
      global ? device_global_sin.copy_to_host()
             : device_local_sin.copy_to_host();
  const BFloat16* device_cosine =
      global ? device_global_cos.get() : device_local_cos.get();
  const BFloat16* device_sine =
      global ? device_global_sin.get() : device_local_sin.get();

  std::vector<BFloat16> query_token_major = token_major_pattern(
      gemma4_31b::kQueryHeadCount, head_size, global ? 5 : 3, 1.0F / 64.0F);
  std::vector<BFloat16> key_token_major = token_major_pattern(
      kv_heads, head_size, global ? 11 : 7, 1.0F / 64.0F);
  std::vector<BFloat16> value_token_major = token_major_pattern(
      kv_heads, head_size, global ? 19 : 17, 1.0F / 32.0F);
  const std::size_t query_elements = query_token_major.size();
  const std::size_t key_elements = key_token_major.size();
  const std::size_t value_elements = value_token_major.size();
  DeviceBuffer<BFloat16> device_query_token(query_elements);
  DeviceBuffer<BFloat16> device_key_token(key_elements);
  DeviceBuffer<BFloat16> device_value_token(value_elements);
  DeviceBuffer<BFloat16> device_query_head(query_elements);
  DeviceBuffer<BFloat16> device_key_head(key_elements);
  device_query_token.copy_from(query_token_major);
  device_key_token.copy_from(key_token_major);
  device_value_token.copy_from(value_token_major);
  apply_rope_transpose_m1024(
      device_query_token.get(), device_cosine, device_sine,
      device_query_head.get(), gemma4_31b::kQueryHeadCount, kind);
  apply_rope_transpose_m1024(
      device_key_token.get(), device_cosine, device_sine, device_key_head.get(),
      kv_heads, kind);
  const std::vector<BFloat16> query_head_major =
      device_query_head.copy_to_host();
  const std::vector<BFloat16> key_head_major =
      device_key_head.copy_to_host();
  const auto query_heads = selected_query_heads(kv_heads);
  check_rope_rows(query_head_major, query_token_major, cosine, sine,
                  gemma4_31b::kQueryHeadCount, head_size,
                  std::vector<std::uint32_t>(query_heads.begin(),
                                             query_heads.end()),
                  "rope_transpose_m1024_" + label + "_query");
  check_rope_rows(key_head_major, key_token_major, cosine, sine, kv_heads,
                  head_size, {0, kv_heads - 1},
                  "rope_transpose_m1024_" + label + "_key");
  require_equal(device_query_token.copy_to_host(), query_token_major,
                "rope_transpose_m1024_" + label + "_query_read_only");
  require_equal(device_key_token.copy_to_host(), key_token_major,
                "rope_transpose_m1024_" + label + "_key_read_only");

  const std::size_t cache_elements =
      static_cast<std::size_t>(kv_heads) * capacity * head_size;
  const BFloat16 cache_sentinel = to_bf16(-13.0F);
  std::vector<BFloat16> cache_initial(cache_elements, cache_sentinel);
  DeviceBuffer<BFloat16> device_key_cache(cache_elements);
  DeviceBuffer<BFloat16> device_value_cache(cache_elements);
  device_key_cache.copy_from(cache_initial);
  device_value_cache.copy_from(cache_initial);
  write_kv_cache_m1024(
      device_key_head.get(), device_value_token.get(), device_key_cache.get(),
      device_value_cache.get(), capacity, kind);
  check_cache(device_key_cache.copy_to_host(),
              device_value_cache.copy_to_host(), key_head_major,
              value_token_major, capacity, kv_heads, head_size, cache_sentinel,
              "write_kv_cache_m1024_" + label);

  const std::size_t context_elements =
      static_cast<std::size_t>(kTokenCount) *
      gemma4_31b::kQueryHeadCount * head_size;
  DeviceBuffer<BFloat16> device_scores(kAttentionMatrixElements);
  DeviceBuffer<BFloat16> device_probabilities(kAttentionMatrixElements);
  DeviceBuffer<BFloat16> device_context(context_elements);
  device_scores.fill_byte(0xff);
  device_probabilities.fill_byte(0xff);
  device_context.fill_byte(0xff);
  causal_gqa_attention_m1024(
      device_query_head.get(), device_key_head.get(), device_value_token.get(),
      device_scores.get(), device_probabilities.get(), device_context.get(),
      kind);

  for (const std::uint32_t query_position : kSelectedPositions) {
    for (const std::uint32_t query_head : query_heads) {
      const AttentionRowReference expected = attention_row_reference(
          query_head_major, key_head_major, value_token_major, query_head,
          query_position, kv_heads, head_size);
      const std::size_t matrix_row =
          (static_cast<std::size_t>(query_head) * kTokenCount +
           query_position) *
          kTokenCount;
      const std::vector<BFloat16> actual_scores =
          device_scores.copy_slice(matrix_row, kTokenCount);
      const std::vector<BFloat16> actual_probabilities =
          device_probabilities.copy_slice(matrix_row, kTokenCount);
      for (std::uint32_t key_position = 0;
           key_position <= query_position; ++key_position) {
        if (bf16_bits(actual_scores[key_position]) !=
            bf16_bits(expected.scores[key_position])) {
          fail("attention_m1024_" + label,
               "score mismatch at query " +
                   std::to_string(query_position) + " head " +
                   std::to_string(query_head) + " key " +
                   std::to_string(key_position));
        }
        if (bf16_bits(actual_probabilities[key_position]) !=
            bf16_bits(expected.probabilities[key_position])) {
          fail("attention_m1024_" + label,
               "probability mismatch at query " +
                   std::to_string(query_position) + " head " +
                   std::to_string(query_head) + " key " +
                   std::to_string(key_position));
        }
      }
      for (std::uint32_t key_position = query_position + 1;
           key_position < kTokenCount; ++key_position) {
        if (bf16_bits(actual_scores[key_position]) != 0xffffU) {
          fail("attention_m1024_" + label,
               "future score scratch was overwritten");
        }
        if (bf16_bits(actual_probabilities[key_position]) != 0x0000U) {
          fail("attention_m1024_" + label,
               "future probability is not exact BF16 positive zero");
        }
      }
      const std::size_t context_row =
          (static_cast<std::size_t>(query_position) *
               gemma4_31b::kQueryHeadCount +
           query_head) *
          head_size;
      require_equal(device_context.copy_slice(context_row, head_size),
                    expected.context, "attention_m1024_" + label +
                                          "_context");
    }
  }
  check_boundary_m1_equivalence(
      query_head_major, device_key_cache, device_value_cache, device_scores,
      device_probabilities, device_context, head_size, kind,
      "attention_m1024_" + label + "_boundary_m1_equivalence");
  require_equal(device_query_head.copy_to_host(), query_head_major,
                "attention_m1024_" + label + "_query_read_only");
  require_equal(device_key_head.copy_to_host(), key_head_major,
                "attention_m1024_" + label + "_key_read_only");
  require_equal(device_value_token.copy_to_host(), value_token_major,
                "attention_m1024_" + label + "_value_read_only");
  report << "prefill primitive m1024_" << label
         << ": rope_transpose=exact cache_placement=exact "
            "attention_queries=0,1,1022,1023 score/prob/context=exact "
            "boundary_m1_equivalent=1 causal_zero=1 read_only=1\n";
}

void test_runtime_chunk_rope(std::ostream& report) {
  constexpr std::uint32_t kBasePosition = 1'022;
  constexpr std::uint32_t kChunkTokens = 3;
  const std::size_t local_factor_elements =
      static_cast<std::size_t>(kChunkTokens) *
      gemma4_31b::kLocalHeadSize;
  const std::size_t global_factor_elements =
      static_cast<std::size_t>(kChunkTokens) *
      gemma4_31b::kGlobalHeadSize;
  DeviceBuffer<BFloat16> device_local_cos(local_factor_elements);
  DeviceBuffer<BFloat16> device_local_sin(local_factor_elements);
  DeviceBuffer<BFloat16> device_global_cos(global_factor_elements);
  DeviceBuffer<BFloat16> device_global_sin(global_factor_elements);
  generate_rope_factors_chunk(
      device_local_cos.get(), device_local_sin.get(), device_global_cos.get(),
      device_global_sin.get(), kBasePosition, kChunkTokens);

  const std::vector<BFloat16> local_cos = device_local_cos.copy_to_host();
  const std::vector<BFloat16> local_sin = device_local_sin.copy_to_host();
  const std::vector<BFloat16> global_cos = device_global_cos.copy_to_host();
  const std::vector<BFloat16> global_sin = device_global_sin.copy_to_host();
  for (std::uint32_t position = 0; position < kChunkTokens; ++position) {
    for (std::uint32_t dimension = 0;
         dimension < gemma4_31b::kLocalHeadSize; ++dimension) {
      const std::size_t index =
          static_cast<std::size_t>(position) *
              gemma4_31b::kLocalHeadSize +
          dimension;
      const BFloat16 expected_cos = rope_factor_reference(
          gemma4_31b::kLocalHeadSize,
          gemma4_31b::kLocalHeadSize / 2, 10'000.0F,
          kBasePosition + position, dimension, true);
      const BFloat16 expected_sin = rope_factor_reference(
          gemma4_31b::kLocalHeadSize,
          gemma4_31b::kLocalHeadSize / 2, 10'000.0F,
          kBasePosition + position, dimension, false);
      if (bf16_bits(local_cos[index]) != bf16_bits(expected_cos) ||
          bf16_bits(local_sin[index]) != bf16_bits(expected_sin)) {
        fail("runtime_chunk_rope_local", "factor mismatch");
      }
    }
    for (std::uint32_t dimension = 0;
         dimension < gemma4_31b::kGlobalHeadSize; ++dimension) {
      const std::size_t index =
          static_cast<std::size_t>(position) *
              gemma4_31b::kGlobalHeadSize +
          dimension;
      const BFloat16 expected_cos = rope_factor_reference(
          gemma4_31b::kGlobalHeadSize, 64, 1'000'000.0F,
          kBasePosition + position, dimension, true);
      const BFloat16 expected_sin = rope_factor_reference(
          gemma4_31b::kGlobalHeadSize, 64, 1'000'000.0F,
          kBasePosition + position, dimension, false);
      if (bf16_bits(global_cos[index]) != bf16_bits(expected_cos) ||
          bf16_bits(global_sin[index]) != bf16_bits(expected_sin)) {
        fail("runtime_chunk_rope_global", "factor mismatch");
      }
    }
  }

  for (const gemma4_31b::AttentionKind kind :
       {gemma4_31b::AttentionKind::local,
        gemma4_31b::AttentionKind::global}) {
    const bool global = kind == gemma4_31b::AttentionKind::global;
    const std::uint32_t head_size = global ? gemma4_31b::kGlobalHeadSize
                                           : gemma4_31b::kLocalHeadSize;
    const std::uint32_t kv_heads = global
                                       ? gemma4_31b::kGlobalKvHeadCount
                                       : gemma4_31b::kLocalKvHeadCount;
    const std::vector<BFloat16>& cosine = global ? global_cos : local_cos;
    const std::vector<BFloat16>& sine = global ? global_sin : local_sin;
    const BFloat16* device_cosine =
        global ? device_global_cos.get() : device_local_cos.get();
    const BFloat16* device_sine =
        global ? device_global_sin.get() : device_local_sin.get();
    for (const std::uint32_t heads :
         {gemma4_31b::kQueryHeadCount, kv_heads}) {
      const std::vector<BFloat16> input = runtime_token_major_pattern(
          kChunkTokens, heads, head_size, kBasePosition, heads + head_size,
          1.0F / 64.0F);
      DeviceBuffer<BFloat16> device_input(input.size());
      DeviceBuffer<BFloat16> device_output(input.size());
      device_input.copy_from(input);
      device_output.fill_byte(0xff);
      apply_rope_transpose_chunk(
          device_input.get(), device_cosine, device_sine, device_output.get(),
          heads, kChunkTokens, kind);
      const std::vector<BFloat16> output = device_output.copy_to_host();
      for (const std::uint32_t head : {0U, heads - 1}) {
        for (std::uint32_t position = 0; position < kChunkTokens;
             ++position) {
          for (std::uint32_t dimension = 0; dimension < head_size;
               ++dimension) {
            const std::size_t output_index =
                (static_cast<std::size_t>(head) * kChunkTokens + position) *
                    head_size +
                dimension;
            const BFloat16 expected = rope_value_reference(
                input, cosine, sine, heads, head_size, position, head,
                dimension);
            if (bf16_bits(output[output_index]) != bf16_bits(expected)) {
              fail(global ? "runtime_chunk_rope_transpose_global"
                          : "runtime_chunk_rope_transpose_local",
                   "transpose mismatch");
            }
          }
        }
      }
      require_equal(device_input.copy_to_host(), input,
                    "runtime_chunk_rope_input_read_only");
    }
  }
  report << "prefill primitive runtime_chunk_rope: M=3 base=1022 "
            "local/global Q/KV exact=1\n";
}

std::vector<BFloat16> runtime_attention_row_reference(
    const std::vector<BFloat16>& query_head_major,
    const std::vector<BFloat16>& current_key_head_major,
    const std::vector<BFloat16>& current_value_token_major,
    const std::vector<BFloat16>& key_cache,
    const std::vector<BFloat16>& value_cache, std::uint32_t query_head,
    std::uint32_t chunk_query_position, std::uint32_t base_position,
    std::uint32_t token_count, std::uint32_t cache_capacity,
    std::uint32_t kv_heads, std::uint32_t head_size,
    gemma4_31b::AttentionKind kind) {
  const std::uint32_t repeats = gemma4_31b::kQueryHeadCount / kv_heads;
  const std::uint32_t kv_head = query_head / repeats;
  const std::uint32_t absolute_query_position =
      base_position + chunk_query_position;
  const bool local = kind == gemma4_31b::AttentionKind::local;
  const std::uint32_t first_key_position =
      local && absolute_query_position >= gemma4_31b::kLocalWindowSize - 1
          ? absolute_query_position - (gemma4_31b::kLocalWindowSize - 1)
          : 0;
  const std::size_t query_row =
      (static_cast<std::size_t>(query_head) * token_count +
       chunk_query_position) *
      head_size;
  float maximum = -std::numeric_limits<float>::infinity();
  float denominator = 0.0F;
  std::vector<float> numerator(head_size, 0.0F);

  for (std::uint32_t absolute_key_position = first_key_position;;
       ++absolute_key_position) {
    const bool current = absolute_key_position >= base_position;
    const std::uint32_t chunk_key_position =
        current ? absolute_key_position - base_position : 0;
    const std::uint32_t cache_key_position =
        local ? absolute_key_position % gemma4_31b::kLocalWindowSize
              : absolute_key_position;
    const std::size_t key_row =
        current
            ? (static_cast<std::size_t>(kv_head) * token_count +
               chunk_key_position) *
                  head_size
            : (static_cast<std::size_t>(kv_head) * cache_capacity +
               cache_key_position) *
                  head_size;
    const std::size_t value_row =
        current
            ? (static_cast<std::size_t>(chunk_key_position) * kv_heads +
               kv_head) *
                  head_size
            : (static_cast<std::size_t>(kv_head) * cache_capacity +
               cache_key_position) *
                  head_size;
    const std::vector<BFloat16>& key =
        current ? current_key_head_major : key_cache;
    const std::vector<BFloat16>& value =
        current ? current_value_token_major : value_cache;

    std::array<float, kRuntimeAttentionThreads> partial{};
    for (unsigned lane = 0; lane < kRuntimeAttentionThreads; ++lane) {
      for (std::uint32_t dimension = lane; dimension < head_size;
           dimension += kRuntimeAttentionThreads) {
        partial[lane] =
            std::fma(to_float(query_head_major[query_row + dimension]),
                     to_float(key[key_row + dimension]), partial[lane]);
      }
    }
    for (unsigned offset = kRuntimeAttentionThreads / 2; offset != 0;
         offset /= 2) {
      for (unsigned lane = 0; lane < offset; ++lane) {
        partial[lane] += partial[lane + offset];
      }
    }
    const float score = partial[0];
    float old_weight;
    float new_weight;
    if (score <= maximum) {
      old_weight = 1.0F;
      new_weight = std::exp(score - maximum);
      denominator += new_weight;
    } else {
      old_weight = std::exp(maximum - score);
      new_weight = 1.0F;
      denominator = std::fma(denominator, old_weight, new_weight);
      maximum = score;
    }
    for (std::uint32_t dimension = 0; dimension < head_size; ++dimension) {
      numerator[dimension] =
          std::fma(new_weight, to_float(value[value_row + dimension]),
                   numerator[dimension] * old_weight);
    }
    if (absolute_key_position == absolute_query_position) {
      break;
    }
  }

  std::vector<BFloat16> result(head_size);
  for (std::uint32_t dimension = 0; dimension < head_size; ++dimension) {
    result[dimension] = to_bf16(numerator[dimension] / denominator);
  }
  return result;
}

// The matching fixture uses zero Q/K and fills V at absolute position p with
// p+1, so each context element is the mean over the visible key positions.
BFloat16 uniform_image_block_context_reference(
    std::uint32_t absolute_query_position, std::uint32_t image_begin,
    std::uint32_t image_end, bool local) {
  const bool image_query = absolute_query_position >= image_begin &&
                           absolute_query_position < image_end;
  // Transformers create_masks_for_vision_model: global = causal;
  // local = (causal OR same image) AND sliding_window_overlay(1024).
  // Enumerate the mask independently of the kernel's contiguous key bounds.
  float numerator = 0.0F;
  std::uint32_t visible = 0;
  const auto end = std::max(image_end, absolute_query_position + 1);
  for (std::uint32_t key_position = 0; key_position < end; ++key_position) {
    const bool causal = key_position <= absolute_query_position;
    const bool same_image = image_query && key_position >= image_begin &&
                            key_position < image_end;
    const bool within_window = static_cast<std::int64_t>(key_position) >
        static_cast<std::int64_t>(absolute_query_position) -
            gemma4_31b::kLocalWindowSize;
    if (!(local ? (causal || same_image) && within_window : causal)) continue;
    numerator += to_float(to_bf16(static_cast<float>(key_position + 1)));
    ++visible;
  }
  return to_bf16(numerator / static_cast<float>(visible));
}

void check_uniform_image_block_context(
    const std::vector<BFloat16>& context, std::uint32_t base_position,
    std::uint32_t token_count, std::uint32_t image_begin,
    std::uint32_t image_end, std::uint32_t head_size,
    std::string_view label) {
  const std::size_t expected_elements =
      static_cast<std::size_t>(token_count) *
      gemma4_31b::kQueryHeadCount * head_size;
  if (context.size() != expected_elements) {
    fail(label, "context size mismatch");
  }
  for (std::uint32_t chunk_query_position = 0;
       chunk_query_position < token_count; ++chunk_query_position) {
    const std::uint32_t absolute_query_position =
        base_position + chunk_query_position;
    const BFloat16 expected = uniform_image_block_context_reference(
        absolute_query_position, image_begin, image_end,
        head_size == gemma4_31b::kLocalHeadSize);
    const std::string_view region =
        absolute_query_position < image_begin
            ? "before"
            : (absolute_query_position < image_end ? "inside" : "after");
    for (std::uint32_t query_head = 0;
         query_head < gemma4_31b::kQueryHeadCount; ++query_head) {
      const std::size_t row =
          (static_cast<std::size_t>(chunk_query_position) *
               gemma4_31b::kQueryHeadCount +
           query_head) *
          head_size;
      for (std::uint32_t dimension = 0; dimension < head_size; ++dimension) {
        if (bf16_bits(context[row + dimension]) != bf16_bits(expected)) {
          fail(label, std::string(region) +
                          " image-mask mismatch at absolute query " +
                          std::to_string(absolute_query_position) +
                          " head " + std::to_string(query_head) +
                          " dimension " + std::to_string(dimension));
        }
      }
    }
  }
}

void test_image_block_cached_chunk(std::ostream& report,
                                   gemma4_31b::AttentionKind kind,
                                   std::string_view label,
                                   std::uint32_t kBasePosition = 2,
                                   std::uint32_t kChunkTokens = 5,
                                   std::uint32_t kImageBegin = 3,
                                   std::uint32_t kImageEnd = 5) {
  const bool global = kind == gemma4_31b::AttentionKind::global;
  const std::uint32_t head_size = global ? gemma4_31b::kGlobalHeadSize
                                         : gemma4_31b::kLocalHeadSize;
  const std::uint32_t kv_heads = global ? gemma4_31b::kGlobalKvHeadCount
                                        : gemma4_31b::kLocalKvHeadCount;
  const std::uint32_t cache_capacity =
      global ? kBasePosition + kChunkTokens
             : gemma4_31b::kLocalWindowSize;
  const BFloat16 zero = to_bf16(0.0F);
  const BFloat16 sentinel = to_bf16(-13.0F);

  std::vector<BFloat16> query(
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * kChunkTokens *
          head_size,
      zero);
  std::vector<BFloat16> current_key(
      static_cast<std::size_t>(kv_heads) * kChunkTokens * head_size, zero);
  std::vector<BFloat16> current_value(
      static_cast<std::size_t>(kChunkTokens) * kv_heads * head_size);
  for (std::uint32_t position = 0; position < kChunkTokens; ++position) {
    const BFloat16 value =
        to_bf16(static_cast<float>(kBasePosition + position + 1));
    for (std::uint32_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
      const std::size_t row =
          (static_cast<std::size_t>(position) * kv_heads + kv_head) *
          head_size;
      std::fill_n(current_value.begin() + row, head_size, value);
    }
  }

  const std::size_t cache_elements =
      static_cast<std::size_t>(kv_heads) * cache_capacity * head_size;
  std::vector<BFloat16> key_cache(cache_elements, zero);
  std::vector<BFloat16> value_cache(cache_elements, sentinel);
  for (std::uint32_t position = 0; position < kBasePosition; ++position) {
    const BFloat16 value = to_bf16(static_cast<float>(position + 1));
    for (std::uint32_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
      const std::size_t row =
          (static_cast<std::size_t>(kv_head) * cache_capacity +
           (global ? position : position % cache_capacity)) *
          head_size;
      std::fill_n(value_cache.begin() + row, head_size, value);
    }
  }

  DeviceBuffer<BFloat16> device_query(query.size());
  DeviceBuffer<BFloat16> device_current_key(current_key.size());
  DeviceBuffer<BFloat16> device_current_value(current_value.size());
  DeviceBuffer<BFloat16> device_key_cache(key_cache.size());
  DeviceBuffer<BFloat16> device_value_cache(value_cache.size());
  DeviceBuffer<BFloat16> device_context(query.size());
  device_query.copy_from(query);
  device_current_key.copy_from(current_key);
  device_current_value.copy_from(current_value);
  device_key_cache.copy_from(key_cache);
  device_value_cache.copy_from(value_cache);
  device_context.fill_byte(0xff);

  image_block_gqa_attention_cached_chunk(
      device_query.get(), device_current_key.get(), device_current_value.get(),
      device_key_cache.get(), device_value_cache.get(), kBasePosition,
      kChunkTokens, cache_capacity, kImageBegin, kImageEnd,
      device_context.get(), kind);
  const std::vector<BFloat16> separate_context =
      device_context.copy_to_host();
  check_uniform_image_block_context(
      separate_context, kBasePosition, kChunkTokens, kImageBegin, kImageEnd,
      head_size, std::string(label) + "_separate");
  CublasHandle handle;
  GuardedTensorScratch tensor_scratch(kChunkTokens);
  device_context.fill_byte(0xff);
  causal_gqa_attention_cached_chunk_tensor(
      handle.get(), device_query.get(), device_current_key.get(), device_current_value.get(),
      device_key_cache.get(), device_value_cache.get(), kBasePosition, kChunkTokens,
      cache_capacity, tensor_scratch.get(), device_context.get(), kind, nullptr,
      gewell::kv_cache::Format::bf16, nullptr, kImageBegin, kImageEnd);
  check_uniform_image_block_context(device_context.copy_to_host(), kBasePosition,
      kChunkTokens, kImageBegin, kImageEnd, head_size, std::string(label) + "_tensor");
  tensor_scratch.check_guards();
  require_equal(device_key_cache.copy_to_host(), key_cache,
                std::string(label) + "_key_cache_read_only");
  require_equal(device_value_cache.copy_to_host(), value_cache,
                std::string(label) + "_value_cache_read_only");

  if (global) {
    const std::vector<BFloat16> compact_cache =
        compact_global_cache(key_cache, value_cache, cache_capacity);
    std::vector<BFloat16> scale(head_size, to_bf16(1.0F));
    DeviceBuffer<BFloat16> device_compact_cache(compact_cache.size());
    DeviceBuffer<BFloat16> device_scale(scale.size());
    device_compact_cache.copy_from(compact_cache);
    device_scale.copy_from(scale);
    device_context.fill_byte(0xff);
    image_block_gqa_attention_cached_chunk_global_compact(
        device_query.get(), device_current_key.get(),
        device_current_value.get(), device_compact_cache.get(),
        device_scale.get(), kBasePosition, kChunkTokens, cache_capacity,
        kImageBegin, kImageEnd, device_context.get());
    const std::vector<BFloat16> compact_context =
        device_context.copy_to_host();
    check_uniform_image_block_context(
        compact_context, kBasePosition, kChunkTokens, kImageBegin, kImageEnd,
        head_size, std::string(label) + "_compact");
    require_equal(compact_context, separate_context,
                  std::string(label) + "_compact_vs_separate");
    require_equal(device_compact_cache.copy_to_host(), compact_cache,
                  std::string(label) + "_compact_cache_read_only");
    constexpr std::uint32_t kPageTokens = 256;
    const std::size_t page_stride_elements =
        static_cast<std::size_t>(kv_heads) * kPageTokens *
        kGlobalCompactCacheRowElements;
    const std::uint32_t page_count = (cache_capacity + kPageTokens - 1) / kPageTokens;
    std::vector<std::uint64_t> page_offsets(page_count);
    for (std::uint32_t page = 0; page < page_count; ++page)
      page_offsets[page] = std::size_t(page_count - page - 1) * page_stride_elements;
    std::vector<BFloat16> paged_cache(page_count * page_stride_elements, sentinel);
    for (std::uint32_t position = 0; position < cache_capacity; ++position) {
      for (std::uint32_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
        const std::size_t source_row =
            (static_cast<std::size_t>(kv_head) * cache_capacity + position) *
            kGlobalCompactCacheRowElements;
        const std::size_t destination_row =
            page_offsets[position / kPageTokens] +
            (static_cast<std::size_t>(kv_head) * kPageTokens + position % kPageTokens) *
            kGlobalCompactCacheRowElements;
        std::copy_n(compact_cache.begin() + source_row,
                    kGlobalCompactCacheRowElements,
                    paged_cache.begin() + destination_row);
      }
    }
    DeviceBuffer<BFloat16> device_paged_cache(paged_cache.size());
    DeviceBuffer<std::uint64_t> device_page_offsets(page_offsets.size());
    device_paged_cache.copy_from(paged_cache);
    device_page_offsets.copy_from(page_offsets);
    const CompactGlobalPagedCache paged_view{
        device_paged_cache.get(), device_page_offsets.get(), kPageTokens, page_count,
        page_stride_elements, 0};
    device_context.fill_byte(0xff);
    image_block_gqa_attention_cached_chunk_global_compact_paged(
        device_query.get(), device_current_key.get(),
        device_current_value.get(), paged_view, device_scale.get(),
        kBasePosition, kChunkTokens, kImageBegin, kImageEnd,
        device_context.get());
    require_equal(device_context.copy_to_host(), compact_context,
                  std::string(label) + "_paged_compact_exact");
    require_equal(device_paged_cache.copy_to_host(), paged_cache,
                  std::string(label) + "_paged_compact_cache_read_only");
    require_equal(device_scale.copy_to_host(), scale,
                  std::string(label) + "_scale_read_only");
    causal_gqa_attention_cached_chunk_tensor_global_compact(handle.get(), device_query.get(),
        device_current_key.get(), device_current_value.get(), device_compact_cache.get(),
        device_scale.get(), kBasePosition, kChunkTokens, cache_capacity,
        tensor_scratch.get(), device_context.get());
    check_uniform_image_block_context(device_context.copy_to_host(), kBasePosition,
        kChunkTokens, kImageBegin, kImageEnd, head_size, std::string(label) + "_tensor_compact");
    causal_gqa_attention_cached_chunk_tensor_global_compact_paged(handle.get(), device_query.get(),
        device_current_key.get(), device_current_value.get(), paged_view, device_scale.get(),
        kBasePosition, kChunkTokens, tensor_scratch.get(), device_context.get());
    check_uniform_image_block_context(device_context.copy_to_host(), kBasePosition,
        kChunkTokens, kImageBegin, kImageEnd, head_size, std::string(label) + "_tensor_paged");
    tensor_scratch.check_guards();
  }

  require_equal(device_query.copy_to_host(), query,
                std::string(label) + "_query_read_only");
  require_equal(device_current_key.copy_to_host(), current_key,
                std::string(label) + "_current_key_read_only");
  require_equal(device_current_value.copy_to_host(), current_value,
                std::string(label) + "_current_value_read_only");
  report << "prefill primitive " << label
         << ": base=" << kBasePosition << " image=[" << kImageBegin << ',' << kImageEnd
         << ") before/inside/after host_reference=exact";
  if (global) {
    report << " compact_vs_separate=exact paged_compact_vs_separate=exact";
  }
  report << '\n';
}

void test_long_local_image_block(std::ostream& report) {
  constexpr std::uint32_t kBasePosition = 0;
  constexpr std::uint32_t kChunkTokens = 1'122;
  constexpr std::uint32_t kImageBegin = 1;
  constexpr std::uint32_t kImageEnd = 1'121;
  constexpr std::uint32_t kHeadSize = gemma4_31b::kLocalHeadSize;
  constexpr std::uint32_t kKvHeads = gemma4_31b::kLocalKvHeadCount;
  constexpr std::uint32_t kCacheCapacity = gemma4_31b::kLocalWindowSize;
  const BFloat16 zero = to_bf16(0.0F);
  const BFloat16 marker = to_bf16(1'024.0F);

  std::vector<BFloat16> query(
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * kChunkTokens *
          kHeadSize,
      zero);
  std::vector<BFloat16> current_key(
      static_cast<std::size_t>(kKvHeads) * kChunkTokens * kHeadSize, zero);
  std::vector<BFloat16> current_value(
      static_cast<std::size_t>(kChunkTokens) * kKvHeads * kHeadSize, zero);
  for (std::uint32_t kv_head = 0; kv_head < kKvHeads; ++kv_head) {
    const std::size_t first_row =
        (static_cast<std::size_t>(kImageBegin) * kKvHeads + kv_head) *
        kHeadSize;
    const std::size_t last_row =
        (static_cast<std::size_t>(kImageEnd - 1) * kKvHeads + kv_head) *
        kHeadSize;
    for (std::uint32_t dimension = 0; dimension < kHeadSize; ++dimension) {
      current_value[(dimension % 2 == 0 ? first_row : last_row) + dimension] = marker;
    }
  }

  const std::size_t cache_elements =
      static_cast<std::size_t>(kKvHeads) * kCacheCapacity * kHeadSize;
  std::vector<BFloat16> key_cache(cache_elements, zero);
  std::vector<BFloat16> value_cache(cache_elements, zero);

  DeviceBuffer<BFloat16> device_query(query.size());
  DeviceBuffer<BFloat16> device_current_key(current_key.size());
  DeviceBuffer<BFloat16> device_current_value(current_value.size());
  DeviceBuffer<BFloat16> device_key_cache(key_cache.size());
  DeviceBuffer<BFloat16> device_value_cache(value_cache.size());
  DeviceBuffer<BFloat16> device_context(query.size());
  device_query.copy_from(query);
  device_current_key.copy_from(current_key);
  device_current_value.copy_from(current_value);
  device_key_cache.copy_from(key_cache);
  device_value_cache.copy_from(value_cache);

  image_block_gqa_attention_cached_chunk(
      device_query.get(), device_current_key.get(), device_current_value.get(),
      device_key_cache.get(), device_value_cache.get(), kBasePosition,
      kChunkTokens, kCacheCapacity, kImageBegin, kImageEnd,
      device_context.get(), gemma4_31b::AttentionKind::local);
  const std::size_t context_row_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * kHeadSize;
  struct ExpectedRow {
    std::uint32_t position;
    float early;
    float late;
  };
  // Nonzero values at the two image endpoints distinguish past-window expiry
  // from future image visibility. Positions are absolute, with a text prefix.
  const ExpectedRow expected_rows[] = {
      {0, 0.0F, 0.0F},
      {1, 1024.0F / 1121, 1024.0F / 1121},
      {1024, 1024.0F / 1120, 1024.0F / 1120},
      {1025, 0.0F, 1024.0F / 1119},
      {1120, 0.0F, 1.0F},
      {1121, 0.0F, 1.0F},
  };
  for (const auto& expected : expected_rows) {
    const auto actual = device_context.copy_slice(
        static_cast<std::size_t>(expected.position) * context_row_elements,
        context_row_elements);
    for (std::size_t index = 0; index < actual.size(); ++index) {
      const BFloat16 value = to_bf16(index % 2 == 0 ? expected.early : expected.late);
      if (bf16_bits(actual[index]) != bf16_bits(value)) {
        fail("long_local_image_block",
             "endpoint visibility differs at query " + std::to_string(expected.position) +
                 " element " + std::to_string(index));
      }
    }
  }

  const auto scalar_context = device_context.copy_to_host();
  CublasHandle handle;
  GuardedTensorScratch tensor_scratch(kChunkTokens);
  device_context.fill_byte(0xff);
  causal_gqa_attention_cached_chunk_tensor(handle.get(), device_query.get(),
      device_current_key.get(), device_current_value.get(), device_key_cache.get(),
      device_value_cache.get(), kBasePosition, kChunkTokens, kCacheCapacity,
      tensor_scratch.get(), device_context.get(), gemma4_31b::AttentionKind::local,
      nullptr, gewell::kv_cache::Format::bf16, nullptr, kImageBegin, kImageEnd);
  require_equal(device_context.copy_to_host(), scalar_context,
                "long_local_image_block_tensor_endpoints");
  tensor_scratch.check_guards();
  require_equal(device_key_cache.copy_to_host(), key_cache,
                "long_local_image_block_key_cache_read_only");
  require_equal(device_value_cache.copy_to_host(), value_cache,
                "long_local_image_block_value_cache_read_only");
  report << "prefill primitive long_local_image_block: image=[1,1121) "
            "past_window_boundary=exact future_image_visibility=exact "
            "text_prefix_suffix=exact\n";
}

void test_long_local_cache_commit(std::ostream& report,
                                  std::uint32_t chunk_tokens) {
  constexpr std::uint32_t kBasePosition = 37;
  const std::uint32_t kChunkTokens = chunk_tokens;
  constexpr std::uint32_t kCacheCapacity = gemma4_31b::kLocalWindowSize;
  const std::uint32_t kRetainedBegin =
      kChunkTokens - gemma4_31b::kLocalWindowSize;
  constexpr std::uint32_t kHeadSize = gemma4_31b::kLocalHeadSize;
  constexpr std::uint32_t kKvHeads = gemma4_31b::kLocalKvHeadCount;

  const std::vector<BFloat16> current_key = runtime_head_major_pattern(
      kKvHeads, kChunkTokens, kHeadSize, kBasePosition, 79,
      1.0F / 64.0F);
  const std::vector<BFloat16> current_value = runtime_token_major_pattern(
      kChunkTokens, kKvHeads, kHeadSize, kBasePosition, 83,
      1.0F / 32.0F);
  const BFloat16 sentinel = to_bf16(-13.0F);
  const std::size_t cache_elements =
      static_cast<std::size_t>(kKvHeads) * kCacheCapacity * kHeadSize;
  const std::vector<BFloat16> empty_key_cache(cache_elements, sentinel);
  const std::vector<BFloat16> empty_value_cache(cache_elements, sentinel);
  std::vector<BFloat16> expected_key_cache = empty_key_cache;
  std::vector<BFloat16> expected_value_cache = empty_value_cache;
  for (std::uint32_t position = kRetainedBegin; position < kChunkTokens;
       ++position) {
    const std::uint32_t cache_position =
        (kBasePosition + position) % kCacheCapacity;
    for (std::uint32_t kv_head = 0; kv_head < kKvHeads; ++kv_head) {
      const std::size_t cache_row =
          (static_cast<std::size_t>(kv_head) * kCacheCapacity +
           cache_position) *
          kHeadSize;
      const std::size_t key_row =
          (static_cast<std::size_t>(kv_head) * kChunkTokens + position) *
          kHeadSize;
      const std::size_t value_row =
          (static_cast<std::size_t>(position) * kKvHeads + kv_head) *
          kHeadSize;
      std::copy_n(current_key.begin() + key_row, kHeadSize,
                  expected_key_cache.begin() + cache_row);
      std::copy_n(current_value.begin() + value_row, kHeadSize,
                  expected_value_cache.begin() + cache_row);
    }
  }

  DeviceBuffer<BFloat16> device_current_key(current_key.size());
  DeviceBuffer<BFloat16> device_current_value(current_value.size());
  DeviceBuffer<BFloat16> device_key_cache(empty_key_cache.size());
  DeviceBuffer<BFloat16> device_value_cache(empty_value_cache.size());
  device_current_key.copy_from(current_key);
  device_current_value.copy_from(current_value);
  for (int trial = 0; trial < 3; ++trial) {
    device_key_cache.copy_from(empty_key_cache);
    device_value_cache.copy_from(empty_value_cache);
    write_kv_cache_chunk(
        device_current_key.get(), device_current_value.get(),
        device_key_cache.get(), device_value_cache.get(), kBasePosition,
        kChunkTokens, kCacheCapacity, gemma4_31b::AttentionKind::local);
    require_equal(device_key_cache.copy_to_host(), expected_key_cache,
                  "long_local_cache_commit_key");
    require_equal(device_value_cache.copy_to_host(), expected_value_cache,
                  "long_local_cache_commit_value");

  }
  require_equal(device_current_key.copy_to_host(), current_key,
                "long_local_cache_commit_key_input_read_only");
  require_equal(device_current_value.copy_to_host(), current_value,
                "long_local_cache_commit_value_input_read_only");
  report << "prefill primitive long_local_cache_commit: M=" << kChunkTokens
         << " base=37 retained=[" << kRetainedBegin << ',' << kChunkTokens
         << ") separate K/V exact over 3 trials\n";
}

void test_runtime_cached_chunk(std::ostream& report,
                               gemma4_31b::AttentionKind kind,
                               std::uint32_t base_position,
                               std::uint32_t token_count,
                               std::uint32_t cache_capacity,
                               std::string_view label, float query_gain = 1.0F,
                               bool check_mask_isolation = false) {
  const bool local = kind == gemma4_31b::AttentionKind::local;
  const std::uint32_t head_size = local ? gemma4_31b::kLocalHeadSize
                                        : gemma4_31b::kGlobalHeadSize;
  const std::uint32_t kv_heads = local ? gemma4_31b::kLocalKvHeadCount
                                       : gemma4_31b::kGlobalKvHeadCount;
  const std::vector<BFloat16> query = runtime_head_major_pattern(
      gemma4_31b::kQueryHeadCount, token_count, head_size, base_position, 37,
      query_gain / 256.0F);
  const std::vector<BFloat16> current_key = runtime_head_major_pattern(
      kv_heads, token_count, head_size, base_position, 41, 1.0F / 256.0F);
  const std::vector<BFloat16> current_value = runtime_token_major_pattern(
      token_count, kv_heads, head_size, base_position, 43, 1.0F / 32.0F);
  const BFloat16 sentinel = to_bf16(-13.0F);
  const std::size_t cache_elements =
      static_cast<std::size_t>(kv_heads) * cache_capacity * head_size;
  std::vector<BFloat16> key_cache(cache_elements, sentinel);
  std::vector<BFloat16> value_cache(cache_elements, sentinel);
  const std::uint32_t first_history_position =
      local && base_position > gemma4_31b::kLocalWindowSize
          ? base_position - gemma4_31b::kLocalWindowSize
          : 0;
  for (std::uint32_t absolute_position = first_history_position;
       absolute_position < base_position; ++absolute_position) {
    const std::uint32_t cache_position =
        local ? absolute_position % gemma4_31b::kLocalWindowSize
              : absolute_position;
    for (std::uint32_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
      const std::size_t row =
          (static_cast<std::size_t>(kv_head) * cache_capacity +
           cache_position) *
          head_size;
      for (std::uint32_t dimension = 0; dimension < head_size; ++dimension) {
        key_cache[row + dimension] = runtime_pattern_value(
            absolute_position, kv_head, dimension, 41, 1.0F / 256.0F);
        value_cache[row + dimension] = runtime_pattern_value(
            absolute_position, kv_head, dimension, 43, 1.0F / 32.0F);
      }
    }
  }

  DeviceBuffer<BFloat16> device_query(query.size());
  DeviceBuffer<BFloat16> device_current_key(current_key.size());
  DeviceBuffer<BFloat16> device_current_value(current_value.size());
  DeviceBuffer<BFloat16> device_key_cache(cache_elements);
  DeviceBuffer<BFloat16> device_value_cache(cache_elements);
  const std::size_t context_elements =
      static_cast<std::size_t>(token_count) *
      gemma4_31b::kQueryHeadCount * head_size;
  GuardedContext device_context(context_elements);
  device_query.copy_from(query);
  device_current_key.copy_from(current_key);
  device_current_value.copy_from(current_value);
  device_key_cache.copy_from(key_cache);
  device_value_cache.copy_from(value_cache);
  device_context.fill_byte(0xff);

  causal_gqa_attention_cached_chunk(
      device_query.get(), device_current_key.get(), device_current_value.get(),
      device_key_cache.get(), device_value_cache.get(), base_position,
      token_count, cache_capacity, device_context.get(), kind);
  const std::vector<BFloat16> context = device_context.copy_to_host();
  if (token_count <= 32) {
    // A singleton image at the final row has the ordinary causal mask. This
    // entry point retains the original serial kernel, independently checking
    // the short-suffix kernel's exact arithmetic and ring-window behavior.
    image_block_gqa_attention_cached_chunk(
        device_query.get(), device_current_key.get(), device_current_value.get(),
        device_key_cache.get(), device_value_cache.get(), base_position,
        token_count, cache_capacity, base_position + token_count - 1,
        base_position + token_count, device_context.get(), kind);
    require_equal(device_context.copy_to_host(), context,
                  std::string(label) + "_serial_arithmetic_exact");
  }
  require_equal(device_key_cache.copy_to_host(), key_cache,
                std::string(label) + "_attention_key_cache_read_only");
  require_equal(device_value_cache.copy_to_host(), value_cache,
                std::string(label) + "_attention_value_cache_read_only");

  std::vector<BFloat16> tensor_context;
  if (token_count <= kTensorAttentionMaximumQueryRows) {
    GuardedTensorScratch tensor_scratch(token_count);
    CublasHandle tensor_handle;
    device_context.fill_byte(0xff);
    const auto run_tensor = [&] {
      causal_gqa_attention_cached_chunk_tensor(
          tensor_handle.get(), device_query.get(), device_current_key.get(),
          device_current_value.get(), device_key_cache.get(),
          device_value_cache.get(), base_position, token_count, cache_capacity,
          tensor_scratch.get(), device_context.get(), kind);
    };
    run_tensor();
    tensor_context = device_context.copy_to_host();
    tensor_scratch.check_guards();
    for (std::size_t index = 0; index < tensor_context.size(); ++index) {
      if (!std::isfinite(to_float(tensor_context[index]))) {
        fail(std::string(label) + "_tensor_context_finite",
             "non-finite value at element " + std::to_string(index));
      }
    }
    require_close(tensor_context, context, 1.0F / 64.0F,
                  std::string(label) + "_tensor_vs_warp_full");
#ifdef GEWELL_TEST_FUSED_PREFILL
    device_context.fill_byte(0xff);
    experiments::fused_prefill::attention(
        device_query.get(), device_current_key.get(),
        device_current_value.get(), device_key_cache.get(),
        device_value_cache.get(), base_position, token_count, cache_capacity,
        device_context.get(), kind);
    require_close(device_context.copy_to_host(), context, 1.0F / 64.0F,
                  std::string(label) + "_fused_vs_warp_full");
#endif
    require_equal(device_key_cache.copy_to_host(), key_cache,
                  std::string(label) + "_tensor_key_cache_read_only");
    require_equal(device_value_cache.copy_to_host(), value_cache,
                  std::string(label) + "_tensor_value_cache_read_only");
    if (check_mask_isolation) {
      // Cross the first query-tile boundary, including one row of the next
      // slice. Future finite K/V changes must leave this entire prefix exact.
      constexpr std::uint32_t kPrefixRows = 257;
      if (token_count <= kPrefixRows || base_position < kTokenCount)
        fail(label, "mask isolation requires a wide chunk with history");
      auto changed_key = current_key;
      auto changed_value = current_value;
      for (std::uint32_t row = kPrefixRows; row < token_count; ++row) {
        for (std::uint32_t head = 0; head < kv_heads; ++head) {
          for (std::uint32_t d = 0; d < head_size; ++d) {
            changed_key[(std::size_t(head) * token_count + row) * head_size + d] = to_bf16(7.0F);
            changed_value[(std::size_t(row) * kv_heads + head) * head_size + d] = to_bf16(-9.0F);
          }
        }
      }
      device_current_key.copy_from(changed_key);
      device_current_value.copy_from(changed_value);
      run_tensor();
      const std::size_t prefix_elements = std::size_t(kPrefixRows) *
          gemma4_31b::kQueryHeadCount * head_size;
      auto changed_context = device_context.copy_to_host();
      changed_context.resize(prefix_elements);
      require_equal(changed_context,
          std::vector<BFloat16>(tensor_context.begin(), tensor_context.begin() + prefix_elements),
          std::string(label) + "_future_isolation");
      device_current_key.copy_from(current_key);
      device_current_value.copy_from(current_value);

      if (local) {
        // The oldest retained ring row is outside every query's local window.
        auto changed_key_cache = key_cache;
        auto changed_value_cache = value_cache;
        const auto slot = (base_position - kTokenCount) % kTokenCount;
        for (std::uint32_t head = 0; head < kv_heads; ++head) {
          const auto offset = (std::size_t(head) * cache_capacity + slot) * head_size;
          std::fill_n(changed_key_cache.begin() + offset, head_size, to_bf16(11.0F));
          std::fill_n(changed_value_cache.begin() + offset, head_size, to_bf16(-15.0F));
        }
        device_key_cache.copy_from(changed_key_cache);
        device_value_cache.copy_from(changed_value_cache);
        run_tensor();
        require_equal(device_context.copy_to_host(), tensor_context,
                      std::string(label) + "_window_isolation");
        device_key_cache.copy_from(key_cache);
        device_value_cache.copy_from(value_cache);
      }
      tensor_scratch.check_guards();
    }
  }

  float maximum_error = 0.0F;
  float tensor_maximum_error = 0.0F;
  const auto query_heads = selected_query_heads(kv_heads);
  std::vector<std::uint32_t> query_positions;
  if (token_count <= 4) {
    for (std::uint32_t position = 0; position < token_count; ++position) {
      query_positions.push_back(position);
    }
  } else {
    query_positions = {0, 1, token_count / 2, token_count - 2,
                       token_count - 1};
  }
  for (const std::uint32_t chunk_query_position : query_positions) {
    for (const std::uint32_t query_head : query_heads) {
      const std::vector<BFloat16> expected = runtime_attention_row_reference(
          query, current_key, current_value, key_cache, value_cache,
          query_head, chunk_query_position, base_position, token_count,
          cache_capacity, kv_heads, head_size, kind);
      const std::size_t row =
          (static_cast<std::size_t>(chunk_query_position) *
               gemma4_31b::kQueryHeadCount +
           query_head) *
          head_size;
      maximum_error = std::max(
          maximum_error,
          require_close(
              std::vector<BFloat16>(context.begin() + row,
                                    context.begin() + row + head_size),
              expected, 1.0F / 256.0F,
              std::string(label) + "_online_context"));
      if (!tensor_context.empty()) {
        tensor_maximum_error = std::max(
            tensor_maximum_error,
            require_close(
                std::vector<BFloat16>(tensor_context.begin() + row,
                                      tensor_context.begin() + row +
                                          head_size),
                expected, 1.0F / 64.0F,
                std::string(label) + "_tensor_context"));
      }
    }
  }

  std::vector<BFloat16> expected_key_cache = key_cache;
  std::vector<BFloat16> expected_value_cache = value_cache;
  for (std::uint32_t position = 0; position < token_count; ++position) {
    const std::uint32_t absolute_position = base_position + position;
    const std::uint32_t cache_position =
        local ? absolute_position % gemma4_31b::kLocalWindowSize
              : absolute_position;
    for (std::uint32_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
      const std::size_t cache_row =
          (static_cast<std::size_t>(kv_head) * cache_capacity +
           cache_position) *
          head_size;
      const std::size_t key_row =
          (static_cast<std::size_t>(kv_head) * token_count + position) *
          head_size;
      const std::size_t value_row =
          (static_cast<std::size_t>(position) * kv_heads + kv_head) *
          head_size;
      std::copy_n(current_key.begin() + key_row, head_size,
                  expected_key_cache.begin() + cache_row);
      std::copy_n(current_value.begin() + value_row, head_size,
                  expected_value_cache.begin() + cache_row);
    }
  }
  write_kv_cache_chunk(
      device_current_key.get(), device_current_value.get(),
      device_key_cache.get(), device_value_cache.get(), base_position,
      token_count, cache_capacity, kind);
  require_equal(device_key_cache.copy_to_host(), expected_key_cache,
                std::string(label) + "_committed_key_cache");
  require_equal(device_value_cache.copy_to_host(), expected_value_cache,
                std::string(label) + "_committed_value_cache");
  require_equal(device_query.copy_to_host(), query,
                std::string(label) + "_query_read_only");
  require_equal(device_current_key.copy_to_host(), current_key,
                std::string(label) + "_current_key_read_only");
  require_equal(device_current_value.copy_to_host(), current_value,
                std::string(label) + "_current_value_read_only");
  report << "prefill primitive " << label << ": M=" << token_count
         << " base=" << base_position
         << " online_context_max_abs=" << maximum_error
         << " tensor_context_max_abs=" << tensor_maximum_error
         << " attention_before_commit=1 cache_placement=exact\n";
}

void test_runtime_compact_global_cached_chunk(
    std::ostream& report, std::uint32_t base_position,
    std::uint32_t token_count, std::uint32_t cache_capacity,
    std::string_view label, float query_gain = 1.0F) {
  constexpr std::uint32_t kHeadSize = gemma4_31b::kGlobalHeadSize;
  constexpr std::uint32_t kKvHeads = gemma4_31b::kGlobalKvHeadCount;
  const std::vector<BFloat16> query = runtime_head_major_pattern(
      gemma4_31b::kQueryHeadCount, token_count, kHeadSize, base_position, 67,
      query_gain / 256.0F);
  const std::vector<BFloat16> current_key = runtime_head_major_pattern(
      kKvHeads, token_count, kHeadSize, base_position, 71,
      1.0F / 256.0F);
  const std::vector<BFloat16> current_value = runtime_token_major_pattern(
      token_count, kKvHeads, kHeadSize, base_position, 73,
      1.0F / 32.0F);
  std::vector<BFloat16> scale(kHeadSize);
  for (std::uint32_t dimension = 0; dimension < kHeadSize; ++dimension) {
    scale[dimension] =
        to_bf16(0.5F + static_cast<float>(dimension % 11) / 16.0F);
  }

  const BFloat16 sentinel = to_bf16(-13.0F);
  const std::size_t cache_elements =
      static_cast<std::size_t>(kKvHeads) * cache_capacity * kHeadSize;
  std::vector<BFloat16> key_cache(cache_elements, sentinel);
  std::vector<BFloat16> value_cache(cache_elements, sentinel);
  for (std::uint32_t absolute_position = 0;
       absolute_position < base_position; ++absolute_position) {
    for (std::uint32_t kv_head = 0; kv_head < kKvHeads; ++kv_head) {
      const std::size_t row =
          (static_cast<std::size_t>(kv_head) * cache_capacity +
           absolute_position) *
          kHeadSize;
      for (std::uint32_t dimension = 0; dimension < kHeadSize; ++dimension) {
        key_cache[row + dimension] = runtime_pattern_value(
            absolute_position, kv_head, dimension, 71, 1.0F / 256.0F);
        value_cache[row + dimension] = runtime_pattern_value(
            absolute_position, kv_head, dimension, 73, 1.0F / 32.0F);
      }
    }
  }
  const std::vector<BFloat16> compact_cache =
      compact_global_cache(key_cache, value_cache, cache_capacity);
  constexpr std::uint32_t kPageTokens = 256;
  constexpr std::size_t kLayerOffsetElements = 37;
  constexpr std::size_t kPageTrailerElements = 19;
  const std::uint32_t page_count =
      (cache_capacity + kPageTokens - 1) / kPageTokens;
  const std::size_t page_layer_elements =
      static_cast<std::size_t>(kKvHeads) * kPageTokens *
      kGlobalCompactCacheRowElements;
  const std::size_t page_stride_elements =
      kLayerOffsetElements + page_layer_elements + kPageTrailerElements;
  std::vector<std::uint64_t> page_offsets(page_count);
  for (std::uint32_t page = 0; page < page_count; ++page) {
    page_offsets[page] =
        static_cast<std::uint64_t>(page_count - page - 1) *
        page_stride_elements;
  }
  const auto paged_cache_bytes = [&](const std::vector<BFloat16>& compact) {
    std::vector<BFloat16> paged(
        static_cast<std::size_t>(page_count) * page_stride_elements,
        sentinel);
    for (std::uint32_t absolute_position = 0;
         absolute_position < cache_capacity; ++absolute_position) {
      const std::uint32_t page = absolute_position / kPageTokens;
      const std::uint32_t page_position = absolute_position % kPageTokens;
      for (std::uint32_t kv_head = 0; kv_head < kKvHeads; ++kv_head) {
        const std::size_t source_row =
            (static_cast<std::size_t>(kv_head) * cache_capacity +
             absolute_position) *
            kGlobalCompactCacheRowElements;
        const std::size_t destination_row =
            page_offsets[page] + kLayerOffsetElements +
            (static_cast<std::size_t>(kv_head) * kPageTokens +
             page_position) *
                kGlobalCompactCacheRowElements;
        std::copy_n(compact.begin() + source_row,
                    kGlobalCompactCacheRowElements,
                    paged.begin() + destination_row);
      }
    }
    return paged;
  };
  const std::vector<BFloat16> paged_cache =
      paged_cache_bytes(compact_cache);
  std::vector<BFloat16> effective_current_key = current_key;
  reconstruct_unrotated_current_global_keys(
      &effective_current_key, current_value, scale, token_count);
  std::vector<BFloat16> effective_key_cache = key_cache;
  reconstruct_unrotated_cached_global_keys(
      &effective_key_cache, value_cache, scale, cache_capacity);

  constexpr std::array<std::uint32_t, 7> kBoundaryDimensions{
      63, 64, 255, 256, 319, 320, 511};
  for (const std::uint32_t dimension : kBoundaryDimensions) {
    const BFloat16 expected =
        is_compact_global_key_dimension_host(dimension)
            ? current_key[dimension]
            : compact_global_reconstructed_key_host(current_value[dimension],
                                                    scale[dimension]);
    if (bf16_bits(effective_current_key[dimension]) !=
        bf16_bits(expected)) {
      fail(std::string(label) + "_boundary_reconstruction",
           "key mismatch at dimension " + std::to_string(dimension));
    }
  }

  DeviceBuffer<BFloat16> device_query(query.size());
  DeviceBuffer<BFloat16> device_current_key(current_key.size());
  DeviceBuffer<BFloat16> device_effective_current_key(
      effective_current_key.size());
  DeviceBuffer<BFloat16> device_current_value(current_value.size());
  DeviceBuffer<BFloat16> device_effective_key_cache(
      effective_key_cache.size());
  DeviceBuffer<BFloat16> device_value_cache(value_cache.size());
  DeviceBuffer<BFloat16> device_compact_cache(compact_cache.size());
  DeviceBuffer<BFloat16> device_paged_cache(paged_cache.size());
  DeviceBuffer<std::uint64_t> device_page_offsets(page_offsets.size());
  DeviceBuffer<BFloat16> device_scale(scale.size());
  const std::size_t context_elements =
      static_cast<std::size_t>(token_count) *
      gemma4_31b::kQueryHeadCount * kHeadSize;
  GuardedContext device_context(context_elements);
  device_query.copy_from(query);
  device_current_key.copy_from(current_key);
  device_effective_current_key.copy_from(effective_current_key);
  device_current_value.copy_from(current_value);
  device_effective_key_cache.copy_from(effective_key_cache);
  device_value_cache.copy_from(value_cache);
  device_compact_cache.copy_from(compact_cache);
  device_paged_cache.copy_from(paged_cache);
  device_page_offsets.copy_from(page_offsets);
  device_scale.copy_from(scale);
  const CompactGlobalPagedCache device_paged_view{
      device_paged_cache.get(), device_page_offsets.get(), kPageTokens,
      page_count, page_stride_elements, kLayerOffsetElements};

  causal_gqa_attention_cached_chunk(
      device_query.get(), device_effective_current_key.get(),
      device_current_value.get(), device_effective_key_cache.get(),
      device_value_cache.get(), base_position, token_count, cache_capacity,
      device_context.get(), gemma4_31b::AttentionKind::global);
  const std::vector<BFloat16> effective_separate_context =
      device_context.copy_to_host();
  device_context.fill_byte(0xff);
  causal_gqa_attention_cached_chunk_global_compact(
      device_query.get(), device_current_key.get(), device_current_value.get(),
      device_compact_cache.get(), device_scale.get(), base_position,
      token_count, cache_capacity, device_context.get());
  require_equal(device_context.copy_to_host(), effective_separate_context,
                std::string(label) + "_warp_exact");
  require_equal(device_compact_cache.copy_to_host(), compact_cache,
                std::string(label) + "_attention_cache_read_only");
  device_context.fill_byte(0xff);
  causal_gqa_attention_cached_chunk_global_compact_paged(
      device_query.get(), device_current_key.get(), device_current_value.get(),
      device_paged_view, device_scale.get(), base_position, token_count,
      device_context.get());
  require_equal(device_context.copy_to_host(), effective_separate_context,
                std::string(label) + "_paged_warp_exact");
  require_equal(device_paged_cache.copy_to_host(), paged_cache,
                std::string(label) + "_paged_attention_cache_read_only");

  if (token_count <= kTensorAttentionMaximumQueryRows) {
    GuardedTensorScratch tensor_scratch(token_count);
    CublasHandle tensor_handle;
    device_context.fill_byte(0xff);
    causal_gqa_attention_cached_chunk_tensor(
        tensor_handle.get(), device_query.get(),
        device_effective_current_key.get(), device_current_value.get(),
        device_effective_key_cache.get(), device_value_cache.get(),
        base_position, token_count, cache_capacity, tensor_scratch.get(),
        device_context.get(), gemma4_31b::AttentionKind::global);
    const std::vector<BFloat16> effective_separate_tensor_context =
        device_context.copy_to_host();
    tensor_scratch.check_guards();
    require_close(effective_separate_tensor_context, effective_separate_context,
                  1.0F / 64.0F, std::string(label) + "_tensor_vs_warp");
    device_context.fill_byte(0xff);
    causal_gqa_attention_cached_chunk_tensor_global_compact(
        tensor_handle.get(), device_query.get(), device_current_key.get(),
        device_current_value.get(), device_compact_cache.get(),
        device_scale.get(), base_position, token_count, cache_capacity,
        tensor_scratch.get(), device_context.get());
    require_equal(device_context.copy_to_host(),
                  effective_separate_tensor_context,
                  std::string(label) + "_tensor_exact");
    tensor_scratch.check_guards();
    device_context.fill_byte(0xff);
    causal_gqa_attention_cached_chunk_tensor_global_compact_paged(
        tensor_handle.get(), device_query.get(), device_current_key.get(),
        device_current_value.get(), device_paged_view, device_scale.get(),
        base_position, token_count, tensor_scratch.get(), device_context.get());
    require_equal(device_context.copy_to_host(),
                  effective_separate_tensor_context,
                  std::string(label) + "_paged_tensor_exact");
    tensor_scratch.check_guards();
#ifdef GEWELL_TEST_FUSED_PREFILL
    device_context.fill_byte(0xff);
    experiments::fused_prefill::attention(
        device_query.get(), device_effective_current_key.get(),
        device_current_value.get(), device_effective_key_cache.get(),
        device_value_cache.get(), base_position, token_count, cache_capacity,
        device_context.get(), gemma4_31b::AttentionKind::global);
    const auto fused_separate = device_context.copy_to_host();
    require_close(fused_separate, effective_separate_context, 1.0F / 64.0F,
                  std::string(label) + "_fused_vs_warp");
    device_context.fill_byte(0xff);
    experiments::fused_prefill::attention_global_compact(
        device_query.get(), device_current_key.get(),
        device_current_value.get(), device_compact_cache.get(),
        device_scale.get(), base_position, token_count, cache_capacity,
        device_context.get());
    require_equal(device_context.copy_to_host(), fused_separate,
                  std::string(label) + "_fused_compact_exact");
    device_context.fill_byte(0xff);
    experiments::fused_prefill::attention_global_compact_paged(
        device_query.get(), device_current_key.get(),
        device_current_value.get(), device_paged_view, device_scale.get(),
        base_position, token_count, device_context.get());
    require_equal(device_context.copy_to_host(), fused_separate,
                  std::string(label) + "_fused_unaligned_paged_exact");
#endif
    require_equal(device_compact_cache.copy_to_host(), compact_cache,
                  std::string(label) + "_tensor_cache_read_only");
    require_equal(device_paged_cache.copy_to_host(), paged_cache,
                  std::string(label) + "_paged_tensor_cache_read_only");
  }

  std::vector<BFloat16> expected_key_cache = key_cache;
  std::vector<BFloat16> expected_value_cache = value_cache;
  for (std::uint32_t position = 0; position < token_count; ++position) {
    const std::uint32_t absolute_position = base_position + position;
    for (std::uint32_t kv_head = 0; kv_head < kKvHeads; ++kv_head) {
      const std::size_t cache_row =
          (static_cast<std::size_t>(kv_head) * cache_capacity +
           absolute_position) *
          kHeadSize;
      const std::size_t key_row =
          (static_cast<std::size_t>(kv_head) * token_count + position) *
          kHeadSize;
      const std::size_t value_row =
          (static_cast<std::size_t>(position) * kKvHeads + kv_head) *
          kHeadSize;
      std::copy_n(current_key.begin() + key_row, kHeadSize,
                  expected_key_cache.begin() + cache_row);
      std::copy_n(current_value.begin() + value_row, kHeadSize,
                  expected_value_cache.begin() + cache_row);
    }
  }
  const std::vector<BFloat16> expected_compact_cache = compact_global_cache(
      expected_key_cache, expected_value_cache, cache_capacity);
  write_kv_cache_chunk_global_compact(
      device_current_key.get(), device_current_value.get(),
      device_compact_cache.get(), base_position, token_count, cache_capacity);
  const std::vector<BFloat16> committed_compact_cache =
      device_compact_cache.copy_to_host();
  require_equal(committed_compact_cache, expected_compact_cache,
                std::string(label) + "_cache_placement");
  write_kv_cache_chunk_global_compact_paged(
      device_current_key.get(), device_current_value.get(), device_paged_view,
      base_position, token_count);
  require_equal(device_paged_cache.copy_to_host(),
                paged_cache_bytes(expected_compact_cache),
                std::string(label) + "_paged_cache_placement");
  const std::size_t committed_row =
      static_cast<std::size_t>(base_position) *
      kGlobalCompactCacheRowElements;
  for (const std::uint32_t dimension : kBoundaryDimensions) {
    const BFloat16 actual_value =
        committed_compact_cache[
            committed_row + kGlobalCompactRotatedKeyElements + dimension];
    if (bf16_bits(actual_value) != bf16_bits(current_value[dimension])) {
      fail(std::string(label) + "_boundary_value_layout",
           "value mismatch at dimension " + std::to_string(dimension));
    }
    if (is_compact_global_key_dimension_host(dimension)) {
      const BFloat16 actual_key = committed_compact_cache[
          committed_row + compact_global_key_index_host(dimension)];
      if (bf16_bits(actual_key) != bf16_bits(current_key[dimension])) {
        fail(std::string(label) + "_boundary_key_layout",
             "key mismatch at dimension " + std::to_string(dimension));
      }
    }
  }
  require_equal(device_query.copy_to_host(), query,
                std::string(label) + "_query_read_only");
  require_equal(device_current_key.copy_to_host(), current_key,
                std::string(label) + "_current_key_read_only");
  require_equal(device_current_value.copy_to_host(), current_value,
                std::string(label) + "_current_value_read_only");
  require_equal(device_scale.copy_to_host(), scale,
                std::string(label) + "_scale_read_only");
  report << "prefill primitive " << label << ": M=" << token_count
         << " base=" << base_position
         << " warp_vs_effective_separate=exact "
            "paged_warp=exact tensor_vs_effective_separate=exact "
            "paged_tensor=exact cache_placement=exact paged_placement=exact "
            "boundary_dims=63,64,255,256,319,320,511\n";
}

void test_invalid_arguments(std::ostream& report) {
  auto* pointer = reinterpret_cast<BFloat16*>(static_cast<std::uintptr_t>(1));
  auto* handle = reinterpret_cast<cublasHandle_t>(
      static_cast<std::uintptr_t>(1));
  const auto invalid_kind = static_cast<gemma4_31b::AttentionKind>(255);
  require_failure("generate_rope_factors_m1024_null", [&] {
    generate_rope_factors_m1024(nullptr, pointer, pointer, pointer);
  });
  require_failure("apply_rope_transpose_m1024_heads", [&] {
    apply_rope_transpose_m1024(pointer, pointer, pointer, pointer, 3,
                               gemma4_31b::AttentionKind::local);
  });
  require_failure("apply_rope_transpose_m1024_kind", [&] {
    apply_rope_transpose_m1024(pointer, pointer, pointer, pointer,
                               gemma4_31b::kQueryHeadCount, invalid_kind);
  });
  require_failure("write_kv_cache_m1024_local_capacity", [&] {
    write_kv_cache_m1024(pointer, pointer, pointer, pointer, kTokenCount - 1,
                         gemma4_31b::AttentionKind::local);
  });
  require_failure("write_kv_cache_m1024_global_capacity", [&] {
    write_kv_cache_m1024(pointer, pointer, pointer, pointer,
                         kGlobalCacheMinimumCapacity - 1,
                         gemma4_31b::AttentionKind::global);
  });
  require_failure("attention_m1024_null", [&] {
    causal_gqa_attention_m1024(nullptr, pointer, pointer, pointer, pointer,
                               pointer,
                               gemma4_31b::AttentionKind::local);
  });
  require_failure("attention_m1024_kind", [&] {
    causal_gqa_attention_m1024(pointer, pointer, pointer, pointer, pointer,
                               pointer, invalid_kind);
  });
  require_failure("generate_rope_factors_chunk_zero", [&] {
    generate_rope_factors_chunk(pointer, pointer, pointer, pointer, 0, 0);
  });
  require_failure("generate_rope_factors_chunk_too_large", [&] {
    generate_rope_factors_chunk(pointer, pointer, pointer, pointer, 0,
                                kMaxChunkTokenCount + 1);
  });
  require_failure("generate_rope_factors_chunk_overflow", [&] {
    generate_rope_factors_chunk(
        pointer, pointer, pointer, pointer,
        std::numeric_limits<std::uint32_t>::max(), 2);
  });
  require_failure("apply_rope_transpose_chunk_zero", [&] {
    apply_rope_transpose_chunk(
        pointer, pointer, pointer, pointer, gemma4_31b::kQueryHeadCount, 0,
        gemma4_31b::AttentionKind::local);
  });
  require_failure("apply_rope_transpose_chunk_heads", [&] {
    apply_rope_transpose_chunk(pointer, pointer, pointer, pointer, 3, 1,
                               gemma4_31b::AttentionKind::local);
  });
  require_failure("attention_chunk_local_capacity", [&] {
    causal_gqa_attention_cached_chunk(
        pointer, pointer, pointer, pointer, pointer, 1'023, 3,
        kTokenCount - 1, pointer, gemma4_31b::AttentionKind::local);
  });
  require_failure("attention_chunk_global_capacity", [&] {
    causal_gqa_attention_cached_chunk(
        pointer, pointer, pointer, pointer, pointer, 5, 3, 7, pointer,
        gemma4_31b::AttentionKind::global);
  });
  require_failure("image_attention_chunk_empty_span", [&] {
    image_block_gqa_attention_cached_chunk(
        pointer, pointer, pointer, pointer, pointer, 5, 3, kTokenCount, 6,
        6, pointer, gemma4_31b::AttentionKind::local);
  });
  require_failure("image_attention_chunk_outside_span", [&] {
    image_block_gqa_attention_cached_chunk(
        pointer, pointer, pointer, pointer, pointer, 5, 3, 8, 4, 7,
        pointer, gemma4_31b::AttentionKind::global);
  });
  require_failure("image_attention_compact_global_oversized_span", [&] {
    image_block_gqa_attention_cached_chunk_global_compact(
        pointer, pointer, pointer, pointer, pointer, 0,
        gemma4_31b::kVisionMaxSoftTokenCount + 1,
        gemma4_31b::kVisionMaxSoftTokenCount + 1, 0,
        gemma4_31b::kVisionMaxSoftTokenCount + 1, pointer);
  });
  const CompactGlobalPagedCache paged_cache{
      pointer, reinterpret_cast<std::uint64_t*>(pointer), 256, 1,
      static_cast<std::size_t>(gemma4_31b::kGlobalKvHeadCount) * 256 *
          kGlobalCompactCacheRowElements,
      0};
  require_failure("image_attention_paged_compact_global_outside_span", [&] {
    image_block_gqa_attention_cached_chunk_global_compact_paged(
        pointer, pointer, pointer, paged_cache, pointer, 5, 3, 4, 7,
        pointer);
  });
  require_failure("tensor_attention_null_handle", [&] {
    causal_gqa_attention_cached_chunk_tensor(
        nullptr, pointer, pointer, pointer, pointer, pointer, 0,
        kTokenCount, kTokenCount, pointer, pointer,
        gemma4_31b::AttentionKind::local);
  });
  require_failure("tensor_attention_exclusive_end_overflow", [&] {
    causal_gqa_attention_cached_chunk_tensor(
        handle, pointer, pointer, pointer, pointer, pointer,
        std::numeric_limits<std::uint32_t>::max() - (kTokenCount - 1),
        kTokenCount, kTokenCount, pointer, pointer,
        gemma4_31b::AttentionKind::local);
  });
  const CompactGlobalPagedCache tensor_paged_cache{
      pointer, reinterpret_cast<std::uint64_t*>(pointer), 256, 16,
      static_cast<std::size_t>(gemma4_31b::kGlobalKvHeadCount) * 256 *
          kGlobalCompactCacheRowElements,
      0};
  for (const std::uint32_t rows : {0U, kTensorAttentionMaximumQueryRows + 1}) {
    require_failure("tensor_attention_invalid_rows_" + std::to_string(rows), [&] {
      causal_gqa_attention_cached_chunk_tensor(
          handle, pointer, pointer, pointer, pointer, pointer, 0, rows,
          kTokenCount, pointer, pointer, gemma4_31b::AttentionKind::local);
    });
    require_failure("tensor_compact_global_invalid_rows_" + std::to_string(rows), [&] {
      causal_gqa_attention_cached_chunk_tensor_global_compact(
          handle, pointer, pointer, pointer, pointer, pointer, 0, rows,
          4096, pointer, pointer);
    });
    require_failure("tensor_paged_compact_global_invalid_rows_" + std::to_string(rows), [&] {
      causal_gqa_attention_cached_chunk_tensor_global_compact_paged(
          handle, pointer, pointer, pointer, tensor_paged_cache, pointer,
          0, rows, pointer, pointer);
    });
  }
  require_failure("write_kv_cache_chunk_overflow", [&] {
    write_kv_cache_chunk(
        pointer, pointer, pointer, pointer,
        std::numeric_limits<std::uint32_t>::max(), 2, kTokenCount,
        gemma4_31b::AttentionKind::local);
  });
  require_failure("write_kv_cache_chunk_kind", [&] {
    write_kv_cache_chunk(pointer, pointer, pointer, pointer, 0, 1,
                         kTokenCount, invalid_kind);
  });
  report << "prefill primitive invalid_arguments: null/head/kind/capacity "
            "chunk/range guards=ok\n";
}

bool run_tests(std::ostream& report, std::string* failure) {
  try {
    test_invalid_arguments(report);
    test_rope_factors(report);
    test_kind(report, gemma4_31b::AttentionKind::local);
    test_kind(report, gemma4_31b::AttentionKind::global);
    test_runtime_chunk_rope(report);
    test_image_block_cached_chunk(
        report, gemma4_31b::AttentionKind::local,
        "image_block_chunk_local");
    test_image_block_cached_chunk(
        report, gemma4_31b::AttentionKind::global,
        "image_block_chunk_global_compact");
    test_long_local_image_block(report);
    test_image_block_cached_chunk(report, gemma4_31b::AttentionKind::local,
        "image_local_280", 511, 282, 512, 792);
    // A later image spans more than the local window after a wrapped prefix.
    // Its boundaries also cross global pages with reversed physical placement.
    test_image_block_cached_chunk(report, gemma4_31b::AttentionKind::local,
        "later_long_image_local_wrap", 2'303, 1'122, 2'304, 3'424);
    test_image_block_cached_chunk(report, gemma4_31b::AttentionKind::global,
        "later_long_image_paged_global", 2'303, 1'122, 2'304, 3'424);
    for (const std::uint32_t rows : {1280U, 2048U, 2049U, 4096U}) {
      test_long_local_cache_commit(report, rows);
    }
    test_runtime_cached_chunk(
        report, gemma4_31b::AttentionKind::local, 1'023, 3, kTokenCount,
        "runtime_chunk_local_wrap");
    test_runtime_cached_chunk(
        report, gemma4_31b::AttentionKind::global, 5, 3, 10,
        "runtime_chunk_global_history");
    test_runtime_cached_chunk(
        report, gemma4_31b::AttentionKind::local, 0, kTokenCount,
        kTokenCount, "runtime_chunk_local_full_base0");
    test_runtime_cached_chunk(
        report, gemma4_31b::AttentionKind::global, 0, kTokenCount,
        kTokenCount, "runtime_chunk_global_full_base0");
    test_runtime_cached_chunk(
        report, gemma4_31b::AttentionKind::local, kTokenCount, kTokenCount,
        kTokenCount, "runtime_chunk_local_full_base1024");
    test_runtime_cached_chunk(
        report, gemma4_31b::AttentionKind::global, kTokenCount, kTokenCount,
        2 * kTokenCount, "runtime_chunk_global_full_base1024");
    test_runtime_compact_global_cached_chunk(
        report, 5, 3, 10, "runtime_chunk_compact_global_history");
    test_runtime_compact_global_cached_chunk(
        report, kTokenCount, kTokenCount, 2 * kTokenCount,
        "runtime_chunk_compact_global_full_base1024");
    for (const std::uint32_t rows : {1U, 3U, 17U, 31U, 32U}) {
      for (const std::uint32_t base : {15U, 16U, 543U, 1023U, 1024U}) {
        test_runtime_cached_chunk(report, gemma4_31b::AttentionKind::local,
            base, rows, kTokenCount, "runtime_chunk_short_local");
        test_runtime_cached_chunk(report, gemma4_31b::AttentionKind::global,
            base, rows, base + rows, "runtime_chunk_short_global");
        test_runtime_compact_global_cached_chunk(report, base, rows, base + rows,
            "runtime_chunk_short_compact_global");
      }
    }
    for (const std::uint32_t rows : {1U, 17U, 33U, 501U, 509U, 973U, 1024U}) {
      constexpr std::uint32_t kLongBase = 20'477;
      test_runtime_cached_chunk(
          report, gemma4_31b::AttentionKind::local, kLongBase, rows,
          kTokenCount, "runtime_chunk_local_variable_wrap");
      test_runtime_compact_global_cached_chunk(
          report, kLongBase, rows, kLongBase + rows,
          "runtime_chunk_compact_global_variable_long_history");
    }
    for (const std::uint32_t rows : {33U, 34U, 39U, 40U, 41U, 255U, 256U, 257U,
                                    511U, 512U, 513U, 767U, 768U, 769U, 973U, 1023U}) {
      test_runtime_cached_chunk(
          report, gemma4_31b::AttentionKind::local, 2045, rows,
          kTokenCount, "runtime_chunk_local_query_slices", 1.0F, rows == 769 || rows == 973);
    }
    for (const std::uint32_t rows : {767U, 768U, 769U, 973U, 1023U, 1024U}) {
      test_runtime_cached_chunk(
          report, gemma4_31b::AttentionKind::global, 2045, rows,
          2045 + rows, "runtime_chunk_global_query_slices", 1.0F, rows >= 768);
      test_runtime_compact_global_cached_chunk(
          report, 2045, rows, 2045 + rows, "runtime_chunk_compact_global_query_slices");
    }
    for (const std::uint32_t rows : {2048U, 2049U, 4096U}) {
      for (const std::uint32_t base : {0U, 2045U}) {
        test_runtime_cached_chunk(
            report, gemma4_31b::AttentionKind::local, base, rows,
            kTokenCount, "runtime_chunk_local_wide");
        test_runtime_compact_global_cached_chunk(
            report, base, rows, base + rows,
            "runtime_chunk_compact_global_wide");
      }
    }
    test_runtime_cached_chunk(report, gemma4_31b::AttentionKind::local,
        std::numeric_limits<std::uint32_t>::max() - 32, 32, 1024,
        "cached_local_near_uint32_limit");
    test_runtime_cached_chunk(report, gemma4_31b::AttentionKind::local,
        1023, 33, 1024, "cached_local_peaked_softmax", 64.0F);
    test_runtime_cached_chunk(report, gemma4_31b::AttentionKind::local,
        1023, 769, 1024, "cached_local_sliced_peaked_softmax", 64.0F);
    test_runtime_cached_chunk(report, gemma4_31b::AttentionKind::local,
        std::numeric_limits<std::uint32_t>::max() - 769, 769, 1024,
        "cached_local_sliced_near_uint32_limit");
    test_runtime_cached_chunk(report, gemma4_31b::AttentionKind::global,
        259, 33, 512, "cached_global_peaked_softmax", 64.0F);
    test_runtime_cached_chunk(report, gemma4_31b::AttentionKind::global,
        2045, 33, 2078, "modular_global_peaked_multitile", 64.0F);
    test_runtime_cached_chunk(report, gemma4_31b::AttentionKind::global,
        2045, 768, 2813, "modular_global_sliced_peaked_multitile", 64.0F);
    test_runtime_cached_chunk(report, gemma4_31b::AttentionKind::global,
        2045, 973, 3018, "modular_global_ragged_sliced_peaked_multitile", 64.0F, true);
    test_runtime_compact_global_cached_chunk(report, 259, 33, 512,
        "cached_compact_peaked_softmax", 64.0F);
    test_runtime_compact_global_cached_chunk(report, 65'503, 33, 65'536,
        "modular_compact_global_64k_history");
    report << "BF16 prefill primitive self-test: fixed M=1024 and modular cached M=1..4096 ok\n";
    return true;
  } catch (const std::exception& error) {
    if (failure != nullptr) {
      *failure = error.what();
    }
    return false;
  }
}

}  // namespace
}  // namespace gewell::prefill_primitives

int main() {
  std::string failure;
  if (!gewell::prefill_primitives::run_tests(std::cout, &failure)) {
    std::cerr << "BF16 prefill primitive self-test failed: " << failure
              << '\n';
    return 1;
  }
  return 0;
}
