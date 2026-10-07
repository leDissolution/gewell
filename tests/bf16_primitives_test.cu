#include "gewell/bf16_primitives.h"
#include "gewell/prefill_primitives.h"
#include "../src/rope_inverse_frequency.cuh"
#include "../src/models/gemma4/31b/sm120/kernels/bf16_attention_detail.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <mma.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <iostream>

namespace gewell::bf16_primitives {
namespace {
using namespace detail;

constexpr float kEmbeddingScaleBf16 = 73.5F;
constexpr float kRmsNormEpsilon = 1.0e-6F;
constexpr float kGeluCoefficient = 0.7978845608028654F;
constexpr float kGeluCubicCoefficient = 0.044715F;

void check_cublas(cublasStatus_t status, std::string_view operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    fail(operation, "cuBLASLt status " + std::to_string(status));
  }
}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t elements) : elements_(elements) {
    if (elements == 0) {
      fail("DeviceBuffer", "zero-sized allocation");
    }
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer_),
                          elements * sizeof(T)),
               "primitive self-test cudaMalloc");
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
               "primitive self-test host-to-device copy");
  }

  std::vector<T> copy_to_host() const {
    std::vector<T> values(elements_);
    check_cuda(cudaMemcpy(values.data(), pointer_, elements_ * sizeof(T),
                          cudaMemcpyDeviceToHost),
               "primitive self-test device-to-host copy");
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

struct Metric {
  std::size_t elements{};
  std::size_t bit_mismatches{};
  double max_abs{};
  double rmse{};
};

Metric measure(const std::vector<BFloat16>& actual,
               const std::vector<BFloat16>& expected) {
  if (actual.size() != expected.size() || actual.empty()) {
    fail("primitive metric", "invalid vector sizes");
  }
  Metric metric{};
  metric.elements = actual.size();
  double squared_error = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    metric.bit_mismatches +=
        bf16_bits(actual[index]) == bf16_bits(expected[index]) ? 0 : 1;
    const double difference = static_cast<double>(to_float(actual[index])) -
                              static_cast<double>(to_float(expected[index]));
    metric.max_abs = std::max(metric.max_abs, std::abs(difference));
    squared_error += difference * difference;
  }
  metric.rmse = std::sqrt(squared_error / static_cast<double>(metric.elements));
  return metric;
}

void report_metric(std::ostream& report, std::string_view name,
                   const Metric& metric, double absolute_tolerance,
                   bool require_exact) {
  report << "primitive " << name << ": elements=" << metric.elements
         << " bit_mismatches=" << metric.bit_mismatches
         << " max_abs=" << std::setprecision(9) << metric.max_abs
         << " rmse=" << metric.rmse << " tolerance=";
  if (require_exact) {
    report << "bit-exact";
  } else {
    report << "atol=" << absolute_tolerance;
  }
  report << '\n';

  if ((require_exact && metric.bit_mismatches != 0) ||
      (!require_exact && metric.max_abs > absolute_tolerance)) {
    fail(name, "CUDA result exceeds the CPU reference tolerance");
  }
}

std::vector<BFloat16> pattern(std::size_t elements, int modulus, int offset,
                              float scale) {
  std::vector<BFloat16> values(elements);
  for (std::size_t index = 0; index < elements; ++index) {
    const int integer = static_cast<int>(index % modulus) - offset;
    values[index] = to_bf16(static_cast<float>(integer) * scale);
  }
  return values;
}

void test_linear(std::ostream& report) {
  constexpr std::uint32_t kInput = 16;
  constexpr std::uint32_t kOutput = 8;
  const std::vector<BFloat16> input = pattern(kInput, 15, 7, 0.125F);
  std::vector<BFloat16> weight(kInput * kOutput);
  for (std::uint32_t row = 0; row < kOutput; ++row) {
    for (std::uint32_t column = 0; column < kInput; ++column) {
      const int integer = static_cast<int>((row * 5 + column * 3) % 11) - 5;
      weight[row * kInput + column] =
          to_bf16(static_cast<float>(integer) * 0.0625F);
    }
  }
  std::vector<BFloat16> expected(kOutput);
  for (std::uint32_t row = 0; row < kOutput; ++row) {
    float sum = 0.0F;
    for (std::uint32_t column = 0; column < kInput; ++column) {
      sum += to_float(input[column]) *
             to_float(weight[row * kInput + column]);
    }
    expected[row] = to_bf16(sum);
  }

  DeviceBuffer<BFloat16> device_input(input.size());
  DeviceBuffer<BFloat16> device_weight(weight.size());
  DeviceBuffer<BFloat16> device_output(expected.size());
  device_input.copy_from(input);
  device_weight.copy_from(weight);

  cublasLtHandle_t handle = nullptr;
  check_cublas(cublasLtCreate(&handle), "primitive self-test cublasLtCreate");
  try {
    linear_m1(handle, device_input.get(), device_weight.get(),
              device_output.get(), kInput, kOutput);
  } catch (...) {
    cublasLtDestroy(handle);
    throw;
  }
  check_cublas(cublasLtDestroy(handle),
               "primitive self-test cublasLtDestroy");
  report_metric(report, "linear_m1", measure(device_output.copy_to_host(), expected),
                0.0, true);
}

void test_embedding(std::ostream& report) {
  constexpr std::uint32_t kRows = 3;
  constexpr std::uint32_t kToken = 2;
  const std::size_t elements =
      static_cast<std::size_t>(kRows) * gemma4_31b::kHiddenSize;
  const std::vector<BFloat16> table = pattern(elements, 127, 63, 1.0F / 128.0F);
  std::vector<BFloat16> expected(gemma4_31b::kHiddenSize);
  const std::size_t row_offset =
      static_cast<std::size_t>(kToken) * gemma4_31b::kHiddenSize;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    expected[index] =
        to_bf16(to_float(table[row_offset + index]) * kEmbeddingScaleBf16);
  }

  DeviceBuffer<BFloat16> device_table(table.size());
  DeviceBuffer<BFloat16> device_output(expected.size());
  device_table.copy_from(table);
  embedding_lookup(device_table.get(), kToken, device_output.get());
  report_metric(report, "embedding_lookup",
                measure(device_output.copy_to_host(), expected), 0.0, true);

  for (const unsigned rows : {1u, 17u, 1024u, 1025u, 4096u}) {
    const auto width = gemma4_31b::kHiddenSize;
    std::vector<std::uint32_t> tokens(rows);
    std::vector<BFloat16> reference(std::size_t(rows + 2) * width, to_bf16(-17));
    DeviceBuffer<BFloat16> output(reference.size());
    output.copy_from(reference);
    for (unsigned row = 0; row < rows; ++row) {
      tokens[row] = (row * 7 + 2) % kRows;
      for (unsigned d = 0; d < width; ++d)
        reference[std::size_t(row + 1) * width + d] = to_bf16(
            to_float(table[std::size_t(tokens[row]) * width + d]) * kEmbeddingScaleBf16);
    }
    embedding_lookup_host_tokens(device_table.get(), tokens.data(), output.get() + width, rows);
    std::fill(tokens.begin(), tokens.end(), 0); // The asynchronous call owns its IDs.
    report_metric(report, "host_embedding_rows_" + std::to_string(rows),
                  measure(output.copy_to_host(), reference), 0.0, true);
    tokens.back() = gemma4_31b::kVocabSize;
    bool rejected = false;
    try { embedding_lookup_host_tokens(device_table.get(), tokens.data(), output.get() + width, rows); }
    catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) fail("host embedding", "accepted invalid final token");
    report_metric(report, "host_embedding_invalid_unchanged", measure(output.copy_to_host(), reference), 0.0, true);
  }
}

std::vector<BFloat16> rms_reference(const std::vector<BFloat16>& input,
                                    const std::vector<BFloat16>* weight,
                                    std::uint32_t rows, std::uint32_t width,
                                    float epsilon) {
  std::vector<BFloat16> expected(input.size());
  for (std::uint32_t row = 0; row < rows; ++row) {
    const std::size_t row_offset = static_cast<std::size_t>(row) * width;
    float sum = 0.0F;
    for (std::uint32_t index = 0; index < width; ++index) {
      const float converted = to_float(input[row_offset + index]);
      sum += converted * converted;
    }
    const float inverse_rms =
        std::pow(sum / static_cast<float>(width) + epsilon, -0.5F);
    for (std::uint32_t index = 0; index < width; ++index) {
      float value = to_float(input[row_offset + index]) * inverse_rms;
      if (weight != nullptr) {
        value *= to_float((*weight)[index]);
      }
      expected[row_offset + index] = to_bf16(value);
    }
  }
  return expected;
}

void test_rms_width(std::ostream& report, std::uint32_t rows,
                    std::uint32_t width, bool scaled,
                    std::string_view label) {
  const std::vector<BFloat16> input =
      pattern(static_cast<std::size_t>(rows) * width, 31, 15,
              1.0F / 16.0F);
  const std::vector<BFloat16> weight = pattern(width, 9, 2, 0.25F);
  const std::vector<BFloat16> expected =
      rms_reference(input, scaled ? &weight : nullptr, rows, width, 1.0e-6F);

  DeviceBuffer<BFloat16> device_input(input.size());
  DeviceBuffer<BFloat16> device_output(expected.size());
  device_input.copy_from(input);
  if (scaled) {
    DeviceBuffer<BFloat16> device_weight(weight.size());
    device_weight.copy_from(weight);
    rms_norm(device_input.get(), device_weight.get(), device_output.get(), rows,
             width);
    // Keep the weight allocation alive until the asynchronous norm is copied
    // back below.
    report_metric(report, label,
                  measure(device_output.copy_to_host(), expected), 1.0 / 64.0,
                  false);
    return;
  }
  rms_norm_unscaled(device_input.get(), device_output.get(), rows, width);
  report_metric(report, label, measure(device_output.copy_to_host(), expected),
                1.0 / 64.0, false);
}

void test_rms_norm(std::ostream& report) {
  test_rms_width(report, gemma4_31b::kVisionHeadCount,
                 gemma4_31b::kVisionHeadSize, true,
                 "rms_norm_vision_q_16x72");
  test_rms_width(report, 1, gemma4_31b::kVisionHiddenSize, true,
                 "rms_norm_vision_hidden_1x1152");
  test_rms_width(report, gemma4_31b::kQueryHeadCount,
                 gemma4_31b::kLocalHeadSize, true, "rms_norm_q_32x256");
  test_rms_width(report, gemma4_31b::kGlobalKvHeadCount,
                 gemma4_31b::kGlobalHeadSize, true, "rms_norm_k_4x512");
  test_rms_width(report, 1, gemma4_31b::kHiddenSize, true,
                 "rms_norm_hidden_1x5376");
  test_rms_width(report, gemma4_31b::kLocalKvHeadCount,
                 gemma4_31b::kLocalHeadSize, false,
                 "rms_norm_unscaled_v_16x256");
  test_rms_width(report, gemma4_31b::kGlobalKvHeadCount,
                 gemma4_31b::kGlobalHeadSize, false,
                 "rms_norm_unscaled_v_4x512");
  test_rms_width(report, gemma4_31b::kVisionHeadCount,
                 gemma4_31b::kVisionHeadSize, false,
                 "rms_norm_unscaled_vision_v_16x72");
  test_rms_width(report, 1, gemma4_31b::kVisionHiddenSize, false,
                 "rms_norm_unscaled_vision_bridge_1x1152");
}

void test_qkv_norm_fusion(std::ostream& report) {
  for (const bool global : {false, true}) {
    const unsigned d = global ? 512 : 256, heads = global ? 4 : 16;
    const unsigned qw = 32 * d, kw = heads * d, width = qw + (global ? kw : 2 * kw);
    for (const unsigned rows : {1u, 3u, 32u, 128u}) {
      auto joined = pattern(std::size_t(rows) * width, 97, 48, 0.0625F);
      // Vary heads and tokens, including a zero head and an outlier head.
      std::fill(joined.begin(), joined.begin() + d, to_bf16(0));
      joined[d + 17] = to_bf16(512.0F);
      std::vector<BFloat16> q(rows * qw), k(rows * kw), v(rows * kw);
      for (unsigned row = 0; row < rows; ++row) {
        std::copy_n(joined.data() + row * width, qw, q.data() + row * qw);
        std::copy_n(joined.data() + row * width + qw, kw, k.data() + row * kw);
        std::copy_n(joined.data() + row * width + qw + (global ? 0 : kw), kw, v.data() + row * kw);
      }
      DeviceBuffer<BFloat16> input(joined.size()), qi(q.size()), ki(k.size()), vi(v.size());
      DeviceBuffer<BFloat16> qwgt(d), kwgt(d), qo(q.size()), ko(k.size()), vo(v.size());
      DeviceBuffer<BFloat16> qr(q.size()), kr(k.size()), vr(v.size());
      input.copy_from(joined); qi.copy_from(q); ki.copy_from(k); vi.copy_from(v);
      qwgt.copy_from(pattern(d, 17, 5, 0.125F));
      kwgt.copy_from(pattern(d, 13, 3, 0.25F));
      rms_norm(qi.get(), qwgt.get(), qr.get(), rows * 32, d);
      rms_norm(ki.get(), kwgt.get(), kr.get(), rows * heads, d);
      rms_norm_unscaled(vi.get(), vr.get(), rows * heads, d);
      cudaStream_t stream;
      check_cuda(cudaStreamCreate(&stream), "QKV test stream");
      cudaGraph_t graph;
      cudaGraphExec_t exec;
      check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "QKV capture");
      qkv_rms_norm(input.get(), qwgt.get(), kwgt.get(), qo.get(), ko.get(), vo.get(),
                    rows, global ? gemma4_31b::AttentionKind::global : gemma4_31b::AttentionKind::local, stream);
      check_cuda(cudaStreamEndCapture(stream, &graph), "QKV capture end");
      check_cuda(cudaGraphInstantiate(&exec, graph, 0), "QKV instantiate");
      for (unsigned replay = 0; replay < 2; ++replay)
        check_cuda(cudaGraphLaunch(exec, stream), "QKV replay");
      check_cuda(cudaStreamSynchronize(stream), "QKV synchronize");
      report_metric(report, "fused_q_norm", measure(qo.copy_to_host(), qr.copy_to_host()), 0, true);
      report_metric(report, "fused_k_norm", measure(ko.copy_to_host(), kr.copy_to_host()), 0, true);
      report_metric(report, "fused_v_norm", measure(vo.copy_to_host(), vr.copy_to_host()), 0, true);
      cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
    }
  }
}

void test_qkv_rope_batch(std::ostream& report) {
  std::vector<unsigned> ragged;
  for (unsigned i = 0; i < 37; ++i)
    ragged.push_back(i == 0 ? 33 : i == 36 ? 1280 : 1 + i % 9);
  const std::vector<std::vector<unsigned>> cases{
      {1}, {4}, {31}, {32}, {33}, {1280}, {31, 32}, {32, 32},
      std::vector<unsigned>(32, 4), ragged};
  for (const bool global : {false, true}) {
    for (const auto& counts : cases) {
      const auto kind = global ? gemma4_31b::AttentionKind::global : gemma4_31b::AttentionKind::local;
      const unsigned d = global ? 512 : 256, heads = global ? 4 : 16;
      const unsigned qw = 32 * d, kw = heads * d, width = qw + (global ? kw : 2 * kw);
      unsigned rows = 0;
      for (const auto count : counts) rows += count;
      report << "QKV/RoPE kind=" << (global ? "global" : "local")
             << " requests=" << counts.size() << " total_rows=" << rows << '\n';
      DeviceBuffer<BFloat16> input(std::size_t(rows) * width), cosine(rows * d), sine(rows * d);
      DeviceBuffer<BFloat16> qwgt(d), kwgt(d), qn(rows * qw), kn(rows * kw);
      // One guard on each side of every request, with only BF16 alignment.
      const std::vector<BFloat16> qguards(rows * qw + 2 * counts.size(), to_bf16(-17));
      const std::vector<BFloat16> kvguards(rows * kw + 2 * counts.size(), to_bf16(-19));
      DeviceBuffer<BFloat16> qr(qguards.size()), kr(kvguards.size()), vr(kvguards.size());
      DeviceBuffer<BFloat16> qo(qguards.size()), ko(kvguards.size()), vo(kvguards.size());
      std::vector<QkvRopeInput> inputs;
      unsigned first = 0;
      for (const auto count : counts) {
        const auto guard_offset = 2 * inputs.size() + 1;
        inputs.push_back({input.get() + first * width, cosine.get() + first * d,
            sine.get() + first * d, qo.get() + first * qw + guard_offset,
            ko.get() + first * kw + guard_offset, vo.get() + first * kw + guard_offset, count});
        first += count;
      }
      cudaStream_t stream;
      cudaGraph_t graph;
      cudaGraphExec_t exec;
      check_cuda(cudaStreamCreate(&stream), "batched QKV stream");
      check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "batched QKV capture");
      qkv_rms_rope_batch(inputs, qwgt.get(), kwgt.get(), kind, stream);
      check_cuda(cudaStreamEndCapture(stream, &graph), "batched QKV capture end");
      check_cuda(cudaGraphInstantiate(&exec, graph, 0), "batched QKV instantiate");
      for (unsigned replay = 0; replay < 2; ++replay) {
        auto raw = pattern(std::size_t(rows) * width, 97 + 2 * replay, 48, 0.0625F);
        for (std::size_t i = 0; i < raw.size(); i += 17)
          raw[i] = to_bf16(std::ldexp(to_float(raw[i]), int((i / 17) % 21) - 10));
        std::fill(raw.begin(), raw.begin() + d, to_bf16(0));
        raw[d + 17] = to_bf16(512.0F);
        input.copy_from(raw);
        cosine.copy_from(pattern(rows * d, 31 + 2 * replay, 15, 0.0625F));
        sine.copy_from(pattern(rows * d, 29 + 2 * replay, 14, 0.0625F));
        qwgt.copy_from(pattern(d, 17 + 2 * replay, 5, 0.125F));
        kwgt.copy_from(pattern(d, 13 + 2 * replay, 3, 0.25F));
        qo.copy_from(qguards); ko.copy_from(kvguards); vo.copy_from(kvguards);
        qr.copy_from(qguards); kr.copy_from(kvguards); vr.copy_from(kvguards);
        first = 0;
        for (unsigned request = 0; request < counts.size(); ++request) {
          const auto count = counts[request], guard_offset = 2 * request + 1;
          qkv_rms_norm(input.get() + first * width, qwgt.get(), kwgt.get(),
              qn.get() + first * qw, kn.get() + first * kw,
              vr.get() + first * kw + guard_offset, count, kind);
          prefill_primitives::apply_rope_transpose_chunk(qn.get() + first * qw,
              cosine.get() + first * d, sine.get() + first * d,
              qr.get() + first * qw + guard_offset, 32, count, kind);
          prefill_primitives::apply_rope_transpose_chunk(kn.get() + first * kw,
              cosine.get() + first * d, sine.get() + first * d,
              kr.get() + first * kw + guard_offset, heads, count, kind);
          first += count;
        }
        check_cuda(cudaDeviceSynchronize(), "batched QKV reference");
        check_cuda(cudaGraphLaunch(exec, stream), "batched QKV replay");
        check_cuda(cudaStreamSynchronize(stream), "batched QKV synchronize");
        report_metric(report, "batched_q_rope", measure(qo.copy_to_host(), qr.copy_to_host()), 0, true);
        report_metric(report, "batched_k_rope", measure(ko.copy_to_host(), kr.copy_to_host()), 0, true);
        report_metric(report, "batched_v_norm", measure(vo.copy_to_host(), vr.copy_to_host()), 0, true);
        report_metric(report, "batched_qkv_input_unchanged", measure(input.copy_to_host(), raw), 0, true);
      }
      inputs.back().rows = 0;
      bool rejected = false;
      try { qkv_rms_rope_batch(inputs, qwgt.get(), kwgt.get(), kind, stream); }
      catch (const std::runtime_error&) { rejected = true; }
      if (!rejected) fail("batched QKV", "accepted invalid last request");
      cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
    }
  }
}

void test_prefill_norm_fusions(std::ostream& report) {
  constexpr auto width = gemma4_31b::kHiddenSize;
  const auto compare = [&](const std::string& label, const std::vector<BFloat16>& actual,
                           const std::vector<BFloat16>& expected) {
    unsigned maximum_steps = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
      const float a = to_float(actual[i]), b = to_float(expected[i]);
      if (!std::isfinite(a) || !std::isfinite(b)) fail(label, "nonfinite normalization");
      const auto ordered = [](BFloat16 value) {
        const auto bits = bf16_bits(value);
        return (bits & 0x8000) ? 0x8000 - int(bits & 0x7fff) : 0x8000 + int(bits);
      };
      const auto steps = unsigned(std::abs(ordered(actual[i]) - ordered(expected[i])));
      if ((i < width || i >= actual.size() - width) && bf16_bits(actual[i]) != bf16_bits(expected[i]))
        fail(label, "guard changed");
      if (std::abs(a - b) > 1.0e-6F) {
        maximum_steps = std::max(maximum_steps, steps);
        if (steps > 2) fail(label, "exceeds two BF16 steps versus separate prefill norms");
      }
    }
    const auto metric = measure(actual, expected);
    report << "primitive " << label << ": max_bf16_steps=" << maximum_steps
           << " bit_mismatches=" << metric.bit_mismatches << " max_abs=" << metric.max_abs
           << " rmse=" << metric.rmse << " guards=exact\n";
  };
  for (const unsigned rows : {1u, 33u, 1024u, 4096u}) {
    const auto elements = std::size_t(rows + 2) * width;
    auto input = pattern(elements, 61, 30, 1.0F / 32.0F);
    for (std::size_t i = width; i < elements - width; i += 17)
      input[i] = to_bf16(std::ldexp(to_float(input[i]), int((i / 17) % 21) - 10));
    std::fill(input.begin() + width, input.begin() + 2 * width, to_bf16(0));
    DeviceBuffer<BFloat16> branch(elements), residual(elements), expected_branch(elements);
    DeviceBuffer<BFloat16> normalized(elements), expected_norm(elements), temporary(elements);
    DeviceBuffer<BFloat16> post(width), next(width), scalar(1);
    residual.copy_from(pattern(elements, 43, 21, 1.0F / 32.0F));
    post.copy_from(pattern(width, 13, 6, 1.0F / 8.0F));
    next.copy_from(pattern(width, 11, 5, 1.0F / 8.0F));
    scalar.copy_from({to_bf16(0.8125F)});
    for (const bool feedforward : {false, true}) {
      branch.copy_from(input);
      expected_branch.copy_from(input);
      const std::vector<BFloat16> guards(elements, to_bf16(-17));
      normalized.copy_from(guards);
      expected_norm.copy_from(guards);
      rms_norm(branch.get() + width, post.get(), temporary.get() + width, rows, width);
      residual_add(residual.get() + width, temporary.get() + width,
                   expected_branch.get() + width, std::size_t(rows) * width);
      if (feedforward)
        trained_scalar(expected_branch.get() + width, scalar.get(), std::size_t(rows) * width);
      rms_norm(expected_branch.get() + width, next.get(), expected_norm.get() + width, rows, width);
      check_cuda(cudaDeviceSynchronize(), "prefill norm reference");
      cudaStream_t stream;
      cudaGraph_t graph;
      cudaGraphExec_t exec;
      check_cuda(cudaStreamCreate(&stream), "prefill norm stream");
      check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "prefill norm capture");
      if (feedforward)
        post_feedforward_residual_scalar_next_norm_prefill(branch.get() + width, post.get(),
            residual.get() + width, scalar.get(), next.get(), normalized.get() + width, rows, stream);
      else
        post_attention_residual_pre_feedforward_norm_prefill(branch.get() + width, post.get(),
            residual.get() + width, next.get(), normalized.get() + width, rows, stream);
      check_cuda(cudaStreamEndCapture(stream, &graph), "prefill norm capture end");
      check_cuda(cudaGraphInstantiate(&exec, graph, 0), "prefill norm instantiate");
      check_cuda(cudaGraphLaunch(exec, stream), "prefill norm replay");
      check_cuda(cudaStreamSynchronize(stream), "prefill norm synchronize");
      const auto label = std::string(feedforward ? "prefill_ff_" : "prefill_attn_") + std::to_string(rows);
      compare(label + "_state", branch.copy_to_host(), expected_branch.copy_to_host());
      compare(label + "_normalized", normalized.copy_to_host(), expected_norm.copy_to_host());
      cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
      if (feedforward) {
        branch.copy_from(input);
        normalized.copy_from(guards);
        post_feedforward_residual_scalar_next_norm_prefill(branch.get() + width, post.get(),
            residual.get() + width, scalar.get(), nullptr, nullptr, rows);
        compare(label + "_state_only", branch.copy_to_host(), expected_branch.copy_to_host());
        report_metric(report, label + "_unused_norm", measure(normalized.copy_to_host(), guards), 0, true);
      }
    }
  }
}

void test_hidden_norm_fusions(std::ostream& report) {
  constexpr std::size_t kElements = gemma4_31b::kHiddenSize;
  const std::vector<BFloat16> branch =
      pattern(kElements, 61, 30, 1.0F / 32.0F);
  const std::vector<BFloat16> residual =
      pattern(kElements, 43, 21, 1.0F / 32.0F);
  const std::vector<BFloat16> post_weight =
      pattern(kElements, 13, 6, 1.0F / 8.0F);
  const std::vector<BFloat16> next_weight =
      pattern(kElements, 11, 5, 1.0F / 8.0F);
  const std::vector<BFloat16> scalar{to_bf16(0.8125F)};

  const std::vector<BFloat16> expected_hidden =
      rms_reference(branch, &post_weight, 1, gemma4_31b::kHiddenSize,
                    kRmsNormEpsilon);
  DeviceBuffer<BFloat16> device_branch(kElements);
  DeviceBuffer<BFloat16> device_residual(kElements);
  DeviceBuffer<BFloat16> device_post_weight(kElements);
  DeviceBuffer<BFloat16> device_next_weight(kElements);
  DeviceBuffer<BFloat16> device_scalar(1);
  DeviceBuffer<BFloat16> device_normalized(kElements);
  device_branch.copy_from(branch);
  device_residual.copy_from(residual);
  device_post_weight.copy_from(post_weight);
  device_next_weight.copy_from(next_weight);
  device_scalar.copy_from(scalar);

  rms_norm_hidden_m1(device_branch.get(), device_post_weight.get(),
                     device_normalized.get());
  report_metric(report, "rms_norm_hidden_m1_1x5376",
                measure(device_normalized.copy_to_host(), expected_hidden),
                1.0 / 64.0, false);

  std::vector<BFloat16> expected_attention_state(kElements);
  for (std::size_t index = 0; index < kElements; ++index) {
    expected_attention_state[index] =
        to_bf16(to_float(residual[index]) + to_float(expected_hidden[index]));
  }
  const std::vector<BFloat16> expected_pre_feedforward = rms_reference(
      expected_attention_state, &next_weight, 1, gemma4_31b::kHiddenSize,
      kRmsNormEpsilon);
  device_branch.copy_from(branch);
  post_attention_residual_pre_feedforward_norm_m1(
      device_branch.get(), device_post_weight.get(), device_residual.get(),
      device_next_weight.get(), device_normalized.get());
  report_metric(report, "post_attention_residual_state_m1",
                measure(device_branch.copy_to_host(),
                        expected_attention_state),
                1.0 / 64.0, false);
  report_metric(report, "pre_feedforward_norm_m1",
                measure(device_normalized.copy_to_host(),
                        expected_pre_feedforward),
                1.0 / 64.0, false);

  std::vector<BFloat16> expected_feedforward_state(kElements);
  for (std::size_t index = 0; index < kElements; ++index) {
    const BFloat16 summed =
        to_bf16(to_float(residual[index]) + to_float(expected_hidden[index]));
    expected_feedforward_state[index] =
        to_bf16(to_float(summed) * to_float(scalar[0]));
  }
  const std::vector<BFloat16> expected_next_norm = rms_reference(
      expected_feedforward_state, &next_weight, 1,
      gemma4_31b::kHiddenSize, kRmsNormEpsilon);
  device_branch.copy_from(branch);
  post_feedforward_residual_scalar_next_norm_m1(
      device_branch.get(), device_post_weight.get(), device_residual.get(),
      device_scalar.get(), device_next_weight.get(), device_normalized.get());
  report_metric(report, "post_feedforward_residual_scalar_state_m1",
                measure(device_branch.copy_to_host(),
                        expected_feedforward_state),
                1.0 / 64.0, false);
  report_metric(report, "next_input_norm_m1",
                measure(device_normalized.copy_to_host(), expected_next_norm),
                1.0 / 64.0, false);
}

void test_batched_decode_rows(std::ostream& report) {
  constexpr std::size_t kWidth = gemma4_31b::kHiddenSize;
  const auto post_weight = pattern(kWidth, 13, 6, 1.0F / 8.0F);
  const auto next_weight = pattern(kWidth, 11, 5, 1.0F / 8.0F);
  const std::vector<BFloat16> scalar{to_bf16(0.8125F)};
  DeviceBuffer<BFloat16> device_post_weight(kWidth);
  DeviceBuffer<BFloat16> device_next_weight(kWidth);
  DeviceBuffer<BFloat16> device_scalar(1);
  device_post_weight.copy_from(post_weight);
  device_next_weight.copy_from(next_weight);
  device_scalar.copy_from(scalar);

  const auto rejects = [](auto operation) {
    try {
      operation();
    } catch (const std::runtime_error&) {
      return;
    }
    fail("batched decode contract", "invalid call was not rejected");
  };
  for (const std::uint32_t rows : {1U, 2U, 3U, 8U}) {
    const std::size_t elements = rows * kWidth;
    auto branch = pattern(elements, 61, 30, 1.0F / 32.0F);
    const auto residual = pattern(elements, 43, 21, 1.0F / 32.0F);
    // Include a zero row and, when present, a small row where epsilon matters.
    std::fill(branch.begin(), branch.begin() + kWidth, to_bf16(0.0F));
    if (rows > 1) {
      for (std::size_t i = kWidth; i < 2 * kWidth; ++i)
        branch[i] = to_bf16(to_float(branch[i]) * 0.001F);
    }
    DeviceBuffer<BFloat16> device_branch(elements);
    DeviceBuffer<BFloat16> device_residual(elements);
    DeviceBuffer<BFloat16> device_output(elements);
    DeviceBuffer<BFloat16> reference_branch(elements);
    DeviceBuffer<BFloat16> reference_output(elements);
    device_branch.copy_from(branch);
    device_residual.copy_from(residual);
    const auto exact = [&](std::string_view name, const auto& actual,
                           const auto& expected) {
      report_metric(report, std::string(name) + "_rows_" +
                                std::to_string(rows),
                    measure(actual, expected), 0.0, true);
    };

    rms_norm_hidden_rows(device_branch.get(), device_post_weight.get(),
                          device_output.get(), rows);
    for (std::uint32_t row = 0; row < rows; ++row)
      rms_norm_hidden_m1(device_branch.get() + row * kWidth,
                         device_post_weight.get(),
                         reference_output.get() + row * kWidth);
    exact("hidden_norm_batch_vs_m1", device_output.copy_to_host(),
          reference_output.copy_to_host());
    const auto post_reference = rms_reference(
        branch, &post_weight, rows, gemma4_31b::kHiddenSize, kRmsNormEpsilon);
    report_metric(report, "hidden_norm_batch_vs_cpu_rows_" +
                              std::to_string(rows),
                  measure(device_output.copy_to_host(), post_reference),
                  1.0 / 64.0, false);

    for (const bool feedforward : {false, true}) {
      device_branch.copy_from(branch);
      reference_branch.copy_from(branch);
      const std::string label = feedforward ? "feedforward" : "attention";
      const auto batched = [&](BFloat16* state, const BFloat16* skip,
                               BFloat16* output, std::uint32_t count) {
        if (feedforward)
          post_feedforward_residual_scalar_next_norm_rows(
              state, device_post_weight.get(), skip, device_scalar.get(),
              device_next_weight.get(), output, count);
        else
          post_attention_residual_pre_feedforward_norm_rows(
              state, device_post_weight.get(), skip, device_next_weight.get(),
              output, count);
      };
      batched(device_branch.get(), device_residual.get(), device_output.get(),
              rows);
      for (std::uint32_t row = 0; row < rows; ++row) {
        const std::size_t offset = row * kWidth;
        if (feedforward)
          post_feedforward_residual_scalar_next_norm_m1(
              reference_branch.get() + offset, device_post_weight.get(),
              device_residual.get() + offset, device_scalar.get(),
              device_next_weight.get(), reference_output.get() + offset);
        else
          post_attention_residual_pre_feedforward_norm_m1(
              reference_branch.get() + offset, device_post_weight.get(),
              device_residual.get() + offset, device_next_weight.get(),
              reference_output.get() + offset);
      }
      exact(label + "_state_batch_vs_m1", device_branch.copy_to_host(),
            reference_branch.copy_to_host());
      exact(label + "_norm_batch_vs_m1", device_output.copy_to_host(),
            reference_output.copy_to_host());
      exact(label + "_residual_immutable", device_residual.copy_to_host(),
            residual);
      std::vector<BFloat16> state_reference(elements);
      for (std::size_t i = 0; i < elements; ++i) {
        const auto sum = to_bf16(to_float(residual[i]) +
                                to_float(post_reference[i]));
        state_reference[i] = feedforward
            ? to_bf16(to_float(sum) * to_float(scalar[0])) : sum;
      }
      const auto norm_reference = rms_reference(
          state_reference, &next_weight, rows, gemma4_31b::kHiddenSize,
          kRmsNormEpsilon);
      report_metric(report, label + "_state_batch_vs_cpu_rows_" +
                                std::to_string(rows),
                    measure(device_branch.copy_to_host(), state_reference),
                    1.0 / 64.0, false);
      report_metric(report, label + "_norm_batch_vs_cpu_rows_" +
                                std::to_string(rows),
                    measure(device_output.copy_to_host(), norm_reference),
                    1.0 / 64.0, false);
      for (const auto count : {0U, static_cast<std::uint32_t>(INT_MAX) + 1U})
        rejects([&] { batched(device_branch.get(), device_residual.get(),
                             device_output.get(), count); });
      rejects([&] { batched(device_branch.get(), device_branch.get(),
                           device_output.get(), rows); });
      rejects([&] { batched(device_branch.get(), device_residual.get(),
                           device_branch.get(), rows); });
      rejects([&] { batched(device_branch.get(), device_residual.get(),
                           device_residual.get(), rows); });
    }

    const auto table = pattern(5 * kWidth, 127, 63, 1.0F / 128.0F);
    std::vector<std::uint32_t> tokens(rows);
    std::vector<BFloat16> expected_embedding(elements);
    for (std::uint32_t row = 0; row < rows; ++row) {
      // Repeated and non-monotonic IDs detect row/token-index confusion.
      tokens[row] = (row * 3 + 4) % 5;
      for (std::size_t i = 0; i < kWidth; ++i)
        expected_embedding[row * kWidth + i] = to_bf16(
            to_float(table[tokens[row] * kWidth + i]) * kEmbeddingScaleBf16);
    }
    DeviceBuffer<BFloat16> device_table(table.size());
    DeviceBuffer<std::uint32_t> device_tokens(rows);
    device_table.copy_from(table);
    device_tokens.copy_from(tokens);
    embedding_lookup_device_tokens(device_table.get(), device_tokens.get(),
                                    device_output.get(), rows, nullptr);
    for (std::uint32_t row = 0; row < rows; ++row)
      embedding_lookup_device_token(device_table.get(), device_tokens.get() + row,
                                    reference_output.get() + row * kWidth,
                                    nullptr);
    exact("embedding_batch_vs_m1", device_output.copy_to_host(),
          reference_output.copy_to_host());
    exact("embedding_batch_vs_cpu", device_output.copy_to_host(),
          expected_embedding);
    for (const auto count : {0U, static_cast<std::uint32_t>(INT_MAX) + 1U}) {
      rejects([&] {
        rms_norm_hidden_rows(device_branch.get(), device_post_weight.get(),
                              device_output.get(), count);
      });
      rejects([&] {
        embedding_lookup_device_tokens(device_table.get(), device_tokens.get(),
                                        device_output.get(), count, nullptr);
      });
    }
  }
  report << "primitive batched_decode_row_and_alias_contract: ok\n";
}

void test_residual_scalar(std::ostream& report) {
  constexpr std::size_t kElements = 257;
  const std::vector<BFloat16> residual =
      pattern(kElements, 29, 14, 1.0F / 16.0F);
  const std::vector<BFloat16> branch =
      pattern(kElements, 17, 8, 1.0F / 32.0F);
  const std::vector<BFloat16> scalar{to_bf16(0.8125F)};
  std::vector<BFloat16> expected(kElements);
  for (std::size_t index = 0; index < kElements; ++index) {
    const BFloat16 sum = to_bf16(to_float(residual[index]) +
                                 to_float(branch[index]));
    expected[index] = to_bf16(to_float(sum) * to_float(scalar[0]));
  }

  DeviceBuffer<BFloat16> device_residual(kElements);
  DeviceBuffer<BFloat16> device_branch(kElements);
  DeviceBuffer<BFloat16> device_scalar(1);
  DeviceBuffer<BFloat16> device_output(kElements);
  device_residual.copy_from(residual);
  device_branch.copy_from(branch);
  device_scalar.copy_from(scalar);
  residual_add(device_residual.get(), device_branch.get(), device_output.get(),
               kElements);
  trained_scalar(device_output.get(), device_scalar.get(), kElements);
  report_metric(report, "residual+trained_scalar",
                measure(device_output.copy_to_host(), expected), 0.0, true);
}

void test_value_kind(std::ostream& report, gemma4_31b::AttentionKind kind,
                     std::string_view label) {
  const bool global = kind == gemma4_31b::AttentionKind::global;
  const std::uint32_t kv_heads = global ? gemma4_31b::kGlobalKvHeadCount
                                        : gemma4_31b::kLocalKvHeadCount;
  const std::uint32_t head_size = global ? gemma4_31b::kGlobalHeadSize
                                         : gemma4_31b::kLocalHeadSize;
  const std::uint32_t repeats = gemma4_31b::kQueryHeadCount / kv_heads;
  const std::size_t compact_elements =
      static_cast<std::size_t>(kv_heads) * head_size;
  const std::size_t expanded_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * head_size;
  const std::vector<BFloat16> compact =
      pattern(compact_elements, 251, 125, 1.0F / 128.0F);
  std::vector<BFloat16> expected(expanded_elements);
  for (std::uint32_t head = 0; head < gemma4_31b::kQueryHeadCount; ++head) {
    const std::uint32_t source_head = head / repeats;
    std::copy_n(compact.begin() + source_head * head_size, head_size,
                expected.begin() + head * head_size);
  }

  DeviceBuffer<BFloat16> device_compact(compact.size());
  DeviceBuffer<BFloat16> device_expanded(expected.size());
  device_compact.copy_from(compact);
  expand_value_heads(device_compact.get(), device_expanded.get(), kind);
  report_metric(report, label,
                measure(device_expanded.copy_to_host(), expected), 0.0, true);
}

void test_value_expansion(std::ostream& report) {
  test_value_kind(report, gemma4_31b::AttentionKind::local,
                  "expand_value_heads_local");
  test_value_kind(report, gemma4_31b::AttentionKind::global,
                  "expand_value_heads_global");
}

struct RopeFactorReference {
  std::vector<BFloat16> cosine;
  std::vector<BFloat16> sine;
};

RopeFactorReference rope_factor_reference(std::uint32_t head_size,
                                          std::uint32_t rotated_frequencies,
                                          float theta,
                                          std::uint32_t positions = 2) {
  RopeFactorReference result{
      std::vector<BFloat16>(static_cast<std::size_t>(positions) * head_size),
      std::vector<BFloat16>(static_cast<std::size_t>(positions) * head_size)};
  const std::uint32_t half = head_size / 2;
  for (std::uint32_t token = 0; token < positions; ++token) {
    for (std::uint32_t dimension = 0; dimension < head_size; ++dimension) {
      const std::uint32_t frequency = dimension % half;
      float inverse_frequency = 0.0F;
      if (frequency < rotated_frequencies) {
        const float exponent =
            static_cast<float>(2 * frequency) / static_cast<float>(head_size);
        inverse_frequency = 1.0F / std::pow(theta, exponent);
      }
      const float angle = static_cast<float>(token) * inverse_frequency;
      const std::size_t index =
          static_cast<std::size_t>(token) * head_size + dimension;
      result.cosine[index] = to_bf16(std::cos(angle));
      result.sine[index] = to_bf16(std::sin(angle));
    }
  }
  return result;
}

RopeFactorReference rope_factor_row_reference(
    std::uint32_t head_size, std::uint32_t rotated_frequencies, float theta,
    std::uint32_t position) {
  RopeFactorReference result{std::vector<BFloat16>(head_size),
                             std::vector<BFloat16>(head_size)};
  const std::uint32_t half = head_size / 2;
  for (std::uint32_t dimension = 0; dimension < head_size; ++dimension) {
    const std::uint32_t frequency = dimension % half;
    float inverse_frequency = 0.0F;
    if (frequency < rotated_frequencies) {
      const float exponent =
          static_cast<float>(2 * frequency) / static_cast<float>(head_size);
      inverse_frequency = 1.0F / std::pow(theta, exponent);
    }
    const float angle = static_cast<float>(position) * inverse_frequency;
    result.cosine[dimension] = to_bf16(std::cos(angle));
    result.sine[dimension] = to_bf16(std::sin(angle));
  }
  return result;
}

void require_bf16_bits(const std::vector<BFloat16>& values,
                       std::size_t index, std::uint16_t expected,
                       std::string_view label) {
  if (index >= values.size() || bf16_bits(values[index]) != expected) {
    fail(label, "unexpected BF16 factor bits at index " +
                    std::to_string(index));
  }
}

void test_rope_factors(std::ostream& report) {
  constexpr std::size_t kLocalElements =
      2 * gemma4_31b::kLocalHeadSize;
  constexpr std::size_t kGlobalElements =
      2 * gemma4_31b::kGlobalHeadSize;
  DeviceBuffer<BFloat16> device_local_cos(kLocalElements);
  DeviceBuffer<BFloat16> device_local_sin(kLocalElements);
  DeviceBuffer<BFloat16> device_global_cos(kGlobalElements);
  DeviceBuffer<BFloat16> device_global_sin(kGlobalElements);
  generate_rope_factors_m2(
      device_local_cos.get(), device_local_sin.get(), device_global_cos.get(),
      device_global_sin.get());

  const std::vector<BFloat16> local_cos = device_local_cos.copy_to_host();
  const std::vector<BFloat16> local_sin = device_local_sin.copy_to_host();
  const std::vector<BFloat16> global_cos = device_global_cos.copy_to_host();
  const std::vector<BFloat16> global_sin = device_global_sin.copy_to_host();
  const RopeFactorReference local_reference = rope_factor_reference(
      gemma4_31b::kLocalHeadSize, gemma4_31b::kLocalHeadSize / 2, 10'000.0F);
  const RopeFactorReference global_reference =
      rope_factor_reference(gemma4_31b::kGlobalHeadSize, 64, 1'000'000.0F);
  report_metric(report, "rope_factors_m2_local_cos",
                measure(local_cos, local_reference.cosine), 0.0, true);
  report_metric(report, "rope_factors_m2_local_sin",
                measure(local_sin, local_reference.sine), 0.0, true);
  report_metric(report, "rope_factors_m2_global_cos",
                measure(global_cos, global_reference.cosine), 0.0, true);
  report_metric(report, "rope_factors_m2_global_sin",
                measure(global_sin, global_reference.sine), 0.0, true);

  // These values come from the pinned pair oracle and make the test sensitive
  // to the exact theta, exponent denominator, position, and partial-RoPE
  // boundary rather than only replaying the implementation's formula.
  require_bf16_bits(local_cos, 0, 0x3f80, "local RoPE position zero cosine");
  require_bf16_bits(local_sin, 0, 0x0000, "local RoPE position zero sine");
  require_bf16_bits(local_cos, 256, 16138,
                    "local RoPE position one first cosine");
  require_bf16_bits(local_sin, 256, 16215,
                    "local RoPE position one first sine");
  require_bf16_bits(local_sin, 256 + 127, 14561,
                    "local RoPE final frequency");
  require_bf16_bits(global_cos, 512, 16138,
                    "global RoPE position one first cosine");
  require_bf16_bits(global_sin, 512 + 63, 15625,
                    "global RoPE final rotated frequency");
  require_bf16_bits(global_cos, 512 + 64, 0x3f80,
                    "global RoPE first no-position cosine");
  require_bf16_bits(global_sin, 512 + 64, 0x0000,
                    "global RoPE first no-position sine");
  require_bf16_bits(global_sin, 512 + 256 + 63, 15625,
                    "global RoPE split-half duplicate");
  require_bf16_bits(global_sin, 512 + 256 + 64, 0x0000,
                    "global RoPE split-half no-position boundary");
  report << "primitive rope_factors_m2_contract: pos0=identity local_theta=10000 "
            "global_theta=1000000 global_rotated_dims=128 bits=ok\n";
}

void test_rope_factors_24(std::ostream& report) {
  constexpr std::size_t kLocalElements =
      kCachedAttentionM1ShortCapacity * gemma4_31b::kLocalHeadSize;
  constexpr std::size_t kGlobalElements =
      kCachedAttentionM1ShortCapacity * gemma4_31b::kGlobalHeadSize;
  DeviceBuffer<BFloat16> device_local_cos(kLocalElements);
  DeviceBuffer<BFloat16> device_local_sin(kLocalElements);
  DeviceBuffer<BFloat16> device_global_cos(kGlobalElements);
  DeviceBuffer<BFloat16> device_global_sin(kGlobalElements);
  generate_rope_factors_24(
      device_local_cos.get(), device_local_sin.get(), device_global_cos.get(),
      device_global_sin.get());

  const std::vector<BFloat16> local_cos = device_local_cos.copy_to_host();
  const std::vector<BFloat16> local_sin = device_local_sin.copy_to_host();
  const std::vector<BFloat16> global_cos = device_global_cos.copy_to_host();
  const std::vector<BFloat16> global_sin = device_global_sin.copy_to_host();
  const RopeFactorReference local_reference = rope_factor_reference(
      gemma4_31b::kLocalHeadSize, gemma4_31b::kLocalHeadSize / 2, 10'000.0F,
      kCachedAttentionM1ShortCapacity);
  const RopeFactorReference global_reference = rope_factor_reference(
      gemma4_31b::kGlobalHeadSize, 64, 1'000'000.0F,
      kCachedAttentionM1ShortCapacity);

  constexpr std::size_t kLocalM2Elements =
      2 * gemma4_31b::kLocalHeadSize;
  constexpr std::size_t kGlobalM2Elements =
      2 * gemma4_31b::kGlobalHeadSize;
  DeviceBuffer<BFloat16> device_m2_local_cos(kLocalM2Elements);
  DeviceBuffer<BFloat16> device_m2_local_sin(kLocalM2Elements);
  DeviceBuffer<BFloat16> device_m2_global_cos(kGlobalM2Elements);
  DeviceBuffer<BFloat16> device_m2_global_sin(kGlobalM2Elements);
  generate_rope_factors_m2(
      device_m2_local_cos.get(), device_m2_local_sin.get(),
      device_m2_global_cos.get(), device_m2_global_sin.get());
  report_metric(
      report, "rope_factors_24_local_cos_m2_prefix",
      measure(std::vector<BFloat16>(local_cos.begin(),
                                    local_cos.begin() + kLocalM2Elements),
              device_m2_local_cos.copy_to_host()),
      0.0, true);
  report_metric(
      report, "rope_factors_24_local_sin_m2_prefix",
      measure(std::vector<BFloat16>(local_sin.begin(),
                                    local_sin.begin() + kLocalM2Elements),
              device_m2_local_sin.copy_to_host()),
      0.0, true);
  report_metric(
      report, "rope_factors_24_global_cos_m2_prefix",
      measure(std::vector<BFloat16>(global_cos.begin(),
                                    global_cos.begin() + kGlobalM2Elements),
              device_m2_global_cos.copy_to_host()),
      0.0, true);
  report_metric(
      report, "rope_factors_24_global_sin_m2_prefix",
      measure(std::vector<BFloat16>(global_sin.begin(),
                                    global_sin.begin() + kGlobalM2Elements),
              device_m2_global_sin.copy_to_host()),
      0.0, true);

  const auto check_row = [&](std::string_view name,
                             const std::vector<BFloat16>& actual,
                             const std::vector<BFloat16>& expected,
                             std::uint32_t position,
                             std::uint32_t width) {
    const std::size_t begin = static_cast<std::size_t>(position) * width;
    report_metric(
        report,
        std::string(name) + "_position_" + std::to_string(position),
        measure(std::vector<BFloat16>(actual.begin() + begin,
                                      actual.begin() + begin + width),
                std::vector<BFloat16>(expected.begin() + begin,
                                      expected.begin() + begin + width)),
        0.0, true);
  };
  for (const std::uint32_t position : {7U, 15U, 23U}) {
    check_row("rope_factors_24_local_cos", local_cos,
              local_reference.cosine, position, gemma4_31b::kLocalHeadSize);
    check_row("rope_factors_24_local_sin", local_sin, local_reference.sine,
              position, gemma4_31b::kLocalHeadSize);
    check_row("rope_factors_24_global_cos", global_cos,
              global_reference.cosine, position,
              gemma4_31b::kGlobalHeadSize);
    check_row("rope_factors_24_global_sin", global_sin,
              global_reference.sine, position,
              gemma4_31b::kGlobalHeadSize);
  }
  report << "primitive rope_factors_24_contract: positions=0..23 "
            "m2_prefix_exact=1 host_positions=7,15,23 exact=1 ok\n";
}

void test_rope_factors_m1_boundary(std::ostream& report) {
  static_assert(kCachedAttentionM1BoundaryPositionCount == 1'026);
  DeviceBuffer<BFloat16> device_local_cos(gemma4_31b::kLocalHeadSize);
  DeviceBuffer<BFloat16> device_local_sin(gemma4_31b::kLocalHeadSize);
  DeviceBuffer<BFloat16> device_global_cos(gemma4_31b::kGlobalHeadSize);
  DeviceBuffer<BFloat16> device_global_sin(gemma4_31b::kGlobalHeadSize);

  constexpr std::size_t kLocalShortElements =
      kCachedAttentionM1ShortCapacity * gemma4_31b::kLocalHeadSize;
  constexpr std::size_t kGlobalShortElements =
      kCachedAttentionM1ShortCapacity * gemma4_31b::kGlobalHeadSize;
  DeviceBuffer<BFloat16> device_short_local_cos(kLocalShortElements);
  DeviceBuffer<BFloat16> device_short_local_sin(kLocalShortElements);
  DeviceBuffer<BFloat16> device_short_global_cos(kGlobalShortElements);
  DeviceBuffer<BFloat16> device_short_global_sin(kGlobalShortElements);
  generate_rope_factors_24(
      device_short_local_cos.get(), device_short_local_sin.get(),
      device_short_global_cos.get(), device_short_global_sin.get());
  const std::vector<BFloat16> short_local_cos =
      device_short_local_cos.copy_to_host();
  const std::vector<BFloat16> short_local_sin =
      device_short_local_sin.copy_to_host();
  const std::vector<BFloat16> short_global_cos =
      device_short_global_cos.copy_to_host();
  const std::vector<BFloat16> short_global_sin =
      device_short_global_sin.copy_to_host();

  constexpr std::size_t kLocalM2Elements =
      2 * gemma4_31b::kLocalHeadSize;
  constexpr std::size_t kGlobalM2Elements =
      2 * gemma4_31b::kGlobalHeadSize;
  DeviceBuffer<BFloat16> device_m2_local_cos(kLocalM2Elements);
  DeviceBuffer<BFloat16> device_m2_local_sin(kLocalM2Elements);
  DeviceBuffer<BFloat16> device_m2_global_cos(kGlobalM2Elements);
  DeviceBuffer<BFloat16> device_m2_global_sin(kGlobalM2Elements);
  generate_rope_factors_m2(
      device_m2_local_cos.get(), device_m2_local_sin.get(),
      device_m2_global_cos.get(), device_m2_global_sin.get());
  const std::vector<BFloat16> m2_local_cos =
      device_m2_local_cos.copy_to_host();
  const std::vector<BFloat16> m2_local_sin =
      device_m2_local_sin.copy_to_host();
  const std::vector<BFloat16> m2_global_cos =
      device_m2_global_cos.copy_to_host();
  const std::vector<BFloat16> m2_global_sin =
      device_m2_global_sin.copy_to_host();

  const RopeFactorReference local_reference = rope_factor_reference(
      gemma4_31b::kLocalHeadSize, gemma4_31b::kLocalHeadSize / 2, 10'000.0F,
      kCachedAttentionM1BoundaryPositionCount);
  const RopeFactorReference global_reference = rope_factor_reference(
      gemma4_31b::kGlobalHeadSize, 64, 1'000'000.0F,
      kCachedAttentionM1BoundaryPositionCount);
  const auto row = [](const std::vector<BFloat16>& values,
                      std::uint32_t position, std::uint32_t width) {
    const auto begin = values.begin() +
                       static_cast<std::size_t>(position) * width;
    return std::vector<BFloat16>(begin, begin + width);
  };
  const auto check = [&](std::string_view stem, std::uint32_t position,
                         const std::vector<BFloat16>& actual,
                         const std::vector<BFloat16>& expected) {
    report_metric(report,
                  std::string(stem) + "_position_" +
                      std::to_string(position),
                  measure(actual, expected), 0.0, true);
  };

  for (const std::uint32_t position : {0U, 1U, 23U, 1'022U, 1'023U,
                                       1'024U, 1'025U}) {
    generate_rope_factors_m1(
        device_local_cos.get(), device_local_sin.get(),
        device_global_cos.get(), device_global_sin.get(), position);
    const std::vector<BFloat16> local_cos =
        device_local_cos.copy_to_host();
    const std::vector<BFloat16> local_sin =
        device_local_sin.copy_to_host();
    const std::vector<BFloat16> global_cos =
        device_global_cos.copy_to_host();
    const std::vector<BFloat16> global_sin =
        device_global_sin.copy_to_host();

    if (position < kCachedAttentionM1ShortCapacity) {
      check("rope_factors_m1_local_cos_short_equivalence", position,
            local_cos,
            row(short_local_cos, position, gemma4_31b::kLocalHeadSize));
      check("rope_factors_m1_local_sin_short_equivalence", position,
            local_sin,
            row(short_local_sin, position, gemma4_31b::kLocalHeadSize));
      check("rope_factors_m1_global_cos_short_equivalence", position,
            global_cos,
            row(short_global_cos, position, gemma4_31b::kGlobalHeadSize));
      check("rope_factors_m1_global_sin_short_equivalence", position,
            global_sin,
            row(short_global_sin, position, gemma4_31b::kGlobalHeadSize));
    }
    if (position <= 1) {
      check("rope_factors_m1_local_cos_m2_equivalence", position, local_cos,
            row(m2_local_cos, position, gemma4_31b::kLocalHeadSize));
      check("rope_factors_m1_local_sin_m2_equivalence", position, local_sin,
            row(m2_local_sin, position, gemma4_31b::kLocalHeadSize));
      check("rope_factors_m1_global_cos_m2_equivalence", position, global_cos,
            row(m2_global_cos, position, gemma4_31b::kGlobalHeadSize));
      check("rope_factors_m1_global_sin_m2_equivalence", position, global_sin,
            row(m2_global_sin, position, gemma4_31b::kGlobalHeadSize));
    }
    if (position >= 1'022) {
      check("rope_factors_m1_local_cos_host", position, local_cos,
            row(local_reference.cosine, position,
                gemma4_31b::kLocalHeadSize));
      check("rope_factors_m1_local_sin_host", position, local_sin,
            row(local_reference.sine, position,
                gemma4_31b::kLocalHeadSize));
      check("rope_factors_m1_global_cos_host", position, global_cos,
            row(global_reference.cosine, position,
                gemma4_31b::kGlobalHeadSize));
      check("rope_factors_m1_global_sin_host", position, global_sin,
            row(global_reference.sine, position,
                gemma4_31b::kGlobalHeadSize));
    }
  }
  generate_rope_factors_m1(
      device_local_cos.get(), device_local_sin.get(),
      device_global_cos.get(), device_global_sin.get(),
      kMaxContextTokenCount - 1);
  const std::vector<BFloat16> max_local_cos =
      device_local_cos.copy_to_host();
  const std::vector<BFloat16> max_local_sin =
      device_local_sin.copy_to_host();
  const std::vector<BFloat16> max_global_cos =
      device_global_cos.copy_to_host();
  const std::vector<BFloat16> max_global_sin =
      device_global_sin.copy_to_host();
  const auto finite = [](BFloat16 value) {
    return std::isfinite(to_float(value));
  };
  if (!std::all_of(max_local_cos.begin(), max_local_cos.end(), finite) ||
      !std::all_of(max_local_sin.begin(), max_local_sin.end(), finite) ||
      !std::all_of(max_global_cos.begin(), max_global_cos.end(), finite) ||
      !std::all_of(max_global_sin.begin(), max_global_sin.end(), finite)) {
    fail("generate_rope_factors_m1", "max-context factor is not finite");
  }
  require_bf16_bits(max_global_cos, 64, 0x3f80,
                    "max-context global unrotated cosine");
  require_bf16_bits(max_global_sin, 64, 0x0000,
                    "max-context global unrotated sine");
  report << "primitive rope_factors_m1_boundary_contract: positions=0..262143 "
            "m2_positions=0,1 short_positions=0,1,23 "
            "host_positions=1022,1023,1024,1025 exact=1 "
            "max_position_finite=1 ok\n";
}

std::vector<BFloat16> rope_transpose_reference(
    const std::vector<BFloat16>& input,
    const RopeFactorReference& factors, std::uint32_t heads,
    std::uint32_t head_size) {
  std::vector<BFloat16> expected(input.size());
  const std::uint32_t half = head_size / 2;
  for (std::uint32_t token = 0; token < 2; ++token) {
    for (std::uint32_t head = 0; head < heads; ++head) {
      const std::size_t input_row =
          (static_cast<std::size_t>(token) * heads + head) * head_size;
      const std::size_t output_row =
          (static_cast<std::size_t>(head) * 2 + token) * head_size;
      for (std::uint32_t dimension = 0; dimension < head_size; ++dimension) {
        const std::uint32_t paired =
            dimension < half ? dimension + half : dimension - half;
        float rotated = to_float(input[input_row + paired]);
        if (dimension < half) {
          rotated = -rotated;
        }
        const std::size_t factor_index =
            static_cast<std::size_t>(token) * head_size + dimension;
        const BFloat16 direct = to_bf16(
            to_float(input[input_row + dimension]) *
            to_float(factors.cosine[factor_index]));
        const BFloat16 crossed =
            to_bf16(rotated * to_float(factors.sine[factor_index]));
        expected[output_row + dimension] =
            to_bf16(to_float(direct) + to_float(crossed));
      }
    }
  }
  return expected;
}

void test_rope_transpose_shape(std::ostream& report,
                               gemma4_31b::AttentionKind kind,
                               std::uint32_t heads,
                               std::string_view label) {
  const bool global = kind == gemma4_31b::AttentionKind::global;
  const std::uint32_t head_size = global ? gemma4_31b::kGlobalHeadSize
                                         : gemma4_31b::kLocalHeadSize;
  const std::size_t elements =
      static_cast<std::size_t>(2) * heads * head_size;
  std::vector<BFloat16> input(elements);
  for (std::uint32_t token = 0; token < 2; ++token) {
    for (std::uint32_t head = 0; head < heads; ++head) {
      for (std::uint32_t dimension = 0; dimension < head_size; ++dimension) {
        const int integer = static_cast<int>(
                                (token * 37 + head * 19 + dimension * 7) %
                                101) -
                            50;
        const std::size_t index =
            (static_cast<std::size_t>(token) * heads + head) * head_size +
            dimension;
        input[index] = to_bf16(static_cast<float>(integer) / 32.0F);
      }
    }
  }
  const RopeFactorReference factors =
      global ? rope_factor_reference(head_size, 64, 1'000'000.0F)
             : rope_factor_reference(head_size, head_size / 2, 10'000.0F);
  const std::vector<BFloat16> expected =
      rope_transpose_reference(input, factors, heads, head_size);

  DeviceBuffer<BFloat16> device_input(elements);
  DeviceBuffer<BFloat16> device_output(elements);
  DeviceBuffer<BFloat16> device_cos(factors.cosine.size());
  DeviceBuffer<BFloat16> device_sin(factors.sine.size());
  device_input.copy_from(input);
  device_cos.copy_from(factors.cosine);
  device_sin.copy_from(factors.sine);
  apply_rope_transpose_m2(device_input.get(), device_cos.get(),
                          device_sin.get(), device_output.get(), heads, kind);
  const std::vector<BFloat16> actual = device_output.copy_to_host();
  report_metric(report, label, measure(actual, expected), 0.0, true);

  // Position zero must be an exact transpose/identity, including the first and
  // last dimensions on the last head.
  for (std::uint32_t head = 0; head < heads; ++head) {
    for (std::uint32_t dimension : {0U, head_size / 2 - 1,
                                    head_size / 2, head_size - 1}) {
      const std::size_t source =
          static_cast<std::size_t>(head) * head_size + dimension;
      const std::size_t destination =
          static_cast<std::size_t>(head) * 2 * head_size + dimension;
      if (bf16_bits(actual[destination]) != bf16_bits(input[source])) {
        fail(label, "position-zero identity/transpose boundary failed");
      }
    }
  }
}

void test_rope_transpose(std::ostream& report) {
  test_rope_transpose_shape(report, gemma4_31b::AttentionKind::local,
                            gemma4_31b::kQueryHeadCount,
                            "rope_transpose_m2_local_q");
  test_rope_transpose_shape(report, gemma4_31b::AttentionKind::local,
                            gemma4_31b::kLocalKvHeadCount,
                            "rope_transpose_m2_local_k");
  test_rope_transpose_shape(report, gemma4_31b::AttentionKind::global,
                            gemma4_31b::kQueryHeadCount,
                            "rope_transpose_m2_global_q");
  test_rope_transpose_shape(report, gemma4_31b::AttentionKind::global,
                            gemma4_31b::kGlobalKvHeadCount,
                            "rope_transpose_m2_global_k");
}

std::vector<BFloat16> rope_m1_reference(
    const std::vector<BFloat16>& input,
    const std::vector<BFloat16>& cosine,
    const std::vector<BFloat16>& sine, std::uint32_t heads,
    std::uint32_t head_size) {
  if (input.size() != static_cast<std::size_t>(heads) * head_size ||
      cosine.size() != head_size || sine.size() != head_size) {
    fail("rope_m1_reference", "invalid reference shape");
  }
  std::vector<BFloat16> expected(input.size());
  const std::uint32_t half = head_size / 2;
  for (std::uint32_t head = 0; head < heads; ++head) {
    const std::size_t row = static_cast<std::size_t>(head) * head_size;
    for (std::uint32_t dimension = 0; dimension < head_size; ++dimension) {
      const std::uint32_t paired =
          dimension < half ? dimension + half : dimension - half;
      float rotated = to_float(input[row + paired]);
      if (dimension < half) {
        rotated = -rotated;
      }
      const BFloat16 direct =
          to_bf16(to_float(input[row + dimension]) *
                  to_float(cosine[dimension]));
      const BFloat16 crossed =
          to_bf16(rotated * to_float(sine[dimension]));
      expected[row + dimension] =
          to_bf16(to_float(direct) + to_float(crossed));
    }
  }
  return expected;
}

void test_rope_m1_shape(std::ostream& report,
                        gemma4_31b::AttentionKind kind,
                        std::uint32_t heads, std::uint32_t position,
                        std::string_view label) {
  const bool global = kind == gemma4_31b::AttentionKind::global;
  const std::uint32_t head_size = global ? gemma4_31b::kGlobalHeadSize
                                         : gemma4_31b::kLocalHeadSize;
  const std::size_t elements = static_cast<std::size_t>(heads) * head_size;
  std::vector<BFloat16> input(elements);
  for (std::uint32_t head = 0; head < heads; ++head) {
    for (std::uint32_t dimension = 0; dimension < head_size; ++dimension) {
      const int integer =
          static_cast<int>((head * 23 + dimension * 11) % 127) - 63;
      input[static_cast<std::size_t>(head) * head_size + dimension] =
          to_bf16(static_cast<float>(integer) / 32.0F);
    }
  }
  const RopeFactorReference factors =
      global ? rope_factor_reference(head_size, 64, 1'000'000.0F)
             : rope_factor_reference(head_size, head_size / 2, 10'000.0F);
  const auto factor_begin =
      factors.cosine.begin() + static_cast<std::size_t>(position) * head_size;
  const auto sine_begin =
      factors.sine.begin() + static_cast<std::size_t>(position) * head_size;
  const std::vector<BFloat16> cosine(factor_begin, factor_begin + head_size);
  const std::vector<BFloat16> sine(sine_begin, sine_begin + head_size);
  const std::vector<BFloat16> expected =
      rope_m1_reference(input, cosine, sine, heads, head_size);

  DeviceBuffer<BFloat16> device_input(elements);
  DeviceBuffer<BFloat16> device_cosine(head_size);
  DeviceBuffer<BFloat16> device_sine(head_size);
  DeviceBuffer<BFloat16> device_output(elements);
  device_input.copy_from(input);
  device_cosine.copy_from(cosine);
  device_sine.copy_from(sine);
  apply_rope_m1(device_input.get(), device_cosine.get(), device_sine.get(),
                device_output.get(), heads, kind);
  report_metric(report, label,
                measure(device_output.copy_to_host(), expected), 0.0, true);
}

void test_rope_m1(std::ostream& report) {
  for (const std::uint32_t position : {0U, 1U}) {
    const std::string suffix = "_position_" + std::to_string(position);
    test_rope_m1_shape(report, gemma4_31b::AttentionKind::local,
                       gemma4_31b::kQueryHeadCount, position,
                       "rope_m1_local_q" + suffix);
    test_rope_m1_shape(report, gemma4_31b::AttentionKind::local,
                       gemma4_31b::kLocalKvHeadCount, position,
                       "rope_m1_local_k" + suffix);
    test_rope_m1_shape(report, gemma4_31b::AttentionKind::global,
                       gemma4_31b::kQueryHeadCount, position,
                       "rope_m1_global_q" + suffix);
    test_rope_m1_shape(report, gemma4_31b::AttentionKind::global,
                       gemma4_31b::kGlobalKvHeadCount, position,
                       "rope_m1_global_k" + suffix);
  }
}

struct AttentionReference {
  std::vector<BFloat16> probabilities;
  std::vector<BFloat16> context;
};

AttentionReference attention_m2_reference(
    const std::vector<BFloat16>& query, const std::vector<BFloat16>& key,
    const std::vector<BFloat16>& value_token_major, std::uint32_t kv_heads,
    std::uint32_t head_size) {
  AttentionReference result{
      std::vector<BFloat16>(gemma4_31b::kQueryHeadCount * 2 * 2),
      std::vector<BFloat16>(static_cast<std::size_t>(2) *
                            gemma4_31b::kQueryHeadCount * head_size)};
  const std::uint32_t repeats = gemma4_31b::kQueryHeadCount / kv_heads;
  for (std::uint32_t query_head = 0;
       query_head < gemma4_31b::kQueryHeadCount; ++query_head) {
    const std::uint32_t kv_head = query_head / repeats;
    for (std::uint32_t query_token = 0; query_token < 2; ++query_token) {
      float scores[2] = {0.0F, 0.0F};
      for (std::uint32_t key_token = 0; key_token < 2; ++key_token) {
        for (std::uint32_t dimension = 0; dimension < head_size;
             ++dimension) {
          const std::size_t query_index =
              (static_cast<std::size_t>(query_head) * 2 + query_token) *
                  head_size +
              dimension;
          const std::size_t key_index =
              (static_cast<std::size_t>(kv_head) * 2 + key_token) *
                  head_size +
              dimension;
          scores[key_token] =
              std::fma(to_float(query[query_index]),
                       to_float(key[key_index]), scores[key_token]);
        }
        scores[key_token] = to_float(to_bf16(scores[key_token]));
      }

      BFloat16 probability[2];
      if (query_token == 0) {
        probability[0] = to_bf16(1.0F);
        probability[1] = to_bf16(0.0F);
      } else {
        const float maximum = std::max(scores[0], scores[1]);
        const float exponential_zero = std::exp(scores[0] - maximum);
        const float exponential_one = std::exp(scores[1] - maximum);
        const float denominator = exponential_zero + exponential_one;
        probability[0] = to_bf16(exponential_zero / denominator);
        probability[1] = to_bf16(exponential_one / denominator);
      }
      const std::size_t probability_offset =
          (static_cast<std::size_t>(query_head) * 2 + query_token) * 2;
      result.probabilities[probability_offset] = probability[0];
      result.probabilities[probability_offset + 1] = probability[1];

      for (std::uint32_t dimension = 0; dimension < head_size; ++dimension) {
        const std::size_t value_zero_index =
            static_cast<std::size_t>(kv_head) * head_size + dimension;
        const std::size_t value_one_index =
            (static_cast<std::size_t>(kv_heads) + kv_head) * head_size +
            dimension;
        const float first = to_float(probability[0]) *
                            to_float(value_token_major[value_zero_index]);
        const float sum = std::fma(
            to_float(probability[1]),
            to_float(value_token_major[value_one_index]), first);
        const std::size_t context_index =
            (static_cast<std::size_t>(query_token) *
                 gemma4_31b::kQueryHeadCount +
             query_head) *
                head_size +
            dimension;
        result.context[context_index] = to_bf16(sum);
      }
    }
  }
  return result;
}

void test_attention_kind(std::ostream& report,
                         gemma4_31b::AttentionKind kind,
                         std::string_view label) {
  const bool global = kind == gemma4_31b::AttentionKind::global;
  const std::uint32_t kv_heads = global ? gemma4_31b::kGlobalKvHeadCount
                                        : gemma4_31b::kLocalKvHeadCount;
  const std::uint32_t head_size = global ? gemma4_31b::kGlobalHeadSize
                                         : gemma4_31b::kLocalHeadSize;
  const std::uint32_t repeats = gemma4_31b::kQueryHeadCount / kv_heads;
  const std::size_t query_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * 2 * head_size;
  const std::size_t kv_elements =
      static_cast<std::size_t>(kv_heads) * 2 * head_size;
  std::vector<BFloat16> query(query_elements, to_bf16(0.0F));
  std::vector<BFloat16> key(kv_elements, to_bf16(0.0F));
  std::vector<BFloat16> value(kv_elements, to_bf16(0.0F));

  for (std::uint32_t head = 0; head < gemma4_31b::kQueryHeadCount;
       ++head) {
    query[(static_cast<std::size_t>(head) * 2) * head_size] =
        to_bf16(1.0F);
    query[(static_cast<std::size_t>(head) * 2 + 1) * head_size +
          head_size - 1] = to_bf16(1.0F);
  }
  for (std::uint32_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
    key[(static_cast<std::size_t>(kv_head) * 2) * head_size +
        head_size - 1] = to_bf16(-1.0F);
    key[(static_cast<std::size_t>(kv_head) * 2 + 1) * head_size] =
        to_bf16(8.0F);
    key[(static_cast<std::size_t>(kv_head) * 2 + 1) * head_size +
        head_size - 1] = to_bf16(1.0F);
    for (std::uint32_t token = 0; token < 2; ++token) {
      for (std::uint32_t dimension = 0; dimension < head_size; ++dimension) {
        const float base = static_cast<float>(kv_head + 1) / 16.0F;
        float item = token == 0 ? base : -0.5F * base;
        if (dimension == head_size - 1) {
          item += token == 0 ? 1.0F : -0.5F;
        }
        const std::size_t index =
            (static_cast<std::size_t>(token) * kv_heads + kv_head) *
                head_size +
            dimension;
        value[index] = to_bf16(item);
      }
    }
  }

  const AttentionReference expected =
      attention_m2_reference(query, key, value, kv_heads, head_size);
  DeviceBuffer<BFloat16> device_query(query.size());
  DeviceBuffer<BFloat16> device_key(key.size());
  DeviceBuffer<BFloat16> device_value(value.size());
  DeviceBuffer<BFloat16> device_probabilities(expected.probabilities.size());
  DeviceBuffer<BFloat16> device_context(expected.context.size());
  device_query.copy_from(query);
  device_key.copy_from(key);
  device_value.copy_from(value);
  causal_gqa_attention_m2(device_query.get(), device_key.get(),
                          device_value.get(), device_probabilities.get(),
                          device_context.get(), kind);
  const std::vector<BFloat16> actual_probabilities =
      device_probabilities.copy_to_host();
  const std::vector<BFloat16> actual_context = device_context.copy_to_host();
  report_metric(report, std::string(label) + "_probabilities",
                measure(actual_probabilities, expected.probabilities), 0.0,
                true);
  report_metric(report, std::string(label) + "_context",
                measure(actual_context, expected.context), 0.0, true);

  for (std::uint32_t head = 0; head < gemma4_31b::kQueryHeadCount; ++head) {
    const std::size_t offset = static_cast<std::size_t>(head) * 4;
    if (bf16_bits(actual_probabilities[offset]) != 0x3f80 ||
        bf16_bits(actual_probabilities[offset + 1]) != 0x0000 ||
        bf16_bits(actual_probabilities[offset + 2]) != 15860 ||
        bf16_bits(actual_probabilities[offset + 3]) != 16225) {
      fail(label, "causal mask or FP32-softmax BF16 boundary failed");
    }
    const std::uint32_t expected_kv_head = head / repeats;
    const std::size_t source =
        static_cast<std::size_t>(expected_kv_head) * head_size;
    const std::size_t destination =
        static_cast<std::size_t>(head) * head_size;
    if (bf16_bits(actual_context[destination]) != bf16_bits(value[source]) ||
        bf16_bits(actual_context[destination + head_size - 1]) !=
            bf16_bits(value[source + head_size - 1])) {
      fail(label, "GQA repeat mapping or head-dimension boundary failed");
    }
  }
  report << "primitive " << label << "_contract: causal=1 repeat="
         << repeats << " last_dimension=" << (head_size - 1) << " ok\n";
}

void test_attention_m2(std::ostream& report) {
  test_attention_kind(report, gemma4_31b::AttentionKind::local,
                      "causal_gqa_attention_m2_local");
  test_attention_kind(report, gemma4_31b::AttentionKind::global,
                      "causal_gqa_attention_m2_global");
}

void test_cached_attention_kind(std::ostream& report,
                                gemma4_31b::AttentionKind kind,
                                std::uint32_t capacity,
                                std::string_view label) {
  const bool global = kind == gemma4_31b::AttentionKind::global;
  const std::uint32_t kv_heads = global ? gemma4_31b::kGlobalKvHeadCount
                                        : gemma4_31b::kLocalKvHeadCount;
  const std::uint32_t head_size = global ? gemma4_31b::kGlobalHeadSize
                                         : gemma4_31b::kLocalHeadSize;
  const std::size_t query_m2_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * 2 * head_size;
  const std::size_t kv_m2_elements =
      static_cast<std::size_t>(kv_heads) * 2 * head_size;
  const std::size_t query_m1_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * head_size;
  const std::size_t kv_m1_elements =
      static_cast<std::size_t>(kv_heads) * head_size;
  const std::size_t cache_elements =
      static_cast<std::size_t>(kv_heads) * capacity * head_size;

  std::vector<BFloat16> query_m2(query_m2_elements, to_bf16(0.0F));
  std::vector<BFloat16> key_m2(kv_m2_elements, to_bf16(0.0F));
  std::vector<BFloat16> value_m2(kv_m2_elements, to_bf16(0.0F));
  for (std::uint32_t head = 0; head < gemma4_31b::kQueryHeadCount;
       ++head) {
    query_m2[(static_cast<std::size_t>(head) * 2) * head_size] =
        to_bf16(1.0F);
    query_m2[(static_cast<std::size_t>(head) * 2 + 1) * head_size +
             head_size - 1] = to_bf16(1.0F);
  }
  for (std::uint32_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
    key_m2[(static_cast<std::size_t>(kv_head) * 2) * head_size +
           head_size - 1] = to_bf16(-1.0F);
    key_m2[(static_cast<std::size_t>(kv_head) * 2 + 1) * head_size] =
        to_bf16(8.0F);
    key_m2[(static_cast<std::size_t>(kv_head) * 2 + 1) * head_size +
           head_size - 1] = to_bf16(1.0F);
    for (std::uint32_t token = 0; token < 2; ++token) {
      for (std::uint32_t dimension = 0; dimension < head_size; ++dimension) {
        const float base = static_cast<float>(kv_head + 1) / 16.0F;
        float item = token == 0 ? base : -0.5F * base;
        if (dimension == head_size - 1) {
          item += token == 0 ? 1.0F : -0.5F;
        }
        const std::size_t index =
            (static_cast<std::size_t>(token) * kv_heads + kv_head) *
                head_size +
            dimension;
        value_m2[index] = to_bf16(item);
      }
    }
  }

  const AttentionReference expected_m2 = attention_m2_reference(
      query_m2, key_m2, value_m2, kv_heads, head_size);
  std::array<std::vector<BFloat16>, 2> query_m1{
      std::vector<BFloat16>(query_m1_elements),
      std::vector<BFloat16>(query_m1_elements)};
  std::array<std::vector<BFloat16>, 2> key_m1{
      std::vector<BFloat16>(kv_m1_elements),
      std::vector<BFloat16>(kv_m1_elements)};
  std::array<std::vector<BFloat16>, 2> value_m1{
      std::vector<BFloat16>(kv_m1_elements),
      std::vector<BFloat16>(kv_m1_elements)};
  for (std::uint32_t token = 0; token < 2; ++token) {
    for (std::uint32_t head = 0; head < gemma4_31b::kQueryHeadCount;
         ++head) {
      const std::size_t source =
          (static_cast<std::size_t>(head) * 2 + token) * head_size;
      const std::size_t destination =
          static_cast<std::size_t>(head) * head_size;
      std::copy_n(query_m2.begin() + source, head_size,
                  query_m1[token].begin() + destination);
    }
    for (std::uint32_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
      const std::size_t key_source =
          (static_cast<std::size_t>(kv_head) * 2 + token) * head_size;
      const std::size_t value_source =
          (static_cast<std::size_t>(token) * kv_heads + kv_head) * head_size;
      const std::size_t destination =
          static_cast<std::size_t>(kv_head) * head_size;
      std::copy_n(key_m2.begin() + key_source, head_size,
                  key_m1[token].begin() + destination);
      std::copy_n(value_m2.begin() + value_source, head_size,
                  value_m1[token].begin() + destination);
    }
  }

  const BFloat16 key_sentinel = to_bf16(-7.5F);
  const BFloat16 value_sentinel = to_bf16(6.25F);
  std::vector<BFloat16> expected_key_cache(cache_elements, key_sentinel);
  std::vector<BFloat16> expected_value_cache(cache_elements, value_sentinel);
  DeviceBuffer<BFloat16> device_key_cache(cache_elements);
  DeviceBuffer<BFloat16> device_value_cache(cache_elements);
  device_key_cache.copy_from(expected_key_cache);
  device_value_cache.copy_from(expected_value_cache);
  DeviceBuffer<BFloat16> device_key(kv_m1_elements);
  DeviceBuffer<BFloat16> device_value(kv_m1_elements);
  DeviceBuffer<BFloat16> device_query(query_m1_elements);
  DeviceBuffer<BFloat16> device_probabilities(
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * 2);
  DeviceBuffer<BFloat16> device_context(query_m1_elements);

  const auto update_expected_cache =
      [&](const std::vector<BFloat16>& key,
          const std::vector<BFloat16>& value, std::uint32_t slot) {
        for (std::uint32_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
          const std::size_t source =
              static_cast<std::size_t>(kv_head) * head_size;
          const std::size_t destination =
              (static_cast<std::size_t>(kv_head) * capacity + slot) *
              head_size;
          std::copy_n(key.begin() + source, head_size,
                      expected_key_cache.begin() + destination);
          std::copy_n(value.begin() + source, head_size,
                      expected_value_cache.begin() + destination);
        }
      };

  const auto check_position = [&](std::uint32_t position) {
    std::vector<BFloat16> expected_probabilities(
        static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * 2);
    std::vector<BFloat16> expected_context(query_m1_elements);
    for (std::uint32_t head = 0; head < gemma4_31b::kQueryHeadCount;
         ++head) {
      const std::size_t probability_source =
          (static_cast<std::size_t>(head) * 2 + position) * 2;
      const std::size_t probability_destination =
          static_cast<std::size_t>(head) * 2;
      std::copy_n(expected_m2.probabilities.begin() + probability_source, 2,
                  expected_probabilities.begin() + probability_destination);
      const std::size_t context_source =
          (static_cast<std::size_t>(position) *
               gemma4_31b::kQueryHeadCount +
           head) *
          head_size;
      const std::size_t context_destination =
          static_cast<std::size_t>(head) * head_size;
      std::copy_n(expected_m2.context.begin() + context_source, head_size,
                  expected_context.begin() + context_destination);
    }

    device_query.copy_from(query_m1[position]);
    causal_gqa_attention_cached_m1(
        device_query.get(), device_key_cache.get(), device_value_cache.get(),
        position, capacity, device_probabilities.get(), device_context.get(),
        kind);
    const std::string position_label =
        std::string(label) + "_position_" + std::to_string(position);
    report_metric(report, position_label + "_probabilities",
                  measure(device_probabilities.copy_to_host(),
                          expected_probabilities),
                  0.0, true);
    report_metric(report, position_label + "_context",
                  measure(device_context.copy_to_host(), expected_context),
                  0.0, true);
  };

  for (std::uint32_t position = 0; position < 2; ++position) {
    device_key.copy_from(key_m1[position]);
    device_value.copy_from(value_m1[position]);
    write_kv_cache_m1(
        device_key.get(), device_value.get(), device_key_cache.get(),
        device_value_cache.get(), position, capacity, kind);
    update_expected_cache(key_m1[position], value_m1[position], position);
    const std::string write_label =
        std::string(label) + "_write_position_" + std::to_string(position);
    report_metric(report, write_label + "_key",
                  measure(device_key_cache.copy_to_host(),
                          expected_key_cache),
                  0.0, true);
    report_metric(report, write_label + "_value",
                  measure(device_value_cache.copy_to_host(),
                          expected_value_cache),
                  0.0, true);
    check_position(position);
  }

  const std::vector<BFloat16> wrap_key =
      pattern(kv_m1_elements, 67, 33, 1.0F / 16.0F);
  const std::vector<BFloat16> wrap_value =
      pattern(kv_m1_elements, 71, 35, 1.0F / 32.0F);
  device_key.copy_from(wrap_key);
  device_value.copy_from(wrap_value);
  write_kv_cache_m1(
      device_key.get(), device_value.get(), device_key_cache.get(),
      device_value_cache.get(), capacity + 1, capacity, kind);
  update_expected_cache(wrap_key, wrap_value, 1);
  report_metric(report, std::string(label) + "_modulo_write_key",
                measure(device_key_cache.copy_to_host(), expected_key_cache),
                0.0, true);
  report_metric(report, std::string(label) + "_modulo_write_value",
                measure(device_value_cache.copy_to_host(),
                        expected_value_cache),
                0.0, true);
  report << "primitive " << label << "_contract: capacity=" << capacity
         << " head_stride=" << capacity * head_size
         << " positions=0,1 modulo_slot=1 ok\n";
}

void test_cached_attention_m1(std::ostream& report) {
  test_cached_attention_kind(
      report, gemma4_31b::AttentionKind::local, 5,
      "causal_gqa_attention_cached_m1_local");
  test_cached_attention_kind(
      report, gemma4_31b::AttentionKind::global, 3,
      "causal_gqa_attention_cached_m1_global");
}

AttentionReference attention_cached_m1_24_reference(
    const std::vector<BFloat16>& query,
    const std::vector<BFloat16>& key_cache,
    const std::vector<BFloat16>& value_cache,
    std::uint32_t absolute_position, std::uint32_t capacity,
    std::uint32_t kv_heads, std::uint32_t head_size) {
  AttentionReference result{
      std::vector<BFloat16>(
          static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) *
              kCachedAttentionM1ShortCapacity,
          to_bf16(0.0F)),
      std::vector<BFloat16>(
          static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * head_size)};
  const std::uint32_t repeats = gemma4_31b::kQueryHeadCount / kv_heads;

  for (std::uint32_t query_head = 0;
       query_head < gemma4_31b::kQueryHeadCount; ++query_head) {
    const std::uint32_t kv_head = query_head / repeats;
    std::array<float, kCachedAttentionM1ShortCapacity> scores{};
    std::array<BFloat16, kCachedAttentionM1ShortCapacity> probability{};
    for (std::uint32_t token = 0; token <= absolute_position; ++token) {
      std::array<float, kThreads> partial{};
      const std::uint32_t slot = token % capacity;
      const std::size_t query_offset =
          static_cast<std::size_t>(query_head) * head_size;
      const std::size_t key_offset =
          (static_cast<std::size_t>(kv_head) * capacity + slot) * head_size;
      for (std::uint32_t thread = 0; thread < kThreads; ++thread) {
        for (std::uint32_t dimension = thread; dimension < head_size;
             dimension += kThreads) {
          partial[thread] = std::fma(
              to_float(query[query_offset + dimension]),
              to_float(key_cache[key_offset + dimension]), partial[thread]);
        }
      }
      for (unsigned offset = kThreads / 2; offset != 0; offset /= 2) {
        for (unsigned thread = 0; thread < offset; ++thread) {
          partial[thread] += partial[thread + offset];
        }
      }
      scores[token] = to_float(to_bf16(partial[0]));
    }

    float maximum = -std::numeric_limits<float>::infinity();
    for (std::uint32_t token = 0; token <= absolute_position; ++token) {
      maximum = std::max(maximum, scores[token]);
    }
    float denominator = 0.0F;
    for (std::uint32_t token = 0; token <= absolute_position; ++token) {
      scores[token] = std::exp(scores[token] - maximum);
      denominator += scores[token];
    }
    const std::size_t probability_offset =
        static_cast<std::size_t>(query_head) *
        kCachedAttentionM1ShortCapacity;
    for (std::uint32_t token = 0; token <= absolute_position; ++token) {
      probability[token] = to_bf16(scores[token] / denominator);
      result.probabilities[probability_offset + token] = probability[token];
    }

    for (std::uint32_t dimension = 0; dimension < head_size; ++dimension) {
      const std::size_t value_zero =
          (static_cast<std::size_t>(kv_head) * capacity) * head_size +
          dimension;
      float sum = to_float(probability[0]) * to_float(value_cache[value_zero]);
      for (std::uint32_t token = 1; token <= absolute_position; ++token) {
        const std::uint32_t slot = token % capacity;
        const std::size_t value_index =
            (static_cast<std::size_t>(kv_head) * capacity + slot) *
                head_size +
            dimension;
        sum = std::fma(to_float(probability[token]),
                       to_float(value_cache[value_index]), sum);
      }
      result.context[static_cast<std::size_t>(query_head) * head_size +
                     dimension] = to_bf16(sum);
    }
  }
  return result;
}

void test_cached_attention_m1_24_kind(std::ostream& report,
                                      gemma4_31b::AttentionKind kind,
                                      std::uint32_t capacity,
                                      std::string_view label) {
  const bool global = kind == gemma4_31b::AttentionKind::global;
  const std::uint32_t kv_heads = global ? gemma4_31b::kGlobalKvHeadCount
                                        : gemma4_31b::kLocalKvHeadCount;
  const std::uint32_t head_size = global ? gemma4_31b::kGlobalHeadSize
                                         : gemma4_31b::kLocalHeadSize;
  const std::size_t query_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * head_size;
  const std::size_t cache_elements =
      static_cast<std::size_t>(kv_heads) * capacity * head_size;
  const std::size_t probability_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) *
      kCachedAttentionM1ShortCapacity;

  DeviceBuffer<BFloat16> device_query(query_elements);
  DeviceBuffer<BFloat16> device_key_cache(cache_elements);
  DeviceBuffer<BFloat16> device_value_cache(cache_elements);
  DeviceBuffer<BFloat16> device_probabilities(probability_elements);
  DeviceBuffer<BFloat16> device_context(query_elements);
  DeviceBuffer<BFloat16> pair_probabilities(
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * 2);
  DeviceBuffer<BFloat16> pair_context(query_elements);

  for (const std::uint32_t position : {0U, 1U, 7U, 23U}) {
    std::vector<BFloat16> query(query_elements, to_bf16(0.0F));
    for (std::uint32_t head = 0; head < gemma4_31b::kQueryHeadCount;
         ++head) {
      const std::size_t row = static_cast<std::size_t>(head) * head_size;
      query[row] = to_bf16(1.0F);
      query[row + head_size - 1] =
          to_bf16((static_cast<int>(head % 5) - 2) / 8.0F);
    }

    // Every non-visible slot is a finite sentinel which would dominate the
    // softmax and context if the kernel read beyond the causal prefix.
    std::vector<BFloat16> key_cache(cache_elements, to_bf16(0.0F));
    std::vector<BFloat16> value_cache(cache_elements, to_bf16(12.0F));
    for (std::uint32_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
      for (std::uint32_t slot = 0; slot < capacity; ++slot) {
        const std::size_t row =
            (static_cast<std::size_t>(kv_head) * capacity + slot) * head_size;
        key_cache[row] = to_bf16(32.0F + static_cast<float>(slot));
        key_cache[row + head_size - 1] = to_bf16(-16.0F);
      }
      for (std::uint32_t token = 0; token <= position; ++token) {
        const std::uint32_t slot = token % capacity;
        const std::size_t row =
            (static_cast<std::size_t>(kv_head) * capacity + slot) * head_size;
        key_cache[row] = to_bf16(
            (static_cast<int>(token) - 8) / 16.0F + kv_head / 32.0F);
        key_cache[row + head_size - 1] = to_bf16(
            (static_cast<int>((3 * token + kv_head) % 9) - 4) / 16.0F);
        for (std::uint32_t dimension = 0; dimension < head_size;
             ++dimension) {
          const int integer = static_cast<int>(
                                  (13 * token + 7 * kv_head + 3 * dimension) %
                                  31) -
                              15;
          value_cache[row + dimension] =
              to_bf16(static_cast<float>(integer) / 32.0F);
        }
      }
    }

    const AttentionReference expected = attention_cached_m1_24_reference(
        query, key_cache, value_cache, position, capacity, kv_heads,
        head_size);
    device_query.copy_from(query);
    device_key_cache.copy_from(key_cache);
    device_value_cache.copy_from(value_cache);
    device_probabilities.copy_from(std::vector<BFloat16>(
        probability_elements, to_bf16(-9.0F)));
    device_context.copy_from(
        std::vector<BFloat16>(query_elements, to_bf16(-9.0F)));
    causal_gqa_attention_cached_m1_24(
        device_query.get(), device_key_cache.get(), device_value_cache.get(),
        position, capacity, device_probabilities.get(), device_context.get(),
        kind);

    const std::vector<BFloat16> actual_probabilities =
        device_probabilities.copy_to_host();
    const std::vector<BFloat16> actual_context =
        device_context.copy_to_host();
    const std::string position_label =
        std::string(label) + "_position_" + std::to_string(position);
    report_metric(report, position_label + "_probabilities",
                  measure(actual_probabilities, expected.probabilities), 0.0,
                  true);
    report_metric(report, position_label + "_context",
                  measure(actual_context, expected.context), 0.0, true);
    report_metric(report, position_label + "_key_cache_read_only",
                  measure(device_key_cache.copy_to_host(), key_cache), 0.0,
                  true);
    report_metric(report, position_label + "_value_cache_read_only",
                  measure(device_value_cache.copy_to_host(), value_cache),
                  0.0, true);
    for (std::uint32_t head = 0; head < gemma4_31b::kQueryHeadCount;
         ++head) {
      for (std::uint32_t token = position + 1;
           token < kCachedAttentionM1ShortCapacity; ++token) {
        const std::size_t index =
            static_cast<std::size_t>(head) *
                kCachedAttentionM1ShortCapacity +
            token;
        if (bf16_bits(actual_probabilities[index]) != 0x0000) {
          fail(position_label,
               "future probability is not exact BF16 positive zero");
        }
      }
    }

    if (position <= 1) {
      causal_gqa_attention_cached_m1(
          device_query.get(), device_key_cache.get(),
          device_value_cache.get(), position, capacity,
          pair_probabilities.get(), pair_context.get(), kind);
      const std::vector<BFloat16> actual_pair_probabilities =
          pair_probabilities.copy_to_host();
      std::vector<BFloat16> short_prefix(actual_pair_probabilities.size());
      for (std::uint32_t head = 0; head < gemma4_31b::kQueryHeadCount;
           ++head) {
        std::copy_n(actual_probabilities.begin() +
                        static_cast<std::size_t>(head) *
                            kCachedAttentionM1ShortCapacity,
                    2, short_prefix.begin() + static_cast<std::size_t>(head) *
                                                  2);
      }
      report_metric(report, position_label + "_pair_probability_equivalence",
                    measure(short_prefix, actual_pair_probabilities), 0.0,
                    true);
      report_metric(report, position_label + "_pair_context_equivalence",
                    measure(actual_context, pair_context.copy_to_host()), 0.0,
                    true);
    }
  }
  report << "primitive " << label << "_contract: visible_capacity="
         << kCachedAttentionM1ShortCapacity
         << " cache_capacity=" << capacity
         << " positions=0,1,7,23 future_zero=1 pair_equivalent=1 ok\n";
}

void test_cached_attention_m1_24(std::ostream& report) {
  static_assert(kCachedAttentionM1ShortCapacity == 24);
  test_cached_attention_m1_24_kind(
      report, gemma4_31b::AttentionKind::local, 29,
      "causal_gqa_attention_cached_m1_24_local");
  test_cached_attention_m1_24_kind(
      report, gemma4_31b::AttentionKind::global, 27,
      "causal_gqa_attention_cached_m1_24_global");
}

AttentionReference attention_cached_m1_boundary_reference(
    const std::vector<BFloat16>& query,
    const std::vector<BFloat16>& key_cache,
    const std::vector<BFloat16>& value_cache,
    std::uint32_t absolute_position, std::uint32_t capacity,
    std::uint32_t first_visible_position, std::uint32_t kv_heads,
    std::uint32_t head_size,
    std::uint32_t position_count =
        kCachedAttentionM1BoundaryPositionCount) {
  AttentionReference result{
      std::vector<BFloat16>(
          static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) *
              position_count,
          to_bf16(0.0F)),
      std::vector<BFloat16>(
          static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * head_size)};
  const std::uint32_t repeats = gemma4_31b::kQueryHeadCount / kv_heads;
  std::vector<float> scores(position_count);
  std::vector<BFloat16> probability(position_count, to_bf16(0.0F));

  for (std::uint32_t query_head = 0;
       query_head < gemma4_31b::kQueryHeadCount; ++query_head) {
    const std::uint32_t kv_head = query_head / repeats;
    const std::size_t query_offset =
        static_cast<std::size_t>(query_head) * head_size;
    for (std::uint32_t token = first_visible_position;
         token <= absolute_position; ++token) {
      std::array<float, kThreads> partial{};
      const std::uint32_t slot = token % capacity;
      const std::size_t key_offset =
          (static_cast<std::size_t>(kv_head) * capacity + slot) * head_size;
      for (std::uint32_t thread = 0; thread < kThreads; ++thread) {
        for (std::uint32_t dimension = thread; dimension < head_size;
             dimension += kThreads) {
          partial[thread] = std::fma(
              to_float(query[query_offset + dimension]),
              to_float(key_cache[key_offset + dimension]), partial[thread]);
        }
      }
      for (unsigned offset = kThreads / 2; offset != 0; offset /= 2) {
        for (unsigned thread = 0; thread < offset; ++thread) {
          partial[thread] += partial[thread + offset];
        }
      }
      scores[token] = to_float(to_bf16(partial[0]));
    }

    float maximum = -std::numeric_limits<float>::infinity();
    for (std::uint32_t token = first_visible_position;
         token <= absolute_position; ++token) {
      maximum = std::max(maximum, scores[token]);
    }
    float denominator = 0.0F;
    for (std::uint32_t token = first_visible_position;
         token <= absolute_position; ++token) {
      denominator += std::exp(scores[token] - maximum);
    }
    const std::size_t probability_offset =
        static_cast<std::size_t>(query_head) * position_count;
    for (std::uint32_t token = first_visible_position;
         token <= absolute_position; ++token) {
      probability[token] =
          to_bf16(std::exp(scores[token] - maximum) / denominator);
      result.probabilities[probability_offset + token] = probability[token];
    }

    const std::uint32_t first_slot = first_visible_position % capacity;
    const std::size_t first_value_offset =
        (static_cast<std::size_t>(kv_head) * capacity + first_slot) *
        head_size;
    for (std::uint32_t dimension = 0; dimension < head_size; ++dimension) {
      float sum = to_float(probability[first_visible_position]) *
                  to_float(value_cache[first_value_offset + dimension]);
      for (std::uint32_t token = first_visible_position + 1;
           token <= absolute_position; ++token) {
        const std::uint32_t slot = token % capacity;
        const std::size_t value_index =
            (static_cast<std::size_t>(kv_head) * capacity + slot) *
                head_size +
            dimension;
        sum = std::fma(to_float(probability[token]),
                       to_float(value_cache[value_index]), sum);
      }
      result.context[static_cast<std::size_t>(query_head) * head_size +
                     dimension] = to_bf16(sum);
    }
    std::fill(probability.begin(), probability.end(), to_bf16(0.0F));
  }
  return result;
}

std::vector<BFloat16> boundary_attention_query(std::uint32_t position,
                                               std::uint32_t head_size) {
  std::vector<BFloat16> query(
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * head_size,
      to_bf16(0.0F));
  for (std::uint32_t head = 0; head < gemma4_31b::kQueryHeadCount; ++head) {
    const std::size_t row = static_cast<std::size_t>(head) * head_size;
    query[row] = to_bf16(1.0F);
    query[row + head_size - 1] = to_bf16(
        (static_cast<int>((head + position) % 9) - 4) / 16.0F);
  }
  return query;
}

void boundary_attention_kv(std::uint32_t token, std::uint32_t kv_heads,
                           std::uint32_t head_size,
                           std::vector<BFloat16>* key,
                           std::vector<BFloat16>* value) {
  const std::size_t elements =
      static_cast<std::size_t>(kv_heads) * head_size;
  key->assign(elements, to_bf16(0.0F));
  value->resize(elements);
  for (std::uint32_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
    const std::size_t row = static_cast<std::size_t>(kv_head) * head_size;
    (*key)[row] = to_bf16(
        (static_cast<int>((17 * token + 5 * kv_head) % 257) - 128) /
        64.0F);
    (*key)[row + head_size - 1] = to_bf16(
        (static_cast<int>((7 * token + 3 * kv_head) % 31) - 15) /
        32.0F);
    for (std::uint32_t dimension = 0; dimension < head_size; ++dimension) {
      const int integer = static_cast<int>(
                              (13 * token + 7 * kv_head + 3 * dimension) %
                              63) -
                          31;
      (*value)[row + dimension] =
          to_bf16(static_cast<float>(integer) / 64.0F);
    }
  }
}

std::vector<BFloat16> dense_boundary_attention_query(
    std::uint32_t position, std::uint32_t head_size) {
  std::vector<BFloat16> query =
      boundary_attention_query(position, head_size);
  for (std::uint32_t head = 0; head < gemma4_31b::kQueryHeadCount; ++head) {
    const std::size_t row = static_cast<std::size_t>(head) * head_size;
    for (std::uint32_t dimension = 1; dimension + 1 < head_size;
         ++dimension) {
      const int integer =
          static_cast<int>((11 * head + 5 * position + 3 * dimension) % 17) -
          8;
      query[row + dimension] =
          to_bf16(static_cast<float>(integer) / 256.0F);
    }
  }
  return query;
}

void dense_boundary_attention_kv(std::uint32_t token,
                                 std::uint32_t kv_heads,
                                 std::uint32_t head_size,
                                 std::vector<BFloat16>* key,
                                 std::vector<BFloat16>* value) {
  boundary_attention_kv(token, kv_heads, head_size, key, value);
  for (std::uint32_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
    const std::size_t row = static_cast<std::size_t>(kv_head) * head_size;
    for (std::uint32_t dimension = 1; dimension + 1 < head_size;
         ++dimension) {
      const int integer = static_cast<int>(
                              (17 * token + 5 * kv_head + 11 * dimension) %
                              29) -
                          14;
      (*key)[row + dimension] =
          to_bf16(static_cast<float>(integer) / 256.0F);
    }
  }
}

void place_boundary_attention_kv(const std::vector<BFloat16>& key,
                                 const std::vector<BFloat16>& value,
                                 std::uint32_t token, std::uint32_t capacity,
                                 std::uint32_t kv_heads,
                                 std::uint32_t head_size,
                                 std::vector<BFloat16>* key_cache,
                                 std::vector<BFloat16>* value_cache) {
  const std::uint32_t slot = token % capacity;
  for (std::uint32_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
    const std::size_t source =
        static_cast<std::size_t>(kv_head) * head_size;
    const std::size_t destination =
        (static_cast<std::size_t>(kv_head) * capacity + slot) * head_size;
    std::copy_n(key.begin() + source, head_size,
                key_cache->begin() + destination);
    std::copy_n(value.begin() + source, head_size,
                value_cache->begin() + destination);
  }
}

void initialize_boundary_attention_cache(
    std::uint32_t through_token, std::uint32_t capacity,
    std::uint32_t kv_heads, std::uint32_t head_size,
    std::vector<BFloat16>* key_cache,
    std::vector<BFloat16>* value_cache) {
  const std::size_t elements =
      static_cast<std::size_t>(kv_heads) * capacity * head_size;
  key_cache->assign(elements, to_bf16(0.0F));
  value_cache->assign(elements, to_bf16(12.0F));
  for (std::uint32_t kv_head = 0; kv_head < kv_heads; ++kv_head) {
    for (std::uint32_t slot = 0; slot < capacity; ++slot) {
      const std::size_t row =
          (static_cast<std::size_t>(kv_head) * capacity + slot) * head_size;
      (*key_cache)[row] = to_bf16(64.0F);
      (*key_cache)[row + head_size - 1] = to_bf16(-16.0F);
    }
  }
  std::vector<BFloat16> key;
  std::vector<BFloat16> value;
  for (std::uint32_t token = 0; token <= through_token; ++token) {
    boundary_attention_kv(token, kv_heads, head_size, &key, &value);
    place_boundary_attention_kv(key, value, token, capacity, kv_heads,
                                head_size, key_cache, value_cache);
  }
}

void initialize_dense_boundary_attention_cache(
    std::uint32_t through_token, std::uint32_t capacity,
    std::uint32_t kv_heads, std::uint32_t head_size,
    std::vector<BFloat16>* key_cache,
    std::vector<BFloat16>* value_cache) {
  const std::size_t elements =
      static_cast<std::size_t>(kv_heads) * capacity * head_size;
  key_cache->assign(elements, to_bf16(0.0F));
  value_cache->assign(elements, to_bf16(12.0F));
  std::vector<BFloat16> key;
  std::vector<BFloat16> value;
  for (std::uint32_t token = 0; token <= through_token; ++token) {
    dense_boundary_attention_kv(token, kv_heads, head_size, &key, &value);
    place_boundary_attention_kv(key, value, token, capacity, kv_heads,
                                head_size, key_cache, value_cache);
  }
}

void require_boundary_probability_zeros(
    const std::vector<BFloat16>& probabilities,
    std::uint32_t first_visible_position, std::uint32_t absolute_position,
    std::string_view label,
    std::uint32_t position_count =
        kCachedAttentionM1BoundaryPositionCount) {
  for (std::uint32_t head = 0; head < gemma4_31b::kQueryHeadCount; ++head) {
    const std::size_t row =
        static_cast<std::size_t>(head) * position_count;
    for (std::uint32_t token = 0; token < position_count; ++token) {
      if ((token < first_visible_position || token > absolute_position) &&
          bf16_bits(probabilities[row + token]) != 0x0000) {
        fail(label, "masked probability is not exact BF16 positive zero");
      }
    }
  }
}

void test_cached_attention_m1_boundary_kind(
    std::ostream& report, gemma4_31b::AttentionKind kind,
    std::string_view label) {
  const bool global = kind == gemma4_31b::AttentionKind::global;
  const std::uint32_t capacity =
      global ? kCachedAttentionM1BoundaryGlobalCapacity
             : kCachedAttentionM1BoundaryLocalCapacity;
  const std::uint32_t kv_heads = global ? gemma4_31b::kGlobalKvHeadCount
                                        : gemma4_31b::kLocalKvHeadCount;
  const std::uint32_t head_size = global ? gemma4_31b::kGlobalHeadSize
                                         : gemma4_31b::kLocalHeadSize;
  const std::size_t query_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * head_size;
  const std::size_t kv_elements =
      static_cast<std::size_t>(kv_heads) * head_size;
  const std::size_t cache_elements =
      static_cast<std::size_t>(kv_heads) * capacity * head_size;
  const std::size_t probability_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) *
      kCachedAttentionM1BoundaryPositionCount;
  static_assert(kCachedAttentionM1BoundaryScoreScratchBytes ==
                static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) *
                    kCachedAttentionM1BoundaryPositionCount *
                    sizeof(BFloat16));

  DeviceBuffer<BFloat16> device_query(query_elements);
  DeviceBuffer<BFloat16> device_key(kv_elements);
  DeviceBuffer<BFloat16> device_value(kv_elements);
  DeviceBuffer<BFloat16> device_key_cache(cache_elements);
  DeviceBuffer<BFloat16> device_value_cache(cache_elements);
  DeviceBuffer<BFloat16> device_score_scratch(probability_elements);
  DeviceBuffer<BFloat16> device_probabilities(probability_elements);
  DeviceBuffer<BFloat16> device_context(query_elements);
  DeviceBuffer<BFloat16> device_short_probabilities(
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) *
      kCachedAttentionM1ShortCapacity);
  DeviceBuffer<BFloat16> device_short_context(query_elements);
  const std::size_t runtime_scratch_bytes =
      causal_gqa_attention_cached_m1_fused_scratch_bytes(capacity, kind);
  DeviceBuffer<float> device_runtime_scratch(runtime_scratch_bytes /
                                              sizeof(float));
  DeviceBuffer<BFloat16> device_runtime_context(query_elements);

  for (const std::uint32_t position : {0U, 1U, 7U, 23U}) {
    const std::vector<BFloat16> query =
        boundary_attention_query(position, head_size);
    std::vector<BFloat16> key_cache;
    std::vector<BFloat16> value_cache;
    initialize_boundary_attention_cache(position, capacity, kv_heads,
                                        head_size, &key_cache, &value_cache);
    device_query.copy_from(query);
    device_key_cache.copy_from(key_cache);
    device_value_cache.copy_from(value_cache);
    device_score_scratch.copy_from(std::vector<BFloat16>(
        probability_elements, to_bf16(-7.0F)));
    device_probabilities.copy_from(std::vector<BFloat16>(
        probability_elements, to_bf16(-9.0F)));
    causal_gqa_attention_cached_m1_boundary(
        device_query.get(), device_key_cache.get(), device_value_cache.get(),
        position, device_score_scratch.get(), device_probabilities.get(),
        device_context.get(), kind);
    causal_gqa_attention_cached_m1_24(
        device_query.get(), device_key_cache.get(), device_value_cache.get(),
        position, capacity, device_short_probabilities.get(),
        device_short_context.get(), kind);
    causal_gqa_attention_cached_m1_fused(
        device_query.get(), device_key_cache.get(), device_value_cache.get(),
        position, capacity, device_runtime_scratch.get(),
        device_runtime_context.get(), kind);

    const std::vector<BFloat16> probabilities =
        device_probabilities.copy_to_host();
    const std::vector<BFloat16> short_probabilities =
        device_short_probabilities.copy_to_host();
    std::vector<BFloat16> boundary_short_prefix(
        short_probabilities.size());
    for (std::uint32_t head = 0; head < gemma4_31b::kQueryHeadCount; ++head) {
      std::copy_n(
          probabilities.begin() +
              static_cast<std::size_t>(head) *
                  kCachedAttentionM1BoundaryPositionCount,
          kCachedAttentionM1ShortCapacity,
          boundary_short_prefix.begin() +
              static_cast<std::size_t>(head) *
                  kCachedAttentionM1ShortCapacity);
    }
    const std::string position_label =
        std::string(label) + "_position_" + std::to_string(position);
    report_metric(report, position_label + "_short_probability_equivalence",
                  measure(boundary_short_prefix, short_probabilities), 0.0,
                  true);
    report_metric(report, position_label + "_short_context_equivalence",
                  measure(device_context.copy_to_host(),
                          device_short_context.copy_to_host()),
                  0.0, true);
    report_metric(report, position_label + "_runtime_fused_context",
                  measure(device_runtime_context.copy_to_host(),
                          device_context.copy_to_host()),
                  1.0 / 128.0, false);
    report_metric(report, position_label + "_key_cache_read_only",
                  measure(device_key_cache.copy_to_host(), key_cache), 0.0,
                  true);
    report_metric(report, position_label + "_value_cache_read_only",
                  measure(device_value_cache.copy_to_host(), value_cache),
                  0.0, true);
    require_boundary_probability_zeros(probabilities, 0, position,
                                       position_label);
  }

  std::vector<BFloat16> expected_key_cache;
  std::vector<BFloat16> expected_value_cache;
  initialize_boundary_attention_cache(1'021, capacity, kv_heads, head_size,
                                      &expected_key_cache,
                                      &expected_value_cache);
  device_key_cache.copy_from(expected_key_cache);
  device_value_cache.copy_from(expected_value_cache);
  for (const std::uint32_t position : {1'022U, 1'023U, 1'024U, 1'025U}) {
    std::vector<BFloat16> key;
    std::vector<BFloat16> value;
    boundary_attention_kv(position, kv_heads, head_size, &key, &value);
    device_key.copy_from(key);
    device_value.copy_from(value);
    write_kv_cache_m1(device_key.get(), device_value.get(),
                      device_key_cache.get(), device_value_cache.get(),
                      position, capacity, kind);
    place_boundary_attention_kv(key, value, position, capacity, kv_heads,
                                head_size, &expected_key_cache,
                                &expected_value_cache);
    const std::string position_label =
        std::string(label) + "_position_" + std::to_string(position);
    report_metric(report, position_label + "_modulo_write_key",
                  measure(device_key_cache.copy_to_host(),
                          expected_key_cache),
                  0.0, true);
    report_metric(report, position_label + "_modulo_write_value",
                  measure(device_value_cache.copy_to_host(),
                          expected_value_cache),
                  0.0, true);

    const std::vector<BFloat16> query =
        boundary_attention_query(position, head_size);
    const std::uint32_t first_visible_position =
        global || position < kCachedAttentionM1BoundaryLocalCapacity
            ? 0
            : position - (kCachedAttentionM1BoundaryLocalCapacity - 1);
    const AttentionReference expected =
        attention_cached_m1_boundary_reference(
            query, expected_key_cache, expected_value_cache, position,
            capacity, first_visible_position, kv_heads, head_size);
    device_query.copy_from(query);
    device_score_scratch.copy_from(std::vector<BFloat16>(
        probability_elements, to_bf16(-7.0F)));
    device_probabilities.copy_from(std::vector<BFloat16>(
        probability_elements, to_bf16(-9.0F)));
    device_context.copy_from(
        std::vector<BFloat16>(query_elements, to_bf16(-9.0F)));
    causal_gqa_attention_cached_m1_boundary(
        device_query.get(), device_key_cache.get(), device_value_cache.get(),
        position, device_score_scratch.get(), device_probabilities.get(),
        device_context.get(), kind);
    causal_gqa_attention_cached_m1_fused(
        device_query.get(), device_key_cache.get(), device_value_cache.get(),
        position, capacity, device_runtime_scratch.get(),
        device_runtime_context.get(), kind);

    const std::vector<BFloat16> probabilities =
        device_probabilities.copy_to_host();
    report_metric(report, position_label + "_probabilities",
                  measure(probabilities, expected.probabilities), 0.0, true);
    report_metric(report, position_label + "_context",
                  measure(device_context.copy_to_host(), expected.context),
                  0.0, true);
    const std::vector<BFloat16> runtime_context =
        device_runtime_context.copy_to_host();
    if (!std::all_of(runtime_context.begin(), runtime_context.end(),
                     [](BFloat16 value) {
                       return std::isfinite(to_float(value));
                     })) {
      fail(position_label + "_runtime_fused_context",
           "CUDA result contains a non-finite value");
    }
    report_metric(report, position_label + "_runtime_fused_context",
                  measure(runtime_context, expected.context),
                  1.0 / 128.0, false);
    report_metric(report, position_label + "_key_cache_read_only",
                  measure(device_key_cache.copy_to_host(),
                          expected_key_cache),
                  0.0, true);
    report_metric(report, position_label + "_value_cache_read_only",
                  measure(device_value_cache.copy_to_host(),
                          expected_value_cache),
                  0.0, true);
    require_boundary_probability_zeros(
        probabilities, first_visible_position, position, position_label);
  }
  report << "primitive " << label << "_contract: positions=0..1025 capacity="
         << capacity << " boundary_positions=1022,1023,1024,1025 "
         << "short_equivalence=0,1,7,23 masked_positive_zero=1 "
         << "runtime_fused=0,1,7,23,1022,1023,1024,1025 "
         << "cache_read_only=1 score_scratch_bytes="
         << kCachedAttentionM1BoundaryScoreScratchBytes << " ok\n";
}

void test_cached_attention_m1_boundary(std::ostream& report) {
  static_assert(kCachedAttentionM1BoundaryLocalCapacity == 1'024);
  static_assert(kCachedAttentionM1BoundaryGlobalCapacity == 1'026);
  static_assert(kMaxContextTokenCount == 262'144);
  const auto expected_scratch = [](std::uint32_t capacity,
                                   std::uint32_t head_size, bool global) {
    const std::size_t split_count =
        (static_cast<std::size_t>(capacity) +
         kRuntimeAttentionTokensPerSplit - 1) /
        kRuntimeAttentionTokensPerSplit;
    const std::size_t fine_bytes =
        static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * split_count *
        (static_cast<std::size_t>(head_size) *
             (global ? sizeof(__half) : sizeof(float)) +
         2 * sizeof(float));
    return fine_bytes +
           (global
                ? runtime_global_coarse_scratch_bytes_for_splits(split_count)
                : 0);
  };
  for (const std::uint32_t capacity : {1U, 31U, 32U, 33U, 131'583U,
                                       kMaxContextTokenCount}) {
    const std::size_t actual =
        causal_gqa_attention_cached_m1_fused_scratch_bytes(
            capacity, gemma4_31b::AttentionKind::global);
    if (actual != expected_scratch(
                      capacity, gemma4_31b::kGlobalHeadSize, true)) {
      fail("causal_gqa_attention_cached_m1_fused_scratch_bytes",
           "global split sizing mismatch");
    }
  }
  const std::size_t local_scratch =
      causal_gqa_attention_cached_m1_fused_scratch_bytes(
          kGraphAttentionLocalCapacity,
          gemma4_31b::AttentionKind::local);
  if (local_scratch != expected_scratch(
                           kGraphAttentionLocalCapacity,
                           gemma4_31b::kLocalHeadSize, false)) {
    fail("causal_gqa_attention_cached_m1_fused_scratch_bytes",
         "local split sizing mismatch");
  }
  constexpr std::size_t kTargetGlobalScratchBytes = 140'071'168;
  constexpr std::size_t kMaximumGlobalScratchBytes = 278'953'984;
  constexpr std::size_t kLocalScratchBytes = 1'056'768;
  if (causal_gqa_attention_cached_m1_fused_scratch_bytes(
          131'583, gemma4_31b::AttentionKind::global) !=
          kTargetGlobalScratchBytes ||
      causal_gqa_attention_cached_m1_fused_scratch_bytes(
          kMaxContextTokenCount,
          gemma4_31b::AttentionKind::global) !=
          kMaximumGlobalScratchBytes ||
      local_scratch != kLocalScratchBytes) {
    fail("causal_gqa_attention_cached_m1_fused_scratch_bytes",
         "known-capacity byte contract mismatch");
  }
  test_cached_attention_m1_boundary_kind(
      report, gemma4_31b::AttentionKind::local,
      "causal_gqa_attention_cached_m1_boundary_local");
  test_cached_attention_m1_boundary_kind(
      report, gemma4_31b::AttentionKind::global,
      "causal_gqa_attention_cached_m1_boundary_global");
  report << "primitive runtime_fused_attention_scratch_contract: "
            "local_splits=32 global_max_splits=8192 max_context=262144 "
            "global_coarse_group=64 global_coarse_min_splits=1024 "
            "target_capacity=131583 target_global_bytes="
         << kTargetGlobalScratchBytes << " local_bytes="
         << kLocalScratchBytes << " "
            "max_global_bytes="
         << causal_gqa_attention_cached_m1_fused_scratch_bytes(
                kMaxContextTokenCount,
                gemma4_31b::AttentionKind::global)
         << " ok\n";
}

bool compact_global_rotated_key_dimension(std::uint32_t dimension) {
  return dimension < 64 || (dimension >= 256 && dimension < 320);
}

std::size_t compact_global_key_offset(std::uint32_t dimension) {
  if (dimension < 64) {
    return dimension;
  }
  if (dimension >= 256 && dimension < 320) {
    return 64 + dimension - 256;
  }
  fail("compact_global_key_offset", "dimension is not rotated");
}

float host_warp_sum(std::array<float, kWarpThreads> values) {
  for (unsigned offset = kWarpThreads / 2; offset != 0; offset /= 2) {
    for (unsigned lane = 0; lane < offset; ++lane) {
      values[lane] += values[lane + offset];
    }
  }
  return values[0];
}

// Mirrors the compact runtime kernel's split reduction. The only intentional
// host/device difference is std::exp versus CUDA's fast __expf.
std::vector<BFloat16> compact_global_folded_query_reference(
    const std::vector<BFloat16>& query,
    const std::vector<BFloat16>& compact_cache,
    const std::vector<BFloat16>& k_norm_scale,
    std::uint32_t absolute_position, std::uint32_t capacity) {
  constexpr std::uint32_t kHeadSize = gemma4_31b::kGlobalHeadSize;
  constexpr std::uint32_t kKvHeads = gemma4_31b::kGlobalKvHeadCount;
  constexpr std::uint32_t kQueriesPerKv =
      gemma4_31b::kQueryHeadCount / kKvHeads;
  const std::size_t query_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * kHeadSize;
  const std::size_t cache_elements =
      static_cast<std::size_t>(kKvHeads) * capacity * kGlobalCompactKvSize;
  if (query.size() != query_elements ||
      compact_cache.size() != cache_elements ||
      k_norm_scale.size() != kHeadSize || absolute_position >= capacity) {
    fail("compact_global_folded_query_reference", "invalid fixture shape");
  }

  const std::uint32_t visible_tokens = absolute_position + 1;
  const std::uint32_t split_count =
      (visible_tokens + kRuntimeAttentionTokensPerSplit - 1) /
      kRuntimeAttentionTokensPerSplit;
  std::vector<BFloat16> context(query_elements);

  for (std::uint32_t query_head = 0;
       query_head < gemma4_31b::kQueryHeadCount; ++query_head) {
    const std::uint32_t kv_head = query_head / kQueriesPerKv;
    const std::size_t query_offset =
        static_cast<std::size_t>(query_head) * kHeadSize;
    std::array<float, kHeadSize> folded_query{};
    for (std::uint32_t dimension = 0; dimension < kHeadSize; ++dimension) {
      float value = to_float(query[query_offset + dimension]);
      if (!compact_global_rotated_key_dimension(dimension)) {
        value *= to_float(k_norm_scale[dimension]);
      }
      folded_query[dimension] = value;
    }

    std::vector<float> split_maximum(
        split_count, -std::numeric_limits<float>::infinity());
    std::vector<float> split_denominator(split_count);
    std::vector<float> partial_context(
        static_cast<std::size_t>(split_count) * kHeadSize);
    for (std::uint32_t split = 0; split < split_count; ++split) {
      const std::uint32_t split_begin = static_cast<std::uint32_t>(
          static_cast<std::uint64_t>(visible_tokens) * split / split_count);
      const std::uint32_t split_end = static_cast<std::uint32_t>(
          static_cast<std::uint64_t>(visible_tokens) * (split + 1) /
          split_count);
      const std::uint32_t split_tokens = split_end - split_begin;
      std::array<float, kRuntimeAttentionTokensPerSplit> scores{};
      for (std::uint32_t token = 0; token < split_tokens; ++token) {
        const std::size_t compact_row =
            (static_cast<std::size_t>(kv_head) * capacity + split_begin +
             token) *
            kGlobalCompactKvSize;
        std::array<float, kWarpThreads> partial{};
        for (std::uint32_t lane = 0; lane < kWarpThreads; ++lane) {
          for (std::uint32_t dimension = lane; dimension < kHeadSize;
               dimension += kWarpThreads) {
            const std::size_t cache_offset =
                compact_global_rotated_key_dimension(dimension)
                    ? compact_global_key_offset(dimension)
                    : kGlobalCompactKeySize + dimension;
            partial[lane] = std::fma(
                folded_query[dimension],
                to_float(compact_cache[compact_row + cache_offset]),
                partial[lane]);
          }
        }
        scores[token] = host_warp_sum(partial);
        split_maximum[split] =
            std::max(split_maximum[split], scores[token]);
      }

      std::array<float, kWarpThreads> denominator_lanes{};
      for (std::uint32_t token = 0; token < split_tokens; ++token) {
        scores[token] =
            std::exp(scores[token] - split_maximum[split]);
        denominator_lanes[token] = scores[token];
      }
      split_denominator[split] = host_warp_sum(denominator_lanes);

      for (std::uint32_t dimension = 0; dimension < kHeadSize;
           ++dimension) {
        float sum = 0.0F;
        for (std::uint32_t token = 0; token < split_tokens; ++token) {
          const std::size_t value_index =
              (static_cast<std::size_t>(kv_head) * capacity + split_begin +
               token) *
                  kGlobalCompactKvSize +
              kGlobalCompactKeySize + dimension;
          sum = std::fma(scores[token],
                         to_float(compact_cache[value_index]), sum);
        }
        partial_context[static_cast<std::size_t>(split) * kHeadSize +
                        dimension] = sum;
      }
    }

    const float maximum =
        *std::max_element(split_maximum.begin(), split_maximum.end());
    std::vector<float> split_scale(split_count);
    std::array<float, kWarpThreads> denominator_lanes{};
    for (std::uint32_t split = 0; split < split_count; ++split) {
      split_scale[split] = std::exp(split_maximum[split] - maximum);
      const std::uint32_t lane = split % kWarpThreads;
      denominator_lanes[lane] =
          std::fma(split_denominator[split], split_scale[split],
                   denominator_lanes[lane]);
    }
    const float inverse_denominator =
        1.0F / host_warp_sum(denominator_lanes);
    for (std::uint32_t dimension = 0; dimension < kHeadSize; ++dimension) {
      float sum = 0.0F;
      for (std::uint32_t split = 0; split < split_count; ++split) {
        sum = std::fma(
            partial_context[static_cast<std::size_t>(split) * kHeadSize +
                            dimension],
            split_scale[split], sum);
      }
      context[query_offset + dimension] =
          to_bf16(sum * inverse_denominator);
    }
  }
  return context;
}

void launch_global_fp32_partial_reference(
    const BFloat16* query, const BFloat16* key_cache,
    const BFloat16* value_cache, std::uint32_t absolute_position,
    std::uint32_t capacity, float* scratch, BFloat16* context) {
  constexpr std::uint32_t kHeadSize = gemma4_31b::kGlobalHeadSize;
  const std::uint32_t split_count =
      (absolute_position + 1 + kRuntimeAttentionTokensPerSplit - 1) /
      kRuntimeAttentionTokensPerSplit;
  const std::size_t partial_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * split_count *
      kHeadSize;
  const std::size_t metadata_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * split_count;
  float* const split_maximum = scratch + partial_elements;
  float* const split_denominator = split_maximum + metadata_elements;
  const dim3 split_grid{gemma4_31b::kGlobalKvHeadCount, split_count};
  causal_gqa_attention_cached_m1_fused_split_kernel<
      float, kHeadSize, gemma4_31b::kGlobalKvHeadCount, false>
      <<<split_grid, kThreads>>>(
          query, key_cache, value_cache, absolute_position, capacity,
          split_count, scratch, split_maximum, split_denominator);
  check_cuda(cudaGetLastError(),
             "global FP32 partial reference split kernel launch");

  constexpr std::uint32_t kDimensionTile =
      kHeadSize / kGraphAttentionFusedFinalizeTiles;
  constexpr dim3 kFinalizeGrid{gemma4_31b::kQueryHeadCount,
                               kGraphAttentionFusedFinalizeTiles};
  const std::size_t shared_bytes =
      static_cast<std::size_t>(split_count) * sizeof(float);
  causal_gqa_attention_cached_m1_fused_finalize_kernel<
      float, kHeadSize, kDimensionTile>
      <<<kFinalizeGrid, kDimensionTile, shared_bytes>>>(
          scratch, split_maximum, split_denominator, split_count, context);
  check_cuda(cudaGetLastError(),
             "global FP32 partial reference finalize kernel launch");
}

void launch_global_compact_fp32_partial_reference(
    const BFloat16* query, const BFloat16* compact_kv_cache,
    const BFloat16* k_norm_scale, std::uint32_t absolute_position,
    std::uint32_t capacity, float* scratch, BFloat16* context) {
  constexpr std::uint32_t kHeadSize = gemma4_31b::kGlobalHeadSize;
  const std::uint32_t split_count =
      (absolute_position + 1 + kRuntimeAttentionTokensPerSplit - 1) /
      kRuntimeAttentionTokensPerSplit;
  const std::size_t partial_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * split_count *
      kHeadSize;
  const std::size_t metadata_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * split_count;
  float* const split_maximum = scratch + partial_elements;
  float* const split_denominator = split_maximum + metadata_elements;
  const dim3 split_grid{gemma4_31b::kGlobalKvHeadCount, split_count};
  causal_gqa_attention_cached_m1_fused_global_compact_scalar_split_kernel<
      float, false>
      <<<split_grid, kThreads>>>(
          query, compact_kv_cache, nullptr, k_norm_scale, 0, 0,
          absolute_position, capacity, split_count, scratch, split_maximum,
          split_denominator);
  check_cuda(cudaGetLastError(),
             "global compact FP32 partial reference split kernel launch");

  constexpr std::uint32_t kDimensionTile =
      kHeadSize / kGraphAttentionFusedFinalizeTiles;
  constexpr dim3 kFinalizeGrid{gemma4_31b::kQueryHeadCount,
                               kGraphAttentionFusedFinalizeTiles};
  const std::size_t shared_bytes =
      static_cast<std::size_t>(split_count) * sizeof(float);
  causal_gqa_attention_cached_m1_fused_finalize_kernel<
      float, kHeadSize, kDimensionTile>
      <<<kFinalizeGrid, kDimensionTile, shared_bytes>>>(
          scratch, split_maximum, split_denominator, split_count, context);
  check_cuda(
      cudaGetLastError(),
      "global compact FP32 partial reference finalize kernel launch");
}

void launch_global_compact_scalar_half_reference(
    const BFloat16* query, const BFloat16* compact_kv_cache,
    const BFloat16* k_norm_scale, std::uint32_t absolute_position,
    std::uint32_t capacity, void* scratch, BFloat16* context) {
  constexpr std::uint32_t kHeadSize = gemma4_31b::kGlobalHeadSize;
  const std::uint32_t split_count =
      (absolute_position + 1 + kRuntimeAttentionTokensPerSplit - 1) /
      kRuntimeAttentionTokensPerSplit;
  const RuntimeFusedScratchLayout storage = runtime_fused_scratch_layout(
      scratch, split_count, kHeadSize, true);
  auto* const partial_context =
      static_cast<__half*>(storage.partial_context);
  const dim3 split_grid{gemma4_31b::kGlobalKvHeadCount, split_count};
  causal_gqa_attention_cached_m1_fused_global_compact_scalar_split_kernel<
      __half, false><<<split_grid, kThreads>>>(
      query, compact_kv_cache, nullptr, k_norm_scale, 0, 0,
      absolute_position, capacity, split_count, partial_context,
      storage.split_maximum, storage.split_denominator);
  check_cuda(cudaGetLastError(),
             "global compact scalar half reference split kernel launch");

  constexpr std::uint32_t kDimensionTile =
      kHeadSize / kGraphAttentionFusedFinalizeTiles;
  constexpr dim3 kFinalizeGrid{gemma4_31b::kQueryHeadCount,
                               kGraphAttentionFusedFinalizeTiles};
  if (split_count >= kRuntimeGlobalCoarseMinimumSplitCount) {
    const RuntimeGlobalCoarseScratchLayout coarse =
        runtime_global_coarse_scratch_layout(scratch, storage.bytes,
                                             split_count);
    const dim3 coarse_grid{gemma4_31b::kQueryHeadCount,
                           coarse.split_count};
    causal_gqa_attention_cached_m1_fused_coarse_reduce_kernel<__half,
                                                              kHeadSize>
        <<<coarse_grid, kThreads>>>(
            partial_context, storage.split_maximum,
            storage.split_denominator, split_count, coarse.split_count,
            coarse.partial_context, coarse.split_maximum,
            coarse.split_denominator);
    check_cuda(cudaGetLastError(),
               "global compact scalar half reference coarse launch");
    const std::size_t shared_bytes =
        static_cast<std::size_t>(coarse.split_count) * sizeof(float);
    causal_gqa_attention_cached_m1_fused_finalize_kernel<
        float, kHeadSize, kDimensionTile>
        <<<kFinalizeGrid, kDimensionTile, shared_bytes>>>(
            coarse.partial_context, coarse.split_maximum,
            coarse.split_denominator, coarse.split_count, context);
  } else {
    const std::size_t shared_bytes =
        static_cast<std::size_t>(split_count) * sizeof(float);
    causal_gqa_attention_cached_m1_fused_finalize_kernel<
        __half, kHeadSize, kDimensionTile>
        <<<kFinalizeGrid, kDimensionTile, shared_bytes>>>(
            partial_context, storage.split_maximum,
            storage.split_denominator, split_count, context);
  }
  check_cuda(cudaGetLastError(),
             "global compact scalar half reference finalize kernel launch");
}

void launch_global_compact_tensor_half_test(
    const BFloat16* query, const BFloat16* compact_kv_cache,
    const BFloat16* k_norm_scale, std::uint32_t absolute_position,
    std::uint32_t capacity, void* scratch, BFloat16* context) {
  constexpr std::uint32_t kHeadSize = gemma4_31b::kGlobalHeadSize;
  const std::uint32_t split_count =
      (absolute_position + 1 + kRuntimeAttentionTokensPerSplit - 1) /
      kRuntimeAttentionTokensPerSplit;
  if (split_count >= kRuntimeGlobalCoarseMinimumSplitCount) {
    fail("global compact tensor half test",
         "test helper does not cover hierarchical reduction");
  }
  const RuntimeFusedScratchLayout storage = runtime_fused_scratch_layout(
      scratch, split_count, kHeadSize, true);
  auto* const partial_context =
      static_cast<__half*>(storage.partial_context);
  const dim3 split_grid{gemma4_31b::kGlobalKvHeadCount, split_count};
  causal_gqa_attention_cached_m1_fused_global_compact_tensor_split_kernel<
      false>
      <<<split_grid, kThreads>>>(
      query, compact_kv_cache, nullptr, k_norm_scale, 0, 0,
      absolute_position, capacity, split_count, partial_context,
      storage.split_maximum, storage.split_denominator);
  check_cuda(cudaGetLastError(),
             "global compact tensor half test split kernel launch");

  constexpr std::uint32_t kDimensionTile =
      kHeadSize / kGraphAttentionFusedFinalizeTiles;
  constexpr dim3 kFinalizeGrid{gemma4_31b::kQueryHeadCount,
                               kGraphAttentionFusedFinalizeTiles};
  const std::size_t shared_bytes =
      static_cast<std::size_t>(split_count) * sizeof(float);
  causal_gqa_attention_cached_m1_fused_finalize_kernel<
      __half, kHeadSize, kDimensionTile>
      <<<kFinalizeGrid, kDimensionTile, shared_bytes>>>(
          partial_context, storage.split_maximum,
          storage.split_denominator, split_count, context);
  check_cuda(cudaGetLastError(),
             "global compact tensor half test finalize kernel launch");
}

struct PagedCompactFixture {
  std::vector<BFloat16> pool;
  std::vector<std::uint64_t> offsets;
  std::uint32_t page_tokens{};
  std::size_t page_stride_elements{};
  std::size_t layer_offset_elements{};
};

PagedCompactFixture make_paged_compact_fixture(
    const std::vector<BFloat16>& compact_cache,
    std::uint32_t cache_capacity) {
  constexpr std::uint32_t kPageTokens = 256;
  constexpr std::size_t kLayerOffsetElements = 37;
  constexpr std::size_t kPageTrailerElements = 19;
  constexpr std::uint32_t kKvHeads = gemma4_31b::kGlobalKvHeadCount;
  const std::uint32_t page_count =
      (cache_capacity + kPageTokens - 1) / kPageTokens;
  const std::size_t page_stride_elements =
      kLayerOffsetElements +
      static_cast<std::size_t>(kKvHeads) * kPageTokens *
          kGlobalCompactKvSize +
      kPageTrailerElements;
  PagedCompactFixture fixture;
  fixture.pool.assign(
      static_cast<std::size_t>(page_count) * page_stride_elements,
      to_bf16(-13.0F));
  fixture.offsets.resize(page_count);
  fixture.page_tokens = kPageTokens;
  fixture.page_stride_elements = page_stride_elements;
  fixture.layer_offset_elements = kLayerOffsetElements;
  for (std::uint32_t page = 0; page < page_count; ++page) {
    fixture.offsets[page] =
        static_cast<std::uint64_t>(page_count - page - 1) *
        page_stride_elements;
  }
  for (std::uint32_t position = 0; position < cache_capacity; ++position) {
    const std::uint32_t page = position / kPageTokens;
    const std::uint32_t page_position = position % kPageTokens;
    for (std::uint32_t head = 0; head < kKvHeads; ++head) {
      const std::size_t source =
          (static_cast<std::size_t>(head) * cache_capacity + position) *
          kGlobalCompactKvSize;
      const std::size_t destination =
          fixture.offsets[page] + kLayerOffsetElements +
          (static_cast<std::size_t>(head) * kPageTokens + page_position) *
              kGlobalCompactKvSize;
      std::copy_n(compact_cache.begin() + source, kGlobalCompactKvSize,
                  fixture.pool.begin() + destination);
    }
  }
  return fixture;
}

void test_runtime_compact_global_reference(std::ostream& report) {
  constexpr std::uint32_t kCapacity = 1'025;
  constexpr std::uint32_t kHeadSize = gemma4_31b::kGlobalHeadSize;
  constexpr std::uint32_t kKvHeads = gemma4_31b::kGlobalKvHeadCount;
  constexpr std::array<std::uint32_t, 9> kPositions{
      0, 14, 15, 16, 30, 31, 32, 1'023, 1'024};
  const std::size_t query_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * kHeadSize;
  const std::size_t cache_elements =
      static_cast<std::size_t>(kKvHeads) * kCapacity * kHeadSize;
  const std::size_t compact_elements =
      static_cast<std::size_t>(kKvHeads) * kCapacity *
      kGlobalCompactKvSize;

  const std::vector<BFloat16> query =
      pattern(query_elements, 37, 18, 1.0F / 128.0F);
  std::vector<BFloat16> scale(kHeadSize);
  for (std::uint32_t dimension = 0; dimension < kHeadSize; ++dimension) {
    scale[dimension] = to_bf16(
        0.5F + static_cast<float>(dimension % 17) / 16.0F);
  }

  std::vector<BFloat16> key_cache(cache_elements);
  std::vector<BFloat16> value_cache(cache_elements);
  std::vector<BFloat16> compact_cache(compact_elements);
  for (std::uint32_t head = 0; head < kKvHeads; ++head) {
    for (std::uint32_t token = 0; token < kCapacity; ++token) {
      const std::size_t full_row =
          (static_cast<std::size_t>(head) * kCapacity + token) * kHeadSize;
      const std::size_t compact_row =
          (static_cast<std::size_t>(head) * kCapacity + token) *
          kGlobalCompactKvSize;
      for (std::uint32_t dimension = 0; dimension < kHeadSize;
           ++dimension) {
        const int value_integer = static_cast<int>(
                                      (13 * token + 7 * head +
                                       3 * dimension) %
                                      63) -
                                  31;
        const BFloat16 value =
            to_bf16(static_cast<float>(value_integer) / 128.0F);
        const bool rotated = dimension < 64 ||
                             (dimension >= 256 && dimension < 320);
        const int key_integer = static_cast<int>(
                                    (5 * token + 11 * head +
                                     7 * dimension) %
                                    47) -
                                23;
        const BFloat16 key =
            rotated
                ? to_bf16(static_cast<float>(key_integer) / 64.0F)
                : to_bf16(to_float(value) * to_float(scale[dimension]));
        key_cache[full_row + dimension] = key;
        value_cache[full_row + dimension] = value;
        compact_cache[compact_row + kGlobalCompactKeySize + dimension] =
            value;
        if (dimension < 64) {
          compact_cache[compact_row + dimension] = key;
        } else if (dimension >= 256 && dimension < 320) {
          compact_cache[compact_row + 64 + dimension - 256] = key;
        }
      }
    }
  }
  const PagedCompactFixture paged =
      make_paged_compact_fixture(compact_cache, kCapacity);

  DeviceBuffer<BFloat16> device_query(query_elements);
  DeviceBuffer<BFloat16> device_scale(scale.size());
  DeviceBuffer<BFloat16> device_key_cache(cache_elements);
  DeviceBuffer<BFloat16> device_value_cache(cache_elements);
  DeviceBuffer<BFloat16> device_compact_cache(compact_elements);
  DeviceBuffer<BFloat16> device_paged_cache(paged.pool.size());
  DeviceBuffer<std::uint64_t> device_page_offsets(paged.offsets.size());
  DeviceBuffer<BFloat16> device_key(kKvHeads * kHeadSize);
  DeviceBuffer<BFloat16> device_value(kKvHeads * kHeadSize);
  DeviceBuffer<BFloat16> device_separate_context(query_elements);
  DeviceBuffer<BFloat16> device_compact_context(query_elements);
  DeviceBuffer<BFloat16> device_compact_scalar_context(query_elements);
  DeviceBuffer<BFloat16> device_paged_context(query_elements);
  DeviceBuffer<BFloat16> device_separate_fp32_context(query_elements);
  DeviceBuffer<BFloat16> device_compact_fp32_context(query_elements);
  const std::size_t scratch_bytes =
      causal_gqa_attention_cached_m1_fused_scratch_bytes(
          kCapacity, gemma4_31b::AttentionKind::global);
  DeviceBuffer<float> device_scratch(scratch_bytes / sizeof(float));
  const std::size_t maximum_split_count =
      (kCapacity + kRuntimeAttentionTokensPerSplit - 1) /
      kRuntimeAttentionTokensPerSplit;
  const std::size_t fp32_scratch_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) *
      maximum_split_count * (kHeadSize + 2);
  DeviceBuffer<float> device_fp32_scratch(fp32_scratch_elements);

  device_query.copy_from(query);
  device_scale.copy_from(scale);
  device_key_cache.copy_from(key_cache);
  device_value_cache.copy_from(value_cache);
  device_compact_cache.copy_from(compact_cache);
  device_paged_cache.copy_from(paged.pool);
  device_page_offsets.copy_from(paged.offsets);
  const compact_global_cache::PagedView<BFloat16> paged_view{
      device_paged_cache.get(), device_page_offsets.get(),
      paged.page_tokens, static_cast<std::uint32_t>(paged.offsets.size()),
      paged.page_stride_elements, paged.layer_offset_elements};

  // Idempotent writes into nonuniform caches catch both row-stride and
  // 31/32/1023/1024 boundary indexing mistakes without a second fixture.
  for (const std::uint32_t position : kPositions) {
    std::vector<BFloat16> key(kKvHeads * kHeadSize);
    std::vector<BFloat16> value(kKvHeads * kHeadSize);
    for (std::uint32_t head = 0; head < kKvHeads; ++head) {
      const std::size_t source =
          (static_cast<std::size_t>(head) * kCapacity + position) *
          kHeadSize;
      const std::size_t destination =
          static_cast<std::size_t>(head) * kHeadSize;
      std::copy_n(key_cache.begin() + source, kHeadSize,
                  key.begin() + destination);
      std::copy_n(value_cache.begin() + source, kHeadSize,
                  value.begin() + destination);
    }
    device_key.copy_from(key);
    device_value.copy_from(value);
    write_kv_cache_m1_global_compact(
        device_key.get(), device_value.get(), device_compact_cache.get(),
        position, kCapacity);
  }
  report_metric(report, "write_kv_cache_m1_global_compact_layout",
                measure(device_compact_cache.copy_to_host(), compact_cache),
                0.0, true);

  bool folded_recipe_exercised = false;
  double maximum_old_reconstruction_drift = 0.0;
  double maximum_half_partial_drift = 0.0;
  double maximum_compact_half_partial_drift = 0.0;
  double maximum_compact_tensor_drift = 0.0;
  for (const std::uint32_t position : kPositions) {
    causal_gqa_attention_cached_m1_fused(
        device_query.get(), device_key_cache.get(), device_value_cache.get(),
        position, kCapacity, device_scratch.get(),
        device_separate_context.get(),
        gemma4_31b::AttentionKind::global);
    launch_global_compact_tensor_half_test(
        device_query.get(), device_compact_cache.get(), device_scale.get(),
        position, kCapacity, device_scratch.get(),
        device_compact_context.get());
    launch_global_compact_scalar_half_reference(
        device_query.get(), device_compact_cache.get(), device_scale.get(),
        position, kCapacity, device_scratch.get(),
        device_compact_scalar_context.get());
    causal_gqa_attention_cached_m1_fused_global_compact_paged(
        device_query.get(), paged_view, device_scale.get(), position,
        device_scratch.get(), device_paged_context.get());
    launch_global_fp32_partial_reference(
        device_query.get(), device_key_cache.get(),
        device_value_cache.get(), position, kCapacity,
        device_fp32_scratch.get(), device_separate_fp32_context.get());
    launch_global_compact_fp32_partial_reference(
        device_query.get(), device_compact_cache.get(), device_scale.get(),
        position, kCapacity, device_fp32_scratch.get(),
        device_compact_fp32_context.get());
    const std::vector<BFloat16> separate_context =
        device_separate_context.copy_to_host();
    const std::vector<BFloat16> separate_fp32_context =
        device_separate_fp32_context.copy_to_host();
    const std::vector<BFloat16> compact_fp32_context =
        device_compact_fp32_context.copy_to_host();
    const std::vector<BFloat16> folded_reference =
        compact_global_folded_query_reference(
            query, compact_cache, scale, position, kCapacity);
    const std::vector<BFloat16> compact_context =
        device_compact_context.copy_to_host();
    const std::vector<BFloat16> compact_scalar_context =
        device_compact_scalar_context.copy_to_host();
    const std::vector<BFloat16> paged_context =
        device_paged_context.copy_to_host();
    const std::string suffix = "_position_" + std::to_string(position);
    const auto require_finite = [&](const std::vector<BFloat16>& values,
                                    std::string_view name) {
      if (!std::all_of(values.begin(), values.end(), [](BFloat16 value) {
            return std::isfinite(to_float(value));
          })) {
        fail(name, "CUDA result contains a non-finite value");
      }
    };
    require_finite(separate_context, "global FP16 partial context");
    require_finite(compact_context,
                   "global compact FP16 partial context");
    require_finite(compact_scalar_context,
                   "global compact scalar FP16 partial context");
    require_finite(paged_context,
                   "paged global compact FP16 partial context");
    const Metric half_partial_drift =
        measure(separate_context, separate_fp32_context);
    const Metric compact_half_partial_drift =
        measure(compact_context, compact_fp32_context);
    maximum_half_partial_drift =
        std::max(maximum_half_partial_drift, half_partial_drift.max_abs);
    maximum_compact_half_partial_drift = std::max(
        maximum_compact_half_partial_drift,
        compact_half_partial_drift.max_abs);
    const Metric compact_tensor_drift =
        measure(compact_context, compact_scalar_context);
    maximum_compact_tensor_drift = std::max(
        maximum_compact_tensor_drift, compact_tensor_drift.max_abs);
    report_metric(report,
                  "causal_gqa_attention_cached_m1_fused_fp16_partial_drift" +
                      suffix,
                  half_partial_drift, 1.0 / 128.0, false);
    report_metric(
        report,
        "causal_gqa_attention_cached_m1_fused_global_compact_fp16_partial_"
        "drift" + suffix,
        compact_half_partial_drift, 1.0 / 128.0, false);
    report_metric(
        report,
        "causal_gqa_attention_cached_m1_fused_global_compact_tensor_scalar_"
        "drift" + suffix,
        compact_tensor_drift, 1.0 / 128.0, false);
    report_metric(
        report,
        "causal_gqa_attention_cached_m1_fused_global_compact_paged" +
            suffix,
        measure(paged_context, compact_context), 0.0, true);
    report_metric(report,
                  "causal_gqa_attention_cached_m1_fused_global_compact_"
                  "folded_query_reference" + suffix,
                  measure(compact_context, folded_reference), 1.0 / 128.0,
                  false);
    const Metric old_drift = measure(compact_context, separate_context);
    folded_recipe_exercised =
        folded_recipe_exercised || old_drift.bit_mismatches != 0;
    maximum_old_reconstruction_drift =
        std::max(maximum_old_reconstruction_drift, old_drift.max_abs);
    report << "primitive causal_gqa_attention_cached_m1_fused_global_"
              "compact_old_reconstruction_drift"
           << suffix << ": elements=" << old_drift.elements
           << " bit_mismatches=" << old_drift.bit_mismatches
           << " max_abs=" << std::setprecision(9) << old_drift.max_abs
           << " rmse=" << old_drift.rmse << " comparison=observation\n";
  }
  if (!folded_recipe_exercised) {
    fail("causal_gqa_attention_cached_m1_fused_global_compact",
         "fixture does not distinguish folded-Q from reconstructed BF16 K");
  }

  report_metric(report,
                "causal_gqa_attention_cached_m1_fused_global_compact_"
                "cache_read_only",
                measure(device_compact_cache.copy_to_host(), compact_cache),
                0.0, true);
  report_metric(report,
                "causal_gqa_attention_cached_m1_fused_global_compact_paged_"
                "cache_read_only",
                measure(device_paged_cache.copy_to_host(), paged.pool), 0.0,
                true);

  report << "primitive runtime_compact_global_reference_contract: "
            "visible_tokens=1,15,16,17,31,32,33,1024,1025 "
            "paged_exact=1 "
            "compact_folded_query_reference=1 compact_row=128+512 "
            "fp16_partial_max_abs="
         << std::setprecision(9) << maximum_half_partial_drift
         << " compact_fp16_partial_max_abs="
         << maximum_compact_half_partial_drift
         << " compact_tensor_scalar_max_abs="
         << maximum_compact_tensor_drift << " "
            "old_reconstruction_max_abs="
         << std::setprecision(9) << maximum_old_reconstruction_drift
         << " ok\n";
}

void test_runtime_global_compact_tensor_constant_value(
    std::ostream& report) {
  constexpr std::uint32_t kCapacity = 33;
  constexpr std::uint32_t kHeadSize = gemma4_31b::kGlobalHeadSize;
  constexpr std::uint32_t kKvHeads = gemma4_31b::kGlobalKvHeadCount;
  constexpr std::array<std::uint32_t, 7> kVisibleTokens{
      1, 15, 16, 17, 31, 32, 33};
  const std::size_t query_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * kHeadSize;
  const std::size_t cache_elements =
      static_cast<std::size_t>(kKvHeads) * kCapacity *
      kGlobalCompactKvSize;
  const BFloat16 zero = to_bf16(0.0F);
  const BFloat16 one = to_bf16(1.0F);
  const BFloat16 constant = to_bf16(0.25F);
  const std::vector<BFloat16> query(query_elements, zero);
  const std::vector<BFloat16> scale(kHeadSize, one);
  std::vector<BFloat16> compact_cache(cache_elements, zero);
  for (std::uint32_t head = 0; head < kKvHeads; ++head) {
    for (std::uint32_t token = 0; token < kCapacity; ++token) {
      const std::size_t row =
          (static_cast<std::size_t>(head) * kCapacity + token) *
          kGlobalCompactKvSize;
      std::fill_n(compact_cache.begin() + row + kGlobalCompactKeySize,
                  kHeadSize, constant);
    }
  }
  const std::vector<BFloat16> expected(query_elements, constant);

  DeviceBuffer<BFloat16> device_query(query_elements);
  DeviceBuffer<BFloat16> device_scale(scale.size());
  DeviceBuffer<BFloat16> device_cache(cache_elements);
  DeviceBuffer<BFloat16> device_context(query_elements);
  const std::size_t scratch_bytes =
      causal_gqa_attention_cached_m1_fused_scratch_bytes(
          kCapacity, gemma4_31b::AttentionKind::global);
  DeviceBuffer<float> device_scratch(scratch_bytes / sizeof(float));
  device_query.copy_from(query);
  device_scale.copy_from(scale);
  device_cache.copy_from(compact_cache);

  for (const std::uint32_t visible_tokens : kVisibleTokens) {
    launch_global_compact_tensor_half_test(
        device_query.get(), device_cache.get(), device_scale.get(),
        visible_tokens - 1, kCapacity, device_scratch.get(),
        device_context.get());
    report_metric(
        report,
        "causal_gqa_attention_cached_m1_fused_global_compact_tensor_"
        "constant_value_visible_" +
            std::to_string(visible_tokens),
        measure(device_context.copy_to_host(), expected), 0.0, true);
  }
  report_metric(
      report,
      "causal_gqa_attention_cached_m1_fused_global_compact_tensor_"
      "constant_value_cache_read_only",
      measure(device_cache.copy_to_host(), compact_cache), 0.0, true);
  report << "primitive runtime_global_compact_tensor_contract: "
            "visible_tokens=1,15,16,17,31,32,33 constant_value_exact=1 "
            "rounded_probability_denominator=1 padded_rows_zero=1 "
            "cache_row=640 read_only=1 ok\n";
}

void test_runtime_global_compact_tensor_dispatch_nonuniform(
    std::ostream& report) {
  constexpr std::uint32_t kCapacity =
      (kRuntimeGlobalCoarseMinimumSplitCount - 1) *
          kRuntimeAttentionTokensPerSplit +
      1;
  constexpr std::uint32_t kPosition = kCapacity - 1;
  constexpr std::uint32_t kHeadSize = gemma4_31b::kGlobalHeadSize;
  constexpr std::uint32_t kKvHeads = gemma4_31b::kGlobalKvHeadCount;
  constexpr std::uint32_t kPatternTokens =
      kRuntimeAttentionTokensPerSplit;
  static_assert((kCapacity + kRuntimeAttentionTokensPerSplit - 1) /
                    kRuntimeAttentionTokensPerSplit ==
                kRuntimeGlobalCoarseMinimumSplitCount);

  const std::size_t query_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * kHeadSize;
  const std::size_t cache_elements =
      static_cast<std::size_t>(kKvHeads) * kCapacity *
      kGlobalCompactKvSize;
  const std::vector<BFloat16> query =
      pattern(query_elements, 37, 18, 1.0F / 128.0F);
  std::vector<BFloat16> scale(kHeadSize);
  for (std::uint32_t dimension = 0; dimension < kHeadSize; ++dimension) {
    scale[dimension] = to_bf16(
        0.5F + static_cast<float>(dimension % 17) / 16.0F);
  }

  std::vector<BFloat16> tile(
      static_cast<std::size_t>(kKvHeads) * kPatternTokens *
      kGlobalCompactKvSize);
  for (std::uint32_t head = 0; head < kKvHeads; ++head) {
    for (std::uint32_t token = 0; token < kPatternTokens; ++token) {
      const std::size_t row =
          (static_cast<std::size_t>(head) * kPatternTokens + token) *
          kGlobalCompactKvSize;
      for (std::uint32_t dimension = 0;
           dimension < kGlobalCompactKeySize; ++dimension) {
        const int key_integer =
            static_cast<int>((5 * token + 11 * head + 7 * dimension) % 47) -
            23;
        tile[row + dimension] =
            to_bf16(static_cast<float>(key_integer) / 64.0F);
      }
      for (std::uint32_t dimension = 0; dimension < kHeadSize;
           ++dimension) {
        const int value_integer =
            static_cast<int>((13 * token + 7 * head + 3 * dimension) % 63) -
            31;
        tile[row + kGlobalCompactKeySize + dimension] =
            to_bf16(static_cast<float>(value_integer) / 128.0F);
      }
    }
  }
  std::vector<BFloat16> compact_cache(cache_elements);
  for (std::uint32_t head = 0; head < kKvHeads; ++head) {
    for (std::uint32_t token = 0; token < kCapacity; ++token) {
      const std::size_t source =
          (static_cast<std::size_t>(head) * kPatternTokens +
           token % kPatternTokens) *
          kGlobalCompactKvSize;
      const std::size_t destination =
          (static_cast<std::size_t>(head) * kCapacity + token) *
          kGlobalCompactKvSize;
      std::copy_n(tile.begin() + source, kGlobalCompactKvSize,
                  compact_cache.begin() + destination);
    }
  }
  const PagedCompactFixture paged =
      make_paged_compact_fixture(compact_cache, kCapacity);

  DeviceBuffer<BFloat16> device_query(query_elements);
  DeviceBuffer<BFloat16> device_scale(scale.size());
  DeviceBuffer<BFloat16> device_cache(cache_elements);
  DeviceBuffer<BFloat16> device_paged_cache(paged.pool.size());
  DeviceBuffer<std::uint64_t> device_page_offsets(paged.offsets.size());
  DeviceBuffer<BFloat16> device_scalar_context(query_elements);
  DeviceBuffer<BFloat16> device_tensor_context(query_elements);
  DeviceBuffer<BFloat16> device_paged_context(query_elements);
  const std::size_t scratch_bytes =
      causal_gqa_attention_cached_m1_fused_scratch_bytes(
          kCapacity, gemma4_31b::AttentionKind::global);
  DeviceBuffer<std::uint8_t> device_scratch(scratch_bytes);
  device_query.copy_from(query);
  device_scale.copy_from(scale);
  device_cache.copy_from(compact_cache);
  device_paged_cache.copy_from(paged.pool);
  device_page_offsets.copy_from(paged.offsets);
  const compact_global_cache::PagedView<BFloat16> paged_view{
      device_paged_cache.get(), device_page_offsets.get(),
      paged.page_tokens, static_cast<std::uint32_t>(paged.offsets.size()),
      paged.page_stride_elements, paged.layer_offset_elements};

  launch_global_compact_scalar_half_reference(
      device_query.get(), device_cache.get(), device_scale.get(), kPosition,
      kCapacity, device_scratch.get(), device_scalar_context.get());
  causal_gqa_attention_cached_m1_fused_global_compact(
      device_query.get(), device_cache.get(), device_scale.get(), kPosition,
      kCapacity, device_scratch.get(), device_tensor_context.get());
  causal_gqa_attention_cached_m1_fused_global_compact_paged(
      device_query.get(), paged_view, device_scale.get(), kPosition,
      device_scratch.get(), device_paged_context.get());
  report_metric(
      report,
      "causal_gqa_attention_cached_m1_fused_global_compact_tensor_"
      "dispatch_nonuniform_context",
      measure(device_tensor_context.copy_to_host(),
              device_scalar_context.copy_to_host()),
      1.0 / 128.0, false);
  report_metric(
      report,
      "causal_gqa_attention_cached_m1_fused_global_compact_paged_tensor_"
      "dispatch_nonuniform_context",
      measure(device_paged_context.copy_to_host(),
              device_tensor_context.copy_to_host()),
      0.0, true);
  report_metric(
      report,
      "causal_gqa_attention_cached_m1_fused_global_compact_paged_tensor_"
      "dispatch_cache_read_only",
      measure(device_paged_cache.copy_to_host(), paged.pool), 0.0, true);
  report << "primitive runtime_global_compact_tensor_dispatch_contract: "
            "visible_tokens="
         << kCapacity << " fine_splits="
         << kRuntimeGlobalCoarseMinimumSplitCount
         << " nonuniform_scalar_reference=1 paged_exact=1 hierarchy=1 ok\n";
}

void test_runtime_global_coarse_reduction(std::ostream& report) {
  constexpr std::uint32_t kHeadSize = gemma4_31b::kGlobalHeadSize;
  constexpr std::uint32_t kFineSplitCount =
      2 * kRuntimeGlobalCoarseGroupSize + 1;
  constexpr std::uint32_t kCoarseSplitCount =
      (kFineSplitCount + kRuntimeGlobalCoarseGroupSize - 1) /
      kRuntimeGlobalCoarseGroupSize;
  const std::size_t fine_metadata_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) *
      kFineSplitCount;
  const std::size_t fine_partial_elements =
      fine_metadata_elements * kHeadSize;
  const std::size_t coarse_metadata_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) *
      kCoarseSplitCount;
  const std::size_t coarse_partial_elements =
      coarse_metadata_elements * kHeadSize;

  std::vector<__half> fine_partial(fine_partial_elements);
  std::vector<float> fine_maximum(fine_metadata_elements);
  std::vector<float> fine_denominator(fine_metadata_elements);
  for (std::uint32_t query_head = 0;
       query_head < gemma4_31b::kQueryHeadCount; ++query_head) {
    for (std::uint32_t split = 0; split < kFineSplitCount; ++split) {
      const std::size_t metadata_index =
          static_cast<std::size_t>(query_head) * kFineSplitCount + split;
      fine_maximum[metadata_index] =
          0.001F * static_cast<float>(query_head) -
          0.015625F * static_cast<float>(split % 17);
      fine_denominator[metadata_index] =
          1.0F + 0.125F * static_cast<float>(split % 5);
      for (std::uint32_t dimension = 0; dimension < kHeadSize;
           ++dimension) {
        const int centered = static_cast<int>(
                                 (dimension + 3 * split + query_head) % 19) -
                             9;
        fine_partial[metadata_index * kHeadSize + dimension] =
            __float2half_rn(static_cast<float>(centered) / 128.0F);
      }
    }
  }

  DeviceBuffer<__half> device_fine_partial(fine_partial_elements);
  DeviceBuffer<float> device_fine_maximum(fine_metadata_elements);
  DeviceBuffer<float> device_fine_denominator(fine_metadata_elements);
  DeviceBuffer<float> device_coarse_partial(coarse_partial_elements);
  DeviceBuffer<float> device_coarse_maximum(coarse_metadata_elements);
  DeviceBuffer<float> device_coarse_denominator(coarse_metadata_elements);
  constexpr std::size_t kContextElements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * kHeadSize;
  DeviceBuffer<BFloat16> device_flat_context(kContextElements);
  DeviceBuffer<BFloat16> device_coarse_context(kContextElements);
  device_fine_partial.copy_from(fine_partial);
  device_fine_maximum.copy_from(fine_maximum);
  device_fine_denominator.copy_from(fine_denominator);

  constexpr dim3 kFinalizeGrid{gemma4_31b::kQueryHeadCount,
                               kGraphAttentionFusedFinalizeTiles};
  constexpr std::uint32_t kDimensionTile =
      kHeadSize / kGraphAttentionFusedFinalizeTiles;
  causal_gqa_attention_cached_m1_fused_finalize_kernel<
      __half, kHeadSize, kDimensionTile>
      <<<kFinalizeGrid, kDimensionTile,
         kFineSplitCount * sizeof(float)>>>(
          device_fine_partial.get(), device_fine_maximum.get(),
          device_fine_denominator.get(), kFineSplitCount,
          device_flat_context.get());
  check_cuda(cudaGetLastError(),
             "global coarse reduction flat reference launch");

  const dim3 coarse_grid{gemma4_31b::kQueryHeadCount,
                         kCoarseSplitCount};
  causal_gqa_attention_cached_m1_fused_coarse_reduce_kernel<__half,
                                                            kHeadSize>
      <<<coarse_grid, kThreads>>>(
          device_fine_partial.get(), device_fine_maximum.get(),
          device_fine_denominator.get(), kFineSplitCount,
          kCoarseSplitCount, device_coarse_partial.get(),
          device_coarse_maximum.get(), device_coarse_denominator.get());
  check_cuda(cudaGetLastError(), "global coarse reduction kernel launch");
  causal_gqa_attention_cached_m1_fused_finalize_kernel<
      float, kHeadSize, kDimensionTile>
      <<<kFinalizeGrid, kDimensionTile,
         kCoarseSplitCount * sizeof(float)>>>(
          device_coarse_partial.get(), device_coarse_maximum.get(),
          device_coarse_denominator.get(), kCoarseSplitCount,
          device_coarse_context.get());
  check_cuda(cudaGetLastError(),
             "global coarse reduction finalizer launch");

  report_metric(report, "runtime_global_coarse_reduction_context",
                measure(device_coarse_context.copy_to_host(),
                        device_flat_context.copy_to_host()),
                1.0 / 128.0, false);
  report << "primitive runtime_global_coarse_reduction_contract: "
            "group_size="
         << kRuntimeGlobalCoarseGroupSize
         << " fine_splits=" << kFineSplitCount
         << " coarse_splits=" << kCoarseSplitCount
         << " fp32_coarse_partial=1 ok\n";
}

void test_runtime_global_coarse_dispatch_boundary(std::ostream& report) {
  constexpr std::uint32_t kHeadSize = gemma4_31b::kGlobalHeadSize;
  constexpr std::uint32_t kCapacity =
      (kRuntimeGlobalCoarseMinimumSplitCount - 1) *
          kRuntimeAttentionTokensPerSplit +
      1;
  constexpr std::uint32_t kFlatPosition = kCapacity - 2;
  constexpr std::uint32_t kCoarsePosition = kCapacity - 1;
  static_assert((kFlatPosition + 1 + kRuntimeAttentionTokensPerSplit - 1) /
                    kRuntimeAttentionTokensPerSplit ==
                kRuntimeGlobalCoarseMinimumSplitCount - 1);
  static_assert((kCoarsePosition + 1 + kRuntimeAttentionTokensPerSplit - 1) /
                    kRuntimeAttentionTokensPerSplit ==
                kRuntimeGlobalCoarseMinimumSplitCount);

  constexpr std::size_t kQueryElements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * kHeadSize;
  constexpr std::size_t kCacheElements =
      static_cast<std::size_t>(gemma4_31b::kGlobalKvHeadCount) * kCapacity *
      kGlobalCompactKvSize;
  DeviceBuffer<BFloat16> device_query(kQueryElements);
  DeviceBuffer<BFloat16> device_scale(kHeadSize);
  DeviceBuffer<BFloat16> device_cache(kCacheElements);
  DeviceBuffer<BFloat16> device_context(kQueryElements);
  const std::size_t scratch_bytes =
      causal_gqa_attention_cached_m1_fused_scratch_bytes(
          kCapacity, gemma4_31b::AttentionKind::global);
  DeviceBuffer<std::uint8_t> device_scratch(scratch_bytes);
  device_query.copy_from(
      std::vector<BFloat16>(kQueryElements, to_bf16(0.0F)));
  device_scale.copy_from(
      std::vector<BFloat16>(kHeadSize, to_bf16(1.0F)));
  check_cuda(cudaMemset(device_cache.get(), 0,
                        kCacheElements * sizeof(BFloat16)),
             "global coarse dispatch cache clear");

  for (const std::uint32_t position : {kFlatPosition, kCoarsePosition}) {
    device_context.copy_from(
        std::vector<BFloat16>(kQueryElements, to_bf16(-1.0F)));
    causal_gqa_attention_cached_m1_fused_global_compact(
        device_query.get(), device_cache.get(), device_scale.get(), position,
        kCapacity, device_scratch.get(), device_context.get());
    const std::vector<BFloat16> context = device_context.copy_to_host();
    if (!std::all_of(context.begin(), context.end(), [](BFloat16 value) {
          return bf16_bits(value) == 0;
        })) {
      fail("runtime global coarse dispatch boundary",
           "zero-value attention did not produce exact zero context");
    }
  }
  report << "primitive runtime_global_coarse_dispatch_contract: "
            "flat_position="
         << kFlatPosition << " coarse_position=" << kCoarsePosition
         << " threshold_splits="
         << kRuntimeGlobalCoarseMinimumSplitCount
         << " scratch_bytes=" << scratch_bytes << " exact_zero=1 ok\n";
}

__global__ void fill_batch_cache(BFloat16* data, std::size_t count, unsigned seed) {
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) data[index] = __float2bfloat16_rn(
      (int((index * 17 + seed * 23) % 113) - 56) * 0.015625F);
}

__global__ void compare_batch_cache(const BFloat16* actual, const BFloat16* expected,
                                    std::size_t count, unsigned* mismatch) {
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count && __bfloat16_as_ushort(actual[index]) != __bfloat16_as_ushort(expected[index]))
    atomicOr(mismatch, 1U);
}

void test_decode_attention_batch(std::ostream& report) {
  for (const bool global : {false, true}) {
    const auto kind = global ? gemma4_31b::AttentionKind::global : gemma4_31b::AttentionKind::local;
    const unsigned d = global ? 512 : 256, heads = global ? 4 : 16;
    const auto scratch_bytes = causal_gqa_attention_cached_m1_fused_scratch_bytes(
        global ? 262144 : 1024, kind) * (global ? 1 : 3);
    DeviceBuffer<std::uint8_t> scratch(scratch_bytes + 256);
    check_cuda(cudaMemset(scratch.get(), 0xCD, scratch_bytes + 256), "batch scratch guard");
    DeviceBuffer<BFloat16> scale(d);
    scale.copy_from(pattern(d, 13, 3, 0.25F));
    struct Fixture {
      std::size_t cache_elements;
      DeviceBuffer<BFloat16> q, k, v, cos, sin, qr, kr, out, saved_out, cache, saved_cache;
      DeviceBuffer<std::uint64_t> pages;
      Fixture(unsigned d, unsigned heads, unsigned capacity, bool global)
          : cache_elements(global ? (capacity / 256) * (4 * 256 * 640 + 32) + 2
                                  : 2 * 1024 * 16 * 256 + 2),
            q(32 * d), k(heads * d), v(heads * d), cos(d), sin(d), qr(32 * d),
            kr(heads * d), out(32 * d), saved_out(32 * d), cache(cache_elements),
            saved_cache(cache_elements), pages(global ? capacity / 256 : 1) {}
    };
    std::vector<std::unique_ptr<Fixture>> fixtures;
    std::vector<DecodeAttentionInput> inputs;
    constexpr unsigned positions[] = {0, 1, 31, 255, 256, 1023, 1024, 4095};
    constexpr unsigned long_positions[] = {32735, 32736, 32737, 65535, 262143};
    for (unsigned i = 0; i < 37; ++i) {
      const unsigned position = i < 32 ? positions[i % 8] : long_positions[i - 32];
      const unsigned capacity = global ? (position / 256 + 1) * 256 : 1024;
      auto owned = std::make_unique<Fixture>(d, heads, capacity, global);
      auto& f = *owned;
      f.q.copy_from(pattern(32 * d, 97 + i, 48, 0.03125F));
      f.k.copy_from(pattern(heads * d, 89 + i, 44, 0.03125F));
      f.v.copy_from(pattern(heads * d, 79 + i, 39, 0.03125F));
      f.cos.copy_from(pattern(d, 31, 15, 0.0625F));
      f.sin.copy_from(pattern(d, 29, 14, 0.0625F));
      fill_batch_cache<<<blocks_for(f.cache_elements), kThreads>>>(f.cache.get(), f.cache_elements, i);
      DecodeAttentionInput input{f.q.get(), f.k.get(), f.v.get(), f.cos.get(), f.sin.get(),
          f.qr.get(), global ? nullptr : f.cache.get() + 1,
          global ? nullptr : f.cache.get() + 1 + 1024 * 16 * 256,
          {}, position, f.out.get()};
      apply_rope_m1(f.q.get(), f.cos.get(), f.sin.get(), f.qr.get(), 32, kind);
      apply_rope_m1(f.k.get(), f.cos.get(), f.sin.get(), f.kr.get(), heads, kind);
      if (global) {
        constexpr std::size_t stride = 4 * 256 * 640 + 32;
        std::vector<std::uint64_t> offsets(capacity / 256);
        for (unsigned p = 0; p < offsets.size(); ++p) offsets[p] = (offsets.size() - 1 - p) * stride;
        f.pages.copy_from(offsets);
        input.global_cache = {f.cache.get() + 1, f.pages.get(), 256, capacity / 256, stride, 16};
        prefill_primitives::write_kv_cache_chunk_global_compact_paged(
            f.kr.get(), f.v.get(), input.global_cache, position, 1);
        causal_gqa_attention_cached_m1_fused_global_compact_paged(
            f.qr.get(), input.global_cache, scale.get(), position, scratch.get(), f.out.get());
      } else {
        write_kv_cache_m1(f.kr.get(), f.v.get(), input.local_key, input.local_value, position, 1024, kind);
        causal_gqa_attention_cached_m1_fused(f.qr.get(), input.local_key, input.local_value,
            position, 1024, scratch.get(), f.out.get(), kind);
      }
      check_cuda(cudaMemcpy(f.saved_out.get(), f.out.get(), 32 * d * sizeof(BFloat16), cudaMemcpyDeviceToDevice), "save serial context");
      check_cuda(cudaMemcpy(f.saved_cache.get(), f.cache.get(), f.cache_elements * sizeof(BFloat16), cudaMemcpyDeviceToDevice), "save serial cache");
      // The batch must perform its own cache write, including the current row.
      fill_batch_cache<<<blocks_for(f.cache_elements), kThreads>>>(f.cache.get(), f.cache_elements, i);
      inputs.push_back(input);
      fixtures.push_back(std::move(owned));
    }
    check_cuda(cudaDeviceSynchronize(), "serial decode batch reference");
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    check_cuda(cudaStreamCreate(&stream), "decode batch stream");
    check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "decode batch capture");
    decode_attention_batch(inputs, scale.get(), scratch.get(), scratch_bytes, kind, stream);
    check_cuda(cudaStreamEndCapture(stream, &graph), "decode batch capture end");
    check_cuda(cudaGraphInstantiate(&exec, graph, 0), "decode batch instantiate");
    for (unsigned replay = 0; replay < 2; ++replay)
      check_cuda(cudaGraphLaunch(exec, stream), "decode batch replay");
    check_cuda(cudaStreamSynchronize(stream), "decode batch synchronize");
    DeviceBuffer<unsigned> mismatch(1);
    mismatch.copy_from({0});
    for (const auto& f : fixtures) {
      compare_batch_cache<<<blocks_for(f->cache_elements), kThreads>>>(
          f->cache.get(), f->saved_cache.get(), f->cache_elements, mismatch.get());
      compare_batch_cache<<<blocks_for(32 * d), kThreads>>>(
          f->out.get(), f->saved_out.get(), 32 * d, mismatch.get());
    }
    if (mismatch.copy_to_host()[0]) fail("decode batch", "serial context or cache mismatch");
    fixtures[0]->q.copy_from(std::vector<BFloat16>(32 * d, to_bf16(0.25F)));
    check_cuda(cudaGraphLaunch(exec, stream), "decode batch isolation replay");
    check_cuda(cudaStreamSynchronize(stream), "decode batch isolation synchronize");
    for (unsigned i = 1; i < fixtures.size(); ++i)
      compare_batch_cache<<<blocks_for(32 * d), kThreads>>>(fixtures[i]->out.get(),
          fixtures[i]->saved_out.get(), 32 * d, mismatch.get());
    if (mismatch.copy_to_host()[0]) fail("decode batch", "request isolation failed");
    std::vector<std::uint8_t> guard(256);
    check_cuda(cudaMemcpy(guard.data(), scratch.get() + scratch_bytes, 256, cudaMemcpyDeviceToHost), "decode scratch guard");
    if (!std::all_of(guard.begin(), guard.end(), [](auto b) { return b == 0xCD; }))
      fail("decode batch", "scratch guard overwritten");
    inputs.back().position = 262144;
    bool rejected = false;
    try { decode_attention_batch(inputs, scale.get(), scratch.get(), scratch_bytes, kind, stream); }
    catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) fail("decode batch", "accepted invalid last request");
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
    report << "batched decode attention global=" << global
           << " requests=37 serial=exact cache=exact isolation=exact cold_graph=passed scratch_guard=passed\n";
  }
}

void test_runtime_global_fp16_partial_range(std::ostream& report) {
  constexpr std::uint32_t kCapacity = 32;
  constexpr std::uint32_t kPosition = kCapacity - 1;
  constexpr std::uint32_t kHeadSize = gemma4_31b::kGlobalHeadSize;
  constexpr float kValue = 2'000.0F;
  constexpr float kExpectedPartial = kCapacity * kValue;
  static_assert(kExpectedPartial < 65'504.0F);
  const std::size_t query_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * kHeadSize;
  const std::size_t cache_elements =
      static_cast<std::size_t>(gemma4_31b::kGlobalKvHeadCount) * kCapacity *
      kHeadSize;
  const std::vector<BFloat16> query(query_elements, to_bf16(0.0F));
  const std::vector<BFloat16> key_cache(cache_elements, to_bf16(0.0F));
  const std::vector<BFloat16> value_cache(cache_elements,
                                          to_bf16(kValue));

  DeviceBuffer<BFloat16> device_query(query_elements);
  DeviceBuffer<BFloat16> device_key_cache(cache_elements);
  DeviceBuffer<BFloat16> device_value_cache(cache_elements);
  DeviceBuffer<BFloat16> device_context(query_elements);
  DeviceBuffer<BFloat16> device_fp32_context(query_elements);
  const std::size_t scratch_bytes =
      causal_gqa_attention_cached_m1_fused_scratch_bytes(
          kCapacity, gemma4_31b::AttentionKind::global);
  DeviceBuffer<__half> device_scratch(scratch_bytes / sizeof(__half));
  constexpr std::size_t kPartialElements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * kHeadSize;
  constexpr std::size_t kFp32ScratchElements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) *
      (kHeadSize + 2);
  DeviceBuffer<float> device_fp32_scratch(kFp32ScratchElements);
  device_query.copy_from(query);
  device_key_cache.copy_from(key_cache);
  device_value_cache.copy_from(value_cache);

  causal_gqa_attention_cached_m1_fused(
      device_query.get(), device_key_cache.get(), device_value_cache.get(),
      kPosition, kCapacity, device_scratch.get(), device_context.get(),
      gemma4_31b::AttentionKind::global);
  launch_global_fp32_partial_reference(
      device_query.get(), device_key_cache.get(), device_value_cache.get(),
      kPosition, kCapacity, device_fp32_scratch.get(),
      device_fp32_context.get());

  const std::vector<__half> scratch = device_scratch.copy_to_host();
  float maximum_absolute_partial = 0.0F;
  for (std::size_t index = 0; index < kPartialElements; ++index) {
    const float value = __half2float(scratch[index]);
    if (!std::isfinite(value)) {
      fail("runtime global FP16 partial range",
           "partial numerator is non-finite");
    }
    maximum_absolute_partial =
        std::max(maximum_absolute_partial, std::abs(value));
  }
  if (maximum_absolute_partial != kExpectedPartial) {
    fail("runtime global FP16 partial range",
         "unexpected near-limit partial numerator");
  }
  const std::vector<BFloat16> context = device_context.copy_to_host();
  if (!std::all_of(context.begin(), context.end(), [](BFloat16 value) {
        return std::isfinite(to_float(value));
      })) {
    fail("runtime global FP16 partial range",
         "final context is non-finite");
  }
  report_metric(report, "runtime_global_fp16_partial_range_context",
                measure(context, device_fp32_context.copy_to_host()), 0.0,
                true);
  report << "primitive runtime_global_fp16_partial_range_contract: "
            "tokens=32 value=2000 maximum_partial="
         << maximum_absolute_partial << " fp16_max=65504 finite=1 ok\n";
}

void test_graph_device_lookup_and_rope(std::ostream& report) {
  static_assert(kGraphPromptTokenCount == 1'024);
  static_assert(kGraphOutputTokenCount == 512);
  static_assert(kGraphAttentionPositionCount == 1'536);
  static_assert(kGraphDecodeFirstPosition == 1'024);
  static_assert(kGraphDecodeLastPosition == 1'534);
  static_assert(kGraphDecodeFinalPosition == 1'535);

  constexpr std::uint32_t kEmbeddingRows = 8;
  const std::vector<BFloat16> table = pattern(
      static_cast<std::size_t>(kEmbeddingRows) * gemma4_31b::kHiddenSize,
      127, 63, 1.0F / 128.0F);
  DeviceBuffer<BFloat16> device_table(table.size());
  DeviceBuffer<BFloat16> device_scalar_embedding(gemma4_31b::kHiddenSize);
  DeviceBuffer<BFloat16> device_indirect_embedding(gemma4_31b::kHiddenSize);
  DeviceBuffer<std::uint32_t> device_token(1);
  device_table.copy_from(table);
  for (const std::uint32_t token : {2U, 7U}) {
    device_token.copy_from({token});
    embedding_lookup(device_table.get(), token, device_scalar_embedding.get(),
                     nullptr);
    embedding_lookup_device_token(device_table.get(), device_token.get(),
                                  device_indirect_embedding.get(), nullptr);
    report_metric(
        report,
        "embedding_lookup_device_token_equivalence_token_" +
            std::to_string(token),
        measure(device_indirect_embedding.copy_to_host(),
                device_scalar_embedding.copy_to_host()),
        0.0, true);
  }

  DeviceBuffer<BFloat16> device_local_cos(gemma4_31b::kLocalHeadSize);
  DeviceBuffer<BFloat16> device_local_sin(gemma4_31b::kLocalHeadSize);
  DeviceBuffer<BFloat16> device_global_cos(gemma4_31b::kGlobalHeadSize);
  DeviceBuffer<BFloat16> device_global_sin(gemma4_31b::kGlobalHeadSize);
  DeviceBuffer<BFloat16> device_scalar_local_cos(
      gemma4_31b::kLocalHeadSize);
  DeviceBuffer<BFloat16> device_scalar_local_sin(
      gemma4_31b::kLocalHeadSize);
  DeviceBuffer<BFloat16> device_scalar_global_cos(
      gemma4_31b::kGlobalHeadSize);
  DeviceBuffer<BFloat16> device_scalar_global_sin(
      gemma4_31b::kGlobalHeadSize);
  DeviceBuffer<std::uint32_t> device_position(1);

  for (const std::uint32_t position : {1'024U, 1'025U}) {
    device_position.copy_from({position});
    generate_rope_factors_m1(
        device_scalar_local_cos.get(), device_scalar_local_sin.get(),
        device_scalar_global_cos.get(), device_scalar_global_sin.get(),
        position, nullptr);
    generate_rope_factors_m1_device_position(
        device_local_cos.get(), device_local_sin.get(),
        device_global_cos.get(), device_global_sin.get(),
        device_position.get(), nullptr);
    const std::string suffix = "_position_" + std::to_string(position);
    report_metric(report, "rope_device_position_local_cos" + suffix,
                  measure(device_local_cos.copy_to_host(),
                          device_scalar_local_cos.copy_to_host()),
                  0.0, true);
    report_metric(report, "rope_device_position_local_sin" + suffix,
                  measure(device_local_sin.copy_to_host(),
                          device_scalar_local_sin.copy_to_host()),
                  0.0, true);
    report_metric(report, "rope_device_position_global_cos" + suffix,
                  measure(device_global_cos.copy_to_host(),
                          device_scalar_global_cos.copy_to_host()),
                  0.0, true);
    report_metric(report, "rope_device_position_global_sin" + suffix,
                  measure(device_global_sin.copy_to_host(),
                          device_scalar_global_sin.copy_to_host()),
                  0.0, true);
  }

  device_position.copy_from({kGraphDecodeLastPosition});
  generate_rope_factors_m1_device_position(
      device_local_cos.get(), device_local_sin.get(),
      device_global_cos.get(), device_global_sin.get(), device_position.get(),
      nullptr);
  const RopeFactorReference local_reference = rope_factor_row_reference(
      gemma4_31b::kLocalHeadSize, gemma4_31b::kLocalHeadSize / 2, 10'000.0F,
      kGraphDecodeLastPosition);
  const RopeFactorReference global_reference = rope_factor_row_reference(
      gemma4_31b::kGlobalHeadSize, 64, 1'000'000.0F,
      kGraphDecodeLastPosition);
  report_metric(report, "rope_device_position_local_cos_position_1534_host",
                measure(device_local_cos.copy_to_host(),
                        local_reference.cosine),
                0.0, true);
  report_metric(report, "rope_device_position_local_sin_position_1534_host",
                measure(device_local_sin.copy_to_host(), local_reference.sine),
                0.0, true);
  report_metric(report,
                "rope_device_position_global_cos_position_1534_host",
                measure(device_global_cos.copy_to_host(),
                        global_reference.cosine),
                0.0, true);
  report_metric(report,
                "rope_device_position_global_sin_position_1534_host",
                measure(device_global_sin.copy_to_host(),
                        global_reference.sine),
                0.0, true);
  report << "primitive graph_device_lookup_rope_contract: "
            "embedding_tokens=2,7 positions=1024,1025 scalar_exact=1 "
            "position_1534_host_exact=1 ok\n";
}

void test_graph_attention_kind(std::ostream& report,
                               gemma4_31b::AttentionKind kind,
                               std::string_view label) {
  const bool global = kind == gemma4_31b::AttentionKind::global;
  const std::uint32_t graph_capacity =
      global ? kGraphAttentionGlobalCapacity
             : kGraphAttentionLocalCapacity;
  const std::uint32_t boundary_capacity =
      global ? kCachedAttentionM1BoundaryGlobalCapacity
             : kCachedAttentionM1BoundaryLocalCapacity;
  const std::uint32_t kv_heads = global ? gemma4_31b::kGlobalKvHeadCount
                                        : gemma4_31b::kLocalKvHeadCount;
  const std::uint32_t head_size = global ? gemma4_31b::kGlobalHeadSize
                                         : gemma4_31b::kLocalHeadSize;
  const std::size_t query_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) * head_size;
  const std::size_t kv_elements =
      static_cast<std::size_t>(kv_heads) * head_size;
  const std::size_t graph_cache_elements =
      static_cast<std::size_t>(kv_heads) * graph_capacity * head_size;
  const std::size_t boundary_cache_elements =
      static_cast<std::size_t>(kv_heads) * boundary_capacity * head_size;
  const std::size_t graph_probability_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) *
      kGraphAttentionPositionCount;
  const std::size_t boundary_probability_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) *
      kCachedAttentionM1BoundaryPositionCount;
  static_assert(kGraphAttentionScoreScratchBytes ==
                static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) *
                    kGraphAttentionPositionCount * sizeof(BFloat16));

  DeviceBuffer<BFloat16> device_query(query_elements);
  DeviceBuffer<BFloat16> device_key(kv_elements);
  DeviceBuffer<BFloat16> device_value(kv_elements);
  DeviceBuffer<BFloat16> device_graph_key_cache(graph_cache_elements);
  DeviceBuffer<BFloat16> device_graph_value_cache(graph_cache_elements);
  DeviceBuffer<BFloat16> device_boundary_key_cache(boundary_cache_elements);
  DeviceBuffer<BFloat16> device_boundary_value_cache(
      boundary_cache_elements);
  DeviceBuffer<BFloat16> device_graph_score(graph_probability_elements);
  DeviceBuffer<BFloat16> device_graph_probabilities(
      graph_probability_elements);
  DeviceBuffer<BFloat16> device_graph_context(query_elements);
  DeviceBuffer<float> device_fused_scratch(
      kGraphAttentionFusedScratchBytes / sizeof(float));
  DeviceBuffer<BFloat16> device_fused_context(query_elements);
  const std::size_t runtime_scratch_bytes =
      causal_gqa_attention_cached_m1_fused_scratch_bytes(graph_capacity,
                                                          kind);
  DeviceBuffer<float> device_runtime_scratch(runtime_scratch_bytes /
                                              sizeof(float));
  DeviceBuffer<BFloat16> device_runtime_context(query_elements);
  DeviceBuffer<BFloat16> device_boundary_score(
      boundary_probability_elements);
  DeviceBuffer<BFloat16> device_boundary_probabilities(
      boundary_probability_elements);
  DeviceBuffer<BFloat16> device_boundary_context(query_elements);
  DeviceBuffer<std::uint32_t> device_position(1);

  std::vector<BFloat16> graph_key_cache;
  std::vector<BFloat16> graph_value_cache;
  std::vector<BFloat16> boundary_key_cache;
  std::vector<BFloat16> boundary_value_cache;
  initialize_boundary_attention_cache(
      1'023, graph_capacity, kv_heads, head_size, &graph_key_cache,
      &graph_value_cache);
  initialize_boundary_attention_cache(
      1'023, boundary_capacity, kv_heads, head_size, &boundary_key_cache,
      &boundary_value_cache);
  device_graph_key_cache.copy_from(graph_key_cache);
  device_graph_value_cache.copy_from(graph_value_cache);
  device_boundary_key_cache.copy_from(boundary_key_cache);
  device_boundary_value_cache.copy_from(boundary_value_cache);

  for (const std::uint32_t position : {1'024U, 1'025U}) {
    std::vector<BFloat16> key;
    std::vector<BFloat16> value;
    boundary_attention_kv(position, kv_heads, head_size, &key, &value);
    device_key.copy_from(key);
    device_value.copy_from(value);
    device_position.copy_from({position});
    write_kv_cache_m1_device_position(
        device_key.get(), device_value.get(), device_graph_key_cache.get(),
        device_graph_value_cache.get(), device_position.get(), kind, nullptr);
    write_kv_cache_m1(device_key.get(), device_value.get(),
                      device_boundary_key_cache.get(),
                      device_boundary_value_cache.get(), position,
                      boundary_capacity, kind, nullptr);
    place_boundary_attention_kv(key, value, position, graph_capacity,
                                kv_heads, head_size, &graph_key_cache,
                                &graph_value_cache);
    place_boundary_attention_kv(key, value, position, boundary_capacity,
                                kv_heads, head_size, &boundary_key_cache,
                                &boundary_value_cache);

    const std::string position_label =
        std::string(label) + "_position_" + std::to_string(position);
    report_metric(report, position_label + "_device_write_key",
                  measure(device_graph_key_cache.copy_to_host(),
                          graph_key_cache),
                  0.0, true);
    report_metric(report, position_label + "_device_write_value",
                  measure(device_graph_value_cache.copy_to_host(),
                          graph_value_cache),
                  0.0, true);

    const std::vector<BFloat16> query =
        boundary_attention_query(position, head_size);
    device_query.copy_from(query);
    device_graph_score.copy_from(std::vector<BFloat16>(
        graph_probability_elements, to_bf16(-7.0F)));
    device_graph_probabilities.copy_from(std::vector<BFloat16>(
        graph_probability_elements, to_bf16(-9.0F)));
    device_boundary_score.copy_from(std::vector<BFloat16>(
        boundary_probability_elements, to_bf16(-7.0F)));
    device_boundary_probabilities.copy_from(std::vector<BFloat16>(
        boundary_probability_elements, to_bf16(-9.0F)));
    causal_gqa_attention_cached_m1_device_position(
        device_query.get(), device_graph_key_cache.get(),
        device_graph_value_cache.get(), device_position.get(),
        device_graph_score.get(), device_graph_probabilities.get(),
        device_graph_context.get(), kind, nullptr);
    causal_gqa_attention_cached_m1_device_position_fused(
        device_query.get(), device_graph_key_cache.get(),
        device_graph_value_cache.get(), device_position.get(),
        device_fused_scratch.get(), device_fused_context.get(), kind,
        nullptr);
    causal_gqa_attention_cached_m1_boundary(
        device_query.get(), device_boundary_key_cache.get(),
        device_boundary_value_cache.get(), position,
        device_boundary_score.get(), device_boundary_probabilities.get(),
        device_boundary_context.get(), kind, nullptr);

    const std::vector<BFloat16> graph_scores =
        device_graph_score.copy_to_host();
    const std::vector<BFloat16> boundary_scores =
        device_boundary_score.copy_to_host();
    std::vector<BFloat16> graph_boundary_score_prefix(
        boundary_probability_elements);
    for (std::uint32_t head = 0; head < gemma4_31b::kQueryHeadCount;
         ++head) {
      std::copy_n(
          graph_scores.begin() +
              static_cast<std::size_t>(head) *
                  kGraphAttentionPositionCount,
          kCachedAttentionM1BoundaryPositionCount,
          graph_boundary_score_prefix.begin() +
              static_cast<std::size_t>(head) *
                  kCachedAttentionM1BoundaryPositionCount);
    }
    const std::vector<BFloat16> graph_probabilities =
        device_graph_probabilities.copy_to_host();
    const std::vector<BFloat16> boundary_probabilities =
        device_boundary_probabilities.copy_to_host();
    std::vector<BFloat16> graph_boundary_prefix(
        boundary_probability_elements);
    for (std::uint32_t head = 0; head < gemma4_31b::kQueryHeadCount;
         ++head) {
      std::copy_n(
          graph_probabilities.begin() +
              static_cast<std::size_t>(head) *
                  kGraphAttentionPositionCount,
          kCachedAttentionM1BoundaryPositionCount,
          graph_boundary_prefix.begin() +
              static_cast<std::size_t>(head) *
                  kCachedAttentionM1BoundaryPositionCount);
    }
    report_metric(report, position_label + "_boundary_score_exact",
                  measure(graph_boundary_score_prefix, boundary_scores), 0.0,
                  true);
    report_metric(report, position_label + "_boundary_probability_exact",
                  measure(graph_boundary_prefix, boundary_probabilities), 0.0,
                  true);
    report_metric(report, position_label + "_boundary_context_exact",
                  measure(device_graph_context.copy_to_host(),
                          device_boundary_context.copy_to_host()),
                  0.0, true);
    const std::vector<BFloat16> fused_context =
        device_fused_context.copy_to_host();
    if (!std::all_of(fused_context.begin(), fused_context.end(),
                     [](BFloat16 value) {
                       return std::isfinite(to_float(value));
                     })) {
      fail(position_label + "_fused_context",
           "CUDA result contains a non-finite value");
    }
    report_metric(report, position_label + "_fused_context_boundary",
                  measure(fused_context,
                          device_boundary_context.copy_to_host()),
                  1.0 / 1'024.0, false);
    const std::uint32_t first_visible_position =
        global ? 0 : position - (kGraphAttentionLocalCapacity - 1);
    require_boundary_probability_zeros(
        graph_probabilities, first_visible_position, position,
        position_label, kGraphAttentionPositionCount);
  }

  initialize_boundary_attention_cache(
      kGraphDecodeLastPosition - 1, graph_capacity, kv_heads, head_size,
      &graph_key_cache, &graph_value_cache);
  device_graph_key_cache.copy_from(graph_key_cache);
  device_graph_value_cache.copy_from(graph_value_cache);
  std::vector<BFloat16> final_key;
  std::vector<BFloat16> final_value;
  boundary_attention_kv(kGraphDecodeLastPosition, kv_heads, head_size,
                        &final_key, &final_value);
  device_key.copy_from(final_key);
  device_value.copy_from(final_value);
  device_position.copy_from({kGraphDecodeLastPosition});
  write_kv_cache_m1_device_position(
      device_key.get(), device_value.get(), device_graph_key_cache.get(),
      device_graph_value_cache.get(), device_position.get(), kind, nullptr);
  place_boundary_attention_kv(
      final_key, final_value, kGraphDecodeLastPosition, graph_capacity,
      kv_heads, head_size, &graph_key_cache, &graph_value_cache);
  const std::vector<BFloat16> final_query =
      boundary_attention_query(kGraphDecodeLastPosition, head_size);
  const std::uint32_t first_visible_position =
      global ? 0
             : kGraphDecodeLastPosition -
                   (kGraphAttentionLocalCapacity - 1);
  const AttentionReference final_expected =
      attention_cached_m1_boundary_reference(
          final_query, graph_key_cache, graph_value_cache,
          kGraphDecodeLastPosition, graph_capacity, first_visible_position,
          kv_heads, head_size, kGraphAttentionPositionCount);
  device_query.copy_from(final_query);
  device_graph_score.copy_from(std::vector<BFloat16>(
      graph_probability_elements, to_bf16(-7.0F)));
  device_graph_probabilities.copy_from(std::vector<BFloat16>(
      graph_probability_elements, to_bf16(-9.0F)));
  causal_gqa_attention_cached_m1_device_position(
      device_query.get(), device_graph_key_cache.get(),
      device_graph_value_cache.get(), device_position.get(),
      device_graph_score.get(), device_graph_probabilities.get(),
      device_graph_context.get(), kind, nullptr);
  causal_gqa_attention_cached_m1_device_position_fused(
      device_query.get(), device_graph_key_cache.get(),
      device_graph_value_cache.get(), device_position.get(),
      device_fused_scratch.get(), device_fused_context.get(), kind,
      nullptr);
  const std::string final_label = std::string(label) + "_position_1534";
  report_metric(report, final_label + "_device_write_key",
                measure(device_graph_key_cache.copy_to_host(),
                        graph_key_cache),
                0.0, true);
  report_metric(report, final_label + "_device_write_value",
                measure(device_graph_value_cache.copy_to_host(),
                        graph_value_cache),
                0.0, true);
  const std::vector<BFloat16> final_probabilities =
      device_graph_probabilities.copy_to_host();
  report_metric(report, final_label + "_probabilities_cpu",
                measure(final_probabilities, final_expected.probabilities),
                0.0, true);
  report_metric(report, final_label + "_context_cpu",
                measure(device_graph_context.copy_to_host(),
                        final_expected.context),
                0.0, true);
  const std::vector<BFloat16> final_fused_context =
      device_fused_context.copy_to_host();
  if (!std::all_of(final_fused_context.begin(), final_fused_context.end(),
                   [](BFloat16 value) {
                     return std::isfinite(to_float(value));
                   })) {
    fail(final_label + "_fused_context",
         "CUDA result contains a non-finite value");
  }
  report_metric(report, final_label + "_fused_context_cpu",
                measure(final_fused_context, final_expected.context),
                1.0 / 1'024.0, false);
  require_boundary_probability_zeros(
      final_probabilities, first_visible_position, kGraphDecodeLastPosition,
      final_label, kGraphAttentionPositionCount);

  initialize_dense_boundary_attention_cache(
      kGraphDecodeLastPosition - 1, graph_capacity, kv_heads, head_size,
      &graph_key_cache, &graph_value_cache);
  device_graph_key_cache.copy_from(graph_key_cache);
  device_graph_value_cache.copy_from(graph_value_cache);
  std::vector<BFloat16> dense_final_key;
  std::vector<BFloat16> dense_final_value;
  dense_boundary_attention_kv(kGraphDecodeLastPosition, kv_heads, head_size,
                              &dense_final_key, &dense_final_value);
  device_key.copy_from(dense_final_key);
  device_value.copy_from(dense_final_value);
  write_kv_cache_m1_device_position(
      device_key.get(), device_value.get(), device_graph_key_cache.get(),
      device_graph_value_cache.get(), device_position.get(), kind, nullptr);
  place_boundary_attention_kv(
      dense_final_key, dense_final_value, kGraphDecodeLastPosition,
      graph_capacity, kv_heads, head_size, &graph_key_cache,
      &graph_value_cache);
  const std::vector<BFloat16> dense_final_query =
      dense_boundary_attention_query(kGraphDecodeLastPosition, head_size);
  const AttentionReference dense_final_expected =
      attention_cached_m1_boundary_reference(
          dense_final_query, graph_key_cache, graph_value_cache,
          kGraphDecodeLastPosition, graph_capacity, first_visible_position,
          kv_heads, head_size, kGraphAttentionPositionCount);
  device_query.copy_from(dense_final_query);
  causal_gqa_attention_cached_m1_device_position_fused(
      device_query.get(), device_graph_key_cache.get(),
      device_graph_value_cache.get(), device_position.get(),
      device_fused_scratch.get(), device_fused_context.get(), kind, nullptr);
  causal_gqa_attention_cached_m1_fused(
      device_query.get(), device_graph_key_cache.get(),
      device_graph_value_cache.get(), kGraphDecodeLastPosition,
      graph_capacity, device_runtime_scratch.get(),
      device_runtime_context.get(), kind, nullptr);
  const std::vector<BFloat16> dense_fused_context =
      device_fused_context.copy_to_host();
  if (!std::all_of(dense_fused_context.begin(), dense_fused_context.end(),
                   [](BFloat16 value) {
                     return std::isfinite(to_float(value));
                   })) {
    fail(final_label + "_fused_dense_context",
         "CUDA result contains a non-finite value");
  }
  report_metric(report, final_label + "_fused_dense_context_cpu",
                measure(dense_fused_context, dense_final_expected.context),
                1.0 / 1'024.0, false);
  report_metric(report, final_label + "_runtime_fused_dense_context",
                measure(device_runtime_context.copy_to_host(),
                        dense_fused_context),
                global ? 1.0 / 1'024.0 : 0.0, !global);
  report << "primitive " << label
         << "_contract: device_positions=1024,1025 score_boundary_exact=1 "
            "probability_context_boundary_exact=1 "
            "fused_context_atol=0.0009765625 "
            "position_1534_cpu_exact=1 fused_dense_qk=1 "
         << (global ? "runtime_global_fp16_partial_atol=0.0009765625 "
                    : "runtime_local_exact=1 ")
         << "capacity="
         << graph_capacity << " score_row=1536 masked_positive_zero=1 ok\n";
}

void test_graph_attention(std::ostream& report) {
  static_assert(kGraphAttentionLocalCapacity == 1'024);
  static_assert(kGraphAttentionGlobalCapacity == 1'536);
  test_graph_attention_kind(
      report, gemma4_31b::AttentionKind::local,
      "causal_gqa_attention_cached_m1_device_position_local");
  test_graph_attention_kind(
      report, gemma4_31b::AttentionKind::global,
      "causal_gqa_attention_cached_m1_device_position_global");
}

void test_graph_decode_state(std::ostream& report) {
  constexpr std::uint32_t kCanary = 0xdecafbadU;
  DeviceBuffer<std::uint32_t> device_first_token(1);
  DeviceBuffer<std::uint32_t> device_next_token(1);
  DeviceBuffer<std::uint32_t> device_current_token(1);
  DeviceBuffer<std::uint32_t> device_position(1);
  DeviceBuffer<std::uint32_t> device_outputs(kGraphOutputTokenCount + 2);
  const std::vector<std::uint32_t> initial_outputs(
      kGraphOutputTokenCount + 2, kCanary);
  device_outputs.copy_from(initial_outputs);
  device_first_token.copy_from({17U});
  seed_graph_decode_state(device_first_token.get(),
                          device_current_token.get(), device_position.get(),
                          device_outputs.get() + 1, nullptr);
  if (device_current_token.copy_to_host()[0] != 17U ||
      device_position.copy_to_host()[0] != kGraphDecodeFirstPosition) {
    fail("seed_graph_decode_state", "unexpected seeded scalar state");
  }
  std::vector<std::uint32_t> outputs = device_outputs.copy_to_host();
  if (outputs.front() != kCanary || outputs.back() != kCanary ||
      outputs[1] != 17U || outputs[2] != kCanary) {
    fail("seed_graph_decode_state", "output-zero or canary mismatch");
  }

  device_next_token.copy_from({23U});
  commit_graph_decode_state(device_next_token.get(),
                            device_current_token.get(), device_position.get(),
                            device_outputs.get() + 1, nullptr);
  outputs = device_outputs.copy_to_host();
  if (device_current_token.copy_to_host()[0] != 23U ||
      device_position.copy_to_host()[0] != 1'025U || outputs[2] != 23U ||
      outputs.front() != kCanary || outputs.back() != kCanary) {
    fail("commit_graph_decode_state", "first commit or canary mismatch");
  }

  device_position.copy_from({kGraphDecodeLastPosition});
  device_next_token.copy_from({29U});
  commit_graph_decode_state(device_next_token.get(),
                            device_current_token.get(), device_position.get(),
                            device_outputs.get() + 1, nullptr);
  outputs = device_outputs.copy_to_host();
  if (device_current_token.copy_to_host()[0] != 29U ||
      device_position.copy_to_host()[0] != kGraphDecodeFinalPosition ||
      outputs[kGraphOutputTokenCount] != 29U ||
      outputs.front() != kCanary || outputs.back() != kCanary) {
    fail("commit_graph_decode_state", "last commit or canary mismatch");
  }

  const std::vector<std::uint32_t> final_outputs = outputs;
  device_next_token.copy_from({31U});
  commit_graph_decode_state(device_next_token.get(),
                            device_current_token.get(), device_position.get(),
                            device_outputs.get() + 1, nullptr);
  if (device_current_token.copy_to_host()[0] != 29U ||
      device_position.copy_to_host()[0] != kGraphDecodeFinalPosition ||
      device_outputs.copy_to_host() != final_outputs) {
    fail("commit_graph_decode_state", "upper-bound commit was not a no-op");
  }

  device_current_token.copy_from({37U});
  device_position.copy_from({kGraphDecodeFirstPosition - 1});
  commit_graph_decode_state(device_next_token.get(),
                            device_current_token.get(), device_position.get(),
                            device_outputs.get() + 1, nullptr);
  if (device_current_token.copy_to_host()[0] != 37U ||
      device_position.copy_to_host()[0] !=
          kGraphDecodeFirstPosition - 1 ||
      device_outputs.copy_to_host() != final_outputs) {
    fail("commit_graph_decode_state", "lower-bound commit was not a no-op");
  }
  report << "primitive graph_decode_state_contract: seed_position=1024 "
            "first_output=0 commit_indices=1,511 final_position=1535 "
            "lower_upper_bounds=no-op canaries=exact ok\n";
}

void test_graph_capture_replay(std::ostream& report) {
  constexpr std::uint32_t kEmbeddingRows = 8;
  constexpr std::uint32_t kCanary = 0xc001cafeU;
  const std::vector<BFloat16> table = pattern(
      static_cast<std::size_t>(kEmbeddingRows) * gemma4_31b::kHiddenSize,
      127, 63, 1.0F / 128.0F);
  const std::size_t kv_elements =
      static_cast<std::size_t>(gemma4_31b::kLocalKvHeadCount) *
      gemma4_31b::kLocalHeadSize;
  const std::size_t cache_elements =
      static_cast<std::size_t>(gemma4_31b::kLocalKvHeadCount) *
      kGraphAttentionLocalCapacity * gemma4_31b::kLocalHeadSize;
  const std::size_t context_elements =
      static_cast<std::size_t>(gemma4_31b::kQueryHeadCount) *
      gemma4_31b::kLocalHeadSize;
  static_assert(kGraphAttentionFusedScratchBytes % sizeof(float) == 0);

  const std::vector<BFloat16> query(context_elements, to_bf16(0.0F));
  const std::vector<BFloat16> key =
      pattern(kv_elements, 31, 15, 1.0F / 32.0F);
  const std::vector<BFloat16> value =
      pattern(kv_elements, 29, 14, 1.0F / 32.0F);

  DeviceBuffer<BFloat16> device_table(table.size());
  DeviceBuffer<BFloat16> device_embedding(gemma4_31b::kHiddenSize);
  DeviceBuffer<BFloat16> device_local_cos(gemma4_31b::kLocalHeadSize);
  DeviceBuffer<BFloat16> device_local_sin(gemma4_31b::kLocalHeadSize);
  DeviceBuffer<BFloat16> device_global_cos(gemma4_31b::kGlobalHeadSize);
  DeviceBuffer<BFloat16> device_global_sin(gemma4_31b::kGlobalHeadSize);
  DeviceBuffer<BFloat16> device_query(context_elements);
  DeviceBuffer<BFloat16> device_key(kv_elements);
  DeviceBuffer<BFloat16> device_value(kv_elements);
  DeviceBuffer<BFloat16> device_key_cache(cache_elements);
  DeviceBuffer<BFloat16> device_value_cache(cache_elements);
  DeviceBuffer<float> device_attention_scratch(
      kGraphAttentionFusedScratchBytes / sizeof(float));
  DeviceBuffer<BFloat16> device_context(context_elements);
  DeviceBuffer<std::uint32_t> device_first_token(1);
  DeviceBuffer<std::uint32_t> device_next_token(1);
  DeviceBuffer<std::uint32_t> device_current_token(1);
  DeviceBuffer<std::uint32_t> device_position(1);
  DeviceBuffer<std::uint32_t> device_outputs(kGraphOutputTokenCount + 2);

  device_table.copy_from(table);
  device_query.copy_from(query);
  device_key.copy_from(key);
  device_value.copy_from(value);
  device_key_cache.copy_from(
      std::vector<BFloat16>(cache_elements, to_bf16(0.0F)));
  device_value_cache.copy_from(
      std::vector<BFloat16>(cache_elements, to_bf16(0.0F)));
  device_context.copy_from(
      std::vector<BFloat16>(context_elements, to_bf16(-9.0F)));
  device_first_token.copy_from({1U});
  device_next_token.copy_from({7U});
  device_outputs.copy_from(std::vector<std::uint32_t>(
      kGraphOutputTokenCount + 2, kCanary));

  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
             "graph primitive self-test create nonblocking stream");
  try {
    seed_graph_decode_state(
        device_first_token.get(), device_current_token.get(),
        device_position.get(), device_outputs.get() + 1, stream);
    check_cuda(cudaStreamSynchronize(stream),
               "graph primitive self-test seed synchronize");
    check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
               "graph primitive self-test begin capture");
    embedding_lookup_device_token(device_table.get(),
                                  device_current_token.get(),
                                  device_embedding.get(), stream);
    generate_rope_factors_m1_device_position(
        device_local_cos.get(), device_local_sin.get(),
        device_global_cos.get(), device_global_sin.get(),
        device_position.get(), stream);
    write_kv_cache_m1_device_position(
        device_key.get(), device_value.get(), device_key_cache.get(),
        device_value_cache.get(), device_position.get(),
        gemma4_31b::AttentionKind::local, stream);
    causal_gqa_attention_cached_m1_device_position_fused(
        device_query.get(), device_key_cache.get(), device_value_cache.get(),
        device_position.get(), device_attention_scratch.get(),
        device_context.get(),
        gemma4_31b::AttentionKind::local, stream);
    commit_graph_decode_state(
        device_next_token.get(), device_current_token.get(),
        device_position.get(), device_outputs.get() + 1, stream);
    check_cuda(cudaStreamEndCapture(stream, &graph),
               "graph primitive self-test end capture");
    check_cuda(cudaGraphInstantiate(&executable, graph, 0),
               "graph primitive self-test instantiate");
    check_cuda(cudaGraphLaunch(executable, stream),
               "graph primitive self-test first launch");
    check_cuda(cudaGraphLaunch(executable, stream),
               "graph primitive self-test second launch");
    check_cuda(cudaStreamSynchronize(stream),
               "graph primitive self-test replay synchronize");
  } catch (...) {
    if (executable != nullptr) {
      cudaGraphExecDestroy(executable);
    }
    if (graph != nullptr) {
      cudaGraphDestroy(graph);
    }
    cudaStreamDestroy(stream);
    throw;
  }
  check_cuda(cudaGraphExecDestroy(executable),
             "graph primitive self-test destroy executable");
  check_cuda(cudaGraphDestroy(graph),
             "graph primitive self-test destroy graph");
  check_cuda(cudaStreamDestroy(stream),
             "graph primitive self-test destroy stream");

  if (device_current_token.copy_to_host()[0] != 7U ||
      device_position.copy_to_host()[0] != 1'026U) {
    fail("graph capture replay", "device state did not advance twice");
  }
  const std::vector<std::uint32_t> outputs =
      device_outputs.copy_to_host();
  if (outputs.front() != kCanary || outputs.back() != kCanary ||
      outputs[1] != 1U || outputs[2] != 7U || outputs[3] != 7U) {
    fail("graph capture replay", "outputs or canaries do not match");
  }
  std::vector<BFloat16> expected_embedding(gemma4_31b::kHiddenSize);
  const std::size_t embedding_row =
      static_cast<std::size_t>(7) * gemma4_31b::kHiddenSize;
  for (std::size_t index = 0; index < expected_embedding.size(); ++index) {
    expected_embedding[index] = to_bf16(
        to_float(table[embedding_row + index]) * kEmbeddingScaleBf16);
  }
  report_metric(report, "graph_capture_replay_embedding",
                measure(device_embedding.copy_to_host(), expected_embedding),
                0.0, true);
  const RopeFactorReference expected_rope = rope_factor_row_reference(
      gemma4_31b::kLocalHeadSize, gemma4_31b::kLocalHeadSize / 2, 10'000.0F,
      1'025);
  report_metric(report, "graph_capture_replay_rope_position_1025",
                measure(device_local_cos.copy_to_host(),
                        expected_rope.cosine),
                0.0, true);
  std::vector<BFloat16> expected_context(context_elements);
  constexpr std::uint32_t kQueryHeadsPerKvHead =
      gemma4_31b::kQueryHeadCount / gemma4_31b::kLocalKvHeadCount;
  constexpr float kTwoCachedValuesWeight =
      2.0F / static_cast<float>(kGraphAttentionLocalCapacity);
  for (std::uint32_t query_head = 0;
       query_head < gemma4_31b::kQueryHeadCount; ++query_head) {
    const std::uint32_t kv_head = query_head / kQueryHeadsPerKvHead;
    for (std::uint32_t dimension = 0;
         dimension < gemma4_31b::kLocalHeadSize; ++dimension) {
      const std::size_t context_index =
          static_cast<std::size_t>(query_head) *
              gemma4_31b::kLocalHeadSize +
          dimension;
      const std::size_t value_index =
          static_cast<std::size_t>(kv_head) *
              gemma4_31b::kLocalHeadSize +
          dimension;
      expected_context[context_index] = to_bf16(
          to_float(value[value_index]) * kTwoCachedValuesWeight);
    }
  }
  report_metric(report, "graph_capture_replay_fused_attention_context",
                measure(device_context.copy_to_host(), expected_context),
                0.0, true);
  report << "primitive graph_capture_replay_contract: "
            "stream=nonblocking captured_surface=embedding,rope,kv,"
            "fused_attention_split_finalize,commit replays=2 "
            "final_position=1026 state_context=exact ok\n";
}

std::vector<BFloat16> gelu_reference(const std::vector<BFloat16>& gate,
                                    const std::vector<BFloat16>& up) {
  std::vector<BFloat16> expected(gate.size());
  for (std::size_t index = 0; index < gate.size(); ++index) {
    const float value = to_float(gate[index]);
    const float cube = value * value * value;
    const float inner =
        kGeluCoefficient * (value + kGeluCubicCoefficient * cube);
    const float activated =
        0.5F * value * (1.0F + std::tanh(inner));
    const BFloat16 activation_bf16 = to_bf16(activated);
    expected[index] =
        to_bf16(to_float(activation_bf16) * to_float(up[index]));
  }
  return expected;
}

void test_gelu(std::ostream& report) {
  constexpr std::size_t kElements = 257;
  const std::vector<BFloat16> gate =
      pattern(kElements, 33, 16, 0.25F);
  const std::vector<BFloat16> up =
      pattern(kElements, 11, 5, 0.125F);
  const std::vector<BFloat16> expected = gelu_reference(gate, up);
  DeviceBuffer<BFloat16> device_gate(kElements);
  DeviceBuffer<BFloat16> device_up(kElements);
  DeviceBuffer<BFloat16> device_output(kElements);
  device_gate.copy_from(gate);
  device_up.copy_from(up);
  gelu_tanh_multiply(device_gate.get(), device_up.get(), device_output.get(),
                     kElements);
  report_metric(report, "gelu_tanh_multiply",
                measure(device_output.copy_to_host(), expected), 1.0 / 32.0,
                false);
  for (const unsigned rows : {1u, 3u, 32u, 128u}) {
    constexpr unsigned width = gemma4_31b::kMlpSize;
    const auto gates = pattern(std::size_t(rows) * width, 97, 48, 0.25F);
    const auto ups = pattern(gates.size(), 43, 21, 0.125F);
    std::vector<BFloat16> joined(2 * gates.size());
    for (unsigned row = 0; row < rows; ++row) {
      std::copy_n(gates.data() + row * width, width, joined.data() + row * 2 * width);
      std::copy_n(ups.data() + row * width, width, joined.data() + row * 2 * width + width);
    }
    DeviceBuffer<BFloat16> g(gates.size()), u(ups.size()), gu(joined.size());
    DeviceBuffer<BFloat16> reference(gates.size()), fused(gates.size());
    g.copy_from(gates); u.copy_from(ups); gu.copy_from(joined);
    gelu_tanh_multiply(g.get(), u.get(), reference.get(), gates.size());
    gelu_tanh_multiply_interleaved(gu.get(), fused.get(), rows, width);
    report_metric(report, "fused_gate_up_activation",
                  measure(fused.copy_to_host(), reference.copy_to_host()), 0, true);
  }
}

std::vector<BFloat16> softcap_reference(const std::vector<BFloat16>& logits,
                                       float cap) {
  std::vector<BFloat16> expected(logits.size());
  for (std::size_t index = 0; index < logits.size(); ++index) {
    const BFloat16 divided = to_bf16(to_float(logits[index]) / cap);
    const BFloat16 squashed = to_bf16(std::tanh(to_float(divided)));
    expected[index] = to_bf16(to_float(squashed) * cap);
  }
  return expected;
}

void test_softcap_argmax(std::ostream& report) {
  constexpr std::uint32_t kElements = 257;
  constexpr float kCap = 30.0F;
  std::vector<BFloat16> logits = pattern(kElements, 121, 60, 1.0F);
  logits[7] = to_bf16(100.0F);
  logits[201] = to_bf16(100.0F);
  const std::vector<BFloat16> expected = softcap_reference(logits, kCap);

  DeviceBuffer<BFloat16> device_logits(logits.size());
  DeviceBuffer<BFloat16> device_capped(logits.size());
  DeviceBuffer<std::uint32_t> device_argmax(1);
  device_logits.copy_from(logits);
  softcap_and_argmax(device_logits.get(), device_capped.get(),
                     device_argmax.get(), kElements, kCap);
  report_metric(report, "softcap",
                measure(device_capped.copy_to_host(), expected), 0.125, false);
  const std::vector<std::uint32_t> argmax = device_argmax.copy_to_host();
  report << "primitive deterministic_argmax: token=" << argmax[0]
         << " expected=7 tie=7,201\n";
  if (argmax[0] != 7) {
    fail("deterministic_argmax", "lowest-id tie break failed");
  }
}

void test_batched_softcap_argmax(std::ostream& report) {
  for (const unsigned rows : {1U, 3U, 32U, 1280U}) {
    const unsigned elements = rows == 32 ? gemma4_31b::kVocabSize : 257;
    const auto count = std::size_t(rows) * elements;
    auto values = pattern(count, 121, 60, 1.0F);
    for (unsigned row = 0; row < rows; ++row) {
      // Exact ties, different winners per row, NaNs, and a saturated row.
      values[std::size_t(row) * elements + row % 7] = to_bf16(1000.0F);
      values[std::size_t(row) * elements + 201] = to_bf16(1000.0F);
      values[std::size_t(row) * elements + 15] = to_bf16(std::numeric_limits<float>::quiet_NaN());
    }
    if (rows > 1)
      std::fill(values.end() - elements, values.end(), to_bf16(-std::numeric_limits<float>::infinity()));
    DeviceBuffer<BFloat16> logits(count), reference(count), capped(count + 16);
    DeviceBuffer<std::uint32_t> expected(rows), selected(rows + 4);
    logits.copy_from(values);
    capped.copy_from(std::vector<BFloat16>(count + 16, to_bf16(0.75F)));
    selected.copy_from(std::vector<std::uint32_t>(rows + 4, UINT_MAX));
    for (unsigned row = 0; row < rows; ++row)
      softcap_and_argmax(logits.get() + std::size_t(row) * elements,
          reference.get() + std::size_t(row) * elements, expected.get() + row, elements);
    check_cuda(cudaDeviceSynchronize(), "batched argmax reference");
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    check_cuda(cudaStreamCreate(&stream), "batched argmax stream");
    check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "batched argmax capture");
    softcap_and_argmax_rows(logits.get(), capped.get(), selected.get(), rows, elements, 30.0F, stream);
    check_cuda(cudaStreamEndCapture(stream, &graph), "batched argmax capture end");
    check_cuda(cudaGraphInstantiate(&exec, graph, 0), "batched argmax instantiate");
    check_cuda(cudaGraphLaunch(exec, stream), "batched argmax replay");
    check_cuda(cudaGraphLaunch(exec, stream), "batched argmax replay");
    check_cuda(cudaStreamSynchronize(stream), "batched argmax synchronize");
    const auto actual = capped.copy_to_host(), serial = reference.copy_to_host();
    const auto tokens = selected.copy_to_host(), serial_tokens = expected.copy_to_host();
    if (std::memcmp(actual.data(), serial.data(), count * sizeof(BFloat16)) ||
        !std::equal(serial_tokens.begin(), serial_tokens.end(), tokens.begin()))
      fail("batched argmax", "serial softcap or selection changed");
    for (std::size_t i = count; i < actual.size(); ++i)
      if (bf16_bits(actual[i]) != bf16_bits(to_bf16(0.75F))) fail("batched argmax", "capped guard overwritten");
    for (unsigned i = rows; i < tokens.size(); ++i)
      if (tokens[i] != UINT_MAX) fail("batched argmax", "token guard overwritten");
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
    report << "batched softcap/argmax rows=" << rows << " elements=" << elements
           << " serial=exact ties=exact guards=passed cold_graph=passed\n";
  }
}

void test_device_sampling(std::ostream& report) {
  const std::vector<BFloat16> logits{
      to_bf16(0.0F), to_bf16(0.0F), to_bf16(-2.0F),
      to_bf16(-4.0F), to_bf16(-8.0F)};
  DeviceBuffer<BFloat16> device_logits(logits.size());
  DeviceBuffer<std::uint8_t> device_scratch(
      sampling_scratch_bytes(static_cast<std::uint32_t>(logits.size())));
  DeviceBuffer<std::uint32_t> device_selected(1);
  device_logits.copy_from(logits);

  sample_top_k_top_p(
      device_logits.get(), static_cast<std::uint32_t>(logits.size()), 1.0F,
      1.0F, 2, 0.75, device_scratch.get(),
      sampling_scratch_bytes(static_cast<std::uint32_t>(logits.size())),
      device_selected.get());
  if (device_selected.copy_to_host()[0] != 1) {
    fail("sample_top_k_top_p", "top-k tie ordering or draw changed");
  }

  sample_top_k_top_p(
      device_logits.get(), static_cast<std::uint32_t>(logits.size()), 1.0F,
      0.4F, 0, 0.99, device_scratch.get(),
      sampling_scratch_bytes(static_cast<std::uint32_t>(logits.size())),
      device_selected.get());
  if (device_selected.copy_to_host()[0] != 0) {
    fail("sample_top_k_top_p", "top-p cutoff changed");
  }
  report << "primitive device_sampling: elements=5 top_k=2 top_p=0.4 "
            "tie_order=lowest_token_id ok\n";
}

bool run_self_tests(std::ostream& report, std::string* failure) {
  try {
    test_linear(report);
    test_embedding(report);
    test_rms_norm(report);
    test_qkv_norm_fusion(report);
    test_qkv_rope_batch(report);
    test_hidden_norm_fusions(report);
    test_prefill_norm_fusions(report);
    test_batched_decode_rows(report);
    test_residual_scalar(report);
    test_value_expansion(report);
    test_rope_factors(report);
    test_rope_factors_24(report);
    test_rope_factors_m1_boundary(report);
    test_rope_transpose(report);
    test_rope_m1(report);
    test_attention_m2(report);
    test_cached_attention_m1(report);
    test_cached_attention_m1_24(report);
    test_cached_attention_m1_boundary(report);
    test_runtime_compact_global_reference(report);
    test_runtime_global_compact_tensor_constant_value(report);
    test_runtime_global_compact_tensor_dispatch_nonuniform(report);
    test_runtime_global_coarse_reduction(report);
    test_runtime_global_coarse_dispatch_boundary(report);
    test_decode_attention_batch(report);
    test_runtime_global_fp16_partial_range(report);
    test_graph_device_lookup_and_rope(report);
    test_graph_attention(report);
    test_graph_decode_state(report);
    test_graph_capture_replay(report);
    test_gelu(report);
    test_softcap_argmax(report);
    test_batched_softcap_argmax(report);
    test_device_sampling(report);
    report << "BF16 primitive self-test: ok\n";
    return true;
  } catch (const std::exception& error) {
    if (failure != nullptr) {
      *failure = error.what();
    }
    return false;
  }
}

}  // namespace
}  // namespace gewell::bf16_primitives

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--qkv-rope") {
    try {
      gewell::bf16_primitives::test_qkv_rope_batch(std::cout);
      return 0;
    } catch (const std::exception& error) {
      std::cerr << error.what() << '\n';
      return 1;
    }
  }
  if (argc != 1) {
    std::cerr << "usage: gewell_bf16_primitives_test [--qkv-rope]\n";
    return 1;
  }
  std::string failure;
  if (!gewell::bf16_primitives::run_self_tests(std::cout, &failure)) {
    std::cerr << failure << "\n";
    return 1;
  }
  return 0;
}
