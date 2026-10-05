#include "gewell/mtp_sampling.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace mtp = gewell::mtp_sampling;

namespace {

void check_cuda(cudaError_t error, const char* operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

void expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename T>
class Device {
 public:
  explicit Device(std::size_t count) : count_(count) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer_),
                           std::max<std::size_t>(count, 1) * sizeof(T)),
               "allocate test device storage");
  }
  ~Device() { cudaFree(pointer_); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  T* get() const { return pointer_; }
  void put(const std::vector<T>& values) {
    expect(values.size() <= count_, "test upload exceeds allocation");
    if (!values.empty()) {
      check_cuda(cudaMemcpy(pointer_, values.data(), values.size() * sizeof(T),
                             cudaMemcpyHostToDevice),
                 "upload test values");
    }
  }
  std::vector<T> read(std::size_t count) const {
    expect(count <= count_, "test read exceeds allocation");
    std::vector<T> values(count);
    if (count != 0) {
      check_cuda(cudaMemcpy(values.data(), pointer_, count * sizeof(T),
                             cudaMemcpyDeviceToHost),
                 "read test values");
    }
    return values;
  }

 private:
  std::size_t count_;
  T* pointer_{};
};

struct Fixture {
  explicit Fixture(std::uint32_t vocabulary)
      : vocabulary(vocabulary),
        bytes(mtp::scratch_bytes(vocabulary)),
        scratch(bytes),
        status(1) {}
  void clear() { mtp::clear_status(status.get()); }
  void success() const {
    const mtp::Status value = status.read(1)[0];
    if (value != mtp::Status::success) {
      throw std::runtime_error(mtp::status_message(value));
    }
  }
  std::uint32_t vocabulary;
  std::size_t bytes;
  Device<unsigned char> scratch;
  Device<mtp::Status> status;
};

std::vector<__nv_bfloat16> bf16(const std::vector<float>& values) {
  std::vector<__nv_bfloat16> result;
  for (const float value : values) {
    result.push_back(__float2bfloat16(value));
  }
  return result;
}

std::vector<float> distribution_reference(
    const std::vector<__nv_bfloat16>& logits, float temperature, float top_p,
    std::uint32_t top_k, const std::vector<std::uint32_t>& mask = {}) {
  const bool greedy = temperature == 0.0F || top_p == 0.0F || top_k == 1;
  std::vector<float> scores(logits.size());
  std::vector<std::uint32_t> order(logits.size());
  std::iota(order.begin(), order.end(), 0);
  if (!mask.empty()) {
    order.erase(std::remove_if(order.begin(), order.end(), [&](auto token) {
      return !(mask[token / 32] & (std::uint32_t{1} << (token % 32)));
    }), order.end());
  }
  expect(!order.empty(), "reference allowed set is empty");
  for (std::size_t index = 0; index < logits.size(); ++index) {
    scores[index] = __bfloat162float(logits[index]) /
                    (greedy ? 1.0F : temperature);
  }
  std::stable_sort(order.begin(), order.end(), [&](auto left, auto right) {
    return scores[left] > scores[right];
  });
  const std::size_t retained =
      greedy ? 1 : (top_k == 0 ? order.size()
                               : std::min<std::size_t>(top_k, order.size()));
  std::vector<float> weights(retained);
  float total = 0.0F;
  for (std::size_t rank = 0; rank < retained; ++rank) {
    weights[rank] = std::exp(scores[order[rank]] - scores[order[0]]);
    total += weights[rank];
  }
  const float threshold = top_p * total;
  float mass = 0.0F;
  std::size_t cutoff = 0;
  do {
    mass += weights[cutoff++];
  } while (cutoff < retained && mass < threshold);
  std::vector<float> result(logits.size(), 0.0F);
  for (std::size_t rank = 0; rank < cutoff; ++rank) {
    result[order[rank]] = weights[rank] / mass;
  }
  return result;
}

std::uint32_t sample_reference(const std::vector<float>& probabilities,
                               float uniform) {
  float total = std::accumulate(probabilities.begin(), probabilities.end(), 0.0F);
  const float threshold =
      std::min(uniform * total, std::nextafter(total, 0.0F));
  float sum = 0.0F;
  for (std::uint32_t token = 0; token < probabilities.size(); ++token) {
    sum += probabilities[token];
    if (sum > threshold) {
      return token;
    }
  }
  throw std::runtime_error("reference CDF exhausted");
}

void expect_close(float actual, float expected, const char* message) {
  expect(std::fabs(actual - expected) < 1.0e-6F, message);
}

void logprob_summary_tests() {
  constexpr std::uint32_t rows = 4;
  constexpr std::uint32_t vocabulary = 7;
  constexpr std::uint32_t top_logprobs = 4;
  const std::vector<float> probabilities{
      0.1F, 0.3F, 0.3F, 0.0F, 0.2F, 0.1F, 0.0F,
      0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F,
      0.05F, 0.15F, 0.0F, 0.4F, 0.1F, 0.2F, 0.1F,
      0.2F, 0.2F, 0.2F, 0.2F, 0.2F, 0.0F, 0.0F,
  };
  const std::vector<std::uint32_t> selected{5, 2, 1, 4};
  Device<float> device_probabilities(probabilities.size());
  Device<std::uint32_t> device_selected(rows);
  Device<gewell::TokenLogprobs> device_output(rows);
  device_probabilities.put(probabilities);
  device_selected.put(selected);
  mtp::summarize_logprobs(device_probabilities.get(), device_selected.get(),
                          rows, vocabulary, top_logprobs,
                          device_output.get());
  const auto output = device_output.read(rows);

  const std::array<std::array<std::uint32_t, top_logprobs>, rows>
      expected_tokens{{{{1, 2, 4, 0}}, {{2, 0, 0, 0}},
                       {{3, 5, 1, 4}}, {{0, 1, 2, 3}}}};
  const std::array<std::array<float, top_logprobs>, rows> expected_probs{{
      {{0.3F, 0.3F, 0.2F, 0.1F}},
      {{1.0F, 0.0F, 0.0F, 0.0F}},
      {{0.4F, 0.2F, 0.15F, 0.1F}},
      {{0.2F, 0.2F, 0.2F, 0.2F}},
  }};
  const std::array<std::uint32_t, rows> expected_counts{4, 1, 4, 4};
  for (std::uint32_t row = 0; row < rows; ++row) {
    expect_close(output[row].logprob,
                 std::log(probabilities[row * vocabulary + selected[row]]),
                 "selected-token natural logprob mismatch");
    expect(output[row].count == expected_counts[row],
           "positive top-logprob support count mismatch");
    for (std::uint32_t rank = 0; rank < expected_counts[row]; ++rank) {
      expect(output[row].top[rank].token == expected_tokens[row][rank],
             "top-logprob probability/tie ordering mismatch");
      expect_close(output[row].top[rank].logprob,
                   std::log(expected_probs[row][rank]),
                   "alternative natural logprob mismatch");
    }
  }
  expect(output[0].top[3].token != selected[0] &&
             output[3].top[3].token != selected[3],
         "selected token outside top N was forced into alternatives");
  expect(output[1].top[1].token == 0 && output[1].top[1].logprob == 0,
         "unused compact alternatives were not cleared");

  mtp::summarize_logprobs(device_probabilities.get(), device_selected.get(),
                          rows, vocabulary, 0, device_output.get());
  const auto selected_only = device_output.read(rows);
  for (std::uint32_t row = 0; row < rows; ++row) {
    expect(selected_only[row].count == 0,
           "top_logprobs=0 returned alternatives");
    expect_close(selected_only[row].logprob, output[row].logprob,
                 "top_logprobs=0 changed selected-token score");
  }
  std::cout << "mtp_logprobs: ties_sparse_selected_outside_top_multirow_zero_ok\n";
}

void full_top_logprob_summary_test() {
  constexpr std::uint32_t vocabulary = 521;
  constexpr std::uint32_t support_count = 24;
  std::vector<float> probabilities(vocabulary);
  std::vector<std::uint32_t> support;
  for (std::uint32_t stride = 0; stride < 3; ++stride) {
    for (std::uint32_t lane = 1; lane <= 8; ++lane) {
      const std::uint32_t token = stride * 256 + lane;
      probabilities[token] = 1.0F / support_count;
      support.push_back(token);
    }
  }
  std::sort(support.begin(), support.end());
  Device<float> device_probabilities(vocabulary);
  Device<std::uint32_t> device_selected(1);
  Device<gewell::TokenLogprobs> device_output(1);
  device_probabilities.put(probabilities);
  device_selected.put({support.back()});
  mtp::summarize_logprobs(
      device_probabilities.get(), device_selected.get(), 1, vocabulary,
      gewell::kMaxTopLogprobs, device_output.get());
  const auto output = device_output.read(1)[0];
  expect(output.count == gewell::kMaxTopLogprobs,
         "maximum top-logprob count was not filled from positive support");
  expect_close(output.logprob, std::log(1.0F / support_count),
               "large-row selected-token logprob mismatch");
  for (std::uint32_t rank = 0; rank < gewell::kMaxTopLogprobs; ++rank) {
    expect(output.top[rank].token == support[rank],
           "cross-stride tie ordering or local candidate advance failed");
    expect_close(output.top[rank].logprob, std::log(1.0F / support_count),
                 "full alternative logprob mismatch");
  }
  expect(std::none_of(output.top,
                      output.top + gewell::kMaxTopLogprobs,
                      [&](const auto& alternative) {
                        return alternative.token == support.back();
                      }),
         "large-row selected token outside top 20 was forced into alternatives");
  std::cout << "mtp_logprobs_full_top: cross_stride_ties_20_of_24_ok\n";
}

void invalid_logprob_argument_tests() {
  Device<float> probabilities(1);
  Device<std::uint32_t> selected(1);
  Device<gewell::TokenLogprobs> output(1);
  probabilities.put({1.0F});
  selected.put({0});
  const auto rejects = [&](const auto& call) {
    try {
      call();
    } catch (const std::invalid_argument&) {
      return;
    }
    throw std::runtime_error("invalid logprob host argument was accepted");
  };
  rejects([&] {
    mtp::summarize_logprobs(nullptr, selected.get(), 1, 1, 0, output.get());
  });
  rejects([&] {
    mtp::summarize_logprobs(probabilities.get(), nullptr, 1, 1, 0,
                            output.get());
  });
  rejects([&] {
    mtp::summarize_logprobs(probabilities.get(), selected.get(), 1, 1, 0,
                            nullptr);
  });
  rejects([&] {
    mtp::summarize_logprobs(probabilities.get(), selected.get(), 0, 1, 0,
                            output.get());
  });
  rejects([&] {
    mtp::summarize_logprobs(probabilities.get(), selected.get(), UINT_MAX, 1,
                            0, output.get());
  });
  rejects([&] {
    mtp::summarize_logprobs(probabilities.get(), selected.get(), 1, 0, 0,
                            output.get());
  });
  rejects([&] {
    mtp::summarize_logprobs(probabilities.get(), selected.get(), 1, UINT_MAX,
                            0, output.get());
  });
  rejects([&] {
    constexpr auto maximum = std::numeric_limits<int>::max();
    mtp::summarize_logprobs(probabilities.get(), selected.get(), maximum,
                            maximum, 0, output.get());
  });
  rejects([&] {
    mtp::summarize_logprobs(probabilities.get(), selected.get(), 1, 1,
                            gewell::kMaxTopLogprobs + 1, output.get());
  });
  mtp::summarize_logprobs(probabilities.get(), selected.get(), 1, 1,
                          gewell::kMaxTopLogprobs, output.get());
  expect(output.read(1)[0].count == 1,
         "maximum top_logprobs value was not accepted");
  std::cout << "mtp_logprob_arguments: null_zero_bounds_ok\n";
}

void distribution_tests() {
  const auto logits = bf16({0.0F, -0.0F, 1.5F, 1.5F, -2.0F, 0.75F, -90.0F});
  Fixture fixture(logits.size());
  Device<__nv_bfloat16> input(logits.size());
  Device<float> probabilities(logits.size());
  Device<float> uniform(1);
  Device<std::uint32_t> token(1);
  input.put(logits);
  struct Settings {
    float temperature;
    float top_p;
    std::uint32_t top_k;
  };
  const std::array<Settings, 8> cases{{
      {0.8F, 1.0F, 0}, {1.0F, 0.5F, 2}, {0.7F, 0.83F, 4},
      {1.0F, 1.0F, 100}, {0.0F, 1.0F, 0}, {0.8F, 0.0F, 0},
      {0.8F, 1.0F, 1}, {1.0F, 1.0F, 2},
  }};
  for (const auto& settings : cases) {
    fixture.clear();
    mtp::build_distribution(input.get(), fixture.vocabulary,
                            settings.temperature, settings.top_p, settings.top_k,
                            probabilities.get(), fixture.scratch.get(),
                            fixture.bytes, fixture.status.get());
    fixture.success();
    const auto actual = probabilities.read(logits.size());
    const auto expected = distribution_reference(
        logits, settings.temperature, settings.top_p, settings.top_k);
    for (std::size_t index = 0; index < actual.size(); ++index) {
      expect(std::fabs(actual[index] - expected[index]) < 2.0e-6F,
             "temperature/top-k/top-p probability mismatch");
    }
    for (const float draw : {0.0F, 0.125F, 0.499F, 0.751F,
                             std::nextafter(1.0F, 0.0F)}) {
      uniform.put({draw});
      mtp::sample_distribution(probabilities.get(), fixture.vocabulary,
                               uniform.get(), token.get(), fixture.scratch.get(),
                               fixture.bytes, fixture.status.get());
      fixture.success();
      expect(token.read(1)[0] == sample_reference(expected, draw),
             "canonical probability draw mismatch");
    }
  }
  // An exactly tied nucleus cutoff keeps only the lower token ID.
  fixture.clear();
  mtp::build_distribution(input.get(), fixture.vocabulary, 1.0F, 0.5F, 2,
                          probabilities.get(), fixture.scratch.get(),
                          fixture.bytes, fixture.status.get());
  fixture.success();
  expect(probabilities.read(logits.size())[2] == 1.0F,
         "top-p equality or token-ID tie ordering changed");
  std::cout << "mtp_distribution: temperature_top_k_top_p_ties_ok\n";
}

void masked_distribution_tests() {
  // The forbidden tokens outrank every allowed token. Masking after top-k or
  // top-p would produce empty support or retain the wrong allowed nucleus.
  std::vector<float> values(35, 20.0F);
  values[2] = values[5] = 1.5F;
  values[32] = -1.0F;
  values[34] = -80.0F;
  const auto logits = bf16(values);
  const std::vector<std::uint32_t> mask{(1U << 2) | (1U << 5),
                                      (1U << 0) | (1U << 2) | (1U << 31)};
  // The final set bit is padding, outside this 35-token vocabulary.
  Fixture fixture(logits.size());
  Device<__nv_bfloat16> input(logits.size());
  Device<std::uint32_t> allowed(mtp::mask_words(fixture.vocabulary));
  Device<float> probabilities(logits.size()), uniform(1);
  Device<std::uint32_t> token(1);
  input.put(logits);
  allowed.put(mask);
  const std::array<std::array<float, 3>, 7> cases{{
      {0.8F, 1.0F, 0}, {1.0F, 0.5F, 2}, {0.7F, 0.93F, 3},
      {1.0F, 1.0F, 100}, {0.0F, 1.0F, 0}, {0.8F, 0.0F, 0},
      {0.8F, 1.0F, 1},
  }};
  for (const auto& settings : cases) {
    fixture.clear();
    const auto top_k = static_cast<std::uint32_t>(settings[2]);
    mtp::build_distribution(input.get(), fixture.vocabulary, settings[0],
        settings[1], top_k, probabilities.get(), fixture.scratch.get(),
        fixture.bytes, fixture.status.get(), nullptr, allowed.get());
    fixture.success();
    const auto actual = probabilities.read(logits.size());
    const auto expected = distribution_reference(
        logits, settings[0], settings[1], top_k, mask);
    for (std::size_t index = 0; index < actual.size(); ++index)
      expect(std::fabs(actual[index] - expected[index]) < 2.0e-6F,
             "mask must precede filters and normalization");
    for (const float draw : {0.0F, 0.49F, 0.9F, std::nextafter(1.0F, 0.0F)}) {
      uniform.put({draw});
      mtp::sample_distribution(probabilities.get(), fixture.vocabulary,
          uniform.get(), token.get(), fixture.scratch.get(), fixture.bytes,
          fixture.status.get());
      fixture.success();
      expect(token.read(1)[0] == sample_reference(expected, draw),
             "masked draw differs from allowed-token reference");
    }
  }

  // No token is legal; padding bits cannot create probability mass.
  allowed.put({0, 1U << 31});
  fixture.clear();
  mtp::build_distribution(input.get(), fixture.vocabulary, 1, 1, 0,
      probabilities.get(), fixture.scratch.get(), fixture.bytes,
      fixture.status.get(), nullptr, allowed.get());
  expect(fixture.status.read(1)[0] == mtp::Status::invalid_distribution,
         "all-masked row must fail instead of drawing a forbidden token");

  // Masks do not hide corrupt raw model logits.
  values[0] = std::numeric_limits<float>::infinity();
  input.put(bf16(values));
  allowed.put(mask);
  fixture.clear();
  mtp::build_distribution(input.get(), fixture.vocabulary, 0, 1, 0,
      probabilities.get(), fixture.scratch.get(), fixture.bytes,
      fixture.status.get(), nullptr, allowed.get());
  expect(fixture.status.read(1)[0] == mtp::Status::invalid_logits,
         "forbidden non-finite raw logit must still fail");
  std::cout << "mtp_mask_distribution: filter_order_greedy_ties_word_bounds_empty_ok\n";
}

void greedy_batch_tests() {
  constexpr std::uint32_t rows = 129, vocabulary = 8193;
  const auto words = mtp::mask_words(vocabulary);
  std::vector<__nv_bfloat16> logits(std::size_t(rows) * vocabulary,
                                    __float2bfloat16(-2.0F));
  std::vector<std::uint32_t> expected(rows);
  for (std::uint32_t row = 0; row < rows; ++row) {
    const auto low = row + 3;
    const auto high = 7000 + row;
    logits[std::size_t(row) * vocabulary + low] = __float2bfloat16(3.0F);
    logits[std::size_t(row) * vocabulary + high] = __float2bfloat16(3.0F);
    expected[row] = low;
  }
  std::vector<std::uint32_t> masks(2 * words);
  masks[expected[0] / 32] = 1U << (expected[0] % 32);
  const auto masked_winner = 7000U;
  masks[masked_winner / 32] |= 1U << (masked_winner % 32);
  // Only the higher tied token is legal in row zero.
  masks[expected[0] / 32] &= ~(1U << (expected[0] % 32));
  expected[0] = masked_winner;
  logits[std::size_t(rows - 1) * vocabulary + 17] =
      __float2bfloat16(std::numeric_limits<float>::infinity());

  Device<__nv_bfloat16> device_logits(logits.size());
  Device<float> probabilities(std::size_t(rows) * vocabulary);
  Device<std::uint32_t> device_masks(masks.size()), best(rows), selected(rows);
  Device<mtp::Status> statuses(rows);
  device_logits.put(logits);
  device_masks.put(masks);
  check_cuda(cudaMemset(statuses.get(), 0, rows * sizeof(mtp::Status)),
             "clear batched greedy statuses");
  std::vector<mtp::GreedyDistributionInput> inputs;
  inputs.reserve(rows);
  for (std::uint32_t row = 0; row < rows; ++row) {
    const std::uint32_t* allowed = nullptr;
    if (row == 0) allowed = device_masks.get();
    if (row == rows - 2) allowed = device_masks.get() + words;
    inputs.push_back({
        device_logits.get() + std::size_t(row) * vocabulary,
        probabilities.get() + std::size_t(row) * vocabulary,
        allowed, statuses.get() + row, best.get() + row,
        selected.get() + row});
  }
  mtp::build_greedy_distributions(inputs, vocabulary);
  const auto actual = probabilities.read(std::size_t(rows) * vocabulary);
  const auto ids = selected.read(rows);
  const auto status = statuses.read(rows);
  for (std::uint32_t row = 0; row < rows; ++row) {
    if (row == rows - 2) {
      expect(status[row] == mtp::Status::invalid_distribution && ids[row] == 0,
             "all-masked batched greedy row did not fail safely");
    } else if (row == rows - 1) {
      expect(status[row] == mtp::Status::invalid_logits && ids[row] == 0,
             "non-finite batched greedy row did not fail safely");
    } else {
      expect(status[row] == mtp::Status::success && ids[row] == expected[row],
             "batched greedy winner or lower-ID tie order changed");
    }
    for (std::uint32_t token = 0; token < vocabulary; ++token) {
      const float wanted = row == rows - 2
                               ? 0.0F
                               : token == expected[row] ? 1.0F : 0.0F;
      expect(actual[std::size_t(row) * vocabulary + token] == wanted,
             "batched greedy probability row is not exact one-hot");
    }
  }

  Device<__nv_bfloat16> select_logits(4);
  Device<std::uint32_t> select_best(1), select_id(1);
  Device<mtp::Status> select_status(1);
  select_logits.put(bf16({-1, 4, 2, 3}));
  mtp::clear_status(select_status.get());
  mtp::build_greedy_distributions(
      {{select_logits.get(), nullptr, nullptr, select_status.get(),
        select_best.get(), select_id.get()}},
      4);
  expect(select_status.read(1)[0] == mtp::Status::success &&
             select_id.read(1)[0] == 1,
         "selection-only batched greedy path changed its output contract");
  std::cout << "mtp_greedy_batch: rows=129 bounded_groups_ties_masks_errors_ok\n";
}

void greedy_verification_batch_tests() {
  constexpr std::uint32_t rows = 129, depth = 3, vocabulary = 8;
  std::vector<std::uint32_t> target_ids(std::size_t(rows) * (depth + 1));
  std::vector<std::uint32_t> draft_ids(std::size_t(rows) * depth);
  std::vector<float> uniforms(std::size_t(rows) * (depth + 1), 0.5F);
  std::vector<mtp::Status> initial_status(rows, mtp::Status::success);
  for (std::uint32_t row = 0; row < rows; ++row) {
    std::copy_n(std::array<std::uint32_t, 4>{1, 2, 3, 4}.begin(), 4,
                target_ids.begin() + std::size_t(row) * 4);
    std::copy_n(std::array<std::uint32_t, 3>{1, 2, 3}.begin(), 3,
                draft_ids.begin() + std::size_t(row) * 3);
  }
  target_ids[1] = 5;  // Row zero rejects after one accepted proposal.
  draft_ids[3] = 6;   // Row one rejects immediately.
  initial_status[125] = mtp::Status::invalid_logits;
  target_ids[std::size_t(126) * 4] = vocabulary;
  draft_ids[std::size_t(127) * 3] = vocabulary;
  uniforms[std::size_t(128) * 4 + 3] = 1.0F;

  Device<std::uint32_t> device_targets(target_ids.size());
  Device<std::uint32_t> device_drafts(draft_ids.size());
  Device<float> device_uniforms(uniforms.size());
  Device<std::uint32_t> outputs(std::size_t(rows) * (depth + 1));
  Device<mtp::Result> results(rows);
  Device<mtp::Status> statuses(rows);
  device_targets.put(target_ids);
  device_drafts.put(draft_ids);
  device_uniforms.put(uniforms);
  statuses.put(initial_status);
  std::vector<mtp::GreedyVerificationInput> inputs;
  inputs.reserve(rows);
  for (std::uint32_t row = 0; row < rows; ++row) {
    const auto row_depth = row == 2 ? 0U : depth;
    inputs.push_back({
        device_targets.get() + std::size_t(row) * (depth + 1),
        row_depth ? device_drafts.get() + std::size_t(row) * depth : nullptr,
        row_depth ? device_uniforms.get() + std::size_t(row) * (depth + 1)
                  : nullptr,
        device_uniforms.get() + std::size_t(row) * (depth + 1) + row_depth,
        row_depth, outputs.get() + std::size_t(row) * (depth + 1),
        results.get() + row, statuses.get() + row});
  }
  mtp::verify_greedy_sequences(inputs, vocabulary);
  const auto actual_status = statuses.read(rows);
  const auto actual_results = results.read(rows);
  const auto actual_outputs = outputs.read(std::size_t(rows) * (depth + 1));
  expect(actual_status[0] == mtp::Status::success &&
             actual_results[0].accepted_drafts == 1 &&
             actual_results[0].output_count == 2 &&
             actual_results[0].rejected_index == 1 &&
             actual_outputs[0] == 1 && actual_outputs[1] == 5,
         "batched greedy first mismatch is wrong");
  expect(actual_status[1] == mtp::Status::success &&
             actual_results[1].accepted_drafts == 0 &&
             actual_results[1].output_count == 1 &&
             actual_outputs[4] == 1,
         "batched greedy immediate correction is wrong");
  expect(actual_status[2] == mtp::Status::success &&
             actual_results[2].accepted_drafts == 0 &&
             actual_results[2].output_count == 1 &&
             actual_outputs[8] == 1,
         "batched depth-zero greedy bonus is wrong");
  for (std::uint32_t row = 3; row < 125; ++row) {
    expect(actual_status[row] == mtp::Status::success &&
               actual_results[row].accepted_drafts == depth &&
               actual_results[row].output_count == depth + 1 &&
               std::equal(actual_outputs.begin() + std::size_t(row) * 4,
                          actual_outputs.begin() + std::size_t(row + 1) * 4,
                          std::array<std::uint32_t, 4>{1, 2, 3, 4}.begin()),
           "batched greedy all-accepted sequence is wrong");
  }
  expect(actual_status[125] == mtp::Status::invalid_logits &&
             actual_results[125].output_count == 0,
         "batched greedy verification ignored sticky status");
  expect(actual_status[126] == mtp::Status::invalid_distribution &&
             actual_results[126].output_count == 0,
         "batched greedy verification accepted invalid target ID");
  expect(actual_status[127] == mtp::Status::invalid_draft_token &&
             actual_results[127].output_count == 0,
         "batched greedy verification accepted invalid draft ID");
  expect(actual_status[128] == mtp::Status::invalid_uniform &&
             actual_results[128].output_count == 0,
         "batched greedy verification skipped final uniform validation");
  std::cout << "mtp_greedy_verification_batch: rows=129_mismatch_bonus_errors_ok\n";
}

__global__ void pack_probabilities(const float* source, mtp::TokenProbability* output,
                                    unsigned count, unsigned vocabulary) {
  const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) output[index] = {index % vocabulary, source[index]};
}

struct Verification {
  Verification(std::uint32_t vocabulary, std::uint32_t depth, bool compact = false)
      : fixture(vocabulary), depth(depth),
        target((depth + 1) * vocabulary), draft(depth * vocabulary),
        draft_ids(depth), uniforms(depth + 1), output(depth + 1), result(1),
        compact(compact), compact_target(compact ? (depth + 1) * vocabulary : 0),
        compact_draft(compact ? depth * vocabulary : 0) {}

  mtp::Result run(const std::vector<float>& p, const std::vector<float>& q,
                  const std::vector<std::uint32_t>& ids,
                  const std::vector<float>& draws) {
    fixture.clear();
    target.put(p);
    draft.put(q);
    return run_device(ids, draws);
  }

  mtp::Result run_device(const std::vector<std::uint32_t>& ids,
                         const std::vector<float>& draws) {
    draft_ids.put(ids);
    uniforms.put(draws);
    if (compact) {
      const auto size = fixture.vocabulary;
      pack_probabilities<<<((depth + 1) * size + 255) / 256, 256>>>(
          target.get(), compact_target.get(), (depth + 1) * size, size);
      if (depth)
        pack_probabilities<<<(depth * size + 255) / 256, 256>>>(
            draft.get(), compact_draft.get(), depth * size, size);
      mtp::verify_compact_sequence(compact_target.get(), compact_draft.get(),
          draft_ids.get(), depth, size, size, uniforms.get(), uniforms.get() + depth,
          output.get(), result.get(), fixture.status.get());
    } else mtp::verify_sequence(
        target.get(), depth ? draft.get() : nullptr,
        depth ? draft_ids.get() : nullptr, depth, fixture.vocabulary,
        depth ? uniforms.get() : nullptr, uniforms.get() + depth, output.get(),
        result.get(), fixture.scratch.get(), fixture.bytes, fixture.status.get());
    return result.read(1)[0];
  }
  Fixture fixture;
  std::uint32_t depth;
  Device<float> target;
  Device<float> draft;
  Device<std::uint32_t> draft_ids;
  Device<float> uniforms;
  Device<std::uint32_t> output;
  Device<mtp::Result> result;
  bool compact;
  Device<mtp::TokenProbability> compact_target, compact_draft;
};

void verification_cases(bool compact = false) {
  for (const std::uint32_t depth : {0U, 1U, 3U, 5U, 8U, 17U, mtp::kMaxDraftTokens}) {
    Verification test(3, depth, compact);
    std::vector<float> p((depth + 1) * 3);
    std::vector<float> q(depth * 3);
    const std::vector<float> row{0.25F, 0.5F, 0.25F};
    for (std::uint32_t index = 0; index <= depth; ++index) {
      std::copy(row.begin(), row.end(), p.begin() + 3 * index);
      if (index < depth) {
        std::copy(row.begin(), row.end(), q.begin() + 3 * index);
      }
    }
    const std::vector<std::uint32_t> ids(depth, 1);
    std::vector<float> draws(depth + 1, std::nextafter(1.0F, 0.0F));
    const auto result = test.run(p, q, ids, draws);
    test.fixture.success();
    expect(result.accepted_drafts == depth && result.output_count == depth + 1 &&
               result.rejected_index == depth,
           "identical probabilities must accept every proposal");
    const auto outputs = test.output.read(result.output_count);
    expect(outputs.back() == 2, "all-accepted bonus draw is wrong");
    for (std::uint32_t index = 0; index < depth; ++index) {
      expect(outputs[index] == 1, "accepted ID changed");
    }
  }

  Verification test(3, 5, compact);
  const std::vector<float> q{0.5F, 0.25F, 0.25F};
  for (std::uint32_t rejected = 0; rejected < 5; ++rejected) {
    std::vector<float> targets(18);
    std::vector<float> drafts(15);
    for (std::uint32_t index = 0; index < 6; ++index) {
      std::copy(q.begin(), q.end(), targets.begin() + 3 * index);
      if (index < 5) {
        std::copy(q.begin(), q.end(), drafts.begin() + 3 * index);
      }
    }
    // Only the selected rejection row differs. Its residual is [0,.25,.25].
    targets[3 * rejected] = 0.0F;
    targets[3 * rejected + 1] = 0.5F;
    targets[3 * rejected + 2] = 0.5F;
    auto draws = std::vector<float>(6, 0.0F);
    draws.back() = 0.75F;
    const auto result = test.run(targets, drafts, std::vector<std::uint32_t>(5, 0),
                                 draws);
    test.fixture.success();
    expect(result.accepted_drafts == rejected &&
               result.rejected_index == rejected &&
               result.output_count == rejected + 1,
           "first-rejection prefix/count is wrong");
    expect(test.output.read(result.output_count).back() == 2,
           "residual token differs or zero-target proposal was accepted at u=0");
  }

  Verification greedy(3, 3, compact);
  const auto result = greedy.run(
      {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0},
      {1, 0, 0, 1, 0, 0, 0, 0, 1}, {0, 0, 2}, {0, 0, 0, 0.9F});
  greedy.fixture.success();
  expect(result.output_count == 2 && result.accepted_drafts == 1 &&
             greedy.output.read(2) == std::vector<std::uint32_t>({0, 1}),
         "greedy one-hot verification differs from first-mismatch behavior");
  std::cout << "mtp_verification: depths_0_1_3_5_8_17_1279_and_each_rejection_ok\n";
}

void masked_verification_cases() {
  constexpr std::uint32_t vocabulary = 3, depth = 3;
  Verification test(vocabulary, depth);
  Device<__nv_bfloat16> logits(vocabulary);
  Device<std::uint32_t> masks(depth + 1);
  logits.put(bf16({0, 0, 0}));
  for (const float temperature : {0.0F, 1.0F}) {
    const auto build = [&](const std::vector<std::uint32_t>& allowed) {
      test.fixture.clear();
      masks.put(allowed);
      for (std::uint32_t row = 0; row <= depth; ++row) {
        mtp::build_distribution(logits.get(), vocabulary, temperature, 1, 0,
            test.target.get() + row * vocabulary, test.fixture.scratch.get(),
            test.fixture.bytes, test.fixture.status.get(), nullptr, masks.get() + row);
        if (row < depth)
          mtp::build_distribution(logits.get(), vocabulary, temperature, 1, 0,
              test.draft.get() + row * vocabulary, test.fixture.scratch.get(),
              test.fixture.bytes, test.fixture.status.get());
      }
      test.fixture.success();
    };
    for (const float draw : {0.0F, 0.9F}) {
      // A legal first token, then an illegal draft. Later prefix masks are
      // unrestricted because those prefixes cannot survive verification.
      build({7, 6, UINT32_MAX, UINT32_MAX});
      const auto original_q = test.draft.read(depth * vocabulary);
      const auto rejected = test.run_device({0, 0, 0}, {0, 0, 0, draw});
      test.fixture.success();
      const auto expected = temperature == 0 || draw == 0 ? 1U : 2U;
      expect(rejected.accepted_drafts == 1 && rejected.rejected_index == 1 &&
                 rejected.output_count == 2 &&
                 test.output.read(2) == std::vector<std::uint32_t>({0, expected}),
             "grammar-invalid proposal must reject at uniform zero with legal residual");
      expect(test.draft.read(depth * vocabulary) == original_q,
             "target masks changed the saved unconstrained proposal distribution");

      // All proposals are legal and accepted; the bonus must obey its own
      // prefix mask rather than the previous row's allowed set.
      build({1, 1, 1, 6});
      const auto bonus = test.run_device({0, 0, 0}, {0.99F, 0.99F, 0.99F, draw});
      test.fixture.success();
      expect(bonus.accepted_drafts == depth && bonus.output_count == depth + 1 &&
                 test.output.read(depth + 1) ==
                     std::vector<std::uint32_t>({0, 0, 0, expected}),
             "all-accepted bonus must sample its constrained target row");
    }
  }
  std::cout << "mtp_mask_verification: illegal_q_preserved_residual_bonus_greedy_sample_ok\n";
}

void invalid_data_cases(bool compact = false) {
  Verification test(2, 1, compact);
  const auto check = [&](const std::vector<float>& p, const std::vector<float>& q,
                         std::uint32_t draft, const std::vector<float>& draws,
                         mtp::Status expected) {
    const auto result = test.run(p, q, {draft}, draws);
    expect(test.fixture.status.read(1)[0] == expected,
           "unexpected invalid-input status");
    expect(result.output_count == 0, "failed verification exposed outputs");
  };
  check({0.5F, 0.5F, 0.5F, 0.5F}, {0.5F, 0.5F}, 2, {0, 0},
        mtp::Status::invalid_draft_token);
  check({0.5F, 0.5F, 0.5F, 0.5F}, {0, 1}, 0, {0, 0},
        mtp::Status::zero_draft_probability);
  check({0.5F, 0.5F, 0.5F, 0.5F}, {0.5F, 0.5F}, 0, {1, 0},
        mtp::Status::invalid_uniform);
  check({0.5F, 0.5F, 0.5F, 0.5F}, {0.5F, 0.5F}, 0, {0, -0.1F},
        mtp::Status::invalid_uniform);
  check({0.5F, 0.5F, 0.5F, 0.5F}, {0.1F, 0.1F}, 0, {0, 0},
        mtp::Status::invalid_distribution);
  check({std::numeric_limits<float>::quiet_NaN(), 0.5F, 0.5F, 0.5F},
        {0.5F, 0.5F}, 0, {0, 0}, mtp::Status::invalid_distribution);
  check({-0.1F, 1.1F, 0.5F, 0.5F}, {0.5F, 0.5F}, 0, {0, 0},
        mtp::Status::invalid_distribution);
  // Within the declared normalization tolerance, rounding can still create a
  // rejection with zero positive residual. It must fail instead of redrawing p.
  check({0.9999F, 0.0F, 0.5F, 0.5F}, {1.0F, 0.0F}, 0,
        {0.99999F, 0}, mtp::Status::zero_residual_mass);

  Fixture fixture(2);
  Device<__nv_bfloat16> logits(2);
  Device<float> probs(2);
  logits.put(bf16({std::numeric_limits<float>::infinity(), 0.0F}));
  fixture.clear();
  mtp::build_distribution(logits.get(), 2, 1.0F, 1.0F, 0, probs.get(),
                          fixture.scratch.get(), fixture.bytes,
                          fixture.status.get());
  expect(fixture.status.read(1)[0] == mtp::Status::invalid_logits,
         "non-finite input logits were accepted");
  logits.put(bf16({0.0F, 0.0F}));
  mtp::build_distribution(logits.get(), 2, 1.0F, 1.0F, 0, probs.get(),
                          fixture.scratch.get(), fixture.bytes,
                          fixture.status.get());
  expect(fixture.status.read(1)[0] == mtp::Status::invalid_logits,
         "device failure status was not sticky");
  std::cout << "mtp_invalid_data: zero_mass_nan_bounds_and_sticky_status_ok\n";
}

float random_uniform(std::mt19937_64& rng) {
  return std::min(std::uniform_real_distribution<float>(0.0F, 1.0F)(rng),
                  std::nextafter(1.0F, 0.0F));
}

void sequence_distribution_test(bool compact = false) {
  // Compare the first two emitted tokens against a nontrivial Markov target.
  // At first rejection, the next target draw is needed to finish that pair.
  constexpr std::uint32_t trials = 6'000;
  const std::array<std::vector<float>, 2> target_next{{{0.8F, 0.2F},
                                                     {0.1F, 0.9F}}};
  const std::array<std::vector<float>, 2> draft_next{{{0.4F, 0.6F},
                                                    {0.9F, 0.1F}}};
  const std::vector<float> target_first{0.3F, 0.7F};
  const std::vector<float> draft_first{0.65F, 0.35F};
  const std::array<double, 4> expected{{0.24, 0.06, 0.07, 0.63}};
  std::array<std::uint32_t, 4> counts{};
  std::array<std::uint32_t, 3> acceptance{};
  std::mt19937_64 rng(0x18fc41533621ULL);
  Verification test(2, 2, compact);
  Device<float> continuation_probs(2);
  Device<float> continuation_uniform(1);
  Device<std::uint32_t> continuation_token(1);
  for (std::uint32_t trial = 0; trial < trials; ++trial) {
    const auto first = sample_reference(draft_first, random_uniform(rng));
    const auto second = sample_reference(draft_next[first], random_uniform(rng));
    const std::vector<float> p{
        target_first[0], target_first[1], target_next[first][0],
        target_next[first][1], target_next[second][0], target_next[second][1]};
    const std::vector<float> q{
        draft_first[0], draft_first[1], draft_next[first][0], draft_next[first][1]};
    const auto result = test.run(
        p, q, {first, second},
        {random_uniform(rng), random_uniform(rng), random_uniform(rng)});
    test.fixture.success();
    ++acceptance[result.accepted_drafts];
    const auto output = test.output.read(result.output_count);
    std::uint32_t output_second;
    if (result.output_count >= 2) {
      output_second = output[1];
    } else {
      continuation_probs.put(target_next[output[0]]);
      continuation_uniform.put({random_uniform(rng)});
      mtp::sample_distribution(
          continuation_probs.get(), 2, continuation_uniform.get(),
          continuation_token.get(), test.fixture.scratch.get(), test.fixture.bytes,
          test.fixture.status.get());
      test.fixture.success();
      output_second = continuation_token.read(1)[0];
    }
    ++counts[2 * output[0] + output_second];
  }
  for (std::size_t pair = 0; pair < counts.size(); ++pair) {
    const double observed = static_cast<double>(counts[pair]) / trials;
    const double tolerance =
        6.0 * std::sqrt(expected[pair] * (1.0 - expected[pair]) / trials) + 0.002;
    expect(std::fabs(observed - expected[pair]) <= tolerance,
           "two-token output frequencies do not follow the target distribution");
  }
  expect(std::all_of(acceptance.begin(), acceptance.end(), [](auto count) {
           return count > 100;
         }), "statistical fixture failed to exercise each acceptance outcome");
  std::cout << "mtp_sequence_distribution: trials=" << trials
            << " joint_counts=[" << counts[0] << ',' << counts[1] << ','
            << counts[2] << ',' << counts[3] << "] threshold=6_sigma_plus_0.002\n";
}

void compact_distribution_tests() {
  for (const unsigned vocabulary : {7U, 35U, 1025U, 4097U, 262144U}) {
    Fixture fixture(vocabulary);
    Device<__nv_bfloat16> input(vocabulary);
    Device<std::uint32_t> mask_device(mtp::mask_words(vocabulary)), selected(1);
    Device<mtp::TokenProbability> output(mtp::kMaxCompactTopK);
    Device<float> uniform(1);
    Device<gewell::TokenLogprobs> summary(1);
    std::vector<float> values(vocabulary);
    for (unsigned token = 0; token < vocabulary; ++token)
      values[token] = float(int((token * 2654435761U) % 97) - 48) / 8;
    values[0] = -0.0F;
    values[1] = 0.0F;
    for (const bool masked : {false, true}) {
      std::vector<std::uint32_t> mask(mtp::mask_words(vocabulary));
      // Fewer allowed tokens than K, including the final partial tile.
      for (const unsigned token : {2U, vocabulary / 2, vocabulary - 1})
        mask[token / 32] |= 1U << (token % 32);
      mask_device.put(mask);
      for (const unsigned top_k : {2U, 40U, 64U, 255U, 256U}) {
        const auto size = std::min(top_k, vocabulary);
        for (const float top_p : {0.5F, 0.95F, 1.0F}) {
          const auto logits = bf16(values);
          input.put(logits);
          fixture.clear();
          mtp::build_compact_distribution(input.get(), vocabulary, 0.8F, top_p,
              size, output.get(), fixture.scratch.get(), fixture.bytes,
              fixture.status.get(), nullptr, masked ? mask_device.get() : nullptr);
          fixture.success();
          const auto compact = output.read(size);
          const auto expected = distribution_reference(logits, 0.8F, top_p, top_k,
                                                       masked ? mask : std::vector<unsigned>{});
          std::vector<float> actual(vocabulary);
          for (unsigned index = 0; index < size; ++index) {
            expect(compact[index].token < vocabulary &&
                       (!index || compact[index - 1].token < compact[index].token),
                   "compact IDs are not unique and ordered");
            actual[compact[index].token] = compact[index].probability;
          }
          for (unsigned token = 0; token < vocabulary; ++token)
            expect(std::fabs(actual[token] - expected[token]) < 2e-6F,
                   "compact top-k/top-p/mask differs from CPU reference");
          for (float draw : {0.0F, 0.371F, std::nextafter(1.0F, 0.0F)}) {
            uniform.put({draw});
            mtp::sample_compact_distribution(output.get(), size, vocabulary,
                uniform.get(), selected.get(), fixture.status.get());
            fixture.success();
            const auto chosen = selected.read(1)[0];
            if (chosen != sample_reference(expected, draw)) {
              std::cerr << "draw=" << draw << " actual=" << chosen << " expected="
                        << sample_reference(expected, draw) << " prob=" << expected[chosen] << '\n';
              throw std::runtime_error("compact CDF token differs from CPU reference");
            }
          }
          mtp::summarize_compact_logprobs(output.get(), selected.get(), 1, size,
                                          20, summary.get());
          const auto record = summary.read(1)[0];
          const auto token = selected.read(1)[0];
          expect(std::fabs(record.logprob - std::log(actual[token])) < 2e-6F,
                 "compact selected logprob differs from target distribution");
          std::vector<unsigned> order;
          for (unsigned id = 0; id < vocabulary; ++id)
            if (expected[id] > 0) order.push_back(id);
          std::stable_sort(order.begin(), order.end(), [&](unsigned a, unsigned b) {
            return expected[a] > expected[b];
          });
          expect(record.count == std::min<std::size_t>(20, order.size()),
                 "compact alternative count differs");
          for (unsigned rank = 0; rank < record.count; ++rank)
            expect(record.top[rank].token == order[rank],
                   "compact logprob tie order differs");
        }
      }
    }
    // Fully tied rows exercise tie preservation across all selection levels.
    input.put(bf16(std::vector<float>(vocabulary, -0.0F)));
    const auto size = std::min(mtp::kMaxCompactTopK, vocabulary);
    fixture.clear();
    mtp::build_compact_distribution(input.get(), vocabulary, 1, 0.5F, size,
        output.get(), fixture.scratch.get(), fixture.bytes, fixture.status.get());
    fixture.success();
    const auto ties = output.read(size);
    const unsigned support = (size + 1) / 2;
    for (unsigned index = 0; index < size; ++index) {
      expect(ties[index].token == index, "compact selection lost lower-ID ties");
      expect_close(ties[index].probability, index < support ? 1.0F / support : 0,
                   "compact exact nucleus boundary changed");
    }
    mask_device.put(std::vector<unsigned>(mtp::mask_words(vocabulary), 0));
    fixture.clear();
    mtp::build_compact_distribution(input.get(), vocabulary, 1, 1, size,
        output.get(), fixture.scratch.get(), fixture.bytes, fixture.status.get(),
        nullptr, mask_device.get());
    expect(fixture.status.read(1)[0] == mtp::Status::invalid_distribution,
           "compact all-masked row succeeded");
    values.back() = std::numeric_limits<float>::infinity();
    input.put(bf16(values));
    fixture.clear();
    mtp::build_compact_distribution(input.get(), vocabulary, 1, 1, size,
        output.get(), fixture.scratch.get(), fixture.bytes, fixture.status.get(),
        nullptr, mask_device.get());
    expect(fixture.status.read(1)[0] == mtp::Status::invalid_logits,
           "compact filtering hid a non-finite raw logit");
  }
  expect(mtp::compact_row_size(262144, 0.8F, 0.95F, 40) == 40 &&
             mtp::compact_row_size(262144, 0.8F, 0.95F, 0) == 0 &&
             mtp::compact_row_size(262144, 0.8F, 0.95F, 257) == 0 &&
             mtp::compact_row_size(262144, 0, 0.95F, 40) == 0,
         "compact dispatch bounds changed");
  std::cout << "mtp_compact_distribution: full_vocab_partial_tiles_masks_ties_cdf_logprobs_ok\n";
}

void compact_batch_tests() {
  constexpr unsigned count = 33, stride = mtp::kMaxCompactTopK + 2;
  for (unsigned vocabulary : {7U, 1025U, 4097U, 262144U}) {
    const auto bytes = mtp::scratch_bytes(vocabulary);
    const auto scratch_stride = bytes + 256;
    Device<unsigned char> scratch(count * scratch_stride);
    Device<__nv_bfloat16> logits(std::size_t(count) * vocabulary);
    Device<unsigned> masks(count * mtp::mask_words(vocabulary));
    Device<mtp::TokenProbability> reference(count * stride), actual(count * stride);
    Device<mtp::Status> ref_status(count), status(count);
    std::vector<mtp::TokenProbability> guarded(count * stride, {UINT32_MAX, -123.0F});
    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream), "create compact batch stream");
    for (unsigned mode = 0; mode < 4; ++mode) {
      std::vector<__nv_bfloat16> values(std::size_t(count) * vocabulary);
      std::vector<unsigned> allowed(count * mtp::mask_words(vocabulary), 0);
      for (unsigned row = 0; row < count; ++row) {
        for (unsigned token = 0; token < vocabulary; ++token)
          values[std::size_t(row) * vocabulary + token] = __float2bfloat16(
              mode == 1 ? -0.0F : float(int((token * 2654435761U + row * 97) % 113) - 56) / 8);
        for (unsigned token : {2U, vocabulary / 2, vocabulary - 1})
          allowed[row * mtp::mask_words(vocabulary) + token / 32] |= 1U << (token % 32);
      }
      if (mode == 2) {
        std::fill_n(allowed.begin() + mtp::mask_words(vocabulary), mtp::mask_words(vocabulary), 0);
        values.back() = __float2bfloat16(std::numeric_limits<float>::infinity());
      }
      logits.put(values); masks.put(allowed);
      reference.put(guarded); actual.put(guarded);
      auto statuses = std::vector<mtp::Status>(count, mtp::Status::success);
      if (mode == 2) statuses[3] = mtp::Status::invalid_uniform;
      ref_status.put(statuses); status.put(statuses);
      check_cuda(cudaMemset(scratch.get(), 0xa5, count * scratch_stride), "guard compact scratch");
      std::vector<mtp::CompactDistributionInput> batch;
      for (unsigned row = 0; row < count; ++row) {
        const unsigned size = std::min(vocabulary, mode == 3
            ? (row % 2 ? 40U : 256U) : (mode == 0 || row >= 16 ? 40U : 256U));
        const float temperature = 0.5F + float(row % 5) / 8;
        const float top_p = row % 3 == 0 ? 1.0F : row % 3 == 1 ? 0.5F : 0.95F;
        const auto* mask = (row % 2 || mode == 2) ? masks.get() + row * mtp::mask_words(vocabulary) : nullptr;
        mtp::build_compact_distribution(logits.get() + std::size_t(row) * vocabulary,
            vocabulary, temperature, top_p, size, reference.get() + row * stride + 1,
            scratch.get() + row * scratch_stride, bytes, ref_status.get() + row, stream, mask);
        batch.push_back({logits.get() + std::size_t(row) * vocabulary,
            temperature, top_p, size, actual.get() + row * stride + 1,
            scratch.get() + row * scratch_stride, bytes, status.get() + row, mask});
      }
      check_cuda(cudaStreamSynchronize(stream), "finish compact row reference");
      mtp::build_compact_distributions(batch, vocabulary, stream);
      check_cuda(cudaStreamSynchronize(stream), "finish compact batch");
      const auto compare = [&] {
        const auto expected = reference.read(count * stride), observed = actual.read(count * stride);
        expect(std::memcmp(expected.data(), observed.data(), expected.size() * sizeof(expected[0])) == 0,
               "compact batch differs from serial rows or overwrites output guards");
        expect(ref_status.read(count) == status.read(count), "compact batch sticky statuses differ");
      };
      compare();
      if (mode == 2) {
        const auto observed = status.read(count);
        expect(observed[1] == mtp::Status::invalid_distribution &&
                   observed[3] == mtp::Status::invalid_uniform &&
                   observed.back() == mtp::Status::invalid_logits,
               "compact batch error isolation failed");
      }
      // Capture and repeated replay reuse only caller-owned, disjoint scratch.
      cudaGraph_t graph;
      cudaGraphExec_t executable;
      check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture compact batch");
      mtp::build_compact_distributions(batch, vocabulary, stream);
      check_cuda(cudaStreamEndCapture(stream, &graph), "end compact capture");
      check_cuda(cudaGraphInstantiate(&executable, graph, 0), "instantiate compact graph");
      for (unsigned repeat = 0; repeat < 2; ++repeat)
        check_cuda(cudaGraphLaunch(executable, stream), "replay compact graph");
      check_cuda(cudaStreamSynchronize(stream), "finish compact replay");
      compare();
      check_cuda(cudaGraphExecDestroy(executable), "destroy compact executable");
      check_cuda(cudaGraphDestroy(graph), "destroy compact graph");
      const auto unchanged = logits.read(values.size());
      expect(std::memcmp(values.data(), unchanged.data(), values.size() * sizeof(values[0])) == 0,
             "compact batch modified input logits");
      const auto workspace = scratch.read(count * scratch_stride);
      for (unsigned row = 0; row < count; ++row)
        expect(std::all_of(workspace.begin() + row * scratch_stride + bytes,
                           workspace.begin() + (row + 1) * scratch_stride,
                           [](auto byte) { return byte == 0xa5; }), "compact batch scratch guard changed");
      const auto invalid = [&](auto rows) {
        bool rejected = false;
        try { mtp::build_compact_distributions(rows, vocabulary, stream); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect(rejected, "invalid compact batch inputs accepted");
      };
      invalid(std::vector<mtp::CompactDistributionInput>{});
      auto bad = batch; bad.back().row_size = 0; invalid(bad);
      bad = batch; bad.back().scratch_size = bytes - 1; invalid(bad);
      bad = batch; bad.back().temperature = 0; invalid(bad);
      bad = batch; bad.back().top_p = std::numeric_limits<float>::quiet_NaN(); invalid(bad);
      bad = batch; bad.back().output = nullptr; invalid(bad);
    }
    check_cuda(cudaStreamDestroy(stream), "destroy compact batch stream");
  }
  std::cout << "mtp_compact_batch: exact_rows_mixed_settings_masks_errors_guards_capture_boundary_ok\n";
}

void compact_disjoint_support_test() {
  Fixture fixture(100);
  Device<mtp::TokenProbability> p(6), q(3);
  Device<unsigned> ids(1), output(2);
  Device<float> uniforms(2);
  Device<mtp::Result> result(1);
  p.put({{3, 0}, {73, 0.75F}, {99, 0.25F},
         {3, 0}, {73, 0.75F}, {99, 0.25F}});
  q.put({{3, 0.5F}, {42, 0.25F}, {99, 0.25F}});
  ids.put({3});
  uniforms.put({0, 0.999F});
  fixture.clear();
  mtp::verify_compact_sequence(p.get(), q.get(), ids.get(), 1, 3, 100,
      uniforms.get(), uniforms.get() + 1, output.get(), result.get(), fixture.status.get());
  fixture.success();
  expect(result.read(1)[0].accepted_drafts == 0 && output.read(1)[0] == 73,
         "compact correction lost target support absent from draft");
  q.put({{3, 0.5F}, {3, 0.25F}, {99, 0.25F}});
  fixture.clear();
  mtp::verify_compact_sequence(p.get(), q.get(), ids.get(), 1, 3, 100,
      uniforms.get(), uniforms.get() + 1, output.get(), result.get(), fixture.status.get());
  expect(fixture.status.read(1)[0] == mtp::Status::invalid_distribution &&
             result.read(1)[0].output_count == 0, "duplicate compact token ID accepted");
  std::cout << "mtp_compact_support: disjoint_residual_duplicate_validation_ok\n";
}

void full_vocabulary_test() {
  constexpr std::uint32_t vocabulary = 262'144;
  Fixture fixture(vocabulary);
  Device<__nv_bfloat16> logits(vocabulary);
  Device<float> probabilities(vocabulary);
  Device<float> uniform(1);
  Device<std::uint32_t> token(1);
  check_cuda(cudaMemset(logits.get(), 0, vocabulary * sizeof(__nv_bfloat16)),
             "initialize full-vocabulary logits");
  fixture.clear();
  mtp::build_distribution(logits.get(), vocabulary, 0.8F, 1.0F, 0,
                          probabilities.get(), fixture.scratch.get(),
                          fixture.bytes, fixture.status.get());
  uniform.put({0.5F});
  mtp::sample_distribution(probabilities.get(), vocabulary, uniform.get(),
                           token.get(), fixture.scratch.get(), fixture.bytes,
                           fixture.status.get());
  fixture.success();
  expect(token.read(1)[0] == vocabulary / 2,
         "full-vocabulary canonical uniform draw mismatch");
  uniform.put({std::nextafter(1.0F, 0.0F)});
  mtp::sample_distribution(probabilities.get(), vocabulary, uniform.get(),
                           token.get(), fixture.scratch.get(), fixture.bytes,
                           fixture.status.get());
  fixture.success();
  expect(token.read(1)[0] == vocabulary - 1,
         "full-vocabulary final CDF bucket mismatch");

  // Argmax must preserve lower-ID ties across reduction partitions and masks.
  std::vector<float> values(vocabulary, -2.0F);
  for (const auto index : {255U, 256U, 8192U, vocabulary - 1}) values[index] = 3.0F;
  logits.put(bf16(values));
  for (const auto settings : std::array<std::array<float, 3>, 3>{{
           {0, 1, 0}, {0.8F, 0, 0}, {0.8F, 1, 1}}}) {
    fixture.clear();
    mtp::build_distribution(logits.get(), vocabulary, settings[0], settings[1],
        static_cast<std::uint32_t>(settings[2]), probabilities.get(),
        fixture.scratch.get(), fixture.bytes, fixture.status.get());
    mtp::sample_distribution(probabilities.get(), vocabulary, uniform.get(),
        token.get(), fixture.scratch.get(), fixture.bytes, fixture.status.get());
    fixture.success();
    expect(token.read(1)[0] == 255, "full-vocabulary greedy tie order changed");
    const auto row = probabilities.read(vocabulary);
    for (unsigned index = 0; index < vocabulary; ++index)
      expect(row[index] == (index == 255 ? 1.0F : 0.0F), "greedy row is not one-hot");
  }
  Device<std::uint32_t> allowed(mtp::mask_words(vocabulary));
  std::vector<std::uint32_t> mask(mtp::mask_words(vocabulary), 0);
  for (const auto index : {256U, 8192U, vocabulary - 1}) mask[index / 32] |= 1U << (index % 32);
  allowed.put(mask);
  fixture.clear();
  mtp::build_distribution(logits.get(), vocabulary, 0, 1, 0, probabilities.get(),
      fixture.scratch.get(), fixture.bytes, fixture.status.get(), nullptr, allowed.get());
  fixture.success();
  expect(probabilities.read(vocabulary)[256] == 1.0F, "masked greedy argmax selected a forbidden winner");
  allowed.put(std::vector<std::uint32_t>(mask.size(), 0));
  fixture.clear();
  mtp::build_distribution(logits.get(), vocabulary, 0, 1, 0, probabilities.get(),
      fixture.scratch.get(), fixture.bytes, fixture.status.get(), nullptr, allowed.get());
  expect(fixture.status.read(1)[0] == mtp::Status::invalid_distribution,
         "empty full-vocabulary greedy mask was accepted");
  std::cout << "mtp_full_vocabulary: vocabulary=" << vocabulary
            << " scratch_bytes=" << fixture.bytes << '\n';
}

void dense_zero_mass_rounding_test() {
  constexpr unsigned vocabulary = 262144;
  Fixture fixture(vocabulary);
  Device<__nv_bfloat16> logits(vocabulary);
  Device<float> probabilities(vocabulary), target(2 * vocabulary), draft(vocabulary);
  Device<float> uniform(1);
  Device<unsigned> token(1), output(2);
  Device<mtp::Result> result(1);
  // Reproduces the production failure with finite BF16 logits, top-k disabled
  // and top-p=1. A parallel FP32 scan can increase across a zero-weight bucket.
  std::mt19937 rng(20260922);
  std::uniform_real_distribution<float> log_weight(-12.0F, 0.0F);
  std::vector<float> weights(vocabulary, 0.0F);
  const auto support = 8 + rng() % 249;
  for (unsigned i = 0; i < support; ++i) {
    const float weight = std::exp(log_weight(rng));
    weights[rng() % vocabulary] += weight;
  }
  for (auto& weight : weights) weight = weight > 0 ? std::log(weight) : -30.0F;
  logits.put(bf16(weights));
  fixture.clear();
  mtp::build_distribution(logits.get(), vocabulary, 0.7F, 1.0F, 0,
      probabilities.get(), fixture.scratch.get(), fixture.bytes, fixture.status.get());
  fixture.success();
  const auto row = probabilities.read(vocabulary);
  check_cuda(cudaMemcpy(target.get(), probabilities.get(), vocabulary * sizeof(float),
                        cudaMemcpyDeviceToDevice), "copy rounding target row");
  check_cuda(cudaMemcpy(target.get() + vocabulary, probabilities.get(), vocabulary * sizeof(float),
                        cudaMemcpyDeviceToDevice), "copy rounding bonus row");
  const auto zero = static_cast<unsigned>(std::find(row.begin(), row.end(), 0.0F) - row.begin());
  expect(zero < vocabulary, "rounding fixture has no zero-probability token");
  std::vector<float> one_hot(vocabulary, 0.0F);
  one_hot[zero] = 1.0F;
  draft.put(one_hot);

  for (const float draw : {0.2074185908F, 0.0F, 0.5F, std::nextafter(1.0F, 0.0F)}) {
    uniform.put({draw});
    fixture.clear();
    mtp::sample_distribution(probabilities.get(), vocabulary, uniform.get(), token.get(),
        fixture.scratch.get(), fixture.bytes, fixture.status.get());
    fixture.success();
    const auto selected = token.read(1)[0];
    expect(selected < vocabulary && row[selected] > 0,
           "dense CDF rounding selected a zero-probability proposal");
    mtp::verify_sequence(target.get(), probabilities.get(), token.get(), 1, vocabulary,
        uniform.get(), uniform.get(), output.get(), result.get(), fixture.scratch.get(),
        fixture.bytes, fixture.status.get());
    fixture.success();
    expect(result.read(1)[0].accepted_drafts == 1,
           "identical target rejected the rounded-CDF proposal");
    for (const auto id : output.read(2))
      expect(id < vocabulary && row[id] > 0, "rounded CDF emitted a zero-mass bonus");

    // Force rejection: q is one-hot where p is zero, so the residual is p.
    token.put({zero});
    fixture.clear();
    mtp::verify_sequence(target.get(), draft.get(), token.get(), 1, vocabulary,
        uniform.get(), uniform.get(), output.get(), result.get(), fixture.scratch.get(),
        fixture.bytes, fixture.status.get());
    fixture.success();
    expect(result.read(1)[0].accepted_drafts == 0,
           "disjoint proposal was not rejected");
    const auto correction = output.read(1)[0];
    expect(correction < vocabulary && row[correction] > 0,
           "rounded CDF emitted a zero-mass correction");
  }
  std::cout << "mtp_dense_rounding: proposal_bonus_correction_support_ok\n";
}

void capture_scores_test() {
  constexpr unsigned vocabulary = 5, depth = 2;
  Fixture fixture(vocabulary);
  Device<float> target(vocabulary * depth), draft(vocabulary * depth);
  Device<std::uint32_t> ids(depth), greedy(depth);
  Device<gewell::MtpCaptureScores> scores(depth);
  const std::vector<float> p = {0.25F,0,0.75F,0,0, 0,0,0,1,0};
  const std::vector<float> q = {0.5F,0,0.5F,0,0, 0,0,0,0.25F,0.75F};
  target.put(p); draft.put(q); ids.put({2,4}); greedy.put({2,3}); fixture.clear();
  mtp::capture_scores(target.get(), draft.get(), ids.get(), depth, vocabulary, 0,
      nullptr, fixture.status.get(), scores.get());
  fixture.success();
  const auto dense = scores.read(depth);
  expect(dense[0].draft_probability == 0.5F && dense[0].target_probability == 0.75F &&
      dense[1].draft_probability == 0.75F && dense[1].target_probability == 0 &&
      std::abs(dense[0].draft_entropy - std::log(2.0F)) < 1e-6F &&
      dense[1].draft_max_probability == 0.75F, "dense capture alignment or entropy changed");
  Device<mtp::TokenProbability> compact_target(4), compact_draft(4);
  compact_target.put({{0,0.25F},{2,0.75F},{1,0},{3,1}});
  compact_draft.put({{0,0.5F},{2,0.5F},{3,0.25F},{4,0.75F}});
  mtp::capture_scores(compact_target.get(), compact_draft.get(), ids.get(), depth, vocabulary, 2,
      nullptr, fixture.status.get(), scores.get());
  const auto compact = scores.read(depth);
  for (unsigned i = 0; i < depth; ++i)
    expect(compact[i].draft_probability == dense[i].draft_probability &&
        compact[i].target_probability == dense[i].target_probability &&
        std::abs(compact[i].draft_entropy - dense[i].draft_entropy) < 1e-6F &&
        compact[i].draft_max_probability == dense[i].draft_max_probability,
        "compact capture differs from dense with disjoint support");
  mtp::capture_scores(nullptr, nullptr, ids.get(), depth, vocabulary, 0,
      greedy.get(), fixture.status.get(), scores.get());
  const auto winners = scores.read(depth);
  expect(winners[0].target_probability == 1 && winners[1].target_probability == 0 &&
      winners[1].draft_probability == 1 && winners[1].draft_entropy == 0 &&
      winners[1].draft_max_probability == 1, "ID-only greedy capture read probability scratch");
  expect(target.read(p.size()) == p && draft.read(q.size()) == q && ids.read(depth) == std::vector<std::uint32_t>({2,4}),
      "capture mutated inference inputs");
  fixture.status.put({mtp::Status::invalid_distribution});
  mtp::capture_scores(nullptr, nullptr, ids.get(), depth, vocabulary, 0,
      greedy.get(), fixture.status.get(), scores.get());
  expect(scores.read(depth)[0].draft_probability == 0 &&
      fixture.status.read(1)[0] == mtp::Status::invalid_distribution, "capture changed failed status");
  Device<std::uint16_t> probes(12), selected_rows(4);
  Device<mtp::Result> result(1);
  probes.put({10,11,20,21,30,31, 40,41,50,51,60,61});
  fixture.clear();
  for (unsigned count : {1U,2U,3U}) {
    result.put({{count - 1, count, count - 1}});
    mtp::gather_capture_rows(probes.get(), 2, 3, 2, result.get(), fixture.status.get(), selected_rows.get());
    const std::vector<std::uint16_t> expected = {std::uint16_t(count * 10), std::uint16_t(count * 10 + 1),
        std::uint16_t(count * 10 + 30), std::uint16_t(count * 10 + 31)};
    expect(selected_rows.read(4) == expected, "target probes did not select the last committed input row");
  }
  fixture.status.put({mtp::Status::invalid_distribution});
  mtp::gather_capture_rows(probes.get(), 2, 3, 2, result.get(), fixture.status.get(), selected_rows.get());
  expect(selected_rows.read(4) == std::vector<std::uint16_t>(4, 0), "failed verifier leaked stale probe rows");
  std::cout << "mtp_capture_scores: dense_compact_greedy_entropy_and_read_only_ok\n";
}

void partitioned_validation_test() {
  constexpr unsigned vocabulary = 8193, depth = 3;
  Verification test(vocabulary, depth);
  const std::vector<float> valid_target((depth + 1) * vocabulary, 1.0F / vocabulary);
  const std::vector<float> valid_draft(depth * vocabulary, 1.0F / vocabulary);
  const std::vector<std::uint32_t> ids(depth, vocabulary - 1);
  const std::vector<float> draws(depth + 1, std::nextafter(1.0F, 0.0F));
  const auto result = test.run(valid_target, valid_draft, ids, draws);
  test.fixture.success();
  expect(result.accepted_drafts == depth && test.output.read(depth + 1).back() == vocabulary - 1,
         "partitioned validation rejected a normalized multi-row distribution");
  for (const unsigned column : {0U, 255U, 256U, 8192U}) {
    auto corrupted = valid_target;
    corrupted[2 * vocabulary + column] = std::numeric_limits<float>::quiet_NaN();
    const auto failed = test.run(corrupted, valid_draft, ids, draws);
    expect(test.fixture.status.read(1)[0] == mtp::Status::invalid_distribution && failed.output_count == 0,
           "partitioned validation missed a corrupt token or row");
  }
  auto wrong_mass = valid_draft;
  std::fill(wrong_mass.begin() + vocabulary, wrong_mass.begin() + 2 * vocabulary, 0.0F);
  const auto failed = test.run(valid_target, wrong_mass, ids, draws);
  expect(test.fixture.status.read(1)[0] == mtp::Status::invalid_distribution && failed.output_count == 0,
         "partitioned validation missed an unnormalized draft row");
  std::cout << "mtp_partitioned_validation: normalized_rows_nan_partitions_and_mass_ok\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::strcmp(argv[1], "--capture") == 0) {
      capture_scores_test(); return 0;
    }
    if (argc == 2 && std::strcmp(argv[1], "--compact-batch") == 0) {
      compact_batch_tests();
      std::cout << "mtp_sampling_compact_batch: PASS\n";
      return 0;
    }
    expect(argc == 1, "expected no arguments or --compact-batch");
    capture_scores_test();
    logprob_summary_tests();
    full_top_logprob_summary_test();
    invalid_logprob_argument_tests();
    distribution_tests();
    masked_distribution_tests();
    greedy_batch_tests();
    greedy_verification_batch_tests();
    verification_cases();
    masked_verification_cases();
    invalid_data_cases();
    sequence_distribution_test();
    full_vocabulary_test();
    dense_zero_mass_rounding_test();
    partitioned_validation_test();
    compact_distribution_tests();
    compact_batch_tests();
    compact_disjoint_support_test();
    verification_cases(true);
    invalid_data_cases(true);
    sequence_distribution_test(true);
    std::cout << "mtp_sampling: PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "mtp_sampling: FAIL: " << error.what() << '\n';
    return 1;
  }
}
