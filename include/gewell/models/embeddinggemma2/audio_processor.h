#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace gewell::embeddinggemma2::audio {
inline constexpr int kSampleRate = 16000;
inline constexpr int kFeatureWidth = 128;
inline constexpr std::size_t kMaxDataUrlBytes = 8 * 1024 * 1024;
inline constexpr std::size_t kMaxDecodedFrameBytes = 16 * 1024 * 1024;

struct Input {
  std::vector<float> features;
  std::vector<std::uint8_t> mask;
  std::uint32_t samples{}, begin{}, end{};
};
struct Metadata {
  int input_rate{}, channels{};
  std::uint32_t samples{};
  std::string layout, codec;
};
struct Prepared {
  std::shared_ptr<Input> input;
  Metadata metadata;
};
using Poll = std::function<void()>;
// Reserve/own the retained feature and mask capacity before it is allocated.
using Acquire = std::function<std::shared_ptr<Input>(std::uint32_t feature_rows)>;
using ObserveSamples = std::function<void(const std::vector<float>&)>;

std::uint32_t feature_rows(std::uint32_t samples);
std::uint32_t max_samples(int max_tokens);
std::shared_ptr<Input> prepare_samples(const std::vector<float>& samples,
    int max_tokens = 8192, const Acquire& acquire = {}, const Poll& poll = {});
Prepared prepare_data_url(std::string_view url, int max_tokens = 8192,
    const Acquire& acquire = {}, const Poll& poll = {}, const ObserveSamples& observe = {});
void validate_features(const Input& input);
}  // namespace gewell::embeddinggemma2::audio
