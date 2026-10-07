#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace gewell::gemma4_26b_a4b {

inline constexpr std::uint32_t kLayerCount = 30;
inline constexpr std::uint32_t kLocalLayerCount = 25;
inline constexpr std::uint32_t kGlobalLayerCount = 5;
inline constexpr std::uint32_t kHiddenSize = 2'816;
inline constexpr std::uint32_t kMlpSize = 2'112;
inline constexpr std::uint32_t kExpertMlpSize = 704;
inline constexpr std::uint32_t kExpertCount = 128;
inline constexpr std::uint32_t kSelectedExpertCount = 8;
inline constexpr std::uint32_t kQueryHeadCount = 16;
inline constexpr std::uint32_t kLocalKvHeadCount = 8;
inline constexpr std::uint32_t kGlobalKvHeadCount = 2;
inline constexpr std::uint32_t kLocalHeadSize = 256;
inline constexpr std::uint32_t kGlobalHeadSize = 512;
inline constexpr std::uint32_t kLocalWindowSize = 1'024;
inline constexpr std::uint32_t kVocabSize = 262'144;
inline constexpr std::uint32_t kMaxPositions = 262'144;
inline constexpr float kEmbeddingScale = 53.0f;
inline constexpr float kRmsEpsilon = 1e-6f;
inline constexpr float kFinalLogitSoftcap = 30.0f;
inline constexpr std::size_t kTextPhysicalTensorCount = 12'117;
inline constexpr std::size_t kEmbeddingPhysicalId = 0;
inline constexpr std::size_t kFinalNormPhysicalId = kTextPhysicalTensorCount - 1;
inline constexpr std::size_t kLmHeadPhysicalId = kEmbeddingPhysicalId;
inline constexpr std::size_t kLmHeadLogicalId = kTextPhysicalTensorCount;
inline constexpr std::int16_t kNoLayer = -1;
inline constexpr std::int16_t kNoExpert = -1;
inline constexpr std::uint64_t kStorageAlignment = 4'096;

// Values are local to this model's artifact, not 31B tensor IDs.
enum class TensorRole : std::uint16_t {
  embedding, input_norm, q_proj, k_proj, v_proj, q_norm, k_norm, o_proj,
  post_attention_norm, pre_feedforward_norm, gate_proj, up_proj, down_proj,
  post_feedforward_norm, layer_scalar, final_norm,
  pre_feedforward_norm_2, post_feedforward_norm_1, post_feedforward_norm_2,
  router_proj, router_scale, router_per_expert_scale,
  expert_gate_proj, expert_up_proj, expert_down_proj,
};

struct TensorSpec {
  TensorRole role{};
  std::int16_t layer{kNoLayer};
  std::int16_t expert{kNoExpert};
  std::uint8_t rank{};
  std::uint32_t rows{};
  std::uint32_t columns{};

  [[nodiscard]] constexpr std::uint64_t element_count() const {
    return static_cast<std::uint64_t>(rows) * (rank == 1 ? 1 : columns);
  }
  [[nodiscard]] constexpr std::uint64_t byte_count() const { return element_count() * 2; }
};

[[nodiscard]] constexpr bool is_global_layer(std::uint32_t layer) { return layer % 6 == 5; }
[[nodiscard]] constexpr std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment = kStorageAlignment) {
  return (value + alignment - 1) / alignment * alignment;
}

[[nodiscard]] constexpr auto make_text_tensors() {
  std::array<TensorSpec, kTextPhysicalTensorCount> result{};
  std::size_t id = 0;
  result[id++] = {TensorRole::embedding, kNoLayer, kNoExpert, 2, kVocabSize, kHiddenSize};
  for (std::int16_t layer = 0; layer < kLayerCount; ++layer) {
    const bool global = is_global_layer(layer);
    const auto head = global ? kGlobalHeadSize : kLocalHeadSize;
    const auto query = kQueryHeadCount * head;
    const auto kv = (global ? kGlobalKvHeadCount : kLocalKvHeadCount) * head;
    const auto vector = [&](TensorRole role, std::uint32_t width) constexpr {
      result[id++] = {role, layer, kNoExpert, 1, width, 0};
    };
    const auto matrix = [&](TensorRole role, std::uint32_t rows, std::uint32_t columns) constexpr {
      result[id++] = {role, layer, kNoExpert, 2, rows, columns};
    };
    vector(TensorRole::input_norm, kHiddenSize);
    matrix(TensorRole::q_proj, query, kHiddenSize);
    matrix(TensorRole::k_proj, kv, kHiddenSize);
    if (!global) matrix(TensorRole::v_proj, kv, kHiddenSize);
    vector(TensorRole::q_norm, head);
    vector(TensorRole::k_norm, head);
    matrix(TensorRole::o_proj, kHiddenSize, query);
    vector(TensorRole::post_attention_norm, kHiddenSize);
    vector(TensorRole::pre_feedforward_norm, kHiddenSize);
    matrix(TensorRole::gate_proj, kMlpSize, kHiddenSize);
    matrix(TensorRole::up_proj, kMlpSize, kHiddenSize);
    matrix(TensorRole::down_proj, kHiddenSize, kMlpSize);
    vector(TensorRole::post_feedforward_norm, kHiddenSize);
    vector(TensorRole::layer_scalar, 1);
    vector(TensorRole::pre_feedforward_norm_2, kHiddenSize);
    vector(TensorRole::post_feedforward_norm_1, kHiddenSize);
    vector(TensorRole::post_feedforward_norm_2, kHiddenSize);
    matrix(TensorRole::router_proj, kExpertCount, kHiddenSize);
    vector(TensorRole::router_scale, kHiddenSize);
    vector(TensorRole::router_per_expert_scale, kExpertCount);
    for (std::int16_t expert = 0; expert < kExpertCount; ++expert) {
      result[id++] = {TensorRole::expert_gate_proj, layer, expert, 2, kExpertMlpSize, kHiddenSize};
      result[id++] = {TensorRole::expert_up_proj, layer, expert, 2, kExpertMlpSize, kHiddenSize};
      result[id++] = {TensorRole::expert_down_proj, layer, expert, 2, kHiddenSize, kExpertMlpSize};
    }
  }
  result[id++] = {TensorRole::final_norm, kNoLayer, kNoExpert, 1, kHiddenSize, 0};
  return result;
}

inline constexpr auto kTextTensors = make_text_tensors();
[[nodiscard]] constexpr std::uint64_t text_weight_bytes() {
  std::uint64_t result = 0;
  for (const auto& tensor : kTextTensors) result += tensor.byte_count();
  return result;
}
static_assert(kTextTensors[kFinalNormPhysicalId].role == TensorRole::final_norm);
static_assert(text_weight_bytes() == 50'466'283'580ULL);

}  // namespace gewell::gemma4_26b_a4b
