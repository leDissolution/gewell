#pragma once

#include "gewell/component_weights.h"
#include "json.hpp"

namespace gewell::embeddinggemma2 {
inline constexpr int kLayers = 24;
inline constexpr int kHidden = 512;
inline constexpr int kIntermediate = 2048;
inline constexpr int kVocabulary = 262144;
inline constexpr int kMaxTokens = 8192;
inline constexpr int kDimension = 768;
inline constexpr int kHeads = 4;
inline constexpr int kWindow = 512;
inline constexpr float kEpsilon = 1.e-6f;
inline constexpr int kSharedTensors = 5;
inline constexpr int kLayerTensors = 17;
inline constexpr int kTensorCount = kSharedTensors + kLayers * kLayerTensors;
enum SharedWeight { Embedding, PleProjection, PleNorm, FinalNorm, OutputProjection };
enum LayerWeight {
  InputNorm, Query, Key, Value, QueryNorm, KeyNorm, AttentionOutput,
  PostAttentionNorm, PreFeedforwardNorm, Gate, Up, Down, PostFeedforwardNorm,
  PleGate, PleOutput, PlePostNorm, Scalar
};
constexpr bool global_layer(int layer) { return layer % 6 == 5; }
constexpr int head_dim(int layer) { return global_layer(layer) ? 512 : 256; }
constexpr int kv_heads(int layer) { return global_layer(layer) ? 1 : 2; }
constexpr int weight_id(int layer, LayerWeight role) {
  return kSharedTensors + layer * kLayerTensors + role;
}
constexpr bool supported_dimension(int dimension) {
  return dimension == 128 || dimension == 256 || dimension == 512 || dimension == 768;
}
// Production import is structural: compatible fine-tunes need no repository/revision pin.
std::vector<component::TensorSpec> tensor_specs();
void validate_config(const nlohmann::json& config);
void validate_bundle_config(const std::string& directory);
}  // namespace gewell::embeddinggemma2
