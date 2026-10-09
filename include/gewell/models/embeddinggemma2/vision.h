#pragma once

#include "gewell/component_weights.h"
#include "json.hpp"

namespace gewell::embeddinggemma2::vision {
inline constexpr int kLayers = 16;
inline constexpr int kHidden = 768;
inline constexpr int kIntermediate = 3072;
inline constexpr int kHeads = 12;
inline constexpr int kHeadDim = 64;
inline constexpr int kPatchWidth = 768;
inline constexpr int kPositionCount = 10240;
inline constexpr int kMaxSoftTokens = 1120;
inline constexpr int kMaxPatches = kMaxSoftTokens * 9;
inline constexpr int kSharedTensors = 3;
inline constexpr int kLayerTensors = 13;
inline constexpr int kTensorCount = kSharedTensors + kLayers * kLayerTensors;
enum SharedWeight { PatchProjection, PositionEmbedding, BridgeProjection };
enum LayerWeight {
  InputNorm, Query, Key, Value, QueryNorm, KeyNorm, AttentionOutput,
  PostAttentionNorm, PreFeedforwardNorm, Gate, Up, Down, PostFeedforwardNorm
};
constexpr int weight_id(int layer, LayerWeight role) {
  return kSharedTensors + layer * kLayerTensors + role;
}
std::vector<component::TensorSpec> tensor_specs();
void validate_config(const nlohmann::json& config);
void validate_processor(const nlohmann::json& processor);
void validate_bundle_config(const std::string& directory);
}  // namespace gewell::embeddinggemma2::vision
