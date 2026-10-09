#pragma once

#include "gewell/component_weights.h"
#include "json.hpp"

namespace gewell::embeddinggemma2::audio {
inline constexpr int kLayers = 12;
inline constexpr int kHidden = 1024;
inline constexpr int kIntermediate = 4096;
inline constexpr int kHeads = 8;
inline constexpr int kHeadDim = 128;
inline constexpr int kOutputWidth = 1536;
inline constexpr int kSharedTensors = 8;
inline constexpr int kLayerTensors = 62;
inline constexpr int kTensorCount = kSharedTensors + kLayers * kLayerTensors;
enum SharedWeight {Conv0, Conv0Norm, Conv1, Conv1Norm, InputProjection, OutputProjection, OutputBias, Bridge};
// Clipped linear IDs name their weight, followed by input_min/input_max and
// output_min/output_max. All four bounds are scalar BF16 source tensors.
enum LayerWeight {
  FF1PreNorm=0, FF1Up=1, FF1Down=6, FF1PostNorm=11,
  FF2PreNorm=12, FF2Up=13, FF2Down=18, FF2PostNorm=23,
  AttentionPreNorm=24, Query=25, Key=30, Value=35, AttentionPost=40,
  RelativeKey=45, PerDimScale=46, AttentionPostNorm=47,
  ConvPreNorm=48, ConvStart=49, Depthwise=54, ConvNorm=55, ConvEnd=56, OutputNorm=61
};
inline constexpr int weight_id(int layer, LayerWeight role) {return kSharedTensors+layer*kLayerTensors+role;}

std::vector<component::TensorSpec> tensor_specs();
void validate_config(const nlohmann::json& config);
void validate_processor(const nlohmann::json& processor);
void validate_bundle_config(const std::string& directory);
}  // namespace gewell::embeddinggemma2::audio
