#pragma once
#include "gewell/component_weights.h"
#include <cstddef>
#include <cstdint>

namespace gewell::gemma4_26b_a4b {
inline constexpr std::uint32_t kAssistantLayerCount = 4;
inline constexpr std::uint32_t kAssistantHiddenSize = 1024;
inline constexpr std::uint32_t kAssistantMlpSize = 8192;
// IDs belong to this separate component, never to the text artifact or 31B.
inline constexpr std::size_t kAssistantTensorCount = 48;
inline constexpr std::size_t kAssistantEmbeddingId = 0;
inline constexpr std::size_t kAssistantLayerFirstId = 1;
inline constexpr std::size_t kAssistantLayerTensorCount = 11;
inline constexpr std::size_t kAssistantFinalNormId = 45;
inline constexpr std::size_t kAssistantPreProjectionId = 46;
inline constexpr std::size_t kAssistantPostProjectionId = 47;
std::vector<component::TensorSpec> assistant_tensor_specs();
inline constexpr std::uint32_t kVisionLayerCount = 27;
inline constexpr std::uint32_t kVisionHiddenSize = 1152;
inline constexpr std::uint32_t kVisionMlpSize = 4304;
inline constexpr std::size_t kVisionTensorCount = 356;
inline constexpr std::size_t kVisionPatchProjectionId = 0;
inline constexpr std::size_t kVisionPositionEmbeddingId = 1;
inline constexpr std::size_t kVisionLayerFirstId = 2;
inline constexpr std::size_t kVisionLayerTensorCount = 13;
inline constexpr std::size_t kVisionStdBiasId = 353;
inline constexpr std::size_t kVisionStdScaleId = 354;
inline constexpr std::size_t kVisionProjectionId = 355;
std::vector<component::TensorSpec> vision_tensor_specs();
}  // namespace gewell::gemma4_26b_a4b
