#include "gewell/models/gemma4/31b/component_weights.h"
#include "gewell/models/gemma4/31b/model.h"
#include <array>
#include <stdexcept>

namespace gewell::gemma4_31b {
namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
std::string component_tensor_name(std::size_t id) {
  static constexpr std::array<const char*, 11> assistant_layers{{
      "input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.q_norm.weight",
      "self_attn.o_proj.weight", "post_attention_layernorm.weight",
      "pre_feedforward_layernorm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
      "mlp.down_proj.weight", "post_feedforward_layernorm.weight", "layer_scalar"}};
  static constexpr std::array<const char*, 13> vision_layers{{
      "input_layernorm.weight", "self_attn.q_proj.linear.weight", "self_attn.k_proj.linear.weight",
      "self_attn.v_proj.linear.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
      "self_attn.o_proj.linear.weight", "post_attention_layernorm.weight",
      "pre_feedforward_layernorm.weight", "mlp.gate_proj.linear.weight",
      "mlp.up_proj.linear.weight", "mlp.down_proj.linear.weight", "post_feedforward_layernorm.weight"}};
  switch (id) {
    case kAssistantEmbeddingPhysicalId: return "model.embed_tokens.weight";
    case kAssistantFinalNormPhysicalId: return "model.norm.weight";
    case kAssistantPreProjectionPhysicalId: return "pre_projection.weight";
    case kAssistantPostProjectionPhysicalId: return "post_projection.weight";
    case kVisionPatchProjectionPhysicalId: return "model.vision_tower.patch_embedder.input_proj.weight";
    case kVisionPositionEmbeddingPhysicalId: return "model.vision_tower.patch_embedder.position_embedding_table";
    case kVisionStdBiasPhysicalId: return "model.vision_tower.std_bias";
    case kVisionStdScalePhysicalId: return "model.vision_tower.std_scale";
    case kVisionProjectionPhysicalId: return "model.embed_vision.embedding_projection.weight";
  }
  if (id >= kAssistantLayerWeightsFirstPhysicalId && id < kAssistantFinalNormPhysicalId) {
    const auto offset = id - kAssistantLayerWeightsFirstPhysicalId;
    return "model.layers." + std::to_string(offset / assistant_layers.size()) + "." +
        assistant_layers[offset % assistant_layers.size()];
  }
  require(id >= kVisionLayerWeightsFirstPhysicalId && id < kVisionStdBiasPhysicalId,
          "invalid component tensor id");
  const auto offset = id - kVisionLayerWeightsFirstPhysicalId;
  return "model.vision_tower.encoder.layers." + std::to_string(offset / vision_layers.size()) + "." +
      vision_layers[offset % vision_layers.size()];
}

}

std::vector<component::TensorSpec> component_specs(Component kind) {
  const auto first = kind == Component::assistant ? kAssistantEmbeddingPhysicalId : kVisionPatchProjectionPhysicalId;
  const auto count = kind == Component::assistant ? kAssistantPhysicalTensorCount : kVisionPhysicalTensorCount;
  std::vector<component::TensorSpec> result;
  for (std::size_t id = first; id < first + count; ++id) {
    const auto& shape = kPhysicalTensors[id].shape;
    result.push_back({id, component_tensor_name(id),
        {shape.dimensions.begin(), shape.dimensions.begin() + shape.rank}});
  }
  return result;
}
}  // namespace gewell::gemma4_31b
