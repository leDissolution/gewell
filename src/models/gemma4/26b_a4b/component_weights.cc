#include "gewell/models/gemma4/26b_a4b/component_weights.h"
#include "gewell/models/gemma4/26b_a4b/model.h"
#include <utility>

namespace gewell::gemma4_26b_a4b {
std::vector<component::TensorSpec> assistant_tensor_specs() {
  std::vector<component::TensorSpec> result;
  const auto add = [&](const std::string& name, std::vector<std::uint32_t> shape) {
    result.push_back({result.size(), name, std::move(shape)});
  };
  add("model.embed_tokens.weight", {kVocabSize, kAssistantHiddenSize});
  for (unsigned layer = 0; layer < kAssistantLayerCount; ++layer) {
    const auto prefix = "model.layers." + std::to_string(layer) + ".";
    const auto head = layer == 3 ? kGlobalHeadSize : kLocalHeadSize;
    const auto query = kQueryHeadCount * head;
    add(prefix + "input_layernorm.weight", {kAssistantHiddenSize});
    add(prefix + "self_attn.q_proj.weight", {query, kAssistantHiddenSize});
    add(prefix + "self_attn.q_norm.weight", {head});
    add(prefix + "self_attn.o_proj.weight", {kAssistantHiddenSize, query});
    add(prefix + "post_attention_layernorm.weight", {kAssistantHiddenSize});
    add(prefix + "pre_feedforward_layernorm.weight", {kAssistantHiddenSize});
    add(prefix + "mlp.gate_proj.weight", {kAssistantMlpSize, kAssistantHiddenSize});
    add(prefix + "mlp.up_proj.weight", {kAssistantMlpSize, kAssistantHiddenSize});
    add(prefix + "mlp.down_proj.weight", {kAssistantHiddenSize, kAssistantMlpSize});
    add(prefix + "post_feedforward_layernorm.weight", {kAssistantHiddenSize});
    add(prefix + "layer_scalar", {1});
  }
  add("model.norm.weight", {kAssistantHiddenSize});
  add("pre_projection.weight", {kAssistantHiddenSize, 2 * kHiddenSize});
  add("post_projection.weight", {kHiddenSize, kAssistantHiddenSize});
  return result;
}
std::vector<component::TensorSpec> vision_tensor_specs() {
  std::vector<component::TensorSpec> result;
  const auto add = [&](const std::string& name, std::vector<std::uint32_t> shape) {
    result.push_back({result.size(), name, std::move(shape)});
  };
  add("model.vision_tower.patch_embedder.input_proj.weight", {kVisionHiddenSize, 768});
  add("model.vision_tower.patch_embedder.position_embedding_table", {2, 10240, kVisionHiddenSize});
  for (unsigned layer = 0; layer < kVisionLayerCount; ++layer) {
    const auto prefix = "model.vision_tower.encoder.layers." + std::to_string(layer) + ".";
    add(prefix + "input_layernorm.weight", {kVisionHiddenSize});
    for (const auto* projection : {"q_proj", "k_proj", "v_proj"})
      add(prefix + "self_attn." + projection + ".linear.weight", {kVisionHiddenSize, kVisionHiddenSize});
    add(prefix + "self_attn.q_norm.weight", {72});
    add(prefix + "self_attn.k_norm.weight", {72});
    add(prefix + "self_attn.o_proj.linear.weight", {kVisionHiddenSize, kVisionHiddenSize});
    add(prefix + "post_attention_layernorm.weight", {kVisionHiddenSize});
    add(prefix + "pre_feedforward_layernorm.weight", {kVisionHiddenSize});
    add(prefix + "mlp.gate_proj.linear.weight", {kVisionMlpSize, kVisionHiddenSize});
    add(prefix + "mlp.up_proj.linear.weight", {kVisionMlpSize, kVisionHiddenSize});
    add(prefix + "mlp.down_proj.linear.weight", {kVisionHiddenSize, kVisionMlpSize});
    add(prefix + "post_feedforward_layernorm.weight", {kVisionHiddenSize});
  }
  add("model.vision_tower.std_bias", {kVisionHiddenSize});
  add("model.vision_tower.std_scale", {kVisionHiddenSize});
  add("model.embed_vision.embedding_projection.weight", {kHiddenSize, kVisionHiddenSize});
  return result;
}
}  // namespace gewell::gemma4_26b_a4b
