#include "gewell/models/embeddinggemma2/model.h"

#include <fstream>
#include <stdexcept>

namespace gewell::embeddinggemma2 {
std::vector<component::TensorSpec> tensor_specs() {
  std::vector<component::TensorSpec> result;
  auto add = [&](std::string name, std::vector<std::uint32_t> shape) {
    result.push_back({result.size(), "language_model." + name, std::move(shape)});
  };
  add("embed_tokens.weight", {kVocabulary, kHidden});
  add("ple.per_layer_model_projection.weight", {kLayers * kHidden, kHidden});
  add("ple.per_layer_projection_norm.weight", {kHidden});
  add("norm.weight", {kHidden});
  add("embedding_projection.weight", {kDimension, kHidden});
  for (int layer = 0; layer < kLayers; ++layer) {
    const auto prefix = "layers." + std::to_string(layer) + ".";
    const auto d = static_cast<std::uint32_t>(head_dim(layer));
    add(prefix + "input_layernorm.weight", {kHidden});
    add(prefix + "self_attn.q_proj.weight", {kHeads * d, kHidden});
    add(prefix + "self_attn.k_proj.weight", {kHidden, kHidden});
    add(prefix + "self_attn.v_proj.weight", {kHidden, kHidden});
    add(prefix + "self_attn.q_norm.weight", {d});
    add(prefix + "self_attn.k_norm.weight", {d});
    add(prefix + "self_attn.o_proj.weight", {kHidden, kHeads * d});
    add(prefix + "post_attention_layernorm.weight", {kHidden});
    add(prefix + "pre_feedforward_layernorm.weight", {kHidden});
    add(prefix + "mlp.gate_proj.weight", {kIntermediate, kHidden});
    add(prefix + "mlp.up_proj.weight", {kIntermediate, kHidden});
    add(prefix + "mlp.down_proj.weight", {kHidden, kIntermediate});
    add(prefix + "post_feedforward_layernorm.weight", {kHidden});
    add(prefix + "ple_block.per_layer_input_gate.weight", {kHidden, kHidden});
    add(prefix + "ple_block.per_layer_projection.weight", {kHidden, kHidden});
    add(prefix + "ple_block.post_per_layer_input_norm.weight", {kHidden});
    add(prefix + "layer_scalar", {1});
  }
  return result;
}

void validate_config(const nlohmann::json& config) {
  using Json = nlohmann::json;
  auto check = [](bool condition, const std::string& field) {
    if (!condition) throw std::runtime_error("embeddinggemma2: unsupported config " + field);
  };
  check(config.at("model_type") == "embedding_gemma2", "model_type");
  for (const auto& [name, id] : std::vector<std::pair<std::string,int>>{
           {"image_token_id",258880}, {"audio_token_id",258881}, {"video_token_id",258884}})
    check(config.contains(name) && config.at(name)==id, name);
  const auto& text = config.at("text_config");
  const auto expected = Json::parse(R"({
    "model_type":"embedding_gemma2_text", "vocab_size":262144,
    "hidden_size":512, "intermediate_size":2048, "num_hidden_layers":24,
    "num_attention_heads":4, "num_key_value_heads":2, "head_dim":256,
    "hidden_size_per_layer_input":512, "embedding_dim":768,
    "attention_bias":false, "hidden_activation":"gelu_pytorch_tanh",
    "rms_norm_eps":0.000001, "sliding_window":512,
    "bos_token_id":2, "eos_token_id":1, "pad_token_id":0,
    "per_layer_config":{"05":{"head_dim":512,"num_key_value_heads":1},
      "11":{"head_dim":512,"num_key_value_heads":1},
      "17":{"head_dim":512,"num_key_value_heads":1},
      "23":{"head_dim":512,"num_key_value_heads":1}},
    "rope_parameters":{"full_attention":{"rope_theta":1000000.0,"rope_type":"default"},
      "sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}}
  })");
  for (auto it = expected.begin(); it != expected.end(); ++it)
    check(text.contains(it.key()) && text.at(it.key()) == it.value(), it.key());
  Json types = Json::array();
  for (int i = 0; i < kLayers; ++i)
    types.push_back(global_layer(i) ? "full_attention" : "sliding_attention");
  check(text.at("layer_types") == types, "layer_types");
  check(text.at("max_position_embeddings").is_number_integer() &&
        text.at("max_position_embeddings").get<std::int64_t>() >= kMaxTokens,
        "max_position_embeddings");
}

void validate_bundle_config(const std::string& directory) {
  std::ifstream input(directory + "/config.json");
  if (!input) throw std::runtime_error("embeddinggemma2: cannot read config.json");
  validate_config(nlohmann::json::parse(input));
}
}  // namespace gewell::embeddinggemma2
