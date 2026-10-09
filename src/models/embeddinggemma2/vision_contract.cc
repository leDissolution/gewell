#include "gewell/models/embeddinggemma2/vision.h"

#include <fstream>
#include <stdexcept>

namespace gewell::embeddinggemma2::vision {
namespace {
void fields(const nlohmann::json& actual, const nlohmann::json& expected) {
  for (auto it = expected.begin(); it != expected.end(); ++it)
    if (!actual.contains(it.key()) || actual.at(it.key()) != it.value())
      throw std::runtime_error("embeddinggemma2 vision: unsupported " + it.key());
}
}

std::vector<component::TensorSpec> tensor_specs() {
  std::vector<component::TensorSpec> result;
  auto add = [&](std::string name, std::vector<std::uint32_t> shape) {
    result.push_back({result.size(), std::move(name), std::move(shape)});
  };
  add("vision_tower.patch_embedder.input_proj.weight", {kHidden, kPatchWidth});
  add("vision_tower.patch_embedder.position_embedding_table", {2, kPositionCount, kHidden});
  add("embed_vision.embedding_projection.weight", {512, kHidden});
  for (int layer = 0; layer < kLayers; ++layer) {
    const auto prefix = "vision_tower.encoder.layers." + std::to_string(layer) + ".";
    add(prefix + "input_layernorm.weight", {kHidden});
    add(prefix + "self_attn.q_proj.linear.weight", {kHidden, kHidden});
    add(prefix + "self_attn.k_proj.linear.weight", {kHidden, kHidden});
    add(prefix + "self_attn.v_proj.linear.weight", {kHidden, kHidden});
    add(prefix + "self_attn.q_norm.weight", {kHeadDim});
    add(prefix + "self_attn.k_norm.weight", {kHeadDim});
    add(prefix + "self_attn.o_proj.linear.weight", {kHidden, kHidden});
    add(prefix + "post_attention_layernorm.weight", {kHidden});
    add(prefix + "pre_feedforward_layernorm.weight", {kHidden});
    add(prefix + "mlp.gate_proj.linear.weight", {kIntermediate, kHidden});
    add(prefix + "mlp.up_proj.linear.weight", {kIntermediate, kHidden});
    add(prefix + "mlp.down_proj.linear.weight", {kHidden, kIntermediate});
    add(prefix + "post_feedforward_layernorm.weight", {kHidden});
  }
  return result;
}

void validate_config(const nlohmann::json& config) {
  fields(config, {{"boi_token_id",255999}, {"eoi_token_id",258882}, {"image_token_id",258880}, {"video_token_id",258884}});
  fields(config.at("vision_config"), nlohmann::json::parse(R"({
    "model_type":"gemma4_vision", "hidden_size":768, "intermediate_size":3072,
    "num_hidden_layers":16, "num_attention_heads":12, "num_key_value_heads":12,
    "head_dim":64, "global_head_dim":64, "patch_size":16, "pooling_kernel_size":3,
    "position_embedding_size":10240, "default_output_length":280,
    "attention_bias":false, "attention_dropout":0.0, "rms_norm_eps":0.000001,
    "hidden_activation":"gelu_pytorch_tanh", "standardize":false, "use_clipped_linears":false,
    "rope_parameters":{"rope_type":"axial","rope_theta":100.0}
  })"));
}

void validate_processor(const nlohmann::json& processor) {
  fields(processor, {{"processor_class","EmbeddingGemma2Processor"}, {"image_seq_length",280}});
  fields(processor.at("image_processor"), nlohmann::json::parse(R"({
    "image_processor_type":"Gemma4ImageProcessor", "do_convert_rgb":true,
    "do_normalize":false, "do_rescale":true, "do_resize":true,
    "image_seq_length":280, "max_soft_tokens":280, "patch_size":16,
    "pooling_kernel_size":3, "resample":3, "rescale_factor":0.00392156862745098
  })"));
  fields(processor.at("video_processor"), nlohmann::json::parse(R"({
    "video_processor_type":"EmbeddingGemma2VideoProcessor", "do_convert_rgb":true,
    "do_normalize":false, "do_rescale":true, "do_resize":true, "do_sample_frames":true,
    "fps":1, "max_frames":32, "max_soft_tokens":140, "overflow_strategy":"uniform",
    "add_timestamps":false, "patch_size":16, "pooling_kernel_size":3,
    "resample":3, "rescale_factor":0.00392156862745098
  })"));
}

void validate_bundle_config(const std::string& directory) {
  std::ifstream config(directory + "/config.json"), processor(directory + "/processor_config.json");
  if (!config || !processor) throw std::runtime_error("embeddinggemma2 vision: missing bundle configuration");
  validate_config(nlohmann::json::parse(config));
  validate_processor(nlohmann::json::parse(processor));
}
}  // namespace gewell::embeddinggemma2::vision
