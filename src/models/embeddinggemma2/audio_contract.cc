#include "gewell/models/embeddinggemma2/audio.h"

#include <fstream>
#include <stdexcept>

namespace gewell::embeddinggemma2::audio {
namespace {
void fields(const nlohmann::json& actual, const nlohmann::json& expected) {
  for (auto it=expected.begin(); it!=expected.end(); ++it)
    if (!actual.is_object() || !actual.contains(it.key()) || actual.at(it.key())!=it.value() ||
        actual.at(it.key()).is_boolean()!=it.value().is_boolean())
      throw std::runtime_error("embeddinggemma2 audio: unsupported "+it.key());
}
}

std::vector<component::TensorSpec> tensor_specs() {
  std::vector<component::TensorSpec> result;
  auto add=[&](std::string name, std::vector<std::uint32_t> shape) {
    result.push_back({result.size(),std::move(name),std::move(shape)});
  };
  auto clipped=[&](const std::string& name, std::vector<std::uint32_t> shape) {
    add(name+".linear.weight",std::move(shape));
    for (auto bound:{"input_min","input_max","output_min","output_max"}) add(name+"."+bound,{});
  };
  const std::string root="audio_tower.";
  for (int layer=0;layer<2;++layer) {
    const auto prefix=root+"subsample_conv_projection.layer"+std::to_string(layer)+".";
    const std::uint32_t channels=layer?32:128, inputs=layer?128:1;
    add(prefix+"conv.weight",{channels,inputs,3,3});
    add(prefix+"norm.weight",{channels});
  }
  add(root+"subsample_conv_projection.input_proj_linear.weight",{kHidden,kHidden});
  add(root+"output_proj.weight",{kOutputWidth,kHidden});
  add(root+"output_proj.bias",{kOutputWidth});
  add("embed_audio.embedding_projection.weight",{512,kOutputWidth});
  for (int layer=0;layer<kLayers;++layer) {
    const auto prefix=root+"layers."+std::to_string(layer)+".";
    for (auto ff:{"feed_forward1","feed_forward2"}) {
      add(prefix+ff+".pre_layer_norm.weight",{kHidden});
      clipped(prefix+ff+".ffw_layer_1",{kIntermediate,kHidden});
      clipped(prefix+ff+".ffw_layer_2",{kHidden,kIntermediate});
      add(prefix+ff+".post_layer_norm.weight",{kHidden});
    }
    add(prefix+"norm_pre_attn.weight",{kHidden});
    for (auto projection:{"q_proj","k_proj","v_proj","post"})
      clipped(prefix+"self_attn."+projection,{kHidden,kHidden});
    add(prefix+"self_attn.relative_k_proj.weight",{kHidden,kHidden});
    add(prefix+"self_attn.per_dim_scale",{kHeadDim});
    add(prefix+"norm_post_attn.weight",{kHidden});
    add(prefix+"lconv1d.pre_layer_norm.weight",{kHidden});
    clipped(prefix+"lconv1d.linear_start",{2*kHidden,kHidden});
    add(prefix+"lconv1d.depthwise_conv1d.weight",{kHidden,1,5});
    add(prefix+"lconv1d.conv_norm.weight",{kHidden});
    clipped(prefix+"lconv1d.linear_end",{kHidden,kHidden});
    add(prefix+"norm_out.weight",{kHidden});
  }
  return result;
}

void validate_config(const nlohmann::json& config) {
  fields(config,{{"audio_token_id",258881},{"boa_token_id",256000},{"eoa_token_index",258883}});
  fields(config.at("audio_config"),nlohmann::json::parse(R"({
    "model_type":"gemma4_audio", "hidden_size":1024, "num_hidden_layers":12,
    "num_attention_heads":8, "output_proj_dims":1536,
    "subsampling_conv_channels":[128,32], "conv_kernel_size":5,
    "attention_chunk_size":12, "attention_context_left":13, "attention_context_right":0,
    "attention_logit_cap":50.0, "attention_invalid_logits_value":-1000000000.0,
    "gradient_clipping":10000000000.0, "residual_weight":0.5, "rms_norm_eps":0.000001,
    "hidden_act":"silu", "use_clipped_linears":true
  })"));
}

void validate_processor(const nlohmann::json& processor) {
  fields(processor,{{"processor_class","EmbeddingGemma2Processor"}});
  const auto& feature=processor.at("feature_extractor");
  fields(feature,nlohmann::json::parse(R"({
    "feature_extractor_type":"Gemma4AudioFeatureExtractor", "sampling_rate":16000,
    "feature_size":128, "frame_length":320, "hop_length":160, "fft_length":512,
    "fft_overdrive":false, "min_frequency":0.0, "max_frequency":8000.0,
    "input_scale_factor":1.0, "dither":0.0, "preemphasis":0.0,
    "preemphasis_htk_flavor":true, "mel_floor":0.001,
    "per_bin_mean":null, "per_bin_stddev":null,
    "padding_side":"right", "padding_value":0.0, "return_attention_mask":true
  })"));
  if (feature.contains("frame_length_ms")) fields(feature,{{"frame_length_ms",20.0}});
  if (feature.contains("hop_length_ms")) fields(feature,{{"hop_length_ms",10.0}});
}

void validate_bundle_config(const std::string& directory) {
  std::ifstream config(directory+"/config.json"),processor(directory+"/processor_config.json");
  if (!config || !processor) throw std::runtime_error("embeddinggemma2 audio: missing bundle configuration");
  validate_config(nlohmann::json::parse(config));
  validate_processor(nlohmann::json::parse(processor));
}
}  // namespace gewell::embeddinggemma2::audio
