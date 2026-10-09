"""Concrete BF16 EmbeddingGemma 2 audio tower, bridge and feature contract."""
from tools.embeddinggemma2_contract import TensorSpec


def tensor_specs():
    specs = []
    def add(name, shape):
        specs.append(TensorSpec(len(specs), name, shape))
    def clipped(name, shape):
        add(name+".linear.weight", shape)
        for bound in ("input_min","input_max","output_min","output_max"):
            add(name+"."+bound, ())
    root = "audio_tower."
    for layer, channels, inputs in ((0,128,1),(1,32,128)):
        prefix = root+f"subsample_conv_projection.layer{layer}."
        add(prefix+"conv.weight", (channels,inputs,3,3))
        add(prefix+"norm.weight", (channels,))
    add(root+"subsample_conv_projection.input_proj_linear.weight", (1024,1024))
    add(root+"output_proj.weight", (1536,1024))
    add(root+"output_proj.bias", (1536,))
    add("embed_audio.embedding_projection.weight", (512,1536))
    for layer in range(12):
        prefix = root+f"layers.{layer}."
        for ff in ("feed_forward1", "feed_forward2"):
            add(prefix+ff+".pre_layer_norm.weight", (1024,))
            clipped(prefix+ff+".ffw_layer_1", (4096,1024))
            clipped(prefix+ff+".ffw_layer_2", (1024,4096))
            add(prefix+ff+".post_layer_norm.weight", (1024,))
        add(prefix+"norm_pre_attn.weight", (1024,))
        for projection in ("q_proj", "k_proj", "v_proj", "post"):
            clipped(prefix+"self_attn."+projection, (1024,1024))
        add(prefix+"self_attn.relative_k_proj.weight", (1024,1024))
        add(prefix+"self_attn.per_dim_scale", (128,))
        add(prefix+"norm_post_attn.weight", (1024,))
        add(prefix+"lconv1d.pre_layer_norm.weight", (1024,))
        clipped(prefix+"lconv1d.linear_start", (2048,1024))
        add(prefix+"lconv1d.depthwise_conv1d.weight", (1024,1,5))
        add(prefix+"lconv1d.conv_norm.weight", (1024,))
        clipped(prefix+"lconv1d.linear_end", (1024,1024))
        add(prefix+"norm_out.weight", (1024,))
    return specs


def fields(actual, expected):
    for name, value in expected.items():
        if (not isinstance(actual, dict) or name not in actual or actual[name] != value
                or isinstance(actual[name], bool) != isinstance(value, bool)):
            raise ValueError(f"unsupported embeddinggemma2 audio: {name}")


def validate_config(config):
    fields(config, {"audio_token_id":258881, "boa_token_id":256000, "eoa_token_index":258883})
    fields(config.get("audio_config"), {
        "model_type":"gemma4_audio", "hidden_size":1024, "num_hidden_layers":12,
        "num_attention_heads":8, "output_proj_dims":1536,
        "subsampling_conv_channels":[128,32], "conv_kernel_size":5,
        "attention_chunk_size":12, "attention_context_left":13, "attention_context_right":0,
        "attention_logit_cap":50.0, "attention_invalid_logits_value":-1e9,
        "gradient_clipping":1e10, "residual_weight":0.5, "rms_norm_eps":1e-6,
        "hidden_act":"silu", "use_clipped_linears":True,
    })


def validate_processor(processor):
    fields(processor, {"processor_class":"EmbeddingGemma2Processor"})
    feature = processor.get("feature_extractor")
    fields(feature, {
        "feature_extractor_type":"Gemma4AudioFeatureExtractor", "sampling_rate":16000,
        "feature_size":128, "frame_length":320, "hop_length":160, "fft_length":512,
        "fft_overdrive":False, "min_frequency":0.0, "max_frequency":8000.0,
        "input_scale_factor":1.0, "dither":0.0, "preemphasis":0.0,
        "preemphasis_htk_flavor":True, "mel_floor":0.001,
        "per_bin_mean":None, "per_bin_stddev":None,
        "padding_side":"right", "padding_value":0.0, "return_attention_mask":True,
    })
    # These constructor options can override the derived sample-count fields.
    for name, value in (("frame_length_ms",20.0),("hop_length_ms",10.0)):
        if name in feature:
            fields(feature, {name:value})
