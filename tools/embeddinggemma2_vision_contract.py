"""Concrete BF16 EmbeddingGemma 2 image tower, bridge and processor contract."""
from tools.embeddinggemma2_contract import TensorSpec


def tensor_specs():
    specs = []

    def add(name, shape):
        specs.append(TensorSpec(len(specs), name, shape))

    add("vision_tower.patch_embedder.input_proj.weight", (768, 768))
    add("vision_tower.patch_embedder.position_embedding_table", (2, 10240, 768))
    add("embed_vision.embedding_projection.weight", (512, 768))
    for layer in range(16):
        prefix = f"vision_tower.encoder.layers.{layer}."
        for name, shape in (
            ("input_layernorm.weight", (768,)),
            ("self_attn.q_proj.linear.weight", (768, 768)),
            ("self_attn.k_proj.linear.weight", (768, 768)),
            ("self_attn.v_proj.linear.weight", (768, 768)),
            ("self_attn.q_norm.weight", (64,)),
            ("self_attn.k_norm.weight", (64,)),
            ("self_attn.o_proj.linear.weight", (768, 768)),
            ("post_attention_layernorm.weight", (768,)),
            ("pre_feedforward_layernorm.weight", (768,)),
            ("mlp.gate_proj.linear.weight", (3072, 768)),
            ("mlp.up_proj.linear.weight", (3072, 768)),
            ("mlp.down_proj.linear.weight", (768, 3072)),
            ("post_feedforward_layernorm.weight", (768,)),
        ):
            add(prefix + name, shape)
    return specs


def fields(actual, expected):
    for name, value in expected.items():
        if (not isinstance(actual, dict) or name not in actual or actual[name] != value
                or isinstance(actual[name], bool) != isinstance(value, bool)):
            raise ValueError(f"unsupported embeddinggemma2 vision: {name}")


def validate_config(config):
    fields(config, {"boi_token_id":255999, "eoi_token_id":258882, "image_token_id":258880, "video_token_id":258884})
    fields(config.get("vision_config"), {
        "model_type":"gemma4_vision", "hidden_size":768, "intermediate_size":3072,
        "num_hidden_layers":16, "num_attention_heads":12, "num_key_value_heads":12,
        "head_dim":64, "global_head_dim":64, "patch_size":16, "pooling_kernel_size":3,
        "position_embedding_size":10240, "default_output_length":280,
        "attention_bias":False, "attention_dropout":0.0, "rms_norm_eps":1e-6,
        "hidden_activation":"gelu_pytorch_tanh", "standardize":False, "use_clipped_linears":False,
        "rope_parameters":{"rope_type":"axial", "rope_theta":100.0},
    })


def validate_processor(processor):
    fields(processor, {"processor_class":"EmbeddingGemma2Processor", "image_seq_length":280})
    fields(processor.get("image_processor"), {
        "image_processor_type":"Gemma4ImageProcessor", "do_convert_rgb":True,
        "do_normalize":False, "do_rescale":True, "do_resize":True,
        "image_seq_length":280, "max_soft_tokens":280, "patch_size":16,
        "pooling_kernel_size":3, "resample":3, "rescale_factor":1/255,
    })
    fields(processor.get("video_processor"), {
        "video_processor_type":"EmbeddingGemma2VideoProcessor", "do_convert_rgb":True,
        "do_normalize":False, "do_rescale":True, "do_resize":True, "do_sample_frames":True,
        "fps":1, "max_frames":32, "max_soft_tokens":140, "overflow_strategy":"uniform",
        "add_timestamps":False, "patch_size":16, "pooling_kernel_size":3,
        "resample":3, "rescale_factor":1/255,
    })
