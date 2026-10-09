"""The concrete BF16 text component and input geometry for EmbeddingGemma 2."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class TensorSpec:
    physical_id: int
    source_name: str
    shape: tuple[int, ...]

    @property
    def byte_length(self):
        return 2 * math.prod(self.shape)


def tensor_specs():
    specs = []

    def add(name, shape):
        specs.append(TensorSpec(len(specs), "language_model." + name, shape))

    add("embed_tokens.weight", (262144, 512))
    add("ple.per_layer_model_projection.weight", (12288, 512))
    add("ple.per_layer_projection_norm.weight", (512,))
    add("norm.weight", (512,))
    add("embedding_projection.weight", (768, 512))
    for layer in range(24):
        prefix = f"layers.{layer}."
        d = 512 if layer % 6 == 5 else 256
        for name, shape in (
            ("input_layernorm.weight", (512,)),
            ("self_attn.q_proj.weight", (4 * d, 512)),
            ("self_attn.k_proj.weight", (512, 512)),
            ("self_attn.v_proj.weight", (512, 512)),
            ("self_attn.q_norm.weight", (d,)),
            ("self_attn.k_norm.weight", (d,)),
            ("self_attn.o_proj.weight", (512, 4 * d)),
            ("post_attention_layernorm.weight", (512,)),
            ("pre_feedforward_layernorm.weight", (512,)),
            ("mlp.gate_proj.weight", (2048, 512)),
            ("mlp.up_proj.weight", (2048, 512)),
            ("mlp.down_proj.weight", (512, 2048)),
            ("post_feedforward_layernorm.weight", (512,)),
            ("ple_block.per_layer_input_gate.weight", (512, 512)),
            ("ple_block.per_layer_projection.weight", (512, 512)),
            ("ple_block.post_per_layer_input_norm.weight", (512,)),
            ("layer_scalar", (1,)),
        ):
            add(prefix + name, shape)
    return specs


def validate_config(config):
    expected = {
        "model_type": "embedding_gemma2_text", "vocab_size": 262144,
        "hidden_size": 512, "intermediate_size": 2048, "num_hidden_layers": 24,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 256,
        "hidden_size_per_layer_input": 512, "embedding_dim": 768,
        "attention_bias": False, "hidden_activation": "gelu_pytorch_tanh",
        "rms_norm_eps": 1e-6, "sliding_window": 512,
        "bos_token_id": 2, "eos_token_id": 1, "pad_token_id": 0,
        "layer_types": ["full_attention" if i % 6 == 5 else "sliding_attention" for i in range(24)],
        "per_layer_config": {f"{i:02}": {"head_dim": 512, "num_key_value_heads": 1} for i in (5, 11, 17, 23)},
        "rope_parameters": {
            "full_attention": {"rope_theta": 1000000.0, "rope_type": "default"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        },
    }
    if config.get("model_type") != "embedding_gemma2":
        raise ValueError("unsupported embeddinggemma2 model_type")
    for name, value in {"image_token_id": 258880, "audio_token_id": 258881, "video_token_id": 258884}.items():
        if type(config.get(name)) is not int or config[name] != value:
            raise ValueError(f"unsupported embeddinggemma2 config: {name}")
    text = config.get("text_config", {})
    for name, value in expected.items():
        if name not in text or text[name] != value or isinstance(text[name], bool) != isinstance(value, bool):
            raise ValueError(f"unsupported embeddinggemma2 config: {name}")
    if type(text.get("max_position_embeddings")) is not int or text["max_position_embeddings"] < 8192:
        raise ValueError("unsupported embeddinggemma2 max_position_embeddings")
