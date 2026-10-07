#!/usr/bin/env python3
"""Write an actual-weight, synthetic-prefix Transformers assistant fixture.

The prefix deliberately tests the assistant independently of target execution.
Run tests/mtp_assistant_oracle.cu against the generated directory. All outputs
are local files; this tool neither converts nor replaces the production asset.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import transformers
from safetensors import safe_open
from transformers import GenerationConfig, Gemma4AssistantConfig, Gemma4AssistantForCausalLM
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb


REVISION = "4735700dca7bd22fad5dc348c228b50ec6cbac6d"
DEFAULT_SOURCE = Path("/mnt/models/huggingface/hub/models--google--gemma-4-31B-it-assistant/snapshots") / REVISION
LAYER_FIELDS = (
    "input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.q_norm.weight",
    "self_attn.o_proj.weight", "post_attention_layernorm.weight",
    "pre_feedforward_layernorm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
    "mlp.down_proj.weight", "post_feedforward_layernorm.weight", "layer_scalar",
)


def fp32_attention(module, query, key, value, attention_mask=None, scaling=1.0, **_kwargs):
    """Independent dense Torch oracle for native FP32 attention boundaries."""
    key = key.repeat_interleave(module.num_key_value_groups, dim=1)
    value = value.repeat_interleave(module.num_key_value_groups, dim=1)
    scores = (query.float() @ key.float().transpose(-1, -2)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask.float()
    probabilities = scores.softmax(dim=-1)
    output = (probabilities @ value.float()).to(query.dtype)
    return output.transpose(1, 2).contiguous(), probabilities


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--config", type=Path,
                        help="explicit source config when stored separately from the weights")
    parser.add_argument("--prefix-length", type=int, default=1031)
    parser.add_argument("--attention-reference", choices=("fp32", "eager"), default="fp32",
                        help="FP32 matches native boundaries; eager measures alternate BF16 score/probability drift")
    args = parser.parse_args()
    if not 1 <= args.prefix_length < 262144:
        parser.error("prefix length must be in 1..262143")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(9173)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    attention = "eager"
    if args.attention_reference == "fp32":
        attention = "gewell_mtp_fp32_reference"
        ALL_ATTENTION_FUNCTIONS.register(attention, fp32_attention)
        ALL_MASK_ATTENTION_FUNCTIONS.register(attention, ALL_MASK_ATTENTION_FUNCTIONS["eager"])
    config_path = args.config or args.source / "config.json"
    source_config = Gemma4AssistantConfig.from_json_file(str(config_path))
    model = Gemma4AssistantForCausalLM.from_pretrained(
        args.source, dtype=torch.bfloat16, attn_implementation=attention,
        local_files_only=True, config=source_config,
        # This fixture calls forward directly. A weights-only directory need
        # not supply an otherwise unused generation_config.json.
        generation_config=GenerationConfig.from_model_config(source_config) if args.config else None,
    ).to("cuda").eval()
    config = model.model.config
    target_width = model.backbone_hidden_size
    local_attention = model.model.layers[0].self_attn
    global_attention = model.model.layers[3].self_attn
    query_heads = local_attention.q_proj.out_features // local_attention.head_dim
    local_heads = local_attention.q_proj.out_features // local_attention.head_dim // local_attention.num_key_value_groups
    global_heads = global_attention.q_proj.out_features // global_attention.head_dim // global_attention.num_key_value_groups
    embedding_scale = torch.tensor(target_width ** 0.5, device="cuda", dtype=torch.bfloat16)
    files = {}

    def save(name, tensor):
        path = args.output / name
        data = tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        path.write_bytes(data.view(torch.uint16).numpy().tobytes())
        files[name] = {"shape": list(data.shape), "sha256": digest(path)}

    names = ["model.embed_tokens.weight"]
    names += [f"model.layers.{layer}.{field}" for layer in range(4) for field in LAYER_FIELDS]
    names += ["model.norm.weight", "pre_projection.weight", "post_projection.weight"]
    with safe_open(args.source / "model.safetensors", framework="pt", device="cpu") as source:
        for index, name in enumerate(names):
            save(f"weight-{index:02}.bf16", source.get_tensor(name))

    for kind, layer_index, width in (("local", 0, 256), ("global", 3, 512)):
        raw = (((torch.arange(query_heads * width, device="cuda") * 13 + 3) % 97).float() / 16 - 3)
        raw = raw.to(torch.bfloat16).view(1, 1, query_heads, width)
        normalized = model.model.layers[layer_index].self_attn.q_norm(raw)
        save(f"rope-{kind}-raw.bf16", raw)
        save(f"rope-{kind}-norm.bf16", normalized)
        for p in (1, 31, 255, 1024, 1031):
            cos, sin = model.model.rotary_emb(raw, torch.tensor([[p]], device="cuda"),
                                             model.model.config.layer_types[layer_index])
            save(f"rope-{kind}-{p}-cos.bf16", cos)
            save(f"rope-{kind}-{p}-sin.bf16", sin)
            save(f"rope-{kind}-{p}-query.bf16", apply_rotary_pos_emb(normalized, cos, sin, unsqueeze_dim=2))

    length = args.prefix_length
    local_length = min(1024, length)
    # Synthetic BF16 states preserve compact K's reconstruction invariant.
    embedding = (torch.randn(target_width) / 64).to(device="cuda", dtype=torch.bfloat16)
    hidden = torch.randn(1, 1, target_width).to(device="cuda", dtype=torch.bfloat16)
    local_k = (torch.randn(1, local_heads, local_length, 256) / 4).to(device="cuda", dtype=torch.bfloat16)
    local_v = (torch.randn_like(local_k.float()) / 4).to(torch.bfloat16)
    scale = (torch.rand(512) + 0.5).to(device="cuda", dtype=torch.bfloat16)
    global_v = (torch.randn(1, global_heads, length, 512) / 4).to(device="cuda", dtype=torch.bfloat16)
    global_k = global_v * scale
    global_k[..., :64] = (torch.randn_like(global_k[..., :64].float()) / 4).to(torch.bfloat16)
    global_k[..., 256:320] = (torch.randn_like(global_k[..., 256:320].float()) / 4).to(torch.bfloat16)
    compact = torch.cat((global_k[..., :64], global_k[..., 256:320], global_v), dim=-1)
    local_ring_k = torch.full((local_heads, 1024, 256), 13, device="cuda", dtype=torch.bfloat16)
    local_ring_v = local_ring_k.clone()
    slots = torch.arange(length - local_length, length, device="cuda") % 1024
    local_ring_k[:, slots] = local_k[0]
    local_ring_v[:, slots] = local_v[0]
    save("target-embedding-row.bf16", embedding)
    save("target-hidden.bf16", hidden)
    save("local-key.bf16", local_ring_k)
    save("local-value.bf16", local_ring_v)
    save("global-compact.bf16", compact)
    save("target-global-k-norm.bf16", scale)
    shared = {"sliding_attention": (local_k, local_v), "full_attention": (global_k, global_v)}
    scaled_embedding = (embedding * embedding_scale).view(1, 1, target_width)
    position = torch.tensor([[length]], device="cuda")
    for step in range(2):
        hooks = []
        for name, module in [("pre-projection", model.pre_projection),
                             *[(f"layer-{i}", layer) for i, layer in enumerate(model.model.layers)],
                             ("final-norm", model.model.norm)]:
            hooks.append(module.register_forward_hook(
                lambda _module, _inputs, output, name=name: save(f"step-{step}-{name}.bf16", output)))
        for i, layer in enumerate(model.model.layers):
            def capture_query(_module, _inputs, output, i=i):
                save(f"step-{step}-layer-{i}-query-norm.bf16", output)
                kind = model.model.config.layer_types[i]
                cos, sin = model.model.rotary_emb(output, position, kind)
                save(f"step-{step}-layer-{i}-query-rope.bf16", apply_rotary_pos_emb(output, cos, sin, unsqueeze_dim=2))
            hooks.append(layer.self_attn.q_norm.register_forward_hook(capture_query))
            hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(
                lambda _module, inputs, i=i: save(f"step-{step}-layer-{i}-attention.bf16", inputs[0])))
        inputs = torch.cat((scaled_embedding, hidden), dim=-1)
        save(f"step-{step}-inputs.bf16", inputs)
        result = model(inputs_embeds=inputs, position_ids=position, shared_kv_states=shared)
        for hook in hooks:
            hook.remove()
        save(f"step-{step}-logits.bf16", result.logits)
        save(f"step-{step}-feedback.bf16", result.last_hidden_state)
        hidden = result.last_hidden_state
    # Length is also stored in a trivial native-readable text file.
    (args.output / "prefix-length.txt").write_text(f"{length}\n")
    source_modules = [
        transformers.models.gemma4_assistant.modeling_gemma4_assistant,
        transformers.models.gemma4.modeling_gemma4,
    ]
    metadata = {
        "schema": 1, "source": str(args.source), "revision": args.source.name,
        "source_weights_sha256": digest(args.source / "model.safetensors"),
        "source_config": str(config_path), "source_config_sha256": digest(config_path),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "reference_sources": {str(Path(module.__file__)): digest(Path(module.__file__)) for module in source_modules},
        "prefix_length": length, "position_convention": "fixed_L",
        "target_width": target_width, "assistant_width": config.hidden_size,
        "query_heads": query_heads,
        "local_kv_heads": local_heads, "global_kv_heads": global_heads,
        "embedding_scale": embedding_scale.item(),
        "input_token": 0, "steps": 2, "seed": 9173,
        "attention_implementation": attention,
        "generator_sha256": digest(Path(__file__)),
        "allow_bf16_reduced_precision_reduction": False, "allow_tf32": False,
        "prefix": "synthetic BF16 shared KV, compact-reconstructible global K",
        "files": files,
    }
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"fixture": str(args.output), "prefix_length": length, "files": len(files)}))


if __name__ == "__main__":
    main()
