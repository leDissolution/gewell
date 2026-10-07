#!/usr/bin/env python3
"""Capture the pinned 26B BF16 reference, independently of native execution.

Run with PYTHONPATH pointing to the specification-pinned Transformers src.
Publishes only after two bit-identical eager executions per fixture.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

os.environ['USE_HUB_KERNELS'] = 'NO'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
import transformers
from safetensors.torch import save_file

from audit_gemma4_26b_source import sha256

REVISION = '5a27d2d2706cbd1ee36d1a8e8441656bb88e7f04'
TARGET_REVISION = '4d7ae4984b7db7de8f8457170b3f1a419ee76d52'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=('bos', 'pair', 'text', 'cached', 'phrase', 'cached_phrase', 'wide_phrase', 'cached_wide_phrase', 'boundary', 'serial', 'serial_cached', 'serial_boundary', 'serial_phrase'),
                        default=('bos', 'pair', 'text', 'cached'))
    parser.add_argument('--attention-control', action='store_true',
                        help='independent compact/tiled attention model control (cold bos/pair/text only)')
    parser.add_argument('--grouped-control', action='store_true',
                        help='independent grouped arithmetic control (up to256 rows/cached tokens)')
    args = parser.parse_args()
    if args.grouped_control:
        if args.attention_control or any(case not in ('bos', 'pair', 'text', 'cached', 'phrase', 'cached_phrase', 'wide_phrase', 'cached_wide_phrase')
                                         for case in args.cases):
            parser.error('--grouped-control supports cold/cached bos, pair, text, phrase and wide_phrase only')
        import gemma4_26b_grouped_reference
        gemma4_26b_grouped_reference.install()
    if args.attention_control:
        if any(case not in ('bos', 'pair', 'text') for case in args.cases):
            parser.error('--attention-control supports cold bos/pair/text cases only')
        from transformers.models.gemma4 import modeling_gemma4
        from generate_gemma4_26b_attention_oracle import reference as tiled_reference
        original_attention = modeling_gemma4.eager_attention_forward

        def controlled_attention(module, query, key, value, attention_mask, **kwargs):
            if not isinstance(module, modeling_gemma4.Gemma4TextAttention):
                return original_attention(module, query, key, value, attention_mask, **kwargs)
            if query.shape[0] != 1 or key.shape[2] != query.shape[2]:
                raise ValueError('compact/tiled model control requires a single cold text sequence')
            result, _ = tiled_reference(query[0].transpose(0, 1), key[0].transpose(0, 1),
                value[0].transpose(0, 1), module.k_norm.weight, 0,
                not module.is_sliding, False)
            return result.unsqueeze(0), None

        modeling_gemma4.eager_attention_forward = controlled_attention
    source = Path(transformers.__file__).resolve().parents[2]
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError(f'wrong Transformers revision: {revision}')
    if subprocess.check_output(['git', '-C', str(source), 'diff', 'HEAD', '--', 'src'], text=True):
        raise ValueError('modified Transformers source')
    if args.snapshot.name != TARGET_REVISION:
        raise ValueError('oracle requires the pinned target snapshot')
    partial = args.output.with_name(args.output.name + '.partial')
    if args.output.exists() or partial.exists():
        raise ValueError('oracle destination already exists; preserve existing captures')
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    model = transformers.Gemma4ForConditionalGeneration.from_pretrained(
        args.snapshot, dtype=torch.bfloat16,
        attn_implementation='eager', experts_implementation='eager', local_files_only=True,
    ).to('cuda').eval()
    text = model.model.language_model
    assert len(text.layers) == 30 and text.embed_tokens.weight.shape == (262144, 2816)
    assert model.lm_head.weight.data_ptr() == text.embed_tokens.weight.data_ptr()
    partial.mkdir(parents=True)
    captures = {}

    def hook(name):
        def capture(module, inputs, output):
            values = output if isinstance(output, tuple) else (output,)
            for i, value in enumerate(values):
                if isinstance(value, torch.Tensor):
                    captures[f'{name}.{i}'] = value.detach().cpu().contiguous().clone()
        return capture

    text.embed_tokens.register_forward_hook(hook('embedding'))
    text.norm.register_forward_hook(hook('final_norm'))
    for i, layer in enumerate(text.layers):
        layer.register_forward_hook(hook(f'layer.{i}.output'))
        for name in ('router', 'router.norm', 'router.proj',
                     'pre_feedforward_layernorm', 'pre_feedforward_layernorm_2',
                     'post_feedforward_layernorm_1', 'post_feedforward_layernorm_2',
                     'post_feedforward_layernorm', 'experts', 'mlp',
                     'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.q_norm',
                     'self_attn.k_norm', 'self_attn.o_proj'):
            layer.get_submodule(name).register_forward_hook(hook(f'layer.{i}.{name}'))

    manifest = {
        'target_revision': TARGET_REVISION, 'transformers_revision': revision,
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
        'attention': 'tiled-compact-control' if args.attention_control else 'eager', 'experts': 'eager', 'dtype': 'bfloat16', 'tf32': False,
        'bf16_reduced_precision_reduction': False, 'tied_head': True,
        'source_config_sha256': sha256(args.snapshot / 'config.json'),
        'cases': {},
    }
    if args.grouped_control:
        manifest.update(attention='compact-bf16-bmm-control', experts='torch-lane32-or-padded-gemm-control',
                        routing='stable lower-ID cutoff ties',
                        grouped_reference_sha256=sha256(Path(gemma4_26b_grouped_reference.__file__)),
                        scope='Independent production arithmetic control through256 rows/tokens; not eager arithmetic equivalence')
    cases = {'bos': [2], 'pair': [2, 105],
             'text': [2, 105, 2364, 107, 9259, 236761],
             'cached': [2, 105, 2364, 107, 9259, 236761],
             'serial': [2, 105, 2364, 107, 9259, 236761],
             'serial_cached': [2, 105, 2364, 107, 9259, 236761]}
    if any(name in args.cases for name in ('boundary', 'serial_boundary', 'serial_phrase', 'phrase', 'cached_phrase', 'wide_phrase', 'cached_wide_phrase')):
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
        pattern = tokenizer.encode('The quick brown fox jumps over the lazy dog. ', add_special_tokens=False)
        cases['boundary'] = [2] + (pattern * (1025 // len(pattern) + 1))[:1025]
        cases['serial_boundary'] = cases['boundary']
        cases['serial_phrase'] = cases['boundary'][:32]
        cases['phrase'] = cases['serial_phrase']
        cases['cached_phrase'] = cases['serial_phrase']
        cases['wide_phrase'] = cases['boundary'][:64]
        cases['cached_wide_phrase'] = cases['wide_phrase']
    for name in args.cases:
        tokens = cases[name]
        def run():
            captures.clear()
            if name.startswith('serial'):
                result = {}
                cache = None
                with torch.inference_mode():
                    for step in range(len(tokens) + (8 if name == 'serial_cached' else 0)):
                        token = tokens[step] if step < len(tokens) else output.logits[0, -1].argmax().item()
                        captures.clear()
                        output = model(input_ids=torch.tensor([[token]], device='cuda'),
                                       past_key_values=cache, use_cache=True)
                        cache = output.past_key_values
                        captures['logits'] = output.logits.cpu().contiguous().clone()
                        if name != 'serial_boundary' or step in (0, 254, 255, 256, 1022, 1023, 1024, 1025):
                            captures['input_ids'] = torch.tensor([[token]])
                            result.update({f'step.{step}.{key}': value for key, value in captures.items()})
                        if step and step % 128 == 0:
                            print(f'{name}: step {step}', flush=True)
                result['logits'] = output.logits.cpu().contiguous().clone()
                return result
            with torch.inference_mode():
                output = model(input_ids=torch.tensor([tokens], device='cuda'), use_cache=name in ('cached', 'cached_phrase', 'cached_wide_phrase'))
                captures['logits'] = output.logits.cpu().contiguous().clone()
                result = dict(captures)
                if name in ('cached', 'cached_phrase', 'cached_wide_phrase'):
                    # Preserve each committed step's independent target states.
                    for step in range(8):
                        token = output.logits[:, -1].argmax(-1, keepdim=True)
                        captures.clear()
                        output = model(input_ids=token, past_key_values=output.past_key_values, use_cache=True)
                        captures['input_ids'] = token.cpu()
                        captures['logits'] = output.logits.cpu().contiguous().clone()
                        result.update({f'decode.{step}.{key}': value for key, value in captures.items()})
            return result
        first, second = run(), run()
        if first.keys() != second.keys() or any(not torch.equal(first[k], second[k]) for k in first):
            raise ValueError(f'non-repeatable reference capture: {name}')
        path = partial / f'{name}.safetensors'
        save_file(first, path)
        manifest['cases'][name] = {
            'input_ids': tokens, 'sha256': sha256(path), 'captures': len(first),
            'argmax': first['logits'].argmax(-1).tolist(),
        }
        (partial / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(name, manifest['cases'][name], flush=True)
    # Explicit cutoff-tie fixture records this backend's top-k choice; it is
    # evidence, not a claim that torch.topk specifies stable cross-device ties.
    tied = torch.full((1, 128), 1 / 128, device='cuda', dtype=torch.float32)
    ties = [torch.topk(tied, 8).indices.cpu().tolist() for _ in range(2)]
    assert ties[0] == ties[1]
    manifest['uniform_topk_indices'] = ties[0]
    manifest['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    (partial / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    partial.rename(args.output)


if __name__ == '__main__':
    main()
