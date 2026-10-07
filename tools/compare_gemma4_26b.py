#!/usr/bin/env python3
"""Compare native BF16 boundary captures to the independent pinned oracle.

Reports errors without silently choosing or widening numerical tolerances.
--exact is the strict same-shape gate; other runs produce diagnostic evidence.
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native', type=Path)
    parser.add_argument('reference', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--serial', action='store_true')
    mode.add_argument('--cached', action='store_true', help='prefill call0 followed by eight cached decode calls')
    parser.add_argument('--exact', action='store_true')
    parser.add_argument('--steps', type=int, help='compare only this many leading serial steps')
    parser.add_argument('--allow-extra', action='store_true', help='report native-only diagnostics without treating them as verified')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference = load_file(args.reference)
    if args.steps is not None:
        if not args.serial or args.steps <= 0:
            raise ValueError('--steps requires serial comparison and a positive count')
        reference = {key: value for key, value in reference.items()
                     if key.startswith('step.') and int(key.split('.')[1]) < args.steps}
    # A missing token/layer must not turn a partial capture into a passing gate.
    boundaries = {'embedding.0', 'final_norm.0', 'logits'}
    layer_boundaries = ('experts.0', 'output.0', 'post_feedforward_layernorm_1.0',
                        'post_feedforward_layernorm_2.0', 'pre_feedforward_layernorm_2.0',
                        'router.proj.0', 'self_attn.k_norm.0', 'self_attn.k_proj.0',
                        'self_attn.o_proj.0', 'self_attn.q_norm.0', 'self_attn.q_proj.0')
    boundaries.update(f'layer.{layer}.{name}' for layer in range(30) for name in layer_boundaries)
    expected = {key for key in reference if
                (key.startswith('step.') and key.split('.', 2)[2] in boundaries
                 if args.serial else key in boundaries)}
    if args.cached:
        boundaries.update(f'layer.{layer}.router.2' for layer in range(30))
        expected = {prefix + name for prefix in ('', *(f'decode.{i}.' for i in range(8)))
                    for name in boundaries}
        required_reference = expected | {f'decode.{i}.input_ids' for i in range(8)}
        missing_reference = required_reference - reference.keys()
        if missing_reference:
            raise ValueError(f'incomplete cached reference: {sorted(missing_reference)[:8]}')
        decode_indices = {int(key.split('.')[1]) for key in reference if key.startswith('decode.')}
        if decode_indices != set(range(8)):
            raise ValueError('cached reference must contain exactly eight decode calls')
        calls = {path.name for path in args.native.iterdir() if path.is_dir() and path.name.isdigit()}
        if calls != {str(i) for i in range(9)}:
            raise ValueError('cached native captures must contain prefill plus eight decode calls')
    observed = set()
    rows = []
    unverified = []
    for directory in sorted(args.native.iterdir()):
        if not directory.is_dir() or not directory.name.isdigit():
            continue
        step = int(directory.name)
        if args.steps is not None and step >= args.steps:
            raise ValueError(f'unexpected native step beyond requested prefix: {step}')
        for path in sorted(directory.glob('*.bf16')):
            name = path.name.removesuffix('.bf16')
            key = f'step.{step}.{name}' if args.serial else name
            if args.cached and step:
                key = f'decode.{step-1}.{name}'
            if key not in reference:
                if not args.allow_extra:
                    raise ValueError(f'no independent capture for {key}')
                unverified.append(key)
                continue
            observed.add(key)
            x = reference[key].flatten().float()
            y = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.bfloat16).float()
            if x.shape != y.shape:
                raise ValueError(f'capture shape mismatch: {key}')
            if name.endswith('router.2'):
                # Expert accumulation is in ascending ID order. topk rank order
                # among equal scores is not a selected-set difference.
                x = x.reshape(-1, 8).sort(dim=-1).values.flatten()
                y = y.reshape(-1, 8).sort(dim=-1).values.flatten()
            rows.append({'position': step, 'name': name, 'exact': torch.equal(x, y),
                         'relative_l2': ((x-y).norm()/x.norm().clamp_min(1e-12)).item(),
                         'max_abs': (x-y).abs().max().item()})
    missing = expected - observed
    if missing:
        raise ValueError(f'missing required native captures ({len(missing)}): {sorted(missing)[:8]}')
    if not rows:
        raise ValueError('no native captures')
    summary = {'compared_captures': len(rows), 'unverified_native_diagnostics': len(unverified),
               'exact_captures': sum(r['exact'] for r in rows),
               'logits': [r for r in rows if r['name'] == 'logits']}
    args.output.write_text(json.dumps({'summary': summary, 'captures': rows, 'unverified': unverified}, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if args.exact and summary['exact_captures'] != len(rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
