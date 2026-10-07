#!/usr/bin/env python3
"""Independent BF16 expert/dispatch fixtures for the 26B grouped primitive."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
from safetensors import safe_open
from safetensors.torch import load_file


def expert_reference(x, ids, weights, gate_up, down):
    output = torch.zeros_like(x)
    for expert in range(128):
        token, slot = (ids == expert).nonzero(as_tuple=True)
        if not len(token): continue
        gate, up = torch.nn.functional.linear(x[token],gate_up[expert]).chunk(2,-1)
        activated = torch.nn.functional.gelu(gate,approximate='tanh')*up
        expert_output = torch.nn.functional.linear(activated,down[expert])
        output[token] += (expert_output*weights[token,slot,None]).bfloat16()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--serial-reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', help='generate only named cases, preserving other fixture packs')
    parser.add_argument('--verify-existing', action='store_true', help='check repeatability without changing saved fixtures')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=args.verify_existing)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    index = json.loads((args.snapshot/'model.safetensors.index.json').read_text())['weight_map']
    prefix = 'model.language_model.layers.0.'
    def weight(name):
        key = prefix+name
        with safe_open(args.snapshot/index[key], framework='pt', device='cpu') as f:
            return f.get_tensor(key).cuda()
    gate_up = weight('experts.gate_up_proj')
    down = weight('experts.down_proj')
    scales = weight('router.per_expert_scale')
    ref = load_file(args.serial_reference)
    pool = torch.cat([ref[f'step.{i}.layer.0.pre_feedforward_layernorm_2.0'].reshape(1,2816)
                      for i in range(32)]).cuda()
    cases = [('one',1,'hot',False), ('tail',17,'distributed',False),
             ('uneven',129,'uneven',False), ('distributed',257,'distributed',False),
             ('all_hot',4096,'hot',False), ('max_padding',4096,'padding',False), ('uniform_ties',17,'ties',False),
             ('max_dispatch',4096,'distributed',True), ('empty',0,'hot',True)]
    if args.cases:
        if set(args.cases)-{case[0] for case in cases}: raise ValueError('unknown case')
        cases = [case for case in cases if case[0] in args.cases]
    def save(path, tensor):
        raw = tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes()
        if args.verify_existing:
            if path.read_bytes() != raw: raise ValueError(f'non-repeatable reference: {path}')
        else: path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()
    manifest = {'snapshot':str(args.snapshot), 'torch':torch.__version__,
                'torch_git':torch.version.git_version, 'tf32':False,
                'bf16_reduced_precision_reduction':False,
                'tie_policy':'stable lower expert ID', 'cases':{}}
    lines = []
    with torch.inference_mode():
        for name, rows, mode, dispatch_only in cases:
            path = args.output/name; path.mkdir(exist_ok=args.verify_existing)
            x = pool[torch.arange(rows,device='cuda')%32].contiguous()
            scores = torch.full((rows,128),-4.,device='cuda',dtype=torch.bfloat16)
            row_ids = torch.arange(rows,device='cuda')
            starts = ((row_ids*13)%128 if mode == 'distributed' else
                      torch.where(row_ids%11 == 0,120,0) if mode == 'uneven' else
                      torch.zeros_like(row_ids))
            ranks = torch.arange(8,device='cuda')
            scores.scatter_(1,(starts[:,None]+ranks)%128,
                            (5.-ranks*.125).bfloat16().expand(rows,8))
            if mode == 'padding':
                changed = torch.arange(120,device='cuda')
                replaced = changed%8
                scores[changed,replaced] = -4.
                scores[changed,changed+8] = (5.-replaced*.125).bfloat16()
            if mode == 'ties': scores.zero_()
            probabilities = scores.softmax(-1,dtype=torch.float32)
            ids = probabilities.argsort(dim=-1,descending=True,stable=True)[:,:8]
            weights = probabilities.gather(1,ids)
            weights = weights/weights.sum(-1,keepdim=True)*scales[ids]
            ids, order = ids.sort(-1)
            weights = weights.gather(1,order)
            output = torch.zeros_like(x)
            if not dispatch_only:
                output = expert_reference(x,ids,weights,gate_up,down)
                if not args.verify_existing and not torch.equal(output,expert_reference(x,ids,weights,gate_up,down)):
                    raise ValueError(f'non-repeatable reference: {name}')
            hashes = {file:save(path/file,tensor) for file,tensor in
                      [('input.bf16',x),('scores.bf16',scores),('ids.i32',ids.int()),
                       ('weights.f32',weights),('output.bf16',output)]}
            manifest['cases'][name] = {'rows':rows,'mode':mode,'dispatch_only':dispatch_only,'sha256':hashes}
            lines.append(f'{name} {rows} {int(dispatch_only)}\n')
            print(name,rows,flush=True)
    if not args.verify_existing:
        manifest['repeat_verified'] = True
        (args.output/'cases.tsv').write_text(''.join(lines))
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__ == '__main__': main()
