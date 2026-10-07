#!/usr/bin/env python3
"""Pinned-source local/proportional RoPE controls through position262143."""
import argparse
import json
from pathlib import Path
import subprocess
import generate_gemma4_26b_oracle as oracle
import torch
import transformers
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding, apply_rotary_pos_emb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Path(transformers.__file__).resolve().parents[2]
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != oracle.REVISION or subprocess.check_output(['git', '-C', str(source), 'diff', 'HEAD', '--', 'src']):
        raise ValueError('requires unmodified pinned Transformers source')
    if args.snapshot.name != oracle.TARGET_REVISION:
        raise ValueError('requires pinned target snapshot')
    partial = args.output.with_name(args.output.name + '.partial')
    if args.output.exists() or partial.exists():
        raise ValueError('preserve existing reference output')
    config = transformers.AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    # Match the full oracle's CPU model initialization followed by CUDA transfer.
    rotary = Gemma4TextRotaryEmbedding(config).to('cuda')
    positions = torch.tensor([[262143, 0, 1024, 131072, 255, 257, 131071, 1023, 256, 1025]], device='cuda')
    torch.manual_seed(262143)
    torch.use_deterministic_algorithms(True)
    partial.mkdir(parents=True)
    (partial/'positions.u32').write_bytes(positions.int().cpu().numpy().astype('<u4').tobytes())
    hashes = {}
    for global_ in (False, True):
        width = 512 if global_ else 256
        for heads in (16, 2 if global_ else 8):
            name = f'{"global" if global_ else "local"}-{heads}'
            x = torch.randn(1, positions.numel(), heads, width, device='cuda').bfloat16()
            def run():
                cosine, sine = rotary(x, positions, 'full_attention' if global_ else 'sliding_attention')
                return apply_rotary_pos_emb(x, cosine, sine, unsqueeze_dim=2)
            expected = run()
            if not torch.equal(expected, run()):
                raise ValueError('non-repeatable RoPE reference')
            for suffix, tensor in (('input', x), ('expected', expected)):
                path = partial/f'{name}.{suffix}.bf16'
                path.write_bytes(tensor.contiguous().view(torch.uint16).cpu().numpy().astype('<u2').tobytes())
                hashes[path.name] = oracle.sha256(path)
    manifest = {'target_revision': oracle.TARGET_REVISION, 'transformers_revision': revision,
                'torch': torch.__version__, 'torch_git': torch.version.git_version,
                'positions': positions.cpu().tolist(), 'seed': 262143, 'repeat_exact': True,
                'config_sha256': oracle.sha256(args.snapshot/'config.json'),
                'sha256': hashes, 'scope': 'RoPE primitive only; not full-model horizon parity'}
    (partial/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    partial.rename(args.output)


if __name__ == '__main__':
    main()
