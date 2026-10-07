#!/usr/bin/env python3
"""Independent Torch controls for tiled 26B compact-cache BF16 attention."""
import argparse
import json
from pathlib import Path
import torch


def save(path, tensor):
    path.write_bytes(tensor.contiguous().cpu().view(torch.uint16).numpy().tobytes())


def load(path, shape):
    return torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.bfloat16).reshape(shape)


def quantize(x):
    # Tensor/scalar division in Torch multiplies by a rounded reciprocal.
    # Cache storage specifies correctly rounded FP32 division by 448.
    scale = (x.float().abs().amax(-1, keepdim=True).double() / 448).float()
    scale = torch.where(scale > 0, scale, 1)
    return ((x.float() / scale).to(torch.float8_e4m3fn).float() * scale).bfloat16()


def reference(q, k, v, norm, base, global_, fp8):
    rows, heads, d = q.shape
    kvheads = k.shape[1]
    k, v = k.clone(), v.clone()
    rotated = torch.cat((torch.arange(64), torch.arange(256, 320))).to(q.device)
    if fp8 and base:
        v[:base] = quantize(v[:base])
        if global_:
            k[:base, :, rotated] = quantize(k[:base, :, rotated])
        else:
            k[:base] = quantize(k[:base])
    if global_:
        reconstructed = (v.float() * norm.float()).bfloat16()
        reconstructed[:, :, rotated] = k[:, :, rotated]
        k = reconstructed
    # Independent batched matmul implementation, accumulating exact BF16
    # inputs in FP32. TF32 is disabled. No native storage/layout helpers.
    q = q.transpose(0, 1).float()
    k = k.repeat_interleave(heads // kvheads, 1).transpose(0, 1).float()
    v = v.repeat_interleave(heads // kvheads, 1).transpose(0, 1).float()
    out, eager = [], []
    for start in range(0, rows, 256):
        count = min(256, rows-start)
        absolute = base+start
        first = 0 if global_ else max(0, absolute-1023)
        end = absolute+count
        queries = q[:, start:start+count]
        scores = queries @ k[:, first:end].transpose(1, 2)
        qp = torch.arange(absolute, end, device=q.device)[:, None]
        kp = torch.arange(first, end, device=q.device)[None, :]
        visible = (kp <= qp) & (True if global_ else qp-kp < 1024)
        scores = scores.masked_fill(~visible, -torch.inf)
        eager_p = torch.softmax(scores.bfloat16().float(), -1).bfloat16().float()
        eager.append((eager_p @ v[:, first:end]).bfloat16())
        maximum = torch.full((heads, count, 1), -torch.inf, device=q.device)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros((heads, count, d), device=q.device)
        for begin in range(0, end-first, 1024):
            tile = scores[:, :, begin:begin+1024]
            next_max = torch.maximum(maximum, tile.amax(-1, keepdim=True))
            old = torch.where(torch.isneginf(maximum), 0., (maximum-next_max).exp())
            weights = (tile-next_max).exp().bfloat16().float()
            numerator = numerator*old + weights @ v[:, first+begin:min(end, first+begin+1024)]
            denominator = denominator*old + weights.sum(-1, keepdim=True)
            maximum = next_max
        out.append((numerator/denominator).bfloat16())
    return tuple(torch.cat(x, 1).transpose(0, 1).contiguous() for x in (out, eager))


def generate(output):
    output.mkdir(exist_ok=False)
    torch.manual_seed(26128)
    cases = []
    for global_ in (False, True):
        d, heads = (512, 2) if global_ else (256, 8)
        for base, rows in ((0, 1), (255, 3), (1023, 17), (2049, 257), (0, 4096), (4099, 1)):
            q = (torch.randn(rows, 16, d, device='cuda') / d**0.5).bfloat16()
            k = torch.randn(base+rows, heads, d, device='cuda').bfloat16()
            v = torch.randn_like(k)
            norm = (torch.randn(512, device='cuda')*.1+1).bfloat16()
            for fp8 in (False, True):
                name = f'{"global" if global_ else "local"}-{base}-{rows}-{int(fp8)}'
                directory = output/name
                directory.mkdir()
                for stem, tensor in [('q', q), ('k', k), ('v', v), ('norm', norm)]:
                    if fp8:
                        (directory/f'{stem}.bf16').symlink_to(f'../{name[:-1]}0/{stem}.bf16')
                    else:
                        save(directory/f'{stem}.bf16', tensor)
                expected, eager = reference(q, k, v, norm, base, global_, fp8)
                repeat, _ = reference(q, k, v, norm, base, global_, fp8)
                assert torch.equal(expected, repeat), name
                save(directory/'expected.bf16', expected)
                save(directory/'eager.bf16', eager)
                cases.append((name, int(global_), int(fp8), base, rows))
                print(name, 'reference repeat exact', flush=True)
    (output/'cases.tsv').write_text(''.join('\t'.join(map(str, row))+'\n' for row in cases))
    (output/'manifest.json').write_text(json.dumps({'torch': torch.__version__,
        'torch_git': torch.version.git_version, 'seed': 26128,
        'compute': 'BF16 inputs, FP32 QK/PV, BF16 unnormalized tile weights; same compact contract as 31B tensor prefill',
        'scope': 'Primitive controls only; full-model numerical policy remains separate',
        'cases': cases}, indent=2)+'\n')


def metrics(a, b):
    a, b = a.float(), b.float()
    return {'differing': int((a != b).sum()), 'max_abs': float((a-b).abs().max()),
            'relative_l2': float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b))}


def fused_decode_reference(q, k, v, norm, base, global_, fp8, batch_size):
    """Independent Torch arithmetic for one or two identical fused inputs.

    Reuses saved source tensors, never native scores or partials. Local splits
    retain 32-key tiles; global decode uses up to 256 splits for a lone long
    request or 128 for two. FP32 exponentials form the denominator; only P*V casts
    probabilities to BF16, matching the reused 31B MTP attention recipe.
    """
    assert q.shape[0] == 1
    _, heads, d = q.shape
    kvheads = k.shape[1]
    k, v = k.clone(), v.clone()
    rotated = torch.cat((torch.arange(64), torch.arange(256, 320))).to(q.device)
    if fp8 and base:
        v[:base] = quantize(v[:base])
        if global_:
            k[:base, :, rotated] = quantize(k[:base, :, rotated])
        else:
            k[:base] = quantize(k[:base])
    if global_:
        reconstructed = (v.float() * norm.float()).bfloat16()
        reconstructed[:, :, rotated] = k[:, :, rotated]
        k = reconstructed
    first = 0 if global_ else max(0, base-1023)
    visible = base-first+1
    split_limit = (256 if batch_size==1 and visible>=16384 else 128) if global_ else 32
    splits = min(split_limit, (visible+31)//32)
    q = q.reshape(kvheads, heads//kvheads, d).float()
    k = k[first:].transpose(0, 1).float()
    v = v[first:].transpose(0, 1).float()
    scores = q @ k.transpose(1, 2)
    maximum = torch.full((kvheads, splits, heads//kvheads, 1), -torch.inf, device=q.device)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros((*maximum.shape[:-1], d), device=q.device)
    indices = torch.arange(splits*32, device=q.device).reshape(splits, 32)
    for begin in range(0, visible, splits*32):
        positions = indices+begin
        valid = positions < visible
        selected = positions.clamp_max(visible-1)
        tile = scores[:, :, selected].transpose(1, 2).masked_fill(~valid[:, None, :], -torch.inf)
        next_max = torch.maximum(maximum, tile.amax(-1, keepdim=True))
        old = torch.where(torch.isneginf(maximum), 0., (maximum-next_max).exp())
        weights = torch.where(valid[:, None, :], (tile-next_max).exp(), 0.)
        numerator = numerator*old + weights.bfloat16().float() @ v[:, selected]
        denominator = denominator*old + weights.sum(-1, keepdim=True)
        maximum = next_max
    scale = (maximum-maximum.amax(1, keepdim=True)).exp()
    return ((numerator*scale).sum(1)/(denominator*scale).sum(1)).reshape(1, heads, d).bfloat16().cpu()


def compare(fixtures, native, report_path=None, fused_decode=False, batch_size=2):
    report = {}
    passed = True
    for line in (fixtures/'cases.tsv').read_text().splitlines():
        name, global_, fp8, base, rows = line.split()
        shape = (int(rows), 16, 512 if int(global_) else 256)
        expected = load(fixtures/name/'expected.bf16', shape)
        actual = load(native/f'{name}.bf16', shape)
        eager = load(fixtures/name/'eager.bf16', shape)
        recipe = 'tiled'
        if fused_decode and int(rows) == 1:
            global_, fp8, base = bool(int(global_)), bool(int(fp8)), int(base)
            kvshape = (base+1, 2 if global_ else 8, shape[-1])
            tensors = [load(fixtures/name/(stem+'.bf16'), dims).cuda() for stem, dims in
                       [('q', shape), ('k', kvshape), ('v', kvshape), ('norm', (512,))]]
            expected = fused_decode_reference(*tensors, base, global_, fp8, batch_size)
            repeat = fused_decode_reference(*tensors, base, global_, fp8, batch_size)
            assert torch.equal(expected, repeat), name
            save(native/(name+'.fused-control.bf16'), expected)
            recipe = 'fused_decode'
        # This is the primitive's same-algorithm control, not a tolerance for
        # differences from eager attention or full-model routing/logits.
        m = metrics(actual, expected)
        passed &= m['relative_l2'] <= 0.001 and bool(torch.isfinite(actual).all())
        report[name] = {f'native_vs_{recipe}_control': m,
                        f'{recipe}_control_vs_eager': metrics(expected, eager)}
        print(name, m)
    (report_path or native/'comparison.json').write_text(json.dumps({'passed': passed, 'cases': report}, indent=2)+'\n')
    if not passed:
        raise SystemExit('attention primitive comparison failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('generate', 'compare'))
    parser.add_argument('fixtures', type=Path)
    parser.add_argument('native', nargs='?', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--fused-decode', action='store_true',
                        help='compare one-row cases to the shared 31B recipe')
    parser.add_argument('--batch-size', type=int, choices=(1, 2), default=2,
                        help='number of identical inputs used by the fused decode fixture runner')
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if args.mode == 'generate':
        generate(args.fixtures)
    else:
        if args.native is None:
            parser.error('compare requires a native output directory')
        compare(args.fixtures, args.native, args.report, args.fused_decode, args.batch_size)


if __name__ == '__main__':
    main()
