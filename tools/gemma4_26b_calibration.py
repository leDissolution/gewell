"""Collect observed 26B projection input ranges; preserve each finished history.

Only calibration-split text records are consumed. The profile reports uncovered
projections explicitly and never supplies an invented range for an unseen expert.
"""
import argparse
from collections import defaultdict, deque
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import bf16_artifact as wire
from tools.gemma4_26b_contract import native_tensor_specs
from tools.gemma4_26b_quantization import TARGET_ROLES

TARGET_REVISION = '4d7ae4984b7db7de8f8457170b3f1a419ee76d52'
TRANSFORMERS_REVISION = '5a27d2d2706cbd1ee36d1a8e8441656bb88e7f04'


def publish(path, data):
    """Publish one completed record without replacing previous successful work."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            output.write(wire.canonical_json_bytes(data))
            output.flush()
            os.fsync(output.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def calibration_records(path):
    buckets = defaultdict(deque)
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['split'] != 'calibration' or row['workload'] == 'vision':
                continue
            # Reject image/video content rather than silently dropping it.
            if any(isinstance(m.get('content'), list) and
                   any(part.get('type') != 'text' for part in m['content']) for m in row['messages']):
                continue
            key = (row['workload'], row.get('languages', {}).get('input', ''),
                   row.get('provenance', {}).get('repo_id', ''))
            buckets[key].append(row)
    workloads = defaultdict(deque)
    for key in sorted(buckets):
        workloads[key[0]].append(key)
    while workloads:
        for workload in sorted(tuple(workloads)):
            key = workloads[workload].popleft()
            yield buckets[key].popleft()
            if buckets[key]:
                workloads[workload].append(key)
            if not workloads[workload]:
                del workloads[workload]


def aggregate(directory):
    """Maxima and actual row counts, without discarding earlier measurements."""
    names = {s.name for s in native_tensor_specs() if s.role in TARGET_ROLES}
    maxima, counts, records = {}, defaultdict(int), []
    for path in sorted((directory/'records').glob('*.json')):
        data = wire.load_json_object(path)
        wire._require(set(data['input_amax']) == set(data['observations']) and
                      not (data['input_amax'].keys()-names), 'invalid calibration projection inventory')
        for name, value in data['input_amax'].items():
            count = data['observations'][name]
            wire._require(type(value) in (int,float) and math.isfinite(value) and value >= 0 and
                          type(count) is int and count > 0, f'invalid measured range/count: {name}')
            maxima[name] = max(maxima.get(name,0.), value)
            counts[name] += count
        records.append({'file': path.name, 'sha256': wire.sha256_file(path), 'id': data['id'],
                        'tokens': len(data['input_ids']), 'workload': data['workload']})
    usable = {name:value for name,value in maxima.items() if value > 0}
    return {'profile': 'gemma4-26b-a4b-bf16-observed-v1',
            'provenance': {'method': 'Maximum absolute BF16 projection input on observed routed tokens; eager reference arithmetic.',
                           'source': wire.load_json_object(directory/'source.json'), 'records': records,
                           'scope': 'Text calibration only; no claim of quantization quality or image coverage.'},
            'input_amax': usable, 'observations': dict(counts),
            'uncovered': sorted(names-usable.keys())}


class Collector:
    """Observe the reference computation without changing its arithmetic."""
    def __init__(self):
        self.maxima, self.counts, self.handles, self.originals = {}, defaultdict(int), [], []

    def reset(self):
        self.maxima.clear()
        self.counts.clear()

    def observe(self, name, value):
        import torch
        maximum = value.detach().abs().amax().float()
        self.maxima[name] = torch.maximum(self.maxima[name], maximum) if name in self.maxima else maximum
        self.counts[name] += value.numel()//value.shape[-1]

    def values(self):
        import torch
        names = list(self.maxima)
        values = torch.stack([self.maxima[name] for name in names]).cpu().tolist()
        wire._require(all(math.isfinite(v) and v >= 0 for v in values), 'nonfinite calibration activation')
        return dict(zip(names,values)), dict(self.counts)

    def install(self, text):
        for layer_id, layer in enumerate(text.layers):
            prefix = f'layers.{layer_id}.'
            for family, projections in (('self_attn', ('q_proj','k_proj','v_proj','o_proj')),
                                        ('mlp', ('gate_proj','up_proj','down_proj'))):
                parent = getattr(layer,family)
                for projection in projections:
                    module = getattr(parent,projection,None)
                    if module is not None:
                        name = prefix+family+'.'+projection+'.weight'
                        self.handles.append(module.register_forward_pre_hook(
                            lambda module, inputs, name=name: self.observe(name,inputs[0])))
            experts = layer.experts
            self.originals.append((experts,experts.forward))
            experts.forward = types.MethodType(self.expert_forward(prefix),experts)
        return self

    def close(self):
        for handle in self.handles:
            handle.remove()
        for module, original in self.originals:
            module.forward = original
        self.handles.clear(); self.originals.clear()

    def expert_forward(self, prefix):
        observer = self
        def forward(module, hidden_states, top_k_index, top_k_weights):
            import torch
            # Preserve the pinned Gemma4TextExperts eager implementation's
            # operation order, casts and index_add; only observations are added.
            final_hidden_states = torch.zeros_like(hidden_states)
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=module.num_experts+1)
                expert_mask = expert_mask.permute(2,1,0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1,-2)),0).nonzero()
            for expert_idx in expert_hit:
                expert_idx = expert_idx[0]
                if expert_idx == module.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                expert = int(expert_idx)
                base = prefix+f'experts.{expert}.'
                observer.observe(base+'gate_proj.weight',current_state)
                observer.observe(base+'up_proj.weight',current_state)
                gate, up = torch.nn.functional.linear(current_state,module.gate_up_proj[expert_idx]).chunk(2,dim=-1)
                current_hidden_states = module.act_fn(gate)*up
                observer.observe(base+'down_proj.weight',current_hidden_states)
                current_hidden_states = torch.nn.functional.linear(current_hidden_states,module.down_proj[expert_idx])
                current_hidden_states = current_hidden_states*top_k_weights[token_idx,top_k_pos,None]
                final_hidden_states.index_add_(0,token_idx,current_hidden_states.to(final_hidden_states.dtype))
            return final_hidden_states
        return forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=32, help='total selected corpus histories; completed histories are reused')
    parser.add_argument('--max-tokens', type=int, default=512)
    args = parser.parse_args()
    if args.limit < 1 or not 1 <= args.max_tokens <= 4096:
        parser.error('limit must be positive; max-tokens must be 1..4096')
    os.environ['USE_HUB_KERNELS'] = 'NO'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    import transformers
    from tools.gemma4_26b_source import read_snapshot
    from tools.gemma4_26b_artifact import snapshot_identity
    wire._require(snapshot_identity(args.snapshot) == {'repository': 'google/gemma-4-26B-A4B-it', 'revision': TARGET_REVISION},
                  'bundled reference calibration requires the pinned Google26B BF16 snapshot')
    source = read_snapshot(args.snapshot)
    wire._require(all(w.dtype=='BF16' for w in source.weights), 'reference calibration requires BF16 source weights')
    implementation = Path(transformers.__file__).resolve().parents[2]
    revision = subprocess.check_output(['git','-C',str(implementation),'rev-parse','HEAD'],text=True).strip()
    wire._require(revision == TRANSFORMERS_REVISION and not subprocess.check_output(
        ['git','-C',str(implementation),'diff','HEAD','--','src'],text=True), 'use the pinned unmodified Transformers source')
    provenance = {'architecture': 'gemma4_26b_a4b', **snapshot_identity(args.snapshot),
                  'config_sha256':source.config_sha256.hex(),'index_sha256':source.index_sha256.hex(),
                  'transformers_revision':revision,'weight_recipe':'BF16','attention':'eager','experts':'eager'}
    args.output.mkdir(parents=True,exist_ok=True)
    source_path = args.output/'source.json'
    if source_path.exists():
        wire._require(wire.load_json_object(source_path)==provenance,'calibration directory belongs to a different source')
    else:
        publish(source_path,provenance)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.snapshot,local_files_only=True)
    pending = []
    for index,row in enumerate(calibration_records(args.corpus)):
        if index >= args.limit:
            break
        tokens = tokenizer.apply_chat_template(row['messages'],tokenize=True,add_generation_prompt=True,return_dict=False)
        tokens = tokens[:args.max_tokens]
        key = hashlib.sha256(wire.canonical_json_bytes({'id':row['id'],'input_ids':tokens})).hexdigest()
        path = args.output/'records'/f'{key}.json'
        if not path.exists():
            pending.append((row,tokens,path))
    if pending:
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        model = transformers.Gemma4ForConditionalGeneration.from_pretrained(args.snapshot,dtype=torch.bfloat16,
            attn_implementation='eager',experts_implementation='eager',local_files_only=True).to('cuda').eval()
        text = model.model.language_model
        collector = Collector().install(text)
        corpus_hash = wire.sha256_file(args.corpus)
        for row,tokens,path in pending:
            collector.reset()
            with torch.inference_mode():
                text(input_ids=torch.tensor([tokens],device='cuda'),use_cache=False)
            maxima,counts = collector.values()
            source.assert_unchanged()
            publish(path,{'id':row['id'],'workload':row['workload'],'input_ids':tokens,
                          'record_provenance':row['provenance'],'corpus_sha256':corpus_hash,
                          'torch':torch.__version__,'collector_sha256':wire.sha256_file(Path(__file__)),
                          'input_amax':maxima,'observations':counts})
            print(json.dumps({'completed':row['id'],'tokens':len(tokens),'observed_projections':len(maxima)}),flush=True)
        collector.close()
    profile = aggregate(args.output)
    # A new aggregate is an immutable result; the measured records stay reusable.
    digest = hashlib.sha256(wire.canonical_json_bytes(profile)).hexdigest()
    profile_path = args.output/f'profile-{digest[:16]}.json'
    if not profile_path.exists():
        publish(profile_path,profile)
    print(json.dumps({'profile':str(profile_path),'records':len(profile['provenance']['records']),
                      'covered':len(profile['input_amax']),'uncovered':len(profile['uncovered'])}),flush=True)


if __name__ == '__main__':
    main()
