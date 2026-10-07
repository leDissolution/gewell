#!/usr/bin/env python3
"""Freeze real target KV and the pinned 26B assistant candidate-generator loop.

Both attention recipes consume independently computed BF16 states; separate
controls expose window and compact-K differences. Each case repeats before
publication; finished cases are
preserved. These are reference captures, not native MTP acceptance results.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import torch
import transformers
from safetensors.torch import save_file
from transformers import GenerationConfig, Gemma4AssistantConfig, Gemma4AssistantForCausalLM
from transformers.generation.candidate_generator import SinglePositionMultiTokenCandidateGenerator
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from generate_mtp_assistant_oracle import fp32_attention

REVISION = '5a27d2d2706cbd1ee36d1a8e8441656bb88e7f04'
TARGET_REVISION = '4d7ae4984b7db7de8f8457170b3f1a419ee76d52'
ASSISTANT_REVISION = '6e5aaaf4c42b98394530b8fda2e95cadd65c151c'
CASES = {'text': [2,105,2364,107,9259,236761],
         'boundary': [2]+[9259,236761]*128,
         'ring': [2]+[9259,236761]*515}


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def capture(target, assistant, tokens):
    captures = {}
    def save(name, tensor):
        captures[name] = tensor.detach().cpu().contiguous().clone()
    ids = torch.tensor([tokens], device='cuda')
    output = target(input_ids=ids, use_cache=True, output_hidden_states=True,
                    return_shared_kv_states=True)
    text = target.model.language_model
    assert text.layers[28].self_attn.store_full_length_kv
    assert text.layers[29].self_attn.store_full_length_kv
    pending = output.logits[:, -1:].argmax(-1)
    save('target.pending', pending)
    save('target.hidden', output.hidden_states[-1][:, -1:])
    save('target.logits', output.logits[:, -1:])
    for kind, pair in output.shared_kv_states.items():
        assert pair[0].shape[2] == len(tokens) and pair[1].shape == pair[0].shape
        save('target.'+kind+'.key', pair[0])
        save('target.'+kind+'.value', pair[1])
    global_key, global_value = output.shared_kv_states['full_attention']
    scale = text.layers[29].self_attn.k_norm.weight
    reconstructed = global_value * scale
    # Compact K reconstructs from already-rounded V; preserve and measure
    # the known difference from the source's separately normalized K.
    reconstructed[...,:64] = global_key[...,:64]
    reconstructed[...,256:320] = global_key[...,256:320]
    save('target.reconstructed_global_key', reconstructed)
    compact_output = copy.copy(output)
    compact_output.shared_kv_states = {
        'full_attention': (reconstructed, global_value),
        'sliding_attention': tuple(t[:,:,-1024:] for t in output.shared_kv_states['sliding_attention'])}
    save('target.global_k_norm', scale)
    save('target.global_compact', torch.cat((global_key[...,:64], global_key[...,256:320], global_value),-1))
    window_output = copy.copy(output)
    window_output.shared_kv_states = dict(output.shared_kv_states)
    window_output.shared_kv_states['sliding_attention'] = compact_output.shared_kv_states['sliding_attention']
    full_ids = torch.cat((ids,pending),-1)
    recipe_records = {}
    labels = ['eager','fp32','eager_compact','fp32_compact']
    if len(tokens)>1024:
        labels += ['eager_window','fp32_window']
    for label in labels:
        assistant.set_attn_implementation('eager' if label.startswith('eager') else 'gewell_mtp_fp32_reference')
        source_output = compact_output if label.endswith('_compact') else window_output if label.endswith('_window') else output
        assistant.generation_config.num_assistant_tokens = 3
        generator = SinglePositionMultiTokenCandidateGenerator(
            input_ids=full_ids, assistant_model=assistant,
            target_model_input_embeddings=text.embed_tokens,
            generation_config=GenerationConfig(max_length=len(tokens)+5), model_kwargs={})
        skipped, logits = generator.get_candidates(full_ids, {}, source_output, True, 0)
        assert torch.equal(skipped,full_ids) and logits is None
        state = {'step':0, 'positions':[], 'visibility':{}}
        def record(name, value):
            save(f'{label}.step.{state["step"]}.{name}', value)
        def inputs_hook(_module, _args, kwargs):
            position=kwargs['position_ids']
            assert position.tolist()==[[len(tokens)]]
            state['positions'].append(position.item())
            record('inputs',kwargs['inputs_embeds'])
            record('embedding',kwargs['inputs_embeds'][...,:2816])
        def output_hook(_module, _args, result):
            record('logits',result.logits)
            record('feedback',result.last_hidden_state)
            state['step']+=1
        def masks_hook(_module, _args, kwargs):
            for kind, mask in kwargs['attention_mask'].items():
                # The pinned source's bidirectional mask is inclusive and
                # exposes1025 rows. Native's explicit contract is1024; the
                # window/compact controls enforce that by cropping inputs.
                window=1024 if label.endswith(('_window','_compact')) else 1025
                expected=min(len(tokens),window) if kind=='sliding_attention' else len(tokens)
                available=kwargs['shared_kv_states'][kind][0].shape[2]
                if mask is None:
                    assert expected==available
                else:
                    valid=(mask[0,0,0]==0).nonzero().flatten()
                    assert valid.tolist()==list(range(available-expected,available)), (label,kind,valid.tolist())
                state['visibility'][kind]=[len(tokens)-expected,len(tokens)]
        hooks=[assistant.register_forward_pre_hook(inputs_hook,with_kwargs=True),
               assistant.register_forward_hook(output_hook),
               assistant.model.register_forward_pre_hook(masks_hook,with_kwargs=True)]
        for name,module in [('pre_projection',assistant.pre_projection),
                            *[(f'layer.{i}',layer) for i,layer in enumerate(assistant.model.layers)],
                            ('final_norm',assistant.model.norm)]:
            hooks.append(module.register_forward_hook(lambda _m,_a,result,name=name:record(name,result)))
        try:
            candidates,logits=generator.get_candidates(full_ids,{},source_output,False,0)
        finally:
            for hook in hooks:hook.remove()
        assert state['step']==3 and state['positions']==[len(tokens)]*3
        assert candidates.shape[1]==len(tokens)+4 and logits.shape==(1,3,262144)
        save(label+'.candidates',candidates)
        for step in range(3):
            assert torch.equal(captures[f'{label}.step.{step}.logits'],logits[:,step:step+1].cpu())
            token=candidates[:,len(tokens)+step:len(tokens)+step+1]
            assert torch.equal(captures[f'{label}.step.{step}.embedding'],text.embed_tokens(token).cpu())
            if step:
                assert torch.equal(captures[f'{label}.step.{step}.inputs'][...,2816:],
                                   captures[f'{label}.step.{step-1}.feedback'])
            else:
                assert torch.equal(captures[f'{label}.step.0.inputs'][...,2816:],captures['target.hidden'])
        recipe_records[label]={'draft_tokens':candidates[0,-3:].tolist(),'positions':state['positions'],
                               'local_kv_rows':source_output.shared_kv_states['sliding_attention'][0].shape[2],
                               'visibility':state['visibility']}
    assert torch.equal(captures['eager.step.0.inputs'],captures['fp32.step.0.inputs'])
    for name,value in captures.items():
        assert torch.isfinite(value).all(),name
    return captures,recipe_records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',type=Path,required=True)
    p.add_argument('--assistant',type=Path,required=True)
    p.add_argument('--assistant-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',nargs='+',choices=tuple(CASES),default=list(CASES))
    args=p.parse_args()
    source=Path(transformers.__file__).resolve().parents[2]
    assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==REVISION
    assert not subprocess.check_output(['git','-C',str(source),'diff','HEAD','--','src'])
    assert args.target.name==TARGET_REVISION and args.assistant.name==ASSISTANT_REVISION
    args.output.mkdir(parents=True,exist_ok=True)
    provenance={'target_revision':TARGET_REVISION,'assistant_revision':ASSISTANT_REVISION,
                'transformers_revision':REVISION,'assistant_config_sha256':digest(args.assistant_config),
                'assistant_weights_sha256':digest(args.assistant/'model.safetensors'),
                'target_config_sha256':digest(args.target/'config.json'),
                'target_index_sha256':digest(args.target/'model.safetensors.index.json')}
    pending=[]
    for name in args.cases:
        record=args.output/(name+'.json')
        if record.exists():
            existing=json.loads(record.read_text())
            assert existing['provenance']==provenance and existing['prompt_ids']==CASES[name]
            assert digest(args.output/(name+'.safetensors'))==existing['capture_sha256']
        else:
            assert not (args.output/(name+'.safetensors')).exists(), 'inspect unfinished case before retrying it'
            pending.append(name)
    if not pending:return
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    ALL_ATTENTION_FUNCTIONS.register('gewell_mtp_fp32_reference',fp32_attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register('gewell_mtp_fp32_reference',ALL_MASK_ATTENTION_FUNCTIONS['eager'])
    target=transformers.Gemma4ForConditionalGeneration.from_pretrained(
        args.target,dtype=torch.bfloat16,attn_implementation='eager',experts_implementation='eager',
        local_files_only=True).to('cuda').eval()
    config=Gemma4AssistantConfig.from_json_file(str(args.assistant_config))
    assistant=Gemma4AssistantForCausalLM.from_pretrained(args.assistant,config=config,
        generation_config=GenerationConfig.from_model_config(config),dtype=torch.bfloat16,
        attn_implementation='eager',local_files_only=True).to('cuda').eval()
    for name in pending:
        tensors,record=capture(target,assistant,CASES[name])
        repeated,repeated_record=capture(target,assistant,CASES[name])
        assert record==repeated_record and tensors.keys()==repeated.keys()
        assert all(torch.equal(value,repeated[key]) for key,value in tensors.items()),'reference is not repeatable'
        path=args.output/(name+'.safetensors')
        save_file(tensors,str(path))
        metadata={'provenance':provenance,'prompt_ids':CASES[name],'recipes':record,
                  'target_shared_layers':{'sliding_attention':28,'full_attention':29},
                  'local_visibility':[max(0,len(CASES[name])-1024),len(CASES[name])],
                  'global_visibility':[0,len(CASES[name])],'repeated_bit_exactly':True,
                  'torch':torch.__version__,'transformers':transformers.__version__,
                  'generator_sha256':digest(Path(__file__)),
                  'attention_helper_sha256':digest(Path(__file__).with_name('generate_mtp_assistant_oracle.py')),
                  'candidate_generator_sha256':digest(source/'src/transformers/generation/candidate_generator.py'),
                  'capture_sha256':digest(path),'tensors':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in tensors.items()}}
        with (args.output/(name+'.json')).open('x') as f:json.dump(metadata,f,indent=2);f.write('\n')
        print(json.dumps({'case':name,'tensors':len(tensors),'recipes':record}),flush=True)


if __name__=='__main__':main()
