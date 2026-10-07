import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import gemma4_26b_calibration as calibration


def test_corpus_selection_excludes_holdout_and_media(tmp_path):
    rows = []
    for split in ('calibration','selection','holdout'):
        for workload in ('instructions','vision','verifiable'):
            rows.append({'id':f'{split}-{workload}','split':split,'workload':workload,
                         'messages':[{'role':'user','content':'hello'}]})
    rows.append({'id':'image-in-text','split':'calibration','workload':'instructions',
                 'messages':[{'role':'user','content':[{'type':'image','path':'x'}]}]})
    path=tmp_path/'corpus.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    assert {r['id'] for r in calibration.calibration_records(path)}=={'calibration-instructions','calibration-verifiable'}


def test_completed_records_aggregate_without_inventing_coverage(tmp_path):
    name='layers.0.experts.0.down_proj.weight'
    calibration.publish(tmp_path/'source.json',{'model':'26b'})
    for i,(amax,count) in enumerate(((3.,2),(2.,5))):
        calibration.publish(tmp_path/'records'/f'{i}.json',{'id':str(i),'workload':'instructions',
             'input_ids':[2,3], 'input_amax':{name:amax}, 'observations':{name:count}})
    profile=calibration.aggregate(tmp_path)
    assert profile['input_amax']=={name:3.}
    assert profile['observations']=={name:7}
    assert name not in profile['uncovered']
    assert 'layers.0.experts.1.down_proj.weight' in profile['uncovered']
    before=(tmp_path/'records/0.json').read_bytes()
    with pytest.raises(FileExistsError): calibration.publish(tmp_path/'records/0.json',{})
    assert (tmp_path/'records/0.json').read_bytes()==before
    assert sorted(p.name for p in (tmp_path/'records').iterdir())==['0.json','1.json']


def test_invalid_or_zero_measurements_never_supply_default(tmp_path):
    name='layers.0.mlp.gate_proj.weight'
    calibration.publish(tmp_path/'source.json',{})
    path=tmp_path/'records/0.json'
    record={'id':'a','workload':'instructions','input_ids':[2], 'input_amax':{name:0.},'observations':{name:1}}
    calibration.publish(path,record)
    result=calibration.aggregate(tmp_path)
    assert name in result['uncovered'] and name not in result['input_amax']
    record['observations'][name]=0
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='range/count'): calibration.aggregate(tmp_path)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_expert_observer_preserves_pinned_reference_math(device):
    torch=pytest.importorskip('torch')
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    torch.manual_seed(12)
    config=SimpleNamespace(num_experts=4,hidden_size=64,moe_intermediate_size=16,
                           hidden_activation='gelu_pytorch_tanh',_experts_implementation='eager')
    experts=Gemma4TextExperts(config).to(device=device,dtype=torch.bfloat16)
    with torch.no_grad():
        experts.gate_up_proj.normal_(0,.125); experts.down_proj.normal_(0,.125)
    hidden=torch.randn(5,64,dtype=torch.bfloat16,device=device)
    indices=torch.tensor([[0,1],[0,1],[0,2],[0,2],[0,1]],device=device)
    weights=torch.rand(5,2,dtype=torch.float32,device=device)
    observer=calibration.Collector()
    with torch.inference_mode():
        expected=experts(hidden,indices,weights)
        actual=observer.expert_forward('layers.0.')(experts,hidden,indices,weights)
    assert torch.equal(actual,expected)
    maxima,counts=observer.values()
    assert counts['layers.0.experts.0.gate_proj.weight']==5
    assert counts['layers.0.experts.1.up_proj.weight']==3
    assert counts['layers.0.experts.2.down_proj.weight']==2
    assert not any('experts.3.' in name for name in counts)
    with torch.inference_mode():
        gate,up=torch.nn.functional.linear(hidden,experts.gate_up_proj[0]).chunk(2,dim=-1)
        activated=experts.act_fn(gate)*up
    assert maxima['layers.0.experts.0.down_proj.weight']==activated.abs().max().float().item()
    assert maxima['layers.0.experts.0.gate_proj.weight']==hidden.abs().max().float().item()
    observer.reset()
    assert not observer.counts and not observer.maxima
