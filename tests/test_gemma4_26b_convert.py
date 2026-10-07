import json
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from tools import bf16_artifact as wire
from tools import gemma4_26b_artifact as bf16
from tools import gemma4_26b_convert as convert
from tools import gemma4_26b_mixed_artifact as mixed
from tools import gemma4_26b_source as source_reader
from tools import nvfp4_artifact as native
from tools.gemma4_26b_contract import Role, TensorSpec, PREFIX
from tools.gemma4_26b_quantization import StorageType
from tools.safetensors_source import materialize_bf16


def test_bundled_profile_coverage_and_missing_scale_plans():
    from tools.gemma4_26b_contract import native_tensor_specs
    from tools.gemma4_26b_quantization import TARGET_ROLES
    profile = wire.load_json_object(convert.DEFAULT_CALIBRATION)
    specs = native_tensor_specs()
    names = {s.name for s in specs if s.role in TARGET_ROLES}
    observed = set(profile['input_amax'])
    missing = set(profile['uncovered'])
    assert observed | missing == names and not observed & missing
    assert all(type(profile['observations'][n]) is int and profile['observations'][n] > 0 for n in observed)
    assert profile['provenance']['source']['architecture'] == 'gemma4_26b_a4b'
    assert profile['provenance']['source']['weight_recipe'] == 'BF16'
    assert profile['provenance']['corpus']['split'] == 'calibration'
    assert profile['provenance']['corpus']['histories'] == len(profile['provenance']['records'])
    assert profile['provenance']['coverage'] == {
        'usable': len(observed), 'uncovered': len(missing), 'total': len(names)}
    for storage, denominator in ((StorageType.FP8_W8A8,448.), (StorageType.NVFP4_W4A4,2688.)):
        targets = [storage if s.name in names else StorageType.BF16 for s in specs]
        scales, evidence = convert.resolve_scales(specs,targets,{}, {})
        assert set(scales) == observed
        assert set(evidence['missing_input_scales']) == missing
        assert set(evidence['origins'].values()) == {'default'}
        assert all(scales[n] == convert.common.validate_input_scale(profile['input_amax'][n]/denominator,n)
                   for n in observed)
        # Explicitly retaining uncovered entries as BF16 makes this recipe
        # complete; the converter never silently makes that choice itself.
        targets = [storage if s.name in observed else StorageType.BF16 for s in specs]
        _, evidence = convert.resolve_scales(specs,targets,{}, {})
        assert evidence['missing_input_scales'] == []


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    specs = (TensorSpec(0,Role.EMBEDDING,-1,-1,(2,64),PREFIX+'embed_tokens.weight'),
             TensorSpec(1,Role.GATE_PROJ,0,-1,(2112,64),PREFIX+'layers.0.mlp.gate_proj.weight'),
             TensorSpec(2,Role.EXPERT_GATE_PROJ,0,0,(704,64),'stacked'),
             TensorSpec(3,Role.EXPERT_UP_PROJ,0,0,(704,64),'stacked'),
             TensorSpec(4,Role.EXPERT_DOWN_PROJ,0,0,(64,704),'stacked_down'))
    for module in (convert,mixed,bf16,source_reader):
        monkeypatch.setattr(module,'native_tensor_specs',lambda:specs)
    monkeypatch.setattr(convert,'DEFAULT_CALIBRATION',tmp_path/'missing-default.json')
    snapshot = tmp_path/'snapshot'
    snapshot.mkdir()
    (snapshot/'tokenizer.json').write_text('{"fixture":true}')
    payload, records = bytearray(), {}
    for spec in specs:
        raw = np.full(spec.shape,0x3f80,dtype='<u2').tobytes()
        records[spec.separate_source_name]={'dtype':'BF16','shape':list(spec.shape),'data_offsets':[len(payload),len(payload)+len(raw)]}
        payload.extend(raw)
    header=json.dumps(records).encode()
    (snapshot/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+payload)
    def args(**kwargs):
        result=dict(snapshot=snapshot,artifact=None,mask=None,input_scales=None,serving_snapshot=snapshot,
                    output=tmp_path/'output',plan=False,verify=None)
        result.update(kwargs)
        return SimpleNamespace(**result)
    def textfile(name,text):
        path=tmp_path/name; path.write_text(text); return path
    return specs,snapshot,args,textfile


def test_plan_missing_scales_does_not_write_or_use_31b_defaults(fixture):
    specs,snapshot,args,textfile=fixture
    mask=textfile('mask','0 experts.*.gate_proj nvfp4_w4a4')
    result=convert.convert_bundle(args(mask=mask,plan=True))
    assert not result['ready']
    assert result['calibration']['missing_input_scales']==[specs[2].name]
    assert not args().output.exists()
    with pytest.raises(ValueError,match='missing input scales'):
        convert.convert_bundle(args(mask=mask))
    assert not args().output.exists()


def test_scales_precedence_and_26b_profile(fixture,monkeypatch):
    specs,_,_,textfile=fixture
    profile=textfile('profile.json',json.dumps({'profile':'26b-fixture','provenance':{'model':'26b'},
                                             'input_amax':{s.name:2688. for s in specs[1:]}}))
    monkeypatch.setattr(convert,'DEFAULT_CALIBRATION',profile)
    targets=[StorageType.BF16,StorageType.NVFP4_W4A4,StorageType.FP8_W8A8,StorageType.NVFP4_W4A4,StorageType.BF16]
    scales,evidence=convert.resolve_scales(specs,targets,{specs[1].name:.5},{specs[1].name:.25,specs[2].name:.125})
    assert scales=={specs[1].name:.5,specs[2].name:.125,specs[3].name:1.}
    assert list(evidence['origins'].values())==['explicit','source','default']
    assert evidence['default_profile']['profile']=='26b-fixture'
    with pytest.raises(ValueError,match='unknown input scale'):
        convert.resolve_scales(specs,targets,{'layers.60.invalid':1.},{})


def test_snapshot_mixed_then_change_one_entry_preserves_others(fixture):
    specs,snapshot,args,textfile=fixture
    mask=textfile('mask','0 gate_proj nvfp4_w4a4\n0 experts.*.gate_proj nvfp4_w4a4\n0 experts.*.up_proj fp8_w8a8')
    scales=textfile('scales',json.dumps({s.name:.25 for s in specs[1:4]}))
    result=convert.convert_bundle(args(mask=mask,input_scales=scales))
    first=Path(result['artifact'])
    header,entries,_=convert.read_native(first,verify=True)
    assert [e.storage_type for e in entries]==[0,1,1,2,0]
    assert json.loads(Path(result['manifest']).read_text())['model']['conversion']['calibration']['origins'][specs[2].name]=='explicit'
    second=args().output.parent/'second'
    overrides=textfile('override',json.dumps({specs[2].name:.5}))
    changed=convert.convert_bundle(args(snapshot=None,artifact=first,output=second,input_scales=overrides))
    _,new,_=convert.read_native(Path(changed['artifact']),verify=True)
    with first.open('rb') as old, Path(changed['artifact']).open('rb') as out:
        for i,(a,b) in enumerate(zip(entries,new)):
            old.seek(a.file_offset); out.seek(b.file_offset)
            previous,current=old.read(a.byte_length),out.read(b.byte_length)
            if i==2:
                assert previous[:-4]==current[:-4]
                assert struct.unpack('<f',current[-4:])==(.5,)
            else:
                assert previous==current
                assert a.sha256==b.sha256
    # Widen only one narrow-output expert, comparing against source values with
    # its actual retained quantization (not against unrecoverable original BF16).
    widen=textfile('widen','0 experts.0.gate_proj bf16')
    scratch=second/'decode'; scratch.mkdir()
    decoded=materialize_bf16(convert.native_weight(first,specs[2],entries[2],scratch),scratch/'bf16')
    expected=decoded.path.read_bytes()
    with pytest.warns(UserWarning,match='does not recover'):
        third=convert.convert_bundle(args(snapshot=None,artifact=first,output=second.parent/'third',mask=widen))
    _,third_entries,_=convert.read_native(Path(third['artifact']),verify=True)
    entry=third_entries[2]
    with Path(third['artifact']).open('rb') as out:
        out.seek(entry.file_offset); assert out.read(entry.byte_length)==expected


def test_native_unchanged_copy_and_inplace_packaging(fixture):
    _,_,args,_=fixture
    first=convert.convert_bundle(args())
    path=Path(first['artifact'])
    copied=convert.convert_bundle(args(snapshot=None,artifact=path,output=path.parent.parent/'copy'))
    assert Path(copied['artifact']).read_bytes()==path.read_bytes()
    standalone=path.parent.parent/'standalone'; standalone.mkdir()
    native_path=standalone/'weights.gwt'; native_path.write_bytes(path.read_bytes())
    before=native_path.stat()
    packaged=convert.convert_bundle(args(snapshot=None,artifact=native_path,output=standalone))
    after=native_path.stat()
    assert (before.st_ino,before.st_mtime_ns)==(after.st_ino,after.st_mtime_ns)
    assert Path(packaged['manifest']).is_file()
    with pytest.raises(ValueError,match='existing output'):
        convert.convert_bundle(args(snapshot=None,artifact=native_path,output=standalone))


def test_plan_reads_no_matrix_payload(fixture,monkeypatch):
    _,_,args,_=fixture
    def forbidden(*a,**kw): raise AssertionError('payload conversion during planning')
    monkeypatch.setattr(convert,'materialize_bf16',forbidden)
    monkeypatch.setattr(mixed,'verify',forbidden)
    monkeypatch.setattr(mixed,'write_artifact',forbidden)
    assert convert.convert_bundle(args(plan=True))['ready']


def test_retained_nvfp4_snapshot_import_is_exact(fixture):
    specs,snapshot,args,textfile=fixture
    mask=textfile('mask','0 gate_proj nvfp4_w4a4\n0 experts.*.gate_proj nvfp4_w4a4\n0 experts.*.up_proj fp8_w8a8')
    scales=textfile('scales',json.dumps({s.name:.125 for s in specs[1:4]}))
    baseline=Path(convert.convert_bundle(args(mask=mask,input_scales=scales))['artifact'])
    _,entries,_=convert.read_native(baseline)
    packed_source=snapshot.parent/'packed-source'; packed_source.mkdir()
    (packed_source/'tokenizer.json').write_bytes((snapshot/'tokenizer.json').read_bytes())
    payload,records=bytearray(),{}
    def add(name,dtype,shape,raw):
        records[name]={'dtype':dtype,'shape':list(shape),'data_offsets':[len(payload),len(payload)+len(raw)]}
        payload.extend(raw)
    for spec,entry in zip(specs,entries):
        scratch=packed_source/str(spec.physical_id); scratch.mkdir()
        weight=convert.native_weight(baseline,spec,entry,scratch)
        with weight.tensor.path.open('rb') as stream:
            stream.seek(weight.tensor.offset); raw=stream.read(weight.tensor.byte_length)
        shape=spec.shape if weight.dtype!='U8' else (spec.shape[0],spec.shape[1]//2)
        add(spec.separate_source_name,weight.dtype,shape,raw)
        prefix=spec.separate_source_name.removesuffix('weight')
        if weight.block_scales:
            add(prefix+'weight_scale','F8_E4M3',(spec.shape[0],spec.shape[1]//16),weight.block_scales.path.read_bytes())
        if entry.storage_type:
            add(prefix+('weight_scale_2' if weight.dtype=='U8' else 'weight_scale'),'F32',(),struct.pack('<f',weight.weight_scale))
            add(prefix+'input_scale','F32',(),struct.pack('<f',weight.input_scale))
    header=json.dumps(records).encode()
    (packed_source/'model.safetensors').write_bytes(struct.pack('<Q',len(header))+header+payload)
    imported=convert.convert_bundle(args(snapshot=packed_source,output=packed_source/'out'))
    assert Path(imported['artifact']).read_bytes()==baseline.read_bytes()
    assert set(imported['calibration']['origins'].values())=={'source'}


def test_reject_changed_inplace_before_touching_weights(fixture):
    specs,_,args,textfile=fixture
    original=convert.convert_bundle(args())
    path=Path(original['artifact'])
    # Remove only the test manifest to expose the in-place precision check.
    Path(original['manifest']).unlink()
    before=path.read_bytes()
    mask=textfile('mask','0 experts.0.gate_proj nvfp4_w4a4')
    scales=textfile('scales',json.dumps({specs[2].name:1.}))
    with pytest.raises(ValueError,match='cannot change precision or scales in place'):
        convert.convert_bundle(args(snapshot=None,artifact=path,mask=mask,input_scales=scales))
    assert path.read_bytes()==before
