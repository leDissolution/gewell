import json
import math
import struct

import pytest

from tools.gemma4_26b_contract import Role, native_tensor_specs, source_shapes
from tools.gemma4_26b_quantization import StorageType
from tools.gemma4_26b_source import read_snapshot


def sparse_source(path, entries, scalars=None):
    """Real safetensors headers and sparse payloads avoid allocating model weights."""
    widths = {'BF16': 2, 'F16': 2, 'F32': 4, 'F8_E4M3': 1, 'U8': 1}
    header, offset = {}, 0
    for name, (dtype, shape) in entries.items():
        size = math.prod(shape) * widths[dtype]
        header[name] = {'dtype': dtype, 'shape': list(shape), 'data_offsets': [offset, offset+size]}
        offset += size
    raw = json.dumps(header).encode()
    raw += b' ' * (-len(raw) % 8)
    base = 8 + len(raw)
    with path.open('wb') as out:
        out.write(struct.pack('<Q', len(raw)) + raw)
        out.truncate(base + offset)
        for name, value in (scalars or {}).items():
            out.seek(base + header[name]['data_offsets'][0])
            out.write(struct.pack('<f', value))
    return base, header


@pytest.mark.parametrize('dtype,unit', [('BF16', 2), ('F16', 2), ('F32', 4)])
def test_first_last_expert_and_gate_up_slices(tmp_path, dtype, unit):
    specs = tuple(s for s in native_tensor_specs() if s.layer == 0 and s.expert in (0, 127))
    shapes = source_shapes()
    entries = {s.source_name: (dtype, shapes[s.source_name]) for s in specs}
    path = tmp_path/'weights.safetensors'
    base, header = sparse_source(path, entries)
    with path.open('r+b') as out:
        for i, spec in enumerate(specs):
            out.seek(base + header[spec.source_name]['data_offsets'][0] + unit*spec.source_element_offset)
            out.write((i+1).to_bytes(unit, 'little'))
    snapshot = read_snapshot(path, specs)
    assert len(snapshot.weights) == 6
    for i, (spec, weight) in enumerate(zip(specs, snapshot.weights)):
        assert weight.dtype == dtype and weight.shape == spec.shape
        assert weight.tensor.byte_length == unit*math.prod(spec.shape)
        with path.open('rb') as stream:
            stream.seek(weight.tensor.offset)
            assert int.from_bytes(stream.read(unit), 'little') == i+1
    snapshot.assert_unchanged()


def nvfp4_fixture(tmp_path, *, scale_shape=None, weight_scale=1.25, input_scale=0.5):
    spec = next(s for s in native_tensor_specs() if s.role == Role.EXPERT_DOWN_PROJ and s.expert == 127)
    name = spec.separate_source_name
    prefix = name.removesuffix('weight')
    entries = {name: ('U8', (2816, 352)), prefix+'weight_scale': ('F8_E4M3', scale_shape or (2816, 44)),
               prefix+'weight_scale_2': ('F32', ()), prefix+'input_scale': ('F32', (1,))}
    path = tmp_path/'weights.safetensors'
    sparse_source(path, entries, {prefix+'weight_scale_2': weight_scale, prefix+'input_scale': input_scale})
    return spec, path


def test_separate_nvfp4_retains_values_scales_and_source_identity(tmp_path):
    spec, path = nvfp4_fixture(tmp_path)
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {
        spec.separate_source_name.removesuffix('weight')+suffix: path.name
        for suffix in ('weight', 'weight_scale', 'weight_scale_2', 'input_scale')}}))
    snapshot = read_snapshot(tmp_path, (spec,))
    weight, = snapshot.weights
    assert weight.storage == StorageType.NVFP4_W4A4
    assert weight.shape == (2816, 704) and weight.tensor.byte_length == 2816*352
    assert weight.block_scales.byte_length == 2816*44
    assert (weight.weight_scale, weight.input_scale) == (1.25, 0.5)
    assert snapshot.index_sha256 != bytes(32)
    with path.open('ab') as out:
        out.write(b'x')
    with pytest.raises(ValueError, match='source changed'):
        snapshot.assert_unchanged()


@pytest.mark.parametrize('kwargs', [{'scale_shape': (2816, 45)}, {'weight_scale': 0},
                                   {'input_scale': float('nan')}])
def test_bad_quantized_scales_rejected(tmp_path, kwargs):
    spec, path = nvfp4_fixture(tmp_path, **kwargs)
    with pytest.raises(ValueError):
        read_snapshot(path, (spec,))


def test_stacked_quantized_scale_mapping_is_not_inferred(tmp_path):
    spec = next(s for s in native_tensor_specs() if s.role == Role.EXPERT_GATE_PROJ)
    path = tmp_path/'weights.safetensors'
    sparse_source(path, {spec.source_name: ('F8_E4M3', source_shapes()[spec.source_name])})
    with pytest.raises(ValueError, match='explicit scale mapping'):
        read_snapshot(path, (spec,))


def test_duplicate_expert_representations_rejected(tmp_path):
    spec = next(s for s in native_tensor_specs() if s.role == Role.EXPERT_UP_PROJ)
    path = tmp_path/'weights.safetensors'
    sparse_source(path, {spec.source_name: ('BF16', source_shapes()[spec.source_name]),
                         spec.separate_source_name: ('BF16', spec.shape)})
    with pytest.raises(ValueError, match='ambiguous'):
        read_snapshot(path, (spec,))


def test_quantized_router_rejected(tmp_path):
    spec = next(s for s in native_tensor_specs() if s.role == Role.ROUTER_PROJ)
    path = tmp_path/'weights.safetensors'
    sparse_source(path, {spec.source_name: ('F8_E4M3', spec.shape)})
    with pytest.raises(ValueError, match='not a supported projection'):
        read_snapshot(path, (spec,))
