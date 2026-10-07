"""Cross-language metadata and padded payload checks at real model geometry."""
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import struct
import subprocess

import pytest

from tools import bf16_artifact as wire
from tools import gemma4_26b_mixed_artifact as artifact
from tools.gemma4_26b_quantization import parse_mask, tensor_bytes


@pytest.fixture
def native_file(tmp_path):
    binary = Path(os.environ.get('GEWELL_26B_ARTIFACT_PROBE',
                                'build/cleanup-host/gewell_gemma4_26b_artifact_probe'))
    if not binary.is_file():
        pytest.skip('build the native 26B artifact probe')
    specs = artifact.native_tensor_specs()
    storages = parse_mask('0 gate_proj nvfp4_w4a4\n0 up_proj fp8_w8a8\n'
                          '0 experts.0.gate_proj nvfp4_w4a4\n'
                          '0 experts.0.up_proj fp8_w8a8\n0 experts.0.down_proj nvfp4_w4a4')
    sizes = artifact.plan(storages)
    table, offsets = bytearray(), []
    cursor = sizes['data_offset']
    for spec, storage in zip(specs, storages):
        length = tensor_bytes(spec, storage)
        offsets.append(cursor)
        table.extend(wire.ENTRY_STRUCT.pack(spec.physical_id, spec.layer, int(spec.role), len(spec.shape),
            int(storage), spec.shape[0], spec.shape[1] if len(spec.shape) == 2 else 0, 0,
            cursor, length, bytes(32)))
        cursor += wire.align_up(length)
    path = tmp_path/'mixed.gwt'

    def publish():
        with path.open('wb') as stream:
            stream.write(artifact.encode_header(sizes, bytes(32), bytes(32), hashlib.sha256(table).digest(), bytes(32)))
            stream.write(table)
            stream.truncate(sizes['file_bytes'])
            for spec, storage, offset in zip(specs, storages, offsets):
                if storage:
                    stream.seek(offset + tensor_bytes(spec, storage) - 8)
                    stream.write(struct.pack('<2f', .25, .125))

    def probe(verify=False):
        return subprocess.run([str(binary), str(path), *(['--verify'] if verify else [])],
                              text=True, capture_output=True)

    publish()
    return path, specs, storages, offsets, table, publish, probe


def test_mixed_metadata_matches_python(native_file):
    path, specs, kinds, offsets, _, _, probe = native_file
    _, entries = artifact.read_metadata(path)
    assert [e.file_offset for e in entries] == offsets
    assert [e.storage_type for e in entries] == list(kinds)
    result = probe()
    assert result.returncode == 0, result.stderr
    assert '12117 tensors' in result.stdout


@pytest.mark.parametrize('kind', ['unknown_storage', 'router_quantized', 'unpadded_size', 'wrong_expert_shape'])
def test_rehashed_invalid_metadata_rejected(native_file, kind):
    _, specs, _, _, table, publish, probe = native_file
    target = next(s for s in specs if s.name == 'layers.0.mlp.gate_proj.weight')
    if kind == 'router_quantized':
        target = next(s for s in specs if s.role.name == 'ROUTER_PROJ')
    elif kind == 'wrong_expert_shape':
        target = next(s for s in specs if s.expert == 127 and s.layer == 29)
    start = target.physical_id * wire.ENTRY_BYTES
    if kind in ('unknown_storage', 'router_quantized'):
        table[start+7] = 3 if kind == 'unknown_storage' else 1
    elif kind == 'unpadded_size':
        struct.pack_into('<Q', table, start+28, target.shape[0]*target.shape[1]//2 + 8)
    else:
        table[start+8] ^= 1
    publish()
    result = probe()
    assert result.returncode != 0
    assert '26B' in result.stderr


@pytest.mark.parametrize('scales', [(0., 1.), (1., float('nan')), (1e-30, 1e-30), (1., 1e-40)])
def test_native_globals_executable_range(native_file, scales):
    path, specs, kinds, offsets, _, _, probe = native_file
    index = next(i for i, k in enumerate(kinds) if k)
    with path.open('r+b') as stream:
        stream.seek(offsets[index] + tensor_bytes(specs[index], kinds[index]) - 8)
        stream.write(struct.pack('<2f', *scales))
    result = probe()
    assert result.returncode != 0
    assert 'global scales' in result.stderr


@lru_cache(maxsize=None)
def zero_digest(size):
    state = hashlib.sha256()
    block = bytes(8*1024*1024)
    while size:
        count = min(size, len(block))
        state.update(memoryview(block)[:count])
        size -= count
    return state.digest()


@pytest.mark.parametrize('kind', ['weight_padding', 'scale_padding', 'scale_nan', 'fp8_nan'])
def test_native_full_verification_checks_quantized_payload(native_file, kind):
    path, specs, kinds, offsets, table, publish, probe = native_file
    index = next(i for i, k in enumerate(kinds) if k == (2 if kind == 'fp8_nan' else 1))
    # Hash the sparse prefix so verification reaches the changed payload. The
    # verifier checks quantized values before its tensor checksum, so no bogus
    # hash error can accidentally satisfy these semantic rejection assertions.
    for i in range(index):
        length = tensor_bytes(specs[i], kinds[i])
        if not kinds[i]:
            digest = zero_digest(length)
        else:
            with path.open('rb') as stream:
                stream.seek(offsets[i])
                digest = hashlib.sha256(stream.read(length)).digest()
        table[i*wire.ENTRY_BYTES+36:i*wire.ENTRY_BYTES+68] = digest
    publish()
    rows, columns = specs[index].shape
    packed = wire.align_up(rows, 128)*columns//2
    if kind == 'weight_padding':
        relative, value = rows*columns//2, 1
    elif kind == 'scale_padding':
        # First padded row in the last 128-row swizzled tile, column zero.
        relative = packed + (rows//128)*128*(columns//16) + ((rows%32)*4 + (rows%128)//32)*4
        value = 1
    elif kind == 'scale_nan':
        relative, value = packed, 0x7f
    else:
        relative, value = 0, 0xff
    with path.open('r+b') as stream:
        stream.seek(offsets[index] + relative)
        stream.write(bytes([value]))
    assert probe().returncode == 0  # Startup never scans matrix payloads.
    result = probe(True)
    assert result.returncode != 0
    assert {'weight_padding': 'weight padding', 'scale_padding': 'scale padding',
            'scale_nan': 'block scale', 'fp8_nan': 'FP8 payload'}[kind] in result.stderr
