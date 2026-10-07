import hashlib
import struct

import numpy as np
import pytest

from tools import bf16_artifact as wire
from tools import nvfp4_artifact as native
from tools import gemma4_26b_mixed_artifact as artifact
from tools.gemma4_26b_contract import Role, TensorSpec
from tools.gemma4_26b_quantization import StorageType, physical_shape


@pytest.fixture
def packed(tmp_path, monkeypatch):
    specs = (TensorSpec(0, Role.EMBEDDING, -1, -1, (2, 3), 'embedding'),
             TensorSpec(1, Role.EXPERT_GATE_PROJ, 0, 0, (704, 64), 'stacked_gate'),
             TensorSpec(2, Role.EXPERT_UP_PROJ, 0, 0, (704, 64), 'stacked_up'),
             TensorSpec(3, Role.GATE_PROJ, 0, -1, (2112, 64), 'shared_gate'))
    monkeypatch.setattr(artifact, 'native_tensor_specs', lambda: specs)
    kinds = (StorageType.BF16, StorageType.NVFP4_W4A4, StorageType.FP8_W8A8, StorageType.NVFP4_W4A4)
    sources = []
    for i, (spec, kind) in enumerate(zip(specs, kinds)):
        count = spec.byte_length if kind == StorageType.BF16 else np.prod(spec.shape)//(2 if kind == StorageType.NVFP4_W4A4 else 1)
        path = tmp_path/f'weight{i}'
        path.write_bytes(bytes([0x38])*int(count))
        view = wire.SourceTensor(path, path.name, spec.source_name, 0, int(count))
        scales = None
        if kind == StorageType.NVFP4_W4A4:
            raw = (np.arange(spec.shape[0]*spec.shape[1]//16) % 126).astype('u1').tobytes()
            scale_path = tmp_path/f'scale{i}'
            scale_path.write_bytes(raw)
            scales = wire.SourceTensor(scale_path, scale_path.name, 'scales', 0, len(raw))
        sources.append(native.TensorSource(view, kind, scales,
            struct.pack('<2f', .25, .125) if kind != StorageType.BF16 else b''))
    path = tmp_path/'mixed.gwt'
    artifact.write_artifact(path, sources)
    return path, specs, sources


def test_source_values_scales_and_logical_shapes_preserved(packed):
    path, specs, sources = packed
    header, entries = artifact.verify(path)
    assert header.file_bytes == path.stat().st_size
    with path.open('rb') as stream:
        for spec, source, entry in zip(specs, sources, entries):
            stream.seek(entry.file_offset)
            assert stream.read(source.weight.byte_length) == source.weight.path.read_bytes()
            if source.storage_type == StorageType.NVFP4_W4A4:
                rows, columns = spec.shape
                padded, _ = physical_shape(spec, source.storage_type)
                assert not any(stream.read((padded-rows)*columns//2))
                decoded = []
                for row in range(0, padded, 128):
                    tile = np.frombuffer(stream.read(128*(columns//16)), dtype='u1')
                    decoded.append(tile.reshape(-1, 32, 4, 4).transpose(2, 1, 0, 3).reshape(128, columns//16))
                scales = np.concatenate(decoded)
                assert scales[:rows].tobytes() == source.block_scales.path.read_bytes()
                assert not scales[rows:].any()
            if source.storage_type != StorageType.BF16:
                assert stream.read(8) == source.globals


def test_native_passthrough_preserves_whole_entries(packed, tmp_path):
    path, specs, _ = packed
    _, entries = artifact.read_metadata(path)
    sources = [native.TensorSource(wire.SourceTensor(path, path.name, spec.separate_source_name,
                entry.file_offset, entry.byte_length), entry.storage_type,
                expected_sha256=entry.sha256, payload_passthrough=True)
               for spec, entry in zip(specs, entries)]
    target = tmp_path/'copy.gwt'
    artifact.write_artifact(target, sources)
    artifact.verify(target)
    assert path.read_bytes() == target.read_bytes()
    with pytest.raises(FileExistsError):
        artifact.write_artifact(target, sources)


def repair_payload_checksums(path, specs):
    raw = bytearray(path.read_bytes())
    header, entries = artifact.read_metadata(path)
    for i, entry in enumerate(entries):
        checksum = hashlib.sha256(raw[entry.file_offset:entry.file_offset+entry.byte_length]).digest()
        offset = wire.HEADER_BYTES + i*wire.ENTRY_BYTES + 36
        raw[offset:offset+32] = checksum
    sizes = artifact.plan(tuple(e.storage_type for e in entries))
    table = raw[wire.HEADER_BYTES:wire.HEADER_BYTES+len(specs)*wire.ENTRY_BYTES]
    raw[:wire.HEADER_BYTES] = artifact.encode_header(sizes, header.config_sha256, header.source_index_sha256,
        hashlib.sha256(table).digest(), hashlib.sha256(raw[header.data_offset:]).digest())
    path.write_bytes(raw)


def test_nonzero_physical_output_padding_rejected_even_with_valid_hashes(packed):
    path, specs, _ = packed
    _, entries = artifact.read_metadata(path)
    with path.open('r+b') as stream:
        stream.seek(entries[1].file_offset + 704*64//2)
        stream.write(b'\x01')
    repair_payload_checksums(path, specs)
    with pytest.raises(ValueError, match='NVFP4 weight padding'):
        artifact.verify(path)


def test_nonzero_scale_padding_rejected_even_with_valid_hashes(packed):
    path, specs, _ = packed
    _, entries = artifact.read_metadata(path)
    with path.open('r+b') as stream:
        # Row704 is in the third32-row group of the final128-row scale tile.
        stream.seek(entries[1].file_offset + 768*64//2 + 5*128*4 + 2*4)
        stream.write(b'\x01')
    repair_payload_checksums(path, specs)
    with pytest.raises(ValueError, match='scale padding'):
        artifact.verify(path)


def test_metadata_does_not_scan_weight_payload(packed):
    path, _, _ = packed
    _, entries = artifact.read_metadata(path)
    with path.open('r+b') as stream:
        stream.seek(entries[0].file_offset)
        stream.write(b'\xff')
    artifact.read_metadata(path)
    with pytest.raises(ValueError, match='checksum'):
        artifact.verify(path)


def test_invalid_globals_rejected_during_metadata_loading(packed):
    path, _, _ = packed
    _, entries = artifact.read_metadata(path)
    with path.open('r+b') as stream:
        stream.seek(entries[1].file_offset + entries[1].byte_length - 8)
        stream.write(struct.pack('<2f', 0, 1))
    with pytest.raises(ValueError, match='finite and positive'):
        artifact.read_metadata(path)


def test_changed_native_entry_is_not_silently_copied(packed, tmp_path):
    path, specs, _ = packed
    _, entries = artifact.read_metadata(path)
    sources = [native.TensorSource(wire.SourceTensor(path, path.name, spec.separate_source_name,
               entry.file_offset, entry.byte_length), entry.storage_type,
               expected_sha256=entry.sha256, payload_passthrough=True) for spec, entry in zip(specs, entries)]
    with path.open('r+b') as stream:
        stream.seek(entries[1].file_offset)
        stream.write(b'\xff')
    with pytest.raises(ValueError, match='source tensor checksum'):
        artifact.write_artifact(tmp_path/'bad-copy.gwt', sources)


def test_wrong_model_identity_rejected(packed):
    path, _, _ = packed
    with path.open('r+b') as stream:
        stream.write(native.MIXED_MAGIC)
    with pytest.raises(ValueError, match='identity'):
        artifact.read_metadata(path)


def test_nonfinite_fp8_rejected_even_with_valid_hashes(packed):
    path, specs, _ = packed
    _, entries = artifact.read_metadata(path)
    with path.open('r+b') as stream:
        stream.seek(entries[2].file_offset)
        stream.write(b'\x7f')
    repair_payload_checksums(path, specs)
    with pytest.raises(ValueError, match='nonfinite FP8'):
        artifact.verify(path)
