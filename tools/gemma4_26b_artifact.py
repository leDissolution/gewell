"""Streaming BF16 artifact I/O for the fixed Gemma 4 26B A4B inventory.

Uses the existing 72-byte entry encoding and hash/alignment helpers. Expert
identity is determined by the fixed inventory; the model magic is distinct.
"""
from contextlib import ExitStack
import hashlib
import os
from pathlib import Path

from tools import bf16_artifact as wire
from tools.gemma4_26b_contract import native_tensor_specs, source_shapes

MAGIC = b'G4A4BF16'
VERSION = 1
FORMAT_NAME = 'gemma4-26b-a4b-bf16-v1'
require = wire._require


def read_source(snapshot: Path):
    """Return bounded native views into validated stacked BF16 source tensors."""
    directory = snapshot if snapshot.is_dir() else snapshot.parent
    index_path = directory / 'model.safetensors.index.json'
    indexed = index_path.exists() and snapshot.is_dir()
    index = wire.load_json_object(index_path).get('weight_map') if indexed else None
    if indexed:
        require(isinstance(index, dict) and index, 'source index has no weight_map')
        require(all(isinstance(k, str) and isinstance(v, str) for k, v in index.items()), 'invalid weight_map')
        files = [directory / wire.safe_basename(s, 'source shard') for s in sorted(set(index.values()))]
    else:
        files = sorted(snapshot.glob('*.safetensors')) if snapshot.is_dir() else [snapshot]
    records = {}
    for path in files:
        for name, record in wire.read_safetensors_header(path).items():
            require(name not in records, f'duplicate source tensor: {name}')
            records[name] = (path, record)
    if index is not None:
        require(index == {k: p.name for k, (p, _) in records.items()}, 'index/shard inventory mismatch')
    for name, shape in source_shapes().items():
        require(name in records, f'missing 26B source tensor: {name}')
        record = records[name][1]
        require(record.dtype == 'BF16' and record.shape == shape, f'wrong BF16 source tensor: {name}')
    views = []
    for spec in native_tensor_specs():
        path, record = records[spec.source_name]
        views.append(wire.SourceTensor(path, path.name, spec.source_name,
                                       record.offset + 2 * spec.source_element_offset, spec.byte_length))
    config = directory / 'config.json'
    return views, (bytes.fromhex(wire.sha256_file(config)) if config.exists() else bytes(32)), \
        (bytes.fromhex(wire.sha256_file(index_path)) if index is not None else bytes(32))


def plan():
    specs = native_tensor_specs()
    data = wire.data_offset_for_count(len(specs))
    payload = sum(wire.align_up(s.byte_length) for s in specs)
    return {'format': FORMAT_NAME, 'tensor_count': len(specs), 'data_offset': data,
            'logical_data_bytes': sum(s.byte_length for s in specs),
            'payload_bytes': payload, 'file_bytes': data + payload}


def encode_header(config_hash, index_hash, table_hash, payload_hash):
    sizes = plan()
    count = sizes['tensor_count']
    prefix = wire.HEADER_STRUCT.pack(
        MAGIC, VERSION, wire.HEADER_BYTES, wire.ENTRY_BYTES, count, count + 1,
        1, wire.ALIGNMENT, 1, 1, 1, wire.HEADER_BYTES,
        sizes['data_offset'], sizes['logical_data_bytes'], sizes['payload_bytes'], sizes['file_bytes'],
        count, 0, bytes(32), bytes(40), config_hash, index_hash, table_hash, payload_hash, bytes(32))
    header = bytearray(prefix + bytes(wire.HEADER_BYTES - len(prefix)))
    header[wire.HEADER_HASH_OFFSET:wire.HEADER_HASH_OFFSET + 32] = hashlib.sha256(header).digest()
    return bytes(header)


def write_artifact(path: Path, sources, config_hash, index_hash):
    """Exclusively create the artifact; caller publishes only after completion."""
    specs = native_tensor_specs()
    require(len(sources) == len(specs), 'source count mismatch')
    table = bytearray()
    payload_hash = hashlib.sha256()
    with ExitStack() as stack:
        output = stack.enter_context(path.open('xb'))
        output.write(bytes(plan()['data_offset']))
        handles = {}
        for spec, source in zip(specs, sources, strict=True):
            require(source.source_name == spec.source_name and source.byte_length == spec.byte_length,
                    f'source view mismatch: {spec.physical_id}')
            if source.path not in handles:
                handles[source.path] = stack.enter_context(source.path.open('rb'))
            digest = hashlib.sha256()
            offset = output.tell()
            wire._copy_source_range(handles[source.path], output, source.offset, source.byte_length,
                                    digest, payload_hash)
            padding = bytes(wire.align_up(source.byte_length) - source.byte_length)
            output.write(padding)
            payload_hash.update(padding)
            table.extend(wire.ENTRY_STRUCT.pack(spec.physical_id, spec.layer, int(spec.role),
                                                len(spec.shape), 0, spec.shape[0],
                                                spec.shape[1] if len(spec.shape) == 2 else 0, 0,
                                                offset, spec.byte_length, digest.digest()))
        require(output.tell() == plan()['file_bytes'], 'artifact size mismatch')
        output.seek(0)
        output.write(encode_header(config_hash, index_hash, hashlib.sha256(table).digest(), payload_hash.digest()))
        output.write(table)
        output.flush()
        os.fsync(output.fileno())
    return plan()


def inspect_artifact(path: Path, *, verify=False):
    specs = native_tensor_specs()
    sizes = plan()
    with path.open('rb') as stream:
        raw = wire._read_exact(stream, wire.HEADER_BYTES, '26B header')
        header, _ = wire.decode_header(raw)
        expected = encode_header(header.config_sha256, header.source_index_sha256,
                                 header.entry_table_sha256, header.payload_sha256)
        require(raw == expected, 'invalid 26B artifact identity, geometry, reserved bytes or header hash')
        require(path.stat().st_size == sizes['file_bytes'], '26B artifact length mismatch')
        table = wire._read_exact(stream, len(specs) * wire.ENTRY_BYTES, '26B entries')
        require(hashlib.sha256(table).digest() == header.entry_table_sha256, '26B entry table hash mismatch')
        cursor = sizes['data_offset']
        entries = []
        for i, spec in enumerate(specs):
            raw_entry = table[i * wire.ENTRY_BYTES:(i + 1) * wire.ENTRY_BYTES]
            entry = wire.ENTRY_STRUCT.unpack(raw_entry)
            expected_fields = (i, spec.layer, int(spec.role), len(spec.shape), 0, spec.shape[0],
                               spec.shape[1] if len(spec.shape) == 2 else 0, 0, cursor, spec.byte_length)
            require(entry[:10] == expected_fields and raw_entry[-4:] == bytes(4), f'invalid 26B entry {i}')
            entries.append(entry)
            cursor += wire.align_up(spec.byte_length)
        if verify:
            gap = sizes['data_offset'] - stream.tell()
            require(wire._read_exact(stream, gap, 'entry padding') == bytes(gap), 'nonzero entry padding')
            payload = hashlib.sha256()
            for entry in entries:
                digest = hashlib.sha256()
                for chunk in wire.iter_file_chunks(stream, entry[9]):
                    digest.update(chunk)
                    payload.update(chunk)
                require(digest.digest() == entry[10], f'tensor {entry[0]} checksum mismatch')
                padding = wire._read_exact(stream, wire.align_up(entry[9]) - entry[9], 'tensor padding')
                require(not any(padding), f'nonzero tensor {entry[0]} padding')
                payload.update(padding)
            require(payload.digest() == header.payload_sha256, '26B payload hash mismatch')
    return sizes


def snapshot_identity(path):
    path = Path(path).absolute()
    if path.parent.name == 'snapshots' and path.parent.parent.name.startswith('models--'):
        return {'repository': path.parent.parent.name[8:].replace('--', '/'), 'revision': path.name}
    return {'repository': '', 'revision': ''}
