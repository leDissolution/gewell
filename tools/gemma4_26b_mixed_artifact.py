"""26B mixed artifacts with logical shapes and padded NVFP4 output rows."""
from dataclasses import dataclass
import hashlib
import os

import numpy as np

from tools import bf16_artifact as wire
from tools import nvfp4_artifact as native
from tools.gemma4_26b_contract import native_tensor_specs
from tools.gemma4_26b_quantization import StorageType, physical_shape, tensor_bytes

MAGIC = b'G4A4MIX1'
VERSION = 1
FORMAT_NAME = 'gemma4-26b-a4b-mixed-v1'
require = wire._require


@dataclass(frozen=True)
class Entry:
    file_offset: int
    byte_length: int
    storage_type: StorageType
    sha256: bytes


def plan(storages):
    specs = native_tensor_specs()
    sizes = [tensor_bytes(spec, storage) for spec, storage in zip(specs, storages, strict=True)]
    data_offset = wire.data_offset_for_count(len(specs))
    payload = sum(map(wire.align_up, sizes))
    return {'format': FORMAT_NAME, 'tensor_count': len(specs), 'data_offset': data_offset,
            'logical_data_bytes': sum(sizes), 'payload_bytes': payload, 'file_bytes': data_offset + payload}


def encode_header(sizes, config_hash, index_hash, table_hash, payload_hash):
    count = sizes['tensor_count']
    raw = bytearray(wire.HEADER_STRUCT.pack(
        MAGIC, VERSION, wire.HEADER_BYTES, wire.ENTRY_BYTES, count, count+1,
        1, wire.ALIGNMENT, 1, 1, 1, wire.HEADER_BYTES,
        sizes['data_offset'], sizes['logical_data_bytes'], sizes['payload_bytes'], sizes['file_bytes'],
        count, 0, bytes(32), bytes(40), config_hash, index_hash, table_hash, payload_hash, bytes(32)))
    raw.extend(bytes(wire.HEADER_BYTES - len(raw)))
    raw[wire.HEADER_HASH_OFFSET:wire.HEADER_HASH_OFFSET+32] = hashlib.sha256(raw).digest()
    return bytes(raw)


def read_metadata(path):
    specs = native_tensor_specs()
    with path.open('rb') as stream:
        raw = wire._read_exact(stream, wire.HEADER_BYTES, '26B mixed header')
        header, _ = wire.decode_header(raw)
        table = wire._read_exact(stream, len(specs)*wire.ENTRY_BYTES, '26B mixed entries')
        require(hashlib.sha256(table).digest() == header.entry_table_sha256, '26B entry table hash mismatch')
        entries = []
        cursor = wire.data_offset_for_count(len(specs))
        for i, spec in enumerate(specs):
            encoded = table[i*wire.ENTRY_BYTES:(i+1)*wire.ENTRY_BYTES]
            fields = wire.ENTRY_STRUCT.unpack(encoded)
            storage = StorageType(fields[4])
            size = tensor_bytes(spec, storage)
            expected = (i, spec.layer, int(spec.role), len(spec.shape), int(storage),
                        spec.shape[0], spec.shape[1] if len(spec.shape) == 2 else 0, 0, cursor, size)
            require(fields[:10] == expected and encoded[-4:] == bytes(4), f'invalid 26B mixed entry {i}')
            entries.append(Entry(cursor, size, storage, fields[10]))
            cursor += wire.align_up(size)
        sizes = plan(tuple(entry.storage_type for entry in entries))
        require(raw == encode_header(sizes, header.config_sha256, header.source_index_sha256,
                                     header.entry_table_sha256, header.payload_sha256),
                'invalid 26B mixed identity, geometry, reserved bytes or header hash')
        require(path.stat().st_size == sizes['file_bytes'], '26B mixed artifact length mismatch')
        for entry in entries:
            if entry.storage_type != StorageType.BF16:
                stream.seek(entry.file_offset + entry.byte_length - 8)
                native.validate_gemm_globals(wire._read_exact(stream, 8, '26B global scales'))
    return header, tuple(entries)


def write_artifact(path, sources, config_hash=bytes(32), index_hash=bytes(32)):
    specs = native_tensor_specs()
    entries, table = [], bytearray()
    payload = hashlib.sha256()
    data_offset = wire.data_offset_for_count(len(specs))
    with path.open('xb') as output:
        output.write(bytes(data_offset))
        for spec, declaration in zip(specs, sources, strict=True):
            storage = declaration.storage_type
            length = tensor_bytes(spec, storage)
            offset, copied = output.tell(), 0
            digest = hashlib.sha256()

            def emit(raw):
                nonlocal copied
                output.write(raw)
                digest.update(raw)
                payload.update(raw)
                copied += len(raw)

            require(declaration.weight.source_name in (spec.source_name, spec.separate_source_name),
                    f'source identity mismatch: {spec.name}')
            source_bytes = (length if declaration.payload_passthrough else spec.byte_length
                            if storage == StorageType.BF16 else native.fp8_packed_bytes(*spec.shape)
                            if storage == StorageType.FP8_W8A8 else native.packed_bytes(*spec.shape))
            require(declaration.weight.byte_length == source_bytes, f'source length mismatch: {spec.name}')
            with declaration.weight.path.open('rb') as stream:
                stream.seek(declaration.weight.offset)
                for chunk in wire.iter_file_chunks(stream, source_bytes):
                    if storage == StorageType.FP8_W8A8 and not declaration.payload_passthrough:
                        require(not np.any((np.frombuffer(chunk, dtype=np.uint8) & 0x7f) == 0x7f),
                                'nonfinite FP8 source')
                    emit(chunk)
            if declaration.payload_passthrough:
                require(declaration.expected_sha256 is not None and declaration.block_scales is None
                        and not declaration.globals, 'invalid native passthrough declaration')
            elif storage == StorageType.NVFP4_W4A4:
                rows, columns = spec.shape
                padded_rows, _ = physical_shape(spec, storage)
                emit(bytes((padded_rows-rows)*columns//2))
                require(declaration.block_scales is not None and
                        declaration.block_scales.byte_length == rows*(columns//16), 'source block scale size mismatch')
                with declaration.block_scales.path.open('rb') as stream:
                    stream.seek(declaration.block_scales.offset)
                    for row in range(0, rows, 128):
                        count = min(128, rows-row)
                        raw = wire._read_exact(stream, count*(columns//16), 'source block scales')
                        emit(native.swizzle_scale_chunk(raw, count, columns//16))
                native.validate_gemm_globals(declaration.globals)
                emit(declaration.globals)
            elif storage == StorageType.FP8_W8A8:
                require(declaration.block_scales is None, 'FP8 entry has block scales')
                native.validate_gemm_globals(declaration.globals)
                emit(declaration.globals)
            else:
                require(declaration.block_scales is None and not declaration.globals, 'BF16 entry has quantization metadata')
            require(copied == length, f'written tensor length mismatch: {spec.name}')
            checksum = digest.digest()
            if declaration.expected_sha256 is not None:
                require(checksum == declaration.expected_sha256, f'source tensor checksum mismatch: {spec.name}')
            table.extend(wire.ENTRY_STRUCT.pack(spec.physical_id, spec.layer, int(spec.role), len(spec.shape),
                int(storage), spec.shape[0], spec.shape[1] if len(spec.shape) == 2 else 0, 0, offset, length, checksum))
            entries.append(Entry(offset, length, storage, checksum))
            padding = bytes(wire.align_up(length)-length)
            output.write(padding)
            payload.update(padding)
        sizes = plan(tuple(entry.storage_type for entry in entries))
        require(output.tell() == sizes['file_bytes'], '26B mixed artifact size mismatch')
        output.seek(0)
        output.write(encode_header(sizes, config_hash, index_hash, hashlib.sha256(table).digest(), payload.digest()))
        output.write(table)
        output.flush()
        os.fsync(output.fileno())
    return sizes


def verify(path):
    header, entries = read_metadata(path)
    payload = hashlib.sha256()
    with path.open('rb') as stream:
        table_end = wire.HEADER_BYTES + len(entries)*wire.ENTRY_BYTES
        stream.seek(table_end)
        require(not any(wire._read_exact(stream, header.data_offset-table_end, 'table padding')),
                'nonzero table padding')
        for spec, entry in zip(native_tensor_specs(), entries, strict=True):
            digest = hashlib.sha256()
            fp8_remaining = native.fp8_packed_bytes(*spec.shape) if entry.storage_type == StorageType.FP8_W8A8 else 0
            for chunk in wire.iter_file_chunks(stream, entry.byte_length):
                if fp8_remaining:
                    count = min(fp8_remaining, len(chunk))
                    require(not np.any((np.frombuffer(chunk[:count], dtype=np.uint8) & 0x7f) == 0x7f),
                            'nonfinite FP8 payload')
                    fp8_remaining -= count
                digest.update(chunk)
                payload.update(chunk)
            require(digest.digest() == entry.sha256, f'26B tensor checksum mismatch: {spec.name}')
            padding = wire._read_exact(stream, wire.align_up(entry.byte_length)-entry.byte_length, 'tensor padding')
            require(not any(padding), 'nonzero tensor padding')
            payload.update(padding)
            next_offset = stream.tell()
            if entry.storage_type == StorageType.NVFP4_W4A4:
                rows, columns = spec.shape
                padded_rows, _ = physical_shape(spec, entry.storage_type)
                stream.seek(entry.file_offset + native.packed_bytes(rows, columns))
                require(not any(wire._read_exact(stream, (padded_rows-rows)*columns//2, 'NVFP4 weight padding')),
                        'nonzero NVFP4 weight padding')
                scale_columns = wire.align_up(columns//16, 4)
                for row in range(0, padded_rows, 128):
                    raw = wire._read_exact(stream, 128*scale_columns, 'NVFP4 scales')
                    linear = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 32, 4, 4).transpose(2, 1, 0, 3).reshape(128, scale_columns)
                    count = min(rows-row, 128)
                    require(np.all(linear[:count, :columns//16] < 0x7f), 'invalid NVFP4 block scales')
                    require(not np.any(linear[count:]) and not np.any(linear[:, columns//16:]), 'nonzero NVFP4 scale padding')
            stream.seek(next_offset)
    require(payload.digest() == header.payload_sha256, '26B mixed payload checksum mismatch')
    return header, entries
