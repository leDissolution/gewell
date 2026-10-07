#!/usr/bin/env python3
"""Freeze local source headers and payload hashes for the 26B oracle.

Reads one file at a time; never materializes model tensors. This is an evidence
collector, not a substitute for numerical reference captures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def audit(snapshot: Path):
    tensors = {}
    files = {}
    for path in sorted(snapshot.glob('*.safetensors')):
        with path.open('rb') as stream:
            header_size, = struct.unpack('<Q', stream.read(8))
            header = json.loads(stream.read(header_size))
        end = 0
        for name, record in sorted(
            ((k, v) for k, v in header.items() if k != '__metadata__'),
            key=lambda item: item[1]['data_offsets'][0],
        ):
            start, stop = record['data_offsets']
            if name in tensors or start != end or stop < start:
                raise ValueError(f'invalid or duplicate tensor: {name}')
            tensors[name] = dict(record, file=path.name)
            end = stop
        if 8 + header_size + end != path.stat().st_size:
            raise ValueError(f'payload length mismatch: {path}')
        files[path.name] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    if not tensors:
        raise ValueError(f'no weights in {snapshot}')
    index = snapshot / 'model.safetensors.index.json'
    if index.exists():
        declared = json.loads(index.read_text())['weight_map']
        if declared != {k: v['file'] for k, v in tensors.items()}:
            raise ValueError('index and shard inventories differ')
    for path in sorted(snapshot.iterdir()):
        if path.suffix in ('.json', '.jinja'):
            files[path.name] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    return {'snapshot': str(snapshot.resolve()), 'files': files, 'tensors': tensors,
            'tensor_count': len(tensors),
            'payload_bytes': sum(r['data_offsets'][1] - r['data_offsets'][0]
                                 for r in tensors.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.snapshot)
    # Preserve previous evidence; use a new destination for a new observation.
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('tensor_count', 'payload_bytes')}))


if __name__ == '__main__':
    main()
