"""Bounded mixed-format conversion for the fixed 26B text inventory."""
import hashlib
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile

import numpy as np

from tools import bf16_artifact as wire
from tools import gemma4_26b_artifact as bf16
from tools import gemma4_26b_mixed_artifact as mixed
from tools import nvfp4_artifact as native
from tools import convert as common
from tools.gemma4_26b_contract import native_tensor_specs
from tools.gemma4_26b_quantization import StorageType, TARGET_ROLES, parse_mask, physical_shape
from tools.gemma4_26b_source import read_snapshot
from tools.safetensors_source import Weight, identity, materialize_bf16, warn_precision
from tools.serving_assets import read_serving_assets, publish_serving_assets

DEFAULT_CALIBRATION = Path(__file__).with_name('gemma4_26b_default_calibration.json')
require = wire._require


def read_native(path, *, verify=False):
    with path.open('rb') as stream:
        magic = stream.read(8)
    if magic == mixed.MAGIC:
        header, entries = mixed.verify(path) if verify else mixed.read_metadata(path)
        return header, entries, mixed.FORMAT_NAME
    require(magic == bf16.MAGIC, 'wrong 26B artifact identity')
    bf16.inspect_artifact(path, verify=verify)
    with path.open('rb') as stream:
        header, _ = wire.decode_header(stream.read(wire.HEADER_BYTES))
        table = stream.read(len(native_tensor_specs())*wire.ENTRY_BYTES)
    entries = []
    for i in range(len(native_tensor_specs())):
        entry = wire.ENTRY_STRUCT.unpack_from(table, i*wire.ENTRY_BYTES)
        entries.append(mixed.Entry(entry[8], entry[9], StorageType.BF16, entry[10]))
    return header, tuple(entries), bf16.FORMAT_NAME


def resolve_scales(specs, selected, overrides, retained):
    require(isinstance(overrides, dict), 'input scales must be a JSON object')
    names = {s.name for s in specs if s.role in TARGET_ROLES}
    require(not (overrides.keys()-names), f'unknown input scale names: {sorted(overrides.keys()-names)}')
    scales, origins, missing, profile = {}, {}, [], None
    for spec, storage in zip(specs, selected, strict=True):
        if storage == StorageType.BF16:
            continue
        if spec.name in overrides:
            value, origin = overrides[spec.name], 'explicit'
        elif retained.get(spec.name) is not None:
            value, origin = retained[spec.name], 'source'
        else:
            if profile is None:
                profile = wire.load_json_object(DEFAULT_CALIBRATION) if DEFAULT_CALIBRATION.is_file() else {}
            amax = profile.get('input_amax', {}).get(spec.name)
            if not (type(amax) in (int, float) and math.isfinite(amax) and amax > 0):
                missing.append(spec.name)
                continue
            value, origin = amax/(448. if storage == StorageType.FP8_W8A8 else 2688.), 'default'
        scales[spec.name] = common.validate_input_scale(value, spec.name)
        origins[spec.name] = origin
    calibration = {'origins': origins, 'missing_input_scales': missing}
    if 'default' in origins.values():
        calibration['default_profile'] = {'profile': profile['profile'],
            'sha256': wire.sha256_file(DEFAULT_CALIBRATION), 'provenance': profile['provenance']}
    return scales, calibration


def native_weight(path, spec, entry, scratch):
    """Expose only logical values, removing physical NVFP4 output padding."""
    storage = entry.storage_type
    length = spec.byte_length if storage == StorageType.BF16 else math.prod(spec.shape)//(2 if storage == StorageType.NVFP4_W4A4 else 1)
    view = wire.SourceTensor(path, path.name, spec.separate_source_name, entry.file_offset, length)
    if storage == StorageType.BF16:
        return Weight(view, 'BF16', spec.shape)
    with path.open('rb') as stream:
        stream.seek(entry.file_offset + entry.byte_length-8)
        weight_scale, input_scale = native.validate_gemm_globals(wire._read_exact(stream, 8, 'global scales'))
        if storage == StorageType.FP8_W8A8:
            return Weight(view, 'F8_E4M3', spec.shape, weight_scale=weight_scale, input_scale=input_scale)
        rows, columns = spec.shape
        physical_rows, _ = physical_shape(spec, storage)
        stream.seek(entry.file_offset + physical_rows*columns//2)
        scale_path = scratch/'scales'
        scale_columns = wire.align_up(columns//16, 4)
        with scale_path.open('xb') as output:
            for row in range(0, rows, 128):
                raw = wire._read_exact(stream, 128*scale_columns, 'NVFP4 scales')
                linear = np.frombuffer(raw, dtype=np.uint8).reshape(-1,32,4,4).transpose(2,1,0,3).reshape(128,scale_columns)
                output.write(linear[:min(128,rows-row), :columns//16].copy().tobytes())
    scales = wire.SourceTensor(scale_path, scale_path.name, spec.separate_source_name, 0, rows*columns//16)
    return Weight(view, 'U8', spec.shape, scales, weight_scale, input_scale)


def model_identity(source, header, serving_source):
    config, index = serving_source/'config.json', serving_source/'model.safetensors.index.json'
    if (config.is_file() and index.is_file() and
        bytes.fromhex(wire.sha256_file(config)) == header.config_sha256 and
        bytes.fromhex(wire.sha256_file(index)) == header.source_index_sha256):
        return bf16.snapshot_identity(serving_source)
    manifest = source.parent/'manifest.json'
    if manifest.is_file():
        model = wire.load_json_object(manifest)['model']
        require(model['config_sha256'] == header.config_sha256.hex() and
                model['index_sha256'] == header.source_index_sha256.hex(), 'native manifest source hashes differ')
        return {key: model[key] for key in ('repository', 'revision')}
    return {'repository': '', 'revision': ''}


def bundle_manifest(path, serving, provenance, conversion):
    header, entries, format_name = read_native(path)
    tensors = [{'physical_id': spec.physical_id, 'layer': None if spec.layer < 0 else spec.layer,
                'expert': None if spec.expert < 0 else spec.expert, 'role': spec.role.name.lower(),
                'shape': list(spec.shape), 'dtype': 'BF16' if entry.storage_type == StorageType.BF16 else common.STORAGE_NAMES[entry.storage_type],
                'file_offset': entry.file_offset, 'byte_length': entry.byte_length, 'sha256': entry.sha256.hex()}
               for spec, entry in zip(native_tensor_specs(), entries, strict=True)]
    return {'schema_version': 6, 'architecture': 'gemma4_26b_a4b', 'format': format_name,
        'model': {**provenance, 'config_sha256': header.config_sha256.hex(),
                  'index_sha256': header.source_index_sha256.hex(), 'conversion': conversion},
        'artifact': {'file': 'weights.gwt', 'file_bytes': header.file_bytes,
                     'header_sha256': header.header_sha256.hex(), 'entry_table_sha256': header.entry_table_sha256.hex(),
                     'payload_sha256': header.payload_sha256.hex()},
        'layout': {'alignment': wire.ALIGNMENT, 'physical_tensor_count': len(entries),
                   'logical_tensor_count': len(entries)+1, 'target': 'sm_120a',
                   'logical_data_bytes': header.logical_data_bytes, 'payload_bytes': header.payload_bytes},
        'aliases': [{'logical_id': len(entries), 'name': 'lm_head.weight', 'target': 'embed_tokens.weight', 'target_physical_id': 0}],
        'serving': serving, 'tensors': tensors}


def convert_bundle(args):
    if args.verify is not None:
        header, entries, format_name = read_native(args.verify, verify=True)
        return {'format': format_name, 'tensor_count': len(entries), 'file_bytes': header.file_bytes,
                'payload_sha256': header.payload_sha256.hex()}
    path = args.snapshot or args.artifact
    require(path is not None, '--snapshot or --artifact is required')
    specs = native_tensor_specs()
    mask = args.mask.read_text() if args.mask else ''
    overrides = wire.load_json_object(args.input_scales) if args.input_scales else {}
    original = None
    if args.snapshot:
        source = read_snapshot(path)
        initial = tuple(w.storage for w in source.weights)
        config_hash, index_hash = source.config_sha256, source.index_sha256
    else:
        original = identity(path)
        header, entries, source_format = read_native(path)
        initial = tuple(e.storage_type for e in entries)
        config_hash, index_hash = header.config_sha256, header.source_index_sha256
    selected = parse_mask(mask, specs, initial=initial)
    retained = {}
    if args.snapshot:
        warn_precision(source.weights, selected)
        retained = {s.name: w.input_scale for s,w,target in zip(specs,source.weights,selected,strict=True) if w.storage == target}
    else:
        with path.open('rb') as stream:
            for spec, entry, target in zip(specs, entries, selected, strict=True):
                if target == entry.storage_type and target != StorageType.BF16:
                    stream.seek(entry.file_offset+entry.byte_length-4)
                    retained[spec.name], = struct.unpack('<f', wire._read_exact(stream,4,'input scale'))
        weights = [Weight(wire.SourceTensor(path,path.name,s.separate_source_name,e.file_offset,e.byte_length),
                          'BF16' if e.storage_type == StorageType.BF16 else 'U8' if e.storage_type == StorageType.NVFP4_W4A4 else 'F8_E4M3',s.shape)
                   for s,e in zip(specs,entries,strict=True)]
        warn_precision(weights, selected)
    scales, calibration = resolve_scales(specs, selected, overrides, retained)
    unchanged = not args.snapshot and selected == initial and all(
        scales.get(s.name) == retained.get(s.name) for s in specs)
    sizes = mixed.plan(selected)
    if unchanged:
        sizes.update(format=source_format, file_bytes=header.file_bytes)
    result = {**sizes, 'calibration': calibration, 'ready': not calibration['missing_input_scales']}
    if args.snapshot:
        source.assert_unchanged()
    else:
        require(identity(path) == original, 'source changed during planning')
    if args.plan:
        return result
    require(result['ready'], f"missing input scales for {len(calibration['missing_input_scales'])} projections; first: "
            f"{next(iter(calibration['missing_input_scales']), '')}; supply --input-scales or measure 26B calibration")
    require(args.output is not None, '--output is required unless --plan or --verify is selected')
    output = args.output
    artifact, partial = output/'weights.gwt', output/'weights.gwt.partial'
    manifest, manifest_partial = output/'manifest.json', output/'manifest.json.partial'
    in_place = args.artifact is not None and path.resolve() == artifact.resolve()
    require(not in_place or unchanged, 'cannot change precision or scales in place; choose a new output')
    require(not in_place or not artifact.is_symlink(), 'cannot package a serving artifact symlink')
    for target in (manifest, partial, manifest_partial, *(() if in_place else (artifact,))):
        require(not target.exists() and not target.is_symlink(), f'refusing existing output: {target}')
    serving_source = args.serving_snapshot or (path if path.is_dir() else path.parent)
    assets = read_serving_assets(serving_source)
    provenance = bf16.snapshot_identity(path) if args.snapshot else model_identity(path,header,serving_source)
    output.mkdir(parents=True, exist_ok=True)
    if not in_place:
        if unchanged:
            with path.open('rb') as src, partial.open('xb') as dst:
                shutil.copyfileobj(src,dst,wire.CHUNK_BYTES)
                dst.flush(); os.fsync(dst.fileno())
        else:
            if not args.snapshot:
                read_native(path,verify=True)
            def declarations():
                for i,(spec,target) in enumerate(zip(specs,selected,strict=True)):
                    if (not args.snapshot and target == initial[i] and
                        scales.get(spec.name) == retained.get(spec.name)):
                        entry = entries[i]
                        view = wire.SourceTensor(path,path.name,spec.separate_source_name,entry.file_offset,entry.byte_length)
                        yield native.TensorSource(view,target,expected_sha256=entry.sha256,payload_passthrough=True)
                        continue
                    with tempfile.TemporaryDirectory(prefix='.convert-',dir=output) as directory:
                        scratch = Path(directory)
                        weight = source.weights[i] if args.snapshot else native_weight(path,spec,entries[i],scratch)
                        if target == weight.storage and target != StorageType.BF16:
                            yield native.TensorSource(weight.tensor,target,weight.block_scales,
                                                      struct.pack('<2f',weight.weight_scale,scales[spec.name]))
                        else:
                            decoded = materialize_bf16(weight,scratch/'decoded')
                            if target == StorageType.BF16:
                                yield native.TensorSource(decoded,target)
                            else:
                                quantizer = common.quantize_tensor if target == StorageType.FP8_W8A8 else common.quantize_nvfp4_tensor
                                yield quantizer(decoded,scratch/'packed',scales[spec.name])
            mixed.write_artifact(partial,declarations(),config_hash,index_hash)
        if args.snapshot:
            source.assert_unchanged()
        else:
            require(identity(path) == original, 'source changed during conversion')
        read_native(partial,verify=True)
        os.link(partial,artifact)
        partial.unlink()
    serving = publish_serving_assets(assets,output)
    serving.update(bf16.snapshot_identity(serving_source))
    conversion = {'mask': mask, 'input_scales': scales, 'calibration': calibration,
                  'source': str(path)}
    data = wire.canonical_json_bytes(bundle_manifest(artifact,serving,provenance,conversion))
    require(len(data) <= 8*1024*1024, '26B serving manifest exceeds 8 MiB')
    with manifest_partial.open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.link(manifest_partial,manifest)
    manifest_partial.unlink()
    descriptor = os.open(output,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {**result, 'artifact': str(artifact), 'manifest': str(manifest), 'manifest_bytes': len(data)}
