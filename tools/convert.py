#!/usr/bin/env python3
"""Convert target text projections to BF16, FP8 W8A8, or NVFP4 W4A4 storage.

The sparse mask leaves omitted entries in their original storage. Activation
scales use explicit overrides, retained source scales, then bundled reference
calibration. Online calibration during conversion is coming later.
Safetensors and native weights are accepted by structure, independently of provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import bf16_artifact as bf16
from tools import nvfp4_artifact as native
from tools.safetensors_source import (Weight, read_snapshot, materialize_bf16, warn_precision)
from tools.serving_assets import MANIFEST_SCHEMA_VERSION, publish_serving_assets, read_serving_assets

ARTIFACT_FILENAME = "weights.gwt"
DEFAULT_CALIBRATION = Path(__file__).with_name("default_calibration.json")
STORAGE_NAMES = {native.StorageType.BF16: "bf16", native.StorageType.NVFP4_W4A4: "nvfp4_w4a4",
                 native.StorageType.FP8_W8A8: "fp8_w8a8"}


def e4m3_values() -> np.ndarray:
    codes = np.arange(127, dtype=np.int32)
    exponents, mantissas = codes >> 3, codes & 7
    return np.where(exponents == 0, mantissas / 512.0,
                    (1.0 + mantissas / 8.0) * np.exp2(exponents - 7)).astype(np.float32)


E4M3_VALUES = e4m3_values()


def quantize_e4m3(values: np.ndarray, scale: float) -> bytes:
    """Encode nearest-even finite E4M3 values, saturating at +/-448."""
    native.require(math.isfinite(scale) and scale > 0, "FP8 weight scale must be finite and positive")
    values = np.asarray(values, dtype=np.float32)
    native.require(bool(np.all(np.isfinite(values))), "FP8 source contains nonfinite weights")
    with np.errstate(over="ignore"):
        magnitude = np.minimum(np.abs(values) / np.float32(scale), np.float32(448))
    upper = np.searchsorted(E4M3_VALUES, magnitude).clip(0, 126)
    lower = (upper - 1).clip(0, 126)
    below, above = magnitude - E4M3_VALUES[lower], E4M3_VALUES[upper] - magnitude
    choose_upper = (above < below) | ((above == below) & ((lower & 1) != 0))
    codes = np.where(choose_upper, upper, lower).astype(np.uint8)
    codes |= np.signbit(values).astype(np.uint8) << 7
    return codes.tobytes()


def bf16_chunks(source: bf16.SourceTensor, block_elements: int = 1):
    block_bytes = 2 * block_elements
    native.require(source.byte_length > 0 and source.byte_length % block_bytes == 0,
                   "BF16 weight byte length must contain complete quantization blocks")
    chunk_bytes = max(block_bytes, bf16.CHUNK_BYTES // block_bytes * block_bytes)
    with source.path.open("rb") as stream:
        stream.seek(source.offset)
        remaining = source.byte_length
        while remaining:
            count = min(remaining, chunk_bytes)
            raw = bf16._read_exact(stream, count, "BF16 weights")
            yield (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
            remaining -= count


def quantize_tensor(source: bf16.SourceTensor, output: Path, input_scale: float) -> native.TensorSource:
    absmax = 0.0
    for values in bf16_chunks(source):
        native.require(bool(np.all(np.isfinite(values))), "FP8 source contains nonfinite weights")
        absmax = max(absmax, float(np.max(np.abs(values))))
    weight_scale = float(np.float32(absmax / 448.0)) if absmax else 1.0
    globals_raw = struct.pack("<2f", weight_scale, input_scale)
    native.validate_gemm_globals(globals_raw)
    with output.open("xb") as target:
        for values in bf16_chunks(source):
            target.write(quantize_e4m3(values, weight_scale))
    declaration = bf16.SourceTensor(output, output.name, source.source_name, 0, source.byte_length // 2)
    return native.TensorSource(declaration, native.StorageType.FP8_W8A8, globals=globals_raw)


def quantize_e2m1(values: np.ndarray) -> bytes:
    """Pack nearest-even E2M1 pairs, with the first element in the low nibble."""
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    native.require(values.size % 2 == 0 and bool(np.all(np.isfinite(values))),
                   "E2M1 packing requires finite pairs")
    levels = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
    magnitude = np.minimum(np.abs(values), np.float32(6))
    upper = np.searchsorted(levels, magnitude).clip(0, 7)
    lower = (upper - 1).clip(0, 7)
    below, above = magnitude - levels[lower], levels[upper] - magnitude
    codes = np.where((above < below) | ((above == below) & ((lower & 1) != 0)),
                     upper, lower).astype(np.uint8)
    codes |= np.signbit(values).astype(np.uint8) << 3
    return (codes[::2] | (codes[1::2] << 4)).tobytes()


def quantize_nvfp4_tensor(source: bf16.SourceTensor, output: Path,
                          input_scale: float) -> native.TensorSource:
    """Pack the max-calibrated group-16 weight recipe used by weight QDQ."""
    absmax = 0.0
    for values in bf16_chunks(source, 16):
        native.require(bool(np.all(np.isfinite(values))), "NVFP4 source contains nonfinite weights")
        absmax = max(absmax, float(np.max(np.abs(values))))
    weight_scale = np.float32(absmax / (6 * 448)) if absmax else np.float32(1)
    globals_raw = struct.pack("<2f", weight_scale, input_scale)
    # Both native GEMMs require executable FP32 reciprocal/product scales.
    native.validate_gemm_globals(globals_raw)
    scales_path = output.with_suffix(".scales")
    with output.open("xb") as target, scales_path.open("xb") as scales:
        for values in bf16_chunks(source, 16):
            blocks = values.reshape(-1, 16)
            block_max = np.max(np.abs(blocks), axis=1)
            unrounded = block_max / (np.float32(6) * weight_scale)
            unrounded = np.where(block_max == 0, np.float32(1),
                                 np.clip(unrounded, np.float32(1 / 512), np.float32(448)))
            raw_scales = quantize_e4m3(unrounded, 1.0)
            scales.write(raw_scales)
            decoded = E4M3_VALUES[np.frombuffer(raw_scales, dtype=np.uint8)] * weight_scale
            target.write(quantize_e2m1(blocks / decoded[:, None]))
    weight = bf16.SourceTensor(output, output.name, source.source_name, 0, source.byte_length // 4)
    scales = bf16.SourceTensor(scales_path, scales_path.name, source.source_name + ".scales",
                               0, source.byte_length // 32)
    return native.TensorSource(weight, native.StorageType.NVFP4_W4A4, scales, globals_raw)


def identity(path: Path) -> tuple[int, int, int, int]:
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def validate_input_scale(value, name):
    native.require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                   f"missing or invalid calibrated input scale: {name}")
    try:
        _, value = native.validate_gemm_globals(struct.pack("<2f", 1.0, value))
    except (OverflowError, struct.error) as error:
        raise bf16.ArtifactError(f"input scale is not finite positive FP32: {name}") from error
    return value


def resolve_input_scales(specs, selected, overrides, retained):
    """Resolve target-format dequantization scales without executing the model."""
    native.require(isinstance(overrides, dict), "input scales must be a JSON object")
    names = {spec.name for spec in specs if spec.role in native.TARGET_ROLES}
    native.require(not (overrides.keys() - names),
                   f"unknown input scale names: {sorted(overrides.keys() - names)}")
    scales, origins, profile = {}, {}, None
    for spec, storage in zip(specs, selected, strict=True):
        if storage == native.StorageType.BF16:
            continue
        if spec.name in overrides:
            value, origin = overrides[spec.name], "explicit"
        elif retained.get(spec.name) is not None:
            value, origin = retained[spec.name], "source"
        else:
            if profile is None:
                profile = bf16.load_json_object(DEFAULT_CALIBRATION)
            amax = profile["input_amax"].get(spec.name)
            native.require(type(amax) in (int, float) and math.isfinite(amax) and amax > 0,
                           f"missing or invalid default activation range: {spec.name}; supply --input-scales")
            divisor = 448.0 if storage == native.StorageType.FP8_W8A8 else 2688.0
            value, origin = amax / divisor, "default"
        scales[spec.name] = validate_input_scale(value, spec.name)
        origins[spec.name] = origin
    calibration = {"origins": origins}
    if profile is not None:
        calibration["default_profile"] = {"profile": profile["profile"],
            "sha256": bf16.sha256_file(DEFAULT_CALIBRATION), "provenance": profile["provenance"]}
        count = sum(origin == "default" for origin in origins.values())
        print(f"Using bundled reference calibration {profile['profile']} for {count} projection(s); "
              "these ranges were not measured on this source or mask. "
              "Override with --input-scales; online calibration is coming later.", file=sys.stderr)
    return scales, calibration


def prepare_source(path: Path, specs: Sequence[bf16.TensorSpec], mask: str,
                   input_scales: dict, *, verify: bool = True):
    original = identity(path)
    with path.open("rb") as source:
        magic = source.read(8)
    native.require(magic in (bf16.MAGIC, native.MAGIC, native.MIXED_MAGIC), "unsupported source artifact magic")
    is_native = magic != bf16.MAGIC
    if verify:
        artifact = native.verify_artifact_file(path, specs, native=is_native)
    else:
        header, entries = native.read_metadata(path, specs, native=is_native)
        artifact = bf16.VerifiedArtifact(path, header, entries, "")
    selected = native.parse_mask(mask, specs, initial=tuple(entry.storage_type for entry in artifact.entries))
    retained = {}
    with path.open("rb") as source:
        for spec, entry, storage in zip(specs, artifact.entries, selected, strict=True):
            if storage == entry.storage_type and storage != native.StorageType.BF16:
                source.seek(entry.file_offset + entry.byte_length - 4)
                retained[spec.name], = struct.unpack("<f", bf16._read_exact(source, 4, "input scale"))
    scales, calibration = resolve_input_scales(specs, selected, input_scales, retained)
    native.require(identity(path) == original, "source changed during verification")
    warn_precision([Weight(bf16.SourceTensor(path, path.name, spec.source_name, entry.file_offset, entry.byte_length),
                           "BF16" if entry.storage_type == native.StorageType.BF16 else
                           "F8_E4M3" if entry.storage_type == native.StorageType.FP8_W8A8 else "U8", spec.shape)
                    for spec, entry in zip(specs, artifact.entries, strict=True)], selected)
    return artifact, selected, original, scales, calibration


def native_weight(path, spec, entry, temporary):
    """Expose packed native data in the same layout as safetensors input."""
    storage = entry.storage_type
    length = (spec.byte_length if storage == native.StorageType.BF16 else
              native.fp8_packed_bytes(*spec.shape) if storage == native.StorageType.FP8_W8A8 else
              native.packed_bytes(*spec.shape))
    tensor = bf16.SourceTensor(path, path.name, spec.source_name, entry.file_offset, length)
    if storage == native.StorageType.BF16:
        return Weight(tensor, "BF16", spec.shape)
    with path.open("rb") as source:
        source.seek(entry.file_offset + entry.byte_length - 8)
        weight_scale, input_scale = native.validate_globals(bf16._read_exact(source, 8, "global scales"))
        if storage == native.StorageType.FP8_W8A8:
            return Weight(tensor, "F8_E4M3", spec.shape, weight_scale=weight_scale, input_scale=input_scale)
        rows, columns = spec.shape
        scale_columns = bf16.align_up(columns // 16, 4)
        scale_path = temporary / f"{spec.physical_id}.scales"
        source.seek(entry.file_offset + length)
        with scale_path.open("xb") as scales:
            for row in range(0, rows, 128):
                raw = bf16._read_exact(source, 128 * scale_columns, "block scales")
                linear = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 32, 4, 4).transpose(2, 1, 0, 3).reshape(128, scale_columns)
                scales.write(linear[:min(rows - row, 128), :columns // 16].copy().tobytes())
    scales = bf16.SourceTensor(scale_path, scale_path.name, spec.source_name, 0, rows * columns // 16)
    return Weight(tensor, "U8", spec.shape, scales, weight_scale, input_scale)


def prepare_snapshot(path, specs, mask, input_scales):
    source = read_snapshot(path, specs)
    initial = [weight.storage if spec.role in native.TARGET_ROLES else native.StorageType.BF16
               for spec, weight in zip(specs, source.weights, strict=True)]
    selected = native.parse_mask(mask, specs, initial=initial)
    warn_precision(source.weights, selected)
    retained = {spec.name: weight.input_scale
                for spec, weight, storage in zip(specs, source.weights, selected, strict=True)
                if storage == weight.storage}
    scales, calibration = resolve_input_scales(specs, selected, input_scales, retained)
    return source, selected, scales, calibration


def build_manifest(artifact, specs, origins, mask, input_scales, source_info, serving):
    tensors = []
    with artifact.path.open("rb") as stream:
        for spec, entry, origin in zip(specs, artifact.entries, origins, strict=True):
            storage = entry.storage_type
            value = {"physical_id": spec.physical_id, "layer": None if spec.layer < 0 else spec.layer,
                     "role": bf16.ROLE_NAMES[spec.role], "name": spec.name, "shape": list(spec.shape),
                     "dtype": "BF16" if storage == native.StorageType.BF16 else STORAGE_NAMES[storage],
                     "layout": "E4M3_ROW_MAJOR" if storage == native.StorageType.FP8_W8A8 else
                               "E2M1_ROW_MAJOR_E4M3_128X4" if storage == native.StorageType.NVFP4_W4A4 else "C_ORDER",
                     "file_offset": entry.file_offset, "byte_length": entry.byte_length,
                     "slot_length": bf16.align_up(entry.byte_length), "sha256": entry.sha256_hex,
                     "source_name": spec.source_name, "source_component": spec.source_component,
                     "source_shard": origin.shard_name, "source_offset": origin.offset}
            if storage != native.StorageType.BF16:
                stream.seek(entry.file_offset + entry.byte_length - 8)
                weight_scale, input_scale = native.validate_globals(bf16._read_exact(stream, 8, "global scales"))
                value["weight_scale" if storage == native.StorageType.FP8_W8A8 else "weight_scale_2"] = weight_scale
                value["input_scale"] = input_scale
            tensors.append(value)
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION, "format": native.MIXED_FORMAT_NAME,
        "model": {"repository": "", "revision": "",
                  "config_sha256": artifact.header.config_sha256.hex(),
                  "index_sha256": artifact.header.source_index_sha256.hex()},
        "artifact": {"file": ARTIFACT_FILENAME, "file_bytes": artifact.header.file_bytes,
                     "file_sha256": artifact.file_sha256, "header_sha256": artifact.header.header_sha256.hex(),
                     "entry_table_sha256": artifact.header.entry_table_sha256.hex(),
                     "payload_sha256": artifact.header.payload_sha256.hex()},
        "layout": {"alignment": bf16.ALIGNMENT, "logical_data_bytes": artifact.header.logical_data_bytes,
                   "payload_bytes": artifact.header.payload_bytes, "physical_tensor_count": len(specs),
                   "logical_tensor_count": len(specs) + 1, "target": "sm_120a"},
        "aliases": [{"logical_id": bf16.LM_HEAD_LOGICAL_ID, "name": "lm_head.weight",
                     "target": "embed_tokens.weight", "target_physical_id": 0}],
        "serving": serving,
        "source": {**source_info, "mask": mask, "mask_sha256": hashlib.sha256(mask.encode()).hexdigest(),
                   "input_scales": input_scales,
                   "resolved_mask": [{"layer": spec.layer, "projection": bf16.ROLE_NAMES[spec.role],
                                      "type": STORAGE_NAMES[entry.storage_type]}
                                     for spec, entry in zip(specs, artifact.entries, strict=True) if spec.role in native.TARGET_ROLES]},
        "tensors": tensors,
    }


def is_snapshot(path):
    return path.is_dir() or path.suffix == ".safetensors"


def _write_partial_manifest(path, manifest):
    with path.open("xb") as output:
        output.write(bf16.canonical_json_bytes(manifest))
        output.flush()
        os.fsync(output.fileno())


def _publish_no_replace(partial, final):
    os.link(partial, final)
    partial.unlink()


def convert(source_path: Path, mask: str, input_scales: dict, output: Path,
            serving_snapshot: Path | None = None) -> tuple[Path, Path]:
    specs = bf16.expected_tensor_specs()
    snapshot = is_snapshot(source_path)
    if snapshot:
        source, selected, scales, calibration = prepare_snapshot(source_path, specs, mask, input_scales)
        config_hash, index_hash = source.config_sha256, source.index_sha256
        source_info = {"snapshot": str(source.path)}
    else:
        source, selected, original, scales, calibration = prepare_source(source_path, specs, mask, input_scales)
        config_hash, index_hash = source.header.config_sha256, source.header.source_index_sha256
        source_info = {"artifact": str(source.path), "file_sha256": source.file_sha256}
    source_info["calibration"] = calibration
    assets = read_serving_assets(serving_snapshot or (source_path if source_path.is_dir() else source_path.parent))
    output.mkdir(parents=True, exist_ok=True)
    artifact_path, manifest_path = output / ARTIFACT_FILENAME, output / "manifest.json"
    partial, manifest_partial = output / (ARTIFACT_FILENAME + ".partial"), output / "manifest.json.partial"
    for path in (artifact_path, manifest_path, partial, manifest_partial):
        native.require(not path.exists() and not path.is_symlink(), f"refusing to replace existing output: {path}")
    published = False
    try:
        # Each temporary decoded tensor is removed after packing. The native writer streams
        # this iterator, so conversion never stages a second full BF16 model.
        with tempfile.TemporaryDirectory(prefix=".convert-", dir=output) as temporary:
            origins = []
            def declarations():
                for i, (spec, storage) in enumerate(zip(specs, selected, strict=True)):
                    with tempfile.TemporaryDirectory(dir=temporary) as tensor_directory:
                        scratch = Path(tensor_directory)
                        if (not snapshot and storage == source.entries[i].storage_type and
                                calibration["origins"].get(spec.name) != "explicit"):
                            entry = source.entries[i]
                            weight = bf16.SourceTensor(source_path, source_path.name, spec.source_name, entry.file_offset, entry.byte_length)
                            origins.append(weight)
                            yield native.TensorSource(weight, storage, expected_sha256=entry.sha256, payload_passthrough=True)
                            continue
                        weight = source.weights[i] if snapshot else native_weight(source_path, spec, source.entries[i], scratch)
                        origins.append(weight.tensor)
                        if storage == weight.storage and storage != native.StorageType.BF16:
                            yield native.TensorSource(weight.tensor, storage, weight.block_scales,
                                                      struct.pack("<2f", weight.weight_scale, scales[spec.name]))
                        else:
                            decoded = materialize_bf16(weight, scratch / "decoded")
                            if storage == native.StorageType.BF16:
                                yield native.TensorSource(decoded, storage)
                            else:
                                print(f"Quantizing {spec.name}", flush=True)
                                quantizer = quantize_tensor if storage == native.StorageType.FP8_W8A8 else quantize_nvfp4_tensor
                                yield quantizer(decoded, scratch / "packed", scales[spec.name])
            artifact = native.write_artifact_partial(partial, specs, declarations(),
                config_sha256=config_hash, index_sha256=index_hash, mixed=True)
            if snapshot:
                source.assert_unchanged()
            else:
                native.require(identity(source_path) == original, "source changed during conversion")
            verified = native.verify_artifact_file(partial, specs)
            native.require(verified.file_sha256 == artifact.file_sha256, "artifact changed during verification")
            serving = publish_serving_assets(assets, output)
            _write_partial_manifest(manifest_partial, build_manifest(artifact, specs, origins, mask, scales, source_info, serving))
        _publish_no_replace(partial, artifact_path)
        published = True
        _publish_no_replace(manifest_partial, manifest_path)
        descriptor = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        partial.unlink(missing_ok=True)
        manifest_partial.unlink(missing_ok=True)
        if published:
            artifact_path.unlink(missing_ok=True)
        raise
    return artifact_path, manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("gemma4_31b", "gemma4_26b_a4b"),
                        default="gemma4_31b", help="fixed model architecture (independent of serving alias)")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--snapshot", type=Path, help="safetensors file or directory for the selected architecture")
    inputs.add_argument("--artifact", type=Path, help="native source weights")
    parser.add_argument("--mask", type=Path, help="LAYER PROJECTION TYPE mask; omitted projections retain source storage")
    parser.add_argument("--input-scales", type=Path,
                        help="optional JSON map of logical tensor names to target-format activation scales; "
                             "overrides source scales and bundled reference defaults")
    parser.add_argument("--serving-snapshot", type=Path, help="directory containing tokenizer.json; defaults to the source directory")
    parser.add_argument("--output", type=Path, help="bundle directory containing weights.gwt and manifest.json")
    parser.add_argument("--plan", action="store_true", help="validate structure/scales and report sizes without writing weights")
    parser.add_argument("--verify", type=Path, help="fully verify existing native weights")
    args = parser.parse_args(argv)
    try:
        if args.architecture == "gemma4_26b_a4b":
            from tools.gemma4_26b_convert import convert_bundle
            print(json.dumps(convert_bundle(args), indent=2))
            return 0
        specs = bf16.expected_tensor_specs()
        if args.verify is not None:
            with args.verify.open("rb") as stream:
                is_native = stream.read(8) != bf16.MAGIC
            result = native.verify_artifact_file(args.verify, specs, native=is_native)
            print(f"Verified {result.path}: {result.file_sha256}")
            return 0
        source_path = args.snapshot or args.artifact
        if source_path is None:
            parser.error("--snapshot or --artifact is required for conversion and planning")
        mask = args.mask.read_text() if args.mask else ""
        scales = bf16.load_json_object(args.input_scales) if args.input_scales else {}
        if args.plan:
            if args.snapshot:
                _, selected, _, calibration = prepare_snapshot(source_path, specs, mask, scales)
            else:
                _, selected, _, _, calibration = prepare_source(source_path, specs, mask, scales, verify=False)
            read_serving_assets(args.serving_snapshot or (source_path if source_path.is_dir() else source_path.parent))
            sizes = [native.tensor_bytes(spec, storage) for spec, storage in zip(specs, selected, strict=True)]
            print(json.dumps({"format": native.MIXED_FORMAT_NAME,
                              "fp8_projections": sum(x == native.StorageType.FP8_W8A8 for x in selected),
                              "nvfp4_projections": sum(x == native.StorageType.NVFP4_W4A4 for x in selected),
                              "activation_scales": {origin: sum(value == origin for value in calibration["origins"].values())
                                                    for origin in ("explicit", "source", "default")},
                              "default_calibration": calibration.get("default_profile"),
                              "logical_data_bytes": sum(sizes),
                              "file_bytes": bf16.data_offset_for_count(len(specs)) + sum(map(bf16.align_up, sizes))}, indent=2))
        else:
            if args.output is None:
                parser.error("--output is required unless --plan or --verify is selected")
            for path in convert(source_path, mask, scales, args.output, args.serving_snapshot):
                print(path)
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
