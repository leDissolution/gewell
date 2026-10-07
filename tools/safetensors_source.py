"""Structural safetensors input shared by text conversion and vision extraction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
import warnings

import numpy as np

from tools import bf16_artifact as bf16
from tools import nvfp4_artifact as native


def identity(path: Path) -> tuple[int, int, int, int]:
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


@dataclass(frozen=True)
class Weight:
    tensor: bf16.SourceTensor
    dtype: str
    shape: tuple[int, ...]
    block_scales: bf16.SourceTensor | None = None
    weight_scale: float = 1.0
    input_scale: float | None = None

    @property
    def storage(self) -> native.StorageType:
        if self.dtype == "U8":
            return native.StorageType.NVFP4_W4A4
        if self.dtype == "F8_E4M3":
            return native.StorageType.FP8_W8A8
        return native.StorageType.BF16


@dataclass(frozen=True)
class Snapshot:
    path: Path
    weights: tuple[Weight, ...]
    identities: dict[Path, tuple[int, int, int, int]]
    config_sha256: bytes
    index_sha256: bytes

    def assert_unchanged(self) -> None:
        for path, original in self.identities.items():
            bf16._require(identity(path) == original, f"source changed during conversion: {path}")


class SourceFiles:
    """Validated shard inventory shared by the 31B and 26B source readers."""
    def __init__(self, path: Path):
        path = path.resolve()
        directory = path if path.is_dir() else path.parent
        index_path = directory / "model.safetensors.index.json"
        identities = {}
        weight_map = None
        if path.is_dir() and index_path.exists():
            identities[index_path] = identity(index_path)
            weight_map = bf16.load_json_object(index_path).get("weight_map")
            bf16._require(isinstance(weight_map, dict) and bool(weight_map), "source index has no weight_map")
            bf16._require(all(isinstance(name, str) and isinstance(shard, str)
                              for name, shard in weight_map.items()), "invalid source weight_map")
            files = [directory / bf16.safe_basename(name, "source shard") for name in sorted(set(weight_map.values()))]
        else:
            files = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
        bf16._require(bool(files), f"no safetensors files in {path}")
        records = {}
        for shard in files:
            identities[shard] = identity(shard)
            for name, record in bf16.read_safetensors_header(shard).items():
                bf16._require(name not in records, f"duplicate source tensor: {name}")
                records[name] = (shard, record)
        if weight_map is not None:
            bf16._require({name: shard.name for name, (shard, _) in records.items()} == weight_map,
                          "source index/shard tensor inventory mismatch")
        self.path, self.directory = path, directory
        self.records, self.identities, self.index_path = records, identities, index_path if weight_map is not None else None

    def tensor(self, name, dtype=None, shape=None):
        bf16._require(name in self.records, f"missing source tensor: {name}")
        shard, record = self.records[name]
        bf16._require(dtype is None or record.dtype == dtype, f"wrong source dtype: {name}")
        bf16._require(shape is None or record.shape == shape, f"wrong source shape: {name}")
        return bf16.SourceTensor(shard, shard.name, name, record.offset, record.byte_length)

    def scalar(self, name, default=None):
        if name not in self.records:
            return default
        declaration = self.tensor(name, "F32")
        bf16._require(self.records[name][1].shape in ((), (1,)) and declaration.byte_length == 4,
                      f"expected scalar scale: {name}")
        with declaration.path.open("rb") as stream:
            stream.seek(declaration.offset)
            value, = struct.unpack("<f", bf16._read_exact(stream, 4, name))
        bf16._require(np.isfinite(value) and value > 0, f"scale must be finite and positive: {name}")
        return value

    def weight(self, name, shape):
        declaration = self.tensor(name)
        dtype = self.records[name][1].dtype
        bf16._require(dtype in ("BF16", "F16", "F32", "F8_E4M3", "U8"),
                      f"unsupported source dtype {dtype}: {name}")
        prefix = name.removesuffix("weight")
        scales = None
        weight_scale, input_scale = 1.0, None
        if dtype == "U8":
            bf16._require(len(shape) == 2 and shape[1] % 16 == 0,
                          f"NVFP4 requires a matrix with K divisible by 16: {name}")
            rows, columns = shape
            self.tensor(name, dtype, (rows, columns // 2))
            scales = self.tensor(prefix + "weight_scale", "F8_E4M3", (rows, columns // 16))
            weight_scale = self.scalar(prefix + "weight_scale_2")
            bf16._require(weight_scale is not None, f"missing NVFP4 weight_scale_2: {name}")
        else:
            self.tensor(name, dtype, shape)
            if dtype == "F8_E4M3":
                weight_scale = self.scalar(prefix + "weight_scale", 1.0)
        if dtype in ("U8", "F8_E4M3"):
            input_scale = self.scalar(prefix + "input_scale")
        return Weight(declaration, dtype, shape, scales, weight_scale, input_scale)

    def snapshot(self, weights):
        hashes = []
        for metadata in (self.directory / "config.json", self.index_path):
            if metadata is not None and metadata.exists():
                self.identities.setdefault(metadata, identity(metadata))
                hashes.append(bytes.fromhex(bf16.sha256_file(metadata)))
            else:
                hashes.append(bytes(32))
        snapshot = Snapshot(self.path, tuple(weights), self.identities, *hashes)
        snapshot.assert_unchanged()
        return snapshot


def read_snapshot(path: Path, specs) -> Snapshot:
    """Accept a single file or arbitrarily sharded directory; extras are ignored."""
    source = SourceFiles(path)
    return source.snapshot(source.weight(spec.source_name, spec.shape) for spec in specs)


def warn_precision(weights, selected) -> None:
    precision = {native.StorageType.BF16: 16, native.StorageType.FP8_W8A8: 8,
                 native.StorageType.NVFP4_W4A4: 4}
    widened = [weight.tensor.source_name for weight, target in zip(weights, selected, strict=True)
               if precision[weight.storage] < precision[target]]
    if widened:
        warnings.warn(f"{len(widened)} source tensor(s) have lower precision than the target mask "
                      f"(first: {widened[0]}). Widening does not recover lost precision.",
                      UserWarning, stacklevel=2)


def decode_e4m3(raw: bytes) -> np.ndarray:
    codes = np.frombuffer(raw, dtype=np.uint8)
    magnitude = codes & 0x7f
    bf16._require(not np.any(magnitude == 127), "nonfinite E4M3 source values")
    exponent, mantissa = magnitude >> 3, magnitude & 7
    values = np.where(exponent == 0, mantissa / 512.0,
                      (1 + mantissa / 8.0) * np.exp2(exponent.astype(np.int32) - 7)).astype(np.float32)
    return np.copysign(values, np.where(codes & 128, -1.0, 1.0)).astype(np.float32)


def bf16_bytes(values: np.ndarray) -> bytes:
    values = np.asarray(values, dtype=np.float32)
    bf16._require(bool(np.all(np.isfinite(values))), "nonfinite decoded source weights")
    bits = values.view(np.uint32)
    rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype("<u2")
    bf16._require(not np.any((rounded & 0x7fff) == 0x7f80), "decoded weights exceed the finite BF16 range")
    return rounded.tobytes()


def materialize_bf16(weight: Weight, output: Path) -> bf16.SourceTensor:
    """Decode one tensor with bounded memory and BF16 round-to-nearest-even."""
    if weight.dtype == "BF16":
        return weight.tensor
    source = weight.tensor
    # NVFP4 chunks contain complete groups of 16 weights (eight packed bytes).
    unit = {"F16": 2, "F32": 4, "F8_E4M3": 1, "U8": 8}[weight.dtype]
    chunk_bytes = max(unit, bf16.CHUNK_BYTES // unit * unit)
    with source.path.open("rb") as stream, output.open("xb") as target:
        stream.seek(source.offset)
        scales = weight.block_scales.path.open("rb") if weight.block_scales else None
        try:
            if scales:
                scales.seek(weight.block_scales.offset)
            remaining = source.byte_length
            while remaining:
                raw = bf16._read_exact(stream, min(remaining, chunk_bytes), source.source_name)
                remaining -= len(raw)
                if weight.dtype in ("F16", "F32"):
                    values = np.frombuffer(raw, dtype="<f2" if weight.dtype == "F16" else "<f4").astype(np.float32)
                elif weight.dtype == "F8_E4M3":
                    values = decode_e4m3(raw) * np.float32(weight.weight_scale)
                else:
                    packed = np.frombuffer(raw, dtype=np.uint8)
                    codes = np.empty(len(raw) * 2, dtype=np.uint8)
                    codes[::2], codes[1::2] = packed & 15, packed >> 4
                    levels = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)
                    values = np.copysign(levels[codes & 7], np.where(codes & 8, -1.0, 1.0)).astype(np.float32)
                    scale_raw = bf16._read_exact(scales, len(raw) // 8, "NVFP4 block scales")
                    bf16._require(all(code < 127 for code in scale_raw), "invalid NVFP4 block scales")
                    multipliers = decode_e4m3(scale_raw) * np.float32(weight.weight_scale)
                    values = (values.reshape(-1, 16) * multipliers[:, None]).reshape(-1)
                target.write(bf16_bytes(values))
        finally:
            if scales:
                scales.close()
    return bf16.SourceTensor(output, output.name, source.source_name, 0, output.stat().st_size)
