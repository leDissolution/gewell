"""Bounded source views for stacked floating-point or separate 26B experts."""
import math

from tools import bf16_artifact as wire
from tools.gemma4_26b_contract import native_tensor_specs, source_shapes
from tools.gemma4_26b_quantization import TARGET_ROLES, StorageType
from tools.safetensors_source import SourceFiles, Weight


def read_snapshot(path, specs=None):
    specs = native_tensor_specs() if specs is None else specs
    source = SourceFiles(path)
    stacked = {}
    shapes = source_shapes()
    weights = []
    for spec in specs:
        name = spec.separate_source_name
        if spec.expert >= 0 and spec.source_name in source.records:
            wire._require(name not in source.records, f'ambiguous stacked and separate expert: {name}')
            if spec.source_name not in stacked:
                dtype = source.records[spec.source_name][1].dtype
                wire._require(dtype in ('BF16', 'F16', 'F32'),
                              f'stacked quantized experts require an explicit scale mapping: {spec.source_name}')
                stacked[spec.source_name] = source.weight(spec.source_name, shapes[spec.source_name])
            parent = stacked[spec.source_name]
            unit = 4 if parent.dtype == 'F32' else 2
            view = wire.SourceTensor(parent.tensor.path, parent.tensor.shard_name, spec.source_name,
                                     parent.tensor.offset + unit * spec.source_element_offset,
                                     unit * math.prod(spec.shape))
            weight = Weight(view, parent.dtype, spec.shape)
        else:
            weight = source.weight(name, spec.shape)
        wire._require(weight.storage == StorageType.BF16 or spec.role in TARGET_ROLES,
                      f'quantized source is not a supported projection: {name}')
        weights.append(weight)
    return source.snapshot(weights)
