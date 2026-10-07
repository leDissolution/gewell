"""Fixed 26B projection selectors and native quantized matrix sizes."""
import re

from tools import nvfp4_artifact as native
from tools import bf16_artifact as wire
from tools.gemma4_26b_contract import Role, native_tensor_specs

StorageType = native.StorageType
SHARED_ROLES = {name: Role[name.upper()] for name in
                ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')}
EXPERT_ROLES = {name: Role['EXPERT_' + name.upper()] for name in ('gate_proj', 'up_proj', 'down_proj')}
TARGET_ROLES = frozenset((*SHARED_ROLES.values(), *EXPERT_ROLES.values()))
STORAGE_NAMES = {'bf16': StorageType.BF16, 'fp8_w8a8': StorageType.FP8_W8A8,
                 'nvfp4_w4a4': StorageType.NVFP4_W4A4}


def parse_mask(text, specs=None, *, initial=None):
    """Later rules win; omitted entries retain their source storage."""
    specs = native_tensor_specs() if specs is None else specs
    native.require(len(text.encode()) <= 1024 * 1024, 'mask exceeds 1 MiB')
    selected = list(initial) if initial is not None else [StorageType.BF16] * len(specs)
    native.require(len(selected) == len(specs), 'initial mask length mismatch')
    for number, raw in enumerate(text.splitlines(), 1):
        fields = raw.partition('#')[0].split()
        if not fields:
            continue
        label = f'mask line {number}'
        native.require(len(fields) == 3, f'{label}: expected LAYER PROJECTION TYPE')
        layer, projection, dtype = fields
        native.require(layer == '*' or (layer.isascii() and layer.isdecimal() and 0 <= int(layer) < 30),
                       f'{label}: layer must be * or an integer in [0, 29]')
        native.require(dtype in STORAGE_NAMES, f'{label}: unsupported native artifact type {dtype}')
        expert = None
        if projection in SHARED_ROLES:
            role = SHARED_ROLES[projection]
        else:
            match = re.fullmatch(r'experts\.(\*|[0-9]+)\.(gate_proj|up_proj|down_proj)', projection)
            native.require(match is not None, f'{label}: unknown target projection {projection}')
            expert, name = match.groups()
            native.require(expert == '*' or int(expert) < 128, f'{label}: expert must be * or in [0, 127]')
            role = EXPERT_ROLES[name]
        matches = [i for i, spec in enumerate(specs) if spec.role == role and
                   (layer == '*' or spec.layer == int(layer)) and
                   (expert is None or expert == '*' or spec.expert == int(expert))]
        native.require(bool(matches), f'{label}: projection does not exist for selected layer/expert')
        for i in matches:
            selected[i] = STORAGE_NAMES[dtype]
    for spec, storage in zip(specs, selected, strict=True):
        tensor_bytes(spec, storage)
    return tuple(selected)


def tensor_bytes(spec, storage):
    native.require(storage in tuple(StorageType), f'unsupported storage: {storage}')
    if storage == StorageType.BF16:
        return spec.byte_length
    native.require(spec.role in TARGET_ROLES and 0 <= spec.layer < 30 and len(spec.shape) == 2,
                   f'quantized storage requires a target projection: {spec.separate_source_name}')
    if storage == StorageType.FP8_W8A8:
        return native.fp8_packed_bytes(*spec.shape) + 8
    shape = physical_shape(spec, storage)
    return native.packed_bytes(*shape) + native.scale_bytes(*shape) + 8


def physical_shape(spec, storage):
    if storage == StorageType.NVFP4_W4A4:
        rows, columns = spec.shape
        return wire.align_up(rows, 128), columns
    return spec.shape
