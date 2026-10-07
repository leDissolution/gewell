import pytest

from tools.gemma4_26b_contract import Role, native_tensor_specs
from tools.gemma4_26b_quantization import StorageType, parse_mask, tensor_bytes


def test_expert_masks_are_independent_and_later_rules_win():
    specs = native_tensor_specs()
    result = parse_mask('* gate_proj fp8_w8a8\n* experts.*.gate_proj nvfp4_w4a4\n'
                        '* experts.*.up_proj fp8_w8a8\n12 experts.17.gate_proj bf16', specs)
    for spec, storage in zip(specs, result):
        expected = StorageType.BF16
        if spec.role in (Role.GATE_PROJ, Role.EXPERT_UP_PROJ):
            expected = StorageType.FP8_W8A8
        if spec.role == Role.EXPERT_GATE_PROJ and (spec.layer, spec.expert) != (12, 17):
            expected = StorageType.NVFP4_W4A4
        assert storage == expected


def test_omitted_entries_retain_source_storage():
    specs = native_tensor_specs()
    initial = tuple(StorageType.NVFP4_W4A4 if s.expert >= 0 else StorageType.BF16 for s in specs)
    result = parse_mask('0 experts.127.down_proj fp8_w8a8', specs, initial=initial)
    changed = [s for s, a, b in zip(specs, initial, result) if a != b]
    assert len(changed) == 1
    assert (changed[0].layer, changed[0].expert, changed[0].role) == (0, 127, Role.EXPERT_DOWN_PROJ)


@pytest.mark.parametrize('rule', ['30 gate_proj bf16', '-1 gate_proj bf16',
    '* experts.128.up_proj bf16', '* experts.-1.up_proj bf16', '* experts.*.router_proj fp8_w8a8',
    '* router_proj fp8_w8a8', '* gate_proj int8', '5 v_proj fp8_w8a8',
    '* experts.١.up_proj bf16', '* experts.*.up_proj bf16 extra'])
def test_invalid_selectors(rule):
    with pytest.raises(ValueError):
        parse_mask(rule)


def test_narrow_quantized_sizes_include_scale_padding_and_globals():
    specs = native_tensor_specs()
    gate = next(s for s in specs if s.role == Role.EXPERT_GATE_PROJ)
    down = next(s for s in specs if s.role == Role.EXPERT_DOWN_PROJ)
    shared = next(s for s in specs if s.role == Role.GATE_PROJ)
    assert tensor_bytes(gate, StorageType.NVFP4_W4A4) == 768*2816//2 + 768*176 + 8
    assert tensor_bytes(down, StorageType.NVFP4_W4A4) == 2816*704//2 + 2816*44 + 8
    assert tensor_bytes(shared, StorageType.NVFP4_W4A4) == 2176*2816//2 + 2176*176 + 8
    assert tensor_bytes(down, StorageType.FP8_W8A8) == 2816*704 + 8
    router = next(s for s in specs if s.role == Role.ROUTER_PROJ)
    with pytest.raises(ValueError):
        tensor_bytes(router, StorageType.FP8_W8A8)
