"""Fixed source tensor shapes for Gemma 4 26B A4B text, assistant and vision weights."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math

PREFIX = 'model.language_model.'


def source_shapes() -> dict[str, tuple[int, ...]]:
    shapes = {PREFIX + 'embed_tokens.weight': (262144, 2816),
              PREFIX + 'norm.weight': (2816,)}
    for layer in range(30):
        global_attention = layer % 6 == 5
        head = 512 if global_attention else 256
        query = 16 * head
        kv = (2 if global_attention else 8) * head
        local = {
            'self_attn.q_proj.weight': (query, 2816),
            'self_attn.k_proj.weight': (kv, 2816),
            'self_attn.o_proj.weight': (2816, query),
            'self_attn.q_norm.weight': (head,),
            'self_attn.k_norm.weight': (head,),
            'mlp.gate_proj.weight': (2112, 2816),
            'mlp.up_proj.weight': (2112, 2816),
            'mlp.down_proj.weight': (2816, 2112),
            'experts.gate_up_proj': (128, 1408, 2816),
            'experts.down_proj': (128, 2816, 704),
            'router.proj.weight': (128, 2816),
            'router.scale': (2816,),
            'router.per_expert_scale': (128,),
            'layer_scalar': (1,),
        }
        if not global_attention:
            local['self_attn.v_proj.weight'] = (kv, 2816)
        for norm in ('input_layernorm', 'post_attention_layernorm',
                     'pre_feedforward_layernorm', 'post_feedforward_layernorm',
                     'pre_feedforward_layernorm_2', 'post_feedforward_layernorm_1',
                     'post_feedforward_layernorm_2'):
            local[norm + '.weight'] = (2816,)
        shapes.update({f'{PREFIX}layers.{layer}.{name}': shape for name, shape in local.items()})
    return shapes


def validate_bf16_text(tensors):
    expected = source_shapes()
    actual = {name: record for name, record in tensors.items() if name.startswith(PREFIX)}
    if actual.keys() != expected.keys():
        raise ValueError(f'text inventory mismatch: missing={expected.keys() - actual.keys()}, '
                         f'extra={actual.keys() - expected.keys()}')
    for name, shape in expected.items():
        record = actual[name]
        if tuple(record['shape']) != shape or record['dtype'] != 'BF16':
            raise ValueError(f'wrong BF16 text tensor: {name}')
    return len(expected)


# Stable role IDs for the 26B artifact; deliberately separate from the 31B enum.


class Role(IntEnum):
    EMBEDDING = 0
    INPUT_NORM = 1
    Q_PROJ = 2
    K_PROJ = 3
    V_PROJ = 4
    Q_NORM = 5
    K_NORM = 6
    O_PROJ = 7
    POST_ATTENTION_NORM = 8
    PRE_FEEDFORWARD_NORM = 9
    GATE_PROJ = 10
    UP_PROJ = 11
    DOWN_PROJ = 12
    POST_FEEDFORWARD_NORM = 13
    LAYER_SCALAR = 14
    FINAL_NORM = 15
    PRE_FEEDFORWARD_NORM_2 = 16
    POST_FEEDFORWARD_NORM_1 = 17
    POST_FEEDFORWARD_NORM_2 = 18
    ROUTER_PROJ = 19
    ROUTER_SCALE = 20
    ROUTER_PER_EXPERT_SCALE = 21
    EXPERT_GATE_PROJ = 22
    EXPERT_UP_PROJ = 23
    EXPERT_DOWN_PROJ = 24


@dataclass(frozen=True)
class TensorSpec:
    physical_id: int
    role: Role
    layer: int
    expert: int
    shape: tuple[int, ...]
    source_name: str
    source_element_offset: int = 0

    @property
    def byte_length(self):
        return 2 * math.prod(self.shape)

    @property
    def name(self):
        return self.separate_source_name.removeprefix(PREFIX)

    @property
    def separate_source_name(self):
        if self.expert < 0:
            return self.source_name
        projection = self.role.name.removeprefix('EXPERT_').lower()
        return f'{PREFIX}layers.{self.layer}.experts.{self.expert}.{projection}.weight'


def native_tensor_specs() -> tuple[TensorSpec, ...]:
    """2-D expert views into stacked Google tensors; offsets are in elements.

    NVIDIA supplies the same logical entries under separate_source_name. No
    weight data is read or expanded to construct this metadata-only inventory.
    """
    shapes = source_shapes()
    specs = []

    def add(role, layer, source_name, *, expert=-1, shape=None, offset=0):
        specs.append(TensorSpec(len(specs), role, layer, expert,
                                shapes[source_name] if shape is None else shape,
                                source_name, offset))

    add(Role.EMBEDDING, -1, PREFIX + 'embed_tokens.weight')
    fields = (
        (Role.INPUT_NORM, 'input_layernorm.weight'),
        (Role.Q_PROJ, 'self_attn.q_proj.weight'),
        (Role.K_PROJ, 'self_attn.k_proj.weight'),
        (Role.V_PROJ, 'self_attn.v_proj.weight'),
        (Role.Q_NORM, 'self_attn.q_norm.weight'),
        (Role.K_NORM, 'self_attn.k_norm.weight'),
        (Role.O_PROJ, 'self_attn.o_proj.weight'),
        (Role.POST_ATTENTION_NORM, 'post_attention_layernorm.weight'),
        (Role.PRE_FEEDFORWARD_NORM, 'pre_feedforward_layernorm.weight'),
        (Role.GATE_PROJ, 'mlp.gate_proj.weight'),
        (Role.UP_PROJ, 'mlp.up_proj.weight'),
        (Role.DOWN_PROJ, 'mlp.down_proj.weight'),
        (Role.POST_FEEDFORWARD_NORM, 'post_feedforward_layernorm.weight'),
        (Role.LAYER_SCALAR, 'layer_scalar'),
        (Role.PRE_FEEDFORWARD_NORM_2, 'pre_feedforward_layernorm_2.weight'),
        (Role.POST_FEEDFORWARD_NORM_1, 'post_feedforward_layernorm_1.weight'),
        (Role.POST_FEEDFORWARD_NORM_2, 'post_feedforward_layernorm_2.weight'),
        (Role.ROUTER_PROJ, 'router.proj.weight'),
        (Role.ROUTER_SCALE, 'router.scale'),
        (Role.ROUTER_PER_EXPERT_SCALE, 'router.per_expert_scale'),
    )
    elements = 704 * 2816
    for layer in range(30):
        prefix = f'{PREFIX}layers.{layer}.'
        for role, suffix in fields:
            if role == Role.V_PROJ and layer % 6 == 5:
                continue
            add(role, layer, prefix + suffix)
        for expert in range(128):
            for half, role in enumerate((Role.EXPERT_GATE_PROJ, Role.EXPERT_UP_PROJ)):
                add(role, layer, prefix + 'experts.gate_up_proj', expert=expert,
                    shape=(704, 2816), offset=(2 * expert + half) * elements)
            add(Role.EXPERT_DOWN_PROJ, layer, prefix + 'experts.down_proj', expert=expert,
                shape=(2816, 704), offset=expert * elements)
    add(Role.FINAL_NORM, -1, PREFIX + 'norm.weight')
    return tuple(specs)


@dataclass(frozen=True)
class ComponentTensorSpec:
    # IDs are local to each standalone component.
    physical_id: int
    source_name: str
    shape: tuple[int, ...]

    @property
    def byte_length(self):
        return 2 * math.prod(self.shape)


def assistant_tensor_specs():
    shapes = {'model.embed_tokens.weight': (262144, 1024)}
    for layer in range(4):
        head = 512 if layer == 3 else 256
        fields = {
            'input_layernorm.weight': (1024,),
            'self_attn.q_proj.weight': (16 * head, 1024),
            'self_attn.q_norm.weight': (head,),
            'self_attn.o_proj.weight': (1024, 16 * head),
            'post_attention_layernorm.weight': (1024,),
            'pre_feedforward_layernorm.weight': (1024,),
            'mlp.gate_proj.weight': (8192, 1024),
            'mlp.up_proj.weight': (8192, 1024),
            'mlp.down_proj.weight': (1024, 8192),
            'post_feedforward_layernorm.weight': (1024,),
            'layer_scalar': (1,),
        }
        shapes.update({f'model.layers.{layer}.{name}': shape for name, shape in fields.items()})
    shapes.update({'model.norm.weight': (1024,), 'pre_projection.weight': (1024, 5632),
                   'post_projection.weight': (2816, 1024)})
    return tuple(ComponentTensorSpec(i, name, shape) for i, (name, shape) in enumerate(shapes.items()))


def vision_tensor_specs():
    shapes = {'model.vision_tower.patch_embedder.input_proj.weight': (1152, 768),
              'model.vision_tower.patch_embedder.position_embedding_table': (2, 10240, 1152)}
    for layer in range(27):
        fields = {'input_layernorm.weight': (1152,),
                  'self_attn.q_proj.linear.weight': (1152, 1152),
                  'self_attn.k_proj.linear.weight': (1152, 1152),
                  'self_attn.v_proj.linear.weight': (1152, 1152),
                  'self_attn.q_norm.weight': (72,), 'self_attn.k_norm.weight': (72,),
                  'self_attn.o_proj.linear.weight': (1152, 1152),
                  'post_attention_layernorm.weight': (1152,),
                  'pre_feedforward_layernorm.weight': (1152,),
                  'mlp.gate_proj.linear.weight': (4304, 1152),
                  'mlp.up_proj.linear.weight': (4304, 1152),
                  'mlp.down_proj.linear.weight': (1152, 4304),
                  'post_feedforward_layernorm.weight': (1152,)}
        shapes.update({f'model.vision_tower.encoder.layers.{layer}.{name}': shape
                       for name, shape in fields.items()})
    shapes.update({'model.vision_tower.std_bias': (1152,),
                   'model.vision_tower.std_scale': (1152,),
                   'model.embed_vision.embedding_projection.weight': (2816, 1152)})
    return tuple(ComponentTensorSpec(i, name, shape) for i, (name, shape) in enumerate(shapes.items()))
