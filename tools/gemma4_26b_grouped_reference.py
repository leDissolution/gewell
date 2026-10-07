"""Independent Torch expression of grouped BF16 arithmetic through256 rows.

Uses source weights and tokens only. No native outputs or kernels are imported.
This control specifies changed reduction/cast semantics; the pinned eager
captures remain the separate evidence of differences from upstream arithmetic.
"""
import torch
from transformers.models.gemma4 import modeling_gemma4 as model


def projection(inputs, weights):
    count, width, inner = weights.shape
    assert inner % 32 == 0
    sums = torch.zeros(count, width, 32, device=inputs.device, dtype=torch.float32)
    # BF16 products are exactly representable in FP32. Eager separate operations
    # preserve each FP32 addition instead of introducing a fused reduction.
    for start in range(0, inner, 32):
        products = inputs[:, None, start:start+32].float() * weights[:, :, start:start+32].float()
        sums = sums + products
    for half in (16, 8, 4, 2, 1):
        sums = sums[:, :, :half] + sums[:, :, half:2*half]
    return sums[:, :, 0].bfloat16()


def router(self, hidden_states):
    normalized = self.norm(hidden_states) * self.scale * self.scalar_root_size
    probabilities = torch.softmax(self.proj(normalized), dim=-1, dtype=torch.float32)
    ids = torch.argsort(probabilities, dim=-1, descending=True, stable=True)[..., :8]
    weights = probabilities.gather(-1, ids)
    weights = weights / weights.sum(-1, keepdim=True) * self.per_expert_scale[ids]
    return probabilities, weights, ids


def small_experts(self, hidden_states, top_k_index, top_k_weights):
    assert hidden_states.ndim == 2 and 1 <= hidden_states.shape[0] <= 32
    ids, order = top_k_index.sort(-1)
    weights = top_k_weights.gather(-1, order)
    inputs = hidden_states.repeat_interleave(ids.shape[1], 0)
    gate, up = projection(inputs, self.gate_up_proj[ids.flatten()]).chunk(2, -1)
    activated = self.act_fn(gate) * up
    down = projection(activated, self.down_proj[ids.flatten()]).reshape(*ids.shape, -1)
    result = torch.zeros_like(hidden_states)
    for rank in range(ids.shape[1]):
        result = result + (down[:, rank] * weights[:, rank, None]).bfloat16()
    return result


def experts(self, hidden_states, top_k_index, top_k_weights):
    if hidden_states.shape[0] <= 32:
        return small_experts(self, hidden_states, top_k_index, top_k_weights)
    assert hidden_states.ndim == 2 and hidden_states.shape[0] <= 256
    result = torch.zeros_like(hidden_states)
    # This reference-only padding selects the pinned cuBLAS accumulation shape
    # validated against WMMA. Production processes only selected assignments.
    for expert in torch.unique(top_k_index, sorted=True):
        token, rank = torch.where(top_k_index == expert)
        inputs = torch.zeros(1024, 2816, device=hidden_states.device, dtype=torch.bfloat16)
        inputs[:len(token)] = hidden_states[token]
        gate, up = torch.nn.functional.linear(inputs, self.gate_up_proj[expert]).chunk(2, -1)
        activated = self.act_fn(gate) * up
        output = torch.nn.functional.linear(activated, self.down_proj[expert])[:len(token)]
        result[token] += (output * top_k_weights[token, rank, None]).bfloat16()
    return result


def install():
    original_attention = model.eager_attention_forward

    def attention(module, query, key, value, attention_mask, **kwargs):
        if not isinstance(module, model.Gemma4TextAttention):
            return original_attention(module, query, key, value, attention_mask, **kwargs)
        assert query.shape[0] == 1 and 1 <= query.shape[-2] <= 256 and key.shape[-2] <= 256
        q, k, v = query[0], key[0], value[0]
        heads, rows, width = q.shape
        kvheads, length, _ = k.shape
        repeat = heads // kvheads
        if not module.is_sliding:
            reconstructed = (v.float() * module.k_norm.weight.float()).bfloat16()
            reconstructed[:, :, :64] = k[:, :, :64]
            reconstructed[:, :, 256:320] = k[:, :, 256:320]
            k = reconstructed
        padded = (length + 7) // 8 * 8
        keys = torch.zeros(kvheads, padded, width, device=q.device, dtype=q.dtype)
        values = torch.zeros_like(keys)
        keys[:, :length], values[:, :length] = k, v
        scores = torch.bmm(q.reshape(kvheads, repeat*rows, width), keys.transpose(1, 2),
                           out_dtype=torch.float32).reshape(heads, rows, padded)
        positions = length - rows + torch.arange(rows, device=q.device)[:, None]
        key_positions = torch.arange(padded, device=q.device)[None, :]
        visible = (key_positions <= positions) & (key_positions < length)
        if module.is_sliding:
            visible &= positions - key_positions < 1024
        scores.masked_fill_(~visible, -torch.inf)
        weights = (scores - scores.amax(-1, keepdim=True)).exp().bfloat16()
        numerator = torch.bmm(weights.reshape(kvheads, repeat*rows, padded), values,
                              out_dtype=torch.float32).reshape(heads, rows, width)
        output = (numerator / weights.float().sum(-1, keepdim=True)).bfloat16()
        return output.transpose(0, 1).contiguous().unsqueeze(0), None

    model.Gemma4TextRouter.forward = router
    model.Gemma4TextExperts.forward = experts
    model.eager_attention_forward = attention
