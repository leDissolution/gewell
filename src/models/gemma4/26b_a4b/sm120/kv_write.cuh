#pragma once
#include "gewell/kv_view.h"
#include <cuda_runtime.h>
#include <vector>

namespace gewell::gemma4_26b_a4b::sm120 {
// Validate storage for a visible/reserved prefix before enqueuing work.
// An empty global history needs no page allocation.
void validate_cache_view(const kv_cache::DeviceView& cache, bool global, unsigned end);

// Inputs are [rows, kv_heads, head_dimension], including RoPE on K.
// Write after attention: a prefill chunk may overwrite old local history.
// Only the final 1024 rows of a longer chunk enter the local ring.
// The ledger must reserve/upload every global page before this call.
void write_cache(const __nv_bfloat16* key, const __nv_bfloat16* value,
                 kv_cache::DeviceView cache, bool global,
                 unsigned position, unsigned rows, cudaStream_t stream = nullptr);

struct CacheWriteInput {
  const __nv_bfloat16 *key{}, *value{};
  kv_cache::DeviceView cache{};
  unsigned position{}, rows{};
};
// Coalesce independent writable views, preserving each input's row retention
// and storage format. Validate the whole batch before enqueuing any writes.
void write_cache_batch(const std::vector<CacheWriteInput>& inputs, bool global,
                       cudaStream_t stream = nullptr);
}  // namespace gewell::gemma4_26b_a4b::sm120
