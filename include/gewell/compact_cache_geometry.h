#pragma once
#include "gewell/kv_cache.h"

namespace gewell::kv_cache {
// Both supported Gemma4 decoders use 256-wide local heads and compact
// [Krot128,V512] global rows, with every sixth layer global.
struct CompactGeometry {
  std::uint32_t layers, local_heads, global_heads, hidden_width;
  constexpr std::uint32_t global_layers() const { return layers / 6; }
  constexpr std::uint32_t local_layers() const { return layers - global_layers(); }
};
PoolConfig compact_pool_config(CompactGeometry geometry, std::size_t gpu_bytes,
    std::size_t cpu_bytes = 0, std::size_t index_bytes = kDefaultIndexBytes,
    Format local = Format::bf16, Format global = Format::bf16);
}  // namespace gewell::kv_cache
