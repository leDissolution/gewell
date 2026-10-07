#pragma once

#include "gewell/compact_cache_geometry.h"
#include "gewell/models/gemma4/31b/model.h"

namespace gewell::gemma4_31b::sm120 {

inline constexpr kv_cache::CompactGeometry kCacheGeometry{
    gemma4_31b::kLayerCount, gemma4_31b::kLocalKvHeadCount,
    gemma4_31b::kGlobalKvHeadCount, gemma4_31b::kHiddenSize};

inline constexpr std::uint32_t kGlobalPageTokens = 256;
inline constexpr std::uint32_t kLocalWindowTokens = 1'024;
inline constexpr std::uint32_t kMaximumContextTokens = 262'144;

// The ledger accounts for these explicit physical sizes without model knowledge.
kv_cache::PoolConfig compact_pool_config(
    std::size_t gpu_bytes, std::size_t cpu_bytes = 0,
    std::size_t index_bytes = kv_cache::kDefaultIndexBytes,
    kv_cache::Format local = kv_cache::Format::bf16,
    kv_cache::Format global = kv_cache::Format::bf16);

}  // namespace gewell::gemma4_31b::sm120
