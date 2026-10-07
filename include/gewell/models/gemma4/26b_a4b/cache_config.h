#pragma once
#include "gewell/compact_cache_geometry.h"
#include "gewell/models/gemma4/26b_a4b/model.h"

namespace gewell::gemma4_26b_a4b::sm120 {
inline constexpr kv_cache::CompactGeometry kCacheGeometry{
    kLayerCount, kLocalKvHeadCount, kGlobalKvHeadCount, kHiddenSize};
}
