#include "cache_config.h"
namespace gewell::gemma4_31b::sm120 {
kv_cache::PoolConfig compact_pool_config(std::size_t gpu, std::size_t cpu,
    std::size_t index, kv_cache::Format local, kv_cache::Format global) {
  return kv_cache::compact_pool_config(kCacheGeometry,gpu,cpu,index,local,global);
}
}
