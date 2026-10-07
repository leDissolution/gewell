#include "gewell/compact_cache_geometry.h"

#include "gewell/compact_global_cache.h"
#include <stdexcept>


namespace gewell::kv_cache {

kv_cache::PoolConfig compact_pool_config(CompactGeometry geometry, std::size_t gpu_bytes,
                                    std::size_t cpu_bytes,
                                    std::size_t index_bytes,
                                    kv_cache::Format local, kv_cache::Format global) {
  // Every sixth Gemma layer is global. A compact global row stores the 128
  // position-dependent K elements followed by the complete V vector. Local
  // rows retain both K and V.
  if (!geometry.layers || geometry.layers % 6 || !geometry.local_heads ||
      !geometry.global_heads || !geometry.hidden_width)
    throw std::invalid_argument("invalid compact cache geometry");

  kv_cache::PoolConfig config;
  config.local_format = local;
  config.global_format = global;
  config.global_page_tokens = 256;
  config.local_window_tokens = 1024;
  config.maximum_context_tokens = 262144;
  config.gpu_bytes = gpu_bytes;
  config.cpu_bytes = cpu_bytes;
  config.index_bytes = index_bytes;
  config.global_page_bytes =
      geometry.global_layers() * geometry.global_heads *
      256 * kv_cache::row_bytes(compact_global_cache::kRowElements, global, 2);
  config.local_ring_bytes =
      geometry.local_layers() * geometry.local_heads *
      1024 * 2 * kv_cache::row_bytes(256, local);
  config.local_bytes_per_token =
      geometry.local_layers() * geometry.local_heads *
      2 * kv_cache::row_bytes(256, local);
  config.terminal_hidden_bytes =
      geometry.hidden_width * 2;
  config.page_table_bytes =
      ((static_cast<std::size_t>(262144) +
        256 - 1) /
       256) *
      sizeof(std::uint64_t);
  config.validate();
  return config;
}

}  // namespace gewell::kv_cache
