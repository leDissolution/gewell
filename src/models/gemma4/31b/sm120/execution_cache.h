#pragma once

#include "runtime/physical_cache.h"
#include "execution_types.h"
#include "gewell/runtime/cache.h"
#include <functional>

namespace gewell::gemma4_31b::sm120 {

// The executor borrows physical state and invokes lifecycle callbacks only at
// work boundaries. Layer access stays concrete; policy retains reservations.
struct ExecutionCache {
  explicit ExecutionCache(runtime::PersistentCacheManager& manager)
      : state(static_cast<runtime::PhysicalCache&>(manager.storage())),
        prepare_write([&manager](kv_cache::ExecutionId id, std::uint32_t position,
                                  std::uint32_t rows, cudaStream_t stream) {
          manager.prepare_write(id, position, rows, runtime::CompletionContext{stream});
        }),
        restore_terminal_hidden([&manager](kv_cache::ExecutionId id,
                                            BFloat16* destination, cudaStream_t stream) {
          manager.restore_terminal_hidden(id, runtime::TerminalState{destination},
                                          runtime::CompletionContext{stream});
        }),
        acquire_speculative_staging([&manager, this](kv_cache::ExecutionId id,
                                                     std::size_t bytes) {
          const auto allocation = manager.acquire_speculative_staging(id, bytes);
          return allocation.valid() ? state.device_pointer(allocation) : nullptr;
        }),
        processed_tokens([&manager](kv_cache::ExecutionId id) {
          return manager.processed_tokens(id);
        }) {}

  runtime::PhysicalCache& state;
  std::function<void(kv_cache::ExecutionId, std::uint32_t, std::uint32_t, cudaStream_t)> prepare_write;
  std::function<void(kv_cache::ExecutionId, BFloat16*, cudaStream_t)> restore_terminal_hidden;
  std::function<void*(kv_cache::ExecutionId, std::size_t)> acquire_speculative_staging;
  std::function<std::uint32_t(kv_cache::ExecutionId)> processed_tokens;
  const kv_cache::PoolConfig& config() const { return state.config(); }
  LayerCacheView layer(kv_cache::ExecutionId execution, std::uint32_t index) const {
    return state.layer(execution, index);
  }
};

}  // namespace gewell::gemma4_31b::sm120
