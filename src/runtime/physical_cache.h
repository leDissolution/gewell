#pragma once

#include "gewell/kv_cache.h"
#include "gewell/runtime/cache_storage.h"
#include "gewell/kv_view.h"
#include "gewell/compact_cache_geometry.h"
#include "cuda_memory.cuh"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

namespace gewell::runtime {

// Physical storage for the supported compact-global caches. The caller's ledger owns
// every allocation and decides reservation, retention and publication order.
class PhysicalCache final : public runtime::CacheStorage {
 public:
  PhysicalCache(kv_cache::CompactGeometry geometry, const kv_cache::PoolConfig& config, kv_cache::CacheLedger& ledger,
                std::size_t page_offsets_count);
  PhysicalCache(const PhysicalCache&) = delete;
  PhysicalCache& operator=(const PhysicalCache&) = delete;

  [[nodiscard]] const kv_cache::PoolConfig& config() const { return config_; }
  [[nodiscard]] bool has_cpu_pool() const { return cpu_pool_ != nullptr; }
  [[nodiscard]] void* device_pointer(const kv_cache::Allocation& allocation) const;
  [[nodiscard]] kv_cache::DeviceView layer(kv_cache::ExecutionId execution,
                                          std::uint32_t layer_index) const;
  void clear_page_table(const kv_cache::ExecutionInfo& info) override;
  void upload_page_table(const kv_cache::ExecutionInfo& info, runtime::CompletionContext completion) override;
  void fork_local(const kv_cache::ExecutionInfo& from,
                  const kv_cache::ExecutionInfo& to, runtime::CompletionContext completion) override;
  void restore_local(const kv_cache::ExecutionInfo& destination,
                     const kv_cache::Allocation& snapshot,
                     std::uint32_t local_start, std::uint32_t local_tokens,
                     runtime::CompletionContext completion = {}) override;
  void copy_local_to_snapshot(const kv_cache::ExecutionInfo& source,
                              std::uint32_t local_tokens,
                              const kv_cache::Allocation& destination,
                              runtime::CompletionContext completion) override;
  void restore_terminal(const kv_cache::Allocation& source, runtime::TerminalState destination,
                        runtime::CompletionContext completion) const override;
  void snapshot_terminal(runtime::TerminalState source,
                         const kv_cache::Allocation& destination,
                         runtime::CompletionContext completion, std::string_view operation) override;
  void copy_global_page(const kv_cache::Allocation& source,
                        const kv_cache::Allocation& destination,
                        runtime::CompletionContext completion) override;
  void restore_global_page(const kv_cache::Allocation& source,
                           const kv_cache::Allocation& destination,
                           runtime::CompletionContext completion) override;
  void spill(const kv_cache::Allocation& source,
             const kv_cache::Allocation& destination, std::size_t bytes,
             std::string_view operation) override;
  void synchronize(runtime::CompletionContext completion, std::string_view operation) const override;
  void wait(runtime::CompletionContext completion) const noexcept override;

 private:
  [[nodiscard]] void* host_pointer(const kv_cache::Allocation& allocation) const;
  [[nodiscard]] std::size_t global_layer_elements() const;
  [[nodiscard]] std::size_t local_layer_bytes() const;
  void copy_ring_to_linear(const std::uint8_t* source, std::uint8_t* destination,
                           std::uint32_t absolute_start, std::uint32_t token_count,
                           std::size_t source_kind_bytes,
                           std::size_t destination_kind_bytes,
                           cudaStream_t stream) const;
  void copy_linear_to_ring(const std::uint8_t* source, std::uint8_t* destination,
                           std::uint32_t absolute_start, std::uint32_t token_count,
                           std::size_t source_kind_bytes,
                           std::size_t destination_kind_bytes,
                           cudaStream_t stream) const;
  void copy_host_linear_to_ring(const std::uint8_t* source,
                                std::uint8_t* destination,
                                std::uint32_t absolute_start,
                                std::uint32_t token_count,
                                std::size_t source_kind_bytes,
                                std::size_t destination_kind_bytes,
                                cudaStream_t stream) const;

  const kv_cache::CompactGeometry geometry_;
  const kv_cache::PoolConfig& config_;
  kv_cache::CacheLedger& ledger_;
  const std::size_t page_offsets_count_;
  cuda_detail::DeviceAllocation pool_;
  std::unique_ptr<cuda_detail::PinnedHostAllocation> cpu_pool_;
  std::unique_ptr<std::uint64_t[]> page_offsets_host_;
};

}  // namespace gewell::runtime
