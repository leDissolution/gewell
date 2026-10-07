#include "physical_cache.h"
#include "gewell/compact_global_cache.h"
#include <algorithm>

namespace gewell::runtime {
using cuda_detail::check_cuda;
using cuda_detail::fail;
using cuda_detail::PinnedHostAllocation;
using BFloat16 = __nv_bfloat16;
PhysicalCache::PhysicalCache(kv_cache::CompactGeometry geometry, const kv_cache::PoolConfig& config,
                             kv_cache::CacheLedger& ledger,
                             std::size_t page_offsets_count)
    : geometry_(geometry), config_(config), ledger_(ledger), page_offsets_count_(page_offsets_count),
      pool_(config_.gpu_bytes),
      cpu_pool_(config_.cpu_bytes == 0
          ? nullptr : std::make_unique<PinnedHostAllocation>(config_.cpu_bytes)),
      page_offsets_host_(std::make_unique<std::uint64_t[]>(page_offsets_count_)) {}

void* PhysicalCache::host_pointer(const kv_cache::Allocation& allocation) const {
  if (cpu_pool_ == nullptr || !allocation.valid() ||
      allocation.tier != kv_cache::Tier::cpu ||
      !ledger_.cpu_pool().owns(allocation) || allocation.offset > cpu_pool_->size() ||
      allocation.bytes > cpu_pool_->size() - allocation.offset) {
    fail("persistent KV allocation", "allocation is outside the CPU pool");
  }
  return static_cast<std::uint8_t*>(cpu_pool_->data()) + allocation.offset;
}

std::size_t PhysicalCache::global_layer_elements() const {
  return static_cast<std::size_t>(geometry_.global_heads) *
         config_.global_page_tokens *
         kv_cache::row_words(compact_global_cache::kRowElements, config_.global_format, 2);
}

std::size_t PhysicalCache::local_layer_bytes() const {
  return config_.local_ring_bytes / geometry_.local_layers();
}

void* PhysicalCache::device_pointer(const kv_cache::Allocation& allocation) const {
  if (!allocation.valid() || allocation.tier != kv_cache::Tier::gpu ||
      !ledger_.gpu_pool().owns(allocation) ||
      allocation.offset > pool_.size() ||
      allocation.bytes > pool_.size() - allocation.offset) {
    fail("persistent KV allocation", "allocation is outside the GPU pool");
  }
  return static_cast<std::uint8_t*>(pool_.data()) + allocation.offset;
}

void PhysicalCache::upload_page_table(const kv_cache::ExecutionInfo& info,
                       runtime::CompletionContext completion) {
  const auto stream = static_cast<cudaStream_t>(completion.value);
  if (info.global_pages.size() > page_offsets_count_) {
    fail("persistent KV page table", "page count exceeds page-table capacity");
  }
  std::fill_n(page_offsets_host_.get(), page_offsets_count_, 0);
  for (std::size_t index = 0; index < info.global_pages.size(); ++index) {
    const kv_cache::Allocation& allocation =
        ledger_.page(info.global_pages[index]).storage;
    if (allocation.offset % sizeof(BFloat16) != 0) {
      fail("persistent KV page table", "page offset is not BF16 aligned");
    }
    page_offsets_host_[index] = allocation.offset / sizeof(BFloat16);
  }
  check_cuda(cudaMemcpyAsync(
                 device_pointer(info.page_table), page_offsets_host_.get(),
                 info.page_table.bytes, cudaMemcpyHostToDevice, stream),
             "upload persistent KV page table");
  check_cuda(cudaStreamSynchronize(stream),
             "synchronize persistent KV page table");
}

void PhysicalCache::copy_local_to_snapshot(const kv_cache::ExecutionInfo& source,
                            std::uint32_t local_tokens,
                            const kv_cache::Allocation& destination,
                            runtime::CompletionContext completion) {
  const auto stream = static_cast<cudaStream_t>(completion.value);
  // Each layer/kind snapshot packs [head][chronological token][dimension].
  // The live cache instead has a full circular token window within each head.
  if (local_tokens == 0) {
    return;
  }
  const std::size_t destination_layer_bytes =
      static_cast<std::size_t>(local_tokens) * config_.local_bytes_per_token /
      geometry_.local_layers();
  const std::size_t per_kind_destination_bytes = destination_layer_bytes / 2;
  const std::size_t source_layer_bytes = local_layer_bytes();
  const std::size_t per_kind_source_bytes = source_layer_bytes / 2;
  const std::uint32_t local_start = source.processed_tokens - local_tokens;
  const auto* const source_base =
      static_cast<const std::uint8_t*>(pool_.data()) + source.local_ring.offset;
  auto* const destination_base =
      static_cast<std::uint8_t*>(device_pointer(destination));
  for (std::size_t layer = 0; layer < geometry_.local_layers(); ++layer) {
    for (std::size_t kind = 0; kind < 2; ++kind) {
      const auto* source_kind =
          source_base + layer * source_layer_bytes +
          kind * per_kind_source_bytes;
      auto* destination_kind =
          destination_base + layer * destination_layer_bytes +
          kind * per_kind_destination_bytes;
      copy_ring_to_linear(source_kind, destination_kind, local_start,
                          local_tokens, per_kind_source_bytes,
                          per_kind_destination_bytes, stream);
    }
  }
}

void PhysicalCache::copy_ring_to_linear(const std::uint8_t* source,
                         std::uint8_t* destination,
                         std::uint32_t absolute_start,
                         std::uint32_t token_count,
                         std::size_t source_kind_bytes,
                         std::size_t destination_kind_bytes,
                         cudaStream_t stream) const {
  const std::size_t heads = geometry_.local_heads;
  const std::size_t row_bytes = kv_cache::row_bytes(256, config_.local_format);
  const std::size_t source_pitch = config_.local_window_tokens * row_bytes;
  const std::size_t destination_pitch = token_count * row_bytes;
  if (source_kind_bytes != heads * source_pitch ||
      destination_kind_bytes != heads * destination_pitch) {
    fail("persistent KV snapshot", "local row size mismatch");
  }
  const std::size_t first_slot =
      absolute_start % config_.local_window_tokens;
  const std::size_t first_rows =
      std::min<std::size_t>(token_count,
                            config_.local_window_tokens - first_slot);
  check_cuda(cudaMemcpy2DAsync(
                 destination, destination_pitch,
                 source + first_slot * row_bytes, source_pitch,
                 first_rows * row_bytes, heads, cudaMemcpyDeviceToDevice, stream),
             "copy local KV snapshot prefix");
  if (first_rows < token_count) {
    check_cuda(cudaMemcpy2DAsync(
                   destination + first_rows * row_bytes, destination_pitch,
                   source, source_pitch, (token_count - first_rows) * row_bytes,
                   heads, cudaMemcpyDeviceToDevice, stream),
               "copy local KV snapshot wrapped suffix");
  }
}

void PhysicalCache::copy_linear_to_ring(const std::uint8_t* source,
                         std::uint8_t* destination,
                         std::uint32_t absolute_start,
                         std::uint32_t token_count,
                         std::size_t source_kind_bytes,
                         std::size_t destination_kind_bytes,
                         cudaStream_t stream) const {
  const std::size_t heads = geometry_.local_heads;
  const std::size_t row_bytes = kv_cache::row_bytes(256, config_.local_format);
  const std::size_t source_pitch = token_count * row_bytes;
  const std::size_t destination_pitch = config_.local_window_tokens * row_bytes;
  if (source_kind_bytes != heads * source_pitch ||
      destination_kind_bytes != heads * destination_pitch) {
    fail("persistent KV restore", "local row size mismatch");
  }
  const std::size_t first_slot =
      absolute_start % config_.local_window_tokens;
  const std::size_t first_rows =
      std::min<std::size_t>(token_count,
                            config_.local_window_tokens - first_slot);
  check_cuda(cudaMemcpy2DAsync(
                 destination + first_slot * row_bytes, destination_pitch,
                 source, source_pitch, first_rows * row_bytes, heads,
                 cudaMemcpyDeviceToDevice, stream),
             "restore local KV snapshot prefix");
  if (first_rows < token_count) {
    check_cuda(cudaMemcpy2DAsync(
                   destination, destination_pitch,
                   source + first_rows * row_bytes, source_pitch,
                   (token_count - first_rows) * row_bytes, heads,
                   cudaMemcpyDeviceToDevice, stream),
               "restore local KV snapshot wrapped suffix");
  }
}

void PhysicalCache::copy_host_linear_to_ring(const std::uint8_t* source,
                              std::uint8_t* destination,
                              std::uint32_t absolute_start,
                              std::uint32_t token_count,
                              std::size_t source_kind_bytes,
                              std::size_t destination_kind_bytes,
                              cudaStream_t stream) const {
  const std::size_t heads = geometry_.local_heads;
  const std::size_t row_bytes = kv_cache::row_bytes(256, config_.local_format);
  const std::size_t source_pitch = token_count * row_bytes;
  const std::size_t destination_pitch = config_.local_window_tokens * row_bytes;
  if (source_kind_bytes != heads * source_pitch ||
      destination_kind_bytes != heads * destination_pitch) {
    fail("persistent KV restore", "local row size mismatch");
  }
  const std::size_t first_slot =
      absolute_start % config_.local_window_tokens;
  const std::size_t first_rows =
      std::min<std::size_t>(token_count,
                            config_.local_window_tokens - first_slot);
  check_cuda(cudaMemcpy2DAsync(
                 destination + first_slot * row_bytes, destination_pitch,
                 source, source_pitch, first_rows * row_bytes, heads,
                 cudaMemcpyHostToDevice, stream),
             "restore cold local KV snapshot prefix");
  if (first_rows < token_count) {
    check_cuda(cudaMemcpy2DAsync(
                   destination, destination_pitch,
                   source + first_rows * row_bytes, source_pitch,
                   (token_count - first_rows) * row_bytes, heads,
                   cudaMemcpyHostToDevice, stream),
               "restore cold local KV snapshot wrapped suffix");
  }
}

kv_cache::DeviceView PhysicalCache::layer(kv_cache::ExecutionId execution,
                                   std::uint32_t layer_index) const {
  if (layer_index >= geometry_.layers) {
    fail("persistent KV cache view", "layer is outside the model");
  }
  const kv_cache::ExecutionInfo& info = ledger_.execution(execution);
  if ((layer_index % 6 == 5)) {
    const std::size_t slot = layer_index / 6;
    const std::size_t layer_elements = global_layer_elements();
    return {
        nullptr,
        nullptr,
        config_.maximum_context_tokens,
        static_cast<BFloat16*>(pool_.data()),
        static_cast<std::uint64_t*>(device_pointer(info.page_table)),
        config_.global_page_tokens,
        static_cast<std::uint32_t>(info.global_pages.size()),
        config_.global_page_bytes / sizeof(BFloat16),
        slot * layer_elements,
        config_.global_format,
    };
  }
  const std::size_t slot = layer_index - layer_index / 6;
  const std::size_t layer_bytes = local_layer_bytes();
  auto* const layer_base = static_cast<std::uint8_t*>(pool_.data()) +
                           info.local_ring.offset + slot * layer_bytes;
  const std::size_t per_kind_bytes = layer_bytes / 2;
  return {
      reinterpret_cast<BFloat16*>(layer_base),
      reinterpret_cast<BFloat16*>(layer_base + per_kind_bytes),
      config_.local_window_tokens,
      nullptr, nullptr, 0, 0, 0, 0, config_.local_format,
  };
}

void PhysicalCache::restore_local(const kv_cache::ExecutionInfo& destination,
                   const kv_cache::Allocation& snapshot,
                   std::uint32_t local_start, std::uint32_t local_tokens,
                   runtime::CompletionContext completion) {
  const auto stream = static_cast<cudaStream_t>(completion.value);
  if (local_tokens == 0) {
    return;
  }
  const std::size_t source_layer_bytes =
      static_cast<std::size_t>(local_tokens) * config_.local_bytes_per_token /
      geometry_.local_layers();
  const std::size_t per_kind_source_bytes = source_layer_bytes / 2;
  const std::size_t destination_layer_bytes = local_layer_bytes();
  const std::size_t per_kind_destination_bytes = destination_layer_bytes / 2;
  auto* const destination_base =
      static_cast<std::uint8_t*>(pool_.data()) + destination.local_ring.offset;
  const auto* const source_base =
      static_cast<const std::uint8_t*>(snapshot.tier == kv_cache::Tier::cpu
          ? host_pointer(snapshot) : device_pointer(snapshot));
  for (std::size_t layer = 0; layer < geometry_.local_layers(); ++layer) {
    for (std::size_t kind = 0; kind < 2; ++kind) {
      const auto* source_kind =
          source_base + layer * source_layer_bytes +
          kind * per_kind_source_bytes;
      auto* destination_kind =
          destination_base + layer * destination_layer_bytes +
          kind * per_kind_destination_bytes;
      if (snapshot.tier == kv_cache::Tier::cpu) {
        copy_host_linear_to_ring(source_kind, destination_kind, local_start,
                                 local_tokens, per_kind_source_bytes,
                                 per_kind_destination_bytes, stream);
      } else {
        copy_linear_to_ring(source_kind, destination_kind, local_start,
                            local_tokens, per_kind_source_bytes,
                            per_kind_destination_bytes, stream);
      }
    }
  }
}

void PhysicalCache::fork_local(const kv_cache::ExecutionInfo& from,
                               const kv_cache::ExecutionInfo& to,
                               runtime::CompletionContext completion) {
  const auto stream = static_cast<cudaStream_t>(completion.value);
  const auto rows = std::min(from.processed_tokens, config_.local_window_tokens);
  const auto start = from.processed_tokens - rows;
  const auto slot = start % config_.local_window_tokens;
  const auto first = std::min(rows, config_.local_window_tokens - slot);
  const auto kind_bytes = local_layer_bytes() / 2;
  const auto row_bytes = kv_cache::row_bytes(256, config_.local_format);
  const auto head_bytes = config_.local_window_tokens * row_bytes;
  for (std::size_t kind = 0; kind < 2 * geometry_.local_layers(); ++kind) {
    auto* src = static_cast<const std::uint8_t*>(device_pointer(from.local_ring)) + kind * kind_bytes;
    auto* dst = static_cast<std::uint8_t*>(device_pointer(to.local_ring)) + kind * kind_bytes;
    check_cuda(cudaMemcpy2DAsync(dst + slot * row_bytes, head_bytes,
        src + slot * row_bytes, head_bytes, first * row_bytes,
        geometry_.local_heads, cudaMemcpyDeviceToDevice, stream), "fork local KV prefix");
    if (first < rows)
      check_cuda(cudaMemcpy2DAsync(dst, head_bytes, src, head_bytes,
          (rows - first) * row_bytes, geometry_.local_heads,
          cudaMemcpyDeviceToDevice, stream), "fork wrapped local KV prefix");
  }
}

void PhysicalCache::clear_page_table(const kv_cache::ExecutionInfo& info) {
  check_cuda(cudaMemset(device_pointer(info.page_table), 0, info.page_table.bytes),
             "clear persistent KV page table");
}

void PhysicalCache::restore_terminal(const kv_cache::Allocation& source,
                                     runtime::TerminalState terminal_destination,
                                     runtime::CompletionContext completion) const {
  const auto stream = static_cast<cudaStream_t>(completion.value);
  auto* destination = static_cast<BFloat16*>(terminal_destination.value);
  if (source.tier == kv_cache::Tier::gpu) {
    check_cuda(cudaMemcpyAsync(destination, device_pointer(source),
        config_.terminal_hidden_bytes, cudaMemcpyDeviceToDevice, stream),
        "restore persistent terminal hidden state");
  } else {
    check_cuda(cudaMemcpyAsync(destination, host_pointer(source),
        config_.terminal_hidden_bytes, cudaMemcpyHostToDevice, stream),
        "restore persistent cold terminal hidden state");
  }
}

void PhysicalCache::snapshot_terminal(runtime::TerminalState terminal_source,
                                      const kv_cache::Allocation& destination,
                                      runtime::CompletionContext completion,
                                      std::string_view operation) {
  const auto stream = static_cast<cudaStream_t>(completion.value);
  const auto* source = static_cast<const BFloat16*>(terminal_source.value);
  check_cuda(cudaMemcpyAsync(device_pointer(destination), source,
      config_.terminal_hidden_bytes, cudaMemcpyDeviceToDevice, stream), operation);
}

void PhysicalCache::copy_global_page(const kv_cache::Allocation& source,
                                     const kv_cache::Allocation& destination,
                                     runtime::CompletionContext completion) {
  const auto stream = static_cast<cudaStream_t>(completion.value);
  check_cuda(cudaMemcpyAsync(device_pointer(destination), device_pointer(source),
      config_.global_page_bytes, cudaMemcpyDeviceToDevice, stream),
      "copy persistent KV shared page");
}

void PhysicalCache::restore_global_page(const kv_cache::Allocation& source,
                                        const kv_cache::Allocation& destination,
                                        runtime::CompletionContext completion) {
  const auto stream = static_cast<cudaStream_t>(completion.value);
  check_cuda(cudaMemcpyAsync(device_pointer(destination), host_pointer(source),
      config_.global_page_bytes, cudaMemcpyHostToDevice, stream),
      "restore persistent cold KV page");
}

void PhysicalCache::spill(const kv_cache::Allocation& source,
                          const kv_cache::Allocation& destination,
                          std::size_t bytes, std::string_view operation) {
  check_cuda(cudaMemcpy(host_pointer(destination), device_pointer(source),
                       bytes, cudaMemcpyDeviceToHost), operation);
}

void PhysicalCache::synchronize(runtime::CompletionContext completion,
                                std::string_view operation) const {
  const auto stream = static_cast<cudaStream_t>(completion.value);
  check_cuda(cudaStreamSynchronize(stream), operation);
}

void PhysicalCache::wait(runtime::CompletionContext completion) const noexcept {
  const auto stream = static_cast<cudaStream_t>(completion.value);
  cudaStreamSynchronize(stream);
}

}  // namespace gewell::runtime
