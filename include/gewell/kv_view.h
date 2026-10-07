#pragma once
#include "gewell/kv_format.h"
#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>

namespace gewell::kv_cache {
// The serving backend's concrete compact-global cache: local K and V are
// separate head-major rings; global rows are [Krot128,V512], contiguous or paged.
struct DeviceView {
  __nv_bfloat16* key{};
  __nv_bfloat16* value{};
  std::uint32_t capacity{};
  __nv_bfloat16* page_pool{};
  std::uint64_t* page_offsets{};
  std::uint32_t page_tokens{};
  std::uint32_t page_count{};
  std::size_t page_stride_elements{};
  std::size_t layer_offset_elements{};
  kv_cache::Format format{kv_cache::Format::bf16};
};
}  // namespace gewell::kv_cache
