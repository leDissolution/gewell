#pragma once

#include "gewell/kv_format.h"


#include <cstddef>
#include <cstdint>

namespace gewell::compact_global_cache {

inline constexpr std::uint32_t kRotatedKeyElements = 128;
inline constexpr std::uint32_t kRowElements =
    kRotatedKeyElements + 512;

template <typename Element>
struct PagedView {
  // Offsets/strides are 16-bit storage words in either format. FP8 page and
  // layer offsets must keep records 16-byte aligned, as in the physical pool.
  Element* page_pool{};
  const std::uint64_t* page_offsets{};
  std::uint32_t page_tokens{};
  std::uint32_t page_count{};
  std::size_t page_stride_elements{};
  std::size_t layer_offset_elements{};
  kv_cache::Format format{kv_cache::Format::bf16};
};

#ifdef __CUDACC__
template <typename Element>
__device__ __forceinline__ Element* paged_row(
    Element* page_pool, const std::uint64_t* page_offsets,
    std::uint32_t page_tokens, std::size_t layer_offset_elements,
    std::uint32_t kv_head, std::uint32_t absolute_position,
    kv_cache::Format format = kv_cache::Format::bf16) {
  const std::uint32_t page_index = absolute_position / page_tokens;
  const std::uint32_t page_position = absolute_position % page_tokens;
  return page_pool + page_offsets[page_index] + layer_offset_elements +
         (static_cast<std::size_t>(kv_head) * page_tokens + page_position) *
             kv_cache::row_words(kRowElements, format, 2);
}
#endif

}  // namespace gewell::compact_global_cache
