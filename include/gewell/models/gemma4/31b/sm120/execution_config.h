#pragma once
#include <cstdint>

namespace gewell::gemma4_31b::sm120 {

inline constexpr std::uint32_t kDefaultPrefillChunkTokens = 1024;
inline constexpr std::uint32_t kMaxPrefillChunkTokens = 4096;
inline constexpr std::uint32_t kDefaultPrefillBatchTokens = 2048;
inline constexpr std::uint32_t kMaxPrefillBatchTokens = 4096;

}  // namespace gewell::gemma4_31b::sm120
