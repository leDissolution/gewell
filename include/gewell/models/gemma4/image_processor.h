#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace gewell::gemma4 {

inline constexpr std::size_t kImageMaxDataUrlBytes = 8 * 1024 * 1024;
inline constexpr std::uint32_t kImageMaxDimension = 8192;
inline constexpr std::size_t kImageMaxPixels = 16 * 1024 * 1024;
inline constexpr std::uint32_t kDefaultImageMaxSoftTokens = 280;

// Little-endian FP32 [max_soft_tokens*9,768] RGB patches and int32
// [max_soft_tokens*9,2] positions.
// Padding is zero pixels and (-1,-1) positions. No CUDA allocation is involved.
struct PreparedImage {
  std::vector<std::uint8_t> pixels;
  std::vector<std::uint8_t> positions;
  std::uint32_t padded_patch_rows{};
  std::uint32_t soft_token_count{};
};

// Accepts strict base64 data:image/png or data:image/jpeg URLs. Decoded images
// are bounded before pixel allocation. Throws invalid_argument for bad input.
// Matches the pinned Gemma4 processor's RGB conversion, uint8 antialiased
// bicubic resize, FP32 rescaling, and patch order. Supported token budgets are
// 70, 140, 280 (default), 560, and 1120.
[[nodiscard]] PreparedImage prepare_image_data_url(
    std::string_view data_url,
    std::uint32_t max_soft_tokens = kDefaultImageMaxSoftTokens);

}  // namespace gewell::gemma4
