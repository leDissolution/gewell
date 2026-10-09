#pragma once

#include "gewell/models/gemma4/31b/model.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gewell::vision_engine {

enum class Model { gemma4_31b, gemma4_26b_a4b };
std::uint32_t output_width(Model model);

inline constexpr std::array<std::uint32_t, 5>
    kSupportedSoftTokenCapacities{70, 140, 280, 560, 1'120};
static_assert(kSupportedSoftTokenCapacities[4] ==
              gemma4_31b::kVisionMaxSoftTokenCount);

[[nodiscard]] constexpr bool is_supported_soft_token_capacity(
    std::uint32_t capacity) {
  for (const std::uint32_t supported : kSupportedSoftTokenCapacities) {
    if (capacity == supported) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] constexpr std::uint32_t padded_patch_rows_for_capacity(
    std::uint32_t soft_token_capacity) {
  return soft_token_capacity * gemma4_31b::kVisionPoolSize *
         gemma4_31b::kVisionPoolSize;
}

[[nodiscard]] constexpr bool is_supported_padded_patch_rows(
    std::uint32_t padded_patch_rows) {
  for (const std::uint32_t capacity : kSupportedSoftTokenCapacities) {
    if (padded_patch_rows == padded_patch_rows_for_capacity(capacity)) {
      return true;
    }
  }
  return false;
}

// Prepared pixels are uint8 RGB values; the tower rescales them by 1/255 on
// device, which reproduces the processor's FP32 values exactly.
[[nodiscard]] constexpr std::size_t prepared_pixel_bytes(
    std::uint32_t padded_patch_rows) {
  return static_cast<std::size_t>(padded_patch_rows) *
         gemma4_31b::kVisionPatchWidth;
}

// External pixel_values files keep the processor's FP32 [rows,768] layout.
[[nodiscard]] constexpr std::size_t prepared_pixel_file_bytes(
    std::uint32_t padded_patch_rows) {
  return prepared_pixel_bytes(padded_patch_rows) * sizeof(float);
}

[[nodiscard]] constexpr std::size_t prepared_position_bytes(
    std::uint32_t padded_patch_rows) {
  return static_cast<std::size_t>(padded_patch_rows) * 2 *
         sizeof(std::int32_t);
}

// One image after the pinned external Gemma 4 image preprocessing step.
//
// patch_values_device is row-major uint8 [padded_patch_rows,768]. Values are
// RGB patches whose processor values are value/255, with pixels laid out in
// patch-height, patch-width, channel order. position_ids_device is row-major int32
// [padded_patch_rows,2] containing (x,y). The supported padded capacities are
// 630, 1260, 2520, 5040, and 10080 rows. The first soft_token_count*9 rows
// describe the real rectangular patch grid in row-major order; every
// remaining position is (-1,-1) and its patch row is zero. Both pointers name
// CUDA device memory.
// Vision execution compacts to that valid prefix; padded rows are part of the
// processor-facing ABI but are not evaluated by the tower.
struct PreparedImage {
  const std::uint8_t* patch_values_device{};
  const std::int32_t* position_ids_device{};
  std::uint32_t padded_patch_rows{};
  std::uint32_t soft_token_count{};
};

// Host-side description of one prefill-only vision invocation. Tensor pointers
// point to CUDA device memory. The output is contiguous row-major BF16
// [image.soft_token_count,output_width(model)]. Input and output device regions must not
// overlap.
struct PrefillRequest {
  PreparedImage image{};
  void* soft_features_bf16_device{};
  std::size_t soft_feature_row_capacity{};
};

// Validates host-visible request metadata and returns the exact number of
// output rows. Device tensor contents are part of PreparedImage's caller
// contract and are not inspected here. Throws std::invalid_argument on an
// invalid request.
[[nodiscard]] std::size_t validate_prefill_request(
    const PrefillRequest& request, Model model);

// Validates the little-endian processor-facing files before copying them to
// CUDA. Pixel and position buffers must select the same supported padded
// capacity. The valid prefix must be an x-fastest rectangular patch grid,
// padding positions must be (-1,-1), and padding pixels must be exact zero.
void validate_prepared_image_bytes(const void* pixel_bytes,
                                   std::size_t pixel_byte_count,
                                   const void* position_bytes,
                                   std::size_t position_byte_count,
                                   std::uint32_t soft_token_count);

// Converts little-endian FP32 pixel_values file bytes to prepared uint8
// pixels. Every value must be exactly k/255 (k in 0..255) as produced by the
// processor's uint8 rescale. Throws std::invalid_argument otherwise.
[[nodiscard]] std::vector<std::uint8_t> prepared_pixels_from_file_bytes(
    const std::vector<std::uint8_t>& file_bytes);

}  // namespace gewell::vision_engine
