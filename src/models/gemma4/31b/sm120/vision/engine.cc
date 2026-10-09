#include "gewell/vision_engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace gewell::vision_engine {
namespace {

struct AddressRange {
  std::uintptr_t begin{};
  std::uintptr_t end{};
};

AddressRange address_range(const void* pointer, std::size_t bytes,
                           std::string_view label) {
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
    throw std::invalid_argument(std::string(label) +
                                " address range overflows");
  }
  return {begin, begin + bytes};
}

bool overlaps(AddressRange first, AddressRange second) {
  return first.begin < second.end && second.begin < first.end;
}

std::uint32_t prepared_rows(std::size_t bytes, std::size_t row_bytes,
                            std::string_view label) {
  if (bytes == 0 || bytes % row_bytes != 0 ||
      bytes / row_bytes > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument(std::string(label) +
                                " has an invalid byte length");
  }
  return static_cast<std::uint32_t>(bytes / row_bytes);
}

std::int32_t read_i32_le(const std::uint8_t* bytes) {
  const std::uint32_t value = static_cast<std::uint32_t>(bytes[0]) |
                              (static_cast<std::uint32_t>(bytes[1]) << 8) |
                              (static_cast<std::uint32_t>(bytes[2]) << 16) |
                              (static_cast<std::uint32_t>(bytes[3]) << 24);
  std::int32_t result = 0;
  static_assert(sizeof(result) == sizeof(value));
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

}  // namespace

std::uint32_t output_width(Model model) {
  switch (model) {
    case Model::gemma4_31b: return 5376;
    case Model::gemma4_26b_a4b: return 2816;
  }
  throw std::invalid_argument("unknown vision model");
}

std::size_t validate_prefill_request(const PrefillRequest& request, Model model) {
  const auto width = output_width(model);
  if (request.soft_features_bf16_device == nullptr) {
    throw std::invalid_argument("vision prefill output is null");
  }

  const PreparedImage& image = request.image;
  if (image.patch_values_device == nullptr) {
    throw std::invalid_argument("vision prefill patch values are null");
  }
  if (image.position_ids_device == nullptr) {
    throw std::invalid_argument("vision prefill position IDs are null");
  }
  if (!is_supported_padded_patch_rows(image.padded_patch_rows)) {
    throw std::invalid_argument(
        "vision prefill padded patch rows must select a supported capacity");
  }
  if (image.soft_token_count == 0 ||
      image.soft_token_count > image.padded_patch_rows /
                                   (gemma4_31b::kVisionPoolSize *
                                    gemma4_31b::kVisionPoolSize)) {
    throw std::invalid_argument(
        "vision prefill soft-token count exceeds prepared capacity");
  }
  if (request.soft_feature_row_capacity < image.soft_token_count) {
    throw std::invalid_argument("vision prefill output capacity is too small");
  }

  const std::size_t patch_bytes =
      prepared_pixel_bytes(image.padded_patch_rows);
  const std::size_t position_bytes =
      prepared_position_bytes(image.padded_patch_rows);
  const std::size_t output_bytes =
      static_cast<std::size_t>(image.soft_token_count) *
      width * sizeof(std::uint16_t);
  const AddressRange patches =
      address_range(image.patch_values_device, patch_bytes, "patch input");
  const AddressRange positions = address_range(
      image.position_ids_device, position_bytes, "position input");
  const AddressRange output = address_range(
      request.soft_features_bf16_device, output_bytes, "soft-feature output");
  if (overlaps(patches, positions) || overlaps(patches, output) ||
      overlaps(positions, output)) {
    throw std::invalid_argument("vision prefill device regions overlap");
  }
  return image.soft_token_count;
}

void validate_prepared_image_bytes(const void* pixel_bytes,
                                   std::size_t pixel_byte_count,
                                   const void* position_bytes,
                                   std::size_t position_byte_count,
                                   std::uint32_t soft_token_count) {
  if (pixel_bytes == nullptr) {
    throw std::invalid_argument("vision prepared pixels are null");
  }
  if (position_bytes == nullptr) {
    throw std::invalid_argument("vision prepared positions are null");
  }
  const std::uint32_t pixel_rows = prepared_rows(
      pixel_byte_count, gemma4_31b::kVisionPatchWidth,
      "vision prepared pixels");
  const std::uint32_t position_rows = prepared_rows(
      position_byte_count, 2 * sizeof(std::int32_t),
      "vision prepared positions");
  if (pixel_rows != position_rows) {
    throw std::invalid_argument(
        "vision prepared pixel and position row counts differ");
  }
  if (!is_supported_padded_patch_rows(pixel_rows)) {
    throw std::invalid_argument(
        "vision prepared padded rows do not select a supported capacity");
  }
  const std::uint32_t soft_token_capacity =
      pixel_rows /
      (gemma4_31b::kVisionPoolSize * gemma4_31b::kVisionPoolSize);
  if (soft_token_count == 0 || soft_token_count > soft_token_capacity) {
    throw std::invalid_argument(
        "vision prepared soft-token count exceeds prepared capacity");
  }

  const auto* pixels = static_cast<const std::uint8_t*>(pixel_bytes);
  const auto* positions = static_cast<const std::uint8_t*>(position_bytes);
  const std::size_t valid_rows =
      static_cast<std::size_t>(soft_token_count) *
      gemma4_31b::kVisionPoolSize * gemma4_31b::kVisionPoolSize;
  const auto coordinate = [&](std::size_t row, std::size_t axis) {
    return read_i32_le(positions + (row * 2 + axis) * sizeof(std::int32_t));
  };
  if (coordinate(0, 0) != 0 || coordinate(0, 1) != 0) {
    throw std::invalid_argument(
        "vision prepared valid position grid must start at (0,0)");
  }

  std::size_t grid_width = 1;
  while (grid_width < valid_rows && coordinate(grid_width, 1) == 0) {
    ++grid_width;
  }
  if (valid_rows % grid_width != 0 ||
      grid_width % gemma4_31b::kVisionPoolSize != 0 ||
      (valid_rows / grid_width) % gemma4_31b::kVisionPoolSize != 0) {
    throw std::invalid_argument(
        "vision prepared valid position grid is not rectangular and 3x3-divisible");
  }
  for (std::size_t row = 0; row < valid_rows; ++row) {
    const std::int32_t expected_x =
        static_cast<std::int32_t>(row % grid_width);
    const std::int32_t expected_y =
        static_cast<std::int32_t>(row / grid_width);
    if (coordinate(row, 0) != expected_x ||
        coordinate(row, 1) != expected_y) {
      throw std::invalid_argument(
          "vision prepared valid positions are not an x-fastest rectangular prefix");
    }
  }
  for (std::size_t row = valid_rows; row < pixel_rows; ++row) {
    if (coordinate(row, 0) != -1 || coordinate(row, 1) != -1) {
      throw std::invalid_argument(
          "vision prepared padding positions must be (-1,-1)");
    }
  }

  const std::size_t valid_pixel_bytes =
      valid_rows * gemma4_31b::kVisionPatchWidth;
  if (!std::all_of(pixels + valid_pixel_bytes,
                   pixels + pixel_byte_count,
                   [](std::uint8_t value) { return value == 0; })) {
    throw std::invalid_argument(
        "vision prepared padding patch rows must be exact zero");
  }
}

std::vector<std::uint8_t> prepared_pixels_from_file_bytes(
    const std::vector<std::uint8_t>& file_bytes) {
  if (file_bytes.size() % sizeof(float) != 0) {
    throw std::invalid_argument("vision pixel file has an invalid byte length");
  }
  std::vector<std::uint8_t> pixels(file_bytes.size() / sizeof(float));
  for (std::size_t i = 0; i < pixels.size(); ++i) {
    float value;
    std::memcpy(&value, file_bytes.data() + i * sizeof(float), sizeof(value));
    const float level = value * 255.0F;
    if (!(level >= 0.0F && level <= 255.0F)) {
      throw std::invalid_argument("vision pixel file values must be in [0,1]");
    }
    pixels[i] = static_cast<std::uint8_t>(std::lrint(level));
    if (pixels[i] * (1.0F / 255.0F) != value) {
      throw std::invalid_argument("vision pixel file values must be uint8/255");
    }
  }
  return pixels;
}

}  // namespace gewell::vision_engine
