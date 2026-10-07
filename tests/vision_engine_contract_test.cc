#include "gewell/vision_engine.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace gewell::vision_engine {
namespace {

[[noreturn]] void fail(std::string_view message) {
  throw std::runtime_error(std::string(message));
}

template <typename Function>
void require_invalid(Function&& function, std::string_view label) {
  try {
    function();
  } catch (const std::invalid_argument&) {
    return;
  }
  fail(label);
}

constexpr std::uintptr_t kPatchAddress = 0x1000'0000ULL;
constexpr std::uintptr_t kPositionAddress = 0x2000'0000ULL;
constexpr std::uintptr_t kOutputAddress = 0x3000'0000ULL;

PreparedImage image(std::uint32_t soft_token_capacity,
                    std::uint32_t soft_token_count) {
  return {
      reinterpret_cast<const float*>(kPatchAddress),
      reinterpret_cast<const std::int32_t*>(kPositionAddress),
      padded_patch_rows_for_capacity(soft_token_capacity),
      soft_token_count,
  };
}

PrefillRequest request(PreparedImage image, std::size_t capacity) {
  return {
      image,
      reinterpret_cast<void*>(kOutputAddress),
      capacity,
  };
}

void test_shape_constants() {
  static_assert(kSupportedSoftTokenCapacities[0] == 70);
  static_assert(kSupportedSoftTokenCapacities[1] == 140);
  static_assert(kSupportedSoftTokenCapacities[2] == 280);
  static_assert(kSupportedSoftTokenCapacities[3] == 560);
  static_assert(kSupportedSoftTokenCapacities[4] == 1'120);
  static_assert(padded_patch_rows_for_capacity(70) == 630);
  static_assert(padded_patch_rows_for_capacity(140) == 1'260);
  static_assert(padded_patch_rows_for_capacity(280) == 2'520);
  static_assert(padded_patch_rows_for_capacity(560) == 5'040);
  static_assert(padded_patch_rows_for_capacity(1'120) == 10'080);
  static_assert(gemma4_31b::kVisionPatchWidth == 768);
  static_assert(gemma4_31b::kHiddenSize == 5'376);
  static_assert(prepared_pixel_bytes(2'520) == 7'741'440);
  static_assert(prepared_position_bytes(2'520) == 20'160);
  static_assert(prepared_pixel_bytes(10'080) == 30'965'760);
  static_assert(prepared_position_bytes(10'080) == 80'640);
}

void write_i32_le(std::uint8_t* bytes, std::int32_t signed_value) {
  const std::uint32_t value = static_cast<std::uint32_t>(signed_value);
  bytes[0] = static_cast<std::uint8_t>(value);
  bytes[1] = static_cast<std::uint8_t>(value >> 8);
  bytes[2] = static_cast<std::uint8_t>(value >> 16);
  bytes[3] = static_cast<std::uint8_t>(value >> 24);
}

void test_prepared_host_bytes() {
  for (const std::uint32_t capacity : kSupportedSoftTokenCapacities) {
    const std::uint32_t padded_rows =
        padded_patch_rows_for_capacity(capacity);
    std::vector<std::uint8_t> pixels(prepared_pixel_bytes(padded_rows), 0);
    std::vector<std::uint8_t> positions(
        prepared_position_bytes(padded_rows), 0xff);
    for (std::size_t row = 0; row < 9; ++row) {
      write_i32_le(positions.data() + (row * 2) * 4,
                   static_cast<std::int32_t>(row % 3));
      write_i32_le(positions.data() + (row * 2 + 1) * 4,
                   static_cast<std::int32_t>(row / 3));
    }
    validate_prepared_image_bytes(pixels.data(), pixels.size(),
                                  positions.data(), positions.size(), 1);
  }

  const std::uint32_t padded_rows = padded_patch_rows_for_capacity(70);
  std::vector<std::uint8_t> pixels(prepared_pixel_bytes(padded_rows), 0);
  std::vector<std::uint8_t> positions(
      prepared_position_bytes(padded_rows), 0xff);
  for (std::size_t row = 0; row < 9; ++row) {
    write_i32_le(positions.data() + (row * 2) * 4,
                 static_cast<std::int32_t>(row % 3));
    write_i32_le(positions.data() + (row * 2 + 1) * 4,
                 static_cast<std::int32_t>(row / 3));
  }
  require_invalid(
      [&] {
        validate_prepared_image_bytes(pixels.data(), pixels.size() - 1,
                                      positions.data(), positions.size(), 1);
      },
      "short pixel input was accepted");
  positions[0] = 1;
  require_invalid(
      [&] {
        validate_prepared_image_bytes(pixels.data(), pixels.size(),
                                      positions.data(), positions.size(), 1);
      },
      "invalid position grid was accepted");

  const std::uint32_t arbitrary_rows = 100 * 9;
  std::vector<std::uint8_t> arbitrary_pixels(
      prepared_pixel_bytes(arbitrary_rows), 0);
  std::vector<std::uint8_t> arbitrary_positions(
      prepared_position_bytes(arbitrary_rows), 0xff);
  require_invalid(
      [&] {
        validate_prepared_image_bytes(
            arbitrary_pixels.data(), arbitrary_pixels.size(),
            arbitrary_positions.data(), arbitrary_positions.size(), 1);
      },
      "unsupported prepared capacity was accepted");
  require_invalid(
      [&] {
        validate_prepared_image_bytes(
            pixels.data(), pixels.size(), positions.data(),
            positions.size() - 2 * sizeof(std::int32_t), 1);
      },
      "mismatched prepared row counts were accepted");
  require_invalid(
      [&] {
        validate_prepared_image_bytes(pixels.data(), pixels.size(),
                                      positions.data(), positions.size(), 71);
      },
      "soft-token count exceeding prepared capacity was accepted");
}

void test_valid_requests() {
  for (const std::uint32_t capacity : kSupportedSoftTokenCapacities) {
    PreparedImage prepared = image(capacity, capacity);
    for (auto model : {Model::gemma4_31b, Model::gemma4_26b_a4b})
      if (validate_prefill_request(request(prepared, capacity), model) != capacity)
        fail("single-image output row count mismatch");
  }
}

void test_request_validation() {
  if (output_width(Model::gemma4_31b) != 5376 || output_width(Model::gemma4_26b_a4b) != 2816)
    fail("vision model output width mismatch");
  require_invalid([] { static_cast<void>(output_width(static_cast<Model>(99))); },
      "unknown vision model accepted");
  auto adjacent = request(image(70, 1), 1);
  adjacent.soft_features_bf16_device = reinterpret_cast<void*>(kPositionAddress - 2816 * 2);
  if (validate_prefill_request(adjacent, Model::gemma4_26b_a4b) != 1)
    fail("26B adjacent output region rejected");
  require_invalid([&] { static_cast<void>(validate_prefill_request(adjacent, Model::gemma4_31b)); },
      "31B overlapping output region accepted");

  PreparedImage valid = image(280, 280);

  PrefillRequest null_output = request(valid, 280);
  null_output.soft_features_bf16_device = nullptr;
  require_invalid(
      [&] { static_cast<void>(validate_prefill_request(null_output, Model::gemma4_31b)); },
      "null output was accepted");

  PreparedImage null_patches = valid;
  null_patches.patch_values_device = nullptr;
  require_invalid(
      [&] {
        static_cast<void>(
            validate_prefill_request(request(null_patches, 280), Model::gemma4_31b));
      },
      "null patch values were accepted");

  PreparedImage null_positions = valid;
  null_positions.position_ids_device = nullptr;
  require_invalid(
      [&] {
        static_cast<void>(
            validate_prefill_request(request(null_positions, 280), Model::gemma4_31b));
      },
      "null positions were accepted");

  PreparedImage wrong_rows = valid;
  wrong_rows.padded_patch_rows = padded_patch_rows_for_capacity(100);
  require_invalid(
      [&] {
        static_cast<void>(
            validate_prefill_request(request(wrong_rows, 280), Model::gemma4_31b));
      },
      "wrong padded row count was accepted");

  PreparedImage no_tokens = valid;
  no_tokens.soft_token_count = 0;
  require_invalid(
      [&] {
        static_cast<void>(
            validate_prefill_request(request(no_tokens, 280), Model::gemma4_31b));
      },
      "zero soft tokens were accepted");

  PreparedImage too_many_tokens = valid;
  too_many_tokens.soft_token_count = 281;
  require_invalid(
      [&] {
        static_cast<void>(
            validate_prefill_request(request(too_many_tokens, 281), Model::gemma4_31b));
      },
      "excess soft tokens were accepted");

  require_invalid(
      [&] {
        static_cast<void>(validate_prefill_request(request(valid, 279), Model::gemma4_31b));
      },
      "short output capacity was accepted");

  PrefillRequest output_overlaps_patches = request(valid, 280);
  output_overlaps_patches.soft_features_bf16_device =
      const_cast<float*>(valid.patch_values_device);
  require_invalid(
      [&] {
        static_cast<void>(
            validate_prefill_request(output_overlaps_patches, Model::gemma4_31b));
      },
      "output overlapping patch values was accepted");

  PreparedImage overlapping_inputs = valid;
  overlapping_inputs.position_ids_device =
      reinterpret_cast<const std::int32_t*>(kPatchAddress + 4);
  require_invalid(
      [&] {
        static_cast<void>(
            validate_prefill_request(request(overlapping_inputs, 280), Model::gemma4_31b));
      },
      "overlapping inputs were accepted");

  PreparedImage overflowing_input = valid;
  overflowing_input.patch_values_device = reinterpret_cast<const float*>(
      std::numeric_limits<std::uintptr_t>::max() - 1);
  require_invalid(
      [&] {
        static_cast<void>(
            validate_prefill_request(request(overflowing_input, 280), Model::gemma4_31b));
      },
      "overflowing input range was accepted");
}

}  // namespace
}  // namespace gewell::vision_engine

int main() {
  try {
    gewell::vision_engine::test_shape_constants();
    gewell::vision_engine::test_valid_requests();
    gewell::vision_engine::test_request_validation();
    gewell::vision_engine::test_prepared_host_bytes();
    std::cout << "vision engine contract tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "vision engine contract tests failed: " << error.what()
              << '\n';
    return 1;
  }
}
