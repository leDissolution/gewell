#pragma once
#include <cstdint>

namespace gewell::artifact {
enum class StorageType : std::uint8_t { bf16 = 0, nvfp4_w4a4 = 1, fp8_w8a8 = 2 };
}
