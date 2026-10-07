#pragma once

#include "gewell/vision_engine.h"
#include <cstdint>
#include <string>

namespace gewell::vision_runner {

// Runs one externally prepared image through the prefill-only vision tower.
// Input files are raw little-endian FP32 [padded_patch_rows,768] patches and
// int32 [padded_patch_rows,2] positions at one of the processor's supported
// capacities. The capture directory must not already exist.
int run(vision_engine::Model selected, const std::string& artifact_path, const std::string& pixel_values_path,
        const std::string& position_ids_path,
        std::uint32_t soft_token_count,
        const std::string& capture_directory);

}  // namespace gewell::vision_runner
