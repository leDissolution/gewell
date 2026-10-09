#pragma once

#include "gewell/runtime/image.h"
#include "gewell/models/embeddinggemma2/audio_processor.h"
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

namespace gewell::embeddinggemma2 {
struct ContextLengthError : std::invalid_argument {using std::invalid_argument::invalid_argument;};
struct PreparedInput {
  std::vector<std::uint32_t> tokens;
  // Still images and selected video frames in assembled sequence order.
  std::vector<std::shared_ptr<const runtime::ImageInput>> visuals;
  std::uint32_t video_count{};
  std::vector<std::shared_ptr<const audio::Input>> audios;
};
// During preparation, each media [begin,end) supplies its actual feature count.
// Assembly assigns final spans, then publishes immutable media references.
struct InputPart {
  std::string text;
  std::shared_ptr<runtime::ImageInput> image;
  std::vector<std::shared_ptr<runtime::ImageInput>> video;
  std::shared_ptr<audio::Input> audio;
};
void validate_input(const PreparedInput& input, int max_tokens = 8192);
}  // namespace gewell::embeddinggemma2
