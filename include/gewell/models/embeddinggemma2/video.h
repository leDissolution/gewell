#pragma once

#include "gewell/models/gemma4/image_processor.h"
#include "gewell/runtime/image.h"
#include <functional>
#include <memory>
#include <string_view>
#include <vector>

namespace gewell::embeddinggemma2 {
inline constexpr std::uint32_t kVideoFrameBudget = 140;
struct VideoMetadata {
  std::uint64_t total_frames{};
  double fps{}, duration{};
  std::vector<std::uint64_t> indices;
  // Presentation timestamps of selected frames; NaN denotes missing timestamps.
  std::vector<double> timestamps;
};
struct PreparedVideo {
  std::vector<std::shared_ptr<runtime::ImageInput>> frames;
  VideoMetadata metadata;
};
using PrepareVideoFrame = std::function<std::shared_ptr<runtime::ImageInput>(gemma4::RgbImage)>;
using PreparationPoll = std::function<void()>;
using ObserveVideoFrame = std::function<void(std::uint64_t, const gemma4::RgbImage&)>;

// Official fixed 1-FPS/32-frame/uniform sampling; absent or invalid FPS means
// uniform sampling from every decoded frame. Allocates at most 32 indices.
std::vector<std::uint64_t> sample_video_frames(std::uint64_t total_frames, double fps);

// Strict bounded data:video/mp4 (H.264) or data:video/webm (VP8/VP9).
// Two streaming passes count frames, then prepare only selected frames.
// A supplied prepare callback reserves/owns each frame's tensor lease; otherwise
// frames are prepared directly. Poll may throw to cancel preparation.
PreparedVideo prepare_video_data_url(std::string_view url, const PrepareVideoFrame& prepare = {},
    const PreparationPoll& poll = {}, const ObserveVideoFrame& observe = {});
}  // namespace gewell::embeddinggemma2
