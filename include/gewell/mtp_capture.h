#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gewell {
struct MtpCaptureSettings {
  std::string path;
  std::uint32_t every{32}, max_samples{250'000};
  // Completed target-layer counts, one-based; final norm is captured separately.
  std::vector<std::uint32_t> layers;  // Empty selects the loaded model's defaults.
};

struct MtpCaptureGeometry {
  std::uint32_t target_width{}, assistant_width{}, layer_count{};
  std::vector<std::uint32_t> default_layers;
};

struct MtpTargetProbes {
  std::uint32_t width{}, position{}, pending_token{};
  std::vector<std::uint32_t> layers;
  std::vector<std::uint16_t> hidden;
};

// Filtered distributions actually used by the verifier. Target probability is
// a training label, never a feature available to the draft-time predictor.
struct MtpCaptureScores {
  float draft_probability{}, target_probability{}, draft_entropy{}, draft_max_probability{};
};

// Host-owned output, filled before run_batch_mtp returns. BF16 values are kept
// as raw bits. Assistant rows predict draft_tokens[i], before sampling that ID.
struct MtpCaptureFeatures {
  std::uint32_t target_width{}, assistant_width{};
  std::vector<std::uint16_t> target_hidden, assistant_hidden;
  std::vector<std::uint32_t> draft_tokens;
  std::vector<MtpCaptureScores> scores;
  MtpTargetProbes probes;
};
}  // namespace gewell
