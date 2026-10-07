#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace gewell {
struct MtpHeadSettings {
  std::string path;
  // Minimum P(accepted >= k) eligible for allocation beyond forced minima.
  float threshold{0.5F};
  // Minimum batch fraction still drafting at the deepest step after refill.
  // Zero disables smoothing; positive values require a shared width budget.
  float tail_fraction{};
};

struct MtpDepthPrediction {
  std::uint32_t minimum{}, maximum{};
  // P(A >= k), not conditional acceptance. Empty means probes are unavailable.
  std::vector<float> survival;
};

// Width includes pending tokens. Forced minima take precedence over width.
// Missing predictions reserve the uniform width share for one fallback round.
std::vector<std::uint32_t> choose_mtp_depths(const std::vector<MtpDepthPrediction>& predictions,
                                           std::uint32_t width, float threshold,
                                           float tail_fraction);
}  // namespace gewell
