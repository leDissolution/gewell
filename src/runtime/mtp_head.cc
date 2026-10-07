#include "gewell/mtp_head.h"
#include <algorithm>
#include <cmath>
#include <queue>
#include <stdexcept>

namespace gewell {
namespace {
std::vector<std::uint32_t> allocate(const std::vector<MtpDepthPrediction>& predictions,
                                  std::uint32_t width, float threshold, std::uint32_t cap) {
  struct Candidate {
    float probability;
    std::uint32_t position;
    std::size_t request;
    bool operator<(const Candidate& other) const {
      if (probability != other.probability) return probability < other.probability;
      if (position != other.position) return position > other.position;
      return request > other.request;
    }
  };
  std::priority_queue<Candidate> candidates;
  std::vector<std::uint32_t> depths(predictions.size());
  std::uint64_t used = predictions.size();
  const auto share = width / predictions.size();
  for (std::size_t row = 0; row < predictions.size(); ++row) {
    const auto& prediction = predictions[row];
    const auto maximum = std::max(prediction.minimum, std::min(cap, prediction.maximum));
    auto& depth = depths[row];
    if (prediction.survival.empty()) {
      depth = width ? std::clamp<std::uint32_t>(share ? share - 1 : 0,
          prediction.minimum, maximum) : maximum;
    } else {
      depth = prediction.minimum;
      if (!width) {
        while (depth < maximum && prediction.survival[depth] >= threshold) ++depth;
      } else if (depth < maximum) {
        candidates.push({prediction.survival[depth], depth, row});
      }
    }
    used += depth;
  }
  while (used < width && !candidates.empty() && candidates.top().probability >= threshold) {
    const auto row = candidates.top().request;
    candidates.pop();
    const auto depth = ++depths[row];
    ++used;
    const auto& prediction = predictions[row];
    if (depth < std::min(cap, prediction.maximum))
      candidates.push({prediction.survival[depth], depth, row});
  }
  return depths;
}
}  // namespace

std::vector<std::uint32_t> choose_mtp_depths(const std::vector<MtpDepthPrediction>& predictions,
                                           std::uint32_t width, float threshold,
                                           float tail_fraction) {
  if (!std::isfinite(threshold) || threshold < 0 || threshold > 1)
    throw std::invalid_argument("MTP head threshold must be in [0,1]");
  if (!std::isfinite(tail_fraction) || tail_fraction < 0 || tail_fraction > 1 ||
      (tail_fraction > 0 && !width))
    throw std::invalid_argument("MTP head tail fraction must be in [0,1] and requires positive width");
  if (predictions.empty()) return {};
  std::uint32_t maximum = 0, minimum = 1;
  for (const auto& prediction : predictions) {
    if (prediction.minimum > prediction.maximum ||
        (!prediction.survival.empty() && prediction.survival.size() < prediction.maximum))
      throw std::invalid_argument("MTP head prediction has invalid depth bounds");
    float previous = 1;
    for (const float probability : prediction.survival) {
      if (!std::isfinite(probability) || probability < 0 || probability > previous)
        throw std::invalid_argument("MTP head requires nonincreasing survival probabilities");
      previous = probability;
    }
    maximum = std::max(maximum, prediction.maximum);
    minimum = std::max(minimum, prediction.minimum);
  }
  auto depths = allocate(predictions, width, threshold, maximum);
  if (!tail_fraction) return depths;
  const auto required = static_cast<std::size_t>(std::ceil(tail_fraction * predictions.size()));
  // Try caps from largest to smallest, reallocating before checking occupancy.
  // Forced minima take precedence; the first draft step is never removed merely
  // because too few requests are eligible for speculation.
  auto cap = *std::max_element(depths.begin(), depths.end());
  while (cap > minimum &&
         static_cast<std::size_t>(std::count_if(depths.begin(), depths.end(),
             [cap](auto depth) { return depth >= cap; })) < required) {
    depths = allocate(predictions, width, threshold, --cap);
  }
  return depths;
}
}  // namespace gewell
