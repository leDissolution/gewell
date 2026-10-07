#include "gewell/mtp_head.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using gewell::MtpDepthPrediction;
using gewell::choose_mtp_depths;
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void budget() {
  const std::vector<MtpDepthPrediction> predictions{
      {0, 4, {.95F, .9F, .8F, .7F}}, {0, 4, {.85F, .4F, .2F, .1F}}};
  require(choose_mtp_depths(predictions, 7, 0, 0) == std::vector<std::uint32_t>{4, 1},
          "shared budget did not favor the more promising prefix");
  require(choose_mtp_depths(predictions, 7, .85F, 0) == std::vector<std::uint32_t>{2, 1},
          "threshold did not stop budget allocation");
  require(choose_mtp_depths(predictions, 0, .7F, 0) == std::vector<std::uint32_t>{4, 1},
          "unbounded width changed independent threshold policy");
  require(choose_mtp_depths(predictions, 1, 0, 0) == std::vector<std::uint32_t>{0, 0},
          "pending tokens were not charged to width");
  require(choose_mtp_depths(predictions, 100, 0, 0) == std::vector<std::uint32_t>{4, 4},
          "budget exceeded maximum depth");
  // Enumerate every feasible prefix pair: selection maximizes the sum of
  // survival probabilities (expected accepted drafts) under the row budget.
  for (unsigned width = 2; width <= 10; ++width) {
    const auto depths = choose_mtp_depths(predictions, width, 0, 0);
    float chosen = 0;
    for (unsigned row = 0; row < 2; ++row)
      for (unsigned k = 0; k < depths[row]; ++k) chosen += predictions[row].survival[k];
    require(2 + depths[0] + depths[1] == width, "available budget was not filled");
    for (unsigned a = 0; a <= 4; ++a) for (unsigned b = 0; b <= 4; ++b) {
      if (2 + a + b > width) continue;
      float alternative = 0;
      for (unsigned k = 0; k < a; ++k) alternative += predictions[0].survival[k];
      for (unsigned k = 0; k < b; ++k) alternative += predictions[1].survival[k];
      require(chosen + 1e-6F >= alternative, "selection missed a better feasible prefix allocation");
    }
  }
}
void ties_and_limits() {
  const std::vector<MtpDepthPrediction> tied(32, {0, 15, std::vector<float>(15, .5F)});
  require(choose_mtp_depths(tied, 256, 0, 0) == std::vector<std::uint32_t>(32, 7),
          "equal scores should favor shallow positions across the batch");
  require(choose_mtp_depths({{0, 3, {.5F,.5F,.5F}}, {0, 3, {.5F,.5F,.5F}}}, 5, 0, 0)
              == std::vector<std::uint32_t>{2, 1}, "ties must be deterministic and preserve prefixes");
  require(choose_mtp_depths({{2, 3, {.1F,.05F,0}}, {1, 1, {}}, {0, 0, {}}}, 1, 1, 0)
              == std::vector<std::uint32_t>{2, 1, 0}, "minima and request limits lost priority");
  require(choose_mtp_depths({{0, 0, {}}, {0, 4, {.9F,.8F,.7F,.6F}}}, 6, 0, 0)
              == std::vector<std::uint32_t>{0, 4}, "unused request share was not redistributed");
  require(choose_mtp_depths({{0, 4, {}}, {0, 4, {.9F,.8F,.7F,.6F}}}, 7, 0, 0)
              == std::vector<std::uint32_t>{2, 3}, "missing probes did not reserve a uniform fallback share");
  require(choose_mtp_depths({{0, 4, {}}, {2, 3, {}}}, 0, 1, 0)
              == std::vector<std::uint32_t>{4, 3}, "unbounded fallback lost configured maximum");
  require(choose_mtp_depths({}, 256, .5F, 0).empty(), "empty batch acquired draft rows");
}
void invalid_curves() {
  for (const auto prediction : std::vector<MtpDepthPrediction>{
      {0, 3, {.9F,.2F,.7F}}, {0, 1, {1.01F}}, {0, 1, {-1}},
      {0, 1, {std::numeric_limits<float>::quiet_NaN()}},
      {0, 1, {std::numeric_limits<float>::infinity()}}, {0, 2, {.5F}}, {2, 1, {}}}) {
    bool rejected = false;
    try { choose_mtp_depths({prediction}, 256, 0, 0); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid survival curve or bounds were accepted");
  }
}
void redistribution() {
  const std::vector<MtpDepthPrediction> predictions{
      {0, 4, {.99F,.9F,.8F,.7F}}, {0, 4, {.95F,.85F,.65F,.3F}},
      {0, 4, {.6F,.3F,.2F,.1F}}, {0, 4, {.55F,.2F,.1F,.05F}}};
  require(choose_mtp_depths(predictions, 12, 0, 0) == std::vector<std::uint32_t>{4,3,1,0},
          "redistribution fixture lost its uneven original allocation");
  require(choose_mtp_depths(predictions, 12, 0, .5F) == std::vector<std::uint32_t>{3,3,1,1},
          "cap must be checked after redistributing the released slots");
  require(choose_mtp_depths(predictions, 12, .75F, .5F) == std::vector<std::uint32_t>{2,2,0,0},
          "redistribution must still respect the confidence threshold");
  for (unsigned width = 1; width <= 20; ++width) for (float fraction : {.4F,.5F,1.F}) {
    const auto depths = choose_mtp_depths(predictions, width, 0, fraction);
    const auto deepest = *std::max_element(depths.begin(), depths.end());
    const auto required = std::ceil(fraction * predictions.size());
    require(deepest <= 1 || std::count(depths.begin(), depths.end(), deepest) >= required,
            "deepest step violated the requested batch fraction");
    // Compare all caps: the selected shape must use the largest cap whose
    // optimal refilled allocation satisfies the active-fraction constraint.
    for (unsigned cap = deepest + 1; cap <= 4; ++cap) {
      auto bounded = predictions;
      for (auto& row : bounded) row.maximum = cap;
      const auto alternative = choose_mtp_depths(bounded, width, 0, 0);
      const auto actual = *std::max_element(alternative.begin(), alternative.end());
      require(actual <= deepest || std::count(alternative.begin(), alternative.end(), actual) < required,
              "redistribution skipped a larger feasible cap");
    }
  }
  require(choose_mtp_depths({{0,4,{.9F,.8F,.7F,.6F}}}, 5, 0, .5F)
              == std::vector<std::uint32_t>{4}, "smoothing shortened batch-one speculation");
  require(choose_mtp_depths({{4,4,{}},{0,4,{.5F,.4F,.3F,.2F}}}, 3, 0, 1)
              == std::vector<std::uint32_t>{4,0}, "smoothing overrode a forced minimum");
  require(choose_mtp_depths({{0,0,{}},{0,4,{.5F,.4F,.3F,.2F}}}, 6, 0, 1)
              == std::vector<std::uint32_t>{0,1}, "smoothing removed the first eligible draft step");
  require(choose_mtp_depths({{0,4,{}},{0,4,{.9F,.8F,.7F,.6F}}}, 7, 0, 1)
              == std::vector<std::uint32_t>{2,2}, "smoothing lost the missing-probe fallback bounds");
  for (float fraction : {-1.F, 1.1F, std::numeric_limits<float>::quiet_NaN()}) {
    bool rejected = false;
    try { choose_mtp_depths(predictions, 12, 0, fraction); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid tail fraction was accepted");
  }
  bool rejected = false;
  try { choose_mtp_depths(predictions, 0, 0, .5F); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, "tail redistribution silently accepted unlimited width");
}
}
int main() {
  try { budget(); ties_and_limits(); invalid_curves(); redistribution(); }
  catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
