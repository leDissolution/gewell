#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

namespace gewell::runtime {
struct BatchLimits;
struct BatchRequest;

// Times describe the whole shared decode batch, not a per-request GPU slice.
struct MtpStatsSample {
  std::uint64_t batch_id{};
  std::uint32_t batch_size{}, depth{}, accepted{};
  std::size_t output_begin{}, emitted{};
  bool speculative{}, failed{};
  double wall_seconds{}, gpu_seconds{}, draft_seconds{}, verify_seconds{}, select_seconds{};
};

// Optional JSONL sink. Keeps only the current non-overlapping window for each
// active request; completed windows are flushed and discarded.
class MtpStats {
 public:
  explicit MtpStats(const BatchLimits& limits);
  ~MtpStats();
  bool enabled() const;
  // First unused request sequence after the existing statistics.
  std::uint64_t next_sequence() const;
  void observe(const BatchRequest& request, const MtpStatsSample& sample);
  void finish(const BatchRequest& request, std::string_view status);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gewell::runtime
