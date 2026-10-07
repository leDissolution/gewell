#pragma once

#include "gewell/mtp_capture.h"
#include <memory>

namespace gewell::runtime {
struct BatchLimits;
struct BatchRequest;
struct BatchMtpInput;
struct MtpOutcome;

class MtpCapture {
 public:
  MtpCapture(const BatchLimits& limits, const MtpCaptureGeometry& geometry);
  ~MtpCapture();
  bool enabled() const;
  const std::vector<std::uint32_t>& layers() const;
  // First unused request sequence after the existing capture.
  std::uint64_t next_sequence() const;
  // Deterministic sampling, independent of inference RNG and acceptance. Each
  // call reserves one of the bounded capture attempts, including failed rounds.
  bool select(std::uint64_t sequence, std::uint64_t cycle);
  void record(const BatchRequest&, const BatchMtpInput&, const MtpOutcome&,
              std::uint64_t batch_id, std::uint32_t batch_size, std::uint32_t emitted);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gewell::runtime
