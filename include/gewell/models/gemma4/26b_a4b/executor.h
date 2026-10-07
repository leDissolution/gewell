#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace gewell::gemma4_26b_a4b::sm120 {
// BF16 bit patterns at the documented reference boundaries.
using Capture = std::function<void(const std::string&, const std::vector<std::uint16_t>&)>;
class Executor {
 public:
  Executor(const std::string& weights, std::uint32_t capacity);
  ~Executor();
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;
  std::vector<std::uint16_t> ForwardToken(std::uint32_t token, const Capture& capture = {});
  void Reset();
  [[nodiscard]] std::uint32_t position() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gewell::gemma4_26b_a4b::sm120
