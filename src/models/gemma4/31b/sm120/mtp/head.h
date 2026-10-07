#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gewell::mtp_head {
struct Input {
  const __nv_bfloat16* probes{};  // GPU [intermediate layers, target width].
  const __nv_bfloat16* final_hidden{};
  float context[9]{};
};

// Small FP32 transformer over captured layers. Loads the training checkpoint
// directly; no Python or training dependency is needed by the serving binary.
class Head {
 public:
  Head(const std::string& directory, std::uint32_t capacity);
  ~Head();
  const std::vector<std::uint32_t>& layers() const;
  std::uint32_t max_depth() const;
  std::size_t bytes() const;
  // Returns batch-major survival probabilities, including the host wait.
  std::vector<float> predict(const std::vector<Input>& inputs, cudaStream_t stream);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gewell::mtp_head
