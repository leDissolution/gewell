#pragma once
#include "gewell/models/embeddinggemma2/input.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace gewell::embeddinggemma2 {
// Diagnostic callbacks receive exact BF16 values expanded to FP32 on the host.
using Capture = std::function<void(const std::string&, const std::vector<int>&,
                                   const std::vector<float>&)>;
using Cancelled = std::function<bool()>;
// One GPU owner, resident weights, no persistent request state or KV cache.
// Packed sequences are masked per input, so attention never crosses inputs.
class Encoder {
 public:
  explicit Encoder(const std::string& directory, int max_tokens = 8192, bool vision = false, bool audio = false);
  ~Encoder();
  Encoder(const Encoder&) = delete;
  Encoder& operator=(const Encoder&) = delete;
  std::vector<std::vector<float>> encode(const PreparedInput& input,
      const std::vector<int>& dimensions = {768}, Capture capture = {}, Cancelled cancelled = {});
  // Packs inputs of similar length into shared forwards; vectors keep input order.
  std::vector<std::vector<float>> encode_batch(const std::vector<const PreparedInput*>& inputs,
      int dimension = 768, Cancelled cancelled = {});
  // Exact-token diagnostics intentionally allow unfilled modality placeholders.
  std::vector<std::vector<float>> encode_raw(const std::vector<std::uint32_t>& tokens,
      const std::vector<int>& dimensions = {768}, Capture capture = {});
  // Diagnostic replay from official inputs isolates a layer's arithmetic from
  // drift accumulated in previous layers. PLE is [tokens,24,512].
  void capture_layer(int layer, const std::vector<float>& hidden,
      const std::vector<float>& ple, Capture capture);
  void capture_vision_layer(int layer, const runtime::ImageInput& image,
      const std::vector<float>& hidden, Capture capture);
  void capture_vision_bridge(const std::vector<float>& hidden, Capture capture);
  void capture_audio_subsample(const audio::Input& input, Capture capture);
  void capture_audio_layer(int layer, const std::vector<float>& hidden, Capture capture);
  void capture_audio_bridge(const std::vector<float>& hidden, Capture capture);
  std::size_t weight_bytes() const;
  // Unique device allocation: shared text/vision working storage is counted once.
  std::size_t scratch_bytes() const;
  std::size_t vision_weight_bytes() const;
  // Vision's working-memory requirement, including storage shared with text.
  std::size_t vision_scratch_bytes() const;
  std::size_t audio_weight_bytes() const;
  std::size_t audio_scratch_bytes() const;
  int max_tokens() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace gewell::embeddinggemma2
