#include "gewell/models/gemma4/26b_a4b/weight_qdq.h"
#include "weight_qdq_detail.cuh"
#include <chrono>

namespace gewell::gemma4_26b_a4b {
weight_qdq::ApplySummary apply_qdq_in_place(const QdqMask& mask,
    const ArtifactFile& artifact, void* device_payload) {
  weight_qdq::ApplySummary result;
  result.selection = mask.summarize(artifact); // Validate the entire mask before mutation.
  if (!mask.has_qdq()) return result;
  if (!device_payload) throw std::invalid_argument("26B QDQ device payload is null");
  const auto started = std::chrono::steady_clock::now();
  weight_qdq::detail::DeviceState state;
  auto* base = static_cast<std::uint8_t*>(device_payload);
  for (std::size_t id = 0; id < kTextTensors.size(); ++id) {
    const auto type = mask.type_for(id);
    if (type != weight_qdq::Type::fp8 && type != weight_qdq::Type::nvfp4) continue;
    const auto& tensor = kTextTensors[id];
    auto* values = reinterpret_cast<bf16_primitives::BFloat16*>(
        base + artifact.entries()[id].offset - kArtifactDataOffset);
    weight_qdq::detail::apply_tensor(state, type, values, tensor.element_count());
  }
  weight_qdq::detail::check_cuda(cudaStreamSynchronize(nullptr), "synchronize 26B weight QDQ");
  result.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  return result;
}
}  // namespace gewell::gemma4_26b_a4b
