#pragma once
#include "gewell/models/gemma4/26b_a4b/artifact.h"
#include "gewell/weight_qdq.h"
#include <array>
#include <string>
#include <string_view>

namespace gewell::gemma4_26b_a4b {
// Ordered rules: LAYER PROJECTION TYPE [EXPERT]. Layers are '*' or 0..29.
// Attention/shared MLP roles use the existing three-field format. Expert roles
// are expert_gate_proj/expert_up_proj/expert_down_proj; omitted EXPERT or '*'
// selects all experts, otherwise EXPERT is 0..127. Unspecified entries are BF16.
class QdqMask {
 public:
  static QdqMask Parse(std::string_view source, std::string name = "<memory>");
  static QdqMask Load(const std::string& path);
  [[nodiscard]] weight_qdq::Type type_for(std::size_t physical_id) const;
  [[nodiscard]] weight_qdq::SelectionSummary summarize() const;
  [[nodiscard]] weight_qdq::SelectionSummary summarize(const ArtifactFile& artifact) const;
  [[nodiscard]] bool has_qdq() const;
  [[nodiscard]] bool has_source() const { return has_source_; }
  [[nodiscard]] const Digest& source_sha256() const { return digest_; }
  // Validate every storage assertion before any in-place quantization.
  void validate(const ArtifactFile& artifact) const;
 private:
  std::array<weight_qdq::Type, kTextPhysicalTensorCount> types_{};
  bool has_source_{};
  Digest digest_{};
};

// Apply only to the freshly uploaded payload, before inference uses it.
// BF16 storage stays BF16; packed types are assertions, never reconstructions.
// The artifact mapping is read-only. The call synchronizes before returning.
weight_qdq::ApplySummary apply_qdq_in_place(const QdqMask& mask,
    const ArtifactFile& artifact, void* device_payload);
}  // namespace gewell::gemma4_26b_a4b
