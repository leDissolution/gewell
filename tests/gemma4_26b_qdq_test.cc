#include "gewell/models/gemma4/26b_a4b/weight_qdq.h"
#include <iostream>
#include <stdexcept>

using namespace gewell::gemma4_26b_a4b;
using Type = gewell::weight_qdq::Type;
void require(bool value) { if (!value) throw std::runtime_error("26B QDQ contract failed"); }
template<class F> void rejects(F action) {
  try { action(); } catch (const std::exception&) { return; }
  throw std::runtime_error("invalid QDQ selection accepted");
}
int main(int argc, char** argv) {
  try {
    const QdqMask defaults;
    require(!defaults.has_source() && !defaults.has_qdq());
    require(defaults.summarize().bf16.tensor_count == 11725);
    auto empty = QdqMask::Parse("");
    require(empty.has_source() && empty.source_sha256().front() == 0xe3 && empty.source_sha256().back() == 0x55);
    auto mask = QdqMask::Parse("* v_proj fp8\n* expert_down_proj nvfp4\n29 expert_down_proj bf16 127 # override\n0 q_proj fp8_w8a8\n");
    const auto summary = mask.summarize();
    require(summary.fp8.tensor_count == 25 && summary.nvfp4.tensor_count == 3839 && summary.fp8_w8a8.tensor_count == 1);
    require(summary.nvfp4.element_count == std::uint64_t(3839) * 2816 * 704);
    require(summary.nvfp4.source_bf16_bytes == summary.nvfp4.element_count * 2);
    require(mask.type_for(kFinalNormPhysicalId - 1) == Type::bf16);
    require(mask.type_for(kFinalNormPhysicalId - 4) == Type::nvfp4);
    require(mask.type_for(0) == Type::bf16 && mask.type_for(1) == Type::bf16);
    require(mask.type_for(kFinalNormPhysicalId) == Type::bf16 && mask.has_qdq());
    auto expert = QdqMask::Parse("* expert_gate_proj fp8 0\n29 expert_gate_proj nvfp4 127\n");
    require(expert.summarize().fp8.tensor_count == 30 && expert.summarize().nvfp4.tensor_count == 1);
    require(QdqMask::Parse("* expert_up_proj fp8 *\n").summarize().fp8.tensor_count == 3840);
    for (const auto* invalid : {"30 q_proj fp8", "-1 q_proj fp8", "1x q_proj fp8",
        "5 v_proj fp8", "29 v_proj bf16", "0 router_proj fp8", "0 input_norm fp8",
        "0 q_proj nope", "0 q_proj", "0 q_proj fp8 1", "0 q_proj fp8 *",
        "0 expert_gate_proj fp8 128", "0 expert_up_proj fp8 -1", "0 expert_down_proj fp8 0 extra"})
      rejects([&] { QdqMask::Parse(invalid); });
    rejects([&] { QdqMask::Parse(std::string((1U << 20) + 1, '#')); });
    rejects([&] { mask.type_for(kTextPhysicalTensorCount); });
    if (argc == 3) {
      auto bf16 = ArtifactFile::Open(argv[1]);
      auto packed = ArtifactFile::Open(argv[2]);
      defaults.validate(bf16);
      defaults.validate(packed);
      rejects([&] { mask.validate(bf16); });
      QdqMask::Parse("* expert_gate_proj fp8\n").validate(bf16);
      rejects([&] { empty.validate(packed); });
      rejects([&] { QdqMask::Parse("* expert_gate_proj fp8\n").validate(packed); });
      QdqMask::Parse("* expert_gate_proj nvfp4_w4a4\n* expert_up_proj nvfp4_w4a4\n* expert_down_proj nvfp4_w4a4\n").validate(packed);
    }
    std::cout << "26B QDQ mask contract passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
