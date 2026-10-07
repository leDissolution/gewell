#include "model_selection.h"
#include "models/gemma4/serving_assets_detail.h"
#include "gewell/models/gemma4/text_contract.h"
#include <array>
#include <fstream>
#include <stdexcept>

namespace gewell::app {
ModelKind artifact_model(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  std::array<char,8> magic{};
  if (!file.read(magic.data(), magic.size()))
    throw std::runtime_error("cannot read model artifact header: " + path);
  const std::string value(magic.data(), magic.size());
  if ((value == "G4A4BF16" || value == "G4A4MIX1")) return ModelKind::gemma4_26b_a4b;
  if (value == std::string("GEWBF16\0",8) || value == std::string("GEWNVF4\0",8) ||
      value == std::string("GEWMIX1\0",8)) return ModelKind::gemma4_31b;
  throw std::runtime_error("unsupported model artifact: " + path);
}
ModelKind serving_model(const std::string& directory) {
  using namespace gemma4::serving_detail;
  const auto manifest = json::parse(read_file(fs::path(directory) / "manifest.json", 8 * 1024 * 1024));
  const auto& format = manifest.at("format");
  if ((format == "gemma4-26b-a4b-bf16-v1" || format == "gemma4-26b-a4b-mixed-v1")) return ModelKind::gemma4_26b_a4b;
  if (format == "gemma4-31b-bf16-v4" || format == "gemma4-31b-nvfp4-w4a4-v2" ||
      format == "gemma4-31b-mixed-v2") return ModelKind::gemma4_31b;
  throw std::runtime_error("unsupported serving model format");
}
const text::TextContract& text_contract(ModelKind model) {
  return model == ModelKind::gemma4_26b_a4b ? gemma4::text_contract_26b_a4b() : gemma4::text_contract_31b();
}
}
