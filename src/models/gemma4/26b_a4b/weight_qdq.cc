#include "gewell/models/gemma4/26b_a4b/weight_qdq.h"
#include <openssl/evp.h>
#include <charconv>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace gewell::gemma4_26b_a4b {
namespace {
using Type = weight_qdq::Type;
constexpr std::size_t maximum_bytes = 1U << 20;
constexpr TensorRole roles[] = {TensorRole::q_proj, TensorRole::k_proj,
    TensorRole::v_proj, TensorRole::o_proj, TensorRole::gate_proj,
    TensorRole::up_proj, TensorRole::down_proj, TensorRole::expert_gate_proj,
    TensorRole::expert_up_proj, TensorRole::expert_down_proj};
constexpr std::string_view names[] = {"q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj", "expert_gate_proj", "expert_up_proj", "expert_down_proj"};
constexpr std::string_view types[] = {"bf16", "fp8", "nvfp4", "nvfp4_w4a4", "fp8_w8a8"};
bool projection(TensorRole role) {
  for (auto candidate : roles) if (candidate == role) return true;
  return false;
}
int coordinate(const std::string& value, unsigned limit) {
  if (value == "*") return -1;
  unsigned result{};
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  if (value.empty() || parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result >= limit)
    throw std::invalid_argument("coordinate outside model range");
  return static_cast<int>(result);
}
weight_qdq::TypeStats& stats(weight_qdq::SelectionSummary& result, Type type) {
  switch (type) {
    case Type::bf16: return result.bf16;
    case Type::fp8: return result.fp8;
    case Type::nvfp4: return result.nvfp4;
    case Type::nvfp4_w4a4: return result.nvfp4_w4a4;
    case Type::fp8_w8a8: return result.fp8_w8a8;
  }
  throw std::invalid_argument("invalid QDQ type");
}
}

QdqMask QdqMask::Parse(std::string_view source, std::string name) {
  if (source.size() > maximum_bytes) throw std::invalid_argument(name + ": QDQ mask exceeds 1 MiB");
  QdqMask result;
  result.has_source_ = true;
  unsigned digest_bytes{};
  if (EVP_Digest(source.data(), source.size(), result.digest_.data(), &digest_bytes, EVP_sha256(), nullptr) != 1 ||
      digest_bytes != result.digest_.size()) throw std::runtime_error("QDQ mask SHA256 failed");
  std::istringstream input{std::string(source)};
  std::string line;
  std::size_t number = 0;
  while (std::getline(input, line)) {
    ++number;
    line.resize(line.find('#') == std::string::npos ? line.size() : line.find('#'));
    std::istringstream fields(line);
    std::string layer_text, role_text, type_text, expert_text = "*", extra;
    if (!(fields >> layer_text)) continue;
    try {
      if (!(fields >> role_text >> type_text)) throw std::invalid_argument("expected LAYER PROJECTION TYPE [EXPERT]");
      const bool explicit_expert = bool(fields >> expert_text);
      if (fields >> extra) throw std::invalid_argument("too many fields");
      std::size_t role = 0, type = 0;
      while (role < std::size(names) && names[role] != role_text) ++role;
      while (type < std::size(types) && types[type] != type_text) ++type;
      if (role == std::size(names) || type == std::size(types)) throw std::invalid_argument("unknown projection or type");
      if (explicit_expert && role < 7) throw std::invalid_argument("expert selector requires an expert projection");
      const int layer = coordinate(layer_text, kLayerCount);
      const int expert = coordinate(expert_text, kExpertCount);
      bool matched = false;
      for (std::size_t id = 0; id < kTextTensors.size(); ++id) {
        const auto& tensor = kTextTensors[id];
        if (tensor.role != roles[role] || (layer >= 0 && tensor.layer != layer) ||
            (role >= 7 && expert >= 0 && tensor.expert != expert)) continue;
        result.types_[id] = static_cast<Type>(type);
        matched = true;
      }
      if (!matched) throw std::invalid_argument("projection does not exist at selected coordinates");
    } catch (const std::invalid_argument& error) {
      throw std::invalid_argument(name + ":" + std::to_string(number) + ": " + error.what());
    }
  }
  return result;
}

QdqMask QdqMask::Load(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("cannot open QDQ mask " + path);
  const auto size = input.tellg();
  if (size < 0 || static_cast<std::uint64_t>(size) > maximum_bytes)
    throw std::invalid_argument("QDQ mask exceeds 1 MiB or cannot be sized");
  std::string source(static_cast<std::size_t>(size), '\0');
  input.seekg(0);
  if (!source.empty()) input.read(source.data(), source.size());
  if (!input) throw std::runtime_error("cannot read QDQ mask " + path);
  return Parse(source, path);
}

weight_qdq::Type QdqMask::type_for(std::size_t id) const { return types_.at(id); }
bool QdqMask::has_qdq() const {
  for (auto type : types_) if (type == Type::fp8 || type == Type::nvfp4) return true;
  return false;
}
weight_qdq::SelectionSummary QdqMask::summarize() const {
  weight_qdq::SelectionSummary result;
  for (std::size_t id = 0; id < kTextTensors.size(); ++id) {
    const auto& tensor = kTextTensors[id];
    if (!projection(tensor.role)) continue;
    auto& item = stats(result, types_[id]);
    ++item.tensor_count;
    item.element_count += tensor.element_count();
    item.source_bf16_bytes += tensor.byte_count();
  }
  return result;
}
void QdqMask::validate(const ArtifactFile& artifact) const {
  for (std::size_t id = 0; id < kTextTensors.size(); ++id) {
    const auto storage = artifact.entries().at(id).storage_type;
    const auto stored = storage == StorageType::nvfp4_w4a4 ? Type::nvfp4_w4a4 :
        storage == StorageType::fp8_w8a8 ? Type::fp8_w8a8 : Type::bf16;
    const auto selected = types_[id];
    if (((selected == Type::nvfp4_w4a4 || selected == Type::fp8_w8a8) && selected != stored) ||
        (stored != Type::bf16 && has_source_ && selected != stored))
      throw std::invalid_argument("QDQ mask differs from packed storage at physical tensor " + std::to_string(id));
  }
}
weight_qdq::SelectionSummary QdqMask::summarize(const ArtifactFile& artifact) const {
  validate(artifact);
  auto result = summarize();
  if (has_source_) return result;
  for (std::size_t id = 0; id < kTextTensors.size(); ++id) {
    const auto storage = artifact.entries()[id].storage_type;
    if (storage == StorageType::bf16) continue;
    const auto& tensor = kTextTensors[id];
    auto& packed = stats(result, storage == StorageType::nvfp4_w4a4 ? Type::nvfp4_w4a4 : Type::fp8_w8a8);
    --result.bf16.tensor_count;
    result.bf16.element_count -= tensor.element_count();
    result.bf16.source_bf16_bytes -= tensor.byte_count();
    ++packed.tensor_count;
    packed.element_count += tensor.element_count();
    packed.source_bf16_bytes += tensor.byte_count();
  }
  return result;
}
}  // namespace gewell::gemma4_26b_a4b
