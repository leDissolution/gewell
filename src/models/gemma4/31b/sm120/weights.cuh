#pragma once

#include "resources.cuh"
#include "gewell/models/gemma4/31b/component_weights.h"
#include "gewell/models/gemma4/31b/sm120/projection_fusion.h"

namespace gewell::gemma4_31b::sm120 {

struct LayerWeightIds {
  std::size_t input_norm{};
  std::size_t q_proj{};
  std::size_t k_proj{};
  std::size_t v_proj{std::numeric_limits<std::size_t>::max()};
  std::size_t q_norm{};
  std::size_t k_norm{};
  std::size_t o_proj{};
  std::size_t post_attention_norm{};
  std::size_t pre_feedforward_norm{};
  std::size_t gate_proj{};
  std::size_t up_proj{};
  std::size_t down_proj{};
  std::size_t post_feedforward_norm{};
  std::size_t layer_scalar{};
};

constexpr std::array<LayerWeightIds, model::kLayerCount> make_weight_ids() {
  std::array<LayerWeightIds, model::kLayerCount> result{};
  for (std::size_t layer = 0; layer < result.size(); ++layer) {
    const std::size_t base = 1 + 14 * layer - layer / 6;
    LayerWeightIds ids{};
    ids.input_norm = base;
    ids.q_proj = base + 1;
    ids.k_proj = base + 2;
    if (model::is_global_layer(layer)) {
      ids.q_norm = base + 3;
      ids.k_norm = base + 4;
      ids.o_proj = base + 5;
      ids.post_attention_norm = base + 6;
      ids.pre_feedforward_norm = base + 7;
      ids.gate_proj = base + 8;
      ids.up_proj = base + 9;
      ids.down_proj = base + 10;
      ids.post_feedforward_norm = base + 11;
      ids.layer_scalar = base + 12;
    } else {
      ids.v_proj = base + 3;
      ids.q_norm = base + 4;
      ids.k_norm = base + 5;
      ids.o_proj = base + 6;
      ids.post_attention_norm = base + 7;
      ids.pre_feedforward_norm = base + 8;
      ids.gate_proj = base + 9;
      ids.up_proj = base + 10;
      ids.down_proj = base + 11;
      ids.post_feedforward_norm = base + 12;
      ids.layer_scalar = base + 13;
    }
    result[layer] = ids;
  }
  return result;
}

inline constexpr auto kWeightIds = make_weight_ids();

constexpr bool tensor_is(std::size_t id, std::size_t layer,
                         model::TensorRole role) {
  return id < model::kPhysicalTensors.size() &&
         model::kPhysicalTensors[id].layer == static_cast<std::int16_t>(layer) &&
         model::kPhysicalTensors[id].role == role;
}

constexpr bool weight_ids_are_valid() {
  for (std::size_t layer = 0; layer < model::kLayerCount; ++layer) {
    const LayerWeightIds& ids = kWeightIds[layer];
    if (!tensor_is(ids.input_norm, layer, model::TensorRole::input_norm) ||
        !tensor_is(ids.q_proj, layer, model::TensorRole::q_proj) ||
        !tensor_is(ids.k_proj, layer, model::TensorRole::k_proj) ||
        !tensor_is(ids.q_norm, layer, model::TensorRole::q_norm) ||
        !tensor_is(ids.k_norm, layer, model::TensorRole::k_norm) ||
        !tensor_is(ids.o_proj, layer, model::TensorRole::o_proj) ||
        !tensor_is(ids.post_attention_norm, layer,
                   model::TensorRole::post_attention_norm) ||
        !tensor_is(ids.pre_feedforward_norm, layer,
                   model::TensorRole::pre_feedforward_norm) ||
        !tensor_is(ids.gate_proj, layer, model::TensorRole::gate_proj) ||
        !tensor_is(ids.up_proj, layer, model::TensorRole::up_proj) ||
        !tensor_is(ids.down_proj, layer, model::TensorRole::down_proj) ||
        !tensor_is(ids.post_feedforward_norm, layer,
                   model::TensorRole::post_feedforward_norm) ||
        !tensor_is(ids.layer_scalar, layer,
                   model::TensorRole::layer_scalar)) {
      return false;
    }
    if (model::is_global_layer(layer)) {
      if (ids.v_proj != std::numeric_limits<std::size_t>::max()) {
        return false;
      }
    } else if (!tensor_is(ids.v_proj, layer, model::TensorRole::v_proj)) {
      return false;
    }
  }
  return model::kPhysicalTensors[model::kFinalNormPhysicalId].role ==
             model::TensorRole::final_norm &&
         model::kPhysicalTensors[model::kFinalNormPhysicalId].layer ==
             model::kNoLayer;
}

static_assert(weight_ids_are_valid());

class WeightArena {
 public:
  explicit WeightArena(const artifact::ArtifactFile& file)
      : WeightArena(file, qdq::Mask{}) {}

  WeightArena(const artifact::ArtifactFile& file, const qdq::Mask& mask,
              const std::string& assistant_path = {}, const std::string& vision_path = {})
      : allocation_(static_cast<std::size_t>(file.header().payload_bytes)) {
    const auto* source = file.payload_data();
    auto* destination = static_cast<std::uint8_t*>(allocation_.data());
    std::size_t copied = 0;
    while (copied < allocation_.size()) {
      const std::size_t chunk =
          std::min(artifact::kIoChunkBytes, allocation_.size() - copied);
      check_cuda(cudaMemcpy(destination + copied, source + copied, chunk,
                            cudaMemcpyHostToDevice),
                 "BF16 weight host-to-device copy");
      copied += chunk;
    }
    qdq_summary_ = qdq::apply_in_place(mask, file, allocation_.data());
    for (const artifact::TensorEntry& entry : file.entries()) {
      const std::uint64_t offset =
          entry.file_offset - file.header().data_offset;
      const auto* data = destination + static_cast<std::size_t>(offset);
      if (entry.storage_type == artifact::StorageType::nvfp4_w4a4) {
        native_weights_[entry.physical_id] = {
            data, data + artifact::nvfp4_packed_bytes(entry.dim0, entry.dim1),
            entry.input_scale, entry.weight_scale_2};
        ++native_count_;
      } else if (entry.storage_type == artifact::StorageType::fp8_w8a8) {
        fp8_weights_[entry.physical_id] = {data, entry.input_scale, entry.weight_scale_2};
        ++fp8_count_;
      } else {
        pointers_[entry.physical_id] = reinterpret_cast<const BFloat16*>(data);
      }
    }
    // Reuse the two original payload regions; only one layer's temporary
    // storage is needed during loading, with no persistent weight duplication.
    std::unique_ptr<DeviceAllocation> joined;
    constexpr auto packed = std::size_t(model::kMlpSize) * model::kHiddenSize / 2;
    const auto scales = gewell::nvfp4::scale_storage_bytes(model::kMlpSize, model::kHiddenSize);
    for (const auto& ids : kWeightIds) {
      auto& gate = native_weights_[ids.gate_proj];
      auto& up = native_weights_[ids.up_proj];
      if (!gate.data || !up.data || gate.input_scale != up.input_scale ||
          gate.weight_scale != up.weight_scale) continue;
      if (!joined) joined = std::make_unique<DeviceAllocation>(2 * (packed + scales));
      auto* temp = static_cast<std::uint8_t*>(joined->data());
      check_cuda(cudaMemcpy(temp, gate.data, packed, cudaMemcpyDeviceToDevice), "join gate weights");
      check_cuda(cudaMemcpy(temp + packed, up.data, packed, cudaMemcpyDeviceToDevice), "join up weights");
      check_cuda(cudaMemcpy(temp + 2 * packed, gate.scales, scales, cudaMemcpyDeviceToDevice), "join gate scales");
      check_cuda(cudaMemcpy(temp + 2 * packed + scales, up.scales, scales, cudaMemcpyDeviceToDevice), "join up scales");
      auto* base = const_cast<std::uint8_t*>(gate.data);
      check_cuda(cudaMemcpy(base, temp, 2 * (packed + scales), cudaMemcpyDeviceToDevice), "store joined gate/up");
      gate.scales = base + 2 * packed;
      up.data = base + packed;
      up.scales = gate.scales + scales;
    }
    // Remove FP8 per-tensor trailers/padding in place, retaining every byte
    // of quantized weights and each projection's calibrated scales. Scalar
    // views remain valid when a mixed path needs an individual projection.
    std::vector<std::vector<std::size_t>> fp8_groups;
    std::vector<float> fp8_scales;
    for (unsigned layer = 0; layer < model::kLayerCount; ++layer) {
      const auto& ids = kWeightIds[layer];
      std::vector<std::size_t> qkv{ids.q_proj, ids.k_proj};
      if (!model::is_global_layer(layer)) qkv.push_back(ids.v_proj);
      for (auto group : {qkv, std::vector<std::size_t>{ids.gate_proj, ids.up_proj}}) {
        const auto input_scale = fp8_weights_[group.front()].input_scale;
        if (!std::all_of(group.begin(), group.end(), [&](auto id) {
              return fp8_weights_[id].data && fp8_weights_[id].input_scale == input_scale;
            })) continue;
        fp8_groups.push_back(group);
        for (auto id : group) {
          const auto& w = fp8_weights_[id];
          fp8_scales.insert(fp8_scales.end(), model::kPhysicalTensors[id].shape.dimensions[0],
                            w.input_scale * w.weight_scale);
        }
      }
    }
    if (!fp8_scales.empty()) {
      fp8_channel_scales_ = std::make_unique<DeviceAllocation>(fp8_scales.size() * sizeof(float));
      check_cuda(cudaMemcpy(fp8_channel_scales_->data(), fp8_scales.data(),
                            fp8_channel_scales_->size(), cudaMemcpyHostToDevice), "FP8 channel scales");
      std::size_t scale_offset = 0;
      for (const auto& group : fp8_groups) {
        std::size_t bytes = 0;
        for (auto id : group) {
          const auto& shape = model::kPhysicalTensors[id].shape;
          bytes += std::size_t(shape.dimensions[0]) * shape.dimensions[1];
        }
        if (!joined || joined->size() < bytes) joined = std::make_unique<DeviceAllocation>(bytes);
        auto* temp = static_cast<std::uint8_t*>(joined->data());
        std::size_t offset = 0;
        for (auto id : group) {
          const auto& shape = model::kPhysicalTensors[id].shape;
          const auto count = std::size_t(shape.dimensions[0]) * shape.dimensions[1];
          check_cuda(cudaMemcpy(temp + offset, fp8_weights_[id].data, count, cudaMemcpyDeviceToDevice),
                     "join FP8 weight rows");
          offset += count;
        }
        auto* base = const_cast<std::uint8_t*>(fp8_weights_[group.front()].data);
        check_cuda(cudaMemcpy(base, temp, bytes, cudaMemcpyDeviceToDevice), "store joined FP8 weights");
        fp8_weights_[group.front()].channel_scales =
            static_cast<const float*>(fp8_channel_scales_->data()) + scale_offset;
        offset = 0;
        for (auto id : group) {
          const auto& shape = model::kPhysicalTensors[id].shape;
          fp8_weights_[id].data = base + offset;
          offset += std::size_t(shape.dimensions[0]) * shape.dimensions[1];
          scale_offset += shape.dimensions[0];
        }
      }
    }
    if (!assistant_path.empty())
      assistant_ = load_component(assistant_path, model::Component::assistant);
    if (!vision_path.empty())
      vision_ = load_component(vision_path, model::Component::vision);
    console::field("text_weight_bytes", allocation_.size());
    console::field("assistant_weight_bytes", assistant_ ? assistant_->size() : 0);
    console::field("vision_weight_bytes", vision_ ? vision_->size() : 0);
    pointers_[model::kAssistantLmHeadLogicalId] = pointers_[model::kAssistantLmHeadPhysicalId];
    pointers_[model::kLmHeadLogicalId] =
        pointers_[file.header().lm_head_target_id];
    if (pointers_[model::kLmHeadLogicalId] !=
        pointers_[model::kEmbeddingPhysicalId]) {
      fail("weight arena", "tied LM head does not alias the embedding");
    }
    if (native_count_ || fp8_count_ || console::json_enabled()) {
      console::field("weight_execution_storage",
                     fp8_count_ && native_count_ ? "mixed_bf16_fp8_w8a8_nvfp4_w4a4" :
                     fp8_count_ ? "mixed_bf16_fp8_w8a8" :
                     native_count_ ? "mixed_bf16_nvfp4_w4a4" : "bf16");
      console::field("weight_native_nvfp4_w4a4_projection_tensors", native_count_);
      console::field("weight_native_fp8_w8a8_projection_tensors", fp8_count_);
      console::field("weight_native_activation_quantization", native_count_ != 0 || fp8_count_ != 0);
      console::field("weight_native_backend",
                     fp8_count_ && native_count_ ? "cublasLt_sm120_fp8_fp4" :
                     fp8_count_ ? "cublasLt_sm120_fp8" :
                     native_count_ ? "cublasLt_sm120_fp4" : "disabled");
    }
  }

  [[nodiscard]] const BFloat16* pointer(std::size_t id) const {
    if (id >= pointers_.size() || pointers_[id] == nullptr) {
      fail("BF16 weight pointer", "tensor is absent or uses native packed storage");
    }
    return pointers_[id];
  }
  const mtp_target::Weights& pointers() const { return pointers_; }
  const gewell::nvfp4::Weights& native_weights() const { return native_weights_; }
  bool has_native() const { return native_count_ != 0; }
  bool has_fp8() const { return fp8_count_ != 0; }
  bool has_vision() const { return vision_ != nullptr; }
  const gewell::fp8::Weights& fp8_weights() const { return fp8_weights_; }
  const BFloat16* projection_pointer(std::size_t id) const {
    return native_weights_.at(id).data || fp8_weights_.at(id).data ? nullptr : pointer(id);
  }
  [[nodiscard]] std::size_t size() const {
    return allocation_.size() + (fp8_channel_scales_ ? fp8_channel_scales_->size() : 0) +
        (assistant_ ? assistant_->size() : 0) + (vision_ ? vision_->size() : 0);
  }
  [[nodiscard]] const qdq::ApplySummary& qdq_summary() const {
    return qdq_summary_;
  }
  [[nodiscard]] std::uintptr_t address_mod_4096() const {
    return reinterpret_cast<std::uintptr_t>(allocation_.data()) %
           model::kStorageAlignment;
  }

 private:
  std::unique_ptr<DeviceAllocation> load_component(const std::string& path, model::Component component) {
    gewell::component::File file(path, model::component_specs(component));
    auto allocation = std::make_unique<DeviceAllocation>(file.device_bytes());
    auto* base = static_cast<std::uint8_t*>(allocation->data());
    check_cuda(cudaMemset(base, 0, allocation->size()), "zero component weight padding");
    std::size_t offset = 0;
    for (const auto& tensor : file.tensors()) {
      check_cuda(cudaMemcpy(base + offset, tensor.data, tensor.bytes, cudaMemcpyHostToDevice),
                 "copy component weights");
      pointers_[tensor.physical_id] = reinterpret_cast<const BFloat16*>(base + offset);
      offset += model::align_up(tensor.bytes, model::kStorageAlignment);
    }
    return allocation;
  }

  DeviceAllocation allocation_;
  std::unique_ptr<DeviceAllocation> assistant_, vision_;
  std::unique_ptr<DeviceAllocation> fp8_channel_scales_;
  std::array<const BFloat16*, model::kLogicalTensorCount> pointers_{};
  gewell::nvfp4::Weights native_weights_{};
  std::uint32_t native_count_{};
  gewell::fp8::Weights fp8_weights_{};
  std::uint32_t fp8_count_{};
  qdq::ApplySummary qdq_summary_{};
};

struct LayerWeights {
  const BFloat16* input_norm{};
  const BFloat16* q_proj{};
  const BFloat16* k_proj{};
  const BFloat16* v_proj{};
  const BFloat16* q_norm{};
  const BFloat16* k_norm{};
  const BFloat16* o_proj{};
  const BFloat16* post_attention_norm{};
  const BFloat16* pre_feedforward_norm{};
  const BFloat16* gate_proj{};
  const BFloat16* up_proj{};
  const BFloat16* down_proj{};
  const BFloat16* post_feedforward_norm{};
  const BFloat16* layer_scalar{};
};


}  // namespace gewell::gemma4_31b::sm120
