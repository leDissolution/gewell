#include "gewell/vision_executor.h"

#include "gewell/bf16_primitives.h"
#include "gewell/models/gemma4/31b/model.h"
#include "gewell/models/gemma4/31b/component_weights.h"
#include "gewell/models/gemma4/26b_a4b/component_weights.h"
#include "gewell/vision_primitives.h"

#include <cublasLt.h>

#include <array>
#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace gewell::vision_executor {
namespace {

namespace model = gemma4_31b;
namespace primitives = bf16_primitives;
namespace vision = vision_primitives;

constexpr std::size_t kScratchAlignment = 256;

[[noreturn]] void fail(std::string_view operation, std::string_view detail) {
  throw std::runtime_error(std::string(operation) + ": " +
                           std::string(detail));
}

void check_cublas(cublasStatus_t status, std::string_view operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    fail(operation, "cuBLAS status " + std::to_string(status));
  }
}

constexpr std::size_t align_up(std::size_t value, std::size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

struct WeightLayout {
  std::array<std::size_t, gemma4_26b_a4b::kVisionTensorCount> offsets{};
  std::size_t bytes{};
};

WeightLayout weight_layout(vision_engine::Model selected) {
  static_cast<void>(vision_engine::output_width(selected));
  const auto specs = selected == vision_engine::Model::gemma4_26b_a4b
      ? gemma4_26b_a4b::vision_tensor_specs()
      : model::component_specs(model::Component::vision);
  WeightLayout layout;
  for (std::size_t i = 0; i < specs.size(); ++i) {
    layout.offsets[i] = layout.bytes;
    std::size_t bytes = sizeof(BFloat16);
    for (const auto dimension : specs[i].shape) bytes *= dimension;
    layout.bytes += align_up(bytes, 4096);
  }
  return layout;
}

struct AddressRange {
  std::uintptr_t begin{};
  std::uintptr_t end{};
};

AddressRange address_range(const void* pointer, std::size_t bytes,
                           std::string_view label) {
  if (pointer == nullptr) {
    fail(label, "pointer is null");
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
    fail(label, "address range overflows");
  }
  return {begin, begin + bytes};
}

bool overlaps(AddressRange first, AddressRange second) {
  return first.begin < second.end && second.begin < first.end;
}

class LtHandle {
 public:
  LtHandle() { check_cublas(cublasLtCreate(&handle_), "cublasLtCreate"); }
  ~LtHandle() {
    if (handle_ != nullptr) {
      cublasLtDestroy(handle_);
    }
  }
  LtHandle(const LtHandle&) = delete;
  LtHandle& operator=(const LtHandle&) = delete;
  [[nodiscard]] cublasLtHandle_t get() const { return handle_; }

 private:
  cublasLtHandle_t handle_{};
};

class MatrixLayout {
 public:
  MatrixLayout(std::uint64_t rows, std::uint64_t columns,
               std::int64_t leading_dimension) {
    check_cublas(cublasLtMatrixLayoutCreate(&layout_, CUDA_R_16BF, rows,
                                            columns, leading_dimension),
                 "create vision linear matrix layout");
    const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    check_cublas(cublasLtMatrixLayoutSetAttribute(
                     layout_, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                     sizeof(order)),
                 "set vision linear row-major layout");
  }
  ~MatrixLayout() {
    if (layout_ != nullptr) {
      cublasLtMatrixLayoutDestroy(layout_);
    }
  }
  MatrixLayout(const MatrixLayout&) = delete;
  MatrixLayout& operator=(const MatrixLayout&) = delete;
  [[nodiscard]] cublasLtMatrixLayout_t get() const { return layout_; }
  void set_rows(std::uint64_t rows) {
    check_cublas(cublasLtMatrixLayoutSetAttribute(
        layout_, CUBLASLT_MATRIX_LAYOUT_ROWS, &rows, sizeof(rows)),
        "set vision linear rows");
  }

 private:
  cublasLtMatrixLayout_t layout_{};
};

class LinearPlan {
 public:
  LinearPlan(std::uint32_t rows, std::uint32_t input_width,
             std::uint32_t output_width)
      : input_(rows, input_width, input_width),
        weight_(output_width, input_width, input_width),
        output_(rows, output_width, output_width) {
    check_cublas(cublasLtMatmulDescCreate(&operation_, CUBLAS_COMPUTE_32F,
                                          CUDA_R_32F),
                 "create vision linear operation");
    const cublasOperation_t no_transpose = CUBLAS_OP_N;
    const cublasOperation_t transpose = CUBLAS_OP_T;
    check_cublas(cublasLtMatmulDescSetAttribute(
                     operation_, CUBLASLT_MATMUL_DESC_TRANSA, &no_transpose,
                     sizeof(no_transpose)),
                 "set vision linear TRANSA");
    check_cublas(cublasLtMatmulDescSetAttribute(
                     operation_, CUBLASLT_MATMUL_DESC_TRANSB, &transpose,
                     sizeof(transpose)),
                 "set vision linear TRANSB");
  }
  ~LinearPlan() {
    if (operation_ != nullptr) {
      cublasLtMatmulDescDestroy(operation_);
    }
  }
  LinearPlan(const LinearPlan&) = delete;
  LinearPlan& operator=(const LinearPlan&) = delete;

  void set_rows(std::uint32_t rows) {
    input_.set_rows(rows);
    output_.set_rows(rows);
  }

  void run(cublasLtHandle_t handle, const BFloat16* input,
           const BFloat16* weight, BFloat16* output,
           cudaStream_t stream) const {
    const float alpha = 1.0F;
    const float beta = 0.0F;
    check_cublas(
        cublasLtMatmul(handle, operation_, &alpha, input, input_.get(), weight,
                       weight_.get(), &beta, output, output_.get(), output,
                       output_.get(), nullptr, nullptr, 0, stream),
        "run vision linear");
  }

 private:
  MatrixLayout input_;
  MatrixLayout weight_;
  MatrixLayout output_;
  cublasLtMatmulDesc_t operation_{};
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
};

class ScratchLayout {
 public:
  explicit ScratchLayout(std::uint32_t patch_rows) {
    const std::size_t hidden_bytes =
        static_cast<std::size_t>(patch_rows) * model::kVisionHiddenSize *
        sizeof(BFloat16);
    const std::size_t mlp_bytes =
        static_cast<std::size_t>(patch_rows) * model::kVisionMlpSize *
        sizeof(BFloat16);
    const std::size_t pool_bytes =
        static_cast<std::size_t>(patch_rows / 9) *
        model::kVisionHiddenSize * sizeof(float);

    h0 = take(hidden_bytes);
    h1 = take(hidden_bytes);
    h2 = take(hidden_bytes);
    raw = take(hidden_bytes);
    query = take(hidden_bytes);
    key = take(hidden_bytes);
    value = take(hidden_bytes);
    gate = take(mlp_bytes);
    up = take(mlp_bytes);
    product = take(mlp_bytes);
    pooled = take(pool_bytes);
    bytes = align_up(cursor_, kScratchAlignment);
  }

  std::size_t h0{};
  std::size_t h1{};
  std::size_t h2{};
  std::size_t raw{};
  std::size_t query{};
  std::size_t key{};
  std::size_t value{};
  std::size_t gate{};
  std::size_t up{};
  std::size_t product{};
  std::size_t pooled{};
  std::size_t bytes{};

 private:
  std::size_t take(std::size_t size) {
    cursor_ = align_up(cursor_, kScratchAlignment);
    const std::size_t result = cursor_;
    if (size > std::numeric_limits<std::size_t>::max() - cursor_) {
      fail("vision scratch layout", "byte count overflows");
    }
    cursor_ += size;
    return result;
  }

  std::size_t cursor_{};
};

static_assert(static_cast<std::size_t>(model::kVisionHeadCount) *
                      model::kVisionHeadSize ==
                  model::kVisionHiddenSize);

}  // namespace

std::size_t vision_weight_slice_bytes(vision_engine::Model selected) {
  return weight_layout(selected).bytes;
}

class VisionExecutor::Impl {
 public:
  Impl(vision_engine::Model selected, VisionWeightSlice weights, std::uint32_t soft_token_count)
      : selected_(selected), output_width_(vision_engine::output_width(selected)),
        layout_(weight_layout(selected)), weights_(checked_weights(weights, layout_.bytes)),
        soft_token_count_(soft_token_count),
        patch_rows_(checked_patch_rows(soft_token_count)),
        scratch_(patch_rows_),
        patch_plan_(patch_rows_, model::kVisionPatchWidth,
                    model::kVisionHiddenSize),
        hidden_plan_(patch_rows_, model::kVisionHiddenSize,
                     model::kVisionHiddenSize),
        hidden_to_mlp_(patch_rows_, model::kVisionHiddenSize,
                       model::kVisionMlpSize),
        mlp_to_hidden_(patch_rows_, model::kVisionMlpSize,
                       model::kVisionHiddenSize),
        bridge_(soft_token_count_, model::kVisionHiddenSize,
                output_width_) {
    bind_weights();
  }

  [[nodiscard]] std::uint32_t soft_token_count() const {
    return soft_token_count_;
  }
  [[nodiscard]] std::uint32_t patch_rows() const { return patch_rows_; }
  [[nodiscard]] std::size_t scratch_bytes() const { return scratch_.bytes; }

  void set_soft_token_count(std::uint32_t soft_token_count) {
    const auto patch_rows = checked_patch_rows(soft_token_count);
    if (soft_token_count == soft_token_count_) return;
    patch_plan_.set_rows(patch_rows);
    hidden_plan_.set_rows(patch_rows);
    hidden_to_mlp_.set_rows(patch_rows);
    mlp_to_hidden_.set_rows(patch_rows);
    bridge_.set_rows(soft_token_count);
    soft_token_count_ = soft_token_count;
    patch_rows_ = patch_rows;
    scratch_ = ScratchLayout(patch_rows);
  }

  void run(const vision_engine::PrefillRequest& request, void* scratch_device,
           std::size_t scratch_capacity_bytes, cudaStream_t stream,
           CaptureSink* captures) const {
    static_cast<void>(vision_engine::validate_prefill_request(request, selected_));
    if (request.image.soft_token_count != soft_token_count_) {
      throw std::invalid_argument(
          "vision request soft-token count differs from executor plan");
    }
    if (scratch_device == nullptr) {
      throw std::invalid_argument("vision executor scratch is null");
    }
    if (reinterpret_cast<std::uintptr_t>(scratch_device) %
            kScratchAlignment != 0) {
      throw std::invalid_argument(
          "vision executor scratch must be 256-byte aligned");
    }
    if (scratch_capacity_bytes < scratch_.bytes) {
      throw std::invalid_argument("vision executor scratch capacity is too small");
    }
    validate_scratch_ranges(request, scratch_device);

    auto* const base = static_cast<std::uint8_t*>(scratch_device);
    BFloat16* const h0 = at<BFloat16>(base, scratch_.h0);
    BFloat16* const h1 = at<BFloat16>(base, scratch_.h1);
    BFloat16* const h2 = at<BFloat16>(base, scratch_.h2);
    BFloat16* const raw = at<BFloat16>(base, scratch_.raw);
    BFloat16* const query = at<BFloat16>(base, scratch_.query);
    BFloat16* const key = at<BFloat16>(base, scratch_.key);
    BFloat16* const value = at<BFloat16>(base, scratch_.value);
    BFloat16* const gate = at<BFloat16>(base, scratch_.gate);
    BFloat16* const up = at<BFloat16>(base, scratch_.up);
    BFloat16* const product = at<BFloat16>(base, scratch_.product);
    float* const pooled = at<float>(base, scratch_.pooled);
    const auto* const positions = request.image.position_ids_device;

    capture(captures, "input.pixel_values",
            request.image.patch_values_device, CaptureDType::f32,
            request.image.padded_patch_rows, model::kVisionPatchWidth,
            stream);
    capture(captures, "input.image_position_ids", positions,
            CaptureDType::i32, request.image.padded_patch_rows, 2, stream);
    vision::normalize_patch_values(request.image.patch_values_device, raw,
                                   patch_rows_, stream);
    patch_plan_.run(linear_handle_.get(), raw, patch_projection_, h2, stream);
    vision::add_patch_position_embeddings(
        h2, positions, position_embedding_, h0, patch_rows_, stream);
    capture(captures, "vision.patch_embedding", h0, CaptureDType::bf16,
            patch_rows_, model::kVisionHiddenSize, stream);

    const std::size_t hidden_elements =
        static_cast<std::size_t>(patch_rows_) * model::kVisionHiddenSize;
    const std::size_t mlp_elements =
        static_cast<std::size_t>(patch_rows_) * model::kVisionMlpSize;
    for (std::uint32_t layer = 0; layer < model::kVisionLayerCount; ++layer) {
      const LayerWeights& weight = layers_[layer];
      primitives::rms_norm(h0, weight.input_norm, h1, patch_rows_,
                           model::kVisionHiddenSize, 1.0e-6F, stream);

      project_norm_and_rope(h1, weight.q_proj, weight.q_norm, raw, h2, query,
                            positions, stream);
      project_norm_and_rope(h1, weight.k_proj, weight.k_norm, raw, h2, key,
                            positions, stream);
      hidden_plan_.run(linear_handle_.get(), h1, weight.v_proj, raw, stream);
      primitives::rms_norm_unscaled(raw, h2,
                                    patch_rows_ * model::kVisionHeadCount,
                                    model::kVisionHeadSize, 1.0e-6F, stream);
      vision::token_heads_to_head_tokens(h2, value, patch_rows_, stream);

      vision::full_attention(query, key, value, raw, patch_rows_, stream);
      hidden_plan_.run(linear_handle_.get(), raw, weight.o_proj, h2, stream);
      primitives::rms_norm(h2, weight.post_attention_norm, h1, patch_rows_,
                           model::kVisionHiddenSize, 1.0e-6F, stream);
      primitives::residual_add(h0, h1, h2, hidden_elements, stream);

      primitives::rms_norm(h2, weight.pre_feedforward_norm, h1, patch_rows_,
                           model::kVisionHiddenSize, 1.0e-6F, stream);
      hidden_to_mlp_.run(linear_handle_.get(), h1, weight.gate_proj, gate,
                         stream);
      hidden_to_mlp_.run(linear_handle_.get(), h1, weight.up_proj, up, stream);
      primitives::gelu_tanh_multiply(gate, up, product, mlp_elements, stream);
      mlp_to_hidden_.run(linear_handle_.get(), product, weight.down_proj, h0,
                         stream);
      primitives::rms_norm(h0, weight.post_feedforward_norm, h1, patch_rows_,
                           model::kVisionHiddenSize, 1.0e-6F, stream);
      primitives::residual_add(h2, h1, h0, hidden_elements, stream);

      char name[40];
      const int length =
          std::snprintf(name, sizeof(name), "vision.layer.%02u.output", layer);
      if (length < 0 || static_cast<std::size_t>(length) >= sizeof(name)) {
        fail("vision capture", "layer name exceeds fixed buffer");
      }
      capture(captures,
              std::string_view(name, static_cast<std::size_t>(length)), h0,
              CaptureDType::bf16, patch_rows_, model::kVisionHiddenSize,
              stream);
    }

    vision::pool_3x3_scaled(h0, positions, pooled, patch_rows_, stream);
    capture(captures, "vision.post_pool_scaled", pooled, CaptureDType::f32,
            soft_token_count_, model::kVisionHiddenSize, stream);
    vision::standardize(pooled, std_bias_, std_scale_, h1,
                        soft_token_count_, stream);
    capture(captures, "vision.post_standardization", h1,
            CaptureDType::bf16, soft_token_count_, model::kVisionHiddenSize,
            stream);
    primitives::rms_norm_unscaled(h1, h2, soft_token_count_,
                                  model::kVisionHiddenSize, 1.0e-6F, stream);
    capture(captures, "vision.bridge_norm", h2, CaptureDType::bf16,
            soft_token_count_, model::kVisionHiddenSize, stream);
    bridge_.run(linear_handle_.get(), h2, projection_,
                static_cast<BFloat16*>(request.soft_features_bf16_device),
                stream);
    capture(captures, "vision.soft_features",
            request.soft_features_bf16_device, CaptureDType::bf16,
            soft_token_count_, output_width_, stream);
  }

 private:
  static VisionWeightSlice checked_weights(VisionWeightSlice weights, std::size_t expected_bytes) {
    if (weights.base == nullptr) {
      throw std::invalid_argument("vision weight slice base is null");
    }
    if (reinterpret_cast<std::uintptr_t>(weights.base) %
            kScratchAlignment != 0) {
      throw std::invalid_argument(
          "vision weight slice must be 256-byte aligned");
    }
    if (weights.bytes != expected_bytes) {
      throw std::invalid_argument(
          "vision weight slice size does not match selected model");
    }
    static_cast<void>(
        address_range(weights.base, weights.bytes, "vision weight slice"));
    return weights;
  }

  static std::uint32_t checked_patch_rows(std::uint32_t soft_token_count) {
    if (soft_token_count == 0 ||
        soft_token_count > model::kVisionMaxSoftTokenCount) {
      throw std::invalid_argument(
          "vision executor soft-token count must be in [1,1120]");
    }
    return soft_token_count * model::kVisionPoolSize * model::kVisionPoolSize;
  }

  const BFloat16* weight(std::size_t id) const {
    const auto* base = reinterpret_cast<const std::uint8_t*>(weights_.base);
    return reinterpret_cast<const BFloat16*>(base + layout_.offsets.at(id));
  }

  void bind_weights() {
    namespace ids = gemma4_26b_a4b;
    patch_projection_ = weight(ids::kVisionPatchProjectionId);
    position_embedding_ = weight(ids::kVisionPositionEmbeddingId);
    for (std::size_t layer = 0; layer < layers_.size(); ++layer) {
      const auto base = ids::kVisionLayerFirstId + layer * ids::kVisionLayerTensorCount;
      layers_[layer] = {
          weight(base), weight(base + 1), weight(base + 2), weight(base + 3),
          weight(base + 4), weight(base + 5), weight(base + 6), weight(base + 7),
          weight(base + 8), weight(base + 9), weight(base + 10), weight(base + 11),
          weight(base + 12)};
    }
    std_bias_ = weight(ids::kVisionStdBiasId);
    std_scale_ = weight(ids::kVisionStdScaleId);
    projection_ = weight(ids::kVisionProjectionId);
  }

  void validate_scratch_ranges(const vision_engine::PrefillRequest& request,
                               void* scratch_device) const {
    const AddressRange scratch =
        address_range(scratch_device, scratch_.bytes, "vision scratch");
    const AddressRange weights =
        address_range(weights_.base, weights_.bytes, "vision weight slice");
    const std::size_t prepared_patch_bytes =
        vision_engine::prepared_pixel_bytes(
            request.image.padded_patch_rows);
    const std::size_t prepared_position_bytes =
        vision_engine::prepared_position_bytes(
            request.image.padded_patch_rows);
    const std::size_t output_bytes =
        static_cast<std::size_t>(soft_token_count_) * output_width_ *
        sizeof(BFloat16);
    const std::array<AddressRange, 4> forbidden{{
        weights,
        address_range(request.image.patch_values_device, prepared_patch_bytes,
                      "vision patch input"),
        address_range(request.image.position_ids_device,
                      prepared_position_bytes, "vision position input"),
        address_range(request.soft_features_bf16_device, output_bytes,
                      "vision output"),
    }};
    for (const AddressRange range : forbidden) {
      if (overlaps(scratch, range)) {
        throw std::invalid_argument(
            "vision executor scratch overlaps weights, input, or output");
      }
    }
    for (std::size_t index = 1; index < forbidden.size(); ++index) {
      if (overlaps(weights, forbidden[index])) {
        throw std::invalid_argument(
            "vision weights overlap request input or output");
      }
    }
  }

  void project_norm_and_rope(const BFloat16* input,
                             const BFloat16* projection,
                             const BFloat16* norm_weight, BFloat16* raw,
                             BFloat16* normalized, BFloat16* head_major,
                             const std::int32_t* positions,
                             cudaStream_t stream) const {
    hidden_plan_.run(linear_handle_.get(), input, projection, raw, stream);
    primitives::rms_norm(raw, norm_weight, normalized,
                         patch_rows_ * model::kVisionHeadCount,
                         model::kVisionHeadSize, 1.0e-6F, stream);
    vision::apply_2d_rope_transpose(normalized, positions, head_major,
                                    patch_rows_, stream);
  }

  static void capture(CaptureSink* sink, std::string_view name,
                      const void* data, CaptureDType dtype,
                      std::uint32_t rows, std::uint32_t columns,
                      cudaStream_t stream) {
    if (sink != nullptr) {
      sink->capture({name, data, dtype, rows, columns}, stream);
    }
  }

  template <typename T>
  static T* at(std::uint8_t* base, std::size_t offset) {
    return reinterpret_cast<T*>(base + offset);
  }

  vision_engine::Model selected_;
  std::uint32_t output_width_;
  WeightLayout layout_;
  VisionWeightSlice weights_{};
  std::uint32_t soft_token_count_{};
  std::uint32_t patch_rows_{};
  ScratchLayout scratch_;
  LtHandle linear_handle_;
  LinearPlan patch_plan_;
  LinearPlan hidden_plan_;
  LinearPlan hidden_to_mlp_;
  LinearPlan mlp_to_hidden_;
  LinearPlan bridge_;
  const BFloat16* patch_projection_{};
  const BFloat16* position_embedding_{};
  std::array<LayerWeights, model::kVisionLayerCount> layers_{};
  const BFloat16* std_bias_{};
  const BFloat16* std_scale_{};
  const BFloat16* projection_{};
};

VisionExecutor::VisionExecutor(vision_engine::Model selected, VisionWeightSlice weights,
                               std::uint32_t soft_token_count)
    : impl_(std::make_unique<Impl>(selected, weights, soft_token_count)) {}

VisionExecutor::~VisionExecutor() = default;
VisionExecutor::VisionExecutor(VisionExecutor&&) noexcept = default;
VisionExecutor& VisionExecutor::operator=(VisionExecutor&&) noexcept = default;

std::uint32_t VisionExecutor::soft_token_count() const {
  return impl_->soft_token_count();
}

std::uint32_t VisionExecutor::patch_rows() const { return impl_->patch_rows(); }

std::size_t VisionExecutor::scratch_bytes() const {
  return impl_->scratch_bytes();
}

void VisionExecutor::set_soft_token_count(std::uint32_t soft_token_count) {
  impl_->set_soft_token_count(soft_token_count);
}

void VisionExecutor::run(const vision_engine::PrefillRequest& request,
                         void* scratch_device,
                         std::size_t scratch_capacity_bytes,
                         cudaStream_t stream, CaptureSink* captures) const {
  impl_->run(request, scratch_device, scratch_capacity_bytes, stream,
             captures);
}

}  // namespace gewell::vision_executor
