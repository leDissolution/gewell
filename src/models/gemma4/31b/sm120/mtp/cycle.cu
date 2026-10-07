#include "gewell/mtp_cycle.h"

#include "gewell/mtp_assistant.h"
#include "gewell/models/gemma4/26b_a4b/model.h"
#include "cuda.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace gewell::mtp_cycle {
namespace {

namespace model = gemma4_31b;
using mtp_cuda::check;
constexpr std::size_t kVocabulary = model::kVocabSize;
constexpr std::size_t kMaskWords = mtp_sampling::mask_words(kVocabulary);

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::invalid_argument(std::string("MTP cycle: ") + message);
  }
}

std::uint32_t batch_capacity(std::uint32_t capacity, std::uint32_t depth,
                              void* staging, std::size_t bytes, std::size_t stride) {
  require(capacity > 0 && depth > 0 && depth <= mtp_target::kMaxDepth &&
              std::uint64_t(capacity) * (depth + 1) <= mtp_target::kMaxDepth + 1,
          "batch verification exceeds 1280 rows");
  require(staging && bytes / capacity >= stride,
          "batch staging is too small");
  return capacity;
}

class Event final {
 public:
  Event() { check(cudaEventCreate(&event_), "create MTP phase event"); }
  ~Event() { cudaEventDestroy(event_); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;

  void record(cudaStream_t stream) const {
    check(cudaEventRecord(event_, stream), "record MTP phase event");
  }

  float elapsed_until(const Event& end) const {
    float milliseconds{};
    check(cudaEventElapsedTime(&milliseconds, event_, end.event_),
          "measure MTP phase duration");
    return milliseconds;
  }

  void synchronize() const {
    check(cudaEventSynchronize(event_), "wait for MTP draft block");
  }

 private:
  cudaEvent_t event_{};
};

class PinnedWords final {
 public:
  explicit PinnedWords(std::size_t count) : bytes_(count * sizeof(std::uint32_t)) {
    check(cudaMallocHost(reinterpret_cast<void**>(&data_), bytes_),
          "allocate MTP constraint host scratch");
  }
  ~PinnedWords() { cudaFreeHost(data_); }
  PinnedWords(const PinnedWords&) = delete;
  PinnedWords& operator=(const PinnedWords&) = delete;
  std::uint32_t* data() const { return data_; }
  std::size_t size() const { return bytes_; }

 private:
  std::uint32_t* data_{};
  std::size_t bytes_{};
};

// Masks are contiguous across requests so a constrained batch needs one copy
// in each direction. The small pending IDs travel beside the proposal block.
struct ConstraintBuffers {
  PinnedWords tokens, masks;
  mtp_cuda::Buffer device_masks;
  Event copied;

  explicit ConstraintBuffers(std::uint32_t rows)
      : tokens(rows), masks(std::size_t(rows) * kMaskWords),
        device_masks(masks.size()) {}

  std::size_t host_bytes() const { return tokens.size() + masks.size(); }

  void download(const std::uint32_t* ids, std::uint32_t rows, cudaStream_t stream) {
    check(cudaMemcpyAsync(tokens.data(), ids, std::size_t(rows) * sizeof(*ids),
                          cudaMemcpyDeviceToHost, stream),
          "download MTP draft block for constraints");
    copied.record(stream);
  }

  void initialize(std::uint32_t rows) {
    copied.synchronize();
    std::fill_n(masks.data(), std::size_t(rows) * kMaskWords, UINT32_MAX);
  }

  void fill(const ConstraintMask& callback, std::uint32_t offset,
            std::uint32_t depth) {
    const auto* drafts = tokens.data() + offset + 1;
    auto* rows = masks.data() + std::size_t(offset) * kMaskWords;
    callback(drafts, depth, rows);
    // Verification stops at this first disallowed proposal, regardless of its
    // acceptance uniform. Later prefixes have no grammar state; never let
    // their masks create a spurious zero-mass failure in distribution setup.
    for (std::uint32_t index = 0; index < depth; ++index) {
      const auto token = drafts[index];
      require(token < kVocabulary, "constraint draft token exceeds vocabulary");
      if (!(rows[std::size_t(index) * kMaskWords + token / 32] &
            (std::uint32_t{1} << (token % 32)))) {
        std::fill_n(rows + std::size_t(index + 1) * kMaskWords,
                    std::size_t(depth - index) * kMaskWords, UINT32_MAX);
        break;
      }
    }
  }

  void upload(std::uint32_t rows, cudaStream_t stream) {
    check(cudaMemcpyAsync(device_masks.data(), masks.data(),
                          std::size_t(rows) * kMaskWords * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream),
          "upload MTP target constraint masks");
  }

  const std::uint32_t* row(std::uint32_t index) const {
    return device_masks.at<std::uint32_t>() + std::size_t(index) * kMaskWords;
  }
};

constexpr std::size_t global_k_norm_id() {
  for (std::size_t id = 0; id < model::kTextPhysicalTensorCount; ++id) {
    const auto& tensor = model::kPhysicalTensors[id];
    if (tensor.layer == 59 && tensor.role == model::TensorRole::k_norm) {
      return id;
    }
  }
  return model::kPhysicalTensorCount;
}
static_assert(global_k_norm_id() < model::kTextPhysicalTensorCount);
static_assert(!model::is_global_layer(58) && model::is_global_layer(59));

mtp_assistant::Weights assistant_weights(const mtp_target::Weights& weights) {
  mtp_assistant::Weights result{};
  result.target_embedding = weights[model::kEmbeddingPhysicalId];
  result.target_global_k_norm = weights[global_k_norm_id()];
  std::copy_n(weights.begin() + model::kAssistantEmbeddingPhysicalId,
              result.assistant.size(), result.assistant.begin());
  return result;
}

template<class Caches>
mtp_assistant::FrozenCache frozen_cache(const Caches& caches,
                                       std::uint32_t processed, unsigned local_layer = 58) {
  const auto& local = caches[local_layer];
  const auto& global = caches[local_layer + 1];
  mtp_assistant::FrozenCache frozen{};
  frozen.local_key = local.key;
  frozen.local_format = local.format;
  frozen.global_format = global.format;
  frozen.local_value = local.value;
  frozen.global = {global.page_pool,
                   global.page_offsets,
                   global.page_tokens,
                   global.page_count,
                   global.page_stride_elements,
                   global.layer_offset_elements, global.format};
  frozen.global_compact = global.page_pool == nullptr ? global.key : nullptr;
  frozen.global_capacity = global.capacity;
  frozen.processed_tokens = processed;
  return frozen;
}

struct Layout {
  std::size_t target_probs;
  std::size_t draft_probs;
  std::size_t sampling;
  std::size_t sampling_bytes;
  std::size_t input_ids;
  std::size_t output_ids;
  std::size_t greedy_ids;
  std::size_t assistant_logits;
  std::size_t feedback;
  std::size_t uniforms;
  std::size_t status;
  std::size_t result;
  std::size_t bytes{};

  explicit Layout(std::uint32_t depth, unsigned hidden_width = model::kHiddenSize) {
    require(depth > 0 && depth <= mtp_target::kMaxDepth,
            "maximum depth must be in 1..1279");
    const auto add = [this](std::size_t length) {
      require(length <= std::numeric_limits<std::size_t>::max() - bytes - 255,
              "device scratch size overflows");
      const std::size_t offset = bytes;
      bytes = mtp_cuda::align(bytes + length);
      return offset;
    };
    target_probs = add((static_cast<std::size_t>(depth) + 1) * kVocabulary *
                       sizeof(float));
    draft_probs = add(static_cast<std::size_t>(depth) * kVocabulary * sizeof(float));
    sampling_bytes = mtp_sampling::scratch_bytes(kVocabulary);
    sampling = add(sampling_bytes);
    input_ids = add((static_cast<std::size_t>(depth) + 1) * sizeof(std::uint32_t));
    output_ids = add((static_cast<std::size_t>(depth) + 1) * sizeof(std::uint32_t));
    greedy_ids = add((static_cast<std::size_t>(depth) + 1) * sizeof(std::uint32_t));
    assistant_logits = add(kVocabulary * sizeof(mtp_target::BFloat16));
    feedback = add(hidden_width * sizeof(mtp_target::BFloat16));
    uniforms = add((2 * static_cast<std::size_t>(depth) + 1) * sizeof(float));
    status = add(sizeof(mtp_sampling::Status));
    result = add(sizeof(mtp_sampling::Result));
  }
};

// Reuse the fixed worst-case arenas: bounded requests pack K token/probability
// pairs per row; unbounded and greedy requests retain V floats per row.
void build_probability_row(
    const mtp_target::BFloat16* logits, float temperature, float top_p,
    std::uint32_t top_k, float* rows, std::uint32_t row,
    std::uint32_t compact_size, void* scratch, std::size_t scratch_size,
    mtp_sampling::Status* status, cudaStream_t stream,
    const std::uint32_t* allowed = nullptr) {
  if (compact_size)
    mtp_sampling::build_compact_distribution(
        logits, kVocabulary, temperature, top_p, compact_size,
        reinterpret_cast<mtp_sampling::TokenProbability*>(rows) +
            std::size_t(row) * compact_size,
        scratch, scratch_size, status, stream, allowed);
  else
    mtp_sampling::build_distribution(
        logits, kVocabulary, temperature, top_p, top_k,
        rows + std::size_t(row) * kVocabulary, scratch, scratch_size,
        status, stream, allowed);
}

void sample_probability_row(
    const float* rows, std::uint32_t row, std::uint32_t compact_size,
    const float* uniform, std::uint32_t* output, void* scratch,
    std::size_t scratch_size, mtp_sampling::Status* status, cudaStream_t stream) {
  if (compact_size)
    mtp_sampling::sample_compact_distribution(
        reinterpret_cast<const mtp_sampling::TokenProbability*>(rows) +
            std::size_t(row) * compact_size,
        compact_size, kVocabulary, uniform, output, status, stream);
  else
    mtp_sampling::sample_distribution(rows + std::size_t(row) * kVocabulary,
        kVocabulary, uniform, output, scratch, scratch_size, status, stream);
}

void verify_probability_rows(
    const float* target, const float* draft, const std::uint32_t* draft_ids,
    std::uint32_t depth, std::uint32_t compact_size, const float* uniforms,
    const float* final_uniform, std::uint32_t* output,
    mtp_sampling::Result* result, void* scratch, std::size_t scratch_size,
    mtp_sampling::Status* status, cudaStream_t stream) {
  if (compact_size)
    mtp_sampling::verify_compact_sequence(
        reinterpret_cast<const mtp_sampling::TokenProbability*>(target),
        reinterpret_cast<const mtp_sampling::TokenProbability*>(draft),
        draft_ids, depth, compact_size, kVocabulary, uniforms, final_uniform,
        output, result, status, stream);
  else
    mtp_sampling::verify_sequence(target, draft, draft_ids, depth, kVocabulary,
        uniforms, final_uniform, output, result, scratch, scratch_size,
        status, stream);
}

// A failed earlier sample must not feed UINT_MAX/uninitialized data to a later
// embedding lookup while asynchronous work is still queued. Preserve the error
// and substitute a valid, unused token until the final status check aborts.
__global__ void copy_safe_token(const std::uint32_t* source,
                                std::uint32_t* destination,
                                mtp_sampling::Status* status) {
  const std::uint32_t token = *source;
  if (*status != mtp_sampling::Status::success) {
    *destination = 0;
  } else if (token >= model::kVocabSize) {
    *status = mtp_sampling::Status::invalid_draft_token;
    *destination = 0;
  } else {
    *destination = token;
  }
}

}  // namespace

struct Cycle::Impl {
  std::uint32_t context_capacity;
  std::uint32_t max_depth;
  void* staging;
  std::size_t staging_size;
  Layout layout;
  mtp_cuda::Buffer own;
  ConstraintBuffers constraints;
  mtp_assistant::Executor assistant;
  mtp_target::Verifier target;
  Event draft_begin;
  Event verify_begin;
  Event select_begin;
  Event select_end;
  bool ready_to_commit{};
  std::uint32_t last_position{};
  std::uint32_t selected_count{};

  Impl(cublasLtHandle_t handle, const mtp_target::Weights& weights,
       std::uint32_t capacity, std::uint32_t depth, void* stage,
       std::size_t stage_size, const nvfp4::Weights* native_weights,
       nvfp4::ActivationPolicy activation_policy, const fp8::Weights* fp8_weights,
       attention::Compute local_compute, attention::Compute global_compute)
      : context_capacity(capacity),
        max_depth(depth),
        staging(stage),
        staging_size(stage_size),
        layout(depth),
        own(layout.bytes),
        constraints(depth + 1),
        assistant(mtp_assistant::Model::gemma4_31b, handle, assistant_weights(weights), capacity, 1, local_compute, global_compute),
        target(handle, weights, depth + 1, capacity, native_weights,
               activation_policy, fp8_weights, local_compute, global_compute) {
    require(capacity > 0 && capacity <= 262'144,
            "context capacity is outside the model range");
    require(stage != nullptr &&
                stage_size >= mtp_target::Verifier::staging_bytes(depth + 1),
            "KV staging does not cover maximum verification width");
  }

  template <typename T>
  T* at(std::size_t offset) const {
    return own.at<T>(offset);
  }
};

Cycle::Cycle(cublasLtHandle_t handle, const mtp_target::Weights& weights,
             std::uint32_t context_capacity, std::uint32_t max_depth,
             void* staging, std::size_t staging_size,
             const nvfp4::Weights* native_weights,
             nvfp4::ActivationPolicy activation_policy,
             const fp8::Weights* fp8_weights,
             attention::Compute local_compute, attention::Compute global_compute)
    : impl_(std::make_unique<Impl>(handle, weights, context_capacity, max_depth,
                                   staging, staging_size, native_weights,
                                   activation_policy, fp8_weights, local_compute, global_compute)) {}

Cycle::~Cycle() = default;

std::size_t Cycle::scratch_bytes() const {
  return impl_->own.size() + impl_->assistant.scratch_bytes() +
         impl_->target.scratch_bytes() + impl_->constraints.device_masks.size();
}

std::size_t Cycle::host_scratch_bytes() const {
  return impl_->constraints.host_bytes();
}

void Cycle::prepare(std::uint32_t depth) {
  require(depth <= impl_->max_depth, "requested depth exceeds allocated capacity");
  impl_->target.prepare(depth + 1);
}

Outcome Cycle::run(const std::uint32_t* pending,
                   const mtp_target::BFloat16* target_hidden, std::uint32_t L,
                   std::uint32_t depth, const mtp_target::Caches& caches,
                   float temperature, float top_p, std::uint32_t top_k,
                   const std::vector<float>& uniforms, cudaStream_t stream,
                   const ConstraintMask& constraint_mask) {
  auto& state = *impl_;
  state.ready_to_commit = false;
  require(pending != nullptr && target_hidden != nullptr,
          "pending token and target hidden row are required");
  require(L > 0 && static_cast<std::uint64_t>(L) + depth + 1 <=
                         state.context_capacity,
          "verification exceeds remaining context capacity");
  require(std::isfinite(temperature) && temperature >= 0.0F &&
              std::isfinite(top_p) && top_p >= 0.0F && top_p <= 1.0F,
          "invalid sampling settings");
  require(uniforms.size() == 2 * static_cast<std::size_t>(depth) + 1,
          "uniform count must equal 2*N+1");
  for (const float uniform : uniforms) {
    require(std::isfinite(uniform) && uniform >= 0.0F && uniform < 1.0F,
            "uniform draw is outside [0,1)");
  }
  // Descriptor/host output allocations occur before the first proposal draw.
  prepare(depth);
  Outcome outcome;
  outcome.tokens.resize(static_cast<std::size_t>(depth) + 1);
  const auto& layout = state.layout;
  auto* status = state.at<mtp_sampling::Status>(layout.status);
  auto* input_ids = state.at<std::uint32_t>(layout.input_ids);
  auto* output_ids = state.at<std::uint32_t>(layout.output_ids);
  auto* device_uniforms = state.at<float>(layout.uniforms);
  auto* draft_probs = state.at<float>(layout.draft_probs);
  auto* target_probs = state.at<float>(layout.target_probs);
  const auto compact_size = mtp_sampling::compact_row_size(
      kVocabulary, temperature, top_p, top_k);
  auto* assistant_logits = state.at<mtp_target::BFloat16>(layout.assistant_logits);
  auto* feedback = state.at<mtp_target::BFloat16>(layout.feedback);
  void* sampling = state.at<void>(layout.sampling);
  state.draft_begin.record(stream);
  mtp_sampling::clear_status(status, stream);
  check(cudaMemcpyAsync(device_uniforms, uniforms.data(),
                         uniforms.size() * sizeof(float), cudaMemcpyHostToDevice,
                         stream),
         "upload MTP cycle uniforms");
  // pending may point into the previous cycle's outputs, so copy it first.
  copy_safe_token<<<1, 1, 0, stream>>>(pending, input_ids, status);
  check(cudaGetLastError(), "copy MTP pending token");
  check(cudaMemsetAsync(output_ids, 0,
                         outcome.tokens.size() * sizeof(std::uint32_t), stream),
         "initialize MTP cycle outputs");

  const auto frozen = frozen_cache(caches, L);
  for (std::uint32_t index = 0; index < depth; ++index) {
    state.assistant.forward(input_ids + index,
                            index == 0 ? target_hidden : feedback, frozen,
                            assistant_logits, feedback, stream);
    build_probability_row(
        assistant_logits, temperature, top_p, top_k, draft_probs, index,
        compact_size, sampling, layout.sampling_bytes, status, stream);
    sample_probability_row(
        draft_probs, index, compact_size, device_uniforms + index, input_ids + index + 1,
        sampling, layout.sampling_bytes, status, stream);
    copy_safe_token<<<1, 1, 0, stream>>>(input_ids + index + 1,
                                       input_ids + index + 1, status);
    check(cudaGetLastError(), "guard MTP sampled proposal");
  }

  if (constraint_mask) {
    state.constraints.download(input_ids, depth + 1, stream);
    outcome.constraint_draft_downloads = 1;
    outcome.constraint_draft_bytes = std::size_t(depth + 1) * sizeof(std::uint32_t);
  }
  state.verify_begin.record(stream);
  state.target.run(input_ids, L, depth + 1, caches, state.staging,
                    state.staging_size, stream);
  state.select_begin.record(stream);
  if (constraint_mask) {
    try {
      state.constraints.initialize(depth + 1);
      state.constraints.fill(constraint_mask, 0, depth);
      state.constraints.upload(depth + 1, stream);
    } catch (...) {
      // Target work is already queued against caller-owned caches/staging.
      // Complete it before propagating a host grammar failure to the caller.
      cudaStreamSynchronize(stream);
      throw;
    }
    outcome.constraint_mask_uploads = 1;
    outcome.constraint_mask_bytes =
        std::size_t(depth + 1) * kMaskWords * sizeof(std::uint32_t);
  }
  for (std::uint32_t row = 0; row <= depth; ++row) {
    const std::size_t offset = static_cast<std::size_t>(row) * kVocabulary;
    build_probability_row(
        state.target.logits() + offset, temperature, top_p, top_k,
        target_probs, row, compact_size, sampling, layout.sampling_bytes, status, stream,
        constraint_mask ? state.constraints.row(row) : nullptr);
  }
  verify_probability_rows(
      target_probs, depth ? draft_probs : nullptr,
      depth ? input_ids + 1 : nullptr, depth, compact_size,
      depth ? device_uniforms + depth : nullptr, device_uniforms + 2 * depth,
      output_ids, state.at<mtp_sampling::Result>(layout.result), sampling,
      layout.sampling_bytes, status, stream);
  state.select_end.record(stream);

  mtp_sampling::Status host_status{};
  check(cudaMemcpyAsync(&host_status, status, sizeof(host_status),
                         cudaMemcpyDeviceToHost, stream),
         "copy MTP cycle status");
  check(cudaMemcpyAsync(&outcome.verification,
                         state.at<mtp_sampling::Result>(layout.result),
                         sizeof(outcome.verification), cudaMemcpyDeviceToHost,
                         stream),
         "copy MTP verification metadata");
  check(cudaMemcpyAsync(outcome.tokens.data(), output_ids,
                         outcome.tokens.size() * sizeof(std::uint32_t),
                         cudaMemcpyDeviceToHost, stream),
         "copy bounded MTP output IDs");
  check(cudaStreamSynchronize(stream), "synchronize MTP cycle selection");
  if (host_status != mtp_sampling::Status::success) {
    throw std::runtime_error(std::string("MTP cycle sampling: ") +
                             mtp_sampling::status_message(host_status));
  }
  require(outcome.verification.output_count > 0 &&
              outcome.verification.output_count <= depth + 1,
          "verification returned an invalid output count");
  outcome.draft_gpu_milliseconds =
      state.draft_begin.elapsed_until(state.verify_begin);
  outcome.verify_gpu_milliseconds =
      state.verify_begin.elapsed_until(state.select_begin);
  outcome.select_gpu_milliseconds =
      state.select_begin.elapsed_until(state.select_end);
  outcome.tokens.resize(outcome.verification.output_count);
  state.ready_to_commit = true;
  state.last_position = L;
  state.selected_count = outcome.verification.output_count;
  return outcome;
}

void Cycle::commit(const mtp_target::Caches& caches, std::uint32_t L,
                    std::uint32_t count, cudaStream_t stream) {
  auto& state = *impl_;
  require(state.ready_to_commit && L == state.last_position && count > 0 &&
              count <= state.selected_count,
          "commit must consume the selected prefix of the latest cycle once");
  state.target.commit(caches, L, count, stream);
  state.ready_to_commit = false;
}

const mtp_target::BFloat16* Cycle::logits() const {
  return impl_->target.logits();
}

const mtp_target::BFloat16* Cycle::hidden() const {
  return impl_->target.hidden();
}

const std::uint32_t* Cycle::output_ids() const {
  return impl_->at<std::uint32_t>(impl_->layout.output_ids);
}

namespace {
class Target31 final : public BatchTarget {
 public:
  Target31(cublasLtHandle_t handle, const mtp_target::Weights& weights,
      unsigned rows, unsigned context, const nvfp4::Weights* native,
      nvfp4::ActivationPolicy policy, const fp8::Weights* fp8,
      attention::Compute local, attention::Compute global)
      : assistant_(mtp_cycle::assistant_weights(weights)),
        verifier_(handle, weights, rows, context, native, policy, fp8, local, global) {}
  mtp_assistant::Model model() const override { return mtp_assistant::Model::gemma4_31b; }
  mtp_assistant::Weights assistant_weights() const override { return assistant_; }
  std::size_t staging_bytes(std::uint32_t rows) const override { return mtp_target::Verifier::staging_bytes(rows); }
  std::size_t scratch_bytes() const override { return verifier_.scratch_bytes(); }
  void prepare(std::uint32_t rows) override { verifier_.prepare(rows); }
  void run_batch(const std::uint32_t* tokens, const std::vector<TargetInput>& inputs, cudaStream_t stream) override {
    std::vector<mtp_target::BatchInput> native;
    for (const auto& i : inputs) {
      require(i.caches.size() == 60, "31B target requires60 cache layers");
      mtp_target::Caches caches;std::copy(i.caches.begin(), i.caches.end(), caches.begin());
      native.push_back({i.base_position, i.rows, caches, i.staging, i.staging_size,
                        i.staging_capacity_rows, i.captures});
    }
    verifier_.run_batch(tokens, native, stream);
  }
  void commit_batch(const std::vector<TargetCommit>& inputs, cudaStream_t stream) override {
    std::vector<mtp_target::Caches> caches(inputs.size());
    std::vector<mtp_target::CommitInput> commits;
    for (std::size_t j=0;j<inputs.size();++j) {
      const auto& i=inputs[j];
      require(i.caches && i.caches->size()==60, "31B commit requires60 cache layers");
      std::copy(i.caches->begin(),i.caches->end(),caches[j].begin());
      commits.push_back({&caches[j],i.base_position,i.source_rows,i.capacity_rows,
                         i.staging,i.staging_size,i.commit_rows});
    }
    mtp_target::commit_staged_rows_batch(commits,stream);
  }
  const mtp_target::BFloat16* logits() const override { return verifier_.logits(); }
  const mtp_target::BFloat16* hidden() const override { return verifier_.hidden(); }
 private:
  mtp_assistant::Weights assistant_;
  mtp_target::Verifier verifier_;
};
}  // namespace

std::unique_ptr<BatchTarget> make_31b_batch_target(cublasLtHandle_t handle,
    const mtp_target::Weights& weights, std::uint32_t rows, std::uint32_t context,
    const nvfp4::Weights* native, nvfp4::ActivationPolicy policy, const fp8::Weights* fp8,
    attention::Compute local, attention::Compute global) {
  return std::make_unique<Target31>(handle,weights,rows,context,native,policy,fp8,local,global);
}

static std::unique_ptr<BatchTarget> checked_target(std::unique_ptr<BatchTarget> target) {
  require(target && (target->model()==mtp_assistant::Model::gemma4_31b ||
      target->model()==mtp_assistant::Model::gemma4_26b_a4b), "missing or invalid batch target");
  return target;
}

struct Batch::Impl {
  std::unique_ptr<BatchTarget> target;
  unsigned hidden_width, layer_count;
  std::uint32_t capacity, max_depth, context_capacity;
  void* staging;
  std::size_t stage_stride;
  Layout layout;
  mtp_cuda::Buffer own, tokens;
  // Allocated only when capture is requested; reused across sampled batches.
  std::unique_ptr<mtp_cuda::Buffer> capture, probe_capture;
  ConstraintBuffers constraints;
  mtp_assistant::Executor assistant;
  std::vector<TargetInput> verification;
  std::vector<std::uint32_t> offsets, selected, compact_sizes;
  Event draft_begin, verify_begin, select_begin, select_end;

  Impl(cublasLtHandle_t handle, std::unique_ptr<BatchTarget> backend,
       std::uint32_t context, std::uint32_t cap, std::uint32_t depth,
       void* stage, std::size_t bytes,
       attention::Compute local_compute, attention::Compute global_compute)
      : target(checked_target(std::move(backend))),
        hidden_width(target->model()==mtp_assistant::Model::gemma4_26b_a4b ? gemma4_26b_a4b::kHiddenSize : model::kHiddenSize),
        layer_count(hidden_width==gemma4_26b_a4b::kHiddenSize ? 30 : 60),
        capacity(batch_capacity(cap, depth, stage, bytes, target->staging_bytes(depth+1))),
        max_depth(depth), context_capacity(context), staging(stage),
        stage_stride(target->staging_bytes(depth+1)), layout(depth, hidden_width),
        own(std::size_t(cap) * layout.bytes),
        tokens(std::size_t(cap) * (depth + 1) * sizeof(std::uint32_t)),
        constraints(cap * (depth + 1)),
        assistant(target->model(),
            handle,target->assistant_weights(),context,cap,local_compute,global_compute) {}

  template <typename T> T* at(std::size_t request, std::size_t offset) const {
    return own.at<T>(request * layout.bytes + offset);
  }
};

Batch::Batch(cublasLtHandle_t handle, std::unique_ptr<BatchTarget> target,
             std::uint32_t context_capacity, std::uint32_t capacity,
             std::uint32_t max_depth, void* staging, std::size_t staging_size,
             attention::Compute local_compute, attention::Compute global_compute)
    : impl_(std::make_unique<Impl>(handle, std::move(target), context_capacity, capacity,
                                   max_depth, staging, staging_size, local_compute, global_compute)) {}
Batch::~Batch() = default;

std::size_t Batch::scratch_bytes() const {
  return impl_->own.size() + impl_->tokens.size() + impl_->assistant.scratch_bytes() +
      impl_->target->scratch_bytes() + impl_->constraints.device_masks.size() +
      (impl_->capture ? impl_->capture->size() : 0) +
      (impl_->probe_capture ? impl_->probe_capture->size() : 0);
}

std::size_t Batch::host_scratch_bytes() const {
  return impl_->constraints.host_bytes();
}

BatchOutcome Batch::run(const std::vector<BatchInput>& inputs, cudaStream_t stream) {
  auto& s = *impl_;
  const auto& l = s.layout;
  require(!inputs.empty() && inputs.size() <= s.capacity, "invalid active batch size");
  s.verification.clear();
  s.offsets.clear();
  s.selected.assign(inputs.size(), 0);
  s.compact_sizes.resize(inputs.size());
  BatchOutcome outcome;
  outcome.requests.resize(inputs.size());
  std::vector<mtp_sampling::Status> statuses(inputs.size());
  std::vector<std::size_t> capture_offsets(inputs.size());
  std::size_t capture_rows = 0, probe_elements = 0, probe_outputs = 0;
  std::vector<std::size_t> probe_offsets(inputs.size()), probe_output_offsets(inputs.size());
  std::vector<std::size_t> head_probe_offsets(inputs.size());
  std::uint32_t total_rows = 0;
  bool constrained = false;
  bool greedy = true;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto& input = inputs[i];
    constrained |= bool(input.constraint_mask);
    s.compact_sizes[i] = mtp_sampling::compact_row_size(
        kVocabulary, input.temperature, input.top_p, input.top_k);
    greedy &= input.temperature == 0.0F || input.top_p == 0.0F ||
              input.top_k == 1;
    require(input.caches.size() == s.layer_count, "batch cache layer count does not match target");
    require(input.target_hidden && input.pending_token < kVocabulary &&
                input.depth <= s.max_depth && input.position > 0 &&
                std::uint64_t(input.position) + input.depth + 1 <= s.context_capacity,
            "batch request exceeds token/context capacity");
    require(std::isfinite(input.temperature) && input.temperature >= 0 &&
                std::isfinite(input.top_p) && input.top_p >= 0 && input.top_p <= 1 &&
                input.uniforms.size() == 2 * std::size_t(input.depth) + 1,
            "invalid batch sampling settings");
    for (const auto u : input.uniforms)
      require(std::isfinite(u) && u >= 0 && u < 1, "batch uniform outside [0,1)");
    s.offsets.push_back(total_rows);
    total_rows += input.depth + 1;
    s.verification.push_back({input.position, input.depth + 1, input.caches,
        static_cast<unsigned char*>(s.staging) + i * s.stage_stride,
        s.stage_stride, s.max_depth + 1});
    outcome.requests[i].tokens.resize(input.depth + 1);
    if (input.capture_next) {
      auto& probes = *input.capture_next;
      require(!probes.layers.empty(), "target probe layers are empty");
      probes.width = s.hidden_width;
      probes.hidden.resize(probes.layers.size() * probes.width);
      probe_offsets[i] = probe_elements;
      probe_output_offsets[i] = probe_outputs;
      probe_elements += (input.depth + 1) * probes.hidden.size();
      probe_outputs += probes.hidden.size();
    }
    if (!input.depth_probes.layers.empty()) {
      require(input.depth_probes.output, "missing depth head probe destination");
      head_probe_offsets[i] = probe_elements;
      probe_elements += (input.depth + 1) * input.depth_probes.layers.size() * s.hidden_width;
    }
    if (input.capture) {
      require(input.depth > 0, "capture requires a positive proposal depth");
      capture_offsets[i] = capture_rows;
      capture_rows += input.depth;
      auto& features = *input.capture;
      features.target_width = s.hidden_width;
      features.assistant_width = model::kAssistantHiddenSize;
      features.target_hidden.resize(features.target_width);
      features.assistant_hidden.resize(std::size_t(input.depth) * features.assistant_width);
      features.draft_tokens.resize(input.depth);
      features.scores.resize(input.depth);
    }
  }
  const auto score_offset = mtp_cuda::align(capture_rows * model::kAssistantHiddenSize * sizeof(mtp_target::BFloat16));
  const auto capture_bytes = score_offset + capture_rows * sizeof(MtpCaptureScores);
  if (capture_bytes && (!s.capture || s.capture->size() < capture_bytes))
    s.capture = std::make_unique<mtp_cuda::Buffer>(capture_bytes);
  const auto probe_output_base = mtp_cuda::align(probe_elements * sizeof(std::uint16_t));
  const auto probe_bytes = probe_output_base + probe_outputs * sizeof(std::uint16_t);
  if (probe_bytes && (!s.probe_capture || s.probe_capture->size() < probe_bytes))
    s.probe_capture = std::make_unique<mtp_cuda::Buffer>(probe_bytes);
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    if (const auto* probes = inputs[i].capture_next)
      for (std::size_t j = 0; j < probes->layers.size(); ++j)
        s.verification[i].captures.push_back({probes->layers[j],
            s.probe_capture->at<mtp_target::BFloat16>() + probe_offsets[i] +
                j * (inputs[i].depth + 1) * probes->width});
    for (std::size_t j = 0; j < inputs[i].depth_probes.layers.size(); ++j)
      s.verification[i].captures.push_back({inputs[i].depth_probes.layers[j],
          s.probe_capture->at<mtp_target::BFloat16>() + head_probe_offsets[i] +
              j * (inputs[i].depth + 1) * s.hidden_width});
  }
  s.target->prepare(total_rows);
  s.draft_begin.record(stream);
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto& input = inputs[i];
    auto* status = s.at<mtp_sampling::Status>(i, l.status);
    auto* ids = s.tokens.at<std::uint32_t>() + s.offsets[i];
    auto* uniforms = s.at<float>(i, l.uniforms);
    mtp_sampling::clear_status(status, stream);
    check(cudaMemcpyAsync(ids, &input.pending_token, sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, stream), "upload batch pending token");
    check(cudaMemcpyAsync(uniforms, input.uniforms.data(), input.uniforms.size() * sizeof(float),
                          cudaMemcpyHostToDevice, stream), "upload batch uniforms");
    check(cudaMemsetAsync(s.at<void>(i, l.output_ids), 0,
                          (input.depth + 1) * sizeof(std::uint32_t), stream),
          "clear batch speculative outputs");
  }
  std::vector<mtp_assistant::Input> drafts;
  std::vector<mtp_sampling::GreedyDistributionInput> greedy_distributions;
  std::vector<mtp_sampling::GreedyVerificationInput> greedy_verifications;
  std::vector<mtp_sampling::CompactDistributionInput> compact_distributions;
  drafts.reserve(inputs.size());
  greedy_distributions.reserve(total_rows);
  greedy_verifications.reserve(inputs.size());
  compact_distributions.reserve(inputs.size());
  for (std::uint32_t step = 0; step < s.max_depth; ++step) {
    drafts.clear();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const auto& input = inputs[i];
      if (step >= input.depth) continue;
      auto* feedback = s.at<mtp_target::BFloat16>(i, l.feedback);
      drafts.push_back({s.tokens.at<std::uint32_t>() + s.offsets[i] + step,
          step ? feedback : input.target_hidden, frozen_cache(input.caches, input.position, s.layer_count - 2),
          s.at<mtp_target::BFloat16>(i, l.assistant_logits), feedback,
          input.capture ? s.capture->at<mtp_target::BFloat16>() +
              (capture_offsets[i] + step) * model::kAssistantHiddenSize : nullptr});
    }
    if (drafts.empty()) break;
    s.assistant.forward_batch(drafts, stream);
    if (greedy) {
      greedy_distributions.clear();
      for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (step >= inputs[i].depth) continue;
        auto* ids = s.tokens.at<std::uint32_t>() + s.offsets[i];
        greedy_distributions.push_back({
            s.at<mtp_target::BFloat16>(i, l.assistant_logits),
            nullptr, nullptr, s.at<mtp_sampling::Status>(i, l.status),
            s.at<std::uint32_t>(i, l.greedy_ids) + step,
            ids + step + 1});
      }
      mtp_sampling::build_greedy_distributions(
          greedy_distributions, kVocabulary, stream);
    } else {
      compact_distributions.clear();
      for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto& input = inputs[i];
        if (step >= input.depth || !s.compact_sizes[i]) continue;
        compact_distributions.push_back({
            s.at<mtp_target::BFloat16>(i, l.assistant_logits),
            input.temperature, input.top_p, s.compact_sizes[i],
            s.at<mtp_sampling::TokenProbability>(i, l.draft_probs) +
                std::size_t(step) * s.compact_sizes[i],
            s.at<void>(i, l.sampling), l.sampling_bytes,
            s.at<mtp_sampling::Status>(i, l.status), nullptr});
      }
      if (!compact_distributions.empty())
        mtp_sampling::build_compact_distributions(compact_distributions,
                                                  kVocabulary, stream);
      for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto& input = inputs[i];
        if (step >= input.depth) continue;
        auto* status = s.at<mtp_sampling::Status>(i, l.status);
        auto* ids = s.tokens.at<std::uint32_t>() + s.offsets[i];
        auto* uniforms = s.at<float>(i, l.uniforms);
        void* sampling = s.at<void>(i, l.sampling);
        auto* probabilities = s.at<float>(i, l.draft_probs);
        if (!s.compact_sizes[i]) build_probability_row(
            s.at<mtp_target::BFloat16>(i, l.assistant_logits),
            input.temperature, input.top_p, input.top_k, probabilities, step,
            s.compact_sizes[i], sampling, l.sampling_bytes, status, stream);
        sample_probability_row(
            probabilities, step, s.compact_sizes[i], uniforms + step, ids + step + 1,
            sampling, l.sampling_bytes, status, stream);
        copy_safe_token<<<1, 1, 0, stream>>>(
            ids + step + 1, ids + step + 1, status);
        check(cudaGetLastError(), "guard batch MTP proposal");
      }
    }
  }
  if (constrained) {
    s.constraints.download(s.tokens.at<std::uint32_t>(), total_rows, stream);
    outcome.constraint_draft_downloads = 1;
    outcome.constraint_draft_bytes = std::size_t(total_rows) * sizeof(std::uint32_t);
  }
  s.verify_begin.record(stream);
  s.target->run_batch(s.tokens.at<std::uint32_t>(), s.verification, stream);
  s.select_begin.record(stream);
  if (constrained) {
    try {
      s.constraints.initialize(total_rows);
      for (std::size_t i = 0; i < inputs.size(); ++i)
        if (inputs[i].constraint_mask)
          s.constraints.fill(inputs[i].constraint_mask, s.offsets[i], inputs[i].depth);
      s.constraints.upload(total_rows, stream);
    } catch (...) {
      cudaStreamSynchronize(stream);
      throw;
    }
    outcome.constraint_mask_uploads = 1;
    outcome.constraint_mask_bytes =
        std::size_t(total_rows) * kMaskWords * sizeof(std::uint32_t);
  }
  if (greedy) {
    greedy_distributions.clear();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const auto& input = inputs[i];
      for (std::uint32_t row = 0; row <= input.depth; ++row) {
        greedy_distributions.push_back({
            logits(i) + std::size_t(row) * kVocabulary,
            input.return_probabilities
                ? s.at<float>(i, l.target_probs) +
                      std::size_t(row) * kVocabulary
                : nullptr,
            input.constraint_mask
                ? s.constraints.row(s.offsets[i] + row)
                : nullptr,
            s.at<mtp_sampling::Status>(i, l.status),
            s.at<std::uint32_t>(i, l.greedy_ids) + row,
            s.at<std::uint32_t>(i, l.output_ids) + row});
      }
    }
    mtp_sampling::build_greedy_distributions(
        greedy_distributions, kVocabulary, stream);
    greedy_verifications.clear();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const auto depth = inputs[i].depth;
      auto* uniforms = s.at<float>(i, l.uniforms);
      greedy_verifications.push_back({
          s.at<std::uint32_t>(i, l.greedy_ids),
          depth ? s.tokens.at<std::uint32_t>() + s.offsets[i] + 1 : nullptr,
          depth ? uniforms + depth : nullptr, uniforms + 2 * depth, depth,
          s.at<std::uint32_t>(i, l.output_ids),
          s.at<mtp_sampling::Result>(i, l.result),
          s.at<mtp_sampling::Status>(i, l.status)});
    }
    mtp_sampling::verify_greedy_sequences(greedy_verifications, kVocabulary,
                                          stream);
  }
  if (!greedy) {
    // Rows of one request share scratch. Batch across requests at each row,
    // then reuse that scratch only after the previous row's kernels finish.
    for (std::uint32_t row = 0; row <= s.max_depth; ++row) {
      compact_distributions.clear();
      for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto& input = inputs[i];
        if (row > input.depth || !s.compact_sizes[i]) continue;
        compact_distributions.push_back({
            logits(i) + std::size_t(row) * kVocabulary,
            input.temperature, input.top_p, s.compact_sizes[i],
            s.at<mtp_sampling::TokenProbability>(i, l.target_probs) +
                std::size_t(row) * s.compact_sizes[i],
            s.at<void>(i, l.sampling), l.sampling_bytes,
            s.at<mtp_sampling::Status>(i, l.status),
            input.constraint_mask ? s.constraints.row(s.offsets[i] + row) : nullptr});
      }
      if (compact_distributions.empty()) break;
      mtp_sampling::build_compact_distributions(compact_distributions,
                                                kVocabulary, stream);
    }
  }
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto& input = inputs[i];
    auto* status = s.at<mtp_sampling::Status>(i, l.status);
    auto* target_probs = s.at<float>(i, l.target_probs);
    auto* uniforms = s.at<float>(i, l.uniforms);
    void* sampling = s.at<void>(i, l.sampling);
    if (!greedy && !s.compact_sizes[i])
      for (std::uint32_t row = 0; row <= input.depth; ++row)
        build_probability_row(
            logits(i) + std::size_t(row) * kVocabulary,
            input.temperature, input.top_p, input.top_k,
            target_probs, row, s.compact_sizes[i], sampling,
            l.sampling_bytes, status, stream,
            input.constraint_mask
                ? s.constraints.row(s.offsets[i] + row)
                : nullptr);
    if (!greedy)
      verify_probability_rows(target_probs,
          s.at<float>(i, l.draft_probs),
          s.tokens.at<std::uint32_t>() + s.offsets[i] + 1, input.depth,
          s.compact_sizes[i], uniforms + input.depth, uniforms + 2 * input.depth,
          s.at<std::uint32_t>(i, l.output_ids),
          s.at<mtp_sampling::Result>(i, l.result), sampling,
          l.sampling_bytes, status, stream);
  }
  s.select_end.record(stream);
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    auto& result = outcome.requests[i];
    if (auto* probes = inputs[i].capture_next) {
      auto* selected_probes = s.probe_capture->at<std::uint16_t>(probe_output_base) + probe_output_offsets[i];
      mtp_sampling::gather_capture_rows(s.probe_capture->at<std::uint16_t>() + probe_offsets[i],
          probes->layers.size(), inputs[i].depth + 1, probes->width,
          s.at<mtp_sampling::Result>(i, l.result), s.at<mtp_sampling::Status>(i, l.status), selected_probes, stream);
      check(cudaMemcpyAsync(probes->hidden.data(), selected_probes, probes->hidden.size() * sizeof(std::uint16_t),
          cudaMemcpyDeviceToHost, stream), "capture selected target layer probes");
    }
    if (!inputs[i].depth_probes.layers.empty())
      mtp_sampling::gather_capture_rows(s.probe_capture->at<std::uint16_t>() + head_probe_offsets[i],
          inputs[i].depth_probes.layers.size(), inputs[i].depth + 1, s.hidden_width,
          s.at<mtp_sampling::Result>(i, l.result), s.at<mtp_sampling::Status>(i, l.status),
          reinterpret_cast<std::uint16_t*>(inputs[i].depth_probes.output), stream);
    if (auto* features = inputs[i].capture) {
      auto* scores = s.capture->at<MtpCaptureScores>(score_offset) + capture_offsets[i];
      mtp_sampling::capture_scores(s.at<void>(i, l.target_probs), s.at<void>(i, l.draft_probs),
          s.tokens.at<std::uint32_t>() + s.offsets[i] + 1, inputs[i].depth, kVocabulary,
          s.compact_sizes[i], greedy ? s.at<std::uint32_t>(i, l.greedy_ids) : nullptr,
          s.at<mtp_sampling::Status>(i, l.status), scores, stream);
      check(cudaMemcpyAsync(features->target_hidden.data(), inputs[i].target_hidden,
          features->target_hidden.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost, stream),
          "capture target hidden");
      check(cudaMemcpyAsync(features->assistant_hidden.data(),
          s.capture->at<mtp_target::BFloat16>() + capture_offsets[i] * model::kAssistantHiddenSize,
          features->assistant_hidden.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost, stream),
          "capture assistant hidden");
      check(cudaMemcpyAsync(features->draft_tokens.data(), s.tokens.at<std::uint32_t>() + s.offsets[i] + 1,
          inputs[i].depth * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream), "capture draft IDs");
      check(cudaMemcpyAsync(features->scores.data(), scores, inputs[i].depth * sizeof(MtpCaptureScores),
          cudaMemcpyDeviceToHost, stream), "capture probability scores");
    }
    check(cudaMemcpyAsync(&statuses[i], s.at<void>(i, l.status), sizeof(statuses[i]),
                          cudaMemcpyDeviceToHost, stream), "copy batch MTP status");
    check(cudaMemcpyAsync(&result.verification, s.at<void>(i, l.result),
                          sizeof(result.verification), cudaMemcpyDeviceToHost, stream),
          "copy batch MTP verification");
    check(cudaMemcpyAsync(result.tokens.data(), s.at<void>(i, l.output_ids),
                          result.tokens.size() * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost, stream), "copy batch MTP outputs");
  }
  check(cudaStreamSynchronize(stream), "complete batch MTP selection");
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    auto& result = outcome.requests[i];
    result.status = statuses[i];
    if (result.status != mtp_sampling::Status::success) {
      result.verification = {};
      result.tokens.clear();
      continue;
    }
    require(result.verification.output_count > 0 &&
                result.verification.output_count <= result.tokens.size(),
            "invalid batch MTP output count");
    result.tokens.resize(result.verification.output_count);
  }
  for (std::size_t i = 0; i < inputs.size(); ++i)
    s.selected[i] = outcome.requests[i].verification.output_count;
  outcome.draft_gpu_milliseconds = s.draft_begin.elapsed_until(s.verify_begin);
  outcome.verify_gpu_milliseconds = s.verify_begin.elapsed_until(s.select_begin);
  outcome.select_gpu_milliseconds = s.select_begin.elapsed_until(s.select_end);
  return outcome;
}

void Batch::commit_batch(const std::vector<BatchCommitInput>& inputs,
                         cudaStream_t stream) {
  auto& s = *impl_;
  require(!inputs.empty(), "batch commit is empty");
  std::vector<TargetCommit> commits;
  std::vector<bool> seen(s.selected.size());
  commits.reserve(inputs.size());
  for (const auto& input : inputs) {
    require(input.request < s.selected.size() && input.caches &&
                !seen[input.request] &&
                input.count > 0 &&
                input.count <= s.selected[input.request],
            "batch commit must consume each selected prefix once");
    seen[input.request] = true;
    const auto& verified = s.verification[input.request];
    commits.push_back({input.caches, verified.base_position, verified.rows,
        verified.staging_capacity_rows, verified.staging,
        verified.staging_size, input.count});
  }
  s.target->commit_batch(commits, stream);
  for (const auto& input : inputs) s.selected[input.request] = 0;
}

const mtp_target::BFloat16* Batch::logits(std::uint32_t request) const {
  return impl_->target->logits() + std::size_t(impl_->offsets.at(request)) * kVocabulary;
}
void Batch::summarize_logprobs(std::uint32_t request, std::uint32_t row_count,
                              std::uint32_t top_logprobs,
                              TokenLogprobs* output, cudaStream_t stream) const {
  require(request < impl_->verification.size() && row_count > 0 &&
              row_count <= impl_->verification[request].rows,
          "logprob rows exceed latest MTP verification");
  const auto* probabilities = impl_->at<float>(request, impl_->layout.target_probs);
  const auto compact_size = impl_->compact_sizes.at(request);
  if (compact_size)
    mtp_sampling::summarize_compact_logprobs(
        reinterpret_cast<const mtp_sampling::TokenProbability*>(probabilities),
        output_ids(request), row_count, compact_size, top_logprobs, output, stream);
  else
    mtp_sampling::summarize_logprobs(probabilities, output_ids(request), row_count,
        kVocabulary, top_logprobs, output, stream);
}
const mtp_target::BFloat16* Batch::hidden(std::uint32_t request) const {
  return impl_->target->hidden() + std::size_t(impl_->offsets.at(request)) * impl_->hidden_width;
}
const std::uint32_t* Batch::output_ids(std::uint32_t request) const {
  (void)impl_->offsets.at(request);
  return impl_->at<std::uint32_t>(request, impl_->layout.output_ids);
}

}  // namespace gewell::mtp_cycle
