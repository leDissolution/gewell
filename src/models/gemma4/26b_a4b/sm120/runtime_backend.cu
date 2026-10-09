#include "runtime_backend.h"
#include "batched_executor.cuh"
#include "mtp_target.cuh"
#include "mtp_staging.cuh"
#include "gewell/models/gemma4/26b_a4b/component_weights.h"
#include "cuda_memory.cuh"
#include "runtime/physical_cache.h"
#include "gewell/models/gemma4/26b_a4b/cache_config.h"
#include "gewell/runtime/scheduler.h"
#include "gewell/mtp_sampling.h"
#include "gewell/vision_executor.h"
#include "gewell/console.h"

#include <cmath>
#include <cstring>
#include <optional>

namespace gewell::gemma4_26b_a4b::sm120 {
namespace {
using namespace runtime;
using cuda_detail::check_cuda;
using Device = cuda_detail::DeviceAllocation;
using Host = cuda_detail::PinnedHostAllocation;
using BF16 = __nv_bfloat16;
constexpr std::size_t kHiddenBytes = kHiddenSize * sizeof(BF16);
constexpr unsigned kMaxRows = 4096;
constexpr unsigned kMaxVerifierRows = 1280;
void require(bool value, const char* message) {
  if (!value) throw std::invalid_argument(std::string("26B runtime: ") + message);
}
struct Stream {
  cudaStream_t value{};
  Stream() { check_cuda(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "create 26B stream"); }
  ~Stream() { cudaStreamDestroy(value); }
};
struct Event {
  cudaEvent_t value{};
  Event() { check_cuda(cudaEventCreate(&value), "create 26B event"); }
  ~Event() { cudaEventDestroy(value); }
};
struct BlasLt {
  cublasLtHandle_t value{};
  BlasLt() {
    if (cublasLtCreate(&value) != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("create 26B MTP cuBLASLt handle");
  }
  ~BlasLt() { cublasLtDestroy(value); }
};
struct SampleInput {
  float uniform;
  mtp_sampling::Status status;
  std::uint32_t mask[mtp_sampling::mask_words(kVocabSize)];
};

class RuntimeBackend final : public ExecutionBackend {
 public:
  RuntimeBackend(ArtifactFile weights, const BatchLimits& requested, nvfp4::ActivationPolicy policy,
                 const std::string& assistant_path, const std::string& vision_path, const std::string& qdq_path)
      : weights_(std::move(weights)), qdq_(qdq_path.empty() ? QdqMask{} : QdqMask::Load(qdq_path)),
        policy_(policy), capacity_(requested.capacity),
        mtp_depth_(requested.mtp_depth) {
    qdq_.validate(weights_);
    require(capacity_ && capacity_ <= kMaxVerifierRows && requested.prefill_batch_tokens <= kMaxRows,
            "invalid batch capacity");
    require(mtp_depth_ < kMaxVerifierRows &&
            std::uint64_t(capacity_) * (mtp_depth_ + 1) <= kMaxVerifierRows,
            "batch size times (depth + 1) must not exceed 1280 verifier rows");
    max_rows_ = std::max(capacity_ * (mtp_depth_ + 1), requested.prefill_batch_tokens);
    require(requested.mtp_head.path.empty(), "learned MTP depth controller is not supported");
    require(!mtp_depth_ || !assistant_path.empty(), "MTP requires an assistant component");
    if (!assistant_path.empty()) {
      auto component = std::make_unique<component::File>(assistant_path, assistant_tensor_specs());
      if (mtp_depth_) assistant_file_ = std::move(component);
    }
    if (!vision_path.empty()) {
      vision_file_ = std::make_unique<component::File>(vision_path, vision_tensor_specs());
      max_rows_ = std::max(max_rows_, 1120U);
    }
    config_ = kv_cache::compact_pool_config(kCacheGeometry, requested.kv_bytes,
        requested.cpu_bytes, requested.index_bytes, requested.local_kv_format, requested.global_kv_format);
    memory_.hidden_slots = requested.kv_bytes / config_.local_ring_bytes;
    memory_.hidden_staging_bytes = memory_.hidden_slots * kHiddenBytes;
    memory_.staging_bytes = mtp_depth_ ? capacity_ * mtp_staging_bytes(mtp_depth_ + 1) : 0;
    require(memory_.hidden_slots && memory_.staging_bytes < requested.kv_bytes &&
            memory_.hidden_staging_bytes < requested.kv_bytes - memory_.staging_bytes,
            "KV budget cannot reserve MTP staging and terminal states");
    memory_.committed_kv_bytes = requested.kv_bytes - memory_.staging_bytes - memory_.hidden_staging_bytes;
    config_.gpu_bytes = memory_.committed_kv_bytes;
    used_.resize(memory_.hidden_slots);
    pending_.resize(capacity_);
  }
  ~RuntimeBackend() override { wait(); }
  const BackendLimits& limits() const override { return limits_; }
  BatchMemoryPlan memory_plan() const override { return memory_; }
  kv_cache::PoolConfig cache_config() const override { return config_; }
  CacheStorageFactory cache_storage_factory() const override {
    return [](const auto& config, auto& ledger, std::size_t offsets) {
      return std::make_unique<PhysicalCache>(kCacheGeometry, config, ledger, offsets);
    };
  }
  void allocate_staging() override {
    hidden_ = std::make_unique<Device>(memory_.hidden_staging_bytes);
    if (memory_.staging_bytes) staging_ = std::make_unique<Device>(memory_.staging_bytes);
  }
  void initialize(PersistentCacheManager& cache, kv_cache::ExecutionId, const BatchLimits& requested) override {
    cache_ = &cache;
    executor_ = std::make_unique<BatchedExecutor>(std::move(weights_), max_rows_, stream_.value, policy_,
        requested.local_attention_compute, requested.global_attention_compute);
    const auto overlay = executor_->apply_qdq(qdq_);
    console::field("weight_qdq_mask_sha256", qdq_.has_source() ? artifact::digest_hex(qdq_.source_sha256()) : "disabled");
    console::field("weight_qdq_seconds", overlay.seconds);
    console::field("nvfp4_activation_policy", policy_ == nvfp4::ActivationPolicy::always ? "always" : "prefill");
    console::field("attention_local_compute", attention::compute_name(requested.local_attention_compute));
    console::field("attention_global_compute", attention::compute_name(requested.global_attention_compute));
    console::field("kv_local_format", kv_cache::format_name(requested.local_kv_format));
    console::field("kv_global_format", kv_cache::format_name(requested.global_kv_format));
    const auto& selection = overlay.selection;
    console::field("weight_qdq_bf16_projection_tensors", selection.bf16.tensor_count);
    console::field("weight_qdq_fp8_projection_tensors", selection.fp8.tensor_count);
    console::field("weight_qdq_nvfp4_projection_tensors", selection.nvfp4.tensor_count);
    console::field("weight_native_fp8_w8a8_projection_tensors", selection.fp8_w8a8.tensor_count);
    console::field("weight_native_nvfp4_w4a4_projection_tensors", selection.nvfp4_w4a4.tensor_count);
    tokens_ = std::make_unique<Device>(max_rows_ * sizeof(std::uint32_t));
    host_tokens_ = std::make_unique<Host>(tokens_->size());
    head_hidden_ = std::make_unique<Device>(kHiddenBytes);
    logits_ = std::make_unique<Device>(std::size_t(capacity_) * limits_.logit_row_bytes);
    ids_ = std::make_unique<Device>(capacity_ * sizeof(std::uint32_t));
    host_ids_ = std::make_unique<Host>(ids_->size());
    sample_ = std::make_unique<Device>(capacity_ * sizeof(SampleInput));
    host_sample_ = std::make_unique<Host>(sample_->size());
    probs_ = std::make_unique<Device>(kVocabSize * sizeof(float));
    sampling_ = std::make_unique<Device>(mtp_sampling::scratch_bytes(kVocabSize));
    best_ = std::make_unique<Device>(capacity_ * sizeof(std::uint32_t));
    if (vision_file_) {
      vision_ = std::make_unique<Device>(vision_file_->device_bytes());
      std::size_t offset = 0;
      for (const auto& tensor : vision_file_->tensors()) {
        check_cuda(cudaMemcpy(static_cast<std::uint8_t*>(vision_->data()) + offset,
            tensor.data, tensor.bytes, cudaMemcpyHostToDevice), "upload26B vision component");
        offset += (tensor.bytes + 4095) / 4096 * 4096;
      }
      vision_file_.reset();
    }
    if (mtp_depth_) {
      assistant_ = std::make_unique<Device>(assistant_file_->device_bytes());
      std::array<const BF16*, kAssistantTensorCount> views{};
      std::size_t offset = 0;
      for (const auto& tensor : assistant_file_->tensors()) {
        auto* destination = static_cast<std::uint8_t*>(assistant_->data()) + offset;
        check_cuda(cudaMemcpy(destination, tensor.data, tensor.bytes, cudaMemcpyHostToDevice),
                   "upload 26B assistant component");
        views[tensor.physical_id] = reinterpret_cast<const BF16*>(destination);
        offset += (tensor.bytes + 4095) / 4096 * 4096;
      }
      assistant_file_.reset();
      mtp_handle_ = std::make_unique<BlasLt>();
      mtp_ = std::make_unique<mtp_cycle::Batch>(mtp_handle_->value,
          make_mtp_target(*executor_, views, capacity_ * (mtp_depth_ + 1), stream_.value),
          requested.max_horizon, capacity_, mtp_depth_, staging_->data(), staging_->size(),
          requested.local_attention_compute, requested.global_attention_compute);
    }
  }
  void initialize_outputs(bool captures, bool logprobs) override {
    const auto rows = std::size_t(capacity_) * (mtp_depth_ + 1);
    if (captures) host_logits_ = std::make_unique<Host>(rows * limits_.logit_row_bytes);
    if (logprobs) {
      logprobs_ = std::make_unique<Device>(rows * sizeof(TokenLogprobs));
      host_logprobs_ = std::make_unique<Host>(rows * sizeof(TokenLogprobs));
    }
    report_memory();
  }
  CompletionContext context() const override { return CompletionContext(stream_.value); }
  void wait() const override { cudaStreamSynchronize(stream_.value); }
  void synchronize(std::string_view op) const override { check_cuda(cudaStreamSynchronize(stream_.value), op); }
  bool ready(std::string_view op) const override {
    auto status = cudaStreamQuery(stream_.value);
    if (status == cudaErrorNotReady) return false;
    check_cuda(status, op);
    return true;
  }
  TerminalState acquire_hidden() override {
    for (std::size_t i = 0; i < used_.size(); ++i) if (!used_[i]) {
      used_[i] = true;
      return TerminalState(static_cast<BF16*>(hidden_->data()) + i * kHiddenSize);
    }
    throw std::runtime_error("26B runtime: no free terminal state");
  }
  void release_hidden(TerminalState state) override {
    if (!state) return;
    auto offset = static_cast<BF16*>(state.value) - static_cast<BF16*>(hidden_->data());
    require(offset >= 0 && std::size_t(offset) < used_.size() * kHiddenSize && offset % kHiddenSize == 0,
            "invalid terminal state");
    used_[offset / kHiddenSize] = false;
  }
  std::size_t occupied_hidden_bytes() const override {
    return std::count(used_.begin(), used_.end(), true) * kHiddenBytes;
  }
  void copy_terminal(TerminalState to, TerminalState from, std::string_view op) override {
    require(bool(to) && bool(from), "missing terminal state");
    check_cuda(cudaMemcpyAsync(to.value, from.value, kHiddenBytes, cudaMemcpyDeviceToDevice, stream_.value), op);
  }
  void save_terminal(std::uint32_t row, TerminalState to) override {
    require(last_hidden_ && row < output_rows_, "invalid terminal row");
    copy_terminal(to, TerminalState(const_cast<BF16*>(last_hidden_ + std::size_t(row) * kHiddenSize)), "save 26B terminal");
  }
  void begin_step(std::string_view op) override { check_cuda(cudaEventRecord(begin_.value, stream_.value), op); }
  void end_step(std::string_view op) override { check_cuda(cudaEventRecord(end_.value, stream_.value), op); }
  float elapsed(std::string_view op) const override {
    float value;
    check_cuda(cudaEventElapsedTime(&value, begin_.value, end_.value), op);
    return value;
  }
  std::vector<bool> prefill_batch(const std::vector<BatchPrefillInput>& inputs,
      const std::function<bool(std::size_t)>& continue_prefill) override {
    require(!inputs.empty(), "empty prefill");
    std::size_t total = 0, image_rows = 0;
    for (const auto& input : inputs) {
      validate(input.position, input.rows, input.tokens);
      total += input.rows;
      if (input.image) {
        require(vision_ != nullptr, "image input requires --vision PATH");
        const auto& image = *input.image;
        require(image.begin == input.position && image.end > image.begin &&
            image.end - image.begin == input.rows && input.rows <= limits_.max_image_tokens &&
            image.end <= limits_.context_tokens, "invalid whole-image segment");
        require(image.pixels.size() == vision_engine::prepared_pixel_bytes(image.padded_patch_rows) &&
            image.positions.size() == vision_engine::prepared_position_bytes(image.padded_patch_rows),
            "prepared image byte lengths disagree with patch rows");
        vision_engine::validate_prepared_image_bytes(image.pixels.data(), image.pixels.size(),
            image.positions.data(), image.positions.size(), input.rows);
        image_rows += input.rows;
      }
    }
    require(total <= max_rows_, "prefill exceeds workspace");
    std::vector<const BF16*> features(inputs.size());
    if (image_rows) {
      reserve(image_features_, image_rows * kHiddenBytes);
      std::size_t offset = 0;
      for (std::size_t i=0;i<inputs.size();++i) {
        const auto& input=inputs[i];
        if (!input.image || !continue_prefill(i)) continue;
        const auto& image=*input.image;
        if (!vision_tower_) vision_tower_=std::make_unique<vision_executor::VisionExecutor>(
            vision_engine::Model::gemma4_26b_a4b,
            vision_executor::VisionWeightSlice{static_cast<const BF16*>(vision_->data()),vision_->size()},
            input.rows);
        vision_tower_->set_soft_token_count(input.rows);
        reserve(image_pixels_,image.pixels.size());
        reserve(image_positions_,image.positions.size());
        reserve(image_scratch_,vision_tower_->scratch_bytes());
        report_memory();
        check_cuda(cudaMemcpyAsync(image_pixels_->data(),image.pixels.data(),image.pixels.size(),
            cudaMemcpyHostToDevice,stream_.value),"upload26B image pixels");
        check_cuda(cudaMemcpyAsync(image_positions_->data(),image.positions.data(),image.positions.size(),
            cudaMemcpyHostToDevice,stream_.value),"upload26B image positions");
        auto* output=static_cast<BF16*>(image_features_->data())+offset*kHiddenSize;
        vision_tower_->run({{static_cast<const std::uint8_t*>(image_pixels_->data()),
            static_cast<const std::int32_t*>(image_positions_->data()),image.padded_patch_rows,input.rows},
            output,input.rows},image_scratch_->data(),image_scratch_->size(),stream_.value);
        features[i]=output;
        offset+=input.rows;
      }
    }
    std::vector<bool> completed(inputs.size(),false);
    std::vector<Segment> segments;
    std::size_t offset = 0;
    for (std::size_t i=0;i<inputs.size();++i) {
      const auto& input=inputs[i];
      // Poll again after serial tower work. Cancellation can leave unused
      // feature rows; they remain private until this dispatch completes.
      if (image_rows && (!continue_prefill(i) || (input.image && !features[i]))) continue;
      std::memcpy(static_cast<std::uint32_t*>(host_tokens_->data()) + offset, input.tokens,
                  input.rows * sizeof(std::uint32_t));
      auto segment=prepare(input.execution,input.position,input.rows);
      segment.image_features=features[i];
      segments.push_back(std::move(segment));
      completed[i]=true;
      offset+=input.rows;
    }
    if (!offset) return completed;
    upload_tokens(offset);
    const auto* hidden = executor_->forward(static_cast<std::uint32_t*>(tokens_->data()), segments, nvfp4::Phase::prefill);
    offset = 0;
    for (std::size_t i=0;i<inputs.size();++i) {
      if (!completed[i]) continue;
      const auto& input=inputs[i];
      offset += input.rows;
      if (input.hidden) copy_terminal(input.hidden,
          TerminalState(const_cast<BF16*>(hidden + (offset - 1) * kHiddenSize)), "save 26B prefill terminal");
    }
    last_hidden_ = nullptr;
    output_rows_ = 0;
    return completed;
  }
  void release_image() override {}
  void prefix_head_step(kv_cache::ExecutionId execution, TerminalState hidden) override {
    if (!hidden) {
      hidden = TerminalState(head_hidden_->data());
      cache_->restore_terminal_hidden(execution, hidden, context());
    }
    last_hidden_ = static_cast<BF16*>(hidden.value);
    output_rows_ = 1;
    distribution_row_.reset();
    executor_->head(last_hidden_, 1, static_cast<BF16*>(logits_->data()));
  }
  void decode_batch(const std::vector<BatchDecodeInput>& inputs) override {
    require(!inputs.empty() && inputs.size() <= capacity_, "invalid decode batch");
    for (const auto& input : inputs) validate(input.position, 1, &input.token);
    std::vector<Segment> segments;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const auto& input = inputs[i];
      static_cast<std::uint32_t*>(host_tokens_->data())[i] = input.token;
      segments.push_back(prepare(input.execution, input.position, 1));
    }
    upload_tokens(inputs.size());
    last_hidden_ = executor_->forward(static_cast<std::uint32_t*>(tokens_->data()), segments, nvfp4::Phase::decode);
    output_rows_ = inputs.size();
    distribution_row_.reset();
    executor_->head(last_hidden_, output_rows_, static_cast<BF16*>(logits_->data()));
  }
  void sample_batch_row(std::uint32_t row, const SamplingSettings& settings,
      std::mt19937_64& rng, const std::uint32_t* mask) override {
    require(row < output_rows_, "invalid sampling row");
    auto* host = static_cast<SampleInput*>(host_sample_->data()) + row;
    auto* device = static_cast<SampleInput*>(sample_->data()) + row;
    const bool greedy = settings.temperature == 0 || settings.top_p == 0 || settings.top_k == 1;
    host->uniform = greedy ? 0 : std::min(std::uniform_real_distribution<float>(0, 1)(rng), std::nextafter(1.0f, 0.0f));
    host->status = mtp_sampling::Status::success;
    if (mask) std::memcpy(host->mask, mask, sizeof(host->mask));
    check_cuda(cudaMemcpyAsync(device, host, mask ? sizeof(SampleInput) : offsetof(SampleInput, mask),
                              cudaMemcpyHostToDevice, stream_.value), "upload 26B sampling inputs");
    distribution_row_.reset();
    auto* selected = static_cast<std::uint32_t*>(ids_->data()) + row;
    if (greedy) {
      mtp_sampling::build_greedy_distributions({{logit_row(row), nullptr, mask ? device->mask : nullptr,
          &device->status, static_cast<std::uint32_t*>(best_->data()) + row, selected}}, kVocabSize, stream_.value);
    } else {
      distribution(row, settings, mask != nullptr);
      mtp_sampling::sample_distribution(static_cast<float*>(probs_->data()), kVocabSize,
          &device->uniform, selected, sampling_->data(), sampling_->size(), &device->status, stream_.value);
    }
    copy_status(row);
  }
  void summarize_batch_row(std::uint32_t row, const SamplingSettings& settings,
      std::uint32_t top, const std::uint32_t* mask) override {
    require(logprobs_ && row < output_rows_, "logprobs not initialized or invalid row");
    if (distribution_row_ != row) distribution(row, settings, mask != nullptr);
    mtp_sampling::summarize_logprobs(static_cast<float*>(probs_->data()),
        static_cast<std::uint32_t*>(ids_->data()) + row, 1, kVocabSize, top,
        static_cast<TokenLogprobs*>(logprobs_->data()) + row, stream_.value);
    copy_status(row);
  }
  void check_constraint_sampling() override {
    for (std::size_t i = 0; i < pending_.size(); ++i) if (pending_[i]) {
      pending_[i] = false;
      auto status = static_cast<SampleInput*>(host_sample_->data())[i].status;
      if (status != mtp_sampling::Status::success)
        throw std::runtime_error(std::string("26B sampling: ") + mtp_sampling::status_message(status));
    }
  }
  BatchMtpOutcome run_batch_mtp(const std::vector<BatchMtpInput>& inputs) override {
    require(bool(mtp_), "MTP is not initialized");
    std::vector<mtp_cycle::BatchInput> proposals;
    for (const auto& input : inputs) {
      // Verification borrows committed views; only accepted commit prepares writes.
      proposals.push_back({input.pending_token, static_cast<const BF16*>(input.target_hidden.value),
          input.position, input.depth, cache_views(input.execution), input.temperature, input.top_p,
          input.top_k, input.return_probabilities, input.uniforms, input.constraint_mask,
          input.capture, input.capture_next, {}});
    }
    auto result = mtp_->run(proposals, stream_.value);
    BatchMtpOutcome output;
    output.draft_gpu_milliseconds = result.draft_gpu_milliseconds;
    output.verify_gpu_milliseconds = result.verify_gpu_milliseconds;
    output.select_gpu_milliseconds = result.select_gpu_milliseconds;
    output.constraint_draft_downloads = result.constraint_draft_downloads;
    output.constraint_mask_uploads = result.constraint_mask_uploads;
    output.constraint_draft_bytes = result.constraint_draft_bytes;
    output.constraint_mask_bytes = result.constraint_mask_bytes;
    for (auto& selected : result.requests)
      output.requests.push_back({{selected.verification.accepted_drafts,
          selected.verification.output_count, selected.verification.rejected_index}, std::move(selected.tokens),
          selected.status == mtp_sampling::Status::success ? std::string{} :
              std::string("MTP cycle: ") + mtp_sampling::status_message(selected.status)});
    last_hidden_ = nullptr;
    output_rows_ = 0;
    distribution_row_.reset();
    return output;
  }
  std::vector<std::uint32_t> predict_mtp_depths(const std::vector<MtpDepthInput>&) override {
    throw std::logic_error("26B learned MTP depth controller is not supported");
  }
  void commit_batch_mtp(const std::vector<BatchMtpCommit>& inputs) override {
    require(bool(mtp_), "MTP is not initialized");
    for (const auto& input : inputs) {
      require(input.row < capacity_ && input.count && input.count <= mtp_depth_ + 1 && bool(input.hidden),
              "invalid MTP commit");
      require(!input.captures || bool(host_logits_), "MTP captures are not initialized");
      require(!input.logprobs || bool(logprobs_), "MTP logprobs are not initialized");
    }
    std::vector<mtp_cycle::BatchCaches> views(inputs.size());
    std::vector<mtp_cycle::BatchCommitInput> commits;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const auto& input = inputs[i];
      cache_->prepare_write(input.execution, input.position, input.count, context());
      views[i] = cache_views(input.execution);
      commits.push_back({input.row, &views[i], input.count});
    }
    mtp_->commit_batch(commits, stream_.value);
    for (const auto& input : inputs) {
      copy_terminal(input.hidden, TerminalState(const_cast<BF16*>(mtp_->hidden(input.row) +
          std::size_t(input.count - 1) * kHiddenSize)), "save 26B MTP terminal");
      const auto offset = std::size_t(input.row) * (mtp_depth_ + 1);
      if (input.captures)
        check_cuda(cudaMemcpyAsync(static_cast<std::uint8_t*>(host_logits_->data()) + offset * limits_.logit_row_bytes,
            mtp_->logits(input.row), input.count * limits_.logit_row_bytes, cudaMemcpyDeviceToHost,
            stream_.value), "download 26B MTP logits");
      if (input.logprobs) {
        auto* records = static_cast<TokenLogprobs*>(logprobs_->data()) + offset;
        mtp_->summarize_logprobs(input.row, input.count, input.top_logprobs, records, stream_.value);
        check_cuda(cudaMemcpyAsync(static_cast<TokenLogprobs*>(host_logprobs_->data()) + offset, records,
            input.count * sizeof(TokenLogprobs), cudaMemcpyDeviceToHost, stream_.value), "download 26B MTP logprobs");
      }
    }
  }
  void download_ids(std::size_t rows) override {
    require(rows <= output_rows_, "invalid ID download count");
    check_cuda(cudaMemcpyAsync(host_ids_->data(), ids_->data(), rows * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost, stream_.value), "download 26B IDs");
  }
  void download_logits(std::size_t row) override {
    require(host_logits_ && row < output_rows_, "logits not initialized or invalid row");
    check_cuda(cudaMemcpyAsync(static_cast<std::uint8_t*>(host_logits_->data()) + row * limits_.logit_row_bytes, logit_row(row), limits_.logit_row_bytes,
                              cudaMemcpyDeviceToHost, stream_.value), "download 26B logits");
  }
  void download_logprobs(std::size_t row) override {
    require(host_logprobs_ && row < output_rows_, "logprobs not initialized or invalid row");
    check_cuda(cudaMemcpyAsync(static_cast<TokenLogprobs*>(host_logprobs_->data()) + row, static_cast<TokenLogprobs*>(logprobs_->data()) + row,
                              sizeof(TokenLogprobs), cudaMemcpyDeviceToHost, stream_.value), "download 26B logprobs");
  }
  std::uint32_t output_id(std::size_t row) const override {
    require(row < output_rows_, "invalid output row");
    return static_cast<std::uint32_t*>(host_ids_->data())[row];
  }
  const std::uint8_t* host_logits() const override { return host_logits_ ? static_cast<std::uint8_t*>(host_logits_->data()) : nullptr; }
  const TokenLogprobs* host_logprobs() const override { return host_logprobs_ ? static_cast<TokenLogprobs*>(host_logprobs_->data()) : nullptr; }
  std::size_t scratch_bytes() const override {
    return (executor_ ? executor_->scratch_bytes() : 0) + bytes(tokens_) + bytes(head_hidden_) +
        bytes(image_pixels_) + bytes(image_positions_) + bytes(image_features_) + bytes(image_scratch_) +
        bytes(logits_) + sampling_scratch_bytes() + bytes(logprobs_) + (mtp_ ? mtp_->scratch_bytes() : 0);
  }
  std::size_t output_bytes() const override { return bytes(ids_); }
  std::size_t host_scratch_bytes() const override { return bytes(host_tokens_) + bytes(host_sample_) + bytes(host_ids_) + bytes(host_logits_) + bytes(host_logprobs_) + (mtp_ ? mtp_->host_scratch_bytes() : 0); }
  std::size_t sampling_scratch_bytes() const override { return bytes(sample_) + bytes(probs_) + bytes(sampling_) + bytes(best_); }
 private:
  template<class T> static std::size_t bytes(const std::unique_ptr<T>& p) { return p ? p->size() : 0; }
  void validate(unsigned position, unsigned rows, const std::uint32_t* tokens) const {
    require(tokens && rows && std::uint64_t(position) + rows <= kMaxPositions, "invalid token range");
    for (unsigned i = 0; i < rows; ++i) require(tokens[i] < kVocabSize, "token outside vocabulary");
  }
  mtp_cycle::BatchCaches cache_views(kv_cache::ExecutionId execution) const {
    mtp_cycle::BatchCaches views(kLayerCount);
    auto& physical = static_cast<PhysicalCache&>(cache_->storage());
    for (unsigned l = 0; l < kLayerCount; ++l) views[l] = physical.layer(execution, l);
    return views;
  }
  Segment prepare(kv_cache::ExecutionId execution, unsigned position, unsigned rows) {
    cache_->prepare_write(execution, position, rows, context());
    Segment result{position, rows, {}};
    auto& physical = static_cast<PhysicalCache&>(cache_->storage());
    for (unsigned l = 0; l < kLayerCount; ++l) result.cache[l] = physical.layer(execution, l);
    return result;
  }
  void upload_tokens(std::size_t count) {
    check_cuda(cudaMemcpyAsync(tokens_->data(), host_tokens_->data(), count * sizeof(std::uint32_t),
                              cudaMemcpyHostToDevice, stream_.value), "upload 26B tokens");
  }
  const BF16* logit_row(std::size_t row) const { return static_cast<BF16*>(logits_->data()) + row * kVocabSize; }
  void distribution(unsigned row, const SamplingSettings& settings, bool masked) {
    auto* input = static_cast<SampleInput*>(sample_->data()) + row;
    mtp_sampling::build_distribution(logit_row(row), kVocabSize, settings.temperature, settings.top_p,
        settings.top_k, static_cast<float*>(probs_->data()), sampling_->data(), sampling_->size(),
        &input->status, stream_.value, masked ? input->mask : nullptr);
    distribution_row_ = row;
  }
  void copy_status(unsigned row) {
    auto* host = static_cast<SampleInput*>(host_sample_->data()) + row;
    auto* device = static_cast<SampleInput*>(sample_->data()) + row;
    check_cuda(cudaMemcpyAsync(&host->status, &device->status, sizeof(host->status),
                              cudaMemcpyDeviceToHost, stream_.value), "download 26B sampling status");
    pending_[row] = true;
  }
  void report_memory() {
    const auto allocated = executor_->weight_bytes() + bytes(assistant_) + bytes(vision_) +
        scratch_bytes() + output_bytes() + memory_.staging_bytes + memory_.hidden_staging_bytes;
    if (allocated == reported_device_bytes_) return;
    reported_device_bytes_ = allocated;
    const auto m = executor_->memory_usage();
    std::size_t free = 0, total = 0;
    check_cuda(cudaMemGetInfo(&free, &total), "26B allocation snapshot");
    console::event("model_device_memory", {
        {"architecture", "gemma4_26b_a4b"}, {"workspace_rows", max_rows_},
        {"weights_bytes", m.weights}, {"expert_pointer_tables_bytes", m.expert_tables},
        {"assistant_weights_bytes", bytes(assistant_)}, {"vision_weights_bytes", bytes(vision_)},
        {"shared_model_activation_scratch_bytes", m.shared_projection}, {"position_bytes", m.positions},
        {"routing_scratch_bytes", m.routing}, {"expert_projection_scratch_bytes", m.expert_projection},
        {"expert_reduction_scratch_bytes", m.expert_reduction}, {"attention_scratch_bytes", m.attention},
        {"quantized_projection_scratch_bytes", m.quantized_projection},
        {"token_input_bytes", bytes(tokens_)}, {"head_hidden_bytes", bytes(head_hidden_)},
        {"head_logits_bytes", bytes(logits_)}, {"sampling_scratch_bytes", sampling_scratch_bytes()},
        {"logprob_output_bytes", bytes(logprobs_)}, {"token_output_bytes", output_bytes()},
        {"mtp_scratch_bytes", mtp_ ? mtp_->scratch_bytes() : 0},
        {"image_input_bytes", bytes(image_pixels_) + bytes(image_positions_)},
        {"image_features_bytes", bytes(image_features_)}, {"image_scratch_bytes", bytes(image_scratch_)},
        {"mtp_kv_staging_bytes", memory_.staging_bytes},
        {"prefix_hidden_staging_bytes", memory_.hidden_staging_bytes},
        {"committed_kv_budget_bytes", memory_.committed_kv_bytes},
        {"backend_allocated_bytes", allocated},
        {"planned_with_kv_budget_bytes", allocated + memory_.committed_kv_bytes},
        {"cuda_device_used_bytes", total - free}, {"cuda_device_total_bytes", total},
        {"measurement", "allocation_snapshot_not_peak; CUDA includes context/libraries/other processes; KV budget is a limit"}});
  }
  std::size_t reported_device_bytes_{};
  static void reserve(std::unique_ptr<Device>& buffer, std::size_t size) {
    if (buffer && buffer->size() >= size) return;
    buffer.reset();
    buffer=std::make_unique<Device>(size);
  }
  ArtifactFile weights_;
  QdqMask qdq_;
  const nvfp4::ActivationPolicy policy_;
  unsigned capacity_, mtp_depth_, max_rows_{}, output_rows_{};
  BackendLimits limits_{kVocabSize, kMaxPositions, kMaxVerifierRows, kMaxVerifierRows - 1, kMaxVerifierRows, kMaxRows, kMaxRows,
                        kLocalWindowSize, kVocabSize * sizeof(BF16), {1, 106, 50}, 1120,
                        {kHiddenSize, kAssistantHiddenSize, kLayerCount, {2, 6, 12, 20, 28}}};
  BatchMemoryPlan memory_;
  kv_cache::PoolConfig config_;
  Stream stream_;
  Event begin_, end_;
  PersistentCacheManager* cache_{};
  std::unique_ptr<BatchedExecutor> executor_;
  std::unique_ptr<Device> hidden_, tokens_, head_hidden_, logits_, ids_, sample_, probs_, sampling_, best_, logprobs_;
  std::unique_ptr<Host> host_tokens_, host_ids_, host_sample_, host_logits_, host_logprobs_;
  std::vector<bool> used_, pending_;
  // Reverse destruction keeps borrowed weights, staging and handle alive through the cycle.
  std::unique_ptr<component::File> vision_file_;
  std::unique_ptr<Device> vision_, image_pixels_, image_positions_, image_features_, image_scratch_;
  std::unique_ptr<vision_executor::VisionExecutor> vision_tower_;
  std::unique_ptr<component::File> assistant_file_;
  std::unique_ptr<Device> assistant_, staging_;
  std::unique_ptr<BlasLt> mtp_handle_;
  std::unique_ptr<mtp_cycle::Batch> mtp_;
  const BF16* last_hidden_{};
  std::optional<unsigned> distribution_row_;
};
}  // namespace
std::unique_ptr<runtime::ExecutionBackend> make_runtime_backend(
    ArtifactFile weights, const runtime::BatchLimits& limits, nvfp4::ActivationPolicy policy,
    const std::string& assistant_path, const std::string& vision_path, const std::string& qdq_path) {
  return std::make_unique<RuntimeBackend>(std::move(weights), limits, policy, assistant_path, vision_path, qdq_path);
}
}  // namespace gewell::gemma4_26b_a4b::sm120
