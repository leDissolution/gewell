#include "runtime_backend.h"
#include "batch_execution.cuh"
#include "vision/prepared_image.h"
#include "gewell/runtime/scheduler.h"

namespace gewell::gemma4_31b::sm120 {
namespace {
class RuntimeBackend final : public runtime::ExecutionBackend {
 public:
  RuntimeBackend(const WeightArena& weights, const runtime::BatchLimits& limits,
                 nvfp4::ActivationPolicy policy)
      : weights_(weights), policy_(policy),
        memory_(limits.kv_bytes, limits.capacity, limits.mtp_depth, limits.local_kv_format, limits.global_kv_format),
        config_(compact_pool_config(memory_.committed_kv_bytes, limits.cpu_bytes, limits.index_bytes, limits.local_kv_format, limits.global_kv_format)),
        execution_(memory_, limits.capacity, limits.mtp_depth) {
    log_nvfp4_policy(weights, policy);
    console::field("attention_local_compute", attention::compute_name(limits.local_attention_compute));
    console::field("attention_global_compute", attention::compute_name(limits.global_attention_compute));
    console::field("kv_local_format", kv_cache::format_name(config_.local_format));
    console::field("kv_global_format", kv_cache::format_name(config_.global_format));
    console::field("kv_local_ring_bytes", config_.local_ring_bytes);
    console::field("kv_global_page_bytes", config_.global_page_bytes);
  }
  ~RuntimeBackend() override { wait(); }
  const runtime::BackendLimits& limits() const override { return limits_; }
  runtime::BatchMemoryPlan memory_plan() const override {
    return {memory_.staging_bytes, memory_.hidden_slots, memory_.hidden_staging_bytes,
            memory_.committed_kv_bytes};
  }
  kv_cache::PoolConfig cache_config() const override { return config_; }
  runtime::CacheStorageFactory cache_storage_factory() const override {
    return [](const auto& config, auto& ledger, std::size_t offsets) {
      return std::make_unique<PhysicalCache>(config, ledger, offsets);
    };
  }
  void allocate_staging() override { execution_.allocate_staging(); }
  void initialize(runtime::PersistentCacheManager& cache, kv_cache::ExecutionId bootstrap,
                  const runtime::BatchLimits& limits) override {
    cache_ = std::make_unique<ExecutionCache>(cache);
    execution_.initialize_executor(weights_, *cache_, bootstrap, limits.plan_rows,
        limits.max_horizon, limits.sampled, limits.prefill_chunk_tokens, limits.prefill_batch_tokens, policy_,
        limits.local_attention_compute, limits.global_attention_compute);
  }
  void initialize_outputs(bool captures, bool logprobs) override {
    execution_.initialize_outputs(captures, logprobs);
  }
  runtime::CompletionContext context() const override {
    return runtime::CompletionContext(execution_.engine->stream());
  }
  void wait() const override { if (execution_.engine) execution_.engine->wait(); }
  void synchronize(std::string_view op) const override { execution_.engine->synchronize(op); }
  bool ready(std::string_view op) const override { return execution_.engine->ready(op); }
  runtime::TerminalState acquire_hidden() override {
    return runtime::TerminalState(execution_.acquire_hidden());
  }
  void release_hidden(runtime::TerminalState state) override {
    execution_.release_hidden(static_cast<BFloat16*>(state.value));
  }
  std::size_t occupied_hidden_bytes() const override { return execution_.occupied_hidden_bytes(); }
  void copy_terminal(runtime::TerminalState to, runtime::TerminalState from,
                     std::string_view op) override {
    execution_.engine->copy_terminal(static_cast<BFloat16*>(to.value),
                                     static_cast<const BFloat16*>(from.value), op);
  }
  void save_terminal(std::uint32_t row, runtime::TerminalState to) override {
    execution_.engine->copy_terminal(static_cast<BFloat16*>(to.value),
        execution_.engine->batch_terminal_hidden(row), "save batch decode terminal hidden");
  }
  void begin_step(std::string_view op) override { execution_.begin_step(op); }
  void end_step(std::string_view op) override { execution_.end_step(op); }
  float elapsed(std::string_view op) const override { return execution_.elapsed(op); }
  std::vector<bool> prefill_batch(const std::vector<runtime::BatchPrefillInput>& inputs,
      const std::function<bool(std::size_t)>& continue_prefill) override {
    std::size_t feature_bytes = 0;
    for (const auto& input : inputs)
      if (input.image) feature_bytes += std::size_t(input.rows) * model::kHiddenSize * sizeof(BFloat16);
    // Text-only forwards already poll after dispatch. Avoid image staging and
    // mid-selection admission/cancellation polls on that established path.
    if (!feature_bytes) {
      execution_.engine->prefill_batch(inputs, {});
      return std::vector<bool>(inputs.size(), true);
    }
    if (feature_bytes && (!image_features_ || image_features_->size() < feature_bytes)) {
      image_features_.reset();
      image_features_ = std::make_unique<DeviceAllocation>(feature_bytes);
    }
    std::vector<VisionPromptSlice> images(inputs.size());
    std::size_t offset = 0;
    const auto stream = execution_.engine->stream();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const auto& input = inputs[i];
      if (!input.image || !continue_prefill(i)) continue;
      if (!weights_.has_vision()) fail("image prefill", "image input requires --vision PATH");
      const auto& image = *input.image;
      if (image.begin != input.position || image.end <= image.begin ||
          image.end - image.begin != input.rows || input.rows > limits_.max_image_tokens ||
          image.end > limits_.context_tokens)
        fail("image prefill", "invalid whole-image segment");
      if (image.pixels.size() != vision_engine::prepared_pixel_bytes(image.padded_patch_rows) ||
          image.positions.size() != vision_engine::prepared_position_bytes(image.padded_patch_rows))
        fail("image prefill", "prepared tensor lengths disagree with patch rows");
      vision_engine::validate_prepared_image_bytes(image.pixels.data(), image.pixels.size(),
          image.positions.data(), image.positions.size(), input.rows);
      if (!image_) image_ = std::make_unique<PreparedImage>(weights_);
      image_->prepare(image.pixels, image.positions, image.padded_patch_rows, input.rows, stream);
      auto* features = static_cast<BFloat16*>(image_features_->data()) + offset;
      // Retain this image before the next serial tower pass reuses its output.
      check_cuda(cudaMemcpyAsync(features, image_->data(),
          std::size_t(input.rows) * model::kHiddenSize * sizeof(BFloat16),
          cudaMemcpyDeviceToDevice, stream), "retain packed image features");
      images[i] = {features, image.begin, image.end};
      offset += std::size_t(input.rows) * model::kHiddenSize;
    }
    std::vector<bool> completed(inputs.size(), false);
    std::vector<runtime::BatchPrefillInput> active;
    std::vector<VisionPromptSlice> active_images;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      if (!continue_prefill(i)) continue;
      active.push_back(inputs[i]);
      active_images.push_back(images[i]);
      completed[i] = true;
    }
    if (!active.empty()) execution_.engine->prefill_batch(active, active_images);
    return completed;
  }
  // Completion ends feature use; retain the bounded cohort buffer and tower workspace.
  void release_image() override {}
  void prefix_head_step(kv_cache::ExecutionId id, runtime::TerminalState hidden) override {
    execution_.engine->prefix_head_step(id, nullptr, static_cast<const BFloat16*>(hidden.value));
  }
  void decode_batch(const std::vector<runtime::BatchDecodeInput>& inputs) override {
    execution_.engine->decode_batch(inputs);
  }
  void sample_batch_row(std::uint32_t row, const runtime::SamplingSettings& sampling,
                        std::mt19937_64& rng, const std::uint32_t* mask) override {
    execution_.engine->sample_batch_row(row, {sampling.temperature, sampling.top_p, sampling.top_k}, rng, mask);
  }
  void summarize_batch_row(std::uint32_t row, const runtime::SamplingSettings& sampling,
                           std::uint32_t top, const std::uint32_t* mask) override {
    execution_.engine->summarize_batch_row(row, {sampling.temperature, sampling.top_p, sampling.top_k},
        top, static_cast<TokenLogprobs*>(execution_.device_logprobs->data()) + row, mask);
  }
  void check_constraint_sampling() override { execution_.engine->check_constraint_sampling(); }
  runtime::BatchMtpOutcome run_batch_mtp(const std::vector<runtime::BatchMtpInput>& inputs) override {
    std::vector<BatchMtpInput> proposals;
    proposals.reserve(inputs.size());
    for (const auto& input : inputs) {
      proposals.push_back({input.execution, input.pending_token, input.position, input.depth,
          static_cast<const BFloat16*>(input.target_hidden.value), input.temperature, input.top_p,
          input.top_k, input.return_probabilities, input.uniforms,
          input.constraint_mask, input.capture, input.capture_next});
    }
    auto result = execution_.engine->run_batch_mtp(proposals);
    runtime::BatchMtpOutcome output;
    output.draft_gpu_milliseconds = result.draft_gpu_milliseconds;
    output.verify_gpu_milliseconds = result.verify_gpu_milliseconds;
    output.select_gpu_milliseconds = result.select_gpu_milliseconds;
    output.constraint_draft_downloads = result.constraint_draft_downloads;
    output.constraint_mask_uploads = result.constraint_mask_uploads;
    output.constraint_draft_bytes = result.constraint_draft_bytes;
    output.constraint_mask_bytes = result.constraint_mask_bytes;
    output.requests.reserve(result.requests.size());
    for (auto& selected : result.requests)
      output.requests.push_back({{selected.verification.accepted_drafts,
          selected.verification.output_count, selected.verification.rejected_index}, std::move(selected.tokens),
          selected.status == mtp_sampling::Status::success ? std::string{} :
              std::string("MTP cycle: ") + mtp_sampling::status_message(selected.status)});
    return output;
  }
  void commit_batch_mtp(
      const std::vector<runtime::BatchMtpCommit>& inputs) override {
    std::vector<BatchMtpCommitInput> commits;
    commits.reserve(inputs.size());
    for (const auto& input : inputs) {
      const auto offset = input.row * std::size_t(execution_.mtp_depth + 1);
      commits.push_back({
          input.row, input.execution, input.position, input.count,
          static_cast<BFloat16*>(input.hidden.value),
          input.captures
              ? static_cast<std::uint8_t*>(execution_.host_logits->data()) +
                    offset * kGenerationLogitRowBytes
              : nullptr,
          input.logprobs
              ? static_cast<TokenLogprobs*>(execution_.device_logprobs->data()) +
                    offset
              : nullptr,
          input.logprobs
              ? static_cast<TokenLogprobs*>(execution_.host_logprobs->data()) +
                    offset
              : nullptr,
          input.top_logprobs,
      });
    }
    execution_.engine->commit_batch_mtp(commits);
  }
  void download_ids(std::size_t rows) override { execution_.download_ids(rows); }
  void download_logits(std::size_t row) override { execution_.download_logits(row); }
  void download_logprobs(std::size_t row) override { execution_.download_logprobs(row); }
  std::uint32_t output_id(std::size_t row) const override {
    return static_cast<const std::uint32_t*>(execution_.host_ids->data())[row];
  }
  const std::uint8_t* host_logits() const override {
    return static_cast<const std::uint8_t*>(execution_.host_logits->data());
  }
  const TokenLogprobs* host_logprobs() const override {
    return static_cast<const TokenLogprobs*>(execution_.host_logprobs->data());
  }
  std::size_t scratch_bytes() const override { return execution_.engine->scratch_bytes(); }
  std::size_t output_bytes() const override { return execution_.engine->output_bytes(); }
  std::size_t host_scratch_bytes() const override { return execution_.engine->host_scratch_bytes(); }
  std::size_t sampling_scratch_bytes() const override { return execution_.engine->sampling_scratch_bytes(); }
 private:
  const WeightArena& weights_;
  const nvfp4::ActivationPolicy policy_;
  const runtime::BackendLimits limits_{model::kVocabSize, primitives::kMaxContextTokenCount,
      kMaxBatchRows, mtp_target::kMaxDepth, mtp_target::kMaxDepth + 1,
      kMaxPrefillChunkTokens, kMaxPrefillBatchTokens, kLocalWindowTokens, kGenerationLogitRowBytes,
      {kGenerationStopTokenIds.begin(), kGenerationStopTokenIds.end()}, model::kVisionMaxSoftTokenCount};
  const BatchMemoryPlan memory_;
  const kv_cache::PoolConfig config_;
  std::unique_ptr<ExecutionCache> cache_;
  BatchExecution execution_;
  std::unique_ptr<PreparedImage> image_;
  std::unique_ptr<DeviceAllocation> image_features_;
};
}
std::unique_ptr<runtime::ExecutionBackend> make_runtime_backend(
    const WeightArena& weights, const runtime::BatchLimits& limits,
    nvfp4::ActivationPolicy activation_policy) {
  return std::make_unique<RuntimeBackend>(weights, limits, activation_policy);
}
}  // namespace gewell::gemma4_31b::sm120
