#pragma once
#include "gewell/attention_compute.h"

#include "weights.cuh"
#include "request_cache.cuh"
#include "gewell/mtp_cycle.h"
#include "gewell/mtp_attention.h"
#include "gewell/logprobs.h"
#include <map>
#include <memory>
#include <optional>
#include <random>

namespace gewell::gemma4_31b::sm120 {

struct RuntimeGenerationScratchLayout {
  explicit constexpr RuntimeGenerationScratchLayout(std::size_t rows = kMultimodalChunkTokens)
      : kRows(rows) {}
  const std::size_t kRows;
  const std::size_t kHiddenElements =
      kRows * model::kHiddenSize;
  const std::size_t kQueryElements =
      kRows * model::kQueryHeadCount * model::kGlobalHeadSize;
  const std::size_t kKvElements =
      kRows * model::kLocalKvHeadCount * model::kLocalHeadSize;
  const std::size_t kMlpElements = kRows * model::kMlpSize;

  const std::size_t kH0 = 0;
  const std::size_t kH1 =
      align_up(kH0 + kHiddenElements * sizeof(BFloat16), kScratchAlignment);
  const std::size_t kH2 =
      align_up(kH1 + kHiddenElements * sizeof(BFloat16), kScratchAlignment);
  const std::size_t kQueryRaw =
      align_up(kH2 + kHiddenElements * sizeof(BFloat16), kScratchAlignment);
  const std::size_t kQueryNorm =
      align_up(kQueryRaw + kRows * fusion::kMaxQkvWidth * sizeof(BFloat16),
               kScratchAlignment);
  const std::size_t kQueryRope =
      align_up(kQueryNorm + kQueryElements * sizeof(BFloat16),
               kScratchAlignment);
  const std::size_t kKeyRaw =
      align_up(kQueryRope + kQueryElements * sizeof(BFloat16),
               kScratchAlignment);
  const std::size_t kKeyNorm =
      align_up(kKeyRaw + kKvElements * sizeof(BFloat16), kScratchAlignment);
  const std::size_t kKeyRope =
      align_up(kKeyNorm + kKvElements * sizeof(BFloat16), kScratchAlignment);
  const std::size_t kValueRaw =
      align_up(kKeyRope + kKvElements * sizeof(BFloat16), kScratchAlignment);
  const std::size_t kValueNorm =
      align_up(kValueRaw + kKvElements * sizeof(BFloat16), kScratchAlignment);
  const std::size_t kContext =
      align_up(kValueNorm + kKvElements * sizeof(BFloat16),
               kScratchAlignment);
  const std::size_t kGate =
      align_up(kContext + kQueryElements * sizeof(BFloat16),
               kScratchAlignment);
  const std::size_t kUp =
      align_up(kGate + kMlpElements * sizeof(BFloat16), kScratchAlignment);
  const std::size_t kProduct =
      align_up(kUp + kMlpElements * sizeof(BFloat16), kScratchAlignment);
  const std::size_t kLogits =
      align_up(kProduct + kMlpElements * sizeof(BFloat16),
               kScratchAlignment);
  const std::size_t kCappedLogits =
      align_up(kLogits + model::kVocabSize * sizeof(BFloat16),
               kScratchAlignment);
  const std::size_t kLocalCos =
      align_up(kCappedLogits + model::kVocabSize * sizeof(BFloat16),
               kScratchAlignment);
  const std::size_t kLocalSin =
      align_up(kLocalCos + kRows * model::kLocalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  const std::size_t kGlobalCos =
      align_up(kLocalSin + kRows * model::kLocalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  const std::size_t kGlobalSin =
      align_up(kGlobalCos + kRows * model::kGlobalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  const std::size_t kArgmax =
      align_up(kGlobalSin + kRows * model::kGlobalHeadSize * sizeof(BFloat16),
               kScratchAlignment);
  const std::size_t kBytes =
      align_up(kArgmax + sizeof(std::uint32_t), kScratchAlignment);
};

static_assert(RuntimeGenerationScratchLayout{}.kRows >= model::kVisionMaxSoftTokenCount);
static_assert(RuntimeGenerationScratchLayout{}.kArgmax + sizeof(std::uint32_t) <=
              RuntimeGenerationScratchLayout{}.kBytes);


class RuntimeChunkPlans {
 public:
  explicit RuntimeChunkPlans(std::uint32_t rows)
      : rows_(rows),
        local_q_(rows, model::kHiddenSize,
                 model::kQueryHeadCount * model::kLocalHeadSize),
        local_kv_(rows, model::kHiddenSize,
                  model::kLocalKvHeadCount * model::kLocalHeadSize),
        global_q_(rows, model::kHiddenSize,
                  model::kQueryHeadCount * model::kGlobalHeadSize),
        global_kv_(rows, model::kHiddenSize,
                   model::kGlobalKvHeadCount * model::kGlobalHeadSize),
        local_qkv_(rows, model::kHiddenSize, fusion::qkv_width(false)),
        global_qkv_(rows, model::kHiddenSize, fusion::qkv_width(true)),
        local_o_(rows, model::kQueryHeadCount * model::kLocalHeadSize,
                 model::kHiddenSize),
        global_o_(rows, model::kQueryHeadCount * model::kGlobalHeadSize,
                  model::kHiddenSize),
        hidden_to_mlp_(rows, model::kHiddenSize, model::kMlpSize),
        gate_up_(rows, model::kHiddenSize, 2 * model::kMlpSize),
        mlp_to_hidden_(rows, model::kMlpSize, model::kHiddenSize) {
    if (rows == 0 || rows > prefill::kMaxChunkTokenCount) {
      fail("generation linear plans", "row count is outside 1..4096");
    }
  }

  [[nodiscard]] std::uint32_t rows() const { return rows_; }
  [[nodiscard]] const LinearPlan& q(bool global) const {
    return global ? global_q_ : local_q_;
  }
  [[nodiscard]] const LinearPlan& kv(bool global) const {
    return global ? global_kv_ : local_kv_;
  }
  [[nodiscard]] const LinearPlan& output(bool global) const {
    return global ? global_o_ : local_o_;
  }
  [[nodiscard]] const LinearPlan& qkv(bool global) const {
    return global ? global_qkv_ : local_qkv_;
  }
  [[nodiscard]] const LinearPlan& gate_up() const { return gate_up_; }
  [[nodiscard]] const LinearPlan& hidden_to_mlp() const {
    return hidden_to_mlp_;
  }
  [[nodiscard]] const LinearPlan& mlp_to_hidden() const {
    return mlp_to_hidden_;
  }

 private:
  std::uint32_t rows_{};
  LinearPlan local_q_;
  LinearPlan local_kv_;
  LinearPlan global_q_;
  LinearPlan global_kv_;
  LinearPlan local_qkv_;
  LinearPlan global_qkv_;
  LinearPlan local_o_;
  LinearPlan global_o_;
  LinearPlan hidden_to_mlp_;
  LinearPlan gate_up_;
  LinearPlan mlp_to_hidden_;
};

inline void log_nvfp4_policy(const WeightArena& weights, gewell::nvfp4::ActivationPolicy policy) {
  console::field("nvfp4_activation_policy", gewell::nvfp4::activation_policy_name(policy));
  console::field("nvfp4_large_mlp_backend", weights.has_native()
      ? "cutlass_sm120_256x128x128_stream_k_rows_512_4096" : "disabled");
  console::field("native_decode_backend", weights.has_fp8()
      ? (weights.has_native()
          ? (policy == gewell::nvfp4::ActivationPolicy::prefill
              ? "cublasLt_sm120_fp8_bf16_dequantized_nvfp4" : "cublasLt_sm120_fp8_fp4")
          : "cublasLt_sm120_fp8")
      : !weights.has_native() ? "disabled" :
      policy == gewell::nvfp4::ActivationPolicy::prefill
          ? "bf16_gemm_dequantized_nvfp4" : "cublasLt_sm120_fp4");
  console::field("native_bf16_weight_scratch_bytes_per_executor",
      weights.has_native() && policy == gewell::nvfp4::ActivationPolicy::prefill
          ? gewell::nvfp4::ProjectionPlans::dequantized_weight_bytes() : 0);
}

struct BatchMtpInput {
  kv_cache::ExecutionId execution{};
  std::uint32_t pending_token{}, position{}, depth{};
  const BFloat16* target_hidden{};
  float temperature{}, top_p{1.0F};
  std::uint32_t top_k{};
  bool return_probabilities{};
  std::vector<float> uniforms;
  std::function<void(const std::uint32_t*, std::uint32_t, std::uint32_t*)> constraint_mask;
  MtpCaptureFeatures* capture{};
  MtpTargetProbes* capture_next{};
};

struct BatchMtpCommitInput {
  std::size_t row{};
  kv_cache::ExecutionId execution{};
  std::uint32_t position{}, count{};
  BFloat16* terminal_hidden{};
  void* host_logits{};
  TokenLogprobs* device_records{};
  TokenLogprobs* host_records{};
  std::uint32_t top_logprobs{};
};

inline std::size_t decode_attention_scratch_bytes(std::uint32_t context,
                                                  std::uint32_t batch_capacity) {
  if (!batch_capacity || batch_capacity > kMaxBatchRows)
    fail("decode attention scratch", "invalid batch capacity");
  const auto one = std::max(
      primitives::causal_gqa_attention_cached_m1_fused_scratch_bytes(kLocalCacheCapacity, model::AttentionKind::local),
      primitives::causal_gqa_attention_cached_m1_fused_scratch_bytes(context, model::AttentionKind::global));
  // Supply disjoint partials for up to 16 concurrent requests, bounded by
  // 128 MiB or the original single-request requirement at long contexts.
  return std::max(one, std::min(one * std::min(batch_capacity, 16U), std::size_t(128) << 20));
}

class Executor {
 public:
  Executor(const WeightArena& weights,
                          std::size_t prompt_tokens,
                          std::uint32_t new_token_count,
                          bool multimodal = false,
                          SamplingSettings sampling = {},
                          ExecutionCache* persistent_cache = nullptr,
                          kv_cache::ExecutionId persistent_execution = 0,
                          std::vector<CheckpointTrigger> checkpoint_triggers = {},
                          std::uint32_t mtp_depth = 0,
                          std::optional<std::uint64_t> seed = {},
                          std::uint32_t chunk_cap = kRuntimeChunkTokens,
                          gewell::nvfp4::ActivationPolicy activation_policy = gewell::nvfp4::ActivationPolicy::always,
                          std::uint32_t decode_batch_capacity = 1,
                          kv_cache::Format local_format = kv_cache::Format::bf16,
                          kv_cache::Format global_format = kv_cache::Format::bf16,
                          attention::Compute local_compute = attention::Compute::bf16,
                          attention::Compute global_compute = attention::Compute::bf16,
                          std::uint32_t packed_rows = 0)
      : weights_(weights),
        activation_policy_(activation_policy),
        prompt_tokens_(prompt_tokens),
        new_token_count_(new_token_count),
        global_capacity_(
            generation_cache_capacity(prompt_tokens, new_token_count)),
        prefill_chunk_tokens_(
            generation_prefill_chunk_tokens(prompt_tokens, chunk_cap)),
        persistent_cache_(persistent_cache),
        persistent_execution_(persistent_execution),
        checkpoint_triggers_(std::move(checkpoint_triggers)),
        scratch_layout_(std::max({kMultimodalChunkTokens, checked_prefill_chunk_tokens(chunk_cap), packed_rows})),
        scratch_(scratch_layout_.kBytes),
        attention_scratch_(std::max(decode_attention_scratch_bytes(global_capacity_, decode_batch_capacity),
            local_compute == attention::Compute::fp8 || global_compute == attention::Compute::fp8
                ? mtp_attention::scratch_bytes(1, global_capacity_) : std::size_t{0})),
        prefill_attention_scratch_(prefill::tensor_attention_scratch_bytes(
            std::max({prefill_chunk_tokens_, chunk_cap, kMultimodalChunkTokens}))),
        caches_(global_capacity_, persistent_cache, persistent_execution, local_format, global_format),
        outputs_(static_cast<std::size_t>(new_token_count) *
                 sizeof(std::uint32_t)),
        primary_plans_(prefill_chunk_tokens_),
        decode_plans_(1),
        lm_head_(1, model::kHiddenSize, model::kVocabSize),
        sampling_(sampling),
        sampling_scratch_(
            sampling.temperature > 0.0F && sampling.top_p > 0.0F &&
                    sampling.top_k != 1
                ? std::make_unique<DeviceAllocation>(
                      primitives::sampling_scratch_bytes(model::kVocabSize))
                : nullptr) {
    local_compute_ = local_compute;
    global_compute_ = global_compute;
    if (local_compute == attention::Compute::fp8 || global_compute == attention::Compute::fp8)
      fp8_attention_ = std::make_unique<prefill::Fp8Attention>(std::max(prefill_chunk_tokens_, chunk_cap));
    if (weights.has_native()) {
      nvfp4_projections_ = std::make_unique<gewell::nvfp4::ProjectionPlans>(
          handle_.get(), weights.native_weights(), activation_policy_);
      nvfp4_projections_->prepare(prefill_chunk_tokens_);
      nvfp4_projections_->prepare(1);
    }
    if (weights.has_fp8()) {
      fp8_projections_ = std::make_unique<gewell::fp8::ProjectionPlans>(handle_.get(), weights.fp8_weights());
      fp8_projections_->prepare(prefill_chunk_tokens_);
      fp8_projections_->prepare(1);
    }
    if (seed) rng_.seed(*seed);
    if (mtp_depth > mtp_target::kMaxDepth)
      fail("MTP depth", "fixed depth must be in 0..1279");
    mtp_depth_ = std::min(mtp_depth, new_token_count > 1 ? new_token_count - 2 : 0);
    if (mtp_depth_) {
      const std::size_t bytes = mtp_target::Verifier::staging_bytes(mtp_depth_ + 1);
      void* staging = nullptr;
      if (persistent_cache_) {
        staging = persistent_cache_->acquire_speculative_staging(
            persistent_execution_, bytes);
      } else {
        mtp_staging_ = std::make_unique<DeviceAllocation>(bytes);
        staging = mtp_staging_->data();
      }
      mtp_ = std::make_unique<mtp_cycle::Cycle>(
          handle_.get(), weights.pointers(), global_capacity_, mtp_depth_,
          staging, bytes, &weights.native_weights(), activation_policy_, &weights.fp8_weights(),
          local_compute_, global_compute_);
      mtp_uniforms_.resize(2 * mtp_depth_ + 1);
    }
    std::uint32_t previous_trigger = 0;
    for (const CheckpointTrigger& trigger : checkpoint_triggers_) {
      if (trigger.processed_tokens == 0 ||
          trigger.processed_tokens > global_capacity_ ||
          trigger.processed_tokens <= previous_trigger) {
        fail("generation checkpoint", "checkpoint triggers are not ordered");
      }
      previous_trigger = trigger.processed_tokens;
    }
    const std::uint32_t tail_rows = multimodal
                                        ? 0
                                        : static_cast<std::uint32_t>(
                                              prompt_tokens %
                                              prefill_chunk_tokens_);
    if (!multimodal && prompt_tokens > prefill_chunk_tokens_ &&
        tail_rows != 0) {
      tail_plans_ = std::make_unique<RuntimeChunkPlans>(tail_rows);
    }
    for (std::size_t layer = 0; layer < model::kLayerCount; ++layer) {
      const LayerWeightIds& ids = kWeightIds[layer];
      layers_[layer] = {
          weights.pointer(ids.input_norm),
          weights.projection_pointer(ids.q_proj),
          weights.projection_pointer(ids.k_proj),
          model::is_global_layer(layer) ? nullptr : weights.projection_pointer(ids.v_proj),
          weights.pointer(ids.q_norm),
          weights.pointer(ids.k_norm),
          weights.projection_pointer(ids.o_proj),
          weights.pointer(ids.post_attention_norm),
          weights.pointer(ids.pre_feedforward_norm),
          weights.projection_pointer(ids.gate_proj),
          weights.projection_pointer(ids.up_proj),
          weights.projection_pointer(ids.down_proj),
          weights.pointer(ids.post_feedforward_norm),
          weights.pointer(ids.layer_scalar),
      };
    }
  }

  [[nodiscard]] std::size_t scratch_bytes() const {
    return scratch_.size() + native_scratch_bytes() + attention_scratch_.size() +
           prefill_attention_scratch_bytes() + sampling_scratch_bytes() +
           (batch_logits_ ? batch_logits_->size() : 0) +
           (batch_tokens_ ? batch_tokens_->size() : 0) +
           (mtp_ ? mtp_->scratch_bytes() : 0) +
           (batch_mtp_ ? batch_mtp_->scratch_bytes() : 0) +
           (mtp_staging_ ? mtp_staging_->size() : 0);
  }
  [[nodiscard]] std::size_t model_scratch_bytes() const {
    return scratch_.size() + native_scratch_bytes();
  }
  [[nodiscard]] std::size_t attention_scratch_bytes() const {
    return attention_scratch_.size();
  }
  [[nodiscard]] std::size_t prefill_attention_scratch_bytes() const {
    return prefill_attention_scratch_.size() +
           (fp8_attention_ ? fp8_attention_->scratch_bytes() : 0);
  }
  [[nodiscard]] std::size_t sampling_scratch_bytes() const {
    return (sampling_scratch_ ? sampling_scratch_->size() : 0) +
           (constraint_sampling_rows_ ? constraint_sampling_rows_->size() : 0) +
           (constraint_probabilities_ ? constraint_probabilities_->size() : 0);
  }
  [[nodiscard]] std::size_t host_scratch_bytes() const {
    return (constraint_host_rows_ ? constraint_host_rows_->size() : 0) +
           constraint_sampling_pending_.capacity() * sizeof(std::uint8_t) +
           batch_host_tokens_.capacity() * sizeof(std::uint32_t) +
           mtp_uniforms_.capacity() * sizeof(float) +
           (mtp_ ? mtp_->host_scratch_bytes() : 0) +
           (batch_mtp_ ? batch_mtp_->host_scratch_bytes() : 0);
  }
  [[nodiscard]] std::size_t local_cache_bytes() const {
    return caches_.local_bytes();
  }
  [[nodiscard]] std::size_t global_cache_bytes() const {
    return caches_.global_bytes();
  }
  [[nodiscard]] std::uint32_t global_capacity() const {
    return global_capacity_;
  }
  [[nodiscard]] std::size_t output_bytes() const { return outputs_.size(); }

  [[nodiscard]] const BFloat16* terminal_hidden() const {
    return at<BFloat16>(scratch_layout_.kH1);
  }
  [[nodiscard]] cudaStream_t stream() const { return stream_.get(); }

  // Forced text uses the existing causal prefill kernels and fresh static KV.
  // h1 stays available until the next chunk, so the large output head can be tiled.
  void replay_chunk(const std::uint32_t* tokens, std::uint32_t position,
                    std::uint32_t rows) {
    if (persistent_cache_ || !rows || rows > prefill_chunk_tokens_ ||
        std::uint64_t(position) + rows > global_capacity_)
      fail("replay chunk", "invalid static-cache replay range");
    prefill_chunk(tokens, position, rows, plans_for(rows), false, nullptr);
    primitives::rms_norm(at<BFloat16>(scratch_layout_.kH0),
        weights_.pointer(model::kFinalNormPhysicalId),
        at<BFloat16>(scratch_layout_.kH1), rows,
        model::kHiddenSize, 1.0e-6F, stream_.get());
    replay_chunk_rows_ = rows;
  }

  void replay_head(std::uint32_t first_row, std::uint32_t rows,
                   BFloat16* logits, BFloat16* capped) {
    if (!rows || std::uint64_t(first_row) + rows > replay_chunk_rows_)
      fail("replay head", "rows are outside the current prefill chunk");
    auto& plan = replay_heads_[rows];
    if (!plan) plan = std::make_unique<LinearPlan>(rows, model::kHiddenSize, model::kVocabSize);
    plan->run(handle_.get(), terminal_hidden() + std::size_t(first_row) * model::kHiddenSize,
              weights_.pointer(model::kLmHeadLogicalId), logits, stream_.get());
    primitives::softcap_logits(logits, capped, rows * model::kVocabSize, 30.0F, stream_.get());
  }

  // The offline scheduler shares this executor and its prefill workspace.
  // Cache views are borrowed for each step; no request owns an executor.
  void initialize_batch(std::uint32_t capacity, bool sampled,
                        std::uint32_t mtp_depth = 0, void* staging = nullptr,
                        std::size_t staging_bytes = 0) {
    if (!persistent_cache_ ||
        capacity == 0 || capacity > kMaxBatchRows || mtp_) {
      fail("batch executor", "invalid compact-global batch configuration");
    }
    batch_capacity_ = capacity;
    batch_logits_ = std::make_unique<DeviceAllocation>(
        std::size_t(capacity) * kGenerationLogitRowBytes * 2);
    batch_tokens_ = std::make_unique<DeviceAllocation>(capacity * sizeof(std::uint32_t));
    batch_host_tokens_.resize(capacity);
    const auto sampler_bytes = std::max(mtp_sampling::scratch_bytes(model::kVocabSize),
        sampled ? primitives::sampling_scratch_bytes(model::kVocabSize) : std::size_t{0});
    if (!sampling_scratch_ || sampling_scratch_->size() < sampler_bytes)
      sampling_scratch_ = std::make_unique<DeviceAllocation>(sampler_bytes);
    // One shared probability/scratch row is safe because all draws use this
    // stream. Upload sources/status destinations need independent pinned rows
    // until the scheduler's existing receive synchronization has completed.
    constraint_sampling_rows_ = std::make_unique<DeviceAllocation>(
        std::size_t(capacity) * sizeof(ConstraintSamplingRow));
    constraint_host_rows_ = std::make_unique<PinnedHostAllocation>(
        std::size_t(capacity) * sizeof(ConstraintSamplingRow));
    constraint_probabilities_ = std::make_unique<DeviceAllocation>(
        model::kVocabSize * sizeof(float));
    constraint_sampling_pending_.assign(capacity, 0);
    for (std::uint32_t rows = 1; rows <= capacity; ++rows) {
      plans_for(rows);
      batch_heads_.emplace(rows, std::make_unique<LinearPlan>(
          rows, model::kHiddenSize, model::kVocabSize));
    }
    if (mtp_depth)
      batch_mtp_ = std::make_unique<mtp_cycle::Batch>(handle_.get(), weights_.pointers(),
          global_capacity_, capacity, mtp_depth, staging, staging_bytes,
          &weights_.native_weights(), activation_policy_, &weights_.fp8_weights(),
          local_compute_, global_compute_);
  }

  void prefill_step(kv_cache::ExecutionId execution, const std::uint32_t* tokens,
                    std::uint32_t position, std::uint32_t rows, bool final,
                    BFloat16* saved_hidden = nullptr, const VisionPromptSlice* vision = nullptr) {
    if (vision && (rows > kMultimodalChunkTokens || !vision->soft_features ||
                   vision->begin < position || vision->begin >= vision->end ||
                   std::uint64_t(vision->end) > std::uint64_t(position) + rows ||
                   vision->end - vision->begin > model::kVisionMaxSoftTokenCount))
      fail("image prefill", "requires one complete image within the chunk");
    persistent_cache_->prepare_write(execution, position, rows, stream_.get());
    prefill_chunk(tokens, position, rows, plans_for(rows), final, vision, execution);
    if (!final && saved_hidden) {
      primitives::rms_norm(
          at<BFloat16>(scratch_layout_.kH0) + std::size_t(rows - 1) * model::kHiddenSize,
          weights_.pointer(model::kFinalNormPhysicalId), saved_hidden, 1,
          model::kHiddenSize, 1.0e-6F, stream_.get());
    }
    if (final) {
      check_cuda(cudaMemcpyAsync(batch_capped_logits(0), capped_logits(),
          kGenerationLogitRowBytes, cudaMemcpyDeviceToDevice, stream_.get()),
          "save batch prefill logits");
      check_cuda(cudaMemcpyAsync(batch_output_ids(), argmax(), sizeof(std::uint32_t),
          cudaMemcpyDeviceToDevice, stream_.get()), "save batch prefill decision");
      if (saved_hidden)
        check_cuda(cudaMemcpyAsync(saved_hidden, batch_terminal_hidden(rows - 1),
            model::kHiddenSize * sizeof(BFloat16), cudaMemcpyDeviceToDevice, stream_.get()),
            "save batch prefill terminal hidden");
    }
  }

  void prefill_batch(const std::vector<runtime::BatchPrefillInput>& inputs,
                     const std::vector<VisionPromptSlice>& images) {
    if (inputs.empty() || inputs.size() > batch_capacity_ ||
        (!images.empty() && images.size() != inputs.size()))
      fail("packed prefill", "invalid segment count");
    std::vector<PrefillSegment> segments;
    std::uint32_t rows = 0;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const auto& input = inputs[i];
      const auto* vision = !images.empty() && images[i].soft_features ? &images[i] : nullptr;
      if (!input.execution || !input.tokens || !input.rows || !input.hidden ||
          input.rows > scratch_layout_.kRows - rows)
        fail("packed prefill", "invalid segment or workspace capacity");
      for (const auto& prior : segments)
        if (prior.execution == input.execution)
          fail("packed prefill", "an execution may occur only once per forward");
      if (vision && (vision->begin != input.position || vision->end - vision->begin != input.rows))
        fail("packed prefill", "image segment must contain exactly one whole image");
      segments.push_back({input.tokens, input.position, input.rows, rows, input.execution, vision});
      rows += input.rows;
    }
    for (const auto& input : inputs)
      persistent_cache_->prepare_write(input.execution, input.position, input.rows, stream_.get());
    prefill_chunks(segments, rows, plans_for(rows), false);
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const auto& segment = segments[i];
      primitives::rms_norm(
          at<BFloat16>(scratch_layout_.kH0) +
              std::size_t(segment.offset + segment.rows - 1) * model::kHiddenSize,
          weights_.pointer(model::kFinalNormPhysicalId),
          static_cast<BFloat16*>(inputs[i].hidden.value), 1, model::kHiddenSize, 1.0e-6F, stream_.get());
    }
  }

  void prefix_head_step(kv_cache::ExecutionId execution,
                        BFloat16* saved_hidden = nullptr,
                        const BFloat16* input_hidden = nullptr) {
    auto* hidden = at<BFloat16>(scratch_layout_.kH1);
    auto* logits = at<BFloat16>(scratch_layout_.kLogits);
    if (input_hidden)
      check_cuda(cudaMemcpyAsync(hidden, input_hidden,
          model::kHiddenSize * sizeof(BFloat16), cudaMemcpyDeviceToDevice, stream_.get()),
          "restore shared prefix terminal hidden");
    else persistent_cache_->restore_terminal_hidden(execution, hidden, stream_.get());
    lm_head_.run(handle_.get(), hidden, weights_.pointer(model::kLmHeadLogicalId),
                 logits, stream_.get());
    primitives::softcap_and_argmax(logits, batch_capped_logits(0), batch_output_ids(),
                                   model::kVocabSize, 30.0F, stream_.get());
    if (saved_hidden)
      check_cuda(cudaMemcpyAsync(saved_hidden, hidden,
          model::kHiddenSize * sizeof(BFloat16), cudaMemcpyDeviceToDevice, stream_.get()),
          "save complete-prefix terminal hidden");
  }

  auto run_batch_mtp(const std::vector<BatchMtpInput>& inputs) {
    std::vector<mtp_cycle::BatchInput> proposals;
    proposals.reserve(inputs.size());
    for (const auto& input : inputs) {
      mtp_cycle::BatchInput proposal;
      proposal.pending_token = input.pending_token;
      proposal.target_hidden = input.target_hidden;
      proposal.position = input.position;
      proposal.depth = input.depth;
      for (std::size_t layer = 0; layer < proposal.caches.size(); ++layer)
        proposal.caches[layer] = persistent_cache_->layer(input.execution, layer);
      proposal.temperature = input.temperature;
      proposal.top_p = input.top_p;
      proposal.top_k = input.top_k;
      proposal.return_probabilities = input.return_probabilities;
      proposal.constraint_mask = input.constraint_mask;
      proposal.uniforms = input.uniforms;
      proposal.capture = input.capture;
      proposal.capture_next = input.capture_next;
      proposals.push_back(std::move(proposal));
    }
    return batch_mtp_->run(proposals, stream_.get());
  }

  void commit_batch_mtp(const std::vector<BatchMtpCommitInput>& inputs) {
    std::vector<mtp_target::Caches> views(inputs.size());
    std::vector<mtp_cycle::BatchCommitInput> commits;
    commits.reserve(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const auto& input = inputs[i];
      persistent_cache_->prepare_write(input.execution, input.position,
                                       input.count, stream_.get());
      for (std::size_t layer = 0; layer < views[i].size(); ++layer)
        views[i][layer] = persistent_cache_->layer(input.execution, layer);
      commits.push_back({static_cast<std::uint32_t>(input.row), &views[i],
                         input.count});
    }
    batch_mtp_->commit_batch(commits, stream_.get());
    for (const auto& input : inputs) {
      check_cuda(cudaMemcpyAsync(input.terminal_hidden,
          batch_mtp_->hidden(input.row) +
              std::size_t(input.count - 1) * model::kHiddenSize,
          model::kHiddenSize * sizeof(BFloat16), cudaMemcpyDeviceToDevice,
          stream_.get()), "save batch MTP terminal hidden");
      if (input.host_logits)
        check_cuda(cudaMemcpyAsync(input.host_logits,
            batch_mtp_->logits(input.row),
            std::size_t(input.count) * kGenerationLogitRowBytes,
            cudaMemcpyDeviceToHost, stream_.get()), "copy batch MTP logits");
      if (!input.device_records) continue;
      batch_mtp_->summarize_logprobs(input.row, input.count,
          input.top_logprobs, input.device_records, stream_.get());
      check_cuda(cudaMemcpyAsync(input.host_records, input.device_records,
          std::size_t(input.count) * sizeof(TokenLogprobs),
          cudaMemcpyDeviceToHost, stream_.get()),
          "copy batch MTP logprobs");
    }
  }

  void copy_terminal(BFloat16* destination, const BFloat16* source,
                     std::string_view operation) {
    check_cuda(cudaMemcpyAsync(destination, source, model::kHiddenSize * sizeof(BFloat16),
        cudaMemcpyDeviceToDevice, stream_.get()), operation);
  }
  void wait() noexcept { cudaStreamSynchronize(stream_.get()); }
  void synchronize(std::string_view operation) {
    check_cuda(cudaStreamSynchronize(stream_.get()), operation);
  }
  bool ready(std::string_view operation) {
    const auto status = cudaStreamQuery(stream_.get());
    if (status == cudaSuccess) return true;
    if (status != cudaErrorNotReady) check_cuda(status, operation);
    return false;
  }

  const BFloat16* batch_terminal_hidden(std::uint32_t row) const {
    return terminal_hidden() + std::size_t(row) * model::kHiddenSize;
  }

  BFloat16* batch_capped_logits(std::uint32_t row) const {
    return static_cast<BFloat16*>(batch_logits_->data()) +
        std::size_t(batch_capacity_ + row) * model::kVocabSize;
  }
  std::uint32_t* batch_output_ids() const {
    return static_cast<std::uint32_t*>(batch_tokens_->data());
  }

  void sample_batch_row(std::uint32_t row, const SamplingSettings& settings,
                        std::mt19937_64& rng,
                        const std::uint32_t* allowed_mask = nullptr) {
    if (row >= batch_capacity_ || constraint_sampling_pending_[row])
      fail("batch sampling", "invalid row or pending constrained sample was not checked");
    const bool greedy = settings.temperature == 0 || settings.top_p == 0 || settings.top_k == 1;
    if (!allowed_mask && greedy) return;
    const float uniform = greedy ? 0.0F :
        std::min(std::uniform_real_distribution<float>(0, 1)(rng),
                 std::nextafter(1.0F, 0.0F));
    if (allowed_mask) {
      auto* host = static_cast<ConstraintSamplingRow*>(constraint_host_rows_->data()) + row;
      auto* device = static_cast<ConstraintSamplingRow*>(constraint_sampling_rows_->data()) + row;
      std::copy_n(allowed_mask, host->mask.size(), host->mask.begin());
      host->uniform = uniform;
      host->status = mtp_sampling::Status::success;
      constraint_sampling_pending_[row] = 1;
      check_cuda(cudaMemcpyAsync(device, host, sizeof(*host), cudaMemcpyHostToDevice,
          stream_.get()), "upload constrained sampling mask and uniform");
      auto* probabilities = static_cast<float*>(constraint_probabilities_->data());
      mtp_sampling::build_distribution(batch_capped_logits(row), model::kVocabSize,
          settings.temperature, settings.top_p, settings.top_k, probabilities,
          sampling_scratch_->data(), sampling_scratch_->size(), &device->status,
          stream_.get(), reinterpret_cast<const std::uint32_t*>(device));
      mtp_sampling::sample_distribution(probabilities, model::kVocabSize,
          &device->uniform, batch_output_ids() + row, sampling_scratch_->data(),
          sampling_scratch_->size(), &device->status, stream_.get());
      check_cuda(cudaMemcpyAsync(&host->status, &device->status, sizeof(host->status),
          cudaMemcpyDeviceToHost, stream_.get()), "download constrained sampling status");
      return;
    }
    primitives::sample_top_k_top_p(batch_capped_logits(row), model::kVocabSize,
        settings.temperature, settings.top_p, settings.top_k, uniform,
        sampling_scratch_->data(), sampling_scratch_->size(),
        batch_output_ids() + row, stream_.get());
  }

  // Build the exact post-control target distribution without disturbing the
  // ordinary sampler's selected ID, then reduce it to the bounded HTTP record.
  // Constrained sampling already left that distribution in the shared row.
  void summarize_batch_row(std::uint32_t row,
                           const SamplingSettings& settings,
                           std::uint32_t top_logprobs,
                           TokenLogprobs* output,
                           const std::uint32_t* allowed_mask = nullptr) {
    if (row >= batch_capacity_ || output == nullptr ||
        top_logprobs > kMaxTopLogprobs ||
        (allowed_mask && !constraint_sampling_pending_[row]))
      fail("batch logprobs", "invalid probability summary request");
    auto* device = static_cast<ConstraintSamplingRow*>(
        constraint_sampling_rows_->data()) + row;
    auto* host = static_cast<ConstraintSamplingRow*>(
        constraint_host_rows_->data()) + row;
    auto* probabilities = static_cast<float*>(constraint_probabilities_->data());
    if (!allowed_mask) {
      mtp_sampling::clear_status(&device->status, stream_.get());
      mtp_sampling::build_distribution(
          batch_capped_logits(row), model::kVocabSize, settings.temperature,
          settings.top_p, settings.top_k, probabilities,
          sampling_scratch_->data(), sampling_scratch_->size(),
          &device->status, stream_.get());
    }
    mtp_sampling::summarize_logprobs(
        probabilities, batch_output_ids() + row, 1, model::kVocabSize,
        top_logprobs, output, stream_.get());
    constraint_sampling_pending_[row] = 1;
    check_cuda(cudaMemcpyAsync(&host->status, &device->status,
        sizeof(host->status), cudaMemcpyDeviceToHost, stream_.get()),
        "download logprob distribution status");
  }

  // The caller must have completed its existing receive stream synchronization.
  // This method reads pinned status words and never waits on the GPU itself.
  void check_constraint_sampling() {
    auto error = mtp_sampling::Status::success;
    const auto* host = constraint_host_rows_
        ? static_cast<const ConstraintSamplingRow*>(constraint_host_rows_->data()) : nullptr;
    for (std::size_t row = 0; row < constraint_sampling_pending_.size(); ++row) {
      if (!constraint_sampling_pending_[row]) continue;
      constraint_sampling_pending_[row] = 0;
      if (error == mtp_sampling::Status::success) error = host[row].status;
    }
    if (error != mtp_sampling::Status::success)
      fail("constrained sampling", mtp_sampling::status_message(error));
  }

  void decode_batch(const std::vector<BatchDecodeInput>& inputs) {
    const auto rows = static_cast<std::uint32_t>(inputs.size());
    if (!rows || rows > batch_capacity_) fail("batch decode", "invalid row count");
    const auto& plans = plans_for(rows);
    auto* h0 = at<BFloat16>(scratch_layout_.kH0);
    auto* h1 = at<BFloat16>(scratch_layout_.kH1);
    auto* h2 = at<BFloat16>(scratch_layout_.kH2);
    auto* q_raw = at<BFloat16>(scratch_layout_.kQueryRaw);
    auto* q_norm = at<BFloat16>(scratch_layout_.kQueryNorm);
    auto* q_rope = at<BFloat16>(scratch_layout_.kQueryRope);
    auto* k_raw = at<BFloat16>(scratch_layout_.kKeyRaw);
    auto* k_norm = at<BFloat16>(scratch_layout_.kKeyNorm);
    auto* k_rope = at<BFloat16>(scratch_layout_.kKeyRope);
    auto* v_raw = at<BFloat16>(scratch_layout_.kValueRaw);
    auto* v_norm = at<BFloat16>(scratch_layout_.kValueNorm);
    auto* context = at<BFloat16>(scratch_layout_.kContext);
    auto* gate = at<BFloat16>(scratch_layout_.kGate);
    auto* up = at<BFloat16>(scratch_layout_.kUp);
    auto* product = at<BFloat16>(scratch_layout_.kProduct);
    auto* local_cos = at<BFloat16>(scratch_layout_.kLocalCos);
    auto* local_sin = at<BFloat16>(scratch_layout_.kLocalSin);
    auto* global_cos = at<BFloat16>(scratch_layout_.kGlobalCos);
    auto* global_sin = at<BFloat16>(scratch_layout_.kGlobalSin);
    const auto stream = stream_.get();
    for (std::uint32_t row = 0; row < rows; ++row) {
      const auto& input = inputs[row];
      persistent_cache_->prepare_write(input.execution, input.position, 1, stream);
      batch_host_tokens_[row] = input.token;
      primitives::generate_rope_factors_m1(local_cos + row * model::kLocalHeadSize,
          local_sin + row * model::kLocalHeadSize,
          global_cos + row * model::kGlobalHeadSize,
          global_sin + row * model::kGlobalHeadSize, input.position, stream);
    }
    check_cuda(cudaMemcpyAsync(batch_output_ids(), batch_host_tokens_.data(),
        rows * sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream), "upload batch input tokens");
    primitives::embedding_lookup_device_tokens(weights_.pointer(model::kEmbeddingPhysicalId),
        batch_output_ids(), h0, rows, stream);
    primitives::rms_norm_hidden_rows(h0, layers_[0].input_norm, h1, rows, stream);
    std::vector<primitives::DecodeAttentionInput> attention_inputs(rows > 1 ? rows : 0);
    std::vector<mtp_attention::BatchInput> fp8_inputs(
        local_compute_ == attention::Compute::fp8 || global_compute_ == attention::Compute::fp8 ? rows : 0);
    for (std::uint32_t layer = 0; layer < model::kLayerCount; ++layer) {
      const bool global = model::is_global_layer(layer);
      const bool fp8_compute = (global ? global_compute_ : local_compute_) == attention::Compute::fp8;
      const auto kind = global ? model::AttentionKind::global : model::AttentionKind::local;
      const auto head_size = global ? model::kGlobalHeadSize : model::kLocalHeadSize;
      const auto kv_heads = global ? model::kGlobalKvHeadCount : model::kLocalKvHeadCount;
      const auto q_width = model::kQueryHeadCount * head_size;
      const auto kv_width = kv_heads * head_size;
      const auto& weight = layers_[layer];
      project_qkv(plans, rows, layer, h1, q_raw, k_raw, v_raw,
                  q_norm, k_norm, v_norm, stream, gewell::nvfp4::Phase::decode);
      for (std::uint32_t row = 0; row < rows; ++row) {
        const auto& input = inputs[row];
        const auto cache = persistent_cache_->layer(input.execution, layer);
        const auto* cosine = (global ? global_cos : local_cos) + row * head_size;
        const auto* sine = (global ? global_sin : local_sin) + row * head_size;
        auto* q = q_rope + std::size_t(row) * q_width;
        auto* k = k_rope + std::size_t(row) * kv_width;
        auto* v = v_norm + std::size_t(row) * kv_width;
        auto* result = context + std::size_t(row) * q_width;
        if (rows > 1 && !fp8_compute) {
          attention_inputs[row] = {q_norm + std::size_t(row) * q_width,
              k_norm + std::size_t(row) * kv_width, v, cosine, sine, q,
              cache.key, cache.value,
              {cache.page_pool, cache.page_offsets, cache.page_tokens, cache.page_count,
               cache.page_stride_elements, cache.layer_offset_elements, cache.format},
              input.position, result, cache.format};
          continue;
        }
        primitives::apply_rope_m1(q_norm + std::size_t(row) * q_width,
                                  cosine, sine, q, model::kQueryHeadCount, kind, stream);
        primitives::apply_rope_m1(k_norm + std::size_t(row) * kv_width,
                                  cosine, sine, k, kv_heads, kind, stream);
        if (global) {
          const prefill::CompactGlobalPagedCache paged{cache.page_pool,
              cache.page_offsets, cache.page_tokens, cache.page_count,
              cache.page_stride_elements, cache.layer_offset_elements, cache.format};
          prefill::write_kv_cache_chunk_global_compact_paged(k, v, paged,
                                                           input.position, 1, stream);
          if (!fp8_compute)
            primitives::causal_gqa_attention_cached_m1_fused_global_compact_paged(
                q, paged, weight.k_norm, input.position, attention_scratch_.data(), result, stream);
        } else {
          primitives::write_kv_cache_m1(k, v, cache.key, cache.value,
                                        input.position, cache.capacity, kind, stream, cache.format);
          if (!fp8_compute)
            primitives::causal_gqa_attention_cached_m1_fused(q, cache.key, cache.value,
                input.position, cache.capacity, attention_scratch_.data(), result, kind, stream, cache.format);
        }
        if (fp8_compute)
          fp8_inputs[row] = {q, nullptr, nullptr, cache, input.position + 1, 1, result};
      }
      if (fp8_compute)
        mtp_attention::run_fp8_batch(fp8_inputs, weight.k_norm, kind,
            attention_scratch_.data(), attention_scratch_.size(), stream, true);
      else if (rows > 1)
        primitives::decode_attention_batch(attention_inputs, weight.k_norm,
            attention_scratch_.data(), attention_scratch_.size(), kind, stream);
      projection_linear(plans.output(global), rows, layer, model::TensorRole::o_proj,
                        context, weight.o_proj, h2, stream, gewell::nvfp4::Phase::decode);
      primitives::post_attention_residual_pre_feedforward_norm_rows(h2,
          weight.post_attention_norm, h0, weight.pre_feedforward_norm, h1, rows, stream);
      project_mlp(plans, rows, layer, h1, gate, up, product, h0,
                  stream, gewell::nvfp4::Phase::decode);
      const auto* next_norm = layer + 1 < model::kLayerCount ? layers_[layer + 1].input_norm
                                           : weights_.pointer(model::kFinalNormPhysicalId);
      primitives::post_feedforward_residual_scalar_next_norm_rows(h0,
          weight.post_feedforward_norm, h2, weight.layer_scalar, next_norm, h1, rows, stream);
    }
    auto* logits = static_cast<BFloat16*>(batch_logits_->data());
    batch_heads_.at(rows)->run(handle_.get(), h1, weights_.pointer(model::kLmHeadLogicalId),
                               logits, stream);
    if (rows > 1)
      primitives::softcap_and_argmax_rows(logits, batch_capped_logits(0), batch_output_ids(),
          rows, model::kVocabSize, 30.0F, stream);
    else
      primitives::softcap_and_argmax(logits, batch_capped_logits(0), batch_output_ids(),
          model::kVocabSize, 30.0F, stream);
  }

  GenerationResult generate(const std::vector<std::uint32_t>& prompt,
                            GenerationLogitsSink* logits_output,
                            GenerationDecisionSink* decision_sink = nullptr,
                            const std::vector<VisionPromptSlice>& images = {},
                            GenerationCheckpointSink* checkpoint_sink =
                                nullptr) {
    if (prompt.size() != prompt_tokens_) {
      fail("generation execution", "prompt length changed after planning");
    }
    std::uint32_t previous_image_end = 0;
    for (const auto& image : images) {
      if (!image.soft_features || image.begin < previous_image_end ||
          image.begin >= image.end || image.end > prompt.size() ||
          image.end - image.begin > model::kVisionMaxSoftTokenCount)
        fail("generation execution", "invalid ordered vision prompt slices");
      previous_image_end = image.end;
    }
    if (!images.empty() && (checkpoint_sink || !checkpoint_triggers_.empty()))
      fail("generation execution", "image prompts do not support checkpoints");
    GenerationResult result;
    result.outputs.resize(new_token_count_);

    const auto prefill_wall_started = std::chrono::steady_clock::now();
    check_cuda(cudaEventRecord(prefill_begin_.get(), stream_.get()),
               "record generation prefill start");
    std::size_t base = persistent_cache_ == nullptr
                           ? 0
                           : persistent_cache_->processed_tokens(
                                 persistent_execution_);
    if (base > prompt.size()) {
      fail("generation execution",
           "persistent checkpoint is not a prefix of the submitted prompt");
    }
    if (!images.empty() && base)
      fail("generation execution", "image prompts do not support prefix reuse");
    const std::size_t resumed_tokens = base;
    std::size_t checkpoint_trigger_index = 0;
    while (checkpoint_trigger_index < checkpoint_triggers_.size() &&
           checkpoint_triggers_[checkpoint_trigger_index].processed_tokens <=
               base) {
      ++checkpoint_trigger_index;
    }
    // Capture boundaries may occur after the response has begun streaming.
    // Keep their complete token history in one bounded allocation acquired
    // before prefill rather than materializing a new prefix at each boundary.
    std::vector<std::uint32_t> checkpoint_history;
    const bool retain_checkpoint_history =
        checkpoint_sink != nullptr &&
        checkpoint_trigger_index < checkpoint_triggers_.size();
    if (retain_checkpoint_history) {
      checkpoint_history.reserve(global_capacity_);
      checkpoint_history.insert(checkpoint_history.end(), prompt.begin(),
                                prompt.begin() + base);
    }
    std::uint32_t final_prefill_rows = 1;
    std::size_t image_index = 0;
    while (base < prompt.size()) {
      const std::size_t chunk_begin = base;
      const VisionPromptSlice* next_image = image_index < images.size()
                                               ? &images[image_index] : nullptr;
      std::size_t rows = multimodal_prefill_chunk_rows(
          static_cast<std::uint32_t>(base), static_cast<std::uint32_t>(prompt.size()),
          prefill_chunk_tokens_, next_image ? next_image->begin : 0,
          next_image ? next_image->end : 0);
      const VisionPromptSlice* vision = next_image && next_image->begin == base
                                            ? next_image : nullptr;
      const CheckpointTrigger* trigger =
          checkpoint_trigger_index < checkpoint_triggers_.size()
              ? &checkpoint_triggers_[checkpoint_trigger_index]
              : nullptr;
      if (trigger != nullptr && trigger->processed_tokens <= prompt.size()) {
        if (trigger->processed_tokens <= base) {
          fail("generation checkpoint", "checkpoint trigger was skipped");
        }
        rows = std::min<std::size_t>(
            rows, static_cast<std::size_t>(trigger->processed_tokens) - base);
      }
      const std::uint32_t chunk_rows = static_cast<std::uint32_t>(rows);
      const bool capture_boundary =
          trigger != nullptr && trigger->processed_tokens == base + rows;
      const bool final_chunk = base + rows == prompt.size();
      if (final_chunk) final_prefill_rows = chunk_rows;
      if (persistent_cache_ != nullptr) {
        persistent_cache_->prepare_write(
            persistent_execution_, static_cast<std::uint32_t>(base), chunk_rows,
            stream_.get());
      }
      prefill_chunk(prompt.data() + base, static_cast<std::uint32_t>(base),
                    chunk_rows, plans_for(chunk_rows),
                    final_chunk || capture_boundary, vision);
      base += rows;
      if (vision) ++image_index;
      if (retain_checkpoint_history) {
        if (rows > checkpoint_history.capacity() -
                       checkpoint_history.size()) {
          fail("generation checkpoint", "checkpoint history exceeds capacity");
        }
        checkpoint_history.insert(checkpoint_history.end(),
                                  prompt.begin() + chunk_begin,
                                  prompt.begin() + base);
      }
      if (capture_boundary) {
        if (checkpoint_sink != nullptr) {
          const BFloat16* const boundary_terminal =
              at<BFloat16>(scratch_layout_.kH1) +
              static_cast<std::size_t>(chunk_rows - 1) * model::kHiddenSize;
          checkpoint_sink->capture_checkpoint(
              trigger->source, checkpoint_history, boundary_terminal,
              stream_.get());
        }
        ++checkpoint_trigger_index;
      }
    }
    BFloat16* const hidden_scratch =
        at<BFloat16>(scratch_layout_.kH1);
    const bool complete_hit = persistent_cache_ != nullptr &&
                              resumed_tokens == prompt.size();
    if (complete_hit) {
      persistent_cache_->restore_terminal_hidden(persistent_execution_,
                                                 hidden_scratch, stream_.get());
      // The saved vector is the final-normed hidden of the checkpoint
      // boundary. No prefill chunk ran, so the output head must run here
      // before the first decision is selected from it.
      lm_head_.run(handle_.get(), hidden_scratch,
                   weights_.pointer(model::kLmHeadLogicalId),
                   at<BFloat16>(scratch_layout_.kLogits),
                   stream_.get());
      primitives::softcap_and_argmax(
          at<BFloat16>(scratch_layout_.kLogits),
          capped_logits(), argmax(), model::kVocabSize, 30.0F, stream_.get());
    }
    if (!complete_hit && final_prefill_rows > 1) {
      check_cuda(cudaMemcpyAsync(hidden_scratch,
          hidden_scratch + std::size_t(final_prefill_rows - 1) * model::kHiddenSize,
          model::kHiddenSize * sizeof(BFloat16), cudaMemcpyDeviceToDevice,
          stream_.get()), "preserve generation terminal hidden");
    }
    select_output_token(capped_logits());
    check_cuda(cudaMemcpyAsync(
                   outputs_.data(), argmax(), sizeof(std::uint32_t),
                   cudaMemcpyDeviceToDevice, stream_.get()),
               "store generation output zero");
    check_cuda(cudaEventRecord(prefill_end_.get(), stream_.get()),
               "record generation prefill end");
    if (logits_output != nullptr) {
      logits_output->write_device_row(capped_logits(), stream_.get());
    }
    std::uint32_t output_count = 1;
    bool continue_generation =
        decision_sink == nullptr ||
        decision_sink->write_device_decision(argmax(), capped_logits(),
                                             stream_.get());
    std::uint32_t pending_output = 0;
    const bool retain_decode_checkpoints =
        retain_checkpoint_history &&
        checkpoint_trigger_index < checkpoint_triggers_.size();
    if ((retain_decode_checkpoints || mtp_) && continue_generation) {
      check_cuda(cudaMemcpyAsync(&pending_output, argmax(),
                                 sizeof(pending_output), cudaMemcpyDeviceToHost,
                                 stream_.get()),
                 "copy generation checkpoint token");
      check_cuda(cudaStreamSynchronize(stream_.get()),
                 "synchronize generation checkpoint token");
    }
    check_cuda(cudaEventSynchronize(prefill_end_.get()),
               "synchronize generation prefill");
    result.prefill_wall_seconds = seconds_since(prefill_wall_started);
    check_cuda(cudaEventElapsedTime(&result.prefill_gpu_milliseconds,
                                    prefill_begin_.get(), prefill_end_.get()),
               "measure generation prefill");

    const auto decode_wall_started = std::chrono::steady_clock::now();
    const bool measure_decode_steps =
        logits_output != nullptr || decision_sink != nullptr;
    if (!measure_decode_steps) {
      check_cuda(cudaEventRecord(decode_begin_.get(), stream_.get()),
                 "record generation decode start");
    }
    for (std::uint32_t output_index = 1;
         continue_generation && output_index < new_token_count_;
         output_index = output_count) {
      if (measure_decode_steps) {
        check_cuda(cudaEventRecord(decode_begin_.get(), stream_.get()),
                   "record generation decode step start");
      }
      const std::uint32_t position =
          static_cast<std::uint32_t>(prompt_tokens_) + output_index - 1;
      std::uint32_t depth = std::min(mtp_depth_, new_token_count_ - output_index - 1);
      if (checkpoint_trigger_index < checkpoint_triggers_.size()) {
        const auto boundary = checkpoint_triggers_[checkpoint_trigger_index].processed_tokens;
        if (boundary > position) depth = std::min(depth, boundary - position - 1);
      }
      if (mtp_ && depth > 0) {
        if (result.mtp_accepted_histogram.empty())
          result.mtp_accepted_histogram.resize(mtp_depth_ + 1);
        mtp_->prepare(depth);
        mtp_target::Caches views;
        for (std::size_t layer = 0; layer < views.size(); ++layer) views[layer] = caches_.layer(layer);
        mtp_uniforms_.resize(2 * depth + 1);
        for (float& u : mtp_uniforms_) u = std::min(
            std::uniform_real_distribution<float>(0.0F, 1.0F)(rng_),
            std::nextafter(1.0F, 0.0F));
        const auto cycle = mtp_->run(argmax(), terminal_hidden(), position, depth, views,
            sampling_.temperature, sampling_.top_p, sampling_.top_k, mtp_uniforms_, stream_.get());
        std::uint32_t emitted = static_cast<std::uint32_t>(cycle.tokens.size());
        if (decision_sink && decision_sink->honors_stop_tokens()) {
          for (std::uint32_t i = 0; i < emitted; ++i) {
            if (is_generation_stop_token(cycle.tokens[i])) { emitted = i + 1; break; }
          }
        }
        if (persistent_cache_) {
          persistent_cache_->prepare_write(persistent_execution_, position, emitted, stream_.get());
          for (std::size_t layer = 0; layer < views.size(); ++layer) views[layer] = caches_.layer(layer);
        }
        mtp_->commit(views, position, emitted, stream_.get());
        check_cuda(cudaMemcpyAsync(at<BFloat16>(scratch_layout_.kH1),
            mtp_->hidden() + std::size_t(emitted - 1) * model::kHiddenSize,
            model::kHiddenSize * sizeof(BFloat16), cudaMemcpyDeviceToDevice, stream_.get()),
            "preserve MTP terminal hidden");
        check_cuda(cudaMemcpyAsync(static_cast<std::uint32_t*>(outputs_.data()) + output_index,
            mtp_->output_ids(), emitted * sizeof(std::uint32_t), cudaMemcpyDeviceToDevice,
            stream_.get()), "store MTP outputs");
        check_cuda(cudaMemcpyAsync(argmax(), mtp_->output_ids() + emitted - 1,
            sizeof(std::uint32_t), cudaMemcpyDeviceToDevice, stream_.get()), "store MTP pending token");
        if (retain_decode_checkpoints) {
          checkpoint_history.push_back(pending_output);
          checkpoint_history.insert(checkpoint_history.end(), cycle.tokens.begin(), cycle.tokens.begin() + emitted - 1);
        }
        pending_output = cycle.tokens[emitted - 1];
        const auto processed = position + emitted;
        if (checkpoint_trigger_index < checkpoint_triggers_.size() &&
            checkpoint_triggers_[checkpoint_trigger_index].processed_tokens == processed) {
          const auto& trigger = checkpoint_triggers_[checkpoint_trigger_index++];
          if (checkpoint_sink) checkpoint_sink->capture_checkpoint(trigger.source, checkpoint_history,
              terminal_hidden(), stream_.get());
        }
        output_count = output_index + emitted;
        ++result.mtp_cycles;
        result.mtp_proposed += depth;
        const auto accepted = std::min(cycle.verification.accepted_drafts, emitted);
        result.mtp_accepted += accepted;
        ++result.mtp_accepted_histogram.at(accepted);
        result.mtp_rejected += cycle.verification.rejected_index < depth &&
                               cycle.verification.rejected_index < emitted;
        result.mtp_draft_gpu_milliseconds += cycle.draft_gpu_milliseconds;
        result.mtp_verify_gpu_milliseconds += cycle.verify_gpu_milliseconds;
        result.mtp_select_gpu_milliseconds += cycle.select_gpu_milliseconds;
        if (measure_decode_steps)
          check_cuda(cudaEventRecord(decode_end_.get(), stream_.get()), "record MTP end");
        for (std::uint32_t i = 0; i < emitted; ++i) {
          const auto* logits = mtp_->logits() + std::size_t(i) * model::kVocabSize;
          if (logits_output) logits_output->write_device_row(logits, stream_.get());
          if (decision_sink) continue_generation = decision_sink->write_device_decision(
              mtp_->output_ids() + i, logits, stream_.get());
        }
        if (measure_decode_steps) {
          check_cuda(cudaEventSynchronize(decode_end_.get()), "synchronize MTP timing");
          float elapsed = 0;
          check_cuda(cudaEventElapsedTime(&elapsed, decode_begin_.get(), decode_end_.get()), "measure MTP cycle");
          result.decode_gpu_milliseconds += elapsed;
        }
        continue;
      }
      if (retain_decode_checkpoints) {
        if (checkpoint_history.size() == checkpoint_history.capacity()) {
          fail("generation checkpoint", "checkpoint history exceeds capacity");
        }
        checkpoint_history.push_back(pending_output);
      }
      if (persistent_cache_ != nullptr) {
        persistent_cache_->prepare_write(persistent_execution_, position, 1,
                                         stream_.get());
      }
      decode_generated_token(position);
      const std::uint32_t processed_boundary = position + 1;
      while (checkpoint_trigger_index < checkpoint_triggers_.size() &&
             checkpoint_triggers_[checkpoint_trigger_index].processed_tokens <=
                 processed_boundary) {
        const CheckpointTrigger& trigger =
            checkpoint_triggers_[checkpoint_trigger_index];
        if (trigger.processed_tokens != processed_boundary) {
          fail("generation checkpoint", "decode checkpoint trigger was skipped");
        }
        if (checkpoint_sink != nullptr) {
          checkpoint_sink->capture_checkpoint(trigger.source, checkpoint_history,
                                               terminal_hidden(), stream_.get());
        }
        ++checkpoint_trigger_index;
      }
      select_output_token(capped_logits());
      check_cuda(cudaMemcpyAsync(
                     static_cast<std::uint32_t*>(outputs_.data()) +
                         output_index,
                     argmax(), sizeof(std::uint32_t),
                     cudaMemcpyDeviceToDevice, stream_.get()),
                 "store generation decode output");
      output_count = output_index + 1;
      if (measure_decode_steps) {
        check_cuda(cudaEventRecord(decode_end_.get(), stream_.get()),
                   "record generation decode step end");
      }
      if (logits_output != nullptr) {
        logits_output->write_device_row(capped_logits(), stream_.get());
      }
      if (decision_sink != nullptr) {
        continue_generation = decision_sink->write_device_decision(
            argmax(), capped_logits(), stream_.get());
      }
      if ((retain_decode_checkpoints || mtp_) && continue_generation) {
        check_cuda(cudaMemcpyAsync(&pending_output, argmax(),
                                   sizeof(pending_output),
                                   cudaMemcpyDeviceToHost, stream_.get()),
                   "copy generation checkpoint token");
        check_cuda(cudaStreamSynchronize(stream_.get()),
                   "synchronize generation checkpoint token");
      }
      if (measure_decode_steps) {
        float step_milliseconds = 0.0F;
        check_cuda(cudaEventElapsedTime(&step_milliseconds,
                                        decode_begin_.get(),
                                        decode_end_.get()),
                   "measure generation decode step");
        result.decode_gpu_milliseconds += step_milliseconds;
      }
    }
    if (!measure_decode_steps) {
      check_cuda(cudaEventRecord(decode_end_.get(), stream_.get()),
                 "record generation decode end");
      check_cuda(cudaEventSynchronize(decode_end_.get()),
                 "synchronize generation decode");
      check_cuda(cudaEventElapsedTime(&result.decode_gpu_milliseconds,
                                      decode_begin_.get(), decode_end_.get()),
                 "measure generation decode");
    }
    result.decode_wall_seconds = seconds_since(decode_wall_started);

    result.outputs.resize(output_count);
    check_cuda(cudaMemcpyAsync(result.outputs.data(), outputs_.data(),
                               result.outputs.size() * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost, stream_.get()),
               "copy generation outputs");
    check_cuda(cudaStreamSynchronize(stream_.get()),
               "synchronize generation output copy");
    return result;
  }

 private:
  struct PrefillSegment {
    const std::uint32_t* tokens;
    std::uint32_t position, rows, offset;
    kv_cache::ExecutionId execution;
    const VisionPromptSlice* vision;
  };

  std::size_t native_scratch_bytes() const {
    return (nvfp4_projections_ ? nvfp4_projections_->scratch_bytes() : 0) +
           (fp8_projections_ ? fp8_projections_->scratch_bytes() : 0);
  }

  void projection_linear(const LinearPlan& bf16_plan, std::uint32_t rows,
                  std::uint32_t layer, model::TensorRole role,
                  const BFloat16* input, const BFloat16* bf16_weight,
                  BFloat16* output, cudaStream_t stream, gewell::nvfp4::Phase phase) {
    const auto& ids = kWeightIds[layer];
    const auto id = role == model::TensorRole::q_proj ? ids.q_proj
                  : role == model::TensorRole::k_proj ? ids.k_proj
                  : role == model::TensorRole::v_proj ? ids.v_proj
                  : role == model::TensorRole::o_proj ? ids.o_proj
                  : role == model::TensorRole::gate_proj ? ids.gate_proj
                  : role == model::TensorRole::up_proj ? ids.up_proj
                                                       : ids.down_proj;
    const auto& fp8_weight = weights_.fp8_weights()[id];
    const auto& shape = model::kPhysicalTensors[id].shape;
    if (fp8_weight.data) {
      fp8_projections_->run(rows, shape.dimensions[1], shape.dimensions[0],
                            input, fp8_weight, output, stream);
      return;
    }
    const auto& weight = weights_.native_weights()[id];
    if (weight.data && nvfp4_projections_->fp4_activations(phase)) {
      nvfp4_projections_->run(rows, shape.dimensions[1], shape.dimensions[0],
                       input, weight, output, stream);
    } else {
      if (weight.data)
        bf16_weight = nvfp4_projections_->decode_weight(
            weight, shape.dimensions[1], shape.dimensions[0], stream);
      bf16_plan.run(handle_.get(), input, bf16_weight, output, stream);
    }
  }

  void project_qkv(const RuntimeChunkPlans& plans, std::uint32_t rows,
                   std::uint32_t layer, const BFloat16* input,
                   BFloat16* qr, BFloat16* kr, BFloat16* vr,
                   BFloat16* qn, BFloat16* kn, BFloat16* vn,
                   cudaStream_t stream, gewell::nvfp4::Phase phase) {
    const bool global = model::is_global_layer(layer);
    const auto kind = global ? model::AttentionKind::global : model::AttentionKind::local;
    const auto d = global ? model::kGlobalHeadSize : model::kLocalHeadSize;
    const auto heads = global ? model::kGlobalKvHeadCount : model::kLocalKvHeadCount;
    const auto& w = layers_[layer];
    const auto fp8_joined = gewell::fp8::qkv_weight(weights_.fp8_weights(), layer);
    if (const auto* joined = fusion::qkv_weight(weights_.pointers(), layer)) {
      plans.qkv(global).run(handle_.get(), input, joined, qr, stream);
      primitives::qkv_rms_norm(qr, w.q_norm, w.k_norm, qn, kn, vn, rows, kind, stream);
    } else if (fp8_joined.data) {
      fp8_projections_->run_joined(rows, fusion::qkv_width(global), input, fp8_joined, qr, stream);
      primitives::qkv_rms_norm(qr, w.q_norm, w.k_norm, qn, kn, vn, rows, kind, stream);
    } else {
      projection_linear(plans.q(global), rows, layer, model::TensorRole::q_proj,
                        input, w.q_proj, qr, stream, phase);
      projection_linear(plans.kv(global), rows, layer, model::TensorRole::k_proj,
                        input, w.k_proj, kr, stream, phase);
      if (!global) projection_linear(plans.kv(false), rows, layer, model::TensorRole::v_proj,
                        input, w.v_proj, vr, stream, phase);
      primitives::rms_norm(qr, w.q_norm, qn, rows * model::kQueryHeadCount, d, 1e-6F, stream);
      primitives::rms_norm(kr, w.k_norm, kn, rows * heads, d, 1e-6F, stream);
      primitives::rms_norm_unscaled(global ? kr : vr, vn, rows * heads, d, 1e-6F, stream);
    }
  }

  void project_mlp(const RuntimeChunkPlans& plans, std::uint32_t rows,
                       std::uint32_t layer, const BFloat16* input,
                       BFloat16* gate, BFloat16* up, BFloat16* product,
                       BFloat16* output,
                       cudaStream_t stream, gewell::nvfp4::Phase phase) {
    const auto& w = layers_[layer];
    const auto& ids = kWeightIds[layer];
    const auto native = gewell::nvfp4::gate_up_weight(
        weights_.native_weights()[ids.gate_proj], weights_.native_weights()[ids.up_proj]);
    const auto* bf16 = fusion::gate_up_weight(w.gate_proj, w.up_proj, rows);
    const auto fp8_joined = gewell::fp8::gate_up_weight(
        weights_.fp8_weights()[ids.gate_proj], weights_.fp8_weights()[ids.up_proj]);
    bool interleaved = true;
    if (native.data && nvfp4_projections_->fp4_activations(phase)) {
      nvfp4_projections_->run(rows, model::kHiddenSize, 2 * model::kMlpSize,
                              input, native, gate, stream);
    } else if (fp8_joined.data) {
      fp8_projections_->run_joined(rows, 2 * model::kMlpSize, input, fp8_joined, gate, stream);
    } else if (bf16) {
      plans.gate_up().run(handle_.get(), input, bf16, gate, stream);
    } else {
      projection_linear(plans.hidden_to_mlp(), rows, layer, model::TensorRole::gate_proj,
                        input, w.gate_proj, gate, stream, phase);
      projection_linear(plans.hidden_to_mlp(), rows, layer, model::TensorRole::up_proj,
                        input, w.up_proj, up, stream, phase);
      interleaved = false;
    }
    const auto down = weights_.native_weights()[ids.down_proj];
    const auto fp8_down = weights_.fp8_weights()[ids.down_proj];
    if (interleaved && fp8_down.data) {
      fp8_projections_->run(rows, model::kMlpSize, model::kHiddenSize,
          gate, fp8_down, output, stream, gewell::fp8::InputTransform::gelu_tanh_multiply);
    } else if (interleaved && down.data && nvfp4_projections_->fp4_activations(phase)) {
      nvfp4_projections_->run(rows, model::kMlpSize, model::kHiddenSize,
          gate, down, output, stream, gewell::nvfp4::InputTransform::gelu_tanh_multiply);
    } else {
      if (interleaved)
        primitives::gelu_tanh_multiply_interleaved(gate, product, rows, model::kMlpSize, stream);
      else
        primitives::gelu_tanh_multiply(gate, up, product, std::size_t(rows) * model::kMlpSize, stream);
      projection_linear(plans.mlp_to_hidden(), rows, layer, model::TensorRole::down_proj,
                        product, w.down_proj, output, stream, phase);
    }
  }

  void select_output_token(const BFloat16* logits) {
    if (sampling_.temperature == 0.0F || sampling_.top_p == 0.0F ||
        sampling_.top_k == 1) {
      return;
    }
    const float uniform = std::min(
        std::uniform_real_distribution<float>(0.0F, 1.0F)(rng_),
        std::nextafter(1.0F, 0.0F));
    primitives::sample_top_k_top_p(
        logits, model::kVocabSize, sampling_.temperature, sampling_.top_p,
        sampling_.top_k, uniform, sampling_scratch_->data(),
        sampling_scratch_->size(), argmax(), stream_.get());
  }

  const RuntimeChunkPlans& plans_for(std::uint32_t rows) {
    if (nvfp4_projections_) nvfp4_projections_->prepare(rows);
    if (fp8_projections_) fp8_projections_->prepare(rows);
    if (rows == primary_plans_.rows()) {
      return primary_plans_;
    }
    if (tail_plans_ != nullptr && rows == tail_plans_->rows()) {
      return *tail_plans_;
    }
    const auto found = extra_plans_.find(rows);
    if (found != extra_plans_.end()) {
      return *found->second;
    }
    auto plans = std::make_unique<RuntimeChunkPlans>(rows);
    const RuntimeChunkPlans* const result = plans.get();
    extra_plans_.emplace(rows, std::move(plans));
    return *result;
  }

  void prefill_chunk(const std::uint32_t* tokens, std::uint32_t base_position,
                     std::uint32_t rows, const RuntimeChunkPlans& plans,
                     bool produce_token, const VisionPromptSlice* vision,
                     kv_cache::ExecutionId execution = 0) {
    prefill_chunks({{tokens, base_position, rows, 0, execution, vision}}, rows, plans, produce_token);
  }

  void prefill_chunks(const std::vector<PrefillSegment>& segments,
                      std::uint32_t rows, const RuntimeChunkPlans& plans, bool produce_token) {
    if (!rows || rows > scratch_layout_.kRows)
      fail("prefill chunk", "row count exceeds the configured workspace");
    for (const auto& segment : segments)
      if (!segment.rows ||
          prefill::tensor_attention_scratch_bytes(segment.rows) > prefill_attention_scratch_.size())
        fail("prefill chunk", "segment exceeds the configured attention workspace");
    if (nvfp4_projections_) nvfp4_projections_->prepare(rows);
    if (fp8_projections_) fp8_projections_->prepare(rows);
    BFloat16* const h0 = at<BFloat16>(scratch_layout_.kH0);
    BFloat16* const h1 = at<BFloat16>(scratch_layout_.kH1);
    BFloat16* const h2 = at<BFloat16>(scratch_layout_.kH2);
    BFloat16* const q_raw =
        at<BFloat16>(scratch_layout_.kQueryRaw);
    BFloat16* const q_norm =
        at<BFloat16>(scratch_layout_.kQueryNorm);
    BFloat16* const q_rope =
        at<BFloat16>(scratch_layout_.kQueryRope);
    BFloat16* const k_raw =
        at<BFloat16>(scratch_layout_.kKeyRaw);
    BFloat16* const k_norm =
        at<BFloat16>(scratch_layout_.kKeyNorm);
    BFloat16* const k_rope =
        at<BFloat16>(scratch_layout_.kKeyRope);
    BFloat16* const v_raw =
        at<BFloat16>(scratch_layout_.kValueRaw);
    BFloat16* const v_norm =
        at<BFloat16>(scratch_layout_.kValueNorm);
    BFloat16* const context =
        at<BFloat16>(scratch_layout_.kContext);
    BFloat16* const gate =
        at<BFloat16>(scratch_layout_.kGate);
    BFloat16* const up = at<BFloat16>(scratch_layout_.kUp);
    BFloat16* const product =
        at<BFloat16>(scratch_layout_.kProduct);
    BFloat16* const logits =
        at<BFloat16>(scratch_layout_.kLogits);
    BFloat16* const capped_logits =
        at<BFloat16>(scratch_layout_.kCappedLogits);
    BFloat16* const local_cos =
        at<BFloat16>(scratch_layout_.kLocalCos);
    BFloat16* const local_sin =
        at<BFloat16>(scratch_layout_.kLocalSin);
    BFloat16* const global_cos =
        at<BFloat16>(scratch_layout_.kGlobalCos);
    BFloat16* const global_sin =
        at<BFloat16>(scratch_layout_.kGlobalSin);
    const cudaStream_t stream = stream_.get();

    for (const auto& segment : segments) {
      prefill::generate_rope_factors_chunk(
          local_cos + std::size_t(segment.offset) * model::kLocalHeadSize,
          local_sin + std::size_t(segment.offset) * model::kLocalHeadSize,
          global_cos + std::size_t(segment.offset) * model::kGlobalHeadSize,
          global_sin + std::size_t(segment.offset) * model::kGlobalHeadSize,
          segment.position, segment.rows, stream);
      primitives::embedding_lookup_host_tokens(
          weights_.pointer(model::kEmbeddingPhysicalId), segment.tokens,
          h0 + std::size_t(segment.offset) * model::kHiddenSize, segment.rows, stream);
      if (const auto* vision = segment.vision) {
        check_cuda(cudaMemcpyAsync(
            h0 + std::size_t(segment.offset + vision->begin - segment.position) * model::kHiddenSize,
            vision->soft_features,
            std::size_t(vision->end - vision->begin) * model::kHiddenSize * sizeof(BFloat16),
            cudaMemcpyDeviceToDevice, stream), "insert vision soft features");
      }
    }

    std::vector<primitives::QkvRopeInput> qkv_inputs;
    qkv_inputs.reserve(segments.size());
    primitives::rms_norm(h0, layers_[0].input_norm, h1, rows,
                         model::kHiddenSize, 1.0e-6F, stream);
    for (std::uint32_t layer = 0; layer < model::kLayerCount; ++layer) {
      const bool global = model::is_global_layer(layer);
      const model::AttentionKind kind =
          global ? model::AttentionKind::global : model::AttentionKind::local;
      const std::uint32_t kv_heads =
          global ? model::kGlobalKvHeadCount : model::kLocalKvHeadCount;
      const LayerWeights& weight = layers_[layer];
      const BFloat16* const cosine = global ? global_cos : local_cos;
      const BFloat16* const sine = global ? global_sin : local_sin;

      const auto d = global ? model::kGlobalHeadSize : model::kLocalHeadSize;
      const auto query_width = model::kQueryHeadCount * d;
      const auto kv_width = kv_heads * d;
      const auto fp8_joined = gewell::fp8::qkv_weight(weights_.fp8_weights(), layer);
      const auto* joined = fusion::qkv_weight(weights_.pointers(), layer);
      if (joined) {
        plans.qkv(global).run(handle_.get(), h1, joined, q_raw, stream);
      } else if (fp8_joined.data) {
        fp8_projections_->run_joined(rows, fusion::qkv_width(global), h1, fp8_joined, q_raw, stream);
      } else {
        project_qkv(plans, rows, layer, h1, q_raw, k_raw, v_raw,
                    q_norm, k_norm, v_norm, stream, gewell::nvfp4::Phase::prefill);
      }
      if (joined || fp8_joined.data) {
        qkv_inputs.clear();
        for (const auto& segment : segments) {
          const auto offset = std::size_t(segment.offset);
          qkv_inputs.push_back({q_raw + offset * fusion::qkv_width(global),
              cosine + offset * d, sine + offset * d, q_rope + offset * query_width,
              k_rope + offset * kv_width, v_norm + offset * kv_width, segment.rows});
        }
        primitives::qkv_rms_rope_batch(qkv_inputs, weight.q_norm, weight.k_norm, kind, stream);
      } else {
        for (const auto& segment : segments) {
          const auto offset = std::size_t(segment.offset);
          prefill::apply_rope_transpose_chunk(
              q_norm + offset * query_width, cosine + offset * d, sine + offset * d,
              q_rope + offset * query_width, model::kQueryHeadCount, segment.rows, kind, stream);
          prefill::apply_rope_transpose_chunk(
              k_norm + offset * kv_width, cosine + offset * d, sine + offset * d,
              k_rope + offset * kv_width, kv_heads, segment.rows, kind, stream);
        }
      }
      // Projections share the packed token dimension; attention sees only one
      // segment's private head-major Q/K and committed execution at a time.
      for (const auto& segment : segments) {
        const auto base_position = segment.position;
        const auto rows = segment.rows;
        const auto* vision = segment.vision;
        const bool tensor_attention = rows <= prefill::kTensorAttentionMaximumQueryRows &&
            (rows >= 32 || base_position >= 1024);
        const auto* query = q_rope + std::size_t(segment.offset) * query_width;
        const auto* key = k_rope + std::size_t(segment.offset) * kv_width;
        const auto* value = v_norm + std::size_t(segment.offset) * kv_width;
        auto* segment_context = context + std::size_t(segment.offset) * query_width;
        const LayerCacheView cache = segment.execution
            ? persistent_cache_->layer(segment.execution, layer) : caches_.layer(layer);
        if (global) {
          if (cache.page_pool != nullptr) {
            const prefill::CompactGlobalPagedCache paged_cache{
                cache.page_pool,
                cache.page_offsets,
                cache.page_tokens,
                cache.page_count,
                cache.page_stride_elements,
                cache.layer_offset_elements,
                cache.format,
            };
            if (vision != nullptr && !tensor_attention) {
              prefill::image_block_gqa_attention_cached_chunk_global_compact_paged(
                  query, key, value, paged_cache, weight.k_norm,
                  base_position, rows, vision->begin, vision->end, segment_context,
                  stream);
            } else if (tensor_attention || (!vision && global_compute_ == attention::Compute::fp8)) {
              prefill::
                  causal_gqa_attention_cached_chunk_tensor_global_compact_paged(
                      prefill_attention_handle_.get(), query, key,
                      value, paged_cache, weight.k_norm, base_position,
                      rows, prefill_attention_scratch_.data(), segment_context, stream,
                      !vision && global_compute_ == attention::Compute::fp8 ? fp8_attention_.get() : nullptr);
            } else {
              prefill::causal_gqa_attention_cached_chunk_global_compact_paged(
                  query, key, value, paged_cache, weight.k_norm,
                  base_position, rows, segment_context, stream);
            }
            prefill::write_kv_cache_chunk_global_compact_paged(
                key, value, paged_cache, base_position, rows, stream);
          } else {
            if (vision != nullptr && !tensor_attention) {
              prefill::image_block_gqa_attention_cached_chunk_global_compact(
                  query, key, value, cache.key, weight.k_norm,
                  base_position, rows, cache.capacity, vision->begin,
                  vision->end, segment_context, stream, cache.format);
            } else if (tensor_attention || (!vision && global_compute_ == attention::Compute::fp8)) {
              prefill::
                  causal_gqa_attention_cached_chunk_tensor_global_compact(
                      prefill_attention_handle_.get(), query, key, value,
                      cache.key, weight.k_norm, base_position, rows, cache.capacity,
                      prefill_attention_scratch_.data(), segment_context, stream, cache.format,
                      !vision && global_compute_ == attention::Compute::fp8 ? fp8_attention_.get() : nullptr);
            } else {
              prefill::causal_gqa_attention_cached_chunk_global_compact(
                  query, key, value, cache.key, weight.k_norm,
                  base_position, rows, cache.capacity, segment_context, stream, cache.format);
            }
            prefill::write_kv_cache_chunk_global_compact(
                key, value, cache.key, base_position, rows,
                cache.capacity, stream, cache.format);
          }
        } else {
          if (vision != nullptr && !tensor_attention) {
            prefill::image_block_gqa_attention_cached_chunk(
                query, key, value, cache.key, cache.value,
                base_position, rows, cache.capacity, vision->begin,
                vision->end, segment_context, kind, stream, cache.format);
          } else if (tensor_attention || (!vision && local_compute_ == attention::Compute::fp8)) {
            prefill::causal_gqa_attention_cached_chunk_tensor(
                prefill_attention_handle_.get(), query, key, value,
                cache.key, cache.value, base_position, rows, cache.capacity,
                prefill_attention_scratch_.data(), segment_context, kind, stream, cache.format,
                !vision && local_compute_ == attention::Compute::fp8 ? fp8_attention_.get() : nullptr,
                vision ? vision->begin : 0, vision ? vision->end : 0);
          } else {
            prefill::causal_gqa_attention_cached_chunk(
                query, key, value, cache.key, cache.value, base_position,
                rows, cache.capacity, segment_context, kind, stream, cache.format);
          }
          prefill::write_kv_cache_chunk(
              key, value, cache.key, cache.value, base_position, rows,
              cache.capacity, kind, stream, cache.format);
        }
      }

      projection_linear(plans.output(global), rows, layer, model::TensorRole::o_proj,
                        context, weight.o_proj, h2, stream, gewell::nvfp4::Phase::prefill);
      primitives::post_attention_residual_pre_feedforward_norm_prefill(
          h2, weight.post_attention_norm, h0, weight.pre_feedforward_norm, h1, rows, stream);
      project_mlp(plans, rows, layer, h1, gate, up, product, h0,
                  stream, gewell::nvfp4::Phase::prefill);
      const auto* next_norm = layer + 1 < model::kLayerCount ? layers_[layer + 1].input_norm
                         : produce_token ? weights_.pointer(model::kFinalNormPhysicalId) : nullptr;
      primitives::post_feedforward_residual_scalar_next_norm_prefill(
          h0, weight.post_feedforward_norm, h2, weight.layer_scalar, next_norm, h1, rows, stream);
    }

    if (produce_token) {
      const BFloat16* const final_row =
          h1 + static_cast<std::size_t>(rows - 1) * model::kHiddenSize;
      lm_head_.run(handle_.get(), final_row,
                   weights_.pointer(model::kLmHeadLogicalId), logits, stream);
      primitives::softcap_and_argmax(logits, capped_logits, argmax(),
                                     model::kVocabSize, 30.0F, stream);
    }
  }

  void decode_generated_token(std::uint32_t position) {
    std::vector<mtp_attention::BatchInput> fp8_inputs(
        local_compute_ == attention::Compute::fp8 || global_compute_ == attention::Compute::fp8 ? 1 : 0);
    BFloat16* const h0 = at<BFloat16>(scratch_layout_.kH0);
    BFloat16* const h1 = at<BFloat16>(scratch_layout_.kH1);
    BFloat16* const h2 = at<BFloat16>(scratch_layout_.kH2);
    BFloat16* const q_raw =
        at<BFloat16>(scratch_layout_.kQueryRaw);
    BFloat16* const q_norm =
        at<BFloat16>(scratch_layout_.kQueryNorm);
    BFloat16* const q_rope =
        at<BFloat16>(scratch_layout_.kQueryRope);
    BFloat16* const k_raw =
        at<BFloat16>(scratch_layout_.kKeyRaw);
    BFloat16* const k_norm =
        at<BFloat16>(scratch_layout_.kKeyNorm);
    BFloat16* const k_rope =
        at<BFloat16>(scratch_layout_.kKeyRope);
    BFloat16* const v_raw =
        at<BFloat16>(scratch_layout_.kValueRaw);
    BFloat16* const v_norm =
        at<BFloat16>(scratch_layout_.kValueNorm);
    BFloat16* const context =
        at<BFloat16>(scratch_layout_.kContext);
    BFloat16* const gate =
        at<BFloat16>(scratch_layout_.kGate);
    BFloat16* const up = at<BFloat16>(scratch_layout_.kUp);
    BFloat16* const product =
        at<BFloat16>(scratch_layout_.kProduct);
    BFloat16* const logits =
        at<BFloat16>(scratch_layout_.kLogits);
    BFloat16* const capped_logits =
        at<BFloat16>(scratch_layout_.kCappedLogits);
    BFloat16* const local_cos =
        at<BFloat16>(scratch_layout_.kLocalCos);
    BFloat16* const local_sin =
        at<BFloat16>(scratch_layout_.kLocalSin);
    BFloat16* const global_cos =
        at<BFloat16>(scratch_layout_.kGlobalCos);
    BFloat16* const global_sin =
        at<BFloat16>(scratch_layout_.kGlobalSin);
    const cudaStream_t stream = stream_.get();

    primitives::embedding_lookup_device_token(
        weights_.pointer(model::kEmbeddingPhysicalId), argmax(), h0, stream);
    primitives::generate_rope_factors_m1(local_cos, local_sin, global_cos,
                                         global_sin, position, stream);
    primitives::rms_norm_hidden_m1(h0, layers_[0].input_norm, h1, stream);
    for (std::uint32_t layer = 0; layer < model::kLayerCount; ++layer) {
      const bool global = model::is_global_layer(layer);
      const model::AttentionKind kind =
          global ? model::AttentionKind::global : model::AttentionKind::local;
      const std::uint32_t kv_heads =
          global ? model::kGlobalKvHeadCount : model::kLocalKvHeadCount;
      const LayerWeights& weight = layers_[layer];
      const LayerCacheView cache = caches_.layer(layer);
      const bool fp8_compute = (global ? global_compute_ : local_compute_) == attention::Compute::fp8;
      const BFloat16* const cosine = global ? global_cos : local_cos;
      const BFloat16* const sine = global ? global_sin : local_sin;
      project_qkv(decode_plans_, 1, layer, h1, q_raw, k_raw, v_raw,
                  q_norm, k_norm, v_norm, stream, gewell::nvfp4::Phase::decode);
      primitives::apply_rope_m1(q_norm, cosine, sine, q_rope,
                                model::kQueryHeadCount, kind, stream);
      primitives::apply_rope_m1(k_norm, cosine, sine, k_rope, kv_heads, kind,
                                stream);
      if (global) {
        if (cache.page_pool != nullptr) {
          const prefill::CompactGlobalPagedCache paged_cache{
              cache.page_pool,
              cache.page_offsets,
              cache.page_tokens,
              cache.page_count,
              cache.page_stride_elements,
              cache.layer_offset_elements,
              cache.format,
          };
          prefill::write_kv_cache_chunk_global_compact_paged(
              k_rope, v_norm, paged_cache, position, 1, stream);
          if (!fp8_compute) primitives::
              causal_gqa_attention_cached_m1_fused_global_compact_paged(
                  q_rope, paged_cache, weight.k_norm, position,
                  attention_scratch_.data(), context, stream);
        } else {
          primitives::write_kv_cache_m1_global_compact(
              k_rope, v_norm, cache.key, position, cache.capacity, stream, cache.format);
          if (!fp8_compute) primitives::causal_gqa_attention_cached_m1_fused_global_compact(
              q_rope, cache.key, weight.k_norm, position, cache.capacity,
              attention_scratch_.data(), context, stream, cache.format);
        }
      } else {
        primitives::write_kv_cache_m1(
            k_rope, v_norm, cache.key, cache.value, position,
            cache.capacity, kind, stream, cache.format);
        if (!fp8_compute) primitives::causal_gqa_attention_cached_m1_fused(
            q_rope, cache.key, cache.value, position, cache.capacity,
              attention_scratch_.data(), context, kind, stream, cache.format);
      }

      if (fp8_compute) {
        fp8_inputs[0] = {q_rope, nullptr, nullptr, cache, position + 1, 1, context};
        mtp_attention::run_fp8_batch(fp8_inputs, weight.k_norm, kind,
            attention_scratch_.data(), attention_scratch_.size(), stream, true);
      }

      const LinearPlan& o_plan =
          decode_plans_.output(global);
      projection_linear(o_plan, 1, layer, model::TensorRole::o_proj,
                        context, weight.o_proj, h2, stream, gewell::nvfp4::Phase::decode);
      primitives::post_attention_residual_pre_feedforward_norm_m1(
          h2, weight.post_attention_norm, h0, weight.pre_feedforward_norm, h1,
          stream);
      project_mlp(decode_plans_, 1, layer, h1, gate, up, product, h0,
                  stream, gewell::nvfp4::Phase::decode);
      const BFloat16* const next_norm_weight =
          layer + 1 < model::kLayerCount
              ? layers_[layer + 1].input_norm
              : weights_.pointer(model::kFinalNormPhysicalId);
      primitives::post_feedforward_residual_scalar_next_norm_m1(
          h0, weight.post_feedforward_norm, h2, weight.layer_scalar,
          next_norm_weight, h1, stream);
    }
    lm_head_.run(handle_.get(), h1,
                 weights_.pointer(model::kLmHeadLogicalId), logits, stream);
    primitives::softcap_and_argmax(logits, capped_logits, argmax(),
                                   model::kVocabSize, 30.0F, stream);
  }

  template <typename T>
  T* at(std::size_t byte_offset) const {
    return reinterpret_cast<T*>(static_cast<std::uint8_t*>(scratch_.data()) +
                                byte_offset);
  }
  std::uint32_t* argmax() const {
    return at<std::uint32_t>(scratch_layout_.kArgmax);
  }
  BFloat16* capped_logits() const {
    return at<BFloat16>(scratch_layout_.kCappedLogits);
  }

  const WeightArena& weights_;
  gewell::nvfp4::ActivationPolicy activation_policy_;
  std::size_t prompt_tokens_{};
  std::uint32_t new_token_count_{};
  std::uint32_t global_capacity_{};
  std::uint32_t prefill_chunk_tokens_{};
  std::uint32_t replay_chunk_rows_{};
  std::map<std::uint32_t, std::unique_ptr<LinearPlan>> replay_heads_;
  ExecutionCache* persistent_cache_{};
  kv_cache::ExecutionId persistent_execution_{};
  std::vector<CheckpointTrigger> checkpoint_triggers_;
  NonblockingCudaStream stream_;
  const RuntimeGenerationScratchLayout scratch_layout_;
  DeviceAllocation scratch_;
  DeviceAllocation attention_scratch_;
  DeviceAllocation prefill_attention_scratch_;
  RuntimeKvCaches caches_;
  DeviceAllocation outputs_;
  CudaEvent prefill_begin_;
  CudaEvent prefill_end_;
  CudaEvent decode_begin_;
  CudaEvent decode_end_;
  CublasHandle prefill_attention_handle_;
  attention::Compute local_compute_{}, global_compute_{};
  std::unique_ptr<prefill::Fp8Attention> fp8_attention_;
  LtHandle handle_;
  std::unique_ptr<gewell::nvfp4::ProjectionPlans> nvfp4_projections_;
  std::unique_ptr<gewell::fp8::ProjectionPlans> fp8_projections_;
  RuntimeChunkPlans primary_plans_;
  std::unique_ptr<RuntimeChunkPlans> tail_plans_;
  std::map<std::uint32_t, std::unique_ptr<RuntimeChunkPlans>> extra_plans_;
  RuntimeChunkPlans decode_plans_;
  LinearPlan lm_head_;
  SamplingSettings sampling_;
  std::unique_ptr<DeviceAllocation> sampling_scratch_;
  std::mt19937_64 rng_{std::random_device{}()};
  std::array<LayerWeights, model::kLayerCount> layers_{};
  std::uint32_t mtp_depth_{};
  std::unique_ptr<DeviceAllocation> mtp_staging_;
  std::unique_ptr<mtp_cycle::Cycle> mtp_;
  std::vector<float> mtp_uniforms_;
  std::uint32_t batch_capacity_{};
  std::unique_ptr<DeviceAllocation> batch_logits_;
  std::unique_ptr<DeviceAllocation> batch_tokens_;
  std::vector<std::uint32_t> batch_host_tokens_;
  std::map<std::uint32_t, std::unique_ptr<LinearPlan>> batch_heads_;
  std::unique_ptr<mtp_cycle::Batch> batch_mtp_;
  struct ConstraintSamplingRow {
    std::array<std::uint32_t, mtp_sampling::mask_words(model::kVocabSize)> mask;
    float uniform;
    mtp_sampling::Status status;
  };
  std::unique_ptr<DeviceAllocation> constraint_sampling_rows_, constraint_probabilities_;
  std::unique_ptr<PinnedHostAllocation> constraint_host_rows_;
  std::vector<std::uint8_t> constraint_sampling_pending_;
};


}  // namespace gewell::gemma4_31b::sm120
