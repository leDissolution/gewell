#pragma once

#include "gewell/runtime/state.h"
#include "gewell/runtime/image.h"
#include "gewell/runtime/cache_storage.h"
#include "gewell/logprobs.h"
#include "gewell/mtp_capture.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace gewell::runtime {
class PersistentCacheManager;
struct BatchLimits;

struct SamplingSettings {
  float temperature{};
  float top_p{1.0F};
  std::uint32_t top_k{};
};
struct BackendLimits {
  std::uint32_t vocab_size{}, context_tokens{}, max_batch_rows{}, max_mtp_depth{};
  std::uint32_t max_verifier_rows{}, max_prefill_chunk_tokens{}, max_prefill_batch_tokens{}, local_window_tokens{};
  std::size_t logit_row_bytes{};
  std::vector<std::uint32_t> stop_tokens;
  std::uint32_t max_image_tokens{};
  MtpCaptureGeometry mtp_capture;
};
struct BatchMemoryPlan {
  std::size_t staging_bytes{}, hidden_slots{}, hidden_staging_bytes{}, committed_kv_bytes{};
};
struct BatchDecodeInput {
  kv_cache::ExecutionId execution{};
  std::uint32_t position{}, token{};
};
struct BatchPrefillInput {
  kv_cache::ExecutionId execution{};
  const std::uint32_t* tokens{};
  std::uint32_t position{}, rows{};
  TerminalState hidden;
  std::shared_ptr<const ImageInput> image;
};
struct MtpDepthInput {
  kv_cache::ExecutionId execution{};
  std::uint32_t position{}, pending_token{}, output_begin{};
  TerminalState target_hidden;
  SamplingSettings sampling;
  std::uint32_t previous_accepted{}, previous_depth{};
  std::uint64_t prior_accepted{}, prior_proposed{};
  std::uint32_t min_depth{}, max_depth{};
};
struct BatchMtpInput {
  kv_cache::ExecutionId execution{};
  std::uint32_t pending_token{}, position{}, depth{};
  TerminalState target_hidden;
  float temperature{}, top_p{1.0F};
  std::uint32_t top_k{};
  bool return_probabilities{};
  MtpCaptureFeatures* capture{};
  MtpTargetProbes* capture_next{};
  std::vector<float> uniforms;
  std::function<void(const std::uint32_t*, std::uint32_t, std::uint32_t*)> constraint_mask;
};
struct VerificationResult {
  std::uint32_t accepted_drafts{}, output_count{}, rejected_index{};
};
struct MtpOutcome {
  VerificationResult verification;
  std::vector<std::uint32_t> tokens;
  // Nonempty only for a request-local sampling failure before KV commit.
  std::string error;
};
struct BatchMtpOutcome {
  std::vector<MtpOutcome> requests;
  float draft_gpu_milliseconds{}, verify_gpu_milliseconds{}, select_gpu_milliseconds{};
  std::uint32_t constraint_draft_downloads{}, constraint_mask_uploads{};
  std::size_t constraint_draft_bytes{}, constraint_mask_bytes{};
};
struct BatchMtpCommit {
  std::uint32_t row{};
  kv_cache::ExecutionId execution{};
  std::uint32_t position{}, count{};
  TerminalState hidden;
  bool captures{}, logprobs{};
  std::uint32_t top_logprobs{};
};

inline std::uint32_t generation_cache_capacity(std::size_t prompt_tokens,
    std::uint32_t new_tokens, std::uint32_t context_tokens) {
  if (!prompt_tokens || !new_tokens || prompt_tokens > context_tokens ||
      std::uint64_t(prompt_tokens) + new_tokens - 1 > context_tokens)
    throw std::invalid_argument("generation request: invalid prompt or fed-token horizon");
  return static_cast<std::uint32_t>(prompt_tokens + new_tokens - 1);
}

// Calls occur only at selected work boundaries. Layer loops, device buffers,
// physical state interpretation and dispatch stay inside the loaded backend.
class ExecutionBackend {
 public:
  virtual ~ExecutionBackend() = default;
  virtual const BackendLimits& limits() const = 0;
  bool is_stop_token(std::uint32_t token) const {
    const auto& stops = limits().stop_tokens;
    return std::find(stops.begin(), stops.end(), token) != stops.end();
  }
  virtual BatchMemoryPlan memory_plan() const = 0;
  virtual kv_cache::PoolConfig cache_config() const = 0;
  virtual CacheStorageFactory cache_storage_factory() const = 0;
  virtual void allocate_staging() = 0;
  virtual void initialize(PersistentCacheManager&, kv_cache::ExecutionId, const BatchLimits&) = 0;
  virtual void initialize_outputs(bool captures, bool logprobs) = 0;
  virtual CompletionContext context() const = 0;
  virtual void wait() const = 0;
  virtual void synchronize(std::string_view) const = 0;
  virtual bool ready(std::string_view) const = 0;
  virtual TerminalState acquire_hidden() = 0;
  virtual void release_hidden(TerminalState) = 0;
  virtual std::size_t occupied_hidden_bytes() const = 0;
  virtual void copy_terminal(TerminalState, TerminalState, std::string_view) = 0;
  virtual void save_terminal(std::uint32_t row, TerminalState) = 0;
  virtual void begin_step(std::string_view) = 0;
  virtual void end_step(std::string_view) = 0;
  virtual float elapsed(std::string_view) const = 0;
  // One bounded forward over independent text or whole-image segments. Image
  // features stay live through the shared forward. Poll between serial image
  // encodes and before their decoder dispatch; false skips a cancelled segment.
  // All-text forwards poll at the scheduler's completion boundary. The returned
  // flags identify segments whose decoder work was enqueued in full.
  virtual std::vector<bool> prefill_batch(const std::vector<BatchPrefillInput>&,
      const std::function<bool(std::size_t)>& continue_prefill) = 0;
  // The scheduler calls this only after queued prefill work has completed.
  virtual void release_image() = 0;
  virtual void prefix_head_step(kv_cache::ExecutionId, TerminalState) = 0;
  virtual void decode_batch(const std::vector<BatchDecodeInput>&) = 0;
  virtual void sample_batch_row(std::uint32_t, const SamplingSettings&, std::mt19937_64&,
                                const std::uint32_t*) = 0;
  virtual void summarize_batch_row(std::uint32_t, const SamplingSettings&, std::uint32_t,
                                   const std::uint32_t*) = 0;
  virtual void check_constraint_sampling() = 0;
  virtual BatchMtpOutcome run_batch_mtp(const std::vector<BatchMtpInput>&) = 0;
  virtual std::vector<std::uint32_t> predict_mtp_depths(const std::vector<MtpDepthInput>&) = 0;
  virtual void commit_batch_mtp(const std::vector<BatchMtpCommit>&) = 0;
  virtual void download_ids(std::size_t) = 0;
  virtual void download_logits(std::size_t) = 0;
  virtual void download_logprobs(std::size_t) = 0;
  virtual std::uint32_t output_id(std::size_t) const = 0;
  // Optional evaluation capture. Normal serving requests only compact scores.
  virtual const std::uint8_t* host_logits() const = 0;
  virtual const TokenLogprobs* host_logprobs() const = 0;
  virtual std::size_t scratch_bytes() const = 0;
  virtual std::size_t output_bytes() const = 0;
  virtual std::size_t host_scratch_bytes() const = 0;
  virtual std::size_t sampling_scratch_bytes() const = 0;
};
}  // namespace gewell::runtime
