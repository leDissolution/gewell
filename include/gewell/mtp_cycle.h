#pragma once

#include "gewell/mtp_sampling.h"
#include "gewell/mtp_target.h"
#include "gewell/mtp_assistant.h"
#include "gewell/mtp_capture.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace gewell::mtp_cycle {

// Called once after the full unconstrained draft block is available on the
// host, with target verification already enqueued. Fill depth+1 packed mask
// rows (mtp_sampling::mask_words(V) words each), beginning at the committed
// grammar state, which already includes pending. Do not mutate that state.
// Stop walking after an invalid draft or terminal token and leave subsequent
// rows all ones. The cycle also clears restrictions after a disallowed draft.
using ConstraintMask = std::function<void(const std::uint32_t* draft_ids,
                                          std::uint32_t depth,
                                          std::uint32_t* packed_masks)>;

struct Outcome {
  mtp_sampling::Result verification{};
  std::vector<std::uint32_t> tokens;
  float draft_gpu_milliseconds{};
  float verify_gpu_milliseconds{};
  float select_gpu_milliseconds{};
  std::uint32_t constraint_draft_downloads{}, constraint_mask_uploads{};
  std::size_t constraint_draft_bytes{}, constraint_mask_bytes{};
  // Batch returns request-local sampling failures without committing any KV.
  // Failed outcomes have no tokens; CUDA/host execution failures still throw.
  mtp_sampling::Status status{mtp_sampling::Status::success};
};

// One frozen-prefix assistant proposal sequence followed by target verification
// and probability-ratio selection. The caller owns KV staging and performs
// persistent-cache allocation/accounting before calling commit(). This class
// neither publishes tokens nor modifies committed KV during run().
class Cycle final {
 public:
  Cycle(cublasLtHandle_t handle, const mtp_target::Weights& weights,
        std::uint32_t context_capacity, std::uint32_t max_depth,
        void* staging, std::size_t staging_size,
        const nvfp4::Weights* native_weights = nullptr,
        nvfp4::ActivationPolicy activation_policy = nvfp4::ActivationPolicy::always,
        const fp8::Weights* fp8_weights = nullptr,
        attention::Compute local_compute = attention::Compute::bf16,
        attention::Compute global_compute = attention::Compute::bf16);
  ~Cycle();
  Cycle(const Cycle&) = delete;
  Cycle& operator=(const Cycle&) = delete;

  // Includes assistant/verifier scratch and all owned device buffers, excluding
  // caller-owned KV staging. Actual depths are bounded by max_depth.
  [[nodiscard]] std::size_t scratch_bytes() const;
  [[nodiscard]] std::size_t host_scratch_bytes() const;
  void prepare(std::uint32_t depth);

  // uniforms = [N draft draws, N acceptance draws, one correction/bonus draw].
  // pending is a device token at position L, already selected but absent in KV.
  // target_hidden is the final-normalized target row at L-1. Every assistant
  // step uses the same committed cache prefix and RoPE position L.
  // No full probability/logit rows move to the host. Constraints add one
  // bounded draft-block download/wait and one packed-mask upload per run.
  // run synchronizes to check status and return bounded token IDs. Truncation is
  // the caller's responsibility; commit exactly the number actually emitted.
  Outcome run(const std::uint32_t* pending,
              const mtp_target::BFloat16* target_hidden, std::uint32_t L,
              std::uint32_t depth, const mtp_target::Caches& caches,
              float temperature, float top_p, std::uint32_t top_k,
              const std::vector<float>& uniforms, cudaStream_t stream,
              const ConstraintMask& constraint_mask = {});

  void commit(const mtp_target::Caches& caches, std::uint32_t L,
              std::uint32_t count, cudaStream_t stream);

  // Latest verifier rows remain valid until the next run. Row i contains the
  // target post-softcap BF16 logits predicting output i. Hidden rows correspond
  // to verifier inputs [pending,draft_1,...]. IDs contain the selected output batch.
  [[nodiscard]] const mtp_target::BFloat16* logits() const;
  [[nodiscard]] const mtp_target::BFloat16* hidden() const;
  [[nodiscard]] const std::uint32_t* output_ids() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// The shared sampling cycle owns no target weights. These two model adapters
// supply verification and accepted-KV commit while preserving one acceptance,
// constraint, logprob and capture implementation.
using BatchCaches = std::vector<kv_cache::DeviceView>;
struct TargetInput {
  std::uint32_t base_position{}, rows{};
  BatchCaches caches;
  void* staging{};
  std::size_t staging_size{};
  std::uint32_t staging_capacity_rows{};
  std::vector<mtp_target::LayerCapture> captures;
};
struct TargetCommit {
  const BatchCaches* caches{};
  std::uint32_t base_position{}, source_rows{}, capacity_rows{};
  const void* staging{};
  std::size_t staging_size{};
  std::uint32_t commit_rows{};
};
class BatchTarget {
 public:
  virtual ~BatchTarget() = default;
  virtual mtp_assistant::Model model() const = 0;
  virtual mtp_assistant::Weights assistant_weights() const = 0;
  virtual std::size_t staging_bytes(std::uint32_t rows) const = 0;
  virtual std::size_t scratch_bytes() const = 0;
  virtual void prepare(std::uint32_t rows) = 0;
  virtual void run_batch(const std::uint32_t* tokens,
                        const std::vector<TargetInput>& inputs, cudaStream_t stream) = 0;
  virtual void commit_batch(const std::vector<TargetCommit>& inputs, cudaStream_t stream) = 0;
  virtual const mtp_target::BFloat16* logits() const = 0;
  virtual const mtp_target::BFloat16* hidden() const = 0;
};
std::unique_ptr<BatchTarget> make_31b_batch_target(
    cublasLtHandle_t handle, const mtp_target::Weights& weights,
    std::uint32_t capacity_rows, std::uint32_t context_capacity,
    const nvfp4::Weights* native_weights, nvfp4::ActivationPolicy activation_policy,
    const fp8::Weights* fp8_weights, attention::Compute local_compute,
    attention::Compute global_compute);

struct DeviceProbes {
  std::vector<std::uint32_t> layers;
  mtp_target::BFloat16* output{};  // Selected terminal row, [layers, hidden width].
};
struct BatchInput {
  std::uint32_t pending_token{};
  const mtp_target::BFloat16* target_hidden{};
  std::uint32_t position{}, depth{};
  BatchCaches caches;
  float temperature{}, top_p{1.0F};
  std::uint32_t top_k{};
  // Greedy verification itself needs only winner IDs. Keep this true for
  // callers that request logprob summaries; the runtime disables it for
  // requests that did not ask for logprobs.
  bool return_probabilities{true};
  std::vector<float> uniforms;
  ConstraintMask constraint_mask;
  MtpCaptureFeatures* capture{};
  MtpTargetProbes* capture_next{};
  DeviceProbes depth_probes;
};

struct BatchOutcome {
  std::vector<Outcome> requests;
  float draft_gpu_milliseconds{}, verify_gpu_milliseconds{}, select_gpu_milliseconds{};
  std::uint32_t constraint_draft_downloads{}, constraint_mask_uploads{};
  std::size_t constraint_draft_bytes{}, constraint_mask_bytes{};
};

struct BatchCommitInput {
  std::uint32_t request{};
  const BatchCaches* caches{};
  std::uint32_t count{};
};

// A shared assistant workspace drafts each independent frozen prefix, then one
// target traversal verifies all request rows together. Staging has one private
// slice per active request and is owned/budgeted by the offline scheduler.
class Batch final {
 public:
  Batch(cublasLtHandle_t handle, std::unique_ptr<BatchTarget> target,
        std::uint32_t context_capacity, std::uint32_t capacity,
        std::uint32_t max_depth, void* staging, std::size_t staging_size,
        attention::Compute local_compute = attention::Compute::bf16,
        attention::Compute global_compute = attention::Compute::bf16);
  ~Batch();
  Batch(const Batch&) = delete;
  Batch& operator=(const Batch&) = delete;
  [[nodiscard]] std::size_t scratch_bytes() const;
  [[nodiscard]] std::size_t host_scratch_bytes() const;
  BatchOutcome run(const std::vector<BatchInput>& inputs, cudaStream_t stream);
  void commit_batch(const std::vector<BatchCommitInput>& inputs,
                    cudaStream_t stream);
  // Latest per-request rows remain valid until the next run and use the same
  // output alignment and target-not-residual contract as Cycle. Greedy
  // probability rows are materialized only when BatchInput requests them.
  [[nodiscard]] const mtp_target::BFloat16* logits(std::uint32_t request) const;
  void summarize_logprobs(std::uint32_t request, std::uint32_t row_count,
                         std::uint32_t top_logprobs, TokenLogprobs* output,
                         cudaStream_t stream) const;
  [[nodiscard]] const mtp_target::BFloat16* hidden(std::uint32_t request) const;
  [[nodiscard]] const std::uint32_t* output_ids(std::uint32_t request) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gewell::mtp_cycle
