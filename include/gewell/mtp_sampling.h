#pragma once

#include "gewell/logprobs.h"
#include "gewell/mtp_capture.h"

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gewell::mtp_sampling {

inline constexpr std::uint32_t kMaxDraftTokens = 1'279;

// Packed token masks use little bit order within each uint32 word. Set bits
// permit tokens; padding bits beyond vocabulary_size are ignored.
inline constexpr std::size_t mask_words(std::uint32_t vocabulary_size) {
  return (static_cast<std::size_t>(vocabulary_size) + 31) / 32;
}

enum class Status : std::uint32_t {
  success = 0,
  invalid_logits,
  invalid_distribution,
  invalid_uniform,
  invalid_draft_token,
  zero_draft_probability,
  zero_residual_mass,
};

// One cycle returns accepted draft IDs followed by one correction or bonus ID.
// rejected_index is zero-based and equals draft_count when all drafts survive.
// The runner applies EOS/output truncation and commits exactly output_count
// verifier input rows after any such truncation.
struct Result {
  std::uint32_t accepted_drafts;
  std::uint32_t output_count;
  std::uint32_t rejected_index;
};

const char* status_message(Status status);

// All pointer arguments below are device pointers. Scratch is reusable across
// operations on one stream, starts at a 256-byte boundary, and must not overlap
// any input or output. Calls allocate no device storage and never synchronize.
// Host contract/CUDA launch errors throw; device data errors set the first
// non-success status. Clear once before a sequence of calls and inspect status
// after the caller's normal synchronization. Outputs are invalid on failure.
std::size_t scratch_bytes(std::uint32_t vocabulary_size);
void clear_status(Status* status, cudaStream_t stream = nullptr);

// Build one canonical token-ID-ordered FP32 probability row from the model's
// BF16 post-model logits (target softcap is applied by the caller). Temperature,
// top-k then top-p, stable lower-token-ID tie order, and top-p relative to the
// retained top-k mass match the ordinary sampler. Temperature zero, top-p zero,
// or top-k one produce a one-hot greedy row. top-k zero retains the vocabulary.
// Input logits must be finite; filters express zero support in output_probs.
// Optional allowed_tokens is a device packed mask applied before top-k/top-p
// and normalization. An empty allowed set fails with invalid_distribution.
void build_distribution(const __nv_bfloat16* logits,
                        std::uint32_t vocabulary_size, float temperature,
                        float top_p, std::uint32_t top_k, float* output_probs,
                        void* scratch, std::size_t scratch_size, Status* status,
                        cudaStream_t stream = nullptr,
                        const std::uint32_t* allowed_tokens = nullptr);

// Bounded top-k rows store only retained candidates, in increasing token-ID
// order. Entries removed by top-p have zero probability. The backing arena may
// reserve space for an unbounded request, but kernels touch only row_size entries.
inline constexpr std::uint32_t kMaxCompactTopK = 256;
struct TokenProbability {
  std::uint32_t token;
  float probability;
};

// Small training diagnostics from the filtered distributions. A nonnull
// greedy_target_ids selects the ID-only greedy path (p is 0/1, q is 1).
// Otherwise compact_size == 0 means dense FP32 rows, >0 TokenProbability rows.
// Does not change distributions, status, RNG, or verifier results.
void capture_scores(const void* target, const void* draft,
    const std::uint32_t* draft_ids, std::uint32_t depth, std::uint32_t vocabulary_size,
    std::uint32_t compact_size, const std::uint32_t* greedy_target_ids,
    const Status* status, MtpCaptureScores* output, cudaStream_t stream = nullptr);

// Select the last committed verifier-input row from layer-major probe rows.
// The selected row precedes the next pending token, so its label is the NEXT
// MTP round. All pointers are device pointers; failed results produce zeros.
void gather_capture_rows(const std::uint16_t* probes, std::uint32_t layers,
    std::uint32_t rows, std::uint32_t width, const Result* result,
    const Status* status, std::uint16_t* output, cudaStream_t stream = nullptr);

// Returns the compact row size, or zero for the dense/greedy paths.
std::uint32_t compact_row_size(std::uint32_t vocabulary_size, float temperature,
                               float top_p, std::uint32_t top_k);
void build_compact_distribution(
    const __nv_bfloat16* logits, std::uint32_t vocabulary_size,
    float temperature, float top_p, std::uint32_t row_size,
    TokenProbability* output, void* scratch, std::size_t scratch_size,
    Status* status, cudaStream_t stream = nullptr,
    const std::uint32_t* allowed_tokens = nullptr);

struct CompactDistributionInput {
  const __nv_bfloat16* logits{};
  float temperature{}, top_p{};
  std::uint32_t row_size{};
  TokenProbability* output{};
  void* scratch{};
  std::size_t scratch_size{};
  Status* status{};
  const std::uint32_t* allowed_tokens{};
};

// Independent rows, with the same arithmetic and tie ordering as the single
// row operation. Each entry needs disjoint scratch and output storage. Rows
// from one request that reuse scratch must be submitted in separate calls.
// Bounded launch parameters avoid allocations/uploads during graph capture.
void build_compact_distributions(
    const std::vector<CompactDistributionInput>& inputs,
    std::uint32_t vocabulary_size, cudaStream_t stream = nullptr);

void sample_compact_distribution(
    const TokenProbability* probs, std::uint32_t row_size,
    std::uint32_t vocabulary_size, const float* uniform,
    std::uint32_t* output_token, Status* status, cudaStream_t stream = nullptr);
void verify_compact_sequence(
    const TokenProbability* target, const TokenProbability* draft,
    const std::uint32_t* draft_ids, std::uint32_t draft_count,
    std::uint32_t row_size, std::uint32_t vocabulary_size,
    const float* accept_uniforms, const float* final_uniform,
    std::uint32_t* output_ids, Result* result, Status* status,
    cudaStream_t stream = nullptr);
void summarize_compact_logprobs(
    const TokenProbability* probs, const std::uint32_t* selected_ids,
    std::uint32_t row_count, std::uint32_t row_size,
    std::uint32_t top_logprobs, TokenLogprobs* output,
    cudaStream_t stream = nullptr);

struct GreedyDistributionInput {
  const __nv_bfloat16* logits{};
  float* output_probs{};
  const std::uint32_t* allowed_tokens{};
  Status* status{};
  std::uint32_t* best_scratch{};
  std::uint32_t* selected_id{};
};

// Batch the exact one-hot distribution used by temperature zero, top-p zero,
// or top-k one. Each row has independent pointers and optional token mask and
// probability output. best_scratch and selected_id are distinct one-word
// device regions. Invalid rows retain the ordinary sticky status behavior and
// publish selected ID 0.
void build_greedy_distributions(
    const std::vector<GreedyDistributionInput>& inputs,
    std::uint32_t vocabulary_size, cudaStream_t stream = nullptr);

struct GreedyVerificationInput {
  const std::uint32_t* target_ids{};
  const std::uint32_t* draft_ids{};
  const float* accept_uniforms{};
  const float* final_uniform{};
  std::uint32_t draft_count{};
  std::uint32_t* output_ids{};
  Result* result{};
  Status* status{};
};

// Verify independent sequences whose target and proposal distributions were
// built by build_greedy_distributions. Exact one-hot rows accept precisely
// while target_ids[i] == draft_ids[i]; the first differing target ID is the
// correction, and the final target ID is the all-accepted bonus. Uniforms are
// still validated to retain verify_sequence's input contract.
void verify_greedy_sequences(
    const std::vector<GreedyVerificationInput>& inputs,
    std::uint32_t vocabulary_size, cudaStream_t stream = nullptr);

// Sample a normalized, nonnegative FP32 row using a device uniform in [0,1).
// FP32 inclusive-scan accumulation and strict CDF comparison define the draw.
// Rounded CDF steps at zero-weight entries advance to positive support (or
// the last positive entry for a trailing plateau); zero mass is never sampled.
void sample_distribution(const float* probs, std::uint32_t vocabulary_size,
                         const float* uniform, std::uint32_t* output_token,
                         void* scratch, std::size_t scratch_size, Status* status,
                         cudaStream_t stream = nullptr);

// Summarize row-major normalized FP32 probability rows without transferring
// the full rows to the host. selected_ids has one in-vocabulary ID with positive
// probability per row. Alternatives contain only positive-probability tokens,
// ordered by descending probability and then lower token ID. Log probabilities
// are natural logarithms. top_logprobs may be in 0..kMaxTopLogprobs.
void summarize_logprobs(const float* probs,
                        const std::uint32_t* selected_ids,
                        std::uint32_t row_count,
                        std::uint32_t vocabulary_size,
                        std::uint32_t top_logprobs, TokenLogprobs* output,
                        cudaStream_t stream = nullptr);

// target_probs is [draft_count+1,V], draft_probs is [draft_count,V]. Row i of
// target_probs predicts draft_ids[i]; its final row supplies the bonus. The
// proposal must have been sampled from its saved draft_probs row. Verification
// uses p>=q or uniform*q<p, then samples max(p-q,0) on the first rejection.
// With draft_count=0, draft arrays/accept_uniforms may be null and the sole
// target row is sampled directly. One-hot rows implement greedy verification.
// output_ids has room for draft_count+1 IDs. final_uniform supplies the one
// correction/bonus draw. EOS belongs to the caller, not the probability rule.
void verify_sequence(const float* target_probs, const float* draft_probs,
                     const std::uint32_t* draft_ids,
                     std::uint32_t draft_count,
                     std::uint32_t vocabulary_size,
                     const float* accept_uniforms, const float* final_uniform,
                     std::uint32_t* output_ids, Result* result, void* scratch,
                     std::size_t scratch_size, Status* status,
                     cudaStream_t stream = nullptr);

}  // namespace gewell::mtp_sampling
