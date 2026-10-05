#pragma once

#include "gewell/mtp_capture.h"

#include "gewell/attention_compute.h"

#include "gewell/kv_cache.h"

#include "gewell/nvfp4_policy.h"
#include "gewell/models/gemma4/31b/sm120/execution_config.h"
#include "gewell/models/gemma4/image_processor.h"

#include <cstdint>
#include <string>
#include <optional>

namespace gewell::http { struct Settings; }

namespace gewell::app {

struct RuntimeSettings {
  std::string assistant_path;
  std::string vision_path;
  std::uint32_t image_max_soft_tokens{gemma4::kDefaultImageMaxSoftTokens};
  attention::Compute local_attention_compute{attention::Compute::bf16};
  attention::Compute global_attention_compute{attention::Compute::bf16};
  kv_cache::Format local_kv_format{kv_cache::Format::bf16};
  kv_cache::Format global_kv_format{kv_cache::Format::bf16};
  nvfp4::ActivationPolicy nvfp4_activation_policy{nvfp4::ActivationPolicy::always};
  // Maximum proposal depth; zero disables MTP. A zero decode width uses fixed depth.
  std::uint32_t mtp_depth{};
  std::uint32_t mtp_min_depth{}, decode_width{};
  std::string mtp_stats_path;
  std::uint32_t mtp_stats_window{64};
  MtpCaptureSettings mtp_capture;
  std::uint32_t prefill_chunk_tokens{gemma4_31b::sm120::kDefaultPrefillChunkTokens};
  std::uint32_t prefill_batch_tokens{gemma4_31b::sm120::kDefaultPrefillBatchTokens};
  std::uint32_t prefill_budget_tokens{};
  // All cache sizes are MiB. The GPU budget is deliberately required for the
  // persistent serving path; the remaining values have the documented
  // engineering defaults.
  std::uint64_t kv_cache_gpu_mib{};
  std::uint64_t kv_cache_cpu_mib{};
  std::uint64_t kv_checkpoint_interval_tokens{8'192};
  std::uint64_t kv_cache_index_mib{kv_cache::kDefaultIndexBytes / kv_cache::kMib};
};

struct GenerationSettings {
  std::string assistant_path;
  std::string vision_path;
  attention::Compute local_attention_compute{attention::Compute::bf16};
  attention::Compute global_attention_compute{attention::Compute::bf16};
  kv_cache::Format local_kv_format{kv_cache::Format::bf16};
  kv_cache::Format global_kv_format{kv_cache::Format::bf16};
  nvfp4::ActivationPolicy nvfp4_activation_policy{nvfp4::ActivationPolicy::always};
  std::uint32_t mtp_depth{};
  // Batch commands only: minimum proposal depth and target verification rows.
  std::uint32_t mtp_min_depth{}, decode_width{};
  std::string mtp_stats_path;
  std::uint32_t mtp_stats_window{64};
  MtpCaptureSettings mtp_capture;
  std::uint32_t prefill_chunk_tokens{gemma4_31b::sm120::kDefaultPrefillChunkTokens};
  std::uint32_t prefill_batch_tokens{gemma4_31b::sm120::kDefaultPrefillBatchTokens};
  std::uint32_t prefill_budget_tokens{};
  float temperature{};
  float top_p{1.0F};
  std::uint32_t top_k{};
  std::optional<std::uint64_t> seed;
};

// Runs one request-sized batch-one generation with compact BF16 KV. The prompt is a nonempty
// little-endian uint32 token stream and the output is created as the same raw
// format. Exactly new_token_count sampled tokens are produced; EOS has no
// special meaning. When logits_output_path is nonempty, it receives token-major
// little-endian BF16 post-softcap logits with one full vocabulary row per
// output token. The prompt plus every generated token that must be fed back
// into the model may occupy at most Gemma 4's 262,144-token context.
int run_generate(const std::string& artifact_path,
                 const std::string& prompt_path,
                 std::uint32_t new_token_count,
                 const std::string& output_path,
                 const std::string& logits_output_path = {},
                 const std::string& qdq_mask_path = {},
                 GenerationSettings settings = {});

// Offline compact-global batching with completed and in-flight prefix reuse.
// The nine-field TSV carries request IDs, prompts, limits, sampling and capture.
// Optional events inject deterministic arrivals, cancellations and failures.
// Output directory must be new. KV budget includes terminal state and MTP staging.
int run_generate_batch(const std::string& artifact_path,
                       const std::string& requests_path,
                       std::uint32_t max_batch,
                       std::uint64_t kv_cache_gpu_mib,
                       const std::string& output_directory,
                       const std::string& qdq_mask_path = {},
                       std::uint32_t mtp_depth = 0,
                       const std::string& events_path = {},
                       std::uint32_t prefill_chunk_tokens = gemma4_31b::sm120::kDefaultPrefillChunkTokens,
                       std::uint32_t prefill_batch_tokens = gemma4_31b::sm120::kDefaultPrefillBatchTokens,
                       nvfp4::ActivationPolicy nvfp4_activation_policy = nvfp4::ActivationPolicy::always,
                       kv_cache::Format local_kv_format = kv_cache::Format::bf16,
                       kv_cache::Format global_kv_format = kv_cache::Format::bf16,
                       attention::Compute local_attention_compute = attention::Compute::bf16,
                       attention::Compute global_attention_compute = attention::Compute::bf16,
                       const std::string& assistant_path = {},
                       const std::string& vision_path = {},
                       std::uint32_t prefill_budget_tokens = 0,
                       std::uint32_t mtp_min_depth = 0,
                       std::uint32_t decode_width = 0,
                       const std::string& mtp_stats_path = {},
                       std::uint32_t mtp_stats_window = 64,
                       const MtpCaptureSettings& mtp_capture = {});

// Native HTTP uses the same scheduler directly, with one GPU owner.
int run_http_server(const std::string& model_directory,
                    std::uint32_t max_batch,
                    RuntimeSettings settings,
                    const http::Settings& http_settings,
                    const std::string& qdq_mask_path = {});

// Teacher-forced text replay through causal prefill. Four TSV fields: ID,
// prompt.u32, continuation.u32, saved generating-model logits.bf16.
// Streams compact per-position comparison metrics, including the final EOS
// prediction, while feeding prompt + continuation[:-1]. No generation/MTP.
int run_replay_rollout(const std::string& artifact_path,
                       const std::string& requests_path,
                       std::uint32_t chunk_rows, std::uint32_t head_rows,
                       const std::string& output_directory,
                       const std::string& qdq_mask_path = {});

// Runs one image-conditioned Gemma 4 caption request. PROMPT.u32 is one
// unpadded sequence and must contain exactly one contiguous run of image
// placeholder tokens immediately inside the begin/end-image markers. Its
// length determines the prepared image's soft-token count. Prefill keeps the
// complete image in one chunk; surrounding text may span further chunks.
// For multiple images use run-jobs. Generation stops after emitting token 1,
// 106, or 50, or after a positive max_new_token_count decisions. The matched
// stop token is included.
// The fed-token horizon is <=262144. Output paths are distinct otherwise-new
// files; output is little-endian u32 and optional logits are token-major BF16.
int run_caption(const std::string& artifact_path,
                const std::string& prompt_path,
                const std::string& pixel_values_path,
                const std::string& position_ids_path,
                std::uint32_t max_new_token_count,
                const std::string& output_path,
                const std::string& logits_output_path = {},
                GenerationSettings settings = {});

std::uint32_t effective_mtp_depth(std::uint32_t depth, const std::string& assistant_path,
                                  std::uint32_t min_depth = 0);

}  // namespace gewell::app
