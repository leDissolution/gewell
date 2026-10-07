#pragma once

#include "resources.cuh"
#include "gewell/runtime/backend.h"
#include "gewell/runtime/cache.h"
#include "gewell/models/gemma4/text_contract.h"
#include "gewell/models/gemma4/31b/sm120/execution_config.h"
#include "gewell/kv_cache.h"

namespace gewell::gemma4_31b::sm120 {

using LayerCacheView = kv_cache::DeviceView;
constexpr auto kLocalCacheCapacity = model::kLocalWindowSize;
constexpr std::size_t kLocalCacheBytesPerKind = std::size_t(model::kLocalLayerCount) * model::kLocalKvHeadCount * kLocalCacheCapacity * model::kLocalHeadSize * sizeof(BFloat16);

constexpr std::uint32_t kRuntimeChunkTokens = kDefaultPrefillChunkTokens;
constexpr std::uint32_t kMultimodalChunkTokens = 1280;
constexpr std::uint32_t kMaxBatchRows = 1280;
static_assert(kMaxPrefillChunkTokens == prefill::kMaxChunkTokenCount);
static_assert(kMultimodalChunkTokens == 1'280);
constexpr std::size_t kGenerationLogitRowBytes =
    static_cast<std::size_t>(model::kVocabSize) * sizeof(BFloat16);
inline const auto& kGenerationStopTokenIds = gemma4::text_contract_31b().stop_tokens;
using runtime::SamplingSettings;

inline bool is_generation_stop_token(std::uint32_t token) {
  return std::find(kGenerationStopTokenIds.begin(),
                   kGenerationStopTokenIds.end(), token) !=
         kGenerationStopTokenIds.end();
}

inline std::uint32_t generation_cache_capacity(std::size_t prompt_tokens,
                                        std::uint32_t new_tokens) {
  if (prompt_tokens == 0) {
    fail("generation request", "prompt is empty");
  }
  if (new_tokens == 0) {
    fail("generation request", "new token count is zero");
  }
  const std::uint64_t fed_tokens =
      static_cast<std::uint64_t>(prompt_tokens) + new_tokens - 1;
  if (prompt_tokens > primitives::kMaxContextTokenCount ||
      fed_tokens > primitives::kMaxContextTokenCount) {
    fail("generation request",
         "prompt plus generated tokens fed back exceeds 262144");
  }
  return static_cast<std::uint32_t>(fed_tokens);
}

inline std::uint32_t checked_prefill_chunk_tokens(std::uint32_t tokens) {
  if (!tokens || tokens > kMaxPrefillChunkTokens)
    fail("prefill chunk", "token cap must be in 1..4096");
  return tokens;
}

inline std::uint32_t checked_prefill_batch_tokens(std::uint32_t tokens, std::uint32_t chunk_tokens) {
  if (!tokens || tokens > kMaxPrefillBatchTokens)
    fail("prefill batch", "token cap must be in 1..4096");
  if (tokens < chunk_tokens)
    fail("prefill batch", "batch token cap must be at least the per-prompt chunk cap");
  return tokens;
}

inline std::uint32_t generation_prefill_chunk_tokens(std::size_t prompt_tokens,
                                              std::uint32_t chunk_cap = kRuntimeChunkTokens) {
  checked_prefill_chunk_tokens(chunk_cap);
  return static_cast<std::uint32_t>(
      std::min<std::size_t>(prompt_tokens, chunk_cap));
}

// Text ends before the next image; an image always occupies one complete
// chunk. This preserves its bidirectional mask without combining image blocks.
inline std::uint32_t multimodal_prefill_chunk_rows(
    std::uint32_t base, std::uint32_t prompt_tokens, std::uint32_t text_chunk_cap,
    std::uint32_t image_begin, std::uint32_t image_end) {
  if (base >= prompt_tokens)
    fail("multimodal prefill", "chunk begins outside the prompt");
  checked_prefill_chunk_tokens(text_chunk_cap);
  if (!image_begin && !image_end)
    return std::min(text_chunk_cap, prompt_tokens - base);
  if (image_begin < base || image_begin >= image_end || image_end > prompt_tokens ||
      image_end - image_begin > model::kVisionMaxSoftTokenCount)
    fail("multimodal prefill", "invalid next image range");
  return base == image_begin ? image_end - image_begin
                             : std::min(text_chunk_cap, image_begin - base);
}

inline std::size_t runtime_global_compact_cache_bytes(std::uint32_t capacity) {
  if (capacity == 0 || capacity > primitives::kMaxContextTokenCount) {
    fail("generation KV cache", "global capacity is outside 1..262144");
  }
  return static_cast<std::size_t>(model::kGlobalLayerCount) *
         model::kGlobalKvHeadCount * capacity *
         primitives::kGlobalCompactKvSize * sizeof(BFloat16);
}

constexpr std::size_t compact_global_layer_elements(
    std::uint32_t capacity) {
  return static_cast<std::size_t>(model::kGlobalKvHeadCount) * capacity *
         primitives::kGlobalCompactKvSize;
}


inline constexpr std::string_view kGenerationKvLayoutId =
    "compact-global-k128-v512-v1";


class GenerationDecisionSink {
 public:
  virtual ~GenerationDecisionSink() = default;
  virtual bool honors_stop_tokens() const { return false; }

  // Returns false after writing a decision when generation should stop.
  virtual bool write_device_decision(const std::uint32_t* token,
                                     const BFloat16* logits,
                                     cudaStream_t stream) = 0;
};

using runtime::CheckpointSource;
using runtime::CheckpointTrigger;

class GenerationCheckpointSink {
 public:
  virtual ~GenerationCheckpointSink() = default;

  virtual void capture_checkpoint(CheckpointSource source,
                                  const std::vector<std::uint32_t>& tokens,
                                  const BFloat16* terminal_hidden,
                                  cudaStream_t stream) = 0;
};

class GenerationLogitsSink {
 public:
  virtual ~GenerationLogitsSink() = default;
  virtual void write_device_row(const BFloat16* logits, cudaStream_t stream) = 0;
};

struct GenerationResult {
  std::uint64_t mtp_cycles{}, mtp_proposed{}, mtp_accepted{}, mtp_rejected{};
  std::vector<std::uint64_t> mtp_accepted_histogram;
  double mtp_draft_gpu_milliseconds{};
  double mtp_verify_gpu_milliseconds{};
  double mtp_select_gpu_milliseconds{};
  std::vector<std::uint32_t> outputs;
  float prefill_gpu_milliseconds{};
  double prefill_wall_seconds{};
  float decode_gpu_milliseconds{};
  double decode_wall_seconds{};
};

struct VisionPromptSlice {
  const BFloat16* soft_features{};
  std::uint32_t begin{};
  std::uint32_t end{};
};

using runtime::BatchDecodeInput;


}  // namespace gewell::gemma4_31b::sm120
