#pragma once

#include "gewell/kv_cache.h"
#include "gewell/json_constraint.h"
#include "gewell/logprobs.h"
#include "gewell/tokenizer.h"
#include "gewell/runtime/image.h"
#include "gewell/models/embeddinggemma2/input.h"
#include "gewell/models/embeddinggemma2/video.h"
#include "json.hpp"

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace gewell::http {

using ClientId = std::uint64_t;
enum class Operation { health, models, model, metrics, cache_index, generate, prefill, finish, stats, embed };
enum class Priority { low, normal, high };

struct CacheControls {
  bool reuse_only = false;
  std::string prompt_id;
  Priority priority = Priority::normal;
  bool finished = false;
};

// Selected by the loaded model. Preparation runs on the HTTP worker; only
// the resident GPU owner interprets the resulting patch tensors.
struct ImageSupport {
  std::function<std::shared_ptr<runtime::ImageInput>(std::string_view, std::uint32_t)> prepare;
  // Embedding videos decode into the same image storage at a fixed frame budget.
  embeddinggemma2::PrepareVideoFrame prepare_video_frame;
  embeddinggemma2::PreparationPoll poll_preparation;
  std::uint32_t begin_token{}, image_token{}, end_token{}, max_image_tokens{};
  std::uint32_t default_max_soft_tokens{};
  // Supported token budgets and their per-image host tensor reservations.
  std::map<std::uint32_t, std::size_t> prepared_bytes;
};

// Runs task(i) for every i < count on the caller plus helper threads shared by
// all requests (a process-wide cap bounds decode/resize scratch). Once a task
// returns false, unclaimed tasks are skipped. Tasks must not throw.
void run_image_tasks(std::size_t count, const std::function<bool(std::size_t)>& task);

struct EmbeddingLimits {
  std::size_t max_inputs = 32;
  std::uint32_t max_batch_tokens = 8192;
  bool audio = false;
};
struct EmbeddingInput {
  std::vector<embeddinggemma2::PreparedInput> inputs;
  int dimensions = 768;
  bool base64 = false;
};
struct EmbeddingResult { std::vector<std::vector<float>> vectors; };

struct Request {
  Operation operation = Operation::generate;
  // Complete HTTP body received; excludes upload but includes preparation.
  std::chrono::steady_clock::time_point arrival_time{};
  bool chat = false;
  bool stream = false;
  bool include_usage = false;
  bool logprobs = false;
  std::uint32_t top_logprobs = 5;
  std::shared_ptr<const std::vector<std::uint32_t>> prompt;
  std::vector<std::shared_ptr<const runtime::ImageInput>> images;
  std::uint32_t max_tokens = 256;
  float temperature = 1;
  float top_p = 1;
  std::uint32_t top_k = 0;
  std::optional<std::int64_t> seed;
  std::vector<std::string> stops;
  bool allow_tool_calls = false;
  bool require_tool_calls = false;
  bool parallel_tool_calls = true;
  bool enforce_tool_calls = false;
  bool enable_thinking = false;
  bool initial_reasoning = false;
  std::shared_ptr<const constraint::Compiled> constraint;
  std::vector<std::string> tool_names;
  CacheControls cache;
  std::shared_ptr<const EmbeddingInput> embeddings;
};

struct Error : std::runtime_error {
  int status;
  std::string param;
  std::string code;
  Error(int status, std::string message, std::string param = {},
        std::string code = {});
};

// Typed execution accounting. No internal binary protocol frames are used.
struct Result {
  std::uint32_t prompt_tokens = 0, completion_tokens = 0, cached_tokens = 0,
                processed_tokens = 0;
  std::uint64_t input_checkpoint = 0, completion_checkpoint = 0;
  bool stopped = false;
  kv_cache::CacheStats cache_stats;
};

Request parse_request(std::string_view method, std::string_view path,
                      std::string_view body, const text::Tokenizer& tokenizer,
                      const std::string& model, constraint::Compiler& compiler,
                      const ImageSupport& images = {});
nlohmann::json error_json(const Error& error);
nlohmann::json model_json(const std::string& model, std::int64_t created);
nlohmann::json immediate_json(const Request& request, const std::string& model,
                             std::int64_t created, bool ready);
nlohmann::json control_json(const Request& request, const Result& result);
Request parse_embedding_request(std::string_view method, std::string_view path,
    std::string_view body, const text::Tokenizer& tokenizer, const std::string& model,
    constraint::Compiler& compiler, const EmbeddingLimits& limits, std::size_t max_output_bytes,
    const ImageSupport& images = {}, const embeddinggemma2::audio::Acquire& acquire_audio = {});
nlohmann::json embedding_json(const Request& request, const EmbeddingResult& result,
                             const std::string& model);

// Called only by the HTTP I/O thread. push() returns SSE text for streams and
// accumulates bounded text for buffered responses. finish() verifies execution
// accounting before emitting successful terminal events or buffered JSON.
class CompletionOutput {
 public:
  CompletionOutput(const text::Tokenizer& tokenizer, const Request& request,
                   std::string model, std::string id, std::int64_t created,
                   std::size_t max_output_bytes);
  ~CompletionOutput();
  std::string start();
  std::string push(const std::uint32_t* tokens, std::size_t count,
                   const TokenLogprobs* logprobs = nullptr);
  std::string finish(const Result& result);
  bool stop_matched() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gewell::http
