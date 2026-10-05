#include "gewell/console.h"
#include "gewell/app.h"
#include "runner_internal.h"
#include "gewell/mtp_cycle.h"
#include "gewell/metrics.h"

#include "gewell/models/gemma4/31b/artifact.h"
#include "gewell/http_server.h"
#include "gewell/json_constraint.h"
#include "gewell/logprobs.h"
#include "gewell/models/gemma4/31b/serving_assets.h"
#include "gewell/models/gemma4/image_processor.h"
#include "gewell/bf16_primitives.h"
#include "gewell/models/gemma4/31b/model.h"
#include "gewell/kv_cache.h"
#include "gewell/prefix_index.h"
#include "gewell/pending_prefix.h"
#include "gewell/prefill_primitives.h"
#include "gewell/replay_metrics.h"
#include "gewell/vision_engine.h"
#include "gewell/vision_executor.h"
#include "gewell/weight_qdq.h"
#include "gewell/models/gemma4/31b/sm120/nvfp4_projections.h"
#include "gewell/models/gemma4/31b/sm120/fp8_projections.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <openssl/evp.h>


#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <functional>
#include <deque>
#include <fcntl.h>
#include <iostream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "gewell/runtime/scheduler.h"
#include "models/gemma4/31b/sm120/runtime_backend.h"
#include "models/gemma4/31b/sm120/runner_support.cuh"
#include "models/gemma4/31b/sm120/batch_execution.cuh"
#include "models/gemma4/31b/sm120/cache_config.h"
#include "models/gemma4/31b/sm120/vision/prepared_image.h"

namespace gewell::app {
namespace {

namespace vision_engine = gewell::vision_engine;
namespace vision_executor = gewell::vision_executor;
namespace sm120 = gewell::gemma4_31b::sm120;
namespace kv_cache = gewell::kv_cache;
namespace prefix_index = gewell::prefix_index;
using namespace sm120;
using runtime::CacheRequestMode;
using runtime::BatchRequest;
using runtime::BatchPhase;
using runtime::BatchFailure;
using runtime::batch_failure;
using runtime::BatchLimits;
using runtime::BatchCallbacks;
using runtime::BatchScheduler;
using runtime::Operation;


// Runtime workspace grows with the configured text chunk cap; the default
// retains enough room for one image prompt and the existing decode batch cap.
std::vector<std::uint32_t> read_generation_prompt(
    const std::string& prompt_path) {
  if (prompt_path.empty() || prompt_path == "-") {
    fail("read generation prompt", "PROMPT.u32 must be a file path");
  }
  const int descriptor = ::open(prompt_path.c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    fail("open generation prompt",
         prompt_path + ": " + std::strerror(errno));
  }
  struct stat metadata {};
  if (::fstat(descriptor, &metadata) != 0) {
    const std::string detail = prompt_path + ": " + std::strerror(errno);
    ::close(descriptor);
    fail("inspect generation prompt", detail);
  }
  constexpr std::size_t kMaximumBytes =
      static_cast<std::size_t>(primitives::kMaxContextTokenCount) *
      sizeof(std::uint32_t);
  if (!S_ISREG(metadata.st_mode) || metadata.st_size <= 0 ||
      static_cast<std::uint64_t>(metadata.st_size) > kMaximumBytes ||
      metadata.st_size % static_cast<off_t>(sizeof(std::uint32_t)) != 0) {
    ::close(descriptor);
    fail("inspect generation prompt",
         "file must contain 1..262144 little-endian uint32 token IDs");
  }
  std::vector<std::uint8_t> encoded(
      static_cast<std::size_t>(metadata.st_size));
  std::size_t read_bytes = 0;
  while (read_bytes < encoded.size()) {
    const ssize_t result =
        ::read(descriptor, encoded.data() + read_bytes,
               encoded.size() - read_bytes);
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result < 0) {
      const std::string detail = prompt_path + ": " + std::strerror(errno);
      ::close(descriptor);
      fail("read generation prompt", detail);
    }
    if (result == 0) {
      ::close(descriptor);
      fail("read generation prompt",
           prompt_path + ": unexpected end of file");
    }
    read_bytes += static_cast<std::size_t>(result);
  }
  std::uint8_t trailing = 0;
  ssize_t trailing_result = 0;
  do {
    trailing_result = ::read(descriptor, &trailing, sizeof(trailing));
  } while (trailing_result < 0 && errno == EINTR);
  if (trailing_result < 0) {
    const std::string detail = prompt_path + ": " + std::strerror(errno);
    ::close(descriptor);
    fail("read generation prompt", detail);
  }
  if (trailing_result != 0) {
    ::close(descriptor);
    fail("read generation prompt",
         prompt_path + ": file grew while it was being read");
  }
  if (::close(descriptor) != 0) {
    fail("close generation prompt",
         prompt_path + ": " + std::strerror(errno));
  }

  std::vector<std::uint32_t> tokens(encoded.size() / sizeof(std::uint32_t));
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    const std::size_t offset = index * sizeof(std::uint32_t);
    tokens[index] = static_cast<std::uint32_t>(encoded[offset]) |
                    (static_cast<std::uint32_t>(encoded[offset + 1]) << 8) |
                    (static_cast<std::uint32_t>(encoded[offset + 2]) << 16) |
                    (static_cast<std::uint32_t>(encoded[offset + 3]) << 24);
    if (tokens[index] >= model::kVocabSize) {
      fail("validate generation prompt",
           "token at index " + std::to_string(index) +
               " is outside the Gemma 4 vocabulary");
    }
  }
  return tokens;
}

std::vector<std::uint8_t> read_exact_input(const std::string& path,
                                           std::size_t expected_bytes,
                                           std::string_view label) {
  if (path.empty() || path == "-") {
    fail(label, "input must be a file path");
  }
  const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    fail(label, path + ": " + std::strerror(errno));
  }
  struct stat metadata {};
  if (::fstat(descriptor, &metadata) != 0) {
    const std::string detail = path + ": " + std::strerror(errno);
    ::close(descriptor);
    fail(label, detail);
  }
  if (!S_ISREG(metadata.st_mode) || metadata.st_size < 0 ||
      static_cast<std::uint64_t>(metadata.st_size) != expected_bytes) {
    const std::string detail =
        metadata.st_size < 0 || !S_ISREG(metadata.st_mode)
            ? "input is not a regular file of the required size"
            : "got " + std::to_string(metadata.st_size) + " bytes, expected " +
                  std::to_string(expected_bytes);
    ::close(descriptor);
    fail(label, path + ": " + detail);
  }
  std::vector<std::uint8_t> bytes(expected_bytes);
  std::size_t read_bytes = 0;
  while (read_bytes < bytes.size()) {
    const ssize_t result =
        ::read(descriptor, bytes.data() + read_bytes,
               bytes.size() - read_bytes);
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result < 0) {
      const std::string detail = path + ": " + std::strerror(errno);
      ::close(descriptor);
      fail(label, detail);
    }
    if (result == 0) {
      ::close(descriptor);
      fail(label, path + ": unexpected end of file");
    }
    read_bytes += static_cast<std::size_t>(result);
  }
  std::uint8_t trailing = 0;
  ssize_t trailing_result = 0;
  do {
    trailing_result = ::read(descriptor, &trailing, sizeof(trailing));
  } while (trailing_result < 0 && errno == EINTR);
  if (trailing_result < 0) {
    const std::string detail = path + ": " + std::strerror(errno);
    ::close(descriptor);
    fail(label, detail);
  }
  if (trailing_result != 0) {
    ::close(descriptor);
    fail(label, path + ": input grew while it was being read");
  }
  if (::close(descriptor) != 0) {
    fail(label, path + ": " + std::strerror(errno));
  }
  return bytes;
}

struct ImagePromptSpan {
  std::uint32_t begin{};
  std::uint32_t end{};
};

ImagePromptSpan find_image_prompt_span(
    const std::vector<std::uint32_t>& prompt) {
  const auto first =
      std::find(prompt.begin(), prompt.end(), model::kImageTokenId);
  if (first == prompt.end()) {
    fail("validate caption prompt", "no image placeholder tokens");
  }
  const auto after =
      std::find_if(first, prompt.end(), [](std::uint32_t token) {
        return token != model::kImageTokenId;
      });
  if (std::find(after, prompt.end(), model::kImageTokenId) != prompt.end()) {
    fail("validate caption prompt",
         "image placeholder tokens must form one contiguous block");
  }
  const std::size_t begin =
      static_cast<std::size_t>(std::distance(prompt.begin(), first));
  const std::size_t end =
      static_cast<std::size_t>(std::distance(prompt.begin(), after));
  if (begin == 0 || end >= prompt.size() ||
      prompt[begin - 1] != model::kBeginImageTokenId ||
      prompt[end] != model::kEndImageTokenId) {
    fail("validate caption prompt",
         "image placeholder block must be immediately bracketed by begin/end-image tokens");
  }
  if (end - begin > model::kVisionMaxSoftTokenCount) {
    fail("validate caption prompt",
         "image placeholder block exceeds 1120 tokens");
  }
  return {static_cast<std::uint32_t>(begin),
          static_cast<std::uint32_t>(end)};
}

void preflight_generation_output(const std::string& output_path,
                                 std::string_view argument) {
  const std::string operation =
      argument == "OUTPUT.u32" ? "inspect generation output"
                               : "inspect generation logits output";
  if (output_path.empty() || output_path == "-") {
    fail(operation, std::string(argument) + " must be a file path");
  }
  const std::filesystem::path path(output_path);
  std::error_code error;
  if (std::filesystem::exists(path, error) || error) {
    fail(operation,
         error ? output_path + ": " + error.message()
               : output_path + " already exists (overwrite refused)");
  }
  const std::filesystem::path parent =
      path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
  if (!std::filesystem::is_directory(parent, error) || error) {
    fail(operation,
         error ? parent.string() + ": " + error.message()
               : parent.string() + " is not a directory");
  }
}

void preflight_generation_outputs(const std::string& output_path,
                                  const std::string& logits_output_path) {
  preflight_generation_output(output_path, "OUTPUT.u32");
  if (logits_output_path.empty()) {
    return;
  }
  preflight_generation_output(logits_output_path, "LOGITS.bf16");
  std::error_code output_error;
  const std::filesystem::path resolved_output =
      std::filesystem::weakly_canonical(output_path, output_error);
  std::error_code logits_error;
  const std::filesystem::path resolved_logits =
      std::filesystem::weakly_canonical(logits_output_path, logits_error);
  if (output_error || logits_error) {
    fail("inspect generation outputs",
         output_error ? output_path + ": " + output_error.message()
                      : logits_output_path + ": " + logits_error.message());
  }
  if (resolved_output == resolved_logits) {
    fail("inspect generation logits output",
         "LOGITS.bf16 must differ from OUTPUT.u32");
  }
}

class GenerationLogitsOutput final : public GenerationLogitsSink {
 public:
  GenerationLogitsOutput(const std::string& path,
                         std::uint32_t expected_rows)
      : path_(path),
        expected_rows_(expected_rows),
        row_(kGenerationLogitRowBytes) {
    const std::filesystem::path final_path(path_);
    const std::filesystem::path parent =
        final_path.has_parent_path() ? final_path.parent_path()
                                     : std::filesystem::path(".");
    const std::filesystem::path temporary_template =
        parent / ".gewell-logits-XXXXXX";
    temporary_path_ = temporary_template.string();
    descriptor_ = ::mkstemp(temporary_path_.data());
    if (descriptor_ < 0) {
      fail("create generation logits output",
           temporary_template.string() + ": " + std::strerror(errno));
    }
  }

  ~GenerationLogitsOutput() {
    if (descriptor_ >= 0) {
      ::close(descriptor_);
    }
    if (!temporary_path_.empty()) {
      ::unlink(temporary_path_.c_str());
    }
  }

  GenerationLogitsOutput(const GenerationLogitsOutput&) = delete;
  GenerationLogitsOutput& operator=(const GenerationLogitsOutput&) = delete;

  void write_device_row(const BFloat16* logits, cudaStream_t stream) {
    if (rows_written_ >= expected_rows_) {
      fail("write generation logits output", "too many decision rows");
    }
    check_cuda(cudaMemcpyAsync(row_.data(), logits, row_.size(),
                               cudaMemcpyDeviceToHost, stream),
               "copy generation logits row");
    check_cuda(cudaStreamSynchronize(stream),
               "synchronize generation logits row");

    std::size_t written = 0;
    while (written < row_.size()) {
      const ssize_t result =
          ::write(descriptor_, row_.data() + written, row_.size() - written);
      if (result < 0 && errno == EINTR) {
        continue;
      }
      if (result <= 0) {
        fail("write generation logits output",
             path_ + ": " + std::strerror(errno));
      }
      written += static_cast<std::size_t>(result);
    }
    ++rows_written_;
  }

  void publish(std::uint32_t actual_rows) {
    if (rows_written_ != actual_rows || actual_rows > expected_rows_) {
      fail("publish generation logits output", "decision row count is wrong");
    }
    const int descriptor = descriptor_;
    descriptor_ = -1;
    if (::close(descriptor) != 0) {
      fail("close generation logits output",
           temporary_path_ + ": " + std::strerror(errno));
    }
    if (::link(temporary_path_.c_str(), path_.c_str()) != 0) {
      fail("publish generation logits output",
           path_ + ": " + std::strerror(errno));
    }
    if (::unlink(temporary_path_.c_str()) == 0) {
      temporary_path_.clear();
    }
  }

 private:
  std::string path_;
  std::string temporary_path_;
  std::uint32_t expected_rows_{};
  std::vector<std::uint8_t> row_;
  std::uint32_t rows_written_{};
  int descriptor_{-1};
};

class StopTokenDecisionSink final : public GenerationDecisionSink {
 public:
  bool honors_stop_tokens() const override { return true; }
  bool write_device_decision(const std::uint32_t* token,
                             const BFloat16*, cudaStream_t stream) override {
    check_cuda(cudaMemcpyAsync(&host_token_, token, sizeof(host_token_),
                               cudaMemcpyDeviceToHost, stream),
               "copy caption decision token");
    check_cuda(cudaStreamSynchronize(stream),
               "synchronize caption decision token");
    return !is_generation_stop_token(host_token_);
  }

 private:
  std::uint32_t host_token_{};
};

std::vector<std::uint8_t> write_generation_output(
    const std::string& output_path,
    const std::vector<std::uint32_t>& tokens) {
  std::vector<std::uint8_t> encoded(tokens.size() * sizeof(std::uint32_t));
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    const std::uint32_t token = tokens[index];
    const std::size_t offset = index * sizeof(std::uint32_t);
    encoded[offset] = static_cast<std::uint8_t>(token);
    encoded[offset + 1] = static_cast<std::uint8_t>(token >> 8);
    encoded[offset + 2] = static_cast<std::uint8_t>(token >> 16);
    encoded[offset + 3] = static_cast<std::uint8_t>(token >> 24);
  }
  write_exclusive(output_path, encoded.data(), encoded.size());
  return encoded;
}



std::optional<double> mtp_accepted_median(
    const std::vector<std::uint64_t>& histogram) {
  std::uint64_t count = 0;
  for (const auto frequency : histogram) count += frequency;
  if (count == 0) return std::nullopt;
  const auto lower = (count - 1) / 2;
  const auto upper = count / 2;
  std::uint64_t cumulative = 0;
  std::size_t lower_value = 0;
  for (std::size_t accepted = 0; accepted < histogram.size(); ++accepted) {
    const auto previous = cumulative;
    cumulative += histogram[accepted];
    if (previous <= lower && lower < cumulative) lower_value = accepted;
    if (upper < cumulative) return (lower_value + accepted) / 2.0;
  }
  return std::nullopt;
}


std::uint64_t batch_integer(const std::string& text, std::uint64_t maximum) {
  std::uint64_t value = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      value > maximum) fail("batch manifest", "invalid unsigned integer: " + text);
  return value;
}

float batch_float(const std::string& text, float maximum) {
  char* end = nullptr;
  const float value = std::strtof(text.c_str(), &end);
  if (text.empty() || end != text.c_str() + text.size() || !std::isfinite(value) ||
      value < 0 || value > maximum) fail("batch manifest", "invalid sampling value: " + text);
  return value;
}

std::vector<BatchRequest> read_batch_requests(const std::string& path) {
  std::ifstream input(path);
  if (!input) fail("batch manifest", "cannot open " + path);
  if (std::filesystem::file_size(path) > 8 * kv_cache::kMib)
    fail("batch manifest", "manifest exceeds 8 MiB");
  std::vector<BatchRequest> requests;
  std::set<std::string> ids;
  std::string line;
  std::size_t prompt_bytes = 0;
  while (std::getline(input, line)) {
    std::vector<std::string> fields;
    std::size_t begin = 0;
    while (true) {
      const auto end = line.find('\t', begin);
      fields.push_back(line.substr(begin, end == std::string::npos ? end : end - begin));
      if (end == std::string::npos) break;
      begin = end + 1;
    }
    if (fields.size() != 9 || requests.size() == 4096)
      fail("batch manifest", "expected nine TSV fields and at most 4096 requests");
    BatchRequest request;
    request.id = fields[0];
    const auto initial_char = [](char c) {
      return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
             (c >= '0' && c <= '9') || c == '_' || c == '-';
    };
    if (request.id.empty() || request.id.size() > 128 || !initial_char(request.id[0]) ||
        !std::all_of(request.id.begin(), request.id.end(), [&](char c) {
          return initial_char(c) || c == '.';
        }) || !ids.insert(request.id).second)
      fail("batch manifest", "invalid or duplicate request ID");
    if (fields[1].empty() || fields[1].find('\0') != std::string::npos ||
        fields[1].find('\r') != std::string::npos)
      fail("batch manifest", "invalid prompt path");
    const auto prompt_path = std::filesystem::path(path).parent_path() / fields[1];
    request.prompt = std::make_shared<const std::vector<std::uint32_t>>(
        read_generation_prompt(prompt_path.string()));
    prompt_bytes += request.prompt->size() * sizeof(std::uint32_t);
    if (prompt_bytes > 256 * kv_cache::kMib)
      fail("batch manifest", "prompt storage exceeds 256 MiB");
    request.max_new_tokens = static_cast<std::uint32_t>(
        batch_integer(fields[2], primitives::kMaxContextTokenCount));
    generation_cache_capacity(request.prompt->size(), request.max_new_tokens);
    request.sampling.temperature = batch_float(fields[3], std::numeric_limits<float>::max());
    request.sampling.top_p = batch_float(fields[4], 1);
    request.sampling.top_k = static_cast<std::uint32_t>(batch_integer(fields[5], model::kVocabSize));
    request.seed = batch_integer(fields[6], std::numeric_limits<std::uint64_t>::max());
    request.honor_eos = batch_integer(fields[7], 1) != 0;
    request.capture_logits = batch_integer(fields[8], 1) != 0;
    request.rng.seed(request.seed);
    requests.push_back(std::move(request));
  }
  if (!input.eof() || requests.empty()) fail("batch manifest", "empty or unreadable manifest");
  return requests;
}

// Deterministic scheduler fixtures use physical prefill progress instead of
// wall-clock races. Ordinary manifests make every request available at startup.
struct BatchEvent {
  std::uint64_t after{};
  bool inflight{};
  std::string action;
  std::size_t request{};
  bool applied{};
};

std::vector<BatchEvent> read_batch_events(
    const std::string& path, std::vector<BatchRequest>& requests) {
  if (path.empty()) return {};
  std::ifstream input(path);
  if (!input || std::filesystem::file_size(path) > kv_cache::kMib)
    fail("batch events", "unreadable file or exceeds 1 MiB");
  std::vector<BatchEvent> events;
  std::set<std::size_t> arrivals;
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream fields(line);
    std::string after, phase, action, id, extra;
    if (!std::getline(fields, after, '\t') || !std::getline(fields, phase, '\t') ||
        !std::getline(fields, action, '\t') || !std::getline(fields, id, '\t') ||
        std::getline(fields, extra, '\t') || events.size() == 8192 ||
        (phase != "idle" && phase != "inflight") ||
        (action != "arrive" && action != "cancel" && action != "fail"))
      fail("batch events", "expected bounded AFTER, PHASE, ACTION, ID TSV records");
    const auto found = std::find_if(requests.begin(), requests.end(),
        [&](const auto& request) { return request.id == id; });
    if (found == requests.end()) fail("batch events", "unknown request ID: " + id);
    const auto index = static_cast<std::size_t>(found - requests.begin());
    if (action == "arrive") {
      if (!arrivals.insert(index).second) fail("batch events", "duplicate arrival");
      found->phase = BatchPhase::withheld;
    }
    events.push_back({batch_integer(after, std::numeric_limits<std::uint64_t>::max()),
                      phase == "inflight", action, index, false});
  }
  if (!input.eof()) fail("batch events", "cannot read events");
  return events;
}

// At most one emitted block per active request is outstanding. That request
// pauses while it drains; slow logit storage cannot block token-only requests.
class BatchLogitWriter {
 public:
  BatchLogitWriter() : worker_([this] {
    for (;;) {
      std::packaged_task<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
        if (queue_.empty()) return;
        task = std::move(queue_.front());
        queue_.pop_front();
      }
      task();
    }
  }) {}
  ~BatchLogitWriter() {
    { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
    ready_.notify_one();
    worker_.join();
  }
  std::future<void> write(std::ofstream& output, std::vector<std::uint8_t> bytes) {
    std::packaged_task<void()> task([&output, bytes = std::move(bytes)] {
      output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
      if (!output) fail("batch logits", "output write failed");
    });
    auto result = task.get_future();
    { std::lock_guard<std::mutex> lock(mutex_); queue_.push_back(std::move(task)); }
    ready_.notify_one();
    return result;
  }
 private:
  std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<std::packaged_task<void()>> queue_;
  bool stopping_{};
  std::thread worker_;
};

volatile std::sig_atomic_t batch_interrupted = 0;
void interrupt_batch(int) { batch_interrupted = 1; }

class BatchSignalScope {
 public:
  BatchSignalScope() {
    batch_interrupted = 0;
    old_int_ = std::signal(SIGINT, interrupt_batch);
    old_term_ = std::signal(SIGTERM, interrupt_batch);
  }
  ~BatchSignalScope() {
    std::signal(SIGINT, old_int_);
    std::signal(SIGTERM, old_term_);
  }
 private:
  using Handler = void (*)(int);
  Handler old_int_{}, old_term_{};
};

}  // namespace

namespace {

int run_loaded_generation(const WeightArena& weights,
                          const std::vector<std::uint32_t>& prompt,
                          std::uint32_t new_token_count,
                          const std::string& output_path,
                          const std::string& logits_output_path,
                          std::size_t free_before, std::size_t total,
                          double load_seconds,
                          const std::vector<VisionPromptSlice>& vision = {},
                          bool honor_stop_tokens = false,
                          const GenerationSettings& settings = {}) {
  Executor engine(weights, prompt.size(),
                                           new_token_count,
                                           !vision.empty(),
                                           {settings.temperature, settings.top_p, settings.top_k},
                                           nullptr, 0, {}, settings.mtp_depth, settings.seed, settings.prefill_chunk_tokens,
                                           settings.nvfp4_activation_policy, 1, settings.local_kv_format, settings.global_kv_format,
                                           settings.local_attention_compute, settings.global_attention_compute);
  log_nvfp4_policy(weights, settings.nvfp4_activation_policy);
  console::field("attention_local_compute", attention::compute_name(settings.local_attention_compute));
  console::field("attention_global_compute", attention::compute_name(settings.global_attention_compute));
  console::field("kv_local_format", kv_cache::format_name(settings.local_kv_format));
  console::field("kv_global_format", kv_cache::format_name(settings.global_kv_format));
  std::unique_ptr<GenerationLogitsOutput> logits_output;
  if (!logits_output_path.empty()) {
    logits_output = std::make_unique<GenerationLogitsOutput>(
        logits_output_path, new_token_count);
  }
  std::size_t free_after_initialization = 0;
  std::size_t total_after_initialization = 0;
  check_cuda(cudaMemGetInfo(&free_after_initialization,
                            &total_after_initialization),
             "cudaMemGetInfo after generation initialization");
  if (total_after_initialization != total ||
      free_after_initialization > free_before) {
    fail("generation CUDA memory", "inconsistent initialization memory");
  }
  const std::size_t cache_bytes =
      engine.local_cache_bytes() + engine.global_cache_bytes();
  std::size_t vision_feature_bytes = 0;
  for (const auto& image : vision)
    vision_feature_bytes += std::size_t(image.end - image.begin) * model::kHiddenSize * sizeof(BFloat16);
  const std::size_t planned_device_bytes =
      weights.size() + engine.scratch_bytes() + cache_bytes +
      engine.output_bytes() + vision_feature_bytes;
  const double weight_copy_seconds =
      std::max(0.0, load_seconds - weights.qdq_summary().seconds);
  console::section("GPU allocation");
  console::field("generation_kv_layout_selector", "compact-global");
  if (console::json_enabled())
    console::field("generation_kv_layout_id", kGenerationKvLayoutId);
  console::field("weight_copy_seconds", weight_copy_seconds);
  console::field("weight_qdq_seconds", weights.qdq_summary().seconds);
  console::field("device_weight_arena_bytes", weights.size());
  console::field("device_model_scratch_bytes", engine.model_scratch_bytes());
  console::field("device_decode_attention_scratch_bytes", engine.attention_scratch_bytes());
  console::field("device_prefill_attention_scratch_bytes", engine.prefill_attention_scratch_bytes());
  console::field("device_sampling_scratch_bytes", engine.sampling_scratch_bytes());
  console::field("local_kv_cache_capacity_tokens", kLocalCacheCapacity);
  console::field("local_kv_cache_bytes", engine.local_cache_bytes());
  console::field("global_kv_cache_capacity_tokens", engine.global_capacity());
  console::field("global_kv_cache_bytes", engine.global_cache_bytes());
  console::field("total_kv_cache_bytes", cache_bytes);
  console::field("device_output_bytes", engine.output_bytes());
  console::field("device_vision_feature_bytes", vision_feature_bytes);
  console::field("planned_device_bytes", planned_device_bytes);
  console::field("gpu_total_bytes", total);
  console::field("gpu_free_before_bytes", free_before);
  console::field("gpu_free_after_initialization_bytes", free_after_initialization);

  StopTokenDecisionSink stop_sink;
  GenerationDecisionSink* const decision_sink =
      honor_stop_tokens ? &stop_sink : nullptr;
  const GenerationResult result =
      engine.generate(prompt, logits_output.get(), decision_sink, vision);
  if (result.outputs.empty() || result.outputs.size() > new_token_count ||
      (!honor_stop_tokens && result.outputs.size() != new_token_count) ||
      !std::all_of(result.outputs.begin(), result.outputs.end(),
                   [](std::uint32_t token) {
                     return token < model::kVocabSize;
                   })) {
    fail("generation result", "output count or vocabulary contract failed");
  }
  if (honor_stop_tokens && result.outputs.size() < new_token_count &&
      !is_generation_stop_token(result.outputs.back())) {
    fail("generation result", "caption stopped without a configured stop token");
  }
  const std::vector<std::uint8_t> encoded_output =
      write_generation_output(output_path, result.outputs);
  if (logits_output != nullptr) {
    try {
      logits_output->publish(static_cast<std::uint32_t>(result.outputs.size()));
    } catch (...) {
      ::unlink(output_path.c_str());
      throw;
    }
  }
  const std::string output_sha256 = artifact::digest_hex(
      sha256_bytes(encoded_output.data(), encoded_output.size()));
  const std::uint32_t decode_tokens =
      static_cast<std::uint32_t>(result.outputs.size() - 1);
  const double prefill_gpu_seconds =
      result.prefill_gpu_milliseconds / 1'000.0;
  const double decode_gpu_seconds = result.decode_gpu_milliseconds / 1'000.0;
  if (console::json_enabled() || settings.mtp_depth != 0) {
    console::section("Speculative decoding");
    console::field("generation_mtp_cycles", result.mtp_cycles);
    console::field("generation_mtp_proposed", result.mtp_proposed);
    console::field("generation_mtp_accepted", result.mtp_accepted);
    console::field("generation_mtp_rejected_cycles", result.mtp_rejected);
    console::field("generation_mtp_draft_gpu_milliseconds", result.mtp_draft_gpu_milliseconds);
    console::field("generation_mtp_verify_gpu_milliseconds", result.mtp_verify_gpu_milliseconds);
    console::field("generation_mtp_select_gpu_milliseconds", result.mtp_select_gpu_milliseconds);
  }
  console::section("Generation performance");
  console::field("generation_prefill_gpu_milliseconds", result.prefill_gpu_milliseconds);
  console::field("generation_prefill_wall_seconds", result.prefill_wall_seconds);
  console::field("generation_prefill_gpu_tokens_per_second", prompt.size() / prefill_gpu_seconds);
  console::field("generation_decode_tokens_fed", decode_tokens);
  console::field("generation_decode_gpu_milliseconds", result.decode_gpu_milliseconds);
  console::field("generation_decode_wall_seconds", result.decode_wall_seconds);
  if (decode_tokens != 0 && decode_gpu_seconds > 0.0) {
    console::field("generation_decode_gpu_tokens_per_second", decode_tokens / decode_gpu_seconds);
  } else {
    console::field("generation_decode_gpu_tokens_per_second", "n/a");
  }
  console::section("Generation output");
  console::field("generation_output_tokens", result.outputs.size());
  if (console::json_enabled())
    console::field("generation_first_token", result.outputs.front());
  if (console::json_enabled())
    console::field("generation_last_token", result.outputs.back());
  if (console::json_enabled())
    console::field("generation_output_sha256", output_sha256);
  console::field("generation_output_file", output_path);
  console::field("generation_output_bytes", result.outputs.size() * sizeof(std::uint32_t));
  if (!logits_output_path.empty()) {
    if (console::json_enabled())
      console::field("generation_logits_dtype", "bf16_le");
    if (console::json_enabled())
      console::field("generation_logits_layout", "token_major_post_softcap");
    console::field("generation_logits_rows", result.outputs.size());
    console::field("generation_logits_columns", model::kVocabSize);
    console::field("generation_logits_file", logits_output_path);
    console::field("generation_logits_bytes", result.outputs.size() * kGenerationLogitRowBytes);
  } else {
    console::field("generation_logits_file", "disabled");
  }
  return 0;
}

}  // namespace

std::uint32_t effective_mtp_depth(std::uint32_t depth, const std::string& assistant_path,
                                  std::uint32_t min_depth) {
  if (min_depth > depth)
    throw std::runtime_error("--mtp-min-depth must not exceed --mtp-depth");
  if (depth && assistant_path.empty()) {
    console::event("mtp_disabled", {{"requested_depth", depth}, {"effective_depth", 0}},
                   "Warning: --mtp-depth requires --assistant PATH; forcing depth to 0.", true);
    return 0;
  }
  return depth;
}

int run_generate(const std::string& artifact_path,
                 const std::string& prompt_path,
                 std::uint32_t new_token_count,
                 const std::string& output_path,
                 const std::string& logits_output_path,
                 const std::string& qdq_mask_path,
                 GenerationSettings settings) {
  settings.mtp_depth = effective_mtp_depth(settings.mtp_depth, settings.assistant_path);
  if (settings.mtp_depth > mtp_target::kMaxDepth)
    fail("MTP settings", "depth must be 0..1279");
  if (!std::isfinite(settings.temperature) || settings.temperature < 0 ||
      !std::isfinite(settings.top_p) || settings.top_p < 0 || settings.top_p > 1 ||
      settings.top_k > model::kVocabSize)
    fail("sampling settings", "invalid temperature, top-p, or top-k");

  const qdq::Mask qdq_mask = qdq_mask_path.empty()
                                 ? qdq::Mask{}
                                 : qdq::Mask::Load(qdq_mask_path);
  const qdq::SelectionSummary qdq_selection = qdq::summarize(qdq_mask);
  const std::vector<std::uint32_t> prompt =
      read_generation_prompt(prompt_path);
  const std::uint32_t cache_capacity =
      generation_cache_capacity(prompt.size(), new_token_count);
  preflight_generation_outputs(output_path, logits_output_path);

  console::section("Text generation");
  if (console::json_enabled())
    console::field("generation_mode", "batch1_free_rollout");
  console::field("generation_prompt_file", prompt_path);
  console::field("generation_prompt_tokens", prompt.size());
  console::field("generation_new_tokens", new_token_count);
  console::field("generation_fed_token_capacity", cache_capacity);
  console::field("generation_max_context_tokens", primitives::kMaxContextTokenCount);
  if (console::json_enabled())
    console::field("generation_chunk_tokens", settings.prefill_chunk_tokens);
  console::section("Sampling");
  if (console::json_enabled())
    console::field("generation_sampling", "softcap_30_temperature_top_k_top_p");
  console::field("generation_temperature", settings.temperature);
  console::field("generation_top_p", settings.top_p);
  console::field("generation_top_k", settings.top_k);
  console::field("generation_mtp_depth", settings.mtp_depth);
  console::field("generation_seed", (settings.seed ? std::to_string(*settings.seed) : "entropy"));
  console::field("generation_eos_stopping", "disabled");
  console::field("generation_logits_returned", !(logits_output_path.empty()));
  if (console::json_enabled() || !qdq_mask.all_bf16()) {
    console::section("Weight quantization");
    console::field("weight_qdq_mask_file", (qdq_mask.has_source() ? qdq_mask.source_name() : "disabled"));
    if (qdq_mask.has_source()) {
      if (console::json_enabled())
        console::field("weight_qdq_mask_sha256", artifact::digest_hex(qdq_mask.source_sha256()));
    } else {
      if (console::json_enabled())
        console::field("weight_qdq_mask_sha256", "disabled");
    }
    console::field("weight_qdq_enabled", qdq_mask.has_qdq());
    console::field("weight_qdq_execution_storage",
                   qdq_mask.has_qdq() ? "reconstructed_bf16" : "disabled");
    console::field("weight_qdq_activation_quantization", false);
    console::field("weight_qdq_fp8_recipe", "e4m3fn_per_tensor_absmax_fp32_scale_v1");
    console::field("weight_qdq_nvfp4_recipe", "e2m1_group16_e4m3fn_absmax_fp32_tensor_scale_v1");
    console::field("weight_qdq_bf16_projection_tensors", qdq_selection.bf16.tensor_count);
    console::field("weight_qdq_bf16_projection_source_bytes", qdq_selection.bf16.source_bf16_bytes);
    console::field("weight_qdq_fp8_projection_tensors", qdq_selection.fp8.tensor_count);
    console::field("weight_qdq_fp8_projection_source_bytes", qdq_selection.fp8.source_bf16_bytes);
    console::field("weight_qdq_nvfp4_projection_tensors", qdq_selection.nvfp4.tensor_count);
    console::field("weight_qdq_nvfp4_projection_source_bytes", qdq_selection.nvfp4.source_bf16_bytes);
  }
  console::section("Model artifact");
  console::field("artifact", artifact_path);
  console::field("verification", "disabled (header+table only)");

  artifact::ArtifactFile file = artifact::ArtifactFile::Open(artifact_path);
  if (console::json_enabled())
    console::field("payload_sha256", artifact::digest_hex(file.header().payload_sha256));

  validate_cuda_device();
  std::size_t free_before = 0;
  std::size_t total = 0;
  check_cuda(cudaMemGetInfo(&free_before, &total),
             "cudaMemGetInfo before generation load");
  const auto load_started = std::chrono::steady_clock::now();
  WeightArena weights(file, qdq_mask, settings.mtp_depth ? settings.assistant_path : "", settings.vision_path);
  const double load_seconds = seconds_since(load_started);
  return run_loaded_generation(
      weights, prompt, new_token_count, output_path, logits_output_path,
      free_before, total, load_seconds, {}, false, settings);
}

int run_caption(const std::string& artifact_path,
                const std::string& prompt_path,
                const std::string& pixel_values_path,
                const std::string& position_ids_path,
                std::uint32_t max_new_token_count,
                const std::string& output_path,
                const std::string& logits_output_path,
                GenerationSettings settings) {
  if (settings.vision_path.empty()) fail("caption", "image input requires --vision PATH");
  settings.mtp_depth = effective_mtp_depth(settings.mtp_depth, settings.assistant_path);
  if (settings.mtp_depth > mtp_target::kMaxDepth)
    fail("MTP settings", "depth must be 0..1279");
  if (!std::isfinite(settings.temperature) || settings.temperature < 0 ||
      !std::isfinite(settings.top_p) || settings.top_p < 0 || settings.top_p > 1 ||
      settings.top_k > model::kVocabSize)
    fail("sampling settings", "invalid temperature, top-p, or top-k");

  const std::vector<std::uint32_t> prompt =
      read_generation_prompt(prompt_path);
  const ImagePromptSpan image = find_image_prompt_span(prompt);
  const std::uint32_t soft_token_count = image.end - image.begin;
  std::error_code pixel_size_error;
  const std::uintmax_t pixel_byte_count =
      std::filesystem::file_size(pixel_values_path, pixel_size_error);
  if (pixel_size_error) {
    fail("inspect caption pixel input",
         pixel_values_path + ": " + pixel_size_error.message());
  }
  std::uint32_t image_max_soft_tokens = 0;
  for (const std::uint32_t capacity :
       vision_engine::kSupportedSoftTokenCapacities) {
    const std::uint32_t rows =
        vision_engine::padded_patch_rows_for_capacity(capacity);
    if (pixel_byte_count == vision_engine::prepared_pixel_bytes(rows)) {
      image_max_soft_tokens = capacity;
      break;
    }
  }
  if (image_max_soft_tokens == 0) {
    fail("inspect caption pixel input",
         "byte length does not select a supported soft-token capacity");
  }
  const std::uint32_t padded_patch_rows =
      vision_engine::padded_patch_rows_for_capacity(image_max_soft_tokens);
  const std::vector<std::uint8_t> pixels = read_exact_input(
      pixel_values_path,
      vision_engine::prepared_pixel_bytes(padded_patch_rows),
      "read caption pixel input");
  const std::vector<std::uint8_t> positions = read_exact_input(
      position_ids_path,
      vision_engine::prepared_position_bytes(padded_patch_rows),
      "read caption position input");
  vision_engine::validate_prepared_image_bytes(
      pixels.data(), pixels.size(), positions.data(), positions.size(),
      soft_token_count);
  const std::uint32_t cache_capacity =
      generation_cache_capacity(prompt.size(), max_new_token_count);
  preflight_generation_outputs(output_path, logits_output_path);

  console::section("Image captioning");
  if (console::json_enabled())
    console::field("caption_mode", "gemma4_31b_bf16_image_conditioned");
  console::field("caption_prompt_file", prompt_path);
  console::field("caption_prompt_tokens", prompt.size());
  if (console::json_enabled())
    console::field("caption_image_begin", image.begin);
  if (console::json_enabled())
    console::field("caption_image_end", image.end);
  console::field("caption_soft_tokens", soft_token_count);
  console::field("caption_image_max_soft_tokens", image_max_soft_tokens);
  if (console::json_enabled())
    console::field("caption_padded_patch_rows", padded_patch_rows);
  console::field("caption_pixel_file", pixel_values_path);
  console::field("caption_position_file", position_ids_path);
  console::field("caption_max_new_tokens", max_new_token_count);
  console::field("caption_fed_token_capacity", cache_capacity);
  console::section("Sampling");
  if (console::json_enabled())
    console::field("caption_prefill_mask", "global_causal_local_sliding_causal_or_same_image");
  if (console::json_enabled())
    console::field("caption_sampling", "softcap_30_temperature_top_k_top_p");
  console::field("caption_mtp_depth", settings.mtp_depth);
  console::field("caption_temperature", settings.temperature);
  console::field("caption_top_p", settings.top_p);
  console::field("caption_top_k", settings.top_k);
  console::field("caption_seed", (settings.seed ? std::to_string(*settings.seed) : "entropy"));
  console::field("caption_stop_token_ids", nlohmann::json::array({1,106,50}));
  console::field("caption_logits_returned", !(logits_output_path.empty()));
  console::field("artifact", artifact_path);
  console::field("verification", "disabled (header+table only)");

  artifact::ArtifactFile file = artifact::ArtifactFile::Open(artifact_path);
  if (console::json_enabled())
    console::field("payload_sha256", artifact::digest_hex(file.header().payload_sha256));

  validate_cuda_device();
  std::size_t free_before = 0;
  std::size_t total = 0;
  check_cuda(cudaMemGetInfo(&free_before, &total),
             "cudaMemGetInfo before caption load");
  const auto load_started = std::chrono::steady_clock::now();
  WeightArena weights(file, qdq::Mask{}, settings.mtp_depth ? settings.assistant_path : "", settings.vision_path);
  const double load_seconds = seconds_since(load_started);

  PreparedImage soft_features(weights);
  soft_features.prepare(pixels, positions, padded_patch_rows, soft_token_count);
  const auto vision_milliseconds = soft_features.gpu_milliseconds();
  const auto vision_scratch_bytes = soft_features.scratch_bytes();
  console::section("Vision encoder");
  if (console::json_enabled())
    console::field("caption_vision_weight_copy", "shared_full_artifact_arena");
  console::field("caption_vision_scratch_bytes", vision_scratch_bytes);
  console::field("caption_soft_feature_bytes", soft_features.size());
  console::field("caption_vision_gpu_milliseconds", vision_milliseconds);

  const VisionPromptSlice slice{
      static_cast<const BFloat16*>(soft_features.data()), image.begin,
      image.end};
  return run_loaded_generation(
      weights, prompt, max_new_token_count, output_path,
      logits_output_path, free_before, total, load_seconds, {slice}, true, settings);
}

int run_replay_rollout(const std::string& artifact_path,
                       const std::string& requests_path,
                       std::uint32_t chunk_rows, std::uint32_t head_rows,
                       const std::string& output_directory,
                       const std::string& qdq_mask_path) {
  if (!chunk_rows || chunk_rows > kMaxPrefillChunkTokens || !head_rows || head_rows > chunk_rows)
    fail("replay configuration", "require 1 <= HEAD_ROWS <= CHUNK_ROWS <= 4096");
  struct Request {
    std::string id, reference;
    std::vector<std::uint32_t> prompt, continuation;
  };
  std::ifstream manifest(requests_path);
  if (!manifest || std::filesystem::file_size(requests_path) > 8 * kv_cache::kMib)
    fail("replay manifest", "cannot read manifest or it exceeds 8 MiB");
  std::vector<Request> requests;
  std::set<std::string> ids;
  std::size_t token_bytes = 0;
  std::string line;
  while (std::getline(manifest, line)) {
    std::vector<std::string> fields;
    std::size_t begin = 0;
    while (true) {
      const auto end = line.find('\t', begin);
      fields.push_back(line.substr(begin, end == std::string::npos ? end : end - begin));
      if (end == std::string::npos) break;
      begin = end + 1;
    }
    if (fields.size() != 4 || requests.size() == 4096)
      fail("replay manifest", "expected four TSV fields and at most 4096 requests");
    const auto initial_char = [](char c) {
      return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
             (c >= '0' && c <= '9') || c == '_' || c == '-';
    };
    if (fields[0].empty() || fields[0].size() > 128 || !initial_char(fields[0][0]) ||
        !std::all_of(fields[0].begin(), fields[0].end(), [&](char c) { return initial_char(c) || c == '.'; }) ||
        !ids.insert(fields[0]).second)
      fail("replay manifest", "invalid or duplicate request ID");
    const auto path = [&](std::size_t index) {
      if (fields[index].empty() || fields[index].find('\0') != std::string::npos ||
          fields[index].find('\r') != std::string::npos)
        fail("replay manifest", "invalid input path");
      return (std::filesystem::path(requests_path).parent_path() / fields[index]).string();
    };
    Request request{fields[0], path(3), read_generation_prompt(path(1)), read_generation_prompt(path(2))};
    generation_cache_capacity(request.prompt.size(), static_cast<std::uint32_t>(request.continuation.size()));
    token_bytes += (request.prompt.size() + request.continuation.size()) * sizeof(std::uint32_t);
    if (token_bytes > 256 * kv_cache::kMib)
      fail("replay manifest", "token storage exceeds 256 MiB");
    if (!std::filesystem::is_regular_file(request.reference) ||
        std::filesystem::file_size(request.reference) != request.continuation.size() * kGenerationLogitRowBytes)
      fail("replay reference", "expected one full BF16 vocabulary row per continuation token");
    requests.push_back(std::move(request));
  }
  if (!manifest.eof() || requests.empty()) fail("replay manifest", "empty or unreadable manifest");
  std::error_code directory_error;
  if (!std::filesystem::create_directory(output_directory, directory_error))
    fail("replay output directory", directory_error ? directory_error.message() : "directory already exists");
  const std::filesystem::path output(output_directory);
  artifact::ArtifactFile file = artifact::ArtifactFile::Open(artifact_path);
  const auto mask = qdq_mask_path.empty() ? qdq::Mask{} : qdq::Mask::Load(qdq_mask_path);
  validate_cuda_device();
  console::section("Recorded-token replay");
  if (console::json_enabled())
    console::field("replay_mode", "causal_chunked_compact_global_bf16");
  console::field("replay_requests", requests.size());
  console::field("replay_chunk_rows", chunk_rows);
  console::field("replay_head_rows", head_rows);
  if (console::json_enabled())
    console::field("payload_sha256", artifact::digest_hex(file.header().payload_sha256));
  if (console::json_enabled())
    console::field("weight_qdq_mask_sha256", (mask.has_source() ? artifact::digest_hex(mask.source_sha256()) : "disabled"));
  const auto load_begin = std::chrono::steady_clock::now();
  WeightArena weights(file, mask);
  console::field("weight_load_seconds", seconds_since(load_begin));

  const auto tile_bytes = std::size_t(head_rows) * kGenerationLogitRowBytes;
  DeviceAllocation raw_logits(tile_bytes), capped_logits(tile_bytes), reference_logits(tile_bytes);
  DeviceAllocation recorded_tokens(head_rows * sizeof(std::uint32_t));
  DeviceAllocation device_metrics(head_rows * sizeof(replay_metrics::Row));
  PinnedHostAllocation host_reference(tile_bytes), host_metrics(head_rows * sizeof(replay_metrics::Row));
  CudaEvent gpu_begin, gpu_end;
  std::ofstream results(output / "results.jsonl");
  results.exceptions(std::ios::failbit | std::ios::badbit);
  results << std::setprecision(17);
  const auto run_begin = std::chrono::steady_clock::now();
  double gpu_seconds_total = 0;
  std::uint64_t scored_total = 0, processed_total = 0, invalid_total = 0;
  for (const auto& request : requests) {
    const auto case_begin = std::chrono::steady_clock::now();
    std::vector<std::uint32_t> input = request.prompt;
    input.insert(input.end(), request.continuation.begin(), request.continuation.end() - 1);
    Executor engine(
        weights, input.size(), 1, false, {}, nullptr, 0, {}, 0, {}, chunk_rows);
    const auto stream = engine.stream();
    std::ifstream reference(request.reference, std::ios::binary);
    if (!reference) fail("replay reference", "cannot open " + request.reference);
    std::ofstream metrics(output / (request.id + ".metrics.tsv"));
    metrics.exceptions(std::ios::failbit | std::ios::badbit);
    metrics << "row\tkl_ref_replay\tkl_replay_ref\treplay_margin\treference_margin\treference_argmax\treplay_argmax\trecorded_token\tflags\n"
            << std::setprecision(17);
    double gpu_seconds = 0;
    std::uint32_t scored = 0, invalid = 0;
    const auto finish_gpu = [&]() {
      check_cuda(cudaEventRecord(gpu_end.get(), stream), "record replay GPU end");
      check_cuda(cudaEventSynchronize(gpu_end.get()), "wait replay GPU work");
      float milliseconds = 0;
      check_cuda(cudaEventElapsedTime(&milliseconds, gpu_begin.get(), gpu_end.get()), "time replay GPU work");
      gpu_seconds += milliseconds / 1000.0;
    };
    for (std::uint32_t base = 0; base < input.size(); base += chunk_rows) {
      const auto rows = static_cast<std::uint32_t>(std::min<std::size_t>(chunk_rows, input.size() - base));
      check_cuda(cudaEventRecord(gpu_begin.get(), stream), "record replay prefill start");
      engine.replay_chunk(input.data() + base, base, rows);
      finish_gpu();
      const auto score_begin = static_cast<std::uint32_t>(std::max<std::size_t>(base, request.prompt.size() - 1));
      for (std::uint32_t position = score_begin; position < base + rows; position += head_rows) {
        const auto count = std::min(head_rows, base + rows - position);
        const auto bytes = std::size_t(count) * kGenerationLogitRowBytes;
        reference.read(static_cast<char*>(host_reference.data()), static_cast<std::streamsize>(bytes));
        if (reference.gcount() != static_cast<std::streamsize>(bytes))
          fail("replay reference", "short reference logit read");
        check_cuda(cudaEventRecord(gpu_begin.get(), stream), "record replay head start");
        check_cuda(cudaMemcpyAsync(reference_logits.data(), host_reference.data(), bytes,
            cudaMemcpyHostToDevice, stream), "copy replay reference logits");
        check_cuda(cudaMemcpyAsync(recorded_tokens.data(), request.continuation.data() + scored,
            count * sizeof(std::uint32_t), cudaMemcpyHostToDevice, stream), "copy replay recorded tokens");
        engine.replay_head(position - base, count, static_cast<BFloat16*>(raw_logits.data()),
                            static_cast<BFloat16*>(capped_logits.data()));
        replay_metrics::compare_rows(static_cast<BFloat16*>(reference_logits.data()),
            static_cast<BFloat16*>(capped_logits.data()), static_cast<std::uint32_t*>(recorded_tokens.data()),
            count, model::kVocabSize, static_cast<replay_metrics::Row*>(device_metrics.data()), stream);
        check_cuda(cudaMemcpyAsync(host_metrics.data(), device_metrics.data(), count * sizeof(replay_metrics::Row),
            cudaMemcpyDeviceToHost, stream), "copy compact replay metrics");
        finish_gpu();
        const auto* values = static_cast<const replay_metrics::Row*>(host_metrics.data());
        for (std::uint32_t row = 0; row < count; ++row) {
          const auto& value = values[row];
          invalid += value.flags != 0;
          metrics << scored + row << '\t' << value.kl_ref_replay << '\t' << value.kl_replay_ref
                  << '\t' << value.replay_margin << '\t' << value.reference_margin
                  << '\t' << value.reference_argmax << '\t' << value.replay_argmax
                  << '\t' << value.recorded_token << '\t' << value.flags << '\n';
        }
        scored += count;
      }
    }
    if (scored != request.continuation.size() || reference.peek() != std::char_traits<char>::eof())
      fail("replay accounting", "prediction rows do not match the complete continuation");
    check_cuda(cudaStreamSynchronize(stream), "finish replay request");
    metrics.close();
    const double wall_seconds = seconds_since(case_begin);
    results << "{\"id\":\"" << request.id << "\",\"prompt_tokens\":" << request.prompt.size()
            << ",\"completion_tokens\":" << scored << ",\"processed_tokens\":" << input.size()
            << ",\"chunk_rows\":" << chunk_rows << ",\"head_rows\":" << head_rows
            << ",\"wall_seconds\":" << wall_seconds << ",\"gpu_seconds\":" << gpu_seconds
            << ",\"invalid_rows\":" << invalid << "}\n" << std::flush;
    scored_total += scored;
    processed_total += input.size();
    invalid_total += invalid;
    gpu_seconds_total += gpu_seconds;
    console::event("replay_progress",
        {{"completed", &request - requests.data() + 1}, {"requests", requests.size()},
         {"id", request.id}, {"rows", scored}, {"wall_seconds", wall_seconds},
         {"invalid_rows", invalid}},
        "Replayed " + request.id + " (" + console::number(&request - requests.data() + 1) +
            "/" + console::number(requests.size()) + "): " + console::number(scored) +
            " rows in " + console::seconds(wall_seconds) + ", " +
            console::number(invalid) + " invalid rows");
  }
  results.close();
  std::ofstream summary(output / "summary.json");
  summary.exceptions(std::ios::failbit | std::ios::badbit);
  summary << std::setprecision(17) << "{\"requests\":" << requests.size()
          << ",\"scored_rows\":" << scored_total << ",\"processed_tokens\":" << processed_total
          << ",\"wall_seconds\":" << seconds_since(run_begin) << ",\"gpu_seconds\":" << gpu_seconds_total
          << ",\"chunk_rows\":" << chunk_rows << ",\"head_rows\":" << head_rows
          << ",\"invalid_rows\":" << invalid_total << "}\n";
  summary.close();
  console::section("Replay results");
  console::field("replay_scored_rows", scored_total);
  console::field("replay_processed_tokens", processed_total);
  console::field("replay_wall_seconds", seconds_since(run_begin));
  console::field("replay_gpu_seconds", gpu_seconds_total);
  console::field("replay_invalid_rows", invalid_total);
  console::field("replay_output_directory", output_directory);
  return invalid_total ? 1 : 0;
}


namespace {


void publish_observability(BatchScheduler& scheduler, http::Server& transport,
    metrics::Snapshot& metrics, const std::string& model, bool include_entries) {
  auto snapshot = scheduler.observability_snapshot(metrics, model, include_entries);
  transport.publish_observability(metrics.render(model) + scheduler.observed_cache_metrics, std::move(snapshot));
}

std::uint32_t operation_record_id(Operation operation) {
  switch (operation) {
    case Operation::generate: return 1;
    case Operation::prefill: return 2;
    case Operation::finish: return 3;
    case Operation::stats: return 4;
  }
  fail("batch record", "invalid operation");
}

void write_batch_result(std::ostream& records, const BatchRequest& request,
                        std::uint32_t mtp_depth) {
  const auto processed = request.cursor;
  const auto decode_tokens = request.completion_tokens > 1 ? request.completion_tokens - 1 : 0;
  const auto rate = [](std::uint64_t tokens, double seconds) -> nlohmann::json {
    return tokens && seconds > 0 ? nlohmann::json(tokens / seconds) : nlohmann::json(nullptr);
  };
  const auto median = mtp_accepted_median(request.mtp_accepted_histogram);
  const bool stopped = request.string_stopped || (request.honor_eos && !request.outputs.empty() &&
                       is_generation_stop_token(request.outputs.back()));
  records << "{\"id\":\"" << request.id << "\",\"prompt_tokens\":" << request.prompt->size()
          << ",\"operation\":" << operation_record_id(request.operation)
          << ",\"input_checkpoint\":" << request.input_checkpoint
          << ",\"completion_checkpoint\":" << request.completion_checkpoint
          << ",\"completion_tokens\":" << request.completion_tokens << ",\"processed_tokens\":" << processed
          << ",\"cached_tokens\":" << request.cached_tokens
          << ",\"shared_tokens\":" << request.shared_tokens
          << ",\"prefill_tokens\":" << request.prefill_tokens
          << ",\"dependency_wait_seconds\":" << request.dependency_wait_seconds
          << ",\"finish_reason\":\"" << (stopped ? "stop" : "length")
          << "\",\"queue_seconds\":" << request.queue_seconds
          << ",\"prefill_seconds\":" << request.prefill_seconds
          << ",\"decode_seconds\":" << request.decode_seconds
          << ",\"prefill_gpu_seconds\":" << request.prefill_gpu_seconds
          << ",\"decode_gpu_seconds\":" << request.decode_gpu_seconds
          << ",\"decode_tokens\":" << decode_tokens
          << ",\"prefill_tokens_per_second\":" << rate(request.prefill_tokens, request.prefill_gpu_seconds).dump()
          << ",\"decode_tokens_per_second\":" << rate(decode_tokens, request.decode_gpu_seconds).dump()
          << ",\"mtp_depth\":" << (request.ordinary_decode ? 0 : mtp_depth)
          << ",\"mtp_cycles\":" << request.mtp_cycles
          << ",\"mtp_proposed\":" << request.mtp_proposed
          << ",\"mtp_accepted\":" << request.mtp_accepted
          << ",\"mtp_rejected_cycles\":" << request.mtp_rejected_cycles
          << ",\"mtp_accepted_median\":" << (median ? nlohmann::json(*median) : nlohmann::json(nullptr)).dump()
          << ",\"mtp_accepted_histogram\":" << nlohmann::json(request.mtp_accepted_histogram).dump()
          << "}\n" << std::flush;
}

void log_batch_result(const char* name, const BatchRequest& request, std::uint32_t mtp_depth) {
  nlohmann::json data;
  if (console::json_enabled()) {
    std::ostringstream record;
    write_batch_result(record, request, mtp_depth);
    data = nlohmann::json::parse(record.str());
  }
  const auto operation = request.operation == Operation::generate ? "Request " :
      request.operation == Operation::prefill ? "Prefill " : "Cache control ";
  std::string summary = operation + request.id +
      (request.phase == BatchPhase::cancelled ? " cancelled" : " completed");
  if (request.prompt && !request.prompt->empty()) {
    summary += " | prompt " + console::number(request.prompt->size()) +
        ", cached " + console::number(request.cached_tokens + request.shared_tokens);
    if (request.operation == Operation::generate)
      summary += ", output " + console::number(request.completion_tokens);
    summary += " tokens | queue " + console::seconds(request.queue_seconds) +
        ", prefill " + console::token_rate(request.prefill_tokens, request.prefill_gpu_seconds) +
        " (GPU " + console::seconds(request.prefill_gpu_seconds) + "), decode " +
        console::token_rate(request.completion_tokens > 1 ? request.completion_tokens - 1 : 0,
                            request.decode_gpu_seconds) +
        " (GPU " + console::seconds(request.decode_gpu_seconds) + ")";
    if (request.mtp_proposed) {
      summary += " | MTP accepted " + console::number(request.mtp_accepted) +
          "/" + console::number(request.mtp_proposed);
      if (const auto median = mtp_accepted_median(request.mtp_accepted_histogram)) {
        std::ostringstream value;
        value << *median;
        summary += ", median " + value.str() + "/step";
      }
    }
  }
  console::event(name, data, summary);
}

}  // namespace

int run_generate_batch(const std::string& artifact_path, const std::string& requests_path,
                       std::uint32_t max_batch, std::uint64_t kv_cache_gpu_mib,
                       const std::string& output_directory, const std::string& qdq_mask_path,
                       std::uint32_t mtp_depth, const std::string& events_path,
                       std::uint32_t prefill_chunk_tokens, std::uint32_t prefill_batch_tokens,
                       gewell::nvfp4::ActivationPolicy activation_policy,
                       kv_cache::Format local_kv_format, kv_cache::Format global_kv_format,
                       attention::Compute local_attention_compute, attention::Compute global_attention_compute,
                       const std::string& assistant_path, const std::string& vision_path,
                       std::uint32_t prefill_budget_tokens, std::uint32_t mtp_min_depth,
                       std::uint32_t decode_width, const std::string& mtp_stats_path,
                       std::uint32_t mtp_stats_window, const MtpCaptureSettings& mtp_capture) {
  mtp_depth = effective_mtp_depth(mtp_depth, assistant_path, mtp_min_depth);
  if (!mtp_depth) mtp_min_depth = 0;
  if (!max_batch || max_batch > kMaxBatchRows || !kv_cache_gpu_mib ||
      kv_cache_gpu_mib > std::numeric_limits<std::size_t>::max() / kv_cache::kMib)
    fail("batch configuration", "invalid batch capacity or GPU KV budget");
  auto requests = read_batch_requests(requests_path);
  auto events = read_batch_events(events_path, requests);
  BatchLimits limits;
  limits.capacity = std::min<std::uint32_t>(max_batch, requests.size());
  limits.kv_bytes = kv_cache_gpu_mib * kv_cache::kMib;
  limits.mtp_depth = mtp_depth;
  limits.mtp_min_depth = mtp_min_depth;
  limits.decode_width = decode_width;
  limits.mtp_stats_path = mtp_stats_path;
  limits.mtp_stats_window = mtp_stats_window;
  limits.mtp_capture = mtp_capture;
  limits.prefill_chunk_tokens = checked_prefill_chunk_tokens(prefill_chunk_tokens);
  limits.prefill_batch_tokens = checked_prefill_batch_tokens(prefill_batch_tokens, limits.prefill_chunk_tokens);
  limits.prefill_budget_tokens = prefill_budget_tokens;
  limits.local_attention_compute = local_attention_compute;
  limits.global_attention_compute = global_attention_compute;
  limits.local_kv_format = local_kv_format;
  limits.global_kv_format = global_kv_format;
  limits.plan_rows = 1;
  limits.max_horizon = 1;
  for (const auto& request : requests) {
    limits.max_horizon = std::max(limits.max_horizon,
        generation_cache_capacity(request.prompt->size(), request.max_new_tokens));
    limits.plan_rows = std::max(limits.plan_rows, static_cast<std::uint32_t>(
        std::min<std::size_t>(request.prompt->size(), limits.prefill_chunk_tokens)));
    limits.sampled |= request.sampling.temperature > 0 && request.sampling.top_p > 0 && request.sampling.top_k != 1;
    limits.captures |= request.capture_logits;
  }
  if (mtp_depth > mtp_target::kMaxDepth ||
      (mtp_depth && std::uint64_t(limits.capacity) * (mtp_depth + 1) > mtp_target::kMaxDepth + 1))
    fail("batch MTP", "batch size times (depth + 1) must not exceed 1280 verifier rows");
  const auto staging_bytes = mtp_depth
      ? limits.capacity * mtp_target::Verifier::staging_bytes(mtp_depth + 1) : 0;
  const auto config = sm120::compact_pool_config(limits.kv_bytes, 0, limits.index_bytes, limits.local_kv_format, limits.global_kv_format);
  const auto hidden_bytes = (limits.kv_bytes / config.local_ring_bytes) * model::kHiddenSize * sizeof(BFloat16);
  if (!hidden_bytes || staging_bytes >= limits.kv_bytes || hidden_bytes >= limits.kv_bytes - staging_bytes)
    fail("batch configuration", "staging exceeds GPU KV budget");
  const auto pages = (std::size_t(limits.max_horizon) + config.global_page_tokens - 1) / config.global_page_tokens;
  if (config.local_ring_bytes + config.page_table_bytes + pages * config.global_page_bytes >
      limits.kv_bytes - staging_bytes - hidden_bytes)
    fail("batch admission", "request exceeds configured GPU KV capacity");
  if (!std::filesystem::create_directory(output_directory))
    fail("batch output directory", "directory must be new");
  const std::filesystem::path output(output_directory);
  BatchSignalScope signals;
  artifact::ArtifactFile file = artifact::ArtifactFile::Open(artifact_path);
  const auto mask = qdq_mask_path.empty() ? qdq::Mask{} : qdq::Mask::Load(qdq_mask_path);
  validate_cuda_device();
  console::section("Batch generation");
  if (console::json_enabled())
    console::field("batch_mode", "offline_compact_global");
  console::field("batch_requests", requests.size());
  console::field("batch_capacity", limits.capacity);
  console::field("batch_mtp_depth", mtp_depth);
  console::field("batch_mtp_min_depth", mtp_min_depth);
  console::field("batch_decode_width", decode_width);
  console::field("batch_mtp_staging_bytes", staging_bytes);
  if (console::json_enabled())
    console::field("payload_sha256", artifact::digest_hex(file.header().payload_sha256));
  if (console::json_enabled())
    console::field("weight_qdq_mask_sha256", (mask.has_source() ? artifact::digest_hex(mask.source_sha256()) : "disabled"));
  const auto load_begin = std::chrono::steady_clock::now();
  WeightArena weights(file, mask, mtp_depth ? assistant_path : "", vision_path);
  console::field("weight_load_seconds", seconds_since(load_begin));
  BatchLogitWriter writer;
  struct Capture { std::unique_ptr<std::ofstream> output; std::future<void> pending; };
  std::map<std::string, Capture> captures;
  std::ofstream records(output / "results.jsonl"), cancellations(output / "cancellations.jsonl"), trace;
  if (!records || !cancellations) fail("batch output", "cannot create records");
  records.precision(10);
  if (!events_path.empty()) trace.open(output / "prefix_work.jsonl");
  BatchScheduler* current = nullptr;
  BatchCallbacks callbacks;
  callbacks.ready = [&](auto& request) {
    if (!captures[request.id].pending.valid()) return true;
    if (captures[request.id].pending.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
    captures[request.id].pending.get();
    return true;
  };
  callbacks.drain = [&](auto& request) {
    auto& pending = captures[request.id].pending;
    if (pending.valid()) pending.get();
  };
  callbacks.start = [&](auto& request) {
    if (request.capture_logits) {
      captures[request.id].output = std::make_unique<std::ofstream>(output / (request.id + ".bf16"), std::ios::binary);
      if (!*captures[request.id].output) fail("batch logits", "cannot create " + request.id);
    }
  };
  callbacks.emit = [&](auto& request, const std::uint32_t*, std::size_t count,
                       const void* logits, const TokenLogprobs*) {
    if (request.capture_logits) {
      const auto* bytes = static_cast<const std::uint8_t*>(logits);
      captures[request.id].pending = writer.write(*captures[request.id].output,
          std::vector<std::uint8_t>(bytes, bytes + count * kGenerationLogitRowBytes));
    }
  };
  callbacks.finish = [&](auto& request) {
    if (captures[request.id].output) {
      captures[request.id].output->close();
      if (!*captures[request.id].output) fail("batch logits", "close failed");
    }
    if (request.phase == BatchPhase::cancelled) {
      cancellations << "{\"id\":\"" << request.id << "\",\"prompt_tokens\":" << request.prompt->size()
                    << ",\"completion_tokens\":" << request.completion_tokens
                    << ",\"processed_tokens\":" << request.cursor
                    << ",\"cached_tokens\":" << request.cached_tokens
                    << ",\"shared_tokens\":" << request.shared_tokens
                    << ",\"prefill_tokens\":" << request.prefill_tokens
                    << ",\"dependency_wait_seconds\":" << request.dependency_wait_seconds << "}\n";
    } else {
      write_generation_output((output / (request.id + ".u32")).string(), request.outputs);
      write_batch_result(records, request, mtp_depth);
    }
    if (!records || !cancellations) fail("batch records", "write failed");
  };
  callbacks.trace = [&](const auto& event, const auto& fields) {
    if (trace.is_open()) {
      trace << "{\"event\":\"" << event << "\"," << fields << "}\n" << std::flush;
      if (!trace) fail("batch trace", "write failed");
    }
  };
  callbacks.poll = [&](bool inflight) {
    for (auto& event : events) {
      if (event.applied || event.inflight != inflight || event.after > current->prefill_tokens) continue;
      event.applied = true;
      if (event.action == "arrive") {
        if (current->requests[event.request].phase != BatchPhase::withheld)
          fail("batch events", "arrival targets request no longer withheld");
        current->requests[event.request].phase = BatchPhase::queued;
      } else if (event.action == "cancel") current->cancel(event.request, inflight);
      else fail("batch compute", "injected failure; partial state cannot be handed off");
    }
  };
  BatchScheduler scheduler(sm120::make_runtime_backend(weights, limits, activation_policy), limits, std::move(callbacks));
  current = &scheduler;
  for (auto& request : requests) scheduler.submit(std::move(request));
  scheduler.register_requests();
  while (scheduler.has_pending()) {
    if (batch_interrupted) fail("batch generation", "interrupted; incomplete requests are not published as successes");
    if (!scheduler.step()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  std::ofstream summary(output / "summary.json");
  summary.precision(10);
  scheduler.write_summary(summary);
  summary.close();
  if (!summary) fail("batch summary", "write failed");
  console::section("Batch results");
  console::field("batch_wall_seconds", scheduler.wall_seconds());
  console::field("batch_decode_tokens_per_second", (scheduler.decode_gpu_seconds ? scheduler.decode_tokens / scheduler.decode_gpu_seconds : 0));
  console::field("batch_completed_requests", scheduler.completed);
  console::field("batch_output_tokens", scheduler.outputs);
  console::field("batch_output_directory", output_directory);
  return 0;
}

BatchLimits live_batch_limits(std::uint32_t max_batch, const RuntimeSettings& settings,
                              std::size_t max_connections) {
  const auto kv_cache_gpu_mib = settings.kv_cache_gpu_mib;
  const auto mtp_depth = settings.mtp_depth;
  if (!max_connections || max_connections > 256 || !max_batch || max_batch > std::min<std::size_t>(max_connections, kMaxBatchRows) ||
      !kv_cache_gpu_mib || kv_cache_gpu_mib > std::numeric_limits<std::size_t>::max() / kv_cache::kMib ||
      mtp_depth > mtp_target::kMaxDepth || settings.mtp_min_depth > mtp_depth ||
      (mtp_depth && std::uint64_t(max_batch) * (mtp_depth + 1) > mtp_target::kMaxDepth + 1))
    fail("batch server", "invalid batch, depth, or GPU budget");
  BatchLimits limits;
  limits.capacity = max_batch;
  limits.mtp_depth = mtp_depth;
  limits.mtp_min_depth = settings.mtp_min_depth;
  limits.decode_width = settings.decode_width;
  limits.mtp_stats_path = settings.mtp_stats_path;
  limits.mtp_stats_window = settings.mtp_stats_window;
  limits.mtp_capture = settings.mtp_capture;
  limits.prefill_chunk_tokens = checked_prefill_chunk_tokens(settings.prefill_chunk_tokens);
  limits.prefill_batch_tokens = checked_prefill_batch_tokens(settings.prefill_batch_tokens, limits.prefill_chunk_tokens);
  limits.prefill_budget_tokens = settings.prefill_budget_tokens;
  limits.local_attention_compute = settings.local_attention_compute;
  limits.global_attention_compute = settings.global_attention_compute;
  limits.local_kv_format = settings.local_kv_format;
  limits.global_kv_format = settings.global_kv_format;
  limits.kv_bytes = kv_cache_gpu_mib * kv_cache::kMib;
  if (settings.kv_cache_cpu_mib > std::numeric_limits<std::size_t>::max() / kv_cache::kMib ||
      !settings.kv_cache_index_mib || settings.kv_cache_index_mib > std::numeric_limits<std::size_t>::max() / kv_cache::kMib)
    fail("batch server", "invalid CPU or index budget");
  limits.cpu_bytes = settings.kv_cache_cpu_mib * kv_cache::kMib;
  limits.index_bytes = settings.kv_cache_index_mib * kv_cache::kMib;
  limits.checkpoint_interval = settings.kv_checkpoint_interval_tokens <= primitives::kMaxContextTokenCount
      ? settings.kv_checkpoint_interval_tokens : 0;
  limits.max_requests = max_connections;
  limits.sampled = limits.captures = limits.live = true;
  const auto config = sm120::compact_pool_config(limits.kv_bytes, 0, limits.index_bytes, limits.local_kv_format, limits.global_kv_format);
  const auto staging = mtp_depth ? max_batch * mtp_target::Verifier::staging_bytes(mtp_depth + 1) : 0;
  const auto hidden = (limits.kv_bytes / config.local_ring_bytes) * model::kHiddenSize * sizeof(BFloat16);
  if (staging >= limits.kv_bytes || hidden >= limits.kv_bytes - staging ||
      limits.kv_bytes - staging - hidden < config.local_ring_bytes + config.page_table_bytes + config.global_page_bytes)
    fail("batch server", "GPU budget cannot hold one execution and staging");
  const auto pages = (limits.kv_bytes - staging - hidden - config.local_ring_bytes - config.page_table_bytes) /
      config.global_page_bytes;
  limits.max_horizon = static_cast<std::uint32_t>(std::min<std::size_t>(
      primitives::kMaxContextTokenCount, pages * config.global_page_tokens));
  limits.plan_rows = std::min(limits.prefill_chunk_tokens, limits.max_horizon);
  return limits;
}

int run_http_server(const std::string& model_directory, std::uint32_t max_batch,
                    RuntimeSettings settings, const http::Settings& http_settings,
                    const std::string& qdq_mask_path) {
  settings.mtp_depth = effective_mtp_depth(settings.mtp_depth, settings.assistant_path, settings.mtp_min_depth);
  if (!settings.mtp_depth) settings.mtp_min_depth = 0;
  auto limits = live_batch_limits(max_batch, settings, http_settings.max_connections);
  limits.captures = false;
  limits.logprobs = true;
  console::section("GeWell - HTTP inference server");
  console::field("server_model", http_settings.model);
  BatchSignalScope signals;
  auto assets = gemma4_31b::ServingAssets::Open(model_directory);
  http::ImageSupport images;
  if (!settings.vision_path.empty()) {
    images.begin_token = model::kBeginImageTokenId;
    images.image_token = model::kImageTokenId;
    images.end_token = model::kEndImageTokenId;
    images.max_image_tokens = model::kVisionMaxSoftTokenCount;
    images.default_max_soft_tokens = settings.image_max_soft_tokens;
    for (const auto budget : vision_engine::kSupportedSoftTokenCapacities) {
      const auto image_rows = vision_engine::padded_patch_rows_for_capacity(budget);
      images.prepared_bytes.emplace(budget, vision_engine::prepared_pixel_bytes(image_rows) +
          vision_engine::prepared_position_bytes(image_rows));
    }
    images.prepare = [](std::string_view url, std::uint32_t max_soft_tokens) {
      auto prepared = gemma4::prepare_image_data_url(url, max_soft_tokens);
      auto image = std::make_shared<runtime::ImageInput>();
      image->pixels = std::move(prepared.pixels);
      image->positions = std::move(prepared.positions);
      image->padded_patch_rows = prepared.padded_patch_rows;
      image->end = prepared.soft_token_count;
      return image;
    };
  }
  http::Server transport(assets.tokenizer, http_settings, settings.mtp_depth + 1, std::move(images));
  const auto mask = qdq_mask_path.empty() ? qdq::Mask{} : qdq::Mask::Load(qdq_mask_path);
  validate_cuda_device();
  const auto load_begin = std::chrono::steady_clock::now();
  WeightArena weights(assets.weights, mask, settings.mtp_depth ? settings.assistant_path : "", settings.vision_path);
  console::field("weight_load_seconds", seconds_since(load_begin));
  BatchScheduler* current = nullptr;
  std::map<http::ClientId, std::size_t> clients;
  std::vector<std::size_t> retired;
  metrics::Snapshot metrics;
  metrics.mtp_depth = settings.mtp_depth;
  metrics.mtp_accepted_per_position.resize(settings.mtp_depth);
  BatchCallbacks callbacks;
  callbacks.metrics = &metrics;
  callbacks.ready = [&](auto& request) { return transport.ready(request.client); };
  callbacks.start = [](auto&) {};
  callbacks.emit = [&](auto& request, const auto* tokens, std::size_t count,
                       const void*, const TokenLogprobs* logprobs) {
    transport.emit(request.client, tokens, count, logprobs);
  };
  callbacks.finish = [&](auto& request) {
    current->observe_finished(request);
    http::Result result;
    result.prompt_tokens = static_cast<std::uint32_t>(request.prompt->size());
    result.completion_tokens = static_cast<std::uint32_t>(request.completion_tokens);
    result.cached_tokens = request.cached_tokens + request.shared_tokens;
    result.processed_tokens = request.cursor;
    result.input_checkpoint = request.input_checkpoint;
    result.completion_checkpoint = request.completion_checkpoint;
    result.stopped = request.string_stopped ||
        (!request.outputs.empty() && is_generation_stop_token(request.outputs.back()));
    result.cache_stats = request.control_stats;
    transport.finish(request.client, result);
  };
  callbacks.done = [&](auto& request) {
    if (request.phase == BatchPhase::cancelled) current->observe_finished(request);
    log_batch_result(request.phase == BatchPhase::cancelled
                     ? "server_http_cancelled" : "server_http_result", request, settings.mtp_depth);
    retired.push_back(clients.at(request.client));
  };
  callbacks.reject = [&](auto& request, const auto& message, BatchFailure reason) {
    current->observe_finished(request, true);
    transport.reject(request.client, http::Error(
        reason == BatchFailure::capacity ? 503 : 500, message, {},
        reason == BatchFailure::capacity ? "capacity_exceeded" : "execution_failed"));
  };
  callbacks.trace = [](const auto& event, const auto& fields) {
    if (console::json_enabled())
      console::event("server_http_event", nlohmann::json::parse(
          "{\"event\":\"" + std::string(event) + "\"," + fields + "}"));
  };
  callbacks.poll = [&](bool inflight) {
    for (auto& admission : transport.take_admissions()) {
      if (batch_interrupted) {
        transport.reject(admission.client, http::Error(503, "server shutting down", {}, "server_unavailable"));
        continue;
      }
      const auto& input = admission.request;
      if ((input.operation == http::Operation::generate || input.operation == http::Operation::prefill) &&
          generation_cache_capacity(input.prompt->size(), std::max(1u, input.max_tokens)) > limits.max_horizon) {
        if (input.operation == http::Operation::generate) ++metrics.failed;
        transport.reject(admission.client, http::Error(400,
            "prompt and output budget exceed the configured GPU KV capacity of " +
                std::to_string(limits.max_horizon) + " tokens",
            input.chat ? "messages" : "prompt", "context_length_exceeded"));
        continue;
      }
      BatchRequest request;
      request.client = admission.client;
      request.metrics_arrival = input.arrival_time;
      request.id = std::to_string(admission.client);
      request.prompt = input.prompt ? input.prompt : std::make_shared<const std::vector<std::uint32_t>>();
      request.images = input.images;
      switch (input.operation) {
        case http::Operation::generate: request.operation = Operation::generate; break;
        case http::Operation::prefill: request.operation = Operation::prefill; break;
        case http::Operation::finish: request.operation = Operation::finish; break;
        case http::Operation::stats: request.operation = Operation::stats; break;
        default: fail("HTTP admission", "unexpected scheduler operation");
      }
      request.controls.mode = input.cache.reuse_only ? CacheRequestMode::reuse_only : CacheRequestMode::automatic;
      request.controls.prompt_id = input.cache.prompt_id;
      request.controls.priority = input.cache.priority == http::Priority::low ? prefix_index::RetentionPriority::low :
          input.cache.priority == http::Priority::high ? prefix_index::RetentionPriority::high : prefix_index::RetentionPriority::normal;
      request.controls.finished = input.cache.finished;
      request.max_new_tokens = request.operation == Operation::prefill ? 1 : input.max_tokens;
      request.sampling = {input.temperature, input.top_p, input.top_k};
      request.honor_eos = true;
      request.logprobs = input.logprobs;
      request.top_logprobs = input.top_logprobs;
      request.ordinary_decode = !input.stops.empty();
      request.seed = input.seed ? static_cast<std::uint64_t>(*input.seed) : std::random_device{}();
      request.rng.seed(request.seed);
      try {
        if (input.constraint)
          request.constraint = std::make_unique<constraint::State>(
              input.constraint, input.enable_thinking, input.initial_reasoning);
        clients.emplace(admission.client, current->submit(std::move(request)));
      } catch (const std::exception& error) {
        if (input.operation == http::Operation::generate) ++metrics.failed;
        const bool capacity = batch_failure(error) == BatchFailure::capacity;
        transport.reject(admission.client, http::Error(capacity ? 503 : 500,
            error.what(), {}, capacity ? "capacity_exceeded" : "execution_failed"));
      }
    }
    for (const auto client : transport.take_stops()) {
      const auto found = clients.find(client);
      if (found != clients.end()) current->stop_at_output(found->second);
    }
    for (const auto client : transport.take_completions()) {
      const auto found = clients.find(client);
      if (found != clients.end()) current->requests[found->second].delivered = true;
    }
    for (const auto client : transport.take_cancellations()) {
      const auto found = clients.find(client);
      if (found != clients.end()) current->cancel(found->second, inflight);
    }
  };
  BatchScheduler scheduler(sm120::make_runtime_backend(weights, limits, settings.nvfp4_activation_policy), limits, std::move(callbacks));
  current = &scheduler;
  console::section("Serving");
  console::field("server_batch_capacity", max_batch);
  console::field("server_mtp_depth", settings.mtp_depth);
  console::field("server_mtp_min_depth", settings.mtp_min_depth);
  console::field("server_decode_width", settings.decode_width);
  console::field("server_prefill_chunk_tokens", settings.prefill_chunk_tokens);
  console::field("server_prefill_batch_tokens", settings.prefill_batch_tokens);
  console::field("server_prefill_budget_tokens", settings.prefill_budget_tokens);
  scheduler.write_startup_capacity();
  publish_observability(scheduler, transport, metrics, http_settings.model, true);
  auto metrics_published = std::chrono::steady_clock::now();
  auto inventory_published = metrics_published;
  transport.set_ready(true);
  console::section("HTTP limits");
  console::field("server_max_connections", http_settings.max_connections);
  console::field("server_max_body_bytes", http_settings.max_body_bytes);
  console::field("server_max_body_total_bytes", http_settings.max_body_total_bytes);
  console::field("server_max_output_bytes", http_settings.max_output_bytes);
  console::field("server_socket_timeout_seconds", http_settings.socket_timeout_seconds);
  console::field("server_image_limit", 1);
  console::field("server_image_prompt_tokens", kMultimodalChunkTokens);
  console::field("server_image_max_soft_tokens", settings.image_max_soft_tokens);
  console::field("server_image_admission", "packed_prefill");
  console::field("server_image_prefix_cache", true);
  if (console::json_enabled()) {
    console::field("server_mode", "native_http");
    console::field("server_bind", http_settings.host + ":" + std::to_string(http_settings.port));
    console::field("server_max_context_tokens", limits.max_horizon);
    console::field("server_kv_cache_gpu_mib", settings.kv_cache_gpu_mib);
    console::field("server_kv_cache_cpu_mib", settings.kv_cache_cpu_mib);
    console::field("server_kv_cache_index_mib", settings.kv_cache_index_mib);
    console::field("payload_sha256", artifact::digest_hex(assets.weights.header().payload_sha256));
    console::field("server_ready", true);
  } else {
    console::message("\nReady - http://" + http_settings.host + ":" + std::to_string(http_settings.port) +
        "\n  Metrics: /metrics    Cache inventory: /v1/cache/index");
  }
  try {
    while (!batch_interrupted) {
      if (!transport.healthy()) fail("HTTP transport", "network worker failed");
      const bool progressed = scheduler.step();
      for (const auto slot : retired) {
        clients.erase(scheduler.requests[slot].client);
        scheduler.forget(slot);
      }
      if (!retired.empty()) {
        retired.clear();
        if (console::json_enabled()) {
          std::ostringstream stats;
          scheduler.write_live_stats(stats);
          console::event("server_http_stats", nlohmann::json::parse(stats.str()));
        }
      }
      const auto now = std::chrono::steady_clock::now();
      if (now - metrics_published >= std::chrono::milliseconds(250)) {
        const bool inventory = now - inventory_published >= std::chrono::seconds(1);
        publish_observability(scheduler, transport, metrics, http_settings.model, inventory);
        metrics_published = now;
        if (inventory) inventory_published = now;
      }
      if (!progressed) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } catch (const std::exception& error) {
    for (auto& request : scheduler.requests)
      if (request.phase != BatchPhase::vacant && !request.metrics_finished)
        scheduler.observe_finished(request, request.phase != BatchPhase::cancelled);
    // Preserve the last consistent cache inventory if execution failed mid-step.
    metrics.running = metrics.waiting = 0;
    transport.publish_observability(metrics.render(http_settings.model) + scheduler.observed_cache_metrics, nullptr);
    transport.fail_all(http::Error(500, error.what(), {}, "execution_failed"));
    // Give responsive clients a bounded opportunity to receive the error.
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    throw;
  }
  transport.set_ready(false);
  transport.stop();
  for (const auto& item : clients) scheduler.cancel(item.second);
  scheduler.step();
  return 0;
}

}  // namespace gewell::app
