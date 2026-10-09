#include "gewell/offline_runner.h"
#include "gewell/app.h"
#include "gewell/console.h"
#include "gewell/models/gemma4/31b/artifact.h"
#include "gewell/models/gemma4/text_contract.h"
#include "gewell/vision_engine.h"
#include "gewell/weight_qdq.h"
#include "models/gemma4/31b/sm120/runtime_backend.h"
#include "models/gemma4/26b_a4b/sm120/runtime_backend.h"
#include "models/gemma4/31b/sm120/weights.cuh"
#include "models/gemma4/31b/sm120/runner_support.cuh"
#include "offline_io.h"
#include "runner_internal.h"

#include <algorithm>
#include <csignal>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <thread>
#include <utility>
#include <unistd.h>

namespace gewell::app {
namespace {
using Json = nlohmann::json;
using namespace runtime;
namespace model = gemma4_31b;
namespace sm120 = gemma4_31b::sm120;
namespace qdq = gewell::weight_qdq;
constexpr std::size_t kMaxPreparedImageBytes = 256ULL * 1024 * 1024;
volatile std::sig_atomic_t interrupted = 0;
void stop_jobs(int) { interrupted = 1; }

struct Signals {
  struct sigaction interrupt{}, terminate{}, pipe{};
  Signals() {
    interrupted = 0;
    struct sigaction action{};
    action.sa_handler = stop_jobs;
    sigemptyset(&action.sa_mask);
    sigaction(SIGINT, &action, &interrupt);
    sigaction(SIGTERM, &action, &terminate);
    action.sa_handler = SIG_IGN;
    sigaction(SIGPIPE, &action, &pipe);
  }
  ~Signals() {
    sigaction(SIGINT, &interrupt, nullptr);
    sigaction(SIGTERM, &terminate, nullptr);
    sigaction(SIGPIPE, &pipe, nullptr);
  }
};

// Scheduler/backend diagnostics remain console output; job events have a
// separate serializer and descriptor so console formatting cannot corrupt them.
struct DiagnosticsToStderr {
  std::streambuf* saved{std::cout.rdbuf(std::cerr.rdbuf())};
  ~DiagnosticsToStderr() { std::cout.rdbuf(saved); }
};

std::uint64_t integer(const Json& object, const char* key, std::uint64_t fallback,
                      std::uint64_t maximum) {
  if (!object.contains(key)) return fallback;
  const auto& value = object.at(key);
  if (!value.is_number_integer() || (value.is_number_integer() && !value.is_number_unsigned() &&
      value.get<std::int64_t>() < 0) || value.get<std::uint64_t>() > maximum)
    throw std::invalid_argument(std::string(key) + " is outside its integer range");
  return value.get<std::uint64_t>();
}

std::string string(const Json& value, const char* key, bool required = false) {
  if (!value.contains(key) && !required) return {};
  const auto text = value.at(key).get<std::string>();
  if (text.empty() || text.find('\0') != std::string::npos)
    throw std::invalid_argument(std::string(key) + " must be nonempty and contain no NUL");
  return text;
}

void keys(const Json& object, std::initializer_list<const char*> allowed) {
  if (!object.is_object()) throw std::invalid_argument("job and nested settings must be JSON objects");
  for (const auto& item : object.items())
    if (std::find(allowed.begin(), allowed.end(), item.key()) == allowed.end())
      throw std::invalid_argument("unknown job field: " + item.key());
}

std::vector<std::uint8_t> read_bytes(const std::string& path, std::size_t exact) {
  if (!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path) != exact)
    throw std::invalid_argument("input must be a regular file of the expected size: " + path);
  std::ifstream input(path, std::ios::binary);
  std::vector<std::uint8_t> bytes(exact);
  input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (!input || input.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("input changed or could not be read: " + path);
  return bytes;
}

struct Job {
  std::size_t slot{};
  std::uint64_t result_event{};
  std::unique_ptr<offline::Output> tokens, logits;
  std::vector<std::pair<std::string, std::string>> image_paths;
  std::size_t declared_image_bytes{};
  bool failed{}, blocked{}, inputs_loaded{};
  void discard() {
    if (tokens) tokens->discard();
    if (logits) logits->discard();
  }
};

BatchRequest parse_request(const Json& command, Job& job, std::uint32_t max_context, const text::TextContract& contract) {
  keys(command, {"op", "id", "prompt_path", "max_tokens", "temperature", "top_p", "top_k",
      "seed", "honor_eos", "cache", "checkpoint_offsets", "prepared_images", "outputs", "prompt_id"});
  BatchRequest request;
  request.id = string(command, "id", true);
  if (request.id.size() > 256) throw std::invalid_argument("id exceeds 256 bytes");
  const auto operation = string(command, "op", true);
  if (operation == "generate") request.operation = Operation::generate;
  else if (operation == "prefill") request.operation = Operation::prefill;
  else if (operation == "finish") request.operation = Operation::finish;
  else if (operation == "stats") request.operation = Operation::stats;
  else throw std::invalid_argument("op must be generate, prefill, finish, stats, or cancel");
  request.prompt = std::make_shared<std::vector<std::uint32_t>>();
  if (!BatchScheduler::computes(request)) {
    keys(command, request.operation == Operation::finish
        ? std::initializer_list<const char*>{"op", "id", "prompt_id"}
        : std::initializer_list<const char*>{"op", "id"});
    if (request.operation == Operation::finish) request.controls.prompt_id = string(command, "prompt_id", true);
    return request;
  }
  if (command.contains("prompt_id")) throw std::invalid_argument("generation prompt_id belongs inside cache");
  const auto path = string(command, "prompt_path", true);
  if (!std::filesystem::is_regular_file(path)) throw std::invalid_argument("prompt_path must name a regular file");
  const auto bytes = std::filesystem::file_size(path);
  if (!bytes || bytes % 4 || bytes / 4 > max_context)
    throw std::invalid_argument("prompt must contain 1..context_tokens little-endian u32 token IDs");
  const auto input = read_bytes(path, bytes);
  auto prompt = std::make_shared<std::vector<std::uint32_t>>(bytes / 4);
  for (std::size_t i = 0; i < prompt->size(); ++i) {
    const auto* p = input.data() + 4 * i;
    (*prompt)[i] = std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) |
        (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
    if ((*prompt)[i] >= contract.vocabulary_size)
      throw std::invalid_argument("prompt token is outside the vocabulary");
  }
  request.prompt = prompt;
  request.max_new_tokens = request.operation == Operation::prefill ? 1 :
      integer(command, "max_tokens", 0, max_context);
  if (!request.max_new_tokens) throw std::invalid_argument("generate requires positive max_tokens");
  (void)generation_cache_capacity(prompt->size(), request.max_new_tokens, max_context);
  request.sampling.temperature = command.value("temperature", 0.0F);
  request.sampling.top_p = command.value("top_p", 1.0F);
  request.sampling.top_k = integer(command, "top_k", 0, contract.vocabulary_size);
  if (!std::isfinite(request.sampling.temperature) || request.sampling.temperature < 0 ||
      !std::isfinite(request.sampling.top_p) || request.sampling.top_p < 0 || request.sampling.top_p > 1)
    throw std::invalid_argument("temperature must be nonnegative and top_p must be in [0,1]");
  request.seed = integer(command, "seed", std::random_device{}(), UINT64_MAX);
  request.rng.seed(request.seed);
  request.honor_eos = command.value("honor_eos", true);
  if (command.contains("cache")) {
    const auto& cache = command.at("cache");
    keys(cache, {"mode", "prompt_id", "priority", "finished"});
    const auto mode = cache.value("mode", std::string("auto"));
    if (mode != "auto" && mode != "reuse_only") throw std::invalid_argument("invalid cache.mode");
    request.controls.mode = mode == "auto" ? CacheRequestMode::automatic : CacheRequestMode::reuse_only;
    request.controls.prompt_id = string(cache, "prompt_id");
    const auto priority = cache.value("priority", std::string("normal"));
    if (priority != "low" && priority != "normal" && priority != "high")
      throw std::invalid_argument("cache.priority must be low, normal, or high");
    request.controls.priority = priority == "low" ? prefix_index::RetentionPriority::low :
        priority == "high" ? prefix_index::RetentionPriority::high : prefix_index::RetentionPriority::normal;
    request.controls.finished = cache.value("finished", false);
    if ((request.controls.finished && !request.controls.named()) ||
        (mode == "reuse_only" && (request.controls.named() || request.controls.finished || priority != "normal")))
      throw std::invalid_argument("invalid cache owner controls");
  }
  if (request.operation == Operation::prefill &&
      (request.controls.finished || request.controls.mode != CacheRequestMode::automatic))
    throw std::invalid_argument("prefill requires automatic cache admission and finished=false");
  if (command.contains("checkpoint_offsets")) {
    const auto& offsets = command.at("checkpoint_offsets");
    if (!offsets.is_array() || offsets.size() > 4096)
      throw std::invalid_argument("checkpoint_offsets must contain at most 4096 offsets");
    for (const auto& offset : offsets) {
      const auto value = integer(Json{{"offset", offset}}, "offset", 0, prompt->size());
      // Offset zero names the empty prefix and needs no physical checkpoint.
      if (!value) continue;
      if (!request.checkpoint_offsets.empty() && value <= request.checkpoint_offsets.back())
        throw std::invalid_argument("nonzero checkpoint_offsets must increase strictly within the prompt");
      request.checkpoint_offsets.push_back(value);
    }
  }
  const auto prepared = command.value("prepared_images", Json::array());
  if (!prepared.is_array() || (command.contains("prepared_images") && prepared.empty()))
    throw std::invalid_argument("prepared_images must be a nonempty array");
  for (std::size_t offset = 0; offset < prompt->size(); ++offset) {
    const auto token = (*prompt)[offset];
    if (token == model::kBeginImageTokenId) {
      const auto begin = ++offset;
      while (offset < prompt->size() && (*prompt)[offset] == model::kImageTokenId) ++offset;
      if (offset == begin || offset == prompt->size() || (*prompt)[offset] != model::kEndImageTokenId)
        throw std::invalid_argument("prepared images require bracketed contiguous image placeholder spans");
      if (request.images.size() == prepared.size())
        throw std::invalid_argument("image spans and prepared_images must have matching counts");
      const auto& entry = prepared.at(request.images.size());
      keys(entry, {"pixel_values", "position_ids", "max_soft_tokens"});
      const auto capacity = integer(entry, "max_soft_tokens", 0, model::kVisionMaxSoftTokenCount);
      if (!vision_engine::is_supported_soft_token_capacity(capacity) || offset - begin > capacity)
        throw std::invalid_argument("image span exceeds its supported prepared-image soft-token capacity");
      auto image = std::make_shared<ImageInput>();
      image->begin = begin;
      image->end = offset;
      image->padded_patch_rows = vision_engine::padded_patch_rows_for_capacity(capacity);
      job.declared_image_bytes += vision_engine::prepared_pixel_bytes(image->padded_patch_rows) +
          vision_engine::prepared_position_bytes(image->padded_patch_rows);
      if (job.declared_image_bytes > kMaxPreparedImageBytes)
        throw std::invalid_argument("prepared image tensors exceed the 256 MiB request limit");
      job.image_paths.emplace_back(string(entry, "pixel_values", true), string(entry, "position_ids", true));
      request.images.push_back(std::move(image));
    } else if (token == model::kImageTokenId || token == model::kEndImageTokenId) {
      throw std::invalid_argument("image tokens must occur in bracketed image placeholder spans");
    }
  }
  if (request.images.size() != prepared.size())
    throw std::invalid_argument("image spans and prepared_images must have matching counts");
  if (command.contains("outputs")) {
    const auto& outputs = command.at("outputs");
    keys(outputs, {"tokens", "logits"});
    if (outputs.contains("tokens")) job.tokens = std::make_unique<offline::Output>(string(outputs, "tokens", true), false);
    if (outputs.contains("logits")) {
      if (request.operation != Operation::generate) throw std::invalid_argument("prefill does not produce logits");
      job.logits = std::make_unique<offline::Output>(string(outputs, "logits", true), true);
      request.capture_logits = true;
    }
  }
  return request;
}

Json stats_json(const kv_cache::CacheStats& stats) {
  const auto pool = [](const auto& p) { return Json{{"capacity", p.capacity}, {"used", p.used},
      {"peak_used", p.peak_used}, {"free", p.free}}; };
  return {{"gpu", pool(stats.gpu)}, {"cpu", pool(stats.cpu)}, {"index_bytes", stats.index_bytes},
      {"index_used", stats.index_used}, {"page_count", stats.page_count},
      {"checkpoint_count", stats.checkpoint_count}, {"execution_count", stats.execution_count},
      {"copy_on_write_pages", stats.copy_on_write_pages}};
}

Json result(const BatchRequest& request, std::uint32_t mtp_depth, const text::TextContract& contract) {
  const auto operation = request.operation == Operation::generate ? "generate" :
      request.operation == Operation::prefill ? "prefill" : request.operation == Operation::finish ? "finish" : "stats";
  const bool stopped = request.honor_eos && !request.outputs.empty() &&
      contract.is_stop(request.outputs.back());
  Json out{{"event", "offline_result"}, {"id", request.id}, {"operation", operation},
      {"prompt_tokens", request.prompt->size()}, {"completion_tokens", request.completion_tokens},
      {"processed_tokens", request.cursor}, {"cached_tokens", request.cached_tokens},
      {"shared_tokens", request.shared_tokens}, {"prefill_tokens", request.prefill_tokens},
      {"input_checkpoint", request.input_checkpoint}, {"completion_checkpoint", request.completion_checkpoint},
      {"finish_reason", stopped ? "stop" : "length"}, {"queue_seconds", request.queue_seconds},
      {"prefill_seconds", request.prefill_seconds}, {"decode_seconds", request.decode_seconds},
      {"prefill_gpu_seconds", request.prefill_gpu_seconds}, {"decode_gpu_seconds", request.decode_gpu_seconds},
      {"dependency_wait_seconds", request.dependency_wait_seconds}, {"mtp_depth", request.ordinary_decode ? 0 : mtp_depth},
      {"mtp_cycles", request.mtp_cycles}, {"mtp_proposed", request.mtp_proposed}, {"mtp_accepted", request.mtp_accepted},
      {"mtp_rejected_cycles", request.mtp_rejected_cycles}, {"mtp_accepted_histogram", request.mtp_accepted_histogram}};
  if (request.operation == Operation::stats) out["cache_stats"] = stats_json(request.control_stats);
  return out;
}
}  // namespace

int run_jobs(const std::string& artifact_path, std::uint32_t max_batch,
             RuntimeSettings settings, const std::string& qdq_mask_path) {
  settings.mtp_depth = effective_mtp_depth(settings.mtp_depth, settings.assistant_path, settings.mtp_min_depth);
  if (!settings.mtp_depth) settings.mtp_min_depth = 0;
  const auto selected_model = artifact_model(artifact_path);
  const auto& contract = text_contract(selected_model);
  const auto limits = live_batch_limits(max_batch, settings, 256, selected_model);
  Signals signals;
  DiagnosticsToStderr diagnostics;
  offline::Channel channel(STDIN_FILENO, STDOUT_FILENO);
  sm120::validate_cuda_device();
  std::unique_ptr<sm120::WeightArena> weights;
  std::unique_ptr<ExecutionBackend> backend;
  if (selected_model == ModelKind::gemma4_26b_a4b) {
    backend = gemma4_26b_a4b::sm120::make_runtime_backend(gemma4_26b_a4b::ArtifactFile::Open(artifact_path), limits, settings.nvfp4_activation_policy, settings.assistant_path, settings.vision_path, qdq_mask_path);
  } else {
    auto file = artifact::ArtifactFile::Open(artifact_path);
    const auto mask = qdq_mask_path.empty() ? qdq::Mask{} : qdq::Mask::Load(qdq_mask_path);
    weights = std::make_unique<sm120::WeightArena>(file, mask, settings.mtp_depth ? settings.assistant_path : "", settings.vision_path);
    backend = sm120::make_runtime_backend(*weights, limits, settings.nvfp4_activation_policy);
  }
  std::map<std::string, Job> jobs;
  const auto prepared_image_bytes = std::make_shared<std::size_t>(0);
  std::vector<std::string> retired;
  BatchScheduler* current = nullptr;
  bool shutdown = false;
  bool stats_dirty = true;
  const auto log = [](const char* event, const Json& data) {
    if (console::json_enabled()) console::event(event, data);
  };
  const auto error = [&](const std::string& id, const std::string& message, const char* code = "request") {
    channel.event(Json{{"event", "offline_error"}, {"id", id}, {"message", message}, {"code", code}}.dump());
  };
  BatchCallbacks callbacks;
  callbacks.ready = [&](auto& request) {
    auto& job = jobs.at(request.id);
    const bool ready = channel.ready() && (!job.logits || (job.logits->ready() && job.logits->error().empty()));
    if (!ready && !job.blocked)
      log("offline_output_blocked", {{"id", request.id}, {"bytes", job.logits ? job.logits->pending_bytes() : 0}});
    job.blocked = !ready;
    return ready && (job.inputs_loaded ||
        job.declared_image_bytes <= kMaxPreparedImageBytes - *prepared_image_bytes);
  };
  callbacks.drain = [&](auto& request) {
    if (!request.delivered) jobs.at(request.id).discard();
  };
  callbacks.start = [&](auto& request) {
    auto& job = jobs.at(request.id);
    if (job.inputs_loaded) return;
    if (job.declared_image_bytes > kMaxPreparedImageBytes - *prepared_image_bytes)
      throw BatchCapacityError("aggregate prepared image capacity is exhausted");
    std::vector<std::shared_ptr<const ImageInput>> loaded;
    loaded.reserve(request.images.size());
    for (std::size_t i = 0; i < request.images.size(); ++i) {
      auto image = std::make_unique<ImageInput>();
      image->begin = request.images[i]->begin;
      image->end = request.images[i]->end;
      image->padded_patch_rows = request.images[i]->padded_patch_rows;
      const auto pixels = vision_engine::prepared_pixel_bytes(image->padded_patch_rows);
      const auto positions = vision_engine::prepared_position_bytes(image->padded_patch_rows);
      const auto bytes = pixels + positions;
      *prepared_image_bytes += bytes;
      // Shared prefix work may outlive its producing job. Charge each tensor
      // until the final request/work reference releases it, including failures.
      auto retained = std::shared_ptr<ImageInput>(image.release(),
          [prepared_image_bytes, bytes](ImageInput* input) {
            delete input;
            *prepared_image_bytes -= bytes;
          });
      retained->pixels = vision_engine::prepared_pixels_from_file_bytes(
          read_bytes(job.image_paths[i].first, vision_engine::prepared_pixel_file_bytes(image->padded_patch_rows)));
      retained->positions = read_bytes(job.image_paths[i].second, positions);
      loaded.push_back(std::move(retained));
    }
    request.images.swap(loaded);
    job.inputs_loaded = true;
  };
  callbacks.emit = [&](auto& request, const auto* tokens, std::size_t count, const void* logits, const TokenLogprobs*) {
    auto& job = jobs.at(request.id);
    if (job.logits) job.logits->append(logits, count * std::size_t(contract.vocabulary_size) * 2);
    channel.event(Json{{"event", "offline_tokens"}, {"id", request.id},
        {"tokens", std::vector<std::uint32_t>(tokens, tokens + count)}}.dump());
  };
  callbacks.finish = [&](auto& request) {
    auto& job = jobs.at(request.id);
    if (job.tokens) {
      job.tokens->append(request.outputs.data(), request.outputs.size() * sizeof(std::uint32_t));
      job.tokens->pump();
      job.tokens->complete();
    }
    if (job.logits) job.logits->complete();
    job.result_event = channel.event(result(request, limits.mtp_depth, contract).dump());
  };
  callbacks.done = [&](auto& request) {
    auto& job = jobs.at(request.id);
    if (request.phase == BatchPhase::cancelled) {
      job.discard();
      if (!job.failed && !shutdown) channel.event(Json{{"event", "offline_cancelled"},
          {"id", request.id}, {"completion_tokens", request.completion_tokens},
          {"processed_tokens", request.cursor}}.dump());
    }
    log(request.phase == BatchPhase::cancelled ? "offline_batch_cancelled" : "offline_batch_result",
        result(request, limits.mtp_depth, contract));
    stats_dirty = true;
    retired.push_back(request.id);
  };
  callbacks.reject = [&](auto& request, const std::string& message, BatchFailure failure) {
    auto& job = jobs.at(request.id);
    job.failed = true;
    job.discard();
    error(request.id, message, failure == BatchFailure::capacity ? "capacity" : "execution");
  };
  callbacks.trace = [&](const auto& event, const auto& fields) {
    if (console::json_enabled()) log("offline_batch_event", Json::parse(
        "{\"event\":\"" + std::string(event) + "\"," + fields + "}"));
    stats_dirty = true;
  };
  callbacks.poll = [&](bool inflight) {
    channel.pump();
    shutdown |= interrupted || !channel.error().empty();
    for (auto& [id, job] : jobs) {
      auto& request = current->requests[job.slot];
      if (job.result_event && job.result_event <= channel.written_events())
        request.delivered = true;
      // A delivered success owns its output files and committed cache demands.
      // Shutdown may still cancel unfinished requests, but cannot undo it.
      if (request.delivered) continue;
      if (job.logits) job.logits->pump();
      if (shutdown || (job.logits && !job.logits->error().empty() && !job.failed)) {
        if (!shutdown) { job.failed = true; error(id, job.logits->error()); }
        job.discard();
        current->cancel(job.slot, inflight);
      }
    }
    // Limit host work per poll as well as admitted slots and retained bytes.
    for (unsigned count = 0; !shutdown && channel.ready() && count < 16; ++count) {
      const auto line = channel.read_line();
      if (!line) break;
      if (line->empty()) continue;
      std::string id;
      try {
        const auto command = Json::parse(*line);
        id = string(command, "id", true);
        if (command.value("op", std::string{}) == "cancel") {
          keys(command, {"op", "id"});
          const auto found = jobs.find(id);
          // Once success is queued it cannot be withdrawn from the JSONL
          // stream. A late cancellation is a no-op, including after retirement.
          if (found == jobs.end() || found->second.result_event) continue;
          found->second.discard();
          current->cancel(found->second.slot, inflight);
          continue;
        }
        if (jobs.count(id)) throw std::invalid_argument("id is already active");
        if (jobs.size() == limits.max_requests) throw BatchCapacityError("offline job queue is full (256 jobs)");
        Job job;
        auto request = parse_request(command, job, limits.max_horizon, contract);
        if (!request.images.empty() && settings.vision_path.empty())
          throw std::invalid_argument("image input requires --vision PATH");
        const auto slot = current->submit(std::move(request));
        job.slot = slot;
        jobs.emplace(id, std::move(job));
        log("offline_admitted", {{"id", id}, {"request", id}});
        stats_dirty = true;
      } catch (const std::exception& failure) {
        error(id, failure.what(), dynamic_cast<const BatchCapacityError*>(&failure) ? "capacity" : "request");
      }
    }
  };
  BatchScheduler scheduler(std::move(backend),
                           limits, std::move(callbacks));
  current = &scheduler;
  scheduler.write_startup_capacity();
  channel.event(Json{{"event", "offline_ready"}, {"max_batch", max_batch}, {"max_pending", limits.max_requests},
      {"prefill_chunk_tokens", limits.prefill_chunk_tokens}, {"prefill_batch_tokens", limits.prefill_batch_tokens},
      {"prefill_budget_tokens", limits.prefill_budget_tokens},
      {"max_context_tokens", limits.max_horizon}, {"vocab_size", contract.vocabulary_size},
      {"mtp_depth", limits.mtp_depth}, {"mtp_min_depth", limits.mtp_min_depth},
      {"decode_width", limits.decode_width}, {"max_logit_chunk_bytes",
          (std::size_t(limits.mtp_depth) + 1) * contract.vocabulary_size * 2}}.dump());
  while (!shutdown && (!channel.eof() || scheduler.has_pending() || !channel.empty())) {
    const bool progressed = scheduler.step();
    for (const auto& id : retired) {
      scheduler.forget(jobs.at(id).slot);
      jobs.erase(id);
    }
    retired.clear();
    if (stats_dirty && console::json_enabled()) {
      std::ostringstream stats;
      scheduler.write_live_stats(stats);
      log("offline_batch_stats", Json::parse(stats.str()));
      stats_dirty = false;
    }
    if (!progressed) std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (!channel.error().empty()) throw std::runtime_error(channel.error());
  return interrupted ? 130 : 0;
}
}  // namespace gewell::app
