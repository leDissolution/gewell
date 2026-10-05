#include "gewell/runtime/scheduler.h"
#include "gewell/runtime/mtp_stats.h"
#include "gewell/runtime/mtp_capture.h"
#include "gewell/console.h"
#include <algorithm>
#include <cmath>
#include <thread>
#include <sstream>
#include <set>
#include <iomanip>
namespace gewell::runtime {
namespace {
CacheEventSink cache_event_sink() {
  if (!console::json_enabled()) return {};
  return [](const nlohmann::json& event) { console::event("server_cache_checkpoint", event); };
}
[[noreturn]] void fail(std::string_view operation, std::string_view message) {
  throw std::runtime_error(std::string(operation) + ": " + std::string(message));
}
double seconds_since(std::chrono::steady_clock::time_point begin) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
}
void release_consumed_images(std::vector<std::shared_ptr<const ImageInput>>& images, std::uint32_t cursor) {
  for (auto& image : images) {
    if (image->end > cursor || (image->pixels.empty() && image->positions.empty())) continue;
    auto metadata = std::make_shared<ImageInput>();
    metadata->begin = image->begin;
    metadata->end = image->end;
    metadata->padded_patch_rows = image->padded_patch_rows;
    image = std::move(metadata);
  }
}
}
BatchScheduler::BatchScheduler(std::unique_ptr<ExecutionBackend> executor, BatchLimits limits, BatchCallbacks adapters)
      : limits(limits), callbacks(std::move(adapters)), capacity(limits.capacity),
        mtp_depth(limits.mtp_depth), owned_backend(std::move(executor)), backend(*owned_backend),
        memory(backend.memory_plan()), config(backend.cache_config()),
        cache(config, backend.cache_storage_factory(), cache_event_sink()) {
    const auto& supported = backend.limits();
    if (!capacity || capacity > supported.max_batch_rows ||
        mtp_depth > supported.max_mtp_depth || limits.mtp_min_depth > mtp_depth ||
        (mtp_depth && std::uint64_t(capacity) * (mtp_depth + 1) > supported.max_verifier_rows) ||
        !limits.max_requests || capacity > limits.max_requests ||
        !limits.prefill_chunk_tokens || limits.prefill_chunk_tokens > supported.max_prefill_chunk_tokens ||
        limits.prefill_batch_tokens < limits.prefill_chunk_tokens ||
        limits.prefill_batch_tokens > supported.max_prefill_batch_tokens ||
        !limits.plan_rows || limits.plan_rows > limits.prefill_chunk_tokens ||
        limits.max_horizon < limits.plan_rows || limits.max_horizon > supported.context_tokens)
      fail("batch configuration", "invalid scheduler limits");
    if (!limits.mtp_stats_path.empty()) mtp_stats = std::make_unique<MtpStats>(limits);
    if (!limits.mtp_capture.path.empty()) mtp_capture = std::make_unique<MtpCapture>(limits);
    // Both sinks use the same admission sequence, including after a restart.
    if (mtp_stats) sequence_begin = mtp_stats->next_sequence();
    if (mtp_capture) sequence_begin = std::max(sequence_begin, mtp_capture->next_sequence());
    requests.reserve(limits.max_requests);
    active.reserve(capacity);
    works.reserve(2 * limits.max_requests);
    backend.allocate_staging();
    const auto bootstrap = cache.try_begin_batch(0, 1, {});
    if (!bootstrap) fail("batch admission", "cannot initialize cache execution");
    try {
      backend.initialize(cache, *bootstrap, limits);
      cache.release(*bootstrap);
    } catch (...) {
      backend.wait();
      cache.release(*bootstrap);
      throw;
    }
    backend.initialize_outputs(limits.captures, limits.logprobs);
    started = std::chrono::steady_clock::now();
  }

BatchScheduler::~BatchScheduler() {
    backend.wait();
    backend.release_image();
    for (auto& request : requests) {
      if (mtp_stats && !request.reported) {
        try { mtp_stats->finish(request, "interrupted"); } catch (...) {}
      }
      if (callbacks.drain) {
        try { callbacks.drain(request); } catch (...) {}
      }
      if (request.execution) cache.release(request.execution);
      if (!request.reported) request.phase = BatchPhase::cancelled;
      release_request_cache(request);
    }
    for (auto& work : works) {
      if (!work) continue;
      if (work->execution) cache.release(work->execution);
      if (work->checkpoint) cache.unpin_batch_checkpoint(work->checkpoint);
    }
  }

bool BatchScheduler::computes(const BatchRequest& request) {
    return request.operation == Operation::generate || request.operation == Operation::prefill;
  }

void BatchScheduler::observe_scheduled(BatchRequest& request) {
    if (!computes(request) ||
        request.phase == BatchPhase::cancelled || request.metrics_scheduled) return;
    request.metrics_scheduled = std::chrono::steady_clock::now();
    request.queue_seconds = std::chrono::duration<double>(
        *request.metrics_scheduled - request.accepted_at).count();
    if (callbacks.metrics && request.operation == Operation::generate)
      callbacks.metrics->queue.observe(request.queue_seconds);
  }

void BatchScheduler::observe_prompt(BatchRequest& request, std::uint32_t cursor, bool cached, bool shared) {
    if (!callbacks.metrics || request.operation != Operation::generate ||
        request.phase == BatchPhase::cancelled) return;
    cursor = std::min<std::uint32_t>(cursor, request.prompt->size());
    if (cursor <= request.metrics_prompt_tokens) return;
    const auto count = cursor - request.metrics_prompt_tokens;
    request.metrics_prompt_tokens = cursor;
    auto& metrics = *callbacks.metrics;
    metrics.prompt_tokens += count;
    if (cached) {
      metrics.cached_prompt_tokens += count;
      metrics.prefix_hits += count;
      if (shared) metrics.shared_prompt_tokens += count;
    } else metrics.computed_prompt_tokens += count;
  }

void BatchScheduler::observe_output(BatchRequest& request, std::size_t count) {
    if (!callbacks.metrics || request.operation != Operation::generate ||
        request.phase == BatchPhase::cancelled) return;
    observe_scheduled(request);
    auto& metrics = *callbacks.metrics;
    const auto now = std::chrono::steady_clock::now();
    metrics.generation_tokens += count;
    if (!request.metrics_first) {
      request.metrics_first = now;
      metrics.ttft.observe(std::chrono::duration<double>(now - request.metrics_arrival).count());
    } else {
      metrics.itl.observe(std::chrono::duration<double>(now - *request.metrics_last).count());
    }
    request.metrics_last = now;
  }

void BatchScheduler::observe_finished(BatchRequest& request, bool failed) {
    if (!callbacks.metrics || request.operation != Operation::generate || request.metrics_finished) return;
    request.metrics_finished = true;
    auto& metrics = *callbacks.metrics;
    if (failed) { ++metrics.failed; return; }
    if (request.phase == BatchPhase::cancelled) { ++metrics.aborted; return; }
    const bool stopped = request.string_stopped ||
        (!request.outputs.empty() && backend.is_stop_token(request.outputs.back()));
    if (stopped) ++metrics.success_stop;
    else ++metrics.success_length;
    metrics.prompt_length.observe(request.prompt->size());
    metrics.generation_length.observe(request.outputs.size());
    if (request.metrics_first && request.metrics_last && request.metrics_scheduled) {
      const auto elapsed = [](auto end, auto begin) {
        return std::chrono::duration<double>(end - begin).count();
      };
      metrics.e2e.observe(elapsed(*request.metrics_last, request.metrics_arrival));
      metrics.prefill.observe(elapsed(*request.metrics_first, *request.metrics_scheduled));
      const auto decode = elapsed(*request.metrics_last, *request.metrics_first);
      metrics.decode.observe(decode);
      metrics.inference.observe(elapsed(*request.metrics_last, *request.metrics_scheduled));
      metrics.tpot.observe(request.outputs.size() > 1 ? decode / (request.outputs.size() - 1) : 0);
    }
  }

bool BatchScheduler::retains(const BatchRequest& request) const {
    return limits.live && computes(request) && request.controls.admits_retained_state() && !request.controls.finished;
  }

bool BatchScheduler::owner_ready(const BatchRequest& request) const {
    return !request.controls.named() || std::none_of(requests.begin(), requests.end(), [&](const auto& other) {
      return other.phase != BatchPhase::vacant && !other.reported &&
          other.accepted_order < request.accepted_order && other.controls.prompt_id == request.controls.prompt_id;
    });
  }

void BatchScheduler::prepare_request(BatchRequest& request) {
    if (request.prepared) return;
    if (!request.images.empty()) {
      initialize_output(request);
      request.image_spans.clear();
      request.image_spans.reserve(request.images.size());
      for (const auto& image : request.images)
        request.image_spans.push_back({image->begin, image->end, image_digest(*image)});
    }
    if (retains(request)) {
      const auto capacity = cache.active_checkpoint_capacity(request.controls.named()) / limits.max_requests;
      if (capacity < 2) throw BatchCapacityError("batch cache: index budget cannot hold request checkpoint bookkeeping");
      const auto match = cache.find_longest(*request.prompt, request.image_spans);
      request.triggers = make_checkpoint_triggers(request.prompt->size(),
          generation_cache_capacity(request.prompt->size(), request.max_new_tokens, backend.limits().context_tokens), 0,
          match.longest_matching_tokens, limits.checkpoint_interval, request.checkpoint_offsets,
          capacity - 2);
      const auto inside_image = [&](std::uint32_t position) {
        return std::any_of(request.image_spans.begin(), request.image_spans.end(), [&](const auto& image) {
          return image.begin < position && position < image.end;
        });
      };
      request.triggers.erase(std::remove_if(request.triggers.begin(), request.triggers.end(),
          [&](const auto& trigger) { return inside_image(trigger.processed_tokens); }), request.triggers.end());
      // Retain safe branches before/after an image, allowing the next request
      // to change a later image without recomputing earlier image features.
      for (const auto& image : request.image_spans) {
        for (const auto position : {image.begin, image.end}) {
          if (!position || position >= request.prompt->size() || request.triggers.size() == capacity - 2) continue;
          if (std::none_of(request.triggers.begin(), request.triggers.end(),
              [&](const auto& trigger) { return trigger.processed_tokens == position; }))
            request.triggers.push_back({position, CheckpointSource::learned_branch});
        }
      }
      std::sort(request.triggers.begin(), request.triggers.end(),
          [](const auto& a, const auto& b) { return a.processed_tokens < b.processed_tokens; });
      const auto slots = std::min(capacity, request.triggers.size() + limits.max_requests + 2);
      request.checkpoints.reserve(slots);
      if (request.controls.named()) {
        request.pinned_checkpoints.reserve(slots);
        request.demand_commit.changes.reserve(slots);
        if (!cache.reserve_owner_demand_slots(slots, &request.demand_reservation))
          throw BatchCapacityError("batch cache: owner demand metadata capacity is exhausted");
      }
    }
    request.prepared = true;
  }

void BatchScheduler::remember_checkpoint(BatchRequest& request, kv_cache::CheckpointId checkpoint) {
    if (!retains(request) || !checkpoint || !cache.has_checkpoint(checkpoint)) return;
    if (cache.checkpoint_tokens(checkpoint) == request.prompt->size()) request.input_checkpoint = checkpoint;
    if (std::find(request.checkpoints.begin(), request.checkpoints.end(), checkpoint) != request.checkpoints.end()) return;
    if (request.checkpoints.size() == request.checkpoints.capacity())
      fail("batch cache", "reserved checkpoint bookkeeping is exhausted");
    request.checkpoints.push_back(checkpoint);
    if (request.controls.named()) {
      cache.pin_batch_checkpoint(checkpoint);
      request.pinned_checkpoints.push_back(checkpoint);
    }
    else (void)cache.add_automatic_demand(checkpoint, request.controls.priority);
  }

prefix_index::CheckpointSource BatchScheduler::source_kind(CheckpointSource source) {
    switch (source) {
      case CheckpointSource::input: return prefix_index::CheckpointSource::input_endpoint;
      case CheckpointSource::periodic: return prefix_index::CheckpointSource::periodic;
      case CheckpointSource::learned_branch: return prefix_index::CheckpointSource::learned_branch;
    }
    fail("batch checkpoint", "unknown checkpoint trigger");
  }

std::optional<prefix_index::CheckpointSource> BatchScheduler::trigger_at(const BatchRequest& request, std::uint32_t cursor) const {
    if (!retains(request)) return {};
    for (const auto& trigger : request.triggers)
      if (trigger.processed_tokens == cursor) return source_kind(trigger.source);
    return {};
  }

void BatchScheduler::capture_work(BatchPrefixWork& work) {
    if (!limits.live || !work.execution || !work.hidden || work.path.releasable()) return;
    const auto cursor = work.path.completed();
    kv_cache::CheckpointId checkpoint = work.last_checkpoint;
    if (checkpoint && (!cache.has_checkpoint(checkpoint) || cache.checkpoint_tokens(checkpoint) != cursor)) checkpoint = 0;
    for (const auto& dep : work.path.dependents()) {
      auto& request = requests[dep.request_id];
      auto source = trigger_at(request, cursor);
      if (!source && retains(request) && work.path.at_boundary() && dep.fork_tokens == cursor &&
          cursor < request.prompt->size()) {
        const auto remaining = std::count_if(request.triggers.begin(), request.triggers.end(),
            [&](const auto& trigger) { return trigger.processed_tokens > cursor; });
        if (request.checkpoints.size() + remaining + 2 < request.checkpoints.capacity())
          source = prefix_index::CheckpointSource::learned_branch;
      }
      if (!source) continue;
      if (!checkpoint && work.capture_suppressed) continue;
      std::vector<std::uint32_t> prefix(work.path.prompt()->begin(), work.path.prompt()->begin() + cursor);
      if (!checkpoint) {
        const auto wall = std::chrono::steady_clock::now();
        checkpoint = cache.try_capture(work.execution, prefix, work.hidden, backend.context(), *source,
            !request.controls.named(), request.controls.priority, true,
            request.operation == Operation::prefill &&
                *source == prefix_index::CheckpointSource::input_endpoint, work.path.images());
        log_event("capture", number("work", work.serial) + "," + number("tokens", cursor) +
            "," + number("checkpoint", checkpoint));
        prefix_capture_seconds += seconds_since(wall);
        if (checkpoint) ++prefix_checkpoints;
      } else cache.add_checkpoint_source(checkpoint, prefix, *source, work.path.images());
      remember_checkpoint(request, checkpoint);
    }
    work.last_checkpoint = checkpoint;
  }

std::uint32_t BatchScheduler::prefill_rows(const BatchPrefixWork& work) const {
    const auto cursor = work.path.completed();
    auto end = cursor + limits.prefill_chunk_tokens;
    for (const auto& dep : work.path.dependents())
      for (const auto& trigger : requests[dep.request_id].triggers)
        if (trigger.processed_tokens > cursor) end = std::min(end, trigger.processed_tokens);
    return end - cursor;
  }

std::uint32_t BatchScheduler::checkpoint_distance(const BatchRequest& request, std::uint32_t cursor) const {
    auto next = backend.limits().context_tokens;
    for (const auto& trigger : request.triggers)
      if (trigger.processed_tokens > cursor) next = std::min(next, trigger.processed_tokens);
    return next - cursor;
  }

std::vector<std::uint32_t> BatchScheduler::processed_history(const BatchRequest& request) const {
    std::vector<std::uint32_t> tokens(request.prompt->begin(), request.prompt->end());
    if (request.outputs.size() > 1) tokens.insert(tokens.end(), request.outputs.begin(), request.outputs.end() - 1);
    return tokens;
  }

void BatchScheduler::finalize_cache(BatchRequest& request) {
    if (request.cache_finalized) return;
    if (retains(request)) {
      request.completion_checkpoint = request.input_checkpoint;
      if (request.execution && request.outputs.size() > 1) {
        const auto tokens = processed_history(request);
        if (!cache.try_resize_batch(request.execution, tokens.size()))
          fail("batch cache", "cannot shrink terminal execution reservation");
        kv_cache::CheckpointId checkpoint = 0;
        for (const auto candidate : request.checkpoints)
          if (cache.has_checkpoint(candidate) && cache.checkpoint_tokens(candidate) == tokens.size()) checkpoint = candidate;
        if (checkpoint) cache.add_checkpoint_source(checkpoint, tokens, prefix_index::CheckpointSource::continuation_endpoint,
                                                   request.image_spans);
        else {
          const auto capture = [&] {
            return cache.try_capture(request.execution, tokens, request.terminal_hidden,
                backend.context(), prefix_index::CheckpointSource::continuation_endpoint,
                !request.controls.named(), request.controls.priority, false,
                request.operation == Operation::prefill, request.image_spans);
          };
          checkpoint = capture();
          if (!checkpoint && trim_provisional_pins(request)) checkpoint = capture();
        }
        remember_checkpoint(request, checkpoint);
        request.completion_checkpoint = checkpoint;
      } else if (request.completion_checkpoint && cache.has_checkpoint(request.completion_checkpoint)) {
        cache.add_checkpoint_source(request.completion_checkpoint, *request.prompt,
            prefix_index::CheckpointSource::continuation_endpoint, request.image_spans);
      }
      if (request.operation == Operation::prefill &&
          (!request.completion_checkpoint || !cache.has_checkpoint(request.completion_checkpoint)))
        throw BatchCapacityError("batch prefill: the prompt endpoint does not fit the remaining KV budget");
    }
    request.cache_finalized = true;
  }

void BatchScheduler::commit_cache(BatchRequest& request) {
    if (retains(request) && request.controls.named() && !request.demands_committed) {
      for (const auto checkpoint : request.checkpoints) {
        if (cache.has_checkpoint(checkpoint) && std::find(request.pinned_checkpoints.begin(),
            request.pinned_checkpoints.end(), checkpoint) == request.pinned_checkpoints.end()) {
          cache.pin_batch_checkpoint(checkpoint);
          request.pinned_checkpoints.push_back(checkpoint);
        }
      }
      auto checkpoints = request.checkpoints;
      std::sort(checkpoints.begin(), checkpoints.end());
      if (!cache.add_owner_demands(checkpoints, request.controls.prompt_id, request.controls.priority,
          &request.demand_reservation, &request.demand_commit))
        throw BatchCapacityError("batch cache: owner demand metadata capacity is exhausted");
      request.demands_committed = true;
    }
  }

bool BatchScheduler::trim_provisional_pins(BatchRequest& request) {
    if (request.demands_committed || request.pinned_checkpoints.empty()) return false;
    for (const auto checkpoint : request.pinned_checkpoints) cache.unpin_batch_checkpoint(checkpoint);
    request.pinned_checkpoints.clear();
    return true;
  }

void BatchScheduler::release_request_cache(BatchRequest& request) {
    if (request.phase == BatchPhase::cancelled && request.demands_committed)
      cache.rollback_owner_demands(request.controls.prompt_id, &request.demand_commit);
    request.demands_committed = false;
    if (request.controls.named()) {
      for (const auto checkpoint : request.pinned_checkpoints) {
        cache.unpin_batch_checkpoint(checkpoint);
        cache.reclaim_undemanded(checkpoint);
      }
    }
    request.pinned_checkpoints.clear();
    request.checkpoints.clear();
    cache.release_owner_demand_reservation(&request.demand_reservation);
    if (request.controls.finished && request.admitted)
      cache.release_owner(request.controls.prompt_id);
  }

std::size_t BatchScheduler::submit(BatchRequest request) {
    if (!request.images.empty()) {
      if (!computes(request) || !backend.limits().max_image_tokens || !request.prompt)
        fail("batch admission", "invalid image request");
      std::uint32_t previous_end = 0;
      for (const auto& image : request.images) {
        if (!image || image->begin < previous_end || image->begin >= image->end ||
            image->end > request.prompt->size() ||
            image->end - image->begin > backend.limits().max_image_tokens)
          fail("batch admission", "invalid or unordered image spans");
        previous_end = image->end;
        for (const auto position : request.checkpoint_offsets)
          if (image->begin < position && position < image->end)
            fail("batch admission", "checkpoint offset falls inside an image feature span");
      }
    }
    const auto horizon = computes(request) ? generation_cache_capacity(request.prompt->size(), request.max_new_tokens, backend.limits().context_tokens) : 0;
    if (computes(request) && (horizon > limits.max_horizon || (config.local_ring_bytes + config.page_table_bytes +
        ((std::size_t(horizon) + config.global_page_tokens - 1) / config.global_page_tokens) *
        config.global_page_bytes > memory.committed_kv_bytes)))
      throw BatchCapacityError("batch admission: request exceeds configured GPU KV capacity");
    if (request.capture_logits && !limits.captures)
      fail("batch admission", "logit capture was not configured");
    if (request.logprobs && !limits.logprobs)
      fail("batch admission", "logprob capture was not configured");
    if (request.top_logprobs > kMaxTopLogprobs)
      fail("batch admission", "top logprob count exceeds 20");
    if (request.sampling.temperature > 0 && request.sampling.top_p > 0 &&
        request.sampling.top_k != 1 && !limits.sampled)
      fail("batch admission", "sampling was not configured");
    const auto bytes = request.prompt->size() * sizeof(std::uint32_t);
    if (bytes > 256 * kv_cache::kMib - retained_prompt_bytes())
      throw BatchCapacityError("batch admission: retained prompt storage exceeds 256 MiB");
    auto slot = std::find_if(requests.begin(), requests.end(),
        [](const auto& value) { return value.phase == BatchPhase::vacant; });
    if (slot == requests.end()) {
      if (requests.size() == limits.max_requests) throw BatchCapacityError("batch admission: request slots exhausted");
      requests.push_back(std::move(request));
      slot = requests.end() - 1;
    } else *slot = std::move(request);
    slot->accepted_order = sequence_begin + submitted++;
    slot->accepted_at = std::chrono::steady_clock::now();
    if (slot->metrics_arrival == std::chrono::steady_clock::time_point{})
      slot->metrics_arrival = slot->accepted_at;
    if (limits.live && computes(*slot)) {
      // A read-only receipt snapshot: actual reuse may change before GPU admission.
      const auto match = !slot->images.empty() ? prefix_index::LookupResult{} : cache.find_longest(*slot->prompt);
      const bool hit = match.has_checkpoint();
      const bool cold = hit && cache.is_cold_checkpoint(match.checkpoint);
      const auto prefix_id = hit ? nlohmann::json(std::to_string(match.checkpoint)) : nlohmann::json(nullptr);
      console::event("server_request_received", {
          {"id", slot->id}, {"prompt_tokens", slot->prompt->size()},
          {"cached_tokens", match.checkpoint_tokens}, {"cached_checkpoint_id", prefix_id},
          {"cached_tier", hit ? nlohmann::json(cold ? "cpu" : "gpu") : nlohmann::json(nullptr)}},
          std::string(slot->operation == Operation::prefill ? "Prefill " : "Request ") + slot->id +
          " received | prompt " + console::number(slot->prompt->size()) + ", cached " +
          console::number(match.checkpoint_tokens) + " tokens | prefix " +
          (hit ? std::to_string(match.checkpoint) + (cold ? " (CPU)" : " (GPU)") : "none"));
    }
    return static_cast<std::size_t>(slot - requests.begin());
  }

void BatchScheduler::forget(std::size_t index) {
    auto& request = requests.at(index);
    if (!request.reported || request.execution || request.terminal_hidden)
      fail("batch retirement", "request is still in use");
    for (const auto& work : works)
      if (work && std::any_of(work->path.dependents().begin(), work->path.dependents().end(),
          [&](const auto& dep) { return dep.request_id == index; }))
        fail("batch retirement", "shared work still references request");
    request = {};
    request.phase = BatchPhase::vacant;
  }

bool BatchScheduler::has_pending() const {
    return std::any_of(requests.begin(), requests.end(), [](const auto& request) {
      return request.phase != BatchPhase::vacant && !request.reported;
    });
  }

double BatchScheduler::wall_seconds() const { return seconds_since(started); }

std::size_t BatchScheduler::retained_prompt_bytes() const {
    std::set<const std::vector<std::uint32_t>*> seen;
    std::size_t bytes = 0;
    const auto count = [&](const pending_prefix::Prompt& prompt) {
      if (prompt && seen.insert(prompt.get()).second) bytes += prompt->size() * sizeof(std::uint32_t);
    };
    for (const auto& request : requests) count(request.prompt);
    for (const auto& work : works) if (work) count(work->path.prompt());
    return bytes;
  }

void BatchScheduler::poll(bool inflight) {
    if (callbacks.poll) callbacks.poll(inflight);
    register_requests();
  }

bool BatchScheduler::step() {
    poll(false);
    bool progressed = retire();
    std::uint64_t prefill_work = 0;
    for (auto& request : requests) {
      if (request.phase != BatchPhase::queued || computes(request) || !owner_ready(request)) continue;
      request.prepared = true;
      if (request.operation == Operation::finish) cache.release_owner(request.controls.prompt_id);
      else request.control_stats = cache.stats();
      request.phase = BatchPhase::finishing;
      progressed = true;
    }
    for (std::size_t wi = 0; wi < works.size(); ++wi) drop_work(wi);
    // Admit compute paths in acceptance order, borrowing completed checkpoints
    // only up to their nearest known branch boundary.
    for (auto work_index : ordered_works()) {
      auto& owned = works[work_index];
      if (!owned || owned->execution || owned->checkpoint || runnable_count() >= capacity) continue;
      auto& work = *owned;
      try {
        const auto boundary = work.path.next_boundary();
        std::vector<std::uint32_t> prefix(work.path.prompt()->begin(), work.path.prompt()->begin() + boundary);
        const auto horizon = work_horizon(work);
        auto match = cache.find_batch_prefix(prefix, horizon, work.path.images());
        const auto admission_begin = std::chrono::steady_clock::now();
        auto execution = cache.try_begin_batch(match.checkpoint, horizon, backend.context());
        if (!execution && active.empty() && cache.stats().execution_count == 0 && match.has_checkpoint()) {
          match = {};
          execution = cache.try_begin_batch(0, horizon, backend.context());
        }
        if (!execution) break;
        work.execution = *execution;
        for (const auto& dep : work.path.dependents()) {
          requests[dep.request_id].admitted = true;
          observe_scheduled(requests[dep.request_id]);
        }
        work.hidden = backend.acquire_hidden();
        work.path.restore_prefix(match.checkpoint_tokens);
        work.captured_endpoint = match.checkpoint_tokens == boundary;
        work.runnable = !work.path.at_boundary();
        auto& request = requests[work.path.dependents().front().request_id];
        if (match.has_checkpoint()) {
          work.last_checkpoint = match.checkpoint;
          cache.restore_terminal_hidden(*execution, work.hidden, backend.context());
          backend.synchronize("complete shared checkpoint restore");
          request.cached_tokens = match.checkpoint_tokens;
          cache.mark_used(match.checkpoint);
          ++prefix_cache_hits;
          cached_tokens += match.checkpoint_tokens;
          prefix_restore_seconds += seconds_since(admission_begin);
          for (const auto& dep : work.path.dependents()) remember_checkpoint(requests[dep.request_id], match.checkpoint);
          for (const auto& dep : work.path.dependents())
            observe_prompt(requests[dep.request_id], match.checkpoint_tokens, true);
        }
        release_consumed_images(work.images, work.path.completed());
        for (const auto& dep : work.path.dependents()) {
          auto& dependent = requests[dep.request_id];
          credit_prefix(dependent, work.path.completed());
          release_consumed_images(dependent.images, dependent.cursor);
        }
        progressed = true;
      } catch (const std::exception& error) {
        reject_work(work_index, error);
        if (!limits.live) throw;
        progressed = true;
      }
    }
    std::optional<std::size_t> blocked_request;
    const auto run_prefill = [&] {
      struct Selected {
        std::size_t work, owner;
        pending_prefix::Step step;
        bool image;
        std::string trace;
      };
      std::vector<Selected> selected;
      std::vector<BatchPrefillInput> inputs;
      std::uint32_t rows = 0;
      for (std::size_t attempt = 0; attempt < works.size() &&
           selected.size() < capacity && rows < limits.prefill_batch_tokens; ++attempt) {
        const auto wi = prefill_turn++ % works.size();
        auto* work = works[wi].get();
        if (!work || !work->execution || work->path.at_boundary() || work->path.releasable()) continue;
        if (!work->runnable && runnable_count() >= capacity) continue;
        const bool image_step = std::any_of(work->path.images().begin(), work->path.images().end(),
            [&](const auto& image) { return image.begin == work->path.completed(); });
        const auto planned = work->path.next_step(prefill_rows(*work));
        // Packing must not introduce new attention/cache chunk boundaries just
        // to consume a small remainder. The candidate leads the next forward.
        if (!selected.empty() && planned.end - planned.begin > limits.prefill_batch_tokens - rows) {
          --prefill_turn;
          break;
        }
        if (!cache.try_resize_batch(work->execution, work_horizon(*work))) {
          work->runnable = false;
          blocked_request = work->path.dependents().front().request_id;
          continue;
        }
        work->runnable = true;
        const auto step = work->path.begin_step(prefill_rows(*work));
        const auto owner = work->path.dependents().front().request_id;
        const auto trace = number("work", work->serial) + "," + named(owner) + "," +
            number("execution", work->execution) + "," + number("begin", step.begin) + "," +
            number("end", step.end) + ",\"image\":" + (image_step ? "true" : "false");
        selected.push_back({wi, owner, step, image_step, trace});
        inputs.push_back({work->execution, work->path.prompt()->data() + step.begin,
                          step.begin, step.end - step.begin, work->hidden});
        if (image_step) {
          const auto image = std::find_if(work->images.begin(), work->images.end(),
              [&](const auto& input) { return input->begin == step.begin; });
          if (image == work->images.end()) fail("shared prefill", "missing image input");
          inputs.back().image = *image;
        }
        rows += step.end - step.begin;
        stop_waiting(requests[owner]);
      }
      if (selected.empty()) return false;
      const auto wall = std::chrono::steady_clock::now();
      backend.begin_step("start shared prefill");
      std::vector<bool> completed;
      try {
        for (const auto& item : selected) log_event("prefill", item.trace);
        log_event("prefill_batch", number("segments", selected.size()) + "," + number("tokens", rows));
        completed = backend.prefill_batch(inputs, [&](std::size_t index) {
          poll(true);
          return !works[selected[index].work]->path.dependents().empty();
        });
        backend.end_step("end shared prefill");
        poll(true);
        synchronize_prefill();
        backend.release_image();
        if (completed.size() != selected.size()) fail("shared prefill", "invalid completion count");
        for (std::size_t i = 0; i < selected.size(); ++i) {
          const auto& item = selected[i];
          auto& work = *works[item.work];
          if (!completed[i] && !work.path.dependents().empty())
            fail("shared prefill", "processed prefix disagrees with completed range");
          work.path.complete_step(completed[i]);
          if (work.path.at_boundary()) work.runnable = false;
          if (completed[i]) {
            log_event("prefill_complete", item.trace);
            if (item.image) log_event("image_encode", item.trace);
          }
        }
      } catch (const std::exception& error) {
        // All selected executions may have pending writes. Retire this cohort
        // after GPU completion; unrelated work remains schedulable.
        for (const auto& item : selected)
          if (works[item.work]) reject_work(item.work, error);
        if (!limits.live) throw;
        progressed = true;
        return true;
      }
      const float milliseconds = backend.elapsed("time shared prefill");
      const auto seconds = seconds_since(wall);
      prefill_gpu_seconds += milliseconds / 1000.0;
      for (std::size_t i = 0; i < selected.size(); ++i) {
        const auto& item = selected[i];
        auto* work = works[item.work].get();
        const auto owner = item.owner;
        const auto step = item.step;
        const auto processed = completed[i] ? step.end : step.begin;
        // As with decode, each owner observes the full batch duration, while
        // aggregate GPU time above counts the physical forward only once.
        requests[owner].prefill_gpu_seconds += milliseconds / 1000.0;
        requests[owner].prefill_seconds += seconds;
        requests[owner].prefill_tokens += processed - step.begin;
        requests[owner].cursor = processed;
        prefill_tokens += processed - step.begin;
        prefill_work += processed - step.begin;
        for (const auto& dep : work->path.dependents()) {
          observe_scheduled(requests[dep.request_id]);
          observe_prompt(requests[dep.request_id], processed, dep.request_id != owner, dep.request_id != owner);
          credit_prefix(requests[dep.request_id], processed);
          release_consumed_images(requests[dep.request_id].images, processed);
        }
        release_consumed_images(work->images, processed);
        late_prefix_replay_tokens += std::min(processed, std::max(step.begin, work->replay_until)) - step.begin;
        work->captured_endpoint = false;
        work->capture_suppressed = false;
        capture_work(*work);
        drop_work(item.work);
      }
      prefer_prefill = false;
      progressed = true;
      return true;
    };
    // Alternate heads and chunks when both can run. If a chunk is blocked,
    // try heads in this same turn so pressure cannot prevent forward progress.
    bool ran_prefill = prefer_prefill && run_prefill();
    // Materialize only completed boundaries. The original ring remains the
    // immutable handoff state while children take private copies; the final
    // group adopts it directly, including after its first client cancels.
    for (std::size_t wi = 0; wi < works.size(); ++wi) {
      auto* work = works[wi].get();
      if (!work || (!work->execution && !work->checkpoint) || !work->path.at_boundary() || work->path.releasable()) continue;
      const auto cursor = work->path.completed();
      if (limits.live) capture_work(*work);
      if (!limits.live && !work->captured_endpoint && work->execution) {
        const bool endpoint = std::any_of(work->path.dependents().begin(), work->path.dependents().end(),
            [&](const auto& dep) { return dep.prompt->size() == cursor; });
        if (endpoint) {
          std::vector<std::uint32_t> prefix(work->path.prompt()->begin(), work->path.prompt()->begin() + cursor);
          const auto wall = std::chrono::steady_clock::now();
          if (cache.try_capture(work->execution, prefix, work->hidden, backend.context(),
              prefix_index::CheckpointSource::input_endpoint, true,
              prefix_index::RetentionPriority::normal, true, false, work->path.images())) ++prefix_checkpoints;
          prefix_capture_seconds += seconds_since(wall);
          work->captured_endpoint = true;
        }
      }
      if (runnable_count() >= capacity) continue;
      std::vector<std::size_t> ready_dependents;
      for (const auto& dep : work->path.dependents())
        if (dep.fork_tokens == cursor) ready_dependents.push_back(dep.request_id);
      // Head-only waiters can leave without another ring. Otherwise try
      // each ready branch: a smaller continuation may fit and release its
      // resources before a larger one adopts the original execution.
      if (work->hidden)
        std::stable_partition(ready_dependents.begin(), ready_dependents.end(), [&](auto i) {
          return requests[i].prompt->size() == cursor && requests[i].max_new_tokens == 1;
        });
      for (const auto index : ready_dependents) {
        auto& request = requests[index];
        if (request.operation == Operation::prefill && request.prompt->size() == cursor) {
          remember_checkpoint(request, work->checkpoint);
          credit_prefix(request, cursor);
          stop_waiting(request);
          request.phase = BatchPhase::finishing;
          work->path.detach(index);
          drop_work(wi);
          progressed = true;
          break;
        }
        // A single decision consumes no additional KV. Sample from the frozen
        // hidden directly, allowing exact waiters even when another ring or
        // full snapshot cannot fit. receive() finishes GPU use before detach.
        if (work->hidden && request.prompt->size() == cursor && request.max_new_tokens == 1) {
          if (ran_prefill || !ready(request)) continue;
          credit_prefix(request, cursor);
          stop_waiting(request);
          initialize_output(request);
          const auto wall = std::chrono::steady_clock::now();
          backend.begin_step("start exact shared head");
          backend.prefix_head_step(0, work->hidden);
          sample_request(0, request);
          backend.end_step("end exact shared head");
          receive({index});
          const float milliseconds = backend.elapsed("time exact shared head");
          prefill_gpu_seconds += milliseconds / 1000.0;
          ++prefill_work;
          request.prefill_gpu_seconds += milliseconds / 1000.0;
          request.prefill_seconds += seconds_since(wall);
          request.phase = BatchPhase::decoding;
          active.push_back(index);
          peak_batch = std::max(peak_batch, active.size());
          log_event("head", number("work", works[wi]->serial) + "," + named(index) + "," +
              number("execution", work->execution) + "," + number("tokens", cursor));
          work->path.detach(index);
          drop_work(wi);
          prefer_prefill = true;
          ran_prefill = progressed = true;
          break;
        }
        auto child = std::make_unique<BatchPrefixWork>(request, index, cursor);
        std::vector<std::size_t> group{index};
        for (const auto& other : work->path.dependents()) {
          if (other.request_id == index || other.fork_tokens != cursor ||
              pending_prefix::common_tokens(*request.prompt, request.image_spans,
                  *other.prompt, requests[other.request_id].image_spans) <= cursor) continue;
          if (!child->path.join(other.request_id, other.prompt, requests[other.request_id].image_spans).joined)
            fail("batch prefix", "nested branch could not attach");
          group.push_back(other.request_id);
        }
        const bool take = group.size() == work->path.dependents().size() && work->execution;
        const auto horizon = work_horizon(*child);
        if (take) {
          if (!cache.try_resize_batch(work->execution, horizon)) {
            blocked_request = index;
            continue;
          }
          child->execution = work->execution;
          child->hidden = work->hidden;
          work->execution = 0;
          work->hidden = {};
        } else {
          if (runnable_count() >= capacity) continue;
          // No source writes until all dependents at this boundary have left.
          if (work->execution && !cache.try_resize_batch(work->execution, cursor))
            fail("batch prefix", "cannot freeze trunk reservation");
          const auto execution = work->execution
              ? cache.try_fork_batch(work->execution, horizon, backend.context())
              : cache.try_begin_batch(work->checkpoint, horizon, backend.context());
          if (!execution) {
            blocked_request = index;
            continue;
          }
          child->execution = *execution;
          try {
            child->hidden = backend.acquire_hidden();
            if (work->hidden)
              backend.copy_terminal(child->hidden, work->hidden, "fork terminal hidden");
            else cache.restore_terminal_hidden(*execution, child->hidden, backend.context());
            backend.synchronize("complete shared prefix fork");
            if (work->checkpoint) cache.mark_used(work->checkpoint);
          } catch (...) {
            backend.wait();
            cache.release(child->execution);
            backend.release_hidden(child->hidden);
            throw;
          }
        }
        child->runnable = !child->path.at_boundary();
        child->last_checkpoint = work->last_checkpoint;
        child->capture_suppressed = work->capture_suppressed;
        child->replay_until = work->replay_until;
        const auto child_id = add_work(std::move(child));
        auto* branch = works[child_id].get();
        log_event(take ? "transfer" : "fork", number("work", works[wi]->serial) + "," + number("child_work", works[child_id]->serial) +
            "," + named(index) + "," + number("execution", take ? branch->execution : work->execution) +
            "," + number("child_execution", branch->execution) + "," + number("tokens", cursor));
        for (auto member : group) {
          credit_prefix(requests[member], cursor);
          work->path.detach(member);
        }
        stop_waiting(request);
        if (request.prompt->size() == cursor) {
          request.execution = branch->execution;
          request.terminal_hidden = branch->hidden;
          branch->execution = 0;
          branch->hidden = {};
          request.phase = BatchPhase::decoding;
          active.push_back(index);
          peak_batch = std::max(peak_batch, active.size());
          initialize_output(request);
          works[child_id].reset();
        }
        drop_work(wi);
        progressed = true;
        break;
      }
    }
    // Each turn runs one head or one bounded packed forward, then one decode
    // batch. Save terminal hidden at every chunk so exact-cursor late joins
    // remain usable after unrelated work overwrites executor scratch.
    for (auto index : active) {
      if (ran_prefill) break;
      auto& request = requests[index];
      if (!request.outputs.empty() || request.phase == BatchPhase::cancelled || !ready(request)) continue;
      const auto wall = std::chrono::steady_clock::now();
      backend.begin_step("start shared prefix head");
      backend.prefix_head_step(request.execution, request.terminal_hidden);
      sample_request(0, request);
      backend.end_step("end shared prefix head");
      receive({index});
      const float milliseconds = backend.elapsed("time shared prefix head");
      prefill_gpu_seconds += milliseconds / 1000.0;
      ++prefill_work;
      request.prefill_gpu_seconds += milliseconds / 1000.0;
      request.prefill_seconds += seconds_since(wall);
      log_event("active_head", named(index) + "," + number("execution", request.execution) + "," +
          number("tokens", request.prompt->size()));
      prefer_prefill = true;
      ran_prefill = progressed = true;
      break;
    }
    if (!ran_prefill) ran_prefill = run_prefill();
    // Batch scheduling work, not GPU rows: retain the selected microbatch
    // shapes and return through poll/retire between forwards. Charge a
    // head as one token so a stream of cache-hit requests cannot starve decode.
    // If no prefill can run, decode immediately, including under KV pressure.
    if (prefill_work) {
      prefill_since_decode += prefill_work;
      if (prefill_since_decode < limits.prefill_budget_tokens) return true;
    }
    prefill_since_decode = 0;
    std::vector<std::size_t> indices;
    std::vector<BatchDecodeInput> inputs;
    for (const auto index : active) {
      auto& request = requests[index];
      if (request.phase == BatchPhase::cancelled ||
          request.cursor != request.prompt->size() || request.outputs.empty() ||
          !ready(request) || terminal(request)) continue;
      inputs.push_back({request.execution,
          static_cast<std::uint32_t>(request.prompt->size() + request.outputs.size() - 1),
          request.outputs.back()});
      indices.push_back(index);
    }
    if (!inputs.empty()) {
      const auto wall = std::chrono::steady_clock::now();
      const auto outputs_before = outputs;
      std::vector<MtpStatsSample> stats(mtp_stats && mtp_stats->enabled() ? inputs.size() : 0);
      backend.begin_step("start batch decode");
      const auto rows_per_request = limits.decode_width / inputs.size();
      const auto depth = limits.decode_width
          ? std::clamp<std::uint32_t>(rows_per_request ? rows_per_request - 1 : 0,
                                      limits.mtp_min_depth, mtp_depth)
          : mtp_depth;
      const bool speculate = depth && std::any_of(indices.begin(), indices.end(),
          [&](std::size_t i) { return !requests[i].ordinary_decode &&
              requests[i].outputs.size() + 1 < requests[i].max_new_tokens; });
      if (callbacks.trace) {
        nlohmann::json ids = nlohmann::json::array(), image_ids = nlohmann::json::array();
        for (const auto i : indices) {
          ids.push_back(requests[i].id);
          if (!requests[i].image_spans.empty()) image_ids.push_back(requests[i].id);
        }
        log_event("decode", "\"requests\":" + ids.dump() + ",\"image_requests\":" + image_ids.dump() +
            ",\"mtp\":" + (speculate ? "true" : "false") + "," + number("mtp_depth", depth));
      }
      if (speculate) {
        std::vector<BatchMtpInput> proposals;
        proposals.reserve(indices.size());
        std::vector<MtpCaptureFeatures> captures(mtp_capture ? indices.size() : 0);
        std::vector<MtpTargetProbes> next_probes(mtp_capture && mtp_capture->enabled() ? indices.size() : 0);
        std::uint32_t verifier_rows = 0;
        for (std::size_t row = 0; row < indices.size(); ++row) {
          auto& request = requests[indices[row]];
          BatchMtpInput proposal;
          proposal.execution = request.execution;
          proposal.pending_token = inputs[row].token;
          proposal.target_hidden = request.terminal_hidden;
          proposal.position = inputs[row].position;
          proposal.depth = request.ordinary_decode ? 0 : std::min<std::uint32_t>(depth,
              request.max_new_tokens - request.outputs.size() - 1);
          proposal.depth = std::min(proposal.depth, checkpoint_distance(request, proposal.position) - 1);
          if (request.mtp_capture_probes) {
            // Pair probes from the preceding committed target row with this
            // round, never with the round that produced those probes.
            const auto& probes = *request.mtp_capture_probes;
            if (proposal.depth && probes.position == proposal.position && probes.pending_token == proposal.pending_token) {
              captures[row].probes = std::move(*request.mtp_capture_probes);
              proposal.capture = &captures[row];
            }
            request.mtp_capture_probes.reset();
          }
          if (!next_probes.empty() && proposal.depth && mtp_capture->select(request.accepted_order, request.mtp_cycles)) {
            next_probes[row].layers = limits.mtp_capture.layers;
            proposal.capture_next = &next_probes[row];
          }
          verifier_rows += proposal.depth + 1;
          proposal.temperature = request.sampling.temperature;
          proposal.top_p = request.sampling.top_p;
          proposal.top_k = request.sampling.top_k;
          proposal.return_probabilities = request.logprobs;
          if (request.constraint)
            proposal.constraint_mask = [state = request.constraint.get()](
                const std::uint32_t* drafts, std::uint32_t depth, std::uint32_t* masks) {
              state->block_masks(drafts, depth, masks);
            };
          proposal.uniforms.resize(2 * proposal.depth + 1);
          for (auto& u : proposal.uniforms)
            u = std::min(std::uniform_real_distribution<float>(0, 1)(request.rng),
                         std::nextafter(1.0F, 0.0F));
          proposals.push_back(std::move(proposal));
        }
        const auto result = backend.run_batch_mtp(proposals);
        constraint_draft_downloads += result.constraint_draft_downloads;
        constraint_mask_uploads += result.constraint_mask_uploads;
        constraint_draft_bytes += result.constraint_draft_bytes;
        constraint_mask_bytes += result.constraint_mask_bytes;
        mtp_verifier_rows += verifier_rows;
        ++verifier_occupancy[verifier_rows];
        mtp_draft_gpu_seconds += result.draft_gpu_milliseconds / 1000.0;
        mtp_verify_gpu_seconds += result.verify_gpu_milliseconds / 1000.0;
        mtp_select_gpu_seconds += result.select_gpu_milliseconds / 1000.0;
        std::vector<std::uint32_t> emitted(indices.size());
        std::vector<BatchMtpCommit> commits;
        commits.reserve(indices.size());
        for (std::size_t row = 0; row < indices.size(); ++row) {
          auto& request = requests[indices[row]];
          const auto& selected = result.requests[row];
          if (!stats.empty()) {
            auto& sample = stats[row];
            sample.speculative = true;
            sample.depth = proposals[row].depth;
            sample.failed = !selected.error.empty();
            sample.draft_seconds = result.draft_gpu_milliseconds / 1000.0;
            sample.verify_seconds = result.verify_gpu_milliseconds / 1000.0;
            sample.select_seconds = result.select_gpu_milliseconds / 1000.0;
          }
          if (!selected.error.empty()) {
            if (!limits.live) throw std::runtime_error(selected.error);
            callbacks.reject(request, selected.error, BatchFailure::execution);
            cancel(indices[row]);
            continue;
          }
          auto count = static_cast<std::uint32_t>(selected.tokens.size());
          if (request.honor_eos)
            for (std::uint32_t i = 0; i < count; ++i)
              if (backend.is_stop_token(selected.tokens[i])) { count = i + 1; break; }
          emitted[row] = count;
          if (proposals[row].capture)
            mtp_capture->record(request, proposals[row], selected, decode_batches + 1, indices.size(), count);
          if (proposals[row].capture_next && count == selected.tokens.size()) {
            auto& probes = next_probes[row];
            probes.position = proposals[row].position + count;
            probes.pending_token = selected.tokens.back();
            request.mtp_capture_probes = std::make_unique<MtpTargetProbes>(std::move(probes));
          }
          if (!stats.empty()) stats[row].accepted = std::min(selected.verification.accepted_drafts, count);
          commits.push_back({static_cast<std::uint32_t>(row), request.execution,
              inputs[row].position, count, request.terminal_hidden,
              request.capture_logits, request.logprobs,
              request.top_logprobs});
          if (proposals[row].depth) {
            const auto accepted = std::min(selected.verification.accepted_drafts, count);
            request.mtp_previous_depth = proposals[row].depth;
            request.mtp_previous_accepted = selected.verification.accepted_drafts;
            ++request.mtp_cycles;
            request.mtp_proposed += proposals[row].depth;
            request.mtp_accepted += accepted;
            if (request.mtp_accepted_histogram.size() <= accepted)
              request.mtp_accepted_histogram.resize(accepted + 1);
            ++request.mtp_accepted_histogram[accepted];
            request.mtp_rejected_cycles += selected.verification.rejected_index < proposals[row].depth &&
                selected.verification.rejected_index < count;
            ++mtp_cycles;
            mtp_proposed += proposals[row].depth;
            mtp_accepted += std::min(selected.verification.accepted_drafts, count);
            mtp_rejected_cycles += selected.verification.rejected_index < proposals[row].depth &&
                selected.verification.rejected_index < count;
          }
        }
        if (!commits.empty()) backend.commit_batch_mtp(commits);
        backend.end_step("end batch MTP decode");
        backend.synchronize("complete batch MTP commit");
        for (std::size_t row = 0; row < indices.size(); ++row) {
          if (!result.requests[row].error.empty()) continue;
          auto& request = requests[indices[row]];
          const auto& tokens = result.requests[row].tokens;
          for (std::uint32_t i = 0; i < emitted[row]; ++i)
            accept_constraint(request, tokens[i]);
          request.outputs.insert(request.outputs.end(), tokens.begin(), tokens.begin() + emitted[row]);
          outputs += emitted[row];
          if (callbacks.metrics && request.phase != BatchPhase::cancelled && proposals[row].depth) {
            auto& metrics = *callbacks.metrics;
            const auto accepted = result.requests[row].verification.accepted_drafts;
            ++metrics.mtp_rounds;
            metrics.mtp_draft_tokens += proposals[row].depth;
            metrics.mtp_accepted_tokens += accepted;
            metrics.mtp_emitted_tokens += emitted[row];
            metrics.mtp_rejected_rounds += accepted < proposals[row].depth;
            for (std::uint32_t pos = 0; pos < accepted; ++pos) ++metrics.mtp_accepted_per_position.at(pos);
          }
          observe_output(request, emitted[row]);
          const auto* bytes = request.capture_logits
              ? backend.host_logits() +
                    row * (mtp_depth + 1) * backend.limits().logit_row_bytes : nullptr;
          const auto* scores = request.logprobs
              ? backend.host_logprobs() +
                    row * (mtp_depth + 1) : nullptr;
          if (request.phase != BatchPhase::cancelled)
            callbacks.emit(request, tokens.data(), emitted[row], bytes, scores);
        }
      } else {
        for (const auto index : indices) requests[index].mtp_capture_probes.reset();
        backend.decode_batch(inputs);
        for (std::uint32_t row = 0; row < indices.size(); ++row) {
          auto& request = requests[indices[row]];
          sample_request(row, request);
          if (request.terminal_hidden)
            backend.save_terminal(row, request.terminal_hidden);
        }
        backend.end_step("end batch decode");
        receive(indices);
      }
      const float milliseconds = backend.elapsed("time batch decode");
      decode_gpu_seconds += milliseconds / 1000.0;
      const auto elapsed = seconds_since(wall);
      for (const auto index : indices) {
        requests[index].decode_seconds += elapsed;
        requests[index].decode_gpu_seconds += milliseconds / 1000.0;
      }
      for (const auto index : indices) {
        auto& request = requests[index];
        const auto cursor = cache.processed_tokens(request.execution);
        const auto source = trigger_at(request, cursor);
        if (source && request.phase != BatchPhase::cancelled) {
          const auto tokens = processed_history(request);
          const auto checkpoint = cache.try_capture(request.execution, tokens, request.terminal_hidden,
              backend.context(), *source, !request.controls.named(), request.controls.priority, true, false,
              request.image_spans);
          remember_checkpoint(request, checkpoint);
        }
      }
      decode_tokens += outputs - outputs_before;
      ++decode_batches;
      ++occupancy[inputs.size()];
      for (std::size_t row = 0; row < stats.size(); ++row) {
        const auto& request = requests[indices[row]];
        auto& sample = stats[row];
        sample.batch_id = decode_batches;
        sample.batch_size = inputs.size();
        sample.output_begin = inputs[row].position - request.prompt->size() + 1;
        sample.emitted = request.outputs.size() - sample.output_begin;
        sample.wall_seconds = elapsed;
        sample.gpu_seconds = milliseconds / 1000.0;
        mtp_stats->observe(request, sample);
      }
      progressed = true;
      if (decode_batches % 128 == 0)
        console::event("batch_progress",
            {{"completed", completed}, {"outputs", outputs},
             {"active", active.size()}, {"decode_batches", decode_batches}},
            limits.live ? std::string{} :
                "Batch progress: " + console::number(completed) + " completed, " +
                console::number(active.size()) + " active, " +
                console::number(outputs) + " output tokens");
    }
    if (!progressed && has_pending()) {
      // Live decoders can only be blocked here by their output writes.
      // Let those drain before paying for a checkpoint handoff.
      if (!active.empty()) {
        return false;
      }
      if (limits.live && std::any_of(requests.begin(), requests.end(), [](const auto& request) {
          return (request.phase == BatchPhase::finishing || request.phase == BatchPhase::responding) &&
              request.controls.named() && !request.pinned_checkpoints.empty();
        })) return false;
      if (limits.live) {
        bool trimmed = false;
        for (auto& request : requests)
          if (request.phase == BatchPhase::waiting) trimmed = trim_provisional_pins(request) || trimmed;
        if (trimmed) {
          for (auto& work : works) if (work) work->capture_suppressed = true;
          return true;
        }
      }
      // A short compact checkpoint can preserve a blocked trunk under a
      // budget that cannot hold a second full local ring. Pins protect it
      // until every queued dependent has borrowed or cancelled.
      for (std::size_t wi = 0; wi < works.size(); ++wi) {
        auto* work = works[wi].get();
        if (!work || !work->execution || !work->path.at_boundary() || work->path.releasable()) continue;
        const auto cursor = work->path.completed();
        if (cursor >= backend.limits().local_window_tokens) continue;
        if (!cache.try_resize_batch(work->execution, cursor)) continue;
        std::vector<std::uint32_t> prefix(work->path.prompt()->begin(), work->path.prompt()->begin() + cursor);
        const auto wall = std::chrono::steady_clock::now();
        const auto checkpoint = cache.try_capture(work->execution, prefix, work->hidden, backend.context(),
            prefix_index::CheckpointSource::learned_branch, !limits.live,
            prefix_index::RetentionPriority::normal, true, false, work->path.images());
        prefix_capture_seconds += seconds_since(wall);
        if (!checkpoint) continue;
        cache.pin_batch_checkpoint(checkpoint);
        log_event("snapshot_handoff", number("work", works[wi]->serial) + "," + number("execution", work->execution) +
            "," + number("checkpoint", checkpoint) + "," + number("tokens", cursor));
        cache.release(work->execution);
        work->execution = 0;
        backend.release_hidden(work->hidden);
        work->hidden = {};
        work->checkpoint = checkpoint;
        progressed = true;
        break;
      }
      if (!progressed && blocked_request) {
        const std::string message = "shared prefix handoff cannot fit the GPU budget";
        if (!limits.live) fail("batch scheduler", message);
        callbacks.reject(requests[*blocked_request], message, BatchFailure::capacity);
        cancel(*blocked_request);
        return true;
      }
      if (!progressed && !limits.live && std::none_of(requests.begin(), requests.end(),
          [](const auto& request) { return request.phase == BatchPhase::finishing; }))
        fail("batch scheduler", "fixture events cannot advance");
    }
    return progressed;
  }

void BatchScheduler::stop_at_output(std::size_t index) {
    auto& request = requests.at(index);
    if (request.reported || request.phase == BatchPhase::cancelled) return;
    if (!request.ordinary_decode || request.outputs.empty())
      fail("batch stop", "string stop requires one ordinary output decision");
    request.string_stopped = true;
    log_event("string_stop", named(index) + "," + number("completion_tokens", request.outputs.size()));
  }

bool BatchScheduler::retire() {
    bool progressed = false;
    for (auto it = active.begin(); it != active.end();) {
      auto& request = requests[*it];
      if (request.phase != BatchPhase::cancelled && !terminal(request)) { ++it; continue; }
      if (request.execution) {
        request.cursor = cache.processed_tokens(request.execution);
        if (request.phase != BatchPhase::cancelled) {
          try { finalize_cache(request); }
          catch (const std::exception& error) {
            if (!limits.live) throw;
            callbacks.reject(request, error.what(), batch_failure(error));
            request.phase = BatchPhase::cancelled;
          }
        }
        cache.release(request.execution);
        request.execution = 0;
      }
      backend.release_hidden(request.terminal_hidden);
      request.terminal_hidden = {};
      if (request.phase != BatchPhase::cancelled) request.phase = BatchPhase::finishing;
      it = active.erase(it);
      progressed = true;
    }
    for (auto& request : requests) {
      if (request.reported || (request.phase != BatchPhase::finishing &&
          request.phase != BatchPhase::responding && request.phase != BatchPhase::cancelled)) continue;
      if (request.phase == BatchPhase::responding && !request.delivered) continue;
      if (request.phase == BatchPhase::finishing && !ready(request)) continue;
      if (!limits.live && callbacks.drain) callbacks.drain(request);
      stop_waiting(request);
      request.completion_tokens = request.outputs.size();
      if (request.phase == BatchPhase::finishing && limits.live) {
        try {
          finalize_cache(request);
          commit_cache(request);
          if (request.input_checkpoint && !cache.has_checkpoint(request.input_checkpoint)) request.input_checkpoint = 0;
          if (request.completion_checkpoint && !cache.has_checkpoint(request.completion_checkpoint)) request.completion_checkpoint = 0;
          callbacks.finish(request);
          request.phase = BatchPhase::responding;
          progressed = true;
          continue;
        } catch (const std::exception& error) {
          callbacks.reject(request, error.what(), batch_failure(error));
          request.phase = BatchPhase::cancelled;
        }
      }
      if (request.phase == BatchPhase::cancelled) {
        ++cancelled;
        cancelled_outputs += request.completion_tokens;
      } else {
        const auto expected = request.prompt->size() +
            (request.operation == Operation::generate ? request.completion_tokens - 1 : 0);
        if (computes(request) && request.cursor != expected)
          fail("batch accounting", "processed prefix disagrees with decisions");
        request.phase = BatchPhase::complete;
        ++completed;
      }
      shared_tokens += request.shared_tokens;
      dependency_wait_seconds += request.dependency_wait_seconds;
      if (!limits.live) callbacks.finish(request);
      release_request_cache(request);
      request.images.clear();
      request.mtp_capture_probes.reset();
      if (mtp_stats) mtp_stats->finish(request, request.phase == BatchPhase::cancelled ? "cancelled" : "complete");
      request.reported = true;
      if (callbacks.done) callbacks.done(request);
      std::vector<std::uint32_t>().swap(request.outputs);
      progressed = true;
    }
    return progressed;
  }

void BatchScheduler::write_live_stats(std::ostream& out) const {
    const auto stats = cache.stats();
    const auto telemetry = cache.telemetry();
    std::size_t live_requests = 0, live_works = 0;
    for (const auto& request : requests)
      live_requests += request.phase != BatchPhase::vacant && !request.reported;
    for (const auto& work : works) live_works += bool(work);
    out << std::setprecision(12)
        << "{\"wall_seconds\":" << seconds_since(started)
        << ",\"active_requests\":" << live_requests << ",\"live_works\":" << live_works
        << ",\"request_slots\":" << requests.size() << ",\"work_slots\":" << works.size()
        << ",\"retained_prompt_bytes\":" << retained_prompt_bytes()
        << ",\"execution_count\":" << stats.execution_count
        << ",\"cold_spill_count\":" << telemetry.cold_spill_count
        << ",\"cold_spill_bytes\":" << telemetry.cold_spill_bytes
        << ",\"cold_restore_count\":" << telemetry.cold_restore_count
        << ",\"cold_restore_bytes\":" << telemetry.cold_restore_bytes
        << ",\"prefill_tokens\":" << prefill_tokens << ",\"shared_tokens\":" << shared_tokens
        << ",\"prefill_gpu_seconds\":" << prefill_gpu_seconds
        << ",\"joins\":" << inflight_joins << ",\"completed\":" << completed
        << ",\"cancelled\":" << cancelled << ",\"kv_peak_bytes\":"
        << stats.gpu.peak_used + memory.staging_bytes + memory.hidden_staging_bytes
        << ",\"constraint_draft_downloads\":" << constraint_draft_downloads
        << ",\"constraint_mask_uploads\":" << constraint_mask_uploads
        << ",\"constraint_draft_bytes\":" << constraint_draft_bytes
        << ",\"constraint_mask_bytes\":" << constraint_mask_bytes
        << ",\"host_sampling_scratch_bytes\":" << backend.host_scratch_bytes()
        << ",\"device_sampling_scratch_bytes\":" << backend.sampling_scratch_bytes()
        << ",\"decode_tokens\":" << decode_tokens << ",\"decode_batches\":" << decode_batches
        << ",\"decode_gpu_seconds\":" << decode_gpu_seconds
        << ",\"mtp_cycles\":" << mtp_cycles << ",\"mtp_proposed\":" << mtp_proposed
        << ",\"mtp_accepted\":" << mtp_accepted << ",\"mtp_verifier_rows\":" << mtp_verifier_rows
        << ",\"mtp_draft_gpu_seconds\":" << mtp_draft_gpu_seconds
        << ",\"mtp_verify_gpu_seconds\":" << mtp_verify_gpu_seconds
        << ",\"mtp_select_gpu_seconds\":" << mtp_select_gpu_seconds
        << ",\"decode_batch_histogram\":{";
    bool comma = false;
    for (const auto& item : occupancy) {
      if (comma) out << ',';
      out << '"' << item.first << "\":" << item.second;
      comma = true;
    }
    out << "}}\n" << std::flush;
  }

void BatchScheduler::write_startup_capacity() const {
    const auto execution_buffers = std::size_t(capacity) *
        (config.local_ring_bytes + config.page_table_bytes);
    const auto global_bytes = memory.committed_kv_bytes > execution_buffers
        ? memory.committed_kv_bytes - execution_buffers : 0;
    const auto pages_per_request = global_bytes / config.global_page_bytes / capacity;
    const auto context_tokens = std::min<std::size_t>(limits.max_horizon,
        pages_per_request * config.global_page_tokens);
    console::event("server_kv_capacity", {
        {"gpu_reserved_bytes", limits.kv_bytes},
        {"cache_pool_bytes", memory.committed_kv_bytes},
        {"mtp_staging_bytes", memory.staging_bytes},
        {"prefix_hidden_staging_bytes", memory.hidden_staging_bytes},
        {"local_ring_bytes_per_execution", config.local_ring_bytes},
        {"page_table_bytes_per_execution", config.page_table_bytes},
        {"max_batch", capacity},
        {"execution_buffer_bytes_at_max_batch", execution_buffers},
        {"available_global_kv_bytes_at_max_batch", global_bytes},
        {"vram_context_tokens_per_request_at_max_batch", context_tokens},
        {"context_basis", "equal_length_independent_requests_empty_pool"}});
    if (!console::json_enabled()) {
      console::section("KV memory");
      console::field("GPU_budget_bytes", limits.kv_bytes);
      console::field("cache_pool_bytes", memory.committed_kv_bytes);
      console::field("local_ring_per_request_bytes", config.local_ring_bytes);
      console::field("page_table_per_request_bytes", config.page_table_bytes);
      console::field("execution_buffers_at_full_batch_bytes", execution_buffers);
      console::field("MTP_staging_bytes", memory.staging_bytes);
      console::field("prefix_hidden_staging_bytes", memory.hidden_staging_bytes);
      console::field("global_KV_at_full_batch_bytes", global_bytes);
      console::field("CPU_cache_bytes", limits.cpu_bytes);
      console::field("cache_index_bytes", limits.index_bytes);
      console::section("Context capacity");
      console::field("single_request_tokens", limits.max_horizon);
      console::field("tokens_per_request_at_full_batch", context_tokens);
      console::message("  Full batch: " + console::number(capacity) +
          " independent, equal-length requests in an empty pool.\n"
          "  Context counts prompt + output - 1 processed tokens.");
      if (!context_tokens)
        console::message("  This KV budget cannot fit a full independent batch; requests run at lower concurrency.");
    }
  }

nlohmann::json BatchScheduler::observability_snapshot(metrics::Snapshot& metrics,
                             const std::string& model, bool include_entries) {
    metrics.running = metrics.waiting = 0;
    for (const auto& request : requests) {
      if (request.operation != Operation::generate || request.metrics_finished ||
          request.phase == BatchPhase::vacant || request.phase == BatchPhase::cancelled ||
          request.phase == BatchPhase::finishing || request.phase == BatchPhase::responding ||
          request.phase == BatchPhase::complete) continue;
      if (request.metrics_scheduled) ++metrics.running;
      else ++metrics.waiting;
    }
    metrics.prefill_gpu_seconds = prefill_gpu_seconds;
    metrics.decode_gpu_seconds = decode_gpu_seconds;
    metrics.mtp_draft_seconds = mtp_draft_gpu_seconds;
    metrics.mtp_verify_seconds = mtp_verify_gpu_seconds;
    metrics.mtp_select_seconds = mtp_select_gpu_seconds;
    auto snapshot = cache.observability_snapshot(include_entries);
    const auto timestamp = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    snapshot["snapshot_time"] = timestamp;
    auto& gpu = snapshot["gpu"];
    gpu["configured_budget_bytes"] = limits.kv_bytes;
    gpu["mtp_staging_bytes"] = memory.staging_bytes;
    gpu["prefix_hidden_staging_bytes"] = memory.hidden_staging_bytes;
    const auto hidden_occupied = backend.occupied_hidden_bytes();
    gpu["prefix_hidden_used_bytes"] = hidden_occupied;
    gpu["executor_scratch_bytes"] = backend.scratch_bytes();
    gpu["executor_output_bytes"] = backend.output_bytes();
    metrics::Writer writer(model);
    writer.gauge("gewell:metrics_snapshot_timestamp_seconds", "Unix time of the last consistent scheduler snapshot.", timestamp);
    const auto capacity_bytes = gpu.at("capacity_bytes").get<double>();
    writer.gauge("vllm:kv_cache_usage_perc", "Nonreclaimable allocated GPU cache bytes divided by cache pool capacity (0 to 1).",
        capacity_bytes ? gpu.at("nonreclaimable_bytes").get<double>() / capacity_bytes : 0);
    for (const auto* tier : {"gpu", "cpu"}) {
      const auto& pool = snapshot.at(tier);
      for (const auto* field : {"capacity_bytes", "used_bytes", "free_bytes", "peak_used_bytes",
           "reclaimable_bytes", "nonreclaimable_bytes", "reserved_growth_bytes", "page_slack_bytes",
           "global_page_bytes", "checkpoint_state_bytes", "execution_buffer_bytes", "pending_capture_bytes",
           "other_bytes", "shared_page_bytes", "page_count", "checkpoint_count"}) {
        writer.gauge(std::string("gewell:kv_cache_") + field,
            std::string("Cache pool ") + field + "; shared physical allocations are counted once.",
            pool.at(field).get<double>(), {{"tier", tier}});
      }
    }
    writer.gauge("gewell:kv_budget_bytes", "Configured GPU KV allowance including fixed staging.", limits.kv_bytes);
    writer.gauge("gewell:kv_staging_bytes", "Fixed GPU staging allocation charged to the KV allowance.", memory.staging_bytes, {{"kind", "mtp"}});
    writer.gauge("gewell:kv_staging_bytes", "Fixed GPU staging allocation charged to the KV allowance.", memory.hidden_staging_bytes, {{"kind", "prefix_hidden"}});
    writer.gauge("gewell:kv_staging_used_bytes", "Occupied prefix hidden slots within fixed GPU staging.", hidden_occupied, {{"kind", "prefix_hidden"}});
    writer.gauge("gewell:executor_scratch_bytes", "Explicit engine device scratch allocated outside the KV allowance.", backend.scratch_bytes());
    writer.gauge("gewell:executor_output_bytes", "Explicit engine output buffer allocated outside the KV allowance.", backend.output_bytes());
    writer.gauge("gewell:cache_index_capacity_bytes", "Configured cache metadata budget.", snapshot.at("index").at("capacity_bytes").get<double>());
    writer.gauge("gewell:cache_index_used_bytes", "Accounted cache metadata bytes; not process RSS.", snapshot.at("index").at("used_bytes").get<double>());
    writer.gauge("gewell:cache_executions", "Active GPU cache executions, including shared prefill work.", snapshot.at("execution_count").get<double>());
    writer.counter("gewell:kv_copy_on_write_pages_total", "Global pages copied before mutation of shared state.", snapshot.at("copy_on_write_pages").get<std::uint64_t>());
    writer.counter("gewell:prefill_tokens_total", "Physical prompt tokens computed, including cache prefill operations and cancelled work.", prefill_tokens);
    writer.counter("gewell:inflight_prefix_joins_total", "Requests joining in-flight shared prefix work, including cache prefill operations.", inflight_joins);
    const auto& telemetry = snapshot.at("telemetry");
    for (const auto* tier : {"gpu", "cpu"})
      writer.counter("gewell:cache_reclaimed_bytes_total", "Physical cache bytes reclaimed.", telemetry.at(std::string(tier) + "_reclaimed_bytes").get<std::uint64_t>(), {{"tier", tier}});
    for (const auto* kind : {"spill", "restore"}) {
      const metrics::Labels labels{{"direction", std::string(kind) == "spill" ? "gpu_to_cpu" : "cpu_to_gpu"}};
      const auto prefix = std::string("cold_") + kind;
      writer.counter("gewell:cache_transfers_total", "Completed cache spill or restore operations.", telemetry.at(prefix + "_count").get<std::uint64_t>(), labels);
      writer.counter("gewell:cache_transfer_bytes_total", "Cache bytes copied between GPU and CPU.", telemetry.at(prefix + "_bytes").get<std::uint64_t>(), labels);
      writer.counter("gewell:cache_transfer_seconds_total", "Cumulative wall time of cache transfers in seconds.", telemetry.at(prefix + "_seconds").get<double>(), labels);
    }
    writer.counter("gewell:cache_spill_avoided_rewrite_bytes_total", "CPU backing bytes reused without another spill copy.", telemetry.at("cold_spill_avoided_rewrite_bytes").get<std::uint64_t>());
    for (const auto* event : {"admissions", "hits", "removals"})
      writer.counter("gewell:cache_checkpoint_events_total", "Retained checkpoint lifecycle events.", telemetry.at(std::string("checkpoint_") + event).get<std::uint64_t>(), {{"event", event}});
    observed_cache_metrics = writer.str();
    return include_entries ? std::move(snapshot) : nlohmann::json(nullptr);
  }

void BatchScheduler::stop_waiting(BatchRequest& request) {
    if (request.waiting_since) {
      request.dependency_wait_seconds += seconds_since(*request.waiting_since);
      request.waiting_since.reset();
    }
  }

void BatchScheduler::credit_prefix(BatchRequest& request, std::uint32_t cursor) {
    if (request.cached_tokens + request.prefill_tokens > cursor)
      fail("batch prefix accounting", "attributed input exceeds inherited state");
    request.shared_tokens = cursor - request.cached_tokens - request.prefill_tokens;
    request.cursor = cursor;
  }

void BatchScheduler::drop_work(std::size_t i) {
    auto& work = works[i];
    if (!work || !work->path.releasable()) return;
    if (work->execution) cache.release(work->execution);
    if (work->checkpoint) cache.unpin_batch_checkpoint(work->checkpoint);
    backend.release_hidden(work->hidden);
    work.reset();
  }

void BatchScheduler::register_requests() {
    std::vector<std::size_t> queued;
    for (std::size_t i = 0; i < requests.size(); ++i)
      if (requests[i].phase == BatchPhase::queued) queued.push_back(i);
    std::sort(queued.begin(), queued.end(), [&](auto a, auto b) {
      return requests[a].accepted_order < requests[b].accepted_order;
    });
    for (auto i : queued) {
      auto& request = requests[i];
      if (request.phase != BatchPhase::queued || !computes(request) || !owner_ready(request)) continue;
      // Loading offline image tensors may wait for the aggregate host budget.
      if (!request.images.empty() && !ready(request)) continue;
      try { prepare_request(request); }
      catch (const std::exception& error) {
        if (!limits.live) throw;
        callbacks.reject(request, error.what(), batch_failure(error));
        cancel(i);
        continue;
      }
      const auto horizon = generation_cache_capacity(request.prompt->size(), request.max_new_tokens, backend.limits().context_tokens);
      const auto retained = cache.find_batch_prefix(*request.prompt, horizon, request.image_spans);
      if (callbacks.metrics && request.operation == Operation::generate)
        callbacks.metrics->prefix_queries += request.prompt->size();
      std::size_t best = works.size();
      std::uint32_t matched = retained.checkpoint_tokens;
      for (std::size_t wi = 0; wi < works.size(); ++wi) {
        if (!works[wi] || works[wi]->path.failed() || works[wi]->path.releasable()) continue;
        auto& path = works[wi]->path;
        const auto common = pending_prefix::common_tokens(*path.prompt(), path.images(),
                                                          *request.prompt, request.image_spans);
        const auto dispatched = path.dispatched_end().value_or(path.completed());
        if (common && common >= dispatched && (common > matched ||
            (common == matched && best == works.size() && retained.has_checkpoint() &&
             cache.is_cold_checkpoint(retained.checkpoint)))) {
          matched = common;
          best = wi;
        } else if (common > retained.checkpoint_tokens && common < dispatched &&
                   common > request.late_boundary) {
          request.late_boundary = common;
          log_event("late_join", named(i) + "," + number("work", works[wi]->serial) + "," +
              number("tokens", common) + "," + number("completed", path.completed()) +
              "," + number("dispatched", dispatched));
        }
      }
      request.phase = BatchPhase::waiting;
      if (best != works.size()) {
        if (!works[best]->path.join(i, request.prompt, request.image_spans).joined)
          fail("batch prefix", "registered match could not join");
        if (works[best]->path.at_boundary()) works[best]->runnable = false;
        request.admitted = works[best]->execution || works[best]->checkpoint;
        if (request.admitted) {
          observe_scheduled(request);
          observe_prompt(request, works[best]->path.completed(), true, true);
        }
        remember_checkpoint(request, works[best]->last_checkpoint);
        request.waiting_since = std::chrono::steady_clock::now();
        ++inflight_joins;
        log_event("join", number("work", works[best]->serial) + "," + named(i) + "," + number("tokens", matched));
      } else {
        auto work = std::make_unique<BatchPrefixWork>(request, i);
        work->replay_until = request.late_boundary;
        add_work(std::move(work));
      }
    }
  }

void BatchScheduler::cancel(std::size_t i, bool inflight) {
    auto& request = requests[i];
    if (request.phase == BatchPhase::vacant || request.phase == BatchPhase::complete || request.phase == BatchPhase::cancelled) return;
    if (request.execution) request.cursor = cache.processed_tokens(request.execution);
    std::string detail = named(i) + ",\"inflight\":" + (inflight ? "true" : "false");
    for (std::size_t wi = 0; wi < works.size(); ++wi) {
      if (!works[wi]) continue;
      const auto& deps = works[wi]->path.dependents();
      if (std::none_of(deps.begin(), deps.end(), [&](const auto& dep) { return dep.request_id == i; })) continue;
      credit_prefix(request, works[wi]->path.completed());
      detail += "," + number("work", works[wi]->serial) + "," + number("execution", works[wi]->execution) +
                "," + number("tokens", works[wi]->path.completed());
      works[wi]->path.cancel(i);
      // Enqueued work survives the cancellation until complete_step after sync.
    }
    stop_waiting(request);
    request.phase = BatchPhase::cancelled;
    log_event("cancel", detail);
  }

std::size_t BatchScheduler::runnable_count() const {
    std::size_t count = active.size();
    for (const auto& work : works)
      count += work && work->execution && work->runnable &&
               !work->path.at_boundary() && !work->path.releasable();
    return count;
  }

std::uint32_t BatchScheduler::work_horizon(const BatchPrefixWork& work) const {
    if (work.path.dependents().size() > 1) return work.path.next_boundary();
    const auto& request = requests[work.path.dependents().front().request_id];
    return generation_cache_capacity(request.prompt->size(), request.max_new_tokens, backend.limits().context_tokens);
  }

void BatchScheduler::write_summary(std::ostream& summary) const {
    const auto elapsed = seconds_since(started);
    const auto stats = cache.stats();
    if (has_pending() || stats.execution_count)
      fail("batch cleanup", "request allocations remain live");
  summary << "{\"requests\":" << submitted << ",\"outputs\":" << outputs
          << ",\"prefill_chunk_tokens\":" << limits.prefill_chunk_tokens
          << ",\"prefill_batch_tokens\":" << limits.prefill_batch_tokens
          << ",\"prefill_budget_tokens\":" << limits.prefill_budget_tokens
          << ",\"completed_requests\":" << completed << ",\"cancelled_requests\":" << cancelled
          << ",\"cancelled_outputs\":" << cancelled_outputs
          << ",\"cached_tokens\":" << cached_tokens << ",\"prefill_tokens\":" << prefill_tokens
          << ",\"shared_tokens\":" << shared_tokens << ",\"inflight_joins\":" << inflight_joins
          << ",\"late_prefix_replay_tokens\":" << late_prefix_replay_tokens
          << ",\"dependency_wait_seconds\":" << dependency_wait_seconds
          << ",\"prefix_hidden_bytes\":" << memory.hidden_staging_bytes
          << ",\"prefix_cache_hits\":" << prefix_cache_hits
          << ",\"prefix_checkpoints\":" << prefix_checkpoints
          << ",\"prefix_restore_seconds\":" << prefix_restore_seconds
          << ",\"prefix_capture_seconds\":" << prefix_capture_seconds
          << ",\"kv_retained_bytes\":" << stats.gpu.used
          << ",\"decode_tokens\":" << decode_tokens << ",\"decode_batches\":" << decode_batches
          << ",\"decode_gpu_seconds\":" << decode_gpu_seconds
          << ",\"prefill_gpu_seconds\":" << prefill_gpu_seconds
          << ",\"wall_seconds\":" << elapsed << ",\"peak_batch_size\":" << peak_batch
          << ",\"kv_peak_bytes\":" << stats.gpu.peak_used + memory.staging_bytes + memory.hidden_staging_bytes
          << ",\"mtp_depth\":" << mtp_depth << ",\"mtp_min_depth\":" << limits.mtp_min_depth
          << ",\"decode_width\":" << limits.decode_width
          << ",\"mtp_staging_bytes\":" << memory.staging_bytes
          << ",\"mtp_cycles\":" << mtp_cycles << ",\"mtp_proposed\":" << mtp_proposed
          << ",\"mtp_accepted\":" << mtp_accepted << ",\"mtp_rejected_cycles\":" << mtp_rejected_cycles
          << ",\"mtp_verifier_rows\":" << mtp_verifier_rows
          << ",\"constraint_draft_downloads\":" << constraint_draft_downloads
          << ",\"constraint_mask_uploads\":" << constraint_mask_uploads
          << ",\"constraint_draft_bytes\":" << constraint_draft_bytes
          << ",\"constraint_mask_bytes\":" << constraint_mask_bytes
          << ",\"host_sampling_scratch_bytes\":" << backend.host_scratch_bytes()
          << ",\"device_sampling_scratch_bytes\":" << backend.sampling_scratch_bytes()
          << ",\"mtp_draft_gpu_seconds\":" << mtp_draft_gpu_seconds
          << ",\"mtp_verify_gpu_seconds\":" << mtp_verify_gpu_seconds
          << ",\"mtp_select_gpu_seconds\":" << mtp_select_gpu_seconds
          << ",\"decode_batch_histogram\":{";
  bool separator = false;
  for (const auto& item : occupancy) {
    if (separator) summary << ',';
    summary << '\"' << item.first << "\":" << item.second;
    separator = true;
  }
  summary << "},\"mtp_verifier_batch_histogram\":{";
  separator = false;
  for (const auto& item : verifier_occupancy) {
    if (separator) summary << ',';
    summary << '\"' << item.first << "\":" << item.second;
    separator = true;
  }
  summary << "}}\n";
  }

std::size_t BatchScheduler::add_work(std::unique_ptr<BatchPrefixWork> work) {
    work->serial = next_work++;
    if (limits.live) {
      for (std::size_t i = 0; i < works.size(); ++i)
        if (!works[i]) { works[i] = std::move(work); return i; }
    }
    if (works.size() == 2 * limits.max_requests)
      fail("batch prefix", "work slots exhausted");
    works.push_back(std::move(work));
    return works.size() - 1;
  }

std::vector<std::size_t> BatchScheduler::ordered_works() const {
    std::vector<std::size_t> indices;
    for (std::size_t i = 0; i < works.size(); ++i) if (works[i]) indices.push_back(i);
    std::sort(indices.begin(), indices.end(), [&](auto a, auto b) {
      return works[a]->serial < works[b]->serial;
    });
    return indices;
  }

void BatchScheduler::reject_work(std::size_t index, const std::exception& error) {
    backend.wait();
    backend.release_image();
    auto& work = *works[index];
    if (work.path.inflight()) work.path.complete_step(false);
    std::vector<std::size_t> dependents;
    for (const auto& dep : work.path.dependents()) dependents.push_back(dep.request_id);
    for (const auto dependent : dependents) {
      cancel(dependent);
      if (limits.live) callbacks.reject(requests[dependent], error.what(), batch_failure(error));
    }
    drop_work(index);
  }

void BatchScheduler::synchronize_prefill() {
    if (limits.live) {
      while (true) {
        if (backend.ready("complete shared prefill")) break;
        poll(true);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    } else backend.synchronize("complete shared prefill");
  }

void BatchScheduler::initialize_output(BatchRequest& request) {
    if (request.output_initialized) return;
    request.outputs.reserve(request.max_new_tokens);
    callbacks.start(request);
    request.output_initialized = true;
  }

bool BatchScheduler::ready(BatchRequest& request) { return callbacks.ready(request); }

bool BatchScheduler::terminal(const BatchRequest& request) {
    if (request.operation == Operation::prefill) return request.cursor == request.prompt->size();
    return !request.outputs.empty() && (request.string_stopped || request.outputs.size() == request.max_new_tokens ||
        (request.honor_eos && backend.is_stop_token(request.outputs.back())));
  }

void BatchScheduler::sample_request(std::uint32_t row, BatchRequest& request) {
    if (request.constraint) {
      request.constraint_mask.resize(request.constraint->mask_words());
      request.constraint->fill_mask(request.constraint_mask.data());
    }
    backend.sample_batch_row(row, request.sampling, request.rng,
        request.constraint ? request.constraint_mask.data() : nullptr);
    if (request.logprobs) {
      backend.summarize_batch_row(row, request.sampling, request.top_logprobs,
          request.constraint ? request.constraint_mask.data() : nullptr);
    }
  }

void BatchScheduler::accept_constraint(BatchRequest& request, std::uint32_t token) {
    if (request.constraint && !request.constraint->accept(token))
      fail("constrained output", "sampler selected a token outside the JSON grammar");
  }

void BatchScheduler::receive(const std::vector<std::size_t>& indices) {
    backend.download_ids(indices.size());
    for (std::size_t row = 0; row < indices.size(); ++row)
      if (requests[indices[row]].capture_logits)
        backend.download_logits(row);
    for (std::size_t row = 0; row < indices.size(); ++row)
      if (requests[indices[row]].logprobs)
        backend.download_logprobs(row);
    backend.synchronize("complete batch step");
    backend.check_constraint_sampling();
    for (std::size_t row = 0; row < indices.size(); ++row) {
      auto& request = requests[indices[row]];
      const auto token = backend.output_id(row);
      if (token >= backend.limits().vocab_size) fail("batch output", "invalid token ID");
      accept_constraint(request, token);
      request.outputs.push_back(token);
      ++outputs;
      observe_output(request, 1);
      const auto* bytes = request.capture_logits
          ? backend.host_logits() + row * backend.limits().logit_row_bytes : nullptr;
      const auto* scores = request.logprobs
          ? backend.host_logprobs() + row : nullptr;
      if (request.phase != BatchPhase::cancelled)
        callbacks.emit(request, &token, 1, bytes, scores);
    }
  }

void BatchScheduler::log_event(const std::string& event, const std::string& fields) {
    if (callbacks.trace) callbacks.trace(event, fields);
  }

std::string BatchScheduler::named(std::size_t i) const {
    return std::string("\"request\":") + nlohmann::json(requests[i].id).dump();
  }

} // namespace gewell::runtime
