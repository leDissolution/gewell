#pragma once
#include "gewell/mtp_head.h"

#include "gewell/attention_compute.h"
#include "gewell/runtime/backend.h"
#include "gewell/runtime/cache.h"
#include "gewell/json_constraint.h"
#include "gewell/metrics.h"
#include "gewell/pending_prefix.h"
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
namespace gewell::runtime {
class MtpStats;
class MtpCapture;
enum class Operation { generate, prefill, finish, stats };
enum class BatchPhase { vacant, withheld, queued, waiting, decoding, finishing, responding, complete, cancelled };

class BatchCapacityError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

enum class BatchFailure { capacity, execution };

inline BatchFailure batch_failure(const std::exception& error) {
  return dynamic_cast<const BatchCapacityError*>(&error)
      ? BatchFailure::capacity : BatchFailure::execution;
}

struct BatchRequest {
  std::string id;
  pending_prefix::Prompt prompt;
  std::vector<std::shared_ptr<const ImageInput>> images;
  // Kept after consumed image tensors are released by input adapters.
  std::vector<prefix_index::ImageSpan> image_spans;
  Operation operation{Operation::generate};
  CacheRequestControls controls;
  std::vector<std::uint32_t> checkpoint_offsets;
  std::vector<CheckpointTrigger> triggers;
  std::vector<kv_cache::CheckpointId> checkpoints;
  std::vector<kv_cache::CheckpointId> pinned_checkpoints;
  PersistentCacheManager::OwnerDemandReservation demand_reservation;
  PersistentCacheManager::OwnerDemandCommit demand_commit;
  kv_cache::CheckpointId input_checkpoint{}, completion_checkpoint{};
  bool prepared{}, admitted{}, output_initialized{}, cache_finalized{}, demands_committed{}, delivered{};
  kv_cache::CacheStats control_stats;
  std::uint32_t max_new_tokens{};
  SamplingSettings sampling;
  std::uint64_t seed{};
  bool honor_eos{}, capture_logits{}, logprobs{}, reported{};
  std::uint32_t top_logprobs{5};
  bool ordinary_decode{}, string_stopped{};
  std::unique_ptr<constraint::State> constraint;
  std::vector<std::uint32_t> constraint_mask;
  std::uint64_t client{};
  std::uint64_t accepted_order{};
  std::chrono::steady_clock::time_point accepted_at{};
  std::chrono::steady_clock::time_point metrics_arrival{};
  std::optional<std::chrono::steady_clock::time_point> metrics_scheduled, metrics_first, metrics_last;
  std::uint32_t metrics_prompt_tokens{};
  bool metrics_finished{};
  kv_cache::ExecutionId execution{};
  std::uint32_t cursor{};
  std::uint32_t cached_tokens{};
  std::uint32_t shared_tokens{}, prefill_tokens{}, late_boundary{};
  BatchPhase phase{BatchPhase::queued};
  std::optional<std::chrono::steady_clock::time_point> waiting_since;
  double dependency_wait_seconds{};
  std::vector<std::uint32_t> outputs;
  std::size_t completion_tokens{};
  std::mt19937_64 rng;
  TerminalState terminal_hidden{};
  double queue_seconds{}, prefill_seconds{}, decode_seconds{};
  double prefill_gpu_seconds{}, decode_gpu_seconds{};
  std::uint64_t mtp_cycles{}, mtp_proposed{}, mtp_accepted{}, mtp_rejected_cycles{};
  std::vector<std::uint64_t> mtp_accepted_histogram;
  std::uint32_t mtp_previous_depth{}, mtp_previous_accepted{};
  std::unique_ptr<MtpTargetProbes> mtp_capture_probes;
};

struct BatchPrefixWork {
  std::uint64_t serial{};
  pending_prefix::Work path;
  // The shared producer owns inputs independently of any one client.
  std::vector<std::shared_ptr<const ImageInput>> images;
  kv_cache::ExecutionId execution{};
  kv_cache::CheckpointId checkpoint{};
  kv_cache::CheckpointId last_checkpoint{};
  TerminalState hidden{};
  std::uint32_t replay_until{};
  bool captured_endpoint{};
  bool capture_suppressed{};
  bool runnable{};

  BatchPrefixWork(const BatchRequest& request, std::size_t index,
                  std::uint32_t completed = 0)
      : path(request.prompt, index, 4096, completed, request.image_spans), images(request.images) {}
};


struct BatchCallbacks {
  metrics::Snapshot* metrics = nullptr;
  std::function<void(BatchRequest&)> drain;
  std::function<bool(BatchRequest&)> ready;
  std::function<void(BatchRequest&)> start;
  std::function<void(BatchRequest&, const std::uint32_t*, std::size_t,
                     const void*, const TokenLogprobs*)> emit;
  std::function<void(BatchRequest&)> finish;
  std::function<void(BatchRequest&)> done;
  std::function<void(BatchRequest&, const std::string&, BatchFailure)> reject;
  std::function<void(const std::string&, const std::string&)> trace;
  std::function<void(bool)> poll;
};

struct BatchLimits {
  attention::Compute local_attention_compute{attention::Compute::bf16};
  attention::Compute global_attention_compute{attention::Compute::bf16};
  kv_cache::Format local_kv_format{kv_cache::Format::bf16};
  kv_cache::Format global_kv_format{kv_cache::Format::bf16};
  std::uint32_t capacity{}, mtp_depth{}, plan_rows{}, max_horizon{};
  // Target rows include one pending token per ready decode request. Zero width
  // selects fixed maximum depth; the minimum takes precedence over the target.
  std::uint32_t mtp_min_depth{}, decode_width{};
  std::string mtp_stats_path;
  std::uint32_t mtp_stats_window{64};
  MtpCaptureSettings mtp_capture;
  MtpHeadSettings mtp_head;
  std::uint32_t prefill_chunk_tokens{0};
  std::uint32_t prefill_batch_tokens{0};
  // Soft prefill-token budget between decode passes; zero runs decode after
  // each packed forward/head. Physical forwards and image spans remain atomic.
  std::uint32_t prefill_budget_tokens{};
  std::size_t kv_bytes{}, max_requests{4096};
  std::size_t cpu_bytes{}, index_bytes{kv_cache::kDefaultIndexBytes};
  std::uint32_t checkpoint_interval{};
  bool sampled{}, captures{}, logprobs{}, live{};
};


class BatchScheduler {
 public:
  BatchScheduler(std::unique_ptr<ExecutionBackend> executor, BatchLimits limits, BatchCallbacks adapters);

  ~BatchScheduler();

  static bool computes(const BatchRequest& request);

  void observe_scheduled(BatchRequest& request);

  // Credit logical prompt progress once per request, including shared work.
  // Physical prefill work remains available separately in prefill_tokens.
  void observe_prompt(BatchRequest& request, std::uint32_t cursor, bool cached, bool shared = false);

  void observe_output(BatchRequest& request, std::size_t count);

  void observe_finished(BatchRequest& request, bool failed = false);

  bool retains(const BatchRequest& request) const;

  bool owner_ready(const BatchRequest& request) const;

  void prepare_request(BatchRequest& request);

  void remember_checkpoint(BatchRequest& request, kv_cache::CheckpointId checkpoint);

  static prefix_index::CheckpointSource source_kind(CheckpointSource source);

  std::optional<prefix_index::CheckpointSource> trigger_at(const BatchRequest& request, std::uint32_t cursor) const;

  void capture_work(BatchPrefixWork& work);

  std::uint32_t prefill_rows(const BatchPrefixWork& work) const;

  std::uint32_t checkpoint_distance(const BatchRequest& request, std::uint32_t cursor) const;

  std::vector<std::uint32_t> processed_history(const BatchRequest& request) const;

  void finalize_cache(BatchRequest& request);

  void commit_cache(BatchRequest& request);

  bool trim_provisional_pins(BatchRequest& request);

  void release_request_cache(BatchRequest& request);

  std::size_t submit(BatchRequest request);

  void forget(std::size_t index);

  bool has_pending() const;

  double wall_seconds() const;

  std::size_t retained_prompt_bytes() const;

  void poll(bool inflight);

  bool step();

  void stop_at_output(std::size_t index);

  // Disconnect cancellation may arrive during prefill. The request slot is
  // retained until that step commits its physical-work accounting.
  bool retire();

  void write_live_stats(std::ostream& out) const;

  void write_startup_capacity() const;

  nlohmann::json observability_snapshot(metrics::Snapshot& metrics,
                             const std::string& model, bool include_entries);

  std::string observed_cache_metrics;
  std::vector<BatchRequest> requests;
  std::uint64_t submitted{}, completed{}, cancelled{}, outputs{}, shared_tokens{}, cancelled_outputs{};
  std::uint64_t inflight_joins{}, late_prefix_replay_tokens{}, decode_tokens{}, decode_batches{};
  std::uint64_t cached_tokens{}, prefill_tokens{}, prefix_cache_hits{}, prefix_checkpoints{};
  double prefix_restore_seconds{}, prefix_capture_seconds{}, dependency_wait_seconds{};
  std::uint64_t mtp_cycles{}, mtp_proposed{}, mtp_accepted{}, mtp_rejected_cycles{}, mtp_verifier_rows{};
  std::uint64_t constraint_draft_downloads{}, constraint_mask_uploads{};
  std::uint64_t constraint_draft_bytes{}, constraint_mask_bytes{};
  double prefill_gpu_seconds{}, decode_gpu_seconds{}, mtp_draft_gpu_seconds{},
         mtp_verify_gpu_seconds{}, mtp_select_gpu_seconds{}, mtp_head_seconds{};
  std::size_t peak_batch{}, prefill_turn{};
  static void stop_waiting(BatchRequest& request);

  static void credit_prefix(BatchRequest& request, std::uint32_t cursor);

  void drop_work(std::size_t i);

  void register_requests();

  void cancel(std::size_t i, bool inflight = false);

  std::size_t runnable_count() const;

  std::uint32_t work_horizon(const BatchPrefixWork& work) const;
  void write_summary(std::ostream& summary) const;

 private:
  std::size_t add_work(std::unique_ptr<BatchPrefixWork> work);

  std::vector<std::size_t> ordered_works() const;

  void reject_work(std::size_t index, const std::exception& error);

  void synchronize_prefill();

  void initialize_output(BatchRequest& request);
  bool ready(BatchRequest& request);
  bool terminal(const BatchRequest& request);
  void sample_request(std::uint32_t row, BatchRequest& request);
  static void accept_constraint(BatchRequest& request, std::uint32_t token);
  void receive(const std::vector<std::size_t>& indices);
  void log_event(const std::string& event, const std::string& fields);
  template <class T> static std::string number(const char* key, T value) {
    return std::string("\"") + key + "\":" + std::to_string(value);
  }
  std::string named(std::size_t i) const;

  BatchLimits limits;
  BatchCallbacks callbacks;
  std::unique_ptr<MtpStats> mtp_stats;
  std::unique_ptr<MtpCapture> mtp_capture;
  std::uint64_t sequence_begin{};
  const std::uint32_t capacity, mtp_depth;
  std::unique_ptr<ExecutionBackend> owned_backend;
  ExecutionBackend& backend;
  const BatchMemoryPlan memory;
  const kv_cache::PoolConfig config;
  PersistentCacheManager cache;
  std::vector<std::size_t> active;
  std::vector<std::unique_ptr<BatchPrefixWork>> works;
  std::uint64_t next_work{};
  bool prefer_prefill{};
  std::uint64_t prefill_since_decode{};

  std::chrono::steady_clock::time_point started{};
  std::map<std::size_t, std::uint64_t> occupancy, verifier_occupancy, mtp_depth_occupancy;
};
} // namespace gewell::runtime
