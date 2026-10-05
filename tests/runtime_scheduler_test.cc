#include "gewell/runtime/scheduler.h"
#include "runtime_test_storage.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

namespace gewell::runtime {
namespace {
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
class TestBackend final : public ExecutionBackend {
 public:
  explicit TestBackend(std::size_t bytes = 8192) : config_(test::small_cache_config(bytes)) {}
  const BackendLimits& limits() const override { return supported; }
  BatchMemoryPlan memory_plan() const override { return {0, 32, 512, config_.gpu_bytes}; }
  kv_cache::PoolConfig cache_config() const override { return config_; }
  CacheStorageFactory cache_storage_factory() const override { return test::byte_storage_factory(); }
  void allocate_staging() override {}
  void initialize(PersistentCacheManager& owner, kv_cache::ExecutionId, const BatchLimits&) override { cache = &owner; }
  void initialize_outputs(bool, bool) override {}
  CompletionContext context() const override { return {}; }
  void wait() const override { ++waits; image_inflight = decode_inflight = prefill_inflight = false; }
  void synchronize(std::string_view) const override { image_inflight = decode_inflight = prefill_inflight = false; }
  bool ready(std::string_view) const override { image_inflight = prefill_inflight = false; return true; }
  TerminalState acquire_hidden() override {
    if (fail_hidden) throw std::bad_alloc();
    auto state = std::make_unique<std::array<std::uint32_t, 4>>();
    auto* value = state->data();
    states.emplace(value, std::move(state));
    return TerminalState(value);
  }
  void release_hidden(TerminalState state) override {
    require(!decode_inflight, "terminal state released during decode");
    require(!prefill_inflight, "terminal state released during prefill");
    if (state) require(states.erase(state.value) == 1, "double terminal release");
  }
  std::size_t occupied_hidden_bytes() const override { return states.size() * 16; }
  void copy_terminal(TerminalState to, TerminalState from, std::string_view) override { std::memcpy(to.value, from.value, 16); }
  void save_terminal(std::uint32_t row, TerminalState to) override { std::memcpy(to.value, terminal[row].data(), 16); }
  void begin_step(std::string_view) override {}
  void end_step(std::string_view) override {}
  float elapsed(std::string_view) const override { return 1; }
  std::vector<bool> prefill_batch(const std::vector<BatchPrefillInput>& inputs,
      const std::function<bool(std::size_t)>& continue_prefill) override {
    require(!image_live, "image features from preceding forward remained live");
    std::uint32_t total = 0;
    const bool has_images = std::any_of(inputs.begin(), inputs.end(),
        [](const auto& input) { return bool(input.image); });
    for (const auto& input : inputs) total += input.rows;
    require(total <= chunk_cap || (inputs.size() == 1 && inputs.front().image),
            "prefill exceeded selected bound");
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      if (!inputs[i].image) continue;
      if (on_image_begin) on_image_begin();
      if (!continue_prefill(i)) continue;
      if (!image_live) ++image_prefills;
      image_live = image_inflight = true;
      const auto& image = *inputs[i].image;
      require(image.begin == inputs[i].position && image.end - image.begin == inputs[i].rows,
              "prefill split an image");
      require(!image.pixels.empty(), "image bytes released before encoding");
      ++encoded_images;
      if (fail_image || encoded_images == fail_image_number)
        throw std::runtime_error("injected image prefill failure");
    }
    std::vector<bool> completed(inputs.size(), false);
    total = 0;
    std::size_t count = 0;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      completed[i] = !has_images || continue_prefill(i);
      if (completed[i]) { total += inputs[i].rows; ++count; }
    }
    if (count) {
      prefill_sizes.push_back(count);
      operations.emplace_back('P', total);
    }
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      if (!completed[i]) continue;
      const auto& [execution, tokens, position, rows, hidden, image] = inputs[i];
      cache->prepare_write(execution, position, rows, {});
      prefill_inflight = true;
      auto* state = static_cast<std::uint32_t*>(hidden.value);
      if (!position) state[0] = state[2] = 0;
      else require(state[1] == position, "packed prefill resumed the wrong cursor");
      for (std::uint32_t j = 0; j < rows; ++j) state[0] += tokens[j];
      if (image) state[0] += image->pixels.at(0) * ++state[2];
      state[1] = position + rows;
      prefill_rows += rows;
      if (fail_prefill) throw std::runtime_error("injected prefill failure");
    }
    return completed;
  }
  void release_image() override {
    require(!image_inflight, "image features were released before GPU completion");
    if (image_live) { ++image_releases; image_live = false; }
  }
  void prefix_head_step(kv_cache::ExecutionId, TerminalState hidden) override {
    operations.emplace_back('H', 1);
    ids[0] = static_cast<const std::uint32_t*>(hidden.value)[0] % 60 + 1;
  }
  void decode_batch(const std::vector<BatchDecodeInput>& inputs) override {
    require(!image_live, "decode overlapped in-flight image features");
    decode_sizes.push_back(inputs.size());
    operations.emplace_back('D', inputs.size());
    decode_depths.emplace_back(inputs.size(), 0);
    for (std::size_t row = 0; row < inputs.size(); ++row) {
      const auto& in = inputs[row];
      cache->prepare_write(in.execution, in.position, 1, {});
      ids[row] = in.token % 60 + 1;
      terminal[row] = {ids[row], in.position + 1, 0, 0};
    }
    decode_inflight = true;
    if (on_decode) on_decode();
  }
  void sample_batch_row(std::uint32_t row, const SamplingSettings& sampling,
                       std::mt19937_64& rng, const std::uint32_t*) override {
    if (sampling.temperature) ids[row] = std::uniform_int_distribution<std::uint32_t>(1, 60)(rng);
  }
  void summarize_batch_row(std::uint32_t, const SamplingSettings&, std::uint32_t, const std::uint32_t*) override {}
  void check_constraint_sampling() override {}
  BatchMtpOutcome run_batch_mtp(const std::vector<BatchMtpInput>& inputs) override {
    require(!image_live, "MTP overlapped in-flight image features");
    if (fail_mtp_execution) throw std::runtime_error("injected MTP execution failure");
    decode_sizes.push_back(inputs.size());
    operations.emplace_back('D', inputs.size());
    BatchMtpOutcome result;
    proposals = inputs;
    decode_depths.emplace_back();
    for (const auto& input : inputs) {
      if (input.capture) {
        input.capture->target_width = 2; input.capture->assistant_width = 1;
        input.capture->target_hidden = {1, 2};
        input.capture->assistant_hidden.assign(input.depth, 3);
        input.capture->draft_tokens.assign(input.depth, input.pending_token + 1);
        input.capture->scores.resize(input.depth);
      }
      decode_depths.back().push_back(input.depth);
      return_probability_flags.push_back(input.return_probabilities);
      require(input.uniforms.size() == 2 * input.depth + 1, "MTP draw block size");
      MtpOutcome output;
      if (failed_mtp_rows.count(result.requests.size())) {
        output.verification = {};
        output.error = "MTP cycle: injected request sampling failure";
        failed_mtp_executions.insert(input.execution);
      } else {
        output.verification = {input.depth, input.depth + 1, input.depth};
        for (std::uint32_t i = 0; i <= input.depth; ++i) output.tokens.push_back(input.pending_token + i + 1);
      }
      if (input.capture_next && output.error.empty()) {
        input.capture_next->width = 2;
        // Encode the committed terminal position; the collector must pair this
        // with the following round, not the round producing the probes.
        input.capture_next->hidden.assign(input.capture_next->layers.size() * 2,
            input.position + output.verification.output_count - 1);
      }
      result.requests.push_back(std::move(output));
    }
    failed_mtp_rows.clear();
    decode_inflight = true;
    if (on_decode) on_decode();
    return result;
  }
  void commit_batch_mtp(const std::vector<BatchMtpCommit>& inputs) override {
    require(!inputs.empty(), "empty MTP commit batch");
    commit_batch_sizes.push_back(inputs.size());
    for (const auto& input : inputs) {
      require(!failed_mtp_executions.count(input.execution), "failed MTP request committed KV");
      commits.push_back(input.count);
      cache->prepare_write(input.execution, input.position, input.count, {});
    }
  }
  void download_ids(std::size_t) override {}
  void download_logits(std::size_t) override {}
  void download_logprobs(std::size_t) override {}
  std::uint32_t output_id(std::size_t row) const override { return ids[row]; }
  const std::uint8_t* host_logits() const override { return nullptr; }
  const TokenLogprobs* host_logprobs() const override { return logprob_rows.data(); }
  std::size_t scratch_bytes() const override { return 0; }
  std::size_t output_bytes() const override { return 0; }
  std::size_t host_scratch_bytes() const override { return 0; }
  std::size_t sampling_scratch_bytes() const override { return 0; }
  BackendLimits supported{64,64,4,3,16,8,16,8,128,{63},8};
  kv_cache::PoolConfig config_;
  PersistentCacheManager* cache{};
  std::uint32_t chunk_cap{4}, prefill_rows{};
  bool fail_prefill{};
  bool fail_mtp_execution{};
  std::set<std::size_t> failed_mtp_rows;
  std::set<kv_cache::ExecutionId> failed_mtp_executions;
  bool fail_image{}, fail_hidden{}, image_live{};
  mutable bool image_inflight{}, decode_inflight{}, prefill_inflight{};
  std::function<void()> on_decode;
  std::function<void()> on_image_begin;
  std::uint32_t image_prefills{}, image_releases{}, encoded_images{}, fail_image_number{};
  mutable std::size_t waits{};
  std::map<void*,std::unique_ptr<std::array<std::uint32_t,4>>> states;
  std::array<std::uint32_t,4> ids{};
  std::array<std::array<std::uint32_t,4>,4> terminal{};
  std::array<TokenLogprobs, 16> logprob_rows{};
  std::vector<BatchMtpInput> proposals;
  std::vector<bool> return_probability_flags;
  std::vector<std::uint32_t> commits;
  std::vector<std::size_t> commit_batch_sizes;
  std::vector<std::size_t> decode_sizes;
  std::vector<std::size_t> prefill_sizes;
  std::vector<std::vector<std::uint32_t>> decode_depths;
  std::vector<std::pair<char, std::uint32_t>> operations;
};
BatchLimits limits(std::uint32_t depth = 0) {
  BatchLimits value;
  value.capacity = 3;
  value.mtp_depth = depth;
  value.plan_rows = value.prefill_chunk_tokens = value.prefill_batch_tokens = 4;
  value.max_horizon = 64;
  value.kv_bytes = 8192 + 512;
  value.max_requests = 8;
  value.sampled = true;
  return value;
}
BatchRequest request(const char* id, std::initializer_list<std::uint32_t> prompt, std::uint32_t count = 4) {
  BatchRequest value;
  value.id = id;
  value.prompt = std::make_shared<const std::vector<std::uint32_t>>(prompt);
  value.max_new_tokens = count;
  value.rng.seed(73);
  return value;
}
struct Output {
  std::map<std::string,std::vector<std::uint32_t>> tokens;
  std::map<std::string,BatchPhase> phases;
  std::map<std::string,std::uint32_t> processed;
  std::set<std::string> blocked;
  BatchCallbacks callbacks() {
    BatchCallbacks result;
    result.ready = [this](auto& r) { return !blocked.count(r.id); };
    result.start = [](auto&) {};
    result.emit = [this](auto& r,const auto* ids,std::size_t n,const void*,const TokenLogprobs*) {
      tokens[r.id].insert(tokens[r.id].end(), ids, ids+n);
    };
    result.finish = [this](auto& r) { phases[r.id] = r.phase; processed[r.id] = r.cursor; };
    result.reject = [](auto&,const auto&,auto) {};
    return result;
  }
};
void run(BatchScheduler& scheduler) {
  for (int turn = 0; scheduler.has_pending() && turn < 1000; ++turn) scheduler.step();
  require(!scheduler.has_pending(), "scheduler failed to make bounded progress");
}
void packed_prefill() {
  for (const auto cap : {4U, 8U}) {
    Output output;
    auto backend = std::make_unique<TestBackend>(); auto* device = backend.get();
    device->chunk_cap = cap;
    auto configured = limits(); configured.prefill_batch_tokens = configured.prefill_chunk_tokens = configured.plan_rows = cap;
    BatchScheduler scheduler(std::move(backend), configured, output.callbacks());
    scheduler.submit(request("a", {2,3,4}, 1));
    scheduler.submit(request("b", {5,6,7}, 1));
    scheduler.submit(request("c", {8,9,10,11,12}, 1));
    run(scheduler);
    require(device->prefill_sizes.front() == (cap == 4 ? 1U : 2U),
            "ready independent prompts did not share a bounded prefill forward");
    require(output.tokens["a"] == std::vector<std::uint32_t>{10} &&
                output.tokens["b"] == std::vector<std::uint32_t>{19} &&
                output.tokens["c"] == std::vector<std::uint32_t>{51},
            "packed prefill mixed histories or lost a tail segment");
    require(scheduler.prefill_tokens == 11 && device->prefill_rows == 11 && device->states.empty(),
            "packed prefill duplicated physical work or leaked terminal state");
  }
}
void packed_prefill_separate_limits() {
  Output output;
  auto backend = std::make_unique<TestBackend>(); auto* device = backend.get();
  device->chunk_cap = 16;
  auto configured = limits(); configured.capacity = 4;
  configured.prefill_batch_tokens = 16;
  BatchScheduler scheduler(std::move(backend), configured, output.callbacks());
  scheduler.submit(request("a", {2,3,4,5,6,7,8,9}, 1));
  scheduler.submit(request("b", {10,11,12,13,14,15,16,17}, 1));
  scheduler.submit(request("c", {18,19,20,21,22,23,24,25}, 1));
  scheduler.submit(request("d", {26,27,28,29,30,31,32,33}, 1));
  run(scheduler);
  require(device->prefill_sizes == std::vector<std::size_t>{4,4},
          "long prompts did not share forwards at the independent chunk cap");
  require(device->operations[0] == std::pair<char,std::uint32_t>{'P',16} &&
              device->operations[1] == std::pair<char,std::uint32_t>{'P',16},
          "four per-prompt chunks were not aggregated into one dense forward");
  require(scheduler.prefill_tokens == 32 && output.tokens["a"] == std::vector<std::uint32_t>{45} &&
              output.tokens["b"] == std::vector<std::uint32_t>{49} &&
              output.tokens["c"] == std::vector<std::uint32_t>{53} &&
              output.tokens["d"] == std::vector<std::uint32_t>{57},
          "independent chunk limits changed request histories or token accounting");
}
void packed_prefill_cancellation_and_failure() {
  for (const bool failure : {false, true}) {
    Output output;
    auto backend = std::make_unique<TestBackend>(); auto* device = backend.get();
    device->chunk_cap = 8;
    device->fail_prefill = failure;
    auto callbacks = output.callbacks();
    BatchScheduler* running = nullptr;
    bool cancelled = false;
    std::size_t rejected = 0;
    callbacks.poll = [&](bool inflight) {
      if (!failure && inflight && device->prefill_inflight && !cancelled) {
        cancelled = true;
        running->cancel(0, true);
      }
    };
    callbacks.reject = [&](auto&, const auto&, auto) { ++rejected; device->fail_prefill = false; };
    callbacks.finish = [](auto& r) { r.delivered = true; };
    auto configured = limits(); configured.live = true; configured.capacity = 2;
    configured.prefill_batch_tokens = configured.prefill_chunk_tokens = configured.plan_rows = 8;
    BatchScheduler scheduler(std::move(backend), configured, std::move(callbacks)); running = &scheduler;
    scheduler.submit(request("a", {2,3,4}, 1));
    scheduler.submit(request("b", {5,6}, 1));
    scheduler.submit(request("later", {7,8}, 1));
    run(scheduler);
    require(device->prefill_sizes.front() == 2 && output.tokens["a"].empty(),
            "cancelled or failed packed prefill emitted output");
    require(output.tokens["later"] == std::vector<std::uint32_t>{16} &&
                (failure ? output.tokens["b"].empty() : output.tokens["b"] == std::vector<std::uint32_t>{12}),
            "packed prefill failure/cancellation affected unrelated successful work");
    require(rejected == (failure ? 2U : 0U) && device->states.empty() &&
                device->cache->stats().execution_count == 0,
            "packed prefill cleanup lost rejection or retained in-flight resources");
    if (!failure) require(scheduler.prefill_tokens == 7, "cancelled packed rows lost physical-work accounting");
  }
}
void shared_prefix_and_seed(std::uint32_t budget = 0) {
  Output output;
  auto backend = std::make_unique<TestBackend>();
  auto* device = backend.get();
  auto configured = limits(); configured.prefill_budget_tokens = budget;
  BatchScheduler scheduler(std::move(backend), configured, output.callbacks());
  auto a = request("a",{2,3,4,5,6,7}); a.sampling.temperature = 0.8F;
  auto b = request("b",{2,3,4,5,6,7}); b.sampling.temperature = 0.8F;
  scheduler.submit(std::move(a)); scheduler.submit(std::move(b));
  run(scheduler);
  require(device->prefill_rows == 6, "shared prompt was recomputed");
  require(output.tokens.at("a") == output.tokens.at("b"), "per-request seeded draws changed with sharing");
  require(output.processed.at("a") == 9 && output.processed.at("b") == 9, "fed and emitted cursors conflated");
  require(device->states.empty() && device->cache->stats().execution_count == 0, "completed execution leaked");
}
void live_decode_accounting() {
  for (const auto depth : {0U, 3U}) {
    Output output;
    auto backend = std::make_unique<TestBackend>();
    auto configured = limits(depth); configured.prefill_budget_tokens = 16;
    BatchScheduler scheduler(std::move(backend), configured, output.callbacks());
    scheduler.submit(request("a", {2,3}, 9));
    scheduler.submit(request("b", {4,5}, 9));
    nlohmann::json previous;
    std::uint64_t observed_tokens = 0, observed_batches = 0;
    for (unsigned turn = 0; scheduler.has_pending() && turn < 100; ++turn) {
      scheduler.step();
      std::ostringstream stream;
      scheduler.write_live_stats(stream);
      const auto current = nlohmann::json::parse(stream.str());
      require(current.at("decode_tokens") == scheduler.decode_tokens &&
                  current.at("prefill_gpu_seconds") == scheduler.prefill_gpu_seconds &&
                  current.at("mtp_proposed") == scheduler.mtp_proposed &&
                  current.at("mtp_accepted") == scheduler.mtp_accepted &&
                  current.at("mtp_verifier_rows") == scheduler.mtp_verifier_rows,
              "live counters disagree with committed scheduler work");
      if (!previous.is_null()) {
        require(current.at("wall_seconds").get<double>() >= previous.at("wall_seconds").get<double>(),
                "live elapsed time moved backwards");
        observed_tokens += current.at("decode_tokens").get<std::uint64_t>() -
                           previous.at("decode_tokens").get<std::uint64_t>();
        observed_batches += current.at("decode_batches").get<std::uint64_t>() -
                            previous.at("decode_batches").get<std::uint64_t>();
      }
      previous = current;
    }
    require(!scheduler.has_pending() && observed_tokens == 16 && observed_batches > 0,
            "live interval accounting lost decode work or included prefill heads");
    require(scheduler.mtp_cycles == (depth ? 4U : 0U) &&
                scheduler.mtp_proposed == (depth ? 12U : 0U), "unexpected measured MTP work");
  }
}
void adaptive_decode_depth() {
  // Three ready requests shrink to two, then one. Remaining output limits
  // still trim individual rows below the chosen batch depth.
  for (const auto width : {0U, 8U, 0xffffffffU}) {
    Output output;
    auto backend = std::make_unique<TestBackend>(); auto* device = backend.get();
    auto configured = limits(3);
    configured.decode_width = width;
    configured.prefill_budget_tokens = 32;
    BatchScheduler scheduler(std::move(backend), configured, output.callbacks());
    scheduler.submit(request("a", {2,3}, 3));
    scheduler.submit(request("b", {4,5}, 6));
    scheduler.submit(request("c", {6,7}, 14));
    run(scheduler);
    const std::vector<std::vector<std::uint32_t>> expected = width == 8
        ? std::vector<std::vector<std::uint32_t>>{{1,1,1}, {2,3}, {3}, {2}}
        : std::vector<std::vector<std::uint32_t>>{{1,3,3}, {0,3}, {3}, {0}};
    require(device->decode_depths == expected, "decode depth did not follow the current batch or maximum");
    require(output.tokens["a"].size() == 3 && output.tokens["b"].size() == 6 &&
                output.tokens["c"].size() == 14, "adaptive depth lost or overproduced output");
    require(device->states.empty() && device->cache->stats().execution_count == 0,
            "adaptive depth leaked execution state");
  }
  // Width below batch size must select ordinary decode without unsigned
  // underflow, then resume MTP as the batch shrinks.
  {
    Output output;
    auto backend = std::make_unique<TestBackend>(); auto* device = backend.get();
    auto configured = limits(3); configured.decode_width = 2; configured.prefill_budget_tokens = 32;
    BatchScheduler scheduler(std::move(backend), configured, output.callbacks());
    scheduler.submit(request("a", {2,3}, 2));
    scheduler.submit(request("b", {4,5}, 3));
    scheduler.submit(request("c", {6,7}, 9));
    run(scheduler);
    require(device->decode_depths == std::vector<std::vector<std::uint32_t>>{
                {0,0,0}, {0,0}, {1}, {1}, {1}}, "ordinary/MTP transition used the wrong depth");
    require(output.tokens["c"] == std::vector<std::uint32_t>{14,15,16,17,18,19,20,21,22},
            "ordinary/MTP transition lost the pending token");
  }
  // A configured floor can exceed the target width; request tails and
  // ordinary-only requests remain correctness exceptions to that floor.
  {
    Output output;
    auto backend = std::make_unique<TestBackend>(); auto* device = backend.get();
    auto configured = limits(3); configured.decode_width = 1; configured.mtp_min_depth = 2;
    configured.prefill_budget_tokens = 32;
    BatchScheduler scheduler(std::move(backend), configured, output.callbacks());
    scheduler.submit(request("tail", {2,3}, 3));
    scheduler.submit(request("mtp", {4,5}, 8));
    auto ordinary = request("ordinary", {6,7}, 8); ordinary.ordinary_decode = true;
    scheduler.submit(std::move(ordinary)); run(scheduler);
    require(device->decode_depths.front() == std::vector<std::uint32_t>{1,2,0},
            "minimum depth overrode a request limit or ordinary fallback");
    require(output.tokens["tail"].size() == 3 && output.tokens["mtp"].size() == 8 &&
                output.tokens["ordinary"].size() == 8, "minimum-depth run lost output");
  }
  // Backpressured requests do not count toward the decode batch. When they
  // become runnable, depth decreases again without restarting the scheduler.
  {
    Output output; output.blocked = {"b", "c"};
    auto backend = std::make_unique<TestBackend>(); auto* device = backend.get();
    device->on_decode = [&] { output.blocked.clear(); };
    auto configured = limits(3); configured.decode_width = 6; configured.prefill_budget_tokens = 32;
    BatchScheduler scheduler(std::move(backend), configured, output.callbacks());
    for (const auto* id : {"a", "b", "c"}) scheduler.submit(request(id, {2,3}, 13));
    run(scheduler);
    require(device->decode_depths.front() == std::vector<std::uint32_t>{3} &&
                std::find(device->decode_depths.begin(), device->decode_depths.end(),
                          std::vector<std::uint32_t>{1,1,1}) != device->decode_depths.end(),
            "depth used admitted capacity or failed to decrease as requests became ready");
  }
  {
    Output output;
    auto configured = limits(3); configured.mtp_min_depth = 4; configured.decode_width = 8;
    bool rejected = false;
    try { BatchScheduler scheduler(std::make_unique<TestBackend>(), configured, output.callbacks()); }
    catch (const std::runtime_error& e) { rejected = std::string(e.what()).find("invalid scheduler limits") != std::string::npos; }
    require(rejected, "minimum depth greater than maximum was accepted");
  }
}
void cancellation_and_backpressure(std::uint32_t budget = 0) {
  Output output; output.blocked.insert("slow");
  auto backend = std::make_unique<TestBackend>(); auto* device = backend.get();
  BatchScheduler* running = nullptr; bool cancelled = false;
  auto callbacks = output.callbacks();
  callbacks.poll = [&](bool inflight) {
    if (inflight && !cancelled) { cancelled = true; running->cancel(0, true); }
  };
  auto configured = limits(); configured.prefill_budget_tokens = budget;
  BatchScheduler scheduler(std::move(backend), configured, std::move(callbacks)); running = &scheduler;
  scheduler.submit(request("producer",{2,3,4,5,6,7}));
  scheduler.submit(request("neighbor",{2,3,4,5,6,7}));
  scheduler.submit(request("slow",{2,3,4,5,6,7}));
  for (int n = 0; n < 30; ++n) scheduler.step();
  require(output.tokens["neighbor"].size() == 4, "slow reader blocked runnable neighbor");
  require(output.tokens["producer"].empty(), "cancelled producer emitted tokens");
  require(output.tokens["slow"].empty(), "backpressured request emitted tokens");
  output.blocked.clear(); run(scheduler);
  require(output.tokens["slow"] == output.tokens["neighbor"], "cancelled producer lost shared state");
  require(device->states.empty(), "cancelled terminal state leaked");
}
void mtp_commit_and_stop_fallback() {
  Output output;
  auto backend = std::make_unique<TestBackend>(); auto* device = backend.get();
  device->supported.stop_tokens = {8};
  auto configured = limits(3);
  configured.logprobs = true;
  BatchScheduler scheduler(std::move(backend), configured, output.callbacks());
  auto a = request("eos",{2,3},8); a.honor_eos = true; a.logprobs = true;
  auto b = request("ordinary",{2,3},5); b.ordinary_decode = true;
  auto c = request("mtp-no-logprobs", {2,3},5);
  scheduler.submit(std::move(a)); scheduler.submit(std::move(b));
  scheduler.submit(std::move(c)); run(scheduler);
  require(output.tokens["eos"] == std::vector<std::uint32_t>({6,7,8}), "MTP EOS inclusion changed");
  require(output.processed["eos"] == 4, "MTP committed beyond emitted EOS");
  require(output.tokens["ordinary"].size() == 5, "ordinary fallback lost output");
  require(output.tokens["mtp-no-logprobs"].size() == 5,
          "non-logprob MTP request lost output");
  require(scheduler.requests[1].mtp_proposed == 0, "stop fallback received speculative proposals");
  require(!device->commits.empty() && device->commits.front() == 2, "MTP failed to trim commit count");
  std::size_t batched_commits = 0;
  for (const auto size : device->commit_batch_sizes) batched_commits += size;
  require(batched_commits == device->commits.size(),
          "MTP commit batch accounting changed");
  require(std::any_of(device->return_probability_flags.begin(),
                      device->return_probability_flags.end(),
                      [](bool value) { return value; }),
          "MTP logprob request did not require probability rows");
  require(std::any_of(device->return_probability_flags.begin(),
                      device->return_probability_flags.end(),
                      [](bool value) { return !value; }),
          "MTP request without logprobs required probability rows");
}
void windowed_mtp_statistics() {
  struct Directory {
    std::filesystem::path path;
    Directory() {
      auto pattern = (std::filesystem::temp_directory_path() / "gewell-scheduler-stats-XXXXXX").string();
      require(::mkdtemp(pattern.data()) != nullptr, "create statistics test directory");
      path = pattern;
    }
    ~Directory() { std::error_code error; std::filesystem::remove_all(path, error); }
  } directory;
  const auto path = directory.path / "stats.jsonl";
  std::map<std::string, std::vector<std::uint32_t>> reference;
  std::map<std::string, std::mt19937_64> reference_rng;
  for (const bool enabled : {false, true}) {
    Output output;
    auto backend = std::make_unique<TestBackend>();
    backend->supported.stop_tokens = {8};
    auto configured = limits(3); configured.prefill_budget_tokens = 16;
    if (enabled) {
      configured.mtp_stats_path = path;
      configured.mtp_capture = {directory.path / "capture", 1, 100};
    }
    configured.mtp_stats_window = 2;
    BatchScheduler scheduler(std::move(backend), configured, output.callbacks());
    auto eos = request("eos", {2,3}, 8); eos.honor_eos = true;
    auto ordinary = request("ordinary", {2,3}, 5); ordinary.ordinary_decode = true;
    scheduler.submit(std::move(eos)); scheduler.submit(std::move(ordinary));
    scheduler.submit(request("long", {2,3}, 14)); run(scheduler);
    if (!enabled) {
      reference = output.tokens;
      for (const auto& r : scheduler.requests) reference_rng[r.id] = r.rng;
      continue;
    }
    for (const auto& r : scheduler.requests)
      require(r.rng == reference_rng[r.id], "capture consumed inference RNG");
    {
      std::ifstream captures(directory.path / "capture" / "samples.jsonl");
      bool final_round = false;
      for (std::string line; std::getline(captures, line);) {
        const auto record = nlohmann::json::parse(line);
        if (record["event"] != "mtp_capture") continue;
        require(record["id"] != "ordinary", "capture recorded depth zero");
        require(record["cycle"] > 0 && record["probe_source_cycle"].get<unsigned>() + 1 == record["cycle"],
                "probe was labeled with its own round");
        std::ifstream binary(directory.path / "capture" / "hidden.bf16", std::ios::binary);
        binary.seekg(record["byte_offset"].get<std::size_t>() + 2 * (2 + record["depth"].get<unsigned>()));
        std::uint16_t probe{}; binary.read(reinterpret_cast<char*>(&probe), 2);
        require(probe + 1 == record["position"], "probe used a rejected or misaligned target row");
        final_round |= record["id"] == "long" && record["cycle"] > 0 && record["previous_depth"] == 3;
      }
      require(final_round, "capture lost per-request history");
    }
    require(output.tokens == reference, "collecting statistics changed decoded output");
    std::ifstream file(path);
    std::map<std::string, std::uint64_t> cycles, proposed, accepted, emitted, windows;
    std::map<std::string, nlohmann::json> endings;
    std::set<std::uint32_t> batches;
    bool ordinary_verifier = false, ordinary_decode = false, clipped_eos = false;
    for (std::string line; std::getline(file, line);) {
      const auto record = nlohmann::json::parse(line);
      if (record["event"] == "mtp_request_end") endings[record["id"]] = record;
      if (record["event"] != "mtp_window") continue;
      const auto id = record["id"].get<std::string>();
      require(record["window"] == windows[id]++, "request window numbering changed");
      require(record["output_begin"] == emitted[id] + 1, "decode windows lost their output offset");
      for (const auto& group : record["groups"]) {
        const auto depth = group["depth"].get<std::uint32_t>();
        const auto count = group["cycles"].get<std::uint64_t>();
        batches.insert(group["batch"].get<std::uint32_t>());
        emitted[id] += group["emitted_tokens"].get<std::uint64_t>();
        require(std::abs(group["gpu_seconds"].get<double>() - count * 0.001) < 1e-9,
                "statistics divided shared batch GPU time among requests");
        require(group["failed_cycles"] == 0, "healthy cycle was marked as failed");
        ordinary_verifier |= depth == 0 && group["mtp"] == true;
        ordinary_decode |= depth == 0 && group["mtp"] == false;
        if (id == "eos") clipped_eos = depth == 3 && group["accepted"][2] == 1;
        if (!depth) continue;
        cycles[id] += count; proposed[id] += depth * count;
        for (std::size_t i = 0; i < group["accepted"].size(); ++i)
          accepted[id] += i * group["accepted"][i].get<std::uint64_t>();
      }
      require(record["output_end"] == emitted[id] + 1, "decode windows lost emitted tokens");
    }
    require(clipped_eos && ordinary_verifier && ordinary_decode && batches.size() > 1,
            "statistics lost EOS clipping, mixed ordinary rows, tails, or actual batch changes");
    for (const auto& r : scheduler.requests) {
      require(cycles[r.id] == r.mtp_cycles && proposed[r.id] == r.mtp_proposed && accepted[r.id] == r.mtp_accepted,
              "window histogram disagrees with committed request MTP totals");
      require(emitted[r.id] + 1 == output.tokens[r.id].size() && endings[r.id]["status"] == "complete" &&
              endings[r.id]["output_tokens"] == output.tokens[r.id].size(),
              "statistics lost final output or request completion");
    }
    require(windows["eos"] == 1, "final partial decode window was dropped");
  }
  // Resume either or both sinks. Their sequence floors can differ because
  // capture samples only some requests and either output can be disabled.
  std::uint64_t next_sequence = 3;
  for (const auto [stats, capture] : {std::pair{true, true}, {false, true}, {true, true}, {true, false}, {true, true}}) {
    Output output;
    auto configured = limits(3);
    if (stats) configured.mtp_stats_path = path;
    if (capture) configured.mtp_capture = {directory.path / "capture", 1, 100};
    BatchScheduler scheduler(std::make_unique<TestBackend>(), configured, output.callbacks());
    scheduler.submit(request("long", {4,5}, 14)); run(scheduler);
    require(scheduler.requests[0].accepted_order == next_sequence && scheduler.submitted == 1,
            "resuming outputs reused a sequence or changed request counts");
    for (const auto& sink : {stats ? path : std::filesystem::path{},
                            capture ? directory.path / "capture" / "samples.jsonl" : std::filesystem::path{}}) {
      if (sink.empty()) continue;
      std::ifstream file(sink);
      nlohmann::json last;
      for (std::string line; std::getline(file, line);) last = nlohmann::json::parse(line);
      require(last["sequence"] == next_sequence, "capture and statistics request identities disagree");
    }
    ++next_sequence;
  }
  // The last selected source round must still be labeled after selection hits
  // its cap. No additional sources or samples may be collected afterward.
  {
    Output output;
    auto configured = limits(3);
    configured.mtp_capture = {directory.path / "capture-cap", 1, 1};
    BatchScheduler scheduler(std::make_unique<TestBackend>(), configured, output.callbacks());
    scheduler.submit(request("capped", {2,3}, 14)); run(scheduler);
    std::ifstream captures(directory.path / "capture-cap" / "samples.jsonl");
    unsigned count = 0;
    for (std::string line; std::getline(captures, line);) {
      const auto record = nlohmann::json::parse(line);
      if (record["event"] != "mtp_capture") continue;
      ++count;
      require(record["cycle"] == 1 && record["probe_source_cycle"] == 0,
              "capture cap discarded the pending label or used the source outcome");
    }
    require(count == 1, "capture cap lost or added a sample");
  }
  // Destruction flushes a partial window for a request that never retired.
  const auto interrupted_path = directory.path / "interrupted.jsonl";
  {
    Output output;
    auto backend = std::make_unique<TestBackend>(); auto* device = backend.get();
    auto configured = limits(3); configured.mtp_stats_path = interrupted_path;
    BatchScheduler scheduler(std::move(backend), configured, output.callbacks());
    scheduler.submit(request("interrupted", {2,3}, 14));
    for (int n = 0; device->decode_sizes.empty() && n < 10; ++n) scheduler.step();
    require(device->decode_sizes.size() == 1, "interruption fixture did not decode one cycle");
  }
  std::ifstream interrupted(interrupted_path);
  nlohmann::json last;
  for (std::string line; std::getline(interrupted, line);) last = nlohmann::json::parse(line);
  require(last["status"] == "interrupted" && last["decode_cycles"] == 1 && last["windows"] == 1,
          "scheduler destruction lost a partial statistics window");
}
void mtp_failure_isolation() {
  for (const std::set<std::size_t> failed : {
           std::set<std::size_t>{0}, {1}, {2}, {0, 1, 2}}) {
    Output output;
    auto backend = std::make_unique<TestBackend>(); auto* device = backend.get();
    device->failed_mtp_rows = failed;
    BatchScheduler* running = nullptr;
    metrics::Snapshot metrics;
    metrics.mtp_accepted_per_position.resize(3);
    std::set<std::string> rejected;
    auto callbacks = output.callbacks();
    callbacks.metrics = &metrics;
    callbacks.reject = [&](auto& r, const auto& error, auto reason) {
      require(reason == BatchFailure::execution && error == "MTP cycle: injected request sampling failure",
              "MTP request error changed or was reported as capacity failure");
      require(rejected.insert(r.id).second, "MTP request rejected twice");
      running->observe_finished(r, true);
    };
    callbacks.finish = [&](auto& r) { running->observe_finished(r); r.delivered = true; };
    callbacks.done = [&](auto& r) { running->observe_finished(r); };
    auto configured = limits(3); configured.live = true;
    configured.prefill_budget_tokens = 16;
    BatchScheduler scheduler(std::move(backend), configured, std::move(callbacks)); running = &scheduler;
    for (const auto* id : {"a", "b", "c"}) scheduler.submit(request(id, {2,3}, 5));
    run(scheduler);
    require(device->decode_sizes == std::vector<std::size_t>{3}, "isolation fixture did not share one MTP batch");
    for (unsigned row = 0; row < 3; ++row) {
      const std::string id(1, 'a' + row);
      require(output.tokens[id].size() == (failed.count(row) ? 1U : 5U),
              "failed MTP row emitted tokens or lost a healthy neighbor");
      require(rejected.count(id) == failed.count(row), "wrong MTP request was rejected");
      require(scheduler.requests[row].cursor == (failed.count(row) ? 2U : 6U),
              "failed MTP request advanced KV or healthy commit used the wrong row");
    }
    require(metrics.failed == failed.size() && metrics.aborted == 0 &&
                metrics.success_length == 3 - failed.size(), "MTP failure metrics double-counted or lost a request");
    require(scheduler.cancelled == failed.size() && device->states.empty() &&
                device->cache->stats().execution_count == 0, "MTP failure leaked execution or hidden state");
    scheduler.submit(request("after", {3,4}, 5));
    run(scheduler);
    require(output.tokens["after"].size() == 5 && rejected.size() == failed.size(),
            "MTP failure prevented a subsequent request from completing");
    require(device->states.empty() && device->cache->stats().execution_count == 0,
            "request following MTP failure leaked state");
  }
  // Offline runs remain fail-fast; execution/CUDA errors still propagate in live mode.
  for (const bool execution_failure : {false, true}) {
    Output output;
    auto backend = std::make_unique<TestBackend>(); auto* device = backend.get();
    device->fail_mtp_execution = execution_failure;
    device->failed_mtp_rows = {0};
    auto configured = limits(3); configured.live = execution_failure;
    BatchScheduler scheduler(std::move(backend), configured, output.callbacks());
    scheduler.submit(request("failed", {2,3}, 5));
    bool threw = false;
    try { run(scheduler); }
    catch (const std::runtime_error& error) {
      threw = std::string(error.what()) == (execution_failure ? "injected MTP execution failure"
          : "MTP cycle: injected request sampling failure");
    }
    require(threw && device->commits.empty(), "fatal MTP failure was swallowed or committed KV");
  }
}
void logical_prefill_budgets() {
  for (const auto depth : {0U, 3U}) {
    std::map<std::string, std::vector<std::uint32_t>> reference;
    for (const auto budget : {0U, 1U, 4U, 16U, 0xffffffffU}) {
      Output output;
      auto backend = std::make_unique<TestBackend>(16384); auto* device = backend.get();
      auto configured = limits(depth);
      configured.kv_bytes = 16384 + 512;
      configured.prefill_budget_tokens = budget;
      BatchScheduler scheduler(std::move(backend), configured, output.callbacks());
      scheduler.submit(request("hot", {1,2}, 56));
      for (int turn = 0; device->decode_sizes.empty() && turn < 20; ++turn) scheduler.step();
      require(!device->decode_sizes.empty(), "budget delayed decode without prefill work");
      device->operations.clear();
      for (const auto* id : {"a", "b"}) {
        auto next = request(id, {2});
        next.prompt = std::make_shared<const std::vector<std::uint32_t>>(24, id[0] == 'a' ? 2 : 3);
        scheduler.submit(std::move(next));
      }
      run(scheduler);
      std::uint64_t work = 0, largest_burst = 0;
      std::uint32_t physical_rows = 0;
      for (const auto& [kind, size] : device->operations) {
        if (kind == 'D') {
          largest_burst = std::max(largest_burst, work);
          // The budget never splits a physical chunk. A final chunk may
          // overshoot the target by at most chunk_cap - 1 text tokens.
          const auto bound = budget ? std::uint64_t(budget) + device->chunk_cap - 1 : device->chunk_cap;
          require(work <= bound, "logical budget failed to bound the prefill burst");
          work = 0;
        } else {
          work += size;
          if (kind == 'P') {
            require(size <= device->chunk_cap, "logical budget enlarged a microbatch");
            physical_rows += size;
          }
        }
      }
      require(physical_rows == 48, "logical budget skipped or repeated prefill tokens");
      if (budget > device->chunk_cap)
        require(largest_burst > device->chunk_cap, "logical budget did not group microbatches");
      if (reference.empty()) reference = output.tokens;
      require(output.tokens == reference, "logical budget changed request-local output/RNG");
      require(device->states.empty() && device->cache->stats().execution_count == 0,
              "logical budget leaked executions or terminal state");
    }
  }
}

void logical_budget_under_kv_pressure() {
  for (const auto depth : {0U, 3U}) {
    Output output;
    auto backend = std::make_unique<TestBackend>(2048); auto* device = backend.get();
    auto configured = limits(depth);
    configured.kv_bytes = 2048 + 512;
    configured.prefill_budget_tokens = 64;
    BatchScheduler scheduler(std::move(backend), configured, output.callbacks());
    scheduler.submit(request("a", {2,2,2,2,2,2,2,2}));
    scheduler.submit(request("b", {3,3,3,3,3,3,3,3}));
    run(scheduler);
    require(output.tokens["a"].size() == 4 && output.tokens["b"].size() == 4,
            "logical budget waited for prefills that could not be admitted");
    require(*std::max_element(device->decode_sizes.begin(), device->decode_sizes.end()) == 1,
            "KV pressure fixture unexpectedly held both requests");
    require(device->states.empty() && device->cache->stats().execution_count == 0,
            "logical budget pressure cleanup leaked");
  }
}

void failure_cleanup() {
  Output output; auto backend = std::make_unique<TestBackend>();
  backend->fail_prefill = true;
  BatchScheduler scheduler(std::move(backend), limits(), output.callbacks());
  scheduler.submit(request("failed",{2,3,4,5,6}));
  bool failed = false;
  try { scheduler.step(); } catch (const std::runtime_error&) { failed = true; }
  require(failed && output.tokens.empty(), "failed partial prefill became visible");
}
BatchRequest image_request(const char* id, std::uint8_t pixel = 1, std::uint32_t count = 3) {
  auto value = request(id, {2,3,4,5,6,7}, count);
  auto image = std::make_shared<ImageInput>();
  image->pixels = {pixel}; image->positions = {0};
  image->padded_patch_rows = 1; image->begin = 1; image->end = 3;
  value.images.push_back(std::move(image));
  return value;
}
void packed_image_prefill() {
  // Mixed whole-image/text cohorts, including cancellation before a later tower
  // pass and failure after an earlier image's features have been retained.
  for (int mode = 0; mode < 4; ++mode) {
    Output output;
    auto backend = std::make_unique<TestBackend>(); auto* device = backend.get();
    auto configured = limits(); configured.live = true;
    configured.prefill_batch_tokens = device->chunk_cap = mode == 3 ? 4 : 8;
    auto callbacks = output.callbacks();
    auto finish = callbacks.finish;
    callbacks.finish = [finish](auto& r) { finish(r); r.delivered = true; };
    BatchScheduler* running = nullptr;
    bool cancelled = false;
    std::size_t rejected = 0;
    callbacks.poll = [&](bool inflight) {
      if (mode == 1 && inflight && device->encoded_images == 1 && !cancelled) {
        cancelled = true;
        running->cancel(1, true);
      }
    };
    callbacks.reject = [&](auto&, const auto&, auto) { ++rejected; };
    if (mode == 2) device->fail_image_number = 2;
    BatchScheduler scheduler(std::move(backend), configured, std::move(callbacks)); running = &scheduler;
    std::vector<std::weak_ptr<const ImageInput>> lifetimes;
    for (int i = 0; i < 2; ++i) {
      auto input = image_request(i ? "b" : "a", i + 1, 1);
      const auto rows = mode == 3 && i == 0 ? 6U : 2U;
      input.prompt = std::make_shared<const std::vector<std::uint32_t>>(rows, i + 2);
      auto image = std::make_shared<ImageInput>(*input.images.front());
      image->begin = 0; image->end = rows;
      input.images = {image}; input.controls.mode = CacheRequestMode::reuse_only;
      lifetimes.push_back(image);
      scheduler.submit(std::move(input));
    }
    auto text = request("text", {4,4,4,4}, 1); text.controls.mode = CacheRequestMode::reuse_only;
    scheduler.submit(std::move(text));
    run(scheduler);
    if (mode == 2) {
      require(rejected == 3 && output.tokens.empty(), "failed packed image cohort leaked partial outputs");
      scheduler.submit(request("after", {5,6}, 1)); run(scheduler);
      require(output.tokens["after"] == std::vector<std::uint32_t>{12}, "image failure blocked later work");
    } else {
      require(output.tokens["a"] == std::vector<std::uint32_t>{mode == 3 ? 14U : 6U} &&
                  output.tokens["text"] == std::vector<std::uint32_t>{17},
              "image packing mixed feature buffers or text history");
      require(mode == 1 ? output.tokens["b"].empty() : output.tokens["b"] == std::vector<std::uint32_t>{9},
              "packed image cancellation affected the wrong request");
      require(device->prefill_sizes.front() == (mode == 3 ? 1U : mode == 1 ? 2U : 3U),
              "whole images and text did not pack within the physical cap");
      require(device->encoded_images == (mode == 1 ? 1U : 2U), "cancelled image still ran the tower");
      require(scheduler.prefill_tokens == (mode == 3 ? 12U : mode == 1 ? 6U : 8U),
              "packed image token accounting lost processed rows");
    }
    require(std::all_of(lifetimes.begin(), lifetimes.end(), [](const auto& image) { return image.expired(); }) &&
                device->image_prefills == device->image_releases && device->states.empty() &&
                device->cache->stats().execution_count == 0,
            "packed image completion retained input, feature lifetime, hidden state or KV");
  }
}
void logical_budget_failed_prefill() {
  for (const auto depth : {0U, 3U}) for (const bool image : {false, true}) {
    Output output;
    auto backend = std::make_unique<TestBackend>(); auto* device = backend.get();
    auto configured = limits(depth); configured.live = true; configured.prefill_budget_tokens = 64;
    auto callbacks = output.callbacks();
    auto finish = callbacks.finish;
    callbacks.finish = [finish](auto& r) { finish(r); r.delivered = true; };
    bool failed = false, cancel_image = false, cancelled = false;
    std::size_t bad = 0;
    BatchScheduler* running = nullptr;
    callbacks.reject = [&](auto&, const auto&, auto) { failed = true; };
    callbacks.poll = [&](bool inflight) {
      if (inflight && cancel_image && !cancelled) {
        cancelled = true;
        running->cancel(bad, true);
      }
    };
    BatchScheduler scheduler(std::move(backend), configured, std::move(callbacks)); running = &scheduler;
    scheduler.submit(request("hot", {1,2}, 32));
    for (int turn = 0; device->decode_sizes.empty() && turn < 20; ++turn) scheduler.step();
    require(!device->decode_sizes.empty(), "failure fixture did not prepare a decoder");
    device->fail_prefill = !image;
    if (image) device->on_image_begin = [&] { cancel_image = true; };
    bad = scheduler.submit(image ? image_request("bad") : request("bad", {8,8,8,8}));
    for (int turn = 0; !failed && !cancelled && turn < 20; ++turn) {
      const auto decodes = device->decode_sizes.size();
      scheduler.step();
      if (failed || cancelled)
        require(device->decode_sizes.size() > decodes,
                "failed/zero-progress prefill incorrectly deferred a ready decoder");
    }
    require(failed || cancelled, "prefill failure/cancellation was not injected");
    device->fail_prefill = false;
    device->on_image_begin = {};
    run(scheduler);
    require(output.tokens["bad"].empty() && output.tokens["hot"].size() == 32,
            "failed prefill emitted output or stopped its neighbor");
    require(device->states.empty() && device->cache->stats().execution_count == 0,
            "failed budgeted prefill leaked state");
  }
}

void images_share_and_mix_with_text(std::uint32_t budget = 0) {
  Output output;
  auto backend = std::make_unique<TestBackend>(); auto* device = backend.get();
  auto callbacks = output.callbacks();
  std::vector<std::string> started;
  callbacks.start = [&](auto& r) { started.push_back(r.id); };
  auto finish = callbacks.finish;
  callbacks.finish = [finish](auto& r) { finish(r); r.delivered = true; };
  auto live = limits(); live.live = true; live.prefill_budget_tokens = budget;
  BatchScheduler scheduler(std::move(backend), live, std::move(callbacks));
  scheduler.submit(request("before",{2,3,4,5,6,7},3));
  scheduler.submit(image_request("image1",1));
  scheduler.submit(image_request("image2",3));
  scheduler.submit(request("after",{2,3,4,5,6,7},3));
  run(scheduler);
  std::sort(started.begin(), started.end());
  require(started == std::vector<std::string>({"after","before","image1","image2"}),
          "shared image request initialized output more than once");
  require(device->encoded_images == 2 && device->image_prefills == device->image_releases,
          "image feature lifetime imbalance");
  require(output.tokens["image1"] != output.tokens["image2"], "different pixels reused equal placeholder KV");
  require(output.tokens["before"] == output.tokens["after"], "image polluted later text prefix reuse");
  require(scheduler.requests[1].shared_tokens == 1 && scheduler.requests[2].shared_tokens == 1,
          "image request failed to share only its safe text prefix");
  require(scheduler.prefill_tokens == 16 && scheduler.prefix_checkpoints > 1,
          "image prefill did not capture or share safe prefixes");
  require(std::any_of(device->decode_sizes.begin(), device->decode_sizes.end(), [](auto size) { return size > 1; }),
          "image requests failed to join decode batches");
  require(device->states.empty() && device->cache->stats().execution_count == 0, "image execution leaked");
}
void shared_image_cancellation_and_fairness() {
  for (const auto depth : {0U, 3U}) {
    Output output;
    auto backend = std::make_unique<TestBackend>(32768); auto* device = backend.get();
    auto live = limits(depth); live.live = true; live.capacity = 4; live.kv_bytes = 32768 + 512;
    const auto input = [](const char* id, std::uint8_t second_pixel = 1) {
      auto value = image_request(id,1,6);
      value.controls.mode = CacheRequestMode::reuse_only;
      value.prompt = std::make_shared<const std::vector<std::uint32_t>>(24,2);
      auto first = std::make_shared<ImageInput>(*value.images.front()); first->end = 7;
      auto second = std::make_shared<ImageInput>(*first);
      second->begin = 14; second->end = 20; second->pixels = {second_pixel};
      value.images = {first,second};
      return value;
    };
    auto callbacks = output.callbacks();
    auto finish = callbacks.finish;
    callbacks.finish = [finish](auto& r) { finish(r); r.delivered = true; };
    callbacks.reject = [](auto&, const auto& error, auto) { throw std::runtime_error(error); };
    std::vector<nlohmann::json> trace;
    callbacks.trace = [&](const auto& event, const auto& fields) {
      auto value = nlohmann::json::parse("{" + fields + "}"); value["event"] = event;
      trace.push_back(std::move(value));
    };
    BatchScheduler* running = nullptr;
    bool cancelled = false;
    callbacks.poll = [&](bool inflight) {
      if (!inflight || device->encoded_images != 1 || cancelled) return;
      cancelled = true;
      running->cancel(0,true);
      running->submit(input("late"));
    };
    BatchScheduler scheduler(std::move(backend), live, std::move(callbacks)); running = &scheduler;
    auto producer = input("producer");
    const std::weak_ptr<const ImageInput> producer_tail = producer.images[1];
    scheduler.submit(std::move(producer));
    scheduler.submit(input("follower"));
    scheduler.submit(input("changed",2));
    auto text = request("text",{3,5},32); text.controls.mode = CacheRequestMode::reuse_only;
    scheduler.submit(std::move(text));
    run(scheduler);
    require(cancelled && output.tokens["producer"].empty() && scheduler.cancelled == 1,
            "shared image producer cancellation did not remain isolated");
    require(output.tokens["follower"] == output.tokens["late"] && output.tokens["follower"].size() == 6 &&
            output.tokens["follower"].front() == 52 && output.tokens["changed"].front() == 54,
            "image sharing lost features, late follower state, or changed-image identity");
    require(device->encoded_images == 3, "shared image blocks were re-encoded after producer cancellation");
    require(std::any_of(trace.begin(),trace.end(),[](const auto& event) {
      return event["event"] == "decode" && !event["image_requests"].empty() &&
             event["image_requests"].size() < event["requests"].size();
    }), "image and text requests never shared a decode batch");
    std::size_t first_image = trace.size(), last_image = 0;
    for (std::size_t i = 0; i < trace.size(); ++i) if (trace[i]["event"] == "image_encode") {
      first_image = std::min(first_image,i); last_image = i;
    }
    require(first_image < last_image && std::any_of(trace.begin()+first_image,trace.begin()+last_image,
        [](const auto& event) { return event["event"] == "decode"; }),
        "image prefill did not yield to active decoders");
    require(producer_tail.expired() && device->states.empty() && device->cache->stats().gpu.used == 0,
            "shared image cancellation retained input, hidden state, or KV");
  }
}
void image_mtp_and_admission_validation() {
  Output output;
  auto backend = std::make_unique<TestBackend>(); auto* device = backend.get();
  device->supported.stop_tokens = {31};
  auto callbacks = output.callbacks();
  auto finish = callbacks.finish;
  callbacks.finish = [finish](auto& r) { finish(r); r.delivered = true; };
  auto live = limits(3); live.live = true;
  BatchScheduler scheduler(std::move(backend), live, std::move(callbacks));
  auto image = image_request("image",1,8); image.honor_eos = true;
  image.controls.mode = CacheRequestMode::reuse_only;
  scheduler.submit(std::move(image)); run(scheduler);
  require(output.tokens["image"] == std::vector<std::uint32_t>({29,30,31}), "image MTP EOS semantics changed");
  require(output.processed["image"] == 8 && scheduler.mtp_proposed == 3,
          "image MTP lost state or committed past EOS");
  require(device->cache->stats().gpu.used == 0 && scheduler.prefix_checkpoints == 0,
          "completed image retained cache state");
  auto over_limit = image_request("too_long");
  auto invalid_span = std::make_shared<ImageInput>(*over_limit.images.front());
  invalid_span->end = 9;
  over_limit.images.front() = std::move(invalid_span);
  auto split_image = image_request("split_image"); split_image.checkpoint_offsets = {2};
  auto invalid_operation = image_request("invalid_operation"); invalid_operation.operation = Operation::stats;
  for (auto* invalid : {&over_limit, &split_image, &invalid_operation}) {
    bool failed = false;
    try { scheduler.submit(std::move(*invalid)); } catch (const std::runtime_error&) { failed = true; }
    require(failed, "invalid image request reached admission");
  }
  require(scheduler.submitted == 1 && device->cache->stats().execution_count == 0,
          "invalid image request acquired runtime state");
}
void image_cancellation_and_failure() {
  for (int failure = 0; failure < 3; ++failure) {
    Output output;
    auto backend = std::make_unique<TestBackend>(); auto* device = backend.get();
    device->fail_image = failure == 1;
    device->fail_hidden = failure == 2;
    BatchScheduler* running = nullptr; bool cancelled = false;
    auto callbacks = output.callbacks();
    callbacks.poll = [&](bool inflight) {
      if (failure == 0 && inflight && device->image_live && !cancelled) {
        cancelled = true;
        running->cancel(0, true);
        require(device->image_live && device->image_inflight, "cancellation released in-flight features");
      }
    };
    callbacks.finish = [&](auto& r) { r.delivered = true; };
    std::size_t rejected = 0;
    callbacks.reject = [&](auto&,const auto&,auto) { ++rejected; };
    auto live = limits(); live.live = true;
    BatchScheduler scheduler(std::move(backend), live, std::move(callbacks)); running = &scheduler;
    auto input = image_request("image");
    const std::weak_ptr<const ImageInput> lifetime = input.images.front();
    scheduler.submit(std::move(input));
    run(scheduler);
    device->fail_image = device->fail_hidden = false;
    scheduler.submit(request("text", {2,3,4,5,6,7},3));
    run(scheduler);
    require(output.tokens["image"].empty() && output.tokens["text"].size() == 3,
            "cancelled/failed image emitted output or blocked following text");
    require(rejected == (failure != 0) && scheduler.cancelled == 1, "image failure was not isolated");
    require(!device->image_live && device->image_prefills == device->image_releases && lifetime.expired(),
            "failed image retained device or host input");
    require(device->states.empty() && device->cache->stats().execution_count == 0,
            "failed image leaked pool reservation or terminal state");
  }
}
void image_decode_cancellation() {
  for (const auto depth : {0U, 3U}) {
    Output output;
    auto backend = std::make_unique<TestBackend>(); auto* device = backend.get();
    auto callbacks = output.callbacks();
    auto finish = callbacks.finish;
    callbacks.finish = [finish](auto& r) { finish(r); r.delivered = true; };
    auto live = limits(depth); live.live = true;
    BatchScheduler scheduler(std::move(backend), live, std::move(callbacks));
    device->on_decode = [&] {
      require(!device->image_live, "decode retained no-longer-used image features");
      scheduler.cancel(0, true);
      require(device->cache->stats().execution_count == 1 && !device->states.empty(),
              "decode cancellation freed in-flight execution");
    };
    auto image = image_request("image",1,8);
    image.controls.mode = CacheRequestMode::reuse_only;
    const std::weak_ptr<const ImageInput> lifetime = image.images.front();
    scheduler.submit(std::move(image)); run(scheduler);
    require(output.tokens["image"] == std::vector<std::uint32_t>({29}),
            "cancelled image decode exposed ordinary or speculative output");
    require(scheduler.cancelled == 1 && device->states.empty() && lifetime.expired() &&
            device->cache->stats().execution_count == 0 && device->cache->stats().gpu.used == 0,
            "image decode cancellation retained request state");
  }
}
void reusable_image_checkpoints() {
  Output output;
  auto backend = std::make_unique<TestBackend>(32768); auto* device = backend.get();
  auto live = limits(); live.live = true; live.max_requests = 16;
  live.kv_bytes = 32768 + 512; live.checkpoint_interval = 2;
  auto callbacks = output.callbacks();
  auto finish = callbacks.finish;
  callbacks.finish = [finish](auto& r) { finish(r); r.delivered = true; };
  callbacks.reject = [](auto&, const auto& error, auto) { throw std::runtime_error(error); };
  BatchScheduler scheduler(std::move(backend), live, std::move(callbacks));
  const auto input = [](const char* id, std::uint8_t first = 1, std::uint8_t second = 3) {
    auto value = image_request(id, first, 1);
    value.prompt = std::make_shared<const std::vector<std::uint32_t>>(
        std::initializer_list<std::uint32_t>{2,3,4,5,6,7,8,9,10,11});
    auto image = std::make_shared<ImageInput>(*value.images.front());
    image->pixels = {second}; image->begin = 6; image->end = 8;
    value.images.push_back(std::move(image));
    return value;
  };
  auto warm = input("warm"); warm.operation = Operation::prefill; warm.controls.prompt_id = "images";
  const auto first = scheduler.submit(std::move(warm)); run(scheduler);
  require(output.tokens["warm"].empty() && scheduler.requests[first].completion_checkpoint,
          "image cache prefill did not retain endpoint without generating");
  const auto encoded = device->encoded_images;
  const auto repeat = scheduler.submit(input("repeat")); run(scheduler);
  require(scheduler.requests[repeat].cached_tokens == 10 && scheduler.requests[repeat].prefill_tokens == 0 &&
          device->encoded_images == encoded, "identical image request recomputed vision or prompt");
  const auto later = scheduler.submit(input("later",1,4)); run(scheduler);
  require(scheduler.requests[later].cached_tokens == 6 && scheduler.requests[later].prefill_tokens == 4 &&
          device->encoded_images == encoded + 1, "changed later image failed to reuse earlier image KV");
  require(output.tokens["later"] != output.tokens["repeat"], "changed later image restored stale hidden");
  const auto reordered = scheduler.submit(input("reordered",3,1)); run(scheduler);
  require(scheduler.requests[reordered].cached_tokens == 1 && device->encoded_images == encoded + 3,
          "reordered images reused an incompatible feature prefix");
  auto followup = input("followup");
  auto tokens = std::make_shared<std::vector<std::uint32_t>>(*followup.prompt);
  tokens->push_back(12); tokens->push_back(13); followup.prompt = tokens;
  const auto extended = scheduler.submit(std::move(followup)); run(scheduler);
  require(scheduler.requests[extended].cached_tokens == 10 && scheduler.requests[extended].prefill_tokens == 2 &&
          device->encoded_images == encoded + 3, "followup re-encoded images already covered by KV");
  auto position_change = input("positions");
  auto changed = std::make_shared<ImageInput>(*position_change.images[1]); changed->positions = {1};
  position_change.images[1] = std::move(changed);
  const auto positions = scheduler.submit(std::move(position_change)); run(scheduler);
  require(scheduler.requests[positions].cached_tokens == 6, "position IDs were omitted from cache identity");
  auto finished = input("finished"); finished.controls.prompt_id = "images"; finished.controls.finished = true;
  const auto released = scheduler.submit(std::move(finished)); run(scheduler);
  require(scheduler.requests[released].cached_tokens == 10 && scheduler.requests[released].checkpoints.empty(),
          "finished image request did not reuse its prompt or release request bookkeeping");
  require(device->states.empty() && device->cache->stats().execution_count == 0,
          "cached image request leaked execution state");
}

void lazy_image_identity_and_reuse_only() {
  Output output;
  auto backend = std::make_unique<TestBackend>(); auto* device = backend.get();
  auto callbacks = output.callbacks();
  std::uint8_t pixel = 1;
  callbacks.start = [&](auto& r) {
    auto image = std::make_shared<ImageInput>(*r.images.front());
    image->pixels = {pixel}; image->positions = {0}; r.images.front() = std::move(image);
  };
  auto finish = callbacks.finish;
  callbacks.finish = [finish](auto& r) { finish(r); r.delivered = true; };
  BatchScheduler* running = nullptr;
  callbacks.poll = [&](bool) {
    for (auto& r : running->requests) {
      if (r.images.empty() || r.cursor < r.images.front()->end) continue;
      auto image = std::make_shared<ImageInput>(*r.images.front());
      image->pixels.clear(); image->positions.clear(); r.images.front() = std::move(image);
    }
  };
  auto live = limits(); live.live = true;
  BatchScheduler scheduler(std::move(backend), live, std::move(callbacks)); running = &scheduler;
  const auto unloaded = [](const char* id) {
    auto value = image_request(id,1,1);
    auto image = std::make_shared<ImageInput>(*value.images.front());
    image->pixels.clear(); image->positions.clear(); value.images.front() = std::move(image);
    return value;
  };
  auto cold = unloaded("cold"); cold.controls.mode = CacheRequestMode::reuse_only;
  scheduler.submit(std::move(cold)); run(scheduler);
  require(!scheduler.prefix_checkpoints && device->cache->stats().gpu.used == 0,
          "reuse_only image miss retained checkpoints");
  scheduler.submit(unloaded("warm")); run(scheduler);
  const auto hit = scheduler.submit(unloaded("hit")); run(scheduler);
  require(scheduler.requests[hit].cached_tokens == 6 && output.tokens["hit"] == output.tokens["warm"],
          "lazy-loaded tensors or discarded payload lost stable cache identity");
  pixel = 3;
  const auto miss = scheduler.submit(unloaded("changed")); run(scheduler);
  require(scheduler.requests[miss].cached_tokens == 1 && output.tokens["changed"] != output.tokens["hit"],
          "cache lookup hashed metadata before lazy tensor loading");
}
void multiple_images_and_long_prompt() {
  Output output;
  auto backend = std::make_unique<TestBackend>(256 * 1024); auto* device = backend.get();
  device->supported.context_tokens = device->config_.maximum_context_tokens = 2048;
  device->config_.page_table_bytes = 512 * sizeof(std::uint64_t);
  device->config_.index_bytes = 4 * 1024 * 1024;
  device->supported.max_image_tokens = 8;
  auto configured = limits(); configured.max_horizon = 2048;
  BatchScheduler scheduler(std::move(backend), configured, output.callbacks());
  auto image = image_request("image",1,1);
  image.prompt = std::make_shared<const std::vector<std::uint32_t>>(1800,2);
  auto second = std::make_shared<ImageInput>(*image.images.front());
  second->begin = 1400; second->end = 1403; second->pixels = {3};
  image.images.push_back(std::move(second));
  scheduler.submit(std::move(image)); run(scheduler);
  require(scheduler.prefill_tokens == 1800 && device->image_prefills == 2 && device->encoded_images == 2,
          "long multi-image prompt lost images or retained the old prompt limit");
  require(output.processed["image"] == 1800 && output.tokens["image"] == std::vector<std::uint32_t>({8}),
          "multi-image prompt lost feature or output accounting");
  require(scheduler.prefix_checkpoints == 1 && device->states.empty(), "multi-image endpoint or execution cleanup failed");
  for (const int invalid : {0, 1, 2, 3}) {
    auto bad = image_request("invalid");
    auto span = std::make_shared<ImageInput>(*bad.images.front());
    if (invalid == 0) span->begin = 0; // Reversed order.
    if (invalid == 1) span->begin = 2; // Overlap.
    if (invalid == 2) { span->begin = 3; span->end = 7; } // Past prompt.
    bad.images.push_back(invalid == 3 ? nullptr : std::move(span));
    bool rejected = false;
    try { scheduler.submit(std::move(bad)); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "invalid multi-image spans reached execution");
  }
}
void multi_image_partial_cleanup() {
  for (const bool failure : {false, true}) {
    Output output;
    auto backend = std::make_unique<TestBackend>(); auto* device = backend.get();
    auto callbacks = output.callbacks();
    BatchScheduler* running = nullptr;
    bool cancelled = false;
    callbacks.poll = [&](bool inflight) {
      if (!failure && inflight && device->encoded_images == 1 && !cancelled) {
        cancelled = true;
        running->cancel(0, true);
      }
    };
    callbacks.finish = [](auto& r) { r.delivered = true; };
    std::size_t rejected = 0;
    callbacks.reject = [&](auto&, const auto&, auto) { ++rejected; };
    auto configured = limits(); configured.live = true;
    BatchScheduler scheduler(std::move(backend), configured, std::move(callbacks)); running = &scheduler;
    auto value = image_request("images");
    value.controls.prompt_id = "cancelled-images";
    auto second = std::make_shared<ImageInput>(*value.images.front());
    second->begin = 3; second->end = 5; second->pixels = {3};
    value.images.push_back(std::move(second));
    std::vector<std::weak_ptr<const ImageInput>> lifetimes(value.images.begin(), value.images.end());
    if (failure) device->fail_image_number = 2;
    scheduler.submit(std::move(value));
    auto after = request("after", {12,13}, 3); after.controls.mode = CacheRequestMode::reuse_only;
    scheduler.submit(std::move(after));
    run(scheduler);
    require(output.tokens["images"].empty() && output.tokens["after"].size() == 3,
            "partial multi-image cancellation/failure blocked later text or emitted output");
    require(device->encoded_images == (failure ? 2U : 1U) && rejected == std::size_t(failure),
            "multi-image cancellation continued encoding or reported an execution failure");
    require(std::all_of(lifetimes.begin(), lifetimes.end(), [](const auto& image) { return image.expired(); }) &&
            device->image_prefills == device->image_releases && device->states.empty() &&
            device->cache->stats().execution_count == 0, "partial multi-image request leaked state");
    require(device->cache->stats().gpu.used == 0 && device->cache->stats().cpu.used == 0 &&
            device->cache->prefix_stats().checkpoint_count == 0,
            "cancelled named image request retained provisional checkpoints");
  }
}

}
}
int main() {
  try {
    using namespace gewell::runtime;
    packed_image_prefill(); packed_prefill(); packed_prefill_separate_limits(); packed_prefill_cancellation_and_failure();
    shared_prefix_and_seed(); cancellation_and_backpressure(); mtp_commit_and_stop_fallback(); failure_cleanup();
    live_decode_accounting(); adaptive_decode_depth();
    mtp_failure_isolation(); windowed_mtp_statistics();
    logical_prefill_budgets(); logical_budget_under_kv_pressure(); cancellation_and_backpressure(16);
    logical_budget_failed_prefill();
    shared_prefix_and_seed(16); images_share_and_mix_with_text(16);
    images_share_and_mix_with_text(); image_mtp_and_admission_validation(); image_cancellation_and_failure();
    shared_image_cancellation_and_fairness();
    image_decode_cancellation(); multiple_images_and_long_prompt(); multi_image_partial_cleanup();
    reusable_image_checkpoints(); lazy_image_identity_and_reuse_only();
    std::cout << "shared runtime scheduler policy tests passed\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
