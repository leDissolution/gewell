// Same-state, full-model oracle cost experiment. Discovery and private-cache
// forks are untimed. Every timed candidate uses normal draft/verify/commit;
// an oracle candidate must actually accept every draft, without bypasses.
#include "models/gemma4/31b/sm120/runtime_backend.h"
#include "models/gemma4/31b/sm120/runner_support.cuh"
#include "gewell/runtime/cache.h"
#include "gewell/runtime/scheduler.h"
#include "gewell/mtp_head.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>

namespace rt = gewell::runtime;
namespace sm = gewell::gemma4_31b::sm120;
namespace model = gewell::gemma4_31b;
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
double milliseconds(Clock::time_point begin) {
  return std::chrono::duration<double, std::milli>(Clock::now()-begin).count();
}
void emit(const Json& record) { std::cout << record.dump() << '\n' << std::flush; }

std::vector<std::uint32_t> read_tokens(const std::string& path) {
  const auto bytes = std::filesystem::file_size(path);
  require(bytes && bytes % 4 == 0 && bytes <= 262144*4, "invalid prompt token file");
  std::vector<std::uint32_t> tokens(bytes/4);
  std::ifstream file(path, std::ios::binary);
  file.read(reinterpret_cast<char*>(tokens.data()), bytes);
  require(bool(file) && file.peek() == std::char_traits<char>::eof(), "prompt read failed");
  require(*std::max_element(tokens.begin(), tokens.end()) < model::kVocabSize, "invalid prompt token");
  return tokens;
}

// Preserve the three independent random streams when changing depth. Simply
// resizing [draft_N, accept_N, final] would use draft draws for acceptance.
std::vector<float> prefix_uniforms(const std::vector<float>& full, unsigned depth) {
  const auto horizon = (full.size()-1)/2;
  require(full.size() == 2*horizon+1 && depth <= horizon, "invalid uniform prefix");
  std::vector<float> result(full.begin(), full.begin()+depth);
  result.insert(result.end(), full.begin()+horizon, full.begin()+horizon+depth);
  result.push_back(full.back());
  return result;
}

std::vector<unsigned> accepted(const rt::BatchMtpOutcome& result) {
  std::vector<unsigned> values;
  for (const auto& r : result.requests) {
    require(r.error.empty(), "MTP sampling failed");
    require(r.tokens.size() == r.verification.accepted_drafts+1, "MTP output accounting differs");
    values.push_back(r.verification.accepted_drafts);
  }
  return values;
}

std::vector<rt::BatchMtpInput> with_depths(const std::vector<rt::BatchMtpInput>& full,
                                         const std::vector<unsigned>& depths) {
  require(full.size() == depths.size(), "depth count differs");
  auto result = full;
  for (std::size_t i=0; i<result.size(); ++i) {
    result[i].depth = depths[i];
    result[i].uniforms = prefix_uniforms(full[i].uniforms, depths[i]);
  }
  return result;
}

std::vector<unsigned> oracle_width(const std::vector<unsigned>& lengths,
                                   unsigned horizon, unsigned width, float tail) {
  std::vector<gewell::MtpDepthPrediction> predictions;
  for (auto a : lengths) {
    predictions.push_back({0, horizon, std::vector<float>(horizon, 0)});
    std::fill_n(predictions.back().survival.begin(), a, 1.0F);
  }
  // Equal certain-survival scores prefer shallow positions before deeper ones.
  // Zero-survival positions never consume unused width.
  return gewell::choose_mtp_depths(predictions, width, 1.0F, tail);
}

void check_equal(const rt::BatchMtpOutcome& a, const rt::BatchMtpOutcome& b) {
  require(accepted(a) == accepted(b), "acceptance changed between discovery and timed replay");
  for (std::size_t i=0; i<a.requests.size(); ++i)
    require(a.requests[i].tokens == b.requests[i].tokens, "tokens changed on identical-state replay");
}

struct Request {
  std::string id;
  std::vector<std::uint32_t> prompt, output;
  gewell::kv_cache::ExecutionId execution{};
  rt::TerminalState hidden;
  std::uint32_t position{}, pending{};
  std::mt19937_64 rng;
};

void cohort(const sm::WeightArena& weights, const Json& settings, const Json& spec) {
  const unsigned horizon = settings.at("max_depth"), width = settings.at("width");
  const unsigned warmup = settings.at("warmup_cycles"), cycles = settings.at("cycles");
  const float tail = settings.at("tail_fraction");
  const unsigned batch = spec.at("requests").size();
  std::vector<Request> requests;
  unsigned context = 0;
  for (const auto& row : spec.at("requests")) {
    Request r;
    r.id = row.at("id"); r.prompt = read_tokens(row.at("path"));
    r.position = r.prompt.size(); r.rng.seed(row.at("seed").get<std::uint64_t>());
    context = std::max(context, r.position+(warmup+cycles+1)*(horizon+1));
    requests.push_back(std::move(r));
  }
  require(batch && batch <= 64 && horizon && batch*(horizon+1) <= 1280 &&
          context <= 262144 && width >= batch && cycles, "invalid oracle dimensions");
  rt::BatchLimits limits;
  limits.capacity = batch; limits.mtp_depth = horizon;
  limits.plan_rows = 1024; limits.max_horizon = context;
  limits.prefill_chunk_tokens = limits.prefill_batch_tokens = 1024;
  limits.kv_bytes = settings.at("kv_mib").get<std::size_t>() << 20;
  limits.sampled = true;
  limits.local_kv_format = limits.global_kv_format = gewell::kv_cache::Format::fp8;
  std::unique_ptr<rt::PersistentCacheManager> cache_owner;
  auto backend = sm::make_runtime_backend(weights, limits, gewell::nvfp4::ActivationPolicy::always);
  cache_owner = std::make_unique<rt::PersistentCacheManager>(backend->cache_config(), backend->cache_storage_factory());
  auto& cache = *cache_owner;
  backend->allocate_staging();
  const auto bootstrap = cache.try_begin_batch(0, 1, {});
  require(bool(bootstrap), "bootstrap cache admission failed");
  backend->initialize(cache, *bootstrap, limits); cache.release(*bootstrap);
  backend->initialize_outputs(false, false);
  const rt::SamplingSettings sampling{settings.at("temperature"), settings.at("top_p"), settings.at("top_k")};
  const auto stream = backend->context();
  // Keep the cache manager alive until the backend has drained and released its
  // borrowed cache facade. No serving state or retained checkpoint is modified.
  for (auto& r : requests) {
    const auto id = cache.try_begin_batch(0, r.position+horizon+1, stream);
    require(bool(id), "source cache admission failed");
    r.execution = *id; r.hidden = backend->acquire_hidden();
    for (unsigned start=0; start<r.prompt.size(); start+=1024) {
      const unsigned rows = std::min<std::size_t>(1024, r.prompt.size()-start);
      backend->prefill_batch({{r.execution, r.prompt.data()+start, start, rows, r.hidden, {}}},
                            [](std::size_t) { return true; });
      backend->synchronize("oracle prefill");
    }
    backend->prefix_head_step(r.execution, r.hidden);
    backend->sample_batch_row(0, sampling, r.rng, nullptr);
    backend->download_ids(1); backend->synchronize("oracle initial token");
    r.pending = backend->output_id(0); r.output.push_back(r.pending);
  }
  sm::DeviceAllocation trial_hidden(std::size_t(batch)*model::kHiddenSize*sizeof(sm::BFloat16));
  std::mt19937_64 order_rng(settings.at("seed").get<std::uint64_t>() + spec.at("repetition").get<unsigned>());
  for (unsigned round=0; round<warmup+cycles; ++round) {
    std::vector<rt::BatchMtpInput> full;
    for (auto& r : requests) {
      require(cache.processed_tokens(r.execution) == r.position, "source position differs");
      // Counterfactual forks need private local rings. Reserve only this
      // source round's growth, not all hypothetical future D15 rounds.
      require(cache.try_resize_batch(r.execution, r.position+horizon+1), "source growth reservation failed");
      rt::BatchMtpInput input;
      input.execution = r.execution; input.position = r.position; input.pending_token = r.pending;
      input.target_hidden = r.hidden; input.depth = horizon;
      input.temperature = sampling.temperature; input.top_p = sampling.top_p; input.top_k = sampling.top_k;
      input.uniforms.resize(2*horizon+1);
      for (auto& u : input.uniforms)
        u = std::min(std::uniform_real_distribution<float>(0,1)(r.rng), std::nextafter(1.0F,0.0F));
      full.push_back(std::move(input));
    }
    const auto truth = backend->run_batch_mtp(full);
    const auto lengths = accepted(truth);
    struct Policy { std::string name; std::vector<unsigned> depths; bool oracle; };
    std::vector<Policy> policies;
    for (unsigned depth : settings.at("fixed_depths")) {
      require(depth <= horizon, "fixed depth exceeds oracle horizon");
      policies.push_back({"fixed-"+std::to_string(depth), std::vector<unsigned>(batch, depth), false});
    }
    policies.push_back({"oracle-all", lengths, true});
    policies.push_back({"oracle-width", oracle_width(lengths, horizon, width, 0), true});
    policies.push_back({"oracle-width-tail", oracle_width(lengths, horizon, width, tail), true});
    std::shuffle(policies.begin(), policies.end(), order_rng);
    Json results = Json::array();
    for (const auto& policy : policies) {
      auto trial_full = full;
      std::vector<gewell::kv_cache::ExecutionId> forks;
      for (unsigned i=0; i<batch; ++i) {
        const auto fork = cache.try_fork_batch(requests[i].execution, requests[i].position+horizon+1, stream);
        if (!fork) throw std::runtime_error("trial fork admission failed at request "+std::to_string(i));
        forks.push_back(*fork); trial_full[i].execution = *fork;
      }
      backend->synchronize("complete untimed oracle forks");
      auto depths = policy.depths;
      auto inputs = with_depths(trial_full, depths);
      auto discovered = backend->run_batch_mtp(inputs);
      unsigned passes = 1;
      // Shapes can change numerical results and hence sampled proposals. Find
      // a self-consistent accepted prefix on this exact unchanged state. Each
      // unsuccessful pass strictly reduces total depth, so this terminates.
      while (policy.oracle && accepted(discovered) != depths) {
        const auto smaller = accepted(discovered);
        require(std::accumulate(smaller.begin(), smaller.end(), 0U) <
                std::accumulate(depths.begin(), depths.end(), 0U), "oracle refinement did not shrink");
        depths = smaller; inputs = with_depths(trial_full, depths);
        discovered = backend->run_batch_mtp(inputs); ++passes;
      }
      const auto start = Clock::now();
      backend->begin_step("begin oracle timed cycle");
      const auto measured = backend->run_batch_mtp(inputs);
      std::vector<rt::BatchMtpCommit> commits;
      for (unsigned i=0; i<batch; ++i)
        commits.push_back({i, forks[i], requests[i].position,
            static_cast<unsigned>(measured.requests[i].tokens.size()),
            rt::TerminalState(static_cast<sm::BFloat16*>(trial_hidden.data())+std::size_t(i)*model::kHiddenSize),
            false, false, 0});
      backend->commit_batch_mtp(commits);
      backend->end_step("end oracle timed cycle");
      backend->synchronize("complete oracle timed commit");
      const auto wall_ms = milliseconds(start);
      const auto gpu_ms = backend->elapsed("oracle full cycle cost");
      check_equal(discovered, measured);
      if (policy.oracle) require(accepted(measured) == depths, "timed oracle rejected a draft");
      const auto a = accepted(measured);
      const auto proposed = std::accumulate(depths.begin(), depths.end(), 0U);
      const auto accepted_count = std::accumulate(a.begin(), a.end(), 0U);
      Json tokens = Json::array();
      for (const auto& r : measured.requests) tokens.push_back(r.tokens);
      results.push_back({{"policy", policy.name}, {"oracle", policy.oracle},
          {"initial_depths", policy.depths}, {"depths", depths}, {"accepted", a},
          {"tokens", tokens}, {"discovery_passes", passes}, {"drafts", proposed},
          {"verifier_rows", batch+proposed}, {"emitted", batch+accepted_count},
          {"wall_ms", wall_ms}, {"gpu_ms", gpu_ms},
          {"draft_ms", measured.draft_gpu_milliseconds},
          {"verify_ms", measured.verify_gpu_milliseconds},
          {"select_ms", measured.select_gpu_milliseconds}});
      for (auto id : forks) cache.release(id);
    }
    // Rerun the source to restore its staging after counterfactual candidates.
    // Exact output equality also checks that none of their commits changed it.
    const auto advance = backend->run_batch_mtp(full);
    check_equal(truth, advance);
    std::vector<rt::BatchMtpCommit> commits;
    Json source_tokens = Json::array(), positions = Json::array();
    for (unsigned i=0; i<batch; ++i) {
      auto& r = requests[i]; const auto& tokens = advance.requests[i].tokens;
      positions.push_back(r.position); source_tokens.push_back(tokens);
      commits.push_back({i, r.execution, r.position, static_cast<unsigned>(tokens.size()), r.hidden, false, false, 0});
      r.position += tokens.size(); r.pending = tokens.back();
      r.output.insert(r.output.end(), tokens.begin(), tokens.end());
    }
    backend->commit_batch_mtp(commits); backend->synchronize("advance common source trajectory");
    emit({{"event", "oracle_round"}, {"batch", batch}, {"repetition", spec.at("repetition")},
          {"round", round}, {"measured", round >= warmup}, {"source_positions", positions},
          {"source_accepted", lengths}, {"source_tokens", source_tokens}, {"policies", results}});
  }
  Json output = Json::array();
  for (auto& r : requests) {
    output.push_back({{"id", r.id}, {"tokens", r.output}});
    cache.release(r.execution); backend->release_hidden(r.hidden);
  }
  emit({{"event", "oracle_cohort_complete"}, {"batch", batch},
        {"repetition", spec.at("repetition")}, {"outputs", output}});
  backend.reset();
}
}  // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 4, "usage: gewell_mtp_oracle_benchmark ARTIFACT ASSISTANT EXPERIMENT.json");
    // A depth-two replay must keep acceptance draws from the original horizon.
    require(prefix_uniforms({0,1,2,3,4,5,6},2) == std::vector<float>({0,1,3,4,6}), "uniform mapping failed");
    require(prefix_uniforms({0,1,2,3,4,5,6},0) == std::vector<float>({6}), "zero-depth mapping failed");
    std::ifstream input(argv[3]); const auto settings = Json::parse(input);
    gewell::console::set_format("json"); sm::validate_cuda_device();
    auto artifact = gewell::artifact::ArtifactFile::Open(argv[1]);
    sm::WeightArena weights(artifact, gewell::weight_qdq::Mask{}, argv[2]);
    emit({{"event", "oracle_settings"}, {"settings", settings},
          {"method", "Same-state cycle replay; free oracle discovery, real draft/verify/commit, exact replay and all-accepted checks. Private-cache fork setup excluded; commit copy-on-write included. Common source advances at fixed maximum depth."}});
    for (const auto& spec : settings.at("cohorts")) cohort(weights, settings, spec);
    emit({{"event", "oracle_complete"}});
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "oracle benchmark: " << error.what() << '\n';
    return 1;
  }
}
