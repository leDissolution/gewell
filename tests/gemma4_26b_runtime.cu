#include "models/gemma4/26b_a4b/sm120/runtime_backend.h"
#include "gewell/runtime/scheduler.h"
#include "gewell/mtp_sampling.h"
#include <cuda_bf16.h>
#include <cmath>
#include <iostream>
#include <map>
#include <fcntl.h>
#include <unistd.h>

using namespace gewell;
using namespace gewell::runtime;
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
int main(int argc, char** argv) {
  try {
    require(argc == 2 || argc == 3, "usage: runtime ARTIFACT [ASSISTANT]");
    BatchLimits limits;
    limits.capacity = 3;
    limits.mtp_depth = argc == 3 ? 3 : 0;
    limits.plan_rows = limits.prefill_chunk_tokens = limits.prefill_batch_tokens = 8;
    limits.max_horizon = 64;
    limits.kv_bytes = 2ULL << 30;
    limits.max_requests = 16;
    limits.captures = limits.logprobs = limits.sampled = true;
    std::map<std::string, std::vector<std::uint32_t>> outputs;
    BatchCallbacks callbacks;
    callbacks.ready = [](auto&) { return true; };
    callbacks.start = [](auto&) {};
    callbacks.finish = [](auto&) {};
    callbacks.reject = [](auto&, const auto& error, auto) { throw std::runtime_error(error); };
    callbacks.emit = [&](auto& request, const auto* ids, std::size_t count, const void* logits, const TokenLogprobs* scores) {
      require(count && count <= limits.mtp_depth + 1 && logits && scores, "missing output capture");
      for (std::size_t row = 0; row < count; ++row) {
        const auto* values = static_cast<const __nv_bfloat16*>(logits) + row * 262144;
        unsigned best = 0;
        for (unsigned i = 1; i < 262144; ++i)
          if (__bfloat162float(values[i]) > __bfloat162float(values[best])) best = i;
        if (request.sampling.temperature == 0) {
          require(ids[row] == best, "greedy ID differs from captured argmax");
          require(scores[row].logprob == 0 && scores[row].count == 1 && scores[row].top[0].token == ids[row],
                  "greedy logprob mismatch");
        } else {
          require(std::isfinite(scores[row].logprob) && scores[row].logprob <= 0 && scores[row].count > 0,
                  "invalid sampled logprob");
        }
        outputs[request.id].push_back(ids[row]);
      }
    };
    // An ephemeral descriptor path proves startup consumes the validated mapping:
    // that path no longer names these weights when the backend initializes.
    const int descriptor = open(argv[1], O_RDONLY | O_CLOEXEC);
    require(descriptor >= 0, "cannot open fixture weights");
    auto artifact = gemma4_26b_a4b::ArtifactFile::Open("/proc/self/fd/" + std::to_string(descriptor));
    close(descriptor);
    auto owned_backend = gemma4_26b_a4b::sm120::make_runtime_backend(std::move(artifact), limits, nvfp4::ActivationPolicy::always, argc == 3 ? argv[2] : "", "", "");
    auto& backend = *owned_backend;
    BatchScheduler scheduler(std::move(owned_backend), limits, callbacks);
    auto submit = [&](const char* id, std::vector<std::uint32_t> tokens, unsigned count, bool sampled) {
      BatchRequest request;
      request.id = id;
      request.prompt = std::make_shared<const std::vector<std::uint32_t>>(std::move(tokens));
      request.max_new_tokens = count;
      request.capture_logits = request.logprobs = true;
      request.seed = 73;
      request.rng.seed(73);
      if (sampled) request.sampling = {0.7f, 0.9f, 16};
      return scheduler.submit(std::move(request));
    };
    auto run = [&] {
      for (unsigned turn = 0; scheduler.has_pending() && turn < 1000; ++turn) scheduler.step();
      require(!scheduler.has_pending(), "scheduler did not finish");
    };
    const auto a = submit("a", {2,105,2364}, 13, false);
    submit("b", {2,9259}, 9, false);
    submit("sample", {2,105,2364}, 11, true);
    run();
    require(outputs["a"].size() == 13 && outputs["b"].size() == 9 && outputs["sample"].size() == 11,
            "wrong completion lengths");
    if (limits.mtp_depth) require(scheduler.requests[a].mtp_cycles > 1, "MTP did not run multiple cycles");
    auto cached = submit("cached", {2,105,2364}, 1, false);
    run();
    require(outputs["cached"].size() == 1 && outputs["cached"][0] == outputs["a"][0], "cached head changed output");
    require(scheduler.requests[cached].prefill_tokens == 0, "cached prompt recomputed");
    submit("sample-repeat", {2,105,2364}, 1, true);
    run();
    require(outputs["sample-repeat"][0] == outputs["sample"][0], "cached seeded sample changed");
    // Exercise the adapter's mask upload/status path using its last decision row.
    std::vector<std::uint32_t> mask(mtp_sampling::mask_words(262144));
    mask[42 / 32] = 1u << (42 % 32);
    std::mt19937_64 rng(99);
    for (float temperature : {0.0f, 0.7f}) {
      SamplingSettings settings{temperature, 0.9f, 16};
      backend.sample_batch_row(0, settings, rng, mask.data());
      backend.summarize_batch_row(0, settings, 20, mask.data());
      backend.download_ids(1);
      backend.download_logprobs(0);
      backend.synchronize("masked runtime check");
      backend.check_constraint_sampling();
      require(backend.output_id(0) == 42 && backend.host_logprobs()->logprob == 0 &&
              backend.host_logprobs()->count == 1, "masked sampling escaped singleton support");
    }
    mask[42 / 32] = 0;
    backend.sample_batch_row(0, {}, rng, mask.data());
    backend.synchronize("empty mask runtime check");
    bool failed = false;
    try { backend.check_constraint_sampling(); } catch (const std::runtime_error&) { failed = true; }
    require(failed, "empty mask did not report failure");
    std::cout << "26B scheduler: heterogeneous prefill/decode, greedy captures, sampled logprobs, cached terminal head, seeded repeat, constrained sampling/status, multi-cycle MTP PASS\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
