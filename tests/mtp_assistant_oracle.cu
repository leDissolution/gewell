#include "gewell/mtp_assistant.h"
#include "gewell/bf16_primitives.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using gewell::mtp_assistant::BFloat16;

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

std::vector<BFloat16> read(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file || file.tellg() <= 0 || file.tellg() % 2) throw std::runtime_error("invalid BF16 fixture: " + path);
  std::vector<BFloat16> values(static_cast<std::size_t>(file.tellg()) / 2);
  file.seekg(0);
  if (!file.read(reinterpret_cast<char*>(values.data()), values.size() * 2)) throw std::runtime_error("fixture read: " + path);
  return values;
}

class Device {
 public:
  explicit Device(std::size_t count) : count_(count) { check(cudaMalloc(reinterpret_cast<void**>(&data_), count * 2)); }
  explicit Device(const std::vector<BFloat16>& values) : Device(values.size()) {
    check(cudaMemcpy(data_, values.data(), count_ * 2, cudaMemcpyHostToDevice));
  }
  ~Device() { cudaFree(data_); }
  Device(const Device&) = delete;
  BFloat16* get() const { return data_; }
  std::vector<BFloat16> host() const {
    std::vector<BFloat16> result(count_);
    check(cudaMemcpy(result.data(), data_, count_ * 2, cudaMemcpyDeviceToHost));
    return result;
  }
 private:
  BFloat16* data_{};
  std::size_t count_;
};

bool compare(const std::vector<BFloat16>& actual, const std::vector<BFloat16>& expected,
             const std::string& name, double relative_rms_bound = 0.025) {
  if (actual.size() != expected.size()) throw std::runtime_error("oracle output shape mismatch");
  double squared_error = 0, squared_reference = 0;
  float maximum = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const float a = __bfloat162float(actual[i]), e = __bfloat162float(expected[i]);
    if (!std::isfinite(a) || !std::isfinite(e)) throw std::runtime_error("non-finite oracle output");
    const float error = a - e;
    maximum = std::max(maximum, std::abs(error));
    squared_error += static_cast<double>(error) * error;
    squared_reference += static_cast<double>(e) * e;
  }
  const double relative_rms = std::sqrt(squared_error / std::max(squared_reference, 1.0e-30));
  std::cout << name << " max_abs=" << maximum << " relative_rms=" << relative_rms << '\n';
  // The default fixture uses independent dense Torch FP32 attention matching
  // the native score/softmax/context boundaries. This is a wiring gate for
  // the complete assistant, not a target quality acceptance gate. Eager BF16
  // attention fixtures are explicit alternate-recipe drift controls.
  return relative_rms <= relative_rms_bound;
}

bool compare_rope(const std::string& directory, const gewell::mtp_assistant::Weights& weights, unsigned heads) {
  namespace primitives = gewell::bf16_primitives;
  using gewell::gemma4_31b::AttentionKind;
  Device local_cos(256), local_sin(256), global_cos(512), global_sin(512);
  bool passed = true;
  for (unsigned global = 0; global < 2; ++global) {
    const std::string kind = global ? "global" : "local";
    const unsigned width = global ? 512 : 256;
    Device raw(read(directory + "/rope-" + kind + "-raw.bf16")), normalized(heads * width), rotated(heads * width);
    primitives::rms_norm(raw.get(), weights.assistant[global ? 36 : 3], normalized.get(), heads, width);
    passed = compare(normalized.host(), read(directory + "/rope-" + kind + "-norm.bf16"), "fixed " + kind + " QNorm", 0) && passed;
    for (unsigned position : {1U, 31U, 255U, 1024U, 1031U}) {
      primitives::generate_rope_factors_m1(local_cos.get(), local_sin.get(), global_cos.get(), global_sin.get(), position);
      auto& cosine = global ? global_cos : local_cos;
      auto& sine = global ? global_sin : local_sin;
      primitives::apply_rope_m1(normalized.get(), cosine.get(), sine.get(), rotated.get(), heads,
                                 global ? AttentionKind::global : AttentionKind::local);
      const auto name = "rope-" + kind + "-" + std::to_string(position);
      passed = compare(cosine.host(), read(directory + "/" + name + "-cos.bf16"), name + " cos", 0) && passed;
      passed = compare(sine.host(), read(directory + "/" + name + "-sin.bf16"), name + " sin", 0) && passed;
      passed = compare(rotated.host(), read(directory + "/" + name + "-query.bf16"), name + " query", 0) && passed;
    }
  }
  return passed;
}

void compare_batch(const std::string& directory, cublasLtHandle_t handle,
                   gewell::mtp_assistant::Weights weights,
                   const gewell::mtp_assistant::FrozenCache& cache,
                   bool report_drift, gewell::mtp_assistant::Model selected_model, unsigned H,
                   gewell::attention::Compute compute) {
  namespace assistant = gewell::mtp_assistant;
  constexpr unsigned V = 262144, N = 3;
  auto embedding = read(directory + "/target-embedding-row.bf16");
  embedding.resize(N * H);
  for (unsigned row = 1; row < N; ++row)
    for (unsigned d = 0; d < H; ++d)
      embedding[row * H + d] = __float2bfloat16_rn(
          __bfloat162float(embedding[d]) * (row == 1 ? -1.0F : 0.5F));
  Device table(embedding), tokens(2 * N), logits(N * V), feedback(N * H);
  Device scalar_logits(V), scalar_feedback(H);
  Device initial(read(directory + "/target-hidden.bf16"));
  weights.target_embedding = table.get();
  const std::uint32_t ids[N] = {2, 0, 1};
  check(cudaMemcpy(tokens.get(), ids, sizeof(ids), cudaMemcpyHostToDevice));
  assistant::Executor batch(selected_model, handle, weights, cache.processed_tokens, N, compute, compute);
  assistant::Executor scalar(selected_model, handle, weights, cache.processed_tokens, 1, compute, compute);
  assistant::Executor isolated(selected_model, handle, weights, cache.processed_tokens, N, compute, compute);
  Device isolated_logits(N * V), isolated_feedback(N * H);
  std::vector<assistant::Input> inputs;
  for (unsigned row = 0; row < N; ++row) {
    auto frozen = cache;
    if (row) frozen.processed_tokens = std::min(cache.processed_tokens, row == 1 ? 1U : 257U);
    inputs.push_back({reinterpret_cast<const std::uint32_t*>(tokens.get()) + row,
        initial.get(), frozen, logits.get() + row * V, feedback.get() + row * H});
  }
  for (unsigned step = 0; step < 2; ++step) {
    // Snapshot each row's feedback before the batched in-place recurrence.
    std::vector<std::vector<BFloat16>> expected_logits, expected_feedback;
    std::vector<std::vector<BFloat16>> fixed_shape_logits, fixed_shape_feedback;
    for (const auto& input : inputs) {
      scalar.forward(input.token, input.hidden, input.cache,
          scalar_logits.get(), scalar_feedback.get());
      expected_logits.push_back(scalar_logits.host());
      expected_feedback.push_back(scalar_feedback.host());
      // Each row also has an independent oracle at the same GEMM shape:
      // repeat only that request in all lanes, with disjoint output storage.
      std::vector<assistant::Input> repeated(N, input);
      for (unsigned row = 0; row < N; ++row) {
        repeated[row].logits = isolated_logits.get() + row * V;
        repeated[row].feedback = isolated_feedback.get() + row * H;
      }
      isolated.forward_batch(repeated);
      auto isolated_output = isolated_logits.host(), isolated_hidden = isolated_feedback.host();
      fixed_shape_logits.emplace_back(isolated_output.begin(), isolated_output.begin() + V);
      fixed_shape_feedback.emplace_back(isolated_hidden.begin(), isolated_hidden.begin() + H);
    }
    batch.forward_batch(inputs);
    const auto output = logits.host(), hidden = feedback.host();
    for (unsigned row = 0; row < N; ++row) {
      const std::string name = "batch step=" + std::to_string(step) + " row=" + std::to_string(row);
      const std::vector<BFloat16> row_logits(output.begin() + row * V, output.begin() + (row + 1) * V);
      const std::vector<BFloat16> row_feedback(hidden.begin() + row * H, hidden.begin() + (row + 1) * H);
      const bool logits_match = compare(row_logits, expected_logits[row], name + " logits");
      const bool feedback_match = compare(row_feedback, expected_feedback[row], name + " feedback");
      if ((!logits_match || !feedback_match) && !report_drift)
        throw std::runtime_error("batched assistant exceeded scalar numerical bound");
      if (!compare(row_logits, fixed_shape_logits[row], name + " fixed-shape logits", 0) ||
          !compare(row_feedback, fixed_shape_feedback[row], name + " fixed-shape feedback", 0))
        throw std::runtime_error("batched assistant changed independent fixed-shape request");
      inputs[row].hidden = inputs[row].feedback;
    }
  }
  // At a fixed dense shape, another row's changed token and prefix must have
  // no effect on the untouched rows. Keep hidden inputs fixed for this check.
  for (auto& input : inputs) input.hidden = initial.get();
  batch.forward_batch(inputs);
  const auto before = logits.host();
  const std::uint32_t changed = 0;
  check(cudaMemcpy(tokens.get(), &changed, sizeof(changed), cudaMemcpyHostToDevice));
  inputs[0].cache.processed_tokens = 1;
  batch.forward_batch(inputs);
  const auto after = logits.host();
  if (!std::memcmp(before.data(), after.data(), V * 2) ||
      std::memcmp(before.data() + V, after.data() + V, (N - 1) * V * 2))
    throw std::runtime_error("batched assistant request isolation failed");
  // A shrinking active set uses its own projection shape and output pointers.
  scalar.forward(inputs[2].token, inputs[2].hidden, inputs[2].cache,
      scalar_logits.get(), scalar_feedback.get());
  batch.forward_batch({inputs[2]});
  const auto shrunk = logits.host();
  if (!compare({shrunk.begin() + 2 * V, shrunk.end()}, scalar_logits.host(), "shrinking assistant batch", 0))
    throw std::runtime_error("one-row batched assistant changed scalar output");
  for (const auto count : {0U, N + 1}) {
    auto invalid = inputs;
    invalid.resize(count, inputs[0]);
    bool rejected = false;
    try { batch.forward_batch(invalid); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("invalid assistant batch size accepted");
  }
  if (selected_model == assistant::Model::gemma4_26b_a4b) {
    // Cross the embedding launch's 128-request boundary. Equal inputs at
    // both ends must agree at the same GEMM shape; changing only the tail
    // token must affect only that request.
    constexpr unsigned Wide = 129;
    assistant::Executor wide(selected_model, handle, weights,
        cache.processed_tokens, Wide, compute, compute);
    Device wide_logits(Wide * V), wide_feedback(Wide * H);
    std::vector<assistant::Input> repeated(Wide, inputs[1]);
    for (unsigned row = 0; row < Wide; ++row) {
      repeated[row].logits = wide_logits.get() + row * V;
      repeated[row].feedback = wide_feedback.get() + row * H;
    }
    wide.forward_batch(repeated);
    const auto baseline = wide_logits.host();
    if (std::memcmp(baseline.data(), baseline.data() + (Wide - 1) * V, V * 2))
      throw std::runtime_error("assistant embedding batch boundary differs");
    repeated.back().token = inputs[2].token;
    wide.forward_batch(repeated);
    const auto changed_output = wide_logits.host();
    if (std::memcmp(baseline.data(), changed_output.data(), (Wide - 1) * V * 2) ||
        !std::memcmp(baseline.data() + (Wide - 1) * V,
            changed_output.data() + (Wide - 1) * V, V * 2))
      throw std::runtime_error("assistant embedding tail request isolation failed");
    std::cout << "assistant 129-row embedding boundary and tail isolation passed\n";
  }
  std::cout << "assistant batch: mixed_positions distinct_tokens recurrent_feedback request_isolation shrinking_batch passed\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 9) throw std::runtime_error("usage: gewell_mtp_assistant_oracle FIXTURE_DIRECTORY [--report-drift] [--free-feedback] [--26b] [--real-target] [--fp8-compute] [--dump DIRECTORY]");
    bool report_drift = false, free_feedback = false, model26 = false, real_target = false, fp8_compute = false;
    std::string dump;
    for (int i = 2; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--report-drift") report_drift = true;
      else if (option == "--free-feedback") free_feedback = true;
      else if (option == "--26b") model26 = true;
      else if (option == "--real-target") real_target = true;
      else if (option == "--fp8-compute") fp8_compute = true;
      else if (option == "--dump" && i + 1 < argc) dump = argv[++i];
      else throw std::runtime_error("unknown oracle option: " + option);
    }
    if (model26 && !report_drift)
      throw std::runtime_error("26B requires --report-drift until its numerical gate is established");
    if (real_target && !model26) throw std::runtime_error("real-target fixture requires --26b");
    if (!dump.empty() && !std::filesystem::create_directory(dump))
      throw std::runtime_error("dump directory must be new");
    if (fp8_compute && !report_drift) throw std::runtime_error("FP8 compute requires --report-drift against the BF16 fixture");
    const auto compute = fp8_compute ? gewell::attention::Compute::fp8 : gewell::attention::Compute::bf16;
    const auto selected_model = model26 ? gewell::mtp_assistant::Model::gemma4_26b_a4b
                                       : gewell::mtp_assistant::Model::gemma4_31b;
    const unsigned H = model26 ? 2816 : 5376, heads = model26 ? 16 : 32;
    const std::string directory = argv[1];
    unsigned length = 0;
    std::ifstream(directory + "/prefix-length.txt") >> length;
    if (!length || length >= 262144) throw std::runtime_error("invalid fixture prefix length");
    std::vector<std::unique_ptr<Device>> storage;
    auto load = [&](const std::string& file) {
      storage.push_back(std::make_unique<Device>(read(directory + "/" + file)));
      return storage.back()->get();
    };
    gewell::mtp_assistant::Weights weights;
    for (unsigned i = 0; i < 48; ++i) {
      std::ostringstream file;
      file << "weight-" << std::setfill('0') << std::setw(2) << i << ".bf16";
      weights.assistant[i] = load(file.str());
    }
    weights.target_embedding = load("target-embedding-row.bf16");
    weights.target_global_k_norm = load("target-global-k-norm.bf16");
    const auto* hidden = load("target-hidden.bf16");
    gewell::mtp_assistant::FrozenCache cache;
    cache.local_key = load("local-key.bf16");
    cache.local_value = load("local-value.bf16");
    cache.global_compact = load("global-compact.bf16");
    cache.global_capacity = length; cache.processed_tokens = length;
    Device logits(262144), feedback(H), token_storage(2), trace_rows(6 * 1024), attention_trace(12 * heads * 512);
    gewell::mtp_assistant::Trace trace;
    trace.pre_projection = trace_rows.get();
    for (unsigned i = 0; i < 4; ++i) trace.layers[i] = trace_rows.get() + (i + 1) * 1024;
    trace.final_norm = trace_rows.get() + 5 * 1024;
    for (unsigned i = 0; i < 4; ++i) {
      trace.query_norm[i] = attention_trace.get() + (3 * i) * heads * 512;
      trace.query_rope[i] = attention_trace.get() + (3 * i + 1) * heads * 512;
      trace.attention[i] = attention_trace.get() + (3 * i + 2) * heads * 512;
    }
    check(cudaMemset(token_storage.get(), 0, 4));
    cublasLtHandle_t handle;
    if (cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLASLt create failed");
    {
      gewell::mtp_assistant::Executor executor(selected_model, handle, weights, length, 1, compute, compute);
      if (!compare_rope(directory, weights, heads)) throw std::runtime_error("fixed-input QNorm/RoPE differs from reference");
      bool passed = true;
      std::cout << "assistant scratch_bytes=" << executor.scratch_bytes() << '\n';
      std::cout << "assistant feedback_input=" << (free_feedback ? "native_recurrent" : "teacher_forced_reference") << '\n';
      for (unsigned step = 0; step < (real_target ? 3U : 2U); ++step) {
        const std::uint32_t token = real_target ? step : 0;
        check(cudaMemcpy(token_storage.get(), &token, 4, cudaMemcpyHostToDevice));
        executor.forward(reinterpret_cast<const std::uint32_t*>(token_storage.get()), hidden,
                         cache, logits.get(), feedback.get(), nullptr, &trace);
        const auto captured = trace_rows.host();
        const auto captured_attention = attention_trace.host();
        for (unsigned layer = 0; !real_target && layer < 4; ++layer) {
          const unsigned width = layer == 3 ? 512 : 256;
          const std::vector<std::string> fields{"query-norm", "query-rope", "attention"};
          for (unsigned field = 0; field < 3; ++field) {
            const auto first = captured_attention.begin() + (3 * layer + field) * heads * 512;
            const std::vector<BFloat16> row(first, first + heads * width);
            const std::string name = "step-" + std::to_string(step) + "-layer-" + std::to_string(layer) + "-" + fields[field];
            passed = compare(row, read(directory + "/" + name + ".bf16"), name) && passed;
          }
        }
        const std::vector<std::string> names{"pre-projection", "layer-0", "layer-1", "layer-2", "layer-3", "final-norm"};
        for (unsigned i = 0; i < names.size(); ++i) {
          const std::vector<BFloat16> row(captured.begin() + i * 1024, captured.begin() + (i + 1) * 1024);
          passed = compare(row, read(directory + "/step-" + std::to_string(step) + "-" + names[i] + ".bf16"),
                  "step " + std::to_string(step) + " " + names[i]) && passed;
        }
        passed = compare(logits.host(), read(directory + "/step-" + std::to_string(step) + "-logits.bf16"), "step " + std::to_string(step) + " logits") && passed;
        passed = compare(feedback.host(), read(directory + "/step-" + std::to_string(step) + "-feedback.bf16"), "step " + std::to_string(step) + " feedback") && passed;
        if (!dump.empty()) {
          const auto save = [&](const std::string& name, const std::vector<BFloat16>& data) {
            std::ofstream file(dump + "/step-" + std::to_string(step) + "-" + name + ".bf16", std::ios::binary);
            if (!file.write(reinterpret_cast<const char*>(data.data()), data.size() * 2))
              throw std::runtime_error("failed to write native oracle dump");
          };
          save("logits", logits.host()); save("feedback", feedback.host()); save("layers", captured);
        }
        // Default to an identical independent input at each complete forward.
        // --free-feedback separately measures recurrent numeric amplification
        // and exercises supported feedback input/output aliasing.
        if (step + 1 < (real_target ? 3U : 2U))
          hidden = free_feedback ? feedback.get() : load("step-" + std::to_string(step) + "-feedback.bf16");
      }
      if (!passed && !report_drift) throw std::runtime_error("assistant exceeded 2.5% relative RMS oracle bound");
      compare_batch(directory, handle, weights, cache, report_drift, selected_model, H, compute);
    }
    cublasLtDestroy(handle);
    std::cout << (report_drift
        ? "assistant fixed-input checks passed; complete-forward drift reported without a numeric pass claim\n"
        : "assistant actual-weight two-step oracle passed\n");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "assistant oracle: " << error.what() << '\n';
    return 1;
  }
}
