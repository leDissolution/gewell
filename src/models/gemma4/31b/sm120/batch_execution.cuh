#pragma once
#include "executor.cuh"
#include "cache_config.h"

namespace gewell::gemma4_31b::sm120 {
struct BatchMemoryPlan {
  static constexpr std::size_t hidden_bytes = std::size_t(model::kHiddenSize) * sizeof(BFloat16);
  std::size_t staging_bytes{}, hidden_slots{}, hidden_staging_bytes{}, committed_kv_bytes{};
  BatchMemoryPlan(std::size_t total, std::uint32_t capacity, std::uint32_t depth,
                  kv_cache::Format local, kv_cache::Format global)
      : staging_bytes(depth ? capacity * mtp_target::Verifier::staging_bytes(depth + 1) : 0),
        hidden_slots(total / compact_pool_config(total, 0, kv_cache::kDefaultIndexBytes, local, global).local_ring_bytes),
        hidden_staging_bytes(hidden_slots * hidden_bytes),
        committed_kv_bytes(committed_bytes(total, staging_bytes, hidden_staging_bytes)) {}
 private:
  static std::size_t committed_bytes(std::size_t total, std::size_t mtp, std::size_t hidden) {
    if (!hidden || mtp >= total || hidden >= total - mtp)
      fail("batch configuration", "staging exceeds GPU KV budget");
    return total - mtp - hidden;
  }

};

class BatchExecution {
 public:
  BatchExecution(const BatchMemoryPlan& memory, std::uint32_t capacity, std::uint32_t depth)
      : memory(memory), capacity(capacity), mtp_depth(depth),
        hidden_staging(memory.hidden_staging_bytes), hidden_used(memory.hidden_slots, false) {}
  ~BatchExecution() { if (engine) engine->wait(); }
  void allocate_staging() {
    if (memory.staging_bytes) staging = std::make_unique<DeviceAllocation>(memory.staging_bytes);
  }
  void initialize_executor(const WeightArena& weights, ExecutionCache& cache,
      kv_cache::ExecutionId bootstrap, std::uint32_t plan_rows, std::uint32_t max_horizon,
      bool sampled, std::uint32_t prefill_chunk_tokens, std::uint32_t prefill_batch_tokens,
      nvfp4::ActivationPolicy activation_policy,
      attention::Compute local_compute, attention::Compute global_compute) {
      engine = std::make_unique<Executor>(
          weights, plan_rows, max_horizon - plan_rows + 1,
          false, SamplingSettings{}, &cache, bootstrap, std::vector<CheckpointTrigger>{},
          0, std::nullopt, prefill_chunk_tokens, activation_policy, capacity,
          kv_cache::Format::bf16, kv_cache::Format::bf16, local_compute, global_compute, prefill_batch_tokens);
      engine->initialize_batch(capacity, sampled, mtp_depth,
                              staging ? staging->data() : nullptr, memory.staging_bytes);
  }
  void initialize_outputs(bool captures, bool logprobs) {
    host_ids = std::make_unique<PinnedHostAllocation>(capacity * sizeof(std::uint32_t));
    if (captures) host_logits = std::make_unique<PinnedHostAllocation>(
        std::size_t(capacity) * (mtp_depth + 1) * kGenerationLogitRowBytes);
    if (logprobs) {
      const auto records = std::size_t(capacity) * (mtp_depth + 1);
      device_logprobs = std::make_unique<DeviceAllocation>(
          records * sizeof(TokenLogprobs));
      host_logprobs = std::make_unique<PinnedHostAllocation>(
          records * sizeof(TokenLogprobs));
    }
  }
  BFloat16* acquire_hidden() {
    const auto slot = std::find(hidden_used.begin(), hidden_used.end(), false);
    if (slot == hidden_used.end()) fail("batch prefix", "execution hidden slots exhausted");
    const auto index = static_cast<std::size_t>(slot - hidden_used.begin());
    hidden_used[index] = true;
    return static_cast<BFloat16*>(hidden_staging.data()) + index * model::kHiddenSize;
  }

  void release_hidden(BFloat16* hidden) {
    if (!hidden) return;
    const auto index = (hidden - static_cast<BFloat16*>(hidden_staging.data())) / model::kHiddenSize;
    if (index < 0 || static_cast<std::size_t>(index) >= hidden_used.size() || !hidden_used[index])
      fail("batch prefix", "invalid terminal-state release");
    hidden_used[index] = false;
  }


  std::size_t occupied_hidden_bytes() const {
    return std::count(hidden_used.begin(), hidden_used.end(), true) * BatchMemoryPlan::hidden_bytes;
  }
  void begin_step(std::string_view operation) {
    check_cuda(cudaEventRecord(begin.get(), engine->stream()), operation);
  }
  void end_step(std::string_view operation) {
    check_cuda(cudaEventRecord(end.get(), engine->stream()), operation);
  }
  float elapsed(std::string_view operation) const {
    float milliseconds = 0;
    check_cuda(cudaEventElapsedTime(&milliseconds, begin.get(), end.get()), operation);
    return milliseconds;
  }
  void download_ids(std::size_t rows) {
    check_cuda(cudaMemcpyAsync(host_ids->data(), engine->batch_output_ids(),
        rows * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, engine->stream()), "copy batch decisions");
  }
  void download_logits(std::size_t row) {
    check_cuda(cudaMemcpyAsync(static_cast<std::uint8_t*>(host_logits->data()) + row * kGenerationLogitRowBytes,
        engine->batch_capped_logits(row), kGenerationLogitRowBytes, cudaMemcpyDeviceToHost,
        engine->stream()), "copy batch logits");
  }
  void download_logprobs(std::size_t row) {
    check_cuda(cudaMemcpyAsync(static_cast<TokenLogprobs*>(host_logprobs->data()) + row,
        static_cast<TokenLogprobs*>(device_logprobs->data()) + row, sizeof(TokenLogprobs),
        cudaMemcpyDeviceToHost, engine->stream()), "copy batch logprobs");
  }
  const BatchMemoryPlan& memory;
  const std::uint32_t capacity, mtp_depth;
  DeviceAllocation hidden_staging;
  std::vector<bool> hidden_used;
  std::unique_ptr<DeviceAllocation> staging;
  std::unique_ptr<Executor> engine;
  std::unique_ptr<DeviceAllocation> device_logprobs;
  std::unique_ptr<PinnedHostAllocation> host_ids, host_logits, host_logprobs;
  CudaEvent begin, end;
};

}
