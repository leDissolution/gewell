#include "cuda_target.cuh"
#include "gewell/console.h"
#include "gewell/models/gemma4/31b/artifact.h"
#include "gewell/models/gemma4/31b/model.h"
#include "gewell/http_server.h"
#include "gewell/app.h"
#include "gewell/offline_runner.h"
#include "gewell/text_codec_cli.h"
#include "gewell/vision_engine.h"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef GEWELL_CUDA_ARCHITECTURE
#error "GEWELL_CUDA_ARCHITECTURE must name the configured CUDA target"
#endif

namespace console = gewell::console;
namespace artifact = gewell::artifact;
namespace model = gewell::gemma4_31b;
namespace app = gewell::app;

namespace {

std::uint32_t parse_positive_u32(std::string_view text,
                                 std::string_view label) {
  std::uint32_t value = 0;
  const char* const begin = text.data();
  const char* const end = begin + text.size();
  const auto parsed = std::from_chars(begin, end, value);
  if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != end ||
      value == 0) {
    throw std::runtime_error(std::string(label) +
                             " must be an integer in 1..4294967295");
  }
  return value;
}

std::uint64_t parse_nonnegative_u64(std::string_view text,
                                    std::string_view label) {
  std::uint64_t value = 0;
  const char* const begin = text.data();
  const char* const end = begin + text.size();
  const auto parsed = std::from_chars(begin, end, value);
  if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != end) {
    throw std::runtime_error(std::string(label) +
                             " must be an integer in 0..18446744073709551615");
  }
  return value;
}

std::uint32_t parse_mtp_depth(std::string_view text, std::string_view option = "--mtp-depth") {
  std::uint32_t value = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || parsed.ec != std::errc{} ||
      parsed.ptr != text.data() + text.size() || value > 1279) {
    throw std::runtime_error(
        std::string(option) + " must be an integer in 0..1279");
  }
  return value;
}

std::uint32_t parse_decode_width(std::string_view text) {
  const auto value = parse_nonnegative_u64(text, "--decode-width");
  if (value > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("--decode-width must be in 0..4294967295");
  return static_cast<std::uint32_t>(value);
}

std::uint32_t parse_prefill_chunk_tokens(std::string_view text) {
  const auto value = parse_positive_u32(text, "--prefill-chunk-tokens");
  if (value > gewell::gemma4_31b::sm120::kMaxPrefillChunkTokens)
    throw std::runtime_error("--prefill-chunk-tokens must be in 1..4096");
  return value;
}

std::uint32_t parse_prefill_batch_tokens(std::string_view text) {
  const auto value = parse_positive_u32(text, "--prefill-batch-tokens");
  if (value > gewell::gemma4_31b::sm120::kMaxPrefillBatchTokens)
    throw std::runtime_error("--prefill-batch-tokens must be in 1..4096");
  return value;
}

std::uint32_t parse_prefill_budget_tokens(std::string_view text) {
  const auto value = parse_nonnegative_u64(text, "--prefill-budget-tokens");
  if (value > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("--prefill-budget-tokens must be in 0..4294967295");
  return static_cast<std::uint32_t>(value);
}

gewell::nvfp4::ActivationPolicy parse_nvfp4_activation_policy(std::string_view text) {
  if (text == "always") return gewell::nvfp4::ActivationPolicy::always;
  if (text == "prefill") return gewell::nvfp4::ActivationPolicy::prefill;
  throw std::runtime_error("--nvfp4-activation-policy must be always or prefill");
}

gewell::kv_cache::Format parse_kv_format(std::string_view text) {
  if (text == "bf16") return gewell::kv_cache::Format::bf16;
  if (text == "fp8") return gewell::kv_cache::Format::fp8;
  throw std::runtime_error("KV format must be bf16 or fp8");
}

gewell::attention::Compute parse_attention_compute(std::string_view text) {
  if (text == "bf16") return gewell::attention::Compute::bf16;
  if (text == "fp8") return gewell::attention::Compute::fp8;
  throw std::runtime_error("Attention compute must be bf16 or fp8");
}

float parse_sampling_float(const char* text, const char* option) {
  char* end = nullptr;
  const float value = std::strtof(text, &end);
  if (end == text || *end || !std::isfinite(value) || value < 0)
    throw std::runtime_error(std::string(option) + " requires a finite nonnegative number");
  return value;
}

std::vector<std::uint32_t> parse_capture_layers(std::string_view value) {
  std::vector<std::uint32_t> layers;
  while (true) {
    const auto comma = value.find(',');
    const auto layer = parse_positive_u32(value.substr(0, comma), "--mtp-capture-layers");
    if (layer > model::kLayerCount || (!layers.empty() && layer <= layers.back()))
      throw std::runtime_error("--mtp-capture-layers must be increasing completed-layer counts in 1..60");
    layers.push_back(layer);
    if (comma == std::string_view::npos) break;
    value.remove_prefix(comma + 1);
  }
  return layers;
}

bool parse_runtime_option(std::string_view option, std::string_view value,
                          app::RuntimeSettings& settings) {
  if (option == "--attention-local-compute") {
    settings.local_attention_compute = parse_attention_compute(value);
  } else if (option == "--attention-global-compute") {
    settings.global_attention_compute = parse_attention_compute(value);
  } else if (option == "--kv-local-format") {
    settings.local_kv_format = parse_kv_format(value);
  } else if (option == "--kv-global-format") {
    settings.global_kv_format = parse_kv_format(value);
  } else if (option == "--mtp-depth") {
    settings.mtp_depth = parse_mtp_depth(value);
  } else if (option == "--mtp-min-depth") {
    settings.mtp_min_depth = parse_mtp_depth(value, option);
  } else if (option == "--decode-width") {
    settings.decode_width = parse_decode_width(value);
  } else if (option == "--mtp-stats") {
    if (value.empty() || value.substr(0, 2) == "--") throw std::runtime_error("--mtp-stats requires a path");
    settings.mtp_stats_path = value;
  } else if (option == "--mtp-capture") {
    if (value.empty() || value.substr(0, 2) == "--") throw std::runtime_error("--mtp-capture requires a path");
    settings.mtp_capture.path = value;
  } else if (option == "--mtp-capture-layers") {
    settings.mtp_capture.layers = parse_capture_layers(value);
  } else if (option == "--mtp-capture-every") {
    settings.mtp_capture.every = parse_positive_u32(value, option);
  } else if (option == "--mtp-capture-max-samples") {
    settings.mtp_capture.max_samples = parse_positive_u32(value, option);
  } else if (option == "--mtp-stats-window") {
    settings.mtp_stats_window = parse_positive_u32(value, option);
  } else if (option == "--prefill-chunk-tokens") {
    settings.prefill_chunk_tokens = parse_prefill_chunk_tokens(value);
  } else if (option == "--prefill-batch-tokens") {
    settings.prefill_batch_tokens = parse_prefill_batch_tokens(value);
  } else if (option == "--prefill-budget-tokens") {
    settings.prefill_budget_tokens = parse_prefill_budget_tokens(value);
  } else if (option == "--nvfp4-activation-policy") {
    settings.nvfp4_activation_policy = parse_nvfp4_activation_policy(value);
  } else if (option == "--kv-cache-gpu-mib") {
    settings.kv_cache_gpu_mib =
        parse_nonnegative_u64(value, "--kv-cache-gpu-mib");
  } else if (option == "--kv-cache-cpu-mib") {
    settings.kv_cache_cpu_mib =
        parse_nonnegative_u64(value, "--kv-cache-cpu-mib");
  } else if (option == "--kv-checkpoint-interval-tokens") {
    settings.kv_checkpoint_interval_tokens = parse_nonnegative_u64(
        value, "--kv-checkpoint-interval-tokens");
  } else if (option == "--kv-cache-index-mib") {
    settings.kv_cache_index_mib =
        parse_nonnegative_u64(value, "--kv-cache-index-mib");
  } else {
    return false;
  }
  return true;
}

void validate_runtime_settings(const app::RuntimeSettings& settings) {
  if (settings.kv_cache_gpu_mib == 0) {
    throw std::runtime_error(
        "GPU KV budget must be positive");
  }
  if (settings.kv_cache_index_mib == 0) {
    throw std::runtime_error("--kv-cache-index-mib must be positive");
  }
}

app::RuntimeSettings parse_runtime_settings(int argc, char** argv,
                                           int first_option,
                                           app::RuntimeSettings settings = {}) {
  for (int index = first_option; index < argc; index += 2) {
    if (index + 1 >= argc) {
      throw std::runtime_error(std::string(argv[index]) +
                               " requires a value");
    }
    const std::string_view option(argv[index]);
    if (!parse_runtime_option(option, argv[index + 1], settings)) {
      throw std::runtime_error("unknown runtime option: " +
                               std::string(option));
    }
  }
  validate_runtime_settings(settings);
  return settings;
}

std::uint16_t parse_port(std::string_view text) {
  const std::uint32_t value = parse_positive_u32(text, "--port");
  if (value > 65'535) {
    throw std::runtime_error("--port must be an integer in 1..65535");
  }
  return static_cast<std::uint16_t>(value);
}

void print_artifact_metadata(const artifact::ArtifactFile& file,
                             std::string_view validation) {
  const artifact::Header& header = file.header();
  console::section("Artifact");
  console::field("artifact", file.path());
  console::field("weight_format", header.native_mixed ? "gemma4-31b-mixed-v2" : header.native_nvfp4
                     ? "gemma4-31b-nvfp4-w4a4-v2" : "gemma4-31b-bf16-v4");
  console::field("validation", validation);
  console::field("physical_tensors", header.physical_tensor_count);
  console::field("logical_tensors", header.logical_tensor_count);
  console::field("aliases", header.alias_count);
  console::field("logical_data_bytes", header.logical_data_bytes);
  console::field("payload_bytes", header.payload_bytes);
  console::field("file_bytes", header.file_bytes);
  console::field("lm_head_logical_id", header.lm_head_logical_id);
  console::field("lm_head_target_physical_id", header.lm_head_target_id);
}

int inspect_artifact(const std::string& path) {
  artifact::ArtifactFile file = artifact::ArtifactFile::Open(path);
  print_artifact_metadata(file, "header+table (payload not scanned)");
  return 0;
}

artifact::Verification verify_and_report(artifact::ArtifactFile& file) {
  console::message("Verifying " + console::bytes(file.header().payload_bytes) +
                   " of payload with SHA-256...");
  const auto started = std::chrono::steady_clock::now();
  const artifact::Verification verification = file.VerifyFull();
  const auto elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - started);
  print_artifact_metadata(file, "full");
  console::field("verification_seconds", elapsed.count());
  console::field("payload_sha256", artifact::digest_hex(verification.payload_sha256));
  return verification;
}

int verify_artifact(const std::string& path) {
  artifact::ArtifactFile file = artifact::ArtifactFile::Open(path);
  verify_and_report(file);
  return 0;
}

const artifact::TensorEntry& find_entry(const artifact::ArtifactFile& file,
                                        model::TensorRole role,
                                        std::int16_t layer) {
  for (const artifact::TensorEntry& entry : file.entries()) {
    if (entry.role == role && entry.layer == layer) {
      return entry;
    }
  }
  throw std::runtime_error("required probe tensor is absent from contract");
}

int probe_artifact(const std::string& path) {
  artifact::ArtifactFile file = artifact::ArtifactFile::Open(path);
  verify_and_report(file);
  if (!validate_cuda_target(true)) {
    return 1;
  }

  const artifact::TensorEntry& scalar =
      find_entry(file, model::TensorRole::layer_scalar, 0);
  const artifact::TensorEntry& final_norm =
      find_entry(file, model::TensorRole::final_norm, model::kNoLayer);
  const std::size_t scalar_bytes = static_cast<std::size_t>(scalar.byte_length);
  const std::size_t norm_bytes = static_cast<std::size_t>(final_norm.byte_length);
  const std::size_t subset_bytes = scalar_bytes + norm_bytes;
  std::vector<std::uint8_t> expected(subset_bytes);
  std::vector<std::uint8_t> actual(subset_bytes);
  std::memcpy(expected.data(), file.tensor_data(scalar.physical_id),
              scalar_bytes);
  std::memcpy(expected.data() + scalar_bytes,
              file.tensor_data(final_norm.physical_id), norm_bytes);

  DeviceAllocation device(subset_bytes);
  require_cuda(cudaMemcpy(device.data(), expected.data(), subset_bytes,
                          cudaMemcpyHostToDevice),
               "probe host-to-device copy");
  require_cuda(cudaMemcpy(actual.data(), device.data(), subset_bytes,
                          cudaMemcpyDeviceToHost),
               "probe device-to-host copy");
  if (actual != expected) {
    throw std::runtime_error("probe byte roundtrip mismatch");
  }
  console::section("GPU roundtrip probe");
  console::field("probe", "passed");
  console::field("probe_physical_tensor_ids",
                 nlohmann::json::array({scalar.physical_id, final_norm.physical_id}));
  console::field("probe_roundtrip_bytes", subset_bytes);
  return 0;
}

int load_artifact(const std::string& path) {
  artifact::ArtifactFile file = artifact::ArtifactFile::Open(path);
  print_artifact_metadata(file, "header+table (payload not scanned)");
  if (!validate_cuda_target(true)) {
    return 1;
  }

  std::size_t free_before = 0;
  std::size_t total_bytes = 0;
  require_cuda(cudaMemGetInfo(&free_before, &total_bytes),
               "cudaMemGetInfo before load");
  const std::size_t payload_bytes =
      static_cast<std::size_t>(file.header().payload_bytes);
  DeviceAllocation arena(payload_bytes);
  const auto* source = file.payload_data();
  auto* destination = static_cast<std::uint8_t*>(arena.data());
  std::size_t copied = 0;
  while (copied < payload_bytes) {
    const std::size_t chunk =
        std::min(artifact::kIoChunkBytes, payload_bytes - copied);
    require_cuda(cudaMemcpy(destination + copied, source + copied, chunk,
                            cudaMemcpyHostToDevice),
                 "payload host-to-device copy");
    copied += chunk;
  }

  std::array<const void*, model::kLogicalTensorCount> tensor_pointers{};
  for (const artifact::TensorEntry& entry : file.entries()) {
    const std::uint64_t arena_offset =
        entry.file_offset - file.header().data_offset;
    tensor_pointers[entry.physical_id] = destination + arena_offset;
  }
  tensor_pointers[model::kLmHeadLogicalId] =
      tensor_pointers[file.header().lm_head_target_id];
  if (tensor_pointers[model::kLmHeadLogicalId] !=
      tensor_pointers[model::kEmbeddingPhysicalId]) {
    throw std::runtime_error("lm_head device pointer is not tied to embedding");
  }

  std::size_t free_after = 0;
  std::size_t total_after = 0;
  require_cuda(cudaMemGetInfo(&free_after, &total_after),
               "cudaMemGetInfo after load");
  if (total_after != total_bytes) {
    throw std::runtime_error("CUDA total memory changed during load");
  }
  if (free_after > free_before) {
    throw std::runtime_error("CUDA free memory increased during load");
  }
  console::section("GPU weight load");
  console::field("load", "passed");
  console::field("gpu_total_bytes", total_bytes);
  console::field("gpu_free_before_bytes", free_before);
  console::field("gpu_free_after_load_bytes", free_after);
  console::field("device_arena_bytes", arena.size());
  console::field("device_arena_address_mod_4096", (reinterpret_cast<std::uintptr_t>(arena.data()) %
                model::kStorageAlignment));
  console::field("device_free_delta_bytes", free_before - free_after);
  console::field("lm_head_pointer_tied", true);

  arena.FreeChecked();
  std::size_t free_after_release = 0;
  require_cuda(cudaMemGetInfo(&free_after_release, &total_after),
               "cudaMemGetInfo after release");
  console::field("gpu_free_after_release_bytes", free_after_release);
  return 0;
}

void print_usage(const char* executable) {
  console::message(std::string("Usage: ") + executable + R"( [--log-format human|json] COMMAND ...

Serving
  serve-http --model-dir DIR --max-batch N --kv-cache-gpu-mib N [options]

Generation
  generate ARTIFACT PROMPT.u32 NEW_TOKENS OUTPUT.u32 [LOGITS.bf16]
  caption ARTIFACT PROMPT.u32 PIXELS.f32 POSITIONS.i32
          MAX_NEW_TOKENS OUTPUT.u32 [LOGITS.bf16]
  run-jobs ARTIFACT MAX_BATCH KV_MIB [runtime options]  # local JSONL stdin/stdout
  generate-batch ARTIFACT REQUESTS.tsv MAX_BATCH KV_MIB OUTPUT_DIR [EVENTS.tsv]
  replay-rollout ARTIFACT REQUESTS.tsv CHUNK_ROWS HEAD_ROWS OUTPUT_DIR

Inspection and validation
  inspect PATH                       Show artifact metadata
  verify PATH                        Verify artifact hashes
  probe PATH                         Check tensor access
  load PATH                          Measure GPU weight loading
  text-codec MODEL_DIR                Run the JSONL tokenizer protocol

Options
  --log-format human|json             Readable console (default) or JSON Lines;
                                     accepted before or after the command
  --assistant PATH                   Original assistant model.safetensors; loaded only for positive MTP depth
  --vision PATH                      Extracted vision + projector safetensors; omission disables image input
  --mtp-depth N                      Maximum proposal depth, 0..1279 (default 0 disables MTP);
                                     warns and falls back to 0 without --assistant
  --mtp-min-depth N                  Minimum adaptive proposal depth (default 0, <= --mtp-depth)
  --decode-width N                   Target pending + proposal rows per decode batch;
                                     0 (default) uses fixed --mtp-depth; batch commands only
  --mtp-stats PATH                   Append windowed MTP statistics to a JSONL file (batch commands)
  --mtp-stats-window N               Decode cycles per request window (default 64, positive)
  --mtp-capture DIR                  Append sampled training pairs in a directory (batch commands)
  --mtp-capture-every N              Capture approximately 1/N MTP rounds (default 32, positive)
  --mtp-capture-max-samples N        Capture attempts per launch (default 250000, positive)
  --mtp-capture-layers LIST          Completed target layers to probe (default 4,12,24,40,56; 1..60)
  --prefill-chunk-tokens N            Per-prompt text chunk cap, 1..4096 (default 1024)
  --prefill-batch-tokens N            Combined text GEMM rows, 1..4096 (default 2048, >= chunk cap)
  --prefill-budget-tokens N           Prefill tokens between batch decodes (0: each forward/head)
  --nvfp4-activation-policy POLICY   always (default) or prefill; prefill keeps NVFP4 decode/MTP activations BF16
  --attention-local-compute bf16|fp8 Local text attention, including MTP (default bf16)
  --attention-global-compute bf16|fp8 Global text attention, including MTP (default bf16)
  --kv-local-format bf16|fp8          Local KV storage (default bf16)
  --kv-global-format bf16|fp8         Compact global KV storage (default bf16)
  --kv-cache-cpu-mib N                Host prefix-cache budget
  --kv-cache-index-mib N              Host cache-index budget (default 512 MiB)
  --kv-checkpoint-interval-tokens N   Periodic checkpoint spacing (0 disables it)

HTTP options (after serve-http)
  --host HOST                        IPv4 bind address (default 127.0.0.1)
  --port N                           HTTP port (default 6311)
  --model NAME                       Public model name
  --image-max-soft-tokens N           Default tokens per image: 70, 140, 280 (default), 560, 1120
  --max-connections N                 Accepted connection limit
  --max-body-bytes N                  Per-request input limit
  --max-body-total-bytes N            Combined input limit
  --max-output-bytes N                Buffered output limit
  --socket-timeout-seconds N          Connection timeout
  --qdq-mask PATH                     Weight quantization mask

Generation options (before generate/caption)
  --temperature T  --top-p P  --top-k K  --seed S  --mtp-depth N
  Batch and server sampling settings belong to individual requests.
  --mtp-depth, --mtp-min-depth, --decode-width, and prefill options may precede batching commands.
  --prefill-chunk-tokens leaves the one-chunk image limit at 1280.
  --qdq-mask PATH precedes generate/generate-batch/run-jobs/replay-rollout.

KV storage: compact-global BF16
See docs/cli.md and README.md for command examples and cache semantics.
)");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    // Console format is global, including commands whose remaining arguments are positional.
    std::string assistant_path, vision_path;
    int remaining = 1;
    for (int index = 1; index < argc; ++index) {
      if (std::string_view(argv[index]) == "--log-format") {
        if (index + 1 == argc) throw std::runtime_error("--log-format requires a value");
        console::set_format(argv[++index]);
      } else if (std::string_view(argv[index]) == "--assistant" || std::string_view(argv[index]) == "--vision") {
        const std::string option(argv[index]);
        if (index + 1 == argc || !argv[index + 1][0] || std::string_view(argv[index + 1]).substr(0, 2) == "--")
          throw std::runtime_error(option + " requires a path");
        auto& path = option == "--assistant" ? assistant_path : vision_path;
        if (!path.empty()) throw std::runtime_error(option + " was supplied more than once");
        path = argv[++index];
      } else argv[remaining++] = argv[index];
    }
    argc = remaining;
    argv[argc] = nullptr;
    if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
      print_usage(argv[0]);
      return 0;
    }
    app::GenerationSettings generation_settings;
    generation_settings.assistant_path = assistant_path;
    generation_settings.vision_path = vision_path;
    bool generation_options = false;
    bool batch_sampling_options = false;
    bool prefill_budget_option = false;
    bool prefill_batch_option = false;
    bool adaptive_depth_options = false;
    bool mtp_stats_options = false;
    bool mtp_capture_options = false;
    while (argc >= 2) {
      const std::string_view option(argv[1]);
      if (option != "--mtp-depth" && option != "--mtp-min-depth" && option != "--decode-width" &&
          option != "--mtp-stats" && option != "--mtp-stats-window" &&
          option != "--mtp-capture" && option != "--mtp-capture-every" && option != "--mtp-capture-max-samples" && option != "--mtp-capture-layers" &&
          option != "--temperature" && option != "--top-p" &&
          option != "--top-k" && option != "--seed" && option != "--prefill-chunk-tokens" && option != "--prefill-batch-tokens" && option != "--prefill-budget-tokens" &&
          option != "--nvfp4-activation-policy" && option != "--kv-local-format" &&
          option != "--kv-global-format" && option != "--attention-local-compute" && option != "--attention-global-compute") break;
      if (argc < 3) throw std::runtime_error(std::string(option) + " requires a value");
      generation_options = true;
      prefill_budget_option |= option == "--prefill-budget-tokens";
      prefill_batch_option |= option == "--prefill-batch-tokens";
      adaptive_depth_options |= option == "--mtp-min-depth" || option == "--decode-width";
      mtp_stats_options |= option == "--mtp-stats" || option == "--mtp-stats-window";
      const bool capture_option = option == "--mtp-capture" || option == "--mtp-capture-every" || option == "--mtp-capture-max-samples" || option == "--mtp-capture-layers";
      mtp_capture_options |= capture_option;
      batch_sampling_options |= !capture_option && option != "--mtp-stats" && option != "--mtp-stats-window" &&
          option != "--mtp-depth" && option != "--mtp-min-depth" && option != "--decode-width" && option != "--prefill-chunk-tokens" && option != "--prefill-batch-tokens" && option != "--prefill-budget-tokens" && option != "--nvfp4-activation-policy" && option != "--kv-local-format" && option != "--kv-global-format" && option != "--attention-local-compute" && option != "--attention-global-compute";
      if (option == "--nvfp4-activation-policy") generation_settings.nvfp4_activation_policy = parse_nvfp4_activation_policy(argv[2]);
      if (option == "--prefill-chunk-tokens") generation_settings.prefill_chunk_tokens = parse_prefill_chunk_tokens(argv[2]);
      if (option == "--prefill-batch-tokens") generation_settings.prefill_batch_tokens = parse_prefill_batch_tokens(argv[2]);
      if (option == "--prefill-budget-tokens") generation_settings.prefill_budget_tokens = parse_prefill_budget_tokens(argv[2]);
      if (option == "--attention-local-compute") generation_settings.local_attention_compute = parse_attention_compute(argv[2]);
      if (option == "--attention-global-compute") generation_settings.global_attention_compute = parse_attention_compute(argv[2]);
      if (option == "--kv-local-format") generation_settings.local_kv_format = parse_kv_format(argv[2]);
      if (option == "--kv-global-format") generation_settings.global_kv_format = parse_kv_format(argv[2]);
      if (option == "--mtp-depth") generation_settings.mtp_depth = parse_mtp_depth(argv[2]);
      if (option == "--mtp-min-depth") generation_settings.mtp_min_depth = parse_mtp_depth(argv[2], option);
      if (option == "--decode-width") generation_settings.decode_width = parse_decode_width(argv[2]);
      if (option == "--mtp-stats-window") generation_settings.mtp_stats_window = parse_positive_u32(argv[2], option);
      if (option == "--mtp-capture-layers") generation_settings.mtp_capture.layers = parse_capture_layers(argv[2]);
      if (option == "--mtp-capture-every") generation_settings.mtp_capture.every = parse_positive_u32(argv[2], option);
      if (option == "--mtp-capture-max-samples") generation_settings.mtp_capture.max_samples = parse_positive_u32(argv[2], option);
      if (option == "--mtp-capture") {
        if (!argv[2][0] || std::string_view(argv[2]).substr(0, 2) == "--")
          throw std::runtime_error("--mtp-capture requires a path");
        generation_settings.mtp_capture.path = argv[2];
      }
      if (option == "--mtp-stats") {
        if (!argv[2][0] || std::string_view(argv[2]).substr(0, 2) == "--")
          throw std::runtime_error("--mtp-stats requires a path");
        generation_settings.mtp_stats_path = argv[2];
      }
      if (option == "--temperature") generation_settings.temperature = parse_sampling_float(argv[2], "--temperature");
      if (option == "--top-p") {
        generation_settings.top_p = parse_sampling_float(argv[2], "--top-p");
        if (generation_settings.top_p > 1) throw std::runtime_error("--top-p must be in [0,1]");
      }
      if (option == "--top-k") {
        const auto k = parse_nonnegative_u64(argv[2], "--top-k");
        if (k > model::kVocabSize) throw std::runtime_error("--top-k exceeds vocabulary");
        generation_settings.top_k = static_cast<std::uint32_t>(k);
      }
      if (option == "--seed") generation_settings.seed = parse_nonnegative_u64(argv[2], "--seed");
      argc -= 2; argv += 2;
    }
    std::string qdq_mask;
    if (argc >= 2 && std::string_view(argv[1]) == "--qdq-mask") {
      if (argc < 4) throw std::runtime_error("--qdq-mask requires a path and command");
      qdq_mask = argv[2];
      argc -= 2; argv += 2;
    }
    if (generation_options && argc >= 2) {
      const std::string_view command(argv[1]);
      if (mtp_capture_options && command != "generate-batch" &&
          command != "serve-http" && command != "run-jobs")
        throw std::runtime_error("--mtp-capture options require a batching command");
      if (mtp_stats_options && command != "generate-batch" &&
          command != "serve-http" && command != "run-jobs")
        throw std::runtime_error("--mtp-stats and --mtp-stats-window require a batching command");
      if (adaptive_depth_options && command != "generate-batch" &&
          command != "serve-http" && command != "run-jobs")
        throw std::runtime_error("--mtp-min-depth and --decode-width require a batching command");
      if (prefill_batch_option && command != "generate-batch" && command != "run-jobs" && command != "serve-http")
        throw std::runtime_error("--prefill-batch-tokens requires a batch command");
      if (prefill_budget_option && command != "generate-batch" &&
          command != "serve-http" && command != "run-jobs")
        throw std::runtime_error("--prefill-budget-tokens requires a batching command");
      if (command != "generate" && command != "caption" && command != "generate-batch" &&
          command != "serve-http" && command != "run-jobs")
        throw std::runtime_error("generation options precede generate/caption or batching commands");
    }
    if (!qdq_mask.empty() && argc >= 2) {
      const std::string_view command(argv[1]);
      if (command != "generate" && command != "generate-batch" && command != "run-jobs" &&
          command != "serve-http" && command != "replay-rollout")
        throw std::runtime_error("--qdq-mask requires generate, generate-batch, run-jobs, serve-http or replay-rollout");
    }
    if (argc == 3) {
      const std::string_view command = argv[1];
      if (command == "text-codec") {
        return gewell::text::run_text_codec(argv[2]);
      }
      if (command == "inspect") {
        return inspect_artifact(argv[2]);
      }
      if (command == "verify") {
        return verify_artifact(argv[2]);
      }
      if (command == "probe") {
        return probe_artifact(argv[2]);
      }
      if (command == "load") {
        return load_artifact(argv[2]);
      }
    }
    if ((argc == 7 || argc == 8) && std::string_view(argv[1]) == "generate-batch") {
      if (batch_sampling_options) throw std::runtime_error("batch sampling belongs in REQUESTS.tsv");
      return app::run_generate_batch(argv[2], argv[3],
          parse_positive_u32(argv[4], "MAX_BATCH"),
          parse_nonnegative_u64(argv[5], "KV_MIB"), argv[6], qdq_mask, generation_settings.mtp_depth,
          argc == 8 ? argv[7] : "", generation_settings.prefill_chunk_tokens, generation_settings.prefill_batch_tokens, generation_settings.nvfp4_activation_policy,
          generation_settings.local_kv_format, generation_settings.global_kv_format,
          generation_settings.local_attention_compute, generation_settings.global_attention_compute, assistant_path, vision_path,
          generation_settings.prefill_budget_tokens, generation_settings.mtp_min_depth, generation_settings.decode_width,
          generation_settings.mtp_stats_path, generation_settings.mtp_stats_window, generation_settings.mtp_capture);
    }
    if (argc >= 2 && std::string_view(argv[1]) == "serve-http") {
      if (batch_sampling_options) throw std::runtime_error("serve-http sampling settings belong to each request");
      app::RuntimeSettings settings;
      settings.assistant_path = assistant_path;
      settings.vision_path = vision_path;
      settings.mtp_depth = generation_settings.mtp_depth;
      settings.mtp_min_depth = generation_settings.mtp_min_depth;
      settings.decode_width = generation_settings.decode_width;
      settings.mtp_stats_path = generation_settings.mtp_stats_path;
      settings.mtp_stats_window = generation_settings.mtp_stats_window;
      settings.mtp_capture = generation_settings.mtp_capture;
      settings.prefill_chunk_tokens = generation_settings.prefill_chunk_tokens;
      settings.prefill_batch_tokens = generation_settings.prefill_batch_tokens;
      settings.prefill_budget_tokens = generation_settings.prefill_budget_tokens;
      settings.nvfp4_activation_policy = generation_settings.nvfp4_activation_policy;
      settings.local_attention_compute = generation_settings.local_attention_compute;
      settings.global_attention_compute = generation_settings.global_attention_compute;
      settings.local_kv_format = generation_settings.local_kv_format;
      settings.global_kv_format = generation_settings.global_kv_format;
      gewell::http::Settings http_settings;
      http_settings.model = "gewell-gemma-4-31b-bf16";
      std::string model_directory;
      std::uint32_t max_batch = 0;
      bool has_gpu_budget = false;
      std::string mask = qdq_mask;
      for (int index = 2; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::runtime_error(std::string(argv[index]) + " requires a value");
        const std::string_view option(argv[index]);
        if (option == "--model-dir") model_directory = argv[index + 1];
        else if (option == "--host") http_settings.host = argv[index + 1];
        else if (option == "--port") http_settings.port = parse_port(argv[index + 1]);
        else if (option == "--max-batch") max_batch = parse_positive_u32(argv[index + 1], "--max-batch");
        else if (option == "--model") http_settings.model = argv[index + 1];
        else if (option == "--image-max-soft-tokens") {
          settings.image_max_soft_tokens = parse_positive_u32(argv[index + 1], option);
          if (!gewell::vision_engine::is_supported_soft_token_capacity(settings.image_max_soft_tokens))
            throw std::runtime_error("--image-max-soft-tokens must be one of 70, 140, 280, 560, 1120");
        }
        else if (option == "--qdq-mask") mask = argv[index + 1];
        else if (option == "--max-connections") http_settings.max_connections = parse_positive_u32(argv[index + 1], "--max-connections");
        else if (option == "--max-body-bytes") http_settings.max_body_bytes = parse_positive_u32(argv[index + 1], "--max-body-bytes");
        else if (option == "--max-body-total-bytes") http_settings.max_body_total_bytes = parse_positive_u32(argv[index + 1], "--max-body-total-bytes");
        else if (option == "--max-output-bytes") http_settings.max_output_bytes = parse_positive_u32(argv[index + 1], "--max-output-bytes");
        else if (option == "--socket-timeout-seconds") http_settings.socket_timeout_seconds = parse_positive_u32(argv[index + 1], "--socket-timeout-seconds");
        else {
          if (!parse_runtime_option(option, argv[index + 1], settings))
            throw std::runtime_error("unknown serve-http option: " + std::string(option));
          has_gpu_budget |= option == "--kv-cache-gpu-mib";
        }
      }
      if (model_directory.empty()) throw std::runtime_error("--model-dir is required for serve-http");
      if (!max_batch) throw std::runtime_error("--max-batch is required for serve-http");
      if (!has_gpu_budget)
        throw std::runtime_error("--kv-cache-gpu-mib is required for serve-http");
      validate_runtime_settings(settings);
      return app::run_http_server(model_directory, max_batch,
          settings, http_settings, mask);
    }
    if (argc >= 5 && std::string_view(argv[1]) == "run-jobs") {
      if (batch_sampling_options) throw std::runtime_error("run-jobs sampling belongs to each job");
      app::RuntimeSettings settings;
      settings.assistant_path = assistant_path;
      settings.vision_path = vision_path;
      settings.kv_cache_gpu_mib = parse_nonnegative_u64(argv[4], "KV_MIB");
      settings.mtp_depth = generation_settings.mtp_depth;
      settings.mtp_min_depth = generation_settings.mtp_min_depth;
      settings.decode_width = generation_settings.decode_width;
      settings.mtp_stats_path = generation_settings.mtp_stats_path;
      settings.mtp_stats_window = generation_settings.mtp_stats_window;
      settings.mtp_capture = generation_settings.mtp_capture;
      settings.prefill_chunk_tokens = generation_settings.prefill_chunk_tokens;
      settings.prefill_batch_tokens = generation_settings.prefill_batch_tokens;
      settings.prefill_budget_tokens = generation_settings.prefill_budget_tokens;
      settings.nvfp4_activation_policy = generation_settings.nvfp4_activation_policy;
      settings.local_attention_compute = generation_settings.local_attention_compute;
      settings.global_attention_compute = generation_settings.global_attention_compute;
      settings.local_kv_format = generation_settings.local_kv_format;
      settings.global_kv_format = generation_settings.global_kv_format;
      for (int index = 5; index < argc; index += 2) {
        if (index + 1 >= argc) throw std::runtime_error(std::string(argv[index]) + " requires a value");
        if (std::string_view(argv[index]) == "--qdq-mask") qdq_mask = argv[index + 1];
        else settings = parse_runtime_settings(2, argv + index, 0, settings);
      }
      return app::run_jobs(argv[2], parse_positive_u32(argv[3], "MAX_BATCH"), settings, qdq_mask);
    }
    if (argc == 7 && std::string_view(argv[1]) == "replay-rollout") {
      return app::run_replay_rollout(argv[2], argv[3],
          parse_positive_u32(argv[4], "CHUNK_ROWS"),
          parse_positive_u32(argv[5], "HEAD_ROWS"), argv[6], qdq_mask);
    }
    if ((argc == 6 || argc == 7) && std::string_view(argv[1]) == "generate") {
      return app::run_generate(argv[2], argv[3], parse_positive_u32(argv[4], "NEW_TOKENS"),
          argv[5], argc == 7 ? argv[6] : "", qdq_mask, generation_settings);
    }
    if ((argc == 8 || argc == 9) && std::string_view(argv[1]) == "caption") {
      return app::run_caption(argv[2], argv[3], argv[4], argv[5],
          parse_positive_u32(argv[6], "MAX_NEW_TOKENS"), argv[7],
          argc == 9 ? argv[8] : "", generation_settings);
    }
    print_usage(argv[0]);
    return 2;
  } catch (const std::exception& error) {
    console::message(error.what(), true);
    return 1;
  }
}
