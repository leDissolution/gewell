#include "mtp_staging.cuh"
#include "kv_write.cuh"
#include "cuda_memory.cuh"
#include "gewell/models/gemma4/26b_a4b/model.h"
#include <stdexcept>

namespace gewell::gemma4_26b_a4b::sm120 {
namespace {
using B = __nv_bfloat16;
constexpr unsigned kMaxVerifierRows = 1280;
constexpr std::size_t layer_width(unsigned layer) {
  return is_global_layer(layer) ? kGlobalKvHeadCount * kGlobalHeadSize
                               : kLocalKvHeadCount * kLocalHeadSize;
}
B* layer_key(const MtpStaging& staging, unsigned layer) {
  std::size_t offset = 0;
  for (unsigned l = 0; l < layer; ++l) offset += 2 * layer_width(l);
  return staging.data + offset * staging.capacity_rows;
}
}  // namespace

std::size_t mtp_staging_bytes(unsigned rows) {
  if (!rows || rows > kMaxVerifierRows)
    throw std::invalid_argument("26B MTP staging capacity must be 1..1280");
  return std::size_t(rows) * 2 * sizeof(B) *
      (kLocalLayerCount * kLocalKvHeadCount * kLocalHeadSize +
       kGlobalLayerCount * kGlobalKvHeadCount * kGlobalHeadSize);
}
void validate_mtp_staging(const MtpStaging& staging, unsigned rows) {
  const auto required = mtp_staging_bytes(staging.capacity_rows);
  if (!staging.data || !rows || rows > staging.capacity_rows || staging.bytes < required)
    throw std::invalid_argument("26B MTP staging does not cover verifier rows");
}
void stage_mtp_layer(const MtpStaging& staging, unsigned layer, unsigned rows,
                     const B* key, const B* value, cudaStream_t stream) {
  validate_mtp_staging(staging, rows);
  if (layer >= kLayerCount || !key || !value)
    throw std::invalid_argument("26B MTP invalid layer KV input");
  const auto width = layer_width(layer);
  auto* destination = layer_key(staging, layer);
  cuda_detail::check_cuda(cudaMemcpyAsync(destination, key, rows * width * sizeof(B),
      cudaMemcpyDeviceToDevice, stream), "stage 26B MTP K");
  cuda_detail::check_cuda(cudaMemcpyAsync(destination + staging.capacity_rows * width,
      value, rows * width * sizeof(B), cudaMemcpyDeviceToDevice, stream), "stage 26B MTP V");
}
void commit_mtp_rows(const std::vector<MtpCommit>& inputs, cudaStream_t stream) {
  if (inputs.empty()) throw std::invalid_argument("26B MTP empty commit batch");
  for (const auto& input : inputs) {
    validate_mtp_staging(input.staging, input.source_rows);
    if (input.accepted_rows > input.source_rows ||
        std::uint64_t(input.position) + input.source_rows > kMaxPositions)
      throw std::invalid_argument("26B MTP invalid accepted prefix");
    if (!input.accepted_rows) continue;
    for (unsigned l = 0; l < kLayerCount; ++l)
      validate_cache_view(input.cache[l], is_global_layer(l), input.position + input.accepted_rows);
  }
  for (const auto& input : inputs) {
    if (!input.accepted_rows) continue;
    for (unsigned l = 0; l < kLayerCount; ++l) {
      const auto* key = layer_key(input.staging, l);
      write_cache(key, key + input.staging.capacity_rows * layer_width(l), input.cache[l],
          is_global_layer(l), input.position, input.accepted_rows, stream);
    }
  }
}
}  // namespace gewell::gemma4_26b_a4b::sm120
