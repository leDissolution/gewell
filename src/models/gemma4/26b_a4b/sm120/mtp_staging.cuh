#pragma once
#include "gewell/kv_view.h"
#include <array>
#include <cuda_runtime.h>
#include <vector>

namespace gewell::gemma4_26b_a4b::sm120 {
// Full BF16 current K/V, token-major within each layer. Layer slots retain
// capacity_rows stride even when the active verifier or accepted prefix shrinks.
// This allocation is caller-owned and charged to the speculative KV budget.
struct MtpStaging {
  __nv_bfloat16* data{};
  std::size_t bytes{};
  unsigned capacity_rows{};
};
std::size_t mtp_staging_bytes(unsigned capacity_rows);
void validate_mtp_staging(const MtpStaging& staging, unsigned source_rows);
void stage_mtp_layer(const MtpStaging& staging, unsigned layer, unsigned rows,
                     const __nv_bfloat16* key, const __nv_bfloat16* value,
                     cudaStream_t stream);
struct MtpCommit {
  std::array<kv_cache::DeviceView, 30> cache{};
  MtpStaging staging{};
  unsigned position{}, source_rows{}, accepted_rows{};
};
// All requests/layers are validated before any cache write. Only accepted rows
// are compacted/quantized into committed storage. Local rings retain their last
// 1024 accepted rows; rejected rows never overwrite committed history.
void commit_mtp_rows(const std::vector<MtpCommit>& inputs, cudaStream_t stream);
}  // namespace gewell::gemma4_26b_a4b::sm120
