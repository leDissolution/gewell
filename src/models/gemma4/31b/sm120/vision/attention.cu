#include "gewell/vision_primitives.h"

#include "kernel_forward.h"

#include <stdexcept>
#include <string>

namespace gewell::vision_primitives {
namespace {
// The pinned CUTLASS fused-attention implementation uses SM80 tensor-core
// instructions supported by SM120. These shapes hold the 72-wide output in
// registers while streaming keys; no per-launch device workspace is needed.
template <int QueryRows>
using Attention = AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80,
                                  true, QueryRows, 128, 128, false, false>;

void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template <int QueryRows>
void run_attention(const BFloat16* query, const BFloat16* key,
                   const BFloat16* value, BFloat16* output,
                   std::uint32_t patch_rows, cudaStream_t stream) {
  using Kernel = Attention<QueryRows>;
  static_assert(!Kernel::kNeedsOutputAccumulatorBuffer);
  constexpr auto kernel = attention_kernel_batched_impl<Kernel>;
  constexpr int shared_bytes = sizeof(typename Kernel::SharedStorage);
  static const bool configured = [] {
    if constexpr (shared_bytes > 48 * 1024)
      check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes),
            "configure vision attention shared memory");
    return true;
  }();
  (void)configured;

  typename Kernel::Params p;
  p.query_ptr = reinterpret_cast<cutlass::bfloat16_t*>(const_cast<BFloat16*>(query));
  p.key_ptr = reinterpret_cast<cutlass::bfloat16_t*>(const_cast<BFloat16*>(key));
  p.value_ptr = reinterpret_cast<cutlass::bfloat16_t*>(const_cast<BFloat16*>(value));
  p.output_ptr = reinterpret_cast<cutlass::bfloat16_t*>(output);
  p.scale = 1.0F;
  p.num_heads = gemma4_31b::kVisionHeadCount;
  p.num_batches = 1;
  p.head_dim = p.head_dim_value = gemma4_31b::kVisionHeadSize;
  p.num_queries = p.num_keys = patch_rows;
  p.q_strideM = p.k_strideM = p.v_strideM = p.head_dim;
  p.q_strideH = p.k_strideH = p.v_strideH = patch_rows * p.head_dim;
  p.q_strideB = p.k_strideB = p.v_strideB = p.num_heads * p.q_strideH;
  p.o_strideM = p.num_heads * p.head_dim_value;
  if (!Kernel::check_supported(p))
    throw std::invalid_argument("vision attention: unsupported tensor alignment");
  kernel<<<p.getBlocksGrid(), p.getThreadsGrid(), shared_bytes, stream>>>(p);
  check(cudaGetLastError(), "launch fused vision attention");
}
}  // namespace

void full_attention(const BFloat16* query, const BFloat16* key,
                    const BFloat16* value, BFloat16* output,
                    std::uint32_t patch_rows, cudaStream_t stream) {
  if (!query || !key || !value || !output || !patch_rows || patch_rows > kMaxPatchRows)
    throw std::invalid_argument("vision attention: invalid tensor or patch count");
  // Offline SM120 measurements: smaller query tiles improve the default image
  // tier, while large images benefit from reusing each K/V tile for more queries.
  // The key tile and reduction order stay fixed at 128 in every specialization.
  if (patch_rows >= 8192)
    run_attention<128>(query, key, value, output, patch_rows, stream);
  else if (patch_rows >= 1536 && patch_rows <= 3072)
    run_attention<32>(query, key, value, output, patch_rows, stream);
  else
    run_attention<64>(query, key, value, output, patch_rows, stream);
}
}  // namespace gewell::vision_primitives
