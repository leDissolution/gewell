#include "attention.h"
#include "gewell/models/embeddinggemma2/vision.h"

#include "kernel_forward.h"

#include <stdexcept>
#include <string>

namespace gewell::embeddinggemma2 {
namespace {
template <int QueryRows>
using Attention = AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80,
                                  true, QueryRows, 128, vision::kHeadDim, false, false>;

template <int QueryRows>
void run(const __nv_bfloat16* query, const __nv_bfloat16* key, const __nv_bfloat16* value,
         __nv_bfloat16* output, int rows, const int* starts, int images) {
  using Kernel = Attention<QueryRows>;
  static_assert(!Kernel::kNeedsOutputAccumulatorBuffer);
  constexpr auto kernel = attention_kernel_batched_impl<Kernel>;
  constexpr int shared_bytes = sizeof(typename Kernel::SharedStorage);
  static const cudaError_t configured = shared_bytes > 48 * 1024
      ? cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes)
      : cudaSuccess;
  if (configured != cudaSuccess)
    throw std::runtime_error(std::string("configure embedding vision attention: ") + cudaGetErrorString(configured));
  typename Kernel::Params p;
  p.query_ptr = reinterpret_cast<cutlass::bfloat16_t*>(const_cast<__nv_bfloat16*>(query));
  p.key_ptr = reinterpret_cast<cutlass::bfloat16_t*>(const_cast<__nv_bfloat16*>(key));
  p.value_ptr = reinterpret_cast<cutlass::bfloat16_t*>(const_cast<__nv_bfloat16*>(value));
  p.output_ptr = reinterpret_cast<cutlass::bfloat16_t*>(output);
  p.scale = 1.0F;
  p.num_heads = vision::kHeads;
  p.num_batches = images;
  p.seqstart_q_ptr = p.seqstart_k_ptr = const_cast<int*>(starts);
  p.head_dim = p.head_dim_value = vision::kHeadDim;
  p.num_queries = p.num_keys = rows;
  p.q_strideM = p.k_strideM = p.v_strideM = vision::kHidden;
  p.q_strideH = p.k_strideH = p.v_strideH = vision::kHeadDim;
  p.q_strideB = p.k_strideB = p.v_strideB = static_cast<int64_t>(rows) * vision::kHidden;
  p.o_strideM = vision::kHidden;
  if (!Kernel::check_supported(p))
    throw std::invalid_argument("embedding vision attention: unsupported tensor alignment");
  kernel<<<p.getBlocksGrid(), p.getThreadsGrid(), shared_bytes>>>(p);
  const auto status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("launch embedding vision attention: ") + cudaGetErrorString(status));
}
}  // namespace

void vision_attention(const __nv_bfloat16* query, const __nv_bfloat16* key,
                      const __nv_bfloat16* value, __nv_bfloat16* output, int rows,
                      const int* starts, int images) {
  if (rows <= 0 || rows > vision::kMaxPatches || images < 1 || (images > 1 && !starts))
    throw std::invalid_argument("embedding vision attention: invalid patch count");
  if (rows >= 4096) run<64>(query, key, value, output, rows, starts, images);
  else run<32>(query, key, value, output, rows, starts, images);
}
}  // namespace gewell::embeddinggemma2
