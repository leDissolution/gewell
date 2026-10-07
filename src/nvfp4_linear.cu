#include "gewell/nvfp4_linear.h"
#include "bf16_activation.cuh"
#include "nvfp4_prefill.cuh"

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace gewell::nvfp4 {
namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("NVFP4 ") + operation + ": " +
                             cudaGetErrorString(status));
  }
}

void check_cublas(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("NVFP4 ") + operation +
                             ": cuBLAS status " + std::to_string(status));
  }
}

std::size_t align256(std::size_t value) { return (value + 255) & ~std::size_t(255); }

__device__ __forceinline__ float reciprocal_approximate(float value) {
  float result;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

__global__ void dequantize_weights(Weight weight, std::uint32_t width,
                                   std::size_t pairs, __nv_bfloat162* output) {
  const std::size_t pair = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= pairs) return;
  const auto row = pair / (width / 2);
  const auto block = (pair % (width / 2)) / 8;
  const auto offset = ((row / 128) * (width / 64) + block / 4) * 512 +
      (row % 32) * 16 + ((row % 128) / 32) * 4 + block % 4;
  const float scale = __fmul_rn(
      __half2float(__nv_cvt_fp8_to_halfraw(weight.scales[offset], __NV_E4M3)),
      weight.weight_scale);
  const float2 values = __half22float2(
      __nv_cvt_fp4x2_to_halfraw2(weight.data[pair], __NV_E2M1));
  // Match ModelOpt's packed reconstruction: materialize block*global first,
  // multiply FP4 values in FP32, then round once to BF16.
  output[pair] = __floats2bfloat162_rn(__fmul_rn(values.x, scale),
                                      __fmul_rn(values.y, scale));
}

// Eight lanes handle one K16 block, each loading and packing two BF16 values.
// Also write padding so scratch can be reused across recipes and graph replays
// without relying on initialization or scales left by an earlier invocation.
template<bool Gelu>
__global__ void quantize(const __nv_bfloat16* input, std::uint8_t* packed,
                         std::uint8_t* scales, std::uint32_t rows,
                         std::uint32_t padded_rows, std::uint32_t width,
                         float inverse_global_scale,
                         std::size_t scale_blocks) {
  const std::size_t pair = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t block = pair / 8;
  if (block >= scale_blocks) return;
  const std::uint32_t blocks_per_row = width / 16;
  const std::uint32_t row = block / blocks_per_row;
  const std::uint32_t column_block = block % blocks_per_row;
  const unsigned lane = threadIdx.x % 8;
  float2 value = make_float2(0.0F, 0.0F);
  if (row < rows) {
    const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(input);
    if constexpr (Gelu) {
      const auto source = std::size_t(row) * width + column_block * 8 + lane;
      const auto gate = __bfloat1622float2(pairs[source]);
      const auto up = __bfloat1622float2(pairs[source + width / 2]);
      value.x = gewell::detail::gelu_tanh_multiply_bf16(gate.x, up.x);
      value.y = gewell::detail::gelu_tanh_multiply_bf16(gate.y, up.y);
    } else {
      value = __bfloat1622float2(pairs[pair]);
    }
  }
  float amax = fmaxf(fabsf(value.x), fabsf(value.y));
  const unsigned mask = __activemask();
  amax = fmaxf(amax, __shfl_xor_sync(mask, amax, 4, 8));
  amax = fmaxf(amax, __shfl_xor_sync(mask, amax, 2, 8));
  amax = fmaxf(amax, __shfl_xor_sync(mask, amax, 1, 8));
  // Match NVIDIA/vLLM's operation order and approximate reciprocal recipe.
  // Reassociation or division can cross FP8/FP4 rounding boundaries with the
  // calibrated checkpoint globals, changing the quantized model's results.
  const std::uint8_t scale = __nv_cvt_float_to_fp8(
      inverse_global_scale * (amax * reciprocal_approximate(6.0F)),
      __NV_SATFINITE, __NV_E4M3);
  if (lane == 0) {
    const std::size_t offset =
        ((std::size_t(row) / 128) * (width / 64) + column_block / 4) * 512 +
        (row % 32) * 16 + ((row % 128) / 32) * 4 + column_block % 4;
    scales[offset] = scale;
  }
  if (row < padded_rows) {
    const float block_scale =
        __half2float(__nv_cvt_fp8_to_halfraw(scale, __NV_E4M3));
    const float factor =
        block_scale == 0.0F
            ? 0.0F
            : reciprocal_approximate(
                  block_scale * reciprocal_approximate(inverse_global_scale));
    value.x *= factor;
    value.y *= factor;
    packed[pair] = __nv_cvt_float2_to_fp4x2(value, __NV_E2M1, cudaRoundNearest);
  }
}

}  // namespace

void dequantize(Weight weight, std::uint32_t input_width,
                std::uint32_t output_width, __nv_bfloat16* output,
                cudaStream_t stream) {
  if (!weight.data || !weight.scales || !output ||
      !input_width || input_width % 64 || !output_width || output_width % 128 ||
      !(weight.weight_scale > 0.0F) || !std::isfinite(weight.weight_scale)) {
    throw std::invalid_argument("NVFP4 dequantize requires valid buffers, K divisible by64, N divisible by128, and a finite positive weight scale");
  }
  const std::size_t pairs = std::size_t(input_width) * output_width / 2;
  dequantize_weights<<<(pairs + 255) / 256, 256, 0, stream>>>(
      weight, input_width, pairs, reinterpret_cast<__nv_bfloat162*>(output));
  check_cuda(cudaGetLastError(), "dequantize weights");
}

struct Plan::Impl {
  cublasLtHandle_t handle{};
  cublasLtMatmulDesc_t operation{};
  cublasLtMatrixLayout_t weight_layout{}, input_layout{}, output_layout{};
  cublasLtMatmulAlgo_t algorithm{};
  void* query_scales{};
  std::uint32_t rows{}, padded_rows{}, input_width{}, output_width{};
  std::size_t scale_offset{}, output_offset{}, workspace_offset{};
  std::size_t workspace_bytes{}, total_bytes{};
  std::unique_ptr<detail::PrefillGemm> prefill;

  ~Impl() {
    if (output_layout) cublasLtMatrixLayoutDestroy(output_layout);
    if (input_layout) cublasLtMatrixLayoutDestroy(input_layout);
    if (weight_layout) cublasLtMatrixLayoutDestroy(weight_layout);
    if (operation) cublasLtMatmulDescDestroy(operation);
    if (query_scales) cudaFree(query_scales);
  }
};

Plan::Plan(cublasLtHandle_t handle, std::uint32_t rows,
           std::uint32_t input_width, std::uint32_t output_width)
    : impl_(std::make_unique<Impl>()) {
  if (!handle || rows == 0 ||
      rows > std::uint32_t(std::numeric_limits<int>::max() - 127) ||
      input_width == 0 || input_width % 64 ||
      output_width == 0 || output_width % 128) {
    throw std::invalid_argument(
        "NVFP4 plan requires positive rows, K divisible by 64, "
        "N divisible by 128, and a cuBLASLt handle");
  }
  auto& p = *impl_;
  p.handle = handle;
  p.rows = rows;
  p.padded_rows = (rows + 31) / 32 * 32;
  p.input_width = input_width;
  p.output_width = output_width;
  check_cublas(cublasLtMatmulDescCreate(&p.operation, CUBLAS_COMPUTE_32F,
                                       CUDA_R_32F),
               "create matmul descriptor");
  const cublasOperation_t transpose = CUBLAS_OP_T;
  check_cublas(cublasLtMatmulDescSetAttribute(
                   p.operation, CUBLASLT_MATMUL_DESC_TRANSA,
                   &transpose, sizeof(transpose)),
               "set weight transpose");
  const auto mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  check_cublas(cublasLtMatmulDescSetAttribute(
                   p.operation, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                   &mode, sizeof(mode)),
               "set weight block scale mode");
  check_cublas(cublasLtMatmulDescSetAttribute(
                   p.operation, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                   &mode, sizeof(mode)),
               "set activation block scale mode");
  // cuBLASLt requires non-null scale pointers even for a shape-only heuristic
  // query. No scale data is read by that query; run replaces both pointers.
  check_cuda(cudaMalloc(&p.query_scales, 16),
             "allocate heuristic scale placeholder");
  check_cublas(cublasLtMatmulDescSetAttribute(
                   p.operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                   &p.query_scales, sizeof(p.query_scales)),
               "set heuristic weight scale pointer");
  check_cublas(cublasLtMatmulDescSetAttribute(
                   p.operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                   &p.query_scales, sizeof(p.query_scales)),
               "set heuristic activation scale pointer");
  // Column-major (W^T)^T @ X^T produces row-major X @ W^T directly.
  check_cublas(cublasLtMatrixLayoutCreate(
                   &p.weight_layout, CUDA_R_4F_E2M1, input_width,
                   output_width, input_width),
               "create weight layout");
  check_cublas(cublasLtMatrixLayoutCreate(
                   &p.input_layout, CUDA_R_4F_E2M1, input_width,
                   p.padded_rows, input_width),
               "create activation layout");
  check_cublas(cublasLtMatrixLayoutCreate(
                   &p.output_layout, CUDA_R_16BF, output_width,
                   p.padded_rows, output_width),
               "create output layout");
  cublasLtMatmulPreference_t preference{};
  check_cublas(cublasLtMatmulPreferenceCreate(&preference), "create preference");
  constexpr std::size_t cublas_workspace_limit = 32 * 1024 * 1024;
  const auto preference_status = cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &cublas_workspace_limit, sizeof(cublas_workspace_limit));
  if (preference_status != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulPreferenceDestroy(preference);
    check_cublas(preference_status, "set workspace limit");
  }
  cublasLtMatmulHeuristicResult_t result{};
  int count = 0;
  const auto heuristic_status = cublasLtMatmulAlgoGetHeuristic(
      handle, p.operation, p.weight_layout, p.input_layout, p.output_layout,
      p.output_layout, preference, 1, &result, &count);
  cublasLtMatmulPreferenceDestroy(preference);
  check_cublas(heuristic_status, "select native FP4 algorithm");
  if (count != 1 || result.state != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        "NVFP4: no native FP4 cuBLASLt algorithm for rows=" +
        std::to_string(rows) + " K=" + std::to_string(input_width) +
        " N=" + std::to_string(output_width));
  }
  if (detail::PrefillGemm::selected(rows, input_width, output_width)) {
    p.prefill = std::make_unique<detail::PrefillGemm>(p.padded_rows, input_width, output_width);
    p.workspace_bytes = p.prefill->workspace_bytes();
    if (p.workspace_bytes > kMaxWorkspaceBytes)
      throw std::runtime_error("NVFP4 prefill GEMM exceeds the workspace bound");
  } else {
    p.algorithm = result.algo;
    p.workspace_bytes = result.workspaceSize;
  }
  p.scale_offset = align256(std::size_t(p.padded_rows) * input_width / 2);
  p.output_offset =
      align256(p.scale_offset + scale_storage_bytes(rows, input_width));
  const std::size_t padded_output_bytes =
      p.rows == p.padded_rows
          ? 0 : std::size_t(p.padded_rows) * output_width * sizeof(__nv_bfloat16);
  p.workspace_offset = align256(p.output_offset + padded_output_bytes);
  p.total_bytes = p.workspace_offset + p.workspace_bytes;
  // The placeholder is needed only for algorithm selection. run supplies
  // real block-scale pointers; keeping this allocation in every cached plan
  // would add device memory outside the caller's bounded scratch allocation.
  void* no_scales = nullptr;
  check_cublas(cublasLtMatmulDescSetAttribute(
                   p.operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                   &no_scales, sizeof(no_scales)), "clear heuristic weight scale pointer");
  check_cublas(cublasLtMatmulDescSetAttribute(
                   p.operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                   &no_scales, sizeof(no_scales)), "clear heuristic activation scale pointer");
  check_cuda(cudaFree(p.query_scales), "release heuristic scale placeholder");
  p.query_scales = nullptr;
}

Plan::~Plan() = default;
Plan::Plan(Plan&&) noexcept = default;
Plan& Plan::operator=(Plan&&) noexcept = default;

std::size_t Plan::scratch_bytes() const { return impl_->total_bytes; }

void Plan::run(const __nv_bfloat16* input, Weight weight, __nv_bfloat16* output,
                void* scratch, std::size_t scratch_capacity, cudaStream_t stream,
                InputTransform transform) {
  auto& p = *impl_;
  const float alpha = weight.input_scale * weight.weight_scale;
  if (!input || !weight.data || !weight.scales || !output || !scratch ||
      std::uintptr_t(scratch) % 256 || scratch_capacity < p.total_bytes ||
      !(weight.input_scale > 0.0F) || !std::isfinite(weight.input_scale) ||
      !(weight.weight_scale > 0.0F) || !std::isfinite(weight.weight_scale) ||
      !(alpha > 0.0F) || !std::isfinite(alpha) ||
      !std::isfinite(1.0F / weight.input_scale) ||
      (transform != InputTransform::identity && transform != InputTransform::gelu_tanh_multiply)) {
    throw std::invalid_argument(
        "NVFP4 run requires valid device buffers, aligned sufficient scratch, "
        "and finite positive global scales");
  }
  auto* packed = static_cast<std::uint8_t*>(scratch);
  auto* scales = packed + p.scale_offset;
  auto* destination =
      p.rows == p.padded_rows
          ? output : reinterpret_cast<__nv_bfloat16*>(packed + p.output_offset);
  const std::size_t scale_blocks = scale_storage_bytes(p.rows, p.input_width);
  const auto grid = (scale_blocks * 8 + 255) / 256;
  if (transform == InputTransform::gelu_tanh_multiply)
    quantize<true><<<grid, 256, 0, stream>>>(input, packed, scales, p.rows,
        p.padded_rows, p.input_width, 1.0F / weight.input_scale, scale_blocks);
  else
    quantize<false><<<grid, 256, 0, stream>>>(input, packed, scales, p.rows,
        p.padded_rows, p.input_width, 1.0F / weight.input_scale, scale_blocks);
  check_cuda(cudaGetLastError(), "quantize activations");
  if (p.prefill) {
    p.prefill->run(packed, weight.data, scales, weight.scales, alpha,
                  destination, packed + p.workspace_offset, stream);
  } else {
    check_cublas(cublasLtMatmulDescSetAttribute(
                   p.operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                   &weight.scales, sizeof(weight.scales)),
               "set weight block scales");
    check_cublas(cublasLtMatmulDescSetAttribute(
                   p.operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                   &scales, sizeof(scales)),
               "set activation block scales");
    constexpr float beta = 0.0F;
    check_cublas(cublasLtMatmul(
                   p.handle, p.operation, &alpha, weight.data, p.weight_layout,
                   packed, p.input_layout, &beta, destination, p.output_layout,
                   destination, p.output_layout, &p.algorithm,
                   packed + p.workspace_offset, p.workspace_bytes, stream),
               "native FP4 matmul");
  }
  if (destination != output) {
    check_cuda(cudaMemcpyAsync(
                   output, destination,
                   std::size_t(p.rows) * p.output_width * sizeof(__nv_bfloat16),
                   cudaMemcpyDeviceToDevice, stream),
               "copy unpadded output");
  }
}

}  // namespace gewell::nvfp4
