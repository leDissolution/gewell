#pragma once

#include "gewell/kv_format.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace gewell::kv_storage {
using BF16 = __nv_bfloat16;
using Format = kv_cache::Format;

template <typename T>
__device__ __forceinline__ T* row(T* base, std::size_t index,
    unsigned elements, Format format, unsigned scales = 1) {
  return base + index * kv_cache::row_words(elements, format, scales);
}

__device__ __forceinline__ BF16 load(const BF16* record, unsigned d,
    unsigned elements, Format format, unsigned split = 0) {
  if (format == Format::bf16) return record[d];
  const auto* bytes = reinterpret_cast<const unsigned char*>(record);
  const auto* scales = reinterpret_cast<const float*>(bytes + elements);
  const float scale = scales[split && d >= split ? 1 : 0];
  const auto half = __nv_cvt_fp8_to_halfraw(bytes[d], __NV_E4M3);
  return __float2bfloat16_rn(__half2float(static_cast<__half>(half)) * scale);
}

__device__ __forceinline__ uint4 unpack_eight(uint2 packed, float scale) {
  uint4 result;
  auto* pairs = reinterpret_cast<__nv_bfloat162*>(&result);
#pragma unroll
  for (unsigned i = 0; i < 4; ++i) {
    const unsigned word = i < 2 ? packed.x : packed.y;
    const auto half = __nv_cvt_fp8x2_to_halfraw2(
        static_cast<__nv_fp8x2_storage_t>(word >> ((i % 2) * 16)), __NV_E4M3);
    const auto values = __half22float2(static_cast<__half2>(half));
    // Keep the scalar reader's FP32 scaling and final BF16 rounding.
    pairs[i] = __float22bfloat162_rn(
        make_float2(values.x * scale, values.y * scale));
  }
  return result;
}

__device__ __forceinline__ uint4 load_eight(const BF16* record, unsigned d,
    unsigned elements, Format format, unsigned split = 0) {
  if (format == Format::bf16 &&
      !(reinterpret_cast<std::uintptr_t>(record + d) & 15U))
    return *reinterpret_cast<const uint4*>(record + d);
  const auto* bytes = reinterpret_cast<const unsigned char*>(record);
  if (format == Format::fp8 &&
      !(reinterpret_cast<std::uintptr_t>(bytes + d) & 7U) &&
      !(split && d < split && d + 8 > split)) {
    const auto packed = *reinterpret_cast<const uint2*>(bytes + d);
    const float scale = reinterpret_cast<const float*>(bytes + elements)
        [split && d >= split ? 1 : 0];
    return unpack_eight(packed, scale);
  }
  uint4 result;
  auto* values = reinterpret_cast<BF16*>(&result);
#pragma unroll
  for (unsigned i = 0; i < 8; ++i)
    values[i] = load(record, d + i, elements, format, split);
  return result;
}

// One 256-thread block owns one record. Each thread supplies consecutive
// 256-element stripes; no atomics or mutable page-wide scales are needed.
template <unsigned Elements, unsigned Split = Elements>
__device__ __forceinline__ void store_fp8(
    BF16* record, const float (&values)[(Elements + 255) / 256]) {
  static_assert(Elements % 16 == 0 && Split <= Elements);
  __shared__ float maxima[2][8];
  float a = 0.0F, b = 0.0F;
#pragma unroll
  for (unsigned i = 0; i < (Elements + 255) / 256; ++i) {
    const unsigned d = threadIdx.x + i * 256;
    if (d < Elements) {
      const float magnitude = fabsf(values[i]);
      if (d < Split) a = fmaxf(a, magnitude);
      else b = fmaxf(b, magnitude);
    }
  }
#pragma unroll
  for (unsigned delta = 16; delta; delta /= 2) {
    a = fmaxf(a, __shfl_xor_sync(0xffffffffU, a, delta));
    b = fmaxf(b, __shfl_xor_sync(0xffffffffU, b, delta));
  }
  if (!(threadIdx.x % 32)) {
    maxima[0][threadIdx.x / 32] = a;
    maxima[1][threadIdx.x / 32] = b;
  }
  __syncthreads();
  a = b = 0.0F;
#pragma unroll
  for (unsigned i = 0; i < 8; ++i) {
    a = fmaxf(a, maxima[0][i]);
    b = fmaxf(b, maxima[1][i]);
  }
  a = a > 0 && isfinite(a) ? a / 448.0F : 1.0F;
  b = b > 0 && isfinite(b) ? b / 448.0F : 1.0F;
  auto* bytes = reinterpret_cast<unsigned char*>(record);
  auto* scales = reinterpret_cast<float*>(bytes + Elements);
  if (threadIdx.x == 0) {
    scales[0] = a;
    if constexpr (Split < Elements) scales[1] = b;
  }
#pragma unroll
  for (unsigned i = 0; i < (Elements + 255) / 256; ++i) {
    const unsigned d = threadIdx.x + i * 256;
    if (d < Elements)
      bytes[d] = __nv_cvt_float_to_fp8(values[i] / (d < Split ? a : b),
                                      __NV_SATFINITE, __NV_E4M3);
  }
  constexpr auto ScaleCount = Split < Elements ? 2 : 1;
  constexpr auto Used = Elements + ScaleCount * sizeof(float);
  constexpr auto Bytes = kv_cache::row_bytes(Elements, Format::fp8, ScaleCount);
  if (threadIdx.x < Bytes - Used) bytes[Used + threadIdx.x] = 0;
  // Consecutive calls may reuse the shared reduction storage.
  __syncthreads();
}

template <unsigned D>
__device__ __forceinline__ void store_separate(
    BF16* key_record, BF16* value_record, const BF16* key, const BF16* value) {
  float k[(D + 255) / 256], v[(D + 255) / 256];
#pragma unroll
  for (unsigned i = 0; i < (D + 255) / 256; ++i) {
    const unsigned d = threadIdx.x + 256 * i;
    k[i] = __bfloat162float(key[d]);
    v[i] = __bfloat162float(value[d]);
  }
  store_fp8<D>(key_record, k);
  store_fp8<D>(value_record, v);
}

__device__ __forceinline__ void store_compact(
    BF16* record, const BF16* key, const BF16* value) {
  float values[3]{};
#pragma unroll
  for (unsigned i = 0; i < 3; ++i) {
    const unsigned d = threadIdx.x + 256 * i;
    if (d < 128) values[i] = __bfloat162float(key[d < 64 ? d : d + 192]);
    else if (d < 640) values[i] = __bfloat162float(value[d - 128]);
  }
  store_fp8<640, 128>(record, values);
}
}  // namespace gewell::kv_storage
