#include "kv_storage.cuh"
#include "../kernels/fp8_cache.cuh"
#include "fp8_quantize.cuh"
#include "gewell/mtp_attention.h"

#include "gewell/compact_global_cache.h"
#include <math_constants.h>
#include <mma.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace gewell::mtp_attention {
namespace {
using BF16 = mtp_target::BFloat16;
using Cache = kv_cache::DeviceView;
constexpr unsigned kThreads = 256;
constexpr unsigned kWarp = 32;
constexpr unsigned kHeads = gemma4_31b::kQueryHeadCount;
constexpr std::uint32_t kMaxContext = 262144;

std::uint32_t splits_for(std::uint32_t visible) {
  return std::min(kMaxSplits, (visible + kKeysPerTile - 1) / kKeysPerTile);
}

unsigned default_splits(unsigned base, unsigned rows, bool global, bool frozen = false) {
  auto splits = splits_for(global ? base + (frozen ? 0 : rows)
                                 : std::min(1024U, base + rows));
  if (global && base + rows < 16384)
    splits = std::min(splits, 128U / std::min(rows, kQueryTileRows));
  return splits;
}

std::size_t bytes_for(std::uint32_t rows, std::uint32_t splits,
                      std::uint32_t dimension, unsigned heads = kHeads) {
  return std::size_t(std::min(rows, kQueryTileRows)) * heads * splits *
         (dimension + 2) * sizeof(float);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (unsigned offset = 16; offset; offset /= 2)
    value += __shfl_down_sync(0xffffffffU, value, offset);
  return value;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (unsigned offset = 16; offset; offset /= 2)
    value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
  return value;
}

__device__ __forceinline__ __nv_bfloat162 load_pair(const BF16* source) {
  // Page and layer offsets are expressed in BF16 elements, so a legal view
  // need not preserve the alignment required by a packed 32-bit load.
  if ((reinterpret_cast<std::uintptr_t>(source) & 3U) == 0)
    return *reinterpret_cast<const __nv_bfloat162*>(source);
  return __halves2bfloat162(source[0], source[1]);
}

__device__ __forceinline__ uint4 load_eight(const BF16* source) {
  if ((reinterpret_cast<std::uintptr_t>(source) & 15U) == 0)
    return *reinterpret_cast<const uint4*>(source);
  uint4 result;
  auto* words = reinterpret_cast<unsigned*>(&result);
#pragma unroll
  for (unsigned i = 0; i < 4; ++i)
    words[i] = unsigned(__bfloat16_as_ushort(source[2 * i])) |
               (unsigned(__bfloat16_as_ushort(source[2 * i + 1])) << 16);
  return result;
}

__device__ __forceinline__ void prefetch_l2(const void* address) {
  const auto global = __cvta_generic_to_global(address);
  asm volatile("prefetch.global.L2 [%0];" : : "l"(global) : "memory");
}



// Eight warps compute disjoint 32-column output tiles for eight query rows.
// Probabilities round once to V's BF16 dtype, matching vLLM's P*V boundary.
// WMMA accumulates the products in FP32. The shared output tile changes
// ownership back to one query per warp.
__device__ void tensor_pv(const BF16* probabilities, const BF16* values,
                         unsigned stride, unsigned width,
                         float* output_tile, float* numerator) {
  const unsigned warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
  for (unsigned column = 0; column < width; column += 256) {
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, float> sum;
    nvcuda::wmma::fill_fragment(sum, 0.0F);
#pragma unroll
    for (unsigned key = 0; key < 32; key += 16) {
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, BF16, nvcuda::wmma::row_major> p;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, BF16, nvcuda::wmma::row_major> v;
      nvcuda::wmma::load_matrix_sync(p, probabilities + key, 32);
      nvcuda::wmma::load_matrix_sync(v, values + key * stride + column + warp * 32, stride);
      nvcuda::wmma::mma_sync(sum, p, v, sum);
    }
    nvcuda::wmma::store_matrix_sync(output_tile + warp * 32, sum, 264, nvcuda::wmma::mem_row_major);
    __syncthreads();
#pragma unroll
    for (unsigned d = 0; d < 8; ++d)
      numerator[column / 32 + d] += output_tile[warp * 264 + lane + d * 32];
    __syncthreads();
  }
}

// Every CTA shares one KV tile among all query heads in its GQA group. A
// split normally processes 32 keys. Long contexts assign it more tiles with a
// fixed stride, keeping the partial-result allocation bounded.
template <bool Paged>
__device__ __forceinline__ const BF16* compact_record(
    const kv_cache::DeviceView& cache, unsigned head, unsigned position) {
  if constexpr (Paged)
    return compact_global_cache::paged_row(cache.page_pool, cache.page_offsets,
        256, cache.layer_offset_elements, head, position, cache.format);
  return kv_storage::row(cache.key, std::size_t(head) * cache.capacity + position,
                          640, cache.format, 2);
}

constexpr unsigned kCompactTileBytes =
    kKeysPerTile * kv_cache::row_bytes(640, kv_cache::Format::fp8, 2);

// Only complete staged FP8 tiles use this reader. Their rows and eight-byte
// vectors are aligned, and no vector straddles the rotated-K / V scale split.
__device__ __forceinline__ uint4 load_compact_shared_eight(
    const BF16* raw, unsigned token, unsigned d) {
  const unsigned record = static_cast<unsigned>(__cvta_generic_to_shared(raw)) +
      token * kv_cache::row_bytes(640, kv_cache::Format::fp8, 2);
  uint2 packed;
  float scale;
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];"
      : "=r"(packed.x), "=r"(packed.y) : "r"(record + d) : "memory");
  asm volatile("ld.shared.f32 %0, [%1];"
      : "=f"(scale) : "r"(record + 640 + (d >= 128 ? 4 : 0)) : "memory");
  return kv_storage::unpack_eight(packed, scale);
}

// A complete 32-key tile stays inside one 256-token page. Stage the packed
// bytes (including their scales) while the previous tile's tensor work runs.
// Tails, staged rows and unaligned views retain the original direct reader.
template <bool Paged>
__device__ __forceinline__ bool stage_compact_tile(
    BF16* destination, const Cache& cache, unsigned head,
    unsigned begin, unsigned base) {
  if (cache.format != kv_cache::Format::fp8 || begin + kKeysPerTile > base)
    return false;
  const auto* source = compact_record<Paged>(cache, head, begin);
  if (reinterpret_cast<std::uintptr_t>(source) & 15U) return false;
  for (unsigned offset = threadIdx.x * 16; offset < kCompactTileBytes;
       offset += kThreads * 16) {
    const auto shared = static_cast<unsigned>(__cvta_generic_to_shared(
        reinterpret_cast<unsigned char*>(destination) + offset));
    const auto global = __cvta_generic_to_global(
        reinterpret_cast<const unsigned char*>(source) + offset);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
        : : "r"(shared), "l"(global) : "memory");
  }
  asm volatile("cp.async.commit_group;" : : : "memory");
  return true;
}

template <bool Global, bool Paged, bool Frozen = false, bool Buffered = false, unsigned Heads = kHeads>
__device__ __forceinline__ void partial_attention_body(
    const BF16* query, const BF16* staged_key, const BF16* staged_value,
    Cache cache, const BF16* k_norm, std::uint32_t base,
    std::uint32_t rows, std::uint32_t first_row, std::uint32_t splits,
    float* partial, float* maxima, float* denominators, unsigned grid_x, unsigned grid_y, unsigned grid_z) {
  constexpr unsigned Threads = kThreads;
  constexpr unsigned D = Global ? 512 : 256;
  constexpr unsigned KvHeads = Global ? Heads / 8 : Heads / 2;
  constexpr unsigned QueriesPerKv = Heads / KvHeads;
  constexpr unsigned WarpsPerQuery = (kThreads / kWarp) / QueriesPerKv;
  // Reconstruct each global BF16 key once, then reuse the tile for V. The
  // eight-element padding avoids WMMA bank conflicts without changing the
  // persistent compact layout. Unbuffered calls allow two resident CTAs;
  // buffered FP8 drafting uses one CTA and keeps its invariant inputs resident.
  constexpr unsigned TileWidth = Global ? D + 8 : D;
  constexpr unsigned ValuesPerThread = D / (WarpsPerQuery * kWarp);
  static_assert(!Buffered || (Global && Frozen));
  extern __shared__ __align__(16) unsigned char raw_workspace[];
  auto* raw = reinterpret_cast<BF16*>(raw_workspace);
  __shared__ __align__(32) BF16 tile[kKeysPerTile * TileWidth];
  __shared__ float scores[QueriesPerKv * kKeysPerTile];
  __shared__ __align__(32) float qk_scratch[Global ? 8 * 264 : 1];
  __shared__ __align__(32) BF16
      pv_probabilities[QueriesPerKv * kKeysPerTile];
  __shared__ float next_maximum[QueriesPerKv];
  __shared__ float added_denominator[QueriesPerKv];

  const unsigned kv_head = grid_x;
  const unsigned tile_row = grid_y;
  const unsigned row = first_row + tile_row;
  const unsigned split = grid_z;
  const unsigned warp = threadIdx.x / kWarp;
  const unsigned lane = threadIdx.x % kWarp;
  const unsigned query_in_group = warp / WarpsPerQuery;
  const unsigned score_group = query_in_group;
  const unsigned warp_in_query = warp % WarpsPerQuery;
  const unsigned query_head = kv_head * QueriesPerKv + query_in_group;
  const std::size_t query_offset = (std::size_t(query_head) * rows + row) * D;
  const unsigned last = Frozen ? base - 1 : base + row;
  const unsigned first = !Global && last >= 1023 ? last - 1023 : 0;

  // Q and k_norm do not change across KV tiles. The buffered frozen kernel's
  // shared memory already limits it to one CTA, leaving room for these registers.
  using GlobalQuery = nvcuda::wmma::fragment<
      nvcuda::wmma::matrix_a, 8, 32, 16, BF16, nvcuda::wmma::row_major>;
  GlobalQuery frozen_query[Buffered ? 8 : 1];
  uint4 frozen_norm{};
  if constexpr (Buffered) {
    const unsigned d = (threadIdx.x * 8) % D;
    if (!(d < 64 || (d >= 256 && d < 320)))
      frozen_norm = load_eight(k_norm + d);
    if (warp < 4) {
#pragma unroll
      for (unsigned k = 0; k < 8; ++k) {
        const auto* source = query +
            (std::size_t(kv_head * 8) * rows + row) * D + warp * 128 + k * 16;
        if ((reinterpret_cast<std::uintptr_t>(query) & 31U) == 0) {
          nvcuda::wmma::load_matrix_sync(frozen_query[k], source, rows * D);
        } else {
          auto* aligned_query = reinterpret_cast<BF16*>(qk_scratch) + warp * 128;
          for (unsigned i = lane; i < 128; i += kWarp)
            aligned_query[i] = source[(i / 16) * rows * D + i % 16];
          __syncwarp();
          nvcuda::wmma::load_matrix_sync(frozen_query[k], aligned_query, 16);
          __syncwarp();
        }
      }
    }
  }

  float query_values[D / kWarp];
#pragma unroll
  for (unsigned i = 0; i < D / kWarp; ++i) {
    if constexpr (!Global)
      query_values[i] =
          __bfloat162float(query[query_offset + lane + i * kWarp]);
  }
  float numerator[ValuesPerThread];
#pragma unroll
  for (unsigned i = 0; i < ValuesPerThread; ++i) numerator[i] = 0.0F;
  float maximum = -CUDART_INF_F;
  float denominator = 0.0F;

  bool ready = false;
  if constexpr (Buffered)
    ready = stage_compact_tile<Paged>(raw, cache, kv_head,
                                      split * kKeysPerTile, base);

  for (unsigned begin = first + split * kKeysPerTile; begin <= last;
       begin += splits * kKeysPerTile) {
    if (ready) {
      asm volatile("cp.async.wait_group 0;" : : : "memory");
      __syncthreads();
    }
    const unsigned load_count = min(kKeysPerTile, last - begin + 1);
    const unsigned count = load_count;
    if constexpr (Global) {
      // Stage eight BF16 elements per copy, including frozen assistant KV.
      // Each vector stays wholly inside or outside the rotated dimensions.
      for (unsigned i = threadIdx.x * 8; i < load_count * D; i += Threads * 8) {
        const unsigned position = begin + i / D;
        const unsigned d = i % D;
        const bool rotated = d < 64 || (d >= 256 && d < 320);
        uint4 value;
        if (position >= base) {
          const auto* source = rotated
              ? staged_key + (std::size_t(kv_head) * rows + position - base) * D + d
              : staged_value + (std::size_t(position - base) * KvHeads + kv_head) * D + d;
          value = load_eight(source);
        } else {
          const auto* record = ready
              ? raw + (position - begin) * kv_cache::row_words(640, cache.format, 2)
              : compact_record<Paged>(cache, kv_head, position);
          const unsigned column = rotated ? (d < 64 ? d : d - 192) : 128 + d;
          value = ready ? load_compact_shared_eight(raw, position - begin, column)
                        : kv_storage::load_eight(record, column, 640, cache.format, 128);
        }
        if (!rotated) {
          const auto n = Buffered ? frozen_norm : load_eight(k_norm + d);
          auto* v = reinterpret_cast<__nv_bfloat162*>(&value);
          const auto* scale = reinterpret_cast<const __nv_bfloat162*>(&n);
#pragma unroll
          for (unsigned p = 0; p < 4; ++p) v[p] = __hmul2(v[p], scale[p]);
        }
        *reinterpret_cast<uint4*>(tile + (i / D) * TileWidth + i % D) = value;
      }
    } else {
      for (unsigned i = threadIdx.x * 8; i < load_count * D; i += Threads * 8) {
        const unsigned position = begin + i / D;
        const unsigned d = i % D;
        *reinterpret_cast<uint4*>(tile + i) = position >= base
                      ? load_eight(staged_key + (std::size_t(kv_head) * rows + position - base) * D + d)
                      : kv_storage::load_eight(kv_storage::row(cache.key, std::size_t(kv_head) * 1024 + position % 1024, D, cache.format), d, D, cache.format);
      }
    }
    if constexpr (Global) {
      // WMMA loads a complete tile; absent keys never read future staged rows.
      for (unsigned i = threadIdx.x; i < (kKeysPerTile - load_count) * D;
           i += Threads)
        tile[(load_count + i / D) * TileWidth + i % D] = __float2bfloat16_rn(0.0F);
    }
    __syncthreads();

    if constexpr (Global) {
      using Accumulator = nvcuda::wmma::fragment<
          nvcuda::wmma::accumulator, 8, 32, 16, float>;
      using Query = GlobalQuery;
      using Key = nvcuda::wmma::fragment<
          nvcuda::wmma::matrix_b, 8, 32, 16, BF16, nvcuda::wmma::col_major>;
      Accumulator accumulator;
      if (warp < 4) {
        nvcuda::wmma::fill_fragment(accumulator, 0.0F);
#pragma unroll
        for (unsigned tile_index = 0; tile_index < 8; ++tile_index) {
          const unsigned d = warp * 128 + tile_index * 16;
          Query q;
          Key k;
          const auto* query_tile =
              query + (std::size_t(kv_head * 8) * rows + row) * D + d;
          if constexpr (Buffered) {
            q = frozen_query[tile_index];
          } else if ((reinterpret_cast<std::uintptr_t>(query) & 31U) == 0) {
            nvcuda::wmma::load_matrix_sync(q, query_tile, rows * D);
          } else {
            // The target supplies aligned queries. Preserve the public
            // primitive's BF16 alignment contract for other callers too.
            auto* aligned_query = reinterpret_cast<BF16*>(qk_scratch) + warp * 128;
            for (unsigned i = lane; i < 128; i += kWarp)
              aligned_query[i] = query_tile[(i / 16) * rows * D + i % 16];
            __syncwarp();
            nvcuda::wmma::load_matrix_sync(q, aligned_query, 16);
          }
          nvcuda::wmma::load_matrix_sync(k, tile + d, TileWidth);
          nvcuda::wmma::mma_sync(accumulator, q, k, accumulator);
          __syncwarp();
        }
      }
      __syncthreads();
      if (warp < 4)
        nvcuda::wmma::store_matrix_sync(
            qk_scratch + warp * 256, accumulator, 32,
            nvcuda::wmma::mem_row_major);
      __syncthreads();
      const unsigned score_index = threadIdx.x;
      scores[threadIdx.x] = qk_scratch[score_index] +
                           qk_scratch[256 + score_index] +
                           qk_scratch[512 + score_index] +
                           qk_scratch[768 + score_index];
    } else {
      for (unsigned token = warp_in_query; token < count; token += WarpsPerQuery) {
        float score = 0.0F;
#pragma unroll
        for (unsigned i = 0; i < D / kWarp; ++i) {
          const unsigned dimension = lane + i * kWarp;
          const BF16 key = tile[token * TileWidth + dimension];
          score = fmaf(query_values[i], __bfloat162float(key), score);
        }
        score = warp_sum(score);
        if (lane == 0) scores[score_group * kKeysPerTile + token] = score;
      }
    }
    __syncthreads();

    if (warp_in_query == 0) {
      const unsigned probability_index =
          score_group * kKeysPerTile + lane;
      float tile_max = lane < count
                           ? scores[probability_index]
                           : -CUDART_INF_F;
      tile_max = warp_max(tile_max);
      tile_max = fmaxf(maximum, __shfl_sync(0xffffffffU, tile_max, 0));
      float weight = 0.0F;
      if (lane < count) {
        weight = expf(scores[probability_index] - tile_max);
        scores[probability_index] = weight;
      }
      pv_probabilities[probability_index] = __float2bfloat16_rn(weight);
      const float sum = warp_sum(weight);
      if (lane == 0) {
        next_maximum[score_group] = tile_max;
        added_denominator[score_group] = sum;
      }
    }
    __syncthreads();

    if constexpr (Global) {
      for (unsigned i = threadIdx.x * 8; i < load_count * D; i += Threads * 8) {
        const unsigned position = begin + i / D;
        const unsigned d = i % D;
        const auto* record = position >= base ? nullptr : ready
            ? raw + (position - begin) * kv_cache::row_words(640, cache.format, 2)
            : compact_record<Paged>(cache, kv_head, position);
        const auto value = position >= base
            ? load_eight(staged_value + (std::size_t(position - base) * KvHeads + kv_head) * D + d)
            : ready ? load_compact_shared_eight(raw, position - begin, 128 + d)
                    : kv_storage::load_eight(record, 128 + d, 640, cache.format, 128);
        *reinterpret_cast<uint4*>(tile + (i / D) * TileWidth + i % D) = value;
      }
    } else {
      for (unsigned i = threadIdx.x * 8; i < load_count * D; i += Threads * 8) {
        const unsigned position = begin + i / D;
        const unsigned d = i % D;
        *reinterpret_cast<uint4*>(tile + i) = position >= base
                      ? load_eight(staged_value + (std::size_t(position - base) * KvHeads + kv_head) * D + d)
                      : kv_storage::load_eight(kv_storage::row(cache.value, std::size_t(kv_head) * 1024 + position % 1024, D, cache.format), d, D, cache.format);
      }
    }
    __syncthreads();
    // K and V have both consumed the packed tile. Only now may the next
    // asynchronous copy overwrite it; the existing arithmetic stays intact.
    if constexpr (Buffered)
      ready = stage_compact_tile<Paged>(raw, cache, kv_head,
          begin + splits * kKeysPerTile, base);
    const float new_maximum = next_maximum[score_group];
    const float old_scale = expf(maximum - new_maximum);
    denominator = fmaf(denominator, old_scale,
                       added_denominator[score_group]);
    maximum = new_maximum;
    if constexpr (Global) {
#pragma unroll
      for (unsigned i = 0; i < ValuesPerThread; ++i) numerator[i] *= old_scale;
      bool nonfinite = false;
      for (unsigned i = threadIdx.x; i < count * D; i += kThreads)
        nonfinite |= (__bfloat16_as_ushort(tile[(i / D) * TileWidth + i % D]) & 0x7f80U) == 0x7f80U;
      const bool scalar_pv = __syncthreads_or(nonfinite);
      if (!scalar_pv)
        tensor_pv(pv_probabilities, tile, TileWidth, D, qk_scratch, numerator);
      if (scalar_pv) {
        for (unsigned token = 0; token < count; ++token) {
          const float probability = __bfloat162float(
              pv_probabilities[score_group * kKeysPerTile + token]);
#pragma unroll
          for (unsigned i = 0; i < ValuesPerThread; ++i) {
            const unsigned d = lane + i * kWarp;
            numerator[i] = fmaf(probability,
                                __bfloat162float(tile[token * TileWidth + d]),
                                numerator[i]);
          }
        }
      }
    } else {
#pragma unroll
      for (unsigned i = 0; i < ValuesPerThread; ++i) {
        const unsigned dimension = warp_in_query * kWarp + lane +
                                   i * WarpsPerQuery * kWarp;
        float sum = numerator[i] * old_scale;
        for (unsigned token = 0; token < count; ++token) {
          sum = fmaf(__bfloat162float(
                         pv_probabilities[score_group * kKeysPerTile + token]),
                     __bfloat162float(tile[token * TileWidth + dimension]), sum);
        }
        numerator[i] = sum;
      }
    }
    __syncthreads();
  }

  const std::size_t result = (std::size_t(tile_row) * Heads + query_head) *
                            splits + split;
#pragma unroll
  for (unsigned i = 0; i < ValuesPerThread; ++i) {
    const unsigned dimension = warp_in_query * kWarp + lane +
                               i * WarpsPerQuery * kWarp;
    partial[result * D + dimension] = numerator[i];
  }
  if (warp_in_query == 0 && lane == 0) {
    maxima[result] = maximum;
    denominators[result] = denominator;
  }
}

template <bool Global, bool Paged, bool Frozen = false, unsigned Heads = kHeads>
__global__ __launch_bounds__(kThreads, 2) void partial_attention(
    const BF16* query, const BF16* staged_key, const BF16* staged_value,
    Cache cache, const BF16* k_norm, std::uint32_t base,
    std::uint32_t rows, std::uint32_t first_row, std::uint32_t splits,
    float* partial, float* maxima, float* denominators) {
  partial_attention_body<Global, Paged, Frozen, false, Heads>(query, staged_key, staged_value, cache, k_norm, base, rows, first_row, splits, partial, maxima, denominators,
      blockIdx.x, blockIdx.y, blockIdx.z);
}

// Keep the original V alongside reconstructed K, and share both across four
// causal rows and all eight GQA heads. Query fragments and the output stay in
// MMA registers. Once QK is complete, the K tile becomes the probability
// workspace. No expanded K is written to global memory.
constexpr unsigned kGlobalSharedBytes =
    2 * kKeysPerTile * 520 * sizeof(BF16) + 2 * 8 * 16 * sizeof(float);
constexpr unsigned kGlobalFp8SharedBytes = kGlobalSharedBytes + kCompactTileBytes;

__device__ __forceinline__ unsigned pair_bits(const BF16* address) {
  const auto pair = load_pair(address);
  return reinterpret_cast<const unsigned&>(pair);
}

__device__ __forceinline__ void mma_16x8(float (&c)[4],
    const unsigned (&a)[4], const unsigned (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void matrix_load_a(unsigned (&a)[4], const BF16* address) {
  const unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(address));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(shared));
}

template<bool Transpose>
__device__ __forceinline__ void matrix_load_b(unsigned (&b)[2], const BF16* address) {
  const unsigned shared = static_cast<unsigned>(__cvta_generic_to_shared(address));
  if constexpr (Transpose)
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
        : "=r"(b[0]), "=r"(b[1]) : "r"(shared));
  else
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
        : "=r"(b[0]), "=r"(b[1]) : "r"(shared));
}

template <bool Paged>
__device__ __forceinline__ void global_four_rows_body(
    const BF16* query, const BF16* staged_key, const BF16* staged_value,
    Cache cache, const BF16* k_norm, std::uint32_t base, std::uint32_t rows,
    std::uint32_t first_row, std::uint32_t splits,
    float* partial, float* maxima, float* denominators,
    unsigned grid_x, unsigned grid_y, unsigned grid_z) {
  constexpr unsigned D = 512, Stride = 520;
  extern __shared__ __align__(32) unsigned char workspace[];
  auto* keys = reinterpret_cast<BF16*>(workspace);
  auto* values = keys + kKeysPerTile * Stride;
  auto* warp_maxima = reinterpret_cast<float*>(values + kKeysPerTile * Stride);
  auto* warp_sums = warp_maxima + 8 * 16;
  auto* raw = reinterpret_cast<BF16*>(warp_sums + 8 * 16);
  auto* probabilities = keys;
  const unsigned head = grid_x, warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
  const unsigned pair = warp / 4, column_warp = warp % 4;
  const unsigned query_head = head * 8 + lane / 4;
  const unsigned group_first = first_row + grid_y * 4;
  const unsigned row0 = group_first + pair * 2;
  const unsigned last = base + min(rows - 1, group_first + 3);
  const unsigned thread_dimension = (threadIdx.x * 8) % D;
  const bool thread_rotated = thread_dimension < 64 ||
                              (thread_dimension >= 256 && thread_dimension < 320);
  const uint4 norm = thread_rotated ? uint4{} : load_eight(k_norm + thread_dimension);
  // Each four-warp group computes a 16-query matrix (two rows, eight GQA
  // heads). Keep Q and the 16x512 output in their MMA register layout.
  unsigned q[32][4];
#pragma unroll
  for (unsigned k = 0; k < 32; ++k) {
    const unsigned d = k * 16 + (lane % 4) * 2;
    const auto* q0 = query + (std::size_t(query_head) * rows + min(row0, rows - 1)) * D;
    const auto* q1 = query + (std::size_t(query_head) * rows + min(row0 + 1, rows - 1)) * D;
    q[k][0] = pair_bits(q0 + d);
    q[k][1] = pair_bits(q1 + d);
    q[k][2] = pair_bits(q0 + d + 8);
    q[k][3] = pair_bits(q1 + d + 8);
  }
  float numerator[16][4]{};
  float maximum[2]{-CUDART_INF_F, -CUDART_INF_F}, denominator[2]{};

  bool ready = stage_compact_tile<Paged>(raw, cache, head,
                                        grid_z * kKeysPerTile, base);

  for (unsigned begin = grid_z * kKeysPerTile; begin <= last;
       begin += splits * kKeysPerTile) {
    if (ready) {
      asm volatile("cp.async.wait_group 0;" : : : "memory");
      __syncthreads();
    }
    const unsigned count = min(kKeysPerTile, last - begin + 1);
    const BF16* committed = ready ? raw :
        begin < base ? compact_record<Paged>(cache, head, begin) : nullptr;
    bool nonfinite = false;
    // Stage eight elements per copy, amortizing page/row addressing. Each
    // vector lies entirely inside or outside the rotated K dimensions.
    if (ready) {
      // Complete committed tiles need no per-vector tail, format, alignment
      // or staged-row checks. Dimension and normalization are loop-invariant.
      const unsigned d = thread_dimension;
      for (unsigned token = threadIdx.x / 64; token < kKeysPerTile; token += 4) {
        const auto v = load_compact_shared_eight(raw, token, 128 + d);
        uint4 k;
        if (thread_rotated)
          k = load_compact_shared_eight(raw, token, d < 64 ? d : d - 192);
        else {
          auto* kp = reinterpret_cast<__nv_bfloat162*>(&k);
          const auto* vp = reinterpret_cast<const __nv_bfloat162*>(&v);
          const auto* np = reinterpret_cast<const __nv_bfloat162*>(&norm);
#pragma unroll
          for (unsigned pair = 0; pair < 4; ++pair) kp[pair] = __hmul2(vp[pair], np[pair]);
        }
        *reinterpret_cast<uint4*>(keys + token * Stride + d) = k;
        *reinterpret_cast<uint4*>(values + token * Stride + d) = v;
      }
    } else for (unsigned i = threadIdx.x * 8; i < kKeysPerTile * D; i += kThreads * 8) {
      const unsigned token = i / D, position = begin + token, d = i % D;
      uint4 v{}, k{};
      if (token < count) {
        const bool rotated = d < 64 || (d >= 256 && d < 320);
        const BF16* source;
        const BF16* rotated_key = nullptr;
        if (position >= base) {
          source = staged_value + (std::size_t(position - base) * 4 + head) * D + d;
          if (rotated)
            rotated_key = staged_key + (std::size_t(head) * rows + position - base) * D + d;
        } else {
          const BF16* compact = committed + std::size_t(token) * kv_cache::row_words(640, cache.format, 2);
          source = compact + 128 + d;
          if (rotated) rotated_key = compact + (d < 64 ? d : d - 192);
        }
        const auto* record = position < base ? committed + std::size_t(token) *
            kv_cache::row_words(640, cache.format, 2) : nullptr;
        v = position < base ? kv_storage::load_eight(record, 128 + d, 640, cache.format, 128)
                            : load_eight(source);
        if (rotated) k = position < base
            ? kv_storage::load_eight(record, d < 64 ? d : d - 192, 640, cache.format, 128)
            : load_eight(rotated_key);
        else {
          auto* kp = reinterpret_cast<__nv_bfloat162*>(&k);
          const auto* vp = reinterpret_cast<const __nv_bfloat162*>(&v);
          const auto* np = reinterpret_cast<const __nv_bfloat162*>(&norm);
#pragma unroll
          for (unsigned pair = 0; pair < 4; ++pair) kp[pair] = __hmul2(vp[pair], np[pair]);
        }
      }
      *reinterpret_cast<uint4*>(keys + token * Stride + d) = k;
      // A future nonfinite value must not contaminate a masked tensor P*V.
      // Restore its contribution only for the rows that actually see it.
      auto* words = reinterpret_cast<unsigned*>(&v);
      if (position >= base) {
#pragma unroll
        for (unsigned pair = 0; pair < 4; ++pair) {
          const bool bad_lo = (words[pair] & 0x7f80U) == 0x7f80U;
          const bool bad_hi = (words[pair] & 0x7f800000U) == 0x7f800000U;
          nonfinite |= bad_lo || bad_hi;
          if (bad_lo) words[pair] &= 0xffff0000U;
          if (bad_hi) words[pair] &= 0x0000ffffU;
        }
      }
      *reinterpret_cast<uint4*>(values + token * Stride + d) = v;
    }
    bool has_nonfinite = false;
    if (begin + count > base)
      has_nonfinite = __syncthreads_or(nonfinite);
    else
      __syncthreads();

    // The shared K/V matrices no longer depend on raw. Overlap the next
    // packed copy with QK/PV, retaining L2 prefetch for the direct reader.
    const unsigned prefetch_begin = begin + splits * kKeysPerTile;
    ready = stage_compact_tile<Paged>(raw, cache, head, prefetch_begin, base);
    if (!ready && prefetch_begin < base && prefetch_begin <= last) {
      const auto* prefetch_row = compact_record<Paged>(cache, head, prefetch_begin);
      constexpr unsigned LineBytes = 128;
      const unsigned prefetch_bytes =
          min(kKeysPerTile, base - prefetch_begin) *
          kv_cache::row_bytes(640, cache.format, 2);
#pragma unroll 1
      for (unsigned offset = threadIdx.x * LineBytes; offset < prefetch_bytes;
           offset += kThreads * LineBytes)
        prefetch_l2(
            reinterpret_cast<const unsigned char*>(prefetch_row) + offset);
    }

    float score[4]{};
#pragma unroll
    for (unsigned k = 0; k < 32; ++k) {
      unsigned key[2];
      matrix_load_b<false>(key, keys + (column_warp * 8 + lane % 8) * Stride +
          k * 16 + ((lane / 8) % 2) * 8);
      mma_16x8(score, q[k], key);
    }
#pragma unroll
    for (unsigned r = 0; r < 2; ++r) {
      const unsigned row = row0 + r;
#pragma unroll
      for (unsigned c = 0; c < 2; ++c) {
        const unsigned token = column_warp * 8 + (lane % 4) * 2 + c;
        if (row >= rows || token >= count || begin + token > base + row)
          score[r * 2 + c] = -CUDART_INF_F;
      }
      float local = fmaxf(score[r * 2], score[r * 2 + 1]);
      local = fmaxf(local, __shfl_xor_sync(0xffffffffU, local, 1));
      local = fmaxf(local, __shfl_xor_sync(0xffffffffU, local, 2));
      if (lane % 4 == 0) warp_maxima[warp * 16 + lane / 4 + r * 8] = local;
    }
    __syncthreads();
    float alpha[2];
#pragma unroll
    for (unsigned r = 0; r < 2; ++r) {
      float next = maximum[r];
#pragma unroll
      for (unsigned w = 0; w < 4; ++w)
        next = fmaxf(next, warp_maxima[(pair * 4 + w) * 16 + lane / 4 + r * 8]);
      float sum = 0;
#pragma unroll
      for (unsigned c = 0; c < 2; ++c) {
        const unsigned token = column_warp * 8 + (lane % 4) * 2 + c;
        const bool visible = row0 + r < rows && token < count && begin + token <= base + row0 + r;
        const float p = visible ? expf(score[r * 2 + c] - next) : 0.0F;
        probabilities[(pair * 16 + lane / 4 + r * 8) * 32 + token] = __float2bfloat16_rn(p);
        sum += p;
      }
      sum += __shfl_xor_sync(0xffffffffU, sum, 1);
      sum += __shfl_xor_sync(0xffffffffU, sum, 2);
      if (lane % 4 == 0) warp_sums[warp * 16 + lane / 4 + r * 8] = sum;
      alpha[r] = next == -CUDART_INF_F ? 1.0F : expf(maximum[r] - next);
      maximum[r] = next;
    }
    __syncthreads();
#pragma unroll
    for (unsigned r = 0; r < 2; ++r) {
      float sum = 0;
#pragma unroll
      for (unsigned w = 0; w < 4; ++w)
        sum += warp_sums[(pair * 4 + w) * 16 + lane / 4 + r * 8];
      denominator[r] = fmaf(denominator[r], alpha[r], sum);
    }
#pragma unroll
    for (unsigned n = 0; n < 16; ++n) {
#pragma unroll
      for (unsigned c = 0; c < 4; ++c) numerator[n][c] *= alpha[c / 2];
    }
#pragma unroll
    for (unsigned k = 0; k < 2; ++k) {
      unsigned p[4];
      matrix_load_a(p, probabilities + pair * 16 * 32 + (lane % 16) * 32 +
          (lane / 16) * 8 + k * 16);
#pragma unroll
      for (unsigned n = 0; n < 16; ++n) {
        unsigned v[2];
        matrix_load_b<true>(v, values + (k * 16 + lane % 16) * Stride + column_warp * 8 + n * 32);
        mma_16x8(numerator[n], p, v);
      }
    }
    if (has_nonfinite) {
      // Restore visible nonfinite values after the finite tensor product;
      // masked future rows must not contaminate earlier queries through 0*NaN.
      for (unsigned token = 0; token < count; ++token) {
        if (begin + token < base) continue;
        const auto* source = staged_value + (std::size_t(begin + token - base) * 4 + head) * D;
#pragma unroll
        for (unsigned n = 0; n < 16; ++n) {
#pragma unroll
          for (unsigned c = 0; c < 4; ++c) {
            const unsigned row = row0 + c / 2;
            if (row >= rows || begin + token > base + row) continue;
            const unsigned d = column_warp * 8 + n * 32 + (lane % 4) * 2 + c % 2;
            const BF16 value = source[d];
            if ((__bfloat16_as_ushort(value) & 0x7f80U) == 0x7f80U)
              numerator[n][c] = fmaf(__bfloat162float(probabilities[
                  (pair * 16 + lane / 4 + (c / 2) * 8) * 32 + token]),
                  __bfloat162float(value), numerator[n][c]);
          }
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (unsigned r = 0; r < 2; ++r) {
    if (row0 + r >= rows) continue;
    const std::size_t result = (std::size_t(grid_y * 4 + pair * 2 + r) * kHeads + query_head) * splits + grid_z;
#pragma unroll
    for (unsigned n = 0; n < 16; ++n) {
      const unsigned d = column_warp * 8 + n * 32 + (lane % 4) * 2;
      partial[result * D + d] = numerator[n][r * 2];
      partial[result * D + d + 1] = numerator[n][r * 2 + 1];
    }
    if (column_warp == 0 && lane % 4 == 0) {
      maxima[result] = maximum[r]; denominators[result] = denominator[r];
    }
  }
}

template <bool Paged>
__global__ __launch_bounds__(kThreads, 1) void global_four_rows(
    const BF16* query, const BF16* staged_key, const BF16* staged_value,
    Cache cache, const BF16* k_norm, std::uint32_t base, std::uint32_t rows,
    std::uint32_t first_row, std::uint32_t splits,
    float* partial, float* maxima, float* denominators) {
  global_four_rows_body<Paged>(query, staged_key, staged_value, cache, k_norm,
      base, rows, first_row, splits, partial, maxima, denominators,
      blockIdx.x, blockIdx.y, blockIdx.z);
}

template <bool Paged>
void launch_global_four_rows(dim3 grid, const BF16* query, const BF16* staged_key,
    const BF16* staged_value, Cache cache, const BF16* k_norm, unsigned base,
    unsigned rows, unsigned first, unsigned splits, float* partial,
    float* maxima, float* denominators, cudaStream_t stream) {
  static const auto status = cudaFuncSetAttribute(global_four_rows<Paged>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kGlobalFp8SharedBytes);
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("MTP shared attention: ") + cudaGetErrorString(status));
  const auto shared = cache.format == kv_cache::Format::fp8
      ? kGlobalFp8SharedBytes : kGlobalSharedBytes;
  global_four_rows<Paged><<<grid, kThreads, shared, stream>>>(query,
      staged_key, staged_value, cache, k_norm, base, rows, first, splits,
      partial, maxima, denominators);
}

// Eight causal rows and their two GQA heads fill the 16 MMA rows. Each lane
// group retains the sliding-window bounds for its query. Softmax stays FP32;
// P*V rounds probabilities once to BF16 and accumulates in FP32 registers.
template<bool Prefetch>
__device__ __forceinline__ void local_eight_rows_body(
    const BF16* query, const BF16* staged_key, const BF16* staged_value,
    Cache cache, std::uint32_t base, std::uint32_t rows,
    std::uint32_t first_row, std::uint32_t splits,
    float* partial, float* maxima, float* denominators,
    unsigned grid_x, unsigned grid_y, unsigned grid_z) {
  constexpr unsigned D = 256, Stride = 264;
  __shared__ __align__(32) BF16 keys[32 * Stride], values[32 * Stride];
  __shared__ __align__(32) BF16 queries[16 * Stride];
  __shared__ float warp_maxima[4 * 16], warp_sums[4 * 16];
  auto* probabilities = keys;
  const unsigned head = grid_x, warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
  const unsigned group_first = first_row + grid_y * 8;
  const unsigned row = group_first + lane / 4;
  const unsigned first = base + group_first > 1023 ? base + group_first - 1023 : 0;
  const unsigned visible_first = base + row > 1023 ? base + row - 1023 : 0;
  const unsigned last = base + min(rows - 1, group_first + 7);
  // The 16 MMA rows cover eight causal positions for each of two GQA heads.
  // Share Q instead of replicating all 64 fragment registers in each compute
  // warp. This avoids register spills while retaining two resident CTAs.
  // The first KV-tile barrier publishes these immutable query copies too.
  for (unsigned i = threadIdx.x * 8; i < 16 * D; i += kThreads * 8) {
    const unsigned r = i / D, d = i % D;
    const auto* source = query +
        (std::size_t(head * 2 + r / 8) * rows + min(group_first + r % 8, rows - 1)) * D + d;
    *reinterpret_cast<uint4*>(queries + r * Stride + d) = load_eight(source);
  }
  float numerator[8][4]{};
  float maximum[2]{-CUDART_INF_F, -CUDART_INF_F}, denominator[2]{};
  for (unsigned begin = first + grid_z * 32; begin <= last; begin += splits * 32) {
    const unsigned count = min(32U, last - begin + 1);
    bool nonfinite = false;
    for (unsigned i = threadIdx.x * 8; i < 32 * D; i += kThreads * 8) {
      const unsigned token = i / D, position = begin + token, d = i % D;
      uint4 key{}, value{};
      if (token < count) {
        key = position >= base ? load_eight(staged_key + (std::size_t(head) * rows + position - base) * D + d)
            : kv_storage::load_eight(kv_storage::row(cache.key,
                std::size_t(head) * 1024 + position % 1024, D, cache.format), d, D, cache.format);
        value = position >= base ? load_eight(staged_value + (std::size_t(position - base) * 16 + head) * D + d)
            : kv_storage::load_eight(kv_storage::row(cache.value,
                std::size_t(head) * 1024 + position % 1024, D, cache.format), d, D, cache.format);
      }
      *reinterpret_cast<uint4*>(keys + token * Stride + d) = key;
      auto* words = reinterpret_cast<unsigned*>(&value);
#pragma unroll
      for (unsigned p = 0; p < 4; ++p) {
        const bool bad_lo = (words[p] & 0x7f80U) == 0x7f80U;
        const bool bad_hi = (words[p] & 0x7f800000U) == 0x7f800000U;
        nonfinite |= bad_lo || bad_hi;
        if (bad_lo) words[p] &= 0xffff0000U;
        if (bad_hi) words[p] &= 0x0000ffffU;
      }
      *reinterpret_cast<uint4*>(values + token * Stride + d) = value;
    }
    const bool has_nonfinite = __syncthreads_or(nonfinite);
    if constexpr (Prefetch) {
      // The four load-assisting warps are free during QK. Pull the next FP8
      // tile into L2, covering each 272-byte record and wrapping within its ring.
      const unsigned prefetch_begin = begin + splits * 32;
      if (warp >= 4 && prefetch_begin < base && prefetch_begin <= last) {
        const unsigned token = (threadIdx.x - 128) / 3;
        const unsigned offset = (threadIdx.x - 128) % 3 * 128;
        if (token < min(32U, base - prefetch_begin)) {
          const auto record = std::size_t(head) * 1024 + (prefetch_begin + token) % 1024;
          const auto* k = kv_storage::row(cache.key, record, D, cache.format);
          const auto* v = kv_storage::row(cache.value, record, D, cache.format);
          prefetch_l2(reinterpret_cast<const unsigned char*>(k) + offset);
          prefetch_l2(reinterpret_cast<const unsigned char*>(v) + offset);
        }
      }
    }
    float score[4]{};
    if (warp < 4) {
#pragma unroll
      for (unsigned k = 0; k < 16; ++k) {
        unsigned query_fragment[4];
        matrix_load_a(query_fragment, queries + (lane % 16) * Stride + (lane / 16) * 8 + k * 16);
        unsigned key[2];
        matrix_load_b<false>(key, keys + (warp * 8 + lane % 8) * Stride + k * 16 + ((lane / 8) % 2) * 8);
        mma_16x8(score, query_fragment, key);
      }
#pragma unroll
      for (unsigned h = 0; h < 2; ++h) {
#pragma unroll
        for (unsigned c = 0; c < 2; ++c) {
          const unsigned token = warp * 8 + (lane % 4) * 2 + c;
          if (row >= rows || token >= count || begin + token < visible_first || begin + token > base + row)
            score[h * 2 + c] = -CUDART_INF_F;
        }
        float local = fmaxf(score[h * 2], score[h * 2 + 1]);
        local = fmaxf(local, __shfl_xor_sync(0xffffffffU, local, 1));
        local = fmaxf(local, __shfl_xor_sync(0xffffffffU, local, 2));
        if (lane % 4 == 0) warp_maxima[warp * 16 + lane / 4 + h * 8] = local;
      }
    }
    __syncthreads();
    float alpha[2];
    if (warp < 4) {
#pragma unroll
      for (unsigned h = 0; h < 2; ++h) {
        float next = maximum[h], sum = 0;
#pragma unroll
        for (unsigned w = 0; w < 4; ++w) next = fmaxf(next, warp_maxima[w * 16 + lane / 4 + h * 8]);
#pragma unroll
        for (unsigned c = 0; c < 2; ++c) {
          const unsigned token = warp * 8 + (lane % 4) * 2 + c;
          const bool visible = row < rows && token < count && begin + token >= visible_first && begin + token <= base + row;
          const float p = visible ? expf(score[h * 2 + c] - next) : 0;
          probabilities[(h * 8 + lane / 4) * 32 + token] = __float2bfloat16_rn(p);
          sum += p;
        }
        sum += __shfl_xor_sync(0xffffffffU, sum, 1);
        sum += __shfl_xor_sync(0xffffffffU, sum, 2);
        if (lane % 4 == 0) warp_sums[warp * 16 + lane / 4 + h * 8] = sum;
        alpha[h] = next == -CUDART_INF_F ? 1.0F : expf(maximum[h] - next);
        maximum[h] = next;
      }
    }
    __syncthreads();
    if (warp < 4) {
#pragma unroll
      for (unsigned h = 0; h < 2; ++h) {
        float sum = 0;
#pragma unroll
        for (unsigned w = 0; w < 4; ++w) sum += warp_sums[w * 16 + lane / 4 + h * 8];
        denominator[h] = fmaf(denominator[h], alpha[h], sum);
      }
#pragma unroll
      for (unsigned n = 0; n < 8; ++n) {
#pragma unroll
        for (unsigned c = 0; c < 4; ++c) numerator[n][c] *= alpha[c / 2];
      }
#pragma unroll
      for (unsigned k = 0; k < 2; ++k) {
        unsigned p[4];
        matrix_load_a(p, probabilities + (lane % 16) * 32 + (lane / 16) * 8 + k * 16);
#pragma unroll
        for (unsigned n = 0; n < 8; ++n) {
          unsigned v[2];
          matrix_load_b<true>(v, values + (k * 16 + lane % 16) * Stride + warp * 8 + n * 32);
          mma_16x8(numerator[n], p, v);
        }
      }
      if (has_nonfinite) {
        for (unsigned token = 0; token < count; ++token) {
          const unsigned position = begin + token;
          if (row >= rows || position < visible_first || position > base + row) continue;
#pragma unroll
          for (unsigned n = 0; n < 8; ++n) {
#pragma unroll
            for (unsigned c = 0; c < 4; ++c) {
              const unsigned d = warp * 8 + n * 32 + (lane % 4) * 2 + c % 2;
              const BF16 value = position >= base
                  ? staged_value[(std::size_t(position - base) * 16 + head) * D + d]
                  : kv_storage::load(kv_storage::row(cache.value, std::size_t(head) * 1024 + position % 1024, D, cache.format), d, D, cache.format);
              if ((__bfloat16_as_ushort(value) & 0x7f80U) == 0x7f80U)
                numerator[n][c] = fmaf(__bfloat162float(probabilities[((c / 2) * 8 + lane / 4) * 32 + token]),
                                      __bfloat162float(value), numerator[n][c]);
            }
          }
        }
      }
    }
    __syncthreads();
  }
  if (warp >= 4 || row >= rows) return;
#pragma unroll
  for (unsigned h = 0; h < 2; ++h) {
    const std::size_t result = (std::size_t(row - first_row) * kHeads + head * 2 + h) * splits + grid_z;
#pragma unroll
    for (unsigned n = 0; n < 8; ++n) {
      const unsigned d = warp * 8 + n * 32 + (lane % 4) * 2;
      partial[result * D + d] = numerator[n][h * 2];
      partial[result * D + d + 1] = numerator[n][h * 2 + 1];
    }
    if (warp == 0 && lane % 4 == 0) { maxima[result] = maximum[h]; denominators[result] = denominator[h]; }
  }
}

__global__ __launch_bounds__(kThreads, 2) void local_eight_rows(
    const BF16* query, const BF16* staged_key, const BF16* staged_value,
    Cache cache, std::uint32_t base, std::uint32_t rows,
    std::uint32_t first_row, std::uint32_t splits,
    float* partial, float* maxima, float* denominators) {
  local_eight_rows_body<false>(query, staged_key, staged_value, cache, base, rows, first_row, splits, partial, maxima, denominators,
      blockIdx.x, blockIdx.y, blockIdx.z);
}

template <unsigned D, unsigned Heads = kHeads>
__device__ __forceinline__ void finalize_attention_body(const float* partial, const float* maxima,
                                   const float* denominators,
                                   std::uint32_t first_row,
                                   std::uint32_t splits, BF16* context, unsigned grid_x, unsigned grid_y) {
  __shared__ float warp_reduction[kThreads / kWarp];
  __shared__ float weights[kMaxSplits];
  __shared__ float total_denominator;
  const unsigned query_head = grid_x;
  const unsigned row = grid_y;
  const unsigned lane = threadIdx.x % kWarp;
  const unsigned warp = threadIdx.x / kWarp;
  const std::size_t first = (std::size_t(row) * Heads + query_head) * splits;
  float maximum = threadIdx.x < splits ? maxima[first + threadIdx.x]
                                      : -CUDART_INF_F;
  maximum = warp_max(maximum);
  if (!lane) warp_reduction[warp] = maximum;
  __syncthreads();
  maximum = lane < kThreads / kWarp ? warp_reduction[lane] : -CUDART_INF_F;
  maximum = warp_max(maximum);
  maximum = __shfl_sync(0xffffffffU, maximum, 0);
  float sum = 0.0F;
  if (threadIdx.x < splits) {
    const float weight = expf(maxima[first + threadIdx.x] - maximum);
    weights[threadIdx.x] = weight;
    sum = weight * denominators[first + threadIdx.x];
  }
  sum = warp_sum(sum);
  // Every warp has read the maxima reduction before its shared slots are reused.
  __syncthreads();
  if (!lane) warp_reduction[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    sum = lane < kThreads / kWarp ? warp_reduction[lane] : 0.0F;
    sum = warp_sum(sum);
    if (!lane) total_denominator = sum;
  }
  __syncthreads();
  for (unsigned d = threadIdx.x; d < D; d += kThreads) {
    float numerator = 0.0F;
    for (unsigned split = 0; split < splits; ++split)
      numerator = fmaf(weights[split], partial[(first + split) * D + d],
                       numerator);
    context[(std::size_t(first_row + row) * Heads + query_head) * D + d] =
        __float2bfloat16_rn(numerator / total_denominator);
  }
}

template <unsigned D, unsigned Heads = kHeads>
__global__ void finalize_attention(const float* partial, const float* maxima,
                                   const float* denominators,
                                   std::uint32_t first_row,
                                   std::uint32_t splits, BF16* context) {
  finalize_attention_body<D, Heads>(partial, maxima, denominators, first_row, splits,
                             context, blockIdx.x, blockIdx.y);
}


void check_launch() {
  const auto status = cudaGetLastError();
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("MTP parallel attention: ") +
                             cudaGetErrorString(status));
}

// Launch parameters hold a bounded group of independent tiles. This avoids a
// device descriptor upload and keeps CUDA graph capture allocation-free.
constexpr unsigned kBatchTiles = 32;
struct AttentionTile {
  BatchInput input;
  unsigned first{}, count{}, splits{}, groups{}, end_blocks{}, end_rows{};
  float *partial{}, *maxima{}, *denominators{};
};
struct AttentionBatch { AttentionTile tiles[kBatchTiles]; };

// A soft block target for the SM120 kernels: about 2.7 resident waves on the
// 188-SM tuning GPU. Multirow global/local kernels permit one/two resident
// CTAs per SM; single-row kernels permit two/four. Count only this launch's
// tiles, and distribute splits in proportion to visible KV work. A lone tile
// retains its original split count. Scratch capacity is a separate hard limit.
bool plan_batch(AttentionBatch& batch, unsigned count, bool global,
                void* scratch, std::size_t bytes, bool frozen = false, unsigned heads = kHeads) {
  const bool four_rows = batch.tiles[0].input.rows > 1;
  const unsigned group_budget = (global ? 128 : 64) * (four_rows ? 1 : 2);
  const unsigned D = global ? 512 : 256;
  std::array<unsigned, kBatchTiles> visible{}, splits{};
  std::uint64_t work = 0;
  for (unsigned i = 0; i < count; ++i) {
    const auto& tile = batch.tiles[i];
    const auto end = tile.input.base_position +
                     (frozen ? 0 : tile.first + tile.count);
    visible[i] = global ? end : std::min(1024U, end);
    work += std::uint64_t(tile.groups) * visible[i];
  }
  std::size_t needed = 0;
  for (unsigned i = 0; i < count; ++i) {
    const auto& tile = batch.tiles[i];
    const auto limit = default_splits(tile.input.base_position,
                                      tile.input.rows, global, frozen);
    const auto budget = unsigned((std::uint64_t(visible[i]) * group_budget + work - 1) / work);
    splits[i] = count == 1 ? limit : std::min(limit, std::max(1U, budget));
    needed += bytes_for(tile.count, splits[i], D, heads);
  }
  if (needed > bytes) return false;

  unsigned blocks = 0, rows = 0;
  std::size_t used = 0;
  for (unsigned i = 0; i < count; ++i) {
    auto& tile = batch.tiles[i];
    tile.splits = splits[i];
    tile.end_blocks = blocks += tile.groups * tile.splits;
    tile.end_rows = rows += tile.count;
    const auto entries = std::size_t(tile.count) * heads * tile.splits;
    tile.partial = reinterpret_cast<float*>(static_cast<unsigned char*>(scratch) + used);
    tile.maxima = tile.partial + entries * D;
    tile.denominators = tile.maxima + entries;
    used += bytes_for(tile.count, tile.splits, D, heads);
  }
  return true;
}

template<bool Global, bool Paged, bool FourRows, bool Frozen, bool Fp8>
__global__ __launch_bounds__(kThreads, (Global && (FourRows || (Frozen && Fp8))) ? 1 : 2)
void batched_partial_attention(const __grid_constant__ AttentionBatch batch,
                               const BF16* k_norm) {
  unsigned index = 0;
  while (blockIdx.y >= batch.tiles[index].end_blocks) ++index;
  const auto& tile = batch.tiles[index];
  auto input = tile.input;
  input.cache.format = Fp8 ? kv_cache::Format::fp8 : kv_cache::Format::bf16;
  const auto block = blockIdx.y - (index ? batch.tiles[index - 1].end_blocks : 0);
  const auto group = block % tile.groups, split = block / tile.groups;
  static_assert(!Frozen || (Global && !FourRows));
  if constexpr (Global && FourRows)
    global_four_rows_body<Paged>(input.query, input.staged_key, input.staged_value,
        input.cache, k_norm, input.base_position, input.rows, tile.first, tile.splits,
        tile.partial, tile.maxima, tile.denominators, blockIdx.x, group, split);
  else if constexpr (FourRows)
    local_eight_rows_body<Fp8>(input.query, input.staged_key, input.staged_value,
        input.cache, input.base_position, input.rows, tile.first, tile.splits,
        tile.partial, tile.maxima, tile.denominators, blockIdx.x, group, split);
  else
    partial_attention_body<Global, Paged, Frozen, Global && Frozen && Fp8>(input.query, input.staged_key, input.staged_value,
        input.cache, k_norm, input.base_position, input.rows, tile.first, tile.splits,
        tile.partial, tile.maxima, tile.denominators, blockIdx.x, group, split);
}

template<unsigned D, unsigned Heads = kHeads>
__global__ void batched_finalize_attention(const __grid_constant__ AttentionBatch batch) {
  unsigned index = 0;
  while (blockIdx.y >= batch.tiles[index].end_rows) ++index;
  const auto& tile = batch.tiles[index];
  const auto row = blockIdx.y - (index ? batch.tiles[index - 1].end_rows : 0);
  finalize_attention_body<D, Heads>(tile.partial, tile.maxima, tile.denominators,
      tile.first, tile.splits, tile.input.context, blockIdx.x, row);
}

template<bool Global, bool Paged, bool FourRows, bool Frozen, bool Fp8>
void launch_batch_format(const AttentionBatch& batch, unsigned count,
                         const BF16* k_norm, cudaStream_t stream) {
  constexpr auto shared = Global && FourRows
      ? (Fp8 ? kGlobalFp8SharedBytes : kGlobalSharedBytes)
      : Global && Frozen && Fp8 ? kCompactTileBytes : 0;
  if constexpr (shared) {
    static const auto status = cudaFuncSetAttribute(
        batched_partial_attention<Global, Paged, FourRows, Frozen, Fp8>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
    if (status != cudaSuccess)
      throw std::runtime_error(std::string("MTP batched shared attention: ") + cudaGetErrorString(status));
  }
  const auto& last = batch.tiles[count - 1];
  batched_partial_attention<Global, Paged, FourRows, Frozen, Fp8>
      <<<dim3(Global ? 4 : 16, last.end_blocks), kThreads, shared, stream>>>(batch, k_norm);
  check_launch();
  batched_finalize_attention<Global ? 512 : 256>
      <<<dim3(kHeads, last.end_rows), kThreads, 0, stream>>>(batch);
  check_launch();
}

template<bool Global, bool Paged, bool FourRows, bool Frozen = false>
void launch_batch(const AttentionBatch& batch, unsigned count,
                  const BF16* k_norm, cudaStream_t stream) {
  if (batch.tiles[0].input.cache.format == kv_cache::Format::fp8)
    launch_batch_format<Global, Paged, FourRows, Frozen, true>(batch, count, k_norm, stream);
  else
    launch_batch_format<Global, Paged, FourRows, Frozen, false>(batch, count, k_norm, stream);
}

#include "fp8_attention.cuh"
}  // namespace

std::size_t scratch_bytes(std::uint32_t rows, std::uint32_t context) {
  if (!rows || rows > mtp_target::kMaxDepth + 1 || !context ||
      context > kMaxContext)
    throw std::invalid_argument("MTP attention scratch geometry outside capacity");
  return bytes_for(rows, splits_for(context), 512);
}

static unsigned checked_splits(const BF16* query, const BF16* staged_key, const BF16* staged_value,
         const Cache& cache, const BF16* k_norm, std::uint32_t base,
         std::uint32_t rows, gemma4_31b::AttentionKind kind, BF16* context,
         void* scratch, std::size_t bytes, bool frozen, unsigned heads = kHeads, unsigned max_rows = mtp_target::kMaxDepth + 1) {
  const bool global = kind == gemma4_31b::AttentionKind::global;
  if ((kind != gemma4_31b::AttentionKind::local && !global) || !query ||
      (!frozen && (!staged_key || !staged_value)) || !context || !scratch || !rows ||
      rows > max_rows ||
      std::uint64_t(base) + (frozen ? 0 : rows) > kMaxContext ||
      (frozen && (!base || rows != 1)))
    throw std::invalid_argument("MTP attention inputs outside capacity");
  if (global) {
    if (!k_norm)
      throw std::invalid_argument("MTP global attention requires K norm scale");
    // A cold causal prefill reads staged keys only.
    if (base && cache.page_pool) {
      const std::size_t layer_elements = (heads / 8) * 256 * kv_cache::row_words(640, cache.format, 2);
      if (!cache.page_offsets || cache.page_tokens != 256 ||
          (std::uint64_t(base) + 255) / 256 > cache.page_count ||
          cache.layer_offset_elements > cache.page_stride_elements ||
          layer_elements > cache.page_stride_elements - cache.layer_offset_elements)
        throw std::invalid_argument("MTP paged attention does not cover committed KV");
    } else if (base && (!cache.key || base > cache.capacity)) {
      throw std::invalid_argument("MTP global attention does not cover committed KV");
    }
  } else if (base && (!cache.key || !cache.value || cache.capacity != 1024)) {
    throw std::invalid_argument("MTP local attention requires separate 1024-row rings");
  }
  const unsigned D = global ? 512 : 256;
  const auto splits = default_splits(base, rows, global, frozen);
  if (bytes < bytes_for(rows, splits, D, heads))
    throw std::invalid_argument("MTP attention scratch is too small");
  return splits;
}

std::size_t frozen_scratch_bytes(unsigned heads, std::uint32_t context) {
  if ((heads != 16 && heads != 32) || !context || context > kMaxContext)
    throw std::invalid_argument("MTP frozen attention geometry outside capacity");
  // The existing local planner includes one row when sizing frozen splits.
  // At L=32 its two 256-wide splits need slightly more than one 512-wide
  // global split. Cover both without changing the established split policy.
  return std::max(bytes_for(1, splits_for(context), 512, heads),
      bytes_for(1, default_splits(context, 1, false, true), 256, heads));
}

template<unsigned Heads, bool Global, bool Paged>
static void frozen_prefix(const BF16* query, const Cache& cache, const BF16* norm,
                          unsigned base, BF16* output, void* scratch,
                          unsigned splits, cudaStream_t stream) {
  constexpr unsigned D = Global ? 512 : 256;
  constexpr unsigned KvHeads = Global ? Heads / 8 : Heads / 2;
  const auto entries = std::size_t(Heads) * splits;
  auto* partial = static_cast<float*>(scratch);
  auto* maxima = partial + entries * D;
  auto* denominator = maxima + entries;
  partial_attention<Global, Paged, true, Heads>
      <<<dim3(KvHeads, 1, splits), kThreads, 0, stream>>>(query, nullptr,
          nullptr, cache, norm, base, 1, 0, splits, partial, maxima, denominator);
  check_launch();
  finalize_attention<D, Heads><<<Heads, kThreads, 0, stream>>>(
      partial, maxima, denominator, 0, splits, output);
  check_launch();
}

void run_frozen_prefix(const BF16* query, const Cache& cache, const BF16* norm,
                       std::uint32_t processed, unsigned heads,
                       gemma4_31b::AttentionKind kind, BF16* output,
                       void* scratch, std::size_t bytes, cudaStream_t stream) {
  if ((heads != 16 && heads != 32) ||
      (cache.format != kv_cache::Format::bf16 && cache.format != kv_cache::Format::fp8))
    throw std::invalid_argument("MTP frozen attention invalid heads or cache format");
  const auto splits = checked_splits(query, nullptr, nullptr, cache, norm,
      processed, 1, kind, output, scratch, bytes, true, heads);
  const bool global = kind == gemma4_31b::AttentionKind::global;
  const auto launch = [&](auto head_count) {
    constexpr unsigned Heads = decltype(head_count)::value;
    if (!global) frozen_prefix<Heads, false, false>(query, cache, norm, processed,
        output, scratch, splits, stream);
    else if (cache.page_pool) frozen_prefix<Heads, true, true>(query, cache, norm,
        processed, output, scratch, splits, stream);
    else frozen_prefix<Heads, true, false>(query, cache, norm, processed,
        output, scratch, splits, stream);
  };
  if (heads == 16) launch(std::integral_constant<unsigned, 16>{});
  else launch(std::integral_constant<unsigned, 32>{});
}

namespace {
template<unsigned Heads, bool Global, bool Paged, bool Frozen = true>
__global__ __launch_bounds__(kThreads, 2) void single_row_batch_partial(
    const __grid_constant__ AttentionBatch batch, const BF16* norm) {
  unsigned index = 0;
  while (blockIdx.y >= batch.tiles[index].end_blocks) ++index;
  const auto& tile = batch.tiles[index];
  const unsigned split = blockIdx.y - (index ? batch.tiles[index - 1].end_blocks : 0);
  const auto& input = tile.input;
  partial_attention_body<Global, Paged, Frozen, false, Heads>(input.query,
      input.staged_key, input.staged_value, input.cache, norm, input.base_position, 1, 0,
      tile.splits, tile.partial, tile.maxima, tile.denominators,
      blockIdx.x, 0, split);
}

template<unsigned Heads, bool Global, bool Paged, bool Frozen = true>
void launch_single_row_batch(const AttentionBatch& batch, unsigned count,
                         const BF16* norm, cudaStream_t stream) {
  single_row_batch_partial<Heads, Global, Paged, Frozen>
      <<<dim3(Global ? Heads / 8 : Heads / 2, batch.tiles[count - 1].end_blocks),
         kThreads, 0, stream>>>(batch, norm);
  check_launch();
  batched_finalize_attention<Global ? 512 : 256, Heads>
      <<<dim3(Heads, count), kThreads, 0, stream>>>(batch);
  check_launch();
}

template<unsigned Heads, bool Frozen = true>
void fixed_split_batch(const std::vector<BatchInput>& inputs, const BF16* norm,
                  bool global, void* scratch, std::size_t bytes, cudaStream_t stream) {
  AttentionBatch batch{};
  unsigned count = 0, blocks = 0;
  std::size_t used = 0;
  bool paged = false;
  const auto flush = [&] {
    if (!count) return;
    if (!global) launch_single_row_batch<Heads, false, false, Frozen>(batch, count, norm, stream);
    else if (paged) launch_single_row_batch<Heads, true, true, Frozen>(batch, count, norm, stream);
    else launch_single_row_batch<Heads, true, false, Frozen>(batch, count, norm, stream);
    count = blocks = 0;
    used = 0;
  };
  const unsigned D = global ? 512 : 256;
  for (const auto& input : inputs) {
    const auto splits = default_splits(input.base_position, 1, global, Frozen);
    const auto needed = bytes_for(1, splits, D, Heads);
    if (count == kBatchTiles || needed > bytes - used ||
        (count && global && paged != bool(input.cache.page_pool))) flush();
    paged = bool(input.cache.page_pool);
    auto* partial = reinterpret_cast<float*>(static_cast<unsigned char*>(scratch) + used);
    auto* maxima = partial + std::size_t(Heads) * splits * D;
    batch.tiles[count] = {input, 0, 1, splits, 1, blocks + splits, count + 1,
        partial, maxima, maxima + std::size_t(Heads) * splits};
    ++count;
    blocks += splits;
    used += needed;
  }
  flush();
}
}  // namespace

void run_frozen_prefix_batch(unsigned heads, const std::vector<BatchInput>& inputs,
                             const BF16* norm, gemma4_31b::AttentionKind kind,
                             void* scratch, std::size_t bytes, cudaStream_t stream) {
  if ((heads != 16 && heads != 32) || inputs.empty())
    throw std::invalid_argument("MTP frozen attention invalid heads or empty batch");
  for (const auto& input : inputs) {
    if (input.cache.format != kv_cache::Format::bf16 && input.cache.format != kv_cache::Format::fp8)
      throw std::invalid_argument("MTP frozen attention invalid cache format");
    checked_splits(input.query, nullptr, nullptr, input.cache, norm,
        input.base_position, input.rows, kind, input.context, scratch, bytes, true, heads);
  }
  const bool global = kind == gemma4_31b::AttentionKind::global;
  if (heads == 16) fixed_split_batch<16>(inputs, norm, global, scratch, bytes, stream);
  else fixed_split_batch<32>(inputs, norm, global, scratch, bytes, stream);
}

void run_decode_batch(unsigned heads, const std::vector<BatchInput>& inputs,
                      const BF16* norm, gemma4_31b::AttentionKind kind,
                      void* scratch, std::size_t bytes, cudaStream_t stream) {
  if ((heads != 16 && heads != 32) || inputs.empty())
    throw std::invalid_argument("Decode attention invalid heads or empty batch");
  for (const auto& input : inputs) {
    if (input.rows != 1 || input.local_image ||
        (input.cache.format != kv_cache::Format::bf16 && input.cache.format != kv_cache::Format::fp8))
      throw std::invalid_argument("Decode attention requires one causal row per input");
    checked_splits(input.query, input.staged_key, input.staged_value, input.cache, norm,
        input.base_position, 1, kind, input.context, scratch, bytes, false, heads);
  }
  const bool global = kind == gemma4_31b::AttentionKind::global;
  const auto launch = [&](auto head_count) {
    constexpr unsigned Heads = decltype(head_count)::value;
    if (!global) {
      // Keep each local CTA to one 32-key tile. Larger tiles serialize the
      // scalar local P*V loop and lose the benefit of coalescing requests.
      fixed_split_batch<Heads, false>(inputs, norm, false, scratch, bytes, stream);
      return;
    }
    AttentionBatch batch{};
    unsigned count = 0;
    bool paged = false;
    const auto flush = [&] {
      if (!count) return;
      if (paged) launch_single_row_batch<Heads, true, true, false>(batch, count, norm, stream);
      else launch_single_row_batch<Heads, true, false, false>(batch, count, norm, stream);
      count = 0;
    };
    for (const auto& input : inputs) {
      if (count == kBatchTiles || (count && paged != bool(input.cache.page_pool))) flush();
      paged = bool(input.cache.page_pool);
      const AttentionTile tile{input, 0, 1, 0, 1};
      batch.tiles[count] = tile;
      if (!plan_batch(batch, count + 1, global, scratch, bytes, false, Heads)) {
        flush();
        batch.tiles[0] = tile;
        if (!plan_batch(batch, 1, global, scratch, bytes, false, Heads))
          throw std::logic_error("Validated decode attention tile does not fit scratch");
      }
      ++count;
    }
    flush();
  };
  if (heads == 16) launch(std::integral_constant<unsigned, 16>{});
  else launch(std::integral_constant<unsigned, 32>{});
}

template<unsigned Heads>
static void fp8_batch(const std::vector<BatchInput>& inputs, const BF16* k_norm,
                   gemma4_31b::AttentionKind kind, void* scratch,
                   std::size_t bytes, cudaStream_t stream, bool frozen) {
  if (inputs.empty()) throw std::invalid_argument("FP8 attention batch is empty");
  for (const auto& i : inputs) {
    if (i.local_image && (Heads != 16 || frozen || kind != gemma4_31b::AttentionKind::local || i.rows > 1120))
      throw std::invalid_argument("FP8 image attention requires a complete 26B local image span");
    checked_splits(i.query, i.staged_key, i.staged_value, i.cache, k_norm,
        i.base_position, i.rows, kind, i.context, scratch, bytes, frozen, Heads, Heads == 16 ? 4096 : 1280);
  }
  const bool global = kind == gemma4_31b::AttentionKind::global;
  AttentionBatch batch{};
  unsigned count = 0;
  bool paged = false;
  const auto flush = [&] {
    if (!count) return;
    if (global && paged) {
      if (frozen) launch_fp8_attention<Heads, true, true, true>(batch, count, k_norm, stream);
      else launch_fp8_attention<Heads, true, true, false>(batch, count, k_norm, stream);
    } else if (global) {
      if (frozen) launch_fp8_attention<Heads, true, false, true>(batch, count, k_norm, stream);
      else launch_fp8_attention<Heads, true, false, false>(batch, count, k_norm, stream);
    } else {
      if (frozen) launch_fp8_attention<Heads, false, false, true>(batch, count, nullptr, stream);
      else launch_fp8_attention<Heads, false, false, false>(batch, count, nullptr, stream);
    }
    count = 0;
  };
  for (const auto& i : inputs) {
    for (unsigned first = 0; first < i.rows; first += kQueryTileRows) {
      if (count == kBatchTiles || (count && paged != bool(i.cache.page_pool))) flush();
      paged = bool(i.cache.page_pool);
      const unsigned rows = std::min(kQueryTileRows, i.rows - first);
      const unsigned group_rows = global ? (frozen ? 2 : 4) : 8;
      const AttentionTile tile{i, first, rows, 0, (rows + group_rows - 1) / group_rows};
      batch.tiles[count] = tile;
      if (!plan_batch(batch, count + 1, global, scratch, bytes, frozen, Heads)) {
        flush(); batch.tiles[0] = tile;
        if (!plan_batch(batch, 1, global, scratch, bytes, frozen, Heads))
          throw std::logic_error("validated FP8 attention tile does not fit scratch");
      }
      ++count;
    }
  }
  flush();
}

std::size_t fp8_scratch_bytes(unsigned heads, std::uint32_t rows, std::uint32_t context) {
  if ((heads != 16 && heads != 32) || !rows || rows > (heads == 16 ? 4096U : 1280U) ||
      !context || context > kMaxContext)
    throw std::invalid_argument("FP8 attention scratch geometry outside capacity");
  return std::max(bytes_for(rows, splits_for(context), 512, heads),
                 bytes_for(rows, default_splits(context, rows, false), 256, heads));
}

void run_fp8_batch(unsigned heads, const std::vector<BatchInput>& inputs, const BF16* k_norm,
                   gemma4_31b::AttentionKind kind, void* scratch,
                   std::size_t bytes, cudaStream_t stream, bool frozen) {
  if (heads != 16 && heads != 32)
    throw std::invalid_argument("FP8 attention invalid query head count");
  for (const auto& i : inputs)
    if (i.cache.format != kv_cache::Format::bf16 && i.cache.format != kv_cache::Format::fp8)
      throw std::invalid_argument("FP8 attention invalid KV storage format");
  if (heads == 16) fp8_batch<16>(inputs, k_norm, kind, scratch, bytes, stream, frozen);
  else fp8_batch<32>(inputs, k_norm, kind, scratch, bytes, stream, frozen);
}

static void run_impl(const BF16* query, const BF16* staged_key, const BF16* staged_value,
         const Cache& cache, const BF16* k_norm, std::uint32_t base,
         std::uint32_t rows, gemma4_31b::AttentionKind kind, BF16* context,
         void* scratch, std::size_t bytes, cudaStream_t stream, bool frozen) {
  const auto splits = checked_splits(query, staged_key, staged_value, cache,
      k_norm, base, rows, kind, context, scratch, bytes, frozen);
  const bool global = kind == gemma4_31b::AttentionKind::global;
  const unsigned D = global ? 512 : 256;
  // Multi-row calls share four rows per KV tile. Single rows and frozen
  // assistant calls keep the smaller specialization.
  const bool share_rows = global && rows >= 2;
  for (unsigned first = 0; first < rows; first += kQueryTileRows) {
    const unsigned count = std::min(kQueryTileRows, rows - first);
    const std::size_t entries = std::size_t(count) * kHeads * splits;
    auto* partial = static_cast<float*>(scratch);
    auto* maxima = partial + entries * D;
    auto* denominators = maxima + entries;
    const dim3 grid(global ? 4 : 16, share_rows ? (count + 3) / 4 : count, splits);
    if (global && cache.page_pool) {
      if (frozen)
        partial_attention<true, true, true><<<grid, kThreads, 0, stream>>>(
            query, nullptr, nullptr, cache, k_norm, base, rows, first,
            splits, partial, maxima, denominators);
      else if (share_rows)
        launch_global_four_rows<true>(grid,
            query, staged_key, staged_value, cache, k_norm, base, rows, first,
            splits, partial, maxima, denominators, stream);
      else
        partial_attention<true, true><<<grid, kThreads, 0, stream>>>(
            query, staged_key, staged_value, cache, k_norm, base, rows, first,
            splits, partial, maxima, denominators);
    } else if (global) {
      if (frozen)
        partial_attention<true, false, true><<<grid, kThreads, 0, stream>>>(
            query, nullptr, nullptr, cache, k_norm, base, rows, first,
            splits, partial, maxima, denominators);
      else if (share_rows)
        launch_global_four_rows<false>(grid,
            query, staged_key, staged_value, cache, k_norm, base, rows, first,
            splits, partial, maxima, denominators, stream);
      else
        partial_attention<true, false><<<grid, kThreads, 0, stream>>>(
            query, staged_key, staged_value, cache, k_norm, base, rows, first,
            splits, partial, maxima, denominators);
    } else if (rows > 1) {
      local_eight_rows<<<dim3(16, (count + 7) / 8, splits), kThreads, 0, stream>>>(
          query, staged_key, staged_value, cache, base, rows, first,
          splits, partial, maxima, denominators);
    } else {
      partial_attention<false, false><<<grid, kThreads, 0, stream>>>(
          query, staged_key, staged_value, cache, k_norm, base, rows, first,
          splits, partial, maxima, denominators);
    }
    check_launch();
    if (global)
      finalize_attention<512><<<dim3(kHeads, count), kThreads, 0, stream>>>(
          partial, maxima, denominators, first, splits, context);
    else
      finalize_attention<256><<<dim3(kHeads, count), kThreads, 0, stream>>>(
          partial, maxima, denominators, first, splits, context);
    check_launch();
  }
}

void run(const BF16* query, const BF16* staged_key, const BF16* staged_value,
         const Cache& cache, const BF16* k_norm, std::uint32_t base,
         std::uint32_t rows, gemma4_31b::AttentionKind kind, BF16* context,
         void* scratch, std::size_t bytes, cudaStream_t stream) {
  run_impl(query, staged_key, staged_value, cache, k_norm, base, rows, kind,
           context, scratch, bytes, stream, false);
}

void run_batch(const std::vector<BatchInput>& inputs, const BF16* k_norm,
               gemma4_31b::AttentionKind kind, void* scratch,
               std::size_t bytes, cudaStream_t stream) {
  if (inputs.empty()) throw std::invalid_argument("MTP attention batch is empty");
  if (inputs.size() == 1) {
    const auto& i = inputs.front();
    run(i.query, i.staged_key, i.staged_value, i.cache, k_norm, i.base_position,
         i.rows, kind, i.context, scratch, bytes, stream);
    return;
  }
  // Validate before enqueuing a batch; each tile must fit on its own.
  for (const auto& i : inputs)
    checked_splits(i.query, i.staged_key, i.staged_value, i.cache, k_norm,
                    i.base_position, i.rows, kind, i.context, scratch, bytes, false);
  const bool global = kind == gemma4_31b::AttentionKind::global;
  AttentionBatch batch{};
  unsigned count = 0;
  bool paged = false, four_rows = false;
  const auto flush = [&] {
    if (!count) return;
    if (global) {
      if (paged) {
        if (four_rows) launch_batch<true, true, true>(batch, count, k_norm, stream);
        else launch_batch<true, true, false>(batch, count, k_norm, stream);
      } else {
        if (four_rows) launch_batch<true, false, true>(batch, count, k_norm, stream);
        else launch_batch<true, false, false>(batch, count, k_norm, stream);
      }
    } else {
      if (four_rows) launch_batch<false, false, true>(batch, count, nullptr, stream);
      else launch_batch<false, false, false>(batch, count, nullptr, stream);
    }
    count = 0;
  };
  // A depth-zero request needs the single-row specialization. Interleaving
  // those requests with draft prefixes needlessly flushes compatible tiles.
  // Visit each specialization together; request buffers and causal bounds stay
  // attached to their own descriptors, and launches still reuse scratch serially.
  for (const bool multirow : {false, true}) {
    for (const auto& i : inputs) {
      if ((i.rows > 1) != multirow) continue;
      for (unsigned first = 0; first < i.rows; first += kQueryTileRows) {
        const auto tile_rows = std::min(kQueryTileRows, i.rows - first);
        if (count == kBatchTiles ||
            (count && (paged != bool(i.cache.page_pool) ||
                       batch.tiles[0].input.cache.format != i.cache.format))) flush();
        paged = bool(i.cache.page_pool);
        four_rows = i.rows > 1;
        const unsigned group_rows = global ? 4 : 8;
        const auto groups = four_rows ? (tile_rows + group_rows - 1) / group_rows : tile_rows;
        const AttentionTile tile{i, first, tile_rows, 0, groups};
        batch.tiles[count] = tile;
        if (!plan_batch(batch, count + 1, global, scratch, bytes)) {
          flush();
          batch.tiles[0] = tile;
          if (!plan_batch(batch, 1, global, scratch, bytes))
            throw std::logic_error("validated MTP attention tile does not fit scratch");
        }
        ++count;
      }
    }
    flush();
  }
}

void run_frozen_global_batch(const std::vector<BatchInput>& inputs,
                             const BF16* k_norm, void* scratch,
                             std::size_t bytes, cudaStream_t stream) {
  if (inputs.empty())
    throw std::invalid_argument("MTP frozen global attention batch is empty");
  if (inputs.size() == 1) {
    const auto& input = inputs.front();
    run_frozen_global(input.query, input.cache, k_norm,
                      input.base_position, input.context, scratch, bytes,
                      stream);
    return;
  }
  for (const auto& input : inputs)
    checked_splits(input.query, input.staged_key, input.staged_value,
                   input.cache, k_norm, input.base_position, input.rows,
                   gemma4_31b::AttentionKind::global, input.context,
                   scratch, bytes, true);

  AttentionBatch batch{};
  unsigned count = 0;
  bool paged = false;
  const auto flush = [&] {
    if (!count) return;
    if (paged)
      launch_batch<true, true, false, true>(batch, count, k_norm, stream);
    else
      launch_batch<true, false, false, true>(batch, count, k_norm, stream);
    count = 0;
  };
  for (const auto& input : inputs) {
    if (count == kBatchTiles ||
        (count && (paged != bool(input.cache.page_pool) ||
                   batch.tiles[0].input.cache.format != input.cache.format)))
      flush();
    paged = bool(input.cache.page_pool);
    const AttentionTile tile{input, 0, 1, 0, 1};
    batch.tiles[count] = tile;
    if (!plan_batch(batch, count + 1, true, scratch, bytes, true)) {
      flush();
      batch.tiles[0] = tile;
      if (!plan_batch(batch, 1, true, scratch, bytes, true))
        throw std::logic_error(
            "validated frozen global attention tile does not fit scratch");
    }
    ++count;
  }
  flush();
}

void run_frozen_global(const BF16* query, const Cache& cache, const BF16* k_norm,
                       std::uint32_t processed, BF16* context, void* scratch,
                       std::size_t bytes, cudaStream_t stream) {
  run_impl(query, nullptr, nullptr, cache, k_norm, processed, 1,
           gemma4_31b::AttentionKind::global, context, scratch, bytes, stream, true);
}
}  // namespace gewell::mtp_attention
