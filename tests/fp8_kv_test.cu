#include "gewell/bf16_primitives.h"
#include "gewell/models/gemma4/26b_a4b/cache_config.h"
#include "models/gemma4/26b_a4b/sm120/attention.cuh"
#include "gewell/prefill_primitives.h"
#include "gewell/mtp_attention.h"
#include "gewell/mtp_assistant.h"
#include "../src/runtime/physical_cache.h"
#include "../src/models/gemma4/31b/sm120/resources.cuh"
#include "../src/models/gemma4/31b/sm120/cache_config.h"
#include "kv_storage.cuh"
#include "../src/models/gemma4/31b/sm120/mtp/fp8_quantize.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
namespace b = gewell::bf16_primitives;
namespace p = gewell::prefill_primitives;
namespace a = gewell::mtp_attention;
namespace m = gewell::gemma4_31b;
namespace kv = gewell::kv_cache;
using BF16 = __nv_bfloat16;
using Cache = gewell::kv_cache::DeviceView;
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
class Device {
 public:
  explicit Device(std::size_t bytes) : bytes(bytes) { check(cudaMalloc(&data, bytes)); }
  ~Device() { cudaFree(data); }
  template <typename T = BF16> T* get() const { return static_cast<T*>(data); }
  template <typename T> void upload(const std::vector<T>& values) {
    require(values.size() * sizeof(T) == bytes, "upload size");
    check(cudaMemcpy(data, values.data(), bytes, cudaMemcpyHostToDevice));
  }
  std::vector<unsigned char> host() const {
    std::vector<unsigned char> result(bytes);
    check(cudaMemcpy(result.data(), data, bytes, cudaMemcpyDeviceToHost));
    return result;
  }
  std::size_t bytes;
 private:
  void* data{};
};
BF16 bf(float x) { return __float2bfloat16_rn(x); }
float fp(BF16 x) { return __bfloat162float(x); }

// Independent exhaustive E4M3 oracle: enumerate all positive finite codes,
// then choose the nearest, resolving exact ties to an even low mantissa bit.
float decoded(unsigned code) {
  const auto e = code >> 3, fraction = code & 7;
  return e ? std::ldexp(1.0F + fraction / 8.0F, int(e) - 7)
           : std::ldexp(float(fraction), -9);
}
unsigned char encoded(float x) {
  const float magnitude = std::fabs(x);
  unsigned best = 0;
  float error = std::numeric_limits<float>::infinity();
  for (unsigned code = 0; code < 127; ++code) {
    const auto distance = std::fabs(magnitude - decoded(code));
    if (distance < error || (distance == error && !(code & 1))) {
      best = code;
      error = distance;
    }
  }
  return best | (std::signbit(x) ? 128 : 0);
}
float fp8_round(float x) {
  static const auto table = [] {
    std::array<float, 127> values{};
    for (unsigned i = 0; i < 127; ++i) values[i] = decoded(i);
    return values;
  }();
  const float magnitude = std::fabs(x);
  auto upper = std::lower_bound(table.begin(), table.end(), magnitude);
  unsigned code = upper == table.end() ? 126 : unsigned(upper - table.begin());
  if (code) {
    const float above = table[code] - magnitude, below = magnitude - table[code - 1];
    if (below < above || (below == above && (code & 1))) --code;
  }
  return std::copysign(table[code], x);
}
void quantize(const std::vector<BF16>& values, unsigned split,
              unsigned char* record, BF16* restored) {
  const unsigned n = values.size(), groups = split < n ? 2 : 1;
  for (unsigned group = 0; group < groups; ++group) {
    const auto begin = group ? split : 0U, end = group ? n : split;
    float maximum = 0;
    for (unsigned d = begin; d < end; ++d) maximum = std::max(maximum, std::fabs(fp(values[d])));
    const float scale = maximum ? maximum / 448.0F : 1.0F;
    std::memcpy(record + n + group * 4, &scale, 4);
    for (unsigned d = begin; d < end; ++d) {
      const auto code = encoded(fp(values[d]) / scale);
      record[d] = code;
      restored[d] = bf(decoded(code & 127) * (code & 128 ? -scale : scale));
    }
  }
  std::fill(record + n + groups * 4, record + kv::row_bytes(n, kv::Format::fp8, groups), 0);
}
BF16 pattern(unsigned tag, unsigned token, unsigned head, unsigned d) {
  if ((token + head) % 43 == 0) return bf(0);
  unsigned x = tag * 0x9e3779b9U + token * 0x85ebca6bU + head * 0xc2b2ae35U + d;
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15;
  const float magnitude = tag == 1 ? 1.0F / 2048 : std::ldexp(1.0F, int(token % 7) - 9);
  return bf((int(x & 255) - 127) * magnitude);
}

__device__ void check_quantized_ratio(float x, float scale, unsigned& errors) {
  const auto expected = __nv_cvt_float_to_fp8(x / scale, __NV_SATFINITE, __NV_E4M3);
  const auto actual = a::detail::fp8_quantize(x, scale, 1.0F / scale);
  errors += actual != expected;
}

__global__ void quantization_bf16_scales(unsigned* failed) {
  const unsigned m = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (m >= 0x7f80) return;
  const float maximum = __uint_as_float(m << 16);
  const float scale = a::detail::fp8_scale(maximum);
  unsigned errors = 0;
  // All finite positive BF16 maxima, and values through 20 exponent bins
  // below each. Covers underflow, the fast-path boundary, and large scales.
  for (unsigned x = m > 2560 ? m - 2560 : 0; x <= m; ++x)
    check_quantized_ratio(__uint_as_float(x << 16), scale, errors);
  if (errors) atomicAdd(failed, errors);
}

__global__ void quantization_fp32_boundaries(unsigned* failed) {
  const unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= 245 * 128) return;
  const unsigned exponent = id / 128 + 1, mantissa = id % 128;
  const float scale = __uint_as_float(exponent << 23 | mantissa * 65537U);
  unsigned errors = 0;
  // Softmax P is FP32, not BF16. Check both sides and the nearest FP32
  // number at every E4M3 rounding midpoint, over varied scales.
  for (unsigned code = 0; code < 126; ++code) {
    const float lo = __half2float(static_cast<__half>(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3)));
    const float hi = __half2float(static_cast<__half>(__nv_cvt_fp8_to_halfraw(code + 1, __NV_E4M3)));
    const float center = ((lo + hi) * 0.5F) * scale;
    check_quantized_ratio(nextafterf(center, 0.0F), scale, errors);
    check_quantized_ratio(center, scale, errors);
    check_quantized_ratio(nextafterf(center, INFINITY), scale, errors);
  }
  if (errors) atomicAdd(failed, errors);
}

void quantization_rounding() {
  Device failed(4);
  check(cudaMemset(failed.get<void>(), 0, 4));
  quantization_bf16_scales<<<128, 256>>>(failed.get<unsigned>());
  quantization_fp32_boundaries<<<123, 256>>>(failed.get<unsigned>());
  require(failed.host() == std::vector<unsigned char>(4, 0), "FP8 reciprocal quantization differs from direct division");
  std::cout << "FP8 quantization: BF16 vector scales and FP32 rounding boundaries passed\n";
}

__global__ void compare_packed_loads(const unsigned char* records, unsigned count,
    unsigned width, unsigned split, unsigned stride, unsigned displacement,
    const unsigned* columns, unsigned column_count, uint4* actual, uint4* reference) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count * column_count) return;
  const auto* record = reinterpret_cast<const BF16*>(
      records + (i / column_count) * stride + displacement);
  const unsigned d = columns[i % column_count];
  actual[i] = gewell::kv_storage::load_eight(record, d, width, kv::Format::fp8, split);
  uint4 expected;
  auto* values = reinterpret_cast<BF16*>(&expected);
  for (unsigned j = 0; j < 8; ++j)
    values[j] = gewell::kv_storage::load(record, d + j, width, kv::Format::fp8, split);
  reference[i] = expected;
}

void packed_loads() {
  const std::vector<float> scales{0.0F, 1.0F, 0.03125F, 1.00003F,
      1e-38F, 1e-20F, 1e20F, std::numeric_limits<float>::max() / 512};
  const unsigned count = 256 * scales.size();
  for (const unsigned width : {256U, 640U}) {
    const unsigned split = width == 640 ? 128 : 0;
    const unsigned stride = kv::row_bytes(width, kv::Format::fp8, split ? 2 : 1) + 16;
    const std::vector<unsigned> columns{0, 1, 2, 4, 7, 8, 16, 120, 124, 127, 128, width - 8};
    Device positions(columns.size() * 4), records(count * stride);
    positions.upload(columns);
    Device actual(count * columns.size() * 16), reference(actual.bytes);
    for (unsigned displacement : {0U, 4U}) {
      std::vector<unsigned char> data(records.bytes, 0xa5);
      for (unsigned r = 0; r < count; ++r) {
        auto* record = data.data() + r * stride + displacement;
        for (unsigned d = 0; d < width; ++d) record[d] = (r + d * 73) & 255;
        const float scale = scales[r / 256];
        std::memcpy(record + width, &scale, 4);
        if (split) {
          const float other = scales[(r / 256 + 3) % scales.size()];
          std::memcpy(record + width + 4, &other, 4);
        }
      }
      records.upload(data);
      compare_packed_loads<<<(count * columns.size() + 255) / 256, 256>>>(
          records.get<unsigned char>(), count, width, split, stride, displacement,
          positions.get<unsigned>(), columns.size(), actual.get<uint4>(), reference.get<uint4>());
      require(actual.host() == reference.host(), "FP8 packed loads differ from scalar decoding");
    }
  }
  std::cout << "FP8 packed loads: all codes, scales, alignment and split boundaries passed\n";
}

struct Fixture {
  static constexpr unsigned Capacity = 5376, Pages = 21;
  unsigned Base;
  unsigned rows, query_heads, heads, d, width, capacity;
  bool global, paged;
  std::size_t stride, ref_stride;
  Device key_cache, value_cache, ref_key, ref_value, offsets, ref_offsets;
  Device query, key, value, norm, output, reference, scratch;
  Cache cache{}, ref{};
  p::CompactGlobalPagedCache page{}, ref_page{};
  std::vector<BF16> hq, hk, hv;

  Fixture(bool global, bool paged, unsigned rows, unsigned base = 1030, unsigned queries = 32)
      : Base(base), rows(rows), query_heads(queries), heads(global ? queries / 8 : queries / 2), d(global ? 512 : 256),
        width(global ? 640 : 256), capacity(global ? Capacity : 1024),
        global(global), paged(paged),
        stride(heads * 256 * kv::row_words(width, kv::Format::fp8, global ? 2 : 1) + 64),
        ref_stride(heads * 256 * width + 64),
        key_cache(paged ? (Pages * stride + 64) * 2 : heads * capacity * kv::row_bytes(width, kv::Format::fp8, global ? 2 : 1)),
        value_cache(global ? 2 : key_cache.bytes),
        ref_key(paged ? (Pages * ref_stride + 64) * 2 : heads * capacity * width * 2),
        ref_value(global ? 2 : ref_key.bytes), offsets(Pages * 8), ref_offsets(Pages * 8),
        query(queries * rows * d * 2), key(heads * rows * d * 2), value(key.bytes),
        norm(d * 2), output(query.bytes), reference(query.bytes),
        scratch(std::max(p::tensor_attention_scratch_bytes(rows),
                        a::scratch_bytes(std::min(rows, 1280U), Capacity))) {
    require(std::uint64_t(Base) + rows <= Capacity, "FP8 fixture exceeds fixed cache capacity");
    std::vector<std::uint64_t> ho(Pages), hr(Pages);
    for (unsigned i = 0; i < Pages; ++i) {
      ho[i] = (Pages - 1 - i) * stride + 32;
      hr[i] = (Pages - 1 - i) * ref_stride + 32;
    }
    offsets.upload(ho); ref_offsets.upload(hr);
    cache = {key_cache.get(), global ? nullptr : value_cache.get(), capacity,
        paged ? key_cache.get() : nullptr, paged ? offsets.get<std::uint64_t>() : nullptr,
        256, Pages, stride, 32, kv::Format::fp8};
    ref = {ref_key.get(), global ? nullptr : ref_value.get(), capacity,
        paged ? ref_key.get() : nullptr, paged ? ref_offsets.get<std::uint64_t>() : nullptr,
        256, Pages, ref_stride, 32};
    page = {cache.page_pool, cache.page_offsets, 256, Pages, stride, 32, cache.format};
    ref_page = {ref.page_pool, ref.page_offsets, 256, Pages, ref_stride, 32};
    std::vector<BF16> pk(heads * Base * d), pv(pk.size());
    for (unsigned h = 0; h < heads; ++h)
      for (unsigned t = 0; t < Base; ++t)
        for (unsigned dim = 0; dim < d; ++dim) {
          pk[(h * Base + t) * d + dim] = pattern(2, t, h, dim);
          pv[(t * heads + h) * d + dim] = pattern(3, t, h, dim);
        }
    std::vector<unsigned char> expected(key_cache.bytes, 0xa5), expected_v(value_cache.bytes, 0xa5);
    std::vector<BF16> restored(ref_key.bytes / 2, bf(NAN)), restored_v(ref_value.bytes / 2, bf(NAN));
    key_cache.upload(expected); value_cache.upload(expected_v);
    for (unsigned h = 0; h < heads; ++h) {
      for (unsigned t = global || Base < 1024 ? 0 : Base - 1024; t < Base; ++t) {
        const auto row = std::size_t(h) * capacity + (global ? t : t % 1024);
        const auto dst = paged ? 2 * (ho[t / 256] + 32) + (h * 256 + t % 256) * kv::row_bytes(width, kv::Format::fp8, 2)
                              : row * kv::row_bytes(width, kv::Format::fp8, global ? 2 : 1);
        const auto rd = paged ? hr[t / 256] + 32 + (h * 256 + t % 256) * width : row * width;
        std::vector<BF16> payload(width);
        if (global) {
          for (unsigned dim = 0; dim < 128; ++dim) payload[dim] = pk[(h * Base + t) * d + (dim < 64 ? dim : dim + 192)];
          std::copy_n(pv.data() + (t * heads + h) * d, d, payload.data() + 128);
          quantize(payload, 128, expected.data() + dst, restored.data() + rd);
        } else {
          std::copy_n(pk.data() + (h * Base + t) * d, d, payload.data());
          quantize(payload, d, expected.data() + dst, restored.data() + rd);
          std::copy_n(pv.data() + (t * heads + h) * d, d, payload.data());
          quantize(payload, d, expected_v.data() + dst, restored_v.data() + rd);
        }
      }
    }
    if (query_heads == 16) {
      key_cache.upload(expected); value_cache.upload(expected_v);
    } else if (Base) {
      Device dk(pk.size() * 2), dv(pv.size() * 2); dk.upload(pk); dv.upload(pv);
      if (paged) p::write_kv_cache_chunk_global_compact_paged(dk.get(), dv.get(), page, 0, Base);
      else if (global) p::write_kv_cache_chunk_global_compact(dk.get(), dv.get(), cache.key, 0, Base, capacity, nullptr, cache.format);
      else p::write_kv_cache_chunk(dk.get(), dv.get(), cache.key, cache.value, 0, Base, capacity, m::AttentionKind::local, nullptr, cache.format);
    }
    require(key_cache.host() == expected, "FP8 K/compact packing or guards differ from CPU oracle");
    if (!global) require(value_cache.host() == expected_v, "FP8 V packing or guards differ from CPU oracle");
    ref_key.upload(restored); ref_value.upload(restored_v);
    hq.resize(query_heads * rows * d); hk.resize(heads * rows * d); hv.resize(hk.size());
    for (unsigned h = 0; h < query_heads; ++h)
      for (unsigned r = 0; r < rows; ++r)
        for (unsigned dim = 0; dim < d; ++dim) hq[(h * rows + r) * d + dim] = pattern(1, Base + r, h, dim);
    for (unsigned h = 0; h < heads; ++h)
      for (unsigned r = 0; r < rows; ++r)
        for (unsigned dim = 0; dim < d; ++dim) {
          hk[(h * rows + r) * d + dim] = pattern(2, Base + r, h, dim);
          hv[(r * heads + h) * d + dim] = pattern(3, Base + r, h, dim);
        }
    query.upload(hq); key.upload(hk); value.upload(hv);
    std::vector<BF16> hn(d);
    for (unsigned dim = 0; dim < d; ++dim) hn[dim] = bf(0.75F + (dim % 31) / 64.0F);
    norm.upload(hn);
  }
  void equal(const char* label, std::size_t count = 0) {
    const auto x = output.host(), y = reference.host();
    require(!std::memcmp(x.data(), y.data(), count ? count * 2 : x.size()), label);
  }
  struct CachedRow {
    std::vector<float> key, value;
    float ks, vs;
  };
  CachedRow cached_fp8_row(unsigned head, unsigned pos,
      const std::vector<unsigned char>& raw_k, const std::vector<unsigned char>& raw_v) const {
    const auto offset = paged ? 2 * ((Pages - 1 - pos / 256) * stride + 64) +
        (head * 256 + pos % 256) * kv::row_bytes(width, kv::Format::fp8, 2)
        : (head * capacity + (global ? pos : pos % 1024)) * kv::row_bytes(width, kv::Format::fp8, global ? 2 : 1);
    const auto* k = raw_k.data() + offset;
    const auto* v = global ? k + 128 : raw_v.data() + offset;
    CachedRow row{std::vector<float>(d), std::vector<float>(d), 0, 0};
    float rotated_scale;
    std::memcpy(&rotated_scale, k + width, 4);
    std::memcpy(&row.vs, global ? k + width + 4 : v + width, 4);
    float norm_max = 0;
    if (global) for (unsigned dim = 0; dim < d; ++dim)
      if (!(dim < 64 || (dim >= 256 && dim < 320)))
        norm_max = std::max(norm_max, std::fabs(fp(bf(0.75F + (dim % 31) / 64.0F))));
    row.ks = global ? std::max(rotated_scale, row.vs * norm_max) : rotated_scale;
    const auto unpack = [](unsigned char x) { return std::copysign(decoded(x & 127), x & 128 ? -1.0F : 1.0F); };
    for (unsigned dim = 0; dim < d; ++dim) {
      row.value[dim] = unpack(v[dim]);
      if (global && !(dim < 64 || (dim >= 256 && dim < 320)))
        row.key[dim] = fp8_round((row.value[dim] * fp(bf(0.75F + (dim % 31) / 64.0F))) * (row.vs / row.ks));
      else {
        const auto index = global && dim >= 256 ? dim - 192 : dim;
        row.key[dim] = global ? fp8_round(unpack(k[index]) * (rotated_scale / row.ks)) : unpack(k[index]);
      }
    }
    return row;
  }

  // Independent CPU attention oracle: scaled E4M3 Q/K, per-channel V for
  // BF16 storage/local prefill, cached per-token V for compact FP8 storage,
  // rounded unnormalized E4M3 probabilities, causal/window masks, and online
  // accumulation across key tiles. Check multiple heads and query positions.
  void fp8_reference(bool cached = true) {
    const auto kr = ref_key.host(), vr = ref_value.host(), actual = cached ? output.host() : reference.host();
    const auto raw_k = key_cache.host(), raw_v = value_cache.host();
    const bool direct = global && cached;
    const auto* k = reinterpret_cast<const BF16*>(kr.data());
    const auto* v = reinterpret_cast<const BF16*>(vr.data());
    const auto* out = reinterpret_cast<const BF16*>(actual.data());
    for (unsigned head : {0U, 31U}) {
      const unsigned h = head / (32 / heads);
      for (unsigned r : {0U, rows / 2, rows - 1}) {
        const bool sliced = rows >= 768 && (!global || rows % 256 == 0);
        const unsigned tile_rows = 512;
        const unsigned start = sliced ? r / tile_rows * tile_rows : 0;
        const unsigned count = sliced ? std::min(tile_rows, rows - start) : rows;
        const unsigned begin = global || Base + start < 1023 ? 0 : Base + start - 1023;
        const unsigned end = Base + start + count;
        std::vector<float> q(d), numerator(d, 0);
        float qmax = 0;
        for (unsigned dim = 0; dim < d; ++dim) qmax = std::max(qmax, std::fabs(fp(hq[(head * rows + r) * d + dim])));
        const float qs = qmax > 0 ? qmax / 448 : 1;
        for (unsigned dim = 0; dim < d; ++dim) q[dim] = fp8_round(fp(hq[(head * rows + r) * d + dim]) / qs);
        float maximum = -INFINITY, denominator = 0;
        for (unsigned tile = begin; tile < end; tile += 1024) {
          const auto n = std::min(1024U, end - tile);
          std::vector<float> keys(n * d), values(n * d), ks(n), vs(d, 0), token_vs(n), scores(n, -INFINITY);
          for (unsigned t = 0; t < n; ++t) {
            const auto pos = tile + t;
            if (direct && pos < Base) {
              const auto row = cached_fp8_row(h, pos, raw_k, raw_v);
              ks[t] = row.ks; token_vs[t] = row.vs;
              std::copy(row.key.begin(), row.key.end(), keys.begin() + t * d);
              std::copy(row.value.begin(), row.value.end(), values.begin() + t * d);
              continue;
            }
            const auto offset = paged ? (Pages - 1 - pos / 256) * ref_stride + 64 + (h * 256 + pos % 256) * width
                : (h * capacity + (global ? pos : pos % 1024)) * width;
            for (unsigned dim = 0; dim < d; ++dim) {
              const float value = pos >= Base ? fp(hv[((pos - Base) * heads + h) * d + dim])
                  : fp(global ? k[offset + 128 + dim] : v[offset + dim]);
              float key;
              if (global && !(dim < 64 || (dim >= 256 && dim < 320)))
                key = direct ? value * fp(bf(0.75F + (dim % 31) / 64.0F))
                    : fp(bf(value * fp(bf(0.75F + (dim % 31) / 64.0F))));
              else key = pos >= Base ? fp(hk[(h * rows + pos - Base) * d + dim])
                  : fp(k[offset + (global && dim >= 256 ? dim - 192 : dim)]);
              keys[t * d + dim] = key; values[t * d + dim] = value;
              ks[t] = std::max(ks[t], std::fabs(key));
              vs[dim] = std::max(vs[dim], std::fabs(value));
              token_vs[t] = std::max(token_vs[t], std::fabs(value));
            }
            if (direct) {
              ks[t] = ks[t] > 0 ? ks[t] / 448 : 1;
              token_vs[t] = token_vs[t] > 0 ? token_vs[t] / 448 : 1;
              for (unsigned dim = 0; dim < d; ++dim) {
                keys[t * d + dim] = fp8_round(keys[t * d + dim] / ks[t]);
                values[t * d + dim] = fp8_round(values[t * d + dim] / token_vs[t]);
              }
            }
          }
          if (!direct) for (auto& x : ks) x = x > 0 ? x / 448 : 1;
          for (auto& x : vs) x = x > 0 ? x / 448 : 1;
          if (direct) std::fill(vs.begin(), vs.end(), *std::max_element(token_vs.begin(), token_vs.end()));
          float tile_maximum = -INFINITY;
          for (unsigned t = 0; t < n; ++t) {
            float dot = 0;
            for (unsigned dim = 0; dim < d; ++dim) {
              dot = std::fma(q[dim], direct ? keys[t * d + dim] : fp8_round(keys[t * d + dim] / ks[t]), dot);
              if (!direct) values[t * d + dim] = fp8_round(values[t * d + dim] / vs[dim]);
            }
            if (tile + t <= Base + r && (global || Base + r - (tile + t) < 1024))
              scores[t] = dot * (qs * ks[t]);
            tile_maximum = std::max(tile_maximum, scores[t]);
          }
          const float next_maximum = std::max(maximum, tile_maximum);
          const float old_scale = std::isinf(maximum) ? 0 : std::exp(maximum - next_maximum);
          for (auto& x : numerator) x *= old_scale;
          denominator *= old_scale;
          std::vector<float> partial(d, 0);
          for (unsigned t = 0; t < n; ++t) {
            const float exponential = std::isinf(scores[t]) ? 0 : std::exp(scores[t] - next_maximum);
            const float weight = fp8_round(direct ? exponential * token_vs[t] / (vs[0] / 256) : exponential * 256);
            denominator += direct ? exponential : weight / 256;
            for (unsigned dim = 0; dim < d; ++dim) partial[dim] = std::fma(weight, values[t * d + dim], partial[dim]);
          }
          for (unsigned dim = 0; dim < d; ++dim) numerator[dim] += partial[dim] * (vs[dim] / 256);
          maximum = next_maximum;
        }
        for (unsigned dim = 0; dim < d; ++dim) {
          const float expected = fp(bf(numerator[dim] / denominator));
          const float observed = fp(out[(r * 32 + head) * d + dim]);
          if (!std::isfinite(observed) || std::fabs(expected - observed) > 0.00004F + std::fabs(expected) * 0.012F)
            throw std::runtime_error("FP8 CPU attention oracle mismatch rows=" + std::to_string(rows) + " head=" + std::to_string(head) + " query=" + std::to_string(r) + " dim=" + std::to_string(dim) + " expected=" + std::to_string(expected) + " actual=" + std::to_string(observed));
        }
      }
    }
  }
  // Independent small-query oracle, including the split reduction and FP8
  // P*V scale folding. Unlike prefill, V is per-token/head, never scaled using
  // future rows. Decode raw bytes independently of the device loaders.
  void fp8_decode_reference(bool frozen, bool cached = true) {
    const auto kr = ref_key.host(), vr = ref_value.host(), actual = cached ? output.host() : reference.host();
    const auto raw_k = key_cache.host(), raw_v = value_cache.host();
    const auto* k = reinterpret_cast<const BF16*>(kr.data());
    const auto* v = reinterpret_cast<const BF16*>(vr.data());
    const auto* out = reinterpret_cast<const BF16*>(actual.data());
    const unsigned visible = global ? Base + (frozen ? 0 : rows) : std::min(1024U, Base + rows);
    unsigned limit = std::min(256U, (visible + 31) / 32);
    if (global && Base + rows < 16384) limit = std::min(limit, 128U / std::min(rows, 8U));
    const unsigned group_rows = global ? (frozen ? 2 : 4) : 8;
    const auto scale = [](float x) { return x > 0 ? x / 448 : 1; };
    for (unsigned head : {0U, query_heads / 2 + 1, query_heads - 1}) {
      const unsigned h = head / (query_heads / heads);
      for (unsigned r : {0U, rows / 2, rows - 1}) {
        // Shared launch descriptors carry at most 32 eight-row query tiles.
        const unsigned first_tile = r / 256 * 256;
        std::uint64_t work = 0;
        for (unsigned first = first_tile; first < std::min(rows, first_tile + 256); first += 8) {
          const unsigned count = std::min(8U, rows - first), end = Base + (frozen ? 0 : first + count);
          work += std::uint64_t((count + group_rows - 1) / group_rows) * (global ? end : std::min(1024U, end));
        }
        const unsigned tile_end = Base + (frozen ? 0 : std::min(rows, (r / 8 + 1) * 8));
        const unsigned tile_visible = global ? tile_end : std::min(1024U, tile_end);
        const unsigned budget = ((global ? 128U : 64U) * std::uint64_t(tile_visible) + work - 1) / work;
        const unsigned splits = rows <= 8 ? limit : std::min(limit, std::max(1U, budget));
        const unsigned first_row = r / group_rows * group_rows;
        const unsigned position = frozen ? Base - 1 : Base + r;
        const unsigned group_position = frozen ? position : Base + first_row;
        const unsigned begin = global || group_position < 1023 ? 0 : group_position - 1023;
        const unsigned last = frozen ? position : Base + std::min(rows, first_row + group_rows) - 1;
        std::vector<float> q(d), part(splits * d), maxima(splits, -INFINITY), sums(splits, 0);
        float qmax = 0;
        for (unsigned dim = 0; dim < d; ++dim) qmax = std::max(qmax, std::fabs(fp(hq[(head * rows + r) * d + dim])));
        const float qs = scale(qmax);
        for (unsigned dim = 0; dim < d; ++dim) q[dim] = fp8_round(fp(hq[(head * rows + r) * d + dim]) / qs);
        for (unsigned split = 0; split < splits; ++split) {
          for (unsigned tile = begin + split * 32; tile <= last; tile += splits * 32) {
            std::vector<float> values(32 * d), scores(32, -INFINITY), vs(32, 1), probabilities(32);
            float tile_max = -INFINITY;
            for (unsigned t = 0; t < 32; ++t) {
              const unsigned pos = tile + t;
              if (pos > position || (!global && position - pos >= 1024)) continue;
              if (cached && pos < Base) {
                const auto row = cached_fp8_row(h, pos, raw_k, raw_v);
                float dot = 0;
                for (unsigned dim = 0; dim < d; ++dim) dot = std::fma(q[dim], row.key[dim], dot);
                scores[t] = (dot * qs) * row.ks; vs[t] = row.vs;
                std::copy(row.value.begin(), row.value.end(), values.begin() + t * d);
                tile_max = std::max(tile_max, scores[t]);
                continue;
              }
              const auto offset = paged ? (Pages - 1 - pos / 256) * ref_stride + 64 + (h * 256 + pos % 256) * width
                  : (h * capacity + (global ? pos : pos % 1024)) * width;
              std::vector<float> keys(d);
              float kmax = 0, vmax = 0;
              for (unsigned dim = 0; dim < d; ++dim) {
                const float value = !frozen && pos >= Base ? fp(hv[((pos - Base) * heads + h) * d + dim])
                    : fp(global ? k[offset + 128 + dim] : v[offset + dim]);
                float key;
                if (global && !(dim < 64 || (dim >= 256 && dim < 320)))
                  key = fp(bf(value * fp(bf(0.75F + (dim % 31) / 64.0F))));
                else key = !frozen && pos >= Base ? fp(hk[(h * rows + pos - Base) * d + dim])
                    : fp(k[offset + (global && dim >= 256 ? dim - 192 : dim)]);
                keys[dim] = key; values[t * d + dim] = value;
                kmax = std::max(kmax, std::fabs(key)); vmax = std::max(vmax, std::fabs(value));
              }
              const float ks = scale(kmax); vs[t] = scale(vmax);
              float dot = 0;
              for (unsigned dim = 0; dim < d; ++dim) {
                dot = std::fma(q[dim], fp8_round(keys[dim] / ks), dot);
                values[t * d + dim] = fp8_round(values[t * d + dim] / vs[t]);
              }
              scores[t] = (dot * qs) * ks;
              tile_max = std::max(tile_max, scores[t]);
            }
            const float next = std::max(maxima[split], tile_max);
            const float old = std::isinf(maxima[split]) ? 0 : std::exp(maxima[split] - next);
            float ps = 0, sum = 0;
            for (unsigned t = 0; t < 32; ++t) {
              const float p = std::isinf(scores[t]) ? 0 : std::exp(scores[t] - next);
              probabilities[t] = p * vs[t]; sum += p;
              ps = std::max(ps, probabilities[t]);
            }
            ps = scale(ps);
            for (auto& p : probabilities) p = fp8_round(p / ps);
            for (unsigned dim = 0; dim < d; ++dim) {
              float pv = 0;
              for (unsigned t = 0; t < 32; ++t) pv = std::fma(probabilities[t], values[t * d + dim], pv);
              part[split * d + dim] = std::fma(part[split * d + dim], old, pv * ps);
            }
            sums[split] = std::fma(sums[split], old, sum); maxima[split] = next;
          }
        }
        const float maximum = *std::max_element(maxima.begin(), maxima.end());
        float denominator = 0;
        for (unsigned split = 0; split < splits; ++split) denominator += std::exp(maxima[split] - maximum) * sums[split];
        for (unsigned dim = 0; dim < d; ++dim) {
          float numerator = 0;
          for (unsigned split = 0; split < splits; ++split)
            numerator = std::fma(std::exp(maxima[split] - maximum), part[split * d + dim], numerator);
          const float expected = fp(bf(numerator / denominator)), observed = fp(out[(r * query_heads + head) * d + dim]);
          if (!std::isfinite(observed) || std::fabs(expected - observed) > 0.00004F + std::fabs(expected) * 0.012F)
            throw std::runtime_error("FP8 decode oracle mismatch global=" + std::to_string(global) + " frozen=" + std::to_string(frozen) + " rows=" + std::to_string(rows) + " head=" + std::to_string(head) + " row=" + std::to_string(r) + " dim=" + std::to_string(dim) + " expected=" + std::to_string(expected) + " actual=" + std::to_string(observed));
        }
      }
    }
  }

  void fp8_decode_compute() {
    const auto kind = global ? m::AttentionKind::global : m::AttentionKind::local;
    const auto original_k = key_cache.host(), original_v = value_cache.host();
    const auto bytes = a::fp8_scratch_bytes(query_heads, rows, Capacity);
    Device arena(bytes + 64);
    arena.upload(std::vector<unsigned char>(arena.bytes, 0xa5));
    for (bool frozen : {false, true}) {
      if (frozen && (rows != 1 || !Base)) continue;
      const auto input = [&](const Cache& c, BF16* out) {
        return a::BatchInput{query.get(), frozen ? nullptr : key.get(), frozen ? nullptr : value.get(), c, Base, rows, out};
      };
      a::run_fp8_batch(query_heads, {input(cache, output.get())}, norm.get(), kind, arena.get<void>(), bytes, nullptr, frozen);
      fp8_decode_reference(frozen);
      a::run_fp8_batch(query_heads, {input(ref, reference.get())}, norm.get(), kind, arena.get<void>(), bytes, nullptr, frozen);
      fp8_decode_reference(frozen, false);
      if (query_heads == 16 && !frozen && (!global || paged || !Base)) {
        // Qualify the 26B token-major layout adapter against the separately
        // checked head-major FP8 path for both stored-cache formats.
        std::vector<BF16> qt(hq.size()), kt(hk.size());
        for (unsigned r = 0; r < rows; ++r) {
          for (unsigned h = 0; h < query_heads; ++h)
            std::copy_n(hq.data() + (h * rows + r) * d, d, qt.data() + (r * query_heads + h) * d);
          for (unsigned h = 0; h < heads; ++h)
            std::copy_n(hk.data() + (h * rows + r) * d, d, kt.data() + (r * heads + h) * d);
        }
        Device dq(query.bytes), dk(key.bytes), result(output.bytes); dq.upload(qt); dk.upload(kt);
        using Compute = gewell::attention::Compute;
        gewell::gemma4_26b_a4b::sm120::AttentionWorkspace workspace(rows,
            global ? Compute::bf16 : Compute::fp8, global ? Compute::fp8 : Compute::bf16);
        cublasHandle_t handle{}; require(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "create layout adapter handle");
        for (bool packed : {false, true}) {
          workspace.run(handle, dq.get(), dk.get(), value.get(), norm.get(), packed ? cache : ref,
                        global, Base, rows, result.get());
          require(result.host() == (packed ? output.host() : reference.host()), "26B FP8 token-major adapter changed output");
        }
        cublasDestroy(handle);
      }
      const auto original = output.host();
      for (unsigned count : (rows > 256 ? std::vector<unsigned>{2U} : std::vector<unsigned>{2U, 33U})) {
        Device actual(count * output.bytes), control(actual.bytes);
        std::vector<a::BatchInput> inputs, controls;
        for (unsigned i = 0; i < count; ++i) {
          inputs.push_back(input(i % 2 ? ref : cache, actual.get() + i * hq.size()));
          controls.push_back(input(i % 2 ? ref : cache, control.get() + i * hq.size()));
        }
        cudaStream_t stream; cudaGraph_t graph; cudaGraphExec_t executable;
        check(cudaStreamCreate(&stream)); check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        a::run_fp8_batch(query_heads, inputs, norm.get(), kind, arena.get<void>(), bytes, stream, frozen);
        check(cudaStreamEndCapture(stream, &graph)); check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        check(cudaGraphLaunch(executable, stream)); check(cudaStreamSynchronize(stream));
        a::run_fp8_batch(query_heads, controls, norm.get(), kind, arena.get<void>(), bytes, stream, frozen);
        check(cudaStreamSynchronize(stream));
        require(actual.host() == control.host(), "FP8 mixed-cache graph differs from eager execution");
        check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph)); check(cudaStreamDestroy(stream));
      }
      if (!frozen && rows > 1) {
        auto poisoned = hv;
        std::fill(poisoned.end() - heads * d, poisoned.end(), bf(NAN));
        value.upload(poisoned);
        a::run_fp8_batch(query_heads, {input(cache, output.get())}, norm.get(), kind, arena.get<void>(), bytes, nullptr);
        auto changed = output.host();
        require(std::equal(original.begin(), original.begin() + (rows - 1) * query_heads * d * 2, changed.begin()), "FP8 future values changed earlier queries");
        value.upload(hv);
      }
    }
    const auto guard = arena.host();
    require(std::all_of(guard.end() - 64, guard.end(), [](auto x) { return x == 0xa5; }), "FP8 attention scratch overrun");
    require(original_k == key_cache.host() && original_v == value_cache.host(), "FP8 compute changed KV");
    std::cout << "FP8 decode/verification/frozen compute query_heads=" << query_heads << " global=" << global << " paged=" << paged << " rows=" << rows << " base=" << Base << " passed\n";
  }

  void mtp_batch(bool misaligned = false) {
    constexpr unsigned Count = 37;
    const auto kind = global ? m::AttentionKind::global : m::AttentionKind::local;
    const unsigned displacement = misaligned ? 1 : 0;
    Device queries(Count * query.bytes + displacement * 2), actual(Count * output.bytes), expected(actual.bytes);
    Device shifted_norm(norm.bytes + 2);
    check(cudaMemcpy(shifted_norm.get() + 1, norm.get(), norm.bytes, cudaMemcpyDeviceToDevice));
    const auto* norm_values = misaligned ? shifted_norm.get() + 1 : norm.get();
    Device arena(scratch.bytes + 64);
    check(cudaMemset(arena.get<void>(), 0xa5, arena.bytes));
    std::vector<BF16> hqueries(queries.bytes / 2);
    const auto original_k = key_cache.host(), original_v = value_cache.host();
    const auto original_ref_k = ref_key.host(), original_ref_v = ref_value.host();
    cudaStream_t stream;
    check(cudaStreamCreate(&stream));
    for (bool frozen : {false, true}) {
      if (frozen && (!global || rows != 1)) continue;
      for (bool mixed : {false, true}) {
        std::vector<a::BatchInput> inputs, controls;
        for (unsigned i = 0; i < Count; ++i) {
          const bool bf16 = mixed && i % 7 >= 4;
          const auto* q = queries.get() + displacement + i * query.bytes / 2;
          inputs.push_back({q, frozen ? nullptr : key.get(), frozen ? nullptr : value.get(),
              bf16 ? ref : cache, Base, rows, actual.get() + i * output.bytes / 2});
          // Complement the formats so both launches have identical split/group
          // geometry while every FP8 result is checked against CPU-decoded KV.
          controls.push_back({q, frozen ? nullptr : key.get(), frozen ? nullptr : value.get(),
              bf16 ? cache : ref, Base, rows, expected.get() + i * output.bytes / 2});
        }
        const auto invoke = [&](const auto& batch, cudaStream_t s) {
          if (frozen) a::run_frozen_global_batch(batch, norm_values, arena.get<void>(), scratch.bytes, s);
          else a::run_batch(batch, norm_values, kind, arena.get<void>(), scratch.bytes, s);
        };
        for (unsigned i = 0; i < Count; ++i)
          for (unsigned h = 0; h < 32; ++h)
            for (unsigned r = 0; r < rows; ++r)
              for (unsigned dim = 0; dim < d; ++dim)
                hqueries[displacement + (std::size_t(i) * 32 * rows + h * rows + r) * d + dim] =
                    pattern(1, Base + r + i * 13, h, dim);
        queries.upload(hqueries);
        // Capture the first invocation too, including the dynamic shared-memory
        // opt-in for the asynchronously staged compact global kernels.
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        invoke(inputs, stream);
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&executable, graph, 0));
        check(cudaGraphLaunch(executable, stream));
        invoke(controls, stream);
        check(cudaStreamSynchronize(stream));
        require(actual.host() == expected.host(), "FP8 MTP batch differs from decoded KV");
        for (unsigned replay = 0; replay < 2; ++replay) {
          for (auto& x : hqueries) x = bf(fp(x) * -0.75F + 0.00390625F);
          queries.upload(hqueries);
          check(cudaGraphLaunch(executable, stream));
          invoke(controls, stream);
          check(cudaStreamSynchronize(stream));
          require(actual.host() == expected.host(), "FP8 MTP mixed-format graph replay differs from decoded KV");
        }
        const auto before = actual.host();
        for (unsigned i = 0; i < query.bytes / 2; ++i)
          hqueries[displacement + i] = bf(fp(hqueries[displacement + i]) * -2.0F + 0.125F);
        queries.upload(hqueries);
        check(cudaGraphLaunch(executable, stream));
        check(cudaStreamSynchronize(stream));
        const auto after = actual.host();
        require(std::equal(before.begin() + output.bytes, before.end(),
                           after.begin() + output.bytes), "FP8 MTP request isolation failed");
        check(cudaGraphExecDestroy(executable));
        check(cudaGraphDestroy(graph));
      }
    }
    check(cudaStreamDestroy(stream));
    const auto guard = arena.host();
    require(std::all_of(guard.end() - 64, guard.end(), [](auto x) { return x == 0xa5; }),
            "FP8 MTP batch scratch overrun");
    require(key_cache.host() == original_k && value_cache.host() == original_v &&
            ref_key.host() == original_ref_k && ref_value.host() == original_ref_v,
            "FP8 MTP batch mutated committed KV");
  }
  void decode_batch() {
    if (rows != 1 || (global && !paged)) return;
    const auto kind = global ? m::AttentionKind::global : m::AttentionKind::local;
    Device cosine(d * 2), sine(d * 2), qr(query.bytes), kr(key.bytes);
    std::vector<BF16> cos(d, bf(0.75F)), sin(d, bf(0.25F));
    cosine.upload(cos); sine.upload(sin);
    const auto original_k = key_cache.host(), original_v = value_cache.host();
    b::apply_rope_m1(query.get(), cosine.get(), sine.get(), qr.get(), 32, kind);
    b::apply_rope_m1(key.get(), cosine.get(), sine.get(), kr.get(), heads, kind);
    if (global) {
      p::write_kv_cache_chunk_global_compact_paged(kr.get(), value.get(), page, Base, 1);
      b::causal_gqa_attention_cached_m1_fused_global_compact_paged(qr.get(), page, norm.get(), Base, scratch.get<void>(), reference.get());
    } else {
      b::write_kv_cache_m1(kr.get(), value.get(), cache.key, cache.value, Base, capacity, kind, nullptr, cache.format);
      b::causal_gqa_attention_cached_m1_fused(qr.get(), cache.key, cache.value, Base, capacity, scratch.get<void>(), reference.get(), kind, nullptr, cache.format);
    }
    const auto expected_k = key_cache.host(), expected_v = value_cache.host();
    key_cache.upload(original_k); value_cache.upload(original_v);
    b::DecodeAttentionInput input{query.get(), key.get(), value.get(), cosine.get(), sine.get(), qr.get(),
        global ? nullptr : cache.key, global ? nullptr : cache.value, page, Base, output.get(), cache.format};
    cudaStream_t stream; cudaGraph_t graph; cudaGraphExec_t executable;
    check(cudaStreamCreate(&stream));
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    b::decode_attention_batch({input}, norm.get(), scratch.get<void>(), scratch.bytes, kind, stream);
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, 0));
    check(cudaGraphLaunch(executable, stream));
    check(cudaStreamSynchronize(stream));
    equal("FP8 batched decode differs from serial reference");
    require(key_cache.host() == expected_k && value_cache.host() == expected_v, "FP8 batched RoPE/cache write differs from serial");
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph)); check(cudaStreamDestroy(stream));
    // Later read-only checks still describe the prefix before this commit.
    key_cache.upload(original_k); value_cache.upload(original_v);
  }
  void fp8_compute() {
    const auto kind = global ? m::AttentionKind::global : m::AttentionKind::local;
    cublasHandle_t handle;
    require(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cuBLAS create");
    p::Fp8Attention fp8(rows);
    const auto invoke = [&](unsigned control, cudaStream_t stream) {
      const auto& c = control ? ref : cache;
      auto* out = control ? reference.get() : output.get();
      if (paged) p::causal_gqa_attention_cached_chunk_tensor_global_compact_paged(handle, query.get(), key.get(), value.get(), control ? ref_page : page, norm.get(), Base, rows, scratch.get<void>(), out, stream, &fp8);
      else if (global) p::causal_gqa_attention_cached_chunk_tensor_global_compact(handle, query.get(), key.get(), value.get(), c.key, norm.get(), Base, rows, capacity, scratch.get<void>(), out, stream, c.format, &fp8);
      else p::causal_gqa_attention_cached_chunk_tensor(handle, query.get(), key.get(), value.get(), c.key, c.value, Base, rows, capacity, scratch.get<void>(), out, kind, stream, c.format, &fp8);
    };
    for (unsigned control = 0; control < 2; ++control) invoke(control, nullptr);
    if (!global) equal("Native FP8 local attention differs between packed storage and CPU decoded storage");
    fp8_reference();
    fp8_reference(false);
    if (global && rows == 128) {
      // Plans are warmed above. Replay the wide PV path with changed Q/K/V,
      // including changing dynamic scales and accumulation across key tiles.
      cudaStream_t stream; cudaGraph_t graph; cudaGraphExec_t executable;
      check(cudaStreamCreate(&stream));
      check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      invoke(0, stream);
      check(cudaStreamEndCapture(stream, &graph));
      check(cudaGraphInstantiate(&executable, graph, 0));
      for (unsigned iteration = 0; iteration < 3; ++iteration) {
        for (auto& x : hq) x = bf(fp(x) * 0.5F - 0.03125F);
        for (auto& x : hk) x = bf(fp(x) * 1.25F);
        for (auto& x : hv) x = bf(-fp(x) * 0.75F + 0.015625F);
        query.upload(hq); key.upload(hk); value.upload(hv);
        check(cudaGraphLaunch(executable, stream));
        check(cudaStreamSynchronize(stream));
        fp8_reference();
      }
      check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
      check(cudaStreamDestroy(stream));
      std::cout << "FP8 wide global PV changed-input graph replay CPU oracle passed\n";
    }
    cublasDestroy(handle);
  }
  void bf16_prefill() {
    const auto kind = global ? m::AttentionKind::global : m::AttentionKind::local;
    const auto scratch_bytes = p::tensor_attention_scratch_bytes(rows);
    Device guarded_scratch(scratch_bytes + 512);
    check(cudaMemset(guarded_scratch.get<void>(), 0xa5, guarded_scratch.bytes));
    auto* workspace = guarded_scratch.get<unsigned char>() + 256;
    const auto original_k = key_cache.host(), original_v = value_cache.host();
    const auto original_ref_k = ref_key.host(), original_ref_v = ref_value.host();
    cublasHandle_t handle;
    require(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cuBLAS create");
    const auto invoke = [&](bool control, cudaStream_t stream) {
      const auto& c = control ? ref : cache;
      auto* out = control ? reference.get() : output.get();
      if (paged) p::causal_gqa_attention_cached_chunk_tensor_global_compact_paged(handle,
          query.get(), key.get(), value.get(), control ? ref_page : page, norm.get(),
          Base, rows, workspace, out, stream);
      else if (global) p::causal_gqa_attention_cached_chunk_tensor_global_compact(handle,
          query.get(), key.get(), value.get(), c.key, norm.get(), Base, rows,
          capacity, workspace, out, stream, c.format);
      else p::causal_gqa_attention_cached_chunk_tensor(handle, query.get(), key.get(),
          value.get(), c.key, c.value, Base, rows, capacity, workspace, out,
          kind, stream, c.format);
    };
    check(cudaMemset(workspace, 0xff, scratch_bytes));
    invoke(false, nullptr);
    invoke(true, nullptr);
    equal("BF16 ragged prefill differs from CPU-decoded FP8 KV");
    cudaStream_t stream; cudaGraph_t graph; cudaGraphExec_t executable;
    check(cudaStreamCreate(&stream));
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    invoke(false, stream);
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, 0));
    for (unsigned iteration = 0; iteration < 2; ++iteration) {
      for (auto& x : hq) x = bf(fp(x) * 0.5F - 0.03125F);
      for (auto& x : hk) x = bf(fp(x) * 1.25F);
      for (auto& x : hv) x = bf(-fp(x) * 0.75F + 0.015625F);
      query.upload(hq); key.upload(hk); value.upload(hv);
      invoke(true, nullptr);
      check(cudaDeviceSynchronize());
      check(cudaMemset(workspace, 0xff, scratch_bytes));
      check(cudaGraphLaunch(executable, stream));
      check(cudaStreamSynchronize(stream));
      equal("BF16 ragged prefill graph differs from CPU-decoded FP8 KV");
      std::vector<BF16> host(hq.size());
      check(cudaMemcpy(host.data(), output.get(), output.bytes, cudaMemcpyDeviceToHost));
      require(std::all_of(host.begin(), host.end(), [](auto x) { return std::isfinite(fp(x)); }),
              "BF16 ragged prefill produced nonfinite output");
    }
    const auto guard = guarded_scratch.host();
    require(std::all_of(guard.begin(), guard.begin() + 256, [](auto x) { return x == 0xa5; }) &&
            std::all_of(guard.end() - 256, guard.end(), [](auto x) { return x == 0xa5; }),
            "BF16 ragged prefill scratch guard changed");
    require(key_cache.host() == original_k && value_cache.host() == original_v &&
            ref_key.host() == original_ref_k && ref_value.host() == original_ref_v,
            "BF16 ragged prefill mutated committed KV");
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
    check(cudaStreamDestroy(stream)); cublasDestroy(handle);
  }
  void run() {
    const auto kind = global ? m::AttentionKind::global : m::AttentionKind::local;
    cublasHandle_t handle;
    require(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cuBLAS create");
    for (bool tensor : {false, true}) {
      for (unsigned control = 0; control < 2; ++control) {
        const auto& c = control ? ref : cache;
        const auto& pg = control ? ref_page : page;
        auto* out = control ? reference.get() : output.get();
        if (tensor) {
          if (paged) p::causal_gqa_attention_cached_chunk_tensor_global_compact_paged(handle, query.get(), key.get(), value.get(), pg, norm.get(), Base, rows, scratch.get<void>(), out);
          else if (global) p::causal_gqa_attention_cached_chunk_tensor_global_compact(handle, query.get(), key.get(), value.get(), c.key, norm.get(), Base, rows, capacity, scratch.get<void>(), out, nullptr, c.format);
          else p::causal_gqa_attention_cached_chunk_tensor(handle, query.get(), key.get(), value.get(), c.key, c.value, Base, rows, capacity, scratch.get<void>(), out, kind, nullptr, c.format);
        } else {
          if (paged) p::causal_gqa_attention_cached_chunk_global_compact_paged(query.get(), key.get(), value.get(), pg, norm.get(), Base, rows, out);
          else if (global) p::causal_gqa_attention_cached_chunk_global_compact(query.get(), key.get(), value.get(), c.key, norm.get(), Base, rows, capacity, out, nullptr, c.format);
          else p::causal_gqa_attention_cached_chunk(query.get(), key.get(), value.get(), c.key, c.value, Base, rows, capacity, out, kind, nullptr, c.format);
        }
      }
      equal(tensor ? "FP8 tensor prefill differs from decoded reference" : "FP8 scalar prefill differs from decoded reference");
    }
    fp8_compute();
    for (unsigned control = 0; control < 2; ++control) {
      const auto& c = control ? ref : cache;
      auto* out = control ? reference.get() : output.get();
      if (paged) p::image_block_gqa_attention_cached_chunk_global_compact_paged(query.get(), key.get(), value.get(), control ? ref_page : page, norm.get(), Base, rows, Base, Base + rows, out);
      else if (global) p::image_block_gqa_attention_cached_chunk_global_compact(query.get(), key.get(), value.get(), c.key, norm.get(), Base, rows, capacity, Base, Base + rows, out, nullptr, c.format);
      else p::image_block_gqa_attention_cached_chunk(query.get(), key.get(), value.get(), c.key, c.value, Base, rows, capacity, Base, Base + rows, out, kind, nullptr, c.format);
    }
    equal("FP8 image attention differs from decoded reference");
    std::vector<BF16> image_reference(output.bytes / sizeof(BF16));
    check(cudaMemcpy(image_reference.data(), output.get(), output.bytes, cudaMemcpyDeviceToHost));
    for (unsigned control = 0; control < 2; ++control) {
      const auto& c = control ? ref : cache;
      auto* out = control ? reference.get() : output.get();
      if (paged) p::causal_gqa_attention_cached_chunk_tensor_global_compact_paged(handle,
          query.get(), key.get(), value.get(), control ? ref_page : page, norm.get(),
          Base, rows, scratch.get<void>(), out);
      else if (global) p::causal_gqa_attention_cached_chunk_tensor_global_compact(handle,
          query.get(), key.get(), value.get(), c.key, norm.get(), Base, rows, capacity,
          scratch.get<void>(), out, nullptr, c.format);
      else p::causal_gqa_attention_cached_chunk_tensor(handle, query.get(), key.get(), value.get(),
          c.key, c.value, Base, rows, capacity, scratch.get<void>(), out, kind, nullptr,
          c.format, nullptr, Base, Base + rows);
    }
    equal("FP8 image tensor attention differs from decoded reference");
    std::vector<BF16> image_tensor(image_reference.size());
    check(cudaMemcpy(image_tensor.data(), output.get(), output.bytes, cudaMemcpyDeviceToHost));
    // Tensor attention rounds both the softmax numerator weights and their
    // denominator to BF16 (unit roundoff 1/256). With |V| <= 16, their
    // combined error is bounded by 2*16/255, plus output rounding. Also bound
    // vector error so cancellation near zero cannot hide broad disagreement.
    double error_squared = 0, reference_squared = 0;
    float maximum_error = 0;
    for (std::size_t i = 0; i < image_reference.size(); ++i) {
      const float error = fp(image_tensor[i]) - fp(image_reference[i]);
      error_squared += double(error) * error;
      reference_squared += double(fp(image_reference[i])) * fp(image_reference[i]);
      maximum_error = std::max(maximum_error, std::abs(error));
      require(std::isfinite(fp(image_tensor[i])) &&
              std::abs(error) <= 32.0F / 255 + std::abs(fp(image_reference[i])) / 128,
              ("Image tensor attention differs from the nonuniform scalar reference: global=" +
              std::to_string(global) + " paged=" + std::to_string(paged) + " rows=" + std::to_string(rows) +
              " index=" + std::to_string(i) + " actual=" + std::to_string(fp(image_tensor[i])) +
              " expected=" + std::to_string(fp(image_reference[i]))).c_str());
    }
    const double relative_l2 = std::sqrt(error_squared / reference_squared);
    require(relative_l2 <= 1.0 / 128, "Image tensor attention relative L2 exceeds BF16 tolerance");
    std::cout << "Image tensor vs scalar global=" << global << " paged=" << paged
              << " rows=" << rows << " relative_l2=" << relative_l2
              << " max_abs=" << maximum_error << '\n';
    cublasDestroy(handle);
    for (unsigned control = 0; control < 2; ++control)
      a::run(query.get(), key.get(), value.get(), control ? ref : cache, norm.get(), Base, rows,
          kind, control ? reference.get() : output.get(), scratch.get<void>(), scratch.bytes, nullptr);
    equal("FP8 MTP attention differs from decoded reference");
    std::vector<BF16> one(32 * d);
    for (unsigned h = 0; h < 32; ++h) std::copy_n(hq.data() + h * rows * d, d, one.data() + h * d);
    Device q(one.size() * 2); q.upload(one);
    for (unsigned control = 0; control < 2; ++control) {
      const auto& c = control ? ref : cache;
      auto* out = control ? reference.get() : output.get();
      if (paged) b::causal_gqa_attention_cached_m1_fused_global_compact_paged(q.get(), control ? ref_page : page, norm.get(), Base - 1, scratch.get<void>(), out);
      else if (global) b::causal_gqa_attention_cached_m1_fused_global_compact(q.get(), c.key, norm.get(), Base - 1, capacity, scratch.get<void>(), out, nullptr, c.format);
      else b::causal_gqa_attention_cached_m1_fused(q.get(), c.key, c.value, Base - 1, capacity, scratch.get<void>(), out, kind, nullptr, c.format);
    }
    equal("FP8 decode differs from decoded reference", one.size());
    if (!global) {
      b::causal_gqa_attention_cached_m1_fused_local_batch(
          {{q.get(), cache.key, cache.value, Base - 1, output.get(), cache.format}},
          scratch.get<void>(), scratch.bytes);
      equal("FP8 frozen local batch differs from decoded reference", one.size());
    }
    if (global) {
      for (unsigned control = 0; control < 2; ++control)
        a::run_frozen_global(q.get(), control ? ref : cache, norm.get(), Base,
            control ? reference.get() : output.get(), scratch.get<void>(), scratch.bytes, nullptr);
      equal("FP8 frozen global attention differs from decoded reference", one.size());
    }
  }
};

// Each layer owns disjoint records; rejected rows and unwritten slots stay poisoned.
void accepted_commits(bool paged) {
  namespace t = gewell::mtp_target;
  constexpr unsigned Base = 1021, Source = 9, Capacity = 16, Accepted = 5;
  constexpr auto LocalWords = 16 * 1024 * kv::row_words(256, kv::Format::fp8);
  constexpr auto PageWords = 4 * 256 * kv::row_words(640, kv::Format::fp8, 2);
  constexpr auto GlobalWords = 5 * PageWords;
  Device staging(t::Verifier::staging_bytes(Capacity)), local(50 * 2 * LocalWords * 2), global(10 * GlobalWords * 2), offsets(5 * 8);
  std::vector<std::uint64_t> pages(5);
  for (unsigned i = 0; i < 5; ++i) pages[i] = (4 - i) * 10 * PageWords;
  offsets.upload(pages);
  std::vector<BF16> hs(staging.bytes / 2, bf(NAN));
  std::vector<unsigned char> expected_l(local.bytes, 0xa5), expected_g(global.bytes, 0xa5);
  t::Caches caches{};
  std::size_t stage = 0;
  unsigned local_layer = 0;
  for (unsigned layer = 0; layer < 60; ++layer) {
    const bool full = m::is_global_layer(layer);
    const unsigned heads = full ? 4 : 16, d = full ? 512 : 256;
    const auto value_offset = stage + std::size_t(Capacity) * heads * d;
    auto& c = caches[layer];
    c.format = kv::Format::fp8;
    c.capacity = full ? 1280 : 1024;
    if (full) {
      c.key = global.get() + (layer / 6) * GlobalWords;
      if (paged) {
        c.page_pool = global.get(); c.page_offsets = offsets.get<std::uint64_t>();
        c.page_count = 5; c.page_tokens = 256; c.page_stride_elements = 10 * PageWords;
        c.layer_offset_elements = (layer / 6) * PageWords;
      }
    } else {
      c.key = local.get() + local_layer * 2 * LocalWords;
      c.value = c.key + LocalWords;
    }
    for (unsigned h = 0; h < heads; ++h) {
      for (unsigned row = 0; row < Source; ++row) {
        for (unsigned dim = 0; dim < d; ++dim) {
          hs[stage + (h * Source + row) * d + dim] = pattern(2 + layer, row, h, dim);
          hs[value_offset + (row * heads + h) * d + dim] = pattern(3 + layer, row, h, dim);
        }
        if (row >= Accepted) continue;
        const unsigned position = Base + row;
        if (full) {
          const auto offset = paged ? pages[position / 256] + c.layer_offset_elements + (h * 256 + position % 256) * kv::row_words(640, c.format, 2)
              : (layer / 6) * GlobalWords + (h * 1280 + position) * kv::row_words(640, c.format, 2);
          std::vector<BF16> payload(640), decoded_row(640);
          for (unsigned dim = 0; dim < 128; ++dim) payload[dim] = hs[stage + (h * Source + row) * d + (dim < 64 ? dim : dim + 192)];
          std::copy_n(hs.data() + value_offset + (row * heads + h) * d, d, payload.data() + 128);
          quantize(payload, 128, expected_g.data() + 2 * offset, decoded_row.data());
        } else {
          for (unsigned kind = 0; kind < 2; ++kind) {
            const auto offset = (local_layer * 2 + kind) * LocalWords + (h * 1024 + position % 1024) * kv::row_words(256, c.format);
            const auto src = kind ? value_offset + (row * heads + h) * d : stage + (h * Source + row) * d;
            std::vector<BF16> payload(hs.begin() + src, hs.begin() + src + d), decoded_row(d);
            quantize(payload, d, expected_l.data() + 2 * offset, decoded_row.data());
          }
        }
      }
    }
    stage += 2 * std::size_t(Capacity) * heads * d;
    if (!full) ++local_layer;
  }
  staging.upload(hs);
  for (bool batch : {false, true}) {
    check(cudaMemset(local.get<void>(), 0xa5, local.bytes));
    check(cudaMemset(global.get<void>(), 0xa5, global.bytes));
    if (batch) t::commit_staged_rows_batch({{&caches, Base, Source, Capacity, staging.get<void>(), staging.bytes, Accepted}}, nullptr);
    else t::commit_staged_rows(caches, Base, Source, Capacity, staging.get<void>(), staging.bytes, Accepted, nullptr);
    require(local.host() == expected_l, "FP8 accepted local commit or untouched suffix differs from CPU oracle");
    require(global.host() == expected_g, "FP8 accepted global commit or untouched suffix differs from CPU oracle");
  }
  std::cout << "FP8 accepted serial/batched commits paged=" << paged << " passed\n";
}

__global__ void fill_storage(unsigned* data, std::size_t words) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < words) data[i] = unsigned(i * 2654435761U);
}
__global__ void check_storage(const unsigned* data, std::size_t words, unsigned* failed) {
  auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < words && data[i] != unsigned(i * 2654435761U)) atomicOr(failed, 1U);
}
void checkpoint_copies(kv::CompactGeometry geometry, kv::Format format) {
  namespace s = gewell::gemma4_31b::sm120;
  const auto config = kv::compact_pool_config(geometry,1536ULL << 20,512ULL << 20,
      64ULL << 20,format,format);
  const bool moe = geometry.layers == 30;
  const auto expected_local = moe ? (format == kv::Format::bf16 ? 209715200ULL : 111411200ULL) : 445644800ULL;
  const auto expected_global = moe ? (format == kv::Format::bf16 ? 3276800ULL : 1679360ULL) : 6717440ULL;
  require(config.local_ring_bytes == expected_local && config.global_page_bytes == expected_global,
      "physical cache accounting");
  kv::CacheLedger ledger(config);
  gewell::runtime::PhysicalCache storage(geometry, config, ledger, 1024);
  const auto bytes = config.local_ring_bytes;
  kv::ExecutionInfo source{}, destination{};
  source.processed_tokens = 1030;
  source.local_ring = ledger.try_allocate_gpu(bytes);
  destination.local_ring = ledger.try_allocate_gpu(bytes);
  const auto snapshot = ledger.try_allocate_gpu(bytes), cold = ledger.try_allocate_cpu(bytes);
  auto* src = static_cast<unsigned*>(storage.device_pointer(source.local_ring));
  auto* dst = static_cast<unsigned*>(storage.device_pointer(destination.local_ring));
  const auto words = bytes / 4;
  Device failed(4);
  fill_storage<<<(words + 255) / 256, 256>>>(src, words);
  for (unsigned mode = 0; mode < 3; ++mode) {
    check(cudaMemset(dst, 0, bytes)); check(cudaMemset(failed.get<void>(), 0, 4));
    if (mode == 0) storage.fork_local(source, destination, {});
    else {
      storage.copy_local_to_snapshot(source, 1024, snapshot, {});
      storage.synchronize({}, "test snapshot");
      if (mode == 2) storage.spill(snapshot, cold, bytes, "test spill");
      storage.restore_local(destination, mode == 2 ? cold : snapshot, 6, 1024);
    }
    check_storage<<<(words + 255) / 256, 256>>>(dst, words, failed.get<unsigned>());
    require(failed.host() == std::vector<unsigned char>(4, 0), "FP8 fork or checkpoint restore changed payload/scales");
  }
  const auto page_a = ledger.try_allocate_gpu(config.global_page_bytes);
  const auto page_b = ledger.try_allocate_gpu(config.global_page_bytes);
  const auto page_cpu = ledger.try_allocate_cpu(config.global_page_bytes);
  auto* page_source = static_cast<unsigned*>(storage.device_pointer(page_a));
  auto* page_destination = static_cast<unsigned*>(storage.device_pointer(page_b));
  const auto page_words = config.global_page_bytes/4;
  fill_storage<<<(page_words+255)/256,256>>>(page_source,page_words);
  storage.copy_global_page(page_a,page_b,{});
  check(cudaMemset(failed.get<void>(),0,4));
  check_storage<<<(page_words+255)/256,256>>>(page_destination,page_words,failed.get<unsigned>());
  require(failed.host() == std::vector<unsigned char>(4,0),"global page copy changed bytes");
  storage.spill(page_a,page_cpu,config.global_page_bytes,"test global spill");
  check(cudaMemset(page_destination,0,config.global_page_bytes));
  storage.restore_global_page(page_cpu,page_b,{});
  check_storage<<<(page_words+255)/256,256>>>(page_destination,page_words,failed.get<unsigned>());
  require(failed.host() == std::vector<unsigned char>(4,0),"global page restore changed bytes");
  std::cout << "cache layers=" << geometry.layers << " format=" << kv::format_name(format)
            << " ring fork, GPU snapshot, CPU spill/restore and global page copies passed\n";
}

// A kernel diagnostic, not a model/quality benchmark. Every request has its
// own FP8 cache (larger than L2 at B32), with no allocations in timed graphs.
__global__ void benchmark_cache(unsigned char* data, unsigned width, unsigned stride,
                                std::size_t count) {
  const auto row = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= count) return;
  auto* record = data + row * stride;
  for (unsigned d = 0; d < width; ++d)
    record[d] = 32 + ((row * 13 + d * 7) % 80) | ((row + d) % 2 ? 128 : 0);
  auto* scales = reinterpret_cast<float*>(record + width);
  scales[0] = 0.0005F * (1 + row % 7);
  if (width == 640) scales[1] = 0.0005F * (1 + row % 5);
}

void benchmark_decode_compute(unsigned base, unsigned rows, bool global) {
  constexpr unsigned Count = 32;
  require(base > 0 && base <= 65536 && rows > 0 && rows <= 8, "benchmark geometry");
  const unsigned d = global ? 512 : 256, heads = global ? 4 : 16;
  const unsigned width = global ? 640 : 256, pages = (base + rows + 255) / 256;
  const unsigned capacity = global ? pages * 256 : 1024;
  const auto stride = kv::row_bytes(width, kv::Format::fp8, global ? 2 : 1);
  const std::size_t cache_bytes = heads * capacity * stride;
  Device keys(Count * cache_bytes), values(global ? 2 : keys.bytes), offsets(Count * pages * 8);
  Device query(Count * 32 * rows * d * 2), staged_key(Count * heads * rows * d * 2);
  Device staged_value(staged_key.bytes), norm(d * 2), output(query.bytes);
  Device scratch(a::scratch_bytes(8, 262144));
  const auto records = std::size_t(Count) * heads * capacity;
  benchmark_cache<<<(records + 255) / 256, 256>>>(keys.get<unsigned char>(), width, stride, records);
  if (!global) benchmark_cache<<<(records + 255) / 256, 256>>>(values.get<unsigned char>(), width, stride, records);
  std::vector<std::uint64_t> ho(Count * pages);
  for (unsigned i = 0; i < ho.size(); ++i) ho[i] = std::uint64_t(i) * heads * 256 * stride / 2;
  offsets.upload(ho);
  std::vector<BF16> hq(query.bytes / 2), hk(staged_key.bytes / 2), hn(d);
  for (unsigned i = 0; i < hq.size(); ++i) hq[i] = pattern(1, base + i / d, i / (rows * d), i % d);
  for (unsigned i = 0; i < hk.size(); ++i) hk[i] = pattern(2, base + i / d, i / (rows * d), i % d);
  for (unsigned i = 0; i < d; ++i) hn[i] = bf(0.75F + (i % 31) / 64.0F);
  query.upload(hq); staged_key.upload(hk); staged_value.upload(hk); norm.upload(hn);
  const bool frozen = global && rows == 1;
  const auto kind = global ? m::AttentionKind::global : m::AttentionKind::local;
  std::vector<a::BatchInput> inputs;
  for (unsigned i = 0; i < Count; ++i) {
    Cache cache{keys.get() + i * cache_bytes / 2,
        global ? nullptr : values.get() + i * cache_bytes / 2, capacity,
        global ? keys.get() : nullptr, global ? offsets.get<std::uint64_t>() + i * pages : nullptr,
        256, pages, heads * 256 * stride / 2, 0, kv::Format::fp8};
    inputs.push_back({query.get() + i * 32 * rows * d,
        frozen ? nullptr : staged_key.get() + i * heads * rows * d,
        frozen ? nullptr : staged_value.get() + i * heads * rows * d,
        cache, base, rows, output.get() + i * 32 * rows * d});
  }
  cudaStream_t stream; cudaEvent_t start, end;
  check(cudaStreamCreate(&stream)); check(cudaEventCreate(&start)); check(cudaEventCreate(&end));
  for (bool fp8 : {false, true}) {
    const auto run = [&] {
      if (fp8) a::run_fp8_batch(32, inputs, norm.get(), kind, scratch.get<void>(), scratch.bytes, stream, frozen);
      else if (frozen) a::run_frozen_global_batch(inputs, norm.get(), scratch.get<void>(), scratch.bytes, stream);
      else a::run_batch(inputs, norm.get(), kind, scratch.get<void>(), scratch.bytes, stream);
    };
    run(); check(cudaStreamSynchronize(stream));
    cudaGraph_t graph; cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (unsigned i = 0; i < 20; ++i) run();
    check(cudaStreamEndCapture(stream, &graph)); check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    check(cudaGraphLaunch(executable, stream)); check(cudaStreamSynchronize(stream));
    std::vector<float> times;
    for (unsigned sample = 0; sample < 5; ++sample) {
      check(cudaEventRecord(start, stream));
      for (unsigned repeat = 0; repeat < 5; ++repeat) check(cudaGraphLaunch(executable, stream));
      check(cudaEventRecord(end, stream)); check(cudaEventSynchronize(end));
      float ms; check(cudaEventElapsedTime(&ms, start, end)); times.push_back(ms / 100);
    }
    std::sort(times.begin(), times.end());
    const auto result = output.host();
    for (std::size_t i = 0; i < result.size() / 2; ++i)
      require(std::isfinite(fp(reinterpret_cast<const BF16*>(result.data())[i])), "benchmark nonfinite output");
    std::cout << std::setprecision(8) << "attention_benchmark batch=" << Count << " base=" << base
              << " rows=" << rows << " global=" << global << " frozen=" << frozen
              << " compute=" << (fp8 ? "fp8" : "bf16") << " median_ms=" << times[2] << std::endl;
    check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
  }
  check(cudaEventDestroy(start)); check(cudaEventDestroy(end)); check(cudaStreamDestroy(stream));
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--checkpoint-copies") {
      checkpoint_copies(gewell::gemma4_31b::sm120::kCacheGeometry,kv::Format::fp8);
      checkpoint_copies(gewell::gemma4_26b_a4b::sm120::kCacheGeometry,kv::Format::bf16);
      checkpoint_copies(gewell::gemma4_26b_a4b::sm120::kCacheGeometry,kv::Format::fp8);
      return 0;
    }
    if (argc == 2 && (std::string(argv[1]) == "--local-verify" ||
                      std::string(argv[1]) == "--local-verify-sanitize")) {
      const bool sanitize = std::string(argv[1]) == "--local-verify-sanitize";
      const std::vector<unsigned> bases = sanitize ? std::vector<unsigned>{1030, 0}
          : std::vector<unsigned>{0, 1, 31, 257, 1023, 1024, 1030, 4095};
      const std::vector<unsigned> counts = sanitize ? std::vector<unsigned>{4, 9}
          : std::vector<unsigned>{2, 4, 8, 9};
      for (unsigned base : bases)
        for (unsigned rows : counts) {
          Fixture fixture(false, false, rows, base);
          if (!sanitize) fixture.mtp_batch();
          fixture.mtp_batch(true);
          std::cout << "Local verification batch base=" << base << " rows=" << rows
                    << " exact=passed graph=passed guards=passed isolation=passed misaligned=passed\n";
        }
      return 0;
    }
    if (argc == 2 && (std::string(argv[1]) == "--compact-global" ||
                      std::string(argv[1]) == "--compact-global-sanitize")) {
      const bool sanitize = std::string(argv[1]) == "--compact-global-sanitize";
      const std::vector<unsigned> bases = sanitize ? std::vector<unsigned>{257, 1030}
          : std::vector<unsigned>{31, 32, 33, 255, 256, 257, 1030, 4095};
      for (unsigned base : bases)
        for (unsigned rows : {1U, 4U})
          for (bool paged : {false, true}) {
            Fixture fixture(true, paged, rows, base);
            if (!sanitize) fixture.mtp_batch();
            fixture.mtp_batch(true);
            std::cout << "Compact global batch base=" << base << " rows=" << rows
                      << " paged=" << paged << " exact=passed graph=passed guards=passed isolation=passed misaligned=passed\n";
          }
      return 0;
    }
    if (argc == 5 && std::string(argv[1]) == "--benchmark-decode-compute") {
      require(std::string(argv[4]) == "global" || std::string(argv[4]) == "local", "benchmark attention kind");
      benchmark_decode_compute(std::stoul(argv[2]), std::stoul(argv[3]), std::string(argv[4]) == "global");
      return 0;
    }
    if (argc == 2 && (std::string(argv[1]) == "--decode-compute-26b" ||
                      std::string(argv[1]) == "--decode-compute-26b-sanitize" ||
                      std::string(argv[1]) == "--decode-compute-26b-wide")) {
      const bool sanitize = std::string(argv[1]) == "--decode-compute-26b-sanitize";
      const bool wide = std::string(argv[1]) == "--decode-compute-26b-wide";
      if (wide) {
        for (unsigned rows : {257U, 1280U, 4096U}) for (bool global : {false, true}) {
          Fixture fixture(global, global, rows, 0, 16); fixture.fp8_decode_compute();
        }
      } else {
        for (unsigned base : (sanitize ? std::vector<unsigned>{1030U} : std::vector<unsigned>{0U, 1U, 1030U, 4095U}))
          for (unsigned rows : (sanitize ? std::vector<unsigned>{1U, 3U} : std::vector<unsigned>{1U, 3U, 8U, 17U}))
            for (unsigned layout = 0; layout < 3; ++layout) {
              Fixture fixture(layout != 0, layout == 2, rows, base, 16); fixture.fp8_decode_compute();
            }
      }
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--decode-compute") {
      quantization_rounding();
      for (unsigned base : {0U, 1U, 1030U, 4095U})
        for (unsigned rows : {1U, 3U, 8U, 17U})
          for (unsigned layout = 0; layout < 3; ++layout) {
            Fixture fixture(layout != 0, layout == 2, rows, base);
            fixture.fp8_decode_compute();
          }
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--decode-compute-sanitize") {
      for (unsigned rows : {1U, 3U})
        for (unsigned layout = 0; layout < 3; ++layout) {
          Fixture fixture(layout != 0, layout == 2, rows);
          fixture.fp8_decode_compute();
        }
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--prefill-compute-sanitize") {
      for (unsigned base : {0U, 1030U})
        for (unsigned rows : {3U, 129U})
          for (bool paged : {false, true}) {
            Fixture fixture(true, paged, rows, base);
            fixture.fp8_compute();
          }
      return 0;
    }
    quantization_rounding();
    packed_loads();
    // Exercise short cached suffixes with packed storage and an independently
    // decoded BF16 cache, including the local-window boundary.
    for (unsigned base : {16U, 543U, 1023U})
      for (unsigned rows : {1U, 17U, 31U})
        for (unsigned layout = 0; layout < 3; ++layout) {
          Fixture fixture(layout != 0, layout == 2, rows, base);
          fixture.run();
        }
    for (unsigned rows : {1U, 4U, 8U})
      for (unsigned layout = 0; layout < 3; ++layout) {
        Fixture fixture(layout != 0, layout == 2, rows);
        fixture.run();
        fixture.mtp_batch();
        if ((layout && rows == 1) || (!layout && rows > 1)) fixture.mtp_batch(true);
        fixture.decode_batch();
        fixture.fp8_decode_compute();
        std::cout << "FP8 KV layout=" << layout << " rows=" << rows << " passed\n";
      }
    for (unsigned rows : {7U, 127U, 128U, 257U, 768U, 769U, 1021U, 4096U})
      for (bool global : {false, true}) {
        Fixture fixture(global, global, rows);
        fixture.fp8_compute();
        std::cout << "FP8 compute CPU oracle global=" << global << " rows=" << rows << " passed\n";
      }
    for (unsigned rows : {1U, 31U, 513U})
      for (bool global : {false, true}) {
        Fixture fixture(global, global, rows, 1);
        fixture.fp8_compute();
        std::cout << "FP8 cold compute CPU oracle global=" << global << " rows=" << rows << " passed\n";
      }
    for (unsigned rows : {34U, 257U, 769U, 973U, 1023U, 1024U})
      for (unsigned layout = 0; layout < 3; ++layout) {
        Fixture fixture(layout != 0, layout == 2, rows);
        fixture.bf16_prefill();
        std::cout << "BF16 ragged prefill / FP8 KV graph layout=" << layout
                  << " rows=" << rows << " passed\n";
      }
    accepted_commits(false);
    accepted_commits(true);
    checkpoint_copies(gewell::gemma4_31b::sm120::kCacheGeometry,kv::Format::fp8);
    checkpoint_copies(gewell::gemma4_26b_a4b::sm120::kCacheGeometry,kv::Format::bf16);
    checkpoint_copies(gewell::gemma4_26b_a4b::sm120::kCacheGeometry,kv::Format::fp8);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
