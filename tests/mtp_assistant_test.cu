#include "gewell/mtp_assistant.h"
#include "gewell/mtp_attention.h"

#include <cuda_runtime.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using gewell::mtp_assistant::BFloat16;
using gewell::mtp_assistant::FrozenCache;
using gewell::mtp_assistant::attend_frozen_prefix;
using gewell::gemma4_31b::AttentionKind;

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <class T> class Device {
 public:
  explicit Device(std::size_t size) : size_(size) {
    check(cudaMalloc(reinterpret_cast<void**>(&data_), size * sizeof(T)));
  }
  explicit Device(const std::vector<T>& data) : Device(data.size()) {
    check(cudaMemcpy(data_, data.data(), size_ * sizeof(T), cudaMemcpyHostToDevice));
  }
  ~Device() { cudaFree(data_); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  T* get() const { return data_; }
  std::vector<T> host() const {
    std::vector<T> result(size_);
    check(cudaMemcpy(result.data(), data_, size_ * sizeof(T), cudaMemcpyDeviceToHost));
    return result;
  }
 private:
  T* data_{};
  std::size_t size_;
};

void check_frozen_batch(unsigned heads, const BFloat16* query,
                        const gewell::kv_cache::DeviceView& cache,
                        const BFloat16* norm, unsigned length, AttentionKind kind) {
  namespace attention = gewell::mtp_attention;
  constexpr unsigned count = 33, guard = 16;
  const unsigned width = heads * (kind == AttentionKind::global ? 512 : 256);
  const auto bytes = attention::frozen_scratch_bytes(heads, length);
  Device<unsigned char> scratch(bytes * 32 + 64);
  Device<BFloat16> expected(count * width);
  std::vector<BFloat16> sentinel(count * width + 2 * guard, __float2bfloat16_rn(-123.f));
  Device<BFloat16> actual(sentinel);
  std::vector<attention::BatchInput> inputs;
  for (unsigned row = 0; row < count; ++row) {
    const unsigned base = row % 2 ? std::min(32U, length) : length;
    attention::run_frozen_prefix(query, cache, norm, base, heads, kind,
        expected.get() + row * width, scratch.get(), bytes, nullptr);
    inputs.push_back({query, nullptr, nullptr, cache, base, 1,
        actual.get() + guard + row * width});
  }
  const auto reference = expected.host();
  const auto check_output = [&] {
    const auto result = actual.host();
    if (std::memcmp(reference.data(), result.data() + guard, reference.size() * 2) ||
        std::memcmp(sentinel.data(), result.data(), guard * 2) ||
        std::memcmp(sentinel.data(), result.data() + guard + reference.size(), guard * 2))
      throw std::runtime_error("frozen batch differs from scalar or overwrites guards");
  };
  for (auto capacity : {bytes, bytes * 32}) {
    check(cudaMemset(scratch.get() + capacity, 0xa5, 64));
    attention::run_frozen_prefix_batch(heads, inputs, norm, kind,
        scratch.get(), capacity, nullptr);
    check_output();
    unsigned char boundary[64];
    check(cudaMemcpy(boundary, scratch.get() + capacity, 64, cudaMemcpyDeviceToHost));
    if (!std::all_of(std::begin(boundary), std::end(boundary),
          [](unsigned char value) { return value == 0xa5; }))
      throw std::runtime_error("frozen batch wrote beyond declared scratch capacity");
  }
  cudaStream_t stream{}; cudaGraph_t graph{}; cudaGraphExec_t executable{};
  check(cudaStreamCreate(&stream));
  check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  attention::run_frozen_prefix_batch(heads, inputs, norm, kind,
      scratch.get(), bytes * 32, stream);
  check(cudaStreamEndCapture(stream, &graph));
  check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  check(cudaGraphLaunch(executable, stream));
  check(cudaStreamSynchronize(stream));
  check_output();
  check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
  check(cudaStreamDestroy(stream));
  check(cudaMemcpy(actual.get(), sentinel.data(), sentinel.size() * 2, cudaMemcpyHostToDevice));
  inputs.back().rows = 2;
  bool rejected = false;
  try { attention::run_frozen_prefix_batch(heads, inputs, norm, kind,
      scratch.get(), bytes * 32, nullptr); }
  catch (const std::invalid_argument&) { rejected = true; }
  const auto unchanged = actual.host();
  if (!rejected || std::memcmp(unchanged.data(), sentinel.data(), sentinel.size() * 2))
    throw std::runtime_error("frozen batch malformed later input changed output");
}

void attend(unsigned heads, const BFloat16* query, const FrozenCache& frozen,
            const BFloat16* norm, AttentionKind kind, void* scratch,
            BFloat16* output) {
  if (heads == 32) {
    attend_frozen_prefix(query, frozen, norm, kind, scratch, output);
  }
  gewell::kv_cache::DeviceView cache{};
  if (kind == AttentionKind::local) {
    cache.key = const_cast<BFloat16*>(frozen.local_key);
    cache.value = const_cast<BFloat16*>(frozen.local_value);
    cache.capacity = 1024;
  } else {
    cache.key = const_cast<BFloat16*>(frozen.global_compact);
    cache.capacity = frozen.global_capacity;
    cache.page_pool = frozen.global.page_pool;
    cache.page_offsets = const_cast<std::uint64_t*>(frozen.global.page_offsets);
    cache.page_tokens = frozen.global.page_tokens;
    cache.page_count = frozen.global.page_count;
    cache.page_stride_elements = frozen.global.page_stride_elements;
    cache.layer_offset_elements = frozen.global.layer_offset_elements;
  }
  const auto previous = heads == 32 ? [&] {
    std::vector<BFloat16> result(32 * (kind == AttentionKind::global ? 512 : 256));
    check(cudaMemcpy(result.data(), output, result.size() * 2, cudaMemcpyDeviceToHost));
    return result;
  }() : std::vector<BFloat16>{};
  gewell::mtp_attention::run_frozen_prefix(query, cache, norm,
      frozen.processed_tokens, heads, kind, output, scratch,
      gewell::mtp_attention::frozen_scratch_bytes(heads, frozen.processed_tokens), nullptr);
  check_frozen_batch(heads, query, cache, norm, frozen.processed_tokens, kind);
  if (heads == 32 && kind == AttentionKind::global) {
    std::vector<BFloat16> actual(previous.size());
    check(cudaMemcpy(actual.data(), output, actual.size() * 2, cudaMemcpyDeviceToHost));
    if (std::memcmp(actual.data(), previous.data(), actual.size() * 2))
      throw std::runtime_error("new32head global specialization differs from existing path");
  }
}

float value(BFloat16 x) { return __bfloat162float(x); }
BFloat16 bf16(float x) { return __float2bfloat16_rn(x); }

float compare(const std::vector<BFloat16>& actual,
              const std::vector<BFloat16>& expected, float tolerance) {
  if (actual.size() != expected.size()) throw std::runtime_error("output size mismatch");
  float error = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const float delta = std::abs(value(actual[i]) - value(expected[i]));
    if (!std::isfinite(delta)) throw std::runtime_error("non-finite attention output");
    error = std::max(error, delta);
  }
  if (error > tolerance) throw std::runtime_error("attention CPU error " + std::to_string(error));
  return error;
}

std::vector<BFloat16> query(unsigned width, unsigned heads) {
  std::vector<BFloat16> result(heads * width);
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = bf16((static_cast<int>((i * 13 + 3) % 97) - 48) / 256.0F);
  return result;
}

template <class Key, class Value>
std::vector<BFloat16> reference(const std::vector<BFloat16>& q,
                                unsigned width, unsigned kv_heads,
                                unsigned begin, unsigned end,
                                Key key, Value v) {
  std::vector<BFloat16> output(q.size());
  std::vector<double> scores(end - begin);
  std::vector<double> sum(width);
  for (unsigned h = 0; h < q.size() / width; ++h) {
    const unsigned kv = h / (q.size() / width / kv_heads);
    double maximum = -std::numeric_limits<double>::infinity();
    for (unsigned t = begin; t < end; ++t) {
      double score = 0;
      for (unsigned d = 0; d < width; ++d)
        score += static_cast<double>(value(q[h * width + d])) * key(kv, t, d);
      scores[t - begin] = score;
      maximum = std::max(maximum, score);
    }
    std::fill(sum.begin(), sum.end(), 0.0);
    double denominator = 0;
    for (unsigned t = begin; t < end; ++t) {
      const double probability = std::exp(scores[t - begin] - maximum);
      denominator += probability;
      for (unsigned d = 0; d < width; ++d) sum[d] += probability * v(kv, t, d);
    }
    for (unsigned d = 0; d < width; ++d)
      output[h * width + d] = bf16(static_cast<float>(sum[d] / denominator));
  }
  return output;
}

void local(unsigned length, unsigned heads) {
  std::vector<BFloat16> k((heads / 2) * 1024 * 256), v(k.size());
  for (std::size_t i = 0; i < k.size(); ++i) {
    k[i] = bf16((static_cast<int>((i * 17 + 11) % 251) - 125) / 256.0F);
    v[i] = bf16((static_cast<int>((i * 7 + 43) % 239) - 119) / 128.0F);
  }
  const auto q = query(256, heads);
  Device<BFloat16> dk(k), dv(v), dq(q), out(q.size());
  Device<unsigned char> scratch(heads == 32 ? gewell::mtp_assistant::attention_scratch_bytes(length)
      : gewell::mtp_attention::frozen_scratch_bytes(heads, length));
  FrozenCache cache;
  cache.local_key = dk.get(); cache.local_value = dv.get(); cache.processed_tokens = length;
  attend(heads, dq.get(), cache, nullptr, AttentionKind::local, scratch.get(), out.get());
  const auto separate = out.host();
  const auto read = [](const auto& data, unsigned head, unsigned token, unsigned dimension) {
    return value(data[(static_cast<std::size_t>(head) * 1024 + token % 1024) * 256 + dimension]);
  };
  const auto expected = reference(q, 256, heads / 2, length > 1024 ? length - 1024 : 0, length,
      [&](unsigned h, unsigned t, unsigned d) { return read(k, h, t, d); },
      [&](unsigned h, unsigned t, unsigned d) { return read(v, h, t, d); });
  const float error = compare(separate, expected, 0.001953125F);
  if (std::memcmp(dk.host().data(), k.data(), k.size() * sizeof(BFloat16)) ||
      std::memcmp(dv.host().data(), v.data(), v.size() * sizeof(BFloat16)))
    throw std::runtime_error("assistant wrote frozen local cache");
  std::cout << "assistant local heads=" << heads << " prefix=" << length << " cpu_max_abs=" << error
            << " cache_unchanged=1\n";
}

void global(unsigned length, unsigned heads) {
  const unsigned capacity = length + 19;
  std::vector<BFloat16> compact(static_cast<std::size_t>(heads / 8) * capacity * 640), scale(512);
  for (unsigned d = 0; d < 512; ++d) scale[d] = bf16(0.5F + (d % 29) / 32.0F);
  for (std::size_t i = 0; i < compact.size(); ++i)
    compact[i] = bf16((static_cast<int>((i * 23 + 19) % 223) - 111) / 256.0F);
  const unsigned pages = (length + 255) / 256;
  const std::size_t layer_offset = 41;
  const std::size_t page_stride = layer_offset + (heads / 8) * 256 * 640 + 17;
  std::vector<BFloat16> pool(pages * page_stride + 29, bf16(17));
  std::vector<std::uint64_t> offsets(pages);
  for (unsigned p = 0; p < pages; ++p) offsets[p] = (pages - p - 1) * page_stride + 29;
  for (unsigned t = 0; t < length; ++t)
    for (unsigned h = 0; h < heads / 8; ++h)
      std::copy_n(compact.data() + (static_cast<std::size_t>(h) * capacity + t) * 640, 640,
          pool.data() + offsets[t / 256] + layer_offset + (h * 256 + t % 256) * 640);
  const auto q = query(512, heads);
  Device<BFloat16> dc(compact), ds(scale), dp(pool), dq(q), out(q.size());
  Device<std::uint64_t> offsets_device(offsets);
  Device<unsigned char> scratch(heads == 32 ? gewell::mtp_assistant::attention_scratch_bytes(length)
      : gewell::mtp_attention::frozen_scratch_bytes(heads, length));
  FrozenCache cache;
  cache.processed_tokens = length; cache.global_compact = dc.get(); cache.global_capacity = capacity;
  attend(heads, dq.get(), cache, ds.get(), AttentionKind::global, scratch.get(), out.get());
  const auto contiguous = out.host();
  const auto row = [&](unsigned h, unsigned t) {
    return compact.data() + (static_cast<std::size_t>(h) * capacity + t) * 640;
  };
  const auto expected = reference(q, 512, heads / 8, 0, length,
      [&](unsigned h, unsigned t, unsigned d) {
        const auto* r = row(h, t);
        if (d < 64) return value(r[d]);
        if (d >= 256 && d < 320) return value(r[d - 256 + 64]);
        return value(bf16(value(r[128 + d]) * value(scale[d])));
      }, [&](unsigned h, unsigned t, unsigned d) { return value(row(h, t)[128 + d]); });
  const float error = compare(contiguous, expected, 0.001953125F);
  cache.global_compact = nullptr; cache.global_capacity = 0;
  cache.global = {dp.get(), offsets_device.get(), 256, pages, page_stride, layer_offset};
  attend(heads, dq.get(), cache, ds.get(), AttentionKind::global, scratch.get(), out.get());
  compare(out.host(), contiguous, 0);
  if (std::memcmp(dp.host().data(), pool.data(), pool.size() * sizeof(BFloat16)))
    throw std::runtime_error("assistant wrote frozen global cache");
  std::cout << "assistant global heads=" << heads << " prefix=" << length << " cpu_max_abs=" << error
            << " contiguous_paged_exact=1 noncontiguous_pages=1 cache_unchanged=1\n";
}

// Independent host E4M3 records and decoded BF16 values. The CPU attention
// reference consumes decoded values, never a production device storage helper.
std::vector<unsigned char> fp8_records(std::vector<BFloat16>& values, unsigned width, unsigned split) {
  const unsigned scales = split < width ? 2 : 1;
  const unsigned stride = (width + scales * 4 + 15) / 16 * 16;
  std::vector<unsigned char> bytes(values.size() / width * stride);
  for (std::size_t row = 0; row < values.size() / width; ++row) {
    float scale[2]{};
    for (unsigned d = 0; d < width; ++d)
      scale[d >= split] = std::max(scale[d >= split], std::abs(value(values[row * width + d])));
    for (auto& s : scale) s = s ? s / 448.f : 1.f;
    for (unsigned d = 0; d < width; ++d) {
      __nv_fp8_e4m3 code(value(values[row * width + d]) / scale[d >= split]);
      bytes[row * stride + d] = code.__x;
      values[row * width + d] = bf16(float(code) * scale[d >= split]);
    }
    std::memcpy(bytes.data() + row * stride + width, scale, scales * 4);
  }
  return bytes;
}

void fp8_frozen(unsigned length, bool global) {
  constexpr unsigned heads = 16;
  const unsigned width = global ? 512 : 256, kv_heads = global ? 2 : 8;
  const unsigned capacity = global ? length + 1 : 1024, elements = global ? 640 : 256;
  const unsigned stride = (elements + (global ? 8 : 4) + 15) / 16 * 16;
  std::vector<BFloat16> key(std::size_t(kv_heads) * capacity * elements), values(key.size()), scale(512);
  for (std::size_t i = 0; i < key.size(); ++i) {
    key[i] = bf16((int((i * 17 + 11) % 251) - 125) / 256.f);
    values[i] = bf16((int((i * 7 + 43) % 239) - 119) / 128.f);
  }
  for (unsigned d = 0; d < 512; ++d) scale[d] = bf16(0.5f + (d % 29) / 32.f);
  const auto q = query(width, heads);
  auto key_bytes = fp8_records(key, elements, global ? 128 : elements);
  auto value_bytes = global ? std::vector<unsigned char>(1) : fp8_records(values, elements, elements);
  if (global) for (unsigned h = 0; h < kv_heads; ++h)
    std::fill_n(key_bytes.data() + (std::size_t(h) * capacity + length) * stride, elements, 0x7f);
  Device<unsigned char> dk(key_bytes), dv(value_bytes);
  Device<BFloat16> dq(q), ds(scale), output(q.size());
  const auto scratch_size = gewell::mtp_attention::frozen_scratch_bytes(heads, length);
  Device<unsigned char> scratch(scratch_size);
  gewell::kv_cache::DeviceView cache{};
  cache.key = reinterpret_cast<BFloat16*>(dk.get());
  cache.value = global ? nullptr : reinterpret_cast<BFloat16*>(dv.get());
  cache.capacity = capacity; cache.format = gewell::kv_cache::Format::fp8;
  const auto run = [&] {
    gewell::mtp_attention::run_frozen_prefix(dq.get(), cache, global ? ds.get() : nullptr,
        length, heads, global ? AttentionKind::global : AttentionKind::local,
        output.get(), scratch.get(), scratch_size, nullptr);
    check_frozen_batch(heads, dq.get(), cache, global ? ds.get() : nullptr,
        length, global ? AttentionKind::global : AttentionKind::local);
    return output.host();
  };
  const auto actual = run();
  const auto expected = reference(q, width, kv_heads, global || length < 1024 ? 0 : length - 1024, length,
      [&](unsigned h, unsigned t, unsigned d) {
        const auto* row = key.data() + (std::size_t(h) * capacity + (global ? t : t % 1024)) * elements;
        if (!global || d < 64) return value(row[d]);
        if (d >= 256 && d < 320) return value(row[d - 192]);
        return value(bf16(value(row[128 + d]) * value(scale[d])));
      }, [&](unsigned h, unsigned t, unsigned d) {
        const auto index = (std::size_t(h) * capacity + (global ? t : t % 1024)) * elements;
        return value(global ? key[index + 128 + d] : values[index + d]);
      });
  const auto error = compare(actual, expected, 0.001953125f);
  if (global) {
    const unsigned pages = (length + 255) / 256;
    const std::size_t page_stride = 16 + kv_heads * 256 * stride + 16;
    std::vector<unsigned char> pool(16 + pages * page_stride, 0xa5);
    std::vector<std::uint64_t> offsets(pages);
    for (unsigned p = 0; p < pages; ++p) offsets[p] = (16 + (pages - p - 1) * page_stride) / 2;
    for (unsigned t = 0; t < length; ++t) for (unsigned h = 0; h < kv_heads; ++h)
      std::memcpy(pool.data() + offsets[t / 256] * 2 + 16 + (h * 256 + t % 256) * stride,
          key_bytes.data() + (std::size_t(h) * capacity + t) * stride, stride);
    Device<unsigned char> dp(pool); Device<std::uint64_t> dt(offsets);
    cache.key = nullptr; cache.page_pool = reinterpret_cast<BFloat16*>(dp.get());
    cache.page_offsets = dt.get(); cache.page_tokens = 256; cache.page_count = pages;
    cache.page_stride_elements = page_stride / 2; cache.layer_offset_elements = 8;
    compare(run(), actual, 0);
    if (dp.host() != pool) throw std::runtime_error("FP8 frozen attention wrote paged KV");
  }
  if (dk.host() != key_bytes || dv.host() != value_bytes)
    throw std::runtime_error("FP8 frozen attention wrote contiguous KV");
  std::cout << "assistant FP8 storage heads=16 global=" << global << " prefix=" << length
            << " cpu_max_abs=" << error << " readonly=exact"
            << (global ? " paged_contiguous=exact" : "") << '\n';
}

// Uniform scores make the exact window endpoints observable without a numeric
// tolerance. Each KV head has a distinct answer, also checking GQA mapping.
void frozen_bounds() {
  constexpr unsigned heads = 16, length = 1031;
  for (bool global : {false, true}) {
    const unsigned width = global ? 512 : 256, kv_heads = global ? 2 : 8;
    const unsigned capacity = global ? length + 1 : 1024;
    const unsigned stride = global ? 640 : width;
    std::vector<BFloat16> key(std::size_t(kv_heads) * capacity * stride, bf16(0));
    std::vector<BFloat16> values(global ? 1 : key.size(), bf16(0));
    for (unsigned h = 0; h < kv_heads; ++h) {
      for (unsigned t = 0; t < capacity; ++t) {
        for (unsigned d = 0; d < width; ++d) {
          if (global) {
            // The pending slot is intentionally nonfinite and must be excluded.
            key[(std::size_t(h) * capacity + t) * stride + 128 + d] =
                bf16(t == length ? std::numeric_limits<float>::quiet_NaN() : h + 1.f);
          } else {
            float v = 0;
            if (t == (length - 1024) % 1024) v += 1024.f * (h + 1);
            if (t == (length - 1) % 1024) v += 2048.f * (h + 1);
            values[(std::size_t(h) * capacity + t) * stride + d] = bf16(v);
          }
        }
      }
    }
    std::vector<BFloat16> q(heads * width, bf16(0)), scale(512, bf16(0));
    Device<BFloat16> dk(key), dv(values), dq(q), ds(scale), output(q.size());
    const auto bytes = gewell::mtp_attention::frozen_scratch_bytes(heads, length);
    Device<unsigned char> scratch(bytes);
    gewell::kv_cache::DeviceView cache{};
    cache.key = dk.get(); cache.value = global ? nullptr : dv.get(); cache.capacity = capacity;
    gewell::mtp_attention::run_frozen_prefix(dq.get(), cache, global ? ds.get() : nullptr,
        length, heads, global ? AttentionKind::global : AttentionKind::local,
        output.get(), scratch.get(), bytes, nullptr);
    auto expected = q;
    for (unsigned h = 0; h < heads; ++h)
      std::fill_n(expected.data() + h * width, width,
          bf16((global ? 1.f : 3.f) * (h / (heads / kv_heads) + 1)));
    compare(output.host(), expected, 0);
    std::cout << "assistant 26B exact frozen bounds global=" << global << " passed\n";
  }
}

}  // namespace

int main() {
  try {
    frozen_bounds();
    for (unsigned length : {1U, 32U, 257U, 1031U, 4099U})
      for (bool global : {false, true}) fp8_frozen(length, global);
    for (unsigned heads : {16U, 32U}) {
      for (unsigned length : {1U, 32U, 257U, 1024U, 1031U}) local(length, heads);
      for (unsigned length : {1U, 32U, 257U, 1031U, 4099U}) global(length, heads);
    }
    std::cout << "assistant frozen-prefix tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "assistant test: " << error.what() << '\n';
    return 1;
  }
}
