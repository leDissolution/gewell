#include "gewell/mtp_attention.h"
#include "gewell/bf16_primitives.h"
#include "gewell/prefill_primitives.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace a = gewell::mtp_attention;
namespace b = gewell::bf16_primitives;
namespace p = gewell::prefill_primitives;
namespace m = gewell::gemma4_31b;
using BF16 = gewell::mtp_target::BFloat16;
using Kind = m::AttentionKind;

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
template <class F> void invalid(F&& action, const char* label) {
  bool rejected = false;
  try { action(); } catch (const std::invalid_argument&) { rejected = true; }
  require(rejected, std::string("accepted invalid ") + label);
}
class Device {
 public:
  explicit Device(std::size_t bytes) : bytes_(bytes) { check(cudaMalloc(&data_, bytes)); }
  ~Device() { cudaFree(data_); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  template <class T = BF16> T* get() const { return static_cast<T*>(data_); }
  std::size_t bytes() const { return bytes_; }
  template <class T> void upload(const std::vector<T>& values) {
    require(values.size() * sizeof(T) == bytes_, "upload size mismatch");
    check(cudaMemcpy(data_, values.data(), bytes_, cudaMemcpyHostToDevice));
  }
  std::vector<BF16> host() const {
    std::vector<BF16> result(bytes_ / sizeof(BF16));
    check(cudaMemcpy(result.data(), data_, bytes_, cudaMemcpyDeviceToHost));
    return result;
  }
 private:
  void* data_{};
  std::size_t bytes_{};
};

BF16 bf(float value) { return __float2bfloat16_rn(value); }
float fp(BF16 value) { return __bfloat162float(value); }
bool same(const std::vector<BF16>& left, const std::vector<BF16>& right) {
  return left.size() == right.size() &&
      std::memcmp(left.data(), right.data(), left.size() * sizeof(BF16)) == 0;
}
BF16 pattern(unsigned tag, unsigned position, unsigned head, unsigned dimension) {
  std::uint32_t value = tag * 0x9e3779b9U + position * 0x85ebca6bU +
                        head * 0xc2b2ae35U + dimension;
  value ^= value >> 16; value *= 0x7feb352dU;
  value ^= value >> 15; value *= 0x846ca68bU; value ^= value >> 16;
  // Integer fractions give nontrivial but well-conditioned, exactly represented
  // BF16 operands. Q has smaller magnitude to exercise broad softmax support.
  const float scale = tag == 1 ? 1.0F / 1024 : 1.0F / 256;
  return bf((static_cast<int>(value & 255) - 127) * scale);
}
bool rotated(unsigned dimension) {
  return dimension < 64 || (dimension >= 256 && dimension < 320);
}
unsigned packed_dimension(unsigned dimension) {
  return dimension < 64 ? dimension : dimension - 192;
}

struct Fixture {
  unsigned base, rows, heads, width, capacity, page_count;
  bool global, paged;
  std::size_t page_stride, layer_offset;
  std::vector<BF16> query, key, value, cache_key, cache_value, norm;
  std::vector<std::uint64_t> offsets;
  Device dq, dk, dv, dc, dcv, dn, doff, out, old_out, scratch;
  gewell::kv_cache::DeviceView cache{};

  Fixture(unsigned position, unsigned count, bool full, bool pages)
      : base(position), rows(count), heads(full ? 4 : 16), width(full ? 512 : 256),
        capacity(full ? position + count : 1024),
        page_count(std::max(1U, (position + 255) / 256)), global(full), paged(pages),
        page_stride(4 * 256 * 640 + 256), layer_offset(128),
        query(std::size_t(32) * count * width), key(std::size_t(heads) * count * width),
        value(key.size()),
        cache_key(pages ? page_count * page_stride + 128 :
                  std::size_t(heads) * capacity * (full ? 640 : width)),
        cache_value(full ? 1 : cache_key.size()), norm(512), offsets(page_count),
        dq(query.size() * 2), dk(key.size() * 2), dv(value.size() * 2),
        dc(cache_key.size() * 2), dcv(cache_value.size() * 2), dn(norm.size() * 2),
        doff(offsets.size() * 8), out(query.size() * 2), old_out(query.size() * 2),
        scratch(a::scratch_bytes(count, std::max(1024U, position + count))) {
    for (unsigned page = 0; page < page_count; ++page)
      offsets[page] = std::size_t(page_count - 1 - page) * page_stride + 64;
    // Unused cache rows, padding, and physical page gaps are poisonous. Reading
    // past the committed prefix cannot accidentally produce plausible output.
    std::fill(cache_key.begin(), cache_key.end(), bf(std::numeric_limits<float>::quiet_NaN()));
    std::fill(cache_value.begin(), cache_value.end(), bf(std::numeric_limits<float>::quiet_NaN()));
    for (unsigned d = 0; d < norm.size(); ++d)
      norm[d] = bf(0.75F + (d % 31) * (1.0F / 64));
    for (unsigned h = 0; h < 32; ++h)
      for (unsigned row = 0; row < rows; ++row)
        for (unsigned d = 0; d < width; ++d)
          query[(std::size_t(h) * rows + row) * width + d] = pattern(1, base + row, h, d);
    for (unsigned h = 0; h < heads; ++h) {
      for (unsigned row = 0; row < rows; ++row)
        for (unsigned d = 0; d < width; ++d) {
          key[(std::size_t(h) * rows + row) * width + d] = pattern(2, base + row, h, d);
          value[(std::size_t(row) * heads + h) * width + d] = pattern(3, base + row, h, d);
        }
      const unsigned begin = full ? 0 : base > 1024 ? base - 1024 : 0;
      for (unsigned position = begin; position < base; ++position)
        for (unsigned d = 0; d < width; ++d) {
          if (full) {
            const auto offset = cache_offset(h, position);
            if (rotated(d)) cache_key[offset + packed_dimension(d)] = pattern(2, position, h, d);
            cache_key[offset + 128 + d] = pattern(3, position, h, d);
          } else {
            const auto offset = (std::size_t(h) * 1024 + position % 1024) * width + d;
            cache_key[offset] = pattern(2, position, h, d);
            cache_value[offset] = pattern(3, position, h, d);
          }
        }
    }
    dq.upload(query); dk.upload(key); dv.upload(value); dc.upload(cache_key);
    dcv.upload(cache_value); dn.upload(norm); doff.upload(offsets);
    if (paged) {
      cache.page_pool = dc.get(); cache.page_offsets = doff.get<std::uint64_t>();
      cache.page_tokens = 256; cache.page_count = page_count;
      cache.page_stride_elements = page_stride; cache.layer_offset_elements = layer_offset;
    } else {
      cache.key = dc.get(); cache.value = global ? nullptr : dcv.get();
      cache.capacity = capacity;
    }
  }
  std::size_t cache_offset(unsigned head, unsigned position) const {
    return paged ? offsets[position / 256] + layer_offset +
                       (std::size_t(head) * 256 + position % 256) * 640
                 : (std::size_t(head) * capacity + position) * 640;
  }
  Kind kind() const { return global ? Kind::global : Kind::local; }
  std::string label() const {
    return std::string(global ? paged ? "paged" : "compact" : "local") +
        " base=" + std::to_string(base) + " rows=" + std::to_string(rows);
  }
  p::CompactGlobalPagedCache paged_view() const {
    return {cache.page_pool, cache.page_offsets, cache.page_tokens, cache.page_count,
            cache.page_stride_elements, cache.layer_offset_elements};
  }
  void run() {
    a::run(dq.get(), dk.get(), dv.get(), cache, global ? dn.get() : nullptr,
           base, rows, kind(), out.get(), scratch.get<void>(), scratch.bytes(), nullptr);
  }
  void generic() {
    if (!global)
      p::causal_gqa_attention_cached_chunk(dq.get(), dk.get(), dv.get(), dc.get(),
          dcv.get(), base, rows, capacity, old_out.get(), kind());
    else if (paged)
      p::causal_gqa_attention_cached_chunk_global_compact_paged(dq.get(), dk.get(),
          dv.get(), paged_view(), dn.get(), base, rows, old_out.get());
    else
      p::causal_gqa_attention_cached_chunk_global_compact(dq.get(), dk.get(), dv.get(),
          dc.get(), dn.get(), base, rows, capacity, old_out.get());
  }
  double get_value(unsigned head, unsigned position, unsigned d) const {
    if (position >= base)
      return fp(value[(std::size_t(position - base) * heads + head) * width + d]);
    return global ? fp(cache_key[cache_offset(head, position) + 128 + d])
                  : fp(cache_value[(std::size_t(head) * 1024 + position % 1024) * width + d]);
  }
  double get_key(unsigned head, unsigned position, unsigned d) const {
    if (global && !rotated(d)) return fp(bf(float(get_value(head, position, d)) * fp(norm[d])));
    if (position >= base)
      return fp(key[(std::size_t(head) * rows + position - base) * width + d]);
    return global ? fp(cache_key[cache_offset(head, position) + packed_dimension(d)])
                  : fp(cache_key[(std::size_t(head) * 1024 + position % 1024) * width + d]);
  }
  std::vector<BF16> reference(unsigned row, unsigned query_head) const {
    const unsigned head = query_head / (32 / heads), end = base + row;
    const unsigned first = !global && end >= 1023 ? end - 1023 : 0;
    std::vector<double> scores(end - first + 1), numerator(width);
    double maximum = -std::numeric_limits<double>::infinity();
    for (unsigned position = first; position <= end; ++position) {
      double score = 0;
      for (unsigned d = 0; d < width; ++d)
        score += double(fp(query[(std::size_t(query_head) * rows + row) * width + d])) *
                 get_key(head, position, d);
      scores[position - first] = score; maximum = std::max(maximum, score);
    }
    double denominator = 0;
    for (unsigned position = first; position <= end; ++position) {
      const double probability = std::exp(scores[position - first] - maximum);
      denominator += probability;
      const double pv_probability = fp(bf(float(probability)));
      for (unsigned d = 0; d < width; ++d)
        numerator[d] += pv_probability * get_value(head, position, d);
    }
    std::vector<BF16> result(width);
    for (unsigned d = 0; d < width; ++d) result[d] = bf(float(numerator[d] / denominator));
    return result;
  }
};

struct Error {
  double sum_squared{}, reference_squared{}, maximum{};
  std::size_t count{};
  void observe(BF16 actual, BF16 expected, const std::string& label) {
    const double x = fp(actual), y = fp(expected), error = std::abs(x - y);
    if (!std::isfinite(x) || !std::isfinite(y))
      throw std::runtime_error(label + " produced a nonfinite value");
    sum_squared += error * error; reference_squared += y * y;
    maximum = std::max(maximum, error); ++count;
  }
  void add(BF16 actual, BF16 expected, const std::string& label) {
    const double x = fp(actual), y = fp(expected), error = std::abs(x - y);
    const double ulp = std::abs(y) < std::numeric_limits<float>::min()
        ? std::ldexp(1.0, -133) : std::ldexp(1.0, std::ilogb(std::abs(y)) - 7);
    if (!std::isfinite(x) || !std::isfinite(y) || error > std::max(1e-6, ulp))
      throw std::runtime_error(label + " exceeds one BF16 ULP (absolute floor 1e-6): actual=" +
                               std::to_string(x) + " expected=" + std::to_string(y));
    observe(actual, expected, label);
  }
  double relative_rms() const { return std::sqrt(sum_squared / std::max(1e-30, reference_squared)); }
  void finish(const std::string& label, double relative_tolerance,
              double absolute_tolerance) const {
    require(maximum <= absolute_tolerance,
            label + " maximum absolute error exceeds " +
                std::to_string(absolute_tolerance) + ": " +
                std::to_string(maximum));
    require(relative_rms() <= relative_tolerance,
            label + " relative RMS error exceeds " +
                std::to_string(relative_tolerance) + ": " +
                std::to_string(relative_rms()));
  }
};

void geometry(unsigned base, unsigned rows, bool global, bool paged) {
  Fixture f(base, rows, global, paged);
  f.run(); f.generic();
  const auto actual = f.out.host(), generic = f.old_out.host();
  if (rows == 1) {
    a::run_decode_batch(32, {{f.dq.get(), f.dk.get(), f.dv.get(), f.cache, base, 1, f.out.get()}},
        f.dn.get(), f.kind(), f.scratch.get<void>(), f.scratch.bytes(), nullptr);
    require(same(actual, f.out.host()), f.label() + " shared single-row decode changed arithmetic");
  }
  Error generic_drift, cpu_error;
  const auto generic_label = f.label() + " generic";
  // Generic chunk attention retains FP32 P*V probabilities. Keep it as a
  // bounded drift diagnostic; the exact tiny-probability fixture below and
  // the BF16-P CPU oracle enforce the operand boundary. The per-component
  // guard prevents aggregate RMS from hiding a localized regression.
  for (std::size_t i = 0; i < actual.size(); ++i)
    generic_drift.observe(actual[i], generic[i], generic_label);
  require(generic_drift.maximum <= 0.002,
          generic_label + " maximum absolute drift exceeds 0.002");
  require(generic_drift.relative_rms() <= 0.005,
          generic_label + " BF16-probability drift exceeds 0.005");
  std::vector<unsigned> selected_rows{0, rows / 2, rows - 1};
  std::sort(selected_rows.begin(), selected_rows.end());
  selected_rows.erase(std::unique(selected_rows.begin(), selected_rows.end()), selected_rows.end());
  for (unsigned row : selected_rows)
    for (unsigned head : {0U, 17U, 31U}) {
      const auto expected = f.reference(row, head);
      for (unsigned d = 0; d < f.width; ++d)
        cpu_error.observe(
            actual[(std::size_t(row) * 32 + head) * f.width + d], expected[d],
            f.label() + " CPU row=" + std::to_string(row) +
                " head=" + std::to_string(head));
    }
  cpu_error.finish(f.label() + " CPU", 0.005, 0.001);
  require(same(f.dc.host(), f.cache_key) && same(f.dcv.host(), f.cache_value),
          f.label() + " committed cache mutated");
  if (rows > 1) {
    const BF16 poison = bf(std::numeric_limits<float>::quiet_NaN());
    for (unsigned head = 0; head < f.heads; ++head)
      std::fill(f.key.begin() + (std::size_t(head) * rows + 1) * f.width,
                f.key.begin() + std::size_t(head + 1) * rows * f.width, poison);
    std::fill(f.value.begin() + f.heads * f.width, f.value.end(), poison);
    f.dk.upload(f.key); f.dv.upload(f.value); f.run();
    const auto poisoned = f.out.host();
    require(std::memcmp(actual.data(), poisoned.data(), 32 * f.width * sizeof(BF16)) == 0,
            f.label() + " future staged NaN changed row zero");
    require(same(f.dc.host(), f.cache_key) && same(f.dcv.host(), f.cache_value),
            f.label() + " poisoned run mutated committed cache");
  }
  std::cout << f.label() << " generic_max_abs=" << generic_drift.maximum
            << " cpu_max_abs=" << cpu_error.maximum
            << " cpu_relative_rms=" << cpu_error.relative_rms()
            << " causal=exact cache_unchanged=exact\n";
}

void misaligned_local_copies() {
  for (unsigned rows : {1U, 4U}) {
    Fixture f(1023, rows, false, false);
    Device key(f.dc.bytes() + 2), value(f.dcv.bytes() + 2);
    Device staged_key(f.dk.bytes() + 2), staged_value(f.dv.bytes() + 2);
    check(cudaMemcpy(key.get() + 1, f.dc.get(), f.dc.bytes(), cudaMemcpyDeviceToDevice));
    check(cudaMemcpy(value.get() + 1, f.dcv.get(), f.dcv.bytes(), cudaMemcpyDeviceToDevice));
    check(cudaMemcpy(staged_key.get() + 1, f.dk.get(), f.dk.bytes(), cudaMemcpyDeviceToDevice));
    check(cudaMemcpy(staged_value.get() + 1, f.dv.get(), f.dv.bytes(), cudaMemcpyDeviceToDevice));
    auto cache = f.cache;
    cache.key = key.get() + 1; cache.value = value.get() + 1;
    if (rows == 1) {
      Device scratch(b::causal_gqa_attention_cached_m1_fused_scratch_bytes(1024, Kind::local));
      b::causal_gqa_attention_cached_m1_fused(f.dq.get(), f.dc.get(), f.dcv.get(),
          f.base - 1, 1024, scratch.get<void>(), f.old_out.get(), Kind::local, nullptr);
      b::causal_gqa_attention_cached_m1_fused(f.dq.get(), cache.key, cache.value,
          f.base - 1, 1024, scratch.get<void>(), f.out.get(), Kind::local, nullptr);
    } else {
      f.run();
      check(cudaMemcpy(f.old_out.get(), f.out.get(), f.out.bytes(), cudaMemcpyDeviceToDevice));
      a::run(f.dq.get(), staged_key.get() + 1, staged_value.get() + 1, cache, nullptr,
          f.base, rows, Kind::local, f.out.get(), f.scratch.get<void>(), f.scratch.bytes(), nullptr);
    }
    require(same(f.out.host(), f.old_out.host()), "misaligned local KV copy changed attention");
  }
  std::cout << "local KV vector/scalar alignment output=exact\n";
}

void bounds_and_long_context() {
  invalid([] { a::scratch_bytes(0); }, "zero scratch rows");
  invalid([] { a::scratch_bytes(1281); }, "excess scratch rows");
  invalid([] { a::scratch_bytes(1, 0); }, "zero context");
  invalid([] { a::scratch_bytes(1, 262145); }, "excess context");
  require(a::scratch_bytes(1280, 262144) <= 129ULL * 1024 * 1024,
          "maximum scratch exceeds 129 MiB bound");
  require(a::scratch_bytes(1280, 262144) == a::scratch_bytes(a::kQueryTileRows, 262144),
          "scratch grows beyond one query tile");
  Fixture f(255, 3, true, true);
  invalid([&] { a::run(f.dq.get(), f.dk.get(), f.dv.get(), f.cache, f.dn.get(),
      f.base, f.rows, f.kind(), f.out.get(), f.scratch.get<void>(),
      a::scratch_bytes(f.rows, f.base + f.rows) - 1, nullptr); },
      "undersized scratch");
  auto bad = f.cache; bad.page_count = 0;
  invalid([&] { a::run(f.dq.get(), f.dk.get(), f.dv.get(), bad, f.dn.get(),
      f.base, f.rows, f.kind(), f.out.get(), f.scratch.get<void>(), f.scratch.bytes(), nullptr); },
      "incomplete committed page table");
  invalid([&] { a::run(f.dq.get(), f.dk.get(), f.dv.get(), f.cache, f.dn.get(),
      262143, 3, f.kind(), f.out.get(), f.scratch.get<void>(), f.scratch.bytes(), nullptr); },
      "context overflow");

  // Shared read-only physical pages make the maximum logical context test
  // small. Every visible value is exactly 0.25, hence so is every context.
  constexpr unsigned base = 262143, page_elements = 4 * 256 * 640;
  Device query(32 * 512 * 2), key(4 * 512 * 2), value(4 * 512 * 2);
  Device page(page_elements * 2), norm(512 * 2), offsets(1024 * 8), output(32 * 512 * 2);
  Device scratch(a::scratch_bytes(1, 262144));
  query.upload(std::vector<BF16>(32 * 512, bf(0.125F)));
  key.upload(std::vector<BF16>(4 * 512, bf(0.25F)));
  value.upload(std::vector<BF16>(4 * 512, bf(0.25F)));
  page.upload(std::vector<BF16>(page_elements, bf(0.25F)));
  norm.upload(std::vector<BF16>(512, bf(1)));
  offsets.upload(std::vector<std::uint64_t>(1024, 0));
  gewell::kv_cache::DeviceView cache{};
  cache.page_pool = page.get(); cache.page_offsets = offsets.get<std::uint64_t>();
  cache.page_tokens = 256; cache.page_count = 1024; cache.page_stride_elements = page_elements;
  a::run(query.get(), key.get(), value.get(), cache, norm.get(), base, 1,
         Kind::global, output.get(), scratch.get<void>(), scratch.bytes(), nullptr);
  for (auto result : output.host()) require(fp(result) == 0.25F, "maximum context constant-value mismatch");
  a::run_frozen_global(query.get(), cache, norm.get(), 262144,
      output.get(), scratch.get<void>(), scratch.bytes(), nullptr);
  for (auto result : output.host()) require(fp(result) == 0.25F, "maximum frozen context mismatch");
  invalid([&] { a::run_frozen_global(query.get(), cache, norm.get(), 0,
      output.get(), scratch.get<void>(), scratch.bytes(), nullptr); }, "empty frozen prefix");
  auto oversized = cache;
  oversized.page_count = 1025;
  invalid([&] { a::run_frozen_global(query.get(), oversized, norm.get(), 262145,
      output.get(), scratch.get<void>(), scratch.bytes(), nullptr); }, "frozen context overflow");
  invalid([&] { a::run_frozen_global(query.get(), cache, norm.get(), 262144,
      output.get(), scratch.get<void>(), scratch.bytes() - 1, nullptr); }, "undersized frozen scratch");
  std::cout << "max_context=262144 max_rows=1280 scratch_bytes="
            << a::scratch_bytes(1280, 262144) << " bounded=1 validation=passed\n";
}

void mixed_batch_launches() {
  constexpr unsigned batch = 64;
  for (const bool global : {false, true}) {
    Fixture single(4093, 1, global, global), multi(4093, 4, global, global);
    single.run(); multi.run();
    const auto one = single.out.host(), four = multi.out.host();
    Device outputs((one.size()+four.size())*(batch/2)*sizeof(BF16));
    Device scratch(a::scratch_bytes(160, 8192));
    std::vector<a::BatchInput> inputs;
    std::size_t offset = 0;
    for (unsigned i = 0; i < batch; ++i) {
      auto& f = i%2 ? multi : single;
      inputs.push_back({f.dq.get(), f.dk.get(), f.dv.get(), f.cache,
          f.base, f.rows, outputs.get()+offset});
      offset += f.query.size();
    }
    cudaStream_t stream;
    check(cudaStreamCreate(&stream));
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    a::run_batch(inputs, global ? single.dn.get() : nullptr, single.kind(),
                 scratch.get<void>(), scratch.bytes(), stream);
    check(cudaStreamEndCapture(stream, &graph));
    std::size_t nodes = 0;
    check(cudaGraphGetNodes(graph, nullptr, &nodes));
    require(nodes == 4, "interleaved single/multirow requests fragmented attention launches");
    check(cudaGraphInstantiate(&executable, graph, 0));
    check(cudaGraphLaunch(executable, stream));
    check(cudaStreamSynchronize(stream));
    const auto actual = outputs.host();
    Error error;
    offset = 0;
    for (unsigned i = 0; i < batch; ++i) {
      const auto& reference = i%2 ? four : one;
      for (std::size_t j = 0; j < reference.size(); ++j)
        error.observe(actual[offset+j], reference[j], "mixed attention output routing");
      offset += reference.size();
    }
    error.finish("mixed attention/serial", .005, .001);
    cudaGraphExecDestroy(executable); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
    std::cout << "mixed_attention global=" << global << " requests=64 launches=4 routing=passed\n";
  }
}

void batched_equivalence() {
  for (const bool global : {false, true}) {
    std::vector<std::unique_ptr<Fixture>> fixtures;
    std::vector<a::BatchInput> inputs;
    // More than one launch-parameter group, then ragged depths/contexts and
    // both global cache representations. Distinct allocations expose routing
    // mistakes that shared constant-value fixtures would conceal.
    for (unsigned i = 0; i < 37; ++i) {
      constexpr unsigned depths[]{1, 2, 3, 5, 8, 9, 17};
      constexpr unsigned positions[]{1, 31, 255, 1023, 4093, 16379, 16383};
      const unsigned rows = i < 18 ? 4 : depths[i % 7];
      // One long request among short peers exercises work-weighted splits,
      // including the maximum-context tail, without duplicating a large KV.
      const unsigned base = i == 15 ? 262139 : i < 18 ? 3237 + i : positions[i % 7];
      auto f = std::make_unique<Fixture>(base, rows, global, global && (i < 18 || i % 2));
      f->run();
      check(cudaMemcpy(f->old_out.get(), f->out.get(), f->out.bytes(), cudaMemcpyDeviceToDevice));
      inputs.push_back({f->dq.get(), f->dk.get(), f->dv.get(), f->cache,
                         f->base, f->rows, f->out.get()});
      fixtures.push_back(std::move(f));
    }
    Device scratch(a::scratch_bytes(1280, 262144) + 256);
    const auto bytes = scratch.bytes() - 256;
    check(cudaMemset(scratch.get<void>(), 0xCD, scratch.bytes()));
    const auto kind = global ? Kind::global : Kind::local;
    const auto* norm = global ? fixtures.front()->dn.get() : nullptr;
    cudaStream_t stream;
    check(cudaStreamCreate(&stream));
    // First use of each batched specialization is captured, including the
    // dynamic shared-memory attribute setup for both global cache layouts.
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    a::run_batch(inputs, norm, kind, scratch.get<void>(), bytes, stream);
    check(cudaStreamEndCapture(stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, 0));
    check(cudaGraphLaunch(executable, stream));
    check(cudaGraphLaunch(executable, stream));
    check(cudaStreamSynchronize(stream));
    Error serial_error, oracle_error;
    for (const auto& f : fixtures) {
      const auto actual = f->out.host(), serial = f->old_out.host();
      const auto label = "batched " + f->label();
      for (std::size_t j = 0; j < actual.size(); ++j)
        serial_error.observe(actual[j], serial[j], label + " serial");
      const auto reference = f->reference(f->rows - 1, 17);
      for (unsigned d = 0; d < f->width; ++d)
        oracle_error.observe(
            actual[((f->rows - 1) * 32 + 17) * f->width + d], reference[d],
            label + " BF16-P oracle");
      require(same(f->dc.host(), f->cache_key) && same(f->dcv.host(), f->cache_value),
               "batched attention mutated cache " + f->label());
      // Isolation is exact at fixed batch geometry, independently of the
      // harmless summation-order differences from serial execution.
      check(cudaMemcpy(f->old_out.get(), f->out.get(), f->out.bytes(), cudaMemcpyDeviceToDevice));
    }
    serial_error.finish("batched/serial", 0.005, 0.001);
    oracle_error.finish("batched/BF16-P oracle", 0.005, 0.001);
    auto& first = *fixtures.front();
    const auto original_key = first.key, original_value = first.value;
    const auto poison = bf(std::numeric_limits<float>::quiet_NaN());
    for (unsigned h = 0; h < first.heads; ++h)
      std::fill(first.key.begin() + (std::size_t(h) * first.rows + 1) * first.width,
                first.key.begin() + (std::size_t(h) + 1) * first.rows * first.width, poison);
    std::fill(first.value.begin() + first.heads * first.width, first.value.end(), poison);
    first.dk.upload(first.key); first.dv.upload(first.value);
    check(cudaGraphLaunch(executable, stream));
    check(cudaStreamSynchronize(stream));
    const auto poisoned = first.out.host(), expected = first.old_out.host();
    require(std::memcmp(poisoned.data(), expected.data(), 32 * first.width * 2) == 0,
             "batched future NaN changed row zero");
    for (unsigned i = 1; i < fixtures.size(); ++i)
      require(same(fixtures[i]->out.host(), fixtures[i]->old_out.host()),
               "poisoned request changed another request's attention");
    first.key = original_key; first.value = original_value;
    first.dk.upload(first.key); first.dv.upload(first.value);
    std::fill(first.query.begin(), first.query.end(), bf(0.25F));
    first.dq.upload(first.query);
    check(cudaGraphLaunch(executable, stream));
    check(cudaStreamSynchronize(stream));
    require(!same(first.out.host(), first.old_out.host()), "query perturbation had no effect");
    for (unsigned i = 1; i < fixtures.size(); ++i)
      require(same(fixtures[i]->out.host(), fixtures[i]->old_out.host()),
               "one request changed another request's attention");
    std::array<unsigned char, 256> guard;
    check(cudaMemcpy(guard.data(), scratch.get<unsigned char>() + bytes,
                       guard.size(), cudaMemcpyDeviceToHost));
    require(std::all_of(guard.begin(), guard.end(), [](auto v) { return v == 0xCD; }),
             "batched attention exceeded scratch");
    invalid([&] { a::run_batch({}, norm, kind, scratch.get<void>(), bytes, stream); }, "empty batch");
    auto bad = inputs;
    bad.back().rows = 0;
    invalid([&] { a::run_batch(bad, norm, kind, scratch.get<void>(), bytes, stream); }, "invalid batch tail");
    cudaGraphExecDestroy(executable); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
    std::cout << "batched attention global=" << global
              << " requests=37 ragged max_context=262143 serial_relative_rms=" << serial_error.relative_rms()
              << " serial_max_abs=" << serial_error.maximum
              << " bf16_p_oracle_relative_rms=" << oracle_error.relative_rms()
              << " oracle_max_abs=" << oracle_error.maximum
              << " isolation=exact future_nan=exact cold_graph=passed scratch_guard=passed\n";
  }
}

void split_budget_launches() {
  for (const bool global : {false, true}) {
    Fixture f(4096, 4, global, global);
    Device output(32 * f.out.bytes());
    // Short global four-row calls cap at 32 splits. This allocation fits one
    // original request but cannot fit the full launch's planned intermediates.
    const auto single_bytes = a::scratch_bytes(4, 1024);
    for (const bool constrained : {false, true}) {
      const auto bytes = constrained ? single_bytes : a::scratch_bytes(1280, 262144);
      Device scratch(bytes + 256);
      for (unsigned batch : {1U, 2U, 8U, 16U, 32U}) {
        check(cudaMemset(scratch.get<void>(), 0xCD, scratch.bytes()));
        std::vector<a::BatchInput> inputs;
        for (unsigned i = 0; i < batch; ++i)
          inputs.push_back({f.dq.get(), f.dk.get(), f.dv.get(), f.cache, f.base, f.rows,
                            output.get() + i * f.query.size()});
        cudaStream_t stream; check(cudaStreamCreate(&stream));
        cudaGraph_t graph; cudaGraphExec_t executable;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        a::run_batch(inputs, global ? f.dn.get() : nullptr, f.kind(),
                     scratch.get<void>(), bytes, stream);
        check(cudaStreamEndCapture(stream, &graph));
        std::size_t size = 0; check(cudaGraphGetNodes(graph, nullptr, &size));
        std::vector<cudaGraphNode_t> nodes(size);
        check(cudaGraphGetNodes(graph, nodes.data(), &size));
        unsigned blocks = 0, launches = 0;
        for (auto node : nodes) {
          cudaGraphNodeType type; check(cudaGraphNodeGetType(node, &type));
          if (type != cudaGraphNodeTypeKernel) continue;
          cudaKernelNodeParams params{}; check(cudaGraphKernelNodeGetParams(node, &params));
          if (params.gridDim.x != f.heads) continue;
          blocks += params.gridDim.x * params.gridDim.y * params.gridDim.z;
          ++launches;
        }
        if (!constrained) {
          const unsigned groups = (batch + 31) / 32;
          const unsigned splits = std::min(32U, (global ? 128U : 64U) / std::min(batch, 32U));
          require(blocks == batch * f.heads * splits && launches == groups,
                   "split budget did not size the actual launch cohort");
        } else {
          require(blocks == batch * f.heads * 32 && launches == batch,
                   "scratch-constrained launches did not retain enough splits");
        }
        check(cudaGraphInstantiate(&executable, graph, 0));
        check(cudaGraphLaunch(executable, stream)); check(cudaStreamSynchronize(stream));
        f.run();
        const auto actual = output.host(), reference = f.out.host();
        Error error;
        for (std::size_t i = 0; i < batch * reference.size(); ++i)
          error.observe(actual[i], reference[i % reference.size()],
                        "split budget result");
        error.finish("split budget result", 0.005, 0.001);
        std::array<unsigned char, 256> guard;
        check(cudaMemcpy(guard.data(), scratch.get<unsigned char>() + bytes, guard.size(), cudaMemcpyDeviceToHost));
        require(std::all_of(guard.begin(), guard.end(), [](auto x) { return x == 0xCD; }),
                 "split budget exceeded hard scratch limit");
        std::cout << "split_budget global=" << global << " batch=" << batch
                  << " constrained=" << constrained << " blocks=" << blocks
                  << " launches=" << launches
                  << " max_abs=" << error.maximum
                  << " numerical=passed scratch_guard=passed\n";
        check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph)); check(cudaStreamDestroy(stream));
      }
    }
  }
}

void tiny_probability(bool global) {
  Fixture f(1, 4, global, false);
  std::fill(f.query.begin(), f.query.end(), bf(0));
  std::fill(f.key.begin(), f.key.end(), bf(0));
  std::fill(f.value.begin(), f.value.end(), bf(0));
  std::fill(f.cache_key.begin(), f.cache_key.end(), bf(0));
  std::fill(f.cache_value.begin(), f.cache_value.end(), bf(0));
  std::fill(f.norm.begin(), f.norm.end(), bf(1));
  for (unsigned h = 0; h < 32; ++h)
    for (unsigned row = 0; row < f.rows; ++row)
      f.query[(std::size_t(h) * f.rows + row) * f.width] = bf(1);
  for (unsigned h = 0; h < f.heads; ++h) {
    if (global) {
      f.cache_key[f.cache_offset(h, 0)] = bf(-96);
      f.cache_key[f.cache_offset(h, 0) + 129] = bf(std::ldexp(1.0F, 126));
    } else {
      f.cache_key[std::size_t(h) * 1024 * f.width] = bf(-96);
      f.cache_value[std::size_t(h) * 1024 * f.width + 1] = bf(std::ldexp(1.0F, 126));
    }
  }
  f.dq.upload(f.query); f.dk.upload(f.key); f.dv.upload(f.value);
  f.dc.upload(f.cache_key); f.dcv.upload(f.cache_value); f.dn.upload(f.norm);
  f.run();
  const auto output = f.out.host();
  Error error;
  for (unsigned row = 0; row < f.rows; ++row) {
    // exp(-96) is below BF16's smallest subnormal. Even a large, finite V
    // cannot recover a probability after the explicit BF16 P*V boundary.
    const auto expected = bf(0.0F);
    for (unsigned head = 0; head < 32; ++head)
      error.add(output[(std::size_t(row) * 32 + head) * f.width + 1], expected,
                f.label() + " subnormal probability");
  }
  std::cout << f.label() << " BF16_subnormal_probability_dropped=1\n";
}

double median_milliseconds(const std::function<void()>& operation) {
  for (unsigned i = 0; i < 5; ++i) operation();
  check(cudaDeviceSynchronize());
  cudaEvent_t begin{}, end{}; check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
  std::vector<double> measurements;
  for (unsigned sample = 0; sample < 17; ++sample) {
    check(cudaEventRecord(begin));
    for (unsigned repeat = 0; repeat < 5; ++repeat) operation();
    check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
    float milliseconds = 0; check(cudaEventElapsedTime(&milliseconds, begin, end));
    measurements.push_back(milliseconds / 5);
  }
  check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
  std::sort(measurements.begin(), measurements.end());
  return measurements[measurements.size() / 2];
}

void benchmark(unsigned base, bool global, bool paged) {
  Fixture f(base, 3, global, paged);
  Device q1(32 * f.width * 2), k1(f.heads * f.width * 2), result(32 * f.width * 2);
  Device scratch(b::causal_gqa_attention_cached_m1_fused_scratch_bytes(
      global ? paged ? f.page_count * 256 : f.capacity : 1024, f.kind()));
  check(cudaMemcpy2D(q1.get(), f.width * 2, f.dq.get(), f.rows * f.width * 2,
                     f.width * 2, 32, cudaMemcpyDeviceToDevice));
  check(cudaMemcpy2D(k1.get(), f.width * 2, f.dk.get(), f.rows * f.width * 2,
                     f.width * 2, f.heads, cudaMemcpyDeviceToDevice));
  if (!global)
    b::write_kv_cache_m1(k1.get(), f.dv.get(), f.dc.get(), f.dcv.get(), base, 1024, f.kind());
  else if (!paged)
    b::write_kv_cache_m1_global_compact(k1.get(), f.dv.get(), f.dc.get(), base, f.capacity);
  else
    p::write_kv_cache_chunk_global_compact_paged(k1.get(), f.dv.get(), f.paged_view(), base, 1);
  const auto ordinary = [&] {
    if (!global)
      b::causal_gqa_attention_cached_m1_fused(q1.get(), f.dc.get(), f.dcv.get(), base,
          1024, scratch.get<void>(), result.get(), f.kind());
    else if (!paged)
      b::causal_gqa_attention_cached_m1_fused_global_compact(q1.get(), f.dc.get(), f.dn.get(),
          base, f.capacity, scratch.get<void>(), result.get());
    else
      b::causal_gqa_attention_cached_m1_fused_global_compact_paged(q1.get(), f.paged_view(),
          f.dn.get(), base, scratch.get<void>(), result.get());
  };
  const double m1 = median_milliseconds(ordinary);
  const double generic = median_milliseconds([&] { f.generic(); });
  const double parallel = median_milliseconds([&] { f.run(); });
  std::cout << "benchmark " << f.label() << " ordinary_m1_ms=" << m1
            << " generic_m3_ms=" << generic << " parallel_m3_ms=" << parallel
            << " generic_speedup=" << generic / parallel << '\n';
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::cout << std::setprecision(8);
    if (argc == 2 && std::string(argv[1]) == "--benchmark") {
      for (unsigned base : {3237U, 32767U}) {
        benchmark(base, false, false);
        benchmark(base, true, false);
        benchmark(base, true, true);
      }
      return 0;
    }
    require(argc == 1, "usage: mtp_attention_test [--benchmark]");
    constexpr std::array<unsigned, 7> bases{1, 31, 255, 256, 1023, 1024, 3237};
    constexpr std::array<unsigned, 7> rows{1, 3, 5, 17, 3, 5, 17};
    for (unsigned i = 0; i < bases.size(); ++i) {
      geometry(bases[i], rows[i], false, false);
      geometry(bases[i], rows[i], true, false);
      geometry(bases[i], rows[i], true, true);
    }
    for (unsigned base : {0U, 1023U, 3237U}) {
      geometry(base, 4, false, false);
      geometry(base, 4, true, true);
    }
    // Partial four-row groups and both sides of the short-context split
    // boundary exercise the shared global K/V tile and its final reduction.
    for (unsigned base : {4093U, 16379U})
      for (unsigned rows : {2U, 4U, 5U, 8U, 9U})
        geometry(base, rows, true, true);
    tiny_probability(false);
    tiny_probability(true);
    misaligned_local_copies();
    batched_equivalence();
    mixed_batch_launches();
    split_budget_launches();
    geometry(1023, 1280, false, false);
    // Cross the split cap with varying keys/values, including a partial extra
    // tile. Constant-value long-context checks alone cannot verify rescaling
    // when later tiles have different maxima and softmax denominators.
    geometry(8191, 17, true, false);
    geometry(8191, 17, true, true);
    // Dispatch shared-row attention and exercise its odd final row in a second
    // query group. Keep these checks in CTest as well as the opt-in long oracle.
    geometry(16383, 9, true, false);
    geometry(16383, 9, true, true);
    bounds_and_long_context();
    check(cudaDeviceSynchronize());
    std::cout << "MTP parallel attention tests passed\n";
  } catch (const std::exception& error) {
    std::cerr << "MTP parallel attention test failed: " << error.what() << '\n';
    return 1;
  }
}
