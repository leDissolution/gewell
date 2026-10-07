// Opt-in resident-cache attention measurements and long-context numerical checks.
// The default CTest suite lives in mtp_attention_test.cu.
#include "gewell/mtp_attention.h"
#include "gewell/bf16_primitives.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace a = gewell::mtp_attention;
namespace b = gewell::bf16_primitives;
namespace m = gewell::gemma4_31b;
using BF16 = gewell::mtp_target::BFloat16;
using Cache = gewell::kv_cache::DeviceView;

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
class Device {
 public:
  explicit Device(std::size_t bytes) : bytes_(bytes) { check(cudaMalloc(&data_, bytes)); }
  ~Device() { cudaFree(data_); }
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  template <class T = BF16> T* get() const { return static_cast<T*>(data_); }
  std::size_t bytes() const { return bytes_; }
  template <class T> std::vector<T> host() const {
    std::vector<T> result(bytes_ / sizeof(T));
    check(cudaMemcpy(result.data(), data_, bytes_, cudaMemcpyDeviceToHost));
    return result;
  }
 private:
  void* data_{};
  std::size_t bytes_{};
};

__host__ __device__ BF16 pattern(unsigned tag, unsigned position,
                               unsigned head, unsigned dimension, bool outliers) {
  std::uint32_t value = tag * 0x9e3779b9U + position * 0x85ebca6bU +
                        head * 0xc2b2ae35U + dimension;
  value ^= value >> 16; value *= 0x7feb352dU;
  value ^= value >> 15; value *= 0x846ca68bU; value ^= value >> 16;
  float scale = tag == 1 ? 1.0F / 1024 : 1.0F / 256;
  if (outliers && tag != 1 && (position % 8191 == 0 || position % 8191 == 31))
    scale *= 32;
  if (outliers && tag == 1 && (dimension == 0 || dimension == 255)) scale *= 8;
  return __float2bfloat16_rn((static_cast<int>(value & 255) - 127) * scale);
}
__device__ BF16* row_pointer(Cache cache, unsigned head, unsigned position) {
  return cache.page_pool
      ? cache.page_pool + cache.page_offsets[position / 256] + cache.layer_offset_elements +
            (std::size_t(head) * 256 + position % 256) * 640
      : cache.key + (std::size_t(head) * cache.capacity + position) * 640;
}
__global__ void fill_cache(Cache cache, unsigned context, bool outliers) {
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= std::size_t(4) * context * 640) return;
  const unsigned d = i % 640, position = (i / 640) % context, head = i / (640ULL * context);
  row_pointer(cache, head, position)[d] = d < 128
      ? pattern(2, position, head, d < 64 ? d : d + 192, outliers)
      : pattern(3, position, head, d - 128, outliers);
}
__global__ void fill_inputs(BF16* query, BF16* key, BF16* value, BF16* norm,
                            unsigned base, unsigned rows, bool outliers) {
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < 32ULL * rows * 512) {
    const unsigned d = i % 512, row = (i / 512) % rows, head = i / (512ULL * rows);
    query[i] = pattern(1, base + row, head, d, outliers);
  }
  if (i < 4ULL * rows * 512) {
    const unsigned d = i % 512, row = (i / 512) % rows, head = i / (512ULL * rows);
    key[i] = pattern(2, base + row, head, d, outliers);
    value[(std::size_t(row) * 4 + head) * 512 + d] = pattern(3, base + row, head, d, outliers);
  }
  if (i < 512) norm[i] = __float2bfloat16_rn(0.75F + (i % 31) * (1.0F / 64));
}
__global__ void poison_future(BF16* key, BF16* value, unsigned rows) {
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= 4ULL * rows * 512) return;
  if ((i / 512) % rows != 0) key[i] = __float2bfloat16_rn(nanf(""));
  if (i >= 4 * 512) value[i] = __float2bfloat16_rn(nanf(""));
}

struct Fixture {
  unsigned context, rows, base, page_count, page_layers;
  bool paged, outliers, misaligned;
  static constexpr std::size_t kLayerPageElements = 4 * 256 * 640;
  std::size_t page_stride;
  Device cache_storage, offsets, query, key, value, norm, output, query1, output1;
  Device scratch, scratch1;
  Cache cache{};

  Fixture(unsigned total, unsigned count, bool pages, bool spikes, bool misaligned,
          unsigned layers)
      : context(total), rows(count), base(total - count), page_count((total + 255) / 256),
        page_layers(layers),
        paged(pages), outliers(spikes), misaligned(misaligned),
        page_stride(layers * kLayerPageElements),
        cache_storage((pages ? std::size_t(page_count) * page_stride + (misaligned ? 1 : 0) :
                             std::size_t(4) * total * 640 + (misaligned ? 1 : 0)) * 2),
        offsets(page_count * sizeof(std::uint64_t)),
        query((32 * count * 512 + (misaligned ? 1 : 0)) * 2),
        key(4 * count * 512 * 2), value(key.bytes()),
        norm((512 + (misaligned ? 1 : 0)) * 2), output(32 * count * 512 * 2),
        query1((32 * 512 + (misaligned ? 1 : 0)) * 2), output1(32 * 512 * 2),
        scratch(a::scratch_bytes(count, total)),
        scratch1(b::causal_gqa_attention_cached_m1_fused_scratch_bytes(
            pages ? page_count * 256 : total, m::AttentionKind::global)) {
    if (paged) {
      std::vector<std::uint64_t> host_offsets(page_count);
      // Deliberately reversed physical pages exercise noncontiguous addressing.
      for (unsigned page = 0; page < page_count; ++page)
        host_offsets[page] = std::size_t(page_count - 1 - page) * page_stride;
      check(cudaMemcpy(offsets.get<void>(), host_offsets.data(), offsets.bytes(), cudaMemcpyHostToDevice));
      cache.page_pool = cache_storage.get() + (misaligned ? 1 : 0);
      cache.page_offsets = offsets.get<std::uint64_t>();
      cache.page_tokens = 256; cache.page_count = page_count;
      cache.page_stride_elements = page_stride;
      // Production pages contain all ten global layers; use the last layer to
      // exercise its nonzero offset and the complete physical page stride.
      cache.layer_offset_elements = (layers - 1) * kLayerPageElements;
    } else {
      cache.key = cache_storage.get() + (misaligned ? 1 : 0); cache.capacity = total;
    }
    check(cudaMemset(cache_storage.get<void>(), 0xff, cache_storage.bytes()));
    fill_cache<<<(std::size_t(4) * total * 640 + 255) / 256, 256>>>(cache, total, spikes);
    restore_inputs();
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
  }
  BF16* query_ptr() const { return query.get() + (misaligned ? 1 : 0); }
  BF16* query1_ptr() const { return query1.get() + (misaligned ? 1 : 0); }
  BF16* norm_ptr() const { return norm.get() + (misaligned ? 1 : 0); }
  void restore_inputs() {
    fill_inputs<<<(32ULL * rows * 512 + 255) / 256, 256>>>(query_ptr(), key.get(), value.get(),
        norm_ptr(), base, rows, outliers);
    check(cudaMemcpy2D(query1_ptr(), 512 * 2, query_ptr(), rows * 512 * 2,
                       512 * 2, 32, cudaMemcpyDeviceToDevice));
  }
  void mtp() {
    a::run(query_ptr(), key.get(), value.get(), cache, norm_ptr(), base, rows,
           m::AttentionKind::global, output.get(), scratch.get<void>(), scratch.bytes(), nullptr);
  }
  void ordinary() {
    if (paged) {
      const gewell::compact_global_cache::PagedView<BF16> view{
          cache.page_pool, cache.page_offsets, cache.page_tokens, cache.page_count,
          cache.page_stride_elements, cache.layer_offset_elements};
      b::causal_gqa_attention_cached_m1_fused_global_compact_paged(query1_ptr(), view,
          norm_ptr(), base, scratch1.get<void>(), output1.get());
    } else {
      b::causal_gqa_attention_cached_m1_fused_global_compact(query1_ptr(), cache.key,
          norm_ptr(), base, context, scratch1.get<void>(), output1.get());
    }
  }
};

// Separate oracle: one CTA follows each production split's strided 32-key
// tiles, with two dimensions per thread. Scores and softmax metadata use FP64;
// P*V probabilities round once to BF16 after each tile maximum is known. It
// reconstructs keys from the public compact representation and never calls
// either production attention implementation. Only selected output rows/heads
// are computed, keeping the maximum-context oracle under 7 MiB.
__device__ double sum_block(double value, double* scratch) {
  const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (unsigned shift = 16; shift; shift /= 2)
    value += __shfl_down_sync(0xffffffffU, value, shift);
  if (!lane) scratch[warp] = value;
  __syncthreads();
  value = threadIdx.x < 8 ? scratch[lane] : 0;
  if (!warp) {
    for (unsigned shift = 16; shift; shift /= 2)
      value += __shfl_down_sync(0xffffffffU, value, shift);
    if (!lane) scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}
__global__ void reference_partials(const BF16* query, const BF16* staged_key,
    const BF16* staged_value, const BF16* norm, Cache cache,
    unsigned base, unsigned rows, unsigned splits, double* partials) {
  __shared__ double reduction[8];
  __shared__ double scores[a::kKeysPerTile];
  __shared__ double old_scale, weight, pv_weight;
  const unsigned selected = blockIdx.y, row = selected < 3 ? 0 : rows - 1;
  const unsigned hindex = selected % 3, head = hindex == 0 ? 0 : hindex == 1 ? 17 : 31;
  const unsigned kv_head = head / 8, d = threadIdx.x, last = base + row;
  const double q0 = __bfloat162float(query[(std::size_t(head) * rows + row) * 512 + d]);
  const double q1 = __bfloat162float(query[(std::size_t(head) * rows + row) * 512 + d + 256]);
  double numerator0 = 0, numerator1 = 0, maximum = -INFINITY, denominator = 0;
  for (unsigned begin = blockIdx.x * a::kKeysPerTile; begin <= last;
       begin += splits * a::kKeysPerTile) {
    const unsigned count = min(a::kKeysPerTile, last - begin + 1);
    for (unsigned token = 0; token < count; ++token) {
      const unsigned position = begin + token;
      const BF16* compact = position < base ? row_pointer(cache, kv_head, position) : nullptr;
      const BF16* value = compact ? compact + 128 :
          staged_value + (std::size_t(position - base) * 4 + kv_head) * 512;
      const BF16* key = compact ? compact :
          staged_key + (std::size_t(kv_head) * rows + position - base) * 512;
      const double v0 = __bfloat162float(value[d]);
      const double v1 = __bfloat162float(value[d + 256]);
      const double k0 = d < 64 ? __bfloat162float(key[d]) :
          __bfloat162float(__float2bfloat16_rn(float(v0) * __bfloat162float(norm[d])));
      const double k1 = d < 64 ? __bfloat162float(key[compact ? d + 64 : d + 256]) :
          __bfloat162float(__float2bfloat16_rn(float(v1) * __bfloat162float(norm[d + 256])));
      const double score = sum_block(q0 * k0 + q1 * k1, reduction);
      if (threadIdx.x == 0) scores[token] = score;
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      double next_maximum = maximum;
      for (unsigned token = 0; token < count; ++token)
        next_maximum = fmax(next_maximum, scores[token]);
      old_scale = exp(maximum - next_maximum);
      denominator *= old_scale;
      maximum = next_maximum;
    }
    __syncthreads();
    numerator0 *= old_scale;
    numerator1 *= old_scale;
    for (unsigned token = 0; token < count; ++token) {
      const unsigned position = begin + token;
      const BF16* compact = position < base ? row_pointer(cache, kv_head, position) : nullptr;
      const BF16* value = compact ? compact + 128 :
          staged_value + (std::size_t(position - base) * 4 + kv_head) * 512;
      if (threadIdx.x == 0) {
        weight = exp(scores[token] - maximum);
        pv_weight = __bfloat162float(__float2bfloat16_rn(float(weight)));
        denominator += weight;
      }
      __syncthreads();
      numerator0 += pv_weight * __bfloat162float(value[d]);
      numerator1 += pv_weight * __bfloat162float(value[d + 256]);
      __syncthreads();
    }
  }
  double* output = partials + (std::size_t(selected) * splits + blockIdx.x) * 514;
  output[d] = numerator0; output[d + 256] = numerator1;
  if (threadIdx.x == 0) { output[512] = maximum; output[513] = denominator; }
}

unsigned production_splits(unsigned context, unsigned rows) {
  unsigned splits = std::min(a::kMaxSplits,
      (context + a::kKeysPerTile - 1) / a::kKeysPerTile);
  if (context < 16384)
    splits = std::min(splits, 128U / std::min(rows, a::kQueryTileRows));
  return splits;
}

std::uint64_t hash_output(const std::vector<BF16>& values) {
  const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
  std::uint64_t hash = 14695981039346656037ULL;
  for (std::size_t i = 0; i < values.size() * 2; ++i) { hash ^= bytes[i]; hash *= 1099511628211ULL; }
  for (const auto value : values)
    require(std::isfinite(__bfloat162float(value)), "nonfinite attention output");
  return hash;
}
void validate(Fixture& fixture) {
  fixture.mtp();
  const auto actual = fixture.output.host<BF16>();
  const unsigned splits = production_splits(fixture.context, fixture.rows);
  Device partials(std::size_t(6) * splits * 514 * sizeof(double));
  reference_partials<<<dim3(splits, 6), 256>>>(fixture.query_ptr(), fixture.key.get(),
      fixture.value.get(), fixture.norm_ptr(), fixture.cache, fixture.base, fixture.rows,
      splits, partials.get<double>());
  check(cudaGetLastError());
  const auto reference = partials.host<double>();
  double maximum_error = 0, squared_error = 0, squared_reference = 0;
  for (unsigned selected = 0; selected < 6; ++selected) {
    double maximum = -std::numeric_limits<double>::infinity(), denominator = 0;
    std::vector<double> numerator(512);
    for (unsigned split = 0; split < splits; ++split)
      maximum = std::max(maximum, reference[(std::size_t(selected) * splits + split) * 514 + 512]);
    for (unsigned split = 0; split < splits; ++split) {
      const auto* part = reference.data() + (std::size_t(selected) * splits + split) * 514;
      const double weight = std::exp(part[512] - maximum);
      denominator += weight * part[513];
      for (unsigned d = 0; d < 512; ++d) numerator[d] += weight * part[d];
    }
    const unsigned row = selected < 3 ? 0 : fixture.rows - 1;
    const unsigned index = selected % 3, head = index == 0 ? 0 : index == 1 ? 17 : 31;
    for (unsigned d = 0; d < 512; ++d) {
      const double expected = __bfloat162float(__float2bfloat16_rn(float(numerator[d] / denominator)));
      const double result = __bfloat162float(actual[(std::size_t(row) * 32 + head) * 512 + d]);
      const double error = std::abs(result - expected);
      const double ulp = std::abs(expected) < std::numeric_limits<float>::min()
          ? std::ldexp(1.0, -133)
          : std::ldexp(1.0, std::ilogb(std::abs(expected)) - 7);
      if (!std::isfinite(result) || error > std::max(1e-6, ulp))
        throw std::runtime_error("BF16-P reference exceeds one BF16 ULP: row=" +
            std::to_string(row) + " head=" + std::to_string(head) + " dimension=" +
            std::to_string(d) + " actual=" + std::to_string(result) +
            " expected=" + std::to_string(expected));
      maximum_error = std::max(maximum_error, error);
      squared_error += error * error; squared_reference += expected * expected;
    }
  }
  const double relative_rms = std::sqrt(squared_error / std::max(1e-30, squared_reference));
  require(relative_rms <= 0.005,
          "BF16-P reference relative RMS exceeds 0.005");
  if (fixture.rows > 1) {
    poison_future<<<(4ULL * fixture.rows * 512 + 255) / 256, 256>>>(
        fixture.key.get(), fixture.value.get(), fixture.rows);
    fixture.mtp();
    const auto poisoned = fixture.output.host<BF16>();
    require(std::memcmp(actual.data(), poisoned.data(), 32 * 512 * sizeof(BF16)) == 0,
            "future staged NaN changes row zero");
    fixture.restore_inputs();
  }
  std::cout << "validation context=" << fixture.context << " rows=" << fixture.rows
            << " paged=" << fixture.paged << " outliers=" << fixture.outliers
            << " page_layers=" << (fixture.paged ? fixture.page_layers : 0)
            << " max_abs=" << maximum_error << " relative_rms=" << relative_rms
            << " reference=BF16-P-FP64 future_invariance=exact\n";
}

// Reading at least four L2 capacities evicts retained KV and partial results.
// The flush is enqueued before the start event and excluded from the timing.
__global__ void flush_l2(const std::uint32_t* arena, std::size_t count, std::uint32_t* sink) {
  std::uint32_t sum = 0;
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += std::size_t(gridDim.x) * blockDim.x) sum += arena[i];
  for (unsigned shift = 16; shift; shift /= 2)
    sum += __shfl_down_sync(0xffffffffU, sum, shift);
  if (threadIdx.x % 32 == 0) atomicAdd(sink, sum);
}
double measure(const std::function<void()>& operation, Device* flush, Device& sink) {
  for (unsigned i = 0; i < 3; ++i) operation();
  check(cudaDeviceSynchronize());
  cudaEvent_t begin{}, end{}; check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
  std::vector<double> samples;
  const unsigned repeats = flush ? 1 : 3;
  for (unsigned sample = 0; sample < 11; ++sample) {
    if (flush) flush_l2<<<4096, 256>>>(flush->get<std::uint32_t>(), flush->bytes() / 4,
                                      sink.get<std::uint32_t>());
    check(cudaEventRecord(begin));
    for (unsigned repeat = 0; repeat < repeats; ++repeat) operation();
    check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
    float milliseconds = 0; check(cudaEventElapsedTime(&milliseconds, begin, end));
    samples.push_back(milliseconds / repeats);
  }
  check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

std::vector<unsigned> numbers(const char* text) {
  std::vector<unsigned> result; std::istringstream input(text); std::string item;
  while (std::getline(input, item, ',')) {
    std::size_t used = 0; const auto value = std::stoul(item, &used);
    require(used == item.size() && value > 0 && value <= 262144, "invalid numeric list");
    result.push_back(static_cast<unsigned>(value));
  }
  require(!result.empty(), "empty numeric list"); return result;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::vector<unsigned> contexts{1024, 3072, 8192, 32768, 131072, 262144}, rows{1, 3, 4, 8};
    bool validation = false, outliers = false, misaligned = false;
    unsigned page_layers = 10;
    std::string layout = "both", mode = "both";
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--validate") validation = true;
      else if (option == "--outliers") outliers = true;
      else if (option == "--misaligned") misaligned = true;
      else {
        require(i + 1 < argc, "missing option value");
        if (option == "--contexts") contexts = numbers(argv[++i]);
        else if (option == "--rows") rows = numbers(argv[++i]);
        else if (option == "--layout") layout = argv[++i];
        else if (option == "--mode") mode = argv[++i];
        else if (option == "--page-layers") {
          const auto values = numbers(argv[++i]);
          require(values.size() == 1 && (values[0] == 1 || values[0] == 10),
                  "page layers must be 1 or 10");
          page_layers = values[0];
        }
        else throw std::invalid_argument("unknown option " + option);
      }
    }
    require(layout == "both" || layout == "contiguous" || layout == "paged", "invalid layout");
    require(mode == "both" || mode == "warm" || mode == "cold" || mode == "none", "invalid mode");
    cudaDeviceProp properties{}; check(cudaGetDeviceProperties(&properties, 0));
    const std::size_t flush_bytes = std::max(64ULL * 1024 * 1024,
                                            4ULL * properties.l2CacheSize);
    Device flush(flush_bytes), sink(4); check(cudaMemset(flush.get<void>(), 1, flush_bytes));
    check(cudaMemset(sink.get<void>(), 0, 4));
    std::cout << std::setprecision(8) << "device=" << properties.name
              << " l2_bytes=" << properties.l2CacheSize << " flush_bytes=" << flush_bytes
              << " timing=median11 flush_excluded=1\n";
    for (unsigned context : contexts) for (unsigned count : rows) {
      require(count <= 1280 && count <= context, "rows exceed context or kernel capacity");
      for (bool paged : {false, true}) {
        if ((paged && layout == "contiguous") || (!paged && layout == "paged")) continue;
        Fixture fixture(context, count, paged, outliers, misaligned, page_layers);
        if (validation) validate(fixture);
        fixture.mtp(); fixture.ordinary();
        const auto mtp_hash = hash_output(fixture.output.host<BF16>());
        const auto m1_hash = hash_output(fixture.output1.host<BF16>());
        for (bool cold : {false, true}) {
          if (mode == "none" || (cold && mode == "warm") || (!cold && mode == "cold")) continue;
          const double mtp_ms = measure([&] { fixture.mtp(); }, cold ? &flush : nullptr, sink);
          const double m1_ms = measure([&] { fixture.ordinary(); }, cold ? &flush : nullptr, sink);
          const double bytes_per_ms = double(4ULL * context * 640 * 2) / 1e6;
          std::cout << "benchmark context=" << context << " base=" << fixture.base
                    << " rows=" << count << " paged=" << paged << " misaligned=" << misaligned
                    << " page_layers=" << (paged ? page_layers : 0)
                    << " page_stride_bytes=" << (paged ? fixture.page_stride * 2 : 0)
                    << " outliers=" << outliers << " mode=" << (cold ? "cold" : "warm")
                    << " cache_bytes=" << fixture.cache_storage.bytes()
                    << " mtp_ms=" << mtp_ms << " ordinary_m1_ms=" << m1_ms
                    << " mtp_unique_KV_GBps=" << bytes_per_ms / mtp_ms
                    << " ordinary_unique_KV_GBps=" << bytes_per_ms / m1_ms
                    << " mtp_hash=" << mtp_hash << " ordinary_hash=" << m1_hash << '\n';
        }
      }
    }
    check(cudaDeviceSynchronize());
  } catch (const std::exception& error) {
    std::cerr << "compact attention benchmark failed: " << error.what() << '\n'; return 1;
  }
}
