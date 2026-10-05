#include "../kernels/kv_storage.cuh"
#include "gewell/mtp_target.h"

#include "gewell/bf16_primitives.h"
#include "gewell/mtp_attention.h"
#include "gewell/prefill_primitives.h"
#include "gewell/models/gemma4/31b/sm120/projection_fusion.h"
#include "cuda.cuh"
#include "norm.cuh"

#include <algorithm>
#include <cmath>
#include <map>

namespace gewell::mtp_target {
namespace {
namespace m = gemma4_31b;
namespace p = bf16_primitives;
namespace pf = prefill_primitives;
namespace fusion = gemma4_31b::sm120::fusion;
using namespace mtp_cuda;
constexpr std::size_t H = m::kHiddenSize, V = m::kVocabSize;
constexpr std::size_t Q = m::kQueryHeadCount * m::kGlobalHeadSize;
constexpr std::size_t K = m::kLocalKvHeadCount * m::kLocalHeadSize;
constexpr std::size_t F = m::kMlpSize;

std::uint32_t valid_rows(std::uint32_t rows) {
  if (!rows || rows > kMaxDepth + 1)
    throw std::invalid_argument("MTP verification rows must be in 1..1280");
  return rows;
}

struct Plans {
  Linear lq, lk, gq, gk, lqkv, gqkv, lo, go, up, gate_up, down, head;

  explicit Plans(std::uint32_t n)
      : lq(n, H, 8192),
        lk(n, H, 4096),
        gq(n, H, 16384),
        gk(n, H, 2048),
        lqkv(n, H, fusion::qkv_width(false)),
        gqkv(n, H, fusion::qkv_width(true)),
        lo(n, 8192, H),
        go(n, 16384, H),
        up(n, H, F),
        gate_up(n, H, 2 * F),
        down(n, F, H),
        head(n, H, V) {}
};

struct Scratch {
  std::size_t h0, h1, h2, qr, qn, qq, kr, kn, vr, context, gate, up,
      product, logits, lc, ls, gc, gs, bytes = 0;

  explicit Scratch(std::uint32_t rows) {
    auto add = [&](std::size_t count) {
      auto off = bytes;
      bytes = align(bytes + rows * count * sizeof(BFloat16));
      return off;
    };
    h0 = add(H);
    h1 = add(H);
    h2 = add(H);
    qr = add(fusion::kMaxQkvWidth);
    qn = add(Q);
    qq = add(Q);
    kr = add(K);
    kn = add(K);
    vr = add(K);
    context = add(Q);
    gate = add(F);
    up = add(F);
    product = add(F);
    logits = add(V);
    lc = add(256);
    ls = add(256);
    gc = add(512);
    gs = add(512);
  }
};

__global__ void softcap(BFloat16* logits, std::size_t count) {
  const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    // Match the target's separately rounded BF16 softcap operations.
    auto x = __float2bfloat16(__bfloat162float(logits[i]) / 30.0F);
    x = __float2bfloat16(tanhf(__bfloat162float(x)));
    logits[i] = __float2bfloat16(__bfloat162float(x) * 30.0F);
  }
}

__global__ void commit_local(const BFloat16* key, const BFloat16* value,
                             CacheView cache, std::uint32_t base,
                             std::uint32_t source_rows, std::uint32_t count) {
  if (cache.format == kv_cache::Format::fp8) {
    const unsigned head = blockIdx.x % 16, row = blockIdx.x / 16;
    if (row >= count || row + 1024 < count) return;
    const unsigned position = base + row;
    const auto* source_key = key + (std::size_t(head) * source_rows + row) * 256;
    const auto* source_value = value + (std::size_t(row) * 16 + head) * 256;
    const auto index = std::size_t(head) * 1024 + position % 1024;
    kv_storage::store_separate<256>(
        kv_storage::row(cache.key, index, 256, cache.format),
        kv_storage::row(cache.value, index, 256, cache.format), source_key, source_value);
    return;
  }

  const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= std::size_t(count) * 4096) return;
  const auto d = i % 256, head = (i / 256) % 16, row = i / 4096;
  if (row + 1024 < count) return;  // Only the final writer to each ring slot commits.
  const auto dest = (head * 1024 + (base + row) % 1024) * 256 + d;
  cache.key[dest] = key[(head * source_rows + row) * 256 + d];
  cache.value[dest] = value[i];
}

__global__ void commit_global(const BFloat16* key, const BFloat16* value,
                              CacheView cache, std::uint32_t base,
                              std::uint32_t source_rows, std::uint32_t count) {
  if (cache.format == kv_cache::Format::fp8) {
    const unsigned head = blockIdx.x % 4, row = blockIdx.x / 4;
    if (row >= count) return;
    const unsigned position = base + row;
    const auto* source_key = key + (std::size_t(head) * source_rows + row) * 512;
    const auto* source_value = value + (std::size_t(row) * 4 + head) * 512;
    auto* dest = cache.page_pool ? compact_global_cache::paged_row(
        cache.page_pool, cache.page_offsets, cache.page_tokens,
        cache.layer_offset_elements, head, position, cache.format)
        : kv_storage::row(cache.key, std::size_t(head) * cache.capacity + position,
                           640, cache.format, 2);
    kv_storage::store_compact(dest, source_key, source_value);
    return;
  }

  const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= std::size_t(count) * 4 * 640) return;
  const auto d = i % 640, head = (i / 640) % 4, row = i / (640 * 4),
             position = base + row;
  BFloat16* dest =
      cache.page_pool
          ? compact_global_cache::paged_row(
                cache.page_pool, cache.page_offsets, cache.page_tokens,
                cache.layer_offset_elements, head, position)
          : cache.key + (head * cache.capacity + position) * 640;
  if (d < 128) {
    const auto kd = d < 64 ? d : d + 192;
    dest[d] = key[(head * source_rows + row) * 512 + kd];
  } else
    dest[d] = value[(row * 4 + head) * 512 + d - 128];
}

constexpr unsigned kCommitBatchEntries = 32;
struct CommitJob {
  const BFloat16* key{};
  const BFloat16* value{};
  CacheView cache{};
  std::uint32_t base{}, source_rows{}, count{};
};
struct CommitBatch { CommitJob jobs[kCommitBatchEntries]; };

__global__ void commit_local_batch(const __grid_constant__ CommitBatch batch) {
  const auto& job = batch.jobs[blockIdx.y];

  if (job.cache.format == kv_cache::Format::fp8) {
    const unsigned head = blockIdx.x % 16, row = blockIdx.x / 16;
    if (row >= job.count || row + 1024 < job.count) return;
    const unsigned position = job.base + row;
    const auto* source_key = job.key + (std::size_t(head) * job.source_rows + row) * 256;
    const auto* source_value = job.value + (std::size_t(row) * 16 + head) * 256;
    const auto index = std::size_t(head) * 1024 + position % 1024;
    kv_storage::store_separate<256>(
        kv_storage::row(job.cache.key, index, 256, job.cache.format),
        kv_storage::row(job.cache.value, index, 256, job.cache.format), source_key, source_value);
    return;
  }
  const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= std::size_t(job.count) * 4096) return;
  const auto d = i % 256, head = (i / 256) % 16, row = i / 4096;
  if (row + 1024 < job.count) return;
  const auto dest = (head * 1024 + (job.base + row) % 1024) * 256 + d;
  job.cache.key[dest] =
      job.key[(head * job.source_rows + row) * 256 + d];
  job.cache.value[dest] = job.value[i];
}

__global__ void commit_global_batch(const __grid_constant__ CommitBatch batch) {
  const auto& job = batch.jobs[blockIdx.y];

  if (job.cache.format == kv_cache::Format::fp8) {
    const unsigned head = blockIdx.x % 4, row = blockIdx.x / 4;
    if (row >= job.count) return;
    const unsigned position = job.base + row;
    const auto* source_key = job.key + (std::size_t(head) * job.source_rows + row) * 512;
    const auto* source_value = job.value + (std::size_t(row) * 4 + head) * 512;
    auto* dest = job.cache.page_pool ? compact_global_cache::paged_row(
        job.cache.page_pool, job.cache.page_offsets, job.cache.page_tokens,
        job.cache.layer_offset_elements, head, position, job.cache.format)
        : kv_storage::row(job.cache.key, std::size_t(head) * job.cache.capacity + position,
                           640, job.cache.format, 2);
    kv_storage::store_compact(dest, source_key, source_value);
    return;
  }
  const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= std::size_t(job.count) * 4 * 640) return;
  const auto d = i % 640, head = (i / 640) % 4, row = i / (640 * 4),
             position = job.base + row;
  BFloat16* dest =
      job.cache.page_pool
          ? compact_global_cache::paged_row(
                job.cache.page_pool, job.cache.page_offsets,
                job.cache.page_tokens, job.cache.layer_offset_elements,
                head, position)
          : job.cache.key +
                (head * job.cache.capacity + position) * 640;
  if (d < 128) {
    const auto kd = d < 64 ? d : d + 192;
    dest[d] = job.key[(head * job.source_rows + row) * 512 + kd];
  } else {
    dest[d] = job.value[(row * 4 + head) * 512 + d - 128];
  }
}

void validate_commit(const Caches& caches, std::uint32_t base,
                     std::uint32_t source_rows,
                     std::uint32_t capacity_rows, const void* staging,
                     std::size_t staging_size, std::uint32_t count) {
  valid_rows(source_rows);
  valid_rows(capacity_rows);
  if (!staging || !count || count > source_rows ||
      source_rows > capacity_rows ||
      staging_size < Verifier::staging_bytes(capacity_rows) ||
      std::uint64_t(base) + count > p::kMaxContextTokenCount)
    throw std::invalid_argument("MTP staged commit outside capacity");
  const std::uint64_t end = std::uint64_t(base) + count;
  for (std::uint32_t i = 0; i < m::kLayerCount; ++i) {
    const auto& cache = caches[i];
    if (!m::is_global_layer(i)) {
      if (!cache.key || !cache.value || cache.capacity != 1024)
        throw std::invalid_argument(
            "MTP local commit requires separate 1024-row rings");
    } else if (cache.page_pool) {
      const std::size_t layer_elements = 4 * 256 * kv_cache::row_words(640, cache.format, 2);
      if (!cache.page_offsets || cache.page_tokens != 256 ||
          (end + 255) / 256 > cache.page_count ||
          cache.layer_offset_elements > cache.page_stride_elements ||
          layer_elements >
              cache.page_stride_elements - cache.layer_offset_elements)
        throw std::invalid_argument(
            "MTP paged commit does not cover accepted rows");
    } else if (!cache.key || end > cache.capacity) {
      throw std::invalid_argument(
          "MTP contiguous commit does not cover accepted rows");
    }
  }
}
}  // namespace

struct Verifier::Impl {
  cublasLtHandle_t handle;
  Weights weights;
  nvfp4::Weights native_weights{};
  std::unique_ptr<nvfp4::ProjectionPlans> nvfp4_projections;
  fp8::Weights fp8_weights{};
  std::unique_ptr<fp8::ProjectionPlans> fp8_projections;
  std::uint32_t capacity, context_capacity, rows = 0, base = 0;
  Scratch layout;
  Buffer scratch;
  Buffer attention_scratch;
  std::map<std::uint32_t, std::unique_ptr<Plans>> plans;
  BFloat16* staged{};
  attention::Compute local_compute, global_compute;

  Impl(cublasLtHandle_t h, const Weights& w, std::uint32_t cap,
       std::uint32_t context, const nvfp4::Weights* native,
       nvfp4::ActivationPolicy activation_policy, const fp8::Weights* native_fp8,
       attention::Compute local, attention::Compute global)
      : handle(h),
        weights(w),
        capacity(valid_rows(cap)),
        context_capacity(context),
        layout(cap),
        scratch(layout.bytes),
        attention_scratch(mtp_attention::scratch_bytes(cap, context)),
        local_compute(local), global_compute(global) {
    if (!handle)
      throw std::invalid_argument("MTP target requires cuBLAS handle");
    if (native) native_weights = *native;
    if (native_fp8) fp8_weights = *native_fp8;
    bool has_native = false, has_fp8 = false;
    for (std::size_t i = 0; i < m::kTextPhysicalTensorCount; ++i) {
      const auto representations = unsigned(weights[i] != nullptr) +
          unsigned(native_weights[i].data != nullptr) +
          unsigned(fp8_weights[i].data != nullptr);
      if (representations != 1)
        throw std::invalid_argument("MTP target requires exactly one weight representation");
      const auto role = m::kPhysicalTensors[i].role;
      if (native_weights[i].data || fp8_weights[i].data) {
        if (role != m::TensorRole::q_proj && role != m::TensorRole::k_proj &&
            role != m::TensorRole::v_proj && role != m::TensorRole::o_proj &&
            role != m::TensorRole::gate_proj && role != m::TensorRole::up_proj &&
            role != m::TensorRole::down_proj)
          throw std::invalid_argument("invalid native MTP target projection");
      }
      if (native_weights[i].data) {
        has_native = true;
      }
      if (fp8_weights[i].data) {
        const auto& weight = fp8_weights[i];
        if (!(weight.input_scale > 0.0F) || !std::isfinite(weight.input_scale) ||
            !(weight.weight_scale > 0.0F) || !std::isfinite(weight.weight_scale))
          throw std::invalid_argument("FP8 MTP target scales must be finite and positive");
        has_fp8 = true;
      }
    }
    if (has_native) {
      nvfp4_projections = std::make_unique<nvfp4::ProjectionPlans>(handle, native_weights, activation_policy);
      nvfp4_projections->prepare(capacity);
    }
    if (has_fp8) {
      fp8_projections = std::make_unique<fp8::ProjectionPlans>(handle, fp8_weights);
      fp8_projections->prepare(capacity);
    }
    plans.emplace(capacity, std::make_unique<Plans>(capacity));
  }

  BFloat16* at(std::size_t off) { return scratch.at<BFloat16>(off); }
};

Verifier::Verifier(cublasLtHandle_t h, const Weights& w, std::uint32_t rows,
                   std::uint32_t context_capacity,
                   const nvfp4::Weights* native_weights,
                   nvfp4::ActivationPolicy activation_policy,
                   const fp8::Weights* fp8_weights,
                   attention::Compute local_compute, attention::Compute global_compute)
    : impl_(std::make_unique<Impl>(h, w, rows, context_capacity, native_weights,
                                   activation_policy, fp8_weights, local_compute, global_compute)) {}

Verifier::~Verifier() = default;

std::size_t Verifier::staging_bytes(std::uint32_t rows) {
  valid_rows(rows);
  return std::size_t(rows) * 2 * (50 * 4096 + 10 * 2048) * sizeof(BFloat16);
}

std::size_t Verifier::scratch_bytes() const {
  return impl_->scratch.size() + impl_->attention_scratch.size() +
         (impl_->nvfp4_projections ? impl_->nvfp4_projections->scratch_bytes() : 0) +
         (impl_->fp8_projections ? impl_->fp8_projections->scratch_bytes() : 0);
}

const BFloat16* Verifier::logits() const {
  return impl_->scratch.at<BFloat16>(impl_->layout.logits);
}

const BFloat16* Verifier::hidden() const {
  return impl_->scratch.at<BFloat16>(impl_->layout.h1);
}

void Verifier::prepare(std::uint32_t rows) {
  valid_rows(rows);
  if (rows > impl_->capacity)
    throw std::invalid_argument("MTP target rows exceed capacity");
  if (impl_->nvfp4_projections) impl_->nvfp4_projections->prepare(rows);
  if (impl_->fp8_projections) impl_->fp8_projections->prepare(rows);
  if (!impl_->plans.count(rows))
    impl_->plans.emplace(rows, std::make_unique<Plans>(rows));
}

void Verifier::run(const std::uint32_t* tokens, std::uint32_t base,
                   std::uint32_t rows, const Caches& caches, void* staging,
                   std::size_t bytes, cudaStream_t stream) {
  auto& s = *impl_;
  run_batch(tokens, {{base, rows, caches, staging, bytes, s.capacity}}, stream);
  s.rows = rows;
  s.base = base;
  s.staged = static_cast<BFloat16*>(staging);
}

void Verifier::run_batch(const std::uint32_t* tokens,
                         const std::vector<BatchInput>& inputs,
                         cudaStream_t stream) {
  auto& s = *impl_;
  const auto& l = s.layout;
  // A failed or batched verification must not leave a previous serial commit
  // target live. Batched callers commit each private staging buffer explicitly.
  s.rows = 0;
  s.base = 0;
  s.staged = nullptr;
  if (!tokens || inputs.empty())
    throw std::invalid_argument("MTP target batch requires tokens and requests");
  std::uint32_t rows = 0;
  bool capture_layers = false;
  for (const auto& input : inputs) {
    valid_rows(input.rows);
    valid_rows(input.staging_capacity_rows);
    if (!input.staging || input.rows > input.staging_capacity_rows ||
        input.rows > s.capacity - rows ||
        input.staging_size < staging_bytes(input.staging_capacity_rows) ||
        input.base_position == 0 ||
        std::uint64_t(input.base_position) + input.rows > s.context_capacity)
      throw std::invalid_argument("MTP target inputs or staging outside capacity");
    for (const auto& capture : input.captures) {
      if (!capture.output || !capture.completed_layers || capture.completed_layers > m::kLayerCount)
        throw std::invalid_argument("MTP target capture layer outside model");
      capture_layers = true;
    }
    rows += input.rows;
  }
  prepare(rows);
  auto& plan = *s.plans.at(rows);
  auto *h0 = s.at(l.h0), *h1 = s.at(l.h1), *h2 = s.at(l.h2),
       *qr = s.at(l.qr), *qn = s.at(l.qn), *qq = s.at(l.qq),
       *kr = s.at(l.kr), *kn = s.at(l.kn), *vr = s.at(l.vr),
       *context = s.at(l.context), *gate = s.at(l.gate), *up = s.at(l.up),
       *product = s.at(l.product);
  std::size_t first_row = 0;
  for (const auto& input : inputs) {
    pf::generate_rope_factors_chunk(
        s.at(l.lc) + first_row * 256, s.at(l.ls) + first_row * 256,
        s.at(l.gc) + first_row * 512, s.at(l.gs) + first_row * 512,
        input.base_position, input.rows, stream);
    first_row += input.rows;
  }
  p::embedding_lookup_device_tokens(s.weights[0], tokens, h0, rows, stream);
  p::rms_norm(h0, s.weights[1], h1, rows, H, 1e-6F, stream);
  const bool batched = inputs.size() > 1;
  std::vector<mtp_attention::BatchInput> attention_inputs(
      batched || s.local_compute == attention::Compute::fp8 || s.global_compute == attention::Compute::fp8
          ? inputs.size() : 0);
  std::vector<p::QkvRopeInput> rope_inputs(batched ? inputs.size() : 0);
  std::size_t staging_layer_elements = 0;
  for (std::uint32_t i = 0; i < m::kLayerCount; ++i) {
    const bool global = m::is_global_layer(i);
    const bool fp8_attention = (global ? s.global_compute : s.local_compute) == attention::Compute::fp8;
    const auto kind = global ? m::AttentionKind::global : m::AttentionKind::local;
    const std::uint32_t d = global ? 512 : 256, heads = global ? 4 : 16;
    const std::size_t b = 1 + 14 * i - i / 6;
    const auto weight = [&](std::size_t offset) { return s.weights[b + offset]; };
    const auto projection = [&](std::size_t offset, std::uint32_t input_width,
                                std::uint32_t output_width, const Linear& bf16,
                                const BFloat16* input, BFloat16* output) {
      const auto& fp8 = s.fp8_weights[b + offset];
      const auto& native = s.native_weights[b + offset];
      if (fp8.data)
        s.fp8_projections->run(rows, input_width, output_width, input, fp8,
                               output, stream);
      else if (native.data && s.nvfp4_projections->fp4_activations(nvfp4::Phase::decode))
        s.nvfp4_projections->run(rows, input_width, output_width, input, native,
                                output, stream);
      else {
        const auto* bf16_weight = native.data
            ? s.nvfp4_projections->decode_weight(native, input_width, output_width, stream)
            : weight(offset);
        bf16.run(s.handle, input, bf16_weight, output, stream);
      }
    };
    const std::size_t shift = global ? 0 : 1;
    const auto* cos = s.at(global ? l.gc : l.lc);
    const auto* sin = s.at(global ? l.gs : l.ls);

    const auto* qkv_weight = fusion::qkv_weight(s.weights, i);
    const auto fp8_qkv = fp8::qkv_weight(s.fp8_weights, i);
    const bool joined_qkv = qkv_weight || fp8_qkv.data;
    if (qkv_weight) {
      (global ? plan.gqkv : plan.lqkv).run(s.handle, h1, qkv_weight, qr, stream);
    } else if (fp8_qkv.data) {
      s.fp8_projections->run_joined(rows, fusion::qkv_width(global), h1, fp8_qkv, qr, stream);
    } else {
      projection(1, H, 32 * d, global ? plan.gq : plan.lq, h1, qr);
      projection(2, H, heads * d, global ? plan.gk : plan.lk, h1, kr);
      if (!global) projection(3, H, heads * d, plan.lk, h1, vr);
      p::rms_norm(qr, weight(3 + shift), qn, rows * 32, d, 1e-6F, stream);
      p::rms_norm(kr, weight(4 + shift), kn, rows * heads, d, 1e-6F, stream);
    }
    first_row = 0;
    std::size_t request = 0;
    for (const auto& input : inputs) {
      auto* key = static_cast<BFloat16*>(input.staging) +
                  staging_layer_elements * input.staging_capacity_rows;
      auto* value = key + std::size_t(input.staging_capacity_rows) * heads * d;
      const auto query_offset = first_row * 32 * d;
      const auto kv_offset = first_row * heads * d;
      const auto rope_offset = first_row * d;
      auto* query = qq + query_offset;
      if (joined_qkv && batched) {
        rope_inputs[request] = {qr + first_row * fusion::qkv_width(global),
            cos + rope_offset, sin + rope_offset, query, key, value, input.rows};
      } else {
        if (joined_qkv)
          p::qkv_rms_norm(qr + first_row * fusion::qkv_width(global),
                          weight(3 + shift), weight(4 + shift), qn + query_offset,
                          kn + kv_offset, value, input.rows, kind, stream);
        else
          p::rms_norm_unscaled((global ? kr : vr) + kv_offset, value,
                               input.rows * heads, d, 1e-6F, stream);
        // Q/K head strides must use this request's row count.
        pf::apply_rope_transpose_chunk(qn + query_offset, cos + rope_offset,
                                       sin + rope_offset, query, 32, input.rows,
                                       kind, stream);
        pf::apply_rope_transpose_chunk(kn + kv_offset, cos + rope_offset,
                                       sin + rope_offset, key, heads, input.rows,
                                       kind, stream);
      }
      if (batched || fp8_attention)
        attention_inputs[request++] = {query, key, value, input.caches[i],
            input.base_position, input.rows, context + query_offset};
      else
        mtp_attention::run(query, key, value, input.caches[i], global ? weight(4) : nullptr,
            input.base_position, input.rows, kind, context + query_offset,
            s.attention_scratch.data(), s.attention_scratch.size(), stream);
      first_row += input.rows;
    }
    if (joined_qkv && batched)
      p::qkv_rms_rope_batch(rope_inputs, weight(3 + shift), weight(4 + shift), kind, stream);
    if (fp8_attention)
      mtp_attention::run_fp8_batch(attention_inputs, global ? weight(4) : nullptr, kind,
          s.attention_scratch.data(), s.attention_scratch.size(), stream);
    else if (batched)
      mtp_attention::run_batch(attention_inputs, global ? weight(4) : nullptr, kind,
          s.attention_scratch.data(), s.attention_scratch.size(), stream);

    projection(5 + shift, 32 * d, H, global ? plan.go : plan.lo, context, h2);
    detail::residual_norm<false><<<rows, detail::kNormThreads, 0, stream>>>(
        h2, weight(6 + shift), h0, nullptr, weight(7 + shift), h1);
    check(cudaGetLastError(), "MTP post-attention residual norm");
    const auto joined = nvfp4::gate_up_weight(s.native_weights[b + 8 + shift],
                                             s.native_weights[b + 9 + shift]);
    const auto* bf16_joined = fusion::gate_up_weight(weight(8 + shift), weight(9 + shift), rows);
    const auto fp8_joined = fp8::gate_up_weight(s.fp8_weights[b + 8 + shift],
                                               s.fp8_weights[b + 9 + shift]);
    bool interleaved = true;
    if (joined.data && s.nvfp4_projections->fp4_activations(nvfp4::Phase::decode)) {
      s.nvfp4_projections->run(rows, H, 2 * F, h1, joined, gate, stream);
    } else if (fp8_joined.data) {
      s.fp8_projections->run_joined(rows, 2 * F, h1, fp8_joined, gate, stream);
    } else if (bf16_joined) {
      plan.gate_up.run(s.handle, h1, bf16_joined, gate, stream);
    } else {
      projection(8 + shift, H, F, plan.up, h1, gate);
      projection(9 + shift, H, F, plan.up, h1, up);
      interleaved = false;
    }
    const auto down = s.native_weights[b + 10 + shift];
    const auto fp8_down = s.fp8_weights[b + 10 + shift];
    if (interleaved && fp8_down.data) {
      s.fp8_projections->run(rows, F, H, gate, fp8_down, h0, stream,
                            fp8::InputTransform::gelu_tanh_multiply);
    } else if (interleaved && down.data && s.nvfp4_projections->fp4_activations(nvfp4::Phase::decode)) {
      s.nvfp4_projections->run(rows, F, H, gate, down, h0, stream,
                              nvfp4::InputTransform::gelu_tanh_multiply);
    } else {
      if (interleaved) p::gelu_tanh_multiply_interleaved(gate, product, rows, F, stream);
      else p::gelu_tanh_multiply(gate, up, product, rows * F, stream);
      projection(10 + shift, F, H, plan.down, product, h0);
    }
    const std::size_t next_norm = i + 1 == m::kLayerCount
        ? m::kFinalNormPhysicalId : 1 + 14 * (i + 1) - (i + 1) / 6;
    detail::residual_norm<true><<<rows, detail::kNormThreads, 0, stream>>>(
        h0, weight(11 + shift), h2, weight(12 + shift), s.weights[next_norm], h1);
    check(cudaGetLastError(), "MTP post-feedforward residual norm");
    if (capture_layers) {
      std::size_t offset = 0;
      for (const auto& input : inputs) {
        for (const auto& capture : input.captures)
          if (capture.completed_layers == i + 1)
            check(cudaMemcpyAsync(capture.output, h0 + offset * H,
                std::size_t(input.rows) * H * sizeof(BFloat16), cudaMemcpyDeviceToDevice, stream),
                "capture target layer residual");
        offset += input.rows;
      }
    }
    staging_layer_elements += 2 * heads * d;
  }
  plan.head.run(s.handle, h1, s.weights[0], s.at(l.logits), stream);
  softcap<<<(std::size_t(rows) * V + 255) / 256, 256, 0, stream>>>(
      s.at(l.logits), std::size_t(rows) * V);
  check(cudaGetLastError(), "MTP target softcap");
}

void Verifier::commit(const Caches& caches, std::uint32_t base,
                      std::uint32_t count, cudaStream_t stream) {
  const auto& s = *impl_;
  if (!s.staged || !count || count > s.rows || base != s.base)
    throw std::invalid_argument("MTP target commit differs from verification");
  commit_staged_rows(caches, base, s.rows, s.capacity, s.staged,
                     staging_bytes(s.capacity), count, stream);
}

void commit_staged_rows(const Caches& caches, std::uint32_t base,
                         std::uint32_t source_rows,
                         std::uint32_t capacity_rows, const void* staging,
                         std::size_t staging_size, std::uint32_t count,
                         cudaStream_t stream) {
  validate_commit(caches, base, source_rows, capacity_rows, staging,
                  staging_size, count);
  const auto* staged = static_cast<const BFloat16*>(staging);
  std::size_t offset = 0;
  for (std::uint32_t i = 0; i < m::kLayerCount; ++i) {
    const bool global = m::is_global_layer(i);
    const auto elements = global ? 2048 : 4096;
    const auto* key = staged + offset;
    const auto* value = key + std::size_t(capacity_rows) * elements;
    if (global)
      commit_global<<<(std::size_t(count) * 2560 + 255) / 256, 256, 0, stream>>>(
          key, value, caches[i], base, source_rows, count);
    else
      commit_local<<<(std::size_t(count) * 4096 + 255) / 256, 256, 0, stream>>>(
          key, value, caches[i], base, source_rows, count);
    check(cudaGetLastError(), "MTP accepted KV commit");
    offset += 2 * std::size_t(capacity_rows) * elements;
  }
}

void commit_staged_rows_batch(const std::vector<CommitInput>& inputs,
                              cudaStream_t stream) {
  if (inputs.empty())
    throw std::invalid_argument("MTP staged commit batch is empty");
  for (const auto& input : inputs) {
    if (!input.caches)
      throw std::invalid_argument("MTP staged commit has null caches");
    validate_commit(*input.caches, input.base_position, input.source_rows,
                    input.capacity_rows, input.staging, input.staging_size,
                    input.commit_rows);
  }

  std::vector<std::size_t> offsets(inputs.size());
  for (std::uint32_t layer = 0; layer < m::kLayerCount; ++layer) {
    const bool global = m::is_global_layer(layer);
    const std::size_t elements = global ? 2048 : 4096;
    for (std::size_t first = 0; first < inputs.size();
         first += kCommitBatchEntries) {
      const auto count = static_cast<unsigned>(std::min<std::size_t>(
          kCommitBatchEntries, inputs.size() - first));
      CommitBatch batch{};
      std::uint32_t max_rows = 0;
      for (unsigned i = 0; i < count; ++i) {
        const auto index = first + i;
        const auto& input = inputs[index];
        const auto* key = static_cast<const BFloat16*>(input.staging) +
                          offsets[index];
        batch.jobs[i] = {
            key,
            key + std::size_t(input.capacity_rows) * elements,
            (*input.caches)[layer],
            input.base_position,
            input.source_rows,
            input.commit_rows,
        };
        max_rows = std::max(max_rows, input.commit_rows);
      }
      const auto blocks = static_cast<unsigned>(
          (std::size_t(max_rows) * (global ? 2560 : 4096) + 255) / 256);
      if (global)
        commit_global_batch<<<dim3(blocks, count), 256, 0, stream>>>(batch);
      else
        commit_local_batch<<<dim3(blocks, count), 256, 0, stream>>>(batch);
      check(cudaGetLastError(), "MTP accepted KV batch commit");
    }
    for (std::size_t i = 0; i < inputs.size(); ++i)
      offsets[i] += 2 * std::size_t(inputs[i].capacity_rows) * elements;
  }
}

}  // namespace gewell::mtp_target
