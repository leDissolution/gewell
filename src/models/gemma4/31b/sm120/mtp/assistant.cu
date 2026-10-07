#include "gewell/mtp_assistant.h"

#include "gewell/bf16_primitives.h"
#include "gewell/mtp_attention.h"
#include "gewell/models/gemma4/26b_a4b/model.h"
#include "cuda.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>

namespace gewell::mtp_assistant {
namespace {

namespace model = gemma4_31b;
namespace primitives = bf16_primitives;
constexpr std::uint32_t kHidden = model::kAssistantHiddenSize;
constexpr std::uint32_t kMlp = model::kAssistantMlpSize;
static_assert(model::kAssistantPhysicalTensorCount == 48);
static_assert(model::kAssistantLayerCount == 4 && model::kAssistantLayerTensorCount == 11);
static_assert(model::kAssistantEmbeddingPhysicalId == 1188 && model::kAssistantFinalNormPhysicalId == 1233);
static_assert(model::kAssistantPreProjectionPhysicalId == 1234 && model::kAssistantPostProjectionPhysicalId == 1235);
constexpr std::uint32_t kThreads = 256;

void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(std::string("MTP assistant: ") + message);
}

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("MTP assistant ") + operation + ": " + cudaGetErrorString(status));
}

void validate_cache(const FrozenCache& cache, model::AttentionKind kind, unsigned heads = 32) {
  require(cache.processed_tokens > 0 && cache.processed_tokens < 262'144,
          "frozen prefix must leave a valid pending position");
  if (kind == model::AttentionKind::local) {
    require(cache.local_key && cache.local_value,
            "a complete separate local K/V cache view is required");
    return;
  }
  require(kind == model::AttentionKind::global, "invalid attention kind");
  const bool paged = cache.global.page_pool != nullptr;
  require(paged != (cache.global_compact != nullptr),
          "exactly one compact global cache view is required");
  if (paged) {
    require(cache.global.page_offsets && cache.global.page_tokens == 256 &&
                cache.global.page_count >= (cache.processed_tokens + 255) / 256,
            "global page table does not cover the frozen prefix");
    const std::size_t layer_elements = static_cast<std::size_t>(heads / 8) * 256 * kv_cache::row_words(640, cache.global.format, 2);
    require(cache.global.page_stride_elements >= layer_elements &&
                cache.global.layer_offset_elements <= cache.global.page_stride_elements - layer_elements,
            "global layer view exceeds a page");
  } else {
    require(cache.global_capacity >= cache.processed_tokens && cache.global_capacity <= 262'144,
            "contiguous global capacity does not cover the frozen prefix");
  }
}

kv_cache::DeviceView global_cache_view(const FrozenCache& cache) {
  kv_cache::DeviceView view{};
  view.key = const_cast<BFloat16*>(cache.global_compact);
  view.capacity = cache.global_capacity;
  view.page_pool = cache.global.page_pool;
  view.page_offsets = const_cast<std::uint64_t*>(cache.global.page_offsets);
  view.page_tokens = cache.global.page_tokens;
  view.page_count = cache.global.page_count;
  view.page_stride_elements = cache.global.page_stride_elements;
  view.layer_offset_elements = cache.global.layer_offset_elements;
  view.format = cache.global.page_pool ? cache.global.format : cache.global_format;
  return view;
}

kv_cache::DeviceView local_cache_view(const FrozenCache& cache) {
  kv_cache::DeviceView view{};
  view.key = const_cast<BFloat16*>(cache.local_key);
  view.value = const_cast<BFloat16*>(cache.local_value);
  view.capacity = 1024;
  view.format = cache.local_format;
  return view;
}

// 26B target embedding scale is already rounded to BF16 (53). The
// assistant always uses this target table, not its own tied output table.
struct Embedding26Input {
  const std::uint32_t* token;
  const BFloat16* hidden;
  BFloat16* output;
};
constexpr unsigned kEmbedding26Batch = 128;
struct Embedding26Batch { Embedding26Input inputs[kEmbedding26Batch]; };

__global__ void embedding_26b(const BFloat16* table,
                              const __grid_constant__ Embedding26Batch batch) {
  constexpr unsigned width = gemma4_26b_a4b::kHiddenSize;
  const auto& input = batch.inputs[blockIdx.y];
  const unsigned d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= width) return;
  const auto token = *input.token;
  if (token < gemma4_26b_a4b::kVocabSize)
    input.output[d] = __float2bfloat16_rn(
        __bfloat162float(table[std::size_t(token) * width + d]) *
        gemma4_26b_a4b::kEmbeddingScale);
  input.output[width + d] = input.hidden[d];
}

__device__ float block_reduce(float value, float* values) {
  values[threadIdx.x] = value;
  __syncthreads();
  for (unsigned delta = kThreads / 2; delta; delta /= 2) {
    if (threadIdx.x < delta)
      values[threadIdx.x] += values[threadIdx.x + delta];
    __syncthreads();
  }
  return values[0];
}

__global__ void hidden_norm(const BFloat16* input, const BFloat16* scale,
                            BFloat16* output) {
  input += std::size_t(blockIdx.x) * kHidden;
  output += std::size_t(blockIdx.x) * kHidden;
  __shared__ float reduction[kThreads];
  float sum = 0.0F;
  for (unsigned i = threadIdx.x; i < kHidden; i += kThreads) {
    const float x = __bfloat162float(input[i]);
    sum += x * x;
  }
  sum = block_reduce(sum, reduction);
  const float inverse = powf(sum / static_cast<float>(kHidden) + 1.0e-6F, -0.5F);
  for (unsigned i = threadIdx.x; i < kHidden; i += kThreads)
    output[i] = __float2bfloat16_rn(__bfloat162float(input[i]) * inverse * __bfloat162float(scale[i]));
}

void norm(const BFloat16* input, const BFloat16* scale, BFloat16* output,
          unsigned rows, cudaStream_t stream) {
  hidden_norm<<<rows, kThreads, 0, stream>>>(input, scale, output);
  check_cuda(cudaGetLastError(), "hidden normalization launch");
}

}  // namespace

std::size_t attention_scratch_bytes(std::uint32_t context_capacity) {
  require(context_capacity > 0 && context_capacity <= 262'144,
          "context capacity outside 1..262144");
  return std::max(mtp_attention::scratch_bytes(1, context_capacity),
      primitives::causal_gqa_attention_cached_m1_fused_scratch_bytes(1024, model::AttentionKind::local));
}

static std::size_t batch_attention_scratch_bytes(
    std::uint32_t context_capacity, std::uint32_t batch_capacity) {
  const auto local =
      primitives::causal_gqa_attention_cached_m1_fused_scratch_bytes(
          1024, model::AttentionKind::local);
  return std::max(attention_scratch_bytes(context_capacity),
                  local * std::min(batch_capacity, 32U));
}

void attend_frozen_prefix(const BFloat16* query, const FrozenCache& cache,
                         const BFloat16* target_global_k_norm,
                         model::AttentionKind kind, void* scratch,
                         BFloat16* context, cudaStream_t stream) {
  require(query && scratch && context, "null frozen attention input/output");
  validate_cache(cache, kind);
  if (kind == model::AttentionKind::global) {
    require(target_global_k_norm, "global KNorm scale is required");
    mtp_attention::run_frozen_global(query, global_cache_view(cache),
        target_global_k_norm,
        cache.processed_tokens, context, scratch,
        attention_scratch_bytes(cache.processed_tokens), stream);
  } else {
    // The last committed position is L-1. Ordinary local decode therefore
    // exposes exactly the assistant's frozen [max(0,L-1024),L) window.
    primitives::causal_gqa_attention_cached_m1_fused(query, cache.local_key,
        cache.local_value, cache.processed_tokens - 1, 1024, scratch, context,
        model::AttentionKind::local, stream, cache.local_format);
  }
  check_cuda(cudaGetLastError(), "attention finalizer launch");
}

class Executor::Impl {
 public:
  struct Plans {
    mtp_cuda::Linear pre, local_query, global_query, local_out, global_out,
        up, down, head, post;
    Plans(unsigned rows, unsigned target_hidden, unsigned heads)
        : pre(rows, 2 * target_hidden, kHidden),
          local_query(rows, kHidden, heads * 256),
          global_query(rows, kHidden, heads * 512),
          local_out(rows, heads * 256, kHidden),
          global_out(rows, heads * 512, kHidden),
          up(rows, kHidden, kMlp), down(rows, kMlp, kHidden),
          head(rows, kHidden, model::kVocabSize),
          post(rows, kHidden, target_hidden) {}
  };

  Impl(Model selected_model, cublasLtHandle_t handle, const Weights& weights, unsigned capacity,
       unsigned batch_capacity, attention::Compute local, attention::Compute global)
      : handle(handle), weights(weights), capacity(capacity), batch_capacity(batch_capacity),
        target_hidden(selected_model == Model::gemma4_26b_a4b ? gemma4_26b_a4b::kHiddenSize : model::kHiddenSize),
        heads(selected_model == Model::gemma4_26b_a4b ? gemma4_26b_a4b::kQueryHeadCount : model::kQueryHeadCount),
        local_compute(local), global_compute(global) {
    require(selected_model == Model::gemma4_31b || selected_model == Model::gemma4_26b_a4b,
            "unknown assistant model");
    require(handle && weights.target_embedding && weights.target_global_k_norm,
            "handle and target weight views are required");
    require(batch_capacity > 0 && batch_capacity <= 1280, "invalid assistant batch capacity");
    for (const auto* weight : weights.assistant) require(weight, "incomplete assistant weight suffix");
    attention_bytes = heads == 16 ? mtp_attention::frozen_scratch_bytes(heads, capacity)
        : batch_attention_scratch_bytes(capacity, batch_capacity);
    // Single allocation; all BF16 regions have even lengths and the attention
    // arena starts at an explicitly aligned offset.
    const std::size_t elements = 3 * target_hidden + 4 * kHidden +
        4 * heads * 512 + 3 * kMlp + 2 * (256 + 512) + model::kVocabSize;
    const std::size_t activation_bytes = mtp_cuda::align(batch_capacity * elements * sizeof(BFloat16));
    plans.emplace(batch_capacity, std::make_unique<Plans>(batch_capacity, target_hidden, heads));
    bytes = activation_bytes + attention_bytes;
    check_cuda(cudaMalloc(&arena, bytes), "allocate scratch");
    auto* cursor = static_cast<BFloat16*>(arena);
    const auto take = [&cursor, batch_capacity](std::size_t n) {
      auto* result = cursor; cursor += batch_capacity * n; return result;
    };
    combined = take(2 * target_hidden);
    hidden = take(kHidden); normalized = take(kHidden);
    branch = take(kHidden); residual = take(kHidden);
    query_raw = take(heads * 512); query_norm = take(heads * 512);
    query_rope = take(heads * 512); context = take(heads * 512);
    gate = take(kMlp); up = take(kMlp); product = take(kMlp);
    local_cos = take(256); local_sin = take(256);
    global_cos = take(512); global_sin = take(512);
    logits = take(model::kVocabSize); feedback = take(target_hidden);
    attention = static_cast<char*>(arena) + activation_bytes;
  }

  ~Impl() { if (arena) cudaFree(arena); }

  void forward(const std::vector<Input>& inputs, cudaStream_t stream, const Trace* trace);

  cublasLtHandle_t handle;
  Weights weights;
  unsigned capacity, batch_capacity, target_hidden, heads;
  attention::Compute local_compute, global_compute;
  std::map<unsigned, std::unique_ptr<Plans>> plans;
  std::size_t bytes{}, attention_bytes{};
  void* arena{};
  void* attention{};
  BFloat16 *combined{}, *hidden{}, *normalized{}, *branch{}, *residual{};
  BFloat16 *query_raw{}, *query_norm{}, *query_rope{}, *context{};
  BFloat16 *gate{}, *up{}, *product{};
  BFloat16 *local_cos{}, *local_sin{}, *global_cos{}, *global_sin{};
  BFloat16 *logits{}, *feedback{};
};

Executor::Executor(Model selected_model, cublasLtHandle_t handle, const Weights& weights,
                   std::uint32_t context_capacity, std::uint32_t batch_capacity,
                   attention::Compute local_compute, attention::Compute global_compute)
    : impl_(std::make_unique<Impl>(selected_model, handle, weights, context_capacity, batch_capacity,
                                  local_compute, global_compute)) {}

Executor::~Executor() = default;

std::size_t Executor::scratch_bytes() const { return impl_->bytes; }

void Executor::forward(const std::uint32_t* token, const BFloat16* input_hidden,
                       const FrozenCache& cache, BFloat16* logits,
                       BFloat16* feedback, cudaStream_t stream, const Trace* trace) {
  const Input input{token, input_hidden, cache, logits, feedback};
  impl_->forward({input}, stream, trace);
}

void Executor::forward_batch(const std::vector<Input>& inputs, cudaStream_t stream) {
  impl_->forward(inputs, stream, nullptr);
}

void Executor::Impl::forward(const std::vector<Input>& inputs, cudaStream_t stream,
                             const Trace* trace) {
  const unsigned rows = inputs.size();
  require(rows > 0 && rows <= batch_capacity, "active assistant batch exceeds capacity");
  for (const auto& input : inputs) {
    require(input.token && input.hidden && input.logits && input.feedback, "null forward input/output");
    require(input.cache.processed_tokens <= capacity, "frozen prefix exceeds allocated scratch capacity");
    validate_cache(input.cache, model::AttentionKind::local, heads);
    validate_cache(input.cache, model::AttentionKind::global, heads);
  }
  if (!plans.count(rows)) plans.emplace(rows, std::make_unique<Plans>(rows, target_hidden, heads));
  auto& p = *plans.at(rows);
  auto& s = *this;
  const auto capture = [stream](BFloat16* destination, const BFloat16* source, unsigned elements = kHidden) {
    if (destination) check_cuda(cudaMemcpyAsync(destination, source, elements * sizeof(BFloat16),
        cudaMemcpyDeviceToDevice, stream), "copy oracle trace");
  };
  std::vector<primitives::FrozenLocalAttentionInput> local_attention;
  std::vector<mtp_attention::BatchInput> global_attention;
  if (rows > 1) {
    local_attention.reserve(rows);
    global_attention.reserve(rows);
  }
  if (heads == 16) {
    for (unsigned first = 0; first < rows; first += kEmbedding26Batch) {
      Embedding26Batch batch{};
      const unsigned count = std::min(kEmbedding26Batch, rows - first);
      for (unsigned i = 0; i < count; ++i) {
        const unsigned row = first + i;
        batch.inputs[i] = {inputs[row].token, inputs[row].hidden,
            s.combined + std::size_t(row) * 2 * target_hidden};
      }
      embedding_26b<<<dim3((target_hidden + 255) / 256, count), 256, 0, stream>>>(
          s.weights.target_embedding, batch);
      check_cuda(cudaGetLastError(), "26B batch embedding and feedback launch");
    }
    std::vector<primitives::RopeFactorsM1Input> factors;
    factors.reserve(rows);
    for (unsigned row = 0; row < rows; ++row)
      factors.push_back({s.local_cos + row * 256, s.local_sin + row * 256,
          s.global_cos + row * 512, s.global_sin + row * 512,
          inputs[row].cache.processed_tokens});
    primitives::generate_rope_factors_m1_batch(factors, stream);
  } else if (rows == 1) {
    primitives::embedding_lookup_device_token(
        s.weights.target_embedding, inputs[0].token, s.combined, stream);
    check_cuda(cudaMemcpyAsync(s.combined + target_hidden,
        inputs[0].hidden, target_hidden * sizeof(BFloat16),
        cudaMemcpyDeviceToDevice, stream), "copy feedback input");
    primitives::generate_rope_factors_m1(
        s.local_cos, s.local_sin, s.global_cos, s.global_sin,
        inputs[0].cache.processed_tokens, stream);
  } else {
    std::vector<primitives::DeviceTokenEmbeddingInput> embeddings;
    std::vector<primitives::RopeFactorsM1Input> factors;
    embeddings.reserve(rows);
    factors.reserve(rows);
    for (unsigned row = 0; row < rows; ++row) {
      auto* combined_row =
          s.combined + std::size_t(row) * 2 * target_hidden;
      check_cuda(cudaMemcpyAsync(combined_row + target_hidden,
          inputs[row].hidden, target_hidden * sizeof(BFloat16),
          cudaMemcpyDeviceToDevice, stream), "copy feedback input");
      embeddings.push_back({inputs[row].token, combined_row});
      factors.push_back({s.local_cos + row * 256,
          s.local_sin + row * 256, s.global_cos + row * 512,
          s.global_sin + row * 512, inputs[row].cache.processed_tokens});
    }
    primitives::embedding_lookup_device_token_batch(
        s.weights.target_embedding, embeddings, stream);
    primitives::generate_rope_factors_m1_batch(factors, stream);
  }
  p.pre.run(handle, s.combined, s.weights.assistant[46], s.hidden, stream);
  if (trace) capture(trace->pre_projection, s.hidden);
  for (unsigned layer = 0; layer < 4; ++layer) {
    const auto* w = s.weights.assistant.data() + 1 + 11 * layer;
    const bool global = layer == 3;
    const unsigned head_width = global ? 512 : 256;
    const auto kind = global ? model::AttentionKind::global : model::AttentionKind::local;
    norm(s.hidden, w[0], s.normalized, rows, stream);
    (global ? p.global_query : p.local_query).run(handle, s.normalized, w[1], s.query_raw, stream);
    primitives::rms_norm(s.query_raw, w[2], s.query_norm, rows * heads, head_width, 1.0e-6F, stream);
    if (trace) capture(trace->query_norm[layer], s.query_norm, heads * head_width);
    if (rows == 1)
      primitives::apply_rope_m1(s.query_norm,
          global ? s.global_cos : s.local_cos,
          global ? s.global_sin : s.local_sin, s.query_rope, heads, kind,
          stream);
    else
      primitives::apply_rope_m1_batch(s.query_norm,
          global ? s.global_cos : s.local_cos,
          global ? s.global_sin : s.local_sin, s.query_rope, rows, heads,
          kind, stream);
    if (trace) capture(trace->query_rope[layer], s.query_rope, heads * head_width);
    if ((global ? global_compute : local_compute) == attention::Compute::fp8) {
      global_attention.clear();
      for (unsigned row = 0; row < rows; ++row) {
        const auto offset = std::size_t(row) * heads * head_width;
        global_attention.push_back({s.query_rope + offset, nullptr, nullptr,
            global ? global_cache_view(inputs[row].cache) : local_cache_view(inputs[row].cache),
            inputs[row].cache.processed_tokens, 1, s.context + offset});
      }
      mtp_attention::run_fp8_batch(heads, global_attention, s.weights.target_global_k_norm, kind,
          s.attention, s.attention_bytes, stream, true);
    } else if (heads == 16) {
      global_attention.clear();
      for (unsigned row = 0; row < rows; ++row) {
        const auto offset = std::size_t(row) * heads * head_width;
        global_attention.push_back({s.query_rope + offset, nullptr, nullptr,
            global ? global_cache_view(inputs[row].cache) : local_cache_view(inputs[row].cache),
            inputs[row].cache.processed_tokens, 1, s.context + offset});
      }
      mtp_attention::run_frozen_prefix_batch(heads, global_attention,
          s.weights.target_global_k_norm, kind, s.attention, s.attention_bytes, stream);
    } else if (rows == 1) {
      attend_frozen_prefix(s.query_rope, inputs[0].cache,
          s.weights.target_global_k_norm, kind, s.attention, s.context, stream);
    } else if (global) {
      global_attention.clear();
      for (unsigned row = 0; row < rows; ++row) {
        const auto offset = std::size_t(row) * heads * head_width;
        global_attention.push_back({s.query_rope + offset, nullptr, nullptr,
            global_cache_view(inputs[row].cache),
            inputs[row].cache.processed_tokens, 1, s.context + offset});
      }
      mtp_attention::run_frozen_global_batch(global_attention,
          s.weights.target_global_k_norm, s.attention, s.attention_bytes,
          stream);
    } else {
      local_attention.clear();
      for (unsigned row = 0; row < rows; ++row) {
        const auto offset = std::size_t(row) * heads * head_width;
        local_attention.push_back({s.query_rope + offset,
            inputs[row].cache.local_key, inputs[row].cache.local_value,
            inputs[row].cache.processed_tokens - 1, s.context + offset, inputs[row].cache.local_format});
      }
      primitives::causal_gqa_attention_cached_m1_fused_local_batch(
          local_attention, s.attention, s.attention_bytes, stream);
    }
    check_cuda(cudaGetLastError(), "attention finalizer launch");
    if (trace) capture(trace->attention[layer], s.context, heads * head_width);
    (global ? p.global_out : p.local_out).run(handle, s.context, w[3], s.branch, stream);
    norm(s.branch, w[4], s.normalized, rows, stream);
    primitives::residual_add(s.hidden, s.normalized, s.residual, rows * kHidden, stream);
    norm(s.residual, w[5], s.normalized, rows, stream);
    p.up.run(handle, s.normalized, w[6], s.gate, stream);
    p.up.run(handle, s.normalized, w[7], s.up, stream);
    primitives::gelu_tanh_multiply(s.gate, s.up, s.product, rows * kMlp, stream);
    p.down.run(handle, s.product, w[8], s.branch, stream);
    norm(s.branch, w[9], s.normalized, rows, stream);
    primitives::residual_add(s.residual, s.normalized, s.hidden, rows * kHidden, stream);
    primitives::trained_scalar(s.hidden, w[10], rows * kHidden, stream);
    if (trace) capture(trace->layers[layer], s.hidden);
  }
  norm(s.hidden, s.weights.assistant[45], s.normalized, rows, stream);
  if (trace) capture(trace->final_norm, s.normalized);
  for (unsigned row = 0; row < rows; ++row)
    capture(inputs[row].capture_hidden, s.normalized + std::size_t(row) * kHidden);
  p.head.run(handle, s.normalized, s.weights.assistant[0], s.logits, stream);
  p.post.run(handle, s.normalized, s.weights.assistant[47], s.feedback, stream);
  for (unsigned row = 0; row < rows; ++row) {
    capture(inputs[row].logits, s.logits + std::size_t(row) * model::kVocabSize, model::kVocabSize);
    capture(inputs[row].feedback, s.feedback + std::size_t(row) * target_hidden, target_hidden);
  }
}

}  // namespace gewell::mtp_assistant
