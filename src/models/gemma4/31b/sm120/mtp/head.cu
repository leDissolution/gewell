#include "head.h"
#include "cuda.cuh"
#include "gewell/models/gemma4/31b/model.h"
#include <cublas_v2.h>
#include "json.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>

namespace gewell::mtp_head {
namespace {
using mtp_cuda::Buffer;
using mtp_cuda::check;
using Json = nlohmann::json;
void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error("MTP head: " + message);
}
__device__ float sum(float x) {
  for (int d = 16; d; d /= 2) x += __shfl_down_sync(0xffffffff, x, d);
  return __shfl_sync(0xffffffff, x, 0);
}
__device__ float block_sum(float x) {
  __shared__ float partial[8];
  x = sum(x);
  if (!(threadIdx.x % 32)) partial[threadIdx.x / 32] = x;
  __syncthreads();
  x = threadIdx.x < 8 ? partial[threadIdx.x] : 0;
  x = sum(x);
  if (!threadIdx.x) partial[0] = x;
  __syncthreads();
  return partial[0];
}
__global__ void normalize_inputs(const Input* inputs, float* normalized, float* scales,
                                 int layers, int hidden) {
  const int row = blockIdx.x, layer = row % layers;
  const auto& input = inputs[row / layers];
  const auto* source = layer == layers - 1 ? input.final_hidden : input.probes + layer * hidden;
  float squares = 0;
  for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
    const float v = __bfloat162float(source[j]); squares += v * v;
  }
  const float rms = sqrtf(block_sum(squares) / hidden + 1e-6F);
  if (!threadIdx.x) scales[row] = logf(rms);
  for (int j = threadIdx.x; j < hidden; j += blockDim.x)
    normalized[row * hidden + j] = __bfloat162float(source[j]) / rms;
}
__global__ void compose(const Input* inputs, const float* projected, const float* scales,
                        const float* scale_weight, const float* embeddings,
                        const float* context_weight, const float* context_bias,
                        float* x, int layers, int width) {
  const int row = blockIdx.x, layer = row % (layers + 1), batch = row / (layers + 1);
  for (int j = threadIdx.x; j < width; j += blockDim.x) {
    float v;
    if (layer == layers) {
      v = context_bias[j];
      for (int k = 0; k < 9; ++k) v += inputs[batch].context[k] * context_weight[j * 9 + k];
    } else {
      const int i = batch * layers + layer;
      v = projected[i * width + j] + scale_weight[j] * scales[i] + embeddings[layer * width + j];
    }
    x[row * width + j] = v;
  }
}
__global__ void norm(const float* x, float* out, const float* weight, const float* bias, int width) {
  const int row = blockIdx.x;
  float total = 0;
  for (int j = threadIdx.x; j < width; j += blockDim.x) total += x[row * width + j];
  const float mean = block_sum(total) / width;
  // All threads must finish reading the first reduction before reusing shared memory.
  __syncthreads();
  float squares = 0;
  for (int j = threadIdx.x; j < width; j += blockDim.x) {
    const float v = x[row * width + j] - mean; squares += v * v;
  }
  const float inv = rsqrtf(block_sum(squares) / width + 1e-5F);
  for (int j = threadIdx.x; j < width; j += blockDim.x)
    out[row * width + j] = (x[row * width + j] - mean) * inv * weight[j] + bias[j];
}
// One warp per query/head. The sequence is just the captured layers + metadata.
__global__ void attention(const float* qkv, const float* bias, float* out,
                          int tokens, int width, int heads) {
  const int h = blockIdx.x % heads, row = blockIdx.x / heads;
  const int batch = row / tokens, dim = width / heads, lane = threadIdx.x;
  float scores[64], maximum = -INFINITY;
  for (int k = 0; k < tokens; ++k) {
    float dot = 0;
    for (int j = lane; j < dim; j += 32) {
      const int c = h * dim + j;
      dot += (qkv[row * 3 * width + c] + bias[c]) *
          (qkv[(batch * tokens + k) * 3 * width + width + c] + bias[width + c]);
    }
    scores[k] = sum(dot) * rsqrtf(float(dim));
    maximum = fmaxf(maximum, scores[k]);
  }
  float denominator = 0;
  for (int k = 0; k < tokens; ++k) { scores[k] = expf(scores[k] - maximum); denominator += scores[k]; }
  for (int j = lane; j < dim; j += 32) {
    const int c = h * dim + j;
    float v = 0;
    for (int k = 0; k < tokens; ++k)
      v += scores[k] / denominator * (qkv[(batch * tokens + k) * 3 * width + 2 * width + c] + bias[2 * width + c]);
    out[row * width + c] = v;
  }
}
__global__ void residual(float* x, const float* update, const float* bias, int count, int width) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) x[i] += update[i] + bias[i % width];
}
__global__ void gelu(float* x, const float* bias, int count, int width) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) { const float v = x[i] + bias[i % width]; x[i] = v * 0.5F * (1 + erff(v * 0.7071067811865475F)); }
}
__global__ void pool(const float* x, const float* weights, float* out, int tokens, int width) {
  const int batch = blockIdx.x;
  for (int j = threadIdx.x; j < width; j += blockDim.x) {
    float v = 0;
    for (int k = 0; k < tokens; ++k) v += x[(batch * tokens + k) * width + j] * weights[k];
    out[batch * width + j] = v;
  }
}
__global__ void survival(float* x, const float* bias, int depth) {
  if (threadIdx.x) return;
  float p = 1;
  for (int k = 0; k < depth; ++k) {
    const int i = blockIdx.x * depth + k;
    p *= 1 / (1 + expf(-x[i] - bias[k])); x[i] = p;
  }
}
struct Handle {
  cublasHandle_t value{};
  Handle() { check(cublasCreate(&value), "create MTP head cuBLAS"); }
  ~Handle() { if (value) cublasDestroy(value); }
};
}  // namespace

struct Head::Impl {
  std::uint32_t capacity, width{}, hidden{}, depth{}, blocks{}, heads{}, ff{};
  std::vector<std::uint32_t> layers;
  std::unique_ptr<Buffer> weights, scratch, input_buffer, pool_weights;
  std::map<std::string, const float*> tensors;
  Handle handle;
  explicit Impl(const std::string& directory, std::uint32_t batch) : capacity(batch) {
    const auto root = std::filesystem::path(directory);
    std::ifstream cf(root / "config.json"); Json config; cf >> config;
    require(config.at("format_version") == 1 && config.at("normalization") == "unit_rms_plus_log_rms_eps_1e-6" &&
        config.at("output") == "conditional_acceptance_logits", "unsupported checkpoint contract");
    const Json features = {"temperature", "top_p", "log1p_top_k_div_16", "log1p_position_div_16",
        "log1p_output_begin_div_16", "previous_accepted_div_max_depth", "previous_depth_div_max_depth",
        "prior_accepted_div_prior_proposed", "log1p_batch_size_div_8"};
    require(config.at("context_features") == features, "context features differ from engine");
    const auto& model = config.at("model");
    require(model.at("architecture") == "transformer" && model.at("pooling") == "learned", "requires transformer with learned pooling");
    width = model.at("width"); hidden = model.at("target_width"); depth = model.at("max_depth");
    blocks = model.at("blocks"); heads = model.at("heads");
    const unsigned multiplier = model.at("ff_multiplier");
    require(capacity && capacity <= 256 && width && width <= 1024 && hidden == gemma4_31b::kHiddenSize &&
        depth && depth <= 1279 && blocks && blocks <= 32 && heads && width % heads == 0 &&
        multiplier && multiplier <= 8, "invalid model dimensions");
    ff = width * multiplier;
    const auto& names = model.at("layers");
    require(names.size() > 1 && names.size() <= 61 && names.back() == "final", "final hidden must follow intermediate layers");
    for (std::size_t i = 0; i + 1 < names.size(); ++i) {
      const auto name = names[i].get<std::string>();
      std::size_t end = 0; const auto layer = std::stoul(name, &end);
      require(end == name.size() && layer && layer <= gemma4_31b::kLayerCount &&
          (layers.empty() || layer > layers.back()), "invalid intermediate layers");
      layers.push_back(layer);
    }
    std::ifstream file(root / "model.safetensors", std::ios::binary | std::ios::ate);
    require(bool(file), "cannot open model.safetensors");
    const auto size = file.tellg(); file.seekg(0);
    std::uint64_t header_size{}; file.read(reinterpret_cast<char*>(&header_size), 8);
    require(file && header_size <= 1 << 20 && size >= std::streamoff(8 + header_size), "invalid safetensors header");
    std::string header(header_size, '\0'); file.read(header.data(), header.size());
    const auto index = Json::parse(header);
    const auto bytes = std::size_t(size) - 8 - header_size;
    require(bytes && bytes % 4 == 0 && bytes <= 512 * 1024 * 1024, "invalid checkpoint size");
    std::vector<float> data(bytes / 4); file.read(reinterpret_cast<char*>(data.data()), bytes);
    require(bool(file) && std::all_of(data.begin(), data.end(), [](float x) { return std::isfinite(x); }), "invalid checkpoint values");
    weights = std::make_unique<Buffer>(bytes);
    check(cudaMemcpy(weights->data(), data.data(), bytes, cudaMemcpyHostToDevice), "upload MTP head weights");
    auto tensor = [&](const std::string& name, std::vector<std::uint32_t> shape) {
      const auto& entry = index.at(name);
      require(entry.at("dtype") == "F32" && entry.at("shape") == shape, "invalid tensor " + name);
      const auto start = entry.at("data_offsets").at(0).get<std::size_t>();
      const auto finish = entry.at("data_offsets").at(1).get<std::size_t>();
      std::size_t elements = 1; for (auto d : shape) elements *= d;
      require(start % 4 == 0 && finish <= bytes && finish >= start && finish - start == elements * 4, "invalid tensor offsets " + name);
      tensors[name] = weights->at<float>(start);
      return start / 4;
    };
    const unsigned n = layers.size() + 1, tokens = n + 1;
    tensor("projection.weight", {width, hidden}); tensor("scale_projection.weight", {width, 1});
    tensor("layer_embedding", {n, width}); tensor("context_projection.weight", {width, 9});
    tensor("context_projection.bias", {width});
    auto pool_offset = tensor("pool_logits", {tokens});
    std::vector<float> pooling(tokens);
    const float maximum = *std::max_element(data.begin() + pool_offset, data.begin() + pool_offset + tokens);
    float total = 0; for (unsigned i = 0; i < tokens; ++i) { pooling[i] = std::exp(data[pool_offset + i] - maximum); total += pooling[i]; }
    for (auto& v : pooling) v /= total;
    pool_weights = std::make_unique<Buffer>(tokens * 4);
    check(cudaMemcpy(pool_weights->data(), pooling.data(), tokens * 4, cudaMemcpyHostToDevice), "upload MTP head pooling");
    for (unsigned b = 0; b < blocks; ++b) {
      const auto p = "blocks." + std::to_string(b) + ".";
      tensor(p + "self_attn.in_proj_weight", {3 * width, width}); tensor(p + "self_attn.in_proj_bias", {3 * width});
      tensor(p + "self_attn.out_proj.weight", {width, width}); tensor(p + "self_attn.out_proj.bias", {width});
      tensor(p + "linear1.weight", {ff, width}); tensor(p + "linear1.bias", {ff});
      tensor(p + "linear2.weight", {width, ff}); tensor(p + "linear2.bias", {width});
      for (const auto* norm : {"norm1.", "norm2."}) {
        tensor(p + norm + "weight", {width}); tensor(p + norm + "bias", {width});
      }
    }
    tensor("output.0.weight", {width}); tensor("output.0.bias", {width});
    tensor("output.1.weight", {depth, width}); tensor("output.1.bias", {depth});
    const std::size_t rows = std::size_t(capacity) * tokens;
    scratch = std::make_unique<Buffer>(4 * (capacity * std::size_t(n) * (hidden + width + 1) + rows * (7 * width + ff) + capacity * (2 * width + depth)));
    input_buffer = std::make_unique<Buffer>(capacity * sizeof(Input));
  }
  const float* w(const std::string& name) const { return tensors.at(name); }
  void linear(const float* x, const float* weight, float* y, int rows, int in, int out) {
    const float one = 1, zero = 0;
    check(cublasSgemm(handle.value, CUBLAS_OP_T, CUBLAS_OP_N, out, rows, in,
        &one, weight, in, x, in, &zero, y, out), "MTP head linear");
  }
};
Head::Head(const std::string& directory, std::uint32_t capacity) : impl_(std::make_unique<Impl>(directory, capacity)) {}
Head::~Head() = default;
const std::vector<std::uint32_t>& Head::layers() const { return impl_->layers; }
std::uint32_t Head::max_depth() const { return impl_->depth; }
std::size_t Head::bytes() const { const auto& s = *impl_; return s.weights->size() + s.scratch->size() + s.pool_weights->size() + s.input_buffer->size(); }
std::vector<float> Head::predict(const std::vector<Input>& inputs, cudaStream_t stream) {
  auto& s = *impl_; const int batch = inputs.size(), n = s.layers.size() + 1, tokens = n + 1, rows = batch * tokens;
  require(batch > 0 && batch <= int(s.capacity), "prediction batch exceeds capacity");
  check(cublasSetStream(s.handle.value, stream), "MTP head stream");
  check(cudaMemcpyAsync(s.input_buffer->data(), inputs.data(), inputs.size() * sizeof(Input), cudaMemcpyHostToDevice, stream), "upload MTP head inputs");
  auto* input = s.input_buffer->at<Input>(); auto* next = s.scratch->at<float>();
  auto take = [&](std::size_t size) { auto* p = next; next += size; return p; };
  auto* normalized = take(batch * n * s.hidden); auto* projected = take(batch * n * s.width); auto* scales = take(batch * n);
  auto* x = take(rows * s.width); auto* normed = take(rows * s.width); auto* qkv = take(rows * 3 * s.width);
  auto* attended = take(rows * s.width); auto* update = take(rows * s.width); auto* ff = take(rows * s.ff);
  auto* pooled = take(batch * s.width); auto* output_norm = take(batch * s.width); auto* output = take(batch * s.depth);
  normalize_inputs<<<batch * n, 256, 0, stream>>>(input, normalized, scales, n, s.hidden);
  s.linear(normalized, s.w("projection.weight"), projected, batch * n, s.hidden, s.width);
  compose<<<rows, 256, 0, stream>>>(input, projected, scales, s.w("scale_projection.weight"), s.w("layer_embedding"),
      s.w("context_projection.weight"), s.w("context_projection.bias"), x, n, s.width);
  for (unsigned b = 0; b < s.blocks; ++b) {
    const auto p = "blocks." + std::to_string(b) + ".";
    norm<<<rows, 256, 0, stream>>>(x, normed, s.w(p + "norm1.weight"), s.w(p + "norm1.bias"), s.width);
    s.linear(normed, s.w(p + "self_attn.in_proj_weight"), qkv, rows, s.width, 3 * s.width);
    attention<<<rows * s.heads, 32, 0, stream>>>(qkv, s.w(p + "self_attn.in_proj_bias"), attended, tokens, s.width, s.heads);
    s.linear(attended, s.w(p + "self_attn.out_proj.weight"), update, rows, s.width, s.width);
    residual<<<(rows * s.width + 255) / 256, 256, 0, stream>>>(x, update, s.w(p + "self_attn.out_proj.bias"), rows * s.width, s.width);
    norm<<<rows, 256, 0, stream>>>(x, normed, s.w(p + "norm2.weight"), s.w(p + "norm2.bias"), s.width);
    s.linear(normed, s.w(p + "linear1.weight"), ff, rows, s.width, s.ff);
    gelu<<<(rows * s.ff + 255) / 256, 256, 0, stream>>>(ff, s.w(p + "linear1.bias"), rows * s.ff, s.ff);
    s.linear(ff, s.w(p + "linear2.weight"), update, rows, s.ff, s.width);
    residual<<<(rows * s.width + 255) / 256, 256, 0, stream>>>(x, update, s.w(p + "linear2.bias"), rows * s.width, s.width);
  }
  pool<<<batch, 256, 0, stream>>>(x, s.pool_weights->at<float>(), pooled, tokens, s.width);
  norm<<<batch, 256, 0, stream>>>(pooled, output_norm, s.w("output.0.weight"), s.w("output.0.bias"), s.width);
  s.linear(output_norm, s.w("output.1.weight"), output, batch, s.width, s.depth);
  survival<<<batch, 32, 0, stream>>>(output, s.w("output.1.bias"), s.depth);
  check(cudaGetLastError(), "MTP head kernels");
  std::vector<float> result(batch * s.depth);
  check(cudaMemcpyAsync(result.data(), output, result.size() * 4, cudaMemcpyDeviceToHost, stream), "download MTP head survival");
  check(cudaStreamSynchronize(stream), "complete MTP head prediction");
  return result;
}
}  // namespace gewell::mtp_head
