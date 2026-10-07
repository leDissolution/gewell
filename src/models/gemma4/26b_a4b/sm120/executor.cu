#include "gewell/models/gemma4/26b_a4b/executor.h"
#include "gewell/models/gemma4/26b_a4b/artifact.h"
#include "bf16_math.cuh"
#include "moe.cuh"
#include "rope_inverse_frequency.cuh"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace gewell::gemma4_26b_a4b::sm120 {
namespace {
using B = __nv_bfloat16;
void check(cudaError_t code) { if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code)); }
void blas(cublasStatus_t code) { if (code != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("26B cuBLAS error " + std::to_string(code)); }
struct Buffer {
  void* data{};
  explicit Buffer(std::size_t bytes) { check(cudaMalloc(&data, bytes)); }
  ~Buffer() { if (data) cudaFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer(Buffer&& b) noexcept : data(std::exchange(b.data, nullptr)) {}
  B* bf() { return static_cast<B*>(data); }
};
using namespace bf16_math;
__global__ void kv_write(const B* key, const B* val, B* keys, B* vals, unsigned width, unsigned slot) {
  unsigned i = blockIdx.x * 256 + threadIdx.x;
  if (i < width) { keys[std::size_t(slot) * width + i] = key[i]; vals[std::size_t(slot) * width + i] = val[i]; }
}
__global__ void gather_ring(const B* keys, const B* values, B* ordered_keys,
                            B* ordered_values, unsigned begin, unsigned width) {
  unsigned i = blockIdx.x*256+threadIdx.x;
  if (i >= 1024*width) return;
  unsigned source = ((begin+i/width)%1024)*width+i%width;
  ordered_keys[i] = keys[source]; ordered_values[i] = values[source];
}
__global__ void softmax(B* data, unsigned count) {
  if (count <= 2048) {
    if (threadIdx.x >= 32) return;
    unsigned width = 1;
    while (width < count) width *= 2;
    unsigned lane = threadIdx.x, iterations = (width+31)/32;
    B* row = data + std::size_t(blockIdx.x)*count;
    float elements[64], maximum = -INFINITY;
    for (unsigned i = 0; i < iterations; ++i) {
      unsigned index = lane+i*32;
      elements[i] = index < count ? value(row[index]) : -INFINITY;
      maximum = fmaxf(maximum,elements[i]);
    }
    for (unsigned offset = (width < 32 ? width : 32)/2; offset; offset /= 2)
      maximum = fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,offset));
    float total = 0;
    for (unsigned i = 0; i < iterations; ++i) {
      elements[i] = expf(elements[i]-maximum); total += elements[i];
    }
    for (unsigned offset = (width < 32 ? width : 32)/2; offset; offset /= 2)
      total += __shfl_xor_sync(0xffffffff,total,offset);
    for (unsigned i = 0; i < iterations; ++i) {
      unsigned index = lane+i*32;
      if (index < count) row[index] = rounded(elements[i]/total);
    }
    return;
  }
  __shared__ float values[256];
  B* row = data + std::size_t(blockIdx.x) * count;
  float maximum = -INFINITY;
  for (unsigned i = threadIdx.x; i < count; i += 256) maximum = fmaxf(maximum, value(row[i]));
  values[threadIdx.x] = maximum; __syncthreads();
  for (unsigned stride = 128; stride; stride /= 2) {
    if (threadIdx.x < stride) values[threadIdx.x] = fmaxf(values[threadIdx.x], values[threadIdx.x + stride]);
    __syncthreads();
  }
  maximum = values[0]; float sum = 0;
  for (unsigned i = threadIdx.x; i < count; i += 256) sum += expf(value(row[i]) - maximum);
  __syncthreads(); values[threadIdx.x] = sum; __syncthreads();
  for (unsigned stride = 128; stride; stride /= 2) {
    if (threadIdx.x < stride) values[threadIdx.x] += values[threadIdx.x + stride];
    __syncthreads();
  }
  for (unsigned i = threadIdx.x; i < count; i += 256) row[i] = rounded(expf(value(row[i]) - maximum) / values[0]);
}
__global__ void attention_pointers(const B* kv, const B* input, B* output,
                                   const B** a, const B** b, B** c,
                                   unsigned d, unsigned heads, unsigned input_stride,
                                   unsigned output_stride) {
  unsigned head = threadIdx.x;
  a[head] = kv + (head / (16 / heads))*d;
  b[head] = input + head*input_stride;
  c[head] = output + head*output_stride;
}
__global__ void softcap(B* x) {
  unsigned i = blockIdx.x * 256 + threadIdx.x;
  if (i < kVocabSize) x[i] = rounded(value(rounded(tanhf(value(rounded(value(x[i]) / 30.f))))) * 30.f);
}
}  // namespace

struct Executor::Impl {
  ArtifactFile artifact;
  Buffer weights;
  Buffer scratch{2 * 131072}, score_buffer, logits{kVocabSize * sizeof(B)};
  Buffer attention_addresses{48*sizeof(B*)};
  // Reused only for wrapped local attention; no duplicated resident cache.
  Buffer ordered_keys{1024*2048*sizeof(B)}, ordered_values{1024*2048*sizeof(B)};
  std::vector<Buffer> keys, values;
  std::array<std::array<const B*, 25>, 30> layer{};
  Buffer expert_tables{30*128*sizeof(ExpertWeights)};
  MoeWorkspace moe{1};
  cublasHandle_t handle{};
  unsigned capacity, position{};
  explicit Impl(const std::string& path, unsigned cap)
      : artifact([&] {
          auto file = ArtifactFile::Open(path);
          for (const auto& entry : file.entries())
            if (entry.storage_type != StorageType::bf16)
              throw std::invalid_argument("26B serial diagnostic requires BF16 weights");
          return file;
        }()), weights(artifact.file_bytes() - kArtifactDataOffset),
        score_buffer(std::size_t(16) * cap * sizeof(B)), capacity(cap) {
    check(cudaMemcpy(weights.data, artifact.tensor_data(0), artifact.file_bytes() - kArtifactDataOffset, cudaMemcpyHostToDevice));
    std::array<std::array<ExpertWeights,128>,30> expert_pointers{};
    for (std::size_t i = 1; i + 1 < kTextTensors.size(); ++i) {
      const auto& spec = kTextTensors[i];
      auto* pointer = reinterpret_cast<const B*>(static_cast<char*>(weights.data) + artifact.entries()[i].offset - kArtifactDataOffset);
      if (spec.expert < 0) layer[spec.layer][unsigned(spec.role)] = pointer;
      else {
        auto& expert = expert_pointers[spec.layer][spec.expert];
        if (spec.role == TensorRole::expert_gate_proj) expert.gate.bf16 = pointer;
        else if (spec.role == TensorRole::expert_up_proj) expert.up.bf16 = pointer;
        else expert.down.bf16 = pointer;
      }
    }
    check(cudaMemcpy(expert_tables.data,expert_pointers.data(),sizeof(expert_pointers),cudaMemcpyHostToDevice));
    for (unsigned l = 0; l < 30; ++l) {
      std::size_t width = is_global_layer(l) ? 1024 : 2048;
      std::size_t slots = is_global_layer(l) ? cap : 1024;
      keys.emplace_back(width * slots * sizeof(B)); values.emplace_back(width * slots * sizeof(B));
    }
    blas(cublasCreate(&handle));
    try { blas(cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)); }
    catch (...) { cublasDestroy(handle); throw; }
  }
  ~Impl() { if (handle) cublasDestroy(handle); }
  void linear(const B* input, const B* weight, B* output, unsigned k, unsigned n) {
    float alpha = 1, beta = 0;
    blas(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, 1, k, &alpha, weight, CUDA_R_16BF, k,
                       input, CUDA_R_16BF, k, &beta, output, CUDA_R_16BF, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  }
  void attention_product(const B* kv, const B* input, B* out, bool transpose,
                         unsigned count, unsigned d, unsigned heads) {
    auto** a = static_cast<const B**>(attention_addresses.data);
    auto** b = a+16;
    auto** c = reinterpret_cast<B**>(static_cast<char*>(attention_addresses.data)+32*sizeof(B*));
    unsigned m = transpose ? count : d, k = transpose ? d : count;
    attention_pointers<<<1,16>>>(kv,input,out,a,b,c,d,heads,k,m);
    float alpha = 1, beta = 0;
    blas(cublasGemmBatchedEx(handle, transpose ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N,
                            m,1,k,&alpha,reinterpret_cast<const void* const*>(a),CUDA_R_16BF,heads*d,
                            reinterpret_cast<const void* const*>(b),CUDA_R_16BF,k,&beta,
                            reinterpret_cast<void* const*>(c),CUDA_R_16BF,m,16,
                            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
  }
  void capture(const Capture& sink, const std::string& name, const B* data, unsigned size) {
    if (!sink) return;
    std::vector<std::uint16_t> result(size);
    check(cudaMemcpy(result.data(), data, size * 2, cudaMemcpyDeviceToHost));
    sink(name, result);
  }
};
Executor::Executor(const std::string& path, std::uint32_t capacity) {
  if (!capacity || capacity > kMaxPositions) throw std::invalid_argument("26B capacity must be in 1..262144");
  impl_ = std::make_unique<Impl>(path, capacity);
}
Executor::~Executor() = default;
void Executor::Reset() { impl_->position = 0; }
std::uint32_t Executor::position() const { return impl_->position; }
std::vector<std::uint16_t> Executor::ForwardToken(std::uint32_t token, const Capture& sink) {
  auto& s = *impl_;
  if (token >= kVocabSize || s.position >= s.capacity) throw std::invalid_argument("26B token or context capacity exceeded");
  B* h = s.scratch.bf(); B* normalized = h + 8192; B* branch = normalized + 8192;
  B* shared = branch + 8192; B* routed = shared + 8192; B* gate = routed + 8192;
  B* up = gate + 8192; B* product = up + 8192; B* q = product + 8192;
  B* k = q + 8192; B* v = k + 8192; B* temporary = v + 8192;
  constexpr unsigned blocks = (kHiddenSize + 255) / 256;
  embedding<<<blocks,256>>>(s.weights.bf(), h, token); s.capture(sink,"embedding.0",h,kHiddenSize);
  for (unsigned l = 0; l < 30; ++l) {
    auto w = [&](TensorRole role) { return s.layer[l][unsigned(role)]; };
    std::string prefix = "layer." + std::to_string(l) + ".";
    bool global = is_global_layer(l); unsigned d = global ? 512 : 256, heads = global ? 2 : 8;
    normalize(h,w(TensorRole::input_norm),normalized,1,kHiddenSize);
    s.linear(normalized,w(TensorRole::q_proj),q,kHiddenSize,16*d);
    s.capture(sink,prefix+"self_attn.q_proj.0",q,16*d);
    s.linear(normalized,w(TensorRole::k_proj),k,kHiddenSize,heads*d);
    s.capture(sink,prefix+"self_attn.k_proj.0",k,heads*d);
    if (!global) s.linear(normalized,w(TensorRole::v_proj),v,kHiddenSize,heads*d);
    normalize(global ? k : v,nullptr,temporary,heads,d);
    s.capture(sink,prefix+"self_attn.v_norm.0",temporary,heads*d);
    normalize(q,w(TensorRole::q_norm),branch,16,d);
    s.capture(sink,prefix+"self_attn.q_norm.0",branch,16*d);
    rope<<<(16*d+255)/256,256>>>(branch,q,16,global,s.position);
    s.capture(sink,prefix+"query_rope",q,16*d);
    normalize(k,w(TensorRole::k_norm),branch,heads,d);
    s.capture(sink,prefix+"self_attn.k_norm.0",branch,heads*d);
    rope<<<(heads*d+255)/256,256>>>(branch,k,heads,global,s.position);
    s.capture(sink,prefix+"key_rope",k,heads*d);
    unsigned slots = global ? s.capacity : 1024;
    kv_write<<<(heads*d+255)/256,256>>>(k,temporary,s.keys[l].bf(),s.values[l].bf(),heads*d,s.position%slots);
    unsigned count = global ? s.position+1 : std::min(s.position+1,1024u), begin = s.position+1-count;
    const B* attention_keys = s.keys[l].bf();
    const B* attention_values = s.values[l].bf();
    if (begin % slots != 0) {
      gather_ring<<<1024*heads*d/256,256>>>(attention_keys,attention_values,
          s.ordered_keys.bf(),s.ordered_values.bf(),begin,heads*d);
      attention_keys = s.ordered_keys.bf(); attention_values = s.ordered_values.bf();
    }
    s.attention_product(attention_keys,q,s.score_buffer.bf(),true,count,d,heads);
    s.capture(sink,prefix+"scores",s.score_buffer.bf(),16*count);
    softmax<<<16,256>>>(s.score_buffer.bf(),count);
    s.capture(sink,prefix+"probabilities",s.score_buffer.bf(),16*count);
    s.attention_product(attention_values,s.score_buffer.bf(),branch,false,count,d,heads);
    s.capture(sink,prefix+"context",branch,16*d);
    s.linear(branch,w(TensorRole::o_proj),normalized,16*d,kHiddenSize);
    s.capture(sink,prefix+"self_attn.o_proj.0",normalized,kHiddenSize);
    normalize(normalized,w(TensorRole::post_attention_norm),branch,1,kHiddenSize);
    s.capture(sink,prefix+"post_attention_layernorm.0",branch,kHiddenSize);
    add<<<blocks,256>>>(h,branch,h,kHiddenSize);
    normalize(h,w(TensorRole::pre_feedforward_norm),normalized,1,kHiddenSize);
    s.linear(normalized,w(TensorRole::gate_proj),gate,kHiddenSize,kMlpSize);
    s.linear(normalized,w(TensorRole::up_proj),up,kHiddenSize,kMlpSize);
    activation<<<(kMlpSize+255)/256,256>>>(gate,up,product,kMlpSize);
    s.linear(product,w(TensorRole::down_proj),branch,kMlpSize,kHiddenSize);
    normalize(branch,w(TensorRole::post_feedforward_norm_1),shared,1,kHiddenSize);
    s.capture(sink,prefix+"post_feedforward_layernorm_1.0",shared,kHiddenSize);
    normalize(h,nullptr,normalized,1,kHiddenSize);
    router_scale<<<blocks,256>>>(normalized,w(TensorRole::router_scale));
    s.linear(normalized,w(TensorRole::router_proj),branch,kHiddenSize,128);
    s.capture(sink,prefix+"router.proj.0",branch,128);
    normalize(h,w(TensorRole::pre_feedforward_norm_2),normalized,1,kHiddenSize);
    s.capture(sink,prefix+"pre_feedforward_layernorm_2.0",normalized,kHiddenSize);
    s.moe.run(normalized,branch,w(TensorRole::router_per_expert_scale),
              static_cast<const ExpertWeights*>(s.expert_tables.data)+l*128,routed,1,true,false,false,false);
    if (sink) {
      std::array<int,8> selected;
      check(cudaMemcpy(selected.data(),s.moe.selected_experts(),sizeof(selected),cudaMemcpyDeviceToHost));
      std::vector<std::uint16_t> ids;
      for (int expert : selected) {
        float id = float(expert);
        std::uint32_t bits;
        std::memcpy(&bits,&id,sizeof(bits));
        ids.push_back(std::uint16_t(bits >> 16));
      }
      sink(prefix+"router.2",ids);
    }
    s.capture(sink,prefix+"experts.0",routed,kHiddenSize);
    normalize(routed,w(TensorRole::post_feedforward_norm_2),branch,1,kHiddenSize);
    s.capture(sink,prefix+"post_feedforward_layernorm_2.0",branch,kHiddenSize);
    add<<<blocks,256>>>(shared,branch,shared,kHiddenSize);
    normalize(shared,w(TensorRole::post_feedforward_norm),branch,1,kHiddenSize);
    add<<<blocks,256>>>(h,branch,h,kHiddenSize); scale<<<blocks,256>>>(h,w(TensorRole::layer_scalar));
    s.capture(sink,prefix+"output.0",h,kHiddenSize);
  }
  auto final_weight = reinterpret_cast<const B*>(static_cast<char*>(s.weights.data)+s.artifact.entries().back().offset-kArtifactDataOffset);
  normalize(h,final_weight,normalized,1,kHiddenSize); s.capture(sink,"final_norm.0",normalized,kHiddenSize);
  // Vocabulary output is larger than the hidden/projection scratch.
  s.linear(normalized,s.weights.bf(),s.logits.bf(),kHiddenSize,kVocabSize);
  softcap<<<kVocabSize/256,256>>>(s.logits.bf());
  check(cudaGetLastError());
  std::vector<std::uint16_t> result(kVocabSize);
  check(cudaMemcpy(result.data(),s.logits.data,result.size()*2,cudaMemcpyDeviceToHost));
  if (sink) sink("logits",result);
  ++s.position;
  return result;
}
}  // namespace gewell::gemma4_26b_a4b::sm120
