#include "gewell/models/embeddinggemma2/encoder.h"
#include "gewell/models/embeddinggemma2/model.h"
#include "gewell/models/embeddinggemma2/vision.h"
#include "gewell/models/embeddinggemma2/audio.h"
#include "gewell/vision_engine.h"
#include "attention.h"
#include "cuda_memory.cuh"
#include "rope_frequencies.h"

#include <cublas_v2.h>
#include <cudnn.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace gewell::embeddinggemma2 {
namespace {
using B = __nv_bfloat16;
using Device = cuda_detail::DeviceAllocation;
using cuda_detail::check_cuda;
constexpr int kTile = 128;
// Packed sequences use the full-key warp softmax (64 values per lane).
constexpr int kPackedMaxTokens = 2048;
__device__ float f(B x) { return __bfloat162float(x); }
__device__ B b(float x) { return __float2bfloat16_rn(x); }
void blas(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("embeddinggemma2 cuBLAS: " + std::to_string(status));
}
struct Blas {
  cublasHandle_t handle{};
  Blas() {
    blas(cublasCreate(&handle));
    try { blas(cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)); }
    catch (...) { cublasDestroy(handle); throw; }
  }
  ~Blas() { cublasDestroy(handle); }
};

__global__ void scale(B* x, const B* scalar, float constant, int n) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i < n) x[i] = b(f(x[i]) * (scalar ? f(*scalar) : constant));
}
__global__ void add(const B* x, const B* y, B* z, int n) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i < n) z[i] = b(f(x[i]) + f(y[i]));
}
__device__ B gelu_value(float v) {
  const float cube = v * v * v;
  return b(0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * cube))));
}
__global__ void gelu(const B* x, B* y, int n) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i < n) y[i] = gelu_value(f(x[i]));
}
// gelu followed by product(layer=-1) in one pass, with the same BF16 boundaries.
__global__ void gelu_product(const B* gate, const B* up, B* y, int n) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i < n) y[i] = b(f(gelu_value(f(gate[i]))) * f(up[i]));
}
__global__ void product(B* x, const B* y, int n, int stride) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i < n) x[i] = b(f(x[i]) * f(y[i / kHidden * stride + i % kHidden]));
}
// Unfused direct-multiplier RMSNorm with a BF16 boundary on every output.
// Lanes threads share a row; one 256-thread block covers 256/Lanes rows.
// With a residual, y = residual + norm(x) (each BF16-rounded); y may alias it.
template <int Lanes>
__global__ void rms(const B* x, const B* weight, B* y, int rows, int width, const B* residual) {
  constexpr int kWarps = Lanes / 32;
  __shared__ float partial[kWarps > 1 ? 256 / 32 : 1];
  const int row = blockIdx.x * (256 / Lanes) + threadIdx.x / Lanes, lane = threadIdx.x % Lanes;
  const bool active = row < rows;
  x += std::size_t(active ? row : 0) * width;
  y += std::size_t(active ? row : 0) * width;
  if (residual) residual += std::size_t(active ? row : 0) * width;
  float sum = 0;
  if (active)
    for (int i = lane * 4; i < width; i += Lanes * 4) {
      const auto pair = *reinterpret_cast<const uint2*>(x + i);
      const float2 low = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&pair.x));
      const float2 high = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&pair.y));
      sum += (low.x * low.x + low.y * low.y) + (high.x * high.x + high.y * high.y);
    }
  for (int step = (Lanes < 32 ? Lanes : 32) / 2; step; step /= 2) sum += __shfl_xor_sync(0xffffffff, sum, step);
  if constexpr (kWarps > 1) {
    if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = sum;
    __syncthreads();
    sum = 0;
    for (int j = 0; j < kWarps; ++j) sum += partial[threadIdx.x / Lanes * kWarps + j];
  }
  if (!active) return;
  const float inv = rsqrtf(sum * (1.f / width) + kEpsilon);
  for (int i = lane * 4; i < width; i += Lanes * 4)
    for (int j = 0; j < 4; ++j) {
      const B value = b((f(x[i + j]) * inv) * (weight ? f(weight[i + j]) : 1.f));
      y[i + j] = residual ? b(f(residual[i + j]) + f(value)) : value;
    }
}
void normalize(const B* x, const B* weight, B* y, int rows, int width, const B* residual = nullptr) {
  if (width % 4) throw std::invalid_argument("embeddinggemma2 RMSNorm width must be a multiple of four");
  const int quads = width / 4;
  if (quads <= 16) rms<16><<<(rows + 15) / 16, 256>>>(x, weight, y, rows, width, residual);
  else if (quads <= 32) rms<32><<<(rows + 7) / 8, 256>>>(x, weight, y, rows, width, residual);
  else if (quads <= 64) rms<64><<<(rows + 3) / 4, 256>>>(x, weight, y, rows, width, residual);
  else rms<128><<<(rows + 1) / 2, 256>>>(x, weight, y, rows, width, residual);
}
__global__ void rope_factors(const float* frequencies, B* cosine, B* sine, int n, int d, int stride) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i < n * d) {
    const float angle = float(i / d % stride) * frequencies[i % (d / 2)];
    cosine[i] = b(cosf(angle));
    sine[i] = b(sinf(angle));
  }
}
__global__ void rope(const B* x, B* y, const B* cosine, const B* sine, int rows, int heads, int d) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i >= rows * heads * d) return;
  const int channel = i % d, factor = (i / (heads * d)) * d + channel;
  const float rotated = channel < d / 2 ? -f(x[i + d / 2]) : f(x[i - d / 2]);
  y[i] = b(f(b(f(x[i]) * f(cosine[factor]))) + f(b(rotated * f(sine[factor]))));
}
// Scores and probabilities round to BF16, with the softmax reduction in FP32.
__global__ void softmax(B* scores, int queries, int keys, int query_start, int key_start, bool global, B* mask) {
  __shared__ float reduction[256];
  B* row = scores + blockIdx.x * keys;
  const int q = query_start + blockIdx.x % queries;
  float maximum = -INFINITY;
  for (int i = threadIdx.x; i < keys; i += 256)
    if (global || abs(q - key_start - i) <= kWindow) maximum = fmaxf(maximum, f(row[i]));
  reduction[threadIdx.x] = maximum;
  __syncthreads();
  for (int step = 128; step; step /= 2) {
    if (threadIdx.x < step) reduction[threadIdx.x] = fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + step]);
    __syncthreads();
  }
  maximum = reduction[0];
  float sum = 0;
  for (int i = threadIdx.x; i < keys; i += 256)
    if (global || abs(q - key_start - i) <= kWindow) sum += expf(f(row[i]) - maximum);
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int step = 128; step; step /= 2) {
    if (threadIdx.x < step) reduction[threadIdx.x] += reduction[threadIdx.x + step];
    __syncthreads();
  }
  for (int i = threadIdx.x; i < keys; i += 256) {
    const bool visible = global || abs(q - key_start - i) <= kWindow;
    row[i] = visible ? b(expf(f(row[i]) - maximum) / reduction[0]) : b(0);
    if (mask && blockIdx.x < queries) mask[blockIdx.x * keys + i] = b(visible ? 1.f : 0.f);
  }
}
// The pinned FP32 softmax uses a warp reduction for rows of up to 2,048 keys.
// Preserve its full key axis and per-lane accumulation order at this size.
__global__ void encoder_warp_softmax(B* scores,int queries,int keys,int start,bool global,B* mask) {
  B* row=scores+blockIdx.x*keys;
  const int query=start+blockIdx.x%queries,lane=threadIdx.x;
  float values[64],maximum=-INFINITY;
  const int iterations=(keys+31)/32;
  for (int j=0;j<iterations;++j) {
    const int key=lane+j*32;
    values[j]=key<keys && (global || abs(query-key)<=kWindow)?f(row[key]):-INFINITY;
    maximum=fmaxf(maximum,values[j]);
  }
  for (int step=16;step;step/=2) maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,step));
  float sum=0;
  for (int j=0;j<iterations;++j) {values[j]=expf(values[j]-maximum);sum+=values[j];}
  for (int step=16;step;step/=2) sum+=__shfl_xor_sync(0xffffffff,sum,step);
  for (int j=0;j<iterations;++j) {
    const int key=lane+j*32;
    if (key<keys) {
      row[key]=b(values[j]/sum);
      if (mask && blockIdx.x<queries) mask[blockIdx.x*keys+key]=b(global || abs(query-key)<=kWindow?1.f:0.f);
    }
  }
}
// Packed sequences share one padded key axis of at most 2,048 keys. Rows are
// [head][sequence][query]; keys past a sequence's length are invisible, and
// padding queries produce zero probabilities so their context stays finite.
__global__ void packed_softmax(B* scores,int queries,int sequences,int keys,int start,bool global,const int* lengths) {
  B* row=scores+std::size_t(blockIdx.x)*keys;
  const int query=start+blockIdx.x%queries,length=lengths[blockIdx.x/queries%sequences],lane=threadIdx.x;
  const int iterations=(keys+31)/32;
  if (query>=length) {
    for (int key=lane;key<keys;key+=32) row[key]=b(0);
    return;
  }
  float values[64],maximum=-INFINITY;
  for (int j=0;j<iterations;++j) {
    const int key=lane+j*32;
    values[j]=key<length && (global || abs(query-key)<=kWindow)?f(row[key]):-INFINITY;
    maximum=fmaxf(maximum,values[j]);
  }
  for (int step=16;step;step/=2) maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,step));
  float sum=0;
  for (int j=0;j<iterations;++j) {values[j]=expf(values[j]-maximum);sum+=values[j];}
  for (int step=16;step;step/=2) sum+=__shfl_xor_sync(0xffffffff,sum,step);
  for (int j=0;j<iterations;++j) {
    const int key=lane+j*32;
    if (key<keys) row[key]=b(values[j]/sum);
  }
}
// Grid (3, sequences): sequence s owns rows [s*stride, s*stride+length).
__global__ void pool(const B* projected, B* mean, int stride, const int* lengths) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  if (i >= kDimension) return;
  const int rows = lengths ? lengths[blockIdx.y] : stride;
  projected += std::size_t(blockIdx.y) * stride * kDimension;
  float sum = 0;
  for (int row = 0; row < rows; ++row) sum += f(projected[row * kDimension + i]);
  mean[blockIdx.y * kDimension + i] = b(f(b(sum)) / f(b(float(rows))));
}
// One block per vector; input rows have kDimension stride, output rows width.
__global__ void l2_normalize(const B* x, B* y, int width) {
  x += blockIdx.x * kDimension;
  y += blockIdx.x * width;
  __shared__ float sum[256];
  float partial = 0;
  for (int i = threadIdx.x; i < width; i += 256) partial += f(x[i]) * f(x[i]);
  sum[threadIdx.x] = partial;
  __syncthreads();
  for (int step = 128; step; step /= 2) {
    if (threadIdx.x < step) sum[threadIdx.x] += sum[threadIdx.x + step];
    __syncthreads();
  }
  const float denominator = fmaxf(f(b(sqrtf(sum[0]))), f(b(1.e-12f)));
  for (int i = threadIdx.x; i < width; i += 256) y[i] = b(f(x[i]) / denominator);
}
// Per token: hidden, embeddings (PLE source), layer PLE, normalized, branch;
// then one phase-shared region. Attention needs Q, rotated Q, rotated K,
// normalized V, one head-batched score tile and K/V/cos/sin, which context
// replaces after rotation. The MLP's three buffers and the final projection
// and pooling reuse the same region.
constexpr std::size_t kPersistentWidth = 5 * kHidden;
constexpr std::size_t kPhaseWidth = 2 * 2048 + 6 * kHidden + kHeads * kTile;
static_assert(kPhaseWidth >= 3 * std::size_t(kIntermediate) && kPhaseWidth >= 4 * std::size_t(kDimension));
std::size_t scratch_elements(int n) {
  return std::size_t(n) * (kPersistentWidth + kPhaseWidth);
}
int checked_capacity(const std::string& directory, int n) {
  validate_bundle_config(directory);
  if (n < 1 || n > kMaxTokens) throw std::invalid_argument("embeddinggemma2 capacity must be in 1..8192");
  cudaDeviceProp properties{};
  int device = 0;
  check_cuda(cudaGetDevice(&device), "get embedding device");
  check_cuda(cudaGetDeviceProperties(&properties, device), "embedding device properties");
  if (properties.major != 12 || properties.minor != 0)
    throw std::runtime_error("embeddinggemma2 requires an SM120 GPU");
  // The default 1 KiB per-thread stack reserves ~280 MiB; launches grow it to
  // the largest frame actually used.
  check_cuda(cudaDeviceSetLimit(cudaLimitStackSize, 0), "embedding stack limit");
  return n;
}
#include "vision.cuh"
#include "audio.cuh"

std::size_t encoder_scratch_bytes(int n,bool vision) {
  // Sequence embeddings survive visual execution. Everything after hidden is
  // temporary: vision finishes before PLE/text execution on the same stream.
  const auto hidden_bytes=std::size_t(n)*kHidden*sizeof(B);
  return hidden_bytes+std::max(scratch_elements(n)*sizeof(B)-hidden_bytes,
      vision?VisionEncoder::workspace_bytes():std::size_t(0));
}
}  // namespace

struct Encoder::Impl {
  const int capacity;
  Blas blas_handle;
  std::unique_ptr<Device> weights;
  Device scratch, lengths, frequencies;
  // The token table stays in the mapped file; batches gather rows on the host.
  std::unique_ptr<component::File> text_file;
  const B* host_embedding{};
  cuda_detail::PinnedHostAllocation staging;
  std::unique_ptr<VisionEncoder> vision_encoder;
  std::unique_ptr<AudioEncoder> audio_encoder;
  std::vector<const B*> w;
  B *hidden, *embeds, *ple, *normalized, *branch, *phase, *q, *qr, *kr, *vn, *scores, *k, *v, *cosine, *sine, *context;
  B *gate, *up, *activated, *projected;
  // All layers' PLE rows when they fit after this batch's phase buffers.
  B* ple_all{};
  explicit Impl(const std::string& directory, int n, bool vision, bool audio)
      : capacity(checked_capacity(directory, n)), scratch(encoder_scratch_bytes(n,vision)),
        lengths(n * sizeof(int)), frequencies((128 + 256) * sizeof(float)),
        text_file(std::make_unique<component::File>(directory + "/text.safetensors", tensor_specs())),
        staging(std::size_t(n) * kHidden * sizeof(B)) {
    std::size_t device_bytes = 0;
    for (const auto& tensor : text_file->tensors())
      if (tensor.physical_id != Embedding) device_bytes += (tensor.bytes + 4095) / 4096 * 4096;
    weights = std::make_unique<Device>(device_bytes);
    w.resize(kTensorCount);
    std::size_t offset = 0;
    for (const auto& tensor : text_file->tensors()) {
      if (tensor.physical_id == Embedding) {
        host_embedding = reinterpret_cast<const B*>(tensor.data);
        continue;
      }
      auto* destination = static_cast<std::uint8_t*>(weights->data()) + offset;
      check_cuda(cudaMemcpy(destination, tensor.data, tensor.bytes, cudaMemcpyHostToDevice), "copy embedding weights");
      w[tensor.physical_id] = reinterpret_cast<B*>(destination);
      offset += (tensor.bytes + 4095) / 4096 * 4096;
    }
    text_file->drop_resident_pages();
    B* cursor = static_cast<B*>(scratch.data());
    auto take = [&](int width) { B* pointer = cursor; cursor += std::size_t(n) * width; return pointer; };
    hidden=take(kHidden); embeds=take(kHidden); ple=take(kHidden); normalized=take(kHidden); branch=take(kHidden);
    phase = cursor;
    check_cuda(cudaMemcpy(frequencies.data(), kRopeFrequencies.data(), kRopeFrequencies.size()*sizeof(float), cudaMemcpyHostToDevice), "copy RoPE frequencies");
    if (vision) vision_encoder=std::make_unique<VisionEncoder>(directory,embeds,blas_handle);
    if (audio) audio_encoder=std::make_unique<AudioEncoder>(directory,n,blas_handle);
  }
  // Phase buffers are packed for this batch's rows, leaving the tail free.
  void layout(int n) {
    B* cursor = phase;
    auto take = [&](int width) { B* pointer = cursor; cursor += std::size_t(n) * width; return pointer; };
    q=take(2048); qr=take(2048); kr=take(kHidden); vn=take(kHidden); scores=take(kHeads*kTile);
    k=take(kHidden); v=take(kHidden); cosine=take(kHidden); sine=take(kHidden);
    context=k;
    cursor = phase;
    gate=take(kIntermediate); up=take(kIntermediate); activated=take(kIntermediate);
    projected=phase;
    ple_all = std::size_t(n)*(kPhaseWidth+kLayers*kHidden) <= std::size_t(capacity)*kPhaseWidth
        ? phase+std::size_t(n)*kPhaseWidth : nullptr;
  }
  void linear(const B* input, const B* weight, B* output, int n, int out, int in = kHidden, float alpha = 1.f) {
    const float beta=0.f;
    blas(cublasGemmEx(blas_handle.handle, CUBLAS_OP_T, CUBLAS_OP_N, out, n, in,
        &alpha, weight, CUDA_R_16BF, in, input, CUDA_R_16BF, in,
        &beta, output, CUDA_R_16BF, out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  }
  std::vector<float> download(const B* data, std::size_t count) {
    std::vector<B> raw(count);
    check_cuda(cudaMemcpy(raw.data(), data, count*sizeof(B), cudaMemcpyDeviceToHost), "read embedding output");
    std::vector<float> values(count);
    for (std::size_t i=0; i<count; ++i) values[i]=__bfloat162float(raw[i]);
    return values;
  }
  void capture(const Capture& callback, const std::string& name, const B* data, std::vector<int> shape) {
    if (!callback) return;
    const auto count=std::accumulate(shape.begin(), shape.end(), std::size_t(1), std::multiplies<>());
    callback(name, shape, download(data, count));
  }
  void attention(int n, int layer, const Capture& cap) {
    const int d=head_dim(layer), kv=kv_heads(layer);
    const bool global=global_layer(layer);
    const float alpha=1.f, beta=0.f;
    std::vector<float> mask;
    if (cap && n<=1026) mask.resize(n*n);
    for (int start=0; start<n; start+=kTile) {
      const int count=std::min(kTile,n-start);
      const int first=global || n<=2048?0:std::max(0,start-kWindow);
      const int last=global || n<=2048?n:std::min(n,start+count+kWindow);
      const int keys=last-first;
      // Heads sharing a KV head use independent matrices in one batched call.
      // Only the head launch dimension changes; token tiles and BF16 boundaries stay fixed.
      const int group=kHeads/kv;
      for (int kh=0; kh<kv; ++kh) {
        const int head=kh*group;
        const long long score_stride=static_cast<long long>(keys)*count;
        blas(cublasGemmStridedBatchedEx(blas_handle.handle,CUBLAS_OP_T,CUBLAS_OP_N,keys,count,d,
            &alpha,kr+first*kv*d+kh*d,CUDA_R_16BF,kv*d,0,
            qr+start*kHeads*d+head*d,CUDA_R_16BF,kHeads*d,d,
            &beta,scores,CUDA_R_16BF,keys,score_stride,group,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
        if (n<=2048) encoder_warp_softmax<<<count*group,32>>>(scores,count,keys,start,global,!mask.empty() && kh==0?q:nullptr);
        else softmax<<<count*group,256>>>(scores,count,keys,start,first,global,!mask.empty() && kh==0 ? q : nullptr);
        if (!mask.empty() && kh==0) {
          const auto tile=download(q,count*keys);
          for (int row=0;row<count;++row)
            std::copy_n(tile.data()+row*keys,keys,mask.data()+(start+row)*n+first);
        }
        blas(cublasGemmStridedBatchedEx(blas_handle.handle,CUBLAS_OP_N,CUBLAS_OP_N,d,count,keys,
            &alpha,vn+first*kv*d+kh*d,CUDA_R_16BF,kv*d,0,scores,CUDA_R_16BF,keys,score_stride,
            &beta,context+start*kHeads*d+head*d,CUDA_R_16BF,kHeads*d,d,group,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
      }
    }
    if (!mask.empty()) cap("layer."+std::to_string(layer)+".mask",{n,n},mask);
  }
  // Sequences occupy fixed-stride row blocks; one strided batch covers every
  // sequence for a head, so launches do not scale with the sequence count.
  void packed_attention(int sequences, int stride, int layer) {
    const int d=head_dim(layer), kv=kv_heads(layer), group=kHeads/kv;
    const bool global=global_layer(layer);
    const float alpha=1.f, beta=0.f;
    const auto* lengths_device=static_cast<const int*>(lengths.data());
    for (int start=0; start<stride; start+=kTile) {
      const int count=std::min(kTile,stride-start);
      const long long score_stride=static_cast<long long>(stride)*count;
      const long long key_stride=static_cast<long long>(stride)*kv*d, query_stride=static_cast<long long>(stride)*kHeads*d;
      for (int head=0; head<kHeads; ++head)
        blas(cublasGemmStridedBatchedEx(blas_handle.handle,CUBLAS_OP_T,CUBLAS_OP_N,stride,count,d,
            &alpha,kr+head/group*d,CUDA_R_16BF,kv*d,key_stride,
            qr+start*kHeads*d+head*d,CUDA_R_16BF,kHeads*d,query_stride,
            &beta,scores+head*sequences*score_stride,CUDA_R_16BF,stride,score_stride,sequences,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
      packed_softmax<<<kHeads*sequences*count,32>>>(scores,count,sequences,stride,start,global,lengths_device);
      for (int head=0; head<kHeads; ++head)
        blas(cublasGemmStridedBatchedEx(blas_handle.handle,CUBLAS_OP_N,CUBLAS_OP_N,d,count,stride,
            &alpha,vn+head/group*d,CUDA_R_16BF,kv*d,key_stride,
            scores+head*sequences*score_stride,CUDA_R_16BF,stride,score_stride,
            &beta,context+start*kHeads*d+head*d,CUDA_R_16BF,kHeads*d,query_stride,sequences,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    }
  }
  void run_layer(int n, int layer, const Capture& cap, int sequences = 1) {
    const int h=n*kHidden;
    const auto prefix="layer."+std::to_string(layer)+".";
    const int d=head_dim(layer);
    const bool deep=layer==0 || layer==5;
    auto weight=[&](LayerWeight role){return w[weight_id(layer,role)];};
    auto save=[&](const std::string& name,const B* data,int width) {
      if (deep) capture(cap,prefix+name,data,{n,width});
    };
    normalize(hidden,weight(InputNorm),normalized,n,kHidden);
    save("input_norm",normalized,kHidden);
    linear(normalized,weight(Query),q,n,kHeads*d);
    linear(normalized,weight(Key),k,n,kHidden);
    linear(normalized,weight(Value),v,n,kHidden);
    save("q",q,kHeads*d); save("k",k,kHidden); save("v",v,kHidden);
    normalize(q,weight(QueryNorm),q,n*kHeads,d);
    normalize(k,weight(KeyNorm),k,n*kv_heads(layer),d);
    normalize(v,nullptr,vn,n*kv_heads(layer),d);
    save("q_norm",q,kHeads*d); save("k_norm",k,kHidden); save("v_norm",vn,kHidden);
    rope_factors<<<(n*d+255)/256,256>>>(static_cast<float*>(frequencies.data())+(global_layer(layer)?128:0),cosine,sine,n,d,n/sequences);
    save("cos",cosine,d); save("sin",sine,d);
    rope<<<(n*kHeads*d+255)/256,256>>>(q,qr,cosine,sine,n,kHeads,d);
    rope<<<(h+255)/256,256>>>(k,kr,cosine,sine,n,kv_heads(layer),d);
    save("q_rope",qr,kHeads*d); save("k_rope",kr,kHidden);
    if (sequences>1) packed_attention(sequences,n/sequences,layer);
    else attention(n,layer,deep?cap:Capture{});
    save("context",context,kHeads*d);
    linear(context,weight(AttentionOutput),branch,n,kHidden,kHeads*d);
    save("attention_output",branch,kHidden);
    normalize(branch,weight(PostAttentionNorm),branch,n,kHidden);
    save("post_attention_norm",branch,kHidden);
    add<<<(h+255)/256,256>>>(hidden,branch,hidden,h);
    save("attention_residual",hidden,kHidden);
    normalize(hidden,weight(PreFeedforwardNorm),normalized,n,kHidden);
    save("pre_ffn_norm",normalized,kHidden);
    linear(normalized,weight(Gate),gate,n,kIntermediate);
    linear(normalized,weight(Up),up,n,kIntermediate);
    save("mlp_gate",gate,kIntermediate); save("mlp_up",up,kIntermediate);
    gelu<<<(n*kIntermediate+255)/256,256>>>(gate,activated,n*kIntermediate);
    save("mlp_activation",activated,kIntermediate);
    product<<<(n*kIntermediate+255)/256,256>>>(activated,up,n*kIntermediate,kHidden);
    save("mlp_product",activated,kIntermediate);
    linear(activated,weight(Down),branch,n,kHidden,kIntermediate);
    save("mlp_down",branch,kHidden);
    normalize(branch,weight(PostFeedforwardNorm),branch,n,kHidden);
    save("post_ffn_norm",branch,kHidden);
    add<<<(h+255)/256,256>>>(hidden,branch,hidden,h);
    save("ffn_residual",hidden,kHidden);
    linear(hidden,weight(PleGate),gate,n,kHidden);
    save("ple_gate",gate,kHidden);
    gelu<<<(h+255)/256,256>>>(gate,activated,h);
    save("ple_activation",activated,kHidden);
    if (ple_all) product<<<(h+255)/256,256>>>(activated,ple_all+layer*kHidden,h,kLayers*kHidden);
    else product<<<(h+255)/256,256>>>(activated,ple,h,kHidden);
    save("ple_product",activated,kHidden);
    linear(activated,weight(PleOutput),branch,n,kHidden);
    save("ple_output",branch,kHidden);
    normalize(branch,weight(PlePostNorm),branch,n,kHidden);
    save("ple_norm",branch,kHidden);
    add<<<(h+255)/256,256>>>(hidden,branch,hidden,h);
    save("ple_residual",hidden,kHidden);
    scale<<<(h+255)/256,256>>>(hidden,weight(Scalar),0,h);
    capture(cap,prefix+"output",hidden,{n,kHidden});
    check_cuda(cudaGetLastError(),"embedding layer kernels");
  }
  // PLE scale (1/sqrt(512)) is folded into the projection GEMM.
  static constexpr float kPleScale = 0.04419417382415922f;
  // One layer's PLE rows, for batches too large to hold every layer at once.
  void layer_ple(int n, int layer) {
    linear(embeds,w[PleProjection]+std::size_t(layer)*kHidden*kHidden,ple,n,kHidden,kHidden,kPleScale);
    normalize(ple,w[PleNorm],ple,n,kHidden);
  }
  // Inputs share one forward in fixed-stride row blocks. Results are
  // input-major, then in dimension order.
  std::vector<std::vector<float>> run(const std::vector<const PreparedInput*>& inputs,
      const std::vector<int>& dimensions, const Capture& cap, const Cancelled& cancelled) {
    const int sequences=inputs.size();
    if (!sequences) throw std::invalid_argument("embeddinggemma2 missing inputs");
    if (cap && sequences>1) throw std::invalid_argument("embeddinggemma2 capture requires one input");
    std::size_t longest=0;
    for (const auto* input:inputs) {
      const auto& tokens=input->tokens;
      if (tokens.empty() || tokens.size()>static_cast<std::size_t>(capacity))
        throw std::invalid_argument("embeddinggemma2 token count exceeds configured capacity or is empty");
      for (auto token:tokens) if (token>=kVocabulary) throw std::invalid_argument("embeddinggemma2 invalid token ID");
      longest=std::max(longest,tokens.size());
    }
    if (dimensions.empty()) throw std::invalid_argument("embeddinggemma2 missing output dimensions");
    for (int d:dimensions) if (!supported_dimension(d)) throw std::invalid_argument("embeddinggemma2 unsupported dimension");
    const int stride=sequences==1?longest:(longest+7)/8*8;
    if (sequences>1 && (longest>kPackedMaxTokens || std::size_t(stride)*sequences>static_cast<std::size_t>(capacity)))
      throw std::invalid_argument("embeddinggemma2 packed inputs exceed capacity");
    // Polling must not drain the stream; scratch is only released after a sync.
    auto poll = [&] {
      if (cancelled && cancelled()) {
        check_cuda(cudaDeviceSynchronize(), "embedding layer completion");
        throw std::runtime_error("embeddinggemma2 request cancelled");
      }
    };
    poll();
    const int n=stride*sequences, h=n*kHidden;
    std::vector<int> sizes(sequences);
    auto* gathered=static_cast<B*>(staging.data());
    for (int s=0;s<sequences;++s) {
      const auto& tokens=inputs[s]->tokens;
      for (int t=0;t<stride;++t) {
        const auto id=t<static_cast<int>(tokens.size())?tokens[t]:0;
        // The official multimodal wrapper uses the PAD row for soft-token
        // placeholders even when text is supplied without modality features.
        const auto row=(id==258880 || id==258881 || id==258884) ? 0 : id;
        std::memcpy(gathered+(std::size_t(s)*stride+t)*kHidden,host_embedding+std::size_t(row)*kHidden,kHidden*sizeof(B));
      }
      sizes[s]=tokens.size();
    }
    check_cuda(cudaMemcpyAsync(hidden,gathered,std::size_t(h)*sizeof(B),cudaMemcpyHostToDevice),"copy token embeddings");
    check_cuda(cudaMemcpy(lengths.data(),sizes.data(),sequences*sizeof(int),cudaMemcpyHostToDevice),"copy embedding lengths");
    scale<<<(h+255)/256,256>>>(hidden,nullptr,22.625f,h);
    // Images from every packed input share vision forwards up to kMaxPatches
    // rows; captures keep one image per forward so names stay per image.
    std::vector<const runtime::ImageInput*> images;
    std::vector<B*> outputs;
    int rows=0;
    auto flush=[&](const Capture& capture) {
      if (images.empty()) return;
      poll();
      vision_encoder->run(images,outputs,capture,poll);
      poll();
      images.clear(); outputs.clear(); rows=0;
    };
    for (int s=0;s<sequences;++s) {
      B* sequence=hidden+std::size_t(s)*stride*kHidden;
      for (std::size_t i=0;i<inputs[s]->visuals.size();++i) {
        const auto& image=*inputs[s]->visuals[i];
        const int patches=(image.end-image.begin)*9;
        if (rows+patches>vision::kMaxPatches) flush({});
        images.push_back(&image); outputs.push_back(sequence+image.begin*kHidden); rows+=patches;
        if (cap) flush(Capture([&,i](const auto& name,const auto& shape,const auto& values) {
          cap("image."+std::to_string(i)+"."+name,shape,values);
        }));
      }
    }
    flush({});
    for (int s=0;s<sequences;++s) {
      B* sequence=hidden+std::size_t(s)*stride*kHidden;
      for (std::size_t i=0;i<inputs[s]->audios.size();++i) {
        poll();
        const auto& audio=*inputs[s]->audios[i];
        audio_encoder->run(audio,sequence+audio.begin*kHidden,cap ? Capture([&](const auto& name,const auto& shape,const auto& values) {
          cap("audio."+std::to_string(i)+"."+name,shape,values);
        }) : Capture{},poll);
        poll();
      }
    }
    capture(cap,"embedding",hidden,{n,kHidden});
    layout(n);
    if (ple_all) {
      linear(hidden,w[PleProjection],ple_all,n,kLayers*kHidden,kHidden,kPleScale);
      capture(cap,"ple.scaled",ple_all,{n,kLayers,kHidden});
      normalize(ple_all,w[PleNorm],ple_all,n*kLayers,kHidden);
      capture(cap,"ple",ple_all,{n,kLayers,kHidden});
    } else {
      check_cuda(cudaMemcpyAsync(embeds,hidden,std::size_t(h)*sizeof(B),cudaMemcpyDeviceToDevice),"copy PLE source");
    }
    for (int layer=0; layer<kLayers; ++layer) {
      poll();
      if (!ple_all) layer_ple(n,layer);
      run_layer(n,layer,cap,sequences);
    }
    poll();
    normalize(hidden,w[FinalNorm],normalized,n,kHidden);
    capture(cap,"final_norm",normalized,{n,kHidden});
    linear(normalized,w[OutputProjection],projected,n,kDimension);
    capture(cap,"projected",projected,{n,kDimension});
    B *mean=projected+std::size_t(n)*kDimension, *unit=mean+std::size_t(sequences)*kDimension,
      *reduced=unit+std::size_t(sequences)*kDimension;
    pool<<<dim3(3,sequences),256>>>(projected,mean,stride,sequences>1?static_cast<const int*>(lengths.data()):nullptr);
    capture(cap,"mean",mean,{kDimension});
    l2_normalize<<<sequences,256>>>(mean,unit,kDimension);
    capture(cap,"normalized",unit,{kDimension});
    std::vector<std::vector<float>> result(std::size_t(sequences)*dimensions.size());
    for (std::size_t j=0;j<dimensions.size();++j) {
      const int d=dimensions[j];
      l2_normalize<<<sequences,256>>>(unit,reduced,d);
      const auto values=download(reduced,std::size_t(sequences)*d);
      for (int s=0;s<sequences;++s) {
        std::vector<float> vector(values.begin()+std::size_t(s)*d,values.begin()+std::size_t(s+1)*d);
        double norm=0;
        for (float value:vector) {
          if (!std::isfinite(value)) throw std::runtime_error("embeddinggemma2 produced nonfinite output");
          norm+=double(value)*value;
        }
        if (norm==0) throw std::runtime_error("embeddinggemma2 produced a zero vector");
        if (cap) cap("embedding."+std::to_string(d),{d},vector);
        result[s*dimensions.size()+j]=std::move(vector);
      }
    }
    return result;
  }
};
Encoder::Encoder(const std::string& directory,int max_tokens,bool vision,bool audio):impl_(std::make_unique<Impl>(directory,max_tokens,vision,audio)) {}
Encoder::~Encoder()=default;
void Encoder::capture_layer(int layer,const std::vector<float>& hidden,
    const std::vector<float>& ple,Capture capture) {
  if (layer<0 || layer>=kLayers || hidden.empty() || hidden.size()%kHidden ||
      hidden.size()/kHidden>static_cast<std::size_t>(impl_->capacity) || ple.size()!=hidden.size()*kLayers)
    throw std::invalid_argument("embeddinggemma2 invalid layer control shape");
  auto upload=[](const std::vector<float>& values,B* destination) {
    std::vector<B> raw(values.size());
    for (std::size_t i=0;i<values.size();++i) {
      if (!std::isfinite(values[i])) throw std::invalid_argument("nonfinite layer control input");
      raw[i]=__float2bfloat16_rn(values[i]);
    }
    check_cuda(cudaMemcpy(destination,raw.data(),raw.size()*sizeof(B),cudaMemcpyHostToDevice),"load layer control input");
  };
  upload(hidden,impl_->hidden);
  const std::size_t rows=hidden.size()/kHidden;
  std::vector<float> slice(hidden.size());
  for (std::size_t t=0;t<rows;++t)
    std::copy_n(ple.begin()+(t*kLayers+layer)*kHidden,kHidden,slice.begin()+t*kHidden);
  upload(slice,impl_->ple);
  impl_->layout(rows);
  impl_->ple_all=nullptr;
  impl_->run_layer(rows,layer,capture);
  check_cuda(cudaDeviceSynchronize(),"layer control completion");
}
std::vector<std::vector<float>> Encoder::encode(const PreparedInput& input,
    const std::vector<int>& dimensions,Capture capture,Cancelled cancelled) {
  validate_input(input,impl_->capacity);
  if (!input.visuals.empty() && !impl_->vision_encoder) throw std::invalid_argument("embeddinggemma2 vision is disabled");
  if (!input.audios.empty() && !impl_->audio_encoder) throw std::invalid_argument("embeddinggemma2 audio is disabled");
  return impl_->run({&input},dimensions,capture,cancelled);
}
std::vector<std::vector<float>> Encoder::encode_batch(const std::vector<const PreparedInput*>& inputs,
    int dimension,Cancelled cancelled) {
  for (const auto* input:inputs) {
    validate_input(*input,impl_->capacity);
    if (!input->visuals.empty() && !impl_->vision_encoder) throw std::invalid_argument("embeddinggemma2 vision is disabled");
    if (!input->audios.empty() && !impl_->audio_encoder) throw std::invalid_argument("embeddinggemma2 audio is disabled");
  }
  auto length=[&](std::size_t i){return inputs[i]->tokens.size();};
  std::vector<std::size_t> order(inputs.size());
  std::iota(order.begin(),order.end(),std::size_t(0));
  std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return length(a)<length(b);});
  std::vector<std::vector<float>> result(inputs.size());
  for (std::size_t first=0;first<order.size();) {
    // Ascending lengths keep padding low; padding beyond 2x is only allowed in small packs.
    std::size_t last=first+1,total=length(order[first]);
    for (;last<order.size();++last) {
      const auto size=length(order[last]),rows=(last-first+1)*((size+7)/8*8);
      if (size>std::size_t(kPackedMaxTokens) || rows>std::size_t(impl_->capacity) ||
          rows>std::max<std::size_t>(2*(total+size),2048)) break;
      total+=size;
    }
    std::vector<const PreparedInput*> pack;
    for (auto i=first;i<last;++i) pack.push_back(inputs[order[i]]);
    auto vectors=impl_->run(pack,{dimension},{},cancelled);
    for (auto i=first;i<last;++i) result[order[i]]=std::move(vectors[i-first]);
    first=last;
  }
  return result;
}
std::vector<std::vector<float>> Encoder::encode_raw(const std::vector<std::uint32_t>& tokens,
    const std::vector<int>& dimensions,Capture capture) {
  const PreparedInput input{tokens,{}};
  return impl_->run({&input},dimensions,capture,{});
}
void Encoder::capture_vision_layer(int layer,const runtime::ImageInput& image,
    const std::vector<float>& hidden,Capture capture) {
  if (!impl_->vision_encoder) throw std::invalid_argument("embeddinggemma2 vision is disabled");
  impl_->vision_encoder->control_layer(layer,image,hidden,capture);
}
void Encoder::capture_vision_bridge(const std::vector<float>& hidden,Capture capture) {
  if (!impl_->vision_encoder) throw std::invalid_argument("embeddinggemma2 vision is disabled");
  impl_->vision_encoder->control_bridge(hidden,capture);
}
std::size_t Encoder::vision_weight_bytes() const {return impl_->vision_encoder ? impl_->vision_encoder->weights->size() : 0;}
std::size_t Encoder::vision_scratch_bytes() const {return impl_->vision_encoder ? impl_->vision_encoder->scratch_bytes() : 0;}
void Encoder::capture_audio_subsample(const audio::Input& input,Capture capture) {
  if (!impl_->audio_encoder) throw std::invalid_argument("embeddinggemma2 audio is disabled");
  impl_->audio_encoder->control_subsample(input,capture);
}
void Encoder::capture_audio_layer(int layer,const std::vector<float>& hidden,Capture capture) {
  if (!impl_->audio_encoder) throw std::invalid_argument("embeddinggemma2 audio is disabled");
  impl_->audio_encoder->control_layer(layer,hidden,capture);
}
void Encoder::capture_audio_bridge(const std::vector<float>& hidden,Capture capture) {
  if (!impl_->audio_encoder) throw std::invalid_argument("embeddinggemma2 audio is disabled");
  impl_->audio_encoder->control_bridge(hidden,capture);
}
std::size_t Encoder::audio_weight_bytes() const {return impl_->audio_encoder ? impl_->audio_encoder->weights->size() : 0;}
std::size_t Encoder::audio_scratch_bytes() const {return impl_->audio_encoder ? impl_->audio_encoder->scratch_bytes() : 0;}
std::size_t Encoder::weight_bytes() const {return impl_->weights->size()+vision_weight_bytes()+audio_weight_bytes();}
std::size_t Encoder::scratch_bytes() const {
  return impl_->scratch.size()+impl_->frequencies.size()+
      (impl_->vision_encoder?impl_->vision_encoder->frequencies.size():0)+audio_scratch_bytes();
}
int Encoder::max_tokens() const {return impl_->capacity;}
}  // namespace gewell::embeddinggemma2
