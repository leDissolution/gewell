#pragma once
#include "gewell/models/gemma4/26b_a4b/model.h"
#include "bf16_activation.cuh"
#include "rope_inverse_frequency.cuh"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>

namespace gewell::gemma4_26b_a4b::sm120::bf16_math {
using B = __nv_bfloat16;
__device__ inline float value(B x) { return __bfloat162float(x); }
__device__ inline B rounded(float x) { return __float2bfloat16_rn(x); }
static __global__ void embedding(const B* weights, B* out, unsigned token) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < kHiddenSize) out[i] = rounded(value(weights[std::size_t(token) * kHiddenSize + i]) * kEmbeddingScale);
}
struct NormOutput {
  const B* weight;
  B* output;
  __device__ void operator()(float x, unsigned row, unsigned column, unsigned width) const {
    output[row*width+column] = rounded(x * (weight ? value(weight[column]) : 1.f));
  }
};
struct FeedForwardNormOutput {
  const B *shared_weight, *router_weight, *expert_weight;
  B *shared, *router, *expert;
  __device__ void operator()(float x, unsigned row, unsigned column, unsigned width) const {
    const unsigned i = row*width+column;
    shared[i] = rounded(x * value(shared_weight[column]));
    expert[i] = rounded(x * value(expert_weight[column]));
    // The router rounds the unweighted norm, the weighted product, and the
    // final scale separately. Sharing the RMS must preserve all three casts.
    router[i] = rounded(value(rounded(value(rounded(x)) * value(router_weight[column]))) *
                        (1.f / sqrtf(float(kHiddenSize))));
  }
};
template<class Output>
static __global__ void norm(const B* x, Output output, unsigned width) {
  __shared__ float sums[512];
  x += blockIdx.x * width;
  // Match the pinned Torch FP32 mean: four adjacent lanes per vector,
  // independent accumulators, then a descending tree reduction.
  float partial[4] = {0, 0, 0, 0};
  for (unsigned i = threadIdx.x * 4; i < width; i += blockDim.x * 4)
    for (unsigned j = 0; j < 4; ++j) partial[j] += value(x[i+j]) * value(x[i+j]);
  float sum = partial[0];
  for (unsigned j = 1; j < 4; ++j) sum += partial[j];
  sums[threadIdx.x] = sum; __syncthreads();
  for (unsigned stride = blockDim.x/2; stride; stride /= 2) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  // Torch specializes scalar pow(x, -0.5) to reciprocal square root on CUDA.
  float inv = rsqrtf(sums[0] * (1.f / float(width)) + kRmsEpsilon);
  for (unsigned i = threadIdx.x; i < width; i += blockDim.x)
    output(value(x[i]) * inv, blockIdx.x, i, width);
}
// Split the four independent reference accumulators across adjacent lanes,
// then combine them in their original order. More lanes also share the output
// transform; the reduction tree still uses exactly the original logical lanes.
template<class Output>
static __global__ void norm_parallel(const B* x, Output output,
                                    unsigned width, unsigned lanes) {
  __shared__ float sums[128];
  x += blockIdx.x * width;
  const unsigned lane = threadIdx.x / 4, part = threadIdx.x % 4;
  if (lane < lanes) {
    float partial = 0;
    for (unsigned i = lane * 4 + part; i < width; i += lanes * 4)
      partial += value(x[i]) * value(x[i]);
    const unsigned first = (threadIdx.x % 32) & ~3U;
    float sum = __shfl_sync(0xffffffff, partial, first);
    sum += __shfl_sync(0xffffffff, partial, first + 1);
    sum += __shfl_sync(0xffffffff, partial, first + 2);
    sum += __shfl_sync(0xffffffff, partial, first + 3);
    if (!part) sums[lane] = sum;
  }
  __syncthreads();
  for (unsigned stride = lanes / 2; stride; stride /= 2) {
    if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  float inv = rsqrtf(sums[0] * (1.f / float(width)) + kRmsEpsilon);
  for (unsigned i = threadIdx.x; i < width; i += blockDim.x)
    output(value(x[i]) * inv, blockIdx.x, i, width);
}
template<class Output>
inline void normalize_to(const B* x, Output output, unsigned rows, unsigned width, cudaStream_t stream) {
  unsigned dim = 1, height = 1;
  while (dim * 2 <= width / 4 && dim < 512) dim *= 2;
  while (height * 2 <= rows && height < 16) height *= 2;
  unsigned lanes = std::min(dim, 512 / height);
  // The shuffle reduction requires complete warps (four threads per lane).
  if (lanes >= 8 && lanes <= 128)
    norm_parallel<<<rows,512,0,stream>>>(x,output,width,lanes);
  else
    norm<<<rows,lanes,0,stream>>>(x,output,width);
}
inline void normalize(const B* x, const B* w, B* y, unsigned rows, unsigned width, cudaStream_t stream = nullptr) {
  normalize_to(x,NormOutput{w,y},rows,width,stream);
}
inline void normalize_feedforward(const B* x, const B* shared_weight, const B* router_weight,
                                 const B* expert_weight, B* shared, B* router, B* expert,
                                 unsigned rows, cudaStream_t stream) {
  normalize_to(x,FeedForwardNormOutput{shared_weight,router_weight,expert_weight,shared,router,expert},
               rows,kHiddenSize,stream);
}
static __global__ void add(const B* a, const B* b, B* y, unsigned n) {
  unsigned i = blockIdx.x * 256 + threadIdx.x;
  if (i < n) y[i] = rounded(value(a[i]) + value(b[i]));
}
static __global__ void scale(B* x, const B* scalar) {
  unsigned i = blockIdx.x * 256 + threadIdx.x;
  if (i < kHiddenSize) x[i] = rounded(value(x[i]) * value(*scalar));
}
static __global__ void activation(const B* gate, const B* up, B* product, unsigned n) {
  unsigned i = blockIdx.x * 256 + threadIdx.x;
  if (i < n) product[i] = rounded(gewell::detail::gelu_tanh_multiply_bf16(value(gate[i]), value(up[i])));
}
static __global__ void router_scale(B* x, const B* w) {
  unsigned i = blockIdx.x * 256 + threadIdx.x;
  if (i < kHiddenSize) x[i] = rounded(value(rounded(value(x[i]) * value(w[i]))) * (1.f / sqrtf(float(kHiddenSize))));
}
static __global__ void rope(const B* x, B* y, unsigned heads, bool global, unsigned position) {
  unsigned d = global ? 512 : 256, half = d / 2;
  unsigned i = blockIdx.x * 256 + threadIdx.x;
  if (i >= heads * d) return;
  unsigned channel = i % d, frequency = channel % half;
  if (global && frequency >= 64) { y[i] = x[i]; return; }
  float inv = global ? rope_inverse_frequency::get<true>(frequency) : rope_inverse_frequency::get<false>(frequency);
  float angle = float(position) * inv;
  B cosine = rounded(cosf(angle)), sine = rounded(sinf(angle));
  float rotated = channel < half ? -value(x[i + half]) : value(x[i - half]);
  y[i] = rounded(value(rounded(value(x[i]) * value(cosine))) + value(rounded(rotated * value(sine))));
}
static __global__ void row_rope(const B* x,B* y,unsigned heads,bool global,const unsigned* positions,unsigned rows) {
  unsigned d=global?512:256,half=d/2,i=blockIdx.x*256+threadIdx.x;
  if (i>=rows*heads*d) return;
  unsigned channel=i%d,frequency=channel%half;
  if (global && frequency>=64) { y[i]=x[i]; return; }
  float inv=global?rope_inverse_frequency::get<true>(frequency):rope_inverse_frequency::get<false>(frequency);
  float angle=float(positions[i/(heads*d)])*inv;
  B cosine=rounded(cosf(angle)),sine=rounded(sinf(angle));
  float rotated=channel<half?-value(x[i+half]):value(x[i-half]);
  y[i]=rounded(value(rounded(value(x[i])*value(cosine)))+value(rounded(rotated*value(sine))));
}
}
