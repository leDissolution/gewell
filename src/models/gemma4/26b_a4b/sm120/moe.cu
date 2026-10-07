#include "moe.cuh"
#include "moe_nvfp4.cuh"
#include "bf16_activation.cuh"
#include "gewell/models/gemma4/26b_a4b/model.h"
#include <cub/device/device_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <mma.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gewell::gemma4_26b_a4b::sm120 {
namespace {
using B = __nv_bfloat16;
using cuda_detail::check_cuda;
using C = cutlass::bfloat16_t;
// Fixed BF16 expert shapes: pipeline K64 loads, keeping K16 MMA accumulation
// and BF16 projection outputs. Problem sizes and pointers stay on the GPU.
using GroupedBf16 = cutlass::gemm::device::GemmGrouped<
    cutlass::gemm::kernel::DefaultGemmGrouped<
      C,cutlass::layout::RowMajor,cutlass::ComplexTransform::kNone,8,
      C,cutlass::layout::ColumnMajor,cutlass::ComplexTransform::kNone,8,
      C,cutlass::layout::RowMajor,float,cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<32,64,64>,cutlass::gemm::GemmShape<16,32,64>,
      cutlass::gemm::GemmShape<16,8,16>,
      cutlass::epilogue::thread::LinearCombination<C,8,float,float>,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,3>::GemmKernel>;
struct GroupedArguments {
  cutlass::gemm::GemmCoord sizes[256];
  C *input[256], *weight[256], *output[256];
  std::int64_t input_stride[256], weight_stride[256], output_stride[256];
};
template<bool GateUp>
__global__ void prepare_grouped(const B* input,const ExpertWeights* weights,B* output,
                                const int* offsets,GroupedArguments* args) {
  const unsigned group=threadIdx.x,expert=GateUp?group/2:group;
  if(expert>=128) return;
  const auto& projection=GateUp?(group%2?weights[expert].up:weights[expert].gate):weights[expert].down;
  constexpr unsigned K=GateUp?kHiddenSize:kExpertMlpSize,N=GateUp?kExpertMlpSize:kHiddenSize;
  const unsigned first=offsets[expert],rows=projection.storage==StorageType::bf16?offsets[expert+1]-first:0;
  args->sizes[group]={int(rows),int(N),int(K)};
  args->input[group]=reinterpret_cast<C*>(const_cast<B*>(input+std::size_t(first)*K));
  args->weight[group]=reinterpret_cast<C*>(const_cast<B*>(projection.bf16));
  args->output[group]=reinterpret_cast<C*>(output+std::size_t(first)*(GateUp?2*N:N)+(GateUp?group%2*N:0));
  args->input_stride[group]=args->weight_stride[group]=K;
  args->output_stride[group]=GateUp?2*N:N;
}
template<bool GateUp>
void run_grouped(const B* input,const ExpertWeights* weights,B* output,const int* offsets,
                 void* scratch,unsigned blocks,cudaStream_t stream) {
  auto* args=static_cast<GroupedArguments*>(scratch);
  prepare_grouped<GateUp><<<1,256,0,stream>>>(input,weights,output,offsets,args);
  GroupedBf16 gemm;
  GroupedBf16::Arguments call(args->sizes,GateUp?256:128,blocks,{1.f,0.f},
      args->input,args->weight,args->output,args->output,args->input_stride,args->weight_stride,
      args->output_stride,args->output_stride);
  if(gemm(call,nullptr,stream)!=cutlass::Status::kSuccess)
    throw std::runtime_error("26B grouped BF16 GEMM failed");
}
struct Tile { int expert, first; };
constexpr unsigned tile_rows = 16, tile_columns = 64;
unsigned task_capacity(unsigned rows) { return (8*rows+15)/16 + 127; }
std::size_t metadata_bytes(unsigned rows) {
  return (6*std::size_t(8)*rows+130)*sizeof(int)+task_capacity(rows)*sizeof(Tile);
}
struct Metadata {
  int *keys, *sorted_keys, *assignments, *sorted_assignments, *inverse, *offsets;
  float* weights;
  Tile* tiles;
  int* tile_count;
  Metadata(void* base, unsigned rows) {
    auto* p = static_cast<int*>(base);
    const unsigned n = 8*rows;
    keys = p; sorted_keys = p+n; assignments = p+2*n;
    sorted_assignments = p+3*n; inverse = p+4*n;
    weights = reinterpret_cast<float*>(p+5*n);
    offsets = p+6*n;
    tiles = reinterpret_cast<Tile*>(offsets+129);
    tile_count = reinterpret_cast<int*>(tiles+task_capacity(rows));
  }
};
__device__ float f(B x) { return __bfloat162float(x); }
__device__ B b(float x) { return __float2bfloat16_rn(x); }
__device__ float reciprocal_approximate(float x) {
  float y;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ unsigned scale_offset(unsigned row, unsigned block, unsigned width) {
  return ((row/128)*(width/64)+block/4)*512+(row%32)*16+((row%128)/32)*4+block%4;
}
__device__ B decoded_weight(const Projection& weight, unsigned row, unsigned k, unsigned width) {
  if (weight.storage==StorageType::bf16) return weight.bf16[row*width+k];
  const auto& w=weight.nvfp4;
  const unsigned packed=w.data[(row*width+k)/2];
  const float2 values=__half22float2(__nv_cvt_fp4x2_to_halfraw2(packed,__NV_E2M1));
  const float scale=__fmul_rn(__half2float(__nv_cvt_fp8_to_halfraw(w.scales[scale_offset(row,k/16,width)],__NV_E4M3)),w.weight_scale);
  return b(__fmul_rn(k%2?values.y:values.x,scale));
}
__global__ void select(const B* scores, const B* scales, int* keys,
                       int* assignments, float* weights) {
  unsigned lane = threadIdx.x, row = blockIdx.x;
  float elements[4], maximum = -INFINITY;
  for (int i = 0; i < 4; ++i) {
    elements[i] = f(scores[row*128+lane+i*32]);
    maximum = fmaxf(maximum,elements[i]);
  }
  for (int offset = 16; offset; offset /= 2)
    maximum = fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,offset));
  float total = 0;
  for (int i = 0; i < 4; ++i) { elements[i] = expf(elements[i]-maximum); total += elements[i]; }
  for (int offset = 16; offset; offset /= 2)
    total += __shfl_xor_sync(0xffffffff,total,offset);
  for (int i = 0; i < 4; ++i) elements[i] /= total;
  int ids[8]; float probabilities_selected[8], sums[8];
  unsigned used = 0;
  for (int i = 0; i < 8; ++i) {
    // Compare probabilities in parallel; retain lower-ID ties and the original
    // selected-probability normalization/reduction below.
    int best = 128; float probability = -INFINITY;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (!(used & (1u << j)) && (best == 128 || elements[j] > probability)) {
        best = lane+j*32; probability = elements[j];
      }
    }
    for (int offset = 16; offset; offset /= 2) {
      const float other = __shfl_xor_sync(0xffffffff,probability,offset);
      const int expert = __shfl_xor_sync(0xffffffff,best,offset);
      if (other > probability || (other == probability && expert < best)) {
        best = expert; probability = other;
      }
    }
    ids[i] = best; probabilities_selected[i] = sums[i] = probability;
    if (unsigned(best)%32 == lane) used |= 1u << (unsigned(best)/32);
  }
  if (lane) return;
  for (int stride = 4; stride; stride /= 2)
    for (int i = 0; i < stride; ++i) sums[i] += sums[i+stride];
  for (int i = 0; i < 8; ++i)
    probabilities_selected[i] = probabilities_selected[i]/sums[0]*f(scales[ids[i]]);
  // Sort this token's eight assignments once, for deterministic BF16 reduction.
  for (int i = 1; i < 8; ++i)
    for (int j = i; j > 0 && ids[j] < ids[j-1]; --j) {
      int id = ids[j]; ids[j] = ids[j-1]; ids[j-1] = id;
      float weight = probabilities_selected[j];
      probabilities_selected[j] = probabilities_selected[j-1]; probabilities_selected[j-1] = weight;
    }
  for (int i = 0; i < 8; ++i) {
    unsigned at = row*8+i;
    keys[at] = ids[i]; weights[at] = probabilities_selected[i]; assignments[at] = at;
  }
}
__global__ void group_offsets(const int* keys, int* offsets, unsigned assignments) {
  unsigned expert = threadIdx.x;
  if (expert > 128) return;
  unsigned lo = 0, hi = assignments;
  while (lo < hi) {
    unsigned mid = (lo+hi)/2;
    if (keys[mid] < int(expert)) lo = mid+1; else hi = mid;
  }
  offsets[expert] = lo;
}
__global__ void make_tiles(const int* offsets, Tile* tiles, int* tile_count) {
  using Scan = cub::BlockScan<unsigned,128>;
  __shared__ Scan::TempStorage scratch;
  const unsigned expert = threadIdx.x;
  const unsigned count = (offsets[expert+1]-offsets[expert]+tile_rows-1)/tile_rows;
  unsigned first, total;
  Scan(scratch).ExclusiveSum(count,first,total);
  for (unsigned i = 0; i < count; ++i)
    tiles[first+i] = {int(expert),int(offsets[expert]+i*tile_rows)};
  if (!expert) *tile_count = total;
}
__global__ void invert(const int* assignments, int* inverse, unsigned n) {
  unsigned i = blockIdx.x*256+threadIdx.x;
  if (i < n) inverse[assignments[i]] = i;
}
__global__ void gather(const B* input, B* expanded, const int* assignments, unsigned n) {
  unsigned i = blockIdx.x*256+threadIdx.x;
  if (i < n*kHiddenSize) expanded[i] = input[(assignments[i/kHiddenSize]/8)*kHiddenSize+i%kHiddenSize];
}
// Register layout for m16n8k32: each lane owns four adjacent K values,
// two row halves and two K halves. Four warps cover a 16x64 output tile.
// Quantize directly into the MMA registers with this projection's input scale;
// gate and up never share a quantization scale or a padded logical width.
__device__ unsigned pack_fp8(const B* input, unsigned row, unsigned end,
                             unsigned width, unsigned k, float inverse_scale) {
  unsigned packed = 0;
  if (row < end) {
    #pragma unroll
    for (unsigned i = 0; i < 4; ++i)
      packed |= unsigned(__nv_cvt_float_to_fp8(__fmul_rn(f(input[row*width+k+i]),inverse_scale),
                                             __NV_SATFINITE,__NV_E4M3)) << (8*i);
  }
  return packed;
}
__device__ void mma_fp8(float* d, const unsigned* a, unsigned b0, unsigned b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template<bool GateUp>
__global__ void grouped_fp8(const B* input, const ExpertWeights* weights, B* output,
                            const int* offsets, const Tile* tiles, const int* tile_count) {
  if (blockIdx.y >= *tile_count) return;
  constexpr unsigned K = GateUp ? kHiddenSize : kExpertMlpSize;
  constexpr unsigned N = GateUp ? 2*kExpertMlpSize : kHiddenSize;
  const Tile task = tiles[blockIdx.y];
  const unsigned col = blockIdx.x*tile_columns, warp = threadIdx.x/32, lane = threadIdx.x%32;
  const auto& projection = GateUp ? (col < kExpertMlpSize ? weights[task.expert].gate : weights[task.expert].up)
                                 : weights[task.expert].down;
  if (projection.storage != StorageType::fp8_w8a8) return;
  const unsigned wc = (GateUp && col >= kExpertMlpSize ? col-kExpertMlpSize : col)+warp*16;
  const auto& weight = projection.fp8;
  const float inverse_scale = 1.f/weight.input_scale;
  const unsigned row = task.first+lane/4, end = offsets[task.expert+1];
  float accum[8]{};
  for (unsigned k = 0; k < K; k += 32) {
    const unsigned kk = k+(lane%4)*4;
    unsigned a[4] = {pack_fp8(input,row,end,K,kk,inverse_scale),
                     pack_fp8(input,row+8,end,K,kk,inverse_scale),
                     pack_fp8(input,row,end,K,kk+16,inverse_scale),
                     pack_fp8(input,row+8,end,K,kk+16,inverse_scale)};
    #pragma unroll
    for (unsigned half = 0; half < 2; ++half) {
      const auto* w = weight.data+(wc+half*8+lane/4)*K+kk;
      // Promote each native FP8 dot fragment through an explicit FP32 add;
      // retaining the long running sum inside MMA loses low bits on mixed
      // expert activations near BF16 rounding boundaries.
      float partial[4]{};
      mma_fp8(partial,a,*reinterpret_cast<const unsigned*>(w),*reinterpret_cast<const unsigned*>(w+16));
      #pragma unroll
      for(unsigned i=0;i<4;++i) accum[half*4+i]=__fadd_rn(accum[half*4+i],partial[i]);
    }
  }
  const float alpha = __fmul_rn(weight.input_scale,weight.weight_scale);
  #pragma unroll
  for (unsigned half = 0; half < 2; ++half) {
    #pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
      const unsigned r = row+(i/2)*8, c = col+warp*16+half*8+(lane%4)*2+i%2;
      if (r < end) output[r*N+c] = b(__fmul_rn(accum[half*4+i],alpha));
    }
  }
}
// Two adjacent lanes own one K16 activation block. Quantize with the same
// operation order and approximate reciprocal recipe as nvfp4::Plan.
__device__ unsigned pack_fp4(const B* input, unsigned row, unsigned end,
                             unsigned width, unsigned k, float inverse_scale, unsigned& scale) {
  float values[8], maximum=0.f;
  #pragma unroll
  for (unsigned i=0;i<8;++i) {
    values[i]=row<end?f(input[row*width+k+i]):0.f;
    maximum=fmaxf(maximum,fabsf(values[i]));
  }
  maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,1,2));
  scale=__nv_cvt_float_to_fp8(inverse_scale*(maximum*reciprocal_approximate(6.f)),__NV_SATFINITE,__NV_E4M3);
  const float sf=__half2float(__nv_cvt_fp8_to_halfraw(scale,__NV_E4M3));
  const float factor=sf==0.f?0.f:reciprocal_approximate(sf*reciprocal_approximate(inverse_scale));
  unsigned packed=0;
  #pragma unroll
  for (unsigned i=0;i<4;++i)
    packed|=unsigned(__nv_cvt_float2_to_fp4x2(make_float2(values[2*i]*factor,values[2*i+1]*factor),
                                           __NV_E2M1,cudaRoundNearest))<<(8*i);
  return packed;
}
__device__ void mma_fp4(float* d,const unsigned* a,unsigned b0,unsigned b1,unsigned sa,unsigned sb) {
  asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0,0}, {%11}, {0,0};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
}
// Quantize each sorted assignment once per projection. Gate and up keep
// independent input scales; down reuses this scratch after activation.
template<bool GateUp,bool Swizzled=false>
__global__ void quantize_fp4(const B* input,const ExpertWeights* weights,const int* experts,
                             unsigned n,std::uint8_t* packed,std::uint8_t* scales,
                             const int* offsets=nullptr,unsigned scale_rows=0) {
  constexpr unsigned K=GateUp?kHiddenSize:kExpertMlpSize;
  const unsigned row=blockIdx.y,part=blockIdx.z,k=(blockIdx.x*blockDim.x+threadIdx.x)*8;
  const auto& w=weights[experts[row]];
  const auto& projection=GateUp?(part?w.up:w.gate):w.down;
  if (projection.storage!=StorageType::nvfp4_w4a4) return;
  unsigned scale;
  const auto value=pack_fp4(input,row,k<K?n:0,K,k,1.f/projection.nvfp4.input_scale,scale);
  if (k<K) {
    *reinterpret_cast<unsigned*>(packed+(std::size_t(part)*n+row)*(K/2)+k/2)=value;
    if (k%16==0) {
      if constexpr(Swizzled) {
        // Each expert gets a 128-row aligned scale region. Adding one block
        // per preceding expert leaves room for every partial final block.
        const unsigned expert=experts[row],first=offsets[expert];
        scales[(std::size_t(part)*scale_rows+(first/128+expert)*128)*(K/16)+
            scale_offset(row-first,k/16,K)]=scale;
      } else scales[(std::size_t(part)*n+row)*(K/16)+k/16]=scale;
    }
  }
}
__device__ unsigned packed_word(const std::uint8_t* input,unsigned row,unsigned end,unsigned width,unsigned offset) {
  return row<end?*reinterpret_cast<const unsigned*>(input+std::size_t(row)*width+offset):0;
}
// Small input batches retain the low-overhead 16-row register kernel.
template<bool GateUp>
__global__ void grouped_fp4(const std::uint8_t* input,const std::uint8_t* scales,unsigned n,
                            const ExpertWeights* weights,B* output,const int* offsets,
                            const Tile* tiles,const int* tile_count) {
  if(blockIdx.y>=*tile_count) return;
  constexpr unsigned K=GateUp?kHiddenSize:kExpertMlpSize;
  constexpr unsigned N=GateUp?2*kExpertMlpSize:kHiddenSize;
  const Tile task=tiles[blockIdx.y];
  const unsigned col=blockIdx.x*tile_columns,warp=threadIdx.x/32,lane=threadIdx.x%32;
  const auto& projection=GateUp?(col<kExpertMlpSize?weights[task.expert].gate:weights[task.expert].up):weights[task.expert].down;
  if(projection.storage!=StorageType::nvfp4_w4a4) return;
  const unsigned wc=(GateUp && col>=kExpertMlpSize?col-kExpertMlpSize:col)+warp*16;
  const auto& weight=projection.nvfp4;
  const unsigned row=task.first+lane/4,end=offsets[task.expert+1];
  const unsigned part=GateUp && col>=kExpertMlpSize;
  input+=std::size_t(part)*n*(K/2);
  scales+=std::size_t(part)*n*(K/16);
  float accum[8]{};
  for(unsigned k=0;k<K;k+=64) {
    const unsigned kk=k+(lane%4)*8;
    unsigned a[4]={packed_word(input,row,end,K/2,kk/2),
                   packed_word(input,row+8,end,K/2,kk/2),
                   packed_word(input,row,end,K/2,(kk+32)/2),
                   packed_word(input,row+8,end,K/2,(kk+32)/2)};
    const unsigned sa=packed_word(scales,row+(lane%2)*8,end,K/16,k/16);
    #pragma unroll
    for(unsigned half=0;half<2;++half) {
      const unsigned wr=wc+half*8+lane/4;
      const auto* source=weight.data+(wr*K+kk)/2;
      mma_fp4(accum+half*4,a,*reinterpret_cast<const unsigned*>(source),
          *reinterpret_cast<const unsigned*>(source+16),sa,
          *reinterpret_cast<const unsigned*>(weight.scales+scale_offset(wr,k/16,K)));
    }
  }
  const float alpha=__fmul_rn(weight.input_scale,weight.weight_scale);
  #pragma unroll
  for(unsigned half=0;half<2;++half) {
    #pragma unroll
    for(unsigned i=0;i<4;++i) {
      const unsigned r=row+(i/2)*8,c=col+warp*16+half*8+(lane%4)*2+i%2;
      if(r<end) output[r*N+c]=b(__fmul_rn(accum[half*4+i],alpha));
    }
  }
}
// Four warps compute a 16x64 output tile. Only the small final tile of an
// expert is padded; the task list never computes unselected expert/token pairs.
template<bool GateUp>
__global__ void grouped_linear(const B* input, const ExpertWeights* weights, B* output,
                               const int* offsets, const Tile* tiles, const int* tile_count,
                               bool fp4, bool skip_bf16) {
  if (blockIdx.y >= *tile_count) return;
  constexpr unsigned K = GateUp ? kHiddenSize : kExpertMlpSize;
  constexpr unsigned N = GateUp ? 2*kExpertMlpSize : kHiddenSize;
  Tile task = tiles[blockIdx.y];
  unsigned col = blockIdx.x*tile_columns, warp = threadIdx.x/32;
  const auto& projection = GateUp ? (col < kExpertMlpSize ? weights[task.expert].gate : weights[task.expert].up)
                                 : weights[task.expert].down;
  if (projection.storage == StorageType::fp8_w8a8 ||
      (projection.storage == StorageType::nvfp4_w4a4 && fp4) ||
      (projection.storage == StorageType::bf16 && skip_bf16)) return;
  unsigned weight_col = GateUp && col >= kExpertMlpSize ? col-kExpertMlpSize : col;
  __shared__ __align__(32) B a[tile_rows*16], w[tile_columns*16];
  __shared__ __align__(32) float result[tile_rows*tile_columns];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,B,nvcuda::wmma::row_major> af;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,B,nvcuda::wmma::col_major> wf;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> acc;
  nvcuda::wmma::fill_fragment(acc,0.f);
  for (unsigned k = 0; k < K; k += 16) {
    for (unsigned i = threadIdx.x; i < tile_rows*16; i += 128) {
      unsigned row = task.first+i/16;
      a[i] = row < offsets[task.expert+1] ? input[row*K+k+i%16] : b(0);
    }
    for (unsigned i = threadIdx.x; i < tile_columns*16; i += 128)
      w[i] = decoded_weight(projection,weight_col+i/16,k+i%16,K);
    __syncthreads();
    nvcuda::wmma::load_matrix_sync(af,a,16);
    nvcuda::wmma::load_matrix_sync(wf,w+warp*16*16,16);
    nvcuda::wmma::mma_sync(acc,af,wf,acc);
    __syncthreads();
  }
  nvcuda::wmma::store_matrix_sync(result+warp*16,acc,tile_columns,nvcuda::wmma::mem_row_major);
  __syncthreads();
  for (unsigned i = threadIdx.x; i < tile_rows*tile_columns; i += 128) {
    unsigned row = task.first+i/tile_columns;
    if (row < offsets[task.expert+1]) output[row*N+col+i%tile_columns] = b(result[i]);
  }
}
template<bool GateUp,unsigned RowTokens=1>
__global__ void small_linear(const B* input, const ExpertWeights* weights, B* output,
                             const int* experts,const int* offsets) {
  constexpr unsigned K = GateUp ? 2816 : 704, N = GateUp ? 1408 : 2816;
  unsigned row = blockIdx.y, col = blockIdx.x*32+(threadIdx.x/32)*4, lane = threadIdx.x%32;
  const unsigned expert=experts[row],end=offsets[expert+1];
  // Small groups retain single-row occupancy. Populated groups share a
  // weight load across four rows, with the same lane-strided sums per row.
  if constexpr(RowTokens==1) { if(end-offsets[expert]>=8) return; }
  else { if(end-offsets[expert]<8) return; }
  if((row-offsets[expert])%RowTokens) return;
  const auto& w = weights[expert];
  const auto& projection = GateUp ? (col < 704 ? w.gate : w.up) : w.down;
  if (projection.storage != StorageType::bf16) return;
  const B* matrix = projection.bf16;
  unsigned wc = GateUp && col >= 704 ? col-704 : col;
  float sum[RowTokens][4]{};
  for (unsigned k = lane; k < K; k += 32) {
    float weight[4];
    #pragma unroll
    for (unsigned i=0;i<4;++i) weight[i]=f(matrix[(wc+i)*K+k]);
    #pragma unroll
    for (unsigned r=0;r<RowTokens;++r) if(row+r<end) {
      const float x=f(input[(row+r)*K+k]);
      #pragma unroll
      for (unsigned i=0;i<4;++i) sum[r][i]+=x*weight[i];
    }
  }
  for (unsigned delta = 16; delta; delta /= 2) {
    #pragma unroll
    for (unsigned r=0;r<RowTokens;++r) if(row+r<end) {
      #pragma unroll
      for (unsigned i=0;i<4;++i) sum[r][i]+=__shfl_down_sync(0xffffffff,sum[r][i],delta);
    }
  }
  if (!lane) {
    #pragma unroll
    for (unsigned r=0;r<RowTokens;++r) if(row+r<end) {
      #pragma unroll
      for (unsigned i=0;i<4;++i) output[(row+r)*N+col+i]=b(sum[r][i]);
    }
  }
}
__global__ void activate(const B* gate_up, B* product, unsigned n) {
  unsigned i = blockIdx.x*256+threadIdx.x;
  if (i < n*kExpertMlpSize) {
    unsigned row = i/kExpertMlpSize, d = i%kExpertMlpSize;
    product[i] = b(gewell::detail::gelu_tanh_multiply_bf16(f(gate_up[row*1408+d]),f(gate_up[row*1408+704+d])));
  }
}
__global__ void reduce(const B* down, const int* inverse, const float* weights, B* output, unsigned rows) {
  unsigned i = blockIdx.x*256+threadIdx.x;
  if (i >= rows*kHiddenSize) return;
  unsigned row = i/kHiddenSize, d = i%kHiddenSize;
  B sum = b(0);
  for (unsigned j = 0; j < 8; ++j) {
    unsigned assignment = row*8+j;
    sum = b(f(sum)+f(b(f(down[inverse[assignment]*kHiddenSize+d])*weights[assignment])));
  }
  output[i] = sum;
}
unsigned checked_rows(unsigned rows) {
  if (!rows || rows > 4096) throw std::invalid_argument("MoE capacity must be in 1..4096");
  return rows;
}
}
MoeWorkspace::MoeWorkspace(unsigned rows)
    : max_rows_(checked_rows(rows)), metadata_(metadata_bytes(rows)),
      expanded_(std::size_t(8)*rows*kHiddenSize*sizeof(B)),
      gate_up_(std::size_t(8)*rows*2*kExpertMlpSize*sizeof(B)),
      activated_(std::size_t(8)*rows*kExpertMlpSize*sizeof(B)),
      quantized_(std::size_t(8)*rows*kHiddenSize+2*(rows>=16?((8*rows+127)/128+128)*128:8*rows)*(kHiddenSize/16)),
      grouped_(sizeof(GroupedArguments)),grouped_blocks_(GroupedBf16::sufficient()) {
  if (!grouped_blocks_) throw std::runtime_error("26B grouped BF16 GEMM has no available blocks");
  Metadata m(metadata_.data(),max_rows_);
  check_cuda(cub::DeviceRadixSort::SortPairs(nullptr,sort_bytes_,m.keys,m.sorted_keys,
      m.assignments,m.sorted_assignments,8*rows,0,7),"size MoE sort");
  sort_scratch_ = std::make_unique<cuda_detail::DeviceAllocation>(sort_bytes_);
  if (rows>=16) grouped_fp4_ = std::make_unique<GroupedNvfp4>();
}
MoeWorkspace::~MoeWorkspace() = default;
void MoeWorkspace::dispatch(const B* scores, const B* scales, unsigned rows, cudaStream_t stream) {
  if (rows > max_rows_) throw std::invalid_argument("MoE rows exceed workspace");
  if (!rows) return;
  Metadata m(metadata_.data(),max_rows_);
  select<<<rows,32,0,stream>>>(scores,scales,m.keys,m.assignments,m.weights);
  check_cuda(cub::DeviceRadixSort::SortPairs(sort_scratch_->data(),sort_bytes_,m.keys,m.sorted_keys,
      m.assignments,m.sorted_assignments,8*rows,0,7,stream),"sort MoE assignments");
  group_offsets<<<1,256,0,stream>>>(m.sorted_keys,m.offsets,8*rows);
  make_tiles<<<1,128,0,stream>>>(m.offsets,m.tiles,m.tile_count);
  invert<<<(8*rows+255)/256,256,0,stream>>>(m.sorted_assignments,m.inverse,8*rows);
  check_cuda(cudaGetLastError(),"MoE dispatch");
}
void MoeWorkspace::run(const B* input, const B* scores, const B* scales,
                       const ExpertWeights* weights, B* output, unsigned rows, bool has_bf16, bool has_fp8, bool has_nvfp4, bool fp4, cudaStream_t stream) {
  dispatch(scores,scales,rows,stream);
  if (!rows) return;
  Metadata m(metadata_.data(),max_rows_);
  auto* expanded = static_cast<B*>(expanded_.data());
  auto* gate_up = static_cast<B*>(gate_up_.data());
  auto* activated = static_cast<B*>(activated_.data());
  auto* packed=static_cast<std::uint8_t*>(quantized_.data());
  unsigned n = 8*rows, tasks = task_capacity(rows);
  const unsigned scale_rows=((n+127)/128+128)*128;
  const bool staged_fp4=rows>=16;
  auto* gate_scales=packed+std::size_t(n)*kHiddenSize;
  auto* down_scales=packed+std::size_t(n)*kExpertMlpSize/2;
  gather<<<(n*kHiddenSize+255)/256,256,0,stream>>>(input,expanded,m.sorted_assignments,n);
  if (has_nvfp4 && fp4) {
    if (staged_fp4) quantize_fp4<true,true><<<dim3((kHiddenSize/8+127)/128,n,2),128,0,stream>>>(expanded,weights,m.sorted_keys,n,packed,gate_scales,m.offsets,scale_rows);
    else quantize_fp4<true><<<dim3((kHiddenSize/8+127)/128,n,2),128,0,stream>>>(expanded,weights,m.sorted_keys,n,packed,gate_scales);
  }
  if (has_bf16 && rows <= 32) small_linear<true><<<dim3(1408/32,n),256,0,stream>>>(expanded,weights,gate_up,m.sorted_keys,m.offsets);
  if (has_bf16 && rows >= 8 && rows <= 32) small_linear<true,4><<<dim3(1408/32,n),256,0,stream>>>(expanded,weights,gate_up,m.sorted_keys,m.offsets);
  if (has_bf16 && rows > 32) run_grouped<true>(expanded,weights,gate_up,m.offsets,grouped_.data(),grouped_blocks_,stream);
  if (has_nvfp4 && !fp4) grouped_linear<true><<<dim3(1408/tile_columns,tasks),128,0,stream>>>(expanded,weights,gate_up,m.offsets,m.tiles,m.tile_count,false,true);
  if (has_fp8) grouped_fp8<true><<<dim3(1408/tile_columns,tasks),128,0,stream>>>(expanded,weights,gate_up,m.offsets,m.tiles,m.tile_count);
  if (has_nvfp4 && fp4) {
    if (staged_fp4) grouped_fp4_->gate_up.run(packed,gate_scales,n,scale_rows,weights,gate_up,m.offsets,stream);
    else grouped_fp4<true><<<dim3(1408/tile_columns,tasks),128,0,stream>>>(packed,gate_scales,n,weights,gate_up,m.offsets,m.tiles,m.tile_count);
  }
  activate<<<(n*704+255)/256,256,0,stream>>>(gate_up,activated,n);
  // Input gather is dead after gate/up; reuse it for the wider down projection.
  if (has_nvfp4 && fp4) {
    if (staged_fp4) quantize_fp4<false,true><<<dim3((kExpertMlpSize/8+127)/128,n),128,0,stream>>>(activated,weights,m.sorted_keys,n,packed,down_scales,m.offsets,scale_rows);
    else quantize_fp4<false><<<dim3((kExpertMlpSize/8+127)/128,n),128,0,stream>>>(activated,weights,m.sorted_keys,n,packed,down_scales);
  }
  if (has_bf16 && rows <= 32) small_linear<false><<<dim3(2816/32,n),256,0,stream>>>(activated,weights,expanded,m.sorted_keys,m.offsets);
  if (has_bf16 && rows >= 8 && rows <= 32) small_linear<false,4><<<dim3(2816/32,n),256,0,stream>>>(activated,weights,expanded,m.sorted_keys,m.offsets);
  if (has_bf16 && rows > 32) run_grouped<false>(activated,weights,expanded,m.offsets,grouped_.data(),grouped_blocks_,stream);
  if (has_nvfp4 && !fp4) grouped_linear<false><<<dim3(2816/tile_columns,tasks),128,0,stream>>>(activated,weights,expanded,m.offsets,m.tiles,m.tile_count,false,true);
  if (has_fp8) grouped_fp8<false><<<dim3(2816/tile_columns,tasks),128,0,stream>>>(activated,weights,expanded,m.offsets,m.tiles,m.tile_count);
  if (has_nvfp4 && fp4) {
    if (staged_fp4) grouped_fp4_->down.run(packed,down_scales,n,scale_rows,weights,expanded,m.offsets,stream);
    else grouped_fp4<false><<<dim3(2816/tile_columns,tasks),128,0,stream>>>(packed,down_scales,n,weights,expanded,m.offsets,m.tiles,m.tile_count);
  }
  reduce<<<(rows*2816+255)/256,256,0,stream>>>(expanded,m.inverse,m.weights,output,rows);
  check_cuda(cudaGetLastError(),"grouped experts");
}
std::size_t MoeWorkspace::bytes() const {
  return routing_bytes()+projection_bytes()+reduction_bytes();
}
std::size_t MoeWorkspace::routing_bytes() const {
  return metadata_.size()+grouped_.size()+sort_scratch_->size()+(grouped_fp4_?grouped_fp4_->bytes():0);
}
const int* MoeWorkspace::selected_experts() const { return Metadata(metadata_.data(),max_rows_).keys; }
const float* MoeWorkspace::routing_weights() const { return Metadata(metadata_.data(),max_rows_).weights; }
const int* MoeWorkspace::offsets() const { return Metadata(metadata_.data(),max_rows_).offsets; }
const int* MoeWorkspace::sorted_assignments() const { return Metadata(metadata_.data(),max_rows_).sorted_assignments; }
}
