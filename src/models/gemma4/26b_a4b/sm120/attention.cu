#include "attention.cuh"
#include "kv_storage.cuh"
#include "gewell/compact_global_cache.h"
#include "gewell/mtp_attention.h"
#include <algorithm>
#include <stdexcept>

namespace gewell::gemma4_26b_a4b::sm120 {
namespace {
using B = __nv_bfloat16;
constexpr unsigned Tile = 1024, Queries = 256, Heads = 16;
constexpr std::size_t Staged = std::size_t(8)*Tile*256;
constexpr std::size_t QElements = std::size_t(Heads)*Queries*512;
constexpr std::size_t Scores = std::size_t(Heads)*Queries*Tile;
constexpr std::size_t States = Heads*Queries;
constexpr std::size_t Bytes = (2*Staged+QElements+Scores)*2+(Scores+QElements+2*States)*4;
void blas(cublasStatus_t s) {
  if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("26B attention cuBLAS error "+std::to_string(s));
}

template <unsigned D>
__global__ void gather_query(const B* input, B* out, unsigned begin, unsigned count) {
  unsigned row=blockIdx.x, head=blockIdx.y, d=threadIdx.x;
  out[(std::size_t(head)*count+row)*D+d]=input[(std::size_t(begin+row)*Heads+head)*D+d];
}

template <bool Global>
__global__ void gather_kv(const B* key, const B* value, const B* norm,
                          kv_cache::DeviceView cache, unsigned base,
                          unsigned begin, unsigned count, B* out_k, B* out_v) {
  constexpr unsigned D=Global?512:256, H=Global?2:8;
  unsigned d=threadIdx.x, t=blockIdx.x, head=blockIdx.y;
  unsigned absolute=begin+t;
  bool current=absolute>=base;
  std::size_t source=(std::size_t(current?absolute-base:0)*H+head)*D+d;
  B k{},v{};
  // All staged lanes are initialized, including padded tail positions.
  if (t<count) {
    if constexpr (Global) {
      const B* record=current?nullptr:compact_global_cache::paged_row(cache.page_pool,
          cache.page_offsets,cache.page_tokens,cache.layer_offset_elements,head,absolute,cache.format);
      v=current?value[source]:kv_storage::load(record,128+d,640,cache.format,128);
      bool rotated=d<64 || (d>=256 && d<320);
      k=rotated?(current?key[source]:kv_storage::load(record,d<64?d:d-192,640,cache.format,128))
          :__float2bfloat16_rn(__bfloat162float(v)*__bfloat162float(norm[d]));
    } else {
      auto index=std::size_t(head)*1024+absolute%1024;
      k=current?key[source]:kv_storage::load(kv_storage::row(cache.key,index,D,cache.format),d,D,cache.format);
      v=current?value[source]:kv_storage::load(kv_storage::row(cache.value,index,D,cache.format),d,D,cache.format);
    }
  }
  auto destination=(std::size_t(head)*Tile+t)*D+d;
  out_k[destination]=k; out_v[destination]=v;
}

template <unsigned D, bool Global>
__global__ void update(const float* scores, B* probabilities, float* numerator,
                       float* maxima, float* denominators, unsigned base,
                       unsigned rows, unsigned begin, unsigned count, bool first, bool image) {
  unsigned lane=threadIdx.x%32, row=blockIdx.y*8+threadIdx.x/32, head=blockIdx.x;
  if (row>=rows) return;
  auto state=std::size_t(head)*rows+row, offset=state*Tile;
  unsigned absolute=base+row;
  float values[32], maximum=-INFINITY;
  for (unsigned i=0;i<32;++i) {
    unsigned t=lane+i*32, k=begin+t;
    bool visible=t<count && (k<=absolute || image) &&
        (Global || k>=absolute || absolute-k<1024);
    values[i]=visible?scores[offset+t]:-INFINITY;
    maximum=fmaxf(maximum,values[i]);
  }
  for (unsigned d=16;d;d/=2) maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,d));
  maximum=__shfl_sync(0xffffffff,maximum,0);
  float previous=first?-INFINITY:maxima[state];
  float next=fmaxf(previous,maximum);
  float scale=maximum==-INFINITY?(first?0.f:1.f):(previous==-INFINITY?0.f:expf(previous-next));
  if (!first && scale!=1.f)
    for (unsigned d=lane;d<D;d+=32) numerator[state*512+d]*=scale;
  float sums[8]={};
  for (unsigned i=0;i<32;++i) {
    B p=__float2bfloat16_rn(values[i]==-INFINITY?0.f:expf(values[i]-next));
    probabilities[offset+lane+i*32]=p;
    sums[i%8]+=__bfloat162float(p);
  }
  for (unsigned d=4;d;d/=2) for (unsigned i=0;i<d;++i) sums[i]+=sums[i+d];
  float sum=sums[0];
  for (unsigned d=16;d;d/=2) sum+=__shfl_down_sync(0xffffffff,sum,d);
  if (!lane) {
    maxima[state]=next;
    denominators[state]=__fmul_rn(first?0.f:denominators[state],scale)+sum;
  }
}

template <unsigned D>
__global__ void finish(const float* numerator,const float* denominator,B* output,unsigned rows) {
  unsigned row=blockIdx.x,head=blockIdx.y,d=threadIdx.x;
  auto state=std::size_t(head)*rows+row;
  output[(std::size_t(row)*Heads+head)*D+d]=__float2bfloat16_rn(numerator[state*512+d]/denominator[state]);
}

template <bool Global>
void forward(cublasHandle_t handle,const B* query,const B* key,const B* value,const B* norm,
             kv_cache::DeviceView cache,unsigned base,unsigned rows,B* output,void* scratch,cudaStream_t stream,bool image) {
  constexpr unsigned D=Global?512:256,H=Global?2:8,Repeat=Heads/H;
  B* staged_k=static_cast<B*>(scratch),*staged_v=staged_k+Staged,*q=staged_v+Staged;
  B* probabilities=q+QElements;
  float* scores=reinterpret_cast<float*>(probabilities+Scores);
  float* numerator=scores+Scores,*maximum=numerator+QElements,*denominator=maximum+States;
  blas(cublasSetStream(handle,stream));
  for (unsigned qb=0;qb<rows;qb+=Queries) {
    unsigned count=std::min(Queries,rows-qb), absolute=base+qb;
    gather_query<D><<<dim3(count,Heads),D,0,stream>>>(query,q,qb,count);
    unsigned first_key=Global?0:(absolute>1023?absolute-1023:0);
    const unsigned end = image && !Global ? base + rows : absolute + count;
    for (unsigned begin=first_key;begin<end;begin+=Tile) {
      unsigned keys=std::min(Tile,end-begin), padded=(keys+7)/8*8;
      gather_kv<Global><<<dim3(padded,H),D,0,stream>>>(key,value,norm,cache,base,begin,keys,staged_k,staged_v);
      float alpha=1.f,zero=0.f,beta=begin==first_key?0.f:1.f;
      long long matrices=std::size_t(Repeat)*count*Tile;
      blas(cublasGemmStridedBatchedEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,padded,Repeat*count,D,
          &alpha,staged_k,CUDA_R_16BF,D,Tile*D,q,CUDA_R_16BF,D,Repeat*count*D,
          &zero,scores,CUDA_R_32F,Tile,matrices,H,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      update<D,Global><<<dim3(Heads,(count+7)/8),256,0,stream>>>(scores,probabilities,numerator,
          maximum,denominator,absolute,count,begin,keys,begin==first_key,image && !Global);
      blas(cublasGemmStridedBatchedEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,D,Repeat*count,padded,
          &alpha,staged_v,CUDA_R_16BF,D,Tile*D,probabilities,CUDA_R_16BF,Tile,matrices,
          &beta,numerator,CUDA_R_32F,512,Repeat*count*512,H,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    finish<D><<<dim3(count,Heads),D,0,stream>>>(numerator,denominator,output+std::size_t(qb)*Heads*D,count);
  }
}
// Only current Q/K need a layout change: shared FP8 attention consumes
// head-major Q/K, token-major V, and returns token-major output.
__global__ void head_major(const B* input, B* output, unsigned rows, unsigned heads, unsigned width) {
  const auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= std::size_t(rows) * heads * width) return;
  const auto d = i % width, head = i / width % heads, row = i / (width * heads);
  output[(head * rows + row) * width + d] = input[i];
}
void validate(cublasHandle_t handle,const AttentionInput& input,const B* norm,bool global,unsigned max_rows) {
  const auto& cache=input.cache;
  if (!handle || !input.query || !input.key || !input.value || !input.output || input.rows>max_rows ||
      (input.image && input.rows>1120) ||
      std::uint64_t(input.position)+input.rows>262144 || (global && !norm) ||
      (cache.format!=kv_cache::Format::bf16 && cache.format!=kv_cache::Format::fp8))
    throw std::invalid_argument("26B attention: invalid inputs");
  if (input.position && (global ? (!cache.page_pool || !cache.page_offsets || cache.page_tokens!=256 ||
      (std::uint64_t(input.position)+255)/256>cache.page_count) : (!cache.key || !cache.value || cache.capacity!=1024)))
    throw std::invalid_argument("26B attention: missing committed history");
}
}  // namespace

AttentionWorkspace::AttentionWorkspace(unsigned max_rows, attention::Compute local, attention::Compute global)
    : max_rows_(max_rows), local_(local), global_(global) {
  const auto valid = [](auto value) { return value == attention::Compute::bf16 || value == attention::Compute::fp8; };
  if (!max_rows || max_rows > 4096 || !valid(local) || !valid(global))
    throw std::invalid_argument("26B attention: invalid workspace geometry or compute");
  decode_inputs_.reserve(32);
  if (local == attention::Compute::bf16 || global == attention::Compute::bf16)
    storage_ = std::make_unique<cuda_detail::DeviceAllocation>(Bytes);
  if (local == attention::Compute::fp8 || global == attention::Compute::fp8) {
    fp8_ = std::make_unique<cuda_detail::DeviceAllocation>(mtp_attention::fp8_scratch_bytes(16, max_rows, 262144));
    packed_ = std::make_unique<cuda_detail::DeviceAllocation>(std::size_t(max_rows) * (16 * 512 + 8 * 256) * sizeof(B));
  }
}
void AttentionWorkspace::run(cublasHandle_t handle,const B* query,const B* key,const B* value,
    const B* norm,kv_cache::DeviceView cache,bool global,unsigned position,unsigned rows,
    B* output,cudaStream_t stream,bool image) {
  if (!rows) return;
  validate(handle,{query,key,value,cache,position,rows,output,image},norm,global,max_rows_);
  if ((global ? global_ : local_) == attention::Compute::fp8) {
    const unsigned width = global ? 512 : 256, kv_heads = global ? 2 : 8;
    auto* q = static_cast<B*>(packed_->data());
    auto* k = q + std::size_t(max_rows_) * 16 * 512;
    head_major<<<(std::size_t(rows) * 16 * width + 255) / 256, 256, 0, stream>>>(query, q, rows, 16, width);
    head_major<<<(std::size_t(rows) * kv_heads * width + 255) / 256, 256, 0, stream>>>(key, k, rows, kv_heads, width);
    cuda_detail::check_cuda(cudaGetLastError(), "26B FP8 attention packing");
    mtp_attention::run_fp8_batch(16, {{q, k, value, cache, position, rows, output, image && !global}}, norm,
        global ? gemma4_31b::AttentionKind::global : gemma4_31b::AttentionKind::local,
        fp8_->data(), fp8_->size(), stream);
    return;
  }
  if (global) forward<true>(handle,query,key,value,norm,cache,position,rows,output,storage_->data(),stream,image);
  else forward<false>(handle,query,key,value,norm,cache,position,rows,output,storage_->data(),stream,image);
  cuda_detail::check_cuda(cudaGetLastError(),"26B attention launch");
}
void AttentionWorkspace::run_batch(cublasHandle_t handle,const std::vector<AttentionInput>& inputs,
    const B* norm,bool global,cudaStream_t stream) {
  for (const auto& input:inputs)
    if (input.rows) validate(handle,input,norm,global,max_rows_);
  auto& batch=decode_inputs_;
  batch.clear();
  const auto flush=[&] {
    if (batch.empty()) return;
    mtp_attention::run_decode_batch(16,batch,norm,
        global?gemma4_31b::AttentionKind::global:gemma4_31b::AttentionKind::local,
        storage_->data(),storage_->size(),stream);
    batch.clear();
  };
  for (const auto& input:inputs) {
    if (!input.rows) continue;
    if (input.rows==1 && !input.image && (global?global_:local_)==attention::Compute::bf16) {
      batch.push_back({input.query,input.key,input.value,input.cache,input.position,1,input.output});
      if (batch.size()==32) flush();
    } else {
      flush();
      run(handle,input.query,input.key,input.value,norm,input.cache,global,input.position,
          input.rows,input.output,stream,input.image);
    }
  }
  flush();
}
}  // namespace gewell::gemma4_26b_a4b::sm120
