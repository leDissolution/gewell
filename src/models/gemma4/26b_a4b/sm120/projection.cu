#include "projection.cuh"
#include "cuda_memory.cuh"
#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>
#include <tuple>

namespace gewell::gemma4_26b_a4b::sm120 {
namespace {
using B = __nv_bfloat16;
using Device = cuda_detail::DeviceAllocation;
using cuda_detail::check_cuda;
void blas(cublasStatus_t result) {
  if (result != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("26B projection cuBLAS error " + std::to_string(result));
}
struct Lt {
  cublasLtHandle_t value{};
  Lt() { blas(cublasLtCreate(&value)); }
  ~Lt() { cublasLtDestroy(value); }
};
using Shape = std::tuple<StorageType,unsigned,unsigned>;
Shape shape(const Projection& weight) { return {weight.storage,weight.input,weight.physical_output}; }
__global__ void compact(const B* input,B* output,unsigned rows,unsigned logical,unsigned physical) {
  const auto i=std::size_t(blockIdx.x)*256+threadIdx.x;
  if (i<std::size_t(rows)*logical) output[i]=input[(i/logical)*physical+i%logical];
}
}

Projection bind_projection(const TensorSpec& spec,const TensorEntry& entry,
                           const std::uint8_t* host,const void* device) {
  if (spec.rank!=2 || !host || !device) throw std::invalid_argument("invalid26B projection binding");
  Projection result;
  result.storage=entry.storage_type;
  result.input=spec.columns;result.output=spec.rows;
  result.physical_output=result.storage==StorageType::nvfp4_w4a4?align_up(spec.rows,128):spec.rows;
  result.bf16=static_cast<const B*>(device);
  if (result.storage==StorageType::bf16) return result;
  float weight_scale,input_scale;
  std::memcpy(&weight_scale,host+entry.bytes-8,4);
  std::memcpy(&input_scale,host+entry.bytes-4,4);
  const auto* packed=static_cast<const std::uint8_t*>(device);
  if (result.storage==StorageType::nvfp4_w4a4)
    result.nvfp4={packed,packed+std::size_t(result.physical_output)*result.input/2,input_scale,weight_scale};
  else if (result.storage==StorageType::fp8_w8a8)
    result.fp8={packed,input_scale,weight_scale,nullptr};
  else throw std::invalid_argument("invalid26B projection storage");
  return result;
}

struct ProjectionWorkspace::Impl {
  struct Plans {
    std::unique_ptr<nvfp4::Plan> nvfp4;
    std::unique_ptr<fp8::Plan> fp8;
  };
  Lt lt;  // Outlives every plan.
  std::map<Shape,Plans> plans;
  std::unique_ptr<Device> scratch,padded,dequantized;
  const unsigned capacity;
  const nvfp4::ActivationPolicy policy;
  const cudaStream_t stream;
  unsigned rows{};
  bool fp4{};
  Impl(const std::vector<Projection>& weights,unsigned max_rows,nvfp4::ActivationPolicy p,cudaStream_t s)
      :capacity(max_rows),policy(p),stream(s) {
    if (!capacity || capacity>4096 || (policy!=nvfp4::ActivationPolicy::always && policy!=nvfp4::ActivationPolicy::prefill))
      throw std::invalid_argument("invalid26B projection capacity/policy");
    std::size_t scratch_bytes=0,padded_bytes=0,dequantized_bytes=0;
    for (const auto& w:weights) {
      if (w.storage==StorageType::bf16) continue;
      plans.try_emplace(shape(w));
      if (w.storage==StorageType::nvfp4_w4a4) {
        scratch_bytes=std::max(scratch_bytes,nvfp4::scratch_upper_bound_bytes(capacity,w.input,w.physical_output));
        if (w.output!=w.physical_output)
          padded_bytes=std::max(padded_bytes,std::size_t(capacity)*w.physical_output*sizeof(B));
        if (policy==nvfp4::ActivationPolicy::prefill)
          dequantized_bytes=std::max(dequantized_bytes,std::size_t(w.physical_output)*w.input*sizeof(B));
      } else if (w.storage==StorageType::fp8_w8a8)
        scratch_bytes=std::max(scratch_bytes,fp8::scratch_upper_bound_bytes(capacity,w.input));
      else throw std::invalid_argument("invalid26B projection storage");
    }
    if (scratch_bytes) scratch=std::make_unique<Device>(scratch_bytes);
    if (padded_bytes) padded=std::make_unique<Device>(padded_bytes);
    if (dequantized_bytes) dequantized=std::make_unique<Device>(dequantized_bytes);
  }
};
ProjectionWorkspace::ProjectionWorkspace(const std::vector<Projection>& weights,unsigned max_rows,
    nvfp4::ActivationPolicy policy,cudaStream_t stream):impl_(std::make_unique<Impl>(weights,max_rows,policy,stream)) {}
ProjectionWorkspace::~ProjectionWorkspace()=default;
std::size_t ProjectionWorkspace::bytes() const {
  const auto& s=*impl_;
  return (s.scratch?s.scratch->size():0)+(s.padded?s.padded->size():0)+(s.dequantized?s.dequantized->size():0);
}
void ProjectionWorkspace::prepare(unsigned rows,nvfp4::Phase phase) {
  auto& s=*impl_;
  if (!rows || rows>s.capacity || (phase!=nvfp4::Phase::prefill && phase!=nvfp4::Phase::decode))
    throw std::invalid_argument("invalid26B projection rows/phase");
  const bool fp4=nvfp4::fp4_activations(s.policy,phase);
  if (s.rows==rows && s.fp4==fp4) return;
  // Retain a plan when only phase changes; rebuild on a new row count outside
  // the layer traversal. The number of plans is bounded by distinct shapes.
  std::map<Shape,Impl::Plans> replacements;
  for (const auto& [key,plans]:s.plans) {
    const auto [storage,k,n]=key;
    auto& next=replacements[key];
    if (storage==StorageType::fp8_w8a8 && (s.rows!=rows || !plans.fp8))
      next.fp8=std::make_unique<fp8::Plan>(s.lt.value,rows,k,n);
    if (storage==StorageType::nvfp4_w4a4 && fp4 && (s.rows!=rows || !plans.nvfp4))
      next.nvfp4=std::make_unique<nvfp4::Plan>(s.lt.value,rows,k,n);
  }
  // Publish only after every requested plan was constructed successfully.
  for (auto& [key,next]:replacements) {
    auto& plans=s.plans.at(key);
    if (next.fp8) plans.fp8=std::move(next.fp8);
    if (next.nvfp4) plans.nvfp4=std::move(next.nvfp4);
    else if (s.rows!=rows) plans.nvfp4.reset();
  }
  s.rows=rows;s.fp4=fp4;
}
void ProjectionWorkspace::run(cublasHandle_t handle,const Projection& w,const B* input,B* output) {
  auto& s=*impl_;
  if (!s.rows || !input || !output) throw std::invalid_argument("prepare26B projections before run");
  B* destination=(w.output==w.physical_output)?output:static_cast<B*>(s.padded->data());
  if (w.storage==StorageType::fp8_w8a8) {
    s.plans.at(shape(w)).fp8->run(input,w.fp8,destination,s.scratch->data(),s.scratch->size(),s.stream);
  } else if (w.storage==StorageType::nvfp4_w4a4 && s.fp4) {
    s.plans.at(shape(w)).nvfp4->run(input,w.nvfp4,destination,s.scratch->data(),s.scratch->size(),s.stream);
  } else {
    const B* weights=w.bf16;
    if (w.storage==StorageType::nvfp4_w4a4) {
      weights=static_cast<const B*>(s.dequantized->data());
      nvfp4::dequantize(w.nvfp4,w.input,w.physical_output,static_cast<B*>(s.dequantized->data()),s.stream);
    }
    float alpha=1.f,beta=0.f;
    blas(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,w.physical_output,s.rows,w.input,&alpha,
        weights,CUDA_R_16BF,w.input,input,CUDA_R_16BF,w.input,&beta,destination,CUDA_R_16BF,
        w.physical_output,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
  }
  if (destination!=output) {
    const auto count=std::size_t(s.rows)*w.output;
    compact<<<(count+255)/256,256,0,s.stream>>>(destination,output,s.rows,w.output,w.physical_output);
    check_cuda(cudaGetLastError(),"compact26B projection output");
  }
}
}
