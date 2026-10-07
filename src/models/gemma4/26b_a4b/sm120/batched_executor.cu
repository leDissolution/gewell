#include "batched_executor.cuh"
#include "gewell/models/gemma4/26b_a4b/artifact.h"
#include "bf16_math.cuh"
#include "moe.cuh"
#include "projection.cuh"
#include "attention.cuh"
#include "kv_write.cuh"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace gewell::gemma4_26b_a4b::sm120 {
namespace {
using B=__nv_bfloat16;
using cuda_detail::DeviceAllocation;
using cuda_detail::check_cuda;
using namespace bf16_math;
constexpr unsigned ScratchWidth=2816*4+8192*2+2112*3+2048*3;
void blas(cublasStatus_t code) {
  if (code!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("26B grouped cuBLAS error "+std::to_string(code));
}
__global__ void embeddings(const B* weights,const std::uint32_t* tokens,B* h,unsigned n) {
  unsigned i=blockIdx.x*256+threadIdx.x;
  if (i<n) h[i]=rounded(value(weights[std::size_t(tokens[i/kHiddenSize])*kHiddenSize+i%kHiddenSize])*kEmbeddingScale);
}
__global__ void set_positions(unsigned* out,unsigned position,unsigned rows) {
  unsigned i=blockIdx.x*256+threadIdx.x;
  if (i<rows) out[i]=position+i;
}
__global__ void row_scale(B* h,const B* scalar,unsigned n) {
  unsigned i=blockIdx.x*256+threadIdx.x;
  if (i<n) h[i]=rounded(value(h[i])*value(*scalar));
}
__global__ void capped(B* x,std::size_t n) {
  auto i=std::size_t(blockIdx.x)*256+threadIdx.x;
  if (i<n) x[i]=rounded(value(rounded(tanhf(value(rounded(value(x[i])/30.f)))))*30.f);
}
}  // namespace

struct BatchedExecutor::Impl {
  ArtifactFile artifact;
  DeviceAllocation weights, scratch, positions, expert_tables;
  std::array<std::array<const B*,25>,30> layer{};
  std::array<std::array<Projection,25>,30> projections{};
  std::unique_ptr<ProjectionWorkspace> quantized;
  std::array<bool,30> expert_bf16{},expert_fp8{},expert_nvfp4{};
  const nvfp4::ActivationPolicy activation_policy;
  MoeWorkspace moe;
  AttentionWorkspace attention;
  cublasHandle_t handle{};
  unsigned capacity;
  std::vector<AttentionInput> attention_inputs;
  std::vector<CacheWriteInput> cache_writes;
  const cudaStream_t stream;
  Impl(ArtifactFile file,unsigned rows,cudaStream_t execution_stream,nvfp4::ActivationPolicy policy,
       attention::Compute local, attention::Compute global):artifact(std::move(file)),
      weights(artifact.file_bytes()-kArtifactDataOffset),scratch(std::size_t(rows)*ScratchWidth*2),
      positions(rows*4),expert_tables(30*128*sizeof(ExpertWeights)),activation_policy(policy),moe(rows),attention(rows,local,global),capacity(rows),stream(execution_stream) {
    attention_inputs.reserve(rows);
    cache_writes.reserve(rows);
    check_cuda(cudaMemcpy(weights.data(),artifact.tensor_data(0),weights.size(),cudaMemcpyHostToDevice),"load 26B weights");
    std::array<std::array<ExpertWeights,128>,30> experts{};
    std::vector<Projection> bindings;
    bool has_quantized=false;
    for (std::size_t i=1;i+1<kTextTensors.size();++i) {
      const auto& spec=kTextTensors[i];
      auto* p=reinterpret_cast<const B*>(static_cast<const char*>(weights.data())+artifact.entries()[i].offset-kArtifactDataOffset);
      if (spec.expert<0) {
        layer[spec.layer][unsigned(spec.role)]=p;
        if (spec.rank==2) {
          auto& projection=projections[spec.layer][unsigned(spec.role)];
          projection=bind_projection(spec,artifact.entries()[i],artifact.tensor_data(i),p);
          bindings.push_back(projection);
          has_quantized|=projection.storage!=StorageType::bf16;
        }
      }
      else {
        auto& e=experts[spec.layer][spec.expert];
        auto binding=bind_projection(spec,artifact.entries()[i],artifact.tensor_data(i),p);
        if (spec.role==TensorRole::expert_gate_proj) e.gate=binding;
        else if (spec.role==TensorRole::expert_up_proj) e.up=binding;
        else e.down=binding;
        expert_bf16[spec.layer]|=binding.storage==StorageType::bf16;
        expert_fp8[spec.layer]|=binding.storage==StorageType::fp8_w8a8;
        expert_nvfp4[spec.layer]|=binding.storage==StorageType::nvfp4_w4a4;
      }
    }
    check_cuda(cudaMemcpy(expert_tables.data(),experts.data(),sizeof(experts),cudaMemcpyHostToDevice),"load expert table");
    blas(cublasCreate(&handle));
    try {
      blas(cublasSetMathMode(handle,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
      blas(cublasSetStream(handle,stream));
      blas(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST));
      if (has_quantized) quantized=std::make_unique<ProjectionWorkspace>(bindings,rows,policy,stream);
    }
    catch (...) { cublasDestroy(handle); throw; }
  }
  ~Impl() { if (handle) cublasDestroy(handle); }
  void linear(const B* input,const B* weight,B* out,unsigned m,unsigned k,unsigned n) {
    float alpha=1.f,beta=0.f;
    blas(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,n,m,k,&alpha,weight,CUDA_R_16BF,k,
        input,CUDA_R_16BF,k,&beta,out,CUDA_R_16BF,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
  }
  void project(unsigned layer_id,TensorRole role,const B* input,B* output,unsigned rows) {
    const auto& weight=projections[layer_id][unsigned(role)];
    if (quantized) quantized->run(handle,weight,input,output);
    else linear(input,weight.bf16,output,rows,weight.input,weight.output);
  }
  void capture(const Capture& sink,const std::string& name,const B* data,unsigned n,cudaStream_t stream) {
    if (!sink) return;
    std::vector<std::uint16_t> result(n);
    check_cuda(cudaMemcpyAsync(result.data(),data,n*2,cudaMemcpyDeviceToHost,stream),"capture grouped boundary");
    check_cuda(cudaStreamSynchronize(stream),"complete explicit capture");
    sink(name,result);
  }
};

BatchedExecutor::BatchedExecutor(ArtifactFile file,unsigned max_rows,cudaStream_t stream,nvfp4::ActivationPolicy policy,
    attention::Compute local, attention::Compute global) {
  if (!max_rows || max_rows>4096) throw std::invalid_argument("26B grouped capacity must be 1..4096");
  if (policy!=nvfp4::ActivationPolicy::always && policy!=nvfp4::ActivationPolicy::prefill)
    throw std::invalid_argument("26B invalid NVFP4 activation policy");
  impl_=std::make_unique<Impl>(std::move(file),max_rows,stream,policy,local,global);
}
BatchedExecutor::~BatchedExecutor()=default;
weight_qdq::ApplySummary BatchedExecutor::apply_qdq(const QdqMask& mask) {
  return apply_qdq_in_place(mask, impl_->artifact, impl_->weights.data());
}
const B* BatchedExecutor::target_embedding() const { return static_cast<const B*>(impl_->weights.data()); }
const B* BatchedExecutor::shared_global_key_norm() const { return impl_->layer[29][unsigned(TensorRole::k_norm)]; }
BatchedExecutor::MemoryUsage BatchedExecutor::memory_usage() const {
  const auto& s = *impl_;
  return {s.weights.size(), s.expert_tables.size(), s.scratch.size(), s.positions.size(),
      s.moe.routing_bytes(), s.moe.projection_bytes(), s.moe.reduction_bytes(),
      s.attention.bytes(), s.quantized ? s.quantized->bytes() : 0};
}
std::size_t BatchedExecutor::weight_bytes() const { return impl_->weights.size()+impl_->expert_tables.size(); }
std::size_t BatchedExecutor::scratch_bytes() const {
  const auto& s=*impl_;return s.scratch.size()+s.positions.size()+s.moe.bytes()+s.attention.bytes()+(s.quantized?s.quantized->bytes():0);
}

const B* BatchedExecutor::forward(const std::uint32_t* tokens,const std::vector<Segment>& segments,nvfp4::Phase phase,
                                 const Capture& sink) {
  auto& s=*impl_;const auto stream=s.stream;unsigned rows=0;
  // Validate the entire dispatch before modifying any sequence's cache.
  for (const auto& segment:segments) {
    if (!segment.rows || segment.rows>s.capacity-rows || std::uint64_t(segment.position)+segment.rows>kMaxPositions)
      throw std::invalid_argument("26B grouped segment exceeds row/context capacity");
    if (segment.image_features && (segment.rows>1120 || segment.staging.data || phase!=nvfp4::Phase::prefill))
      throw std::invalid_argument("26B image requires a complete prefill segment of at most1120 rows");
    for (const auto& capture : segment.captures)
      if (!capture.output || !capture.completed_layers || capture.completed_layers>kLayerCount)
        throw std::invalid_argument("26B grouped invalid layer capture");
    if (segment.staging.data) {
      validate_mtp_staging(segment.staging, segment.rows);
      if (phase != nvfp4::Phase::decode)
        throw std::invalid_argument("26B MTP verifier requires decode phase");
    } else if (segment.staging.bytes || segment.staging.capacity_rows)
      throw std::invalid_argument("26B MTP incomplete staging view");
    for (unsigned l=0;l<30;++l)
      validate_cache_view(segment.cache[l], is_global_layer(l), segment.position +
          (segment.staging.data ? 0 : segment.rows));
    rows+=segment.rows;
  }
  for (const auto& segment : segments)
    if (segment.staging.data && rows > 1280)
      throw std::invalid_argument("26B MTP verifier exceeds 1280 rows");
  if (!rows || !tokens) throw std::invalid_argument("26B grouped forward needs tokens");
  if (phase!=nvfp4::Phase::prefill && phase!=nvfp4::Phase::decode)
    throw std::invalid_argument("26B invalid execution phase");
  if (s.quantized) s.quantized->prepare(rows,phase);
  B* h=static_cast<B*>(s.scratch.data());
  B* normalized=h+rows*2816,*branch=normalized+rows*2816,*shared=branch+rows*8192;
  B* routed=shared+rows*2816,*gate=routed+rows*2816,*up=gate+rows*2112;
  B* product=up+rows*2112,*q=product+rows*2112,*k=q+rows*8192;
  B* v=k+rows*2048,*temporary=v+rows*2048;
  auto* positions=static_cast<unsigned*>(s.positions.data());
  unsigned offset=0;
  for (const auto& segment:segments) {
    set_positions<<<(segment.rows+255)/256,256,0,stream>>>(positions+offset,segment.position,segment.rows);
    offset+=segment.rows;
  }
  const unsigned hidden=rows*kHiddenSize,blocks=(hidden+255)/256;
  const bool images = std::any_of(segments.begin(), segments.end(),
      [](const Segment& segment) { return segment.image_features != nullptr; });
  if (!images) {
    embeddings<<<blocks,256,0,stream>>>(static_cast<const B*>(s.weights.data()),tokens,h,hidden);
  } else {
    offset=0;
    for (const auto& segment : segments) {
      const auto elements=std::size_t(segment.rows)*kHiddenSize;
      auto* destination=h+std::size_t(offset)*kHiddenSize;
      if (segment.image_features)
        check_cuda(cudaMemcpyAsync(destination,segment.image_features,elements*sizeof(B),
            cudaMemcpyDeviceToDevice,stream),"insert26B image features");
      else
        embeddings<<<(elements+255)/256,256,0,stream>>>(static_cast<const B*>(s.weights.data()),
            tokens+offset,destination,elements);
      offset+=segment.rows;
    }
  }
  s.capture(sink,"embedding.0",h,hidden,stream);
  auto& attention_inputs=s.attention_inputs;
  for (unsigned l=0;l<30;++l) {
    auto w=[&](TensorRole role) { return s.layer[l][unsigned(role)]; };
    std::string prefix=sink?"layer."+std::to_string(l)+".":"";
    auto capture=[&](const char* name,const B* data,unsigned n) {
      if (sink) s.capture(sink,prefix+name,data,n,stream);
    };
    bool global=is_global_layer(l);unsigned d=global?512:256,heads=global?2:8;
    normalize(h,w(TensorRole::input_norm),normalized,rows,kHiddenSize,stream);
    s.project(l,TensorRole::q_proj,normalized,q,rows);
    capture("self_attn.q_proj.0",q,rows*16*d);
    s.project(l,TensorRole::k_proj,normalized,k,rows);
    capture("self_attn.k_proj.0",k,rows*heads*d);
    if (!global) s.project(l,TensorRole::v_proj,normalized,v,rows);
    normalize(global?k:v,nullptr,temporary,rows*heads,d,stream);
    capture("self_attn.v_norm.0",temporary,rows*heads*d);
    normalize(q,w(TensorRole::q_norm),branch,rows*16,d,stream);
    capture("self_attn.q_norm.0",branch,rows*16*d);
    row_rope<<<(rows*16*d+255)/256,256,0,stream>>>(branch,q,16,global,positions,rows);
    normalize(k,w(TensorRole::k_norm),branch,rows*heads,d,stream);
    capture("self_attn.k_norm.0",branch,rows*heads*d);
    row_rope<<<(rows*heads*d+255)/256,256,0,stream>>>(branch,k,heads,global,positions,rows);
    attention_inputs.clear();
    offset=0;
    for (const auto& segment:segments) {
      attention_inputs.push_back({q+std::size_t(offset)*16*d,k+std::size_t(offset)*heads*d,
          temporary+std::size_t(offset)*heads*d,segment.cache[l],segment.position,segment.rows,
          branch+std::size_t(offset)*16*d,segment.image_features!=nullptr});
      offset+=segment.rows;
    }
    s.attention.run_batch(s.handle,attention_inputs,w(TensorRole::k_norm),global,stream);
    s.cache_writes.clear();
    offset=0;
    for (const auto& segment:segments) {
      auto* key=k+std::size_t(offset)*heads*d;
      auto* val=temporary+std::size_t(offset)*heads*d;
      if (segment.staging.data)
        stage_mtp_layer(segment.staging,l,segment.rows,key,val,stream);
      else s.cache_writes.push_back({key,val,segment.cache[l],segment.position,segment.rows});
      offset+=segment.rows;
    }
    write_cache_batch(s.cache_writes,global,stream);
    capture("context",branch,rows*16*d);
    s.project(l,TensorRole::o_proj,branch,normalized,rows);
    capture("self_attn.o_proj.0",normalized,hidden);
    normalize(normalized,w(TensorRole::post_attention_norm),branch,rows,kHiddenSize,stream);
    add<<<blocks,256,0,stream>>>(h,branch,h,hidden);
    // Q is dead after attention. Its 8192-wide storage holds both 2816-wide
    // router and expert inputs while the shared MLP uses normalized.
    B* router_input=q;
    B* expert_input=q+hidden;
    normalize_feedforward(h,w(TensorRole::pre_feedforward_norm),w(TensorRole::router_scale),
        w(TensorRole::pre_feedforward_norm_2),normalized,router_input,expert_input,rows,stream);
    s.project(l,TensorRole::gate_proj,normalized,gate,rows);
    s.project(l,TensorRole::up_proj,normalized,up,rows);
    activation<<<(rows*kMlpSize+255)/256,256,0,stream>>>(gate,up,product,rows*kMlpSize);
    s.project(l,TensorRole::down_proj,product,branch,rows);
    normalize(branch,w(TensorRole::post_feedforward_norm_1),shared,rows,kHiddenSize,stream);
    capture("post_feedforward_layernorm_1.0",shared,hidden);
    s.linear(router_input,w(TensorRole::router_proj),branch,rows,kHiddenSize,128);
    capture("router.proj.0",branch,rows*128);
    capture("pre_feedforward_layernorm_2.0",expert_input,hidden);
    s.moe.run(expert_input,branch,w(TensorRole::router_per_expert_scale),
        static_cast<const ExpertWeights*>(s.expert_tables.data())+l*128,routed,rows,s.expert_bf16[l],s.expert_fp8[l],s.expert_nvfp4[l],nvfp4::fp4_activations(s.activation_policy,phase),stream);
    if (sink) {
      std::vector<int> selected(rows*8);
      check_cuda(cudaMemcpyAsync(selected.data(),s.moe.selected_experts(),selected.size()*sizeof(int),
          cudaMemcpyDeviceToHost,stream),"capture grouped routes");
      check_cuda(cudaStreamSynchronize(stream),"complete explicit routing capture");
      std::vector<std::uint16_t> ids(selected.size());
      for (std::size_t i=0;i<ids.size();++i) {
        float id=float(selected[i]);std::uint32_t bits;
        std::memcpy(&bits,&id,sizeof(bits));ids[i]=std::uint16_t(bits>>16);
      }
      sink(prefix+"router.2",ids);
    }
    capture("experts.0",routed,hidden);
    normalize(routed,w(TensorRole::post_feedforward_norm_2),branch,rows,kHiddenSize,stream);
    capture("post_feedforward_layernorm_2.0",branch,hidden);
    add<<<blocks,256,0,stream>>>(shared,branch,shared,hidden);
    normalize(shared,w(TensorRole::post_feedforward_norm),branch,rows,kHiddenSize,stream);
    add<<<blocks,256,0,stream>>>(h,branch,h,hidden);
    row_scale<<<blocks,256,0,stream>>>(h,w(TensorRole::layer_scalar),hidden);
    capture("output.0",h,hidden);
    offset=0;
    for (const auto& segment : segments) {
      for (const auto& capture : segment.captures)
        if (capture.completed_layers==l+1)
          check_cuda(cudaMemcpyAsync(capture.output,h+std::size_t(offset)*kHiddenSize,
              std::size_t(segment.rows)*kHiddenSize*sizeof(B),cudaMemcpyDeviceToDevice,stream),
              "capture26B target probe rows");
      offset+=segment.rows;
    }
  }
  auto* final=reinterpret_cast<const B*>(static_cast<const char*>(s.weights.data())+s.artifact.entries().back().offset-kArtifactDataOffset);
  normalize(h,final,normalized,rows,kHiddenSize,stream);
  s.capture(sink,"final_norm.0",normalized,hidden,stream);
  check_cuda(cudaGetLastError(),"26B grouped forward launch");
  return normalized;
}

void BatchedExecutor::head(const B* hidden,unsigned rows,B* logits) {
  auto& s=*impl_;
  if (!hidden || !logits || !rows || rows>s.capacity) throw std::invalid_argument("26B head: invalid rows");
  s.linear(hidden,static_cast<const B*>(s.weights.data()),logits,rows,kHiddenSize,kVocabSize);
  auto n=std::size_t(rows)*kVocabSize;
  capped<<<(n+255)/256,256,0,s.stream>>>(logits,n);
  check_cuda(cudaGetLastError(),"26B head launch");
}
}  // namespace gewell::gemma4_26b_a4b::sm120
