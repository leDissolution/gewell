#include "models/gemma4/26b_a4b/sm120/moe.cuh"
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <map>
#include <tuple>
#include <cublas_v2.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstring>

namespace s = gewell::gemma4_26b_a4b::sm120;
using S = gewell::gemma4_26b_a4b::StorageType;
using B = __nv_bfloat16;
using D = gewell::cuda_detail::DeviceAllocation;
using gewell::cuda_detail::check_cuda;
void blas(cublasStatus_t status) { if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("reference cuBLAS failure"); }
template<class T> std::vector<T> download(const void* p, std::size_t count) {
  std::vector<T> v(count);
  if (count) check_cuda(cudaMemcpy(v.data(),p,count*sizeof(T),cudaMemcpyDeviceToHost),"download test data");
  return v;
}
template<class T> void upload(void* p, const std::vector<T>& v) {
  if (!v.empty()) check_cuda(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice),"upload test data");
}
// cuBLAS SGEMM control operates on decoded FP32 values, independently of the
// grouped FP8 instruction/register mapping. Apply globals only in its epilogue.
__global__ void reference_input(const B* input, float* output, unsigned n, float scale, bool fp8) {
  unsigned i=blockIdx.x*256+threadIdx.x;
  if (i<n) {
    float x=float(input[i]);
    if (fp8) x=float(__nv_fp8_e4m3(__fmul_rn(x,1.f/scale)));
    output[i]=x;
  }
}
__global__ void reference_weight(const void* input, float* output, unsigned n, bool fp8) {
  unsigned i=blockIdx.x*256+threadIdx.x;
  if (i<n) output[i]=fp8?float(static_cast<const __nv_fp8_e4m3*>(input)[i]):float(static_cast<const B*>(input)[i]);
}
void compare(const B* actual, const B* expected, std::size_t n, const char* label) {
  for (std::size_t i=0;i<n;++i)
    if (float(actual[i]) != float(expected[i]))
      throw std::runtime_error(std::string(label)+" mismatch at "+std::to_string(i)+" actual="+std::to_string(float(actual[i]))+" expected="+std::to_string(float(expected[i])));
}
int main(int argc,char** argv) {
 try {
  bool quick=false,nv_only=false,scale_stress=false;
  for(int i=1;i<argc;++i) {
    if(std::string(argv[i])=="--quick") quick=true;
    else if(std::string(argv[i])=="--nvfp4") nv_only=true;
    else if(std::string(argv[i])=="--scale-stress") scale_stress=true;
    else throw std::runtime_error("usage: moe_quantized [--quick] [--nvfp4] [--scale-stress]");
  }
  const unsigned capacity=quick?129:4096;
  constexpr unsigned K=2816, H=704, E=128;
  constexpr std::size_t elements=std::size_t(K)*H;
  std::array<D,3> wb={D(elements*2),D(elements*2),D(elements*2)};
  std::array<D,3> wf={D(elements),D(elements),D(elements)};
  std::array<D,3> wn={D(768*K/2),D(768*K/2),D(elements/2)};
  std::array<D,3> sn={D(gewell::nvfp4::scale_storage_bytes(768,K)),D(gewell::nvfp4::scale_storage_bytes(768,K)),D(gewell::nvfp4::scale_storage_bytes(K,H))};
  std::array<D,3> sn_varied={D(sn[0].size()),D(sn[1].size()),D(sn[2].size())};
  for (unsigned role=0;role<3;++role) {
    unsigned width=role==2?H:K, out=role==2?K:H;
    std::vector<B> bf(elements); std::vector<__nv_fp8_e4m3> fp(elements);
    for (unsigned n=0;n<out;++n) for (unsigned k=0;k<width;++k) {
      // Dense signed values vary along K and N, exercising every register and
      // the entire 704/2816 input width. Products are exactly representable.
      float value=(int((n*7+k*11+role*3)%17)-8)/16.f;
      bf[n*width+k]=B(value); fp[n*width+k]=__nv_fp8_e4m3(value);
    }
    upload(wb[role].data(),bf); upload(wf[role].data(),fp);
    std::vector<std::uint8_t> packed(wn[role].size(),0), sf(sn[role].size(),0);
    // All signed E2M1 values and per-K16/per-row E4M3 scale variations.
    for(unsigned n=0;n<out;++n) {
      for(unsigned k=0;k<width;k+=2)
        packed[(n*width+k)/2]=((n*7+k*3+role)%16)|(((n*7+(k+1)*3+role)%16)<<4);
      for(unsigned block=0;block<width/16;++block) {
        __nv_fp8_e4m3 value(std::ldexp(1.f,int((n*3+block*5+role)%7)-4));
        sf[gewell::nvfp4::scale_offset(n,block,width)]=value.__x;
      }
    }
    upload(wn[role].data(),packed); upload(sn[role].data(),sf);
    for(unsigned n=0;n<out;++n) for(unsigned block=0;block<width/16;++block) {
      unsigned pattern=n*3+block*5+role;
      float value=pattern%11==0?0.f:pattern%13==0?float(1+pattern%7)/512.f:
          std::ldexp(float(8+pattern%8)/8.f,int(pattern%7)-4);
      sf[gewell::nvfp4::scale_offset(n,block,width)]=__nv_fp8_e4m3(value).__x;
    }
    upload(sn_varied[role].data(),sf);
  }
  D table(E*sizeof(s::ExpertWeights)), input(capacity*K*2), scores(capacity*E*2), scales(E*2), output(capacity*K*2+256);
  D ref_input(std::size_t(capacity)*K*4), ref_weight(elements*4), ref_output(std::size_t(std::max(capacity,1024u))*K*4);
  D group_input(std::size_t(std::max(capacity,1024u))*K*2);
  s::MoeWorkspace workspace(capacity);
  const auto initial_bytes=workspace.bytes();
  cublasHandle_t handle{}; blas(cublasCreate(&handle));
  blas(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));
  blas(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST));
  cublasLtHandle_t lt{}; blas(cublasLtCreate(&lt));
  std::map<std::tuple<unsigned,unsigned,unsigned>,std::unique_ptr<gewell::nvfp4::Plan>> plans;
  D nv_scratch(gewell::nvfp4::scratch_upper_bound_bytes(capacity,K,K));
  unsigned cases=0,fp64_reference_corrections=0;
  for (unsigned recipe : {0u,1u,2u,3u,4u,5u}) {
    if(nv_only && recipe<3) continue;
    if(scale_stress && recipe!=5) continue;
    const bool mixed=recipe!=0;
    std::vector<s::ExpertWeights> weights(E);
    for (unsigned e=0;e<E;++e) for (unsigned role=0;role<3;++role) {
      auto& p=role==0?weights[e].gate:(role==1?weights[e].up:weights[e].down);
      p.storage=mixed && (e+role)%3==0?S::bf16:S::fp8_w8a8;
      if(recipe>=3) p.storage=recipe==3 || recipe==5?S::nvfp4_w4a4:
          ((e+role)%3==0?S::bf16:((e+role)%3==1?S::fp8_w8a8:S::nvfp4_w4a4));
      p.bf16=static_cast<B*>(wb[role].data());
      p.fp8={static_cast<const std::uint8_t*>(wf[role].data()),std::ldexp(1.f,int((e+role)%3)-2),std::ldexp(1.f,int((e*3+role)%4)-5),nullptr};
      if(recipe==2 || recipe==5) {
        constexpr float input_scales[]={0.0003f,0.3f,1.125f};
        p.fp8.input_scale=input_scales[(e+role)%3];
        p.fp8.weight_scale=0.03f*(role+1);
      }
      p.nvfp4={static_cast<const std::uint8_t*>(wn[role].data()),static_cast<const std::uint8_t*>((recipe==5?sn_varied[role]:sn[role]).data()),p.fp8.input_scale,p.fp8.weight_scale};
    }
    upload(table.data(),weights);
    for(bool fp4 : {true,false}) {
    if(recipe<3 && !fp4) continue;
    for (unsigned rows : {1u,7u,8u,17u,32u,33u,129u,capacity,1u,0u}) for (unsigned distribution : {0u,1u,2u}) {
      if((recipe==2 || recipe==5 || distribution==2) && rows>129) continue;
      auto base=[&](unsigned r) {return distribution==0 || (distribution==2 && r%17)?120u:r*7;};
      std::vector<B> x(rows*K), logits(rows*E), expert_scales(E,B(1.f));
      for (unsigned r=0;r<rows;++r) {
        for (unsigned k=0;k<K;++k) x[r*K+k]=B((int((r*3+k*5)%17)-8)/16.f);
        for (unsigned e=0;e<E;++e) logits[r*E+e]=B(-32.f);
        for (unsigned j=0;j<8;++j) logits[r*E+(base(r)+j)%E]=B(0.f);
      }
      upload(input.data(),x); upload(scores.data(),logits); upload(scales.data(),expert_scales);
      check_cuda(cudaMemset(output.data(),0x5a,output.size()),"poison output");
      workspace.run(static_cast<B*>(input.data()),static_cast<B*>(scores.data()),static_cast<B*>(scales.data()),
                    static_cast<s::ExpertWeights*>(table.data()),static_cast<B*>(output.data()),rows,recipe==1 || recipe==2 || recipe==4,recipe<3 || recipe==4,recipe>=3,fp4);
      check_cuda(cudaDeviceSynchronize(),"run grouped quantized experts");
      auto guards=download<unsigned char>(static_cast<char*>(output.data())+rows*K*2,256);
      for (auto c:guards) if(c!=0x5a) throw std::runtime_error("output guard changed");
      if(workspace.bytes()!=initial_bytes) throw std::runtime_error("workspace grew");
      if(!rows) { ++cases; continue; }
      auto ids=download<int>(workspace.selected_experts(),rows*8);
      for(unsigned r=0;r<rows;++r) {
        std::array<int,8> expected{};
        for(unsigned j=0;j<8;++j) expected[j]=(base(r)+j)%E;
        std::sort(expected.begin(),expected.end());
        for(unsigned j=0;j<8;++j) if(ids[r*8+j]!=expected[j]) throw std::runtime_error("selected set mismatch");
      }
      auto offsets=download<int>(workspace.offsets(),E+1), assignments=download<int>(workspace.sorted_assignments(),rows*8);
      auto gate_up=download<B>(workspace.gate_up_capture(),std::size_t(rows)*8*2*H);
      auto activated=download<B>(workspace.activated_capture(),std::size_t(rows)*8*H);
      auto down=download<B>(workspace.down_capture(),std::size_t(rows)*8*K);
      std::vector<B> ref_down(down.size());
      for (unsigned e=0;e<E;++e) {
        unsigned m=offsets[e+1]-offsets[e]; if(!m) continue;
        if(m>rows) throw std::runtime_error("expert exceeds row count");
        for(unsigned role=0;role<3;++role) {
          unsigned width=role==2?H:K, out=role==2?K:H;
          const auto& p=role==0?weights[e].gate:(role==1?weights[e].up:weights[e].down);
          bool fp8=p.storage==S::fp8_w8a8, nv=p.storage==S::nvfp4_w4a4;
          unsigned control_rows=(!fp8 && !nv && rows>32) || (nv && !fp4)?std::max(m,1024u):m;
          std::vector<B> gx(std::size_t(control_rows)*width,B(0.f));
          for(unsigned r=0;r<m;++r) {
            const B* source=role==2?activated.data()+(offsets[e]+r)*H:x.data()+(assignments[offsets[e]+r]/8)*K;
            std::memcpy(gx.data()+r*width,source,width*2);
          }
          upload(group_input.data(),gx);
          float alpha=fp8?p.fp8.input_scale*p.fp8.weight_scale:1.f,beta=0;
          std::vector<B> expected;
          std::vector<float> expected_float;
          if(nv && fp4) {
            unsigned physical=(out+127)/128*128;
            auto key=std::make_tuple(m,width,physical);
            auto& plan=plans[key];
            if(!plan) plan=std::make_unique<gewell::nvfp4::Plan>(lt,m,width,physical);
            plan->run(static_cast<B*>(group_input.data()),p.nvfp4,static_cast<B*>(ref_output.data()),nv_scratch.data(),nv_scratch.size());
            auto padded=download<B>(ref_output.data(),std::size_t(m)*physical);
            expected.resize(std::size_t(m)*out);
            for(unsigned r=0;r<m;++r) std::memcpy(expected.data()+r*out,padded.data()+r*physical,out*2);
          } else if((!fp8 && rows>32) || nv) {
            const void* matrix=wb[role].data();
            if(nv) {
              gewell::nvfp4::dequantize(p.nvfp4,width,(out+127)/128*128,static_cast<B*>(ref_weight.data()));
              matrix=ref_weight.data();
            }
            blas(cublasSetMathMode(handle,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
            blas(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,out,control_rows,width,&alpha,matrix,CUDA_R_16BF,width,
                             group_input.data(),CUDA_R_16BF,width,&beta,ref_output.data(),CUDA_R_16BF,out,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
            expected=download<B>(ref_output.data(),std::size_t(m)*out);
            blas(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));
          } else {
            reference_input<<<(m*width+255)/256,256>>>(static_cast<B*>(group_input.data()),static_cast<float*>(ref_input.data()),m*width,p.fp8.input_scale,fp8);
            reference_weight<<<(elements+255)/256,256>>>(fp8?wf[role].data():wb[role].data(),static_cast<float*>(ref_weight.data()),elements,fp8);
            blas(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,out,m,width,&alpha,ref_weight.data(),CUDA_R_32F,width,
                             ref_input.data(),CUDA_R_32F,width,&beta,ref_output.data(),CUDA_R_32F,out,CUBLAS_COMPUTE_32F_PEDANTIC,CUBLAS_GEMM_DEFAULT));
            expected_float=download<float>(ref_output.data(),std::size_t(m)*out);
            expected.resize(expected_float.size());
            for(std::size_t i=0;i<expected.size();++i) expected[i]=B(expected_float[i]);
          }
          for(unsigned r=0;r<m;++r) {
            const B* actual=role==2?down.data()+(offsets[e]+r)*K:gate_up.data()+(offsets[e]+r)*2*H+role*H;
            // FP32 SGEMM can itself cross a BF16 rounding midpoint. Resolve
            // disputed FP8 elements using exact decoded E4M3 products summed
            // in FP64; this strengthens the reference, not the tolerance.
            if(fp8) {
              std::vector<float> rx;
              for(unsigned col=0;col<out;++col) if(float(actual[col])!=float(expected[r*out+col])) {
                if(rx.empty()) rx=download<float>(static_cast<float*>(ref_input.data())+r*width,width);
                auto rw=download<float>(static_cast<float*>(ref_weight.data())+col*width,width);
                double dot=0; for(unsigned k=0;k<width;++k) dot+=double(rx[k])*rw[k];
                // One FP32 epilogue multiplication and then the BF16 cast.
                B exact(float(dot)*alpha);
                if(float(actual[col])!=float(exact)) {
                  std::cerr.precision(17);
                  std::cerr<<"FP64dot="<<dot<<" alpha="<<alpha<<" exact_scaled="<<dot*alpha
                           <<" reference_float="<<expected_float[r*out+col]<<" actual="<<float(actual[col])<<'\n';
                  throw std::runtime_error("FP8 disagrees with FP64 rounding control");
                }
                expected[r*out+col]=exact;
                ++fp64_reference_corrections;
              }
            }
            try {compare(actual,expected.data()+r*out,out,role==2?"down":"gate/up");}
            catch(const std::exception& error) {
              throw std::runtime_error("expert="+std::to_string(e)+" role="+std::to_string(role)+" storage="+std::to_string(unsigned(p.storage))+" fp4="+std::to_string(fp4)+" "+error.what());
            }
            if(role==2) std::memcpy(ref_down.data()+(offsets[e]+r)*K,expected.data()+r*out,out*2);
          }
        }
      }
      // Independently restore original row order and apply the specified BF16
      // weighted-product/add boundaries. Equal selected logits yield weight1/8.
      std::vector<int> inverse(rows*8);
      for(unsigned i=0;i<rows*8;++i) inverse[assignments[i]]=i;
      auto actual=download<B>(output.data(),std::size_t(rows)*K);
      std::vector<B> expected(actual.size(),B(0.f));
      for(unsigned r=0;r<rows;++r) for(unsigned k=0;k<K;++k)
        for(unsigned j=0;j<8;++j)
          expected[r*K+k]=B(float(expected[r*K+k])+float(B(float(ref_down[std::size_t(inverse[r*8+j])*K+k])/8.f)));
      compare(actual.data(),expected.data(),expected.size(),"reduction");
      ++cases;
      std::cout<<"recipe="<<recipe<<" rows="<<rows<<" fp4="<<fp4<<" distribution="<<distribution<<" exact PASS\n";
    }
  }
  }
  plans.clear();
  blas(cublasLtDestroy(lt));
  blas(cublasDestroy(handle));
  std::cout<<"cases="<<cases<<" fp64_reference_corrections="<<fp64_reference_corrections<<" workspace_bytes="<<initial_bytes<<" PASS\n";
  return 0;
 } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
