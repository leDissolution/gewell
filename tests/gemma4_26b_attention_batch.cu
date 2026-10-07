#include "models/gemma4/26b_a4b/sm120/attention.cuh"
#include "models/gemma4/26b_a4b/sm120/kv_write.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

namespace s=gewell::gemma4_26b_a4b::sm120;
namespace kv=gewell::kv_cache;
using B=__nv_bfloat16;
using gewell::cuda_detail::DeviceAllocation;
using gewell::cuda_detail::check_cuda;

__global__ void initialize(B* data,unsigned n,unsigned seed,float scale,float bias=0.f) {
  unsigned i=blockIdx.x*256+threadIdx.x;
  if (i>=n) return;
  unsigned x=i+seed; x^=x>>16; x*=0x7feb352d; x^=x>>15; x*=0x846ca68b; x^=x>>16;
  data[i]=__float2bfloat16_rn((int(x%65536)-32768)*(scale/32768.f)+bias);
}

struct Request {
  unsigned rows,base,d,heads,pages;
  DeviceAllocation q,k,v,output,control,cache_data,page_table;
  kv::DeviceView cache{};
  s::AttentionInput input;
  Request(bool global,kv::Format format,unsigned position,unsigned n,unsigned seed,cudaStream_t stream,bool uniform)
      :rows(n),base(position),d(global?512:256),heads(global?2:8),pages((base+rows+255)/256),
       q(std::size_t(rows)*16*d*2),k(std::size_t(base+rows)*heads*d*2),v(k.size()),
       output(q.size()+64),control(q.size()),
       cache_data(global?pages*(2*256*kv::row_bytes(640,format,2)+256):2*8*1024*kv::row_bytes(256,format)),
       page_table(pages*8) {
    auto* key=static_cast<B*>(k.data()); auto* value=static_cast<B*>(v.data());
    initialize<<<(q.size()/2+255)/256,256,0,stream>>>(static_cast<B*>(q.data()),q.size()/2,seed,uniform?0.f:1.f/std::sqrt(float(d)));
    initialize<<<(k.size()/2+255)/256,256,0,stream>>>(key,k.size()/2,seed+1,1.f);
    initialize<<<(v.size()/2+255)/256,256,0,stream>>>(value,v.size()/2,seed+2,1.f);
    check_cuda(cudaMemsetAsync(output.data(),0xa5,output.size(),stream),"poison output guards");
    check_cuda(cudaMemsetAsync(cache_data.data(),0xff,cache_data.size(),stream),"poison cache");
    cache.format=format;cache.capacity=global?262144:1024;
    cache.key=static_cast<B*>(cache_data.data());cache.value=cache.key+cache_data.size()/4;
    cache.page_pool=cache.key;cache.page_tokens=256;cache.page_count=pages;
    cache.layer_offset_elements=64;
    cache.page_stride_elements=global?(2*256*kv::row_bytes(640,format,2)+256)/2:0;
    cache.page_offsets=static_cast<std::uint64_t*>(page_table.data());
    std::vector<std::uint64_t> offsets(pages);
    for (unsigned i=0;i<pages;++i) offsets[i]=(pages-1-i)*cache.page_stride_elements;
    check_cuda(cudaMemcpyAsync(page_table.data(),offsets.data(),pages*8,cudaMemcpyHostToDevice,stream),"page table");
    // Keep the temporary host page table alive until the copy completes.
    check_cuda(cudaStreamSynchronize(stream),"initialize request");
    for (unsigned begin=0;begin<base;begin+=4096)
      s::write_cache(key+std::size_t(begin)*heads*d,value+std::size_t(begin)*heads*d,
          cache,global,begin,std::min(4096u,base-begin),stream);
    input={static_cast<B*>(q.data()),key+std::size_t(base)*heads*d,value+std::size_t(base)*heads*d,
           cache,base,rows,static_cast<B*>(output.data())+16,false};
  }
};

std::vector<B> download(const B* p,std::size_t n) {
  std::vector<B> out(n);check_cuda(cudaMemcpy(out.data(),p,n*2,cudaMemcpyDeviceToHost),"download");return out;
}

int main(int argc,char** argv) {
  try {
    const bool benchmark=argc==3 && std::string(argv[1])=="--bench-position";
    if (argc!=1 && !benchmark) throw std::runtime_error("usage: gemma4_26b_attention_batch [--bench-position N]");
    const unsigned bench_position=benchmark?std::stoul(argv[2]):0;
    cudaStream_t stream{};check_cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"stream");
    cublasHandle_t handle{};
    if (cublasCreate(&handle)!=CUBLAS_STATUS_SUCCESS ||
        cublasSetMathMode(handle,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)!=CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cuBLAS initialization");
    s::AttentionWorkspace workspace;
    DeviceAllocation norm(1024);
    initialize<<<2,256,0,stream>>>(static_cast<B*>(norm.data()),512,73,.1f,1.f);
    const auto* weights=static_cast<B*>(norm.data());
    const unsigned positions[]={0,1,31,255,1023,1024,4099,16383};
    unsigned cases=0;
    for (bool global:{false,true}) for (auto format:{kv::Format::bf16,kv::Format::fp8})
      for (unsigned count:benchmark?std::vector<unsigned>{1,4,16,32}:std::vector<unsigned>{1,4,32,33}) for (bool mixed:{false,true}) {
        if (benchmark && mixed) continue;
        std::vector<std::unique_ptr<Request>> requests;
        std::vector<s::AttentionInput> inputs;
        for (unsigned i=0;i<count;++i) {
          requests.push_back(std::make_unique<Request>(global,format,benchmark?bench_position:positions[(i+3)%8],
              mixed && i%3==1?3:1,1009*(i+1),stream,!benchmark));
          inputs.push_back(requests.back()->input);
        }
        const auto serial=[&] {
          for (const auto& r:requests) workspace.run(handle,r->input.query,r->input.key,r->input.value,
              weights,r->cache,global,r->base,r->rows,static_cast<B*>(r->control.data()),stream);
        };
        const auto batched=[&] {workspace.run_batch(handle,inputs,weights,global,stream);};
        serial();batched();check_cuda(cudaStreamSynchronize(stream),"attention");
        std::vector<std::vector<B>> initial;
        for (const auto& r:requests) initial.push_back(download(r->input.output,r->q.size()/2));
        batched();check_cuda(cudaStreamSynchronize(stream),"repeat attention");
        double worst=0;
        for (unsigned i=0;i<count;++i) {
          const auto& r=*requests[i];
          auto ref=download(static_cast<B*>(r.control.data()),r.q.size()/2);
          auto got=download(r.input.output,r.q.size()/2);
          double error=0,total=0;
          for (std::size_t j=0;j<ref.size();++j) {
            const double a=__bfloat162float(ref[j]),b=__bfloat162float(got[j]);
            if (!std::isfinite(b)) throw std::runtime_error("nonfinite output");
            error+=(a-b)*(a-b);total+=a*a;
          }
          const double relative=std::sqrt(error/total);worst=std::max(worst,relative);
          if (!benchmark && relative>.001) throw std::runtime_error("attention numerical policy failed: "+std::to_string(relative));
          // Zero queries make both recipes the mean of visible V. Different
          // requests, poison, ring wraps and page maps expose visibility leaks.
          // Random-score arithmetic is checked by the independent Torch oracle.
          if (std::memcmp(initial[i].data(),got.data(),r.q.size()))
            throw std::runtime_error("attention repeat changed output");
          std::vector<unsigned char> guards(r.output.size());
          check_cuda(cudaMemcpy(guards.data(),r.output.data(),guards.size(),cudaMemcpyDeviceToHost),"guards");
          for (unsigned j=0;j<32;++j)
            if (guards[j]!=0xa5 || guards[guards.size()-1-j]!=0xa5) throw std::runtime_error("output guard overwritten");
        }
        float serial_ms=0,batched_ms=0;
        if (!mixed && (benchmark || count==1 || count==32)) {
          cudaEvent_t start{},end{};check_cuda(cudaEventCreate(&start),"event");check_cuda(cudaEventCreate(&end),"event");
          const auto measure=[&](const auto& run) {
            for (unsigned i=0;i<3;++i) run();
            check_cuda(cudaEventRecord(start,stream),"start");
            for (unsigned i=0;i<10;++i) run();
            check_cuda(cudaEventRecord(end,stream),"end");check_cuda(cudaEventSynchronize(end),"wait");
            float elapsed;check_cuda(cudaEventElapsedTime(&elapsed,start,end),"elapsed");return elapsed/10;
          };
          serial_ms=measure(serial);batched_ms=measure(batched);
          check_cuda(cudaEventDestroy(start),"event destroy");check_cuda(cudaEventDestroy(end),"event destroy");
        }
        ++cases;
        std::cout<<"global="<<global<<" fp8="<<(format==kv::Format::fp8)<<" batch="<<count
                 <<" mixed="<<mixed<<" uniform="<<!benchmark<<" relative_l2="<<worst<<" serial_ms="<<serial_ms
                 <<" batched_ms="<<batched_ms<<(benchmark?" OBSERVED\n":" PASS\n");
      }
    check_cuda(cudaStreamSynchronize(stream),"finish");
    cublasDestroy(handle);check_cuda(cudaStreamDestroy(stream),"destroy stream");
    std::cout<<"cases="<<cases<<(benchmark?" measured\n":" PASS\n");
    return 0;
  } catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
