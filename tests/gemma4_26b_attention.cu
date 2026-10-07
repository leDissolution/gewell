#include "models/gemma4/26b_a4b/sm120/attention.cuh"
#include "models/gemma4/26b_a4b/sm120/kv_write.cuh"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

using B=__nv_bfloat16;
using gewell::cuda_detail::DeviceAllocation;
using gewell::cuda_detail::check_cuda;
namespace s=gewell::gemma4_26b_a4b::sm120;
namespace fs=std::filesystem;
void read(const fs::path& path,void* output,std::size_t bytes) {
  std::vector<char> data(bytes);
  std::ifstream f(path,std::ios::binary); f.read(data.data(),bytes);
  if (!f || f.peek()!=EOF) throw std::runtime_error("fixture size: "+path.string());
  check_cuda(cudaMemcpy(output,data.data(),bytes,cudaMemcpyHostToDevice),"upload fixture");
}
int main(int argc,char** argv) {
  try {
    if (argc<3 || argc>7) throw std::runtime_error("usage: attention FIXTURES OUTPUT [--image] [--fp8-compute] [--batch-size 1|2]");
    bool image = false, fp8_compute = false;
    unsigned batch_size=0;
    for (int i=3;i<argc;++i) {
      const std::string option(argv[i]);
      if (option=="--image") image=true;
      else if (option=="--fp8-compute") fp8_compute=true;
      else if (option=="--batch-size") {
        if (++i==argc || (std::string(argv[i])!="1" && std::string(argv[i])!="2"))
          throw std::runtime_error("batch size must be 1 or 2");
        batch_size=unsigned(argv[i][0]-'0');
      }
      else throw std::runtime_error("unknown attention test option");
    }
    fs::path fixtures(argv[1]),output(argv[2]);
    if (!fs::create_directory(output)) throw std::runtime_error("refusing existing output");
    cudaStream_t stream{}; check_cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"stream");
    cublasHandle_t handle{};
    if (cublasCreate(&handle)!=CUBLAS_STATUS_SUCCESS ||
        cublasSetMathMode(handle,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)!=CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("cuBLAS initialization");
    const auto compute=fp8_compute ? gewell::attention::Compute::fp8 : gewell::attention::Compute::bf16;
    s::AttentionWorkspace workspace(4096,compute,compute);
    std::ifstream list(fixtures/"cases.tsv");
    std::string name; bool global,fp8; unsigned base,rows;
    while (list>>name>>global>>fp8>>base>>rows) {
      unsigned d=global?512:256,heads=global?2:8,total=base+rows;
      std::size_t qbytes=std::size_t(rows)*16*d*2,kvbytes=std::size_t(total)*heads*d*2;
      DeviceAllocation q(qbytes),k(kvbytes),v(kvbytes),norm(512*2),result(qbytes),second_result(qbytes);
      read(fixtures/name/"q.bf16",q.data(),qbytes); read(fixtures/name/"k.bf16",k.data(),kvbytes);
      read(fixtures/name/"v.bf16",v.data(),kvbytes); read(fixtures/name/"norm.bf16",norm.data(),1024);
      auto format=fp8?gewell::kv_cache::Format::fp8:gewell::kv_cache::Format::bf16;
      unsigned pages=(total+255)/256;
      std::size_t record=gewell::kv_cache::row_bytes(global?640:256,format,global?2:1);
      std::size_t stride=global?2*256*record+256:0;
      std::size_t bytes=global?pages*stride:2*8*1024*record;
      DeviceAllocation cache_data(bytes),table(pages*8);
      std::vector<std::uint64_t> offsets(pages);
      for (unsigned i=0;i<pages;++i) offsets[i]=(pages-1-i)*stride/2;
      check_cuda(cudaMemset(cache_data.data(),0xff,bytes),"poison unused cache");
      check_cuda(cudaMemcpy(table.data(),offsets.data(),pages*8,cudaMemcpyHostToDevice),"page map");
      gewell::kv_cache::DeviceView cache{};
      cache.format=format; cache.capacity=global?262144:1024;
      cache.key=static_cast<B*>(cache_data.data()); cache.value=cache.key+bytes/4;
      cache.page_pool=cache.key; cache.page_offsets=static_cast<std::uint64_t*>(table.data());
      cache.page_tokens=256; cache.page_count=pages; cache.layer_offset_elements=64;
      cache.page_stride_elements=stride/2;
      auto* keys=static_cast<B*>(k.data()); auto* values=static_cast<B*>(v.data());
      for (unsigned begin=0;begin<base;begin+=4096)
        s::write_cache(keys+std::size_t(begin)*heads*d,values+std::size_t(begin)*heads*d,
                       cache,global,begin,std::min(4096u,base-begin),stream);
      if (fp8 && base && base<=1024) {
        check_cuda(cudaStreamSynchronize(stream),"finish prefix writes");
        std::vector<char> record_data(record);
        const char* first=static_cast<const char*>(cache_data.data())+(global?offsets[0]*2+128:0);
        check_cuda(cudaMemcpy(record_data.data(),first,record,cudaMemcpyDeviceToHost),"capture first record");
        std::ofstream record_out(output/(name+".record"),std::ios::binary);
        record_out.write(record_data.data(),record);
      }
      auto run=[&] {
        s::AttentionInput input{static_cast<B*>(q.data()),keys+std::size_t(base)*heads*d,
            values+std::size_t(base)*heads*d,cache,base,rows,static_cast<B*>(result.data()),image};
        if (batch_size) {
          auto second=input;second.output=static_cast<B*>(second_result.data());
          const auto inputs=batch_size==1?std::vector<s::AttentionInput>{input}:std::vector<s::AttentionInput>{input,second};
          workspace.run_batch(handle,inputs,static_cast<B*>(norm.data()),global,stream);
        }
        else workspace.run(handle,input.query,input.key,input.value,static_cast<B*>(norm.data()),cache,
                           global,base,rows,input.output,stream,image);
      };
      std::vector<char> first(qbytes),second(qbytes);
      run(); check_cuda(cudaStreamSynchronize(stream),"attention");
      check_cuda(cudaMemcpy(first.data(),result.data(),qbytes,cudaMemcpyDeviceToHost),"read output");
      run(); check_cuda(cudaStreamSynchronize(stream),"repeat attention");
      check_cuda(cudaMemcpy(second.data(),result.data(),qbytes,cudaMemcpyDeviceToHost),"read repeat");
      if (first!=second) throw std::runtime_error("attention not deterministic");
      std::ofstream out(output/(name+".bf16"),std::ios::binary); out.write(first.data(),qbytes);
      if (!out) throw std::runtime_error("output write failed");
      std::cout<<name<<" completed, scratch="<<workspace.bytes()<<" bytes\n";
    }
    cublasDestroy(handle); check_cuda(cudaStreamDestroy(stream),"destroy stream");
    return 0;
  } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
