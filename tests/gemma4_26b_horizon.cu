#include "models/gemma4/26b_a4b/sm120/batched_executor.cuh"
#include "gewell/models/gemma4/26b_a4b/cache_config.h"
#include "runtime/physical_cache.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace m=gewell::gemma4_26b_a4b;
namespace s=m::sm120;
namespace kv=gewell::kv_cache;
using gewell::cuda_detail::DeviceAllocation;
using gewell::cuda_detail::check_cuda;
using B=__nv_bfloat16;
namespace fs=std::filesystem;
unsigned number(const char* text) {
  std::size_t end;auto value=std::stoul(text,&end);
  if (text[end] || !value || value>m::kMaxPositions) throw std::runtime_error("invalid size");
  return value;
}
int main(int argc,char** argv) {
 try {
  if (argc!=6) throw std::runtime_error("usage: horizon ARTIFACT OUTPUT CHUNK HORIZON PATTERN.u32");
  unsigned chunk=number(argv[3]),horizon=number(argv[4]);
  if (chunk>4096) throw std::runtime_error("chunk exceeds4096");
  std::ifstream pattern_file(argv[5],std::ios::binary|std::ios::ate);
  auto bytes=pattern_file.tellg();
  if (bytes<8 || bytes>4096 || bytes%4) throw std::runtime_error("invalid pattern size");
  std::vector<unsigned> pattern(std::size_t(bytes)/4);pattern_file.seekg(0);
  pattern_file.read(reinterpret_cast<char*>(pattern.data()),bytes);
  if (!pattern_file || pattern[0]!=2 || std::any_of(pattern.begin(),pattern.end(),[](auto t){return t>=m::kVocabSize;}))
    throw std::runtime_error("pattern requires BOS and vocabulary token IDs");
  fs::path output(argv[2]);
  if (!fs::create_directory(output)) throw std::runtime_error("refusing existing output");
  fs::copy_file(argv[5],output/"pattern.u32");
  cudaStream_t stream{};check_cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"stream");
  auto executor=std::make_unique<s::BatchedExecutor>(m::ArtifactFile::Open(argv[1]),chunk,stream,gewell::nvfp4::ActivationPolicy::always);
  auto config=kv::compact_pool_config(s::kCacheGeometry,4ULL<<30,0,64ULL<<20);
  config.gpu_bytes=config.local_ring_bytes+((horizon+255)/256)*config.global_page_bytes+(2ULL<<20);
  kv::CacheLedger ledger(config);
  gewell::runtime::PhysicalCache cache(s::kCacheGeometry,config,ledger,1024);
  auto execution=ledger.try_begin_batch(0,horizon);
  if (!execution) throw std::runtime_error("cache admission failed");
  cache.clear_page_table(ledger.execution(*execution));
  DeviceAllocation tokens(chunk*4),logits(std::size_t(m::kVocabSize)*2);
  std::vector<unsigned> input(chunk);
  std::vector<B> host_logits(m::kVocabSize),host_hidden(m::kHiddenSize);
  std::vector<unsigned> marks;
  for (auto mark:{1u,256u,257u,1024u,1025u,131072u,131073u,262144u}) if (mark<=horizon) marks.push_back(mark);
  if (marks.back()!=horizon) marks.push_back(horizon);
  std::ofstream calls(output/"calls.tsv"),results(output/"boundaries.tsv");
  calls<<"position\trows\tseconds\n";
  results<<"processed_tokens\targmax\tlogits_l2\n";
  unsigned position=0,mark_index=0;
  auto start=std::chrono::steady_clock::now();
  while (position<horizon) {
   auto begin=std::chrono::steady_clock::now();
   unsigned rows=std::min(chunk,marks[mark_index]-position);
   for (unsigned i=0;i<rows;++i) {
    auto p=position+i;input[i]=p?pattern[1+(p-1)%(pattern.size()-1)]:2;
   }
   auto plan=ledger.prepare_write(*execution,position,rows);
   if (!plan.copies.empty()) throw std::runtime_error("unexpected private-cache COW");
   cache.upload_page_table(ledger.execution(*execution),gewell::runtime::CompletionContext(stream));
   s::Segment segment{position,rows,{}};
   for (unsigned l=0;l<m::kLayerCount;++l) segment.cache[l]=cache.layer(*execution,l);
   check_cuda(cudaMemcpyAsync(tokens.data(),input.data(),rows*4,cudaMemcpyHostToDevice,stream),"tokens");
   const auto* hidden=executor->forward(static_cast<unsigned*>(tokens.data()),{segment},gewell::nvfp4::Phase::prefill);
   const bool capture=position+rows==marks[mark_index];
   if (capture) {
    auto* last=hidden+std::size_t(rows-1)*m::kHiddenSize;
    executor->head(last,1,static_cast<B*>(logits.data()));
    check_cuda(cudaMemcpyAsync(host_logits.data(),logits.data(),host_logits.size()*2,cudaMemcpyDeviceToHost,stream),"logits");
    check_cuda(cudaMemcpyAsync(host_hidden.data(),last,host_hidden.size()*2,cudaMemcpyDeviceToHost,stream),"hidden");
   }
   check_cuda(cudaStreamSynchronize(stream),"chunk completion");
   calls<<position<<'\t'<<rows<<'\t'<<std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()<<'\n';
   position+=rows;
   if (capture) {
    unsigned best=0;double square=0;
    for (unsigned i=0;i<m::kVocabSize;++i) {
     float value=__bfloat162float(host_logits[i]);
     if (!std::isfinite(value) || std::abs(value)>30) throw std::runtime_error("invalid logits");
     square+=double(value)*value;
     if (value>__bfloat162float(host_logits[best])) best=i;
    }
    for (auto value:host_hidden) if (!std::isfinite(__bfloat162float(value))) throw std::runtime_error("invalid hidden");
    auto save=[&](const char* name,const std::vector<B>& data) {
     std::ofstream file(output/(std::to_string(position)+name),std::ios::binary);
     file.write(reinterpret_cast<const char*>(data.data()),data.size()*2);
     if (!file) throw std::runtime_error("capture write failed");
    };
    save(".logits.bf16",host_logits);save(".hidden.bf16",host_hidden);
    results<<position<<'\t'<<best<<'\t'<<std::sqrt(square)<<'\n';results.flush();++mark_index;
   }
   if (capture || position%16384<chunk) std::cout<<"processed="<<position<<" seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<std::endl;
  }
  if (ledger.execution(*execution).processed_tokens!=horizon) throw std::runtime_error("incomplete cache horizon");
  std::cout<<"horizon="<<horizon<<" kv_bytes="<<config.gpu_bytes<<" weights="<<executor->weight_bytes()<<" scratch="<<executor->scratch_bytes()<<std::endl;
  ledger.release_execution(*execution);executor.reset();check_cuda(cudaStreamDestroy(stream),"destroy stream");
 } catch (const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}
