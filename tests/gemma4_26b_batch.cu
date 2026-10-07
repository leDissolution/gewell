#include "models/gemma4/26b_a4b/sm120/batched_executor.cuh"
#include "gewell/models/gemma4/26b_a4b/cache_config.h"
#include "runtime/physical_cache.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>
#include <cmath>

namespace m=gewell::gemma4_26b_a4b;
namespace s=m::sm120;
namespace kv=gewell::kv_cache;
namespace fs=std::filesystem;
using B=__nv_bfloat16;
using gewell::cuda_detail::DeviceAllocation;
using gewell::cuda_detail::check_cuda;
// Capture every logit row for the wide64 fixture, bounding the head buffer at32MiB.
constexpr unsigned FullLogitRows=64;
unsigned number(const std::string& text,unsigned maximum) {
  std::size_t consumed=0;auto value=std::stoul(text,&consumed);
  if (consumed!=text.size() || value>maximum) throw std::runtime_error("invalid number: "+text);
  return unsigned(value);
}
int main(int argc,char** argv) {
  try {
    if (argc<5) throw std::runtime_error("usage: batch ARTIFACT OUTPUT CHUNK [--greedy N] [--replay-calls TSV] [--image-features BF16] [--nvfp4-activation-policy always|prefill] TOKEN...");
    fs::path output(argv[2]);unsigned chunk=number(argv[3],4096),greedy=0;
    if (!chunk || chunk>4096) throw std::runtime_error("chunk must be 1..4096");
    auto policy=gewell::nvfp4::ActivationPolicy::always;
    std::string image_features,replay_calls;
    std::vector<unsigned> tokens;
    for (int i=4;i<argc;++i) {
      std::string arg=argv[i];
      if (arg=="--nvfp4-activation-policy") {
        if (++i==argc) throw std::runtime_error("missing activation policy");
        std::string value=argv[i];
        if (value=="always") policy=gewell::nvfp4::ActivationPolicy::always;
        else if (value=="prefill") policy=gewell::nvfp4::ActivationPolicy::prefill;
        else throw std::runtime_error("invalid activation policy");
        continue;
      }
      if (arg=="--image-features") {
        if (++i==argc) throw std::runtime_error("missing image features path");
        image_features=argv[i];continue;
      }
      if (arg=="--replay-calls") {
        if (++i==argc) throw std::runtime_error("missing replay calls path");
        replay_calls=argv[i];continue;
      }
      if (arg=="--greedy") {
        if (++i==argc) throw std::runtime_error("missing greedy count");
        greedy=number(argv[i],m::kMaxPositions);continue;
      }
      unsigned token=number(arg,m::kVocabSize-1);
      if (token>=m::kVocabSize) throw std::runtime_error("invalid token");
      tokens.push_back(token);
    }
    auto prompt=tokens.size(),horizon=prompt+(greedy?greedy-1:0);
    if (!prompt || horizon>m::kMaxPositions) throw std::runtime_error("invalid horizon");
    if (!image_features.empty() && (prompt>1120 || prompt>chunk))
      throw std::runtime_error("image fixture must fit one complete segment");
    if (!fs::create_directory(output)) throw std::runtime_error("refusing existing output");
    cudaStream_t stream{};check_cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"stream");
    auto executor=std::make_unique<s::BatchedExecutor>(gewell::gemma4_26b_a4b::ArtifactFile::Open(argv[1]),chunk,stream,policy);
    auto config=kv::compact_pool_config(s::kCacheGeometry,4ULL<<30,0,64ULL<<20,kv::Format::bf16,kv::Format::bf16);
    config.gpu_bytes=config.local_ring_bytes+((horizon+255)/256)*config.global_page_bytes+(2ULL<<20);
    kv::CacheLedger ledger(config);
    gewell::runtime::PhysicalCache cache(s::kCacheGeometry,config,ledger,1024);
    auto execution=ledger.try_begin_batch(0,horizon);
    if (!execution) throw std::runtime_error("cache admission failed");
    cache.clear_page_table(ledger.execution(*execution));
    DeviceAllocation device_tokens(chunk*4),logits(std::size_t(std::min(chunk,FullLogitRows))*m::kVocabSize*2);
    std::unique_ptr<DeviceAllocation> image;
    if (!image_features.empty()) {
      std::vector<std::uint16_t> values(prompt*m::kHiddenSize);
      std::ifstream file(image_features,std::ios::binary);
      file.read(reinterpret_cast<char*>(values.data()),values.size()*2);
      if (!file || file.peek()!=EOF) throw std::runtime_error("image feature shape mismatch");
      image=std::make_unique<DeviceAllocation>(values.size()*2);
      check_cuda(cudaMemcpy(image->data(),values.data(),values.size()*2,cudaMemcpyHostToDevice),"image features");
    }
    std::ofstream calls(output/"calls.tsv");
    std::ifstream replay;
    if (!replay_calls.empty()) {
      replay.open(replay_calls);
      if (!replay) throw std::runtime_error("cannot read replay calls");
    }
    unsigned position=0,call=0;
    while (position<tokens.size()) {
      unsigned rows=std::min<std::size_t>(chunk,tokens.size()-position);
      auto plan=ledger.prepare_write(*execution,position,rows);
      if (!plan.copies.empty()) throw std::runtime_error("unexpected COW for private fixture");
      cache.upload_page_table(ledger.execution(*execution),gewell::runtime::CompletionContext(stream));
      s::Segment segment{};segment.position=position;segment.rows=rows;
      if (image && !position) segment.image_features=static_cast<const B*>(image->data());
      for (unsigned l=0;l<30;++l) segment.cache[l]=cache.layer(*execution,l);
      check_cuda(cudaMemcpyAsync(device_tokens.data(),tokens.data()+position,rows*4,cudaMemcpyHostToDevice,stream),"tokens");
      auto directory=output/std::to_string(call);fs::create_directory(directory);
      auto capture=[&](const std::string& name,const std::vector<std::uint16_t>& values) {
        std::ofstream file(directory/(name+".bf16"),std::ios::binary);
        file.write(reinterpret_cast<const char*>(values.data()),values.size()*2);
        if (!file) throw std::runtime_error("capture write");
      };
      const auto* hidden=executor->forward(static_cast<unsigned*>(device_tokens.data()),{segment},position<prompt?gewell::nvfp4::Phase::prefill:gewell::nvfp4::Phase::decode,capture);
      unsigned head_rows=rows<=FullLogitRows?rows:1;
      executor->head(hidden+(rows-head_rows)*m::kHiddenSize,head_rows,static_cast<B*>(logits.data()));
      std::vector<std::uint16_t> result(std::size_t(head_rows)*m::kVocabSize);
      check_cuda(cudaMemcpyAsync(result.data(),logits.data(),result.size()*2,cudaMemcpyDeviceToHost,stream),"logits");
      check_cuda(cudaStreamSynchronize(stream),"forward completion");
      capture(head_rows==rows?"logits":"last_logits",result);
      float maximum=-INFINITY;unsigned best=0;
      for (unsigned i=0;i<m::kVocabSize;++i) {
        std::uint32_t bits=std::uint32_t(result[(head_rows-1)*m::kVocabSize+i])<<16;
        float x;std::memcpy(&x,&bits,4);
        if (x>maximum) { maximum=x;best=i; }
      }
      calls<<call<<'\t'<<position<<'\t'<<rows<<'\t'<<best<<'\n';
      unsigned next=best;
      if (replay.is_open()) {
        unsigned expected_call,expected_position,expected_rows;
        if (!(replay>>expected_call>>expected_position>>expected_rows>>next) ||
            expected_call!=call || expected_position!=position || expected_rows!=rows || next>=m::kVocabSize)
          throw std::runtime_error("replay call schedule mismatch");
      }
      position+=rows;++call;
      std::cout<<"position="<<position<<" rows="<<rows<<" argmax="<<best<<std::endl;
      if (position==tokens.size() && tokens.size()<horizon) tokens.push_back(next);
    }
    if (replay.is_open() && (replay>>std::ws).peek()!=EOF)
      throw std::runtime_error("unused replay calls");
    std::cout<<"weights="<<executor->weight_bytes()<<" scratch="<<executor->scratch_bytes()<<std::endl;
    ledger.release_execution(*execution);
    executor.reset();
    check_cuda(cudaStreamDestroy(stream),"destroy stream");
    return 0;
  } catch (const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
