#include "models/gemma4/26b_a4b/sm120/batched_executor.cuh"
#include "gewell/models/gemma4/26b_a4b/cache_config.h"
#include "runtime/physical_cache.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

namespace s=gewell::gemma4_26b_a4b::sm120;
namespace kv=gewell::kv_cache;
using B=__nv_bfloat16;
using gewell::cuda_detail::DeviceAllocation;
using gewell::cuda_detail::check_cuda;
struct Stream {
  cudaStream_t value{};
  Stream() { check_cuda(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking),"stream"); }
  ~Stream() { cudaStreamDestroy(value); }
};
struct Output {
  std::vector<std::uint16_t> hidden,logits;
};

int main(int argc,char** argv) {
  try {
    if (argc!=2) throw std::runtime_error("usage: packed ARTIFACT");
    Stream stream;
    s::BatchedExecutor executor(gewell::gemma4_26b_a4b::ArtifactFile::Open(argv[1]),4096,stream.value,gewell::nvfp4::ActivationPolicy::always);
    auto config=kv::compact_pool_config(s::kCacheGeometry,1ULL<<30,0,64ULL<<20,kv::Format::bf16,kv::Format::bf16);
    kv::CacheLedger ledger(config);
    gewell::runtime::PhysicalCache cache(s::kCacheGeometry,config,ledger,1024);
    std::vector<kv::ExecutionId> executions;
    for (unsigned i=0;i<4;++i) {
      auto id=ledger.try_begin_batch(0,8);
      if (!id) throw std::runtime_error("packed fixture admission failed");
      executions.push_back(*id);cache.clear_page_table(ledger.execution(*id));
    }
    DeviceAllocation tokens(4096*4),logits(std::size_t(5)*262144*2);
    auto run=[&](const std::vector<unsigned>& order,const std::vector<std::vector<unsigned>>& chunks,
                 bool endpoints=false) {
      std::vector<unsigned> input;
      std::vector<s::Segment> segments;
      for (unsigned i=0;i<order.size();++i) {
        auto id=executions[order[i]];s::Segment segment{};
        segment.position=ledger.execution(id).processed_tokens;segment.rows=chunks[i].size();
        auto plan=ledger.prepare_write(id,segment.position,segment.rows);
        if (!plan.copies.empty()) throw std::runtime_error("unexpected fixture COW");
        cache.upload_page_table(ledger.execution(id),gewell::runtime::CompletionContext(stream.value));
        for (unsigned l=0;l<30;++l) segment.cache[l]=cache.layer(id,l);
        segments.push_back(segment);input.insert(input.end(),chunks[i].begin(),chunks[i].end());
      }
      check_cuda(cudaMemcpyAsync(tokens.data(),input.data(),input.size()*4,cudaMemcpyHostToDevice,stream.value),"tokens");
      auto hidden=executor.forward(static_cast<const unsigned*>(tokens.data()),segments,gewell::nvfp4::Phase::prefill);
      unsigned head_rows=endpoints?segments.size():input.size();
      if (head_rows>5) throw std::runtime_error("fixture head buffer exceeded");
      if (endpoints) {
        unsigned end=0;
        for (unsigned i=0;i<segments.size();++i) {
          end+=segments[i].rows;
          executor.head(hidden+std::size_t(end-1)*2816,1,static_cast<B*>(logits.data())+std::size_t(i)*262144);
        }
      } else executor.head(hidden,input.size(),static_cast<B*>(logits.data()));
      Output out;out.hidden.resize(input.size()*2816);out.logits.resize(std::size_t(head_rows)*262144);
      check_cuda(cudaMemcpyAsync(out.hidden.data(),hidden,out.hidden.size()*2,cudaMemcpyDeviceToHost,stream.value),"hidden");
      check_cuda(cudaMemcpyAsync(out.logits.data(),logits.data(),out.logits.size()*2,cudaMemcpyDeviceToHost,stream.value),"logits");
      check_cuda(cudaStreamSynchronize(stream.value),"packed completion");
      return out;
    };
    auto compare=[&](const Output& original,const Output& reversed,unsigned first,unsigned second) {
      for (unsigned width:{2816u,262144u}) {
        const auto& a=width==2816?original.hidden:original.logits;
        const auto& b=width==2816?reversed.hidden:reversed.logits;
        for (unsigned row=0;row<first+second;++row) {
          unsigned mapped=row<first?second+row:row-first;
          if (!std::equal(a.begin()+std::size_t(row)*width,a.begin()+std::size_t(row+1)*width,
                          b.begin()+std::size_t(mapped)*width))
            throw std::runtime_error("packed permutation mismatch: row="+std::to_string(row)+" width="+std::to_string(width));
        }
      }
    };
    // The two requests have different contents, lengths and subsequent positions.
    // A second pair has independent physical allocations and reversed row order.
    auto cold=run({0,1},{{2,105,2364},{2,9259}});
    auto cold_reverse=run({3,2},{{2,9259},{2,105,2364}});
    compare(cold,cold_reverse,3,2);
    std::cout<<"cold heterogeneous packed permutation: hidden and full logits exact\n";
    auto cached=run({0,1},{{107},{2364,107,236761}});
    auto cached_reverse=run({3,2},{{2364,107,236761},{107}});
    compare(cached,cached_reverse,1,3);
    std::cout<<"cached heterogeneous positions 3/2: hidden and full logits exact\n";
    // Continue just one request, after the other request changed its history.
    auto single=run({0},{{9259}});
    auto single_copy=run({2},{{9259}});
    if (single.hidden!=single_copy.hidden || single.logits!=single_copy.logits)
      throw std::runtime_error("request isolation mismatch");
    std::cout<<"single-request continuation after packed work: hidden and logits exact\n";
    for (auto id:executions) ledger.release_execution(id);
    executions.clear();
    for (unsigned i=0;i<4;++i) {
      auto id=ledger.try_begin_batch(0,2082);
      if (!id) throw std::runtime_error("wide packed fixture admission failed");
      executions.push_back(*id);cache.clear_page_table(ledger.execution(*id));
    }
    auto pattern=[](unsigned length,bool second) {
      const std::vector<unsigned> phrase=second?std::vector<unsigned>{9259,236761,105,2364,107}:
          std::vector<unsigned>{818,3823,8864,37423,38167,1024,506,31770,4799,236761};
      std::vector<unsigned> result(length);
      for (unsigned i=0;i<length;++i) result[i]=phrase[i%phrase.size()];
      result[0]=2;return result;
    };
    auto compare_endpoints=[&](const Output& a,const Output& b,unsigned first,unsigned second) {
      for (unsigned row=0;row<first+second;++row) {
        unsigned mapped=row<first?second+row:row-first;
        if (!std::equal(a.hidden.begin()+std::size_t(row)*2816,a.hidden.begin()+std::size_t(row+1)*2816,
                        b.hidden.begin()+std::size_t(mapped)*2816))
          throw std::runtime_error("wide packed hidden permutation mismatch");
      }
      for (unsigned i=0;i<2;++i)
        if (!std::equal(a.logits.begin()+std::size_t(i)*262144,a.logits.begin()+std::size_t(i+1)*262144,
                        b.logits.begin()+std::size_t(1-i)*262144))
          throw std::runtime_error("wide packed endpoint logits mismatch");
    };
    auto first=pattern(2049,false),second=pattern(2047,true);
    auto wide=run({0,1},{first,second},true);
    auto wide_reverse=run({3,2},{second,first},true);
    compare_endpoints(wide,wide_reverse,2049,2047);
    const auto memory=ledger.gpu_memory_stats();
    const auto stats=ledger.stats();
    const auto page_bytes=34*config.global_page_bytes;
    const auto execution_bytes=4*(config.local_ring_bytes+config.page_table_bytes);
    if (stats.execution_count!=4 || stats.page_count!=34 ||
        memory.global_page_bytes!=page_bytes || memory.execution_buffer_bytes!=execution_bytes ||
        stats.gpu.used!=page_bytes+execution_bytes || memory.other_bytes || memory.shared_page_bytes ||
        memory.checkpoint_state_bytes || memory.pending_capture_bytes)
      throw std::runtime_error("wide packed KV accounting mismatch");
    std::cout<<"4096-row packed cap, heterogeneous2049/2047: all hidden rows and endpoint logits exact\n";
    std::cout<<"wide packed KV bytes="<<stats.gpu.used<<" pages="<<stats.page_count
             <<" weights="<<executor.weight_bytes()<<" scratch="<<executor.scratch_bytes()<<'\n';
    auto continuation=pattern(32,false);
    auto wide_cached=run({0,1},{continuation,{107}},true);
    auto wide_cached_reverse=run({3,2},{{107},continuation},true);
    compare_endpoints(wide_cached,wide_cached_reverse,32,1);
    std::cout<<"33-row packed TensorCore continuation at positions2049/2047: hidden and endpoint logits exact\n";
    auto wide_single=run({0},{{9259}});
    auto wide_single_copy=run({2},{{9259}});
    if (wide_single.hidden!=wide_single_copy.hidden || wide_single.logits!=wide_single_copy.logits)
      throw std::runtime_error("wide packed request isolation mismatch");
    std::cout<<"single-request continuation after wide packed work: hidden and logits exact\n";
    for (auto id:executions) ledger.release_execution(id);
    const auto empty=ledger.stats();
    if (empty.gpu.used || empty.execution_count || empty.page_count)
      throw std::runtime_error("wide packed cache cleanup mismatch");
    return 0;
  } catch (const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
