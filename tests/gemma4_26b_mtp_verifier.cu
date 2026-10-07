#include "models/gemma4/26b_a4b/sm120/batched_executor.cuh"
#include "models/gemma4/26b_a4b/sm120/kv_write.cuh"
#include "cuda_memory.cuh"
#include "models/gemma4/26b_a4b/sm120/mtp_target.cuh"
#include "gewell/models/gemma4/26b_a4b/component_weights.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>

namespace m=gewell::gemma4_26b_a4b;
namespace s=m::sm120;
namespace kv=gewell::kv_cache;
using B=__nv_bfloat16;
using D=gewell::cuda_detail::DeviceAllocation;
using gewell::cuda_detail::check_cuda;
void require(bool ok,const char* what) { if (!ok) throw std::runtime_error(what); }

// Test-owned allocations, reversed global pages and unused padding. Expected
// commit images are assembled on the CPU from the ordinary forward's cache,
// without using staging offsets or the production cache writer.
struct Cache {
  unsigned pages,local_record,global_record;
  std::size_t local_layer,global_layer,page_bytes;
  D local,global,table;
  std::array<kv::DeviceView,30> views{};
  Cache(unsigned extent,kv::Format format):pages((extent+255)/256),
      local_record(format==kv::Format::bf16?512:272),global_record(format==kv::Format::bf16?1280:656),
      local_layer(std::size_t(2)*8*1024*local_record),global_layer(std::size_t(2)*256*global_record),
      page_bytes(5*global_layer+256),local(25*local_layer),global(pages*page_bytes),table(pages*8) {
    check_cuda(cudaMemset(local.data(),0xa5,local.size()),"fill local");
    check_cuda(cudaMemset(global.data(),0xa5,global.size()),"fill global");
    std::vector<std::uint64_t> offsets(pages);
    for (unsigned p=0;p<pages;++p) offsets[p]=(pages-1-p)*page_bytes/2;
    check_cuda(cudaMemcpy(table.data(),offsets.data(),table.size(),cudaMemcpyHostToDevice),"page offsets");
    unsigned li=0,gi=0;
    for (unsigned l=0;l<30;++l) {
      auto& v=views[l];v.format=format;
      if (m::is_global_layer(l)) {
        v.page_pool=static_cast<B*>(global.data());v.page_offsets=static_cast<std::uint64_t*>(table.data());
        v.page_count=pages;v.page_tokens=256;v.page_stride_elements=page_bytes/2;
        v.layer_offset_elements=64+gi++*global_layer/2;
      } else {
        v.key=reinterpret_cast<B*>(static_cast<char*>(local.data())+li++*local_layer);
        v.value=reinterpret_cast<B*>(reinterpret_cast<char*>(v.key)+local_layer/2);v.capacity=1024;
      }
    }
  }
  using Image=std::pair<std::vector<unsigned char>,std::vector<unsigned char>>;
  Image snapshot(cudaStream_t stream) {
    Image result{{},{}};result.first.resize(local.size());result.second.resize(global.size());
    check_cuda(cudaMemcpyAsync(result.first.data(),local.data(),local.size(),cudaMemcpyDeviceToHost,stream),"snapshot local");
    check_cuda(cudaMemcpyAsync(result.second.data(),global.data(),global.size(),cudaMemcpyDeviceToHost,stream),"snapshot global");
    check_cuda(cudaStreamSynchronize(stream),"snapshot complete");return result;
  }
  void restore(const Image& image,cudaStream_t stream) {
    check_cuda(cudaMemcpyAsync(local.data(),image.first.data(),local.size(),cudaMemcpyHostToDevice,stream),"restore local");
    check_cuda(cudaMemcpyAsync(global.data(),image.second.data(),global.size(),cudaMemcpyHostToDevice,stream),"restore global");
  }
  Image accepted(const Image& before,const Image& full,unsigned position,unsigned rows) {
    auto result=before;
    for (unsigned t=rows>1024?rows-1024:0;t<rows;++t)
      for (unsigned l=0;l<25;++l) for (unsigned kv=0;kv<2;++kv) for (unsigned h=0;h<8;++h) {
        auto offset=l*local_layer+kv*local_layer/2+(h*1024+(position+t)%1024)*local_record;
        std::copy_n(full.first.data()+offset,local_record,result.first.data()+offset);
      }
    for (unsigned t=0;t<rows;++t) for (unsigned l=0;l<5;++l) for (unsigned h=0;h<2;++h) {
      auto p=position+t;
      auto offset=(pages-1-p/256)*page_bytes+128+l*global_layer+(h*256+p%256)*global_record;
      std::copy_n(full.second.data()+offset,global_record,result.second.data()+offset);
    }
    return result;
  }
};
std::vector<unsigned> tokens(unsigned rows,unsigned salt=0) {
  const unsigned words[]={2,9259,236761,105,2364,107};std::vector<unsigned> out(rows);
  for (unsigned i=0;i<rows;++i) out[i]=words[(i+salt)%6];return out;
}
struct Output { std::vector<B> hidden,logits; };
Output run(s::BatchedExecutor& executor,D& ids,const std::vector<unsigned>& input,
           const std::vector<s::Segment>& segments,cudaStream_t stream,bool head=true,
           gewell::nvfp4::Phase phase=gewell::nvfp4::Phase::decode) {
  check_cuda(cudaMemcpyAsync(ids.data(),input.data(),input.size()*4,cudaMemcpyHostToDevice,stream),"tokens");
  auto hidden=executor.forward(static_cast<unsigned*>(ids.data()),segments,phase);
  Output out;out.hidden.resize(input.size()*2816);
  check_cuda(cudaMemcpyAsync(out.hidden.data(),hidden,out.hidden.size()*2,cudaMemcpyDeviceToHost,stream),"hidden");
  if (head) {
    D logits(input.size()*262144*2);
    executor.head(hidden,input.size(),static_cast<B*>(logits.data()));out.logits.resize(input.size()*262144);
    check_cuda(cudaMemcpyAsync(out.logits.data(),logits.data(),logits.size(),cudaMemcpyDeviceToHost,stream),"logits");
    check_cuda(cudaStreamSynchronize(stream),"head completion");
  }
  check_cuda(cudaStreamSynchronize(stream),"forward completion");return out;
}
bool same(const std::vector<B>& a,const std::vector<B>& b) {
  return a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*2);
}
void scenario(s::BatchedExecutor& executor,D& ids,cudaStream_t stream,kv::Format format,
              std::vector<unsigned> positions,std::vector<unsigned> rows,std::vector<unsigned> capacities) {
  const unsigned n=rows.size();
  std::vector<std::unique_ptr<Cache>> caches;
  std::vector<std::unique_ptr<D>> staging;
  std::vector<Cache::Image> before,full;
  std::vector<s::Segment> segments;
  std::vector<unsigned> input;
  for (unsigned i=0;i<n;++i) {
    caches.push_back(std::make_unique<Cache>(positions[i]+rows[i],format));
    auto& cache=*caches.back();
    if (positions[i]) run(executor,ids,tokens(positions[i],i),{{0,positions[i],cache.views}},stream,false,gewell::nvfp4::Phase::prefill);
    before.push_back(cache.snapshot(stream));
    staging.push_back(std::make_unique<D>(s::mtp_staging_bytes(capacities[i])));
    check_cuda(cudaMemset(staging.back()->data(),0xcd,staging.back()->size()),"poison staging");
    s::MtpStaging view{static_cast<B*>(staging.back()->data()),staging.back()->size(),capacities[i]};
    segments.push_back({positions[i],rows[i],cache.views,view});
    // Verification requires only committed pages, not the reserved future tail.
    for (unsigned l=5;l<30;l+=6)
      segments.back().cache[l].page_count=(positions[i]+255)/256;
    auto part=tokens(rows[i],i+1);input.insert(input.end(),part.begin(),part.end());
  }
  auto verified=run(executor,ids,input,segments,stream);
  for (unsigned i=0;i<n;++i) require(caches[i]->snapshot(stream)==before[i],"verification changed committed KV");
  auto ordinary=segments;
  for (unsigned i=0;i<n;++i) { ordinary[i].staging={};ordinary[i].cache=caches[i]->views; }
  auto baseline=run(executor,ids,input,ordinary,stream);
  require(same(verified.hidden,baseline.hidden) && same(verified.logits,baseline.logits),"verifier differs from ordinary full MoE execution");
  for (unsigned i=0;i<n;++i) full.push_back(caches[i]->snapshot(stream));
  for (unsigned selection=0;selection<4;++selection) {
    std::vector<s::MtpCommit> commits;
    for (unsigned i=0;i<n;++i) {
      caches[i]->restore(before[i],stream);
      unsigned accepted=selection==0?0:selection==1?1:selection==2?rows[i]/2:rows[i];
      // For a >1024-row ordinary chunk, early rows no longer exist in its ring.
      if (rows[i]>1024 && accepted && accepted<rows[i]) accepted=rows[i];
      commits.push_back({caches[i]->views,segments[i].staging,positions[i],rows[i],accepted});
    }
    s::commit_mtp_rows(commits,stream);
    for (unsigned i=0;i<n;++i)
      require(caches[i]->snapshot(stream)==caches[i]->accepted(before[i],full[i],positions[i],commits[i].accepted_rows),
          "accepted prefix cache differs from CPU expected image");
  }
  // A malformed later request must not partially commit an earlier request.
  for (unsigned i=0;i<n;++i) caches[i]->restore(before[i],stream);
  auto good=s::MtpCommit{caches[0]->views,segments[0].staging,positions[0],rows[0],1};
  auto bad=good;bad.staging.bytes--;
  bool rejected=false;try { s::commit_mtp_rows({good,bad},stream); } catch (const std::invalid_argument&) { rejected=true; }
  require(rejected,"short staging accepted");require(caches[0]->snapshot(stream)==before[0],"invalid batch partially committed");
  bad=good;bad.cache[29].page_count=0;rejected=false;
  try { s::commit_mtp_rows({good,bad},stream); } catch (const std::invalid_argument&) { rejected=true; }
  require(rejected,"unreserved commit pages accepted");require(caches[0]->snapshot(stream)==before[0],"invalid cache batch partially committed");
  // Capacity and active row count are separate from accepted count.
  bad=good;bad.accepted_rows=bad.source_rows+1;rejected=false;
  try { s::commit_mtp_rows({good,bad},stream); } catch (const std::invalid_argument&) { rejected=true; }
  require(rejected,"oversized acceptance accepted");require(caches[0]->snapshot(stream)==before[0],"invalid acceptance changed cache");
  std::cout<<"format="<<(format==kv::Format::bf16?"BF16":"FP8")<<" positions=";
  for (auto p:positions) std::cout<<p<<',';
  std::cout<<" verifier_hidden_logits=exact committed_unchanged=exact accepted_prefixes=exact malformed_batch=atomic\n";
}
// Maximum-depth partial commits need their own control: ordinary full forward
// overwrites the earliest rows of a 1280-row chunk in its local ring. Keep
// independent patterned source K/V instead, and compare with direct writers.
void primitive_commits(cudaStream_t stream) {
  constexpr unsigned rows=1280,position=1023;
  D staged(s::mtp_staging_bytes(rows));
  s::MtpStaging view{static_cast<B*>(staged.data()),staged.size(),rows};
  std::vector<std::unique_ptr<D>> keys,values;
  for (unsigned l=0;l<30;++l) {
    const unsigned width=m::is_global_layer(l)?1024:2048;
    std::vector<B> key(rows*width),value(key.size());
    for (unsigned i=0;i<key.size();++i) {
      key[i]=B((int((i*17+l*31)%997)-498)*0.015625f);
      value[i]=B((int((i*29+l*11)%991)-495)*0.001953125f);
    }
    keys.push_back(std::make_unique<D>(key.size()*2));values.push_back(std::make_unique<D>(value.size()*2));
    check_cuda(cudaMemcpyAsync(keys.back()->data(),key.data(),key.size()*2,cudaMemcpyHostToDevice,stream),"patterned K");
    check_cuda(cudaMemcpyAsync(values.back()->data(),value.data(),value.size()*2,cudaMemcpyHostToDevice,stream),"patterned V");
    s::stage_mtp_layer(view,l,rows,static_cast<B*>(keys.back()->data()),static_cast<B*>(values.back()->data()),stream);
    check_cuda(cudaStreamSynchronize(stream),"pattern source uploaded");
  }
  for (auto format:{kv::Format::bf16,kv::Format::fp8}) {
    Cache actual(position+rows,format),expected(position+rows,format);
    for (unsigned accepted:{0u,1u,255u,256u,1023u,1024u,1279u,1280u}) {
      check_cuda(cudaMemsetAsync(actual.local.data(),0xa5,actual.local.size(),stream),"clear actual local");
      check_cuda(cudaMemsetAsync(actual.global.data(),0xa5,actual.global.size(),stream),"clear actual global");
      check_cuda(cudaMemsetAsync(expected.local.data(),0xa5,expected.local.size(),stream),"clear expected local");
      check_cuda(cudaMemsetAsync(expected.global.data(),0xa5,expected.global.size(),stream),"clear expected global");
      s::commit_mtp_rows({{actual.views,view,position,rows,accepted}},stream);
      for (unsigned l=0;l<30;++l)
        s::write_cache(static_cast<B*>(keys[l]->data()),static_cast<B*>(values[l]->data()),
            expected.views[l],m::is_global_layer(l),position,accepted,stream);
      require(actual.snapshot(stream)==expected.snapshot(stream),"maximum-depth partial commit differs from independent source writer");
      std::cout<<"maxdepth format="<<(format==kv::Format::bf16?"BF16":"FP8")
               <<" accepted="<<accepted<<" exact\n";
    }
  }
}

void shared_cycle(const char* artifact,const char* assistant_path) {
  constexpr unsigned N=3,depth=3;
  cudaStream_t stream{};check_cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"cycle stream");
  cublasLtHandle_t handle{};require(cublasLtCreate(&handle)==CUBLAS_STATUS_SUCCESS,"cycle handle");
  {
    s::BatchedExecutor executor(m::ArtifactFile::Open(artifact),1280,stream,gewell::nvfp4::ActivationPolicy::always);
    gewell::component::File component(assistant_path,m::assistant_tensor_specs());
    D assistant(component.device_bytes());std::array<const B*,48> weights{};std::size_t offset=0;
    for (const auto& tensor:component.tensors()) {
      auto* destination=static_cast<char*>(assistant.data())+offset;
      check_cuda(cudaMemcpyAsync(destination,tensor.data,tensor.bytes,cudaMemcpyHostToDevice,stream),"load assistant component");
      weights[tensor.physical_id]=reinterpret_cast<B*>(destination);offset+=(tensor.bytes+4095)/4096*4096;
    }
    D ids(1280*4),staging(N*s::mtp_staging_bytes(depth+1)),terminal(N*2816*2),one_logits(262144*2),logprobs(4*sizeof(gewell::TokenLogprobs));
    gewell::mtp_cycle::Batch cycle(handle,s::make_mtp_target(executor,weights,N*(depth+1),stream),
        2048,N,depth,staging.data(),staging.size());
    std::vector<std::unique_ptr<Cache>> caches;
    std::vector<Cache::Image> before;
    std::vector<gewell::mtp_cycle::BatchInput> inputs;
    std::vector<gewell::MtpCaptureFeatures> captures(N);
    std::vector<gewell::MtpTargetProbes> probes(N);
    const unsigned positions[N]={6,257,1031},depths[N]={0,1,3};
    for (unsigned i=0;i<N;++i) {
      caches.push_back(std::make_unique<Cache>(positions[i]+4,kv::Format::bf16));
      auto output=run(executor,ids,tokens(positions[i],i),{{0,positions[i],caches[i]->views}},stream,false,gewell::nvfp4::Phase::prefill);
      auto* saved=static_cast<B*>(terminal.data())+i*2816;
      check_cuda(cudaMemcpyAsync(saved,output.hidden.data()+(positions[i]-1)*2816,2816*2,cudaMemcpyHostToDevice,stream),"save target hidden");
      executor.head(saved,1,static_cast<B*>(one_logits.data()));std::vector<B> logits(262144);
      check_cuda(cudaMemcpyAsync(logits.data(),one_logits.data(),logits.size()*2,cudaMemcpyDeviceToHost,stream),"pending logits");
      check_cuda(cudaStreamSynchronize(stream),"pending completion");
      auto pending=std::max_element(logits.begin(),logits.end(),[](B a,B b){return float(a)<float(b);})-logits.begin();
      gewell::mtp_cycle::BatchInput input;input.pending_token=pending;input.target_hidden=saved;
      input.position=positions[i];input.depth=depths[i];input.caches.assign(caches[i]->views.begin(),caches[i]->views.end());
      input.uniforms.assign(2*depths[i]+1,0.37f);input.capture=depths[i]?&captures[i]:nullptr;
      probes[i].layers={2,6,12,20,28};input.capture_next=&probes[i];inputs.push_back(input);
      before.push_back(caches[i]->snapshot(stream));
    }
    for (unsigned mode=0;mode<3;++mode) {
      unsigned forced=0;
      for (unsigned i=0;i<N;++i) {
        caches[i]->restore(before[i],stream);
        inputs[i].temperature=mode?0.8f:0.f;inputs[i].top_p=mode?0.9f:1.f;inputs[i].top_k=i==1?32:0;
        inputs[i].constraint_mask={};
      }
      if (mode) inputs[2].constraint_mask=[&](const unsigned* drafts,unsigned,unsigned* mask) {
        forced=drafts[0]==9259?107:9259;
        std::fill_n(mask,gewell::mtp_sampling::mask_words(262144),0u);mask[forced/32]|=1u<<(forced%32);
      };
      if (mode==2) inputs[1].constraint_mask=[](const unsigned*,unsigned,unsigned* mask) {
        std::fill_n(mask,gewell::mtp_sampling::mask_words(262144),0u);
      };
      auto result=cycle.run(inputs,stream);auto repeat=cycle.run(inputs,stream);
      for (unsigned i=0;i<N;++i) {
        require(result.requests[i].tokens==repeat.requests[i].tokens &&
            result.requests[i].status==repeat.requests[i].status,"shared cycle is not repeatable");
        require(caches[i]->snapshot(stream)==before[i],"cycle changed committed KV before commit");
        require(probes[i].width==2816 && probes[i].hidden.size()==5*2816,"26B probe geometry");
        if (depths[i]) require(captures[i].target_width==2816 && captures[i].assistant_width==1024 &&
            captures[i].assistant_hidden.size()==depths[i]*1024,"26B capture geometry");
      }
      if (mode) require(result.requests[2].verification.accepted_drafts==0 &&
          result.requests[2].tokens==std::vector<unsigned>{forced},"constrained residual correction failed");
      if (mode==2) require(result.requests[1].status!=gewell::mtp_sampling::Status::success &&
          result.requests[1].tokens.empty(),"empty constrained distribution did not stay request-local");
      std::vector<s::Segment> ordinary;std::vector<unsigned> full_tokens;
      std::vector<std::unique_ptr<D>> probe_rows;
      for (unsigned i=0;i<N;++i) {
        s::Segment segment{positions[i],depths[i]+1,caches[i]->views};
        probe_rows.push_back(std::make_unique<D>(5*(depths[i]+1)*2816*2));
        for (unsigned j=0;j<5;++j) segment.captures.push_back({probes[i].layers[j],
            static_cast<B*>(probe_rows.back()->data())+j*(depths[i]+1)*2816});
        ordinary.push_back(segment);full_tokens.push_back(inputs[i].pending_token);
        full_tokens.insert(full_tokens.end(),captures[i].draft_tokens.begin(),captures[i].draft_tokens.end());
      }
      auto reference=run(executor,ids,full_tokens,ordinary,stream);
      std::vector<Cache::Image> full;std::vector<gewell::mtp_cycle::BatchCommitInput> commits;
      unsigned row_offset=0;
      for (unsigned i=0;i<N;++i) {
        const unsigned count=result.requests[i].tokens.size();
        std::vector<B> hidden((depths[i]+1)*2816),logits((depths[i]+1)*262144);
        check_cuda(cudaMemcpyAsync(hidden.data(),cycle.hidden(i),hidden.size()*2,cudaMemcpyDeviceToHost,stream),"cycle hidden");
        check_cuda(cudaMemcpyAsync(logits.data(),cycle.logits(i),logits.size()*2,cudaMemcpyDeviceToHost,stream),"cycle logits");
        check_cuda(cudaStreamSynchronize(stream),"cycle views");
        require(!std::memcmp(hidden.data(),reference.hidden.data()+row_offset*2816,hidden.size()*2) &&
            !std::memcmp(logits.data(),reference.logits.data()+std::size_t(row_offset)*262144,logits.size()*2),
            "cycle verifier differs from full ordinary target or views were overwritten");
        if (count) {
          if (!mode) for (unsigned row=0;row<count;++row) {
            auto begin=logits.begin()+std::size_t(row)*262144;
            unsigned best=std::max_element(begin,begin+262144,[](B a,B b){return float(a)<float(b);})-begin;
            require(result.requests[i].tokens[row]==best,"greedy output differs from target winner");
          }
          std::vector<unsigned short> probe(5*(depths[i]+1)*2816);
          check_cuda(cudaMemcpyAsync(probe.data(),probe_rows[i]->data(),probe.size()*2,cudaMemcpyDeviceToHost,stream),"probe rows");
          cycle.summarize_logprobs(i,count,3,static_cast<gewell::TokenLogprobs*>(logprobs.data()),stream);
          std::vector<gewell::TokenLogprobs> scores(count);
          check_cuda(cudaMemcpyAsync(scores.data(),logprobs.data(),scores.size()*sizeof(scores[0]),cudaMemcpyDeviceToHost,stream),"logprobs");
          check_cuda(cudaStreamSynchronize(stream),"cycle diagnostics");
          for (const auto& score:scores) require(std::isfinite(score.logprob) && score.count<=3,"invalid target logprob summary");
          for (unsigned j=0;j<5;++j)
            require(std::equal(probes[i].hidden.begin()+j*2816,probes[i].hidden.begin()+(j+1)*2816,
                probe.begin()+(j*(depths[i]+1)+count-1)*2816),"selected target probe row mismatch");
          commits.push_back({i,&inputs[i].caches,count});
        }
        full.push_back(caches[i]->snapshot(stream));caches[i]->restore(before[i],stream);row_offset+=depths[i]+1;
      }
      cycle.commit_batch(commits,stream);
      for (unsigned i=0;i<N;++i)
        require(caches[i]->snapshot(stream)==caches[i]->accepted(before[i],full[i],positions[i],result.requests[i].tokens.size()),
            "cycle accepted-prefix commit mismatch");
      bool rejected=false;try {cycle.commit_batch(commits,stream);} catch(const std::invalid_argument&){rejected=true;}
      require(rejected,"cycle allowed duplicate commit");
      std::cout<<"cycle mode="<<mode<<" depths=0,1,3 repeatability=exact target=exact capture=exact readonly=exact commit=exact\n";
    }
  }
  cublasLtDestroy(handle);check_cuda(cudaStreamDestroy(stream),"cycle stream destroy");
}

int main(int argc,char** argv) {
  try {
    if (argc==4 && std::string(argv[1])=="--cycle") {shared_cycle(argv[2],argv[3]);return 0;}
    require(argc>=2 && argc<=4,"usage: verifier ARTIFACT [always|prefill] [wide] | --primitive");
    if (argc==2 && std::string(argv[1])=="--primitive") {
      primitive_commits(nullptr);return 0;
    }
    auto policy=argc>=3 && std::string(argv[2])=="prefill"?gewell::nvfp4::ActivationPolicy::prefill:gewell::nvfp4::ActivationPolicy::always;
    cudaStream_t stream{};check_cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"stream");
    {
      s::BatchedExecutor executor(m::ArtifactFile::Open(argv[1]),1280,stream,policy);D ids(1280*4);
      for (auto format:{kv::Format::bf16,kv::Format::fp8}) {
        scenario(executor,ids,stream,format,{0,255},{5,3},{8,7});
        scenario(executor,ids,stream,format,{1023,1031},{5,33},{9,40});
      }
      if (argc==4) scenario(executor,ids,stream,kv::Format::bf16,{0},{1280},{1280});
      std::cout<<"staging_bytes_1280="<<s::mtp_staging_bytes(1280)<<" weights="<<executor.weight_bytes()<<" scratch="<<executor.scratch_bytes()<<'\n';
    }
    check_cuda(cudaStreamDestroy(stream),"destroy stream");return 0;
  } catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
