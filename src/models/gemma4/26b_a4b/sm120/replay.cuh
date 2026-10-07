#pragma once
#include "batched_executor.cuh"
#include "cuda_memory.cuh"
#include "gewell/models/gemma4/26b_a4b/cache_config.h"
#include "runtime/physical_cache.h"
#include <stdexcept>
#include <optional>

namespace gewell::gemma4_26b_a4b::sm120 {
// One private teacher-forced history. The caller retains the resident executor
// across requests and finishes stream work before destroying this cache.
class ReplayCache {
 public:
  ReplayCache(BatchedExecutor& executor,unsigned horizon,unsigned max_rows,cudaStream_t stream)
      :executor_(executor),config_(config(horizon)),ledger_(config_),
       cache_(kCacheGeometry,config_,ledger_,1024),tokens_(std::size_t(max_rows)*4),
       stream_(stream),capacity_(max_rows) {
    execution_=ledger_.try_begin_batch(0,horizon);
    if(!execution_) throw std::runtime_error("26B replay cache admission failed");
    cache_.clear_page_table(ledger_.execution(*execution_));
  }
  ~ReplayCache() { if(execution_) ledger_.release_execution(*execution_); }
  void forward(const std::uint32_t* tokens,unsigned position,unsigned rows,nvfp4::Phase phase) {
    if(!rows || rows>capacity_) throw std::invalid_argument("26B replay rows exceed capacity");
    const auto plan=ledger_.prepare_write(*execution_,position,rows);
    if(!plan.copies.empty()) throw std::logic_error("private26B replay unexpectedly requires COW");
    cache_.upload_page_table(ledger_.execution(*execution_),runtime::CompletionContext(stream_));
    Segment segment{};segment.position=position;segment.rows=rows;
    for(unsigned l=0;l<kLayerCount;++l) segment.cache[l]=cache_.layer(*execution_,l);
    cuda_detail::check_cuda(cudaMemcpyAsync(tokens_.data(),tokens,rows*4,cudaMemcpyHostToDevice,stream_),"copy26B replay tokens");
    hidden_=executor_.forward(static_cast<const std::uint32_t*>(tokens_.data()),{segment},phase);
    rows_=rows;
  }
  void head(unsigned offset,unsigned rows,__nv_bfloat16* logits) {
    if(!hidden_ || !rows || offset>rows_ || rows>rows_-offset)
      throw std::invalid_argument("26B replay head exceeds forward rows");
    executor_.head(hidden_+std::size_t(offset)*kHiddenSize,rows,logits);
  }
 private:
  static kv_cache::PoolConfig config(unsigned horizon) {
    if(!horizon || horizon>kMaxPositions) throw std::invalid_argument("26B replay horizon");
    auto result=kv_cache::compact_pool_config(kCacheGeometry,4ULL<<30,0,64ULL<<20,
                                            kv_cache::Format::bf16,kv_cache::Format::bf16);
    result.gpu_bytes=result.local_ring_bytes+((horizon+255)/256)*result.global_page_bytes+(2ULL<<20);
    return result;
  }
  BatchedExecutor& executor_;
  kv_cache::PoolConfig config_;
  kv_cache::CacheLedger ledger_;
  runtime::PhysicalCache cache_;
  cuda_detail::DeviceAllocation tokens_;
  const cudaStream_t stream_;
  const unsigned capacity_;
  std::optional<kv_cache::ExecutionId> execution_;
  const __nv_bfloat16* hidden_{};
  unsigned rows_{};
};
}
