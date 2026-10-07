#include "mtp_target.cuh"
#include "cuda_memory.cuh"
#include <algorithm>
#include <stdexcept>

namespace gewell::gemma4_26b_a4b::sm120 {
namespace {
using B=__nv_bfloat16;
class Target final : public mtp_cycle::BatchTarget {
 public:
  Target(BatchedExecutor& executor,const std::array<const B*,48>& assistant,
         unsigned capacity,cudaStream_t stream)
      : executor_(executor),assistant_{executor.target_embedding(),assistant,executor.shared_global_key_norm()},
        capacity_(checked_capacity(capacity)),stream_(stream),
        hidden_(std::size_t(capacity_)*kHiddenSize*sizeof(B)),
        logits_(std::size_t(capacity_)*kVocabSize*sizeof(B)) {}
  mtp_assistant::Model model() const override { return mtp_assistant::Model::gemma4_26b_a4b; }
  mtp_assistant::Weights assistant_weights() const override { return assistant_; }
  std::size_t staging_bytes(std::uint32_t rows) const override { return mtp_staging_bytes(rows); }
  std::size_t scratch_bytes() const override { return hidden_.size()+logits_.size(); }
  void prepare(std::uint32_t rows) override {
    if (!rows || rows>capacity_) throw std::invalid_argument("26B MTP target rows exceed capacity");
  }
  void run_batch(const std::uint32_t* tokens,const std::vector<mtp_cycle::TargetInput>& inputs,
                 cudaStream_t stream) override {
    if (stream!=stream_) throw std::invalid_argument("26B MTP target stream differs from resident executor");
    std::vector<Segment> segments;unsigned rows=0;
    for (const auto& i:inputs) {
      if (i.caches.size()!=kLayerCount || i.rows>capacity_-rows)
        throw std::invalid_argument("26B MTP target cache/row geometry");
      Segment segment{i.base_position,i.rows,{},{static_cast<B*>(i.staging),i.staging_size,i.staging_capacity_rows}};
      validate_mtp_staging(segment.staging,i.rows);
      std::copy(i.caches.begin(),i.caches.end(),segment.cache.begin());
      for (const auto& capture:i.captures) segment.captures.push_back({capture.completed_layers,capture.output});
      segments.push_back(std::move(segment));rows+=i.rows;
    }
    prepare(rows);
    const auto* hidden=executor_.forward(tokens,segments,nvfp4::Phase::decode);
    executor_.head(hidden,rows,static_cast<B*>(logits_.data()));
    // Preserve the cycle's latest target rows independently of ordinary work
    // that subsequently reuses the resident executor's per-layer scratch.
    cuda_detail::check_cuda(cudaMemcpyAsync(hidden_.data(),hidden,std::size_t(rows)*kHiddenSize*sizeof(B),
        cudaMemcpyDeviceToDevice,stream),"preserve26B MTP hidden rows");
  }
  void commit_batch(const std::vector<mtp_cycle::TargetCommit>& inputs,cudaStream_t stream) override {
    if (stream!=stream_) throw std::invalid_argument("26B MTP commit stream differs from resident executor");
    std::vector<MtpCommit> commits;
    for (const auto& i:inputs) {
      if (!i.caches || i.caches->size()!=kLayerCount)
        throw std::invalid_argument("26B MTP commit cache geometry");
      MtpCommit commit{{},{const_cast<B*>(static_cast<const B*>(i.staging)),i.staging_size,i.capacity_rows},
          i.base_position,i.source_rows,i.commit_rows};
      std::copy(i.caches->begin(),i.caches->end(),commit.cache.begin());commits.push_back(commit);
    }
    commit_mtp_rows(commits,stream);
  }
  const B* logits() const override { return static_cast<const B*>(logits_.data()); }
  const B* hidden() const override { return static_cast<const B*>(hidden_.data()); }
 private:
  static unsigned checked_capacity(unsigned capacity) {
    if (!capacity || capacity>1280) throw std::invalid_argument("26B MTP capacity must be1..1280");
    return capacity;
  }
  BatchedExecutor& executor_;
  mtp_assistant::Weights assistant_;
  unsigned capacity_;
  cudaStream_t stream_;
  cuda_detail::DeviceAllocation hidden_,logits_;
};
}  // namespace
std::unique_ptr<mtp_cycle::BatchTarget> make_mtp_target(BatchedExecutor& executor,
    const std::array<const B*,48>& assistant,unsigned capacity_rows,cudaStream_t stream) {
  return std::make_unique<Target>(executor,assistant,capacity_rows,stream);
}
}  // namespace gewell::gemma4_26b_a4b::sm120
