#include "prepared_image.h"

namespace gewell::gemma4_31b::sm120 {
namespace {
void reserve(std::unique_ptr<DeviceAllocation>& buffer, std::size_t bytes) {
  if (buffer && buffer->size() >= bytes) return;
  // cudaFree waits for prior uses. Release before growing so a new largest
  // image does not require both the old and new workspace simultaneously.
  buffer.reset();
  buffer = std::make_unique<DeviceAllocation>(bytes);
}
}

PreparedImage::PreparedImage(const WeightArena& weights)
    : tower_({weights.pointer(model::kVisionPatchProjectionPhysicalId),
              vision_executor::kVisionWeightSliceBytes}, 1) {}

void PreparedImage::prepare(const std::vector<std::uint8_t>& pixels,
    const std::vector<std::uint8_t>& positions, std::uint32_t padded_patch_rows,
    std::uint32_t soft_token_count, cudaStream_t stream) {
  tower_.set_soft_token_count(soft_token_count);
  reserve(pixels_, pixels.size());
  reserve(positions_, positions.size());
  reserve(features_, std::size_t(soft_token_count) * model::kHiddenSize * sizeof(BFloat16));
  reserve(scratch_, tower_.scratch_bytes());
  check_cuda(cudaMemcpyAsync(pixels_->data(), pixels.data(), pixels.size(),
                            cudaMemcpyHostToDevice, stream), "copy prepared pixels to device");
  check_cuda(cudaMemcpyAsync(positions_->data(), positions.data(), positions.size(),
                            cudaMemcpyHostToDevice, stream), "copy prepared positions to device");
  const vision_engine::PrefillRequest request{
      {static_cast<const float*>(pixels_->data()),
       static_cast<const std::int32_t*>(positions_->data()), padded_patch_rows, soft_token_count},
      features_->data(), soft_token_count};
  check_cuda(cudaEventRecord(begin_.get(), stream), "record prepared vision start");
  tower_.run(request, scratch_->data(), scratch_->size(), stream);
  check_cuda(cudaEventRecord(end_.get(), stream), "record prepared vision end");
}

float PreparedImage::gpu_milliseconds() const {
  check_cuda(cudaEventSynchronize(end_.get()), "synchronize prepared vision execution");
  float milliseconds;
  check_cuda(cudaEventElapsedTime(&milliseconds, begin_.get(), end_.get()),
             "measure prepared vision execution");
  return milliseconds;
}
}
