#pragma once
#include "gewell/models/gemma4/26b_a4b/artifact.h"
#include "gewell/nvfp4_linear.h"
#include "gewell/fp8_linear.h"
#include "gewell/nvfp4_policy.h"
#include <cublas_v2.h>
#include <memory>
#include <vector>

namespace gewell::gemma4_26b_a4b::sm120 {
struct Projection {
  StorageType storage{StorageType::bf16};
  unsigned input{}, output{}, physical_output{};
  const __nv_bfloat16* bf16{};
  nvfp4::Weight nvfp4;
  fp8::Weight fp8;
};
// Bind validated native metadata once. Device payload points to this entry,
// not the start of the whole artifact. Logical K is never padded.
Projection bind_projection(const TensorSpec&, const TensorEntry&,
                           const std::uint8_t* host_payload, const void* device_payload);

// Shared scratch for sequential projections on one bound stream. prepare is
// called before layer traversal; run never creates plans or allocates storage.
// NVFP4 output padding is removed before returning model-visible values.
// run accepts a projection from the constructor bindings and a cuBLAS handle
// configured for this stream, host scalar pointer mode, and FP32 accumulation
// without reduced precision. The workspace does not change the caller's handle.
class ProjectionWorkspace {
 public:
  ProjectionWorkspace(const std::vector<Projection>&, unsigned max_rows,
                      nvfp4::ActivationPolicy, cudaStream_t);
  ~ProjectionWorkspace();
  void prepare(unsigned rows, nvfp4::Phase);
  void run(cublasHandle_t, const Projection&, const __nv_bfloat16* input,
           __nv_bfloat16* output);
  std::size_t bytes() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
