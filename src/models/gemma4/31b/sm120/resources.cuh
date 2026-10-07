#pragma once
#include "cuda_memory.cuh"

#include "gewell/mtp_target.h"
#include "gewell/console.h"

#include "gewell/models/gemma4/31b/artifact.h"
#include "gewell/bf16_primitives.h"
#include "gewell/models/gemma4/31b/model.h"
#include "gewell/prefill_primitives.h"
#include "gewell/weight_qdq.h"
#include "gewell/models/gemma4/31b/sm120/nvfp4_projections.h"
#include "gewell/models/gemma4/31b/sm120/fp8_projections.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <openssl/evp.h>


#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>


namespace gewell::gemma4_31b::sm120 {

namespace artifact = gewell::artifact;
namespace model = gewell::gemma4_31b;
namespace primitives = gewell::bf16_primitives;
namespace prefill = gewell::prefill_primitives;
namespace qdq = gewell::weight_qdq;

using BFloat16 = primitives::BFloat16;

using cuda_detail::fail;
using cuda_detail::check_cuda;
using cuda_detail::DeviceAllocation;
using cuda_detail::PinnedHostAllocation;

constexpr std::size_t kScratchAlignment = 256;
inline void check_cublas(cublasStatus_t status, std::string_view operation) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    fail(operation, "cuBLASLt status " + std::to_string(status));
  }
}

constexpr std::size_t align_up(std::size_t value, std::size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}


class LtHandle {
 public:
  LtHandle() { check_cublas(cublasLtCreate(&handle_), "cublasLtCreate"); }
  ~LtHandle() {
    if (handle_ != nullptr) {
      cublasLtDestroy(handle_);
    }
  }
  LtHandle(const LtHandle&) = delete;
  LtHandle& operator=(const LtHandle&) = delete;
  [[nodiscard]] cublasLtHandle_t get() const { return handle_; }

 private:
  cublasLtHandle_t handle_{nullptr};
};


class MatrixLayout {
 public:
  MatrixLayout(std::uint64_t rows, std::uint64_t columns,
               std::int64_t leading_dimension) {
    check_cublas(cublasLtMatrixLayoutCreate(&layout_, CUDA_R_16BF, rows,
                                            columns, leading_dimension),
                 "cublasLtMatrixLayoutCreate");
    const cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    check_cublas(cublasLtMatrixLayoutSetAttribute(
                     layout_, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                     sizeof(order)),
                 "set row-major matrix layout");
  }
  ~MatrixLayout() {
    if (layout_ != nullptr) {
      cublasLtMatrixLayoutDestroy(layout_);
    }
  }
  MatrixLayout(const MatrixLayout&) = delete;
  MatrixLayout& operator=(const MatrixLayout&) = delete;
  [[nodiscard]] cublasLtMatrixLayout_t get() const { return layout_; }

 private:
  cublasLtMatrixLayout_t layout_{nullptr};
};

class LinearPlan {
 public:
  LinearPlan(std::uint32_t rows, std::uint32_t input_width,
             std::uint32_t output_width)
      : input_(rows, input_width, input_width),
        weight_(output_width, input_width, input_width),
        output_(rows, output_width, output_width) {
    check_cublas(cublasLtMatmulDescCreate(&operation_, CUBLAS_COMPUTE_32F,
                                          CUDA_R_32F),
                 "cublasLtMatmulDescCreate");
    const cublasOperation_t no_transpose = CUBLAS_OP_N;
    const cublasOperation_t transpose = CUBLAS_OP_T;
    check_cublas(cublasLtMatmulDescSetAttribute(
                     operation_, CUBLASLT_MATMUL_DESC_TRANSA, &no_transpose,
                     sizeof(no_transpose)),
                 "set pair linear TRANSA");
    check_cublas(cublasLtMatmulDescSetAttribute(
                     operation_, CUBLASLT_MATMUL_DESC_TRANSB, &transpose,
                     sizeof(transpose)),
                 "set pair linear TRANSB");
  }
  ~LinearPlan() {
    if (operation_ != nullptr) {
      cublasLtMatmulDescDestroy(operation_);
    }
  }
  LinearPlan(const LinearPlan&) = delete;
  LinearPlan& operator=(const LinearPlan&) = delete;

  void run(cublasLtHandle_t handle, const BFloat16* input,
           const BFloat16* weight, BFloat16* output,
           cudaStream_t stream = nullptr) const {
    const float alpha = 1.0F;
    const float beta = 0.0F;
    check_cublas(
        cublasLtMatmul(handle, operation_, &alpha, input, input_.get(), weight,
                       weight_.get(), &beta, output, output_.get(), output,
                       output_.get(), nullptr, nullptr, 0, stream),
        "pair cublasLtMatmul");
  }

 private:
  cublasLtMatmulDesc_t operation_{nullptr};
  MatrixLayout input_;
  MatrixLayout weight_;
  MatrixLayout output_;
};

class CudaEvent {
 public:
  CudaEvent() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
  ~CudaEvent() {
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
    }
  }
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  [[nodiscard]] cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_{nullptr};
};

class NonblockingCudaStream {
 public:
  NonblockingCudaStream() {
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "create graph-decode nonblocking stream");
  }
  ~NonblockingCudaStream() {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }
  NonblockingCudaStream(const NonblockingCudaStream&) = delete;
  NonblockingCudaStream& operator=(const NonblockingCudaStream&) = delete;
  [[nodiscard]] cudaStream_t get() const { return stream_; }

 private:
  cudaStream_t stream_{nullptr};
};


class CublasHandle {
 public:
  CublasHandle() {
    check_cublas(cublasCreate(&handle_), "cublasCreate");
    check_cublas(cublasSetMathMode(handle_, CUBLAS_TENSOR_OP_MATH),
                 "set cuBLAS tensor-op math mode");
  }
  ~CublasHandle() {
    if (handle_ != nullptr) {
      cublasDestroy(handle_);
    }
  }
  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;
  [[nodiscard]] cublasHandle_t get() const { return handle_; }

 private:
  cublasHandle_t handle_{nullptr};
};


inline double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace gewell::gemma4_31b::sm120
