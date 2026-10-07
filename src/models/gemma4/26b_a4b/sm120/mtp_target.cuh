#pragma once
#include "gewell/mtp_cycle.h"
#include "batched_executor.cuh"

namespace gewell::gemma4_26b_a4b::sm120 {
// Borrows the resident executor (and its weights) and official assistant views.
// The executor and its bound stream must outlive this adapter and the cycle.
std::unique_ptr<mtp_cycle::BatchTarget> make_mtp_target(BatchedExecutor& executor,
    const std::array<const __nv_bfloat16*,48>& assistant, unsigned capacity_rows,
    cudaStream_t stream);
}  // namespace gewell::gemma4_26b_a4b::sm120
