#pragma once
#include "gewell/runtime/backend.h"
#include "gewell/nvfp4_policy.h"
#include "gewell/models/gemma4/26b_a4b/artifact.h"

namespace gewell::gemma4_26b_a4b::sm120 {
std::unique_ptr<runtime::ExecutionBackend> make_runtime_backend(
    ArtifactFile weights, const runtime::BatchLimits& limits, nvfp4::ActivationPolicy policy,
    const std::string& assistant_path, const std::string& vision_path, const std::string& qdq_path);
}
