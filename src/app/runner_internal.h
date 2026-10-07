#pragma once
#include "gewell/app.h"
#include "model_selection.h"
#include "gewell/runtime/scheduler.h"
namespace gewell::app {
runtime::BatchLimits live_batch_limits(std::uint32_t max_batch, const RuntimeSettings&, std::size_t max_requests, ModelKind model);
}
