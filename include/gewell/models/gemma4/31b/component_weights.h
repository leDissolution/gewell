#pragma once
#include "gewell/component_weights.h"

namespace gewell::gemma4_31b {
enum class Component { assistant, vision };
std::vector<component::TensorSpec> component_specs(Component kind);
}  // namespace gewell::gemma4_31b
