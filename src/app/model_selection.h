#pragma once
#include <string>
#include "gewell/text_contract.h"

namespace gewell::app {
enum class ModelKind { gemma4_31b, gemma4_26b_a4b };
// Dispatch only; the selected artifact reader performs complete metadata validation.
ModelKind artifact_model(const std::string& path);
ModelKind serving_model(const std::string& directory);
const text::TextContract& text_contract(ModelKind model);
}
