#pragma once

#include "gewell/tokenizer.h"

namespace gewell::text {
// CPU-only JSON-lines diagnostic for the explicitly selected local model.
int run_text_codec(const Tokenizer& tokenizer);
}  // namespace gewell::text
