#pragma once

#include "gewell/text_contract.h"

namespace gewell::gemma4 {

// Pinned 31B and 26B-A4B assets share token IDs and validated native text mechanics.
[[nodiscard]] const text::TextContract& text_contract_31b();
[[nodiscard]] const text::TextContract& text_contract_26b_a4b();



}  // namespace gewell::gemma4
