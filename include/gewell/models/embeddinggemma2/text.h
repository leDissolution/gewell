#pragma once
#include "gewell/tokenizer.h"
#include "gewell/models/embeddinggemma2/input.h"

namespace gewell::embeddinggemma2 {
const text::TextContract& text_contract();
std::vector<std::uint32_t> tokenize_input(const text::Tokenizer& tokenizer, std::string_view input);
PreparedInput prepare_input(const text::Tokenizer& tokenizer, std::string_view input);
PreparedInput prepare_input(const text::Tokenizer& tokenizer, const std::vector<InputPart>& parts);
}  // namespace gewell::embeddinggemma2
