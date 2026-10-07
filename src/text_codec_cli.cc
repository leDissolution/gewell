#include "gewell/text_codec_cli.h"
#include "gewell/chat_codec.h"

#include <iostream>
#include <stdexcept>

namespace gewell::text {
namespace {

using nlohmann::json;

std::vector<std::uint32_t> token_ids(const json& value, const TextContract& contract) {
  if (!value.is_array()) throw std::invalid_argument("tokens must be an array");
  std::vector<std::uint32_t> tokens;
  for (const auto& item : value) {
    if (!item.is_number_unsigned() || item.get<std::uint64_t>() >= contract.vocabulary_size)
      throw std::invalid_argument("token ID must be an integer in 0.." + std::to_string(contract.vocabulary_size - 1));
    tokens.push_back(item.get<std::uint32_t>());
  }
  return tokens;
}

json process(const Tokenizer& tokenizer, const json& request) {
  if (!request.is_object()) throw std::invalid_argument("request must be an object");
  const auto operation = request.at("operation").get<std::string>();
  if (operation == "completion") {
    if (request.contains("text") == request.contains("tokens"))
      throw std::invalid_argument("completion requires exactly one of text or tokens");
    if (request.contains("tokens"))
      return {{"token_ids", tokenizer.completion_prompt(token_ids(request.at("tokens"), tokenizer.contract()))}};
  }
  if (operation == "encode" || operation == "completion") {
    const auto& text = request.at("text").get_ref<const std::string&>();
    return {{"token_ids", operation == "encode" ? tokenizer.encode(text) : tokenizer.completion_prompt(text)}};
  }
  if (operation == "chat") {
    const auto messages = tokenizer.contract().normalize_messages(request.at("messages"));
    const auto options = tokenizer.contract().template_options(
        request.value("chat_template_kwargs", json(nullptr)));
    const auto tools = tokenizer.contract().normalize_tools(request.value("tools", json(nullptr)));
    const auto rendered = tokenizer.contract().render_chat(messages, options, tools);
    return {{"prompt", rendered}, {"token_ids", tokenizer.encode(rendered)}};
  }
  if (operation == "decode") {
    IncrementalTextDecoder decoder(tokenizer);
    std::string text;
    auto fragments = json::array();
    for (const auto id : token_ids(request.at("tokens"), tokenizer.contract())) {
      auto fragment = decoder.push(id);
      text += fragment;
      fragments.push_back(std::move(fragment));
    }
    auto final = decoder.finish();
    text += final;
    fragments.push_back(std::move(final));
    return {{"text", text}, {"fragments", std::move(fragments)}};
  }
  if (operation == "chat_decode") {
    ChatOutputDecoder decoder(tokenizer);
    std::string content, reasoning;
    auto fragments = json::array();
    auto add = [&](ChatDelta delta) {
      content += delta.content;
      reasoning += delta.reasoning;
      fragments.push_back({{"content", delta.content}, {"reasoning", delta.reasoning}});
    };
    for (const auto id : token_ids(request.at("tokens"), tokenizer.contract())) add(decoder.push(id));
    add(decoder.finish());
    return {{"content", content}, {"reasoning", reasoning}, {"fragments", std::move(fragments)}};
  }
  throw std::invalid_argument("unknown codec operation: " + operation);
}

}  // namespace

int run_text_codec(const Tokenizer& tokenizer) {
  std::string line;
  bool failed = false;
  while (std::getline(std::cin, line)) {
    try {
      std::cout << process(tokenizer, json::parse(line)).dump() << '\n';
    } catch (const std::exception& error) {
      std::cout << json({{"error", error.what()}}).dump() << '\n';
      failed = true;
    }
    std::cout.flush();
  }
  if (std::cin.bad() || !std::cout.good()) throw std::runtime_error("codec JSON-lines I/O failed");
  return failed ? 1 : 0;
}

}  // namespace gewell::text
