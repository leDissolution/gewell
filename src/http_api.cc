#include "gewell/http_api.h"

#include "gewell/chat_codec.h"
#include "gewell/tool_output.h"
#include "gewell/stop_matcher.h"

#include <cmath>
#include <deque>
#include <initializer_list>
#include <limits>
#include <set>
#include <utility>

namespace gewell::http {
namespace {

using nlohmann::json;

[[noreturn]] void invalid(const std::string& param, const std::string& message,
                          const std::string& code = "invalid_parameter") {
  throw Error(400, message, param, code);
}

[[noreturn]] void execution_error(const std::string& message) {
  throw Error(500, message, {}, "invalid_execution_result");
}

bool present(const json& value, const char* key) {
  const auto found = value.find(key);
  return found != value.end() && !found->is_null();
}

void allow_keys(const json& value, const std::set<std::string>& allowed,
                const std::string& prefix = {}) {
  for (const auto& item : value.items()) {
    if (!allowed.count(item.key())) {
      const auto param = prefix + item.key();
      invalid(param, "unsupported field: " + param, "unsupported_parameter");
    }
  }
}

std::uint64_t integer(const json& value, const std::string& param,
                       std::uint64_t minimum, std::uint64_t maximum) {
  if (!value.is_number_integer() ||
      (value.is_number_integer() && !value.is_number_unsigned() &&
       value.get<std::int64_t>() < 0)) {
    invalid(param, param + " must be an integer in " + std::to_string(minimum) +
                       ".." + std::to_string(maximum));
  }
  const auto number = value.get<std::uint64_t>();
  if (number < minimum || number > maximum)
    invalid(param, param + " must be an integer in " + std::to_string(minimum) +
                       ".." + std::to_string(maximum));
  return number;
}

double number(const json& value, const std::string& param,
              double minimum, double maximum) {
  if (!value.is_number()) invalid(param, param + " must be a number");
  const double result = value.get<double>();
  if (!std::isfinite(result) || result < minimum || result > maximum)
    invalid(param, param + " must be finite and in [" + std::to_string(minimum) +
                       ", " + std::to_string(maximum) + "]");
  return result;
}

bool boolean(const json& value, const std::string& param) {
  if (!value.is_boolean()) invalid(param, param + " must be a boolean");
  return value.get<bool>();
}

std::string owner_id(const json& value, const std::string& param) {
  if (!value.is_string()) invalid(param, param + " must be a string");
  const auto id = value.get<std::string>();
  if (id.empty() || id.size() > 128 || id.find('\0') != std::string::npos)
    invalid(param, param + " must contain 1..128 UTF-8 bytes without NUL");
  return id;
}

void validate_model(const json& body, const std::string& model, bool required) {
  if (!present(body, "model")) {
    if (required) invalid("model", "model is required", "missing_required_parameter");
    return;
  }
  if (!body["model"].is_string()) invalid("model", "model must be a string");
  if (body["model"] != model)
    throw Error(404, "model is not served: " + body["model"].get<std::string>(),
                "model", "model_not_found");
}

CacheControls cache_controls(const json& body) {
  CacheControls controls;
  if (!present(body, "cache")) return controls;
  const auto& value = body["cache"];
  if (!value.is_object()) invalid("cache", "cache must be an object or null");
  allow_keys(value, {"mode", "prompt_id", "priority", "finished"}, "cache.");
  if (present(value, "mode")) {
    if (value["mode"] != "auto" && value["mode"] != "reuse_only")
      invalid("cache.mode", "cache.mode must be auto or reuse_only");
    controls.reuse_only = value["mode"] == "reuse_only";
  }
  if (present(value, "prompt_id"))
    controls.prompt_id = owner_id(value["prompt_id"], "cache.prompt_id");
  if (present(value, "priority")) {
    if (value["priority"] == "low") controls.priority = Priority::low;
    else if (value["priority"] == "high") controls.priority = Priority::high;
    else if (value["priority"] != "normal")
      invalid("cache.priority", "cache.priority must be low, normal, or high");
  }
  if (present(value, "finished"))
    controls.finished = boolean(value["finished"], "cache.finished");
  if (controls.reuse_only && (!controls.prompt_id.empty() ||
      controls.priority != Priority::normal || controls.finished))
    invalid("cache", "reuse_only requires no prompt_id, normal priority, and finished=false");
  if (controls.finished && controls.prompt_id.empty())
    invalid("cache.finished", "cache.finished=true requires prompt_id");
  return controls;
}

void unsupported(const std::string& key) {
  invalid(key, key + " is not supported yet", "unsupported_parameter");
}

void neutral_controls(const json& body, Request& request, const text::TextContract& contract) {
  const bool chat = request.chat;
  // Ignore unknown completion-request extensions, but never silently accept
  // a recognized control whose behavior we cannot provide. Descriptive hints
  // such as metadata, safety_identifier and prompt_cache_key need no action.
  for (const char* key : {"audio", "moderation", "prediction", "prompt_cache_retention",
                         "service_tier", "verbosity", "web_search_options"})
    if (present(body, key)) unsupported(key);
  if (present(body, "n") && integer(body["n"], "n", 1, contract.context_tokens) != 1) unsupported("n");
  for (const char* key : {"frequency_penalty", "presence_penalty"}) {
    if (present(body, key) && number(body[key], key, -2, 2) != 0) unsupported(key);
  }
  if (present(body, "logit_bias") &&
      (!body["logit_bias"].is_object() || !body["logit_bias"].empty())) unsupported("logit_bias");
  if (present(body, "user") && !body["user"].is_string())
    invalid("user", "user must be a string or null");
  if (present(body, "logprobs")) {
    if (!chat) {
      (void)integer(body["logprobs"], "logprobs", 0, 5);
      unsupported("logprobs");
    }
    request.logprobs = boolean(body["logprobs"], "logprobs");
  }
  if (chat) {
    for (const char* key : {"prompt", "echo", "best_of", "suffix"})
      if (present(body, key)) unsupported(key);
    if (present(body, "top_logprobs")) {
      request.top_logprobs = static_cast<std::uint32_t>(
          integer(body["top_logprobs"], "top_logprobs", 0, kMaxTopLogprobs));
      if (!request.logprobs)
        invalid("top_logprobs", "top_logprobs requires logprobs=true");
    }
    if (present(body, "store") && boolean(body["store"], "store")) unsupported("store");
    if (present(body, "modalities") && body["modalities"] != json::array({"text"})) unsupported("modalities");
    for (const char* key : {"functions"}) {
      if (present(body, key) && (!body[key].is_array() || !body[key].empty())) unsupported(key);
    }
    if (present(body, "function_call")) unsupported("function_call");
  } else {
    for (const char* key : {"messages", "max_completion_tokens", "chat_template_kwargs",
                           "top_logprobs", "response_format", "store", "modalities", "tools",
                           "tool_choice", "functions", "function_call", "parallel_tool_calls",
                           "reasoning_effort"})
      if (present(body, key)) unsupported(key);
    if (present(body, "echo") && boolean(body["echo"], "echo")) unsupported("echo");
    if (present(body, "best_of") && integer(body["best_of"], "best_of", 1, contract.context_tokens) != 1)
      unsupported("best_of");
    if (present(body, "suffix")) unsupported("suffix");
  }
}

void stop_and_seed(const json& body, Request& request) {
  if (present(body, "stop")) {
    const auto& value = body["stop"];
    if (value.is_string()) request.stops.push_back(value.get<std::string>());
    else if (value.is_array() && !value.empty() && value.size() <= 4) {
      for (const auto& item : value) {
        if (!item.is_string()) invalid("stop", "stop entries must be nonempty strings");
        request.stops.push_back(item.get<std::string>());
      }
    } else invalid("stop", "stop must be a nonempty string or 1..4 nonempty strings");
    for (const auto& stop : request.stops)
      if (stop.empty()) invalid("stop", "stop entries must be nonempty strings");
  }
  if (present(body, "seed")) {
    const auto& value = body["seed"];
    if (!value.is_number_integer() || (value.is_number_unsigned() &&
        value.get<std::uint64_t>() > std::uint64_t(std::numeric_limits<std::int64_t>::max())))
      invalid("seed", "seed must be a signed 64-bit integer");
    request.seed = value.get<std::int64_t>();
  }
}

json tool_controls(const json& body, Request& request, const text::TextContract& contract) {
  json tools;
  try { tools = contract.normalize_tools(body.value("tools", json(nullptr))); }
  catch (const std::invalid_argument& error) { invalid("tools", error.what()); }
  for (const auto& tool : tools) {
    request.tool_names.push_back(tool["function"]["name"].get<std::string>());
    request.enforce_tool_calls |= tool["function"].value("strict", false);
  }
  request.allow_tool_calls = !tools.empty();
  if (present(body, "parallel_tool_calls")) {
    request.parallel_tool_calls = boolean(body["parallel_tool_calls"], "parallel_tool_calls");
    request.enforce_tool_calls |= !request.parallel_tool_calls;
  }
  if (present(body, "tool_choice")) {
    const auto& choice = body["tool_choice"];
    if (choice.is_string()) {
      if (choice != "auto" && choice != "none" && choice != "required")
        invalid("tool_choice", "tool_choice must be auto, none, required, or a function selection");
      request.allow_tool_calls = choice != "none";
      request.require_tool_calls = choice == "required";
      request.enforce_tool_calls |= choice != "auto";
    } else if (choice.is_object()) {
      request.enforce_tool_calls = request.allow_tool_calls = true;
      json selected;
      if (choice.value("type", json()) == "function") {
        allow_keys(choice, {"type", "function"}, "tool_choice.");
        selected = json::array({choice});
        request.require_tool_calls = true;
        request.parallel_tool_calls = false;
      } else if (choice.value("type", json()) == "allowed_tools") {
        allow_keys(choice, {"type", "allowed_tools"}, "tool_choice.");
        const auto allowed = choice.value("allowed_tools", json());
        if (!allowed.is_object()) invalid("tool_choice", "allowed_tools must be an object");
        allow_keys(allowed, {"mode", "tools"}, "tool_choice.allowed_tools.");
        const auto mode = allowed.value("mode", json());
        if (mode != "auto" && mode != "required") invalid("tool_choice", "allowed_tools mode must be auto or required");
        request.require_tool_calls = mode == "required";
        selected = allowed.value("tools", json());
      } else invalid("tool_choice", "unsupported tool selection type");
      if (!selected.is_array() || selected.empty() || selected.size() > 128)
        invalid("tool_choice", "selection must contain 1..128 declared functions");
      std::vector<std::string> names;
      for (const auto& tool : selected) {
        if (!tool.is_object()) invalid("tool_choice", "selected tool must be an object");
        allow_keys(tool, {"type", "function"}, "tool_choice.");
        if (tool.value("type", json()) != "function" || !tool.contains("function") || !tool["function"].is_object())
          invalid("tool_choice", "selected tool must specify a function");
        allow_keys(tool["function"], {"name"}, "tool_choice.function.");
        const auto name = tool["function"].value("name", json());
        if (!name.is_string() || std::find(request.tool_names.begin(), request.tool_names.end(), name.get<std::string>()) == request.tool_names.end())
          invalid("tool_choice", "selected function is not declared");
        if (std::find(names.begin(), names.end(), name.get<std::string>()) != names.end())
          invalid("tool_choice", "selected functions must be unique");
        names.push_back(name.get<std::string>());
      }
      request.tool_names = std::move(names);
    } else invalid("tool_choice", "tool_choice must be a string or object");
    if (request.allow_tool_calls && tools.empty()) invalid("tool_choice", "tool selection requires tools");
  }
  return tools;
}

void response_constraint(const json& body, Request& request, constraint::Compiler& compiler) {
  if (!present(body, "response_format")) return;
  const auto& format = body["response_format"];
  if (!format.is_object()) invalid("response_format", "response_format must be an object");
  if (!format.contains("type") || !format["type"].is_string())
    invalid("response_format.type", "response_format.type must be text, json_object, or json_schema");
  const auto type = format["type"].get<std::string>();
  if (type == "text") {
    allow_keys(format, {"type"}, "response_format.");
    return;
  }
  json schema;
  if (type == "json_object") {
    allow_keys(format, {"type"}, "response_format.");
    schema = {{"type", "object"}};
  } else if (type == "json_schema") {
    allow_keys(format, {"type", "json_schema"}, "response_format.");
    if (!format.contains("json_schema") || !format["json_schema"].is_object())
      invalid("response_format.json_schema", "json_schema must be an object");
    const auto& value = format["json_schema"];
    allow_keys(value, {"name", "description", "schema", "strict"}, "response_format.json_schema.");
    if (!value.contains("name") || !value["name"].is_string())
      invalid("response_format.json_schema.name", "schema name is required and must be a string");
    const auto name = value["name"].get<std::string>();
    if (name.empty() || name.size() > 64 || name.find_first_not_of(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-") != std::string::npos)
      invalid("response_format.json_schema.name", "schema name must contain 1..64 ASCII letters, digits, underscores, or hyphens");
    if (present(value, "description") && !value["description"].is_string())
      invalid("response_format.json_schema.description", "schema description must be a string");
    if (present(value, "strict")) (void)boolean(value["strict"], "response_format.json_schema.strict");
    if (!value.contains("schema"))
      invalid("response_format.json_schema.schema", "schema is required", "missing_required_parameter");
    schema = value["schema"];
  } else invalid("response_format.type", "response_format.type must be text, json_object, or json_schema");
  if (!request.stops.empty())
    invalid("stop", "stop strings cannot be combined with constrained response_format");
  try { request.constraint = compiler.compile(schema); }
  catch (const std::invalid_argument& error) {
    invalid("response_format.json_schema.schema", error.what(), "invalid_schema");
  }
}

std::vector<std::uint32_t> raw_prompt(const json& value, const text::Tokenizer& tokenizer) {
  if (value.is_string()) return tokenizer.completion_prompt(value.get_ref<const std::string&>());
  if (!value.is_array()) invalid("prompt", "prompt must be a string or flat token-ID array");
  if (value.size() > tokenizer.contract().context_tokens) invalid("prompt", "prompt exceeds " + std::to_string(tokenizer.contract().context_tokens) + " tokens", "context_length_exceeded");
  std::vector<std::uint32_t> tokens;
  tokens.reserve(value.size());
  for (std::size_t i = 0; i < value.size(); ++i)
    tokens.push_back(static_cast<std::uint32_t>(integer(value[i], "prompt[" + std::to_string(i) + "]", 0, tokenizer.contract().vocabulary_size - 1)));
  return tokens;
}

std::vector<std::uint32_t> chat_prompt(const json& body, const text::Tokenizer& tokenizer,
                                       const json& tools, Request& request) {
  text::ChatTemplateOptions options;
  try {
    options = tokenizer.contract().template_options(body.value("chat_template_kwargs", json(nullptr)));
  } catch (const std::invalid_argument& error) {
    invalid("chat_template_kwargs", error.what());
  }
  if (present(body, "reasoning_effort")) {
    const auto& effort = body["reasoning_effort"];
    if (effort != "none" && effort != "low" && effort != "medium" && effort != "high")
      invalid("reasoning_effort", "reasoning_effort must be none, low, medium, or high");
    options.enable_thinking = effort != "none";
  }
  request.enable_thinking = options.enable_thinking;
  if (!body.contains("messages")) invalid("messages", "messages is required", "missing_required_parameter");
  try {
    const auto rendered = tokenizer.contract().render_chat(tokenizer.contract().normalize_messages(body["messages"]), options, tools);
    const auto continuation = tokenizer.contract().reasoning_continuation;
    request.initial_reasoning = rendered.size() >= continuation.size() &&
        rendered.compare(rendered.size() - continuation.size(), continuation.size(), continuation) == 0;
    return tokenizer.encode(rendered);
  } catch (const std::invalid_argument& error) {
    invalid("messages", error.what());
  }
}

// Collect image parts in conversation order and validate the complete message
// structure before allocating decoded tensors. Each preparation retains its
// own transport memory lease until the request retires.
std::vector<std::shared_ptr<runtime::ImageInput>> prepare_chat_images(
    json& body, const text::Tokenizer& tokenizer, const ImageSupport& images) {
  if (!body.contains("messages") || !body["messages"].is_array()) return {};
  struct PendingImage {
    json* part;
    std::string url, param;
  };
  std::vector<PendingImage> pending;
  auto& messages = body["messages"];
  for (std::size_t i = 0; i < messages.size(); ++i) {
    auto& message = messages[i];
    if (!message.is_object() || !message.contains("content") || !message["content"].is_array()) continue;
    auto& parts = message["content"];
    for (std::size_t j = 0; j < parts.size(); ++j) {
      auto& part = parts[j];
      if (!part.is_object() || !part.contains("type") || part["type"] != "image_url") continue;
      const auto param = "messages[" + std::to_string(i) + "].content[" + std::to_string(j) + "]";
      if (!images.prepare)
        invalid(param, "image input requires --vision PATH", "unsupported_parameter");
      if (!message.contains("role") || message["role"] != "user")
        invalid(param, "image input requires a user message");
      allow_keys(part, {"type", "image_url"}, param + ".");
      if (!part.contains("image_url") || !part["image_url"].is_object())
        invalid(param + ".image_url", "image_url must be an object");
      const auto& image = part["image_url"];
      allow_keys(image, {"url", "detail"}, param + ".image_url.");
      if (!image.contains("url") || !image["url"].is_string())
        invalid(param + ".image_url.url", "image URL must be a base64 data URL string");
      if (present(image, "detail") && image["detail"] != "auto")
        invalid(param + ".image_url.detail", "only detail=auto is supported", "unsupported_parameter");
      pending.push_back({&part, image["url"].get<std::string>(), param});
    }
  }
  if (pending.empty()) return {};
  // Even the smallest image needs begin, feature, and end tokens.
  if (pending.size() > tokenizer.contract().context_tokens / 3)
    invalid("messages", "image count exceeds the token context", "context_length_exceeded");
  for (auto& image : pending) *image.part = {{"type", "text"}, {"text", ""}};
  try { (void)tokenizer.contract().normalize_messages(messages); }
  catch (const std::invalid_argument& error) { invalid("messages", error.what()); }
  std::vector<std::shared_ptr<runtime::ImageInput>> prepared;
  prepared.reserve(pending.size());
  std::uint64_t image_tokens = 0;
  for (auto& image : pending) {
    std::shared_ptr<runtime::ImageInput> value;
    try { value = images.prepare(image.url); }
    catch (const std::invalid_argument& error) { invalid(image.param + ".image_url.url", error.what(), "invalid_image"); }
    if (!value || value->begin || !value->end || value->end > images.max_image_tokens)
      execution_error("image processor returned an invalid feature span");
    image_tokens += std::uint64_t(value->end) + 2;
    if (image_tokens > tokenizer.contract().context_tokens)
      invalid("messages", "image tokens exceed the token context", "context_length_exceeded");
    std::string placeholder = tokenizer.token_piece(images.begin_token);
    for (std::uint32_t i = 0; i < value->end; ++i) placeholder += tokenizer.token_piece(images.image_token);
    placeholder += tokenizer.token_piece(images.end_token);
    (*image.part)["text"] = std::move(placeholder);
    prepared.push_back(std::move(value));
  }
  return prepared;
}

void validate_image_spans(const std::vector<std::uint32_t>& prompt,
                          const ImageSupport& support,
                          const std::vector<std::shared_ptr<runtime::ImageInput>>& images,
                          bool chat) {
  if (!support.prepare) return;
  const auto param = chat ? "messages" : "prompt";
  std::size_t image_index = 0;
  for (std::size_t position = 0; position < prompt.size(); ++position) {
    const auto token = prompt[position];
    if (token == support.begin_token) {
      if (image_index == images.size())
        invalid(param, "image placeholder tokens require an image content part");
      auto& image = *images[image_index++];
      const auto begin = position + 1;
      const auto end = begin + image.end;
      if (end >= prompt.size() || prompt[end] != support.end_token ||
          !std::all_of(prompt.begin() + begin, prompt.begin() + end,
                       [&](auto feature) { return feature == support.image_token; }))
        invalid(param, "image content must produce ordered bracketed image spans");
      image.begin = static_cast<std::uint32_t>(begin);
      image.end = static_cast<std::uint32_t>(end);
      position = end;
    } else if (token == support.image_token || token == support.end_token) {
      invalid(param, "image placeholder tokens require an image content part");
    }
  }
  if (image_index != images.size())
    invalid(param, "image content must produce ordered bracketed image spans");
}

std::string decoded_model_id(std::string_view value) {
  std::string result;
  auto hex = [](char digit) -> int {
    if (digit >= '0' && digit <= '9') return digit - '0';
    if (digit >= 'a' && digit <= 'f') return digit - 'a' + 10;
    if (digit >= 'A' && digit <= 'F') return digit - 'A' + 10;
    return -1;
  };
  for (std::size_t i = 0; i < value.size(); ++i) {
    if (value[i] != '%') result.push_back(value[i]);
    else {
      if (i + 2 >= value.size() || hex(value[i + 1]) < 0 || hex(value[i + 2]) < 0)
        invalid("model", "invalid model path encoding");
      result.push_back(static_cast<char>((hex(value[i + 1]) << 4) | hex(value[i + 2])));
      i += 2;
    }
  }
  return result;
}

json usage(const Result& result) {
  return {{"prompt_tokens", result.prompt_tokens}, {"completion_tokens", result.completion_tokens},
          {"total_tokens", std::uint64_t(result.prompt_tokens) + result.completion_tokens},
          {"prompt_tokens_details", {{"cached_tokens", result.cached_tokens}}}};
}

int hex_digit(char value) {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return value - 'a' + 10;
  if (value >= 'A' && value <= 'F') return value - 'A' + 10;
  return -1;
}

std::optional<unsigned char> fallback_byte(const text::Tokenizer& tokenizer,
                                           std::uint32_t token) {
  const auto& piece = tokenizer.token_piece(token);
  if (piece.size() != 6 || piece.compare(0, 3, "<0x") != 0 ||
      piece.back() != '>')
    return std::nullopt;
  const auto high = hex_digit(piece[3]), low = hex_digit(piece[4]);
  if (high < 0 || low < 0)
    execution_error("invalid byte-fallback logprob token");
  return static_cast<unsigned char>((high << 4) | low);
}

json token_bytes(const text::Tokenizer& tokenizer, std::uint32_t token) {
  if (token >= tokenizer.contract().vocabulary_size) execution_error("logprob token is outside the vocabulary");
  if (tokenizer.is_special(token)) return nullptr;
  if (const auto byte = fallback_byte(tokenizer, token))
    return json::array({*byte});
  text::IncrementalTextDecoder decoder(tokenizer);
  auto decoded = decoder.push(token);
  decoded += decoder.finish();
  json bytes = json::array();
  for (const unsigned char byte : decoded) bytes.push_back(byte);
  return bytes;
}

std::string token_text(const text::Tokenizer& tokenizer, std::uint32_t token) {
  if (token >= tokenizer.contract().vocabulary_size) execution_error("logprob token is outside the vocabulary");
  text::IncrementalTextDecoder decoder(tokenizer);
  auto decoded = decoder.push(token);
  decoded += decoder.finish();
  return decoded;
}

json scored_token(const text::Tokenizer& tokenizer, std::uint32_t token,
                  float logprob) {
  if (!std::isfinite(logprob) || logprob > 0)
    execution_error("invalid token log probability");
  return {{"token", token_text(tokenizer, token)}, {"bytes", token_bytes(tokenizer, token)},
          {"logprob", logprob}};
}

json logprob_record(const text::Tokenizer& tokenizer, std::uint32_t selected,
                    const TokenLogprobs& scores, std::uint32_t requested) {
  if (requested > kMaxTopLogprobs || scores.count > requested)
    execution_error("invalid top-logprob result count");
  auto result = scored_token(tokenizer, selected, scores.logprob);
  result["top_logprobs"] = json::array();
  float previous = std::numeric_limits<float>::infinity();
  std::uint32_t previous_token = 0;
  std::set<std::uint32_t> seen;
  for (std::uint32_t index = 0; index < scores.count; ++index) {
    const auto& alternative = scores.top[index];
    if (!seen.insert(alternative.token).second ||
        alternative.logprob > previous ||
        (alternative.logprob == previous && index && alternative.token < previous_token))
      execution_error("invalid top-logprob ordering or token IDs");
    result["top_logprobs"].push_back(
        scored_token(tokenizer, alternative.token, alternative.logprob));
    previous = alternative.logprob;
    previous_token = alternative.token;
  }
  return result;
}

void validate_accounting(const Request& request, const Result& result,
                         std::uint32_t seen, bool stopped) {
  if (!request.prompt || result.prompt_tokens != request.prompt->size() ||
      result.completion_tokens != seen || result.cached_tokens > result.prompt_tokens)
    execution_error("execution token accounting does not match the request/output");
  const auto processed = std::uint64_t(result.prompt_tokens) + seen - (seen != 0);
  if (result.processed_tokens != processed || result.stopped != stopped)
    execution_error("execution processed-token or stop accounting is inconsistent");
  if (request.operation == Operation::generate &&
      (seen == 0 || seen > request.max_tokens || (!stopped && seen != request.max_tokens)))
    execution_error("generation ended without EOS or the requested output limit");
}

json pool_json(const kv_cache::PoolStats& stats) {
  return {{"capacity_bytes", stats.capacity}, {"used_bytes", stats.used},
          {"peak_used_bytes", stats.peak_used}, {"free_bytes", stats.free}};
}

}  // namespace

Error::Error(int status_value, std::string message, std::string parameter, std::string error_code)
    : std::runtime_error(std::move(message)), status(status_value),
      param(std::move(parameter)), code(std::move(error_code)) {}

Request parse_request(std::string_view method, std::string_view path,
                      std::string_view encoded, const text::Tokenizer& tokenizer,
                      const std::string& model, constraint::Compiler& compiler,
                      const ImageSupport& images) {
  Request request;
  std::string_view expected_method = "GET";
  if (path == "/health") request.operation = Operation::health;
  else if (path == "/metrics") request.operation = Operation::metrics;
  else if (path == "/v1/cache/index") request.operation = Operation::cache_index;
  else if (path == "/v1/models") request.operation = Operation::models;
  else if (path.substr(0, 11) == "/v1/models/") request.operation = Operation::model;
  else if (path == "/v1/cache/stats") request.operation = Operation::stats;
  else {
    expected_method = "POST";
    if (path == "/v1/chat/completions") { request.operation = Operation::generate; request.chat = true; }
    else if (path == "/v1/completions") request.operation = Operation::generate;
    else if (path == "/v1/cache/prefill") request.operation = Operation::prefill;
    else if (path == "/v1/cache/finish") request.operation = Operation::finish;
    else throw Error(404, "route is not implemented", {}, "not_found");
  }
  if (method != expected_method) throw Error(405, "method is not allowed for this route", {}, "method_not_allowed");
  if (expected_method == "GET") {
    if (!encoded.empty()) invalid("body", "GET requests must not contain a body");
    if (request.operation == Operation::model && decoded_model_id(path.substr(11)) != model)
      throw Error(404, "model is not served", "model", "model_not_found");
    return request;
  }
  json body;
  try { body = json::parse(encoded); }
  catch (const json::exception& error) { throw Error(400, "invalid JSON: " + std::string(error.what()), {}, "invalid_json"); }
  if (!body.is_object()) invalid("body", "request body must be a JSON object");
  if (request.operation == Operation::finish) {
    allow_keys(body, {"prompt_id"});
    if (!present(body, "prompt_id")) invalid("prompt_id", "prompt_id is required", "missing_required_parameter");
    request.cache.prompt_id = owner_id(body["prompt_id"], "prompt_id");
    return request;
  }
  if (request.operation == Operation::prefill) {
    allow_keys(body, {"model", "messages", "prompt", "cache", "chat_template_kwargs",
                      "tools", "tool_choice", "parallel_tool_calls", "reasoning_effort"});
    if (body.contains("messages") == body.contains("prompt"))
      invalid("prompt", "prefill requires exactly one of messages or prompt");
    request.chat = body.contains("messages");
    request.max_tokens = 0;
    request.temperature = 0;
    if (!request.chat && (present(body, "tools") || present(body, "tool_choice") || present(body, "parallel_tool_calls")))
      invalid("tools", "tool controls require messages");
    if (!request.chat && present(body, "chat_template_kwargs") && body["chat_template_kwargs"] != json::object())
      invalid("chat_template_kwargs", "chat_template_kwargs requires messages");
    if (!request.chat && present(body, "reasoning_effort"))
      invalid("reasoning_effort", "reasoning_effort requires messages");
  } else {
    neutral_controls(body, request, tokenizer.contract());
    stop_and_seed(body, request);
    const char* limit = request.chat && present(body, "max_completion_tokens") ? "max_completion_tokens" : "max_tokens";
    if (present(body, limit)) request.max_tokens = static_cast<std::uint32_t>(integer(body[limit], limit, 1, tokenizer.contract().context_tokens));
    if (present(body, "temperature")) request.temperature = static_cast<float>(number(body["temperature"], "temperature", 0, 2));
    if (present(body, "top_p")) request.top_p = static_cast<float>(number(body["top_p"], "top_p", 0, 1));
    if (present(body, "top_k")) request.top_k = static_cast<std::uint32_t>(integer(
        body["top_k"], "top_k", 0, std::numeric_limits<std::uint32_t>::max()));
    if (present(body, "stream")) request.stream = boolean(body["stream"], "stream");
    if (present(body, "stream_options")) {
      const auto& options = body["stream_options"];
      if (!options.is_object()) invalid("stream_options", "stream_options must be an object or null");
      allow_keys(options, {"include_usage"}, "stream_options.");
      if (present(options, "include_usage")) request.include_usage = boolean(options["include_usage"], "stream_options.include_usage");
      if (!request.stream) invalid("stream_options", "stream_options requires stream=true");
    }
  }
  validate_model(body, model, request.operation == Operation::generate);
  request.cache = cache_controls(body);
  if (request.operation == Operation::prefill && (request.cache.reuse_only || request.cache.finished))
    invalid("cache", "prefill requires mode=auto and finished=false");
  std::vector<std::uint32_t> prompt;
  auto prepared_images = request.chat ? prepare_chat_images(body, tokenizer, images)
                                      : std::vector<std::shared_ptr<runtime::ImageInput>>{};
  json tools;
  if (request.chat) {
    tools = tool_controls(body, request, tokenizer.contract());
    prompt = chat_prompt(body, tokenizer, request.allow_tool_calls ? tools : json::array(), request);
  }
  else {
    if (!body.contains("prompt")) invalid("prompt", "prompt is required", "missing_required_parameter");
    prompt = raw_prompt(body["prompt"], tokenizer);
  }
  if (prompt.empty()) invalid(request.chat ? "messages" : "prompt", "tokenized input must not be empty");
  validate_image_spans(prompt, images, prepared_images, request.chat);
  request.images.assign(prepared_images.begin(), prepared_images.end());
  if (prompt.size() > tokenizer.contract().context_tokens || (request.max_tokens && prompt.size() + request.max_tokens - 1 > tokenizer.contract().context_tokens))
    invalid(request.chat ? "messages" : "prompt", "prompt and generation exceed the " + std::to_string(tokenizer.contract().context_tokens) + "-token context", "context_length_exceeded");
  request.prompt = std::make_shared<const std::vector<std::uint32_t>>(std::move(prompt));
  if (request.chat && request.operation == Operation::generate) {
    response_constraint(body, request, compiler);
    if (request.enforce_tool_calls || (request.allow_tool_calls && request.constraint)) {
      try {
        request.constraint = compiler.compile_tools(tools,
            request.allow_tool_calls ? request.tool_names : std::vector<std::string>{}, request.require_tool_calls,
            request.parallel_tool_calls, request.constraint);
      } catch (const std::invalid_argument& error) { invalid("tools", error.what(), "invalid_schema"); }
    }
  }
  return request;
}

json error_json(const Error& error) {
  return {{"error", {{"message", error.what()},
                     {"type", error.status >= 500 ? "server_error" : "invalid_request_error"},
                     {"param", error.param.empty() ? json(nullptr) : json(error.param)},
                     {"code", error.code.empty() ? json(nullptr) : json(error.code)}}}};
}

json model_json(const std::string& model, std::int64_t created) {
  return {{"id", model}, {"object", "model"}, {"created", created}, {"owned_by", "gewell"}};
}

json immediate_json(const Request& request, const std::string& model, std::int64_t created, bool ready) {
  if (request.operation == Operation::health) {
    if (!ready) throw Error(503, "model is not ready", {}, "model_not_ready");
    return {{"status", "ok"}, {"model", model}};
  }
  if (request.operation == Operation::model) return model_json(model, created);
  if (request.operation == Operation::models) return {{"object", "list"}, {"data", json::array({model_json(model, created)})}};
  execution_error("request is not an immediate operation");
}

json control_json(const Request& request, const Result& result) {
  if (request.operation == Operation::finish)
    return {{"prompt_id", request.cache.prompt_id}, {"released", true}};
  if (request.operation == Operation::prefill) {
    validate_accounting(request, result, 0, false);
    if (!result.completion_checkpoint) execution_error("prefill did not retain its endpoint");
    return {{"usage", usage(result)}, {"checkpoint_tokens", result.processed_tokens},
            {"retained", true}, {"prompt_id", request.cache.prompt_id.empty() ? json(nullptr) : json(request.cache.prompt_id)}};
  }
  if (request.operation == Operation::stats) {
    const auto& stats = result.cache_stats;
    return {{"gpu", pool_json(stats.gpu)}, {"cpu", pool_json(stats.cpu)},
            {"index", {{"capacity_bytes", stats.index_bytes}, {"used_bytes", stats.index_used}}},
            {"page_count", stats.page_count}, {"checkpoint_count", stats.checkpoint_count},
            {"execution_count", stats.execution_count}, {"copy_on_write_pages", stats.copy_on_write_pages}};
  }
  execution_error("request is not a cache control operation");
}

struct CompletionOutput::Impl {
  struct PendingLogprob {
    std::uint32_t token;
    json value;
    std::size_t begin = 0, end = 0, bytes = 0;
    bool assigned = false;
  };

  const text::Tokenizer& tokenizer;
  Request request;
  std::string model, id;
  std::int64_t created;
  std::size_t limit, pending_bytes = 0, logprob_bytes = 0;
  text::IncrementalTextDecoder raw_decoder;
  text::ToolOutputDecoder chat_decoder;
  text::StopMatcher stop_matcher;
  std::unique_ptr<constraint::State> constraint;
  std::string content, reasoning, visible_content;
  json content_logprobs = json::array();
  std::deque<PendingLogprob> pending_logprobs;
  std::size_t input_content_bytes = 0, visible_content_bytes = 0,
              emitted_content_bytes = 0;
  std::uint32_t seen = 0;
  bool stopped = false, started = false, finished = false, decoder_finished = false;

  Impl(const text::Tokenizer& tokenizer_value, const Request& request_value,
       std::string model_value, std::string id_value, std::int64_t time, std::size_t capacity)
      : tokenizer(tokenizer_value), request(request_value), model(std::move(model_value)),
        id(std::move(id_value)), created(time), limit(capacity), raw_decoder(tokenizer),
        chat_decoder(tokenizer, request.tool_names, "call_" + id + "_", request.allow_tool_calls, request.initial_reasoning),
        stop_matcher(request.stops) {
    if (request.operation != Operation::generate || !request.prompt || request.prompt->empty() ||
        !request.max_tokens || !limit || (request.logprobs && !request.chat))
      execution_error("invalid completion output configuration");
    if (request.constraint)
      constraint = std::make_unique<constraint::State>(request.constraint, request.enable_thinking,
                                                     request.initial_reasoning);
  }

  void bounded(std::size_t bytes) const {
    if (bytes > limit) throw Error(500, "completion output exceeds the configured byte limit", {}, "output_limit_exceeded");
  }

  void queue_logprob(std::uint32_t token, json value) {
    const auto bytes = value.dump().size() + 1;
    logprob_bytes += bytes;
    pending_logprobs.push_back({token, std::move(value), 0, 0, bytes, false});
  }

  // The text decoder can release a whole byte-fallback run at once. Re-decode
  // that bounded run to recover exact per-token byte spans in the content
  // stream before stop matching hides or releases them.
  void assign_logprob_spans(std::string_view decoded) {
    if (!request.logprobs || decoded.empty()) return;
    std::size_t first = 0;
    while (first < pending_logprobs.size() && pending_logprobs[first].assigned)
      ++first;
    if (first == pending_logprobs.size())
      execution_error("decoded content has no matching logprob tokens");

    std::vector<std::size_t> lengths(pending_logprobs.size() - first);
    std::string expected;
    for (std::size_t index = first; index < pending_logprobs.size();) {
      if (fallback_byte(tokenizer, pending_logprobs[index].token)) {
        const auto begin = index;
        std::string raw;
        text::IncrementalTextDecoder decoder(tokenizer);
        while (index < pending_logprobs.size()) {
          const auto byte = fallback_byte(tokenizer, pending_logprobs[index].token);
          if (!byte) break;
          raw.push_back(static_cast<char>(*byte));
          expected += decoder.push(pending_logprobs[index].token);
          ++index;
        }
        const auto flushed = decoder.finish();
        expected += flushed;
        const bool valid = flushed == raw;
        const auto width = valid ? std::size_t{1} : std::size_t{3};
        if (!valid && flushed.size() != 3 * (index - begin))
          execution_error("byte-fallback logprob decoding is inconsistent");
        for (auto item = begin; item < index; ++item)
          lengths[item - first] = width;
      } else {
        const auto text = token_text(tokenizer, pending_logprobs[index].token);
        expected += text;
        lengths[index - first] = text.size();
        ++index;
      }
    }
    if (expected != decoded)
      execution_error("decoded content does not match its logprob tokens");
    for (std::size_t index = first; index < pending_logprobs.size(); ++index) {
      auto& item = pending_logprobs[index];
      item.begin = input_content_bytes;
      input_content_bytes += lengths[index - first];
      item.end = input_content_bytes;
      item.assigned = true;
    }
  }

  json release_logprobs(std::string& released_content) {
    json released = json::array();
    auto release_through = emitted_content_bytes;
    while (!pending_logprobs.empty() && pending_logprobs.front().assigned) {
      auto& item = pending_logprobs.front();
      const bool complete = item.end <= visible_content_bytes;
      const bool partial = stop_matcher.matched() &&
                           item.begin < visible_content_bytes;
      if (!complete && !partial) break;
      release_through = partial ? visible_content_bytes
                                : std::max(release_through, item.end);
      if (request.stream) {
        released.push_back(std::move(item.value));
        logprob_bytes -= item.bytes;
      } else {
        content_logprobs.push_back(std::move(item.value));
      }
      pending_logprobs.pop_front();
    }
    if (stop_matcher.matched()) {
      release_through = visible_content_bytes;
      while (!pending_logprobs.empty()) {
        logprob_bytes -= pending_logprobs.front().bytes;
        pending_logprobs.pop_front();
      }
    } else if (pending_logprobs.empty()) {
      release_through = visible_content_bytes;
    }
    if (release_through < emitted_content_bytes ||
        release_through - emitted_content_bytes > visible_content.size())
      execution_error("logprob content visibility is inconsistent");
    const auto count = release_through - emitted_content_bytes;
    released_content.assign(visible_content, 0, count);
    visible_content.erase(0, count);
    emitted_content_bytes = release_through;
    return released;
  }

  json envelope() const {
    return {{"id", id}, {"object", request.chat ? "chat.completion.chunk" : "text_completion"},
            {"created", created}, {"model", model}};
  }

  std::string event(json value) const {
    if (request.include_usage && !value.contains("usage")) value["usage"] = nullptr;
    auto output = "data: " + value.dump() + "\n\n";
    bounded(output.size());
    return output;
  }

  std::string choice_event(json choice) const {
    auto value = envelope();
    value["choices"] = json::array({std::move(choice)});
    return event(std::move(value));
  }

  std::string emit(text::ToolDelta delta, bool match_content = true) {
    json released_logprobs = json::array();
    if (request.logprobs) {
      if (match_content) {
        assign_logprob_spans(delta.content);
        delta.content = stop_matcher.push(delta.content);
      }
      visible_content_bytes += delta.content.size();
      visible_content += delta.content;
      released_logprobs = release_logprobs(delta.content);
    } else if (match_content) {
      delta.content = stop_matcher.push(delta.content);
    }
    if (stop_matcher.matched()) stopped = true;
    if (!request.stream) {
      bounded(content.size() + reasoning.size() + delta.content.size() + delta.reasoning.size() +
              visible_content.size() + logprob_bytes + chat_decoder.retained_bytes() +
              stop_matcher.pending_bytes());
      content += delta.content;
      reasoning += delta.reasoning;
      return {};
    }
    std::string output;
    bool logprob_emitted = false;
    const auto event_logprobs = [&](bool content_event) {
      if (!content_event || released_logprobs.empty() || logprob_emitted)
        return json(nullptr);
      logprob_emitted = true;
      return json{{"content", released_logprobs}, {"refusal", nullptr}};
    };
    for (const auto& field : {std::pair<const char*, const std::string&>{"reasoning_content", delta.reasoning},
                             std::pair<const char*, const std::string&>{"content", delta.content}}) {
      if (field.second.empty()) continue;
      if (request.chat)
        output += choice_event({{"index", 0}, {"delta", {{field.first, field.second}}},
                                {"logprobs", event_logprobs(std::string_view(field.first) == "content")},
                                {"finish_reason", nullptr}});
      else output += choice_event({{"index", 0}, {"text", field.second}, {"logprobs", nullptr}, {"finish_reason", nullptr}});
    }
    if (!delta.tool_calls.empty()) {
      json calls = json::array();
      for (const auto& call : delta.tool_calls) {
        json value = {{"index", call.index}, {"function", {{"arguments", call.arguments}}}};
        if (!call.id.empty()) {
          value["id"] = call.id;
          value["type"] = "function";
          value["function"]["name"] = call.name;
        }
        calls.push_back(std::move(value));
      }
      output += choice_event({{"index", 0}, {"delta", {{"tool_calls", std::move(calls)}}},
                              {"logprobs", nullptr}, {"finish_reason", nullptr}});
    }
    if (!released_logprobs.empty() && !logprob_emitted)
      output += choice_event({{"index", 0}, {"delta", json::object()},
                              {"logprobs", event_logprobs(true)},
                              {"finish_reason", nullptr}});
    bounded(output.size() + visible_content.size() + logprob_bytes +
            chat_decoder.retained_bytes() + stop_matcher.pending_bytes());
    return output;
  }

  text::ToolDelta finish_decoder(bool eos) {
    if (decoder_finished) return {};
    decoder_finished = true;
    if (request.chat) {
      try { return chat_decoder.finish(eos); }
      catch (const std::invalid_argument& error) {
        throw Error(500, error.what(), {}, "invalid_tool_output");
      }
    }
    return {{}, raw_decoder.finish(), {}};
  }
};

CompletionOutput::CompletionOutput(const text::Tokenizer& tokenizer, const Request& request,
                                   std::string model, std::string id, std::int64_t created,
                                   std::size_t max_output_bytes)
    : impl_(std::make_unique<Impl>(tokenizer, request, std::move(model), std::move(id), created, max_output_bytes)) {}
CompletionOutput::~CompletionOutput() = default;

std::string CompletionOutput::start() {
  auto& state = *impl_;
  if (state.started || state.finished) execution_error("completion output already started/finished");
  state.started = true;
  if (!state.request.stream || !state.request.chat) return {};
  return state.choice_event({{"index", 0}, {"delta", {{"role", "assistant"}, {"content", ""}}},
                             {"logprobs", nullptr}, {"finish_reason", nullptr}});
}

std::string CompletionOutput::push(const std::uint32_t* tokens, std::size_t count,
                                   const TokenLogprobs* logprobs) {
  auto& state = *impl_;
  if (state.finished || (state.request.stream && !state.started)) execution_error("completion output is not active");
  if (count && !tokens) execution_error("generation returned a null token buffer");
  if (count && state.request.logprobs != (logprobs != nullptr))
    execution_error("generation returned mismatched logprob metadata");
  std::string output;
  for (std::size_t i = 0; i < count; ++i) {
    const auto token = tokens[i];
    if (token >= state.tokenizer.contract().vocabulary_size || state.stopped || state.seen == state.request.max_tokens)
      execution_error("generation returned an invalid or excess token");
    if (state.constraint && !state.constraint->accept(token))
      throw Error(500, "generation returned a token outside the response schema", {}, "invalid_schema_output");
    json token_logprob;
    if (state.request.logprobs && !state.tokenizer.is_special(token) &&
        state.chat_decoder.in_content())
      token_logprob = logprob_record(state.tokenizer, token, logprobs[i],
                                    state.request.top_logprobs);
    if (!token_logprob.is_null())
      state.queue_logprob(token, std::move(token_logprob));
    ++state.seen;
    if (state.tokenizer.contract().is_stop(token)) {
      if (token == state.tokenizer.contract().tool_handoff && state.request.chat) {
        try { output += state.emit(state.chat_decoder.push(token)); }
        catch (const std::invalid_argument& error) { throw Error(500, error.what(), {}, "invalid_tool_output"); }
      }
      // Flush while this token is still owned by the scheduler so stop feedback
      // and parser failures arrive before it finalizes request accounting.
      output += state.emit(state.finish_decoder(true));
      state.stopped = true;
      continue;
    }
    const auto& piece = state.tokenizer.token_piece(token);
    if (piece.size() == 6 && piece.compare(0, 3, "<0x") == 0 && piece.back() == '>') {
      ++state.pending_bytes;
    } else state.pending_bytes = 0;
    text::ToolDelta delta;
    if (state.request.chat) {
      try { delta = state.chat_decoder.push(token); }
      catch (const std::invalid_argument& error) { throw Error(500, error.what(), {}, "invalid_tool_output"); }
    } else delta.content = state.raw_decoder.push(token);
    state.bounded(state.pending_bytes + state.content.size() + state.reasoning.size() +
                  state.chat_decoder.retained_bytes() + state.stop_matcher.pending_bytes());
    output += state.emit(std::move(delta));
    if (!state.request.stops.empty() && !state.stop_matcher.matched()) {
      const auto pending = state.request.chat ? state.chat_decoder.pending().content : state.raw_decoder.pending();
      if (state.stop_matcher.would_match(pending)) output += state.emit(state.finish_decoder(false));
    }
    if (!state.decoder_finished && state.seen == state.request.max_tokens)
      output += state.emit(state.finish_decoder(false));
    state.bounded(output.size() + state.chat_decoder.retained_bytes() + state.stop_matcher.pending_bytes());
  }
  return output;
}

bool CompletionOutput::stop_matched() const { return impl_->stop_matcher.matched(); }

std::string CompletionOutput::finish(const Result& result) {
  auto& state = *impl_;
  if (state.finished || (state.request.stream && !state.started)) execution_error("completion output is not active");
  state.finished = true;
  validate_accounting(state.request, result, state.seen, state.stopped);
  auto output = state.emit(state.finish_decoder(state.stopped && !state.stop_matcher.matched()));
  text::ToolDelta tail;
  tail.content = state.stop_matcher.finish();
  output += state.emit(std::move(tail), false);
  const char* reason = state.chat_decoder.tool_handoff() ? "tool_calls" : state.stopped ? "stop" : "length";
  if (state.request.stream) {
    if (state.request.chat) output += state.choice_event({{"index", 0}, {"delta", json::object()}, {"logprobs", nullptr}, {"finish_reason", reason}});
    else output += state.choice_event({{"index", 0}, {"text", ""}, {"logprobs", nullptr}, {"finish_reason", reason}});
    if (state.request.include_usage) {
      auto value = state.envelope();
      value["choices"] = json::array();
      value["usage"] = usage(result);
      output += state.event(std::move(value));
    }
    output += "data: [DONE]\n\n";
  } else {
    auto value = state.envelope();
    json choice = {{"index", 0}, {"logprobs", nullptr}, {"finish_reason", reason}};
    if (state.request.chat) {
      value["object"] = "chat.completion";
      if (state.request.logprobs)
        choice["logprobs"] = {{"content", std::move(state.content_logprobs)},
                              {"refusal", nullptr}};
      choice["message"] = {{"role", "assistant"}, {"content", state.content}, {"refusal", nullptr}};
      if (!state.reasoning.empty()) choice["message"]["reasoning_content"] = state.reasoning;
      if (!state.chat_decoder.calls().empty()) {
        choice["message"]["tool_calls"] = json::array();
        for (const auto& call : state.chat_decoder.calls())
          choice["message"]["tool_calls"].push_back({{"id", call.id}, {"type", "function"},
              {"function", {{"name", call.name}, {"arguments", call.arguments}}}});
        if (state.content.empty()) choice["message"]["content"] = nullptr;
      }
    } else choice["text"] = state.content;
    value["choices"] = json::array({std::move(choice)});
    value["usage"] = usage(result);
    output = value.dump();
  }
  state.bounded(output.size());
  return output;
}

}  // namespace gewell::http
