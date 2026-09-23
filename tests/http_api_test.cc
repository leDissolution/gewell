#include "gewell/models/gemma4/text_contract.h"
#include "gewell/models/gemma4/31b/model.h"
#include "text_contract_fixture.h"
#include "gewell/http_api.h"
#include "gewell/tool_output.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
const auto& contract = gewell::gemma4::text_contract_31b();
using nlohmann::json;
namespace http = gewell::http;
namespace text = gewell::text;
namespace constraint = gewell::constraint;
const std::string kModel = "google/gemma-4-31B-it";

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Function>
void rejects(int status, const std::string& param, Function action) {
  try { action(); }
  catch (const http::Error& error) {
    require(error.status == status, "wrong error status: " + std::string(error.what()));
    require(error.param == param, "wrong error param: " + error.param + " expected " + param);
    const auto envelope = http::error_json(error);
    require(envelope.at("error").contains("code") && envelope.at("error").contains("type"), "incomplete error envelope");
    return;
  }
  throw std::runtime_error("invalid request/output was accepted: expected " + std::to_string(status) + " " + param);
}

http::Request parse(const text::Tokenizer& tokenizer, constraint::Compiler& compiler, const json& body,
                    std::string_view route = "/v1/completions") {
  return http::parse_request("POST", route, body.dump(), tokenizer, kModel, compiler);
}

http::Result accounting(const http::Request& request, std::size_t tokens, bool stopped) {
  http::Result result;
  result.prompt_tokens = request.prompt->size();
  result.completion_tokens = tokens;
  result.processed_tokens = result.prompt_tokens + tokens - (tokens != 0);
  result.cached_tokens = result.prompt_tokens;
  result.stopped = stopped;
  return result;
}

std::vector<json> events(const std::string& stream, bool done) {
  std::vector<json> result;
  std::size_t offset = 0;
  bool done_seen = false;
  while (offset < stream.size()) {
    const auto end = stream.find("\n\n", offset);
    require(end != std::string::npos, "unterminated SSE event");
    const auto line = stream.substr(offset, end - offset);
    require(line.compare(0, 6, "data: ") == 0, "missing SSE data prefix");
    require(!done_seen, "event followed DONE");
    if (line == "data: [DONE]") done_seen = true;
    else result.push_back(json::parse(line.substr(6)));
    offset = end + 2;
  }
  require(done_seen == done, "wrong DONE termination");
  return result;
}

void request_cases(const text::Tokenizer& tokenizer, constraint::Compiler& compiler) {
  const auto defaults = parse(tokenizer, compiler, {{"model", kModel}, {"prompt", ""}});
  require(*defaults.prompt == std::vector<std::uint32_t>{2} && defaults.max_tokens == 256 &&
      defaults.temperature == 1 && defaults.top_p == 1 && defaults.top_k == 0 &&
      !defaults.stream && !defaults.logprobs && defaults.top_logprobs == 5,
      "completion defaults differ");
  auto raw = json{{"model", kModel}, {"prompt", json::array({7, 8})}, {"n", 1},
      {"presence_penalty", 0}, {"frequency_penalty", 0.0}, {"logit_bias", json::object()},
      {"echo", false}, {"best_of", 1}, {"logprobs", nullptr}, {"stop", nullptr}, {"seed", nullptr},
      {"suffix", nullptr}, {"user", "client-id"}, {"temperature", 2}, {"top_p", 0}, {"top_k", 4294967295ULL}};
  require(*parse(tokenizer, compiler, raw).prompt == std::vector<std::uint32_t>({7, 8}), "raw token IDs changed");
  const auto chat_body = json{{"model", kModel}, {"messages", json::array({{{"role", "user"}, {"content", "Hi"}}})},
      {"max_tokens", 8}, {"max_completion_tokens", 4}, {"logprobs", false},
      {"response_format", {{"type", "text"}}}, {"store", false}, {"modalities", json::array({"text"})},
      {"tools", json::array()}, {"functions", nullptr}, {"parallel_tool_calls", nullptr},
      {"chat_template_kwargs", {{"enable_thinking", true}, {"preserve_thinking", true}}}};
  const auto chat = parse(tokenizer, compiler, chat_body, "/v1/chat/completions");
  require(chat.chat && chat.max_tokens == 4 && chat.prompt->front() == 2, "chat preparation differs");
  for (bool is_chat : {false, true}) {
    auto extended = is_chat ? chat_body : raw;
    extended["id_slot"] = 7;
    extended["cache_prompt"] = true;
    extended["vendor_options"] = {{"nested", json::array({nullptr, false, "future"})}};
    extended["unknown"] = nullptr;
    extended["metadata"] = {{"pipeline", "dsgen"}};
    extended["safety_identifier"] = "client-id";
    extended["prompt_cache_key"] = "routing-hint";
    const auto route = is_chat ? "/v1/chat/completions" : "/v1/completions";
    const auto expected = parse(tokenizer, compiler, is_chat ? chat_body : raw, route);
    const auto actual = parse(tokenizer, compiler, extended, route);
    require(*actual.prompt == *expected.prompt && actual.max_tokens == expected.max_tokens &&
        actual.temperature == expected.temperature && actual.top_k == expected.top_k &&
        actual.top_p == expected.top_p && actual.cache.prompt_id.empty(),
        "ignored extension changed request semantics");
    if (!is_chat) {
      extended["logprobs"] = 1;
      rejects(400, "logprobs", [&] { (void)parse(tokenizer, compiler, extended, route); });
    }
  }
  auto scored_chat = chat_body;
  scored_chat["logprobs"] = true;
  const auto default_scores = parse(tokenizer, compiler, scored_chat, "/v1/chat/completions");
  require(default_scores.logprobs && default_scores.top_logprobs == 5,
          "chat logprob defaults differ");
  scored_chat["top_logprobs"] = 20;
  require(parse(tokenizer, compiler, scored_chat, "/v1/chat/completions").top_logprobs == 20,
          "chat top_logprobs was not retained");
  auto nullable = raw;
  for (const char* key : {"temperature", "top_p", "top_k", "max_tokens", "stream", "cache"}) nullable[key] = nullptr;
  require(parse(tokenizer, compiler, nullable).max_tokens == 256, "optional null did not default");

  const std::vector<std::pair<std::string, json>> bad_raw = {
      {"prompt", json::array()}, {"prompt", true},
      {"max_tokens", 0}, {"max_tokens", true}, {"max_tokens", 1.5}, {"max_tokens", 262145},
      {"temperature", true}, {"temperature", -1}, {"temperature", 2.1},
      {"top_p", 1.1}, {"top_k", -1}, {"top_k", true}, {"top_k", 4294967296ULL}, {"stream", "true"},
      {"n", 2}, {"n", true}, {"frequency_penalty", 1}, {"presence_penalty", false},
      {"logit_bias", {{"2", 1}}}, {"stop", ""}, {"seed", 1.0},
      {"echo", true}, {"best_of", 2}, {"logprobs", 0}, {"logprobs", false},
      {"suffix", "tail"}, {"max_completion_tokens", 4},
      {"response_format", {{"type", "json_object"}}}, {"tools", json::array()},
      {"user", 4}, {"stream_options", {{"include_usage", true}}}};
  for (const auto& [key, value] : bad_raw) {
    auto body = json{{"model", kModel}, {"prompt", "Hi"}};
    body[key] = value;
    rejects(400, key, [&] { (void)parse(tokenizer, compiler, body); });
  }
  for (const auto& value : {json(true), json(-1), json(262144), json(2.0), json::array({2})}) {
    rejects(400, "prompt[0]", [&] { (void)parse(tokenizer, compiler, {{"model", kModel}, {"prompt", json::array({value})}}); });
  }
  for (const auto& [key, value] : std::vector<std::pair<std::string, json>>{
      {"logprobs", 0}, {"top_logprobs", 1}, {"tools", json::array({json::object()})},
      {"tool_choice", "required"}, {"parallel_tool_calls", "false"},
      {"modalities", json::array({"audio"})}, {"store", true}, {"echo", false}}) {
    auto body = chat_body;
    body[key] = value;
    rejects(400, key, [&] { (void)parse(tokenizer, compiler, body, "/v1/chat/completions"); });
  }
  for (const auto& value : {json(-1), json(21), json(true), json(1.5), json("5")}) {
    auto body = chat_body;
    body["logprobs"] = true;
    body["top_logprobs"] = value;
    rejects(400, "top_logprobs", [&] {
      (void)parse(tokenizer, compiler, body, "/v1/chat/completions");
    });
  }
  for (const char* key : {"audio", "moderation", "prediction", "prompt_cache_retention",
                         "service_tier", "verbosity", "web_search_options"}) {
    auto body = chat_body;
    body["id_slot"] = 42;
    body[key] = "requested";
    rejects(400, key, [&] { (void)parse(tokenizer, compiler, body, "/v1/chat/completions"); });
    body[key] = nullptr;
    (void)parse(tokenizer, compiler, body, "/v1/chat/completions");
  }
  rejects(400, "model", [&] { (void)parse(tokenizer, compiler, {{"prompt", "Hi"}}); });
  rejects(404, "model", [&] { (void)parse(tokenizer, compiler, {{"model", "other"}, {"prompt", "Hi"}}); });
  rejects(400, "messages", [&] { (void)parse(tokenizer, compiler, {{"model", kModel}, {"messages", json::array({{{"role", "tool"}, {"content", "x"}}})}}, "/v1/chat/completions"); });
  rejects(400, "chat_template_kwargs", [&] { auto body = chat_body; body["chat_template_kwargs"] = {{"wrong", true}}; (void)parse(tokenizer, compiler, body, "/v1/chat/completions"); });
  rejects(400, "stream_options.include_usage", [&] { auto body = raw; body["stream"] = true; body["stream_options"] = {{"include_usage", 1}}; (void)parse(tokenizer, compiler, body); });
  rejects(400, "stream_options.wrong", [&] { auto body = raw; body["stream"] = true; body["stream_options"] = {{"wrong", nullptr}}; (void)parse(tokenizer, compiler, body); });
  std::vector<std::uint32_t> full_context(262144, 7);
  require(parse(tokenizer, compiler, {{"model", kModel}, {"prompt", full_context}, {"max_tokens", 1}}).prompt->size() == full_context.size(), "exact context boundary rejected");
  rejects(400, "prompt", [&] { (void)parse(tokenizer, compiler, {{"model", kModel}, {"prompt", full_context}, {"max_tokens", 2}}); });

  for (const auto& [path, operation] : std::vector<std::pair<std::string, http::Operation>>{
      {"/health", http::Operation::health}, {"/v1/models", http::Operation::models},
      {"/v1/models/" + kModel, http::Operation::model},
      {"/v1/models/google%2Fgemma-4-31B-it", http::Operation::model},
      {"/v1/cache/stats", http::Operation::stats},
      {"/metrics", http::Operation::metrics}, {"/v1/cache/index", http::Operation::cache_index}}) {
    require(http::parse_request("GET", path, "", tokenizer, kModel, compiler).operation == operation, "GET route mismatch");
    rejects(405, "", [&] { (void)http::parse_request("POST", path, "{}", tokenizer, kModel, compiler); });
  }
  rejects(404, "", [&] { (void)http::parse_request("GET", "/v1/responses", "", tokenizer, kModel, compiler); });
  rejects(404, "model", [&] { (void)http::parse_request("GET", "/v1/models/other", "", tokenizer, kModel, compiler); });
  rejects(400, "body", [&] { (void)http::parse_request("GET", "/health", "{}", tokenizer, kModel, compiler); });
  rejects(400, "", [&] { (void)http::parse_request("POST", "/v1/completions", "{", tokenizer, kModel, compiler); });
  const auto health = http::parse_request("GET", "/health", "", tokenizer, kModel, compiler);
  rejects(503, "", [&] { (void)http::immediate_json(health, kModel, 42, false); });
  require(http::immediate_json(health, kModel, 42, true)["status"] == "ok", "ready health mismatch");
  const auto models = http::parse_request("GET", "/v1/models", "", tokenizer, kModel, compiler);
  require(http::immediate_json(models, kModel, 42, true)["data"][0] == http::model_json(kModel, 42), "model list mismatch");
}

void reasoning_effort_cases(const text::Tokenizer& tokenizer, constraint::Compiler& compiler) {
  const json messages = json::array({{{"role", "user"}, {"content", "Hi"}}});
  const json base = {{"model", kModel}, {"messages", messages}};
  require(!parse(tokenizer, compiler, base, "/v1/chat/completions").enable_thinking,
          "omitted reasoning effort enabled thinking by default");
  for (const auto& effort : std::vector<json>{nullptr, "none", "low", "medium", "high"}) {
    auto body = base;
    body["reasoning_effort"] = effort;
    const bool thinking = !effort.is_null() && effort != "none";
    const auto request = parse(tokenizer, compiler, body, "/v1/chat/completions");
    auto native = base;
    native["chat_template_kwargs"] = {{"enable_thinking", thinking}};
    require(request.enable_thinking == thinking &&
                *request.prompt == *parse(tokenizer, compiler, native, "/v1/chat/completions").prompt,
            "reasoning effort differs from native thinking prompt");
    const auto prefill = parse(tokenizer, compiler, body, "/v1/cache/prefill");
    require(prefill.enable_thinking == thinking && *prefill.prompt == *request.prompt,
            "reasoning effort prefill and generation prompts differ");
    if (!effort.is_null()) {
      body["chat_template_kwargs"] = {{"enable_thinking", !thinking}, {"preserve_thinking", true}};
      native["chat_template_kwargs"]["preserve_thinking"] = true;
      const auto overridden = parse(tokenizer, compiler, body, "/v1/chat/completions");
      require(overridden.enable_thinking == thinking &&
                  *overridden.prompt == *parse(tokenizer, compiler, native, "/v1/chat/completions").prompt,
              "explicit reasoning effort did not override enable_thinking");
    }
  }
  for (bool supplied_null : {false, true}) {
    auto body = base;
    body["chat_template_kwargs"] = {{"enable_thinking", true}};
    if (supplied_null) body["reasoning_effort"] = nullptr;
    require(parse(tokenizer, compiler, body, "/v1/chat/completions").enable_thinking,
            "omitted/null reasoning effort changed explicit template setting");
  }
  for (const auto& value : std::vector<json>{"", "minimal", "xhigh", "HIGH", true, 1, json::array(), json::object()}) {
    auto body = base;
    body["reasoning_effort"] = value;
    for (const auto* route : {"/v1/chat/completions", "/v1/cache/prefill"})
      rejects(400, "reasoning_effort", [&] { (void)parse(tokenizer, compiler, body, route); });
  }
  for (const auto* route : {"/v1/completions", "/v1/cache/prefill"})
    rejects(400, "reasoning_effort", [&] {
      (void)parse(tokenizer, compiler, {{"model", kModel}, {"prompt", "Hi"}, {"reasoning_effort", "low"}}, route);
    });
}

void image_request_cases(const text::Tokenizer& tokenizer, constraint::Compiler& compiler) {
  namespace model = gewell::gemma4_31b;
  unsigned prepared = 0;
  std::vector<std::weak_ptr<gewell::runtime::ImageInput>> allocations;
  http::ImageSupport images;
  images.begin_token = model::kBeginImageTokenId;
  images.image_token = model::kImageTokenId;
  images.end_token = model::kEndImageTokenId;
  images.max_image_tokens = 1120;
  images.prepare = [&](std::string_view url) {
    ++prepared;
    const bool second = url == "data:image/png;base64,second";
    if (!second && url != "data:image/png;base64,fixture") throw std::invalid_argument("invalid fixture image");
    auto image = std::make_shared<gewell::runtime::ImageInput>();
    image->pixels = {std::uint8_t(second ? 9 : 1), 2, 3, 4};
    image->positions = {5, 6, 7, 8};
    image->padded_patch_rows = 2520;
    image->end = second ? 3 : 6;
    allocations.push_back(image);
    return image;
  };
  const json part{{"type", "image_url"}, {"image_url", {{"url", "data:image/png;base64,fixture"}, {"detail", "auto"}}}};
  auto second_part = part;
  second_part["image_url"]["url"] = "data:image/png;base64,second";
  const json user{{"role", "user"}, {"content", json::array({part, {{"type", "text"}, {"text", "Describe this image."}}})}};
  const json original{{"model", kModel}, {"messages", json::array({user})},
      {"max_tokens", 8}, {"seed", 7}, {"logprobs", true}};
  auto parse_image = [&](const json& body, std::string_view route = "/v1/chat/completions") {
    return http::parse_request("POST", route, body.dump(), tokenizer, kModel, compiler, images);
  };
  const auto request = parse_image(original);
  require(prepared == 1 && request.images.size() == 1 && request.images[0]->end - request.images[0]->begin == 6 &&
      request.prompt->at(request.images[0]->begin - 1) == model::kBeginImageTokenId &&
      request.prompt->at(request.images[0]->end) == model::kEndImageTokenId &&
      request.images[0]->pixels == std::vector<std::uint8_t>({1, 2, 3, 4}) && request.logprobs && request.seed == 7,
      "image span, ownership, or ordinary generation controls changed");
  auto multiple = original;
  multiple["messages"][0]["content"] = json::array({
      {{"type", "text"}, {"text", "Before"}}, part,
      {{"type", "text"}, {"text", "Between"}}, second_part,
      {{"type", "text"}, {"text", "After"}}});
  auto placeholder = [&](unsigned count) {
    auto result = tokenizer.token_piece(images.begin_token);
    while (count--) result += tokenizer.token_piece(images.image_token);
    return result + tokenizer.token_piece(images.end_token);
  };
  auto verify_order = [&](const json& body, const std::vector<unsigned>& counts,
                          const std::vector<std::uint8_t>& pixels) {
    const auto parsed = parse_image(body);
    require(parsed.images.size() == counts.size(), "ordered images missing from request");
    std::uint32_t last_end = 0;
    for (std::size_t i = 0; i < counts.size(); ++i) {
      const auto& image = *parsed.images[i];
      require(image.begin > last_end && image.end - image.begin == counts[i] && image.pixels.front() == pixels[i] &&
          parsed.prompt->at(image.begin - 1) == images.begin_token && parsed.prompt->at(image.end) == images.end_token,
          "image tensor order or absolute feature span changed");
      last_end = image.end;
    }
    return parsed;
  };
  const auto ordered = verify_order(multiple, {6, 3}, {1, 9});
  const json expected_messages = json::array({{{"role", "user"},
      {"content", "Before" + placeholder(6) + "Between" + placeholder(3) + "After"}}});
  require(*ordered.prompt == tokenizer.encode(tokenizer.contract().render_chat(
      tokenizer.contract().normalize_messages(expected_messages), {}, json::array())),
      "text/image interleaving was reordered");
  std::swap(multiple["messages"][0]["content"][1], multiple["messages"][0]["content"][3]);
  verify_order(multiple, {3, 6}, {9, 1});
  auto history = original;
  history["messages"].insert(history["messages"].begin(), json{{"role", "system"}, {"content", "Be concise."}});
  history["messages"].push_back({{"role", "assistant"}, {"content", "Earlier caption."}});
  history["messages"].push_back({{"role", "user"}, {"content", json::array({second_part,
      {{"type", "text"}, {"text", "Compare it to the earlier image."}}})}});
  verify_order(history, {6, 3}, {1, 9});
  history["messages"].push_back({{"role", "assistant"}, {"content", "They differ."}});
  history["messages"].push_back({{"role", "user"}, {"content", "Which came first?"}});
  verify_order(history, {6, 3}, {1, 9});
  auto bad = original;
  const auto prepared_before_invalid = prepared;
  bad["messages"].push_back({{"role", "assistant"}, {"content", json::array({part})}});
  rejects(400, "messages[1].content[0]", [&] { (void)parse_image(bad); });
  bad = original; bad["messages"].push_back({{"role", "user"}, {"content", 42}});
  rejects(400, "messages", [&] { (void)parse_image(bad); });
  require(prepared == prepared_before_invalid, "invalid later message allocated image tensors before validation");
  for (const auto& role : {"system", "assistant", "tool"}) {
    bad = original; bad["messages"][0]["role"] = role;
    rejects(400, "messages[0].content[0]", [&] { (void)parse_image(bad); });
  }
  auto controlled = original;
  controlled["cache"] = {{"mode", "auto"}, {"prompt_id", "image-owner"}, {"priority", "high"}, {"finished", true}};
  const auto owned = parse_image(controlled);
  require(owned.cache.prompt_id == "image-owner" && owned.cache.priority == http::Priority::high &&
      owned.cache.finished && !owned.cache.reuse_only && owned.images.size() == 1,
      "image cache owner controls were lost");
  controlled["cache"] = {{"mode", "reuse_only"}};
  require(parse_image(controlled).cache.reuse_only, "image reuse-only control was lost");
  controlled["cache"] = {{"mode", "reuse_only"}, {"prompt_id", "image-owner"}};
  rejects(400, "cache", [&] { (void)parse_image(controlled); });
  for (const auto& detail : {json("low"), json("high"), json(1)}) {
    bad = original; bad["messages"][0]["content"][0]["image_url"]["detail"] = detail;
    rejects(400, "messages[0].content[0].image_url.detail", [&] { (void)parse_image(bad); });
  }
  bad = original; bad["messages"][0]["content"][0]["image_url"]["url"] = "https://example.invalid/image.png";
  rejects(400, "messages[0].content[0].image_url.url", [&] { (void)parse_image(bad); });
  bad = original; bad["messages"][0]["content"].push_back({{"type", "image_url"}, {"image_url", {{"url", "invalid"}}}});
  const auto allocated_before_failure = allocations.size();
  rejects(400, "messages[0].content[2].image_url.url", [&] { (void)parse_image(bad); });
  require(allocations.size() == allocated_before_failure + 1 && allocations.back().expired(),
      "later image failure did not release the earlier prepared tensor");
  auto warm_body = multiple;
  warm_body.erase("seed"); warm_body.erase("max_tokens"); warm_body.erase("logprobs");
  warm_body["cache"] = {{"prompt_id", "image-warm"}};
  const auto warm = parse_image(warm_body, "/v1/cache/prefill");
  require(warm.operation == http::Operation::prefill && warm.images.size() == 2 &&
      warm.max_tokens == 0 && warm.cache.prompt_id == "image-warm",
      "image cache prefill lost images or owner controls");
  warm_body["cache"] = {{"mode", "reuse_only"}};
  rejects(400, "cache", [&] { (void)parse_image(warm_body, "/v1/cache/prefill"); });
  auto long_prompt = original;
  std::string padding;
  for (unsigned i = 0; i < 1400; ++i) padding += "<pad>";
  long_prompt["messages"][0]["content"][1]["text"] = padding;
  require(parse_image(long_prompt).prompt->size() > 1280, "long image prompt was rejected or truncated");
  long_prompt["max_tokens"] = tokenizer.contract().context_tokens;
  rejects(400, "messages", [&] { (void)parse_image(long_prompt); });
  images.max_image_tokens = 5;
  rejects(500, "", [&] { (void)parse_image(original); });
  images.max_image_tokens = 1120;
  for (const auto token : {images.begin_token, images.image_token, images.end_token}) {
    bad = multiple; bad["messages"][0]["content"][2]["text"] = tokenizer.token_piece(token);
    rejects(400, "messages", [&] { (void)parse_image(bad); });
    rejects(400, "prompt", [&] { (void)parse_image({{"model", kModel}, {"prompt", json::array({2, token})}}, "/v1/completions"); });
  }
  bad = multiple; bad["messages"][0]["content"][2]["text"] = placeholder(3);
  rejects(400, "messages", [&] { (void)parse_image(bad); });
}

void cache_cases(const text::Tokenizer& tokenizer, constraint::Compiler& compiler) {
  auto body = json{{"model", kModel}, {"prompt", "Hi"}, {"cache", {{"prompt_id", "owner"}, {"priority", "high"}, {"finished", true}}}};
  require(parse(tokenizer, compiler, body).cache.finished, "cache finished lost");
  for (const auto& [value, param] : std::vector<std::pair<json, std::string>>{
      {json::array(), "cache"}, {{{"unknown", nullptr}}, "cache.unknown"},
      {{{"mode", "bad"}}, "cache.mode"}, {{{"mode", "reuse_only"}, {"prompt_id", "owner"}}, "cache"},
      {{{"priority", "bad"}}, "cache.priority"}, {{{"finished", true}}, "cache.finished"},
      {{{"finished", 1}}, "cache.finished"}, {{{"prompt_id", ""}}, "cache.prompt_id"},
      {{{"prompt_id", std::string(129, 'x')}}, "cache.prompt_id"}, {{{"prompt_id", std::string("x\0y", 3)}}, "cache.prompt_id"}}) {
    body["cache"] = value;
    rejects(400, param, [&] { (void)parse(tokenizer, compiler, body); });
  }
  auto warm_body = json{{"prompt", json::array({2, 7})}, {"cache", {{"prompt_id", "warm"}}}};
  const auto warm = parse(tokenizer, compiler, warm_body, "/v1/cache/prefill");
  auto result = accounting(warm, 0, false);
  result.completion_checkpoint = 9;
  const auto warmed = http::control_json(warm, result);
  require(warmed["retained"] == true && warmed["checkpoint_tokens"] == 2 && warmed["usage"]["completion_tokens"] == 0, "prefill response mismatch");
  result.completion_checkpoint = 0;
  rejects(500, "", [&] { (void)http::control_json(warm, result); });
  for (const char* field : {"stream", "temperature", "max_tokens"}) {
    auto invalid = warm_body; invalid[field] = nullptr;
    rejects(400, field, [&] { (void)parse(tokenizer, compiler, invalid, "/v1/cache/prefill"); });
  }
  rejects(400, "cache", [&] { auto invalid = warm_body; invalid["cache"] = {{"mode", "reuse_only"}}; (void)parse(tokenizer, compiler, invalid, "/v1/cache/prefill"); });
  rejects(400, "prompt", [&] { auto invalid = warm_body; invalid["messages"] = json::array(); (void)parse(tokenizer, compiler, invalid, "/v1/cache/prefill"); });
  const auto release = parse(tokenizer, compiler, {{"prompt_id", "warm"}}, "/v1/cache/finish");
  require(http::control_json(release, {}) == json{{"prompt_id", "warm"}, {"released", true}}, "finish response mismatch");
  rejects(400, "model", [&] { (void)parse(tokenizer, compiler, {{"model", kModel}, {"prompt_id", "warm"}}, "/v1/cache/finish"); });
  const auto stats_request = http::parse_request("GET", "/v1/cache/stats", "", tokenizer, kModel, compiler);
  result.cache_stats.gpu = {100, 20, 30, 80};
  result.cache_stats.index_bytes = 500;
  const auto stats = http::control_json(stats_request, result);
  require(stats["gpu"]["free_bytes"] == 80 && stats["index"]["capacity_bytes"] == 500, "stats mapping differs");
}

void completion_cases(const text::Tokenizer& tokenizer, constraint::Compiler& compiler) {
  auto ids = tokenizer.encode("Hello world");
  ids.push_back(1);
  for (bool chat : {false, true}) {
    for (bool include_usage : {false, true}) {
      auto body = json{{"model", kModel}, {"max_tokens", ids.size()}, {"stream", true},
                      {"stream_options", {{"include_usage", include_usage}}}};
      if (chat) body["messages"] = json::array({{{"role", "user"}, {"content", "Hi"}}});
      else body["prompt"] = json::array({2, 7});
      const auto request = parse(tokenizer, compiler, body, chat ? "/v1/chat/completions" : "/v1/completions");
      http::CompletionOutput output(tokenizer, request, kModel, "id", 42, 1024 * 1024);
      std::string stream = output.start();
      const auto first = output.push(ids.data(), 1);
      require(!first.empty(), "first text was buffered until generation completion");
      stream += first;
      stream += output.push(ids.data() + 1, ids.size() - 1);
      stream += output.finish(accounting(request, ids.size(), true));
      const auto chunks = events(stream, true);
      std::string reconstructed;
      for (const auto& chunk : chunks) {
        require(chunk["id"] == "id" && chunk["created"] == 42 && chunk["model"] == kModel, "SSE identity changed");
        if (!chunk["choices"].empty()) {
          const auto& choice = chunk["choices"][0];
          reconstructed += chat ? choice["delta"].value("content", std::string{}) : choice["text"].get<std::string>();
          require(include_usage ? chunk.contains("usage") && chunk["usage"].is_null() : !chunk.contains("usage"), "wrong per-chunk usage behavior");
        }
      }
      require(reconstructed == "Hello world", "EOS leaked or streamed text changed");
      const auto terminal = chunks[chunks.size() - (include_usage ? 2 : 1)];
      require(terminal["choices"][0]["finish_reason"] == "stop", "last-token EOS did not win over length");
      if (include_usage) require(chunks.back()["choices"].empty() && chunks.back()["usage"]["completion_tokens"] == ids.size(), "missing final usage");
      body["stream"] = false; body.erase("stream_options");
      const auto buffered = parse(tokenizer, compiler, body, chat ? "/v1/chat/completions" : "/v1/completions");
      http::CompletionOutput complete(tokenizer, buffered, kModel, "id", 42, 1024 * 1024);
      require(complete.push(ids.data(), ids.size()).empty(), "buffered push returned SSE");
      const auto document = json::parse(complete.finish(accounting(buffered, ids.size(), true)));
      require(document["choices"][0][chat ? "message" : "text"] == (chat ? json{{"role", "assistant"}, {"content", reconstructed}, {"refusal", nullptr}} : json(reconstructed)), "buffered/streamed text differs");
      require(document["usage"]["completion_tokens"] == ids.size(), "hidden EOS not counted");
    }
  }
  auto reasoning = std::vector<std::uint32_t>{100};
  const auto thought = tokenizer.encode("thought\nplan"); reasoning.insert(reasoning.end(), thought.begin(), thought.end());
  reasoning.push_back(101);
  const auto answer = tokenizer.encode("answer"); reasoning.insert(reasoning.end(), answer.begin(), answer.end());
  reasoning.push_back(106);
  const auto chat = parse(tokenizer, compiler, {{"model", kModel}, {"max_tokens", reasoning.size()},
      {"messages", json::array({{{"role", "user"}, {"content", "Hi"}}})}}, "/v1/chat/completions");
  http::CompletionOutput separated(tokenizer, chat, kModel, "id", 42, 10000);
  separated.push(reasoning.data(), reasoning.size());
  const auto message = json::parse(separated.finish(accounting(chat, reasoning.size(), true)))["choices"][0]["message"];
  require(message["content"] == "answer" && message["reasoning_content"] == "plan", "reasoning was not separated");

  const auto request = parse(tokenizer, compiler, {{"model", kModel}, {"prompt", "Hi"}, {"max_tokens", 1}, {"stream", true}});
  for (int failure = 0; failure < 5; ++failure) {
    http::CompletionOutput output(tokenizer, request, kModel, "id", 42, 10000);
    output.start(); const std::uint32_t token = 107; output.push(&token, 1);
    auto result = accounting(request, 1, false);
    if (failure == 0) ++result.prompt_tokens;
    if (failure == 1) ++result.completion_tokens;
    if (failure == 2) ++result.processed_tokens;
    if (failure == 3) result.cached_tokens = result.prompt_tokens + 1;
    if (failure == 4) result.stopped = true;
    rejects(500, "", [&] { (void)output.finish(result); });
  }
  http::CompletionOutput length(tokenizer, request, kModel, "id", 42, 10000);
  length.start(); const std::uint32_t token = 107; length.push(&token, 1);
  require(events(length.finish(accounting(request, 1, false)), true)[0]["choices"][0]["finish_reason"] == "length", "length finish mismatch");
  http::CompletionOutput excess(tokenizer, request, kModel, "id", 42, 10000);
  excess.start(); const std::uint32_t overflow[] = {1, 107};
  rejects(500, "", [&] { (void)excess.push(overflow, 2); });
  auto buffered = request; buffered.stream = false;
  http::CompletionOutput bounded(tokenizer, buffered, kModel, "id", 42, 1);
  bounded.push(&token, 1);
  rejects(500, "", [&] { (void)bounded.finish(accounting(buffered, 1, false)); });
}

gewell::TokenLogprobs token_scores(std::uint32_t token, std::uint32_t alternatives) {
  gewell::TokenLogprobs result;
  result.logprob = -0.25F;
  result.count = alternatives;
  if (alternatives) result.top[0] = {token, -0.25F};
  for (std::uint32_t index = 1; index < alternatives; ++index)
    result.top[index] = {index - 1, -2.0F - index};
  return result;
}

void logprob_cases(const text::Tokenizer& tokenizer, constraint::Compiler& compiler) {
  auto content = tokenizer.encode("Hello world");
  auto tokens = content;
  tokens.push_back(1);
  std::vector<gewell::TokenLogprobs> scores;
  for (auto token : tokens) scores.push_back(token_scores(token, 2));
  const auto body = json{{"model", kModel}, {"max_tokens", tokens.size()},
      {"logprobs", true}, {"top_logprobs", 2},
      {"messages", json::array({{{"role", "user"}, {"content", "Hi"}}})}};
  const auto buffered = parse(tokenizer, compiler, body, "/v1/chat/completions");
  http::CompletionOutput complete(tokenizer, buffered, kModel, "scores", 42, 1024 * 1024);
  require(complete.push(tokens.data(), tokens.size(), scores.data()).empty(),
          "buffered logprob push returned SSE");
  const auto document = json::parse(complete.finish(accounting(buffered, tokens.size(), true)));
  const auto& choice = document["choices"][0];
  require(choice["message"]["content"] == "Hello world" &&
          choice["logprobs"]["refusal"].is_null(),
          "buffered chat logprob envelope differs");
  const auto& entries = choice["logprobs"]["content"];
  require(entries.size() == content.size(), "hidden EOS received a content logprob");
  std::string decoded;
  for (const auto& entry : entries) {
    require(entry.size() == 4 && entry["top_logprobs"].size() == 2 &&
            entry["logprob"].get<float>() == -0.25F,
            "selected chat logprob shape differs");
    require(entry["top_logprobs"][0]["token"] == entry["token"] &&
            entry["top_logprobs"][0]["bytes"] == entry["bytes"] &&
            entry["top_logprobs"][1]["bytes"].is_null(),
            "top chat logprob token or byte mapping differs");
    decoded += entry["token"].get<std::string>();
  }
  require(decoded == "Hello world", "logprob tokens do not reconstruct content");

  auto streamed_body = body;
  streamed_body["stream"] = true;
  const auto streamed = parse(tokenizer, compiler, streamed_body, "/v1/chat/completions");
  http::CompletionOutput streaming(tokenizer, streamed, kModel, "scores", 42, 1024 * 1024);
  std::string wire = streaming.start();
  wire += streaming.push(tokens.data(), tokens.size(), scores.data());
  wire += streaming.finish(accounting(streamed, tokens.size(), true));
  const auto chunks = events(wire, true);
  json streamed_entries = json::array();
  std::string streamed_text;
  for (const auto& chunk : chunks) {
    if (chunk["choices"].empty()) continue;
    const auto& item = chunk["choices"][0];
    if (item["logprobs"].is_object())
      for (const auto& entry : item["logprobs"]["content"])
        streamed_entries.push_back(entry);
    streamed_text += item["delta"].value("content", "");
  }
  require(streamed_entries == entries && streamed_text == "Hello world" &&
          chunks.front()["choices"][0]["logprobs"].is_null() &&
          chunks.back()["choices"][0]["logprobs"].is_null(),
          "streamed chat logprobs differ from buffered output");

  std::vector<std::uint32_t> bytes;
  for (std::uint32_t token = 0; token < 262144 && bytes.size() < 3; ++token) {
    const auto& piece = tokenizer.token_piece(token);
    if (piece == "<0xE2>" || piece == "<0x82>" || piece == "<0xAC>")
      bytes.push_back(token);
  }
  std::sort(bytes.begin(), bytes.end(), [&](auto left, auto right) {
    const auto order = [](const std::string& piece) {
      return piece == "<0xE2>" ? 0 : piece == "<0x82>" ? 1 : 2;
    };
    return order(tokenizer.token_piece(left)) < order(tokenizer.token_piece(right));
  });
  require(bytes.size() == 3, "missing byte-fallback logprob fixtures");
  std::vector<gewell::TokenLogprobs> byte_scores;
  for (auto token : bytes) byte_scores.push_back(token_scores(token, 0));
  const auto byte_request = parse(tokenizer, compiler,
      {{"model", kModel}, {"max_tokens", 3}, {"logprobs", true}, {"top_logprobs", 0},
       {"messages", json::array({{{"role", "user"}, {"content", "Hi"}}})}},
      "/v1/chat/completions");
  http::CompletionOutput byte_output(tokenizer, byte_request, kModel, "bytes", 42, 10000);
  byte_output.push(bytes.data(), bytes.size(), byte_scores.data());
  const auto byte_document = json::parse(byte_output.finish(accounting(byte_request, 3, false)));
  const auto& byte_entries = byte_document["choices"][0]["logprobs"]["content"];
  require(byte_document["choices"][0]["message"]["content"] == "€" && byte_entries.size() == 3 &&
          byte_entries[0]["bytes"] == json::array({0xE2}) &&
          byte_entries[1]["bytes"] == json::array({0x82}) &&
          byte_entries[2]["bytes"] == json::array({0xAC}) &&
          byte_entries[0]["token"] == "�" && byte_entries[0]["top_logprobs"].empty(),
          "byte-fallback chat logprob mapping differs");

  auto stop_tokens = tokenizer.encode("pre");
  const auto stopped_suffix = tokenizer.encode("END");
  stop_tokens.insert(stop_tokens.end(), stopped_suffix.begin(), stopped_suffix.end());
  std::vector<gewell::TokenLogprobs> stop_scores;
  for (auto item : stop_tokens) stop_scores.push_back(token_scores(item, 0));
  for (const bool stream : {false, true}) {
    const auto stop_request = parse(tokenizer, compiler,
        {{"model", kModel}, {"max_tokens", stop_tokens.size()}, {"stop", "END"},
         {"logprobs", true}, {"top_logprobs", 0}, {"stream", stream},
         {"messages", json::array({{{"role", "user"}, {"content", "Hi"}}})}},
        "/v1/chat/completions");
    http::CompletionOutput stopped(tokenizer, stop_request, kModel, "scores-stop", 42,
                                   1024 * 1024);
    std::string wire = stopped.start();
    std::size_t seen = 0;
    for (; seen < stop_tokens.size(); ++seen) {
      wire += stopped.push(&stop_tokens[seen], 1, &stop_scores[seen]);
      if (stopped.stop_matched()) { ++seen; break; }
    }
    wire += stopped.finish(accounting(stop_request, seen, true));
    std::string visible, scored;
    if (!stream) {
      const auto value = json::parse(wire);
      visible = value["choices"][0]["message"]["content"];
      for (const auto& entry : value["choices"][0]["logprobs"]["content"])
        scored += entry["token"].get<std::string>();
    } else {
      for (const auto& chunk : events(wire, true)) {
        if (chunk["choices"].empty()) continue;
        const auto& choice = chunk["choices"][0];
        visible += choice["delta"].value("content", "");
        if (choice["logprobs"].is_object()) {
          require(!choice["delta"].value("content", "").empty(),
                  "streamed stop scores were detached from content");
          for (const auto& entry : choice["logprobs"]["content"])
            scored += entry["token"].get<std::string>();
        }
      }
    }
    require(visible == "pre" && scored == visible,
            "stop-suppressed content retained chat logprobs");
  }

  for (const bool stream : {false, true}) {
    const auto stopped_bytes = parse(tokenizer, compiler,
        {{"model", kModel}, {"max_tokens", 3}, {"stop", "€"},
         {"logprobs", true}, {"top_logprobs", 0}, {"stream", stream},
         {"messages", json::array({{{"role", "user"}, {"content", "Hi"}}})}},
        "/v1/chat/completions");
    http::CompletionOutput stopped(tokenizer, stopped_bytes, kModel, "byte-stop", 42,
                                   1024 * 1024);
    std::string wire = stopped.start();
    for (std::size_t index = 0; index < bytes.size(); ++index)
      wire += stopped.push(&bytes[index], 1, &byte_scores[index]);
    require(stopped.stop_matched(), "byte-fallback stop was not detected");
    wire += stopped.finish(accounting(stopped_bytes, 3, true));
    if (!stream) {
      const auto value = json::parse(wire);
      require(value["choices"][0]["message"]["content"] == "" &&
              value["choices"][0]["logprobs"]["content"].empty(),
              "byte-fallback stop retained hidden scores");
    } else {
      for (const auto& chunk : events(wire, true)) {
        if (chunk["choices"].empty()) continue;
        require(chunk["choices"][0]["logprobs"].is_null(),
                "streamed byte-fallback stop retained hidden scores");
      }
    }
  }

  auto streamed_bytes_body = json{{"model", kModel}, {"max_tokens", 3},
      {"logprobs", true}, {"top_logprobs", 0}, {"stream", true},
      {"messages", json::array({{{"role", "user"}, {"content", "Hi"}}})}};
  const auto streamed_bytes_request = parse(tokenizer, compiler, streamed_bytes_body,
                                            "/v1/chat/completions");
  http::CompletionOutput streamed_bytes(tokenizer, streamed_bytes_request, kModel,
                                        "byte-stream", 42, 1024 * 1024);
  std::string byte_wire = streamed_bytes.start();
  for (std::size_t index = 0; index < bytes.size(); ++index)
    byte_wire += streamed_bytes.push(&bytes[index], 1, &byte_scores[index]);
  byte_wire += streamed_bytes.finish(accounting(streamed_bytes_request, 3, false));
  std::size_t scored_byte_events = 0;
  for (const auto& chunk : events(byte_wire, true)) {
    if (chunk["choices"].empty()) continue;
    const auto& choice = chunk["choices"][0];
    if (!choice["logprobs"].is_object()) continue;
    ++scored_byte_events;
    require(choice["delta"].value("content", "") == "€" &&
            choice["logprobs"]["content"].size() == 3,
            "byte-fallback scores were detached from streamed content");
  }
  require(scored_byte_events == 1,
          "byte-fallback scores were split across streamed events");

  const auto one = parse(tokenizer, compiler,
      {{"model", kModel}, {"max_tokens", 1}, {"logprobs", true}, {"top_logprobs", 2},
       {"messages", json::array({{{"role", "user"}, {"content", "Hi"}}})}},
      "/v1/chat/completions");
  const auto token = content.front();
  const auto rejected = [&](gewell::TokenLogprobs invalid) {
    http::CompletionOutput output(tokenizer, one, kModel, "bad", 42, 10000);
    rejects(500, "", [&] { (void)output.push(&token, 1, &invalid); });
  };
  {
    http::CompletionOutput output(tokenizer, one, kModel, "bad", 42, 10000);
    rejects(500, "", [&] { (void)output.push(&token, 1); });
  }
  auto invalid = token_scores(token, 3);
  rejected(invalid);
  invalid = token_scores(token, 2); invalid.logprob = 0.1F;
  rejected(invalid);
  invalid = token_scores(token, 2); invalid.top[1] = {7, -0.1F};
  rejected(invalid);
  invalid = token_scores(token, 2); invalid.top[1] = invalid.top[0];
  rejected(invalid);
  invalid = token_scores(token, 2); invalid.top[1].token = 262144;
  rejected(invalid);

  const auto unscored = parse(tokenizer, compiler,
      {{"model", kModel}, {"prompt", "Hi"}, {"max_tokens", 1}});
  http::CompletionOutput unexpected(tokenizer, unscored, kModel, "bad", 42, 10000);
  const auto extra = token_scores(token, 1);
  rejects(500, "", [&] { (void)unexpected.push(&token, 1, &extra); });
}
json tool_declarations() {
  return json::array({
      {{"type", "function"}, {"function", {{"name", "weather"}, {"description", "Get weather"},
          {"parameters", {{"type", "object"}}}}}},
      {{"type", "function"}, {"function", {{"name", "sum"}, {"description", "Add numbers"},
          {"parameters", {{"type", "object"}}}}}},
  });
}

json tool_body() {
  return {{"model", kModel}, {"messages", json::array({{{"role", "user"}, {"content", "Help"}}})},
          {"tools", tool_declarations()}};
}

void tool_request_cases(const text::Tokenizer& tokenizer, constraint::Compiler& compiler) {
  auto body = tool_body();
  const auto automatic = parse(tokenizer, compiler, body, "/v1/chat/completions");
  require(automatic.allow_tool_calls && automatic.tool_names == std::vector<std::string>({"weather", "sum"}),
          "tool defaults or function names differ");
  body["tools"][0]["function"]["parameters"]["properties"] = {
      {"locations", {{"type", "array"}, {"items", {{"type", "object"},
          {"properties", {{"city", {{"type", "string"}}}}}}}}}};
  const auto schema_prompt = parse(tokenizer, compiler, body, "/v1/chat/completions").prompt;
  const auto unannotated = body;
  for (const auto* dialect : {"http://json-schema.org/draft-07/schema#",
                              "https://json-schema.org/draft/2020-12/schema"}) {
    for (const auto* path : {"/tools/0/function/parameters",
                             "/tools/0/function/parameters/properties/locations/items",
                             "/tools/0/function/parameters/properties/locations/items/properties/city"}) {
      body = unannotated;
      auto& schema = body[json::json_pointer(path)];
      schema["$schema"] = dialect;
      const auto annotated = parse(tokenizer, compiler, body, "/v1/chat/completions");
      require(annotated.allow_tool_calls && annotated.tool_names == automatic.tool_names &&
                  *annotated.prompt == *schema_prompt,
              "tool schema dialect changed the prompt or tool controls");
      for (const auto& value : std::vector<json>{nullptr, false, 7, json::array(), json::object()}) {
        schema["$schema"] = value;
        rejects(400, "tools", [&] { (void)parse(tokenizer, compiler, body, "/v1/chat/completions"); });
      }
    }
  }
  body = tool_body();
  body["tool_choice"] = "none";
  body["parallel_tool_calls"] = true;
  const auto none = parse(tokenizer, compiler, body, "/v1/chat/completions");
  auto plain = body; plain.erase("tools");
  require(!none.allow_tool_calls && *none.prompt == *parse(tokenizer, compiler, plain, "/v1/chat/completions").prompt,
          "tool_choice none retained current declarations");
  const auto prefill = parse(tokenizer, compiler, body, "/v1/cache/prefill");
  require(prefill.chat && *prefill.prompt == *none.prompt, "prefill tool prompt differs");
  for (const auto& choice : {json("unknown"), json{{"type", "function"}, {"function", {{"name", "unknown"}}}}, json(true)}) {
    body["tool_choice"] = choice;
    rejects(400, "tool_choice", [&] { (void)parse(tokenizer, compiler, body, "/v1/chat/completions"); });
  }
  body = tool_body(); body["tools"][0]["function"]["strict"] = true;
  rejects(400, "tools", [&] { (void)parse(tokenizer, compiler, body, "/v1/chat/completions"); });
  body = tool_body(); body["tools"].push_back(body["tools"][0]);
  rejects(400, "tools", [&] { (void)parse(tokenizer, compiler, body, "/v1/chat/completions"); });
  body = tool_body(); body["parallel_tool_calls"] = false;
  require(bool(parse(tokenizer, compiler, body, "/v1/chat/completions").constraint), "single-call control was not compiled");
  body["parallel_tool_calls"] = "false";
  rejects(400, "parallel_tool_calls", [&] { (void)parse(tokenizer, compiler, body, "/v1/chat/completions"); });
  body = tool_body();
  body["messages"].push_back({{"role", "assistant"}, {"content", nullptr}, {"tool_calls", json::array({
      {{"id", "history"}, {"type", "function"}, {"function", {{"name", "weather"}, {"arguments", "{}"}}}}
  })}});
  body["messages"].push_back({{"role", "tool"}, {"tool_call_id", "history"}, {"content", "sunny"}});
  body["chat_template_kwargs"] = {{"enable_thinking", true}};
  body["stop"] = "plan";
  const auto continuation = tokenizer.encode("plan<channel|>Answer<turn|>");
  body["max_tokens"] = continuation.size();
  const auto resumed = parse(tokenizer, compiler, body, "/v1/chat/completions");
  require(resumed.initial_reasoning, "tool-result thinking continuation was not detected");
  http::CompletionOutput output(tokenizer, resumed, kModel, "resumed", 42, 10000);
  output.push(continuation.data(), continuation.size());
  require(!output.stop_matched(), "stop scanned reasoning already opened by prompt");
  const auto message = json::parse(output.finish(accounting(resumed, continuation.size(), true)))["choices"][0]["message"];
  require(message["content"] == "Answer" && message["reasoning_content"] == "plan", "tool-result reasoning leaked into content");
}

struct ToolOutputRecord { json message; std::string reason; std::vector<json> chunks; };

ToolOutputRecord tool_output(const text::Tokenizer& tokenizer, constraint::Compiler& compiler, const std::vector<std::uint32_t>& tokens,
                             bool stream) {
  auto body = tool_body();
  body["max_tokens"] = tokens.size(); body["stream"] = stream;
  if (stream) body["stream_options"] = {{"include_usage", true}};
  const auto request = parse(tokenizer, compiler, body, "/v1/chat/completions");
  http::CompletionOutput output(tokenizer, request, kModel, "stable", 42, 100000);
  std::string wire = output.start();
  for (const auto token : tokens) wire += output.push(&token, 1);
  const bool eos = tokens.back() == 1 || tokens.back() == 106 || tokens.back() == 50;
  wire += output.finish(accounting(request, tokens.size(), eos));
  if (!stream) {
    const auto document = json::parse(wire);
    return {document["choices"][0]["message"], document["choices"][0]["finish_reason"], {}};
  }
  auto chunks = events(wire, true);
  json message = {{"role", "assistant"}, {"content", ""}, {"refusal", nullptr}};
  json calls = json::array();
  std::string reason;
  for (const auto& chunk : chunks) {
    if (chunk["choices"].empty()) continue;
    const auto& choice = chunk["choices"][0];
    if (!choice["finish_reason"].is_null()) reason = choice["finish_reason"];
    const auto& delta = choice["delta"];
    for (const char* field : {"content", "reasoning_content"})
      if (delta.contains(field)) message[field] = message.value(field, std::string{}) + delta[field].get<std::string>();
    if (delta.contains("tool_calls")) {
      for (const auto& fragment : delta["tool_calls"]) {
        const auto index = fragment["index"].get<std::size_t>();
        require(index <= calls.size(), "tool index skipped or changed");
        if (index == calls.size()) {
          require(fragment.contains("id") && fragment["type"] == "function", "initial tool delta lacks identity");
          require(fragment["id"] == "call_stable_" + std::to_string(index), "tool ID is not request-index scoped");
          calls.push_back({{"id", fragment["id"]}, {"type", "function"},
                          {"function", {{"name", fragment["function"]["name"]}, {"arguments", ""}}}});
        } else require(!fragment.contains("id") && !fragment["function"].contains("name"),
                       "tool identity repeated or changed after first delta");
        auto& args = calls[index]["function"]["arguments"];
        args = args.get<std::string>() + fragment["function"]["arguments"].get<std::string>();
      }
    }
  }
  if (!calls.empty()) {
    message["tool_calls"] = calls;
    if (message["content"] == "") message["content"] = nullptr;
  }
  require(chunks.back()["usage"]["completion_tokens"] == tokens.size(), "tool syntax/terminal tokens not counted");
  return {std::move(message), std::move(reason), std::move(chunks)};
}

void tool_output_cases(const text::Tokenizer& tokenizer, constraint::Compiler& compiler) {
  const std::string call = "<|tool_call>call:weather{city:<|\"|>Paris \\\"centre\\\" \\ road\n🌧<|\"|>,"
                           "nested:{items:[1,true,null,{x:<|\"|>é<|\"|>}]}}<tool_call|>";
  const std::string second = "<|tool_call>call:sum{values:[1,-2.5,3e2]}<tool_call|>";
  for (const auto& text : {
      std::string("Hello<turn|>"),
      call + "<|tool_response>",
      call + second + "<|tool_response>",
      std::string("<|channel>thought\nplan<channel|>Checking. ") + call + "<|tool_response>"}) {
    const auto tokens = tokenizer.encode(text);
    const auto buffered = tool_output(tokenizer, compiler, tokens, false);
    const auto streamed = tool_output(tokenizer, compiler, tokens, true);
    require(buffered.message == streamed.message && buffered.reason == streamed.reason,
            "buffered/streamed tools differ");
    require(buffered.reason == (text == "Hello<turn|>" ? "stop" : "tool_calls"), "tool handoff finish differs");
    if (buffered.message.contains("tool_calls")) {
      const auto args = json::parse(buffered.message["tool_calls"][0]["function"]["arguments"].get<std::string>());
      require(args["nested"]["items"][3]["x"] == "é" && args["city"].get<std::string>().find("🌧") != std::string::npos,
              "nested or Unicode argument conversion differs");
      if (text == call + second + "<|tool_response>")
        require(buffered.message["tool_calls"].size() == 2, "stopped at first call-end token");
    }
  }
  // Every token-budget cut keeps identical partial argument bytes. A complete
  // 49 without the handoff token remains length; an incomplete name stays hidden.
  const auto complete = tokenizer.encode(call + "<|tool_response>");
  for (std::size_t cut = 1; cut < complete.size(); ++cut) {
    const std::vector<std::uint32_t> tokens(complete.begin(), complete.begin() + cut);
    const auto buffered = tool_output(tokenizer, compiler, tokens, false);
    const auto streamed = tool_output(tokenizer, compiler, tokens, true);
    require(buffered.reason == "length" && streamed.reason == "length" && buffered.message == streamed.message,
            "tool truncation repaired arguments or changed stream parity");
  }
  for (const auto& malformed : {
      "<|tool_call>call:unknown{}<tool_call|><|tool_response>",
      "<|tool_call>call:weather{x:wat}<tool_call|><|tool_response>",
      "<|tool_call>call:weather{key name:1}<tool_call|><|tool_response>",
      "<|tool_call>call:weather{x:1<tool_call|>",
      "<|tool_call>call:weather{}<turn|>",
      "<|tool_call>call:weather{}<tool_call|><turn|>",
      "<tool_call|>", "<|tool_response>", "<|tool_call><|tool_call>"}) {
    rejects(500, "", [&] { (void)tool_output(tokenizer, compiler, tokenizer.encode(malformed), false); });
  }
  auto body = tool_body(); body["tool_choice"] = "none"; body["stream"] = true;
  const auto none = parse(tokenizer, compiler, body, "/v1/chat/completions");
  http::CompletionOutput suppressed(tokenizer, none, kModel, "none", 42, 10000);
  suppressed.start(); const std::uint32_t start = 48;
  rejects(500, "", [&] { (void)suppressed.push(&start, 1); });
  auto bounded_body = tool_body(); bounded_body["max_tokens"] = 500;
  const auto bounded_request = parse(tokenizer, compiler, bounded_body, "/v1/chat/completions");
  http::CompletionOutput bounded(tokenizer, bounded_request, kModel, "bounded", 42, 128);
  const auto long_call = tokenizer.encode("<|tool_call>call:weather{x:<|\"|>" + std::string(256, 'a'));
  rejects(500, "", [&] { (void)bounded.push(long_call.data(), long_call.size()); });
}

void stop_seed_cases(const text::Tokenizer& tokenizer, constraint::Compiler& compiler) {
  auto body = json{{"model", kModel}, {"prompt", "Hi"}, {"stop", json::array({"END", "END!"})},
                   {"seed", std::numeric_limits<std::int64_t>::min()}};
  auto request = parse(tokenizer, compiler, body);
  require(request.stops == std::vector<std::string>({"END", "END!"}) &&
          request.seed == std::numeric_limits<std::int64_t>::min(), "stop/seed values changed");
  body["seed"] = std::numeric_limits<std::int64_t>::max();
  require(parse(tokenizer, compiler, body).seed == std::numeric_limits<std::int64_t>::max(), "large seed lost precision");
  for (const auto& value : {json(true), json(1.0), json("1"), json(std::uint64_t(1) << 63)}) {
    body["seed"] = value;
    rejects(400, "seed", [&] { (void)parse(tokenizer, compiler, body); });
  }
  body["seed"] = nullptr;
  for (const auto& value : {json(""), json::array(), json::array({"x", ""}), json::array({1}),
                          json::array({"a", "b", "c", "d", "e"}), json(true)}) {
    body["stop"] = value;
    rejects(400, "stop", [&] { (void)parse(tokenizer, compiler, body); });
  }
  auto run = [&](std::vector<std::uint32_t> tokens, std::string stop, std::string expected,
                 std::size_t expected_count, bool chat = false) {
    json value = {{"model", kModel}, {"max_tokens", tokens.size()}, {"stop", stop}};
    if (chat) value["messages"] = json::array({{{"role", "user"}, {"content", "Hi"}}});
    else value["prompt"] = "Hi";
    const auto request = parse(tokenizer, compiler, value, chat ? "/v1/chat/completions" : "/v1/completions");
    http::CompletionOutput output(tokenizer, request, kModel, "stop", 42, 10000);
    std::size_t seen = 0;
    for (auto token : tokens) {
      output.push(&token, 1); ++seen;
      if (output.stop_matched()) break;
    }
    require(seen == expected_count && output.stop_matched(), "stop selected a lookahead token or missed boundary");
    const auto response = json::parse(output.finish(accounting(request, seen, true)));
    require(response["choices"][0]["finish_reason"] == "stop" && output.stop_matched(), "stop finish not preserved");
    require((chat ? response["choices"][0]["message"]["content"] : response["choices"][0]["text"]) == expected,
            "stop text leaked or visible prefix changed");
  };
  const auto hello = tokenizer.encode("Hello");
  run(hello, "ell", "H", hello.size());
  auto split = tokenizer.encode("pre");
  const auto end = tokenizer.encode("END"); split.insert(split.end(), end.begin(), end.end());
  run(split, "eEND", "pr", split.size());
  const auto thought = tokenizer.encode("<|channel>thought\nEND plan<channel|>Hi END");
  run(thought, "END", "Hi ", thought.size(), true);
  std::uint32_t e2 = 0, continuation = 0, ac = 0;
  for (std::uint32_t i = 0; i < 262144; ++i) {
    if (tokenizer.token_piece(i) == "<0xE2>") e2 = i;
    if (tokenizer.token_piece(i) == "<0x82>") continuation = i;
    if (tokenizer.token_piece(i) == "<0xAC>") ac = i;
  }
  require(e2 && continuation && ac, "missing fallback fixtures");
  run({e2, continuation, ac, hello.front()}, "€", "", 3);
  run({e2}, "�", "", 1);  // Final invalid byte flush must win over length.
}

void schema_cases(const text::Tokenizer& tokenizer, constraint::Compiler& compiler) {
  const json schema = {{"type", "object"}, {"properties", {{"answer", {{"type", "string"}, {"enum", {"blue"}}}}}},
                       {"required", {"answer"}}, {"additionalProperties", false}};
  const json format = {{"type", "json_schema"}, {"json_schema", {{"name", "answer"}, {"schema", schema}}}};
  const json base = {{"model", kModel}, {"messages", {{{"role", "user"}, {"content", "Return JSON"}}}},
                     {"response_format", format}};
  for (const auto& strict : {json(nullptr), json(true), json(false)}) {
    auto body = base;
    if (!strict.is_null()) body["response_format"]["json_schema"]["strict"] = strict;
    const auto request = parse(tokenizer, compiler, body, "/v1/chat/completions");
    require(bool(request.constraint) && !request.enable_thinking, "schema was not compiled before admission");
  }
  auto object = base; object["response_format"] = {{"type", "json_object"}};
  require(bool(parse(tokenizer, compiler, object, "/v1/chat/completions").constraint), "JSON object mode was not compiled");
  auto plain = base; plain["response_format"] = {{"type", "text"}};
  require(!parse(tokenizer, compiler, plain, "/v1/chat/completions").constraint, "text mode enabled constraints");
  plain.erase("response_format");
  require(*parse(tokenizer, compiler, plain, "/v1/chat/completions").prompt ==
          *parse(tokenizer, compiler, base, "/v1/chat/completions").prompt, "schema changed the chat prompt");

  for (const auto& invalid : std::vector<std::pair<json, std::string>>{
      {"json_schema", "response_format"}, {{{"type", "unknown"}}, "response_format.type"},
      {{{"type", "json_schema"}}, "response_format.json_schema"},
      {{{"type", "json_schema"}, {"json_schema", {{"schema", schema}}}}, "response_format.json_schema.name"},
      {{{"type", "json_schema"}, {"json_schema", {{"name", "bad name"}, {"schema", schema}}}}, "response_format.json_schema.name"},
      {{{"type", "json_object"}, {"extra", true}}, "response_format.extra"}}) {
    auto body = base; body["response_format"] = invalid.first;
    rejects(400, invalid.second, [&] { (void)parse(tokenizer, compiler, body, "/v1/chat/completions"); });
  }
  auto bad = base; bad["response_format"]["json_schema"]["strict"] = "yes";
  rejects(400, "response_format.json_schema.strict", [&] { (void)parse(tokenizer, compiler, bad, "/v1/chat/completions"); });
  bad = base; bad["response_format"]["json_schema"]["schema"]["gewell_unsupported_keyword"] = true;
  rejects(400, "response_format.json_schema.schema", [&] { (void)parse(tokenizer, compiler, bad, "/v1/chat/completions"); });
  bad = base; bad["stop"] = "blue";
  rejects(400, "stop", [&] { (void)parse(tokenizer, compiler, bad, "/v1/chat/completions"); });
  bad = base; bad["tools"] = tool_declarations();
  require(bool(parse(tokenizer, compiler, bad, "/v1/chat/completions").constraint), "tools blocked JSON answer constraints");
  bad["tool_choice"] = "none";
  require(bool(parse(tokenizer, compiler, bad, "/v1/chat/completions").constraint), "disabled tools blocked JSON finalization");
  rejects(400, "response_format", [&] { (void)parse(tokenizer, compiler,
      {{"model", kModel}, {"prompt", "JSON"}, {"response_format", format}}); });

  for (bool thinking : {false, true}) for (bool stream : {false, true}) {
    const auto tokens = tokenizer.encode(thinking ?
        "<|channel>thought\nConsider colors.<channel|>{\"answer\":\"blue\"}<turn|>" :
        "{\"answer\":\"blue\"}<turn|>");
    auto body = base;
    body["max_tokens"] = tokens.size(); body["stream"] = stream;
    body["chat_template_kwargs"] = {{"enable_thinking", thinking}};
    const auto request = parse(tokenizer, compiler, body, "/v1/chat/completions");
    require(request.enable_thinking == thinking, "constraint thinking metadata lost");
    http::CompletionOutput output(tokenizer, request, kModel, "schema", 42, 100000);
    std::string wire = output.start();
    for (auto token : tokens) wire += output.push(&token, 1);
    wire += output.finish(accounting(request, tokens.size(), true));
    std::string content, reasoning;
    if (stream) {
      const auto chunks = events(wire, true);
      require(chunks.back()["choices"][0]["finish_reason"] == "stop", "schema stream did not stop");
      for (const auto& chunk : chunks) {
        const auto& delta = chunk["choices"][0]["delta"];
        content += delta.value("content", ""); reasoning += delta.value("reasoning_content", "");
      }
    } else {
      const auto value = json::parse(wire);
      require(value["choices"][0]["finish_reason"] == "stop", "schema buffered response did not stop");
      content = value["choices"][0]["message"]["content"].get<std::string>();
      reasoning = value["choices"][0]["message"].value("reasoning_content", "");
    }
    require(json::parse(content) == json{{"answer", "blue"}}, "schema content differs");
    require(reasoning == (thinking ? "Consider colors." : ""), "schema reasoning leaked or disappeared");
  }
  auto short_body = base; short_body["max_tokens"] = 1;
  const auto short_request = parse(tokenizer, compiler, short_body, "/v1/chat/completions");
  const auto valid = tokenizer.encode("{\"answer\":\"blue\"}");
  http::CompletionOutput short_output(tokenizer, short_request, kModel, "short", 42, 10000);
  short_output.push(valid.data(), 1);
  require(json::parse(short_output.finish(accounting(short_request, 1, false)))["choices"][0]["finish_reason"] == "length",
          "incomplete JSON was repaired or marked complete");
  const auto request = parse(tokenizer, compiler, base, "/v1/chat/completions");
  for (const auto& invalid : {tokenizer.encode("{\"answer\":\"red\"}"), std::vector<std::uint32_t>{106}}) {
    http::CompletionOutput output(tokenizer, request, kModel, "invalid", 42, 10000);
    rejects(500, "", [&] { (void)output.push(invalid.data(), invalid.size()); });
  }
  auto history = base;
  history["messages"].push_back({{"role", "assistant"}, {"content", nullptr}, {"tool_calls", json::array({
      {{"id", "history"}, {"type", "function"}, {"function", {{"name", "weather"}, {"arguments", "{}"}}}}
  })}});
  history["messages"].push_back({{"role", "tool"}, {"tool_call_id", "history"}, {"content", "blue sky"}});
  history["tools"] = tool_declarations(); history["tool_choice"] = "none";
  history["chat_template_kwargs"] = {{"enable_thinking", true}};
  const auto resumed = parse(tokenizer, compiler, history, "/v1/chat/completions");
  require(resumed.initial_reasoning && resumed.enable_thinking && resumed.constraint, "constrained history lost open reasoning state");
  const auto continuation = tokenizer.encode("Consider colors.<channel|>{\"answer\":\"blue\"}<turn|>");
  http::CompletionOutput output(tokenizer, resumed, kModel, "history", 42, 10000);
  output.push(continuation.data(), continuation.size());
  const auto message = json::parse(output.finish(accounting(resumed, continuation.size(), true)))["choices"][0]["message"];
  require(message["reasoning_content"] == "Consider colors." && json::parse(message["content"].get<std::string>()) == json{{"answer", "blue"}},
          "constrained history reasoning leaked into JSON");
}

void explicit_contract_limits() {
  const auto fixture = small_text_contract();
  const auto tokenizer = small_tokenizer(fixture);
  constraint::Compiler compiler(tokenizer, 0);
  const auto request = http::parse_request("POST", "/v1/completions",
      json{{"model", kModel}, {"prompt", json::array({65})}, {"max_tokens", 2}}.dump(),
      tokenizer, kModel, compiler);
  require(*request.prompt == std::vector<std::uint32_t>{65},
          "HTTP confused the context limit with vocabulary size");
  try {
    (void)http::parse_request("POST", "/v1/completions",
        json{{"model", kModel}, {"prompt", json::array({65})}, {"max_tokens", 17}}.dump(),
        tokenizer, kModel, compiler);
    throw std::runtime_error("HTTP ignored the supplied context limit");
  } catch (const http::Error& error) {
    require(error.status == 400, "context overflow returned an unexpected status");
  }
  http::CompletionOutput output(tokenizer, request, kModel, "contract", 42, 10000);
  const std::uint32_t tokens[] = {65, fixture.stop_tokens.front()};
  output.push(tokens, 2);
  const auto response = json::parse(output.finish(accounting(request, 2, true)));
  require(response["choices"][0]["text"] == "A" &&
              response["choices"][0]["finish_reason"] == "stop",
          "HTTP output ignored the supplied vocabulary or stop IDs");
  text::ToolOutputDecoder tools(tokenizer, {"echo"}, "contract_", true);
  (void)tools.push(fixture.tool_call_start);
  for (const auto token : tokenizer.encode("call:echo{}")) (void)tools.push(token);
  (void)tools.push(fixture.tool_call_end);
  (void)tools.push(fixture.tool_handoff);
  (void)tools.finish(true);
  require(tools.tool_handoff() && tools.calls().size() == 1 &&
              tools.calls().front().arguments == "{}",
          "tool decoding ignored the supplied boundary tokens");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) { std::cerr << "usage: http_api_test TOKENIZER_DIRECTORY\n"; return 1; }
  const auto path = std::filesystem::path(argv[1]) / "tokenizer.json";
  if (!std::filesystem::is_regular_file(path)) { std::cout << "SKIP: local tokenizer is unavailable\n"; return 77; }
  try {
    const text::Tokenizer tokenizer(path.string(), contract);
    constraint::Compiler compiler(tokenizer, 1279);
    explicit_contract_limits();
    request_cases(tokenizer, compiler);
    reasoning_effort_cases(tokenizer, compiler);
    image_request_cases(tokenizer, compiler);
    cache_cases(tokenizer, compiler);
    completion_cases(tokenizer, compiler);
    logprob_cases(tokenizer, compiler);
    tool_request_cases(tokenizer, compiler);
    tool_output_cases(tokenizer, compiler);
    stop_seed_cases(tokenizer, compiler);
    schema_cases(tokenizer, compiler);
    std::cout << "HTTP API parsing, cache controls, serialization, SSE, limits and accounting passed\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << "HTTP API test: " << error.what() << '\n'; return 1; }
}
