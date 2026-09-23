#include "gewell/models/gemma4/text_contract.h"
#include "gewell/chat_codec.h"
#include "gewell/tokenizer.h"

#include <openssl/evp.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
const auto& contract = gewell::gemma4::text_contract_31b();

using nlohmann::json;
namespace text = gewell::text;
namespace fs = std::filesystem;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Function>
void rejects(const std::string& name, Function action) {
  try {
    action();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(name + ": accepted invalid input");
}

void expect_equal(const json& actual, const json& expected, const std::string& name) {
  require(actual == expected, name + "\nexpected: " + expected.dump() +
                                  "\nactual:   " + actual.dump());
}

std::string sha256(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  require(input.good(), "cannot read " + path.string());
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  require(context && EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) == 1,
          "cannot initialize SHA256");
  std::array<char, 65536> buffer{};
  while (input.read(buffer.data(), buffer.size()) || input.gcount()) {
    require(EVP_DigestUpdate(context.get(), buffer.data(), input.gcount()) == 1,
            "cannot update SHA256");
  }
  require(input.eof(), "failed reading " + path.string());
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned size = 0;
  require(EVP_DigestFinal_ex(context.get(), digest.data(), &size) == 1 && size == 32,
          "cannot finish SHA256");
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (unsigned index = 0; index < size; ++index) {
    result += hex[digest[index] >> 4];
    result += hex[digest[index] & 15];
  }
  return result;
}

void golden_cases(const text::Tokenizer& tokenizer, const json& fixture) {
  for (const auto& item : fixture.at("raw")) {
    const auto name = "raw/" + item.at("name").get<std::string>();
    const auto input = item.at("text").get<std::string>();
    expect_equal(tokenizer.encode(input), item.at("token_ids"), name + "/encode");
    expect_equal(tokenizer.completion_prompt(input), item.at("completion_token_ids"),
          name + "/completion BOS");
  }
  for (const auto& item : fixture.at("chat")) {
    const auto name = "chat/" + item.at("name").get<std::string>();
    const auto messages = contract.normalize_messages(item.at("messages"));
    const auto options = contract.template_options({
        {"enable_thinking", item.at("enable_thinking")},
        {"preserve_thinking", item.at("preserve_thinking")}});
    const auto rendered = contract.render_chat(messages, options);
    expect_equal(rendered, item.at("rendered"), name + "/render");
    expect_equal(tokenizer.encode(rendered), item.at("token_ids"), name + "/encode");
  }
  for (const auto& item : fixture.at("tool_chat")) {
    const auto name = "tool_chat/" + item.at("name").get<std::string>();
    const auto tools = contract.normalize_tools(item.at("tools"));
    expect_equal(tools, item.at("normalized_tools"), name + "/normalize tools");
    const auto messages = contract.normalize_messages(item.at("messages"));
    const auto options = contract.template_options({
        {"enable_thinking", item.at("enable_thinking")},
        {"preserve_thinking", item.at("preserve_thinking")}});
    const auto rendered = contract.render_chat(messages, options, tools);
    expect_equal(rendered, item.at("rendered"), name + "/render");
    expect_equal(tokenizer.encode(rendered), item.at("token_ids"), name + "/encode");
  }
  for (const auto& item : fixture.at("decode")) {
    const auto name = "decode/" + item.at("name").get<std::string>();
    text::IncrementalTextDecoder decoder(tokenizer, item.at("skip_special_tokens").get<bool>());
    std::vector<std::string> fragments;
    std::string joined;
    for (const auto id : item.at("token_ids").get<std::vector<std::uint32_t>>())
      fragments.push_back(decoder.push(id));
    fragments.push_back(decoder.finish());
    for (const auto& fragment : fragments) joined += fragment;
    expect_equal(fragments, item.at("fragments"), name + "/fragments");
    expect_equal(joined, item.at("text"), name + "/joined");
  }
  for (const auto& item : fixture.at("chat_decode")) {
    const auto name = "chat_decode/" + item.at("name").get<std::string>();
    text::ChatOutputDecoder decoder(tokenizer);
    json fragments = json::array();
    std::string reasoning, content;
    auto append = [&](text::ChatDelta delta) {
      reasoning += delta.reasoning;
      content += delta.content;
      fragments.push_back({{"reasoning", delta.reasoning}, {"content", delta.content}});
    };
    for (const auto id : item.at("token_ids").get<std::vector<std::uint32_t>>())
      append(decoder.push(id));
    append(decoder.finish());
    expect_equal(fragments, item.at("fragments"), name + "/fragments");
    expect_equal(reasoning, item.at("reasoning"), name + "/joined reasoning");
    expect_equal(content, item.at("content"), name + "/joined content");
  }
}

void invalid_inputs(const text::Tokenizer& tokenizer) {
  const std::vector<std::string> invalid_utf8 = {
      "\x80", "\xc0\xaf", "\xc2", "\xe0\x80\xaf", "\xe2\x82",
      "\xed\xa0\x80", "\xf0\x80\x80\xaf", "\xf0\x9f\x98",
      "\xf4\x90\x80\x80", "\xf5\x80\x80\x80", "\xff"};
  for (std::size_t index = 0; index < invalid_utf8.size(); ++index) {
    const auto& input = invalid_utf8[index];
    const auto name = "invalid UTF-8 " + std::to_string(index);
    rejects(name, [&] { (void)tokenizer.encode(input); });
    rejects(name + " after special", [&] { (void)tokenizer.encode("<bos>" + input); });
    rejects(name + " before special", [&] { (void)tokenizer.encode(input + "<bos>"); });
    rejects(name + " in chat", [&] {
      (void)contract.render_chat(contract.normalize_messages(
          json::array({{{"role", "user"}, {"content", input}}})));
    });
  }
  for (const std::uint32_t id : {262144U, std::numeric_limits<std::uint32_t>::max()}) {
    rejects("out-of-range pretokenized prompt", [&] {
      (void)tokenizer.completion_prompt(std::vector<std::uint32_t>{id});
    });
    rejects("out-of-range piece", [&] { (void)tokenizer.token_piece(id); });
    rejects("out-of-range special", [&] { (void)tokenizer.is_special(id); });
    rejects("out-of-range decode", [&] {
      text::IncrementalTextDecoder decoder(tokenizer);
      (void)decoder.push(id);
    });
    rejects("out-of-range chat decode", [&] {
      text::ChatOutputDecoder decoder(tokenizer);
      (void)decoder.push(id);
    });
  }
  const std::vector<json> invalid_messages = {
      nullptr, json::object(), json::array(), json::array({"user"}),
      json::parse(R"([{"content":"hello"}])"),
      json::parse(R"([{"role":"user"}])"),
      json::parse(R"([{"role":1,"content":"hello"}])"),
      json::parse(R"([{"role":"tool","content":"hello"}])"),
      json::parse(R"([{"role":"user","content":null}])"),
      json::parse(R"([{"role":"user","content":1}])"),
      json::parse(R"([{"role":"user","content":["hello"]}])"),
      json::parse(R"([{"role":"user","content":[{"type":"text","text":1}]}])"),
      json::parse(R"([{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:"}}]}])"),
      json::parse(R"([{"role":"user","content":[{"type":"text","text":"hi","extra":1}]}])"),
      json::parse(R"([{"role":"assistant","content":"","tool_calls":[{}]}])"),
      json::parse(R"([{"role":"assistant","content":"","reasoning":false}])"),
      json::parse(R"([{"role":"assistant","content":"","reasoning":{}}])"),
      json::parse(R"([{"role":"assistant","content":"","reasoning":[]}])"),
      json::parse(R"([{"role":"assistant","content":"","reasoning_content":false}])"),
      json::parse(R"([{"role":"user","content":"hi","unknown":true}])")};
  for (std::size_t index = 0; index < invalid_messages.size(); ++index)
    rejects("invalid messages " + std::to_string(index), [&] {
      (void)contract.normalize_messages(invalid_messages[index]);
    });
  rejects("empty normalized messages", [] { (void)contract.render_chat({}); });
  rejects("invalid normalized role", [] { (void)contract.render_chat({{"tool", "hi"}}); });

  for (const auto& value : std::vector<json>{nullptr, json::object()}) {
    const auto options = contract.template_options(value);
    require(!options.enable_thinking && !options.preserve_thinking,
            "template defaults differ");
  }
  for (bool enable : {false, true}) {
    for (bool preserve : {false, true}) {
      const auto options = contract.template_options({
          {"enable_thinking", enable}, {"preserve_thinking", preserve}});
      require(options.enable_thinking == enable &&
                  options.preserve_thinking == preserve,
              "template boolean controls differ");
    }
  }
  const auto enable_only = contract.template_options({{"enable_thinking", true}});
  const auto preserve_only = contract.template_options({{"preserve_thinking", true}});
  require(enable_only.enable_thinking && !enable_only.preserve_thinking &&
              !preserve_only.enable_thinking && preserve_only.preserve_thinking,
          "omitted template control is not false");
  const std::vector<json> invalid_kwargs = {
      false, "thinking", json::array(), {{"unknown", false}},
      {{"enable_thinking", nullptr}}, {{"enable_thinking", 1}},
      {{"enable_thinking", "true"}}, {{"preserve_thinking", nullptr}},
      {{"preserve_thinking", 1}}, {{"preserve_thinking", "true"}}};
  for (const auto& value : invalid_kwargs)
    rejects("invalid template kwargs " + value.dump(), [&] {
      (void)contract.template_options(value);
    });
}

void repeated_inputs(const text::Tokenizer& tokenizer) {
  for (const auto& tokens : std::vector<std::vector<std::uint32_t>>{
           {}, {107}, {2, 107}, {2, 2, 107}, {100, 101, 262143}})
    expect_equal(tokenizer.completion_prompt(tokens), tokens, "pretokenized prompt stays exact");
  // Independent pinned-oracle identities also exercise many equal-rank merges
  // and adjacent special boundaries without fragile wall-clock assertions.
  expect_equal(tokenizer.encode("aaaaaaa"), json::array({50354, 72004}), "leftmost merge tie");
  std::vector<std::uint32_t> repeated_a(4096, 128820);
  repeated_a.push_back(236746);
  expect_equal(tokenizer.encode(std::string(32769, 'a')), repeated_a, "long repeated text");
  std::string specials;
  for (unsigned index = 0; index < 16384; ++index) specials += "<bos>";
  expect_equal(tokenizer.encode(specials), std::vector<std::uint32_t>(16384, 2),
        "many adjacent specials");
  expect_equal(tokenizer.encode(std::string(64, ' ')), json::array({167, 167, 138}),
        "long whitespace merge boundaries");
  require(tokenizer.token_piece(2) == "<bos>" && tokenizer.is_special(2) &&
              tokenizer.is_special(258884) && !tokenizer.is_special(107) &&
              !tokenizer.is_special(238), "special-token inventory differs");
}

void schema_tools() {
  auto schema = json::parse(R"({
    "$schema":"https://json-schema.org/draft/2020-12/schema",
    "title":"Generated SDK schema", "description":"Root description", "type":"object",
    "$defs":{"value":{"type":"integer","minimum":0}},
    "properties":{
      "query":{"type":["string","null"],"default":null,"examples":["東京"]},
      "count":{"$ref":"#/$defs/value"},
      "mode":{"enum":[1,2,null]},
      "variant":{"anyOf":[{"const":true},{"type":"string","pattern":"^[a-z]+$"}]},
      "encoded":{"type":"string","contentMediaType":"application/json",
        "contentSchema":{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"integer"}},
      "free":{"type":"array"},
      "世界 key":{"type":"object","properties":{},"additionalProperties":{"type":"string"}},
      "$schema":{"type":"string"}
    },
    "default":{"type":"string","$schema":"application data"},
    "dependencies":{"query":["count"], "count":{"$schema":"http://json-schema.org/draft-07/schema#",
      "type":"object","properties":{"query":{"type":"string"}}}},
    "required":["query"],"additionalProperties":false
  })");
  const auto tools = contract.normalize_tools(json::array({{{"type", "function"},
      {"function", {{"name", "lookup"}, {"parameters", schema}}}}}));
  schema.erase("$schema");
  schema["properties"]["encoded"]["contentSchema"].erase("$schema");
  schema["dependencies"]["count"].erase("$schema");
  expect_equal(tools[0]["function"]["parameters"], schema, "SDK schema assertions and annotation data preserved");
  const auto rendered = contract.render_chat(contract.normalize_messages(
      json::array({{{"role", "user"}, {"content", "lookup"}}})), {}, tools);
  for (const char* text : {"#/$defs/value", "anyOf", "^[a-z]+$", "Root description", "世界 key", "application data"})
    require(rendered.find(text) != std::string::npos, std::string("schema detail missing from prompt: ") + text);
  const json arguments = {{"世界 key", {{"", 1}, {"a.b", true}}}, {"$schema", "value"}};
  const auto messages = contract.normalize_messages(json::array({
      {{"role", "assistant"}, {"content", nullptr}, {"tool_calls", json::array({
          {{"id", "call"}, {"type", "function"}, {"function", {{"name", "lookup"}, {"arguments", arguments.dump()}}}}
      })}}, {{"role", "tool"}, {"tool_call_id", "call"}, {"content", "done"}}}));
  expect_equal(messages[0].tool_calls[0].arguments, arguments, "arbitrary JSON argument keys preserved");
  require(contract.render_chat(messages).find("<|\"|>世界 key<|\"|>") != std::string::npos,
          "non-identifier argument keys were not quoted");
}

void invalid_tools() {
  const auto tools = json::parse(R"([{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}])");
  expect_equal(contract.normalize_tools(nullptr), json::array(), "null tools");
  for (const auto& value : std::vector<json>{true, "lookup", json::object()})
    rejects("invalid tools container", [&] { (void)contract.normalize_tools(value); });
  auto duplicate = tools;
  duplicate.push_back(tools[0]);
  rejects("duplicate function names", [&] { (void)contract.normalize_tools(duplicate); });
  for (const auto& [key, value] : std::vector<std::pair<std::string, json>>{
      {"name", ""}, {"name", "has space"}, {"name", "函数"}, {"name", std::string(65, 'x')},
      {"description", 1}, {"strict", 0}, {"unknown", nullptr},
      {"parameters", json::array()}, {"parameters", {{"type", "string"}}},
      {"parameters", {{"type", "object"}, {"additionalProperties", 1}}},
      {"parameters", {{"type", "object"}, {"required", json::array({"query", "query"})}}}}) {
    auto invalid = tools;
    invalid[0]["function"][key] = value;
    rejects("invalid tool function " + key, [&] { (void)contract.normalize_tools(invalid); });
  }
  for (const auto& schema : std::vector<json>{
      {{"type", json::array({"string", "string"})}}, {{"type", "String"}}, {{"$ref", 1}},
      {{"type", "integer"}, {"minimum", "0"}}, {{"type", "integer"}, {"enum", 1}},
      {{"type", "string"}, {"nullable", 1}}, {{"type", "string"}, {"enum", json::array()}},
      {{"type", "array"}, {"items", 1}}, {{"type", "object"}, {"properties", json::array()}},
      {{"anyOf", json::array()}}}) {
    auto invalid = tools;
    invalid[0]["function"]["parameters"]["properties"]["query"] = schema;
    rejects("unsupported property schema " + schema.dump(), [&] { (void)contract.normalize_tools(invalid); });
  }

  const auto history = json::parse(R"([{"role":"user","content":"lookup"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{\"query\":\"世界\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"result"}])");
  const auto normalized = contract.normalize_messages(history);
  expect_equal(normalized[1].tool_calls[0].arguments, {{"query", "世界"}}, "argument strings parse to objects");
  require(normalized[2].name == "lookup", "tool result name was not associated");
  json nested = json::object();
  for (unsigned level = 1; level < 32; ++level) nested = {{"child", std::move(nested)}};
  auto boundary = history;
  boundary[1]["tool_calls"][0]["function"]["arguments"] = nested;
  (void)contract.normalize_messages(boundary);
  boundary[1]["tool_calls"][0]["function"]["arguments"] = {{"child", std::move(nested)}};
  rejects("33 argument container levels", [&] { (void)contract.normalize_messages(boundary); });
  std::vector<json> bad_histories;
  auto changed = history;
  changed.erase(2);
  bad_histories.push_back(changed);
  changed.push_back({{"role", "user"}, {"content", "missing result"}});
  bad_histories.push_back(changed);
  changed = history;
  changed[2]["tool_call_id"] = "other";
  bad_histories.push_back(changed);
  changed = history;
  changed[2]["name"] = "other";
  bad_histories.push_back(changed);
  changed = history;
  changed.push_back(history[2]);
  bad_histories.push_back(changed);
  changed = history;
  changed[1]["tool_calls"].push_back(changed[1]["tool_calls"][0]);
  bad_histories.push_back(changed);
  changed = history;
  changed[1]["role"] = "user";
  bad_histories.push_back(changed);
  changed = history;
  changed[2]["content"] = nullptr;
  bad_histories.push_back(changed);
  for (const auto& [key, value] : std::vector<std::pair<std::string, json>>{
      {"refusal", "refused"}, {"annotations", json::array({json::object()})}, {"audio", json::object()}}) {
    changed = history;
    changed[1][key] = value;
    bad_histories.push_back(changed);
  }
  for (const auto& arguments : std::vector<json>{"{", "[]", "null", json::array(), nullptr}) {
    changed = history;
    changed[1]["tool_calls"][0]["function"]["arguments"] = arguments;
    bad_histories.push_back(changed);
  }
  for (const auto& invalid : bad_histories)
    rejects("invalid tool history " + invalid.dump(), [&] { (void)contract.normalize_messages(invalid); });
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: text_codec_test TOKENIZER_DIRECTORY FIXTURE_JSON\n";
    return 1;
  }
  try {
    const fs::path directory(argv[1]);
    const auto tokenizer_path = directory / "tokenizer.json";
    if (!fs::is_regular_file(tokenizer_path)) {
      std::cout << "SKIP: local tokenizer unavailable: " << tokenizer_path << '\n';
      return 77;
    }
    std::ifstream input(argv[2]);
    require(input.good(), "cannot read committed oracle fixture");
    const json fixture = json::parse(input);
    require(fixture.at("schema_version") == 1 &&
                fixture.at("source").at("repository") == "google/gemma-4-31B-it" &&
                fixture.at("source").at("revision") ==
                    "842da3794eaa0b77d5f08bae87a17459d91ff475",
            "fixture source is not the pinned serving revision");
    expect_equal(sha256(tokenizer_path), fixture.at("source").at("assets").at("tokenizer.json"),
                 "reference tokenizer SHA256");
    const text::Tokenizer tokenizer(tokenizer_path.string(), contract);
    golden_cases(tokenizer, fixture);
    invalid_inputs(tokenizer);
    invalid_tools();
    schema_tools();
    repeated_inputs(tokenizer);
    std::cout << "text codec: " << fixture.at("raw").size() << " raw, "
              << fixture.at("chat").size() << " chat, "
              << fixture.at("tool_chat").size() << " tool chat, "
              << fixture.at("decode").size() << " decode, "
              << fixture.at("chat_decode").size()
              << " chat decode fixtures, invalid inputs and long inputs passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "text codec test: " << error.what() << '\n';
    return 1;
  }
}
