#include "gewell/models/gemma4/text_contract.h"
#include "text_contract_fixture.h"
#include "gewell/json_constraint.h"
#include "gewell/chat_codec.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
const auto& contract = gewell::gemma4::text_contract_31b();
using nlohmann::json;
namespace constraint = gewell::constraint;
namespace text = gewell::text;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <class Action> void rejects(const std::string& name, Action action) {
  try { action(); } catch (const std::exception&) { return; }
  throw std::runtime_error(name + ": accepted unsupported schema");
}

bool bit(const std::vector<std::uint32_t>& mask, std::uint32_t token, std::size_t row = 0) {
  return mask[row * contract.mask_words() + token / 32] & (std::uint32_t{1} << (token % 32));
}

bool accept_tokens(const std::shared_ptr<const constraint::Compiled>& grammar,
                   const std::vector<std::uint32_t>& tokens) {
  constraint::State state(grammar);
  std::vector<std::uint32_t> mask(contract.mask_words());
  for (const auto token : tokens) {
    state.fill_mask(mask.data());
    const bool allowed = bit(mask, token);
    const bool accepted = state.accept(token);
    require(allowed == accepted, "mask and accept disagree for token " + std::to_string(token));
    if (!accepted) return false;
  }
  return state.completed();
}

bool accepts(const text::Tokenizer& tokenizer, const std::shared_ptr<const constraint::Compiled>& grammar,
             const std::string& value) {
  return accept_tokens(grammar, tokenizer.encode(value));
}

void dsgen(const text::Tokenizer& tokenizer, constraint::Compiler& compiler, const json& fixture) {
  std::size_t valid = 0, invalid = 0;
  for (const auto& item : fixture.at("cases")) {
    const auto name = item.at("id").get<std::string>();
    const auto compiled = compiler.compile(item.at("response_format").at("json_schema").at("schema"));
    require(compiled->memory_bytes() > 0, "compiled grammar memory was not reported");
    for (const auto& value : item.at("valid_instances")) {
      require(accepts(tokenizer, compiled, value.dump()), name + ": valid instance rejected: " + value.dump());
      ++valid;
    }
    for (const auto& value : item.at("invalid_instances")) {
      require(!accepts(tokenizer, compiled, value.dump()), name + ": invalid instance accepted: " + value.dump());
      ++invalid;
    }
    std::cout << "constraint fixture " << name << " bytes=" << compiled->memory_bytes() << '\n';
  }
  std::cout << "DSGen native constraints: " << valid << " valid, " << invalid << " invalid examples\n";
}

void numeric_bounds(const text::Tokenizer& tokenizer, constraint::Compiler& compiler) {
  const auto probability = compiler.compile({{"type", "number"}, {"minimum", 0}, {"maximum", 1}});
  for (const auto* value : {"0", "1", "0.1", "0.999999", "1.0"})
    require(accepts(tokenizer, probability, value), std::string("valid probability rejected: ") + value);
  for (const auto* value : {"-0.1", "1.01", "1.000001", "2", "100", "1e2"})
    require(!accepts(tokenizer, probability, value), std::string("invalid probability accepted: ") + value);
  // Bounded numbers use the converter's canonical decimal spellings.
  for (const auto* value : {"-0.0", "1e-1", "0.1234567"})
    require(!accepts(tokenizer, probability, value), std::string("unsupported decimal spelling accepted: ") + value);
  const auto exclusive = compiler.compile({{"type", "number"}, {"minimum", -1}, {"exclusiveMinimum", -0.5},
                                           {"maximum", 1}, {"exclusiveMaximum", 0.5}});
  for (const auto* value : {"-0.499999", "0", "0.499999"})
    require(accepts(tokenizer, exclusive, value), std::string("valid exclusive range value rejected: ") + value);
  for (const auto* value : {"-1", "-0.5", "0.5", "1"})
    require(!accepts(tokenizer, exclusive, value), std::string("invalid exclusive range value accepted: ") + value);
}

void whitespace(const text::Tokenizer& tokenizer, constraint::Compiler& compiler) {
  const auto grammar = compiler.compile({{"type", "object"},
      {"properties", {{"x", {{"type", "boolean"}}}}},
      {"required", {"x"}}, {"additionalProperties", false}});
  require(accepts(tokenizer, grammar, "{\n  \"x\": " + std::string(15, ' ') + "true\n}"),
          "bounded formatted whitespace rejected");
  require(!accepts(tokenizer, grammar, "{\"x\":" + std::string(17, ' ') + "true}"),
          "unbounded structural whitespace accepted");
  const auto string_grammar = compiler.compile({{"type", "string"}});
  require(accepts(tokenizer, string_grammar, json(std::string(128, ' ')).dump()),
          "structural whitespace limit leaked into string content");
}

void object_properties(const text::Tokenizer& tokenizer, constraint::Compiler& compiler) {
  for (const std::string mode : {"default", "true", "schema", "false"}) {
    json schema = {{"type", "object"}, {"properties", {{"x", {{"type", "integer"}}}}}, {"required", {"x"}}};
    if (mode == "true") schema["additionalProperties"] = true;
    if (mode == "false") schema["additionalProperties"] = false;
    if (mode == "schema") schema["additionalProperties"] = {{"type", "string"}};
    const auto grammar = compiler.compile(schema);
    require(accepts(tokenizer, grammar, "{\"x\":1}"), "declared property rejected");
    for (const auto* value : {"{\"x\":1,\"x\":\"bad\"}", "{\"x\":1,\"\\u0078\":\"bad\"}",
                             "{\"x\":1,\"extra\":\"omitted by generation subset\"}"})
      require(!accepts(tokenizer, grammar, value), "additional property bypassed declared key constraints");
    schema["required"] = json::array();
    const auto optional = compiler.compile(schema);
    require(accepts(tokenizer, optional, "{}"), "optional declared property became required");
    require(!accepts(tokenizer, optional, "{\"\\u0078\":\"bad\"}"), "omitted property reappeared as escaped alias");
  }
  const auto dictionary = compiler.compile({{"type", "object"}, {"additionalProperties", {{"type", "integer"}}}});
  for (const auto* value : {"{}", "{\"new\":1}", "{\"\\u0078\":2}", "{\"😀\":3}"})
    require(accepts(tokenizer, dictionary, value), "dictionary key rejected");
  require(!accepts(tokenizer, dictionary, "{\"\\u0078\":\"bad\"}"), "dictionary value constraint ignored");
  json schema = {{"type", "object"}, {"properties", {{"x", {{"type", "integer"}}}}}};
  schema["default"] = {{"properties", {{"data", true}}}, {"additionalProperties", true}};
  const auto normalized = constraint::normalize_schema(schema);
  require(normalized["default"] == schema["default"], "schema normalization changed annotation data");
}

void masks_and_rollback(const text::Tokenizer& tokenizer, constraint::Compiler& compiler) {
  const auto grammar = compiler.compile({{"type", "object"}, {"properties", {{"value", {{"type", "integer"}, {"enum", {0, 1, 2}}}}}},
                                        {"required", {"value"}}, {"additionalProperties", false}});
  const auto tokens = tokenizer.encode("{\"value\":1}");
  constraint::State state(grammar);
  std::vector<std::uint32_t> initial(contract.mask_words()), after(initial.size());
  state.fill_mask(initial.data());
  for (auto token : {1U, 106U, 50U, 48U, 49U, 100U, 101U}) require(!bit(initial, token), "premature EOS/control allowed");
  std::vector<std::uint32_t> masks((tokens.size() + 1) * contract.mask_words());
  state.block_masks(tokens.data(), tokens.size(), masks.data());
  constraint::State expected(grammar);
  for (std::size_t row = 0; row <= tokens.size(); ++row) {
    expected.fill_mask(after.data());
    require(std::equal(after.begin(), after.end(), masks.begin() + row * contract.mask_words()), "block prefix mask differs");
    if (row < tokens.size()) require(expected.accept(tokens[row]), "valid draft rejected");
  }
  state.fill_mask(after.data());
  require(initial == after && !state.completed(), "speculative walk changed live state");
  auto invalid_draft = tokens;
  invalid_draft.insert(invalid_draft.begin() + 1, 48);
  masks.resize((invalid_draft.size() + 1) * contract.mask_words());
  state.block_masks(invalid_draft.data(), invalid_draft.size(), masks.data());
  require(!bit(masks, 48, 1), "invalid draft token was not excluded at reachable row");
  require(std::all_of(masks.begin() + 2 * contract.mask_words(), masks.end(),
                      [](auto word) { return word == ~std::uint32_t{0}; }), "unreachable suffix was not marked");
  state.fill_mask(after.data());
  require(initial == after, "invalid speculative walk changed live state");
  auto terminal_draft = tokens;
  terminal_draft.push_back(106);
  terminal_draft.push_back(48);
  masks.resize((terminal_draft.size() + 1) * contract.mask_words());
  state.block_masks(terminal_draft.data(), terminal_draft.size(), masks.data());
  state.fill_mask(after.data());
  require(initial == after && !state.terminated(), "terminal draft did not roll back");
  for (const auto token : tokens) require(state.accept(token), "committed token rejected");
  state.fill_mask(after.data());
  require(state.completed() && bit(after, 1) && bit(after, 106) && !bit(after, 50), "completion/EOS mask differs");
  require(state.accept(106) && state.terminated(), "terminal EOS not accepted");
}

void tool_constraints(const text::Tokenizer& tokenizer, constraint::Compiler& compiler) {
  const json schema = {{"type", "object"}, {"properties", {
      {"city name", {{"type", "string"}, {"enum", {"café ☃", "東京"}}}},
      {"count", {{"type", "integer"}, {"minimum", 1}, {"maximum", 3}}}}},
      {"required", {"city name", "count"}}, {"additionalProperties", false}};
  const json tools = json::array({{{"type", "function"}, {"function", {
      {"name", "lookup"}, {"strict", true}, {"parameters", schema}}}}});
  const std::string call = "<|tool_call>call:lookup{\"city name\":\"café ☃\",\"count\":2}<tool_call|>";
  const auto required = compiler.compile_tools(tools, {"lookup"}, true, true);
  require(accepts(tokenizer, required, call + "<|tool_response>"), "strict required call rejected");
  require(accepts(tokenizer, required, call + call + "<|tool_response>"), "parallel calls rejected");
  for (const auto& text : std::vector<std::string>{"done<turn|>", "<turn|>", "<|tool_response>",
      call + "<turn|>", "<|tool_call>call:unknown{}<tool_call|><|tool_response>",
      "<|tool_call>call:lookup{\"city name\":\"café ☃\",\"count\":4}<tool_call|><|tool_response>",
      "<|tool_call>call:lookup{\"count\":2}<tool_call|><|tool_response>"})
    require(!accepts(tokenizer, required, text), "invalid required/strict output accepted: " + text);
  const auto single = compiler.compile_tools(tools, {"lookup"}, true, false);
  require(accepts(tokenizer, single, call + "<|tool_response>"), "single call rejected");
  require(!accepts(tokenizer, single, call + call + "<|tool_response>"), "single-call limit ignored");
  const auto automatic = compiler.compile_tools(tools, {"lookup"}, false, false);
  require(accepts(tokenizer, automatic, "A plain answer.<turn|>"), "auto cannot answer");
  require(accepts(tokenizer, automatic, "Looking it up. " + call + "<|tool_response>"), "auto cannot call after content");
  const auto none = compiler.compile_tools(tools, {}, false, true);
  require(accepts(tokenizer, none, "A plain answer.<turn|>"), "none cannot answer");
  require(!accepts(tokenizer, none, call + "<|tool_response>"), "none permits tool calls");
  const auto answer = compiler.compile({{"type", "object"}, {"properties", {{"ok", {{"const", true}}}}},
      {"required", {"ok"}}, {"additionalProperties", false}});
  const auto combined = compiler.compile_tools(tools, {"lookup"}, false, true, answer);
  require(accepts(tokenizer, combined, "{\"ok\":true}<turn|>"), "combined JSON answer rejected");
  require(accepts(tokenizer, combined, call + "<|tool_response>"), "combined tool call rejected");
  require(!accepts(tokenizer, combined, "{\"ok\":false}<turn|>"), "combined answer schema ignored");
  auto loose = tools;
  loose[0]["function"]["strict"] = false;
  require(accepts(tokenizer, compiler.compile_tools(loose, {"lookup"}, true, false),
      "<|tool_call>call:lookup{\"arbitrary key\":false}<tool_call|><|tool_response>"),
      "non-strict arguments were constrained to the parameter schema");
  auto structured = tools;
  structured[0]["function"]["parameters"] = json::parse(R"({
    "type":"object", "properties":{
      "data":{"$ref":"#/$defs/node"},
      "option":{"anyOf":[{"type":"integer","enum":[1,2]},{"type":"null"}]}
    }, "required":["data","option"], "additionalProperties":false,
    "$defs":{"node":{"type":"object","properties":{
      "children":{"type":"array","items":{"$ref":"#/$defs/node"},"maxItems":2},
      "name":{"type":["string","null"]}
    },"required":["children","name"],"additionalProperties":false}}
  })");
  const auto recursive = compiler.compile_tools(structured, {"lookup"}, true, false);
  require(accepts(tokenizer, recursive,
      "<|tool_call>call:lookup{\"data\":{\"children\":[{\"children\":[],\"name\":null}],\"name\":\"東京\"},\"option\":2}<tool_call|><|tool_response>"),
      "recursive references, unions, or anyOf rejected in strict arguments");
  for (const auto& invalid_schema : std::vector<json>{
      {{"type", "object"}, {"properties", {{"x", {{"type", "string"}}}}}, {"additionalProperties", false}},
      {{"type", "object"}, {"properties", json::object()}},
      {{"type", "object"}, {"properties", {{"x", {{"type", "integer"}, {"minimum", 3}, {"maximum", 1}}}}},
       {"required", {"x"}}, {"additionalProperties", false}},
      {{"type", "object"}, {"properties", {{"x", {{"type", "string"}, {"pattern", "[a-z]+"}}}}},
       {"required", {"x"}}, {"additionalProperties", false}}}) {
    auto invalid = tools;
    invalid[0]["function"]["parameters"] = invalid_schema;
    rejects("invalid or unsupported strict arguments", [&] { (void)compiler.compile_tools(invalid, {"lookup"}, true, false); });
    rejects("invalid unselected strict arguments", [&] { (void)compiler.compile_tools(invalid, {}, false, false); });
  }

  // Walk and rewind across thinking, name selection, arguments, repeated calls,
  // and handoff using the same block-mask interface used by MTP.
  const auto tokens = tokenizer.encode("<|channel>thought\nPlan.<channel|>" + call + call + "<|tool_response>");
  constraint::State state(required, true);
  std::vector<std::uint32_t> before(contract.mask_words()), after(before.size());
  std::vector<std::uint32_t> masks((tokens.size() + 1) * contract.mask_words());
  state.fill_mask(before.data());
  state.block_masks(tokens.data(), tokens.size(), masks.data());
  state.fill_mask(after.data());
  require(before == after && !state.terminated(), "tool speculation changed committed state");
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    state.fill_mask(after.data());
    require(std::equal(after.begin(), after.end(), masks.begin() + i * contract.mask_words()), "tool speculative mask differs");
    require(state.accept(tokens[i]), "committed tool token rejected");
  }
  require(state.completed() && state.terminated(), "handoff did not terminate the grammar");
}

void thinking(const text::Tokenizer& tokenizer, constraint::Compiler& compiler) {
  const auto grammar = compiler.compile({{"type", "object"}});
  constraint::State state(grammar, true);
  std::vector<std::uint32_t> mask(contract.mask_words()), before(mask.size());
  state.fill_mask(mask.data());
  require(bit(mask, 100) && !bit(mask, 101), "optional thought channel unavailable");
  std::vector<std::uint32_t> draft = {100};
  const auto plan = tokenizer.encode("thought\nAn explanation before the JSON.");
  draft.insert(draft.end(), plan.begin(), plan.end());
  draft.push_back(101);
  const auto answer = tokenizer.encode("{}");
  draft.insert(draft.end(), answer.begin(), answer.end());
  std::vector<std::uint32_t> block((draft.size() + 1) * contract.mask_words());
  before = mask;
  state.block_masks(draft.data(), draft.size(), block.data());
  state.fill_mask(mask.data());
  require(before == mask, "thinking speculative walk changed phase");
  for (const auto token : draft) require(state.accept(token), "thought/JSON transition rejected");
  require(state.completed(), "JSON following thought not complete");
  constraint::State continuation(grammar, true, true);
  continuation.fill_mask(mask.data());
  require(bit(mask, 101) && !bit(mask, 100) && !bit(mask, 1) && !bit(mask, 50), "initial reasoning mask differs");
  require(continuation.accept(101), "initial reasoning channel cannot end");
  for (const auto token : answer) require(continuation.accept(token), "continued JSON rejected");
  require(continuation.completed(), "continued JSON not complete");
}

void unicode_bytes(const text::Tokenizer& tokenizer, constraint::Compiler& compiler) {
  std::vector<std::uint32_t> bytes(256);
  for (std::uint32_t token = 0; token < contract.vocabulary_size; ++token) {
    const auto& piece = tokenizer.token_piece(token);
    if (piece.size() == 6 && piece.compare(0, 3, "<0x") == 0 && piece.back() == '>')
      bytes[std::stoul(piece.substr(3, 2), nullptr, 16)] = token;
  }
  const auto grammar = compiler.compile({{"type", "string"}, {"enum", {"é", "▁", " 😀 "}}});
  for (const std::string value : {"é", "▁", " 😀 "}) {
    std::vector<std::uint32_t> tokens;
    const auto encoded = json(value).dump();
    for (const unsigned char byte : encoded) tokens.push_back(bytes[byte]);
    require(accept_tokens(grammar, tokens), "valid fallback UTF-8 rejected");
    text::IncrementalTextDecoder decoder(tokenizer);
    std::string output;
    for (auto token : tokens) output += decoder.push(token);
    output += decoder.finish();
    require(output == encoded, "grammar and native fallback decode differ");
  }
  require(!accept_tokens(grammar, {bytes['"'], bytes[0xC3], bytes[0xA9], bytes[0xFF], bytes['"']}),
          "invalid UTF-8 byte suffix allowed");
  const auto generic = compiler.compile({{"type", "string"}});
  require(accepts(tokenizer, generic, "\" spaced 😀 \""), "ordinary space marker decoding differs");
  const auto object = compiler.compile({{"type", "object"}});
  for (const std::string quoted : {
      "\"\\ud800\"", "\"\\udfff\"", "\"\\udc00\\ud800\"",
      "\"\\ud800\\ud800\"", "\"\\ud800\\u0041\""}) {
    require(!accepts(tokenizer, generic, quoted), "unpaired surrogate escape accepted in string");
    require(!accepts(tokenizer, object, "{" + quoted + ":0}"), "unpaired surrogate escape accepted in key");
    std::vector<std::uint32_t> tokens;
    for (const unsigned char byte : quoted) tokens.push_back(bytes[byte]);
    require(!accept_tokens(generic, tokens), "fallback tokens bypassed surrogate escape validation");
  }
  for (const std::string quoted : {
      "\"\\u0000\"", "\"\\u00e9\"", "\"\\ud7ff\"", "\"\\ue000\"",
      "\"\\ud83d\\ude00\"", "\"\\uD83D\\uDE00\"", "\"\\uD800\\udc00\"", "\"\\udbff\\uDFFF\""}) {
    require(accepts(tokenizer, generic, quoted), "valid Unicode escape rejected in string");
    require(accepts(tokenizer, object, "{" + quoted + ":0}"), "valid Unicode escape rejected in key");
  }
  constraint::State escaped(generic);
  for (const unsigned char byte : std::string("\"\\ud800"))
    require(escaped.accept(bytes[byte]), "valid surrogate escape prefix rejected");
  std::vector<std::uint32_t> escape_before(contract.mask_words()), escape_after(escape_before.size());
  escaped.fill_mask(escape_before.data());
  require(!bit(escape_before, bytes['"']) && bit(escape_before, bytes['\\']),
          "incomplete surrogate pair can terminate string");
  std::vector<std::uint32_t> escaped_draft;
  for (const unsigned char byte : std::string("\\udc00\"")) escaped_draft.push_back(bytes[byte]);
  escaped_draft.push_back(106);
  std::vector<std::uint32_t> escaped_masks((escaped_draft.size() + 1) * contract.mask_words());
  escaped.block_masks(escaped_draft.data(), escaped_draft.size(), escaped_masks.data());
  escaped.fill_mask(escape_after.data());
  require(escape_before == escape_after && !escaped.terminated(), "surrogate pair draft did not roll back");
  for (auto token : escaped_draft) require(escaped.accept(token), "valid split surrogate pair rejected");
  require(escaped.terminated(), "split surrogate pair did not terminate");
  for (const std::vector<unsigned> bad : {
      std::vector<unsigned>{0xC0, 0xAF}, {0xC1, 0xBF}, {0xE0, 0x80, 0x80},
      {0xED, 0xA0, 0x80}, {0xF0, 0x80, 0x80, 0x80}, {0xF4, 0x90, 0x80, 0x80},
      {0xF5, 0x80, 0x80, 0x80}, {0xFF}, {0x80}, {0xE2, 0x82}, {0xC3, 0xA9, 0xFF}}) {
    std::vector<std::uint32_t> tokens = {bytes['"']};
    for (auto byte : bad) tokens.push_back(bytes[byte]);
    tokens.push_back(bytes['"']);
    require(!accept_tokens(generic, tokens), "generic string allowed malformed fallback UTF-8");
  }
  for (const std::string encoded : {"\"é\"", "\"😀\"", "\"é\"", "\"\\u00e9\"", "\"\\ud83d\\ude00\""}) {
    std::vector<std::uint32_t> tokens;
    for (const unsigned char byte : encoded) tokens.push_back(bytes[byte]);
    require(accept_tokens(generic, tokens), "generic string rejected valid Unicode");
    text::IncrementalTextDecoder decoder(tokenizer);
    std::string output;
    for (auto token : tokens) output += decoder.push(token);
    output += decoder.finish();
    require(json::parse(output) == json::parse(encoded), "accepted Unicode changed during native decoding");
  }
  for (unsigned control : {0U, 1U, 9U, 10U, 13U, 31U})
    require(!accept_tokens(generic, {bytes['"'], bytes[control], bytes['"']}), "raw JSON control accepted");

  constraint::State partial(generic);
  require(partial.accept(bytes['"']) && partial.accept(bytes[0xE0]), "valid UTF-8 prefix rejected");
  std::vector<std::uint32_t> before(contract.mask_words()), after(before.size());
  partial.fill_mask(before.data());
  require(!bit(before, bytes[0x80]) && bit(before, bytes[0xA0]), "UTF-8 continuation range not masked");
  const std::vector<std::uint32_t> invalid_draft = {bytes[0x80], bytes['"']};
  std::vector<std::uint32_t> masks((invalid_draft.size() + 1) * contract.mask_words());
  partial.block_masks(invalid_draft.data(), invalid_draft.size(), masks.data());
  require(std::all_of(masks.begin() + contract.mask_words(), masks.end(),
                      [](auto word) { return word == ~std::uint32_t{0}; }), "invalid UTF-8 draft suffix reachable");
  partial.fill_mask(after.data());
  require(before == after, "invalid UTF-8 draft changed pending state");
  const std::vector<std::uint32_t> valid_draft = {bytes[0xA0], bytes[0x80], bytes['"'], 106};
  masks.resize((valid_draft.size() + 1) * contract.mask_words());
  partial.block_masks(valid_draft.data(), valid_draft.size(), masks.data());
  partial.fill_mask(after.data());
  require(before == after && !partial.terminated(), "UTF-8 draft completion did not roll back");
  for (auto token : valid_draft) require(partial.accept(token), "valid UTF-8 continuation rejected");
  require(partial.terminated(), "valid UTF-8 JSON did not terminate");
}

void explicit_contract_geometry() {
  const auto fixture = small_text_contract();
  const auto tokenizer = small_tokenizer(fixture);
  require(tokenizer.completion_prompt("A") == std::vector<std::uint32_t>{256, 65},
          "completion ignored the contract BOS/vocabulary");
  constraint::Compiler compiler(tokenizer, 1);
  const auto grammar = compiler.compile(json{{"type", "null"}});
  constraint::State state(grammar, true);
  require(grammar->mask_words() == 9 && state.mask_words() == 9,
          "constraint retained a fixed vocabulary mask shape");
  std::vector<std::uint32_t> mask(state.mask_words());
  state.fill_mask(mask.data());
  require(mask[fixture.channel_start / 32] & (1U << (fixture.channel_start % 32)),
          "constraint omitted the supplied thinking channel");
  require(state.accept(fixture.channel_start) && state.accept(65) && state.accept(fixture.channel_end),
          "constraint ignored the supplied channel semantics");
  for (const auto token : tokenizer.encode("null")) require(state.accept(token), "small vocabulary rejected null");
  require(!state.accept(fixture.tool_handoff) && state.accept(fixture.constraint_stop_tokens.front()),
          "ordinary handoff and constrained stops were conflated");
  constraint::State rollback(grammar);
  const std::uint32_t draft[] = {110, 117};
  std::vector<std::uint32_t> masks(3 * rollback.mask_words());
  rejects("configured speculative depth", [&] { rollback.block_masks(draft, 2, masks.data()); });
  text::ChatOutputDecoder decoder(tokenizer);
  (void)decoder.push(fixture.channel_start);
  text::ChatDelta output;
  for (const auto token : tokenizer.encode("plan\nreason")) {
    const auto delta = decoder.push(token);
    output.reasoning += delta.reasoning;
  }
  output.reasoning += decoder.push(fixture.channel_end).reasoning;
  require(output.reasoning == "reason", "decoder ignored the supplied thought prefix");
}

void invalid_schemas(constraint::Compiler& compiler) {
  for (const json& schema : std::vector<json>{
      false, {{"type", "number"}, {"minimum", 1}, {"maximum", 0}},
      {{"type", "string"}, {"pattern", ".*"}}, {{"type", "integer"}, {"multipleOf", 2}},
      {{"type", "string"}, {"minLength", 1}}, {{"type", "string"}, {"maxLength", 1}},
      {{"$ref", "https://example.com/schema"}}, {{"$ref", "#/$defs/missing"}},
      {{"$ref", "#/default"}, {"default", {{"type", "array"}, {"uniqueItems", true}}}},
      {{"$ref", "#/default"}, {"default", {{"$ref", "https://example.com/schema"}}}},
      {{"type", "string"}, {"$id", "https://example.com/schema"}},
      {{"type", "number"}, {"minimum", 1e-8}, {"maximum", 2e-8}},
      {{"type", "object"}, {"properties", {{"x", {{"type", "string"}}}}}, {"required", {"missing"}}},
      {{"anyOf", json::array({{{"type", "string"}}})}, {"type", "integer"}},
      {{"enum", {1}}, {"type", "string"}}, {{"type", "array"}, {"maxItems", 4097}}})
    rejects(schema.dump(), [&] { (void)compiler.compile(schema); });
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) { std::cerr << "usage: json_constraint_test TOKENIZER_DIR DSGEN_FIXTURE\n"; return 1; }
  try {
    const auto tokenizer_path = std::filesystem::path(argv[1]) / "tokenizer.json";
    if (!std::filesystem::is_regular_file(tokenizer_path)) { std::cout << "SKIP: tokenizer unavailable\n"; return 77; }
    const text::Tokenizer tokenizer(tokenizer_path.string(), contract);
    constraint::Compiler compiler(tokenizer, 1279);
    explicit_contract_geometry();
    std::ifstream input(argv[2]);
    require(input.good(), "DSGen fixture unavailable");
    dsgen(tokenizer, compiler, json::parse(input));
    numeric_bounds(tokenizer, compiler);
    whitespace(tokenizer, compiler);
    object_properties(tokenizer, compiler);
    masks_and_rollback(tokenizer, compiler);
    thinking(tokenizer, compiler);
    tool_constraints(tokenizer, compiler);
    unicode_bytes(tokenizer, compiler);
    invalid_schemas(compiler);
    std::cout << "Native JSON constraints, token masks, speculative rollback, thinking and Unicode passed\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << "JSON constraint test: " << error.what() << '\n'; return 1; }
}
