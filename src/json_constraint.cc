#include "gewell/json_constraint.h"

#include <xgrammar/xgrammar.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gewell::constraint {
namespace {

using nlohmann::json;
constexpr std::size_t kSchemaBytes = 64 * 1024;
constexpr std::size_t kSchemaNodes = 4096;
constexpr unsigned kSchemaDepth = 64;
constexpr std::int64_t kCacheBytes = 64 * 1024 * 1024;

[[noreturn]] void invalid(const std::string& path, const std::string& reason) {
  throw std::invalid_argument(path + ": " + reason);
}

bool annotation(const std::string& key) {
  return key == "title" || key == "description" || key == "default" || key == "examples" ||
      key == "deprecated" || key == "readOnly" || key == "writeOnly" ||
      key == "$comment" || key == "$schema" || key == "$id";
}

bool type_matches(const json& value, const std::string& type) {
  if (type == "null") return value.is_null();
  if (type == "boolean") return value.is_boolean();
  if (type == "object") return value.is_object();
  if (type == "array") return value.is_array();
  if (type == "string") return value.is_string();
  if (type == "number") return value.is_number();
  return value.is_number_integer() || (value.is_number_float() &&
      std::isfinite(value.get<double>()) && std::floor(value.get<double>()) == value.get<double>());
}

std::set<std::string> schema_types(const json& schema, const std::string& path) {
  std::set<std::string> result;
  if (!schema.contains("type")) return result;
  const auto add = [&](const json& value) {
    if (!value.is_string()) invalid(path + ".type", "must contain JSON type names");
    const auto name = value.get<std::string>();
    if (name != "object" && name != "array" && name != "string" && name != "number" &&
        name != "integer" && name != "boolean" && name != "null") invalid(path + ".type", "unsupported JSON type");
    if (!result.insert(name).second) invalid(path + ".type", "types must be unique");
  };
  if (schema["type"].is_array()) {
    if (schema["type"].empty()) invalid(path + ".type", "must not be empty");
    for (const auto& value : schema["type"]) add(value);
  } else add(schema["type"]);
  return result;
}

struct Validation {
  std::size_t nodes = 0;
  std::set<const json*> schemas;
  std::vector<std::pair<const json*, std::string>> references;
};

void validate_schema(const json& schema, const json& root, const std::string& path,
                     unsigned depth, Validation& validation) {
  if (++validation.nodes > kSchemaNodes || depth >= kSchemaDepth) invalid(path, "schema exceeds node/depth limits");
  validation.schemas.insert(&schema);
  if (schema.is_boolean()) return;
  if (!schema.is_object()) invalid(path, "schema must be an object or boolean");
  static const std::set<std::string> keywords = {
      "$ref", "$defs", "definitions", "type", "properties", "required", "additionalProperties",
      "items", "prefixItems", "minItems", "maxItems",
      "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "enum", "const", "anyOf"};
  for (const auto& field : schema.items()) {
    if (!keywords.count(field.key()) && !annotation(field.key()))
      invalid(path + "." + field.key(), "unsupported schema keyword");
    if ((field.key() == "title" || field.key() == "description" || field.key() == "$comment" ||
         field.key() == "$schema" || field.key() == "$id") && !field.value().is_string())
      invalid(path + "." + field.key(), "annotation must be a string");
    if ((field.key() == "deprecated" || field.key() == "readOnly" || field.key() == "writeOnly") &&
        !field.value().is_boolean()) invalid(path + "." + field.key(), "annotation must be boolean");
    if (field.key() == "examples" && !field.value().is_array())
      invalid(path + ".examples", "annotation must be an array");
    if (field.key() == "$id" && !field.value().get_ref<const std::string&>().empty())
      invalid(path + ".$id", "schema identifiers that change reference scope are unsupported");
  }
  const auto types = schema_types(schema, path);
  for (const char* key : {"$defs", "definitions"}) {
    if (!schema.contains(key)) continue;
    if (!schema[key].is_object()) invalid(path + "." + key, "must be an object");
    for (const auto& field : schema[key].items())
      validate_schema(field.value(), root, path + "." + key + "." + field.key(), depth + 1, validation);
  }
  // These dispatching keywords take precedence in the upstream converter.
  // Reject assertion siblings it would otherwise silently ignore.
  for (const char* dispatch : {"$ref", "const", "enum", "anyOf"}) {
    if (!schema.contains(dispatch)) continue;
    for (const auto& field : schema.items()) {
      if (field.key() == dispatch || annotation(field.key()) || field.key() == "$defs" ||
          field.key() == "definitions") continue;
      if ((std::string_view(dispatch) == "const" || std::string_view(dispatch) == "enum") && field.key() == "type") continue;
      invalid(path + "." + field.key(), std::string("assertion sibling of ") + dispatch + " is unsupported");
    }
    if (std::string_view(dispatch) == "$ref") {
      if (!schema[dispatch].is_string()) invalid(path + ".$ref", "must be a local JSON pointer");
      const auto ref = schema[dispatch].get<std::string>();
      if (ref.empty() || ref[0] != '#' || (ref.size() > 1 && ref[1] != '/'))
        invalid(path + ".$ref", "only local JSON pointers are supported");
      try {
        const auto& target = root.at(json::json_pointer(ref.substr(1)));
        if (!target.is_object() && !target.is_boolean()) invalid(path + ".$ref", "target must be a schema");
        validation.references.emplace_back(&target, path + ".$ref");
      } catch (const json::exception&) { invalid(path + ".$ref", "target does not exist"); }
    } else if (std::string_view(dispatch) == "anyOf") {
      if (!schema[dispatch].is_array() || schema[dispatch].empty() || schema[dispatch].size() > 128)
        invalid(path + ".anyOf", "must contain 1..128 schemas");
      for (std::size_t i = 0; i < schema[dispatch].size(); ++i)
        validate_schema(schema[dispatch][i], root, path + ".anyOf[" + std::to_string(i) + "]", depth + 1, validation);
    } else {
      const auto check_value = [&](const json& value) {
        if (!types.empty() && std::none_of(types.begin(), types.end(), [&](const auto& type) { return type_matches(value, type); }))
          invalid(path + "." + dispatch, "value contradicts its type");
      };
      if (std::string_view(dispatch) == "enum") {
        if (!schema[dispatch].is_array() || schema[dispatch].empty() || schema[dispatch].size() > 256)
          invalid(path + ".enum", "must contain 1..256 values");
        for (const auto& value : schema[dispatch]) check_value(value);
      } else check_value(schema[dispatch]);
    }
    return;
  }
  const auto require_type = [&](const std::string& type, const std::string& keyword) {
    if (!types.count(type)) invalid(path + "." + keyword, "requires explicit " + type + " type");
  };
  if (schema.contains("properties")) {
    require_type("object", "properties");
    if (!schema["properties"].is_object()) invalid(path + ".properties", "must be an object");
    for (const auto& field : schema["properties"].items())
      validate_schema(field.value(), root, path + ".properties." + field.key(), depth + 1, validation);
  }
  if (schema.contains("required")) {
    require_type("object", "required");
    if (!schema["required"].is_array()) invalid(path + ".required", "must be an array");
    std::set<std::string> names;
    for (const auto& name : schema["required"]) {
      if (!name.is_string() || !names.insert(name.get<std::string>()).second)
        invalid(path + ".required", "must list unique property names");
      if (!schema.contains("properties") || !schema["properties"].contains(name.get<std::string>()))
        invalid(path + ".required", "must reference declared properties");
    }
  }
  if (schema.contains("additionalProperties")) {
    require_type("object", "additionalProperties");
    validate_schema(schema["additionalProperties"], root, path + ".additionalProperties", depth + 1, validation);
  }
  if (schema.contains("items")) {
    require_type("array", "items");
    validate_schema(schema["items"], root, path + ".items", depth + 1, validation);
  }
  if (schema.contains("prefixItems")) {
    require_type("array", "prefixItems");
    if (!schema["prefixItems"].is_array() || schema["prefixItems"].size() > 128)
      invalid(path + ".prefixItems", "must be an array of at most 128 schemas");
    for (const auto& item : schema["prefixItems"])
      validate_schema(item, root, path + ".prefixItems", depth + 1, validation);
  }
  for (const char* key : {"minItems", "maxItems"}) {
    if (!schema.contains(key)) continue;
    require_type("array", key);
    if (!schema[key].is_number_integer() || schema[key].get<long double>() < 0 || schema[key].get<long double>() > 4096)
      invalid(path + "." + key, "must be an integer in 0..4096");
  }
  if (schema.contains("minItems") && schema.contains("maxItems") && schema["minItems"] > schema["maxItems"])
      invalid(path, "minimum exceeds maximum");
  for (const char* key : {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"}) {
    if (!schema.contains(key)) continue;
    if (!types.count("integer") && !types.count("number")) invalid(path + "." + key, "requires an explicit numeric type");
    if (!schema[key].is_number() || !std::isfinite(schema[key].get<double>()))
      invalid(path + "." + key, "must be finite");
    const auto value = schema[key].get<double>();
    if (std::abs(value) > 1e9) invalid(path + "." + key, "supported bounds are within +/-1000000000");
    if (std::round(value * 1e6) / 1e6 != value)
      invalid(path + "." + key, "numeric bounds support at most six decimal places");
    if (types.count("integer") && !types.count("number") && std::floor(value) != value)
      invalid(path + "." + key, "integer bounds must be whole numbers");
  }
}

void set_bit(std::uint32_t* mask, std::uint32_t token) { mask[token / 32] |= std::uint32_t{1} << (token % 32); }
bool has_bit(const std::uint32_t* mask, std::uint32_t token) { return mask[token / 32] & (std::uint32_t{1} << (token % 32)); }
void clear_bit(std::uint32_t* mask, std::uint32_t token) { mask[token / 32] &= ~(std::uint32_t{1} << (token % 32)); }

int hex(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  return -1;
}

struct ByteTokens {
  std::array<std::uint32_t, 256> ids{};
  std::vector<std::int16_t> values;
  explicit ByteTokens(std::size_t vocabulary_size) : values(vocabulary_size, -1) {}
};

struct Utf8 {
  unsigned remaining = 0;
  unsigned lower = 0x80, upper = 0xBF;

  bool accept(unsigned byte) {
    if (remaining) {
      if (byte < lower || byte > upper) return false;
      --remaining;
      lower = 0x80;
      upper = 0xBF;
      return true;
    }
    if (byte <= 0x7F) return true;
    if (byte >= 0xC2 && byte <= 0xDF) remaining = 1;
    else if (byte >= 0xE0 && byte <= 0xEF) {
      remaining = 2;
      if (byte == 0xE0) lower = 0xA0;
      if (byte == 0xED) upper = 0x9F;
    } else if (byte >= 0xF0 && byte <= 0xF4) {
      remaining = 3;
      if (byte == 0xF0) lower = 0x90;
      if (byte == 0xF4) upper = 0x8F;
    } else return false;
    return true;
  }

  void intersect(std::uint32_t* mask, const ByteTokens& tokens) const {
    if (remaining) {
      // Normal vocabulary pieces contain complete UTF-8 scalars and cannot
      // finish a byte fallback prefix. Only its valid continuation bytes fit.
      std::array<std::uint32_t, 64> allowed{};
      std::size_t count = 0;
      for (auto byte = lower; byte <= upper; ++byte)
        if (has_bit(mask, tokens.ids[byte])) allowed[count++] = tokens.ids[byte];
      std::fill(mask, mask + (tokens.values.size() + 31) / 32, 0);
      for (std::size_t i = 0; i < count; ++i) set_bit(mask, allowed[i]);
    } else {
      for (unsigned byte = 0x80; byte <= 0xC1; ++byte) clear_bit(mask, tokens.ids[byte]);
      for (unsigned byte = 0xF5; byte <= 0xFF; ++byte) clear_bit(mask, tokens.ids[byte]);
    }
  }
};

std::vector<std::string> decoded_vocabulary(const text::Tokenizer& tokenizer) {
  std::vector<std::string> result(tokenizer.contract().vocabulary_size);
  const auto space_marker = tokenizer.contract().space_marker;
  for (std::uint32_t token = 0; token < tokenizer.contract().vocabulary_size; ++token) {
    if (tokenizer.is_special(token)) continue;
    const auto& piece = tokenizer.token_piece(token);
    if (piece.size() == 6 && piece.compare(0, 3, "<0x") == 0 && piece.back() == '>' && hex(piece[3]) >= 0 && hex(piece[4]) >= 0) {
      result[token].push_back(static_cast<char>(hex(piece[3]) * 16 + hex(piece[4])));
      continue;
    }
    for (std::size_t offset = 0; offset < piece.size();) {
      const auto found = piece.find(space_marker, offset);
      result[token].append(piece, offset, found == std::string::npos ? found : found - offset);
      if (found == std::string::npos) break;
      result[token] += ' ';
      offset = found + space_marker.size();
    }
  }
  return result;
}

void fill(xgrammar::GrammarMatcher& matcher, std::uint32_t* mask, std::size_t mask_words) {
  std::int64_t shape[] = {static_cast<std::int64_t>(mask_words)};
  DLTensor tensor{};
  tensor.data = mask;
  tensor.device = {kDLCPU, 0};
  tensor.ndim = 1;
  tensor.dtype = {kDLInt, 32, 1};
  tensor.shape = shape;
  (void)matcher.FillNextTokenBitmask(&tensor);
}

void close_declared_objects(json& schema) {
  if (!schema.is_object()) return;
  for (const char* key : {"$defs", "definitions", "properties"})
    if (schema.contains(key))
      for (auto& child : schema[key]) close_declared_objects(child);
  for (const char* key : {"anyOf", "prefixItems"})
    if (schema.contains(key))
      for (auto& child : schema[key]) close_declared_objects(child);
  for (const char* key : {"items", "additionalProperties"})
    if (schema.contains(key)) close_declared_objects(schema[key]);
  // The upstream additional-key trie excludes only literal property names;
  // escaped aliases can otherwise overwrite a declared key with a wrong type.
  // Omitting extras is a valid generation subset of every accepted schema.
  if (schema.contains("properties") && !schema["properties"].empty())
    schema["additionalProperties"] = false;
}

}  // namespace

json normalize_schema(const json& schema) {
  if (schema.dump().size() > kSchemaBytes) invalid("schema", "exceeds 64 KiB");
  Validation validation;
  validate_schema(schema, schema, "schema", 0, validation);
  for (const auto& [target, path] : validation.references)
    if (!validation.schemas.count(target)) invalid(path, "target must be a declared schema location");
  auto normalized = schema;
  close_declared_objects(normalized);
  return normalized;
}

struct Compiled::Impl {
  const text::TextContract& contract;
  std::uint32_t maximum_speculative_depth;
  xgrammar::CompiledGrammar grammar;
  std::shared_ptr<const std::vector<std::uint32_t>> ordinary_mask;
  std::shared_ptr<const ByteTokens> byte_tokens;
  std::size_t bytes;
  bool tool_mode;
  Impl(xgrammar::CompiledGrammar compiled, std::shared_ptr<const std::vector<std::uint32_t>> mask,
       std::shared_ptr<const ByteTokens> tokens, const text::TextContract& text_contract,
       std::uint32_t maximum_depth, bool tools = false)
      : contract(text_contract), maximum_speculative_depth(maximum_depth), grammar(std::move(compiled)), ordinary_mask(std::move(mask)), byte_tokens(std::move(tokens)),
        bytes(grammar.MemorySizeBytes()), tool_mode(tools) {}
};

struct Compiler::Impl {
  const text::TextContract& contract;
  std::uint32_t maximum_speculative_depth;
  xgrammar::TokenizerInfo tokenizer_info;
  xgrammar::GrammarCompiler compiler;
  std::unique_ptr<xgrammar::GrammarCompiler> tool_compiler;
  std::shared_ptr<const std::vector<std::uint32_t>> ordinary_mask;
  std::shared_ptr<const ByteTokens> byte_tokens;
  std::mutex mutex;
  Impl(const text::Tokenizer& tokenizer, std::uint32_t maximum_depth)
      : contract(tokenizer.contract()), maximum_speculative_depth(maximum_depth),
        tokenizer_info(decoded_vocabulary(tokenizer), xgrammar::VocabType::RAW, contract.vocabulary_size,
                       std::vector<std::int32_t>(contract.constraint_stop_tokens.begin(),
                                                 contract.constraint_stop_tokens.end()), false),
        compiler(tokenizer_info, 1, true, kCacheBytes) {
    auto mask = std::make_shared<std::vector<std::uint32_t>>(contract.mask_words());
    auto bytes = std::make_shared<ByteTokens>(contract.vocabulary_size);
    std::size_t byte_count = 0;
    for (std::uint32_t token = 0; token < tokenizer.contract().vocabulary_size; ++token) {
      if (!tokenizer.is_special(token)) set_bit(mask->data(), token);
      const auto& piece = tokenizer.token_piece(token);
      if (piece.size() == 6 && piece.compare(0, 3, "<0x") == 0 && piece.back() == '>' &&
          hex(piece[3]) >= 0 && hex(piece[4]) >= 0) {
        const auto byte = hex(piece[3]) * 16 + hex(piece[4]);
        bytes->values[token] = static_cast<std::int16_t>(byte);
        bytes->ids[byte] = token;
        ++byte_count;
      }
    }
    if (byte_count != 256) throw std::invalid_argument("JSON constraint requires all 256 byte fallback tokens");
    ordinary_mask = std::move(mask);
    byte_tokens = std::move(bytes);
  }
};

Compiled::Compiled(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
std::size_t Compiled::memory_bytes() const { return impl_->bytes; }
std::size_t Compiled::mask_words() const { return impl_->contract.mask_words(); }
Compiler::Compiler(const text::Tokenizer& tokenizer, std::uint32_t maximum_speculative_depth) {
  if (maximum_speculative_depth >= static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("JSON constraint speculative depth exceeds matcher rollback capacity");
  impl_ = std::make_unique<Impl>(tokenizer, maximum_speculative_depth);
}
Compiler::~Compiler() = default;

std::shared_ptr<const Compiled> Compiler::compile(const json& schema) {
  const auto normalized = normalize_schema(schema);
  try {
    std::lock_guard lock(impl_->mutex);
    auto compiled = impl_->compiler.CompileJSONSchema(normalized.dump(), true, std::nullopt,
        std::nullopt, false, 16);
    auto value = std::shared_ptr<const Compiled>(new Compiled(
        std::make_shared<const Compiled::Impl>(std::move(compiled), impl_->ordinary_mask, impl_->byte_tokens,
                                             impl_->contract, impl_->maximum_speculative_depth)));
    State state(value);
    std::vector<std::uint32_t> initial(value->mask_words());
    state.fill_mask(initial.data());
    if (std::none_of(initial.begin(), initial.end(), [](auto word) { return word != 0; }))
      invalid("schema", "grammar has no valid initial token");
    return value;
  } catch (const std::bad_alloc&) { throw; }
  catch (const std::invalid_argument&) { throw; }
  catch (const std::exception& error) { invalid("schema", error.what()); }
}

namespace {
void strict_objects(const json& schema) {
  if (!schema.is_object()) return;
  const auto types = schema_types(schema, "schema");
  if (types.count("object")) {
    if (!schema.contains("additionalProperties") || schema["additionalProperties"] != false)
      invalid("schema.additionalProperties", "strict objects require false");
    const auto properties = schema.value("properties", json::object());
    const auto required = schema.value("required", json::array());
    if (required.size() != properties.size())
      invalid("schema.required", "strict objects require every property");
  }
  for (const char* key : {"properties", "$defs", "definitions", "anyOf", "prefixItems"})
    if (schema.contains(key)) for (const auto& child : schema[key]) strict_objects(child);
  for (const char* key : {"items", "additionalProperties"})
    if (schema.contains(key)) strict_objects(schema[key]);
}
}  // namespace

std::shared_ptr<const Compiled> Compiler::compile_tools(
    const json& tools, const std::vector<std::string>& selected,
    bool required, bool parallel, std::shared_ptr<const Compiled> answer) {
  try {
    std::vector<xgrammar::NamedGrammar> arguments;
    std::string calls;
    for (std::size_t i = 0; i < tools.size(); ++i) {
      const auto& function = tools[i].at("function");
      json schema = {{"type", "object"}};
      if (function.value("strict", false)) {
        schema = normalize_schema(function.at("parameters"));
        strict_objects(function.at("parameters"));
      }
      auto argument = xgrammar::Grammar::FromJSONSchema(
          schema.dump(), true, std::nullopt, std::nullopt, false, 16);
      const auto name = function.at("name").get<std::string>();
      if (std::find(selected.begin(), selected.end(), name) == selected.end()) continue;
      arguments.push_back({"args" + std::to_string(i), std::move(argument)});
      if (!calls.empty()) calls += " | ";
      calls += json("call:" + name).dump() + " @args" + std::to_string(i);
    }
    if (required && calls.empty()) invalid("tools", "required selection needs a function");
    const auto token = [](std::uint32_t id) { return "<[" + std::to_string(id) + "]>"; };
    const auto& contract = impl_->contract;
    std::string eos;
    for (auto id : contract.constraint_stop_tokens) {
      if (!eos.empty()) eos += " | ";
      eos += token(id);
    }
    std::string grammar = "start: ";
    if (required) grammar += "tool_turn\n";
    else if (answer) {
      arguments.push_back({"answer", answer->impl_->grammar.GetGrammar()});
      grammar += "@answer eos" + std::string(calls.empty() ? "\n" : " | tool_turn\n");
    } else {
      grammar += calls.empty() ? "text eos\n" : "text (eos | tool_turn)\n";
      grammar += "text: /[^\\x00]*/\n";
    }
    grammar += "eos: " + eos + "\n";
    if (!calls.empty()) {
      grammar += "tool_turn: call" + std::string(parallel ? " ~ 1..128 " : " ") + token(contract.tool_handoff) + "\n";
      grammar += "call: " + token(contract.tool_call_start) + " (" + calls + ") " + token(contract.tool_call_end) + "\n";
    }
    auto parsed = xgrammar::Grammar::FromLark(grammar, std::nullopt, arguments);
    std::lock_guard lock(impl_->mutex);
    if (!impl_->tool_compiler) {
      auto vocabulary = impl_->tokenizer_info.GetDecodedVocab();
      // Nonempty placeholders make boundary tokens available to XGrammar's
      // atomic token rules. NUL cannot match JSON or the ordinary-text rule,
      // so a boundary token cannot masquerade as argument/content bytes.
      for (auto id : contract.constraint_stop_tokens) vocabulary[id] = std::string(1, '\0');
      for (auto id : {contract.tool_call_start, contract.tool_call_end, contract.tool_handoff})
        vocabulary[id] = std::string(1, '\0');
      xgrammar::TokenizerInfo info(vocabulary, xgrammar::VocabType::RAW, contract.vocabulary_size,
                                  std::vector<std::int32_t>{int(contract.bos)}, false);
      impl_->tool_compiler = std::make_unique<xgrammar::GrammarCompiler>(info, 1, true, kCacheBytes);
    }
    auto compiled = impl_->tool_compiler->CompileGrammar(parsed);
    auto mask = std::make_shared<std::vector<std::uint32_t>>(*impl_->ordinary_mask);
    for (auto id : contract.constraint_stop_tokens) set_bit(mask->data(), id);
    for (auto id : {contract.tool_call_start, contract.tool_call_end, contract.tool_handoff})
      set_bit(mask->data(), id);
    return std::shared_ptr<const Compiled>(new Compiled(std::make_shared<const Compiled::Impl>(
        std::move(compiled), std::move(mask), impl_->byte_tokens, contract,
        impl_->maximum_speculative_depth, true)));
  } catch (const std::bad_alloc&) { throw; }
  catch (const std::invalid_argument&) { throw; }
  catch (const std::exception& error) { invalid("tools", error.what()); }
}

struct State::Impl {
  enum class Phase { begin, thinking, answer };
  std::shared_ptr<const Compiled> compiled;
  xgrammar::GrammarMatcher matcher;
  Phase phase;
  std::uint64_t grammar_steps = 0;
  Utf8 utf8;
  Impl(std::shared_ptr<const Compiled> value, bool allow_thinking, bool initial_reasoning)
      : compiled(std::move(value)), matcher(compiled->impl_->grammar, std::nullopt,
          compiled->impl_->tool_mode, compiled->impl_->maximum_speculative_depth + 1),
        phase(initial_reasoning ? Phase::thinking : allow_thinking ? Phase::begin : Phase::answer) {}

  void fill_mask(std::uint32_t* mask) {
    if (matcher.IsTerminated()) throw std::logic_error("cannot mask a terminated JSON constraint");
    if (phase == Phase::thinking) {
      std::copy(compiled->impl_->ordinary_mask->begin(), compiled->impl_->ordinary_mask->end(), mask);
      for (auto id : compiled->impl_->contract.stop_tokens) clear_bit(mask, id);
      clear_bit(mask, compiled->impl_->contract.tool_call_start);
      clear_bit(mask, compiled->impl_->contract.tool_call_end);
      set_bit(mask, compiled->impl_->contract.channel_end);
      return;
    }
    fill(matcher, mask, compiled->mask_words());
    utf8.intersect(mask, *compiled->impl_->byte_tokens);
    if (phase == Phase::begin) set_bit(mask, compiled->impl_->contract.channel_start);
  }

  bool accept(std::uint32_t token) {
    if (token >= compiled->impl_->contract.vocabulary_size || matcher.IsTerminated()) return false;
    if (phase == Phase::thinking) {
      if (token == compiled->impl_->contract.channel_end) { phase = Phase::answer; return true; }
      if (compiled->impl_->contract.is_stop(token) || token == compiled->impl_->contract.tool_call_start ||
          token == compiled->impl_->contract.tool_call_end) return false;
      return has_bit(compiled->impl_->ordinary_mask->data(), token);
    }
    if (phase == Phase::begin && token == compiled->impl_->contract.channel_start) { phase = Phase::thinking; return true; }
    if (!compiled->impl_->contract.is_constraint_stop(token) && !has_bit(compiled->impl_->ordinary_mask->data(), token)) return false;
    auto next_utf8 = utf8;
    const auto byte = compiled->impl_->byte_tokens->values[token];
    if (byte >= 0) {
      if (!next_utf8.accept(static_cast<unsigned>(byte))) return false;
    } else if (utf8.remaining) return false;
    if (!matcher.AcceptToken(static_cast<std::int32_t>(token))) return false;
    utf8 = next_utf8;
    ++grammar_steps;
    phase = Phase::answer;
    return true;
  }
};

State::State(std::shared_ptr<const Compiled> compiled, bool allow_thinking, bool initial_reasoning) {
  if (!compiled) throw std::invalid_argument("JSON constraint requires a compiled grammar");
  impl_ = std::make_unique<Impl>(std::move(compiled), allow_thinking, initial_reasoning);
}
State::~State() = default;
std::size_t State::mask_words() const { return impl_->compiled->mask_words(); }
void State::fill_mask(std::uint32_t* mask) {
  if (!mask) throw std::invalid_argument("JSON constraint mask buffer is null");
  impl_->fill_mask(mask);
}
bool State::accept(std::uint32_t token) { return impl_->accept(token); }
bool State::completed() const { return impl_->phase == Impl::Phase::answer && impl_->matcher.IsCompleted(); }
bool State::terminated() const { return impl_->matcher.IsTerminated(); }

void State::block_masks(const std::uint32_t* draft_ids, std::uint32_t depth, std::uint32_t* masks) {
  if (depth > impl_->compiled->impl_->maximum_speculative_depth || !masks || (depth && !draft_ids))
    throw std::invalid_argument("invalid speculative JSON constraint buffers/depth");
  auto& state = *impl_;
  const auto phase = state.phase;
  const auto steps = state.grammar_steps;
  const auto utf8 = state.utf8;
  const auto rollback = [&] {
    const auto count = state.grammar_steps - steps;
    if (count) state.matcher.Rollback(static_cast<int>(count));
    state.grammar_steps = steps;
    state.phase = phase;
    state.utf8 = utf8;
  };
  try {
    bool reachable = !state.matcher.IsTerminated();
    for (std::uint32_t row = 0; row <= depth; ++row) {
      auto* mask = masks + std::size_t(row) * state.compiled->mask_words();
      if (reachable) state.fill_mask(mask);
      else std::fill(mask, mask + state.compiled->mask_words(), ~std::uint32_t{0});
      if (row < depth && reachable)
        reachable = state.accept(draft_ids[row]) && !state.matcher.IsTerminated();
    }
  } catch (...) { rollback(); throw; }
  rollback();
}

}  // namespace gewell::constraint
