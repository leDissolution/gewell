#include "gewell/models/gemma4/text_contract.h"
#include "../../../text/utf8.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace gewell::gemma4::text_detail {
using namespace gewell::text;
namespace {

using gewell::text::detail::read_scalar;
constexpr std::string_view kChannelStart = "<|channel>";
constexpr std::string_view kChannelEnd = "<channel|>";
bool valid_role(std::string_view role) {
  return role == "system" || role == "developer" || role == "user" ||
         role == "assistant" || role == "tool";
}

using nlohmann::json;

[[noreturn]] void invalid(const std::string& path, const std::string& message) {
  throw std::invalid_argument(path + " " + message);
}

void keys(const json& object, std::initializer_list<const char*> allowed,
          const std::string& path) {
  if (!object.is_object()) invalid(path, "must be an object");
  for (const auto& field : object.items()) {
    if (std::none_of(allowed.begin(), allowed.end(), [&](const char* key) { return field.key() == key; }))
      invalid(path + "." + field.key(), "is unsupported");
  }
}

bool identifier(std::string_view value, bool function = false) {
  if (value.empty() || (function && value.size() > 64)) return false;
  return std::all_of(value.begin(), value.end(), [](unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_' || c == '-';
  });
}

std::string string_field(const json& object, const char* key, const std::string& path,
                         bool required = true) {
  const auto found = object.find(key);
  if (found == object.end() || found->is_null()) {
    if (required) invalid(path + "." + key, "must be a string");
    return {};
  }
  if (!found->is_string()) invalid(path + "." + key, "must be a string");
  return found->get<std::string>();
}

void valid_id(const std::string& value, const std::string& path) {
  if (value.empty() || value.size() > 128 || value.find('\0') != std::string::npos)
    invalid(path, "must contain 1..128 bytes without NUL");
}

void validate_argument(const json& value, const std::string& path, unsigned depth = 0) {
  if (depth > 32 || (depth >= 32 && (value.is_object() || value.is_array())))
    invalid(path, "exceeds the 32-level nesting limit");
  if (value.is_object()) {
    for (const auto& field : value.items()) {
      validate_argument(field.value(), path + "." + field.key(), depth + 1);
    }
  } else if (value.is_array()) {
    for (const auto& item : value) validate_argument(item, path, depth + 1);
  } else if (value.is_number_float() && !std::isfinite(value.get<double>())) {
    invalid(path, "must contain finite JSON numbers");
  }
}

// Non-strict declarations describe arguments to the model. Preserve schema
// assertions and annotations; the constraint compiler validates its enforceable
// subset separately when strict mode is requested.
json normalize_schema(json schema, const std::string& path, unsigned depth = 0) {
  if (depth >= 32) invalid(path, "exceeds the 32-level nesting limit");
  if (schema.is_boolean()) return schema;
  if (!schema.is_object()) invalid(path, "must be an object or boolean schema");
  for (const char* key : {"$schema", "$id", "$ref", "$comment", "title", "description", "pattern", "format"})
    if (schema.contains(key) && !schema[key].is_string()) invalid(path + "." + key, "must be a string");
  schema.erase("$schema");
  for (const char* key : {"nullable", "deprecated", "readOnly", "writeOnly", "uniqueItems"})
    if (schema.contains(key) && !schema[key].is_boolean()) invalid(path + "." + key, "must be boolean");
  if (schema.contains("examples") && !schema["examples"].is_array()) invalid(path + ".examples", "must be an array");
  if (schema.contains("type")) {
    const auto types = schema["type"].is_array() ? schema["type"] : json::array({schema["type"]});
    if (types.empty()) invalid(path + ".type", "must not be empty");
    std::set<std::string> seen;
    for (const auto& value : types) {
      if (!value.is_string()) invalid(path + ".type", "must contain JSON type names");
      const auto type = value.get<std::string>();
      if ((type != "string" && type != "integer" && type != "number" && type != "boolean" &&
           type != "object" && type != "array" && type != "null") || !seen.insert(type).second)
        invalid(path + ".type", "must contain unique JSON type names");
    }
  }
  if (schema.contains("enum") && (!schema["enum"].is_array() || schema["enum"].empty()))
    invalid(path + ".enum", "must be a nonempty array");
  for (const char* key : {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"})
    if (schema.contains(key) && (!schema[key].is_number() || !std::isfinite(schema[key].get<double>()) ||
        (std::string_view(key) == "multipleOf" && schema[key].get<double>() <= 0)))
      invalid(path + "." + key, "must be a finite number (positive for multipleOf)");
  for (const char* key : {"minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties", "minContains", "maxContains"})
    if (schema.contains(key) && (!schema[key].is_number_integer() || schema[key].get<long double>() < 0))
      invalid(path + "." + key, "must be a nonnegative integer");
  if (schema.value("type", json()) == "object" && !schema.contains("properties")) schema["properties"] = json::object();
  for (const char* key : {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas", "dependencies"}) {
    if (!schema.contains(key)) continue;
    if (!schema[key].is_object()) invalid(path + "." + key, "must be an object");
    for (auto& field : schema[key].items()) {
      if (std::string_view(key) == "dependencies" && field.value().is_array()) {
        for (const auto& name : field.value())
          if (!name.is_string()) invalid(path + ".dependencies." + field.key(), "must list property names");
      } else field.value() = normalize_schema(field.value(), path + "." + key + "." + field.key(), depth + 1);
    }
  }
  if (schema.contains("required")) {
    if (!schema["required"].is_array()) invalid(path + ".required", "must be an array");
    std::set<std::string> seen;
    for (const auto& item : schema["required"])
      if (!item.is_string() || !seen.insert(item.get<std::string>()).second)
        invalid(path + ".required", "must list unique property names");
  }
  for (const char* key : {"anyOf", "oneOf", "allOf", "prefixItems"}) {
    if (!schema.contains(key)) continue;
    if (!schema[key].is_array() || schema[key].empty()) invalid(path + "." + key, "must be a nonempty schema array");
    for (auto& child : schema[key]) child = normalize_schema(child, path + "." + key, depth + 1);
  }
  for (const char* key : {"items", "additionalProperties", "additionalItems", "contains", "propertyNames",
                          "not", "if", "then", "else", "unevaluatedProperties", "unevaluatedItems", "contentSchema"}) {
    if (!schema.contains(key)) continue;
    if (std::string_view(key) == "items" && schema[key].is_array()) {
      for (auto& child : schema[key]) child = normalize_schema(child, path + ".items", depth + 1);
    } else schema[key] = normalize_schema(schema[key], path + "." + key, depth + 1);
  }
  return schema;
}

std::string lowercase(std::string value) {
  for (char& c : value) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
  return value;
}

std::string uppercase(std::string value) {
  for (char& c : value) if (c >= 'a' && c <= 'z') c -= 'a' - 'A';
  return value;
}

// Jinja dictsort is case-insensitive. Inputs are normalized JSON maps first,
// preserving a deterministic tie order when keys differ only in ASCII case.
std::vector<std::string> sorted_keys(const json& object) {
  std::vector<std::string> result;
  for (const auto& field : object.items()) result.push_back(field.key());
  std::stable_sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
    return lowercase(a) < lowercase(b);
  });
  return result;
}

std::string gemma_quote(std::string_view value) { return "<|\"|>" + std::string(value) + "<|\"|>"; }

std::string format_argument(const json& value, bool escape_keys = true) {
  if (value.is_string()) return gemma_quote(value.get_ref<const std::string&>());
  if (value.is_object()) {
    std::string result = "{";
    for (const auto& key : sorted_keys(value)) {
      if (result.size() > 1) result += ',';
      result += (escape_keys || !identifier(key) ? gemma_quote(key) : key) + ":" + format_argument(value[key], escape_keys);
    }
    return result + "}";
  }
  if (value.is_array()) {
    std::string result = "[";
    for (const auto& item : value) {
      if (result.size() > 1) result += ',';
      result += format_argument(item, escape_keys);
    }
    return result + "]";
  }
  auto result = value.dump();
  // Jinja uses Python's float repr: decimal through exponent 15. The JSON
  // serializer switches to scientific one exponent earlier. Expand only that
  // differing interval, preserving the shortest digits and float suffix.
  if (value.is_number_float()) {
    const auto exponent = result.find('e');
    if (exponent != std::string::npos && result.substr(exponent) == "e+15") {
      result.resize(exponent);
      const auto point = result.find('.');
      if (point != std::string::npos) result.erase(point, 1);
      const auto sign = result.front() == '-' ? 1U : 0U;
      const auto digits = result.size() - sign;
      if (digits <= 16) result += std::string(16 - digits, '0') + ".0";
      else result.insert(sign + 16, ".");
    }
  }
  return result;
}

bool truthy(const json& value) {
  if (value.is_null()) return false;
  if (value.is_boolean()) return value.get<bool>();
  return !value.empty();
}

const json& get(const json& value, const char* key) {
  static const json absent;
  const auto found = value.find(key);
  return found == value.end() ? absent : *found;
}

std::string format_parameters(const json& properties) {
  std::string result;
  for (const auto& key : sorted_keys(properties)) {
    const auto& value = properties[key];
    const auto type = value.at("type").get<std::string>();
    if (!result.empty()) result += ',';
    result += key + ":{";
    std::string fields;
    const auto append = [&](const std::string& field) {
      if (!fields.empty()) fields += ',';
      fields += field;
    };
    if (truthy(get(value, "description"))) append("description:" + gemma_quote(value["description"].get<std::string>()));
    if (type == "string" && truthy(get(value, "enum"))) append("enum:" + format_argument(value["enum"]));
    if (type == "array") {
      std::string items;
      for (const auto& item_key : sorted_keys(value["items"])) {
        const auto& item = value["items"][item_key];
        if (item.is_null()) continue;
        if (!items.empty()) items += ',';
        items += item_key + ":";
        if (item_key == "properties") items += "{" + format_parameters(item) + "}";
        else if (item_key == "type") items += gemma_quote(uppercase(item.get<std::string>()));
        else items += format_argument(item);
      }
      append("items:{" + items + "}");
    }
    if (truthy(get(value, "nullable"))) append("nullable:true");
    if (type == "object") {
      append("properties:{" + format_parameters(value["properties"]) + "}");
      if (truthy(get(value, "required"))) append("required:" + format_argument(value["required"]));
    }
    append("type:" + gemma_quote(uppercase(type)));
    result += fields + "}";
  }
  return result;
}

bool native_schema(const json& schema) {
  if (!schema.is_object() || !schema.contains("type") || !schema["type"].is_string()) return false;
  for (const auto& field : schema.items()) {
    if (field.key() != "type" && field.key() != "description" && field.key() != "enum" &&
        field.key() != "nullable" && field.key() != "properties" && field.key() != "required" &&
        field.key() != "items" && field.key() != "additionalProperties") return false;
  }
  const auto type = schema["type"].get<std::string>();
  if (schema.contains("enum") && (type != "string" ||
      std::any_of(schema["enum"].begin(), schema["enum"].end(), [](const auto& v) { return !v.is_string(); }))) return false;
  if (schema.contains("additionalProperties") && !schema["additionalProperties"].is_boolean()) return false;
  if (schema.contains("properties")) {
    if (type != "object") return false;
    for (const auto& field : schema["properties"].items())
      if (!identifier(field.key()) || !native_schema(field.value())) return false;
  }
  if (type == "array") return schema.contains("items") && native_schema(schema["items"]);
  return !schema.contains("items");
}

json declaration_schema(json schema) {
  if (!schema.is_object()) return schema;
  if (schema.contains("type")) {
    if (schema["type"].is_string()) schema["type"] = uppercase(schema["type"].get<std::string>());
    else for (auto& type : schema["type"]) type = uppercase(type.get<std::string>());
  }
  for (const char* key : {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas", "dependencies",
                          "anyOf", "oneOf", "allOf", "prefixItems"})
    if (schema.contains(key)) for (auto& child : schema[key]) child = declaration_schema(child);
  for (const char* key : {"items", "additionalProperties", "additionalItems", "contains", "propertyNames",
                          "not", "if", "then", "else", "unevaluatedProperties", "unevaluatedItems", "contentSchema"}) {
    if (!schema.contains(key)) continue;
    if (schema[key].is_array()) for (auto& child : schema[key]) child = declaration_schema(child);
    else schema[key] = declaration_schema(schema[key]);
  }
  return schema;
}

std::string format_declaration(const json& tool) {
  const auto& function = tool.at("function");
  std::string result = "declaration:" + function.at("name").get<std::string>() +
      "{description:" + gemma_quote(function.at("description").get<std::string>());
  const auto& parameters = function.at("parameters");
  if (!native_schema(parameters) || parameters.contains("description") || parameters.contains("nullable") || parameters.contains("enum"))
    return result + ",parameters:" + format_argument(declaration_schema(parameters), false) + "}";
  result += ",parameters:{";
  if (!parameters["properties"].empty()) result += "properties:{" + format_parameters(parameters["properties"]) + "},";
  if (truthy(get(parameters, "required"))) result += "required:" + format_argument(parameters["required"]) + ",";
  return result + "type:" + gemma_quote("OBJECT") + "}}";
}

bool python_whitespace(std::uint32_t scalar) {
  return (scalar >= 0x09 && scalar <= 0x0D) ||
         (scalar >= 0x1C && scalar <= 0x20) || scalar == 0x85 ||
         scalar == 0xA0 || scalar == 0x1680 ||
         (scalar >= 0x2000 && scalar <= 0x200A) || scalar == 0x2028 ||
         scalar == 0x2029 || scalar == 0x202F || scalar == 0x205F ||
         scalar == 0x3000;
}

std::string trim(std::string_view text) {
  std::size_t begin = text.size();
  std::size_t end = 0;
  std::size_t offset = 0;
  while (offset < text.size()) {
    const std::size_t previous = offset;
    std::uint32_t scalar = 0;
    if (!read_scalar(text, offset, scalar)) {
      throw std::invalid_argument("chat content must be valid UTF-8");
    }
    if (!python_whitespace(scalar)) {
      if (begin == text.size()) begin = previous;
      end = offset;
    }
  }
  return end == 0 ? std::string{} : std::string(text.substr(begin, end - begin));
}

std::string strip_thinking(std::string_view text) {
  std::string result;
  std::size_t offset = 0;
  for (;;) {
    const auto end = text.find(kChannelEnd, offset);
    const auto part = text.substr(offset, end == std::string_view::npos
                                            ? end : end - offset);
    result.append(part.substr(0, part.find(kChannelStart)));
    if (end == std::string_view::npos) break;
    offset = end + kChannelEnd.size();
  }
  return trim(result);
}

}  // namespace

json normalize_chat_tools(const json& tools) {
  if (tools.is_null()) return json::array();
  if (!tools.is_array() || tools.size() > 128) invalid("tools", "must be an array of at most 128 functions");
  json result = json::array();
  std::set<std::string> names;
  for (std::size_t index = 0; index < tools.size(); ++index) {
    const auto path = "tools[" + std::to_string(index) + "]";
    const auto& tool = tools[index];
    keys(tool, {"type", "function"}, path);
    if (get(tool, "type") != "function") invalid(path + ".type", "must be function");
    const auto& input = get(tool, "function");
    keys(input, {"name", "description", "parameters", "strict"}, path + ".function");
    const auto name = string_field(input, "name", path + ".function");
    if (!identifier(name, true)) invalid(path + ".function.name", "must use 1..64 letters, digits, underscores or hyphens");
    if (!names.insert(name).second) invalid(path + ".function.name", "must be unique");
    const auto& strict = get(input, "strict");
    if (!strict.is_null() && !strict.is_boolean()) invalid(path + ".function.strict", "must be boolean or null");
    json function = {{"name", name}, {"description", string_field(input, "description", path + ".function", false)}};
    if (strict == true) function["strict"] = true;
    json parameters = get(input, "parameters");
    if (parameters.is_null()) {
      parameters = {{"type", "object"}, {"properties", json::object()}};
      if (strict == true) {
        parameters["additionalProperties"] = false;
        parameters["required"] = json::array();
      }
    }
    if (!parameters.is_object()) invalid(path + ".function.parameters", "must be an object schema");
    if (parameters.dump().size() > 64 * 1024) invalid(path + ".function.parameters", "exceeds 64 KiB");
    if (!parameters.contains("type")) parameters["type"] = "object";
    function["parameters"] = normalize_schema(std::move(parameters), path + ".function.parameters");
    if (function["parameters"]["type"] != "object") invalid(path + ".function.parameters.type", "must be object");
    result.push_back({{"type", "function"}, {"function", std::move(function)}});
  }
  return result;
}

std::vector<ChatMessage> normalize_chat_messages(const json& messages) {
  if (!messages.is_array() || messages.empty()) invalid("messages", "must be a nonempty array");
  std::vector<ChatMessage> result;
  result.reserve(messages.size());
  std::set<std::string> call_ids;
  std::map<std::string, std::string> pending;
  for (std::size_t index = 0; index < messages.size(); ++index) {
    const auto& message = messages[index];
    const std::string path = "messages[" + std::to_string(index) + "]";
    keys(message, {"role", "content", "reasoning", "reasoning_content", "tool_calls",
                   "tool_call_id", "name", "tool_responses", "function_call",
                   "refusal", "annotations", "audio"}, path);
    ChatMessage normalized;
    normalized.role = string_field(message, "role", path);
    if (!valid_role(normalized.role)) invalid(path + ".role", "is unsupported");
    if (normalized.role != "tool" && !pending.empty()) invalid(path, "is missing results for the preceding assistant tool calls");
    for (const char* key : {"refusal", "annotations", "audio"}) {
      if (!message.contains(key)) continue;
      const auto& value = message[key];
      if (normalized.role != "assistant" || (!value.is_null() &&
          !(std::string_view(key) == "annotations" && value.is_array() && value.empty())))
        invalid(path + "." + key, "only supports its empty assistant-message value");
    }
    for (const char* key : {"tool_responses", "function_call"}) {
      const auto& value = get(message, key);
      if (!value.is_null() && !(std::string_view(key) == "tool_responses" && value.is_array() && value.empty()))
        invalid(path + "." + key, "is unsupported");
    }
    for (const char* key : {"reasoning", "reasoning_content"}) {
      const auto value = string_field(message, key, path, false);
      if (normalized.reasoning.empty()) normalized.reasoning = value;
    }
    const auto& calls = get(message, "tool_calls");
    if (!calls.is_null()) {
      if (!calls.is_array() || calls.size() > 128) invalid(path + ".tool_calls", "must be an array of at most 128 calls");
      if (!calls.empty() && normalized.role != "assistant") invalid(path + ".tool_calls", "requires assistant role");
      for (std::size_t call_index = 0; call_index < calls.size(); ++call_index) {
        const auto call_path = path + ".tool_calls[" + std::to_string(call_index) + "]";
        const auto& call = calls[call_index];
        keys(call, {"id", "type", "function"}, call_path);
        if (get(call, "type") != "function") invalid(call_path + ".type", "must be function");
        ChatToolCall parsed;
        parsed.id = string_field(call, "id", call_path);
        valid_id(parsed.id, call_path + ".id");
        if (!call_ids.insert(parsed.id).second) invalid(call_path + ".id", "must be unique in the conversation");
        const auto& function = get(call, "function");
        keys(function, {"name", "arguments"}, call_path + ".function");
        parsed.name = string_field(function, "name", call_path + ".function");
        if (!identifier(parsed.name, true)) invalid(call_path + ".function.name", "must use 1..64 letters, digits, underscores or hyphens");
        parsed.arguments = get(function, "arguments");
        if (parsed.arguments.is_string()) {
          try { parsed.arguments = json::parse(parsed.arguments.get_ref<const std::string&>()); }
          catch (const json::exception&) { invalid(call_path + ".function.arguments", "must contain valid JSON"); }
        }
        if (!parsed.arguments.is_object()) invalid(call_path + ".function.arguments", "must be a JSON object or JSON object string");
        validate_argument(parsed.arguments, call_path + ".function.arguments");
        pending.emplace(parsed.id, parsed.name);
        normalized.tool_calls.push_back(std::move(parsed));
      }
    }
    normalized.tool_call_id = string_field(message, "tool_call_id", path, false);
    normalized.name = string_field(message, "name", path, false);
    if (normalized.role == "tool") {
      valid_id(normalized.tool_call_id, path + ".tool_call_id");
      const auto found = pending.find(normalized.tool_call_id);
      if (found == pending.end()) invalid(path + ".tool_call_id", "does not match an outstanding assistant call");
      if (!normalized.name.empty() && normalized.name != found->second) invalid(path + ".name", "does not match the associated function");
      normalized.name = found->second;
      pending.erase(found);
    } else if (!normalized.tool_call_id.empty() || !normalized.name.empty()) {
      invalid(path, "name/tool_call_id require tool role");
    }
    const auto& content = get(message, "content");
    if (content.is_string()) normalized.content = content.get_ref<const std::string&>();
    else if (content.is_array()) {
      for (std::size_t part_index = 0; part_index < content.size(); ++part_index) {
        const auto& part = content[part_index];
        if (!part.is_object() || part.size() != 2 || get(part, "type") != "text" || !get(part, "text").is_string())
          invalid(path + ".content[" + std::to_string(part_index) + "]", "must be a text part");
        normalized.content += part["text"].get_ref<const std::string&>();
      }
    } else if (!(content.is_null() && normalized.role == "assistant" && !normalized.tool_calls.empty())) {
      invalid(path + ".content", "must be text (or null for assistant tool calls)");
    }
    result.push_back(std::move(normalized));
  }
  if (!pending.empty()) invalid("messages", "is missing results for the final assistant tool calls");
  return result;
}

ChatTemplateOptions parse_chat_template_kwargs(const nlohmann::json& value) {
  if (value.is_null()) return {};
  if (!value.is_object()) {
    throw std::invalid_argument("chat_template_kwargs must be an object or null");
  }
  for (const auto& field : value.items()) {
    if (field.key() != "enable_thinking" && field.key() != "preserve_thinking") {
      throw std::invalid_argument("chat_template_kwargs." + field.key() +
                                  " is unsupported");
    }
    if (!field.value().is_boolean()) {
      throw std::invalid_argument("chat_template_kwargs." + field.key() +
                                  " must be boolean");
    }
  }
  return {value.value("enable_thinking", false),
          value.value("preserve_thinking", false)};
}

std::string render_chat(const std::vector<ChatMessage>& messages,
                        ChatTemplateOptions options, const json& tools) {
  if (messages.empty()) throw std::invalid_argument("messages must not be empty");
  for (const auto& message : messages) {
    if (!valid_role(message.role)) throw std::invalid_argument("unsupported chat role: " + message.role);
    if (message.role == "tool" && (message.name.empty() || message.tool_call_id.empty()))
      throw std::invalid_argument("tool messages require a normalized name and tool_call_id");
  }
  std::string result = "<bos>";
  std::size_t first = 0;
  const bool initial_system = messages[0].role == "system" || messages[0].role == "developer";
  if (options.enable_thinking || initial_system || !tools.empty()) {
    result += "<|turn>system\n";
    if (options.enable_thinking) result += "<|think|>\n";
    if (initial_system) {
      result += trim(messages[0].content);
      first = 1;
    }
    for (const auto& tool : tools) result += "<|tool>" + format_declaration(tool) + "<tool|>";
    result += "<turn|>\n";
  }
  std::size_t current_turn_begin = first;
  for (std::size_t index = first; index < messages.size(); ++index)
    if (messages[index].role == "user") current_turn_begin = index + 1;
  std::vector<std::size_t> next_non_tool(messages.size(), messages.size());
  std::size_t next = messages.size();
  for (std::size_t index = messages.size(); index > first;) {
    --index;
    next_non_tool[index] = next;
    if (messages[index].role != "tool") next = index;
  }
  std::string previous_role;
  bool last_call = false, last_response = false;
  for (std::size_t index = first; index < messages.size(); ++index) {
    const auto& message = messages[index];
    if (message.role == "tool") continue;
    last_call = last_response = false;
    const bool assistant = message.role == "assistant";
    if (!(assistant && previous_role == "assistant")) {
      result += "<|turn>";
      result += assistant ? "model" : message.role;
      result += '\n';
    }
    if ((index >= current_turn_begin || (options.preserve_thinking && !message.tool_calls.empty())) &&
        !message.reasoning.empty())
      result += "<|channel>thought\n" + message.reasoning + "\n<channel|>";
    for (const auto& call : message.tool_calls) {
      result += "<|tool_call>call:" + call.name + format_argument(call.arguments, false) + "<tool_call|>";
      last_call = true;
    }
    if (!message.tool_calls.empty()) {
      for (std::size_t follow = index + 1; follow < messages.size() && messages[follow].role == "tool"; ++follow) {
        const auto& response = messages[follow];
        result += "<|tool_response>response:" + response.name + "{value:" + gemma_quote(response.content) + "}<tool_response|>";
        last_response = true;
        last_call = false;
      }
    }
    const auto content = assistant ? strip_thinking(message.content) : trim(message.content);
    result += content;
    const auto following = next_non_tool[index];
    const bool continues = assistant && following < messages.size() &&
        messages[following].role == "assistant" && (message.tool_calls.empty() || last_response);
    if (last_call) result += "<|tool_response>";
    else if (!continues && !(last_response && content.empty() && following == messages.size()))
      result += "<turn|>\n";
    previous_role = message.role;
  }
  if (!last_call && !last_response) {
    result += "<|turn>model\n";
    if (!options.enable_thinking) result += "<|channel>thought\n<channel|>";
  } else if (last_response && options.enable_thinking) result += "<|channel>thought\n";
  return result;
}

}  // namespace gewell::gemma4::text_detail
