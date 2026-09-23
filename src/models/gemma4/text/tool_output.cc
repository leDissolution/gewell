#include "gewell/tool_output.h"

#include "json.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace gewell::text {
namespace {

[[noreturn]] void malformed(const std::string& message) {
  throw std::invalid_argument("invalid generated tool call: " + message);
}

bool space(char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }
bool name_char(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || c == '_' || c == '-';
}

// Escape literal Gemma string bytes without interpreting backslash escapes.
// UTF-8 sequences arrive intact from the pinned incremental text decoder.
void escape_byte(std::string& output, unsigned char c) {
  if (c == '"' || c == '\\') { output.push_back('\\'); output.push_back(c); }
  else if (c < 0x20) {
    static constexpr char hex[] = "0123456789abcdef";
    output += "\\u00";
    output.push_back(hex[c >> 4]); output.push_back(hex[c & 15]);
  } else output.push_back(c);
}

}  // namespace

struct ToolOutputDecoder::Impl {
  struct Frame { char kind; bool key; };
  enum class String { none, gemma, json, bare_key };
  const TextContract& contract;
  ChatOutputDecoder ordinary;
  IncrementalTextDecoder arguments_decoder;
  std::vector<std::string> names;
  std::string id_prefix, prefix;
  std::vector<ToolCallOutput> calls;
  std::vector<Frame> stack;
  bool allow_calls, in_call = false, have_name = false, handoff = false,
       reasoning = false, json_escape = false, key_space = false, finished = false;
  String string = String::none;

  Impl(const Tokenizer& tokenizer, std::vector<std::string> allowed,
       std::string ids, bool enabled, bool initial_reasoning)
      : contract(tokenizer.contract()), ordinary(tokenizer), arguments_decoder(tokenizer), names(std::move(allowed)),
        id_prefix(std::move(ids)), allow_calls(enabled), reasoning(initial_reasoning) {
    if (initial_reasoning) {
      (void)ordinary.push(contract.channel_start);
      for (auto token : tokenizer.encode(contract.thought_prefix)) (void)ordinary.push(token);
    }
  }

  ToolDelta ordinary_delta(ChatDelta value) {
    return {std::move(value.reasoning), std::move(value.content), {}};
  }

  void append(ToolDelta& delta, std::string bytes, bool first = false) {
    if (bytes.empty() && !first) return;
    auto& call = calls.back();
    call.arguments += bytes;
    if (!delta.tool_calls.empty() && delta.tool_calls.back().index == calls.size() - 1) {
      delta.tool_calls.back().arguments += bytes;
      return;
    }
    delta.tool_calls.push_back({calls.size() - 1, first ? call.id : "",
                               first ? call.name : "", std::move(bytes)});
  }

  void argument_text(std::string_view text, ToolDelta& delta) {
    std::string output;
    for (const unsigned char c : text) {
      if (string == String::gemma) { escape_byte(output, c); continue; }
      if (string == String::json) {
        output.push_back(c);
        if (json_escape) json_escape = false;
        else if (c == '\\') json_escape = true;
        else if (c == '"') string = String::none;
        continue;
      }
      if (string == String::bare_key) {
        if (c == ':') { output += "\":"; string = String::none; }
        else {
          if (space(c)) { key_space = true; continue; }
          if (key_space || !name_char(c))
            malformed("object keys must use letters, digits, underscore or hyphen");
          escape_byte(output, c);
        }
        continue;
      }
      if (stack.empty()) {
        if (!space(c)) malformed("trailing bytes after arguments");
        output.push_back(c);
        continue;
      }
      auto& frame = stack.back();
      if (frame.kind == '{' && frame.key && !space(c) && c != '}') {
        frame.key = false;
        if (c == '"') { string = String::json; output.push_back(c); }
        else {
          if (!name_char(c)) malformed("invalid object key");
          string = String::bare_key;
          key_space = false;
          output.push_back('"'); escape_byte(output, c);
        }
        continue;
      }
      if (c == '"') string = String::json;
      else if (c == '{' || c == '[') {
        if (stack.size() >= 32) malformed("argument nesting exceeds 32 levels");
        stack.push_back({static_cast<char>(c), c == '{'});
      }
      else if (c == '}' || c == ']') {
        if ((c == '}') != (frame.kind == '{')) malformed("mismatched argument brackets");
        stack.pop_back();
      } else if (c == ',' && frame.kind == '{') frame.key = true;
      output.push_back(c);
    }
    append(delta, std::move(output));
  }

  void decoded_text(std::string_view text, ToolDelta& delta) {
    if (have_name) { argument_text(text, delta); return; }
    for (std::size_t i = 0; i < text.size(); ++i) {
      const char c = text[i];
      if (prefix.empty() && space(c)) continue;
      if (c != '{') {
        prefix.push_back(c);
        const std::string_view marker = "call:";
        if (prefix.size() <= marker.size()) {
          if (marker.substr(0, prefix.size()) != prefix) malformed("expected call: prefix");
        } else if (!name_char(c) || prefix.size() > marker.size() + 64) {
          malformed("invalid function name");
        }
        continue;
      }
      if (prefix.size() <= 5 || prefix.compare(0, 5, "call:") != 0)
        malformed("missing function name");
      const auto name = prefix.substr(5);
      if (std::find(names.begin(), names.end(), name) == names.end())
        malformed("unknown function: " + name);
      if (calls.size() == 128) malformed("more than 128 tool calls");
      calls.push_back({id_prefix + std::to_string(calls.size()), name, {}, false});
      have_name = true;
      prefix.clear();
      stack.push_back({'{', true});
      append(delta, "{", true);
      argument_text(text.substr(i + 1), delta);
      return;
    }
  }

  void marker(ToolDelta& delta) {
    decoded_text(arguments_decoder.finish(), delta);
    if (!have_name) malformed("string delimiter before function arguments");
    if (string == String::gemma) { string = String::none; append(delta, "\""); return; }
    if (string != String::none || stack.empty()) malformed("unexpected string delimiter");
    if (stack.back().kind == '{' && stack.back().key) stack.back().key = false;
    string = String::gemma;
    append(delta, "\"");
  }

  void close_call(ToolDelta& delta) {
    decoded_text(arguments_decoder.finish(), delta);
    if (!have_name || string != String::none || !stack.empty())
      malformed("incomplete arguments at call end");
    try {
      const auto value = nlohmann::json::parse(calls.back().arguments);
      if (!value.is_object()) malformed("arguments must be an object");
    } catch (const nlohmann::json::exception&) { malformed("arguments are not valid JSON"); }
    calls.back().complete = true;
    in_call = have_name = false;
  }
};

ToolOutputDecoder::ToolOutputDecoder(const Tokenizer& tokenizer,
    std::vector<std::string> names, std::string id_prefix, bool allow_calls, bool initial_reasoning)
    : impl_(std::make_unique<Impl>(tokenizer, std::move(names), std::move(id_prefix), allow_calls, initial_reasoning)) {}
ToolOutputDecoder::~ToolOutputDecoder() = default;

ToolDelta ToolOutputDecoder::push(std::uint32_t token) {
  auto& s = *impl_;
  if (s.finished || s.handoff) malformed("output after handoff or finalization");
  if (token == s.contract.tool_call_start) {
    if (!s.allow_calls) malformed("tool calls are disabled by tool_choice");
    if (s.in_call || s.reasoning) malformed("call starts inside an unfinished call or reasoning channel");
    auto delta = s.ordinary_delta(s.ordinary.finish());
    s.in_call = true;
    return delta;
  }
  if (token == s.contract.tool_call_end) {
    if (!s.in_call) malformed("call end without call start");
    ToolDelta delta;
    s.close_call(delta);
    return delta;
  }
  if (token == s.contract.tool_handoff) {
    if (s.in_call || s.calls.empty()) malformed("handoff requires complete function calls");
    s.handoff = true;
    return s.ordinary_delta(s.ordinary.finish());
  }
  if (s.in_call) {
    if (token == s.contract.channel_start || token == s.contract.channel_end || s.contract.is_constraint_stop(token))
      malformed("control token inside function arguments");
    ToolDelta delta;
    if (token == s.contract.string_delimiter) s.marker(delta);
    else s.decoded_text(s.arguments_decoder.push(token), delta);
    return delta;
  }
  if (token == s.contract.string_delimiter) malformed("argument delimiter outside a function call");
  if (token == s.contract.channel_start || token == s.contract.channel_end) s.reasoning = token == s.contract.channel_start;
  return s.ordinary_delta(s.ordinary.push(token));
}

ToolDelta ToolOutputDecoder::finish(bool eos) {
  auto& s = *impl_;
  if (s.finished) malformed("output already finalized");
  s.finished = true;
  if (eos && !s.handoff && (s.in_call || !s.calls.empty()))
    malformed("model ended a tool turn without handoff");
  if (s.in_call) {
    ToolDelta delta;
    s.decoded_text(s.arguments_decoder.finish(), delta);
    return delta;
  }
  return s.ordinary_delta(s.ordinary.finish());
}

ChatDelta ToolOutputDecoder::pending() const {
  return impl_->in_call ? ChatDelta{} : impl_->ordinary.pending();
}

bool ToolOutputDecoder::in_content() const {
  return !impl_->in_call && !impl_->reasoning && !impl_->handoff;
}

bool ToolOutputDecoder::tool_handoff() const { return impl_->handoff; }
const std::vector<ToolCallOutput>& ToolOutputDecoder::calls() const { return impl_->calls; }
std::size_t ToolOutputDecoder::retained_bytes() const {
  std::size_t bytes = impl_->prefix.size() + impl_->stack.size() * sizeof(Impl::Frame);
  for (const auto& call : impl_->calls) bytes += call.id.size() + call.name.size() + call.arguments.size();
  return bytes;
}

}  // namespace gewell::text
