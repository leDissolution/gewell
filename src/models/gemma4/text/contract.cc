#include "gewell/models/gemma4/text_contract.h"
#include "chat_template.h"

#include <stdexcept>

namespace gewell::gemma4 {
namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error("tokenizer: " + message);
}

void validate_tokenizer_pipeline(const nlohmann::json& data) {
  using nlohmann::json;
    require(data.at("version") == "1.0" && data.at("truncation").is_null() &&
                data.at("padding").is_null(), "unsupported pipeline settings");
    require(data.at("normalizer") == json::parse(
        R"({"type":"Replace","pattern":{"String":" "},"content":"▁"})"),
        "unsupported normalization");
    // Normalization removes every literal space before this split runs.
    require(data.at("pre_tokenizer") == json::parse(
        R"({"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious","invert":false})"),
        "unsupported pre-tokenizer");
    require(data.at("post_processor") == json::parse(
        R"({"type":"TemplateProcessing","single":[{"Sequence":{"id":"A","type_id":0}}],"pair":[{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{}})"),
        "unsupported post-processor");
    require(data.at("decoder") == json::parse(
        R"({"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]})"),
        "unsupported decoder");
    const auto& model = data.at("model");
    require(model.at("type") == "BPE" && model.at("dropout").is_null() &&
                model.at("unk_token") == "<unk>" && model.at("fuse_unk") == true &&
                model.at("byte_fallback") == true && model.at("ignore_merges") == false &&
                model.at("continuing_subword_prefix").is_null() &&
                model.at("end_of_word_suffix").is_null(), "unsupported BPE settings");
}

const text::TextContract& shared_text_contract() {
  static const text::TextContract contract = [] {
    text::TextContract value{};
    value.vocabulary_size = 262144;
    value.context_tokens = 262144;
    value.merge_count = 514906;
    value.bos = 2;
    value.stop_tokens = {1, 106, 50};
    value.constraint_stop_tokens = {1, 106};
    value.channel_start = 100;
    value.channel_end = 101;
    value.tool_call_start = 48;
    value.tool_call_end = 49;
    value.tool_handoff = 50;
    value.string_delimiter = 52;
    value.thought_prefix = "thought\n";
    value.reasoning_continuation = "<|channel>thought\n";
    value.space_marker = "\xE2\x96\x81";
    value.special_tokens = {0,1,2,3,4,46,47,48,49,50,51,52,98,100,101,105,106,
                           255999,256000,258880,258881,258882,258883,258884};
    value.control_tokens = {{0, "<pad>"}, {1, "<eos>"}, {2, "<bos>"},
                            {50, "<|tool_response>"}, {100, "<|channel>"},
                            {101, "<channel|>"}, {106, "<turn|>"}};
    value.validate_tokenizer_pipeline = validate_tokenizer_pipeline;
    value.normalize_messages = text_detail::normalize_chat_messages;
    value.normalize_tools = text_detail::normalize_chat_tools;
    value.template_options = text_detail::parse_chat_template_kwargs;
    value.render = text_detail::render_chat;
    return value;
  }();
  return contract;
}

}  // namespace

const text::TextContract& text_contract_31b() { return shared_text_contract(); }
const text::TextContract& text_contract_26b_a4b() { return shared_text_contract(); }

}  // namespace gewell::gemma4
