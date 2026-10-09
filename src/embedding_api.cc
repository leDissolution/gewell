#include "gewell/http_api.h"
#include "gewell/models/embeddinggemma2/text.h"
#include "gewell/vision_engine.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <map>

namespace gewell::http {
namespace {
using nlohmann::json;
struct PreparedImageSlot {
  std::shared_ptr<runtime::ImageInput> image;
  std::exception_ptr error;
  bool done = false;
};
using ImageSlots = std::map<std::pair<std::size_t, std::size_t>, PreparedImageSlot>;
// Decodes every structurally plausible image part concurrently; validation and
// error ordering stay with the sequential assembly pass.
ImageSlots prepare_images(const json& value, bool single, std::size_t count, const ImageSupport& images,
                          std::uint32_t budget) {
  ImageSlots slots;
  std::vector<std::pair<const std::string*, PreparedImageSlot*>> jobs;
  if (!images.prepare) return slots;
  for (std::size_t i = 0; i < count; ++i) {
    const auto& item = single ? value : value[i];
    if (!item.is_object() || !item.contains("content") || !item["content"].is_array()) continue;
    for (std::size_t j = 0; j < item["content"].size(); ++j) {
      const auto& part = item["content"][j];
      if (!part.is_object() || part.value("type", json()) != "image_url" || !part.contains("image_url")) continue;
      const auto& media = part["image_url"];
      if (!media.is_object() || !media.contains("url") || !media["url"].is_string()) continue;
      jobs.emplace_back(&media["url"].get_ref<const std::string&>(), &slots[{i, j}]);
    }
  }
  if (jobs.size() < 2) return {};
  run_image_tasks(jobs.size(), [&](std::size_t k) {
    auto& slot = *jobs[k].second;
    try { slot.image = images.prepare(*jobs[k].first, budget); }
    catch (...) { slot.error = std::current_exception(); }
    slot.done = true;
    return !slot.error;
  });
  return slots;
}
[[noreturn]] void invalid(const std::string& param, const std::string& message) {
  throw Error(400, message, param, "invalid_parameter");
}
void allow_keys(const json& object,std::initializer_list<std::string_view> allowed,const std::string& param) {
  if (!object.is_object()) invalid(param,"must be an object");
  for (const auto& item:object.items())
    if (std::find(allowed.begin(),allowed.end(),item.key())==allowed.end())
      throw Error(400,"unsupported field: "+item.key(),param.empty()?item.key():param+"."+item.key(),"unsupported_parameter");
}
std::string base64_floats(const std::vector<float>& values) {
  static constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::vector<unsigned char> bytes;
  bytes.reserve(values.size() * 4);
  for (float value : values) {
    std::uint32_t bits;
    static_assert(sizeof(value) == sizeof(bits));
    std::memcpy(&bits, &value, 4);
    for (int shift = 0; shift < 32; shift += 8) bytes.push_back((bits >> shift) & 255);
  }
  std::string result;
  result.reserve((bytes.size() + 2) / 3 * 4);
  for (std::size_t i = 0; i < bytes.size(); i += 3) {
    const auto word = (std::uint32_t(bytes[i]) << 16) |
        (i + 1 < bytes.size() ? std::uint32_t(bytes[i + 1]) << 8 : 0) |
        (i + 2 < bytes.size() ? bytes[i + 2] : 0);
    result += alphabet[word >> 18]; result += alphabet[(word >> 12) & 63];
    result += i + 1 < bytes.size() ? alphabet[(word >> 6) & 63] : '=';
    result += i + 2 < bytes.size() ? alphabet[word & 63] : '=';
  }
  return result;
}
}

Request parse_embedding_request(std::string_view method, std::string_view path,
    std::string_view encoded, const text::Tokenizer& tokenizer, const std::string& model,
    constraint::Compiler& compiler, const EmbeddingLimits& limits, std::size_t max_output_bytes,
    const ImageSupport& images, const embeddinggemma2::audio::Acquire& acquire_audio) {
  if (path == "/health" || path == "/metrics" || path == "/v1/models" || path.substr(0, 11) == "/v1/models/")
    return parse_request(method, path, encoded, tokenizer, model, compiler);
  if (path != "/v1/embeddings") throw Error(404, "route is not implemented", {}, "not_found");
  if (method != "POST") throw Error(405, "method is not allowed for this route", {}, "method_not_allowed");
  json body;
  try { body = json::parse(encoded); }
  catch (const json::exception&) { throw Error(400, "invalid JSON", {}, "invalid_json"); }
  if (!body.is_object()) invalid("body", "request body must be a JSON object");
  allow_keys(body,{"model","input","dimensions","encoding_format","mm_processor_kwargs"},"");
  if (!body.contains("model") || !body["model"].is_string()) invalid("model", "model must be a string");
  if (body["model"] != model) throw Error(404, "model is not served", "model", "model_not_found");
  std::uint32_t budget=280;
  if (body.contains("mm_processor_kwargs")) {
    const auto& options=body["mm_processor_kwargs"];
    allow_keys(options,{"max_soft_tokens"},"mm_processor_kwargs");
    if (options.contains("max_soft_tokens")) {
      const auto& value=options["max_soft_tokens"];
      if (!value.is_number_integer() || value<1 || value>1120 ||
          !vision_engine::is_supported_soft_token_capacity(value.get<std::uint32_t>()))
        invalid("mm_processor_kwargs.max_soft_tokens","max_soft_tokens must be 70, 140, 280, 560 or 1120");
      budget=value.get<std::uint32_t>();
    }
  }
  auto input = std::make_shared<EmbeddingInput>();
  if (body.contains("dimensions")) {
    const auto& d = body["dimensions"];
    if (!d.is_number_integer() || (d != 128 && d != 256 && d != 512 && d != 768))
      invalid("dimensions", "dimensions must be 128, 256, 512 or 768");
    input->dimensions = d.get<int>();
  }
  if (body.contains("encoding_format")) {
    if (body["encoding_format"] != "float" && body["encoding_format"] != "base64")
      invalid("encoding_format", "encoding_format must be float or base64");
    input->base64 = body["encoding_format"] == "base64";
  }
  if (!body.contains("input")) invalid("input", "input is required");
  const auto& value = body["input"];
  const bool single = value.is_string() || value.is_object();
  if (!single && !value.is_array()) invalid("input", "input must be a string, content object, or array of either");
  const auto count = single ? 1 : value.size();
  if (!count || count > limits.max_inputs) invalid("input", "input count exceeds the configured limit or is zero");
  // Reserve a conservative JSON float bound (including separators), or exact
  // base64 payload size, plus objects, escaped model ID and HTTP headers.
  const std::size_t per_vector = 128 + (input->base64 ? (input->dimensions * 4 + 2) / 3 * 4 : input->dimensions * 32);
  const auto overhead = 1024 + json(model).dump().size();
  if (overhead > max_output_bytes || count > (max_output_bytes - overhead) / per_vector)
    throw Error(400, "embedding response exceeds the configured output budget", "input", "output_limit_exceeded");
  input->inputs.reserve(count);
  auto prefetched = prepare_images(value, single, count, images, budget);
  for (std::size_t i = 0; i < count; ++i) {
    const auto& item = single ? value : value[i];
    const auto param = single ? "input" : "input[" + std::to_string(i) + "]";
    std::vector<embeddinggemma2::InputPart> parts;
    if (item.is_string()) {
      if (item.get_ref<const std::string&>().empty()) invalid(param,"input must be a nonempty string");
      parts.push_back({item.get<std::string>(),{}});
    } else {
      allow_keys(item,{"content"},param);
      if (!item.contains("content") || !item["content"].is_array() || item["content"].empty())
        invalid(param+".content","content must be a nonempty array");
      for (std::size_t j=0;j<item["content"].size();++j) {
        const auto& part=item["content"][j];
        const auto location=param+".content["+std::to_string(j)+"]";
        if (!part.is_object() || !part.contains("type") || !part["type"].is_string())
          invalid(location+".type","content part requires a string type");
        if (part["type"]=="text") {
          allow_keys(part,{"type","text"},location);
          if (!part.contains("text") || !part["text"].is_string()) invalid(location+".text","text must be a string");
          parts.push_back({part["text"].get<std::string>(),{}});
        } else if (part["type"]=="image_url" || part["type"]=="video_url") {
          const bool video=part["type"]=="video_url";
          const std::string kind=video?"video_url":"image_url";
          allow_keys(part,{"type",kind},location);
          if (!part.contains(kind)) invalid(location+"."+kind,kind+" is required");
          const auto& media=part[kind];
          allow_keys(media,{"url"},location+"."+kind);
          const auto url_param=location+"."+kind+".url";
          if (!media.contains("url") || !media["url"].is_string()) invalid(url_param,"url must be a string");
          if (video?!bool(images.prepare_video_frame):!bool(images.prepare))
            throw Error(400,video?"video input requires --vision":"image input requires --vision",location,"unsupported_parameter");
          try {
            const auto& url=media["url"].get_ref<const std::string&>();
            if (video) {
              auto prepared=embeddinggemma2::prepare_video_data_url(url,images.prepare_video_frame,images.poll_preparation);
              parts.push_back({{},{},std::move(prepared.frames)});
            } else {
              const auto slot=prefetched.find({i,j});
              if (slot!=prefetched.end() && slot->second.error) std::rethrow_exception(slot->second.error);
              auto prepared=slot!=prefetched.end() && slot->second.done ? std::move(slot->second.image) : images.prepare(url,budget);
              if (!prepared || prepared->begin || !prepared->end || prepared->end>budget)
                throw Error(500,"image processor returned an invalid feature count",{},"execution_failed");
              parts.push_back({{},std::move(prepared)});
            }
          } catch (const std::invalid_argument& error) {
            throw Error(400,error.what(),url_param,video?"invalid_video":"invalid_image");
          }
        } else if (part["type"]=="audio_url") {
          allow_keys(part,{"type","audio_url"},location);
          if (!part.contains("audio_url")) invalid(location+".audio_url","audio_url is required");
          const auto& media=part["audio_url"];
          allow_keys(media,{"url"},location+".audio_url");
          const auto url_param=location+".audio_url.url";
          if (!media.contains("url") || !media["url"].is_string()) invalid(url_param,"url must be a string");
          if (!limits.audio) throw Error(400,"audio input requires --audio",location,"unsupported_parameter");
          try {
            auto prepared=embeddinggemma2::audio::prepare_data_url(media["url"].get_ref<const std::string&>(),
                limits.max_batch_tokens,acquire_audio,images.poll_preparation);
            parts.push_back({{},{},{},std::move(prepared.input)});
          } catch (const embeddinggemma2::ContextLengthError& error) {
            throw Error(400,error.what(),param,"context_length_exceeded");
          } catch (const std::invalid_argument& error) {
            throw Error(400,error.what(),url_param,"invalid_audio");
          }
        } else throw Error(400,"unsupported content type",location+".type","unsupported_parameter");
      }
    }
    embeddinggemma2::PreparedInput prepared;
    try {prepared=embeddinggemma2::prepare_input(tokenizer,parts);}
    catch (const embeddinggemma2::ContextLengthError& error) {throw Error(400,error.what(),param,"context_length_exceeded");}
    catch (const std::invalid_argument& error) { invalid(param, error.what()); }
    if (prepared.tokens.size() > limits.max_batch_tokens)
      throw Error(400, "input exceeds the configured physical batch token limit", param, "context_length_exceeded");
    input->inputs.push_back(std::move(prepared));
  }
  Request request;
  request.operation = Operation::embed;
  request.embeddings = std::move(input);
  return request;
}

json embedding_json(const Request& request, const EmbeddingResult& result, const std::string& model) {
  if (!request.embeddings || result.vectors.size() != request.embeddings->inputs.size())
    throw Error(500, "embedding result count mismatch", {}, "invalid_execution_result");
  const auto& input = *request.embeddings;
  json data = json::array();
  std::uint64_t tokens = 0;
  for (std::size_t i = 0; i < result.vectors.size(); ++i) {
    const auto& vector = result.vectors[i];
    if (vector.size() != static_cast<std::size_t>(input.dimensions))
      throw Error(500, "embedding dimension mismatch", {}, "invalid_execution_result");
    for (float v : vector) if (!std::isfinite(v)) throw Error(500, "nonfinite embedding", {}, "invalid_execution_result");
    data.push_back({{"object", "embedding"}, {"index", i},
                   {"embedding", input.base64 ? json(base64_floats(vector)) : json(vector)}});
    tokens += input.inputs[i].tokens.size();
  }
  return {{"object", "list"}, {"model", model}, {"data", std::move(data)},
          {"usage", {{"prompt_tokens", tokens}, {"total_tokens", tokens}}}};
}
}  // namespace gewell::http
