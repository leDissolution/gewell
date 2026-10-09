#include "gewell/models/embeddinggemma2/text.h"
#include "gewell/models/embeddinggemma2/model.h"
#include "gewell/vision_engine.h"
#include <algorithm>
#include <stdexcept>

namespace gewell::embeddinggemma2 {
namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(std::string("embeddinggemma2 tokenizer: ") + message);
}
void validate_pipeline(const nlohmann::json& data) {
  using Json=nlohmann::json;
  require(data.at("version")=="1.0" && data.at("truncation").is_null() && data.at("padding").is_null(),"unsupported settings");
  require(data.at("normalizer")==Json::parse(R"({"type":"Replace","pattern":{"String":" "},"content":"▁"})"),"unsupported normalizer");
  require(data.at("pre_tokenizer")==Json::parse(R"({"type":"Split","pattern":{"String":" "},"behavior":"MergedWithPrevious","invert":false})"),"unsupported pre-tokenizer");
  require(data.at("decoder")==Json::parse(R"({"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"}]})"),"unsupported decoder");
  require(data.at("post_processor")==Json::parse(R"({"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"<bos>","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"<eos>","type_id":0}}],"pair":[{"SpecialToken":{"id":"<bos>","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"<eos>","type_id":0}},{"SpecialToken":{"id":"<bos>","type_id":1}},{"Sequence":{"id":"B","type_id":1}},{"SpecialToken":{"id":"<eos>","type_id":1}}],"special_tokens":{"<bos>":{"id":"<bos>","ids":[2],"tokens":["<bos>"]},"<eos>":{"id":"<eos>","ids":[1],"tokens":["<eos>"]}}})"),"unsupported post-processor");
  const auto& m=data.at("model");
  require(m.at("type")=="BPE" && m.at("dropout").is_null() && m.at("unk_token")=="<unk>" &&
      m.at("fuse_unk")==true && m.at("byte_fallback")==true && m.at("ignore_merges")==false &&
      m.at("continuing_subword_prefix").is_null() && m.at("end_of_word_suffix").is_null(),"unsupported BPE settings");
}
}
const text::TextContract& text_contract() {
  static const auto contract=[] {
    text::TextContract c{};
    c.vocabulary_size=kVocabulary; c.context_tokens=kMaxTokens; c.merge_count=514906;
    c.bos=2; c.stop_tokens={1}; c.space_marker="▁";
    c.special_tokens={0,1,2,3,4,46,47,48,49,50,51,52,98,100,101,105,106,
                      255999,256000,258880,258881,258882,258883,258884};
    c.control_tokens={{0,"<pad>"},{1,"<eos>"},{2,"<bos>"},{3,"<unk>"}};
    c.validate_tokenizer_pipeline=validate_pipeline;
    return c;
  }();
  return contract;
}
std::vector<std::uint32_t> tokenize_input(const text::Tokenizer& tokenizer,std::string_view input) {
  require(!input.empty(),"input must be nonempty");
  auto tokens=tokenizer.encode(input);
  if (tokens.size()>kMaxTokens-2) throw ContextLengthError("embeddinggemma2 input exceeds 8192 tokens including BOS/EOS");
  tokens.insert(tokens.begin(),2);
  tokens.push_back(1);
  return tokens;
}
void validate_input(const PreparedInput& input, int max_tokens) {
  const auto& tokens=input.tokens;
  require(!tokens.empty() && tokens.size()<=static_cast<std::size_t>(max_tokens),"expanded context exceeds capacity or is empty");
  for (auto token:tokens) require(token<kVocabulary,"invalid token ID");
  std::vector<bool> filled(tokens.size(),false);
  std::size_t cursor=0;
  for (const auto& image:input.visuals) {
    require(image && image->begin>=1 && image->begin>=cursor && image->end>image->begin && image->end<tokens.size(),"invalid image span");
    require(tokens[image->begin-1]==255999 && tokens[image->end]==258882,"image span boundaries differ");
    const auto placeholder=tokens[image->begin];
    require(placeholder==258880 || placeholder==258884,"visual span requires image or video placeholders");
    require(std::all_of(tokens.begin()+image->begin,tokens.begin()+image->end,[&](auto t){return t==placeholder;}),"visual span must contain consistent placeholders");
    require(image->positions.size()==std::size_t(image->padded_patch_rows)*8,"image padded row count differs");
    vision_engine::validate_prepared_image_bytes(image->pixels.data(),image->pixels.size(),
        image->positions.data(),image->positions.size(),image->end-image->begin);
    std::fill(filled.begin()+image->begin,filled.begin()+image->end,true);
    cursor=image->end;
  }
  cursor=0;
  for (const auto& audio:input.audios) {
    require(audio && audio->begin>=1 && audio->begin>=cursor && audio->end>audio->begin && audio->end<tokens.size(),"invalid audio span");
    require(tokens[audio->begin-1]==256000 && tokens[audio->end]==258883,"audio span boundaries differ");
    require(std::all_of(tokens.begin()+audio->begin,tokens.begin()+audio->end,[](auto t){return t==258881;}),"audio span requires audio placeholders");
    audio::validate_features(*audio);
    std::fill(filled.begin()+audio->begin,filled.begin()+audio->end,true);
    cursor=audio->end;
  }
  for (std::size_t i=0;i<tokens.size();++i)
    require(filled[i]==(tokens[i]==258880 || tokens[i]==258881 || tokens[i]==258884),"unfilled modality placeholder");
}
PreparedInput prepare_input(const text::Tokenizer& tokenizer,const std::vector<InputPart>& parts) {
  PreparedInput result;
  std::string rendered, run;
  struct Span {std::uint32_t count,token;std::uint32_t *begin,*end;};
  std::vector<Span> spans;
  auto flush=[&] {
    for (auto token:tokenizer.encode(run))
      require(token!=258880 && token!=258881 && token!=258884,"modality placeholders require media content parts");
    rendered+=run; run.clear();
  };
  auto visual=[&](const std::shared_ptr<runtime::ImageInput>& image,std::uint32_t token) {
    require(image && image->end>image->begin,"visual feature count must be positive");
    const auto count=image->end-image->begin;
    require(count<=1120,"image feature count exceeds supported capacity");
    rendered+=tokenizer.token_piece(255999);
    for (std::uint32_t i=0;i<count;++i) rendered+=tokenizer.token_piece(token);
    rendered+=tokenizer.token_piece(258882);
    spans.push_back({count,token,&image->begin,&image->end});result.visuals.push_back(image);
  };
  for (const auto& part:parts) {
    const int media=bool(part.image)+!part.video.empty()+bool(part.audio);
    if (!media) { run+=part.text; continue; }
    require(part.text.empty() && media==1,"a content part must contain a single media type");
    flush();
    if (part.image) visual(part.image,258880);
    else if (part.audio) {
      audio::validate_features(*part.audio);
      const auto count=part.audio->end-part.audio->begin;
      rendered+=tokenizer.token_piece(256000);
      for (std::uint32_t i=0;i<count;++i) rendered+=tokenizer.token_piece(258881);
      rendered+=tokenizer.token_piece(258883);
      spans.push_back({count,258881,&part.audio->begin,&part.audio->end});result.audios.push_back(part.audio);
    }
    else {
      require(part.video.size()<=32,"video frame count exceeds supported capacity");
      ++result.video_count;
      for (const auto& frame:part.video) visual(frame,258884);
    }
  }
  flush();
  result.tokens=tokenize_input(tokenizer,rendered);
  std::size_t cursor=0;
  for (const auto& span:spans) {
    while (cursor<result.tokens.size() && result.tokens[cursor]!=span.token) ++cursor;
    *span.begin=cursor;*span.end=cursor+span.count;
    cursor=*span.end;
  }
  validate_input(result);
  return result;
}
PreparedInput prepare_input(const text::Tokenizer& tokenizer,std::string_view input) {
  return prepare_input(tokenizer,std::vector<InputPart>{{std::string(input),{}}});
}
}  // namespace gewell::embeddinggemma2
