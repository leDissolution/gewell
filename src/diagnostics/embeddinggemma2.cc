#include "gewell/models/embeddinggemma2/encoder.h"
#include "gewell/models/embeddinggemma2/model.h"
#include "gewell/models/embeddinggemma2/text.h"
#include "gewell/models/embeddinggemma2/video.h"
#include "gewell/models/gemma4/image_processor.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

int main(int argc,char** argv) {
  using Json=nlohmann::json;
  namespace eg=gewell::embeddinggemma2;
  try {
    bool vision=false,audio=false;
    if (argc<2) throw std::invalid_argument("usage: gewell_embeddinggemma2 BUNDLE [--vision] [--audio] < requests.jsonl");
    for (int i=2;i<argc;++i) {
      if (std::string(argv[i])=="--vision" && !vision) vision=true;
      else if (std::string(argv[i])=="--audio" && !audio) audio=true;
      else throw std::invalid_argument("unsupported or duplicate option");
    }
    const std::string directory=argv[1];
    eg::validate_bundle_config(directory);
    gewell::text::Tokenizer tokenizer(directory+"/tokenizer.json",eg::text_contract());
    std::unique_ptr<eg::Encoder> encoder;
    std::string line;
    bool failed=false;
    while (std::getline(std::cin,line)) {
      try {
        const auto request=Json::parse(line);
        eg::PreparedInput input;
        Json videos=Json::array(),audios=Json::array();
        auto& tokens=input.tokens;
        if (request.contains("token_ids")) {
          if (!request["token_ids"].is_array()) throw std::invalid_argument("token_ids must be an array");
          for (const auto& value:request["token_ids"]) {
            if (!value.is_number_integer() || value<0 || value>=eg::kVocabulary)
              throw std::invalid_argument("invalid token ID");
            tokens.push_back(value.get<std::uint32_t>());
          }
          if (tokens.empty() || tokens.size()>eg::kMaxTokens) throw std::invalid_argument("invalid token count");
        } else if (request.contains("content")) {
          if (!request["content"].is_array()) throw std::invalid_argument("content must be an array");
          std::vector<eg::InputPart> parts;
          if (request.contains("max_soft_tokens")) {
            const auto& value=request["max_soft_tokens"];
            if (!value.is_number_integer() || (value!=70 && value!=140 && value!=280 && value!=560 && value!=1120))
              throw std::invalid_argument("unsupported max_soft_tokens");
          }
          const auto budget=request.value("max_soft_tokens",280);
          for (const auto& part:request["content"]) {
            if (part.at("type")=="text") parts.push_back({part.at("text").get<std::string>(),{}});
            else if (part.at("type")=="image_url") {
              if (!vision) throw std::invalid_argument("vision is disabled");
              auto prepared=gewell::gemma4::prepare_image_data_url(part.at("image_url").at("url").get<std::string>(),budget);
              auto image=std::make_shared<gewell::runtime::ImageInput>();
              image->pixels=std::move(prepared.pixels); image->positions=std::move(prepared.positions);
              image->padded_patch_rows=prepared.padded_patch_rows; image->end=prepared.soft_token_count;
              parts.push_back({{},std::move(image)});
            } else if (part.at("type")=="video_url") {
              if (!vision) throw std::invalid_argument("vision is disabled");
              auto video=eg::prepare_video_data_url(part.at("video_url").at("url").get<std::string>());
              const auto& metadata=video.metadata;
              videos.push_back({{"total_num_frames",metadata.total_frames},{"fps",metadata.fps>0?Json(metadata.fps):Json(nullptr)},
                  {"duration",metadata.fps>0?Json(metadata.duration):Json(nullptr)},
                  {"selected_indices",metadata.indices},{"timestamps",metadata.timestamps}});
              parts.push_back({{},{},std::move(video.frames)});
            } else if (part.at("type")=="audio_url") {
              if (!audio) throw std::invalid_argument("audio is disabled");
              auto prepared=eg::audio::prepare_data_url(part.at("audio_url").at("url").get<std::string>());
              const auto& metadata=prepared.metadata;
              audios.push_back({{"input_rate",metadata.input_rate},{"channels",metadata.channels},{"samples",metadata.samples},
                  {"layout",metadata.layout},{"codec",metadata.codec},{"feature_frames",prepared.input->mask.size()},
                  {"soft_tokens",prepared.input->end-prepared.input->begin}});
              parts.push_back({{},{},{},std::move(prepared.input)});
            } else throw std::invalid_argument("unsupported content type");
          }
          input=eg::prepare_input(tokenizer,parts);
        } else {
          const auto value=request.at("text").get<std::string>();
          if (request.value("tokenize_only",false)) tokens=eg::tokenize_input(tokenizer,value);
          else input=eg::prepare_input(tokenizer,value);
        }
        Json response={{"token_ids",tokens}};
        response["videos"]=std::move(videos);
        response["spans"]=Json::array();
        for (const auto& image:input.visuals) response["spans"].push_back({image->begin,image->end});
        if (audio) {
          response["audios"]=std::move(audios);
          response["audio_spans"]=Json::array();response["media_spans"]=Json::array();
          for (std::size_t i=0;i<input.visuals.size();++i) {
            const auto& image=*input.visuals[i];
            response["media_spans"].push_back({{"kind",tokens[image.begin]==258880?"image":"video"},{"index",i},
                {"begin",image.begin},{"end",image.end}});
          }
          for (std::size_t i=0;i<input.audios.size();++i) {
            const auto& clip=*input.audios[i];
            response["audio_spans"].push_back({clip.begin,clip.end});response["spans"].push_back({clip.begin,clip.end});
            response["media_spans"].push_back({{"kind","audio"},{"index",i},{"begin",clip.begin},{"end",clip.end}});
          }
          std::sort(response["spans"].begin(),response["spans"].end());
          std::sort(response["media_spans"].begin(),response["media_spans"].end(),[](const auto& a,const auto& b){return a["begin"]<b["begin"];});
        }
        if (!request.value("tokenize_only",false)) {
          std::vector<int> dimensions={768,512,256,128};
          if (request.contains("dimensions")) {
            dimensions.clear();
            if (!request["dimensions"].is_array()) throw std::invalid_argument("dimensions must be an array");
            for (const auto& value:request["dimensions"]) {
              if (!value.is_number_integer() || value<128 || value>768 || !eg::supported_dimension(value.get<int>()))
                throw std::invalid_argument("unsupported dimension");
              dimensions.push_back(value.get<int>());
            }
          }
          if (dimensions.empty()) throw std::invalid_argument("dimensions must not be empty");
          eg::Capture capture;
          std::filesystem::path capture_dir;
          Json manifest=Json::object();
          if (request.contains("capture_dir")) {
            capture_dir=request.at("capture_dir").get<std::string>();
            if (!std::filesystem::create_directory(capture_dir)) throw std::invalid_argument("capture directory already exists");
            capture=[&](const std::string& name,const std::vector<int>& shape,const std::vector<float>& data) {
              if (request.contains("capture_names") &&
                  std::find(request["capture_names"].begin(),request["capture_names"].end(),name)==request["capture_names"].end()) return;
              const auto filename=name+".f32";
              std::ofstream file(capture_dir/filename,std::ios::binary);
              file.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(float));
              file.close();
              if (!file) throw std::runtime_error("cannot write capture "+name);
              manifest[name]={{"shape",shape},{"file",filename}};
            };
          }
          if (!encoder) encoder=std::make_unique<eg::Encoder>(directory,eg::kMaxTokens,vision,audio);
          const auto start=std::chrono::steady_clock::now();
          std::vector<std::vector<float>> vectors;
          if (request.contains("control_layer") || request.contains("control_vision_layer") || request.contains("control_bridge") ||
              request.contains("control_audio_subsample") || request.contains("control_audio_layer") || request.contains("control_audio_bridge")) {
            const std::filesystem::path source=request.at("control_dir").get<std::string>();
            auto read=[&](const std::string& name,std::size_t count) {
              std::ifstream file(source/(name+".f32"),std::ios::binary|std::ios::ate);
              if (!file || file.tellg()!=std::streamoff(count*sizeof(float)))
                throw std::invalid_argument("invalid layer control file "+name);
              file.seekg(0);
              std::vector<float> data(count);
              file.read(reinterpret_cast<char*>(data.data()),count*sizeof(float));
              if (!file) throw std::runtime_error("cannot read layer control file "+name);
              return data;
            };
            if (request.contains("control_audio_subsample") || request.contains("control_audio_layer") || request.contains("control_audio_bridge")) {
              if (input.audios.empty()) throw std::invalid_argument("audio control requires audio");
              const auto& clip=*input.audios.front();
              const int count=clip.end-clip.begin;
              auto audio_capture=[&](const auto& name,const auto& shape,const auto& values) {
                if (capture) capture("audio.0."+name,shape,values);
              };
              if (request.contains("control_audio_subsample")) {
                auto control=clip;
                control.features=read("audio.0.feature_input",clip.features.size());
                encoder->capture_audio_subsample(control,audio_capture);
              } else if (request.contains("control_audio_bridge"))
                encoder->capture_audio_bridge(read("audio.0.output_projection",count*1536),audio_capture);
              else {
                const auto& value=request["control_audio_layer"];
                if (!value.is_number_integer() || value<0 || value>=12) throw std::invalid_argument("invalid audio control layer");
                const auto layer=value.get<int>();
                const auto name=layer==0 || layer==11?"audio.0.layer."+std::to_string(layer)+".input":
                    "audio.0.layer."+std::to_string(layer-1)+".output";
                encoder->capture_audio_layer(layer,read(name,count*1024),audio_capture);
              }
            } else if (request.contains("control_vision_layer") || request.contains("control_bridge")) {
              if (input.visuals.empty()) throw std::invalid_argument("vision control requires an image");
              const auto& image=*input.visuals.front();
              const int count=image.end-image.begin;
              auto image_capture=[&](const auto& name,const auto& shape,const auto& values) {
                if (capture) capture("image.0."+name,shape,values);
              };
              if (request.contains("control_bridge"))
                encoder->capture_vision_bridge(read("image.0.bridge_input",count*768),image_capture);
              else {
                const auto& value=request["control_vision_layer"];
                if (!value.is_number_integer() || value<0 || value>=16) throw std::invalid_argument("invalid vision control layer");
                const auto layer=value.get<int>();
                encoder->capture_vision_layer(layer,image,read("image.0.layer."+std::to_string(layer)+".input",count*9*768),image_capture);
              }
            } else {
              const auto& value=request.at("control_layer");
              if (!value.is_number_integer() || value<0 || value>=eg::kLayers)
                throw std::invalid_argument("invalid control layer");
              const int layer=value.get<int>();
              encoder->capture_layer(layer,read(layer==0?"embedding":"layer."+std::to_string(layer-1)+".output",tokens.size()*eg::kHidden),
                  read("ple",tokens.size()*eg::kLayers*eg::kHidden),capture);
            }
          } else if (request.contains("token_ids")) vectors=encoder->encode_raw(tokens,dimensions,capture);
          else vectors=encoder->encode(input,dimensions,capture);
          response["milliseconds"]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
          for (std::size_t i=0;i<vectors.size();++i) response["embeddings"][std::to_string(dimensions[i])]=vectors[i];
          response["weight_bytes"]=encoder->weight_bytes();
          response["scratch_bytes"]=encoder->scratch_bytes();
          response["vision_weight_bytes"]=encoder->vision_weight_bytes();
          response["vision_scratch_bytes"]=encoder->vision_scratch_bytes();
          response["audio_weight_bytes"]=encoder->audio_weight_bytes();
          response["audio_scratch_bytes"]=encoder->audio_scratch_bytes();
          if (capture) {
            std::ofstream file(capture_dir/"manifest.json");
            file<<manifest.dump(2)<<'\n';
            file.close();
            if (!file) throw std::runtime_error("cannot write capture manifest");
          }
        }
        std::cout<<response.dump()<<std::endl;
      } catch (const std::exception& e) {
        std::cout<<Json({{"error",e.what()}}).dump()<<std::endl;
        failed=true;
      }
    }
    return failed?1:0;
  } catch (const std::exception& e) {
    std::cerr<<e.what()<<'\n';
    return 1;
  }
}
