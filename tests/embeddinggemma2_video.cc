#include "gewell/models/embeddinggemma2/video.h"
#include <openssl/evp.h>
#include <json.hpp>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace eg=gewell::embeddinggemma2;
using Json=nlohmann::json;
void require(bool ok,const char* message) {if (!ok) throw std::runtime_error(message);}
std::string digest(const std::vector<std::uint8_t>& bytes) {
  std::array<unsigned char,32> result{};
  unsigned int size{};
  require(EVP_Digest(bytes.data(),bytes.size(),result.data(),&size,EVP_sha256(),nullptr)==1 && size==32,"hash failed");
  std::ostringstream text;
  for (auto c:result) text<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(c);
  return text.str();
}
std::string data_url(const std::filesystem::path& path) {
  std::ifstream input(path,std::ios::binary);
  require(bool(input),"missing fixture");
  std::vector<unsigned char> bytes{std::istreambuf_iterator<char>(input),{}};
  std::string encoded(4*((bytes.size()+2)/3)+1,'\0');
  encoded.resize(EVP_EncodeBlock(reinterpret_cast<unsigned char*>(encoded.data()),bytes.data(),bytes.size()));
  return std::string(path.extension()==".mp4"?"data:video/mp4;base64,":"data:video/webm;base64,")+encoded;
}
template<class F> void invalid(F operation) {
  try {operation();} catch (const std::invalid_argument&) {return;}
  throw std::runtime_error("invalid input was accepted");
}
int main(int argc,char** argv) {
  try {
    if (argc==2 && std::string(argv[1])=="--sample") {
      std::string line;
      while (std::getline(std::cin,line)) {
        const auto row=Json::parse(line);
        std::cout<<Json(eg::sample_video_frames(row.at("total_frames"),row.at("fps"))).dump()<<'\n';
      }
      return 0;
    }
    require(eg::sample_video_frames(45,15)==std::vector<std::uint64_t>({0,15,30}),"CFR sampling");
    require(eg::sample_video_frames(95,30000./1001)==std::vector<std::uint64_t>({0,29,59}),"fractional FPS sampling");
    require(eg::sample_video_frames(4,.5)==std::vector<std::uint64_t>({0,0,1,1,2,2,3,3}),"low FPS repetitions");
    require(eg::sample_video_frames(4,30)==std::vector<std::uint64_t>({0}),"short clip sampling");
    require(eg::sample_video_frames(70,2)==std::vector<std::uint64_t>({0,2,4,6,8,10,12,14,16,18,20,24,26,28,30,32,34,36,38,40,42,46,48,50,52,54,56,58,60,62,64,68}),"uniform overflow");
    require(eg::sample_video_frames(3,0)==std::vector<std::uint64_t>({0,1,2}),"missing FPS");
    invalid([]{eg::sample_video_frames(0,30);});
    for (auto url:{"https://example.com/a.mp4","file:///tmp/a.mp4","data:video/avi;base64,AAAA",
                  "data:video/mp4;base64,","data:video/mp4;base64,A","data:video/mp4;base64,AA=A",
                  "data:video/mp4;base64,AB==","data:video/mp4;base64,AAB=","data:video/mp4;base64,====",
                  "data:video/mp4;base64,AA==AAAA","data:video/mp4;base64,AAA\n","data:video/webm;base64,AAAA"})
      invalid([&]{eg::prepare_video_data_url(url);});
    invalid([]{eg::prepare_video_data_url(std::string(gewell::gemma4::kImageMaxDataUrlBytes+1,'x'));});
    invalid([]{(void)gewell::gemma4::prepare_rgb_image({0,1,{}});});
    invalid([]{(void)gewell::gemma4::prepare_rgb_image({1,1,{0,0}});});
    invalid([]{(void)gewell::gemma4::prepare_rgb_image({8193,1,{}});});
    if (argc==3) {
      const std::filesystem::path fixtures=argv[1];
      std::ifstream file(argv[2]); const auto reference=Json::parse(file);
      for (const auto& item:reference.at("cases")) {
        const auto& content=item.at("content");
        if (content.size()!=1 || !content[0].contains("video")) continue;
        const auto url=data_url(fixtures/content[0]["video"].get<std::string>());
        const auto& expected=item.at("videos")[0];
        std::map<std::uint64_t,std::string> rgb;
        auto prepared=eg::prepare_video_data_url(url,{}, {},[&](auto index,const auto& image){rgb[index]=digest(image.pixels);});
        require(prepared.metadata.total_frames==expected["total_num_frames"],"decoded frame count differs");
        require(expected["fps"].is_null() ? prepared.metadata.fps==0 && prepared.metadata.duration==0 :
            prepared.metadata.fps==expected["fps"] && prepared.metadata.duration==expected["duration"],"timing metadata differs");
        require(Json(prepared.metadata.indices)==expected["selected_indices"],"selected indices differ");
        require(prepared.frames.size()==item["images"].size(),"frame preparation count differs");
        for (std::size_t i=0;i<prepared.frames.size();++i) {
          const auto index=prepared.metadata.indices[i];
          require(rgb.at(index)==expected["rgb_sha256"][index],"decoded RGB differs");
          require(expected["timestamps"][index].is_null()?std::isnan(prepared.metadata.timestamps[i]):
              std::abs(prepared.metadata.timestamps[i]-expected["timestamps"][index].get<double>())<=.00000051,"timestamp differs");
          const auto& frame=*prepared.frames[i];
          const auto& hash=item["images"][i];
          require(digest(frame.pixels)==hash["pixels_sha256"] && digest(frame.positions)==hash["positions_sha256"],"prepared frame differs");
          require(frame.begin==0 && frame.end==hash["soft_tokens"] && frame.padded_patch_rows==1260,"frame geometry differs");
        }
        std::cout<<item["id"].get<std::string>()<<": RGB, metadata and "<<prepared.frames.size()<<" prepared frames exact\n";
      }
      if (std::filesystem::exists(fixtures/"red.mp4")) {
      const auto url=data_url(fixtures/"red.mp4");
      int polls=0;
      try {eg::prepare_video_data_url(url,{},[&]{if (++polls==8) throw std::runtime_error("cancel-count");});throw std::logic_error("cancel ignored");}
      catch (const std::runtime_error& error) {require(std::string(error.what())=="cancel-count","wrong counting cancellation");}
      std::weak_ptr<gewell::runtime::ImageInput> released;
      int made=0;
      try {
        eg::prepare_video_data_url(url,[&](auto image) {
          auto source=gewell::gemma4::prepare_rgb_image(std::move(image),140);
          auto frame=std::make_shared<gewell::runtime::ImageInput>();
          frame->pixels=std::move(source.pixels);frame->positions=std::move(source.positions);
          frame->padded_patch_rows=source.padded_patch_rows;frame->end=source.soft_token_count;
          released=frame;++made;return frame;
        },[&]{if (made) throw std::runtime_error("cancel-prepared");});
        throw std::logic_error("cancel ignored");
      } catch (const std::runtime_error& error) {require(std::string(error.what())=="cancel-prepared","wrong prepared cancellation");}
      require(made==1 && released.expired(),"cancelled preparation retained a frame");
      require(eg::prepare_video_data_url(url).frames.size()==3,"decode after cancellation failed");
      auto wrong_mime=url;wrong_mime.replace(11,3,"webm");
      invalid([&]{eg::prepare_video_data_url(wrong_mime);});
      invalid([&]{eg::prepare_video_data_url(url.substr(0,url.size()/2));});
      std::cout<<"counting/preparation cancellation, ownership release and recovery passed\n";
      }
      std::ifstream corpus_file(fixtures/"cases.json");
      const auto corpus=Json::parse(corpus_file);
      if (corpus.contains("rejected")) for (const auto& name:corpus["rejected"]) {
        invalid([&]{eg::prepare_video_data_url(data_url(fixtures/name.get<std::string>()));});
        std::cout<<name.get<std::string>()<<": rejected\n";
      }
    } else if (argc!=1) throw std::invalid_argument("usage: video_test [FIXTURES REFERENCE_JSON] | --sample");
    std::cout<<"video processor tests passed\n";
  } catch (const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
