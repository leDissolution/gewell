#include "gewell/models/embeddinggemma2/audio_processor.h"
#include "gewell/models/embeddinggemma2/input.h"
#include <openssl/evp.h>
#include <json.hpp>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace eg=gewell::embeddinggemma2;
namespace audio=eg::audio;
using Json=nlohmann::json;
void require(bool ok,const char* message) {if (!ok) throw std::runtime_error(message);}
std::string digest(const void* data,std::size_t bytes) {
  std::array<unsigned char,32> result{};
  unsigned int size{};
  require(EVP_Digest(data,bytes,result.data(),&size,EVP_sha256(),nullptr)==1 && size==32,"hash failed");
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
  return std::string(path.extension()==".wav"?"data:audio/wav;base64,":path.extension()==".mp3"?
      "data:audio/mpeg;base64,":"data:audio/flac;base64,")+encoded;
}
Json read_json(const std::filesystem::path& path) {
  std::ifstream input(path);require(bool(input),"missing JSON fixture");return Json::parse(input);
}
std::vector<float> read_floats(const std::filesystem::path& path,std::size_t count) {
  std::ifstream input(path,std::ios::binary);require(bool(input),"missing feature capture");
  std::vector<float> result(count);
  input.read(reinterpret_cast<char*>(result.data()),count*sizeof(float));
  require(input.gcount()==std::streamsize(count*sizeof(float)) && input.peek()==EOF,"capture size differs");
  return result;
}
template<class Error=std::invalid_argument,class F> void rejected(F operation) {
  try {operation();} catch (const Error&) {return;}
  throw std::runtime_error("invalid input was accepted");
}
int main(int argc,char** argv) {
  try {
    for (auto url:{"https://example.com/a.wav","file:///tmp/a.wav","data:audio/ogg;base64,AAAA",
                  "data:audio/wav;base64,","data:audio/wav;base64,A","data:audio/wav;base64,AA=A",
                  "data:audio/wav;base64,AB==","data:audio/wav;base64,AAB=","data:audio/wav;base64,====",
                  "data:audio/wav;base64,AA==AAAA","data:audio/wav;base64,AAA\n","data:audio/flac;base64,AAAA"})
      rejected([&]{audio::prepare_data_url(url);});
    rejected([]{audio::prepare_data_url(std::string(audio::kMaxDataUrlBytes+1,'x'));});
    for (int n:{0,1,160}) rejected([&]{audio::prepare_samples(std::vector<float>(n));});
    rejected([]{audio::prepare_samples(std::vector<float>(161,std::nanf("")));});
    rejected<eg::ContextLengthError>([]{audio::prepare_samples(std::vector<float>(161),4);});
    rejected<eg::ContextLengthError>([]{audio::prepare_samples(std::vector<float>(801),5);});
    auto small=audio::prepare_samples(std::vector<float>(161),5);
    require(small->mask==std::vector<std::uint8_t>{1} && small->end==1,"short audio geometry");
    auto damaged=*small;damaged.mask[0]=0;rejected([&]{audio::validate_features(damaged);});
    damaged=*small;damaged.end=2;rejected([&]{audio::validate_features(damaged);});
    require(audio::max_samples(8192)==5240480 && audio::feature_rows(5240480)==32753,"full-context audio geometry");
    if (argc==3 || argc==4) {
      const std::filesystem::path fixtures=argv[1],reference_path=argv[2];
      const auto reference=read_json(reference_path);
      Json rows=Json::array();std::set<std::string> seen;
      double overall_abs=0,overall_relative=0;
      for (std::size_t index=0;index<reference["cases"].size();++index) {
        const auto& item=reference["cases"][index];
        const auto directory=reference_path.parent_path()/std::to_string(index);
        const auto manifest=read_json(directory/"manifest.json");
        for (std::size_t i=0;i<item["audios"].size();++i) {
          const auto& expected=item["audios"][i];
          const std::string name=expected["file"];
          if (!seen.insert(name).second) continue;
          std::string samples_hash;
          std::size_t reserved=0;
          auto prepared=audio::prepare_data_url(data_url(fixtures/name),8192,
              [&](std::uint32_t frames){reserved=std::size_t(frames)*513;return std::make_shared<audio::Input>();},
              {},[&](const auto& samples){samples_hash=digest(samples.data(),samples.size()*sizeof(float));});
          const auto& input=*prepared.input;
          require(samples_hash==expected["samples_sha256"],"decoded samples differ");
          require(prepared.metadata.samples==expected["samples"] && prepared.metadata.input_rate==expected["input_rate"] &&
              prepared.metadata.channels==expected["channels"] && prepared.metadata.layout==expected["layout"] &&
              prepared.metadata.codec==expected["codec"],"decoded metadata differs");
          require(input.mask.size()==expected["feature_frames"] && input.end==expected["soft_tokens"] &&
              input.begin==0 && reserved==input.features.size()*sizeof(float)+input.mask.size(),"feature allocation/count differs");
          require(digest(input.mask.data(),input.mask.size())==expected["mask_sha256"],"feature mask differs");
          const auto& spec=manifest["audio."+std::to_string(i)+".feature_input"];
          const auto wanted=read_floats(directory/spec["file"].get<std::string>(),input.features.size());
          double max_abs=0,xx=0,dd=0;
          Json bf16_differences=Json::array();
          auto bf16=[](float value) {
            std::uint32_t bits;std::memcpy(&bits,&value,sizeof(bits));
            return (bits+0x7fff+((bits>>16)&1))>>16;
          };
          for (std::size_t j=0;j<wanted.size();++j) {
            const double delta=double(input.features[j])-wanted[j];
            max_abs=std::max(max_abs,std::abs(delta));dd+=delta*delta;xx+=double(wanted[j])*wanted[j];
            if (input.features[j]!=wanted[j] && bf16(input.features[j])!=bf16(wanted[j])) bf16_differences.push_back({{"row",j/128},{"bin",j%128},
                {"reference",wanted[j]},{"native",input.features[j]}});
          }
          const double relative=std::sqrt(dd/std::max(xx,1e-60));
          overall_abs=std::max(overall_abs,max_abs);overall_relative=std::max(overall_relative,relative);
          rows.push_back({{"file",name},{"samples_exact",true},{"mask_exact",true},{"max_abs",max_abs},{"relative_l2",relative},
              {"features_sha256",digest(input.features.data(),input.features.size()*sizeof(float))},
              {"bf16_differences",bf16_differences}});
          std::cout<<name<<": samples/mask exact, features max_abs "<<max_abs<<", relative_l2 "<<relative<<'\n';
          require(max_abs<=1e-5 && relative<=1e-6,"feature numerical gate failed");
        }
      }
      const auto corpus=read_json(fixtures/"cases.json");
      for (const auto& name:corpus["rejected"]) {
        rejected([&]{audio::prepare_data_url(data_url(fixtures/name.get<std::string>()));});
        std::cout<<name.get<std::string>()<<": rejected\n";
      }
      rejected<eg::ContextLengthError>([&]{audio::prepare_data_url(data_url(fixtures/corpus["context_overflow"].get<std::string>()));});
      const auto url=data_url(fixtures/"dog.wav");
      int polls=0;
      try {audio::prepare_data_url(url,8192,{},[&]{if (++polls==4) throw std::runtime_error("cancel-decode");});throw std::logic_error("cancel ignored");}
      catch (const std::runtime_error& error) {require(std::string(error.what())=="cancel-decode","wrong decode cancellation");}
      std::weak_ptr<audio::Input> released;int prepared_polls=0;
      try {
        audio::prepare_data_url(url,8192,[&](auto){auto input=std::make_shared<audio::Input>();released=input;return input;},
            [&]{if (!released.expired() && ++prepared_polls==2) throw std::runtime_error("cancel-features");});
        throw std::logic_error("cancel ignored");
      } catch (const std::runtime_error& error) {require(std::string(error.what())=="cancel-features","wrong feature cancellation");}
      require(released.expired(),"cancelled preparation retained features");
      require(audio::prepare_data_url(url).input->end>0,"decode after cancellation failed");
      rejected<std::bad_alloc>([&]{audio::prepare_data_url(url,8192,[](auto)->std::shared_ptr<audio::Input>{throw std::bad_alloc();});});
      require(audio::prepare_data_url(url).input->end>0,"decode after reservation failure failed");
      Json report{{"files",rows},{"max_feature_abs",overall_abs},{"max_feature_relative_l2",overall_relative},
          {"decode_and_feature_cancellation_passed",true},{"reservation_failure_and_recovery_passed",true},
          {"malformed_fixtures_rejected",corpus["rejected"]},{"context_overflow_rejected",true}};
      if (argc==4) {
        require(!std::filesystem::exists(argv[3]),"preserve existing validation report");
        std::ofstream output(argv[3]);output<<report.dump(2)<<'\n';require(bool(output),"cannot save report");
      }
    } else if (argc!=1) throw std::invalid_argument("usage: audio_processor_test [FIXTURES REFERENCE_JSON [NEW_REPORT]]");
    std::cout<<"audio processor tests passed\n";
  } catch (const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
