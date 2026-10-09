#include "gewell/models/embeddinggemma2/encoder.h"
#include "gewell/models/embeddinggemma2/text.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace eg=gewell::embeddinggemma2;
void require(bool condition,const char* message) {if (!condition) throw std::runtime_error(message);}
int main(int argc,char** argv) {
  try {
    if (argc!=2) throw std::invalid_argument("usage: embeddinggemma2_audios BUNDLE");
    gewell::text::Tokenizer tokenizer(std::string(argv[1])+"/tokenizer.json",eg::text_contract());
    auto clip=[](int count) {
      std::vector<float> values(count);
      for (int i=0;i<count;++i) values[i]=.2f*std::sin(i*.137f);
      return eg::audio::prepare_samples(values);
    };
    auto first=clip(16000),second=clip(801);
    const auto input=eg::prepare_input(tokenizer,std::vector<eg::InputPart>{
        {"before "},{"",{},{},first},{" between "},{"",{},{},second},{" after"}});
    require(input.audios.size()==2 && input.audios[0]->end<input.audios[1]->begin,"audio order differs");
    auto invalid=[&](eg::PreparedInput value) {
      bool rejected=false;
      try {eg::validate_input(value);} catch (const std::invalid_argument&) {rejected=true;}
      require(rejected,"invalid prepared audio accepted");
    };
    auto bad=input;bad.audios.clear();invalid(bad);
    bad=input;bad.audios.push_back(first);invalid(bad);
    bad=input;bad.tokens[first->begin]=42;invalid(bad);
    bad=input;bad.tokens[first->end]=42;invalid(bad);
    bad=input;std::swap(bad.audios[0],bad.audios[1]);invalid(bad);
    for (int mutation=0;mutation<7;++mutation) {
      auto broken=std::make_shared<eg::audio::Input>(*first);
      if (mutation==0) ++broken->begin;
      if (mutation==1) ++broken->samples;
      if (mutation==2) broken->mask.pop_back();
      if (mutation==3) broken->mask.front()=0;
      if (mutation==4) broken->features.pop_back();
      if (mutation==5) broken->features.front()=std::numeric_limits<float>::quiet_NaN();
      if (mutation==6) broken->end=broken->begin;
      bad=input;bad.audios[0]=broken;invalid(bad);
    }
    eg::Encoder encoder(argv[1],1026,false,true);
    const auto baseline=encoder.encode(input,{768,512,256,128});
    const auto text=eg::prepare_input(tokenizer,"A reusable text request.");
    const auto text_baseline=encoder.encode(text,{768,512,256,128});
    for (const std::string boundary:{"audio.0.subsample.projected","audio.0.layer.0.output","audio.0.features"}) {
      bool cancelled=false;std::string last;
      try {
        encoder.encode(input,{768},[&](const auto& name,const auto&,const auto&){last=name;},[&]{return last==boundary;});
      } catch (const std::runtime_error& error) {cancelled=std::string(error.what()).find("cancelled")!=std::string::npos;}
      require(cancelled,"audio boundary cancellation ignored");
      require(encoder.encode(input,{768,512,256,128})==baseline,"cancelled audio contaminated reuse");
      require(encoder.encode(text,{768,512,256,128})==text_baseline,"cancelled audio contaminated text");
    }
    const auto longer=eg::prepare_input(tokenizer,std::vector<eg::InputPart>{{"",{},{},clip(160000)}});
    encoder.encode(longer,{128});
    require(encoder.encode(input,{768,512,256,128})==baseline,"long audio contaminated reuse");
    for (int offset:{511,512,513}) {
      std::string prefix;
      for (int i=0;i<offset-2;++i) prefix+="<mask>";
      const auto shifted=eg::prepare_input(tokenizer,std::vector<eg::InputPart>{{prefix},{"",{},{},clip(161)}});
      require(shifted.audios[0]->begin==std::uint32_t(offset),"audio boundary fixture shifted");
      std::vector<float> mask;
      encoder.encode(shifted,{128},[&](const auto& name,const auto&,const auto& values){if(name=="layer.0.mask")mask=values;});
      const auto n=shifted.tokens.size();
      require(mask.size()==n*n,"missing encoder local mask");
      for (std::size_t i=0;i<n;++i) for (std::size_t j=0;j<n;++j)
        require(mask[i*n+j]==(std::abs(int(i)-int(j))<=512?1.f:0.f),"audio local attention distance differs");
    }
    require(encoder.audio_weight_bytes()>0 && encoder.audio_scratch_bytes()>0 && encoder.vision_weight_bytes()==0,
        "audio allocation accounting differs");
    std::cout<<"audio geometry, ordered spans, boundary cancellation, scratch reuse and local masks passed\n";
    return 0;
  } catch (const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
