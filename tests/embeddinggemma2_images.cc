#include "gewell/models/embeddinggemma2/encoder.h"
#include "gewell/models/embeddinggemma2/text.h"
#include "gewell/models/gemma4/image_processor.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace eg=gewell::embeddinggemma2;
void require(bool condition,const char* message) {if (!condition) throw std::runtime_error(message);}
int main(int argc,char** argv) {
  try {
    if (argc!=3) throw std::invalid_argument("usage: embeddinggemma2_images BUNDLE IMAGE_FIXTURE.json");
    gewell::text::Tokenizer tokenizer(std::string(argv[1])+"/tokenizer.json",eg::text_contract());
    std::ifstream file(argv[2]); nlohmann::json fixtures;file>>fixtures;
    auto prepared=gewell::gemma4::prepare_image_data_url(fixtures.at("cases").at(0).at("data_url").get<std::string>(),70);
    auto image=std::make_shared<gewell::runtime::ImageInput>();
    image->pixels=std::move(prepared.pixels);image->positions=std::move(prepared.positions);
    image->padded_patch_rows=prepared.padded_patch_rows;image->end=prepared.soft_token_count;
    const auto input=eg::prepare_input(tokenizer,std::vector<eg::InputPart>{{"before ",{}},{"",image},{" after",{}}});
    auto invalid=[&](eg::PreparedInput value) {
      bool rejected=false;
      try {eg::validate_input(value);} catch (const std::invalid_argument&) {rejected=true;}
      require(rejected,"invalid prepared input was accepted");
    };
    auto bad=input;bad.visuals.clear();invalid(bad);
    bad=input;bad.tokens[image->begin]=42;invalid(bad);
    bad=input;bad.tokens[image->end]=42;invalid(bad);
    bad=input;bad.visuals.push_back(image);invalid(bad);
    for (int mutation=0;mutation<4;++mutation) {
      auto broken=std::make_shared<gewell::runtime::ImageInput>(*image);
      if (mutation==0) ++broken->begin;
      if (mutation==1) ++broken->padded_patch_rows;
      if (mutation==2) broken->pixels.back()=1;
      if (mutation==3) broken->positions[0]=1;
      bad=input;bad.visuals[0]=broken;invalid(bad);
    }
    eg::Encoder encoder(argv[1],8192,true);
    require(encoder.scratch_bytes()<500ULL*1024*1024,"full-context text/vision workspace exceeds shared budget");
    const auto baseline=encoder.encode(input,{768,512,256,128});
    int polls=0;bool cancelled=false;std::string last;
    try {
      encoder.encode(input,{768},[&](const auto& name,const auto&,const auto&){last=name;},[&]{return ++polls==4;});
    } catch (const std::runtime_error& error) {cancelled=std::string(error.what()).find("cancelled")!=std::string::npos;}
    require(cancelled && last=="image.0.layer.0.output","cancellation did not occur in vision");
    require(encoder.encode(input,{768,512,256,128})==baseline,"cancelled vision contaminated reuse");
    const eg::PreparedInput text={{2,1234,4567,1},{}};
    const auto text_baseline=encoder.encode(text,{768,512,256,128});
    {
      // With a small text capacity the vision requirement determines arena
      // size. Alternating modalities also checks persistent vision constants.
      eg::Encoder compact(argv[1],128,true);
      require(compact.scratch_bytes()<400ULL*1024*1024,"small-context shared workspace exceeds budget");
      require(compact.encode(input,{768,512,256,128})==baseline,"small-capacity image differs");
      require(compact.encode(text,{768,512,256,128})==text_baseline,"small-capacity text differs");
      require(compact.encode(input,{768,512,256,128})==baseline,"text contaminated shared vision workspace");
    }
    auto long_input=input;
    long_input.tokens.insert(long_input.tokens.end()-1,700,1000);
    encoder.encode(long_input,{768});
    require(encoder.encode(input,{768,512,256,128})==baseline,"long input contaminated vision reuse");
    require(encoder.encode(text,{768,512,256,128})==text_baseline,"image input contaminated text reuse");
    auto first_frame=std::make_shared<gewell::runtime::ImageInput>(*image);
    auto second_frame=std::make_shared<gewell::runtime::ImageInput>(*image);
    const auto video=eg::prepare_input(tokenizer,std::vector<eg::InputPart>{{"before ",{}},
        {"",{},{first_frame,second_frame}},{" after",{}}});
    require(video.video_count==1 && video.visuals.size()==2,"video frame assembly count differs");
    const auto video_baseline=encoder.encode(video,{768,512,256,128});
    last.clear();cancelled=false;
    try {
      encoder.encode(video,{768},[&](const auto& name,const auto&,const auto&){last=name;},
          [&]{return last=="image.0.features";});
    } catch (const std::runtime_error& error) {cancelled=std::string(error.what()).find("cancelled")!=std::string::npos;}
    require(cancelled,"cancellation between video frames was ignored");
    require(encoder.encode(video,{768,512,256,128})==video_baseline,"cancelled video contaminated reuse");
    require(encoder.encode(input,{768,512,256,128})==baseline,"video contaminated image reuse");
    require(encoder.encode(text,{768,512,256,128})==text_baseline,"video contaminated text reuse");
    // Shift a feature span across the local attention boundary. The complete
    // sequence, including media rows, must obey the existing +/-512 mask.
    for (bool frame:{false,true}) for (int offset:{511,512,513}) {
      auto shifted=std::make_shared<gewell::runtime::ImageInput>(*image);
      std::string text_prefix;
      for (int i=0;i<offset-2;++i) text_prefix+="<mask>";
      const auto media=frame?eg::InputPart{{},{},{shifted}}:eg::InputPart{{},shifted};
      auto prefix=eg::prepare_input(tokenizer,std::vector<eg::InputPart>{{text_prefix,{}},media});
      require(prefix.visuals.front()->begin==static_cast<std::uint32_t>(offset),"media boundary fixture shifted");
      std::vector<float> mask;
      encoder.encode(prefix,{128},[&](const auto& name,const auto&,const auto& values){if(name=="layer.0.mask")mask=values;});
      const auto n=prefix.tokens.size();
      require(mask.size()==n*n,"missing local attention mask");
      for (std::size_t i=0;i<n;++i) for (std::size_t j=0;j<n;++j)
        require(mask[i*n+j]==(std::abs(static_cast<int>(i)-static_cast<int>(j))<=512?1.f:0.f),"media local attention distance mismatch");
    }
    std::cout<<"prepared spans/geometry, image/video cancellation recovery, visual/text scratch reuse, and local masks passed\n";
    return 0;
  } catch (const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
