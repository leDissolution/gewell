#include "gewell/models/embeddinggemma2/encoder.h"
#include "gewell/models/embeddinggemma2/model.h"
#include <algorithm>
#include <iostream>
#include <map>
#include <stdexcept>

namespace eg=gewell::embeddinggemma2;
void require(bool condition,const char* message) {if (!condition) throw std::runtime_error(message);}
int main(int argc,char** argv) {
  try {
    if (argc!=2) throw std::invalid_argument("usage: embeddinggemma2_encoder BUNDLE");
    eg::Encoder encoder(argv[1],1026);
    const eg::PreparedInput short_input={{2,1234,4567,1},{}};
    const auto before=encoder.encode(short_input,{768,512,256,128});
    int polls=0;
    bool cancelled=false;
    try {encoder.encode(short_input,{768},{},[&]{return ++polls==4;});}
    catch (const std::runtime_error& e) {cancelled=std::string(e.what()).find("cancelled")!=std::string::npos;}
    require(cancelled,"layer-boundary cancellation was not observed");
    require(encoder.encode(short_input,{768,512,256,128})==before,"cancelled work corrupted the next request");
    bool rejected=false;
    try {encoder.encode({std::vector<std::uint32_t>(1027,2),{}});}
    catch (const std::invalid_argument&) {rejected=true;}
    require(rejected,"physical token capacity was not enforced");
    using Captures=std::map<std::string,std::vector<float>>;
    auto run=[&](const std::vector<std::uint32_t>& tokens) {
      Captures captures;
      encoder.encode({tokens,{}},{768},[&](const auto& name,const auto&,const auto& values) {
        if (name=="layer.0.context" || name=="layer.0.output" || name=="layer.5.output")
          captures[name]=values;
      });
      return captures;
    };
    std::vector<std::uint32_t> base(514,1000);
    base.front()=2; base.back()=1;
    auto inside=base,outside=base;
    inside[512]=2000; outside[513]=2000;
    const auto a=run(base), b=run(inside), c=run(outside);
    auto first_row_equal=[](const Captures& x,const Captures& y,const std::string& name,int width) {
      return std::equal(x.at(name).begin(),x.at(name).begin()+width,y.at(name).begin());
    };
    require(!first_row_equal(a,b,"layer.0.context",1024),"future token at +512 did not affect local context");
    require(first_row_equal(a,c,"layer.0.output",512),"future token at +513 leaked into local layer 0");
    require(!first_row_equal(a,c,"layer.5.output",512),"suffix change did not reach earlier global representation");
    require(encoder.encode(short_input,{768,512,256,128})==before,"previous sequence leaked into the next request");
    std::cout<<"cancellation recovery, physical capacity, bidirectional suffix visibility, and request isolation passed\n";
    return 0;
  } catch (const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
