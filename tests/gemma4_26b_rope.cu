#include "models/gemma4/26b_a4b/sm120/bf16_math.cuh"
#include "cuda_memory.cuh"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace math=gewell::gemma4_26b_a4b::sm120::bf16_math;
using gewell::cuda_detail::DeviceAllocation;
using gewell::cuda_detail::check_cuda;
using B=__nv_bfloat16;
namespace fs=std::filesystem;
std::vector<char> read(const fs::path& path,std::size_t bytes) {
  std::vector<char> data(bytes);
  std::ifstream input(path,std::ios::binary);input.read(data.data(),bytes);
  if (!input || input.peek()!=EOF) throw std::runtime_error("fixture size: "+path.string());
  return data;
}
int main(int argc,char** argv) {
  try {
    if (argc!=2) throw std::runtime_error("usage: rope FIXTURES");
    fs::path fixtures(argv[1]);
    auto position_bytes=fs::file_size(fixtures/"positions.u32");
    if (!position_bytes || position_bytes%4 || position_bytes>4096*4)
      throw std::runtime_error("invalid position fixture");
    unsigned rows=position_bytes/4;
    auto positions=read(fixtures/"positions.u32",position_bytes);
    DeviceAllocation device_positions(position_bytes);
    check_cuda(cudaMemcpy(device_positions.data(),positions.data(),position_bytes,cudaMemcpyHostToDevice),"positions");
    for (bool global:{false,true}) for (unsigned heads:{16u,global?2u:8u}) {
      unsigned width=global?512:256;
      std::string name=std::string(global?"global":"local")+"-"+std::to_string(heads);
      std::size_t elements=std::size_t(rows)*heads*width,bytes=elements*2;
      auto input=read(fixtures/(name+".input.bf16"),bytes);
      auto expected=read(fixtures/(name+".expected.bf16"),bytes);
      DeviceAllocation x(bytes),y(bytes);
      check_cuda(cudaMemcpy(x.data(),input.data(),bytes,cudaMemcpyHostToDevice),"input");
      math::row_rope<<<(elements+255)/256,256>>>(static_cast<B*>(x.data()),static_cast<B*>(y.data()),
          heads,global,static_cast<unsigned*>(device_positions.data()),rows);
      check_cuda(cudaGetLastError(),"row RoPE launch");
      std::vector<char> actual(bytes);
      check_cuda(cudaMemcpy(actual.data(),y.data(),bytes,cudaMemcpyDeviceToHost),"output");
      if (actual!=expected) throw std::runtime_error(name+": RoPE differs from independent reference");
      std::cout<<name<<": "<<elements<<" BF16 elements exact\n";
    }
    return 0;
  } catch (const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
