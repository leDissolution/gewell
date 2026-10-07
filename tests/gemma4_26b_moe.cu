#include "models/gemma4/26b_a4b/sm120/moe.cuh"
#include "gewell/models/gemma4/26b_a4b/artifact.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>

namespace m = gewell::gemma4_26b_a4b;
namespace s = gewell::gemma4_26b_a4b::sm120;
namespace fs = std::filesystem;
using B = __nv_bfloat16;
using gewell::cuda_detail::DeviceAllocation;
using gewell::cuda_detail::check_cuda;
std::vector<char> read(const fs::path& path, std::size_t bytes) {
  std::ifstream f(path,std::ios::binary);
  std::vector<char> result(bytes);
  if (bytes) f.read(result.data(),bytes);
  if (!f || f.peek() != EOF) throw std::runtime_error("wrong fixture size: "+path.string());
  return result;
}
std::vector<char> download(const void* data, std::size_t bytes) {
  std::vector<char> result(bytes);
  if (bytes) check_cuda(cudaMemcpy(result.data(),data,bytes,cudaMemcpyDeviceToHost),"download MoE capture");
  return result;
}
void save(const fs::path& path, const void* data, std::size_t bytes) {
  auto result = download(data,bytes);
  std::ofstream f(path,std::ios::binary);
  f.write(result.data(),bytes);
  if (!f) throw std::runtime_error("write fixture output");
}
int main(int argc,char** argv) {
  try {
    if (argc != 4) throw std::runtime_error("usage: moe ARTIFACT FIXTURES OUTPUT");
    fs::path fixtures(argv[2]), output(argv[3]);
    if (!fs::create_directory(output)) throw std::runtime_error("refusing existing output");
    auto artifact = m::ArtifactFile::Open(argv[1]);
    constexpr unsigned capacity = 4096;
    constexpr std::size_t matrix_bytes = std::size_t(704)*2816*2;
    DeviceAllocation weight_data(128*3*matrix_bytes), weight_table(128*sizeof(s::ExpertWeights));
    DeviceAllocation scales(128*2), input(capacity*2816*2), scores(capacity*128*2), result(capacity*2816*2);
    std::array<s::ExpertWeights,128> weights{};
    for (unsigned i = 0; i < m::kTextTensors.size(); ++i) {
      const auto& spec = m::kTextTensors[i];
      if (spec.layer != 0) continue;
      if (spec.role == m::TensorRole::router_per_expert_scale)
        check_cuda(cudaMemcpy(scales.data(),artifact.tensor_data(i),256,cudaMemcpyHostToDevice),"copy scales");
      if (spec.expert < 0) continue;
      unsigned role = unsigned(spec.role)-unsigned(m::TensorRole::expert_gate_proj);
      B* pointer = reinterpret_cast<B*>(static_cast<char*>(weight_data.data())+(spec.expert*3+role)*matrix_bytes);
      check_cuda(cudaMemcpy(pointer,artifact.tensor_data(i),matrix_bytes,cudaMemcpyHostToDevice),"copy expert");
      auto& w = weights[spec.expert];
      if (role == 0) w.gate.bf16 = pointer; else if (role == 1) w.up.bf16 = pointer; else w.down.bf16 = pointer;
    }
    check_cuda(cudaMemcpy(weight_table.data(),weights.data(),sizeof(weights),cudaMemcpyHostToDevice),"copy weight table");
    s::MoeWorkspace workspace(capacity);
    std::ifstream list(fixtures/"cases.tsv");
    std::string name; unsigned rows; bool dispatch_only;
    while (list >> name >> rows >> dispatch_only) {
      if (rows > capacity) throw std::runtime_error("fixture rows exceed capacity");
      auto hx = read(fixtures/name/"input.bf16",rows*2816*2);
      auto hs = read(fixtures/name/"scores.bf16",rows*128*2);
      if (rows) {
        check_cuda(cudaMemcpy(input.data(),hx.data(),hx.size(),cudaMemcpyHostToDevice),"copy input");
        check_cuda(cudaMemcpy(scores.data(),hs.data(),hs.size(),cudaMemcpyHostToDevice),"copy scores");
      }
      auto run = [&] {
        if (dispatch_only) workspace.dispatch(static_cast<B*>(scores.data()),static_cast<B*>(scales.data()),rows);
        else workspace.run(static_cast<B*>(input.data()),static_cast<B*>(scores.data()),static_cast<B*>(scales.data()),
                           static_cast<s::ExpertWeights*>(weight_table.data()),static_cast<B*>(result.data()),rows,true,false,false,false);
      };
      run();
      auto first = dispatch_only ? std::vector<char>{} : download(result.data(),rows*2816*2);
      run();
      if (!dispatch_only && first != download(result.data(),rows*2816*2)) throw std::runtime_error("non-repeatable MoE output");
      fs::create_directory(output/name);
      save(output/name/"ids.i32",workspace.selected_experts(),rows*8*4);
      save(output/name/"weights.f32",workspace.routing_weights(),rows*8*4);
      if (rows) {
        save(output/name/"offsets.i32",workspace.offsets(),129*4);
        save(output/name/"assignments.i32",workspace.sorted_assignments(),rows*8*4);
      }
      if (!dispatch_only) save(output/name/"output.bf16",result.data(),rows*2816*2);
      if (!dispatch_only && rows && rows <= 32) {
        save(output/name/"gate_up.bf16",workspace.gate_up_capture(),rows*8*1408*2);
        save(output/name/"activated.bf16",workspace.activated_capture(),rows*8*704*2);
        save(output/name/"down.bf16",workspace.down_capture(),rows*8*2816*2);
      }
      std::cout << name << " rows=" << rows << " workspace_bytes=" << workspace.bytes() << '\n';
    }
    if (!list.eof()) throw std::runtime_error("invalid fixture list");
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
