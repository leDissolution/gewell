#include "gewell/models/gemma4/26b_a4b/weight_qdq.h"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>
using namespace gewell::gemma4_26b_a4b;
using Type = gewell::weight_qdq::Type;
using B = __nv_bfloat16;
void check(cudaError_t result) { if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result)); }
void require(bool result, const char* message) { if (!result) throw std::runtime_error(message); }
struct Device {
  void* data{};
  explicit Device(std::size_t bytes) { check(cudaMalloc(&data, bytes)); }
  ~Device() { cudaFree(data); }
};
std::vector<B> reference(std::vector<B> values, Type type) {
  float maximum = 0;
  for (auto x : values) maximum = std::max(maximum, std::fabs(__bfloat162float(x)));
  if (maximum == 0) return values;
  if (type == Type::fp8) {
    const float scale = maximum / 448.f;
    for (auto& x : values) x = __float2bfloat16_rn(float(__nv_fp8_e4m3(__bfloat162float(x) / scale)) * scale);
  } else {
    const float global = maximum / (6.f * 448.f);
    for (std::size_t row = 0; row < values.size(); row += 16) {
      float local = 0;
      for (unsigned j = 0; j < 16; ++j) local = std::max(local, std::fabs(__bfloat162float(values[row+j])));
      const float block = local == 0 ? 1.f : float(__nv_fp8_e4m3(std::min(448.f, std::max(0x1p-9f, local/(6.f*global)))));
      const float scale = block * global;
      for (unsigned j = 0; j < 16; ++j)
        values[row+j] = __float2bfloat16_rn(float(__nv_fp4_e2m1(__bfloat162float(values[row+j])/scale)) * scale);
    }
  }
  return values;
}
int main(int argc, char** argv) {
  try {
    require(argc == 3, "expected BF16 and NVIDIA artifacts");
    auto file = ArtifactFile::Open(argv[1]);
    auto packed = ArtifactFile::Open(argv[2]);
    const QdqMask none;
    const auto native = apply_qdq_in_place(none, packed, nullptr);
    require(native.selection.nvfp4_w4a4.tensor_count == 11520 && native.selection.bf16.tensor_count == 205,
            "implicit native summary differs");
    const auto mask = QdqMask::Parse("0 q_proj fp8\n0 gate_proj nvfp4\n0 down_proj fp8\n0 expert_gate_proj nvfp4 0\n29 expert_down_proj fp8 127\n29 k_proj nvfp4\n");
    const auto bytes = file.file_bytes() - kArtifactDataOffset;
    Device device(bytes + 512);
    auto* arena = static_cast<unsigned char*>(device.data) + 256;
    check(cudaMemset(device.data, 0x5a, 256));
    check(cudaMemset(arena + bytes, 0x5a, 256));
    std::set<std::size_t> selected, untouched;
    for (std::size_t id=0; id<kTextTensors.size(); ++id)
      if (mask.type_for(id) != Type::bf16) selected.insert(id);
    for (auto id : selected) {
      untouched.insert(id-1);
      untouched.insert(id+1);
    }
    for (auto id : selected) untouched.erase(id);
    untouched.insert(0); // Tied embedding/head prefix, excluded from QDQ.
    const auto address = [&](std::size_t id) { return arena + file.entries()[id].offset - kArtifactDataOffset; };
    for (auto id : selected) check(cudaMemcpy(address(id),file.tensor_data(id),kTextTensors[id].byte_count(),cudaMemcpyHostToDevice));
    for (auto id : untouched) {
      const auto size = id == 0 ? 256 : kTextTensors[id].byte_count();
      check(cudaMemcpy(address(id),file.tensor_data(id),size,cudaMemcpyHostToDevice));
    }
    bool rejected = false;
    try { apply_qdq_in_place(QdqMask::Parse("0 q_proj fp8\n29 q_proj nvfp4_w4a4\n"),file,arena); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "late storage mismatch accepted");
    for (auto id : selected) {
      std::vector<unsigned char> output(kTextTensors[id].byte_count());
      check(cudaMemcpy(output.data(),address(id),output.size(),cudaMemcpyDeviceToHost));
      require(std::memcmp(output.data(),file.tensor_data(id),output.size()) == 0, "failed validation mutated weights");
    }
    const auto summary = apply_qdq_in_place(mask,file,arena);
    require(summary.selection.fp8.tensor_count == 3 && summary.selection.nvfp4.tensor_count == 3,
            "selected summary differs");
    for (auto id : selected) {
      std::vector<B> input(kTextTensors[id].element_count()), output(input.size());
      std::memcpy(input.data(),file.tensor_data(id),input.size()*2);
      const auto expected = reference(input,mask.type_for(id));
      check(cudaMemcpy(output.data(),address(id),output.size()*2,cudaMemcpyDeviceToHost));
      require(std::memcmp(expected.data(),output.data(),output.size()*2) == 0, "independent QDQ recipe differs");
      require(std::memcmp(input.data(),file.tensor_data(id),input.size()*2) == 0, "source mapping changed");
      std::cout << "tensor " << id << " exact " << output.size() << " BF16 values\n";
    }
    for (auto id : untouched) {
      std::vector<unsigned char> output(id == 0 ? 256 : kTextTensors[id].byte_count());
      check(cudaMemcpy(output.data(),address(id),output.size(),cudaMemcpyDeviceToHost));
      require(std::memcmp(output.data(),file.tensor_data(id),output.size()) == 0, "unselected tensor changed");
    }
    std::vector<unsigned char> guard(256);
    for (auto* location : {static_cast<unsigned char*>(device.data),arena+bytes}) {
      check(cudaMemcpy(guard.data(),location,guard.size(),cudaMemcpyDeviceToHost));
      require(std::all_of(guard.begin(),guard.end(),[](auto x){return x==0x5a;}),"payload guard changed");
    }
    std::cout << "26B QDQ device contract passed\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
