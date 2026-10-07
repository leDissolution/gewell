#include "models/gemma4/26b_a4b/sm120/projection.cuh"
#include "cuda_memory.cuh"
#include <cuda_fp8.h>
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace m=gewell::gemma4_26b_a4b;
namespace s=m::sm120;
namespace fp4=gewell::nvfp4;
using B=__nv_bfloat16;
using Device=gewell::cuda_detail::DeviceAllocation;
using gewell::cuda_detail::check_cuda;
void require(bool value,const char* message) { if (!value) throw std::runtime_error(message); }
void blas(cublasStatus_t result) { require(result==CUBLAS_STATUS_SUCCESS,"cuBLAS failure"); }
struct Handles {
  cudaStream_t stream{};cublasHandle_t blas{};
  Handles() {
    check_cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"stream");
    ::blas(cublasCreate(&blas));::blas(cublasSetStream(blas,stream));
    ::blas(cublasSetPointerMode(blas,CUBLAS_POINTER_MODE_HOST));
    ::blas(cublasSetMathMode(blas,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
  }
  ~Handles() { cublasDestroy(blas);cudaStreamDestroy(stream); }
};
constexpr float levels[]={0,.5f,1,1.5f,2,3,4,6};
constexpr float raw[]={0,.25f,.75f,1.25f,1.75f,2.5f,3.5f,5,6,-.25f,-.75f,-1.25f,-1.75f,-2.5f,-3.5f,-6};
// Independently enumerated E2M1 nearest-even ties for raw[], not device packing.
constexpr float quantized[]={0,0,1,1,2,2,4,4,6,0,-1,-1,-2,-2,-4,-6};
float activation(unsigned row,unsigned column,bool fp4) {
  const float block_scale=std::array<float,3>{.5f,1.f,2.f}[(column/16+row)%3];
  return (fp4?quantized[column%16]:raw[column%16])*.125f*block_scale*(row%2?-1.f:1.f);
}
std::uint8_t code(unsigned row) { return std::uint8_t(1+row%7+(row%2?8:0)); }
float coefficient(unsigned row) { auto c=code(row);return levels[c&7]*(c&8?-1.f:1.f)*.25f; }
unsigned column(unsigned row,unsigned k) { return (row*37+5)%k; }

void test(unsigned n,unsigned k,m::StorageType storage,Handles& handles,unsigned capacity) {
  const unsigned physical=storage==m::StorageType::nvfp4_w4a4?m::align_up(n,128):n;
  const std::size_t packed=std::size_t(physical)*k/(storage==m::StorageType::nvfp4_w4a4?2:1);
  const std::size_t size=storage==m::StorageType::bf16?std::size_t(n)*k*2:
      packed+(storage==m::StorageType::nvfp4_w4a4?fp4::scale_storage_bytes(physical,k):0)+8;
  std::vector<std::uint8_t> data(size);
  for (unsigned row=0;row<n;++row) {
    auto at=std::size_t(row)*k+column(row,k);
    if (storage==m::StorageType::bf16) {
      B value=__float2bfloat16(coefficient(row));std::memcpy(data.data()+at*2,&value,2);
    } else if (storage==m::StorageType::fp8_w8a8)
      data[at]=__nv_cvt_float_to_fp8(coefficient(row)/.25f,__NV_SATFINITE,__NV_E4M3);
    else {
      data[at/2]|=code(row)<<(4*(at%2));
      for (unsigned block=0;block<k/16;++block) data[packed+fp4::scale_offset(row,block,k)]=0x38;
    }
  }
  if (storage!=m::StorageType::bf16) {
    const float globals[]={.25f,.125f};std::memcpy(data.data()+size-8,globals,8);
  }
  Device weights(size);
  check_cuda(cudaMemcpy(weights.data(),data.data(),size,cudaMemcpyHostToDevice),"weights");
  m::TensorSpec spec{m::TensorRole::gate_proj,0,-1,2,n,k};
  m::TensorEntry entry{0,size,{},storage};
  auto projection=s::bind_projection(spec,entry,data.data(),weights.data());
  require(projection.input==k && projection.output==n && projection.physical_output==physical,"binding shape");
  Device input(std::size_t(capacity)*k*2),output(std::size_t(capacity)*n*2+256);
  std::vector<B> x(std::size_t(capacity)*k),y(std::size_t(capacity)*n);
  for (unsigned row=0;row<capacity;++row)
    for (unsigned col=0;col<k;++col) x[std::size_t(row)*k+col]=__float2bfloat16(activation(row,col,false));
  check_cuda(cudaMemcpy(input.data(),x.data(),x.size()*2,cudaMemcpyHostToDevice),"inputs");
  for (auto policy:{fp4::ActivationPolicy::always,fp4::ActivationPolicy::prefill}) {
    s::ProjectionWorkspace workspace({projection},capacity,policy,handles.stream);
    const auto bytes=workspace.bytes();
    if (storage==m::StorageType::bf16) require(bytes==0,"BF16 unnecessary projection scratch");
    for (unsigned rows:{1u,33u,7u,129u,capacity,1u}) {
      for (auto phase:{fp4::Phase::prefill,fp4::Phase::decode}) {
        workspace.prepare(rows,phase);
        check_cuda(cudaMemsetAsync(output.data(),0x5a,output.size(),handles.stream),"output sentinel");
        workspace.run(handles.blas,projection,static_cast<const B*>(input.data()),static_cast<B*>(output.data()));
        check_cuda(cudaMemcpyAsync(y.data(),output.data(),std::size_t(rows)*n*2,cudaMemcpyDeviceToHost,handles.stream),"result");
        std::array<std::uint8_t,256> tail;
        check_cuda(cudaMemcpyAsync(tail.data(),static_cast<const char*>(output.data())+std::size_t(rows)*n*2,
            tail.size(),cudaMemcpyDeviceToHost,handles.stream),"tail");
        check_cuda(cudaStreamSynchronize(handles.stream),"compare");
        const bool fp4=storage==m::StorageType::nvfp4_w4a4 && fp4::fp4_activations(policy,phase);
        for (unsigned row=0;row<rows;++row) for (unsigned col=0;col<n;++col) {
          const B expected=__float2bfloat16(activation(row,column(col,k),fp4)*coefficient(col));
          // Zero sign is immaterial; every nonzero BF16 value must be exact.
          if (float(y[std::size_t(row)*n+col])!=float(expected)) {
            std::cerr<<"N="<<n<<" K="<<k<<" storage="<<int(storage)<<" rows="<<rows<<" fp4="<<fp4
                <<" row="<<row<<" col="<<col<<" got="<<float(y[std::size_t(row)*n+col])<<" expected="<<float(expected)<<'\n';
            throw std::runtime_error("projection numeric mismatch");
          }
        }
        for (auto value:tail) require(value==0x5a,"projection wrote beyond logical output");
        require(workspace.bytes()==bytes,"projection scratch grew during execution");
      }
    }
  }
  std::cout<<"passed N="<<n<<" K="<<k<<" storage="<<int(storage)<<'\n';
}
void mixed_shapes(Handles& handles) {
  const std::array<std::array<unsigned,3>,4> cases{{{704,2816,1},{2112,2816,2},{2816,704,1},{2816,2112,0}}};
  std::vector<std::unique_ptr<Device>> weights;
  std::vector<s::Projection> projections;
  for (const auto& item:cases) {
    const auto n=item[0],k=item[1];
    const auto storage=static_cast<m::StorageType>(item[2]);
    const unsigned physical=storage==m::StorageType::nvfp4_w4a4?m::align_up(n,128):n;
    const std::size_t packed=std::size_t(physical)*k/(storage==m::StorageType::nvfp4_w4a4?2:1);
    const std::size_t size=storage==m::StorageType::bf16?std::size_t(n)*k*2:
        packed+(storage==m::StorageType::nvfp4_w4a4?fp4::scale_storage_bytes(physical,k):0)+8;
    std::vector<std::uint8_t> data(size);
    if (storage==m::StorageType::bf16) {
      const B value=__float2bfloat16(.25f);
      for (std::size_t i=0;i<size;i+=2) std::memcpy(data.data()+i,&value,2);
    } else {
      std::fill_n(data.data(),std::size_t(n)*k/(storage==m::StorageType::nvfp4_w4a4?2:1),
                  storage==m::StorageType::nvfp4_w4a4?0x22:0x38);
      if (storage==m::StorageType::nvfp4_w4a4)
        for (unsigned row=0;row<n;++row) for (unsigned block=0;block<k/16;++block)
          data[packed+fp4::scale_offset(row,block,k)]=0x38;
      const float globals[]={.25f,.125f};std::memcpy(data.data()+size-8,globals,8);
    }
    weights.push_back(std::make_unique<Device>(size));
    check_cuda(cudaMemcpy(weights.back()->data(),data.data(),size,cudaMemcpyHostToDevice),"mixed weights");
    m::TensorSpec spec{m::TensorRole::gate_proj,0,-1,2,n,k};
    m::TensorEntry entry{0,size,{},storage};
    projections.push_back(s::bind_projection(spec,entry,data.data(),weights.back()->data()));
  }
  Device input(33*2816*2),output(33*2816*2+256);
  for (auto policy:{fp4::ActivationPolicy::always,fp4::ActivationPolicy::prefill}) {
    s::ProjectionWorkspace workspace(projections,33,policy,handles.stream);
    const auto bytes=workspace.bytes();
    for (unsigned rows:{3u,33u,3u}) for (auto phase:{fp4::Phase::prefill,fp4::Phase::decode}) {
      workspace.prepare(rows,phase);
      for (const auto& projection:projections) {
        std::vector<B> x(std::size_t(rows)*projection.input),y(std::size_t(rows)*projection.output);
        for (unsigned row=0;row<rows;++row)
          std::fill_n(x.data()+std::size_t(row)*projection.input,projection.input,__float2bfloat16(row%2?1.5f:.75f));
        check_cuda(cudaMemcpyAsync(input.data(),x.data(),x.size()*2,cudaMemcpyHostToDevice,handles.stream),"mixed inputs");
        check_cuda(cudaMemsetAsync(output.data(),0x5a,output.size(),handles.stream),"mixed guard");
        workspace.run(handles.blas,projection,static_cast<const B*>(input.data()),static_cast<B*>(output.data()));
        check_cuda(cudaMemcpyAsync(y.data(),output.data(),y.size()*2,cudaMemcpyDeviceToHost,handles.stream),"mixed result");
        std::array<std::uint8_t,256> tail;
        check_cuda(cudaMemcpyAsync(tail.data(),static_cast<char*>(output.data())+y.size()*2,tail.size(),
                                 cudaMemcpyDeviceToHost,handles.stream),"mixed tail");
        check_cuda(cudaStreamSynchronize(handles.stream),"mixed comparison");
        for (unsigned row=0;row<rows;++row) for (unsigned col=0;col<projection.output;++col)
          require(float(y[std::size_t(row)*projection.output+col])==float(__float2bfloat16(
              (row%2?1.5f:.75f)*.25f*projection.input)),"mixed shape/storage scratch mismatch");
        for (auto value:tail) require(value==0x5a,"mixed output overflow");
        require(workspace.bytes()==bytes,"mixed scratch grew");
      }
    }
  }
  std::cout<<"mixed shape/storage scratch reuse passed\n";
}
int main(int argc,char** argv) {
  try {
    require(argc==1 || (argc==2 && std::string(argv[1])=="--quick"),"usage: projection [--quick]");
    const unsigned capacity=argc==2?129:4096;
    Handles handles;
    for (const auto& shape:std::array<std::array<unsigned,2>,10>{{{704,2816},{2112,2816},{2816,704},
        {2816,2112},{4096,2816},{8192,2816},{2048,2816},{1024,2816},{2816,8192},{2816,4096}}})
      for (auto storage:{m::StorageType::bf16,m::StorageType::fp8_w8a8,m::StorageType::nvfp4_w4a4})
        test(shape[0],shape[1],storage,handles,capacity);
    mixed_shapes(handles);
    std::cout<<"26B projection storage/policy/phase/padding checks passed\n";
  } catch (const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
