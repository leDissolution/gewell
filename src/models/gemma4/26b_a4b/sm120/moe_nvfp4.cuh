#pragma once

#include "moe.cuh"
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <stdexcept>

namespace gewell::gemma4_26b_a4b::sm120 {
namespace grouped_nvfp4 {
// TMA pipelines K128 loads while tensor cores consume earlier stages.
// Problem shapes, pointers and per-projection scales are refreshed on the GPU.
template<bool GateUp>
struct Configuration {
  using Tile = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using Cluster = cute::Shape<cute::_1,cute::_1,cute::_1>;
  using Element = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
  using Output = cutlass::bfloat16_t;
  using Shape = cutlass::gemm::GroupProblemShape<cute::Shape<int,int,int>>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120,cutlass::arch::OpClassTensorOp,Tile,Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,float,float,
      void,cutlass::layout::RowMajor*,8,Output,cutlass::layout::RowMajor*,8,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      cutlass::epilogue::fusion::LinearCombination<Output,float,void,float>>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120,cutlass::arch::OpClassBlockScaledTensorOp,
      Element,cutlass::layout::RowMajor*,32,Element,cutlass::layout::ColumnMajor*,32,
      float,Tile,Cluster,cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)>,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape,Mainloop,Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using Scales = typename Mainloop::Sm1xxBlkScaledConfig;
  struct Arrays {
    typename Shape::UnderlyingProblemShape shapes[256];
    const typename Gemm::ElementA *input[256], *weight[256];
    const typename Mainloop::ElementSF *input_scales[256], *weight_scales[256];
    Output* output[256];
    typename Kernel::InternalStrideA input_stride[256];
    typename Kernel::InternalStrideB weight_stride[256];
    typename Kernel::InternalStrideD output_stride[256];
    typename Mainloop::InternalLayoutSFA input_layout[256];
    typename Mainloop::InternalLayoutSFB weight_layout[256];
    float alpha[256];
    const float* alpha_ptr[256];
  };
};

template<bool GateUp>
__global__ void prepare(const std::uint8_t* input,const std::uint8_t* scales,unsigned n,
                        unsigned scale_rows,const ExpertWeights* weights,__nv_bfloat16* output,
                        const int* offsets,typename Configuration<GateUp>::Arrays* arrays) {
  using C = Configuration<GateUp>;
  const unsigned group=threadIdx.x,expert=GateUp?group/2:group,part=GateUp?group%2:0;
  if (expert>=128) return;
  constexpr unsigned K=GateUp?2816:704,N=GateUp?704:2816,D=GateUp?1408:2816;
  const auto& projection=GateUp?(part?weights[expert].up:weights[expert].gate):weights[expert].down;
  const unsigned first=offsets[expert];
  const unsigned rows=projection.storage==StorageType::nvfp4_w4a4?offsets[expert+1]-first:0;
  arrays->shapes[group]=cute::make_shape(int(rows),int(N),int(K));
  arrays->input[group]=reinterpret_cast<const typename C::Gemm::ElementA*>(input+(std::size_t(part)*n+first)*(K/2));
  arrays->weight[group]=reinterpret_cast<const typename C::Gemm::ElementB*>(projection.nvfp4.data);
  const unsigned padded_first=(first/128+expert)*128;
  arrays->input_scales[group]=reinterpret_cast<const typename C::Mainloop::ElementSF*>(
      scales+(std::size_t(part)*scale_rows+padded_first)*(K/16));
  arrays->weight_scales[group]=reinterpret_cast<const typename C::Mainloop::ElementSF*>(projection.nvfp4.scales);
  arrays->output[group]=reinterpret_cast<typename C::Output*>(output+std::size_t(first)*D+part*N);
  arrays->input_stride[group]=cutlass::make_cute_packed_stride(typename C::Kernel::InternalStrideA{}, {int(rows),int(K),1});
  arrays->weight_stride[group]=cutlass::make_cute_packed_stride(typename C::Kernel::InternalStrideB{}, {int(N),int(K),1});
  arrays->output_stride[group]=cutlass::make_cute_packed_stride(typename C::Kernel::InternalStrideD{}, {int(rows),int(D),1});
  auto shape=cute::make_shape(int(rows),int(N),int(K),1);
  arrays->input_layout[group]=C::Scales::tile_atom_to_shape_SFA(shape);
  arrays->weight_layout[group]=C::Scales::tile_atom_to_shape_SFB(shape);
  arrays->alpha[group]=__fmul_rn(projection.nvfp4.input_scale,projection.nvfp4.weight_scale);
  arrays->alpha_ptr[group]=arrays->alpha+group;
}

template<bool GateUp>
class Plan {
  using C = Configuration<GateUp>;
  using Gemm = typename C::Gemm;
 public:
  explicit Plan(cutlass::KernelHardwareInfo hardware):arrays_(sizeof(typename C::Arrays)) {
    typename Gemm::Arguments args{};
    auto* a=static_cast<typename C::Arrays*>(arrays_.data());
    args.mode=cutlass::gemm::GemmUniversalMode::kGrouped;
    args.problem_shape={GateUp?256:128,a->shapes,nullptr};
    args.mainloop={a->input,a->input_stride,a->weight,a->weight_stride,
        a->input_scales,a->input_layout,a->weight_scales,a->weight_layout};
    args.epilogue.thread.alpha_ptr_array=a->alpha_ptr;
    args.epilogue.thread.dAlpha={cute::_0{},cute::_0{},1};
    args.epilogue.ptr_D=a->output;
    args.epilogue.dD=a->output_stride;
    args.epilogue.dC=a->output_stride;
    args.hw_info=hardware;
    scratch_=std::make_unique<cuda_detail::DeviceAllocation>(Gemm::get_workspace_size(args));
    // The grouped scheduler and linear epilogue have no per-run workspace
    // reset. All changing data lives in arrays_, so encode TMA templates once.
    if (gemm_.initialize(args,scratch_->data())!=cutlass::Status::kSuccess)
      throw std::runtime_error("26B grouped NVFP4 initialization failed");
  }
  void run(const std::uint8_t* input,const std::uint8_t* scales,unsigned n,unsigned scale_rows,
           const ExpertWeights* weights,__nv_bfloat16* output,const int* offsets,cudaStream_t stream) {
    prepare<GateUp><<<1,256,0,stream>>>(input,scales,n,scale_rows,weights,output,offsets,
        static_cast<typename C::Arrays*>(arrays_.data()));
    if (gemm_.run(stream)!=cutlass::Status::kSuccess)
      throw std::runtime_error("26B grouped NVFP4 GEMM failed");
  }
  std::size_t bytes() const { return arrays_.size()+scratch_->size(); }
 private:
  cuda_detail::DeviceAllocation arrays_;
  std::unique_ptr<cuda_detail::DeviceAllocation> scratch_;
  Gemm gemm_;
};
}  // namespace grouped_nvfp4

class GroupedNvfp4 {
  static cutlass::KernelHardwareInfo hardware() {
    cutlass::KernelHardwareInfo result;
    cuda_detail::check_cuda(cudaGetDevice(&result.device_id),"NVFP4 grouped device");
    cuda_detail::check_cuda(cudaDeviceGetAttribute(&result.sm_count,cudaDevAttrMultiProcessorCount,
        result.device_id),"NVFP4 grouped SM count");
    return result;
  }
 public:
  GroupedNvfp4():gate_up(hardware()),down(hardware()) {}
  std::size_t bytes() const { return gate_up.bytes()+down.bytes(); }
  grouped_nvfp4::Plan<true> gate_up;
  grouped_nvfp4::Plan<false> down;
};
}  // namespace gewell::gemma4_26b_a4b::sm120
