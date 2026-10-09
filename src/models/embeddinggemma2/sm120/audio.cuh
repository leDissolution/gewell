// Concrete audio tower; shares verified BF16 arithmetic and cuBLAS ownership.
#include "audio_positions.h"

__global__ void audio_first_convolution(const float* input,const std::uint8_t* mask,const B* weight,B* output,int time) {
  const int i=blockIdx.x*256+threadIdx.x;
  if (i>=((time+1)/2)*64*128) return;
  const int channel=i%128,x=(i/128)%64,t=i/(128*64);
  float sum=0;
  for (int ky=0;ky<3;++ky) for (int kx=0;kx<3;++kx) {
    const int y=t*2+ky-1,col=x*2+kx-1;
    const float value=y>=0 && y<time && col>=0 && col<128 && mask[y]?f(b(input[y*128+col])):0;
    sum+=value*f(weight[channel*9+ky*3+kx]);
  }
  output[i]=b(sum);
}
__global__ void audio_convolution_mask(B* values,const std::uint8_t* mask,int count) {
  const int i=blockIdx.x*256+threadIdx.x;
  if (i<count && !mask[(i/(64*128))*2]) values[i]=b(0.f);
}
__global__ void audio_layer_norm_relu(B* values,const B* weight,int width) {
  B* row=values+blockIdx.x*width;
  const int lane=threadIdx.x;
  float mean=0,variance=0,count=0;
  // The source LayerNorm reduces groups of four with online Welford statistics.
  // Preserve its fused multiply-adds explicitly; other BF16 boundaries remain
  // unfused throughout this translation unit.
  if (lane*4<width) for (int j=0;j<4;++j) {
    const float x=f(row[lane*4+j]),delta=x-mean;
    count+=1;mean=fmaf(delta,1.f/count,mean);
    variance=fmaf(delta,x-mean,variance);
  }
  for (int step=16;step;step/=2) {
    const float other_mean=__shfl_down_sync(0xffffffff,mean,step);
    const float other_variance=__shfl_down_sync(0xffffffff,variance,step);
    const float other_count=__shfl_down_sync(0xffffffff,count,step);
    const float total=count+other_count,delta=mean-other_mean;
    const float inverse=total>0?1.f/total:0;
    const float a=other_count*inverse,b=count*inverse;
    mean=fmaf(a,other_mean,b*mean);
    variance=fmaf((delta*delta)*other_count,b,other_variance+variance);
    count=total;
  }
  mean=__shfl_sync(0xffffffff,mean,0);
  const float inv=rsqrtf(__shfl_sync(0xffffffff,variance,0)/width+1.e-6f);
  if (lane*4<width) for (int j=0;j<4;++j) {
    const int c=lane*4+j;
    row[c]=b(fmaxf(0.f,f(b(f(weight[c])*(inv*(f(row[c])-mean))))));
  }
}
__global__ void audio_clip(const B* input,B* output,const B* minimum,const B* maximum,int count) {
  const int i=blockIdx.x*256+threadIdx.x;
  if (i<count) output[i]=b(fminf(fmaxf(f(input[i]),minimum?f(*minimum):-f(b(1.e10f))),
      maximum?f(*maximum):f(b(1.e10f))));
}
__global__ void audio_clip_transposed(const B* input,B* output,const B* minimum,const B* maximum,int rows,int width) {
  const int i=blockIdx.x*256+threadIdx.x;
  if (i<rows*width) output[(i%width)*rows+i/width]=b(fminf(fmaxf(f(input[i]),f(*minimum)),f(*maximum)));
}
__global__ void audio_silu(const B* input,B* output,int count) {
  const int i=blockIdx.x*256+threadIdx.x;
  if (i<count) {const float x=f(input[i]);output[i]=b(x/(1.f+expf(-x)));}
}
__global__ void audio_glu(const B* input,B* output,int count) {
  const int i=blockIdx.x*256+threadIdx.x;
  if (i<count) {
    const int row=i/1024,c=i%1024;
    const float sigmoid=1.f/(1.f+expf(-f(input[row*2048+1024+c])));
    output[i]=b(f(input[row*2048+c])*sigmoid);
  }
}
__global__ void audio_conv_rms(B* values,const B* weight) {
  // Reproduce the reduction over strided channels after the NCT convolution.
  // Four independent accumulators and a channel-lane reduction match Torch's
  // noncontiguous FP32 mean, while storage can remain row-major here.
  __shared__ float sums[512];
  B* row=values+blockIdx.x*1024;
  float partial[4]={};
  for (int first=threadIdx.x;first<1024;first+=blockDim.x*4)
    for (int j=0;j<4;++j) {
      const int c=first+j*blockDim.x;
      if (c<1024) partial[j]+=f(row[c])*f(row[c]);
    }
  sums[threadIdx.x]=((partial[0]+partial[1])+partial[2])+partial[3];
  __syncthreads();
  for (int step=blockDim.x/2;step;step/=2) {
    if (threadIdx.x<step) sums[threadIdx.x]+=sums[threadIdx.x+step];
    __syncthreads();
  }
  const float inv=rsqrtf(sums[0]*(1.f/1024)+1.e-6f);
  for (int c=threadIdx.x;c<1024;c+=blockDim.x) row[c]=b((f(row[c])*inv)*f(weight[c]));
}
void audio_normalize_convolution(B* values,const B* weight,int rows) {
  if (rows==1) {normalize(values,weight,values,rows,1024);return;}
  int vector=4;
  while (rows%vector) vector/=2;
  const int threads=512/vector;
  int outputs=1;
  while (outputs*2<=rows/vector && outputs<threads) outputs*=2;
  const int lanes=threads/std::min(outputs,32);
  audio_conv_rms<<<rows,lanes>>>(values,weight);
}
__global__ void audio_depthwise(const B* input,const B* weight,B* output,int count) {
  const int i=blockIdx.x*256+threadIdx.x;
  if (i>=count*1024) return;
  const int row=i/1024,c=i%1024;
  float sum=0;
  for (int k=0;k<5;++k) if (row+k>=4) sum+=f(input[(row+k-4)*1024+c])*f(weight[c*5+k]);
  output[i]=b(sum);
}
__global__ void audio_position_embedding(const float* frequencies,B* output) {
  const int i=blockIdx.x*256+threadIdx.x;
  if (i<13*1024) {
    const int row=i/1024,c=i%1024;
    const float angle=f(b(float(12-row)*frequencies[c%512]));
    output[i]=b(c<512?sinf(angle):cosf(angle));
  }
}
__global__ void audio_attention_pack(const B* query,const B* key,const B* value,const B* per_dim,
    float* q,float* original_q,float* k,float* v,int count,int blocks) {
  const int i=blockIdx.x*256+threadIdx.x;
  if (i>=8*blocks*24*128) return;
  const int c=i%128,j=(i/128)%24,block=(i/(128*24))%blocks,head=i/(128*24*blocks);
  const int source=block*12+j-12;
  const int ki=blocks==1?j*1024+head*128+c:((head*blocks+block)*128+c)*24+j;
  const int vi=blocks==1?j*1024+head*128+c:i;
  k[ki]=source>=0 && source<count?f(key[source*1024+head*128+c])*1.8946361239720115f:0;
  v[vi]=source>=0 && source<count?f(value[source*1024+head*128+c]):0;
  if (j<12) {
    const int row=block*12+j;
    const float p=f(per_dim[c]),scale=f(b(p>20?p:log1pf(expf(p))));
    const float value=row<count?
        (f(query[row*1024+head*128+c])*0.1275174308245987f)*scale:0;
    q[((head*blocks+block)*12+j)*128+c]=value;
    original_q[row*1024+head*128+c]=value;
  }
}
__global__ void audio_relative_pack(const B* input,float* output) {
  const int i=blockIdx.x*256+threadIdx.x;
  if (i<13*1024) output[i]=f(input[i]);
}
__global__ void audio_attention_probabilities(float* scores,const float* relative,
    float* mask,int count,int valid,int blocks) {
  const int row=blockIdx.x,lane=threadIdx.x,within=row%12,block=(row/12)%blocks,q=block*12+within;
  const int key=block*12+lane-12,position=lane-within;
  const bool visible=q<count && key>=0 && key<valid && q-key>=0 && q-key<12;
  float value=-INFINITY;
  if (lane<24) {
    const float bias=position>=0 && position<13?relative[row*13+position]:0;
    value=visible?tanhf((scores[row*24+lane]+bias)/50.f)*50.f:-1.e9f;
    if (mask && row<blocks*12 && q<valid) mask[q*24+lane]=visible?1.f:0.f;
  }
  float maximum=value;
  for (int step=16;step;step/=2) maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,step));
  const float exponential=expf(value-maximum);
  float sum=exponential;
  for (int step=16;step;step/=2) sum+=__shfl_xor_sync(0xffffffff,sum,step);
  if (lane<24) scores[row*24+lane]=exponential/sum;
}
__global__ void audio_attention_unpack(const float* input,B* output,int count,int blocks) {
  const int i=blockIdx.x*256+threadIdx.x;
  if (i<count*1024) {
    const int row=i/1024,head=(i%1024)/128,c=i%128;
    output[i]=b(input[((head*blocks+row/12)*12+row%12)*128+c]);
  }
}
__global__ void audio_bias(const B* bias,B* output,int count,int width) {
  const int i=blockIdx.x*256+threadIdx.x;
  if (i<count*width) output[i]=bias[i%width];
}

void audio_cudnn(cudnnStatus_t status) {
  if (status!=CUDNN_STATUS_SUCCESS) throw std::runtime_error(std::string("embeddinggemma2 cuDNN: ")+cudnnGetErrorString(status));
}
// This explicit deterministic algorithm matches the pinned official second
// subsampling convolution. cuBLAS im2col has a different accumulation order.
struct AudioConvolution {
  cudnnHandle_t handle{};
  cudnnTensorDescriptor_t input{},output{};
  cudnnFilterDescriptor_t filter{};
  cudnnConvolutionDescriptor_t conv{};
  static constexpr auto algorithm=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  AudioConvolution() {
    if (cudnnGetVersion()/10000!=9) throw std::runtime_error("EmbeddingGemma 2 audio requires cuDNN 9 runtime");
    try {
      audio_cudnn(cudnnCreate(&handle));
      audio_cudnn(cudnnCreateTensorDescriptor(&input));audio_cudnn(cudnnCreateTensorDescriptor(&output));
      audio_cudnn(cudnnCreateFilterDescriptor(&filter));audio_cudnn(cudnnCreateConvolutionDescriptor(&conv));
      audio_cudnn(cudnnSetFilter4dDescriptor(filter,CUDNN_DATA_BFLOAT16,CUDNN_TENSOR_NCHW,32,128,3,3));
      audio_cudnn(cudnnSetConvolution2dDescriptor(conv,1,1,2,2,1,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT));
      audio_cudnn(cudnnSetConvolutionMathType(conv,CUDNN_TENSOR_OP_MATH));
    } catch (...) {release();throw;}
  }
  ~AudioConvolution() {release();}
  void release() {
    if (conv) cudnnDestroyConvolutionDescriptor(conv);
    if (filter) cudnnDestroyFilterDescriptor(filter);
    if (output) cudnnDestroyTensorDescriptor(output);
    if (input) cudnnDestroyTensorDescriptor(input);
    if (handle) cudnnDestroy(handle);
  }
  std::size_t configure(int rows) {
    audio_cudnn(cudnnSetTensor4dDescriptor(input,CUDNN_TENSOR_NHWC,CUDNN_DATA_BFLOAT16,1,128,rows,64));
    audio_cudnn(cudnnSetTensor4dDescriptor(output,CUDNN_TENSOR_NHWC,CUDNN_DATA_BFLOAT16,1,32,(rows+1)/2,32));
    std::size_t bytes=0;
    audio_cudnn(cudnnGetConvolutionForwardWorkspaceSize(handle,input,filter,conv,output,algorithm,&bytes));
    return bytes;
  }
  void run(const B* x,const B* weights,B* y,int rows,Device& workspace) {
    if (configure(rows)>workspace.size()) throw std::runtime_error("audio convolution exceeds reserved workspace");
    const float alpha=1,beta=0;
    audio_cudnn(cudnnConvolutionForward(handle,&alpha,input,x,filter,weights,conv,algorithm,
        workspace.data(),workspace.size(),&beta,output,y));
  }
};

struct AudioEncoder {
  static constexpr int H=audio::kHidden, M=audio::kIntermediate, O=audio::kOutputWidth;
  const int frames, capacity, max_blocks;
  Blas& handle;
  AudioConvolution convolution_engine;
  std::unique_ptr<Device> weights, convolution_workspace;
  Device scratch, convolution, input_features, input_mask, attention_scratch, frequencies;
  std::vector<const B*> w;
  B *hidden,*branch,*norm,*q,*k,*v,*context,*gate,*activated,*clipped,*projected,*features,*position,*relative;
  float *aq,*ak,*av,*ac,*scores,*relative_scores,*relative_keys,*mask;
  static std::size_t scratch_elements(int n) {return std::size_t(n)*(19*H+O+kHidden)+2*13*H;}
  static std::size_t attention_elements(int n) {
    const int blocks=(n+11)/12;
    return std::size_t(blocks)*(72*H+8*12*(24+13))+13*H+std::size_t(n)*24;
  }
  explicit AudioEncoder(const std::string& directory,int max_tokens,Blas& blas)
      : frames(audio::feature_rows(audio::max_samples(std::max(5,max_tokens)))),
        capacity((frames+3)/4),max_blocks((capacity+11)/12),handle(blas),
        scratch(scratch_elements(capacity)*sizeof(B)),
        convolution(std::size_t((frames+1)/2)*64*128*sizeof(B)),
        input_features(std::size_t(frames)*128*sizeof(float)),input_mask(frames),
        attention_scratch(attention_elements(capacity)*sizeof(float)),frequencies(sizeof(kAudioFrequencies)) {
    audio::validate_bundle_config(directory);
    component::File file(directory+"/audio.safetensors",audio::tensor_specs());
    weights=std::make_unique<Device>(file.device_bytes());w.resize(audio::kTensorCount);
    std::size_t offset=0;
    for (const auto& tensor:file.tensors()) {
      auto* destination=static_cast<std::uint8_t*>(weights->data())+offset;
      check_cuda(cudaMemcpy(destination,tensor.data,tensor.bytes,cudaMemcpyHostToDevice),"copy audio weights");
      w[tensor.physical_id]=reinterpret_cast<B*>(destination);
      offset+=(tensor.bytes+4095)/4096*4096;
    }
    // Query every admitted physical shape at startup, so cuDNN cannot request
    // an unreserved workspace size while serving. No model execution is needed.
    std::size_t convolution_bytes=0;
    for (int rows=1;rows<=(frames+1)/2;++rows)
      convolution_bytes=std::max(convolution_bytes,convolution_engine.configure(rows));
    convolution_workspace=std::make_unique<Device>(convolution_bytes);
    B* cursor=static_cast<B*>(scratch.data());
    auto take=[&](int width){B* pointer=cursor;cursor+=std::size_t(capacity)*width;return pointer;};
    hidden=take(H);branch=take(H);norm=take(H);q=take(H);k=take(H);v=take(H);context=take(H);
    gate=take(M);activated=take(M);clipped=take(M);projected=take(O);features=take(kHidden);
    position=cursor;relative=position+13*H;
    float* fc=static_cast<float*>(attention_scratch.data());
    auto ftake=[&](std::size_t count){float* pointer=fc;fc+=count;return pointer;};
    aq=ftake(std::size_t(max_blocks)*12*H);ak=ftake(std::size_t(max_blocks)*24*H);
    av=ftake(std::size_t(max_blocks)*24*H);ac=ftake(std::size_t(max_blocks)*12*H);
    scores=ftake(std::size_t(max_blocks)*8*12*24);relative_scores=ftake(std::size_t(max_blocks)*8*12*13);
    relative_keys=ftake(13*H);mask=ftake(std::size_t(capacity)*24);
    check_cuda(cudaMemcpy(frequencies.data(),kAudioFrequencies,sizeof(kAudioFrequencies),cudaMemcpyHostToDevice),"copy audio frequencies");
    audio_position_embedding<<<(13*H+255)/256,256>>>(static_cast<float*>(frequencies.data()),position);
    check_cuda(cudaGetLastError(),"audio position kernel");
  }
  std::size_t scratch_bytes() const {
    return scratch.size()+convolution_workspace->size()+convolution.size()+input_features.size()+input_mask.size()+attention_scratch.size()+frequencies.size();
  }
  void linear(const B* input,const B* weight,B* output,int rows,int out,int in=H,bool bias=false) {
    const float alpha=1,beta=bias?1:0;
    blas(cublasGemmEx(handle.handle,CUBLAS_OP_T,CUBLAS_OP_N,out,rows,in,
        &alpha,weight,CUDA_R_16BF,in,input,CUDA_R_16BF,in,&beta,output,CUDA_R_16BF,out,
        CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
  }
  void capture(const Capture& cap,const std::string& name,const B* data,std::vector<int> shape) {
    if (!cap) return;
    const auto count=std::accumulate(shape.begin(),shape.end(),std::size_t(1),std::multiplies<>());
    std::vector<B> raw(count);std::vector<float> values(count);
    check_cuda(cudaMemcpy(raw.data(),data,count*sizeof(B),cudaMemcpyDeviceToHost),"read audio capture");
    std::transform(raw.begin(),raw.end(),values.begin(),[](B x){return __bfloat162float(x);});
    cap(name,shape,values);
  }
  std::vector<float> download(const float* data,std::size_t count) {
    std::vector<float> values(count);
    check_cuda(cudaMemcpy(values.data(),data,count*sizeof(float),cudaMemcpyDeviceToHost),"read audio FP32 capture");
    return values;
  }
  void upload(const std::vector<float>& input,B* output) {
    std::vector<B> raw(input.size());
    for (std::size_t i=0;i<input.size();++i) {
      if (!std::isfinite(input[i])) throw std::invalid_argument("nonfinite audio control input");
      raw[i]=__float2bfloat16_rn(input[i]);
    }
    check_cuda(cudaMemcpy(output,raw.data(),raw.size()*sizeof(B),cudaMemcpyHostToDevice),"upload audio control");
  }
  void clipped_linear(const B* input,int id,B* output,int n,int out,int in,
      const Capture& cap,const std::string& name,int valid,bool transposed=false) {
    if (transposed) {
      // The official depthwise convolution returns NCT storage. Its following
      // RMS/SiLU preserve that layout, so linear_end sees a column-major input.
      audio_clip_transposed<<<(n*in+255)/256,256>>>(input,clipped,w[id+1],w[id+2],n,in);
      const float alpha=1,beta=0;
      blas(cublasGemmEx(handle.handle,CUBLAS_OP_T,CUBLAS_OP_T,out,n,in,
          &alpha,w[id],CUDA_R_16BF,in,clipped,CUDA_R_16BF,n,&beta,output,CUDA_R_16BF,out,
          CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
    } else {
      audio_clip<<<(n*in+255)/256,256>>>(input,clipped,w[id+1],w[id+2],n*in);
      linear(clipped,w[id],output,n,out,in);
    }
    capture(cap,name+".linear",output,{valid,out});
    audio_clip<<<(n*out+255)/256,256>>>(output,output,w[id+3],w[id+4],n*out);
  }
  void feedforward(int n,int valid,int index,bool second,const Capture& cap) {
    const int id=audio::weight_id(index,second?audio::FF2PreNorm:audio::FF1PreNorm);
    const auto prefix="layer."+std::to_string(index)+(second?".feed_forward2.":".feed_forward1.");
    audio_clip<<<(n*H+255)/256,256>>>(hidden,norm,nullptr,nullptr,n*H);
    normalize(norm,w[id],norm,n,H);capture(cap,prefix+"pre_layer_norm",norm,{valid,H});
    clipped_linear(norm,id+1,gate,n,M,H,cap,prefix+"ffw_layer_1",valid);
    audio_silu<<<(n*M+255)/256,256>>>(gate,activated,n*M);
    capture(cap,prefix+"act_fn",activated,{valid,M});
    clipped_linear(activated,id+6,branch,n,H,M,cap,prefix+"ffw_layer_2",valid);
    audio_clip<<<(n*H+255)/256,256>>>(branch,branch,nullptr,nullptr,n*H);
    normalize(branch,w[id+11],branch,n,H);capture(cap,prefix+"post_layer_norm",branch,{valid,H});
    scale<<<(n*H+255)/256,256>>>(branch,nullptr,.5f,n*H);
    add<<<(n*H+255)/256,256>>>(hidden,branch,hidden,n*H);
    capture(cap,prefix+"output",hidden,{valid,H});
  }
  void attention(int n,int valid,int index,const Capture& cap) {
    const int blocks=(n+11)/12;
    const auto prefix="layer."+std::to_string(index)+".";
    auto weight=[&](audio::LayerWeight role){return w[audio::weight_id(index,role)];};
    clipped_linear(norm,audio::weight_id(index,audio::Query),q,n,H,H,cap,prefix+"self_attn.q_proj",valid);
    clipped_linear(norm,audio::weight_id(index,audio::Key),k,n,H,H,cap,prefix+"self_attn.k_proj",valid);
    clipped_linear(norm,audio::weight_id(index,audio::Value),v,n,H,H,cap,prefix+"self_attn.v_proj",valid);
    audio_attention_pack<<<(8*blocks*24*128+255)/256,256>>>(q,k,v,weight(audio::PerDimScale),aq,ac,ak,av,n,blocks);
    linear(position,weight(audio::RelativeKey),relative,13,H);
    capture(cap,prefix+"relative_k",relative,{1,13,H});
    audio_relative_pack<<<(13*H+255)/256,256>>>(relative,relative_keys);
    const float alpha=1,beta=0;
    // Match the concrete strides used by the official matmul reshapes. The
    // head/block flattening copies transposed keys only when blocks > 1.
    blas(cublasSgemmStridedBatched(handle.handle,blocks==1?CUBLAS_OP_T:CUBLAS_OP_N,CUBLAS_OP_N,24,12,128,
        &alpha,ak,blocks==1?H:24,blocks==1?128:24*128,
        blocks==1?ac:aq,blocks==1?H:128,blocks==1?128:12*128,&beta,scores,24,12*24,8*blocks));
    // Relative attention flattens the query sequence, retaining interleaved
    // heads; it is one matrix per head rather than one matrix per chunk.
    blas(cublasSgemmStridedBatched(handle.handle,CUBLAS_OP_T,CUBLAS_OP_N,13,blocks*12,128,
        &alpha,relative_keys,H,128,ac,H,128,&beta,relative_scores,13,blocks*12*13,8));
    audio_attention_probabilities<<<8*blocks*12,32>>>(scores,relative_scores,cap?mask:nullptr,n,valid,blocks);
    if (cap) {
      cap(prefix+"mask",{valid,24},download(mask,valid*24));
      const auto raw=download(scores,8*blocks*12*24);
      std::vector<float> probabilities(std::size_t(valid)*8*24);
      for (int row=0;row<valid;++row) for (int head=0;head<8;++head)
        std::copy_n(raw.data()+(head*blocks*12+row)*24,24,probabilities.data()+(row*8+head)*24);
      cap(prefix+"attention_probabilities",{valid,8,24},probabilities);
    }
    blas(cublasSgemmStridedBatched(handle.handle,CUBLAS_OP_N,CUBLAS_OP_N,128,12,24,
        &alpha,av,blocks==1?H:128,blocks==1?128:24*128,scores,24,12*24,&beta,ac,128,12*128,8*blocks));
    audio_attention_unpack<<<(n*H+255)/256,256>>>(ac,context,n,blocks);
    capture(cap,prefix+"context",context,{valid,H});
    clipped_linear(context,audio::weight_id(index,audio::AttentionPost),branch,n,H,H,cap,prefix+"self_attn.post",valid);
    capture(cap,prefix+"attention_output",branch,{valid,H});
  }
  void layer(int n,int valid,int index,const Capture& capture_all,bool control=false) {
    const auto prefix="layer."+std::to_string(index)+".";
    const Capture cap=(control || index==0 || index==11)?capture_all:Capture{};
    auto weight=[&](audio::LayerWeight role){return w[audio::weight_id(index,role)];};
    capture(cap,prefix+"input",hidden,{valid,H});
    feedforward(n,valid,index,false,cap);
    audio_clip<<<(n*H+255)/256,256>>>(hidden,norm,nullptr,nullptr,n*H);
    normalize(norm,weight(audio::AttentionPreNorm),norm,n,H);capture(cap,prefix+"norm_pre_attn",norm,{valid,H});
    attention(n,valid,index,cap);
    audio_clip<<<(n*H+255)/256,256>>>(branch,branch,nullptr,nullptr,n*H);
    normalize(branch,weight(audio::AttentionPostNorm),branch,n,H);capture(cap,prefix+"norm_post_attn",branch,{valid,H});
    add<<<(n*H+255)/256,256>>>(hidden,branch,hidden,n*H);
    normalize(hidden,weight(audio::ConvPreNorm),norm,n,H);capture(cap,prefix+"lconv1d.pre_layer_norm",norm,{valid,H});
    clipped_linear(norm,audio::weight_id(index,audio::ConvStart),gate,n,2*H,H,cap,prefix+"lconv1d.linear_start",valid);
    audio_glu<<<(n*H+255)/256,256>>>(gate,activated,n*H);
    audio_depthwise<<<(n*H+255)/256,256>>>(activated,weight(audio::Depthwise),branch,n);
    capture(cap,prefix+"lconv1d.depthwise_conv1d",branch,{valid,H});
    audio_clip<<<(n*H+255)/256,256>>>(branch,branch,nullptr,nullptr,n*H);
    audio_normalize_convolution(branch,weight(audio::ConvNorm),n);capture(cap,prefix+"lconv1d.conv_norm",branch,{valid,H});
    audio_silu<<<(n*H+255)/256,256>>>(branch,activated,n*H);capture(cap,prefix+"lconv1d.act_fn",activated,{valid,H});
    clipped_linear(activated,audio::weight_id(index,audio::ConvEnd),branch,n,H,H,cap,prefix+"lconv1d.linear_end",valid,true);
    add<<<(n*H+255)/256,256>>>(hidden,branch,hidden,n*H);capture(cap,prefix+"lconv1d.output",hidden,{valid,H});
    feedforward(n,valid,index,true,cap);
    audio_clip<<<(n*H+255)/256,256>>>(hidden,hidden,nullptr,nullptr,n*H);
    normalize(hidden,weight(audio::OutputNorm),hidden,n,H);capture(cap,prefix+"norm_out",hidden,{valid,H});
    capture(capture_all,prefix+"output",hidden,{valid,H});
    check_cuda(cudaGetLastError(),"audio layer kernels");
  }
  void subsample(const audio::Input& input,const Capture& cap,const std::function<void()>& poll) {
    const int frows=input.mask.size(),valid=input.end-input.begin;
    if (frows>frames) throw std::invalid_argument("audio features exceed configured capacity");
    check_cuda(cudaMemcpy(input_features.data(),input.features.data(),input.features.size()*sizeof(float),cudaMemcpyHostToDevice),"copy audio features");
    check_cuda(cudaMemcpy(input_mask.data(),input.mask.data(),input.mask.size(),cudaMemcpyHostToDevice),"copy audio mask");
    if (cap) {
      cap("feature_input",{frows,128},input.features);
      cap("input.mask",{frows},std::vector<float>(input.mask.begin(),input.mask.end()));
    }
    auto* conv=static_cast<B*>(convolution.data());
    const auto* mask_data=static_cast<std::uint8_t*>(input_mask.data());
    int time=frows,frequency=128;
    for (int i=0;i<2;++i) {
      poll();
      const int ot=(time+1)/2,of=(frequency+1)/2,oc=i?32:128;
      if (i) {
        audio_convolution_mask<<<(time*64*128+255)/256,256>>>(conv,mask_data,time*64*128);
        convolution_engine.run(conv,w[audio::Conv1],branch,time,*convolution_workspace);
        conv=branch;
      } else audio_first_convolution<<<(ot*of*oc+255)/256,256>>>(
          static_cast<float*>(input_features.data()),mask_data,w[audio::Conv0],conv,time);
      const auto prefix="subsample."+std::to_string(i)+".";
      capture(cap,prefix+"conv",conv,{ot,of,oc});
      audio_layer_norm_relu<<<ot*of,32>>>(conv,w[i?audio::Conv1Norm:audio::Conv0Norm],oc);
      capture(cap,prefix+"output",conv,{ot,of,oc});
      if (cap) {
        std::vector<float> mask_values(ot);
        for (int row=0;row<ot;++row) mask_values[row]=input.mask[row*(2<<i)];
        cap(prefix+"mask",{ot},mask_values);
      }
      time=ot;frequency=of;
    }
    linear(conv,w[audio::InputProjection],hidden,time,H);
    capture(cap,"subsample.projected",hidden,{valid,H});
    check_cuda(cudaGetLastError(),"audio subsample kernels");
  }
  void bridge(int n,int valid,const Capture& cap) {
    normalize(projected,nullptr,clipped,n,O);capture(cap,"bridge_norm",clipped,{valid,O});
    linear(clipped,w[audio::Bridge],features,n,kHidden,O);
    capture(cap,"features",features,{valid,kHidden});
  }
  void run(const audio::Input& input,B* output,const Capture& cap,const std::function<void()>& poll) {
    subsample(input,cap,poll);
    poll();
    const int n=(input.mask.size()+3)/4,valid=input.end-input.begin;
    capture(cap,"position_embedding",position,{1,13,H});
    for (int index=0;index<audio::kLayers;++index) {poll();layer(n,valid,index,cap);}
    poll();
    audio_bias<<<(n*O+255)/256,256>>>(w[audio::OutputBias],projected,n,O);
    linear(hidden,w[audio::OutputProjection],projected,n,O,H,true);
    capture(cap,"output_projection",projected,{valid,O});
    if (cap) cap("output.mask",{valid},std::vector<float>(valid,1));
    bridge(n,valid,cap);
    check_cuda(cudaMemcpy(output,features,std::size_t(valid)*kHidden*sizeof(B),cudaMemcpyDeviceToDevice),"insert audio features");
  }
  void control_subsample(const audio::Input& input,const Capture& cap) {
    audio::validate_features(input);subsample(input,cap,[]{});
    check_cuda(cudaDeviceSynchronize(),"audio subsample control completion");
  }
  void control_layer(int index,const std::vector<float>& input,const Capture& cap) {
    if (index<0 || index>=audio::kLayers || input.empty() || input.size()%H || input.size()/H>std::size_t(capacity))
      throw std::invalid_argument("invalid audio layer control shape");
    const int n=input.size()/H;
    upload(input,hidden);layer(n,n,index,cap,true);
    check_cuda(cudaDeviceSynchronize(),"audio layer control completion");
  }
  void control_bridge(const std::vector<float>& input,const Capture& cap) {
    if (input.empty() || input.size()%O || input.size()/O>std::size_t(capacity))
      throw std::invalid_argument("invalid audio bridge control shape");
    const int n=input.size()/O;
    upload(input,projected);bridge(n,n,cap);
    check_cuda(cudaDeviceSynchronize(),"audio bridge control completion");
  }
};
