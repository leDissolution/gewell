// Concrete EmbeddingGemma 2 vision tower. Included inside encoder.cu's private
// namespace to share the verified BF16 arithmetic and cuBLAS primitives.
__global__ void image_pixels(const std::uint8_t* pixels,B* output,int count) {
  const int i=blockIdx.x*256+threadIdx.x;
  if (i<count) output[i]=b(2.f*(__fmul_rn(pixels[i],1.f/255.f)-.5f));
}
__global__ void image_position(B* hidden,const B* table,const int* positions,int count) {
  const int i=blockIdx.x*256+threadIdx.x;
  if (i<count*vision::kHidden) {
    const int row=i/vision::kHidden, c=i%vision::kHidden;
    const B x=table[positions[2*row]*vision::kHidden+c];
    const B y=table[(vision::kPositionCount+positions[2*row+1])*vision::kHidden+c];
    hidden[i]=b(f(hidden[i])+f(b(f(x)+f(y))));
  }
}
__global__ void image_rope(const B* input,B* output,const int* positions,const float* freq,int count) {
  const int i=blockIdx.x*256+threadIdx.x;
  if (i>=count*vision::kHidden) return;
  const int row=i/vision::kHidden, channel=i%vision::kHeadDim, axis=channel/32, offset=channel%32;
  const float angle=positions[2*row+axis]*freq[offset%16];
  const B cosine=b(cosf(angle)), sine=b(sinf(angle));
  const float rotated=offset<16 ? -f(input[i+16]) : f(input[i-16]);
  output[i]=b(f(b(f(input[i])*f(cosine)))+f(b(rotated*f(sine))));
}
__global__ void image_pool(const B* input,B* output,float* scaled,int count,int grid_width) {
  const int i=blockIdx.x*256+threadIdx.x;
  if (i>=count*vision::kHidden) return;
  const int token=i/vision::kHidden,c=i%vision::kHidden, width=grid_width/3;
  const int first=(token/width*3)*grid_width+token%width*3;
  float sum=0;
  for (int y=0;y<3;++y) for (int x=0;x<3;++x)
    sum+=f(input[(first+y*grid_width+x)*vision::kHidden+c])*(1.f/9.f);
  const float value=f(b(sum))*27.712812921102035f;
  scaled[i]=value; output[i]=b(value);
}

struct VisionEncoder {
  static constexpr int H=vision::kHidden, M=vision::kIntermediate, N=vision::kMaxPatches;
  Blas& handle;
  std::unique_ptr<Device> weights;
  // Frequencies persist across requests; all other working storage is borrowed
  // from the encoder's shared text/vision arena.
  Device frequencies, starts;
  // Concatenated images in the current forward; attention stays per image.
  int images=1, longest=0;
  std::uint8_t* pixels; float* pooled_float;
  int* positions;
  std::vector<const B*> w;
  B *hidden,*branch,*norm,*q,*k,*v,*qr,*kr,*vn,*context,*gate,*up,*activated,*pooled,*features;
  // Per patch: hidden, branch, norm, then one phase-shared region holding the
  // uint8 pixels before layer 0, attention or MLP buffers within a layer, and
  // pooling/bridge buffers after the last layer.
  static constexpr std::size_t scratch_elements() {
    return std::size_t(N)*(3*H+2*M);
  }
  static_assert(7*H<=2*M);
  static constexpr std::size_t workspace_bytes() {
    return scratch_elements()*sizeof(B)+N*2*sizeof(int);
  }
  explicit VisionEncoder(const std::string& directory,B* storage,Blas& blas)
      : handle(blas),frequencies(16*sizeof(float)), starts((vision::kMaxSoftTokens+1)*sizeof(int)) {
    vision::validate_bundle_config(directory);
    component::File file(directory+"/vision.safetensors",vision::tensor_specs());
    weights=std::make_unique<Device>(file.device_bytes()); w.resize(vision::kTensorCount);
    std::size_t offset=0;
    for (const auto& tensor:file.tensors()) {
      auto* destination=static_cast<std::uint8_t*>(weights->data())+offset;
      check_cuda(cudaMemcpy(destination,tensor.data,tensor.bytes,cudaMemcpyHostToDevice),"copy vision weights");
      w[tensor.physical_id]=reinterpret_cast<B*>(destination);
      offset+=(tensor.bytes+4095)/4096*4096;
    }
    B* cursor=storage;
    auto take=[&](int width){B* pointer=cursor;cursor+=std::size_t(N)*width;return pointer;};
    hidden=take(H); branch=take(H); norm=take(H);
    B* const phase=cursor;
    q=take(H); k=take(H); v=take(H); qr=take(H); kr=take(H); vn=take(H); context=take(H);
    cursor=phase;
    gate=take(M); up=take(M); activated=gate;
    pixels=reinterpret_cast<std::uint8_t*>(phase);
    pooled=phase; features=pooled+vision::kMaxSoftTokens*H;
    pooled_float=reinterpret_cast<float*>(features+vision::kMaxSoftTokens*kHidden);
    positions=reinterpret_cast<int*>(storage+scratch_elements());
    // Exact CPU FP32 values from the pinned axial RoPE initializer (theta=100).
    constexpr float freq[]={0x1p+0f,0x1.7ff222p-1f,0x1.1feb34p-1f,0x1.afd136p-2f,
      0x1.43d136p-2f,0x1.e5a846p-3f,0x1.6c310ep-3f,0x1.111aeep-3f,
      0x1.99999ap-4f,0x1.33281ap-4f,0x1.ccab84p-5f,0x1.59742ap-5f,
      0x1.030dc6p-5f,0x1.84869ep-6f,0x1.235a72p-6f,0x1.b4f7e4p-7f};
    check_cuda(cudaMemcpy(frequencies.data(),freq,sizeof(freq),cudaMemcpyHostToDevice),"copy vision frequencies");
  }
  std::size_t scratch_bytes() const {
    return workspace_bytes()+frequencies.size()+starts.size();
  }
  void linear(const B* input,const B* weight,B* output,int rows,int out,int in=H) {
    const float alpha=1,beta=0;
    blas(cublasGemmEx(handle.handle,CUBLAS_OP_T,CUBLAS_OP_N,out,rows,in,
        &alpha,weight,CUDA_R_16BF,in,input,CUDA_R_16BF,in,&beta,output,CUDA_R_16BF,out,
        CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT));
  }
  void capture(const Capture& cap,const std::string& name,const B* data,int rows,int width) {
    if (!cap) return;
    std::vector<B> raw(std::size_t(rows)*width);
    check_cuda(cudaMemcpy(raw.data(),data,raw.size()*sizeof(B),cudaMemcpyDeviceToHost),"read vision capture");
    std::vector<float> values(raw.size());
    std::transform(raw.begin(),raw.end(),values.begin(),[](B x){return __bfloat162float(x);});
    cap(name,{rows,width},values);
  }
  void upload(const std::vector<float>& input,B* destination) {
    std::vector<B> raw(input.size());
    for (std::size_t i=0;i<input.size();++i) {
      if (!std::isfinite(input[i])) throw std::invalid_argument("nonfinite vision control input");
      raw[i]=__float2bfloat16_rn(input[i]);
    }
    check_cuda(cudaMemcpy(destination,raw.data(),raw.size()*sizeof(B),cudaMemcpyHostToDevice),"upload vision control");
  }
  void layer(int n,int index,const Capture& cap) {
    const int h=n*H;
    const auto prefix="layer."+std::to_string(index)+".";
    const bool deep=index==0 || index==15;
    auto weight=[&](vision::LayerWeight role){return w[vision::weight_id(index,role)];};
    auto save=[&](const std::string& name,const B* data){if (deep) capture(cap,prefix+name,data,n,H);};
    save("input",hidden);
    normalize(hidden,weight(vision::InputNorm),norm,n,H); save("input_norm",norm);
    linear(norm,weight(vision::Query),q,n,H); linear(norm,weight(vision::Key),k,n,H); linear(norm,weight(vision::Value),v,n,H);
    save("q",q); save("k",k); save("v",v);
    normalize(q,weight(vision::QueryNorm),q,n*vision::kHeads,vision::kHeadDim);
    normalize(k,weight(vision::KeyNorm),k,n*vision::kHeads,vision::kHeadDim);
    normalize(v,nullptr,vn,n*vision::kHeads,vision::kHeadDim);
    save("q_norm",q); save("k_norm",k); save("v_norm",vn);
    image_rope<<<(h+255)/256,256>>>(q,qr,positions,static_cast<float*>(frequencies.data()),n);
    image_rope<<<(h+255)/256,256>>>(k,kr,positions,static_cast<float*>(frequencies.data()),n);
    save("q_rope",qr); save("k_rope",kr);
    if (images>1) vision_attention(qr,kr,vn,context,longest,static_cast<const int*>(starts.data()),images);
    else vision_attention(qr,kr,vn,context,n);
    if (deep && cap && n<=1260) cap(prefix+"mask",{n,n},std::vector<float>(std::size_t(n)*n,1.f));
    save("context",context);
    linear(context,weight(vision::AttentionOutput),branch,n,H);
    normalize(branch,weight(vision::PostAttentionNorm),hidden,n,H,hidden);
    normalize(hidden,weight(vision::PreFeedforwardNorm),norm,n,H);
    linear(norm,weight(vision::Gate),gate,n,M); linear(norm,weight(vision::Up),up,n,M);
    gelu_product<<<(n*M+255)/256,256>>>(gate,up,activated,n*M);
    linear(activated,weight(vision::Down),branch,n,H,M);
    normalize(branch,weight(vision::PostFeedforwardNorm),hidden,n,H,hidden);
    capture(cap,prefix+"output",hidden,n,H);
    check_cuda(cudaGetLastError(),"vision layer kernels");
  }
  void bridge(int count,B* output,const Capture& cap) {
    capture(cap,"bridge_input",pooled,count,H);
    normalize(pooled,nullptr,norm,count,H); capture(cap,"bridge_norm",norm,count,H);
    linear(norm,w[vision::BridgeProjection],output,count,kHidden);
    capture(cap,"features",output,count,kHidden);
  }
  // Images are concatenated row-wise (at most kMaxPatches rows in total) and
  // share every projection; outputs[i] receives image i's bridge features.
  void run(const std::vector<const runtime::ImageInput*>& batch,const std::vector<B*>& outputs,
      const Capture& cap,const std::function<void()>& poll) {
    if (batch.empty() || batch.size()!=outputs.size() || (cap && batch.size()>1))
      throw std::invalid_argument("invalid embeddinggemma2 vision batch");
    std::vector<int> offsets{0};
    for (const auto* image:batch) offsets.push_back(offsets.back()+(image->end-image->begin)*9);
    const int n=offsets.back(), tokens=n/9;
    if (n>N) throw std::invalid_argument("embeddinggemma2 vision batch exceeds capacity");
    images=batch.size(); longest=0;
    for (std::size_t i=0;i<batch.size();++i) {
      const auto& image=*batch[i];
      const int rows=offsets[i+1]-offsets[i];
      longest=std::max(longest,rows);
      check_cuda(cudaMemcpy(pixels+std::size_t(offsets[i])*vision::kPatchWidth,image.pixels.data(),
          std::size_t(rows)*vision::kPatchWidth,cudaMemcpyHostToDevice),"copy image pixels");
      check_cuda(cudaMemcpy(positions+offsets[i]*2,image.positions.data(),rows*2*sizeof(int),cudaMemcpyHostToDevice),"copy image positions");
    }
    if (images>1) check_cuda(cudaMemcpy(starts.data(),offsets.data(),offsets.size()*sizeof(int),cudaMemcpyHostToDevice),"copy image offsets");
    if (cap) {
      const auto& image=*batch[0];
      std::vector<float> values(image.pixels.begin(),image.pixels.end());
      for (float& value:values) value*=1.f/255.f;
      cap("pixels",{static_cast<int>(image.padded_patch_rows),vision::kPatchWidth},values);
      values.resize(image.positions.size()/sizeof(int));
      for (std::size_t i=0;i<values.size();++i) {int position;std::memcpy(&position,image.positions.data()+i*sizeof(int),sizeof(int));values[i]=position;}
      cap("positions",{static_cast<int>(image.padded_patch_rows),2},values);
    }
    image_pixels<<<(n*vision::kPatchWidth+255)/256,256>>>(pixels,norm,n*vision::kPatchWidth);
    linear(norm,w[vision::PatchProjection],hidden,n,H,vision::kPatchWidth);
    image_position<<<(n*H+255)/256,256>>>(hidden,w[vision::PositionEmbedding],positions,n);
    capture(cap,"patch_embedding",hidden,n,H);
    for (int i=0;i<vision::kLayers;++i) {poll();layer(n,i,cap);}
    poll();
    for (std::size_t i=0;i<batch.size();++i) {
      const int rows=offsets[i+1]-offsets[i],count=rows/9;
      int last_x;std::memcpy(&last_x,batch[i]->positions.data()+(rows-1)*2*sizeof(int),sizeof(int));
      const int grid_width=last_x+1;
      if (cap && rows<=1260) {
        std::vector<float> membership(std::size_t(rows)*count,0);
        for (int r=0;r<rows;++r) membership[std::size_t(r)*count+(r%grid_width)/3+(grid_width/3)*(r/grid_width/3)]=1;
        cap("pool_membership",{rows,count},membership);
      }
      const int token=offsets[i]/9;
      image_pool<<<(count*H+255)/256,256>>>(hidden+std::size_t(offsets[i])*H,pooled+std::size_t(token)*H,
          pooled_float+std::size_t(token)*H,count,grid_width);
    }
    if (cap) {
      std::vector<float> values(tokens*H);
      check_cuda(cudaMemcpy(values.data(),pooled_float,values.size()*sizeof(float),cudaMemcpyDeviceToHost),"read vision pooling");
      cap("pooled_scaled",{tokens,H},values);
    }
    bridge(tokens,features,cap);
    for (std::size_t i=0;i<batch.size();++i)
      check_cuda(cudaMemcpyAsync(outputs[i],features+std::size_t(offsets[i]/9)*kHidden,
          std::size_t(offsets[i+1]-offsets[i])/9*kHidden*sizeof(B),cudaMemcpyDeviceToDevice),"copy vision features");
    images=1;
  }
  void control_layer(int index,const runtime::ImageInput& image,const std::vector<float>& input,const Capture& cap) {
    if (index<0 || index>=vision::kLayers || image.end<=image.begin)
      throw std::invalid_argument("invalid vision layer control");
    vision_engine::validate_prepared_image_bytes(image.pixels.data(),image.pixels.size(),image.positions.data(),image.positions.size(),image.end-image.begin);
    const int n=(image.end-image.begin)*9;
    if (input.size()!=std::size_t(n)*H) throw std::invalid_argument("invalid vision control shape");
    check_cuda(cudaMemcpy(positions,image.positions.data(),n*2*sizeof(int),cudaMemcpyHostToDevice),"copy vision control positions");
    images=1;
    upload(input,hidden); layer(n,index,cap);
    check_cuda(cudaDeviceSynchronize(),"vision control completion");
  }
  void control_bridge(const std::vector<float>& input,const Capture& cap) {
    if (input.empty() || input.size()%H || input.size()/H>vision::kMaxSoftTokens)
      throw std::invalid_argument("invalid vision bridge control shape");
    upload(input,pooled); bridge(input.size()/H,features,cap);
    check_cuda(cudaDeviceSynchronize(),"vision bridge control completion");
  }
};
