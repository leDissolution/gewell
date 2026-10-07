#include "models/gemma4/26b_a4b/sm120/kv_write.cuh"
#include "cuda_memory.cuh"
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

using B = __nv_bfloat16;
using gewell::cuda_detail::DeviceAllocation;
using gewell::cuda_detail::check_cuda;
using gewell::kv_cache::Format;
using gewell::gemma4_26b_a4b::sm120::write_cache;
using gewell::gemma4_26b_a4b::sm120::write_cache_batch;
using gewell::gemma4_26b_a4b::sm120::CacheWriteInput;

// Independent host record construction; compares payload, scales, padding and
// untouched storage. No device cache address or storage helper is used here.
void record(std::vector<unsigned char>& destination, std::size_t offset,
            const std::vector<B>& values, unsigned split, Format format) {
  if (format == Format::bf16) {
    std::memcpy(destination.data()+offset,values.data(),values.size()*2);
    return;
  }
  float scale[2] = {0,0};
  for (unsigned i=0;i<values.size();++i)
    scale[i>=split] = std::max(scale[i>=split],std::abs(float(values[i])));
  for (auto& s : scale) s = s > 0 ? s/448.f : 1.f;
  for (unsigned i=0;i<values.size();++i) {
    __nv_fp8_e4m3 value(float(values[i])/scale[i>=split]);
    destination[offset+i] = value.__x;
  }
  unsigned scales = split < values.size() ? 2 : 1;
  std::memcpy(destination.data()+offset+values.size(),scale,scales*4);
  auto used = values.size()+scales*4, padded = (used+15)/16*16;
  std::fill(destination.begin()+offset+used,destination.begin()+offset+padded,0);
}

struct WriteCase {
  const bool global;
  const Format format;
  const unsigned position,rows,heads,width,elements,record_bytes,page_begin,page_end,pages;
  const std::size_t stride,bytes;
  std::vector<B> key,value;
  std::vector<unsigned char> expected,actual;
  std::vector<std::uint64_t> offsets;
  DeviceAllocation input_k,input_v,pool,table;
  gewell::kv_cache::DeviceView cache{};
  WriteCase(bool global,Format format,unsigned position,unsigned rows,cudaStream_t stream,unsigned seed=0)
      :global(global),format(format),position(position),rows(rows),heads(global?2:8),width(global?512:256),
       elements(global?640:256),record_bytes(format==Format::bf16?elements*2:(elements+(global?8:4)+15)/16*16),
       page_begin(position/256),page_end((position+rows+255)/256),pages(page_end-page_begin),
       stride(std::size_t(2)*256*record_bytes+256),bytes(global?pages*stride:std::size_t(2)*8*1024*record_bytes),
       key(std::size_t(rows)*heads*width),value(key.size()),expected(bytes,0xa5),actual(bytes),offsets(page_end),
       input_k(key.size()*2),input_v(value.size()*2),pool(bytes),table(offsets.size()*8) {
  for (std::size_t i=0;i<key.size();++i) {
    // One all-zero head/token tests zero-scale records, others exercise signs
    // and independent K/V scales. Every source dimension has a known value.
    key[i] = B(i/width==0 ? 0.f : (int((i*37+11+seed)%1009)-504)*0.03125f);
    value[i] = B(i/width==0 ? 0.f : (int((i*17+3+seed)%997)-498)*0.001953125f);
  }
  for (unsigned p=page_begin;p<page_end;++p) offsets[p]=(pages-1-(p-page_begin))*stride/2;
  check_cuda(cudaMemcpyAsync(input_k.data(),key.data(),key.size()*2,cudaMemcpyHostToDevice,stream),"copy K");
  check_cuda(cudaMemcpyAsync(input_v.data(),value.data(),value.size()*2,cudaMemcpyHostToDevice,stream),"copy V");
  check_cuda(cudaMemcpyAsync(pool.data(),expected.data(),bytes,cudaMemcpyHostToDevice,stream),"initialize cache");
  check_cuda(cudaMemcpyAsync(table.data(),offsets.data(),offsets.size()*8,cudaMemcpyHostToDevice,stream),"copy pages");
  cache.format=format; cache.capacity=global?262144:1024;
  cache.key=static_cast<B*>(pool.data());
  cache.value=cache.key+bytes/4;
  cache.page_pool=cache.key; cache.page_offsets=static_cast<std::uint64_t*>(table.data());
  cache.page_tokens=256; cache.page_count=page_end; cache.layer_offset_elements=64;
  cache.page_stride_elements=stride/2;
  // Finish the host upload before constructing expected bytes in the same vector.
  check_cuda(cudaStreamSynchronize(stream),"initialize cache case");
  for (unsigned t=global?0:rows-std::min(rows,1024u);t<rows;++t) {
    unsigned absolute=position+t;
    for (unsigned h=0;h<heads;++h) {
      std::size_t input=(std::size_t(t)*heads+h)*width;
      if (global) {
        std::vector<B> packed(640);
        for (unsigned d=0;d<128;++d) packed[d]=key[input+(d<64?d:d+192)];
        std::copy_n(value.begin()+input,512,packed.begin()+128);
        auto offset=offsets[absolute/256]*2+128+(std::size_t(h)*256+absolute%256)*record_bytes;
        record(expected,offset,packed,128,format);
      } else {
        auto offset=(std::size_t(h)*1024+absolute%1024)*record_bytes;
        record(expected,offset,{key.begin()+input,key.begin()+input+width},width,format);
        record(expected,bytes/2+offset,{value.begin()+input,value.begin()+input+width},width,format);
      }
    }
  }
  }
  CacheWriteInput input() const {
    return {static_cast<const B*>(input_k.data()),static_cast<const B*>(input_v.data()),cache,position,rows};
  }
  void verify(cudaStream_t stream) {
  check_cuda(cudaMemcpyAsync(actual.data(),pool.data(),bytes,cudaMemcpyDeviceToHost,stream),"read cache");
  check_cuda(cudaStreamSynchronize(stream),"complete cache writes");
  if (actual!=expected) {
    auto i=std::mismatch(actual.begin(),actual.end(),expected.begin()).first-actual.begin();
    throw std::runtime_error("cache record mismatch at byte "+std::to_string(i));
  }
  }
};

void run(bool global, Format format, unsigned position, unsigned rows, cudaStream_t stream) {
  WriteCase test(global,format,position,rows,stream);
  const auto input=test.input();
  write_cache(input.key,input.value,input.cache,global,position,rows,stream);
  test.verify(stream);
  std::cout<<(global?"global":"local")<<" "<<(format==Format::bf16?"BF16":"FP8")
           <<" position="<<position<<" rows="<<rows<<" exact\n";
}

void run_batch(bool global,unsigned count,bool decode,cudaStream_t stream) {
  const std::pair<unsigned,unsigned> shapes[]={{0,1},{255,2},{1023,3},{255,4096},{262140,4}};
  std::vector<std::unique_ptr<WriteCase>> cases;
  std::vector<CacheWriteInput> inputs;
  for (unsigned i=0;i<count;++i) {
    auto [position,rows]=shapes[i%5];
    if (decode) rows=1;
    cases.push_back(std::make_unique<WriteCase>(global,i%2?Format::fp8:Format::bf16,
        position,rows,stream,101*(i+1)));
    inputs.push_back(cases.back()->input());
  }
  inputs.insert(inputs.begin()+count/2,CacheWriteInput{});  // Empty entries must not consume a tile.
  write_cache_batch(inputs,global,stream);
  for (auto& test:cases) test->verify(stream);
  std::cout<<(global?"global":"local")<<" batch="<<count<<" decode="<<decode
           <<" mixed BF16/FP8 payload, scales, padding and untouched bytes exact\n";
}

int main() {
  try {
    cudaStream_t stream{};
    check_cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create stream");
    for (bool global : {false,true}) for (auto format : {Format::bf16,Format::fp8})
      for (auto shape : {std::pair{0u,1u},{255u,2u},{1023u,3u},{255u,4096u},{262140u,4u}})
        run(global,format,shape.first,shape.second,stream);
    for (bool global:{false,true}) for (unsigned count:{1u,4u,32u,33u}) for (bool decode:{false,true})
      run_batch(global,count,decode,stream);
    write_cache(nullptr,nullptr,{},false,0,0,stream);
    write_cache_batch({},false,stream);
    unsigned rejected=0;
    for (bool global : {false,true}) {
      try { write_cache(nullptr,nullptr,{},global,0,1,stream); }
      catch (const std::invalid_argument&) { ++rejected; }
      WriteCase test(global,Format::bf16,255,2,stream);
      auto bad=test.input();bad.rows=4097;
      try { write_cache_batch({test.input(),bad},global,stream); }
      catch (const std::invalid_argument&) { ++rejected; }
      std::fill(test.expected.begin(),test.expected.end(),0xa5);
      test.verify(stream);  // A malformed tail must not leave an earlier write queued.
    }
    if (rejected!=4) throw std::runtime_error("invalid cache write accepted");
    check_cuda(cudaStreamDestroy(stream),"destroy stream");
    std::cout<<"26B cache writes: 20 scalar and 16 batch cases exact; empty/invalid inputs passed\n";
    return 0;
  } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
