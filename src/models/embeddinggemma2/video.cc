#include "gewell/models/embeddinggemma2/video.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
#include <openssl/evp.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace gewell::embeddinggemma2 {
namespace {
[[noreturn]] void invalid(const char* message) {throw std::invalid_argument(message);}
void checked(int status, const char* operation) {
  if (status >= 0) return;
  if (status == AVERROR(ENOMEM)) throw std::bad_alloc();
  char reason[AV_ERROR_MAX_STRING_SIZE];
  av_strerror(status, reason, sizeof(reason));
  throw std::invalid_argument(std::string("video ") + operation + ": " + reason);
}
bool valid_size(int width, int height) {
  return width > 0 && height > 0 && width <= int(gemma4::kImageMaxDimension) &&
      height <= int(gemma4::kImageMaxDimension) && std::uint64_t(width) * height <= gemma4::kImageMaxPixels;
}
int bounded_frame_buffer(AVCodecContext* context, AVFrame* frame, int flags) {
  if (!valid_size(context->width, context->height)) return AVERROR_INVALIDDATA;
  return avcodec_default_get_buffer2(context, frame, flags);
}
int digit(unsigned char c) {
  if (c >= 'A' && c <= 'Z') return c-'A';
  if (c >= 'a' && c <= 'z') return c-'a'+26;
  if (c >= '0' && c <= '9') return c-'0'+52;
  if (c == '+') return 62;
  if (c == '/') return 63;
  return -1;
}
std::vector<std::uint8_t> base64(std::string_view encoded) {
  if (encoded.empty() || encoded.size()%4) invalid("invalid video base64");
  const std::size_t padding = (encoded.back()=='=') + (encoded[encoded.size()-2]=='=');
  for (std::size_t i=0; i<encoded.size()-padding; ++i)
    if (digit(encoded[i])<0) invalid("invalid video base64");
  if (padding && ((padding==2 && (digit(encoded[encoded.size()-3])&15)) ||
                  (padding==1 && (digit(encoded[encoded.size()-2])&3)))) invalid("invalid video base64 padding");
  std::vector<std::uint8_t> result(encoded.size()/4*3);
  const int size=EVP_DecodeBlock(result.data(),reinterpret_cast<const unsigned char*>(encoded.data()),encoded.size());
  if (size<0) invalid("invalid video base64");
  result.resize(size-padding);
  return result;
}

struct Decoder {
  const std::vector<std::uint8_t>& bytes;
  const PreparationPoll& poll;
  std::size_t offset{};
  AVIOContext* io{};
  AVFormatContext* format{};
  AVCodecContext* codec{};
  AVPacket* packet{};
  AVFrame* frame{};
  SwsContext* scaler{};
  int stream=-1, width=0, height=0;
  double fps{};
  Decoder(const std::vector<std::uint8_t>& source, const PreparationPoll& check) : bytes(source), poll(check) {}
  ~Decoder() {
    sws_freeContext(scaler);
    av_frame_free(&frame); av_packet_free(&packet); avcodec_free_context(&codec);
    avformat_close_input(&format);
    if (io) {av_freep(&io->buffer); avio_context_free(&io);}
  }
  static int read(void* opaque, std::uint8_t* output, int requested) {
    auto& self=*static_cast<Decoder*>(opaque);
    const auto size=std::min<std::size_t>(requested,self.bytes.size()-self.offset);
    if (!size) return AVERROR_EOF;
    std::memcpy(output,self.bytes.data()+self.offset,size); self.offset+=size;
    return size;
  }
  static std::int64_t seek(void* opaque, std::int64_t delta, int whence) {
    auto& self=*static_cast<Decoder*>(opaque);
    if (whence&AVSEEK_SIZE) return self.bytes.size();
    whence &= ~AVSEEK_FORCE;
    const std::int64_t base=whence==SEEK_SET?0:whence==SEEK_CUR?self.offset:whence==SEEK_END?self.bytes.size():-1;
    if (base<0 || delta < -base || delta > std::int64_t(self.bytes.size())-base) return AVERROR(EINVAL);
    self.offset=base+delta; return self.offset;
  }
  static int refuse_external_io(AVFormatContext*, AVIOContext**, const char*, int, AVDictionary**) {
    return AVERROR(EACCES);
  }
  void open(bool mp4) {
    if (poll) poll();
    auto* buffer=static_cast<unsigned char*>(av_malloc(32768));
    if (!buffer) throw std::bad_alloc();
    io=avio_alloc_context(buffer,32768,0,this,read,nullptr,seek);
    if (!io) {av_free(buffer); throw std::bad_alloc();}
    format=avformat_alloc_context();
    if (!format) throw std::bad_alloc();
    format->pb=io; format->flags|=AVFMT_FLAG_CUSTOM_IO;
    format->io_open=refuse_external_io;
    format->error_recognition=AV_EF_CRCCHECK|AV_EF_EXPLODE;
    const auto* demuxer=av_find_input_format(mp4?"mov":"matroska");
    if (!demuxer) invalid("video container decoder is unavailable");
    checked(avformat_open_input(&format,nullptr,demuxer,nullptr),"container");
    // The supported demuxers provide track headers at open. Avoid opening
    // unbounded probe decoders through avformat_find_stream_info.
    for (unsigned i=0;i<format->nb_streams;++i)
      if (format->streams[i]->codecpar->codec_type==AVMEDIA_TYPE_VIDEO) {stream=i;break;}
    if (stream<0) invalid("video has no video stream");
    const auto* parameters=format->streams[stream]->codecpar;
    if (mp4 ? parameters->codec_id!=AV_CODEC_ID_H264 :
        parameters->codec_id!=AV_CODEC_ID_VP8 && parameters->codec_id!=AV_CODEC_ID_VP9)
      invalid("video codec must be H.264 in MP4 or VP8/VP9 in WebM");
    if (!valid_size(parameters->width,parameters->height)) invalid("video frame exceeds decoded image bounds or has no dimensions");
    const auto rate=format->streams[stream]->avg_frame_rate;
    if (rate.num>0 && rate.den>0) fps=av_q2d(rate);
    const auto* implementation=avcodec_find_decoder(parameters->codec_id);
    if (!implementation) invalid("video decoder is unavailable");
    codec=avcodec_alloc_context3(implementation);
    if (!codec) throw std::bad_alloc();
    checked(avcodec_parameters_to_context(codec,parameters),"codec parameters");
    codec->thread_count=1;
    codec->max_pixels=gemma4::kImageMaxPixels;
    codec->get_buffer2=bounded_frame_buffer;
    codec->err_recognition=AV_EF_CRCCHECK|AV_EF_EXPLODE;
    checked(avcodec_open2(codec,implementation,nullptr),"decoder");
    packet=av_packet_alloc(); frame=av_frame_alloc();
    if (!packet || !frame) throw std::bad_alloc();
  }
  template<class Output> std::uint64_t decode(Output output) {
    std::uint64_t count=0;
    auto receive=[&] {
      for (;;) {
        if (poll) poll();
        const int result=avcodec_receive_frame(codec,frame);
        if (result==AVERROR(EAGAIN) || result==AVERROR_EOF) return;
        checked(result,"frame decode");
        if (!valid_size(frame->width,frame->height)) invalid("video frame exceeds decoded image bounds");
        if (frame->flags&AV_FRAME_FLAG_CORRUPT || frame->decode_error_flags) invalid("corrupt video frame");
        if (!count) {width=frame->width;height=frame->height;}
        if (width!=frame->width || height!=frame->height) invalid("video frame dimensions change within the stream");
        output(count++,*frame);
        av_frame_unref(frame);
      }
    };
    for (;;) {
      if (poll) poll();
      const int result=av_read_frame(format,packet);
      if (result==AVERROR_EOF) break;
      checked(result,"packet read");
      if (packet->stream_index==stream) {
        if (packet->flags&AV_PKT_FLAG_CORRUPT) invalid("corrupt video packet");
        checked(avcodec_send_packet(codec,packet),"packet decode");
        receive();
      }
      av_packet_unref(packet);
    }
    checked(avcodec_send_packet(codec,nullptr),"decoder flush");
    receive();
    if (!count) invalid("video has no decoded frames");
    return count;
  }
  gemma4::RgbImage rgb(const AVFrame& source) {
    scaler=sws_getCachedContext(scaler,width,height,static_cast<AVPixelFormat>(source.format),
        width,height,AV_PIX_FMT_RGB24,SWS_BICUBIC,nullptr,nullptr,nullptr);
    if (!scaler) throw std::bad_alloc();
    const auto space=source.colorspace==AVCOL_SPC_UNSPECIFIED?SWS_CS_DEFAULT:
        source.colorspace==AVCOL_SPC_BT2020_CL?SWS_CS_BT2020:int(source.colorspace);
    const int* coefficients=sws_getCoefficients(space);
    checked(sws_setColorspaceDetails(scaler,coefficients,source.color_range==AVCOL_RANGE_JPEG,
        coefficients,1,0,1<<16,1<<16),"color conversion setup");
    const int stride=(width*3+31)&~31;
    gemma4::RgbImage result{std::uint32_t(width),std::uint32_t(height),
        std::vector<std::uint8_t>(std::size_t(stride)*height+AV_INPUT_BUFFER_PADDING_SIZE)};
    std::uint8_t* targets[]={result.pixels.data(),nullptr,nullptr,nullptr};
    const int strides[]={stride,0,0,0};
    if (sws_scale(scaler,source.data,source.linesize,0,height,targets,strides)!=height)
      invalid("video RGB conversion is incomplete");
    for (int y=1;y<height;++y)
      std::memmove(result.pixels.data()+std::size_t(y)*width*3,result.pixels.data()+std::size_t(y)*stride,width*3);
    result.pixels.resize(std::size_t(width)*height*3);
    return result;
  }
};
}

std::vector<std::uint64_t> sample_video_frames(std::uint64_t total_frames,double fps) {
  if (!total_frames) invalid("video has no decoded frames");
  const bool rate=std::isfinite(fps) && fps>0;
  const double duration=rate?double(total_frames)/fps:0;
  if (rate && (!std::isfinite(duration) || duration>=double(UINT64_MAX))) invalid("video duration exceeds sampling capacity");
  const auto samples=rate?std::uint64_t(std::max(1.,std::floor(duration))):total_frames;
  const auto count=std::min<std::uint64_t>(32,samples);
  std::vector<std::uint64_t> result;
  for (std::uint64_t i=0;i<count;++i) {
    // numpy.linspace includes the exact final endpoint, then casts to int.
    const auto sample=samples<=32?i:i==31?samples-1:std::uint64_t(i*(double(samples-1)/31.));
    const auto index=rate?std::min(double(total_frames-1),std::floor(sample*fps)):double(sample);
    result.push_back(std::uint64_t(index));
  }
  return result;
}

PreparedVideo prepare_video_data_url(std::string_view url,const PrepareVideoFrame& prepare,
    const PreparationPoll& poll,const ObserveVideoFrame& observe) {
  if (url.size()>gemma4::kImageMaxDataUrlBytes) invalid("video data URL exceeds 8 MiB");
  constexpr std::string_view mp4="data:video/mp4;base64,",webm="data:video/webm;base64,";
  const bool is_mp4=url.substr(0,mp4.size())==mp4;
  if (!is_mp4 && url.substr(0,webm.size())!=webm) invalid("video must be a base64 MP4 or WebM data URL");
  const auto bytes=base64(url.substr(is_mp4?mp4.size():webm.size()));
  PreparedVideo result;
  {
    Decoder counter(bytes,poll);counter.open(is_mp4);
    result.metadata.total_frames=counter.decode([](auto,const auto&){});
    result.metadata.fps=counter.fps;
  }
  auto& metadata=result.metadata;
  metadata.duration=metadata.fps>0?metadata.total_frames/metadata.fps:0;
  metadata.indices=sample_video_frames(metadata.total_frames,metadata.fps);
  Decoder decoder(bytes,poll);decoder.open(is_mp4);
  std::size_t next=0;
  const auto count=decoder.decode([&](std::uint64_t index,const AVFrame& frame) {
    if (next==metadata.indices.size() || index!=metadata.indices[next]) return;
    bool first=true;
    while (next<metadata.indices.size() && metadata.indices[next]==index) {
      if (poll) poll();
      auto raster=decoder.rgb(frame);
      if (first && observe) observe(index,raster);
      first=false;
      std::shared_ptr<runtime::ImageInput> prepared;
      if (prepare) prepared=prepare(std::move(raster));
      else {
        auto image=gemma4::prepare_rgb_image(std::move(raster),kVideoFrameBudget);
        prepared=std::make_shared<runtime::ImageInput>();
        prepared->pixels=std::move(image.pixels);prepared->positions=std::move(image.positions);
        prepared->padded_patch_rows=image.padded_patch_rows;prepared->end=image.soft_token_count;
      }
      if (!prepared || prepared->begin || !prepared->end || prepared->end>kVideoFrameBudget)
        throw std::runtime_error("video frame processor returned invalid features");
      result.frames.push_back(std::move(prepared));
      const auto timestamp=frame.best_effort_timestamp==AV_NOPTS_VALUE?std::numeric_limits<double>::quiet_NaN():
          frame.best_effort_timestamp*av_q2d(decoder.format->streams[decoder.stream]->time_base);
      metadata.timestamps.push_back(timestamp);
      ++next;
      if (poll) poll();
    }
  });
  if (count!=metadata.total_frames || next!=metadata.indices.size()) invalid("video decode passes disagree");
  return result;
}
}  // namespace gewell::embeddinggemma2
