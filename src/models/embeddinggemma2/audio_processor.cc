#include "gewell/models/embeddinggemma2/audio_processor.h"
#include "gewell/models/embeddinggemma2/input.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
}
#include <openssl/evp.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace gewell::embeddinggemma2::audio {
namespace {
[[noreturn]] void invalid(const char* message) {throw std::invalid_argument(message);}
void checked(int status, const char* operation) {
  if (status >= 0) return;
  if (status == AVERROR(ENOMEM)) throw std::bad_alloc();
  char reason[AV_ERROR_MAX_STRING_SIZE];
  av_strerror(status, reason, sizeof(reason));
  throw std::invalid_argument(std::string("audio ")+operation+": "+reason);
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
  if (encoded.empty() || encoded.size()%4) invalid("invalid audio base64");
  const std::size_t padding=(encoded.back()=='=')+(encoded[encoded.size()-2]=='=');
  for (std::size_t i=0;i<encoded.size()-padding;++i)
    if (digit(encoded[i])<0) invalid("invalid audio base64");
  if (padding && ((padding==2 && (digit(encoded[encoded.size()-3])&15)) ||
                  (padding==1 && (digit(encoded[encoded.size()-2])&3)))) invalid("invalid audio base64 padding");
  std::vector<std::uint8_t> result(encoded.size()/4*3);
  const int size=EVP_DecodeBlock(result.data(),reinterpret_cast<const unsigned char*>(encoded.data()),encoded.size());
  if (size<0) invalid("invalid audio base64");
  result.resize(size-padding);
  return result;
}
bool wav_codec(AVCodecID id) {
  return id==AV_CODEC_ID_PCM_U8 || id==AV_CODEC_ID_PCM_S16LE || id==AV_CODEC_ID_PCM_S24LE ||
      id==AV_CODEC_ID_PCM_S32LE || id==AV_CODEC_ID_PCM_F32LE || id==AV_CODEC_ID_PCM_F64LE;
}
bool bounded_frame(const AVFrame* frame) {
  // Match default_get_buffer2's aligned allocation, including planar padding.
  const int bytes=av_samples_get_buffer_size(nullptr,frame->ch_layout.nb_channels,frame->nb_samples,
      static_cast<AVSampleFormat>(frame->format),0);
  return bytes>0 && std::uint64_t(bytes)<=kMaxDecodedFrameBytes;
}
int allocate_frame(AVCodecContext* context, AVFrame* frame, int flags) {
  if (!bounded_frame(frame)) return AVERROR_INVALIDDATA;
  return avcodec_default_get_buffer2(context,frame,flags);
}

struct Decoder {
  const std::vector<std::uint8_t>& bytes;
  const Poll& poll;
  std::uint32_t limit;
  std::size_t offset{};
  AVIOContext* io{};
  AVFormatContext* format{};
  AVCodecContext* codec{};
  AVPacket* packet{};
  AVFrame* frame{};
  SwrContext* resampler{};
  AVChannelLayout input_layout{};
  int stream=-1, input_rate{}, input_format=-1;
  Metadata metadata;
  std::vector<float> samples;
  std::array<float,16384> output{};
  Decoder(const std::vector<std::uint8_t>& source, std::uint32_t cap, const Poll& check)
      : bytes(source),poll(check),limit(cap) {}
  ~Decoder() {
    swr_free(&resampler);av_channel_layout_uninit(&input_layout);
    av_frame_free(&frame);av_packet_free(&packet);avcodec_free_context(&codec);
    avformat_close_input(&format);
    if (io) {av_freep(&io->buffer);avio_context_free(&io);}
  }
  static int read(void* opaque,std::uint8_t* target,int requested) {
    auto& self=*static_cast<Decoder*>(opaque);
    const auto size=std::min<std::size_t>(requested,self.bytes.size()-self.offset);
    if (!size) return AVERROR_EOF;
    std::memcpy(target,self.bytes.data()+self.offset,size);self.offset+=size;
    return size;
  }
  static std::int64_t seek(void* opaque,std::int64_t delta,int whence) {
    auto& self=*static_cast<Decoder*>(opaque);
    if (whence&AVSEEK_SIZE) return self.bytes.size();
    whence&=~AVSEEK_FORCE;
    const std::int64_t base=whence==SEEK_SET?0:whence==SEEK_CUR?self.offset:whence==SEEK_END?self.bytes.size():-1;
    if (base<0 || delta < -base || delta>std::int64_t(self.bytes.size())-base) return AVERROR(EINVAL);
    self.offset=base+delta;return self.offset;
  }
  static int refuse_external_io(AVFormatContext*,AVIOContext**,const char*,int,AVDictionary**) {
    return AVERROR(EACCES);
  }
  void open(const char* container) {
    if (poll) poll();
    auto* buffer=static_cast<unsigned char*>(av_malloc(32768));
    if (!buffer) throw std::bad_alloc();
    io=avio_alloc_context(buffer,32768,0,this,read,nullptr,seek);
    if (!io) {av_free(buffer);throw std::bad_alloc();}
    format=avformat_alloc_context();
    if (!format) throw std::bad_alloc();
    format->pb=io;format->flags|=AVFMT_FLAG_CUSTOM_IO;
    format->io_open=refuse_external_io;format->error_recognition=AV_EF_CRCCHECK|AV_EF_EXPLODE;
    const auto* demuxer=av_find_input_format(container);
    if (!demuxer) invalid("audio container decoder is unavailable");
    checked(avformat_open_input(&format,nullptr,demuxer,nullptr),"container");
    for (unsigned i=0;i<format->nb_streams;++i)
      if (format->streams[i]->codecpar->codec_type==AVMEDIA_TYPE_AUDIO) {stream=i;break;}
    if (stream<0) invalid("audio has no audio stream");
    const auto* parameters=format->streams[stream]->codecpar;
    if (std::strcmp(container,"wav")==0 ? !wav_codec(parameters->codec_id) :
        std::strcmp(container,"mp3")==0 ? parameters->codec_id!=AV_CODEC_ID_MP3 : parameters->codec_id!=AV_CODEC_ID_FLAC)
      invalid("unsupported audio encoding");
    const auto* implementation=avcodec_find_decoder(parameters->codec_id);
    if (!implementation) invalid("audio decoder is unavailable");
    codec=avcodec_alloc_context3(implementation);
    if (!codec) throw std::bad_alloc();
    checked(avcodec_parameters_to_context(codec,parameters),"codec parameters");
    codec->pkt_timebase=format->streams[stream]->time_base;
    codec->thread_count=1;codec->get_buffer2=allocate_frame;
    codec->err_recognition=AV_EF_CRCCHECK|AV_EF_EXPLODE;
    checked(avcodec_open2(codec,implementation,nullptr),"decoder");
    metadata.codec=avcodec_get_name(parameters->codec_id);
    packet=av_packet_alloc();frame=av_frame_alloc();
    if (!packet || !frame) throw std::bad_alloc();
    // A fixed bounded waveform buffer avoids geometric vector growth past the
    // declared transient budget. Only its actual decoded prefix becomes features.
    samples.reserve(limit+1);
  }
  void setup_resampler() {
    if (frame->sample_rate<=0) invalid("invalid audio sample rate");
    if (frame->ch_layout.order==AV_CHANNEL_ORDER_UNSPEC) {
      if (frame->ch_layout.nb_channels>2) invalid("multichannel audio requires a declared layout");
      av_channel_layout_default(&frame->ch_layout,frame->ch_layout.nb_channels);
    }
    if (!av_channel_layout_check(&frame->ch_layout)) invalid("invalid audio channel layout");
    if (resampler) {
      if (frame->sample_rate!=input_rate || frame->format!=input_format ||
          av_channel_layout_compare(&frame->ch_layout,&input_layout)!=0)
        invalid("audio format changes within the stream");
      return;
    }
    input_rate=frame->sample_rate;input_format=frame->format;
    checked(av_channel_layout_copy(&input_layout,&frame->ch_layout),"channel layout");
    AVChannelLayout mono=AV_CHANNEL_LAYOUT_MONO;
    checked(swr_alloc_set_opts2(&resampler,&mono,AV_SAMPLE_FMT_FLT,kSampleRate,&input_layout,
        static_cast<AVSampleFormat>(input_format),input_rate,0,nullptr),"resampler options");
    checked(av_opt_set_sample_fmt(resampler,"internal_sample_fmt",AV_SAMPLE_FMT_FLTP,0),"resampler precision");
    for (const auto& item:std::array<std::pair<const char*,double>,7>{{
        {"center_mix_level",std::sqrt(.5)},{"surround_mix_level",std::sqrt(.5)},
        {"lfe_mix_level",0},{"rematrix_volume",1},{"rematrix_maxval",1},{"cutoff",.97},{"kaiser_beta",9}}})
      checked(av_opt_set_double(resampler,item.first,item.second,0),"resampler setting");
    for (const auto& item:std::array<std::pair<const char*,int>,8>{{
        {"resampler",SWR_ENGINE_SWR},{"filter_size",32},{"phase_shift",10},{"linear_interp",1},
        {"exact_rational",1},{"filter_type",SWR_FILTER_TYPE_KAISER},{"dither_method",SWR_DITHER_NONE},{"async",0}}})
      checked(av_opt_set_int(resampler,item.first,item.second,0),"resampler setting");
    checked(swr_init(resampler),"resampler initialization");
    char layout[256];checked(av_channel_layout_describe(&input_layout,layout,sizeof(layout)),"layout description");
    metadata.input_rate=input_rate;metadata.channels=input_layout.nb_channels;metadata.layout=layout;
  }
  void append(int count) {
    checked(count,"resampling");
    for (int i=0;i<count;++i) {
      if (!std::isfinite(output[i])) invalid("nonfinite audio samples");
      if (samples.size()==limit) throw ContextLengthError("audio exceeds expanded context capacity");
      samples.push_back(output[i]);
    }
  }
  void convert() {
    if (!bounded_frame(frame)) invalid("audio frame exceeds decoded byte bounds");
    if (frame->flags&AV_FRAME_FLAG_CORRUPT || frame->decode_error_flags) invalid("corrupt audio frame");
    setup_resampler();
    const auto type=static_cast<AVSampleFormat>(input_format);
    const int bytes_per_sample=av_get_bytes_per_sample(type);
    const bool planar=av_sample_fmt_is_planar(type);
    const int planes=planar?input_layout.nb_channels:1;
    std::vector<const std::uint8_t*> source(planes);
    const int chunk=std::min<std::int64_t>(4096,std::max<std::int64_t>(1,16384LL*input_rate/kSampleRate));
    for (int begin=0;begin<frame->nb_samples;begin+=chunk) {
      if (poll) poll();
      const int count=std::min(chunk,frame->nb_samples-begin);
      for (int p=0;p<planes;++p) source[p]=frame->extended_data[p]+
          std::size_t(begin)*bytes_per_sample*(planar?1:input_layout.nb_channels);
      auto* target=reinterpret_cast<std::uint8_t*>(output.data());
      append(swr_convert(resampler,&target,output.size(),source.data(),count));
    }
  }
  void decode() {
    auto receive=[&] {
      for (;;) {
        if (poll) poll();
        const int status=avcodec_receive_frame(codec,frame);
        if (status==AVERROR(EAGAIN) || status==AVERROR_EOF) return;
        checked(status,"frame decode");convert();av_frame_unref(frame);
      }
    };
    for (;;) {
      if (poll) poll();
      const int status=av_read_frame(format,packet);
      if (status==AVERROR_EOF) break;
      checked(status,"packet read");
      if (packet->stream_index==stream) {
        if (packet->flags&AV_PKT_FLAG_CORRUPT) invalid("corrupt audio packet");
        checked(avcodec_send_packet(codec,packet),"packet decode");receive();
      }
      av_packet_unref(packet);
    }
    checked(avcodec_send_packet(codec,nullptr),"decoder flush");receive();
    if (!resampler) invalid("audio has no decoded frames");
    for (;;) {
      if (poll) poll();
      auto* target=reinterpret_cast<std::uint8_t*>(output.data());
      const int count=swr_convert(resampler,&target,output.size(),nullptr,0);
      append(count);if (!count) break;
    }
    metadata.samples=samples.size();
  }
};

struct FeatureTables {
  std::array<float,320> window;
  std::array<std::complex<double>,256> twiddles;
  std::array<std::array<double,128>,257> mel{};
  FeatureTables() {
    constexpr double pi=3.141592653589793238462643383279502884;
    for (int i=0;i<320;++i) window[i]=.5-.5*std::cos(2*pi*i/320);
    for (int i=0;i<256;++i) twiddles[i]={std::cos(-2*pi*i/512),std::sin(-2*pi*i/512)};
    std::array<double,130> frequencies;
    const double top=2595*std::log10(1+8000./700);
    for (int i=0;i<130;++i) frequencies[i]=700*(std::pow(10.,(i*(top/129))/2595)-1);
    for (int f=0;f<257;++f) for (int m=0;m<128;++m) {
      const double hz=f*(16000./512);
      mel[f][m]=std::max(0.,std::min((hz-frequencies[m])/(frequencies[m+1]-frequencies[m]),
          (frequencies[m+2]-hz)/(frequencies[m+2]-frequencies[m+1])));
    }
  }
  void fft(std::array<std::complex<double>,512>& data) const {
    for (unsigned i=1,j=0;i<512;++i) {
      unsigned bit=256;
      for (;j&bit;bit>>=1) j^=bit;
      j^=bit;if (i<j) std::swap(data[i],data[j]);
    }
    for (int size=2;size<=512;size*=2) for (int begin=0;begin<512;begin+=size)
      for (int i=0;i<size/2;++i) {
        const auto left=data[begin+i],right=data[begin+i+size/2]*twiddles[i*(512/size)];
        data[begin+i]=left+right;data[begin+i+size/2]=left-right;
      }
  }
};
}

std::uint32_t feature_rows(std::uint32_t samples) {
  const auto padded=(std::uint64_t(samples)+127)/128*128;
  return padded<161?0:(padded-161)/160+1;
}
std::uint32_t max_samples(int max_tokens) {
  if (max_tokens<1 || max_tokens>8192) invalid("invalid audio context capacity");
  if (max_tokens<5) throw ContextLengthError("audio requires at least five context positions");
  return 160+640*(max_tokens-4);
}
void validate_features(const Input& input) {
  if (input.samples<=160 || input.samples>5240480 || input.mask.size()!=feature_rows(input.samples) ||
      input.features.size()!=input.mask.size()*kFeatureWidth || input.end<=input.begin)
    invalid("invalid prepared audio geometry");
  std::uint32_t count=0;
  for (std::size_t i=0;i<input.mask.size();++i) {
    const bool valid=i*160+160<input.samples;
    if (input.mask[i]!=valid) invalid("invalid prepared audio mask");
    if (i%4==0 && valid) ++count;
    for (int j=0;j<kFeatureWidth;++j) {
      const float value=input.features[i*kFeatureWidth+j];
      if (!std::isfinite(value) || (!valid && value!=0)) invalid("invalid prepared audio features");
    }
  }
  if (input.end-input.begin!=count) invalid("prepared audio span/count differs");
}
std::shared_ptr<Input> prepare_samples(const std::vector<float>& samples,int max_tokens,const Acquire& acquire,const Poll& poll) {
  if (samples.size()>max_samples(max_tokens)) throw ContextLengthError("audio exceeds expanded context capacity");
  if (samples.size()<=160) invalid("audio has no valid features");
  for (std::size_t i=0;i<samples.size();++i) {
    if (i%16384==0 && poll) poll();
    if (!std::isfinite(samples[i])) invalid("nonfinite audio samples");
  }
  const auto rows=feature_rows(samples.size());
  auto input=acquire?acquire(rows):std::make_shared<Input>();
  if (!input) invalid("audio feature acquisition returned no input");
  input->samples=samples.size();input->begin=input->end=0;
  input->features.assign(std::size_t(rows)*kFeatureWidth,0);input->mask.resize(rows);
  static const FeatureTables tables;
  std::array<std::complex<double>,512> spectrum;
  std::array<double,257> magnitude;
  for (std::uint32_t t=0;t<rows;++t) {
    if (poll) poll();
    input->mask[t]=std::uint64_t(t)*160+160<samples.size();
    if (!input->mask[t]) continue;
    if (t%4==0) ++input->end;
    spectrum.fill({0,0});
    for (int i=0;i<320;++i) {
      const auto position=std::int64_t(t)*160+i-160;
      const float value=position>=0 && position<std::int64_t(samples.size())?samples[position]:0;
      spectrum[i]={value*tables.window[i],0};
    }
    tables.fft(spectrum);
    // NumPy's FP32-input FFT computes in double but returns complex64. Its
    // magnitude is FP32 before the float64 mel multiplication. The complex64
    // rounding matters at BF16 input ties even when FP32 features differ by
    // only one ULP.
    for (int f=0;f<257;++f) magnitude[f]=std::hypot(float(spectrum[f].real()),float(spectrum[f].imag()));
    for (int m=0;m<128;++m) {
      double sum=0;
      for (int f=0;f<257;++f) sum+=magnitude[f]*tables.mel[f][m];
      input->features[std::size_t(t)*128+m]=std::log(sum+.001);
    }
  }
  validate_features(*input);
  return input;
}
Prepared prepare_data_url(std::string_view url,int max_tokens,const Acquire& acquire,const Poll& poll,const ObserveSamples& observe) {
  if (url.size()>kMaxDataUrlBytes) invalid("audio data URL exceeds 8 MiB");
  const char* container=nullptr;std::size_t prefix=0;
  for (const auto& item:std::array<std::pair<std::string_view,const char*>,3>{{
      {"data:audio/wav;base64,","wav"},{"data:audio/mpeg;base64,","mp3"},{"data:audio/flac;base64,","flac"}}})
    if (url.substr(0,item.first.size())==item.first) {container=item.second;prefix=item.first.size();break;}
  if (!container) invalid("audio must be a base64 WAV, MP3 or FLAC data URL");
  const auto bytes=base64(url.substr(prefix));
  Decoder decoder(bytes,max_samples(max_tokens),poll);decoder.open(container);decoder.decode();
  if (poll) poll();
  if (observe) observe(decoder.samples);
  return {prepare_samples(decoder.samples,max_tokens,acquire,poll),decoder.metadata};
}
}  // namespace gewell::embeddinggemma2::audio
