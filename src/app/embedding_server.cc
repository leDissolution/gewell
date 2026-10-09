#include "gewell/embedding_server.h"
#include "gewell/console.h"
#include "gewell/http_server.h"
#include "gewell/metrics.h"
#include "gewell/models/embeddinggemma2/encoder.h"
#include "gewell/models/embeddinggemma2/text.h"
#include "gewell/models/gemma4/image_processor.h"
#include "gewell/vision_engine.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <csignal>
#include <deque>
#include <malloc.h>
#include <map>
#include <thread>

namespace gewell::app {
namespace {
using Clock = std::chrono::steady_clock;
volatile std::sig_atomic_t interrupted = 0;
void interrupt(int) { interrupted = 1; }
struct Signals {
  using Handler = void (*)(int);
  Handler previous_int, previous_term;
  Signals() : previous_int(std::signal(SIGINT, interrupt)), previous_term(std::signal(SIGTERM, interrupt)) { interrupted = 0; }
  ~Signals() { std::signal(SIGINT, previous_int); std::signal(SIGTERM, previous_term); }
};
std::uint64_t positive(std::string_view value, const std::string& option, std::uint64_t maximum) {
  std::uint64_t result = 0;
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || !result || result > maximum)
    throw std::invalid_argument(option + " must be in 1.." + std::to_string(maximum));
  return result;
}
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
struct Observations {
  std::uint64_t completed = 0, failed = 0, cancelled = 0, rejected = 0, tokens = 0, inputs = 0;
  std::uint64_t images = 0, image_tokens = 0;
  std::uint64_t videos = 0, video_frames = 0, video_tokens = 0;
  std::uint64_t audios = 0, audio_tokens = 0;
  std::map<int, std::uint64_t> dimensions{{128,0}, {256,0}, {512,0}, {768,0}};
  metrics::Histogram queue{{.001,.005,.01,.05,.1,.5,1,5}}, execution{{.001,.005,.01,.05,.1,.5,1,5}},
      batch{{1,2,4,8,16,32,64,128,256}}, lengths{{32,128,512,1024,2048,4096,8192}};
  std::string render(const std::string& model, std::size_t waiting, bool active, const embeddinggemma2::Encoder& encoder) const {
    metrics::Writer w(model);
    w.counter("gewell:embedding_requests_completed_total", "Embedding responses delivered successfully.", completed);
    w.counter("gewell:embedding_requests_failed_total", "Admitted embedding requests with execution errors.", failed);
    w.counter("gewell:embedding_requests_cancelled_total", "Admitted embedding requests cancelled before delivery.", cancelled);
    w.counter("gewell:embedding_requests_rejected_total", "Prepared embedding requests rejected by queue capacity.", rejected);
    w.counter("gewell:embedding_input_tokens_total", "Tokens in completed input forwards, including BOS/EOS.", tokens);
    w.counter("gewell:embedding_inputs_total", "Completed input forwards, including inputs of subsequently cancelled requests.", inputs);
    w.counter("gewell:embedding_images_total", "Images in completed input forwards.", images);
    w.counter("gewell:embedding_image_soft_tokens_total", "Image feature rows in completed input forwards.", image_tokens);
    w.counter("gewell:embedding_videos_total", "Video clips in completed input forwards.", videos);
    w.counter("gewell:embedding_video_frames_total", "Selected video frames in completed input forwards.", video_frames);
    w.counter("gewell:embedding_video_soft_tokens_total", "Video feature rows in completed input forwards.", video_tokens);
    w.counter("gewell:embedding_audios_total", "Audio clips in completed input forwards.", audios);
    w.counter("gewell:embedding_audio_soft_tokens_total", "Audio feature rows in completed input forwards.", audio_tokens);
    for (auto [dimension, count] : dimensions)
      w.counter("gewell:embedding_vectors_total", "Delivered embedding vectors by dimension.", count, {{"dimensions", std::to_string(dimension)}});
    w.gauge("gewell:embedding_requests_waiting", "Queued embedding requests.", waiting);
    w.gauge("gewell:embedding_requests_running", "Executing embedding requests.", active ? 1 : 0);
    w.gauge("gewell:embedding_weight_bytes", "Explicit resident encoder weight allocation.", encoder.weight_bytes());
    w.gauge("gewell:embedding_scratch_bytes", "Explicit fixed encoder scratch allocation, counting shared storage once.", encoder.scratch_bytes());
    w.gauge("gewell:embedding_vision_weight_bytes", "Explicit resident vision weight allocation.", encoder.vision_weight_bytes());
    w.gauge("gewell:embedding_vision_scratch_bytes", "Vision working-memory requirement, including storage shared with text.", encoder.vision_scratch_bytes());
    w.gauge("gewell:embedding_audio_weight_bytes", "Explicit resident audio weight allocation.", encoder.audio_weight_bytes());
    w.gauge("gewell:embedding_audio_scratch_bytes", "Explicit fixed audio scratch allocation.", encoder.audio_scratch_bytes());
    w.gauge("gewell:embedding_peak_allocated_bytes", "Peak explicit encoder allocation, excluding CUDA, cuBLAS and cuDNN overhead.", encoder.weight_bytes() + encoder.scratch_bytes());
    w.histogram("gewell:embedding_queue_seconds", "Preparation and queue delay before execution.", queue);
    w.histogram("gewell:embedding_execution_seconds", "Request execution time including cancellation polls.", execution);
    w.histogram("gewell:embedding_request_inputs", "Input count in admitted requests.", batch);
    w.histogram("gewell:embedding_batch_tokens", "Token count per completed input.", lengths);
    return w.str();
  }
};
}

int serve_embeddings(int argc, char** argv) {
  if (!argc || std::string_view(argv[0]).substr(0, 2) == "--")
    throw std::invalid_argument("serve-embeddings requires MODEL_DIR");
  const std::string directory = argv[0];
  // Every image-preparation thread would otherwise get its own glibc arena,
  // whose top chunk malloc_trim cannot return.
  mallopt(M_ARENA_MAX, 4);
  http::Settings settings;
  settings.model = "google/embeddinggemma-2";
  settings.embeddings.emplace();
  // Four preparation workers can each hold a full request of prepared images
  // (about 7.7 MB per 280-budget image), which overflows the 256 MiB default.
  settings.max_body_total_bytes = std::size_t(1) << 30;
  std::size_t max_pending = 8;
  bool vision=false;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option=="--vision") {vision=true;continue;}
    if (option=="--audio") {settings.embeddings->audio=true;continue;}
    if (++i == argc) throw std::invalid_argument(option + " requires a value");
    const std::string value = argv[i];
    if (option == "--host") settings.host = value;
    else if (option == "--model") settings.model = value;
    else if (option == "--port") settings.port = positive(value, option, 65535);
    else if (option == "--max-inputs") settings.embeddings->max_inputs = positive(value, option, 256);
    else if (option == "--max-pending") max_pending = positive(value, option, 256);
    else if (option == "--max-batch-tokens") settings.embeddings->max_batch_tokens = positive(value, option, 8192);
    else if (option == "--max-connections") settings.max_connections = positive(value, option, 256);
    else if (option == "--max-body-bytes") settings.max_body_bytes = positive(value, option, UINT32_MAX);
    else if (option == "--max-body-total-bytes") settings.max_body_total_bytes = positive(value, option, UINT32_MAX);
    else if (option == "--max-output-bytes") settings.max_output_bytes = positive(value, option, UINT32_MAX);
    else if (option == "--socket-timeout-seconds") settings.socket_timeout_seconds = positive(value, option, UINT32_MAX);
    else throw std::invalid_argument("unknown serve-embeddings option: " + option);
  }
  text::Tokenizer tokenizer(directory + "/tokenizer.json", embeddinggemma2::text_contract());
  Signals signals;
  http::ImageSupport images;
  if (vision) {
    images.begin_token=255999;images.image_token=258880;images.end_token=258882;
    images.max_image_tokens=1120;images.default_max_soft_tokens=280;
    for (const auto budget:vision_engine::kSupportedSoftTokenCapacities) {
      const auto rows=vision_engine::padded_patch_rows_for_capacity(budget);
      images.prepared_bytes.emplace(budget,vision_engine::prepared_pixel_bytes(rows)+vision_engine::prepared_position_bytes(rows));
    }
    auto make_image=[](gemma4::PreparedImage prepared) {
      auto image=std::make_shared<runtime::ImageInput>();
      image->pixels=std::move(prepared.pixels);image->positions=std::move(prepared.positions);
      image->padded_patch_rows=prepared.padded_patch_rows;image->end=prepared.soft_token_count;
      return image;
    };
    images.prepare=[make_image](std::string_view url,std::uint32_t budget) {
      return make_image(gemma4::prepare_image_data_url(url,budget));
    };
    images.prepare_video_frame=[make_image](gemma4::RgbImage frame) {
      return make_image(gemma4::prepare_rgb_image(std::move(frame),embeddinggemma2::kVideoFrameBudget));
    };
  }
  http::Server transport(tokenizer, settings, 1, std::move(images));
  embeddinggemma2::Encoder encoder(directory, settings.embeddings->max_batch_tokens,vision,settings.embeddings->audio);
  Observations stats;
  std::deque<http::Admission> pending;
  // Results awaiting socket delivery; bounded by the transport connection limit.
  std::map<http::ClientId, std::pair<int, std::size_t>> delivering;
  http::ClientId active = 0;
  auto last_activity = Clock::now();
  bool trimmed = false;
  auto publish = [&] { transport.publish_observability(stats.render(settings.model, pending.size(), active != 0, encoder), nullptr); };
  auto poll = [&] {
    bool changed = false;
    for (auto id : transport.take_completions()) {
      const auto it = delivering.find(id);
      if (it != delivering.end()) {
        ++stats.completed;
        stats.dimensions[it->second.first] += it->second.second;
        delivering.erase(it); changed = true;
      }
    }
    for (auto id : transport.take_cancellations()) {
      auto it = std::find_if(pending.begin(), pending.end(), [id](const auto& p) { return p.client == id; });
      if (it != pending.end()) { pending.erase(it); ++stats.cancelled; changed = true; }
      if (delivering.erase(id)) { ++stats.cancelled; changed = true; }
    }
    for (auto& admission : transport.take_admissions()) {
      changed = true;
      if (pending.size() + (active ? 1 : 0) >= max_pending) {
        transport.reject(admission.client, http::Error(503, "embedding queue is full", {}, "capacity_exceeded"));
        ++stats.rejected;
      } else {
        stats.batch.observe(admission.request.embeddings->inputs.size());
        pending.push_back(std::move(admission));
      }
    }
    if (changed) { publish(); last_activity = Clock::now(); trimmed = false; }
    return interrupted || !transport.healthy() || (active && transport.cancelled(active));
  };
  publish();
  transport.set_ready(true);
  console::event("embedding_server_ready", {{"model", settings.model}, {"host", settings.host}, {"port", settings.port},
      {"max_inputs", settings.embeddings->max_inputs}, {"max_pending", max_pending},
      {"vision",vision},{"vision_weight_bytes",encoder.vision_weight_bytes()},{"vision_scratch_bytes",encoder.vision_scratch_bytes()},
      {"audio",settings.embeddings->audio},{"audio_weight_bytes",encoder.audio_weight_bytes()},{"audio_scratch_bytes",encoder.audio_scratch_bytes()},
      {"max_batch_tokens", encoder.max_tokens()}, {"weight_bytes", encoder.weight_bytes()}, {"scratch_bytes", encoder.scratch_bytes()}},
      "Embeddings ready | " + settings.model + " | http://" + settings.host + ":" + std::to_string(settings.port));
  while (!interrupted && transport.healthy()) {
    poll();
    if (interrupted) break;
    if (pending.empty()) {
      // Image preparation threads leave freed decode/resize buffers in glibc arenas.
      if (!trimmed && delivering.empty() && Clock::now() - last_activity > std::chrono::seconds(1)) {
        malloc_trim(0);
        trimmed = true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    auto admission = std::move(pending.front()); pending.pop_front();
    active = admission.client;
    const auto& input = *admission.request.embeddings;
    stats.queue.observe(seconds(admission.request.arrival_time));
    const auto begin = Clock::now();
    publish();
    try {
      http::EmbeddingResult result;
      std::vector<const embeddinggemma2::PreparedInput*> batch;
      for (const auto& prepared : input.inputs) batch.push_back(&prepared);
      result.vectors = encoder.encode_batch(batch, input.dimensions, poll);
      for (const auto& prepared : input.inputs) {
        stats.tokens += prepared.tokens.size(); ++stats.inputs; stats.lengths.observe(prepared.tokens.size());
        stats.videos+=prepared.video_count;
        stats.audios+=prepared.audios.size();
        for (const auto& audio:prepared.audios) stats.audio_tokens+=audio->end-audio->begin;
        for (const auto& image:prepared.visuals) {
          if (prepared.tokens[image->begin]==258884) {
            ++stats.video_frames; stats.video_tokens+=image->end-image->begin;
          } else {
            ++stats.images; stats.image_tokens+=image->end-image->begin;
          }
        }
      }
      if (poll()) ++stats.cancelled;
      else {
        delivering.emplace(active, std::make_pair(input.dimensions, result.vectors.size()));
        transport.finish_embeddings(active, std::move(result));
      }
    } catch (const std::exception& error) {
      if (interrupted || transport.cancelled(active)) ++stats.cancelled;
      else {
        ++stats.failed;
        transport.fail_all(http::Error(500, "embedding execution failed", {}, "execution_failed"));
        console::event("embedding_execution_error", {{"message", error.what()}}, error.what(), true);
      }
    }
    stats.execution.observe(seconds(begin));
    active = 0; publish();
    last_activity = Clock::now(); trimmed = false;
  }
  const bool healthy = transport.healthy();
  transport.set_ready(false);
  transport.fail_all(http::Error(503, "embedding server is stopping", {}, "server_stopping"));
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  transport.stop();
  return healthy ? 0 : 1;
}
}  // namespace gewell::app
