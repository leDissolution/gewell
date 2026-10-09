#include "gewell/http_server.h"
#include "gewell/chat_codec.h"
#include "gewell/console.h"
#include "gewell/metrics.h"

#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <ctime>
#include <deque>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>
#include <utility>

namespace gewell::http {
namespace {

using Clock = std::chrono::steady_clock;
constexpr std::size_t kHeaderBytes = 32768;

const char* reason(int status) {
  switch (status) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 411: return "Length Required";
    case 413: return "Payload Too Large";
    case 415: return "Unsupported Media Type";
    case 417: return "Expectation Failed";
    case 431: return "Request Header Fields Too Large";
    case 503: return "Service Unavailable";
    default: return "Internal Server Error";
  }
}

std::string headers(int status, std::string_view id, bool stream, std::size_t length = 0,
                    std::string_view content_type = "application/json") {
  return "HTTP/1.1 " + std::to_string(status) + " " + reason(status) +
      "\r\nContent-Type: " + std::string(stream ? "text/event-stream; charset=utf-8" : content_type) +
      "\r\nConnection: close\r\nx-request-id: " + std::string(id) +
      (stream ? "\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n\r\n" :
       "\r\nContent-Length: " + std::to_string(length) + "\r\n\r\n");
}

std::string chunk(std::string_view text) {
  if (text.empty()) return {};
  std::ostringstream size;
  size << std::hex << text.size();
  return size.str() + "\r\n" + std::string(text) + "\r\n";
}

std::string_view trim_header(std::string_view value) {
  while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) value.remove_prefix(1);
  while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) value.remove_suffix(1);
  return value;
}

bool header_character(unsigned char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
      (c >= '0' && c <= '9') || std::string_view("!#$%&'*+-.^_`|~").find(c) != std::string_view::npos;
}

void log_error(std::string_view request_id, std::string_view method,
               std::string_view path, const Error& error, bool response_sent = false) {
  // Readiness probes are expected while weights load; retain them in JSON diagnostics.
  if (!console::json_enabled() && method == "GET" && path.substr(0, path.find('?')) == "/health" &&
      error.status == 503 && error.code == "model_not_ready") return;
  auto event = error_json(error);
  event["request_id"] = request_id;
  event["method"] = method;
  event["path"] = path.substr(0, path.find('?'));
  event["status"] = error.status;
  event["response_sent"] = response_sent;
  // One flushed line on stderr, without headers or request bodies. Invalid
  // bytes in malformed HTTP targets must not prevent the error response.
  console::event("server_http_error", event,
      "HTTP " + std::to_string(error.status) + " | " + std::string(method) + " " +
      std::string(path.substr(0, path.find('?'))) + " | request " + std::string(request_id) +
      " | " + error.what(), true);
}

}  // namespace

struct Server::Impl {
  struct Client {
    ClientId id{};
    int socket = -1;
    std::string request_id, response_id;
    std::int64_t created{};
    // Shared fields below are protected by mutex. The HTTP state below them
    // belongs only to the I/O thread; preparation owns body after enqueue.
    bool alive = true, worker_active = false, admitted = false;
    bool admission_pending = false, cancellation_pending = false, completion_pending = false;
    bool stop_pending = false, stop_notified = false;
    bool reserved = false, occupied = false, terminal = false;
    std::size_t body_charge = 0;
    std::string body;
    std::optional<Request> prepared;
    std::optional<Error> error;
    std::vector<std::uint32_t> tokens;
    std::vector<TokenLogprobs> logprobs;
    std::optional<Result> result;
    std::optional<EmbeddingResult> embedding_result;
    std::string method, path, input, wire;
    std::size_t body_length = 0, body_received = 0, sent = 0;
    bool parsed_header = false, request_complete = false;
    bool response_started = false, response_sent = false, streaming = false;
    bool closing = false, success = false;
    Clock::time_point last_read = Clock::now(), last_write = Clock::now();
    Clock::time_point arrival_time{};
    std::optional<Error> deferred_error;
    std::optional<Request> request;
    std::unique_ptr<CompletionOutput> output;
  };

  const text::Tokenizer& tokenizer;
  constraint::Compiler compiler;
  Settings settings;
  ImageSupport images;
  // The lease follows the shared tensors through preparation, admission and
  // GPU retirement, including a disconnect before queued work is cancelled.
  struct ImageLease {
    std::shared_ptr<runtime::ImageInput> image;
    std::shared_ptr<std::atomic<std::size_t>> total;
    std::size_t bytes{};
    ~ImageLease() { image.reset(); if (bytes) total->fetch_sub(bytes); }
  };
  struct AudioLease {
    embeddinggemma2::audio::Input input;
    std::shared_ptr<std::atomic<std::size_t>> total;
    std::size_t bytes{};
    ~AudioLease() { input={}; if (bytes) total->fetch_sub(bytes); }
  };
  std::shared_ptr<std::atomic<std::size_t>> media_bytes =
      std::make_shared<std::atomic<std::size_t>>(0);
  std::size_t max_burst;
  int listener = -1, wake_fd = -1;
  std::int64_t started = std::time(nullptr);
  std::string instance = std::to_string(Clock::now().time_since_epoch().count());
  mutable std::mutex mutex;
  std::condition_variable work_available;
  std::thread io_thread;
  std::vector<std::thread> worker_threads;
  std::atomic<bool> stopping{false}, healthy{true}, model_ready{false};
  std::map<ClientId, std::shared_ptr<Client>> clients;
  std::deque<std::shared_ptr<Client>> jobs;
  std::vector<Admission> admissions;
  std::vector<ClientId> cancellations, completions, stops;
  std::size_t body_bytes = 0;
  ClientId next_id = 0;
  struct Observability {
    std::string metrics;
    std::shared_ptr<const std::string> cache_index;
  };
  std::shared_ptr<const Observability> observability;

  Impl(const text::Tokenizer& codec, Settings configured, std::size_t burst, ImageSupport image_support)
      : tokenizer(codec), compiler(codec, burst ? burst - 1 : 0), settings(std::move(configured)),
        images(std::move(image_support)), max_burst(burst) {
    if (settings.host.empty() || !settings.port || !settings.max_connections || settings.max_connections > 256 ||
        !settings.max_body_bytes || !settings.max_body_total_bytes ||
        settings.max_output_bytes < 1024 || !settings.socket_timeout_seconds || !max_burst ||
        max_burst > static_cast<std::size_t>(std::numeric_limits<int>::max()) || settings.model.empty() ||
        (settings.embeddings && (!settings.embeddings->max_inputs || settings.embeddings->max_inputs > 256 ||
          settings.embeddings->max_batch_tokens < 3 || settings.embeddings->max_batch_tokens > 8192)) ||
        (images.prepare && (!images.max_image_tokens ||
          !images.prepared_bytes.count(images.default_max_soft_tokens) ||
          !images.prepared_bytes.at(images.default_max_soft_tokens))) ||
        (images.prepare_video_frame && (!images.prepared_bytes.count(embeddinggemma2::kVideoFrameBudget) ||
          !images.prepared_bytes.at(embeddinggemma2::kVideoFrameBudget))))
      throw std::invalid_argument("invalid HTTP bounds, port, model, or token burst (output minimum: 1024 bytes)");
    metrics::Writer initial(settings.model);
    initial.gauge("vllm:kv_cache_usage_perc", "Nonreclaimable allocated GPU cache pool fraction; 1 means full.", 0);
    observability = std::make_shared<const Observability>(Observability{
        settings.embeddings ? std::string{} : metrics::Snapshot{}.render(settings.model) + initial.str(),
        std::make_shared<const std::string>(nlohmann::json{
            {"instance_id", instance}, {"model_name", settings.model}, {"snapshot_time", nullptr},
            {"checkpoints", nlohmann::json::array()}, {"executions", nlohmann::json::array()}}.dump())});
    listener = ::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
    wake_fd = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    try {
      if (listener < 0 || wake_fd < 0) throw std::runtime_error("cannot create HTTP sockets");
      const int reuse = 1;
      if (::setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)))
        throw std::runtime_error("cannot configure HTTP listener");
      sockaddr_in address{};
      address.sin_family = AF_INET;
      address.sin_port = htons(settings.port);
      if (::inet_pton(AF_INET, settings.host.c_str(), &address.sin_addr) != 1)
        throw std::runtime_error("HTTP host must be an IPv4 address: " + settings.host);
      if (::bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)) ||
          ::listen(listener, static_cast<int>(settings.max_connections)))
        throw std::runtime_error("cannot bind/listen on HTTP " + settings.host + ":" +
                                 std::to_string(settings.port));
      io_thread = std::thread([this] { run_io(); });
      // Requests are prepared independently, so several can decode media at once.
      for (int i = 0; i < 4; ++i)
        worker_threads.emplace_back([this] { run_worker(); });
    } catch (...) {
      stopping = true;
      wake();
      work_available.notify_all();
      if (io_thread.joinable()) io_thread.join();
      for (auto& worker : worker_threads) worker.join();
      if (listener >= 0) ::close(listener);
      if (wake_fd >= 0) ::close(wake_fd);
      throw;
    }
  }

  ~Impl() {
    stop();
    ::close(listener);
    ::close(wake_fd);
  }

  void wake() const {
    const std::uint64_t one = 1;
    (void)::write(wake_fd, &one, sizeof(one));
  }

  void stop() {
    stopping = true;
    model_ready = false;
    work_available.notify_all();
    wake();
    if (io_thread.joinable()) io_thread.join();
    for (auto& worker : worker_threads) if (worker.joinable()) worker.join();
    std::lock_guard lock(mutex);
    jobs.clear();
    clients.clear();
    admissions.clear();
    body_bytes = 0;
  }

  std::shared_ptr<Client> find(ClientId id) {
    const auto it = clients.find(id);
    return it == clients.end() || !it->second->alive ? nullptr : it->second;
  }

  void queue_error(Client& client, Error error) {
    std::lock_guard lock(mutex);
    if (!client.alive) return;
    client.error = std::move(error);
    client.terminal = true;
  }

  void disconnect(Client& client, bool delivered = false) {
    if (client.socket >= 0) {
      ::close(client.socket);
      client.socket = -1;
    }
    std::string().swap(client.input);
    std::string().swap(client.wire);
    client.output.reset();
    client.request.reset();
    std::lock_guard lock(mutex);
    if (!client.alive) return;
    client.alive = false;
    if (delivered && client.admitted && client.success) {
      client.completion_pending = true;
      completions.push_back(client.id);
    } else if (client.admitted || client.worker_active) {
      client.cancellation_pending = true;
      cancellations.push_back(client.id);
    }
    if (!client.worker_active) {
      body_bytes -= client.body_charge;
      client.body_charge = 0;
      std::string().swap(client.body);
    }
    client.prepared.reset();
    client.tokens.clear();
    client.logprobs.clear();
    client.result.reset();
    client.embedding_result.reset();
    client.error.reset();
  }

  void append_wire(Client& client, std::string bytes) {
    if (client.sent) {
      client.wire.erase(0, client.sent);
      client.sent = 0;
    }
    if (bytes.size() > settings.max_output_bytes - client.wire.size())
      throw Error(500, "response exceeds configured output capacity", {}, "output_capacity_exceeded");
    if (client.wire.empty() && !bytes.empty()) client.last_write = Clock::now();
    client.wire += bytes;
  }

  void json_reply(Client& client, int status, std::string body, bool success,
                  std::string_view content_type = "application/json") {
    append_wire(client, headers(status, client.request_id, false, body.size(), content_type) + body);
    client.response_started = true;
    client.closing = true;
    client.success = success;
  }

  void error_reply(Client& client, const Error& error) {
    log_error(client.request_id, client.method, client.path, error, client.response_sent);
    const auto body = error_json(error).dump();
    if (!client.response_sent) {
      client.wire.clear();
      client.sent = 0;
      client.streaming = false;
      json_reply(client, error.status, body, false);
    } else if (client.streaming) {
      append_wire(client, chunk("data: " + body + "\n\n") + "0\r\n\r\n");
      client.closing = true;
      client.success = false;
    } else {
      disconnect(client);
    }
  }

  void parse_headers(Client& client, std::string_view block) {
    const auto first_end = block.find("\r\n");
    if (first_end == std::string_view::npos) throw Error(400, "malformed HTTP request line");
    const auto line = block.substr(0, first_end);
    const auto first_space = line.find(' '), last_space = line.rfind(' ');
    if (first_space == std::string_view::npos || first_space == last_space ||
        line.find(' ', first_space + 1) != last_space)
      throw Error(400, "malformed HTTP request line");
    const auto method = line.substr(0, first_space);
    const auto path = line.substr(first_space + 1, last_space - first_space - 1);
    const auto version = line.substr(last_space + 1);
    if (method.empty() || path.empty() || path.front() != '/' ||
        (version != "HTTP/1.0" && version != "HTTP/1.1"))
      throw Error(400, "unsupported HTTP request line");
    for (unsigned char c : method) if (!header_character(c)) throw Error(400, "invalid HTTP method");
    for (unsigned char c : path) if (c <= 32 || c == 127) throw Error(400, "invalid HTTP target");
    client.method = method;
    client.path = path;
    bool length_seen = false, host_seen = false;
    std::size_t length = 0, offset = first_end + 2, count = 0;
    while (offset < block.size()) {
      const auto end = block.find("\r\n", offset);
      if (end == std::string_view::npos) throw Error(400, "malformed HTTP header");
      const auto field = block.substr(offset, end - offset);
      offset = end + 2;
      if (field.empty()) break;
      if (++count > 128) throw Error(431, "too many HTTP headers");
      const auto colon = field.find(':');
      if (!colon || colon == std::string_view::npos) throw Error(400, "malformed HTTP header");
      std::string name(field.substr(0, colon));
      for (char& c : name) {
        if (!header_character(static_cast<unsigned char>(c))) throw Error(400, "invalid HTTP header name");
        if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
      }
      const auto value = trim_header(field.substr(colon + 1));
      for (unsigned char c : value)
        if ((c < 32 && c != '\t') || c == 127) throw Error(400, "invalid HTTP header value");
      if (name == "transfer-encoding") throw Error(400, "Transfer-Encoding request bodies are unsupported");
      if (name == "expect") throw Error(417, "Expect requests are unsupported");
      if (name == "content-encoding" && value != "identity") throw Error(415, "encoded request bodies are unsupported");
      if (name == "host") {
        if (host_seen || value.empty()) throw Error(400, "invalid or duplicate Host header");
        host_seen = true;
      }
      if (name == "content-length") {
        if (length_seen || value.empty()) throw Error(400, "invalid or duplicate Content-Length header");
        length_seen = true;
        const auto parsed = std::from_chars(value.data(), value.data() + value.size(), length);
        if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
          throw Error(400, "invalid Content-Length header");
      }
    }
    if (version == "HTTP/1.1" && !host_seen) throw Error(400, "HTTP/1.1 requires Host");
    if (method == "POST" && !length_seen) throw Error(411, "POST requires Content-Length");
    if (length > settings.max_body_bytes) throw Error(413, "request body is too large");
    {
      std::lock_guard lock(mutex);
      if (length > settings.max_body_total_bytes - body_bytes - media_bytes->load())
        throw Error(503, "aggregate request body capacity is exhausted", {}, "capacity_exceeded");
      body_bytes += length;
      client.body_charge = length;
    }
    client.body.resize(length);
    client.body_length = length;
    client.parsed_header = true;
  }

  void prepare(Client& client) {
    client.request_complete = true;
    client.arrival_time = Clock::now();
    if (client.method == "GET" && client.body.empty()) {
      try {
        auto request = parse(client.method, client.path, {});
        std::lock_guard lock(mutex);
        client.prepared = std::move(request);
      } catch (const Error& error) { queue_error(client, error); }
      return;
    }
    {
      std::lock_guard lock(mutex);
      client.worker_active = true;
      jobs.push_back(clients.at(client.id));
    }
    work_available.notify_one();
  }

  void read(Client& client) {
    if (client.closing) return;
    char buffer[4096];
    std::size_t budget = 65536;
    while (budget) {
      const auto received = ::recv(client.socket, buffer, std::min(sizeof(buffer), budget), MSG_DONTWAIT);
      if (received < 0 && errno == EINTR) continue;
      if (received < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return;
      if (received <= 0) { disconnect(client); return; }
      client.last_read = Clock::now();
      budget -= static_cast<std::size_t>(received);
      if (client.request_complete) { disconnect(client); return; }
      try {
        std::string_view incoming(buffer, received);
        if (!client.parsed_header) {
          client.input.append(incoming);
          const auto end = client.input.find("\r\n\r\n");
          for (std::size_t i = 0; i < (end == std::string::npos ? client.input.size() : end + 4); ++i)
            if (client.input[i] == '\n' && (i == 0 || client.input[i - 1] != '\r'))
              throw Error(400, "HTTP header lines require CRLF");
          if (end == std::string::npos) {
            if (client.input.size() > kHeaderBytes) throw Error(431, "HTTP headers are too large");
            continue;
          }
          if (end + 4 > kHeaderBytes) throw Error(431, "HTTP headers are too large");
          parse_headers(client, std::string_view(client.input).substr(0, end + 4));
          incoming = std::string_view(client.input).substr(end + 4);
        }
        if (incoming.size() > client.body_length - client.body_received)
          throw Error(400, "unexpected bytes after HTTP request body");
        std::copy(incoming.begin(), incoming.end(), client.body.begin() + client.body_received);
        client.body_received += incoming.size();
        std::string().swap(client.input);
        if (client.body_received == client.body_length) {
          prepare(client);
          return;
        }
      } catch (const Error& error) {
        client.request_complete = true;
        queue_error(client, error);
        return;
      } catch (const std::exception&) {
        client.request_complete = true;
        queue_error(client, Error(500, "HTTP request preparation failed"));
        return;
      }
    }
  }

  void run_worker() {
    for (;;) {
      std::shared_ptr<Client> client;
      std::string body;
      bool alive;
      {
        std::unique_lock lock(mutex);
        work_available.wait(lock, [&] { return stopping || !jobs.empty(); });
        if (stopping) return;
        client = std::move(jobs.front());
        jobs.pop_front();
        body.swap(client->body);
        alive = client->alive;
      }
      auto bounded_images = images;
      auto poll_preparation = [this, client] {
        std::lock_guard lock(mutex);
        if (stopping || !client->alive || client->terminal || client->error)
          throw Error(503, "media preparation cancelled", {}, "request_cancelled");
      };
      auto reserve_prepared = [this, client](std::size_t bytes, auto& lease) {
        lease.total = media_bytes;
        std::lock_guard lock(mutex);
        if (stopping || !client->alive || client->terminal || client->error)
          throw Error(503, "media preparation cancelled", {}, "request_cancelled");
        if (bytes > settings.max_body_total_bytes - body_bytes - media_bytes->load())
          throw Error(503, "aggregate media and request body capacity is exhausted", {}, "capacity_exceeded");
        media_bytes->fetch_add(bytes);
        lease.bytes = bytes;
      };
      auto prepare_bounded = [this, reserve_prepared, poll_preparation](std::uint32_t max_soft_tokens, auto prepare) {
        auto lease = std::make_shared<ImageLease>();
        reserve_prepared(images.prepared_bytes.at(max_soft_tokens),*lease);
        lease->image = prepare();
        if (!lease->image || lease->image->pixels.size() > lease->bytes ||
            lease->image->positions.size() > lease->bytes - lease->image->pixels.size())
          throw Error(500, "image processor exceeded its reserved tensor capacity", {}, "execution_failed");
        poll_preparation();
        return std::shared_ptr<runtime::ImageInput>(lease, lease->image.get());
      };
      bounded_images.poll_preparation = poll_preparation;
      if (images.prepare) bounded_images.prepare = [this, prepare_bounded](std::string_view url, std::uint32_t budget) {
        return prepare_bounded(budget, [&] { return images.prepare(url, budget); });
      };
      if (images.prepare_video_frame) bounded_images.prepare_video_frame = [this, prepare_bounded](gemma4::RgbImage frame) {
        return prepare_bounded(embeddinggemma2::kVideoFrameBudget,
            [&] { return images.prepare_video_frame(std::move(frame)); });
      };
      embeddinggemma2::audio::Acquire acquire_audio;
      if (settings.embeddings && settings.embeddings->audio) acquire_audio = [reserve_prepared, poll_preparation](std::uint32_t rows) {
        auto lease=std::make_shared<AudioLease>();
        const auto count=std::size_t(rows)*embeddinggemma2::audio::kFeatureWidth;
        reserve_prepared(count*sizeof(float)+rows,*lease);
        lease->input.features.reserve(count);lease->input.mask.reserve(rows);
        if (lease->input.features.capacity()*sizeof(float)+lease->input.mask.capacity()!=lease->bytes)
          throw Error(500,"audio processor exceeded its reserved tensor capacity",{},"execution_failed");
        poll_preparation();
        return std::shared_ptr<embeddinggemma2::audio::Input>(lease,&lease->input);
      };
      std::optional<Request> request;
      std::optional<Error> error;
      if (alive) {
        try {
          request = parse(client->method, client->path, body, bounded_images, acquire_audio);
          if (settings.verbose && request->prompt) {
            text::IncrementalTextDecoder decoder(tokenizer);
            std::string prompt;
            for (const auto token : *request->prompt) prompt += decoder.push(token);
            prompt += decoder.finish();
            console::event("server_request_prompt",
                {{"request_id", client->request_id}, {"path", client->path.substr(0, client->path.find('?'))},
                 {"prompt", prompt}},
                "Request " + client->request_id + " prompt: " + prompt);
          }
        }
        catch (const Error& problem) { error = problem; }
        catch (const std::exception&) { error = Error(500, "HTTP request preparation failed"); }
      }
      std::string().swap(body);
      {
        std::lock_guard lock(mutex);
        body_bytes -= client->body_charge;
        client->body_charge = 0;
        client->worker_active = false;
        if (client->alive && !client->terminal && !client->error) {
          client->prepared = std::move(request);
          if (error) { client->error = std::move(error); client->terminal = true; }
        }
      }
      wake();
    }
  }

  Request parse(std::string_view method, std::string_view path, std::string_view body,
                const ImageSupport& support = {}, const embeddinggemma2::audio::Acquire& acquire_audio = {}) {
    if (settings.embeddings)
      return parse_embedding_request(method, path, body, tokenizer, settings.model, compiler,
                                     *settings.embeddings, settings.max_output_bytes, support, acquire_audio);
    return parse_request(method, path, body, tokenizer, settings.model, compiler, support);
  }

  void dispatch(Client& client, Request request) {
    request.arrival_time = client.arrival_time;
    if (request.operation == Operation::metrics || request.operation == Operation::cache_index) {
      std::shared_ptr<const Observability> snapshot;
      {
        std::lock_guard lock(mutex);
        snapshot = observability;
      }
      if (request.operation == Operation::metrics) {
        metrics::Writer current(settings.model);
        current.gauge("gewell:ready", "Whether the model is ready to accept inference requests.", model_ready ? 1 : 0);
        json_reply(client, 200, snapshot->metrics + current.str(), true,
                   "text/plain; version=0.0.4; charset=utf-8");
      } else {
        json_reply(client, 200, *snapshot->cache_index, true);
      }
      return;
    }
    if (request.operation == Operation::health || request.operation == Operation::models ||
        request.operation == Operation::model) {
      const auto body = immediate_json(request, settings.model, started, model_ready).dump();
      json_reply(client, request.operation == Operation::health && !model_ready ? 503 : 200, body, true);
      return;
    }
    if (!model_ready) throw Error(503, "model is not ready", {}, "model_not_ready");
    client.request = std::move(request);
    if (client.request->operation == Operation::generate) {
      client.streaming = client.request->stream;
      client.response_id = (client.request->chat ? "chatcmpl-" : "cmpl-") + instance + "-" + std::to_string(client.id);
      client.output = std::make_unique<CompletionOutput>(tokenizer, *client.request,
          settings.model, client.response_id, client.created, settings.max_output_bytes);
    }
    std::lock_guard lock(mutex);
    if (!client.alive) return;
    client.admitted = client.admission_pending = true;
    admissions.push_back({client.id, *client.request});
  }

  void start_stream(Client& client) {
    if (!client.response_started) {
      append_wire(client, headers(200, client.request_id, true));
      append_wire(client, chunk(client.output->start()));
      client.response_started = true;
    }
  }

  void process(Client& client) {
    std::optional<Request> prepared;
    std::optional<Error> error;
    {
      std::lock_guard lock(mutex);
      if (!client.alive) return;
      error.swap(client.error);
      prepared.swap(client.prepared);
    }
    if (error) client.deferred_error = std::move(error);
    try {
      if (client.deferred_error) {
        if (!client.wire.empty() && client.response_sent) return;
        error_reply(client, *client.deferred_error);
        client.deferred_error.reset();
        return;
      }
      if (prepared && !client.closing) dispatch(client, std::move(*prepared));
      if (!client.wire.empty() || client.closing) return;
      std::vector<std::uint32_t> tokens;
      std::vector<TokenLogprobs> logprobs;
      std::optional<Result> result;
      std::optional<EmbeddingResult> embedding_result;
      {
        std::lock_guard lock(mutex);
        tokens.swap(client.tokens);
        logprobs.swap(client.logprobs);
        result.swap(client.result);
        embedding_result.swap(client.embedding_result);
      }
      if (!tokens.empty()) {
        if (!client.output) throw Error(500, "tokens received for a control request");
        if (client.streaming) start_stream(client);
        const auto body = client.output->push(tokens.data(), tokens.size(),
            logprobs.empty() ? nullptr : logprobs.data());
        if (client.streaming) append_wire(client, chunk(body));
        if (client.output->stop_matched()) {
          std::lock_guard lock(mutex);
          if (!client.stop_notified && !client.terminal) {
            client.stop_notified = client.stop_pending = true;
            stops.push_back(client.id);
          }
        }
      }
      if (result) {
        if (client.output) {
          if (client.streaming) {
            start_stream(client);
            append_wire(client, chunk(client.output->finish(*result)) + "0\r\n\r\n");
            client.closing = client.success = true;
          } else {
            json_reply(client, 200, client.output->finish(*result), true);
          }
          client.output.reset();
        } else {
          json_reply(client, 200, control_json(*client.request, *result).dump(), true);
        }
      }
      if (embedding_result) {
        json_reply(client, 200, embedding_json(*client.request, *embedding_result, settings.model).dump(), true);
        // The serialized response contains all remaining delivery state.
        // A slow reader must not retain completed inputs or their media leases.
        client.request.reset();
      }
      if (client.wire.empty()) {
        std::lock_guard lock(mutex);
        if (client.occupied && client.tokens.empty() && client.logprobs.empty() &&
            !client.result && !client.terminal)
          client.reserved = client.occupied = false;
      }
    } catch (const Error& problem) {
      queue_error(client, problem);
    } catch (const std::exception&) {
      queue_error(client, Error(500, "HTTP response serialization failed"));
    }
  }

  void write(Client& client) {
    if (client.wire.empty()) return;
    const auto size = std::min<std::size_t>(262144, client.wire.size() - client.sent);
    const auto sent = ::send(client.socket, client.wire.data() + client.sent, size, MSG_DONTWAIT | MSG_NOSIGNAL);
    if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) return;
    if (sent <= 0) { disconnect(client); return; }
    client.sent += static_cast<std::size_t>(sent);
    client.response_sent = true;
    client.last_write = Clock::now();
    if (client.sent != client.wire.size()) return;
    std::string().swap(client.wire);
    client.sent = 0;
    if (client.closing && !client.deferred_error) { disconnect(client, true); return; }
    std::lock_guard lock(mutex);
    if (client.occupied && client.tokens.empty() && !client.result && !client.terminal)
      client.reserved = client.occupied = false;
  }

  void accept() {
    for (unsigned count = 0; count < 16; ++count) {
      const int fd = ::accept4(listener, nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
      if (fd < 0 && errno == EINTR) continue;
      if (fd < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return;
      if (fd < 0) throw std::runtime_error("HTTP accept failed");
      std::lock_guard lock(mutex);
      const auto id = ++next_id;
      const auto request_id = "req-" + instance + "-" + std::to_string(id);
      if (clients.size() >= settings.max_connections) {
        const Error error(503, "HTTP connection capacity is exhausted", {}, "capacity_exceeded");
        log_error(request_id, {}, {}, error);
        const auto body = error_json(error).dump();
        const auto reply = headers(503, request_id, false, body.size()) + body;
        (void)::send(fd, reply.data(), reply.size(), MSG_DONTWAIT | MSG_NOSIGNAL);
        (void)::shutdown(fd, SHUT_WR);
        ::close(fd);
        continue;
      }
      const int one = 1, socket_bytes = 65536;
      if (::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) ||
          ::setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &socket_bytes, sizeof(socket_bytes)) ||
          ::setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &socket_bytes, sizeof(socket_bytes))) {
        ::close(fd);
        continue;
      }
      auto client = std::make_shared<Client>();
      client->id = id;
      client->socket = fd;
      client->created = std::time(nullptr);
      client->request_id = request_id;
      clients.emplace(client->id, std::move(client));
    }
  }

  void run_io() noexcept {
    try {
      while (!stopping) {
        std::vector<std::shared_ptr<Client>> active;
        {
          std::lock_guard lock(mutex);
          for (auto it = clients.begin(); it != clients.end();) {
            const auto& client = *it->second;
            if (!client.alive && !client.worker_active && !client.admission_pending &&
                !client.cancellation_pending && !client.completion_pending && !client.stop_pending) it = clients.erase(it);
            else { if (client.alive) active.push_back(it->second); ++it; }
          }
        }
        std::vector<pollfd> descriptors{{listener, POLLIN, 0}, {wake_fd, POLLIN, 0}};
        for (auto& client : active) {
          process(*client);
          descriptors.push_back({client->socket, static_cast<short>(POLLIN | POLLRDHUP |
              (client->wire.empty() ? 0 : POLLOUT)), 0});
        }
        const auto polled = ::poll(descriptors.data(), descriptors.size(), 100);
        if (polled < 0 && errno != EINTR) throw std::runtime_error("HTTP poll failed");
        if (descriptors[1].revents & POLLIN) { std::uint64_t value; (void)::read(wake_fd, &value, sizeof(value)); }
        if (descriptors[0].revents & POLLIN) accept();
        for (std::size_t index = 0; index < active.size(); ++index) {
          auto& client = *active[index];
          if (client.socket < 0) continue;
          const auto events = descriptors[index + 2].revents;
          if (events & (POLLERR | POLLNVAL | POLLHUP | POLLRDHUP)) { disconnect(client); continue; }
          if (events & POLLIN) read(client);
          if (client.socket >= 0 && (events & POLLOUT)) write(client);
          if (client.socket < 0) continue;
          const auto now = Clock::now();
          const auto timeout = std::chrono::seconds(settings.socket_timeout_seconds);
          if ((!client.request_complete && now - client.last_read >= timeout) ||
              (!client.wire.empty() && now - client.last_write >= timeout)) disconnect(client);
        }
      }
    } catch (...) {
      healthy = false;
      stopping = true;
      model_ready = false;
      work_available.notify_all();
    }
    std::vector<std::shared_ptr<Client>> remaining;
    { std::lock_guard lock(mutex); for (auto& entry : clients) remaining.push_back(entry.second); }
    for (auto& client : remaining) disconnect(*client);
  }
};

Server::Server(const text::Tokenizer& tokenizer, Settings settings, std::size_t max_token_burst, ImageSupport images)
    : impl_(std::make_unique<Impl>(tokenizer, std::move(settings), max_token_burst, std::move(images))) {}
Server::~Server() = default;
void Server::set_ready(bool ready) { impl_->model_ready = ready && impl_->healthy && !impl_->stopping; impl_->wake(); }
bool Server::healthy() const { return impl_->healthy && !impl_->stopping; }

std::vector<Admission> Server::take_admissions() {
  std::lock_guard lock(impl_->mutex);
  std::vector<Admission> result;
  for (auto& admission : impl_->admissions) {
    const auto it = impl_->clients.find(admission.client);
    if (it != impl_->clients.end()) {
      it->second->admission_pending = false;
      if (it->second->alive) result.push_back(std::move(admission));
    }
  }
  impl_->admissions.clear();
  impl_->wake();
  return result;
}

std::vector<ClientId> Server::take_cancellations() {
  std::lock_guard lock(impl_->mutex);
  auto result = std::move(impl_->cancellations);
  impl_->cancellations.clear();
  for (auto id : result) {
    const auto it = impl_->clients.find(id);
    if (it != impl_->clients.end()) it->second->cancellation_pending = false;
  }
  impl_->wake();
  return result;
}

std::vector<ClientId> Server::take_completions() {
  std::lock_guard lock(impl_->mutex);
  auto result = std::move(impl_->completions);
  impl_->completions.clear();
  for (auto id : result) {
    const auto it = impl_->clients.find(id);
    if (it != impl_->clients.end()) it->second->completion_pending = false;
  }
  impl_->wake();
  return result;
}

std::vector<ClientId> Server::take_stops() {
  std::lock_guard lock(impl_->mutex);
  std::vector<ClientId> result;
  for (auto id : impl_->stops) {
    const auto it = impl_->clients.find(id);
    if (it == impl_->clients.end()) continue;
    it->second->stop_pending = false;
    if (it->second->alive && !it->second->terminal) result.push_back(id);
  }
  impl_->stops.clear();
  impl_->wake();
  return result;
}

bool Server::ready(ClientId id) {
  std::lock_guard lock(impl_->mutex);
  const auto client = impl_->find(id);
  if (!client || !client->admitted || client->terminal || client->stop_pending) return false;
  if (client->occupied) return false;
  if (client->reserved) return true;
  client->reserved = true;
  return true;
}

void Server::emit(ClientId id, const std::uint32_t* tokens, std::size_t count,
                  const TokenLogprobs* logprobs) {
  std::lock_guard lock(impl_->mutex);
  const auto client = impl_->find(id);
  if (!client) return;
  const bool expected_logprobs = client->request && client->request->logprobs;
  if (!client->admitted || client->terminal || !client->reserved || client->occupied ||
      !client->tokens.empty() || !client->logprobs.empty() || !tokens || !count ||
      count > impl_->max_burst || expected_logprobs != (logprobs != nullptr))
    throw std::logic_error("HTTP token burst was not reserved or is malformed");
  client->tokens.assign(tokens, tokens + count);
  if (logprobs) client->logprobs.assign(logprobs, logprobs + count);
  client->occupied = true;
  impl_->wake();
}

void Server::finish(ClientId id, Result result) {
  std::lock_guard lock(impl_->mutex);
  const auto client = impl_->find(id);
  if (!client || client->terminal) return;
  client->reserved = client->occupied = client->terminal = true;
  client->result = std::move(result);
  impl_->wake();
}

void Server::finish_embeddings(ClientId id, EmbeddingResult result) {
  std::lock_guard lock(impl_->mutex);
  const auto client = impl_->find(id);
  if (!client || client->terminal) return;
  client->reserved = client->occupied = client->terminal = true;
  client->embedding_result = std::move(result);
  impl_->wake();
}

bool Server::cancelled(ClientId id) const {
  std::lock_guard lock(impl_->mutex);
  const auto client = impl_->find(id);
  return !client || client->terminal || impl_->stopping;
}

void Server::reject(ClientId id, Error error) {
  std::lock_guard lock(impl_->mutex);
  const auto client = impl_->find(id);
  if (!client) return;
  client->error = std::move(error);
  client->terminal = true;
  impl_->wake();
}

void Server::fail_all(Error error) {
  impl_->healthy = false;
  impl_->model_ready = false;
  std::lock_guard lock(impl_->mutex);
  for (auto& entry : impl_->clients) {
    auto& client = *entry.second;
    if (client.alive && !client.terminal) { client.error = error; client.terminal = true; }
  }
  impl_->wake();
}

void Server::stop() { impl_->stop(); }

void Server::publish_observability(std::string metrics, nlohmann::json cache_index) {
  std::shared_ptr<const std::string> inventory;
  if (!cache_index.is_null()) {
    cache_index["instance_id"] = impl_->instance;
    cache_index["model_name"] = impl_->settings.model;
    inventory = std::make_shared<const std::string>(cache_index.dump());
  }
  std::lock_guard lock(impl_->mutex);
  if (!inventory) inventory = impl_->observability->cache_index;
  impl_->observability = std::make_shared<const Impl::Observability>(
      Impl::Observability{std::move(metrics), std::move(inventory)});
}

}  // namespace gewell::http
