#pragma once

#include "gewell/http_api.h"

namespace gewell::http {

struct Settings {
  std::string host = "127.0.0.1";
  std::uint16_t port = 6311;
  std::string model;
  std::size_t max_connections = 64;
  std::size_t max_body_bytes = 8 * 1024 * 1024;
  std::size_t max_body_total_bytes = 256 * 1024 * 1024;
  // Per connection, including buffered text and pending response bytes.
  std::size_t max_output_bytes = 8 * 1024 * 1024;
  std::uint32_t socket_timeout_seconds = 60;
  bool verbose = false;
};

struct Admission { ClientId client; Request request; };

// One network thread and one request-preparation worker; the GPU owner only
// exchanges bounded admissions, token bursts, outcomes and lifecycle events.
class Server {
 public:
  Server(const text::Tokenizer& tokenizer, Settings settings,
         std::size_t max_token_burst, ImageSupport images = {});
  ~Server();
  void set_ready(bool ready);
  // Publish serialized observations without exposing mutable runtime state to
  // HTTP threads. A null index retains the last inventory and its timestamp.
  void publish_observability(std::string metrics, nlohmann::json cache_index);
  bool healthy() const;
  std::vector<Admission> take_admissions();
  std::vector<ClientId> take_cancellations();
  std::vector<ClientId> take_completions();
  // The decoder found a string stop. Acknowledge it as normal scheduler
  // completion before reserving another output decision for this request.
  std::vector<ClientId> take_stops();
  // Reserve one burst/result slot; false applies backpressure to this request.
  bool ready(ClientId client);
  void emit(ClientId client, const std::uint32_t* tokens, std::size_t count,
            const TokenLogprobs* logprobs = nullptr);
  void finish(ClientId client, Result result);
  void reject(ClientId client, Error error);
  // Mark unhealthy and fail pending requests (JSON before SSE, error after it).
  void fail_all(Error error);
  void stop();
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gewell::http
