#include "gewell/models/gemma4/text_contract.h"
#include "gewell/models/gemma4/31b/model.h"
#include "gewell/http_server.h"
#include "gewell/chat_codec.h"
#include "gewell/console.h"
#include "gewell/metrics.h"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <iostream>
#include <map>
#include <poll.h>
#include <set>
#include <thread>

namespace {
const auto& contract = gewell::gemma4::text_contract_31b();

volatile std::sig_atomic_t interrupted = 0;
void interrupt(int) { interrupted = 1; }

struct Pending {
  gewell::http::Request request;
  std::vector<std::uint32_t> script;
  std::uint32_t seen = 0, cached = 0;
  bool terminal = false, decoded = false, stopped = false;
  std::chrono::steady_clock::time_point submitted = std::chrono::steady_clock::now();
};

}  // namespace

// CPU transport fixture. The real HTTP parser, tokenizer, output codec, and
// threaded transport run against deterministic execution outcomes. Raw prompts
// beginning 42 fail before output, 43 fail after one token, and 44 produce
// newline bursts to exercise slow readers. Prompt 45 delays its first decision
// three seconds, modeling prefill without network activity. Stdin commands
// ready/unready/fail/stop control readiness and whole-server failure.
// Prompts containing fixture:tool:NAME select exact Gemma output sequences;
// startup logs record their selected-token boundaries for truncation tests.
int main(int argc, char** argv) {
  gewell::console::set_format("json");
  if (argc != 3 && argc != 8) {
    std::cerr << "usage: http_server_test TOKENIZER_DIRECTORY PORT "
                 "[MAX_CONNECTIONS MAX_BODY_BYTES MAX_TOTAL_BODY_BYTES MAX_OUTPUT_BYTES TIMEOUT_SECONDS]\n";
    return 1;
  }
  try {
    const gewell::http::Settings defaults;
    if (defaults.host != "127.0.0.1" || defaults.port != 6311)
      throw std::runtime_error("unexpected HTTP bind defaults");
    const auto tokenizer_path = std::filesystem::path(argv[1]) / "tokenizer.json";
    if (!std::filesystem::is_regular_file(tokenizer_path)) {
      std::cout << "SKIP: local tokenizer unavailable\n";
      return 77;
    }
    gewell::http::Settings settings;
    // Bind beyond loopback while clients connect through loopback, exercising
    // the configurable address without exposing a fixed test port.
    settings.host = "0.0.0.0";
    settings.model = "gewell-gemma-4-31b-bf16";
    settings.port = static_cast<std::uint16_t>(std::stoul(argv[2]));
    if (argc == 8) {
      settings.max_connections = std::stoull(argv[3]);
      settings.max_body_bytes = std::stoull(argv[4]);
      settings.max_body_total_bytes = std::stoull(argv[5]);
      settings.max_output_bytes = std::stoull(argv[6]);
      settings.socket_timeout_seconds = std::stoul(argv[7]);
    }
    const gewell::text::Tokenizer tokenizer(tokenizer_path.string(), contract);
    constexpr std::size_t burst = 64;
    const auto ordinary = tokenizer.encode(" Hello");
    const auto long_piece = tokenizer.encode(std::string(31, '\n')).front();
    const std::string call = "<|tool_call>call:echo{value:<|\"|>café ☃ \"quoted\"<|\"|>,"
        "nested:{items:[1,true,null],label:<|\"|>line\nnext<|\"|>}}<tool_call|>";
    const std::map<std::string, std::string> script_text{
        {"tool:one", call + "<|tool_response>"},
        {"tool:multiple", call + "<|tool_call>call:echo{value:<|\"|>second<|\"|>}<tool_call|><|tool_response>"},
        {"tool:mixed", "<|channel>thought\nprivate reasoning<channel|>Visible answer. " + call + "<|tool_response>"},
        {"tool:plain", "READY<turn|>"},
        {"tool:unknown", "<|tool_call>call:undeclared{}<tool_call|><|tool_response>"},
        {"tool:malformed", "<|tool_call>call:echo{value:]}<tool_call|><|tool_response>"},
        {"tool:escaped", R"gemma(<|tool_call>call:echo{value:"line\n\"quoted\"\\end"}<tool_call|><|tool_response>)gemma"},
        {"tool:handoff", "<|tool_response>"},
        {"tool:strict", "<|tool_call>call:echo{\"value\":\"café ☃\"}<tool_call|><|tool_response>"},
        {"tool:strict_multiple", "<|tool_call>call:echo{\"value\":\"café ☃\"}<tool_call|><|tool_call>call:echo{\"value\":\"second\"}<tool_call|><|tool_response>"},
        {"tool:strict_invalid", "<|tool_call>call:echo{\"value\":42}<tool_call|><|tool_response>"},
        {"tool:empty", "<|tool_call>call:empty{}<tool_call|><|tool_response>"},
        {"tool:kilo", R"gemma(<|tool_call>call:todowrite{"todos":[{"content":"Check tools","status":"pending","priority":"high"}]}<tool_call|><|tool_response>)gemma"},
        {"stop:split", "before αβSTOPafter<turn|>"},
        {"schema:valid", "{\"answer\":\"blue\"}<turn|>"},
        {"schema:thinking", "<|channel>thought\nConsider the colors.<channel|>{\"answer\":\"blue\"}<turn|>"},
    };
    std::map<std::string, std::vector<std::uint32_t>> scripts;
    for (const auto& [name, text] : script_text) {
      auto tokens = tokenizer.encode(text);
      const auto close = std::find(tokens.begin(), tokens.end(), 49);
      gewell::text::IncrementalTextDecoder decoder(tokenizer);
      std::string decoded;
      std::vector<std::string> prefixes;
      for (const auto token : tokens) {
        decoded += decoder.push(token);
        prefixes.push_back(decoded);
      }
      std::cout << "http_test_script: " << nlohmann::json{
          {"name", name}, {"tokens", tokens.size()},
          {"call_close", close == tokens.end() ? 0 : close - tokens.begin() + 1},
          {"decoded_prefixes", prefixes}}
                    .dump() << '\n';
      scripts.emplace(name, std::move(tokens));
    }
    gewell::http::ImageSupport images;
    images.begin_token = gewell::gemma4_31b::kBeginImageTokenId;
    images.image_token = gewell::gemma4_31b::kImageTokenId;
    images.end_token = gewell::gemma4_31b::kEndImageTokenId;
    images.max_image_tokens = 1120;
    images.prepared_bytes = 16384;
    images.prepare = [](std::string_view url) {
      // Deliberately tiny synthetic processor output isolates transport leases
      // from decoder correctness, covered by the native processor fixtures.
      if (url == "data:image/png;base64,slow") {
        std::cout << "http_test_image_preparing: true\n" << std::flush;
        std::this_thread::sleep_for(std::chrono::seconds(1));
      } else if (url == "data:image/png;base64,second") {
        std::cout << "http_test_second_image_preparing: true\n" << std::flush;
      } else if (url != "data:image/png;base64,fixture") {
        throw std::invalid_argument("invalid fixture image");
      }
      auto image = std::make_shared<gewell::runtime::ImageInput>();
      image->pixels.resize(16376);
      image->positions.resize(8);
      image->padded_patch_rows = 1;
      image->end = 1;
      return image;
    };
    gewell::http::Server server(tokenizer, settings, burst, std::move(images));
    std::map<gewell::http::ClientId, Pending> requests;
    std::set<std::vector<std::uint32_t>> retained;
    std::set<std::string> owners;
    std::signal(SIGINT, interrupt);
    std::signal(SIGTERM, interrupt);
    server.set_ready(true);
    std::cout << "http_test_ready: true\n" << std::flush;
    bool input_open = true, failed = false;
    while (!interrupted && (server.healthy() || failed)) {
      pollfd control{0, POLLIN, 0};
      if (input_open && ::poll(&control, 1, 0) > 0 && (control.revents & (POLLIN | POLLHUP))) {
        std::string command;
        if (!std::getline(std::cin, command)) input_open = false;
        else {
          if (command == "ready") server.set_ready(true);
          if (command == "unready") server.set_ready(false);
          if (command == "stop") interrupted = 1;
          if (command == "fail") {
            failed = true;
            server.fail_all(gewell::http::Error(500, "injected whole-server failure", {}, "execution_failed"));
          }
          if (command == "publish" || command == "publish-metrics") {
            gewell::metrics::Snapshot snapshot;
            snapshot.prompt_tokens = command == "publish" ? 12 : 20;
            snapshot.computed_prompt_tokens = snapshot.prompt_tokens - 5;
            snapshot.cached_prompt_tokens = 5;
            snapshot.shared_prompt_tokens = 2;
            snapshot.generation_tokens = 4;
            snapshot.success_length = 1;
            snapshot.mtp_depth = 3;
            snapshot.mtp_rounds = 2;
            snapshot.mtp_draft_tokens = 5;
            snapshot.mtp_accepted_tokens = 3;
            snapshot.mtp_accepted_per_position = {2, 1, 0};
            snapshot.ttft.observe(0.04);
            snapshot.ttft.observe(0.5);
            snapshot.e2e.observe(1.5);
            snapshot.queue.observe(0.01);
            gewell::metrics::Writer cache(settings.model);
            cache.gauge("vllm:kv_cache_usage_perc", "Cache pressure.", 0.25);
            server.publish_observability(snapshot.render(settings.model) + cache.str(),
                command == "publish" ? nlohmann::json{
                    {"snapshot_time", 1700000000.5}, {"checkpoints", nlohmann::json::array({{
                        {"id", "18446744073709551615"}, {"tokens", 128}, {"reuse_count", 2}}})},
                    {"executions", nlohmann::json::array()}} : nlohmann::json(nullptr));
          }
          std::cout << "http_test_command: " << command << '\n' << std::flush;
          if (command == "block") std::this_thread::sleep_for(std::chrono::seconds(3));
        }
      }
      for (auto id : server.take_cancellations()) {
        const auto found = requests.find(id);
        std::cout << "http_test_cancelled: " << id << " tokens="
                  << (found == requests.end() ? 0 : found->second.seen) << '\n' << std::flush;
        requests.erase(id);
      }
      for (auto id : server.take_completions()) {
        std::cout << "http_test_completed: " << id << '\n' << std::flush;
        requests.erase(id);
      }
      for (auto& admission : server.take_admissions()) {
        Pending pending{std::move(admission.request)};
        if (pending.request.arrival_time == std::chrono::steady_clock::time_point{} ||
            pending.request.arrival_time > pending.submitted)
          throw std::runtime_error("HTTP admission did not preserve arrival timestamp");
        if (pending.request.prompt) {
          gewell::text::IncrementalTextDecoder decoder(tokenizer);
          std::string prompt_text;
          for (const auto token : *pending.request.prompt) prompt_text += decoder.push(token);
          prompt_text += decoder.finish();
          for (const auto& [name, tokens] : scripts)
            if (prompt_text.find("fixture:" + name) != std::string::npos) pending.script = tokens;
          for (const auto& prefix : retained) {
            if (!pending.request.images.empty()) break;
            if (prefix.size() <= pending.request.prompt->size() &&
                std::equal(prefix.begin(), prefix.end(), pending.request.prompt->begin()))
              pending.cached = std::max<std::uint32_t>(pending.cached, prefix.size());
          }
        }
        requests.emplace(admission.client, std::move(pending));
        std::cout << "http_test_admitted: " << admission.client << '\n' << std::flush;
      }
      for (const auto id : server.take_stops()) {
        const auto found = requests.find(id);
        if (found != requests.end()) found->second.decoded = found->second.stopped = true;
      }
      for (auto& [id, pending] : requests) {
        if (pending.terminal) continue;
        const auto& request = pending.request;
        const auto trigger = request.prompt && !request.prompt->empty() ? request.prompt->front() : 0;
        if ((trigger == 45 || !request.images.empty()) && std::chrono::steady_clock::now() - pending.submitted < std::chrono::seconds(3))
          continue;
        if (request.operation == gewell::http::Operation::generate &&
            (trigger == 42 || (trigger == 43 && pending.seen))) {
          server.reject(id, gewell::http::Error(500, "injected CPU execution failure", {}, "execution_failed"));
          pending.terminal = true;
          continue;
        }
        if (!server.ready(id)) continue;
        // Scheduler reservations may be queried repeatedly before a decision.
        if (!server.ready(id)) throw std::runtime_error("output reservation is not idempotent");
        gewell::http::Result result;
        if (request.prompt) {
          result.prompt_tokens = request.prompt->size();
          result.processed_tokens = request.prompt->size();
          result.cached_tokens = pending.cached;
        }
        if (request.operation == gewell::http::Operation::generate) {
          if (!pending.decoded) {
            const auto count = std::min<std::uint32_t>(trigger == 44 && request.stops.empty() ? burst : 1,
                                                      request.max_tokens - pending.seen);
            std::vector<std::uint32_t> tokens(count);
            for (std::uint32_t i = 0; i < count; ++i)
              tokens[i] = !pending.script.empty() ? pending.script.at(pending.seen + i) :
                  trigger == 44 ? long_piece : ordinary[(pending.seen + i) % ordinary.size()];
            std::vector<gewell::TokenLogprobs> logprobs;
            if (request.logprobs) {
              logprobs.resize(count);
              for (std::uint32_t row = 0; row < count; ++row) {
                auto& scores = logprobs[row];
                scores.logprob = -0.25F;
                scores.count = request.top_logprobs;
                for (std::uint32_t rank = 0; rank < scores.count; ++rank) {
                  scores.top[rank].token = (tokens[row] + rank) % 262144;
                  scores.top[rank].logprob = -0.25F - rank;
                }
              }
            }
            server.emit(id, tokens.data(), tokens.size(),
                logprobs.empty() ? nullptr : logprobs.data());
            pending.seen += count;
            pending.stopped = tokens.back() == 1 || tokens.back() == 106 || tokens.back() == 50;
            pending.decoded = pending.stopped || pending.seen == request.max_tokens;
            // A stop request needs I/O parsing feedback before another token or
            // its terminal result, including a match on the last budget token.
            if (!pending.decoded || !request.stops.empty()) continue;
          }
          result.stopped = pending.stopped;
          result.completion_tokens = pending.seen;
          result.processed_tokens += pending.seen - 1;
          if (request.images.empty() && !request.cache.reuse_only) retained.insert(*request.prompt);
          if (!request.cache.prompt_id.empty()) {
            if (request.cache.finished) owners.erase(request.cache.prompt_id);
            else owners.insert(request.cache.prompt_id);
          }
          std::cout << "http_test_result: " << id << " tokens=" << pending.seen
                    << " stopped=" << (pending.stopped ? "true" : "false")
                    << " owners=" << owners.size() << '\n' << std::flush;
        } else if (request.operation == gewell::http::Operation::prefill) {
          retained.insert(*request.prompt);
          if (!request.cache.prompt_id.empty()) owners.insert(request.cache.prompt_id);
          result.input_checkpoint = result.completion_checkpoint = retained.size();
        } else if (request.operation == gewell::http::Operation::finish) {
          owners.erase(request.cache.prompt_id);
        } else if (request.operation == gewell::http::Operation::stats) {
          result.cache_stats.checkpoint_count = retained.size();
          result.cache_stats.page_count = retained.size();
          result.cache_stats.execution_count = std::count_if(requests.begin(), requests.end(),
              [](const auto& entry) {
                return !entry.second.terminal && entry.second.request.operation == gewell::http::Operation::generate;
              });
        }
        // Exercise result and the final token burst sharing one reserved slot.
        server.finish(id, std::move(result));
        pending.terminal = true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    server.stop();
    std::cout << "http_test_stopped: true\n" << std::flush;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "http test fixture: " << error.what() << '\n';
    return 1;
  }
}
