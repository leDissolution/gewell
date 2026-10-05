#include "gewell/runtime/mtp_stats.h"
#include "gewell/runtime/scheduler.h"

#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using namespace gewell::runtime;
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
struct TempDirectory {
  std::filesystem::path path;
  TempDirectory() {
    auto pattern = (std::filesystem::temp_directory_path() / "gewell-mtp-stats-XXXXXX").string();
    require(::mkdtemp(pattern.data()) != nullptr, "create temporary directory");
    path = pattern;
  }
  ~TempDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
};
std::vector<nlohmann::json> read(const std::filesystem::path& path) {
  std::ifstream file(path);
  std::vector<nlohmann::json> records;
  for (std::string line; std::getline(file, line);) records.push_back(nlohmann::json::parse(line));
  return records;
}
BatchRequest request(std::uint64_t sequence = 0) {
  BatchRequest r;
  r.id = "story-\"quoted\"";
  r.accepted_order = sequence;
  r.prompt = std::make_shared<const std::vector<std::uint32_t>>(25, 3);
  r.max_new_tokens = 4096;
  r.outputs = {5};
  r.seed = 123;
  return r;
}
void windows_and_lifecycle(const std::filesystem::path& path) {
  BatchLimits limits;
  limits.capacity = 16; limits.mtp_depth = 15;
  limits.mtp_stats_path = path; limits.mtp_stats_window = 2;
  MtpStats writer(limits);
  auto r = request();
  MtpStatsSample sample;
  sample.batch_id = 5; sample.batch_size = 2; sample.depth = 15; sample.accepted = 15;
  sample.output_begin = 1; sample.emitted = 16; sample.speculative = true;
  sample.wall_seconds = 0.012; sample.gpu_seconds = 0.01;
  sample.draft_seconds = 0.002; sample.verify_seconds = 0.006; sample.select_seconds = 0.001;
  writer.observe(r, sample);
  require(read(path).size() == 2, "partial window was emitted before the configured interval");
  // Same acceptance length at two depths must keep separate denominators.
  sample.batch_id = 7; sample.depth = 3; sample.accepted = 3;
  sample.output_begin = 17; sample.emitted = 4;
  writer.observe(r, sample);
  auto records = read(path);
  require(records.size() == 3 && records.back()["event"] == "mtp_window", "full window was not flushed");
  const auto& first = records.back();
  require(first["output_begin"] == 1 && first["output_end"] == 21 && first["cycles"] == 2 &&
          first["first_batch"] == 5 && first["last_batch"] == 7, "window boundaries changed");
  const auto& groups = first["groups"];
  require(groups.size() == 2 && groups[0]["depth"] == 3 && groups[1]["depth"] == 15 &&
          groups[0]["accepted"].size() == 4 && groups[1]["accepted"].size() == 16 &&
          groups[1]["accepted"][15] == 1, "depth fifteen or depth grouping lost bins");
  require(groups[1]["gpu_seconds"] == 0.01 && groups[1]["wall_seconds"] == 0.012 &&
          groups[1]["verify_gpu_seconds"] == 0.006, "complete-cycle and stage timing were conflated");
  sample.batch_id = 8; sample.depth = 0; sample.accepted = 0;
  sample.output_begin = 21; sample.emitted = 1; sample.batch_size = 1; sample.speculative = false;
  sample.draft_seconds = sample.verify_seconds = sample.select_seconds = 0;
  writer.observe(r, sample);
  r.outputs.resize(22);
  writer.finish(r, "complete");
  records = read(path);
  require(records.size() == 5 && records[3]["cycles"] == 1 && records[3]["window"] == 1 &&
          records[3]["groups"][0]["mtp"] == false && records[3]["groups"][0]["accepted"][0] == 1,
          "ordinary decode or the final partial window was dropped");
  require(records.back()["decode_cycles"] == 3 && records.back()["windows"] == 2 &&
          records.back()["status"] == "complete", "end record lost totals");
  // The same external ID can be reused; sequence disambiguates it and window state resets.
  r = request(1);
  writer.finish(r, "cancelled");
  records = read(path);
  require(records.size() == 7 && records.back()["sequence"] == 1 && records.back()["windows"] == 0 &&
          records.back()["decode_cycles"] == 0, "zero-decode cancellation leaked or inherited a window");
  r = request(2);
  sample.depth = 15; sample.accepted = 0; sample.emitted = 0; sample.speculative = true; sample.failed = true;
  writer.observe(r, sample);
  sample.failed = false; sample.accepted = 15; sample.emitted = 16;
  writer.observe(r, sample);
  writer.finish(r, "interrupted");
  records = read(path);
  const auto& failed_group = records[8]["groups"][0];
  require(records.size() == 10 && records.back()["windows"] == 1 &&
          failed_group["cycles"] == 2 && failed_group["failed_cycles"] == 1 &&
          failed_group["accepted"][0] == 0 && failed_group["accepted"][15] == 1,
          "failed cycle was treated as an accepted prefix or empty tail was written");
  require(records[0]["mtp_depth"] == 15 && records[0]["timing_scope"] == "shared_decode_batch" &&
          records[1]["id"] == r.id && records[1]["seed"] == 123,
          "configuration, identity, or sampling metadata missing");
  bool rejected = false;
  try { MtpStats duplicate(limits); } catch (const std::runtime_error&) { rejected = true; }
  require(rejected && read(path).size() == 10, "statistics allowed concurrent writers");
}
std::string contents(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(file), {}};
}
void resume(const std::filesystem::path& path) {
  BatchLimits limits; limits.mtp_stats_path = path; limits.mtp_stats_window = 1;
  {
    MtpStats writer(limits);
    writer.finish(request(31), "complete");
    writer.finish(request(7), "cancelled");
  }
  const auto original = contents(path);
  { std::ofstream file(path, std::ios::app); file << "{\"event\":\"mtp_request"; }
  limits.mtp_stats_window = 2; limits.mtp_depth = 15;
  {
    MtpStats writer(limits);
    require(writer.next_sequence() == 32, "statistics restart reused a request sequence");
    auto r = request(writer.next_sequence());
    MtpStatsSample sample; sample.batch_size = 1;
    writer.observe(r, sample); writer.finish(r, "complete");
  }
  const auto rows = read(path);
  require(rows.size() == 9 && rows[5]["event"] == "mtp_stats_config" && rows[5]["window_cycles"] == 2 &&
          rows[5]["mtp_depth"] == 15 && rows.back()["sequence"] == 32 && rows.back()["decode_cycles"] == 1,
          "statistics restart lost configuration or request state");
  require(contents(path).substr(0, original.size()) == original, "statistics restart modified prior records");
  {
    MtpStats writer(limits);
    require(writer.next_sequence() == 33, "second statistics restart failed");
  }
  { std::ofstream file(path, std::ios::app); file << "invalid json\n"; }
  const auto damaged = contents(path);
  bool rejected = false;
  try { MtpStats writer(limits); } catch (const std::exception&) { rejected = true; }
  require(rejected && contents(path) == damaged, "statistics restart erased a malformed complete record");
}
void missing_newline(const std::filesystem::path& path) {
  BatchLimits limits; limits.mtp_stats_path = path;
  { MtpStats writer(limits); writer.finish(request(31), "complete"); }
  const auto original = contents(path);
  std::filesystem::resize_file(path, original.size() - 1);
  MtpStats writer(limits);
  require(writer.next_sequence() == 32 && read(path).size() == 4 &&
          contents(path).substr(0, original.size()) == original,
          "complete final statistics without a newline were discarded");
}
void write_failure(const std::filesystem::path& path) {
  const auto pid = ::fork();
  require(pid >= 0, "fork write failure fixture");
  if (!pid) {
    BatchLimits limits; limits.mtp_stats_path = path; limits.mtp_stats_window = 1;
    MtpStats writer(limits);
    struct rlimit maximum{};
    if (::getrlimit(RLIMIT_FSIZE, &maximum) != 0) ::_exit(2);
    maximum.rlim_cur = 0;
    std::signal(SIGXFSZ, SIG_IGN);
    if (::setrlimit(RLIMIT_FSIZE, &maximum) != 0) ::_exit(3);
    auto r = request();
    MtpStatsSample sample;
    sample.batch_size = 1;
    writer.observe(r, sample);
    if (writer.enabled()) ::_exit(4);
    // Collection stays disabled; subsequent requests and finish must not throw.
    writer.observe(r, sample); writer.finish(r, "complete");
    ::_exit(0);
  }
  int status{};
  require(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "statistics write failure escaped into inference");
}
}
int main() {
  try {
    TempDirectory temp;
    windows_and_lifecycle(temp.path / "windows.jsonl");
    resume(temp.path / "resume.jsonl");
    missing_newline(temp.path / "missing-newline.jsonl");
    write_failure(temp.path / "failure.jsonl");
    BatchLimits limits; limits.mtp_stats_path = temp.path / "invalid.jsonl"; limits.mtp_stats_window = 0;
    bool rejected = false;
    try { MtpStats invalid(limits); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected && !std::filesystem::exists(limits.mtp_stats_path), "invalid window created an output file");
    limits.mtp_stats_window = 64; limits.mtp_stats_path = temp.path / "missing" / "stats.jsonl";
    rejected = false;
    try { MtpStats invalid(limits); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "unwritable statistics path was accepted");
    std::cout << "windowed MTP statistics tests passed\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
