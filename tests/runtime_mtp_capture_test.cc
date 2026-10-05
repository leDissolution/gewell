#include "gewell/runtime/mtp_capture.h"
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
using namespace gewell;
using namespace gewell::runtime;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
struct Directory {
  std::filesystem::path path;
  Directory() {
    auto pattern = (std::filesystem::temp_directory_path() / "gewell-mtp-capture-XXXXXX").string();
    require(::mkdtemp(pattern.data()), "create capture test directory"); path = pattern;
  }
  ~Directory() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};
std::vector<nlohmann::json> read(const std::filesystem::path& root) {
  std::ifstream file(root / "samples.jsonl");
  std::vector<nlohmann::json> rows;
  for (std::string line; std::getline(file, line);) rows.push_back(nlohmann::json::parse(line));
  return rows;
}
struct Fixture {
  BatchRequest request;
  BatchMtpInput input;
  MtpOutcome outcome;
  MtpCaptureFeatures features;
  Fixture() {
    request.id = "quoted-\"story\""; request.accepted_order = 31;
    request.prompt = std::make_shared<const std::vector<std::uint32_t>>(25, 1);
    request.outputs = {3}; request.max_new_tokens = 20;
    request.mtp_cycles = 4; request.mtp_previous_depth = 3; request.mtp_previous_accepted = 0;
    input.depth = 3; input.position = 25; input.pending_token = 3; input.capture = &features;
    features.target_width = 2; features.assistant_width = 1;
    features.target_hidden = {0x3f80, 0x4000}; features.assistant_hidden = {0x4040, 0x4080, 0x40a0};
    features.probes = {2, 25, 3, {4,12,24,40,56}, {11,12,13,14,15,16,17,18,19,20}};
    features.draft_tokens = {4,5,6}; features.scores.resize(3, {0.5, 0.25, 0.69, 0.5});
    outcome.verification = {3,4,3}; outcome.tokens = {4,5,6,7};
  }
};
void bounded_records(const std::filesystem::path& path) {
  BatchLimits limits; limits.mtp_depth = 3; limits.mtp_capture = {path, 1, 2};
  MtpCapture writer(limits);
  Fixture f;
  require(writer.select(31, 4), "first capture was not selected");
  writer.record(f.request, f.input, f.outcome, 6, 8, 1);
  require(writer.select(31, 5) && !writer.enabled() && !writer.select(31, 6), "capture exceeded cap");
  ++f.request.mtp_cycles;
  writer.record(f.request, f.input, f.outcome, 7, 7, 4);
  const auto rows = read(path);
  require(rows.size() == 3 && rows[1]["id"] == f.request.id, "capture identity lost or JSON invalid");
  require(rows[1]["accepted"] == 3 && rows[1]["emitted"] == 1 && rows[1]["eos_truncated"] == true,
          "raw acceptance was clipped to EOS");
  require(rows[1]["batch_size"] == 8 && rows[1]["previous_accepted"] == 0 && rows[1]["prior_cycles"] == 4,
          "capture lost scheduler context");
  require(rows[1]["byte_offset"] == 0 && rows[1]["byte_length"] == 30 && rows[2]["byte_offset"] == 30,
          "capture offsets overlap or skip data");
  require(std::filesystem::file_size(path / "hidden.bf16") == 60, "capture binary size changed");
  std::ifstream payload(path / "hidden.bf16", std::ios::binary);
  std::vector<std::uint16_t> bits(30); payload.read(reinterpret_cast<char*>(bits.data()), 60);
  require(bits == std::vector<std::uint16_t>({0x3f80,0x4000,0x4040,0x4080,0x40a0,11,12,13,14,15,16,17,18,19,20,
      0x3f80,0x4000,0x4040,0x4080,0x40a0,11,12,13,14,15,16,17,18,19,20}),
          "BF16 bits or target/assistant ordering changed");
  bool rejected = false;
  try { MtpCapture duplicate(limits); } catch (const std::exception&) { rejected = true; }
  require(rejected && read(path).size() == 3, "capture allowed concurrent writers");
}
std::string contents(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(file), {}};
}
void resume(const std::filesystem::path& path) {
  BatchLimits limits; limits.mtp_depth = 3; limits.mtp_capture = {path, 1, 2};
  Fixture f;
  // Existing empty directories are also valid destinations.
  std::filesystem::create_directory(path);
  {
    MtpCapture writer(limits);
    require(writer.next_sequence() == 0, "new capture advanced request sequence");
    writer.select(31, 4); writer.record(f.request, f.input, f.outcome, 6, 8, 4);
    f.request.accepted_order = 7;  // Completion order need not match admission order.
    writer.select(7, 4); writer.record(f.request, f.input, f.outcome, 7, 8, 4);
    require(!writer.enabled(), "fixture did not exhaust the per-launch cap");
  }
  const auto original_index = contents(path / "samples.jsonl");
  const auto original_data = contents(path / "hidden.bf16");
  // Simulate a crash after flushing payload but during the index write.
  { std::ofstream tail(path / "samples.jsonl", std::ios::app); tail << "{\"event\":\"mtp_capture\""; }
  { std::ofstream tail(path / "hidden.bf16", std::ios::binary | std::ios::app); tail << "orphaned payload"; }
  limits.capacity = 16; limits.decode_width = 192; limits.mtp_capture.max_samples = 1;
  {
    MtpCapture writer(limits);
    require(writer.next_sequence() == 32, "resume reused an existing request sequence");
    f.request.accepted_order = writer.next_sequence(); f.request.id = "restarted";
    require(writer.select(32, 4) && !writer.enabled(), "sample limit was not reset for this launch");
    writer.record(f.request, f.input, f.outcome, 1, 1, 4);
  }
  const auto rows = read(path);
  require(rows.size() == 5 && rows[3]["event"] == "mtp_capture_config" && rows[3]["sample_start"] == 2 &&
          rows[3]["decode_width"] == 192 && rows[3]["max_samples"] == 1,
          "resume lost configuration or sample count");
  require(rows[4]["sample"] == 2 && rows[4]["byte_offset"] == 60 && rows[4]["sequence"] == 32,
          "resumed sample overlaps existing data or identity");
  require(contents(path / "samples.jsonl").substr(0, original_index.size()) == original_index &&
          contents(path / "hidden.bf16") == original_data + original_data.substr(0, 30),
          "resume changed completed samples or retained orphaned bytes");
  // A second clean restart must handle the intervening configuration record.
  MtpCapture again(limits);
  require(again.next_sequence() == 33 && again.enabled(), "second restart could not continue");
}
void reject_damaged_resume(const std::filesystem::path& root) {
  for (const std::string damage : {"depth", "layers", "payload", "json", "offset"}) {
    const auto path = root / damage;
    BatchLimits limits; limits.mtp_depth = 3; limits.mtp_capture = {path, 1, 2};
    {
      MtpCapture writer(limits); Fixture f;
      writer.select(31, 4); writer.record(f.request, f.input, f.outcome, 6, 8, 4);
    }
    if (damage == "depth") limits.mtp_depth = 4;
    if (damage == "layers") limits.mtp_capture.layers = {4,12};
    if (damage == "payload") std::filesystem::resize_file(path / "hidden.bf16", 29);
    if (damage == "json") { std::ofstream tail(path / "samples.jsonl", std::ios::app); tail << "invalid json\n"; }
    if (damage == "offset") {
      auto rows = read(path); rows.back()["byte_offset"] = 1;
      std::ofstream file(path / "samples.jsonl");
      for (const auto& row : rows) file << row.dump() << '\n';
    }
    const auto index = contents(path / "samples.jsonl"), data = contents(path / "hidden.bf16");
    bool rejected = false;
    try { MtpCapture writer(limits); } catch (const std::exception&) { rejected = true; }
    require(rejected && contents(path / "samples.jsonl") == index && contents(path / "hidden.bf16") == data,
            "invalid resume modified existing capture");
  }
}
void incomplete_config(const std::filesystem::path& path) {
  std::filesystem::create_directory(path);
  { std::ofstream file(path / "samples.jsonl"); file << "{\"event\":"; }
  BatchLimits limits; limits.mtp_depth = 3; limits.mtp_capture = {path, 1, 2};
  MtpCapture writer(limits);
  require(read(path).size() == 1 && read(path)[0]["event"] == "mtp_capture_config" && writer.enabled(),
          "interrupted configuration could not be resumed");
}
void missing_newline(const std::filesystem::path& path) {
  BatchLimits limits; limits.mtp_depth = 3; limits.mtp_capture = {path, 1, 2};
  Fixture f;
  {
    MtpCapture writer(limits);
    writer.select(31, 4); writer.record(f.request, f.input, f.outcome, 1, 1, 4);
  }
  const auto original_index = contents(path / "samples.jsonl");
  const auto original_data = contents(path / "hidden.bf16");
  std::filesystem::resize_file(path / "samples.jsonl", original_index.size() - 1);
  MtpCapture writer(limits);
  require(writer.next_sequence() == 32 && read(path).size() == 3 &&
          contents(path / "samples.jsonl").substr(0, original_index.size()) == original_index &&
          contents(path / "hidden.bf16") == original_data,
          "complete final capture without a newline was discarded");
}
void sampling(const std::filesystem::path& path) {
  BatchLimits limits; limits.mtp_depth = 15; limits.mtp_capture = {path, 32, 10000};
  MtpCapture writer(limits);
  unsigned selected = 0, first = 0, last = 0;
  for (unsigned request = 0; request < 500; ++request)
    for (unsigned cycle = 0; cycle < 100; ++cycle) {
      const auto take = writer.select(request, cycle);
      selected += take; if (!cycle) first += take; if (cycle == 99) last += take;
    }
  require(selected > 1300 && selected < 1900 && first > 5 && last > 5, "sampling is biased by cycle position");
}
void failure(const std::filesystem::path& path) {
  const auto pid = ::fork(); require(pid >= 0, "fork capture failure fixture");
  if (!pid) {
    BatchLimits limits; limits.mtp_depth = 3; limits.mtp_capture = {path, 1, 10};
    MtpCapture writer(limits); Fixture f;
    writer.select(31, 4);
    f.outcome.error = "failed verification";
    writer.record(f.request, f.input, f.outcome, 0, 1, 0);
    if (read(path).size() != 1 || std::filesystem::file_size(path / "hidden.bf16")) ::_exit(2);
    struct rlimit maximum{};
    if (::getrlimit(RLIMIT_FSIZE, &maximum)) ::_exit(3);
    maximum.rlim_cur = 0; std::signal(SIGXFSZ, SIG_IGN);
    if (::setrlimit(RLIMIT_FSIZE, &maximum)) ::_exit(4);
    f.outcome.error.clear(); writer.select(31, 5);
    writer.record(f.request, f.input, f.outcome, 1, 1, 4);
    if (writer.enabled() || writer.select(31, 6)) ::_exit(5);
    writer.record(f.request, f.input, f.outcome, 2, 1, 4);
    ::_exit(0);
  }
  int status{};
  require(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) && !WEXITSTATUS(status),
          "write failure escaped into inference or emitted a failed round");
}
}
int main() {
  try {
    Directory directory;
    bounded_records(directory.path / "bounded");
    resume(directory.path / "resume");
    std::filesystem::create_directory(directory.path / "damaged");
    reject_damaged_resume(directory.path / "damaged");
    incomplete_config(directory.path / "incomplete-config");
    missing_newline(directory.path / "missing-newline");
    sampling(directory.path / "sampling");
    failure(directory.path / "failure");
    BatchLimits limits; limits.mtp_capture.path = directory.path / "invalid";
    bool rejected = false;
    try { MtpCapture writer(limits); } catch (const std::exception&) { rejected = true; }
    require(rejected && !std::filesystem::exists(limits.mtp_capture.path), "disabled MTP capture accepted");
    std::cout << "MTP capture tests passed\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
