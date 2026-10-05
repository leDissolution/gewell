#pragma once

#include "json.hpp"
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gewell::runtime::mtp_file {
struct Close { void operator()(std::FILE* file) const { std::fclose(file); } };
using File = std::unique_ptr<std::FILE, Close>;

inline void fail(const std::string& path, const std::string& error) {
  throw std::runtime_error("MTP output " + path + ": " + error);
}

inline File open(const std::string& path, const char* mode = "a+b") {
  File file(std::fopen(path.c_str(), mode));
  if (!file) fail(path, std::strerror(errno));
  // An append writer must own the entire JSON/payload transaction, not merely
  // individual writes. Closing the file releases this process-crash-safe lock.
  if (::flock(::fileno(file.get()), LOCK_EX | LOCK_NB))
    fail(path, "cannot lock output (another writer may be using it): " + std::string(std::strerror(errno)));
  return file;
}

struct Scan {
  std::uint64_t bytes{}, next_sequence{};
  bool needs_newline{};
};

// Read metadata only. Leave incomplete tails untouched until the caller has
// checked the format and, for capture, the indexed binary extent.
template <class Visit>
Scan scan(const std::string& path, Visit visit) {
  std::ifstream input(path);
  if (!input) fail(path, "cannot read index");
  Scan result;
  for (std::string line; std::getline(input, line);) {
    try {
      nlohmann::json record;
      try {
        record = nlohmann::json::parse(line);
      } catch (const nlohmann::json::parse_error&) {
        if (input.eof()) break;  // Only an unfinished JSON tail can be discarded.
        throw;
      }
      visit(record);
      if (record.contains("sequence")) {
        const auto& sequence = record.at("sequence");
        if (!sequence.is_number_unsigned() || sequence == std::numeric_limits<std::uint64_t>::max())
          throw std::runtime_error("invalid request sequence");
        result.next_sequence = std::max(result.next_sequence, sequence.get<std::uint64_t>() + 1);
      }
    } catch (const std::exception& error) {
      fail(path, "invalid record at byte " + std::to_string(result.bytes) + ": " + error.what());
    }
    result.needs_newline = input.eof();
    result.bytes += line.size() + !result.needs_newline;
  }
  if (input.bad()) fail(path, "read failed");
  return result;
}

inline std::uint64_t size(std::FILE* file, const std::string& path) {
  struct stat info{};
  if (::fstat(::fileno(file), &info)) fail(path, std::strerror(errno));
  return info.st_size;
}

inline void trim(std::FILE* file, const std::string& path, std::uint64_t bytes) {
  if (size(file, path) != bytes && ::ftruncate(::fileno(file), bytes)) fail(path, std::strerror(errno));
  if (::fseeko(file, 0, SEEK_END)) fail(path, std::strerror(errno));
}

inline void resume(std::FILE* file, const std::string& path, const Scan& existing) {
  trim(file, path, existing.bytes);
  // A complete JSON object can survive even if its final newline did not.
  if (existing.needs_newline && (std::fputc('\n', file) == EOF || std::fflush(file)))
    fail(path, std::strerror(errno));
}
}  // namespace gewell::runtime::mtp_file
