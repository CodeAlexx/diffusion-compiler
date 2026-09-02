#pragma once

// Host-side facts a benchmark must state instead of assume: how much of each
// model file is already in the page cache, and an explicit (user-space,
// no-root) way to evict it before a cold-filesystem run.

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace dif::bench {

struct FileResidency {
  std::filesystem::path path;
  bool exists{};
  std::uint64_t bytes{};
  std::uint64_t resident_bytes{};
};

// mincore() over a read-only mapping. Never faults pages in.
FileResidency measure_residency(const std::filesystem::path &path);

// posix_fadvise(DONTNEED) over the whole file. Clean, unmapped pages are
// dropped; anything else stays, which the follow-up residency measurement
// reports honestly.
void drop_file_cache(const std::filesystem::path &path);

struct ResidencySummary {
  std::uint64_t total_bytes{};
  std::uint64_t resident_bytes{};
  double resident_fraction{};
  // "cold" below 5 percent resident, "warm" above 95 percent, "mixed"
  // between, "unknown" when no model file was declared.
  std::string condition;
};

ResidencySummary summarize_residency(const std::vector<FileResidency> &files);

std::string hostname();

} // namespace dif::bench
