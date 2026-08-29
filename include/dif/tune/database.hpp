#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace dif::tune {

struct Measurement {
  std::string candidate_hash;
  std::string program_hash;
  std::string backend;
  std::string device;
  double mean_milliseconds{};
  double minimum_milliseconds{};
  double maximum_milliseconds{};
  double max_absolute_error{};
  double cosine_similarity{};
  double norm_ratio{};
  std::uint64_t nonfinite_count{};
  // Lifetime-aware planned device working set of the candidate, in bytes. Zero
  // when the recording tool did not plan memory.
  std::uint64_t planned_memory_bytes{};
  // Canonical transformation sequence that rebuilds this candidate from
  // program_hash. Empty for a measurement that was not produced by a
  // transformation plan. Together with program_hash this is the reproducible
  // identity of the candidate: replaying it must yield candidate_hash.
  std::string plan;
  // Acceptance verdict, including the rejection reason when rejected.
  std::string status;
  std::int64_t created_unix{};
};

class Database {
public:
  explicit Database(const std::filesystem::path &path);
  ~Database();
  Database(const Database &) = delete;
  Database &operator=(const Database &) = delete;

  void record(const Measurement &measurement);
  std::vector<Measurement> results(const std::string &program_hash) const;
  bool persistent() const;

private:
  void *handle_{};
};

} // namespace dif::tune
