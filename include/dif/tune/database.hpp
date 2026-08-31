#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace dif::tune {

struct Measurement {
  // candidate_hash identifies the complete executable candidate/plan. For
  // DiffIR-only candidates it may equal candidate_program_hash.
  std::string candidate_hash;
  std::string candidate_program_hash;
  std::string program_hash;
  std::string recipe_hash;
  std::string recipe_text;
  std::string backend;
  std::string device;
  std::vector<double> trial_mean_milliseconds;
  std::vector<double> iteration_milliseconds;
  std::string objective_name;
  double objective_milliseconds{};
  double preparation_milliseconds{};
  double mean_milliseconds{};
  double minimum_milliseconds{};
  double maximum_milliseconds{};
  double max_absolute_error{};
  double cosine_similarity{};
  double norm_ratio{};
  std::uint64_t nonfinite_count{};
  std::uint64_t planned_device_bytes{};
  std::uint64_t measured_resident_bytes{};
  std::uint64_t memory_limit_bytes{};
  std::string status;
  std::string rejection_reason;
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
