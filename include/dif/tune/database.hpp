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
