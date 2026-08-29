#include "dif/tune/database.hpp"

#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <span>
#include <utility>
#include <vector>

namespace dif::tune {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic = {'D', 'I', 'F', 'T', 'U', 'N', 'E', '1'};
constexpr std::uint32_t kVersion = 3;
constexpr std::size_t kDigestBytes = 32;
constexpr std::uint32_t kMaxRecords = 1U << 20U;
constexpr std::uint32_t kMaxString = 1U << 20U;

struct Store {
  std::filesystem::path path;
  std::vector<Measurement> measurements;
};

class Writer {
public:
  void u32(std::uint32_t value) {
    for (unsigned shift = 0; shift < 32U; shift += 8U)
      bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  }
  void u64(std::uint64_t value) {
    for (unsigned shift = 0; shift < 64U; shift += 8U)
      bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  }
  void f64(double value) { u64(std::bit_cast<std::uint64_t>(value)); }
  void string(const std::string &value) {
    if (value.size() > kMaxString)
      fail("tuning database string is too large");
    u32(static_cast<std::uint32_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
  }
  std::vector<std::uint8_t> bytes;
};

class Reader {
public:
  explicit Reader(std::span<const std::uint8_t> data) : data_(data) {}
  std::uint32_t u32() {
    require(4);
    std::uint32_t value = 0;
    for (unsigned shift = 0; shift < 32U; shift += 8U)
      value |= static_cast<std::uint32_t>(data_[offset_++]) << shift;
    return value;
  }
  std::uint64_t u64() {
    require(8);
    std::uint64_t value = 0;
    for (unsigned shift = 0; shift < 64U; shift += 8U)
      value |= static_cast<std::uint64_t>(data_[offset_++]) << shift;
    return value;
  }
  double f64() { return std::bit_cast<double>(u64()); }
  std::string string() {
    const auto size = u32();
    if (size > kMaxString)
      fail("tuning database string is too large");
    require(size);
    std::string value(reinterpret_cast<const char *>(data_.data() + offset_), size);
    offset_ += size;
    return value;
  }
  std::span<const std::uint8_t> raw(std::size_t size) {
    require(size);
    const auto value = data_.subspan(offset_, size);
    offset_ += size;
    return value;
  }
  bool done() const { return offset_ == data_.size(); }

private:
  void require(std::size_t size) const {
    if (size > data_.size() - offset_)
      fail("truncated tuning database");
  }
  std::span<const std::uint8_t> data_;
  std::size_t offset_{};
};

std::vector<std::uint8_t> read_all(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    fail("cannot open tuning database: " + path.string());
  const auto end = input.tellg();
  if (end < 0)
    fail("cannot size tuning database: " + path.string());
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
  input.seekg(0);
  input.read(reinterpret_cast<char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input)
    fail("cannot read tuning database: " + path.string());
  return bytes;
}

void save(const Store &store) {
  Writer writer;
  writer.bytes.insert(writer.bytes.end(), kMagic.begin(), kMagic.end());
  writer.u32(kVersion);
  if (store.measurements.size() > kMaxRecords)
    fail("tuning database has too many records");
  writer.u32(static_cast<std::uint32_t>(store.measurements.size()));
  for (const auto &value : store.measurements) {
    writer.string(value.candidate_hash);
    writer.string(value.program_hash);
    writer.string(value.backend);
    writer.string(value.device);
    writer.f64(value.mean_milliseconds);
    writer.f64(value.minimum_milliseconds);
    writer.f64(value.maximum_milliseconds);
    writer.f64(value.max_absolute_error);
    writer.f64(value.cosine_similarity);
    writer.f64(value.norm_ratio);
    writer.u64(value.nonfinite_count);
    writer.u64(value.planned_memory_bytes);
    writer.string(value.plan);
    writer.string(value.status);
    writer.u64(std::bit_cast<std::uint64_t>(value.created_unix));
  }
  const auto digest = sha256(writer.bytes);
  writer.bytes.insert(writer.bytes.end(), digest.begin(), digest.end());
  if (!store.path.parent_path().empty())
    std::filesystem::create_directories(store.path.parent_path());
  const auto temporary = std::filesystem::path(store.path.string() + ".tmp");
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output)
      fail("cannot create tuning database temporary file");
    output.write(reinterpret_cast<const char *>(writer.bytes.data()),
                 static_cast<std::streamsize>(writer.bytes.size()));
    if (!output)
      fail("cannot write tuning database temporary file");
  }
  std::filesystem::rename(temporary, store.path);
}

void load(Store &store) {
  if (!std::filesystem::exists(store.path))
    return;
  const auto file = read_all(store.path);
  if (file.size() < kMagic.size() + 2U * sizeof(std::uint32_t) + kDigestBytes)
    fail("tuning database is too small");
  const auto bytes = std::span<const std::uint8_t>(file);
  const auto payload = bytes.first(bytes.size() - kDigestBytes);
  const auto digest = sha256(payload);
  if (!std::equal(digest.begin(), digest.end(), bytes.end() - 32))
    fail("tuning database SHA-256 mismatch");
  Reader reader(payload);
  const auto magic = reader.raw(kMagic.size());
  if (!std::equal(kMagic.begin(), kMagic.end(), magic.begin()))
    fail("invalid tuning database magic");
  const auto version = reader.u32();
  // Records grew fields in versions 2 and 3; every earlier layout still reads.
  if (version == 0U || version > kVersion)
    fail("unsupported tuning database version");
  const auto count = reader.u32();
  if (count > kMaxRecords)
    fail("tuning database record count is unreasonable");
  store.measurements.reserve(count);
  for (std::uint32_t i = 0; i < count; ++i) {
    Measurement value;
    value.candidate_hash = reader.string();
    value.program_hash = reader.string();
    value.backend = reader.string();
    value.device = reader.string();
    value.mean_milliseconds = reader.f64();
    value.minimum_milliseconds = reader.f64();
    value.maximum_milliseconds = reader.f64();
    value.max_absolute_error = reader.f64();
    value.cosine_similarity = reader.f64();
    if (version >= 2U) {
      value.norm_ratio = reader.f64();
      value.nonfinite_count = reader.u64();
    } else {
      value.norm_ratio = 1.0;
      value.nonfinite_count = 0U;
    }
    if (version >= 3U) {
      value.planned_memory_bytes = reader.u64();
      value.plan = reader.string();
    }
    value.status = reader.string();
    value.created_unix = std::bit_cast<std::int64_t>(reader.u64());
    store.measurements.push_back(std::move(value));
  }
  if (!reader.done())
    fail("trailing bytes in tuning database");
}

} // namespace

Database::Database(const std::filesystem::path &path) {
  auto *store = new Store{path, {}};
  try {
    load(*store);
  } catch (...) {
    delete store;
    throw;
  }
  handle_ = store;
}

Database::~Database() { delete static_cast<Store *>(handle_); }

bool Database::persistent() const { return handle_ != nullptr; }

void Database::record(const Measurement &measurement) {
  auto &store = *static_cast<Store *>(handle_);
  const auto existing = std::find_if(
      store.measurements.begin(), store.measurements.end(),
      [&](const Measurement &value) {
        return value.candidate_hash == measurement.candidate_hash &&
               value.backend == measurement.backend && value.device == measurement.device;
      });
  if (existing == store.measurements.end())
    store.measurements.push_back(measurement);
  else
    *existing = measurement;
  save(store);
}

std::vector<Measurement> Database::results(const std::string &program_hash) const {
  std::vector<Measurement> output;
  const auto &store = *static_cast<const Store *>(handle_);
  std::copy_if(store.measurements.begin(), store.measurements.end(),
               std::back_inserter(output), [&](const Measurement &value) {
                 return value.program_hash == program_hash;
               });
  std::sort(output.begin(), output.end(), [](const Measurement &a, const Measurement &b) {
    return a.mean_milliseconds < b.mean_milliseconds;
  });
  return output;
}

} // namespace dif::tune
