#pragma once

#include "dif/runtime/tensor.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>
#include <sys/stat.h>

namespace dif::runtime::detail {

// Executor-local identity, not a path/content cache. A separately opened
// mapping is a miss. Checkpoints must remain immutable while in use;
// metadata additionally invalidates reuse after an in-place rewrite.
struct MappedWeightKey {
  std::uintptr_t mapping{};
  std::size_t offset{}, bytes{};
  ir::DType dtype{};
  std::vector<std::uint64_t> dims;
  std::array<std::uint64_t, 7> version{};
  auto fields() const {
    return std::tie(mapping, offset, bytes, dtype, dims, version);
  }
  bool operator==(const MappedWeightKey &other) const {
    return fields() == other.fields();
  }
  bool operator<(const MappedWeightKey &other) const {
    return fields() < other.fields();
  }
};

inline std::optional<MappedWeightKey> mapped_weight_key(const Tensor &tensor) {
  if (!tensor.is_mapped() || tensor.byte_size() == 0U)
    return std::nullopt;
  struct stat info {};
  if (::fstat(tensor.mapping->descriptor(), &info) != 0 ||
      !S_ISREG(info.st_mode) || info.st_size < 0 ||
      tensor.mapping_offset > static_cast<std::uint64_t>(info.st_size) ||
      tensor.byte_size() > static_cast<std::uint64_t>(info.st_size) -
                               tensor.mapping_offset)
    return std::nullopt;
  return MappedWeightKey{
      reinterpret_cast<std::uintptr_t>(tensor.mapping.get()),
      tensor.mapping_offset, tensor.byte_size(), tensor.dtype, tensor.dims,
      {static_cast<std::uint64_t>(info.st_dev),
       static_cast<std::uint64_t>(info.st_ino),
       static_cast<std::uint64_t>(info.st_size),
       static_cast<std::uint64_t>(info.st_mtim.tv_sec),
       static_cast<std::uint64_t>(info.st_mtim.tv_nsec),
       static_cast<std::uint64_t>(info.st_ctim.tv_sec),
       static_cast<std::uint64_t>(info.st_ctim.tv_nsec)}};
}

// Exact groups only: partial hits must not retain unused weights from a
// larger group. Weak entries retain neither mappings nor device allocations.
// Access is serialized by the caller, like CUDA streams and telemetry.
template <class Resource> class MappedWeightCache {
public:
  using Key = std::vector<MappedWeightKey>;
  std::shared_ptr<Resource> find(const Key &key) {
    std::erase_if(entries_, [](const auto &entry) {
      return entry.resource.expired();
    });
    for (const auto &entry : entries_)
      if (entry.key == key)
        return entry.resource.lock();
    return {};
  }
  // Only after successful preparation and its upload completion fence.
  void publish(Key key, const std::shared_ptr<Resource> &resource) {
    entries_.push_back({std::move(key), resource});
  }

private:
  struct Entry {
    Key key;
    std::weak_ptr<Resource> resource;
  };
  std::vector<Entry> entries_;
};

} // namespace dif::runtime::detail
