#pragma once

#include "dif/ir/ir.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace dif::runtime {

class MappedStorage {
public:
  ~MappedStorage();
  MappedStorage(const MappedStorage &) = delete;
  MappedStorage &operator=(const MappedStorage &) = delete;

  const std::uint8_t *data() const;
  std::size_t size() const;
  void discard(std::size_t offset, std::size_t bytes) const;
  void evict(std::size_t offset, std::size_t bytes) const;
  // Asynchronous read-ahead of a file range into the page cache
  // (posix_fadvise/madvise WILLNEED). A later memcpy from the mapping then
  // finds warm pages instead of faulting them in one page at a time.
  void prefetch(std::size_t offset, std::size_t bytes) const;
  // Copy a file range into `destination` with O_DIRECT reads, sixteen MiB
  // chunks, eight in flight, bypassing the page cache. Returns false (and
  // writes nothing) when the file has no direct descriptor or a read fails,
  // so the caller can fall back to copying from the mapping.
  bool read_direct(std::size_t offset, std::size_t bytes,
                   void *destination) const;
  // Fraction of the range's pages currently in the page cache (mincore),
  // 0.0 when unknown.
  double resident_fraction(std::size_t offset, std::size_t bytes) const;

private:
  friend std::shared_ptr<const MappedStorage>
  map_readonly_file(const std::filesystem::path &path);
  MappedStorage(void *address, std::size_t size, int descriptor,
                int direct_descriptor);

  void *address_{};
  std::size_t size_{};
  int descriptor_{-1};
  int direct_descriptor_{-1};
};

struct Tensor {
  Tensor() = default;
  Tensor(ir::DType value_dtype, std::vector<std::uint64_t> value_dims,
         std::vector<std::uint8_t> value_bytes)
      : dtype(value_dtype), dims(std::move(value_dims)),
        bytes(std::move(value_bytes)) {}

  ir::DType dtype{};
  std::vector<std::uint64_t> dims;
  std::vector<std::uint8_t> bytes;
  std::shared_ptr<const MappedStorage> mapping;
  std::size_t mapping_offset{};
  std::size_t mapping_bytes{};

  std::uint64_t element_count() const;
  void validate() const;
  const std::uint8_t *data() const;
  std::uint8_t *mutable_data();
  std::size_t byte_size() const;
  bool is_mapped() const { return static_cast<bool>(mapping); }
  void discard_mapped_pages() const;
  void evict_mapped_pages() const;
  void prefetch_mapped_pages() const;
  // Stage this tensor's bytes into host memory: direct IO from the mapped
  // file when available (page cache bypassed), else false.
  bool read_direct_into(void *destination) const;
  double mapped_resident_fraction() const;

  std::span<float> f32();
  std::span<const float> f32() const;
};

struct TensorSlice {
  ir::DType dtype{};
  std::vector<std::uint64_t> dims;
  std::uint64_t file_offset{};
  std::uint64_t byte_count{};
};

Tensor read_tensor(const std::filesystem::path &path);
Tensor map_tensor(const std::filesystem::path &path);
std::shared_ptr<const MappedStorage>
map_readonly_file(const std::filesystem::path &path);
Tensor map_tensor_slice(std::shared_ptr<const MappedStorage> storage,
                        ir::DType dtype, std::vector<std::uint64_t> dims,
                        std::uint64_t file_offset,
                        std::uint64_t byte_count);
Tensor map_tensor_slice(const std::filesystem::path &path, ir::DType dtype,
                        std::vector<std::uint64_t> dims,
                        std::uint64_t file_offset,
                        std::uint64_t byte_count);
std::vector<Tensor>
map_tensor_slices(const std::filesystem::path &path,
                  const std::vector<TensorSlice> &slices);
void write_tensor(const Tensor &tensor, const std::filesystem::path &path);
Tensor zeros(const ir::TensorDesc &desc);
// Materialize a tensor in another floating storage dtype.  This is the shared
// native checkpoint-load boundary used when creator code calls `.to(BF16)` on
// an F32 SafeTensors parameter.  It deliberately performs no shape/layout
// transformation and uses the runtime's round-to-nearest-even conversions.
Tensor convert_float_tensor(const Tensor &source, ir::DType destination);

} // namespace dif::runtime
