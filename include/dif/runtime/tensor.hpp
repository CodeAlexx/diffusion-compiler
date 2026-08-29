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

class MappedStorage;

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
Tensor map_tensor_slice(const std::filesystem::path &path, ir::DType dtype,
                        std::vector<std::uint64_t> dims,
                        std::uint64_t file_offset,
                        std::uint64_t byte_count);
std::vector<Tensor>
map_tensor_slices(const std::filesystem::path &path,
                  const std::vector<TensorSlice> &slices);
void write_tensor(const Tensor &tensor, const std::filesystem::path &path);
Tensor zeros(const ir::TensorDesc &desc);

} // namespace dif::runtime
