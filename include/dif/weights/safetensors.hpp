#pragma once

#include "dif/ir/ir.hpp"
#include "dif/runtime/tensor.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace dif::weights {

struct SafeTensorEntry {
  std::string name;
  ir::DType dtype{};
  std::vector<std::uint64_t> dims;
  std::uint64_t file_offset{};
  std::uint64_t byte_count{};
};

struct SafeTensorMetadataEntry {
  std::string name;
  std::string dtype;
  std::vector<std::uint64_t> dims;
  std::uint64_t file_offset{};
  std::uint64_t byte_count{};
};

struct SafeTensorFile {
  std::filesystem::path path;
  std::uint64_t file_size{};
  std::uint64_t data_offset{};
  // All tensors from one SafeTensors file share one read-only mapping.  This
  // keeps checkpoint mapping O(files), rather than mapping the complete file
  // again for every tensor entry.
  std::shared_ptr<const runtime::MappedStorage> mapping;
  std::map<std::string, SafeTensorEntry, std::less<>> tensors;
  std::map<std::string, SafeTensorMetadataEntry, std::less<>> metadata_tensors;

  const SafeTensorEntry *find(std::string_view name) const;
  const SafeTensorMetadataEntry *find_metadata(std::string_view name) const;
};

struct SafeTensorIndex {
  std::filesystem::path path;
  std::map<std::string, std::filesystem::path, std::less<>> weight_map;
};

struct SafeTensorWriteSpec {
  std::string name;
  ir::DType dtype{};
  std::vector<std::uint64_t> dims;
};

class SafeTensorWriter {
public:
  SafeTensorWriter(const std::filesystem::path &path,
                   std::vector<SafeTensorWriteSpec> tensors);
  ~SafeTensorWriter();
  SafeTensorWriter(const SafeTensorWriter &) = delete;
  SafeTensorWriter &operator=(const SafeTensorWriter &) = delete;
  SafeTensorWriter(SafeTensorWriter &&) noexcept;
  SafeTensorWriter &operator=(SafeTensorWriter &&) noexcept;

  void append(std::string_view name, std::span<const std::uint8_t> bytes);
  SafeTensorFile finish();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

SafeTensorFile read_safetensors(const std::filesystem::path &path);
SafeTensorIndex read_safetensors_index(const std::filesystem::path &path);
runtime::Tensor map_safetensor(const SafeTensorFile &file,
                               std::string_view name);
std::vector<std::uint8_t>
read_safetensor_metadata(const SafeTensorFile &file, std::string_view name);

} // namespace dif::weights
