#pragma once

#include "dif/ir/ir.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/support/sha256.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace dif::weights {

struct BundleShard {
  std::filesystem::path path;
  std::uint64_t file_size{};
  Sha256Digest digest{};
};

struct BundleBinding {
  std::uint32_t tensor_id{};
  std::uint32_t shard_index{};
  std::string tensor_name;
  ir::DType dtype{};
  std::vector<std::uint64_t> dims;
  std::uint64_t file_offset{};
  std::uint64_t byte_count{};
};

struct WeightBundle {
  Sha256Digest program_fingerprint{};
  Sha256Digest index_fingerprint{};
  std::vector<BundleShard> shards;
  std::vector<BundleBinding> bindings;
};

void write_weight_bundle(const WeightBundle &bundle,
                         const std::filesystem::path &path);
WeightBundle read_weight_bundle(const std::filesystem::path &path);
runtime::TensorMap load_weight_bundle(const WeightBundle &bundle,
                                      const ir::Program &program,
                                      bool verify_shard_digests);
void verify_weight_bundle(const WeightBundle &bundle, const ir::Program &program,
                          bool verify_shard_digests);

} // namespace dif::weights
