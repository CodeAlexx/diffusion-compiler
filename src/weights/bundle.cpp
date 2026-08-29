#include "dif/weights/bundle.hpp"

#include "dif/ir/codec.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/weights/safetensors.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <unordered_map>

namespace dif::weights {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic = {'D', 'I', 'F', 'B', 'N', 'D', '0', '1'};
constexpr std::uint32_t kVersion = 1U;
constexpr std::size_t kMaximumBundleBytes = 64U * 1024U * 1024U;

void append_u32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (unsigned shift = 0; shift < 32U; shift += 8U)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void append_u64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64U; shift += 8U)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void append_string(std::vector<std::uint8_t> &bytes, const std::string &value) {
  if (value.empty() || value.size() > std::numeric_limits<std::uint32_t>::max())
    fail("weight bundle string has an invalid size");
  append_u32(bytes, static_cast<std::uint32_t>(value.size()));
  bytes.insert(bytes.end(), value.begin(), value.end());
}

std::uint32_t read_u32(std::span<const std::uint8_t> bytes, std::size_t &offset) {
  if (offset > bytes.size() || bytes.size() - offset < 4U)
    fail("truncated weight bundle");
  std::uint32_t value = 0;
  for (unsigned shift = 0; shift < 32U; shift += 8U)
    value |= static_cast<std::uint32_t>(bytes[offset++]) << shift;
  return value;
}

std::uint64_t read_u64(std::span<const std::uint8_t> bytes, std::size_t &offset) {
  if (offset > bytes.size() || bytes.size() - offset < 8U)
    fail("truncated weight bundle");
  std::uint64_t value = 0;
  for (unsigned shift = 0; shift < 64U; shift += 8U)
    value |= static_cast<std::uint64_t>(bytes[offset++]) << shift;
  return value;
}

std::string read_string(std::span<const std::uint8_t> bytes,
                        std::size_t &offset) {
  const auto size = read_u32(bytes, offset);
  if (size == 0U || offset > bytes.size() || size > bytes.size() - offset)
    fail("invalid weight bundle string");
  std::string value(reinterpret_cast<const char *>(bytes.data() + offset), size);
  offset += size;
  return value;
}

std::vector<std::uint8_t> read_all(const std::filesystem::path &path) {
  const auto size = std::filesystem::file_size(path);
  if (size == 0U || size > kMaximumBundleBytes)
    fail("weight bundle has an invalid size");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  std::ifstream input(path, std::ios::binary);
  if (!input)
    fail("cannot open weight bundle: " + path.string());
  input.read(reinterpret_cast<char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input)
    fail("cannot read weight bundle: " + path.string());
  return bytes;
}

void validate_binding(const WeightBundle &bundle, const ir::Program &program,
                      const BundleBinding &binding) {
  if (binding.shard_index >= bundle.shards.size())
    fail("weight bundle binding references an unknown shard");
  const auto *description = program.tensor(binding.tensor_id);
  if (!description || !description->has_role(ir::TensorRole::Constant))
    fail("weight bundle binding does not target a program constant");
  if (description->dtype != binding.dtype || description->dims != binding.dims ||
      description->byte_count() != binding.byte_count)
    fail("weight bundle binding disagrees with the program descriptor");
}

} // namespace

void write_weight_bundle(const WeightBundle &bundle,
                         const std::filesystem::path &path) {
  if (bundle.shards.empty() || bundle.bindings.empty())
    fail("cannot write an empty weight bundle");
  if (bundle.shards.size() > std::numeric_limits<std::uint32_t>::max() ||
      bundle.bindings.size() > std::numeric_limits<std::uint32_t>::max())
    fail("weight bundle contains too many records");
  std::vector<std::uint8_t> bytes(kMagic.begin(), kMagic.end());
  append_u32(bytes, kVersion);
  bytes.insert(bytes.end(), bundle.program_fingerprint.begin(),
               bundle.program_fingerprint.end());
  bytes.insert(bytes.end(), bundle.index_fingerprint.begin(),
               bundle.index_fingerprint.end());
  append_u32(bytes, static_cast<std::uint32_t>(bundle.shards.size()));
  for (const auto &shard : bundle.shards) {
    append_string(bytes, shard.path.string());
    append_u64(bytes, shard.file_size);
    bytes.insert(bytes.end(), shard.digest.begin(), shard.digest.end());
  }
  append_u32(bytes, static_cast<std::uint32_t>(bundle.bindings.size()));
  for (const auto &binding : bundle.bindings) {
    append_u32(bytes, binding.tensor_id);
    append_u32(bytes, binding.shard_index);
    append_string(bytes, binding.tensor_name);
    append_u32(bytes, static_cast<std::uint32_t>(binding.dtype));
    if (binding.dims.empty() || binding.dims.size() > ir::kMaxRank)
      fail("weight bundle binding has an invalid rank");
    append_u32(bytes, static_cast<std::uint32_t>(binding.dims.size()));
    for (const auto dim : binding.dims)
      append_u64(bytes, dim);
    append_u64(bytes, binding.file_offset);
    append_u64(bytes, binding.byte_count);
  }
  const auto digest = sha256(bytes);
  bytes.insert(bytes.end(), digest.begin(), digest.end());
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    fail("cannot create weight bundle: " + path.string());
  output.write(reinterpret_cast<const char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!output)
    fail("cannot write weight bundle: " + path.string());
}

WeightBundle read_weight_bundle(const std::filesystem::path &path) {
  const auto bytes = read_all(path);
  if (bytes.size() < kMagic.size() + 4U + 64U + 4U + 4U + 32U)
    fail("weight bundle is too small");
  const auto payload =
      std::span<const std::uint8_t>(bytes).first(bytes.size() - 32U);
  const auto actual = sha256(payload);
  if (!std::equal(actual.begin(), actual.end(), bytes.end() - 32))
    fail("weight bundle SHA-256 mismatch");
  std::size_t offset = 0U;
  if (!std::equal(kMagic.begin(), kMagic.end(), payload.begin()))
    fail("invalid weight bundle magic");
  offset += kMagic.size();
  if (read_u32(payload, offset) != kVersion)
    fail("unsupported weight bundle version");
  WeightBundle bundle;
  std::copy_n(payload.begin() + static_cast<std::ptrdiff_t>(offset), 32U,
              bundle.program_fingerprint.begin());
  offset += 32U;
  std::copy_n(payload.begin() + static_cast<std::ptrdiff_t>(offset), 32U,
              bundle.index_fingerprint.begin());
  offset += 32U;
  const auto shard_count = read_u32(payload, offset);
  if (shard_count == 0U || shard_count > 4096U)
    fail("weight bundle shard count is invalid");
  bundle.shards.reserve(shard_count);
  for (std::uint32_t i = 0; i < shard_count; ++i) {
    BundleShard shard;
    shard.path = read_string(payload, offset);
    shard.file_size = read_u64(payload, offset);
    if (offset > payload.size() || payload.size() - offset < shard.digest.size())
      fail("truncated weight bundle shard digest");
    std::copy_n(payload.begin() + static_cast<std::ptrdiff_t>(offset),
                shard.digest.size(), shard.digest.begin());
    offset += shard.digest.size();
    bundle.shards.push_back(std::move(shard));
  }
  const auto binding_count = read_u32(payload, offset);
  if (binding_count == 0U || binding_count > 1000000U)
    fail("weight bundle binding count is invalid");
  bundle.bindings.reserve(binding_count);
  for (std::uint32_t i = 0; i < binding_count; ++i) {
    BundleBinding binding;
    binding.tensor_id = read_u32(payload, offset);
    binding.shard_index = read_u32(payload, offset);
    binding.tensor_name = read_string(payload, offset);
    binding.dtype = static_cast<ir::DType>(read_u32(payload, offset));
    const auto rank = read_u32(payload, offset);
    if (rank == 0U || rank > ir::kMaxRank)
      fail("weight bundle binding rank is invalid");
    binding.dims.reserve(rank);
    for (std::uint32_t axis = 0; axis < rank; ++axis)
      binding.dims.push_back(read_u64(payload, offset));
    binding.file_offset = read_u64(payload, offset);
    binding.byte_count = read_u64(payload, offset);
    bundle.bindings.push_back(std::move(binding));
  }
  if (offset != payload.size())
    fail("weight bundle contains trailing payload bytes");
  return bundle;
}

void verify_weight_bundle(const WeightBundle &bundle, const ir::Program &program,
                          bool verify_shard_digests) {
  if (bundle.program_fingerprint != ir::fingerprint(program))
    fail("weight bundle targets a different DiffIR fingerprint");
  std::set<std::uint32_t> tensor_ids;
  for (const auto &binding : bundle.bindings) {
    validate_binding(bundle, program, binding);
    if (!tensor_ids.insert(binding.tensor_id).second)
      fail("weight bundle contains a duplicate tensor id");
  }
  for (const auto &shard : bundle.shards) {
    if (!std::filesystem::is_regular_file(shard.path) ||
        std::filesystem::file_size(shard.path) != shard.file_size)
      fail("weight bundle shard size mismatch: " + shard.path.string());
    if (verify_shard_digests && sha256_file(shard.path) != shard.digest)
      fail("weight bundle shard SHA-256 mismatch: " + shard.path.string());
  }
}

runtime::TensorMap load_weight_bundle(const WeightBundle &bundle,
                                      const ir::Program &program,
                                      bool verify_shard_digests) {
  verify_weight_bundle(bundle, program, verify_shard_digests);
  runtime::TensorMap tensors;
  for (std::uint32_t shard_index = 0; shard_index < bundle.shards.size();
       ++shard_index) {
    const auto &shard = bundle.shards[shard_index];
    const auto metadata = read_safetensors(shard.path);
    std::vector<const BundleBinding *> selected;
    std::vector<runtime::TensorSlice> slices;
    for (const auto &binding : bundle.bindings) {
      if (binding.shard_index != shard_index)
        continue;
      const auto *entry = metadata.find(binding.tensor_name);
      if (!entry || entry->dtype != binding.dtype || entry->dims != binding.dims ||
          entry->file_offset != binding.file_offset ||
          entry->byte_count != binding.byte_count)
        fail("weight bundle SafeTensors metadata mismatch: " +
             binding.tensor_name);
      selected.push_back(&binding);
      slices.push_back({binding.dtype, binding.dims, binding.file_offset,
                        binding.byte_count});
    }
    if (selected.empty())
      continue;
    auto mapped = runtime::map_tensor_slices(shard.path, slices);
    for (std::size_t i = 0; i < selected.size(); ++i)
      tensors.emplace(selected[i]->tensor_id, std::move(mapped[i]));
  }
  if (tensors.size() != bundle.bindings.size())
    fail("weight bundle did not map every binding");
  return tensors;
}

} // namespace dif::weights
