#include "dif/training/checkpoint.hpp"

#include "dif/support/error.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <vector>

namespace dif::training {
namespace {

constexpr std::array<char, 8> kMagic = {'D', 'I', 'F', 'T', 'R', 'N', '0', '1'};
constexpr std::uint32_t kVersion = 1U;

void write_u32(std::ostream &output, std::uint32_t value) {
  std::array<char, 4> bytes{};
  for (unsigned shift = 0U; shift < 32U; shift += 8U)
    bytes[shift / 8U] = static_cast<char>(value >> shift);
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void write_u64(std::ostream &output, std::uint64_t value) {
  std::array<char, 8> bytes{};
  for (unsigned shift = 0U; shift < 64U; shift += 8U)
    bytes[shift / 8U] = static_cast<char>(value >> shift);
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

std::uint32_t read_u32(std::istream &input) {
  std::array<unsigned char, 4> bytes{};
  input.read(reinterpret_cast<char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input)
    fail("truncated training checkpoint");
  std::uint32_t value = 0U;
  for (unsigned shift = 0U; shift < 32U; shift += 8U)
    value |= static_cast<std::uint32_t>(bytes[shift / 8U]) << shift;
  return value;
}

std::uint64_t read_u64(std::istream &input) {
  std::array<unsigned char, 8> bytes{};
  input.read(reinterpret_cast<char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input)
    fail("truncated training checkpoint");
  std::uint64_t value = 0U;
  for (unsigned shift = 0U; shift < 64U; shift += 8U)
    value |= static_cast<std::uint64_t>(bytes[shift / 8U]) << shift;
  return value;
}

bool valid_dtype(ir::DType dtype) {
  return dtype == ir::DType::F32 || dtype == ir::DType::BF16 ||
         dtype == ir::DType::F16 || dtype == ir::DType::I8 ||
         dtype == ir::DType::I32;
}

} // namespace

void write_checkpoint(const Checkpoint &checkpoint,
                      const std::filesystem::path &path) {
  if (checkpoint.state.empty() ||
      checkpoint.state.size() > std::numeric_limits<std::uint32_t>::max())
    fail("training checkpoint state count is invalid");
  std::vector<std::uint32_t> ids;
  ids.reserve(checkpoint.state.size());
  for (const auto &[id, tensor] : checkpoint.state) {
    if (id == 0U)
      fail("training checkpoint tensor id must be nonzero");
    tensor.validate();
    ids.push_back(id);
  }
  std::sort(ids.begin(), ids.end());
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    fail("cannot create training checkpoint: " + path.string());
  output.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
  write_u32(output, kVersion);
  output.write(reinterpret_cast<const char *>(checkpoint.program_fingerprint.data()),
               static_cast<std::streamsize>(checkpoint.program_fingerprint.size()));
  write_u64(output, checkpoint.completed_steps);
  write_u32(output, static_cast<std::uint32_t>(ids.size()));
  for (const auto id : ids) {
    const auto &tensor = checkpoint.state.at(id);
    write_u32(output, id);
    write_u32(output, static_cast<std::uint32_t>(tensor.dtype));
    write_u32(output, static_cast<std::uint32_t>(tensor.dims.size()));
    for (const auto dimension : tensor.dims)
      write_u64(output, dimension);
    write_u64(output, tensor.byte_size());
    output.write(reinterpret_cast<const char *>(tensor.data()),
                 static_cast<std::streamsize>(tensor.byte_size()));
    if (!output)
      fail("cannot write training checkpoint tensor");
  }
  output.flush();
  if (!output)
    fail("cannot flush training checkpoint");
  const auto payload_bytes = static_cast<std::uint64_t>(output.tellp());
  output.close();
  const auto digest = sha256_file_prefix(path, payload_bytes);
  std::ofstream trailer(path, std::ios::binary | std::ios::app);
  trailer.write(reinterpret_cast<const char *>(digest.data()),
                static_cast<std::streamsize>(digest.size()));
  if (!trailer)
    fail("cannot append training checkpoint digest");
}

Checkpoint read_checkpoint(const std::filesystem::path &path) {
  const auto file_bytes = std::filesystem::file_size(path);
  if (file_bytes < kMagic.size() + 4U + 32U + 8U + 4U + 32U)
    fail("training checkpoint is too small");
  const auto payload_bytes = file_bytes - 32U;
  const auto actual = sha256_file_prefix(path, payload_bytes);
  std::ifstream input(path, std::ios::binary);
  if (!input)
    fail("cannot open training checkpoint: " + path.string());
  std::array<char, 8> magic{};
  input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (magic != kMagic || read_u32(input) != kVersion)
    fail("invalid or unsupported training checkpoint");
  Checkpoint checkpoint;
  input.read(reinterpret_cast<char *>(checkpoint.program_fingerprint.data()),
             static_cast<std::streamsize>(checkpoint.program_fingerprint.size()));
  if (!input)
    fail("truncated training checkpoint fingerprint");
  checkpoint.completed_steps = read_u64(input);
  const auto tensor_count = read_u32(input);
  if (tensor_count == 0U || tensor_count > 1000000U)
    fail("training checkpoint tensor count is invalid");
  for (std::uint32_t index = 0U; index < tensor_count; ++index) {
    const auto id = read_u32(input);
    const auto dtype = static_cast<ir::DType>(read_u32(input));
    const auto rank = read_u32(input);
    if (id == 0U || !valid_dtype(dtype) || rank == 0U || rank > ir::kMaxRank)
      fail("training checkpoint tensor metadata is invalid");
    std::vector<std::uint64_t> dims;
    dims.reserve(rank);
    for (std::uint32_t axis = 0U; axis < rank; ++axis)
      dims.push_back(read_u64(input));
    const auto byte_count = read_u64(input);
    if (byte_count > static_cast<std::uint64_t>(
                         std::numeric_limits<std::size_t>::max()))
      fail("training checkpoint tensor is too large for this host");
    runtime::Tensor tensor{dtype, std::move(dims), {}};
    if (tensor.byte_size() != 0U)
      fail("new training checkpoint tensor unexpectedly owns bytes");
    tensor.bytes.resize(static_cast<std::size_t>(byte_count));
    tensor.validate();
    input.read(reinterpret_cast<char *>(tensor.bytes.data()),
               static_cast<std::streamsize>(tensor.bytes.size()));
    if (!input)
      fail("truncated training checkpoint tensor payload");
    if (!checkpoint.state.emplace(id, std::move(tensor)).second)
      fail("training checkpoint contains duplicate tensor id");
  }
  if (static_cast<std::uint64_t>(input.tellg()) != payload_bytes)
    fail("training checkpoint payload length mismatch");
  Sha256Digest stored{};
  input.read(reinterpret_cast<char *>(stored.data()),
             static_cast<std::streamsize>(stored.size()));
  if (!input || stored != actual)
    fail("training checkpoint SHA-256 mismatch");
  return checkpoint;
}

} // namespace dif::training
