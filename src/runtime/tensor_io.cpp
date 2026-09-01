#include "dif/runtime/tensor.hpp"

#include "dif/runtime/scalar.hpp"
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <limits>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace dif::runtime {

MappedStorage::MappedStorage(void *address, std::size_t size, int descriptor)
    : address_(address), size_(size), descriptor_(descriptor) {}

MappedStorage::~MappedStorage() {
  if (address_ && address_ != MAP_FAILED)
    (void)munmap(address_, size_);
  if (descriptor_ >= 0)
    (void)close(descriptor_);
}

const std::uint8_t *MappedStorage::data() const {
  return static_cast<const std::uint8_t *>(address_);
}

std::size_t MappedStorage::size() const { return size_; }

void MappedStorage::discard(std::size_t offset, std::size_t bytes) const {
  if (!address_ || address_ == MAP_FAILED || bytes == 0U || offset > size_ ||
      bytes > size_ - offset)
    return;
  const auto page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0)
    return;
  const auto page = static_cast<std::size_t>(page_size);
  const auto begin = offset - (offset % page);
  const auto end_unaligned = offset + bytes;
  const auto end = std::min(
      size_, end_unaligned + (page - end_unaligned % page) % page);
  (void)madvise(static_cast<std::uint8_t *>(address_) + begin, end - begin,
                MADV_DONTNEED);
}

void MappedStorage::evict(std::size_t offset, std::size_t bytes) const {
  discard(offset, bytes);
  if (bytes == 0U || offset > size_ || bytes > size_ - offset)
    return;
  const auto page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0)
    return;
  const auto page = static_cast<std::size_t>(page_size);
  const auto begin = offset - (offset % page);
  const auto end_unaligned = offset + bytes;
  const auto end = std::min(
      size_, end_unaligned + (page - end_unaligned % page) % page);
  // A GPU-resident tensor has finished its host lifetime.  fadvise lets the
  // kernel reclaim its clean file-cache range.  Streamed tensors deliberately
  // use discard(), not evict(), so repeated evaluations do not reread disk.
  if (descriptor_ >= 0)
    (void)posix_fadvise(descriptor_, static_cast<off_t>(begin),
                       static_cast<off_t>(end - begin), POSIX_FADV_DONTNEED);
}

namespace {

constexpr std::array<std::uint8_t, 8> kMagic = {'D', 'I', 'F', 'T', 'N', 'S', '0', '1'};
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kDigestBytes = 32;

void append_u32(std::vector<std::uint8_t> &bytes, std::uint32_t value) {
  for (unsigned shift = 0; shift < 32U; shift += 8U)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

void append_u64(std::vector<std::uint8_t> &bytes, std::uint64_t value) {
  for (unsigned shift = 0; shift < 64U; shift += 8U)
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::uint32_t read_u32(std::span<const std::uint8_t> bytes, std::size_t &offset) {
  if (offset + 4U > bytes.size())
    fail("truncated tensor file");
  std::uint32_t result = 0;
  for (unsigned shift = 0; shift < 32U; shift += 8U)
    result |= static_cast<std::uint32_t>(bytes[offset++]) << shift;
  return result;
}

std::uint64_t read_u64(std::span<const std::uint8_t> bytes, std::size_t &offset) {
  if (offset + 8U > bytes.size())
    fail("truncated tensor file");
  std::uint64_t result = 0;
  for (unsigned shift = 0; shift < 64U; shift += 8U)
    result |= static_cast<std::uint64_t>(bytes[offset++]) << shift;
  return result;
}

std::vector<std::uint8_t> read_all(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    fail("cannot open tensor file: " + path.string());
  const auto end = input.tellg();
  if (end < 0)
    fail("cannot size tensor file: " + path.string());
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
  input.seekg(0);
  input.read(reinterpret_cast<char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input)
    fail("cannot read tensor file: " + path.string());
  return bytes;
}

struct TensorHeader {
  ir::DType dtype{};
  std::vector<std::uint64_t> dims;
  std::size_t data_offset{};
  std::size_t byte_count{};
};

TensorHeader parse_tensor_file(std::span<const std::uint8_t> file) {
  if (file.size() < kMagic.size() + 3U * sizeof(std::uint32_t) +
                        sizeof(std::uint64_t) + kDigestBytes)
    fail("tensor file is too small");
  const auto payload = file.first(file.size() - kDigestBytes);
  const auto actual_digest = sha256(payload);
  if (!std::equal(actual_digest.begin(), actual_digest.end(),
                  file.end() - static_cast<std::ptrdiff_t>(kDigestBytes)))
    fail("tensor SHA-256 mismatch");
  std::size_t offset = 0;
  if (!std::equal(kMagic.begin(), kMagic.end(), payload.begin()))
    fail("invalid tensor magic");
  offset += kMagic.size();
  if (read_u32(payload, offset) != kVersion)
    fail("unsupported tensor version");
  TensorHeader header;
  header.dtype = static_cast<ir::DType>(read_u32(payload, offset));
  const auto rank = read_u32(payload, offset);
  if (rank == 0 || rank > ir::kMaxRank)
    fail("invalid tensor rank");
  header.dims.reserve(rank);
  for (std::uint32_t axis = 0; axis < rank; ++axis)
    header.dims.push_back(read_u64(payload, offset));
  const auto byte_count = read_u64(payload, offset);
  if (byte_count > payload.size() - offset || byte_count != payload.size() - offset)
    fail("invalid tensor byte count");
  header.data_offset = offset;
  header.byte_count = static_cast<std::size_t>(byte_count);
  return header;
}

} // namespace

std::uint64_t Tensor::element_count() const {
  std::uint64_t count = 1;
  for (const auto dim : dims) {
    if (dim == 0 || count > std::numeric_limits<std::uint64_t>::max() / dim)
      fail("tensor element count overflow or zero dimension");
    count *= dim;
  }
  return count;
}

void Tensor::validate() const {
  if (dims.empty() || dims.size() > ir::kMaxRank)
    fail("invalid tensor rank");
  const auto count = element_count();
  const auto width = ir::dtype_size(dtype);
  if (count > std::numeric_limits<std::uint64_t>::max() / width ||
      count * width != byte_size())
    fail("tensor byte count does not match shape and dtype");
}

const std::uint8_t *Tensor::data() const {
  if (mapping) {
    if (mapping_offset > mapping->size() ||
        mapping_bytes > mapping->size() - mapping_offset)
      fail("mapped tensor extent is invalid");
    return mapping->data() + mapping_offset;
  }
  return bytes.data();
}

std::uint8_t *Tensor::mutable_data() {
  if (mapping)
    fail("mapped tensor storage is read only");
  return bytes.data();
}

std::size_t Tensor::byte_size() const {
  return mapping ? mapping_bytes : bytes.size();
}

void Tensor::discard_mapped_pages() const {
  if (!mapping || mapping_bytes == 0U)
    return;
  mapping->discard(mapping_offset, mapping_bytes);
}

void Tensor::evict_mapped_pages() const {
  if (!mapping || mapping_bytes == 0U)
    return;
  mapping->evict(mapping_offset, mapping_bytes);
}

std::span<float> Tensor::f32() {
  validate();
  if (dtype != ir::DType::F32)
    fail("tensor is not f32");
  return {reinterpret_cast<float *>(mutable_data()), byte_size() / sizeof(float)};
}

std::span<const float> Tensor::f32() const {
  validate();
  if (dtype != ir::DType::F32)
    fail("tensor is not f32");
  return {reinterpret_cast<const float *>(data()), byte_size() / sizeof(float)};
}

Tensor read_tensor(const std::filesystem::path &path) {
  const auto file = read_all(path);
  const auto header = parse_tensor_file(file);
  Tensor tensor{header.dtype, header.dims, {}};
  tensor.bytes.assign(file.begin() + static_cast<std::ptrdiff_t>(header.data_offset),
                      file.begin() + static_cast<std::ptrdiff_t>(header.data_offset +
                                                                 header.byte_count));
  tensor.validate();
  return tensor;
}

Tensor map_tensor(const std::filesystem::path &path) {
  auto storage = map_readonly_file(path);
  const auto file = std::span<const std::uint8_t>(storage->data(), storage->size());
  const auto header = parse_tensor_file(file);
  Tensor tensor{header.dtype, header.dims, {}};
  tensor.mapping = std::move(storage);
  tensor.mapping_offset = header.data_offset;
  tensor.mapping_bytes = header.byte_count;
  tensor.validate();
  // Checksum verification touched every page. Drop those clean file-backed
  // pages so mapping a full checkpoint does not imply retaining it in RAM.
  tensor.mapping->discard(0U, tensor.mapping->size());
  return tensor;
}

std::shared_ptr<const MappedStorage>
map_readonly_file(const std::filesystem::path &path) {
  const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor < 0)
    fail("cannot open mapped tensor file: " + path.string());
  struct stat status {};
  if (fstat(descriptor, &status) != 0 || status.st_size <= 0) {
    (void)close(descriptor);
    fail("cannot stat mapped tensor file: " + path.string());
  }
  const auto size = static_cast<std::size_t>(status.st_size);
  void *address = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, descriptor, 0);
  if (address == MAP_FAILED) {
    (void)close(descriptor);
    fail("cannot mmap tensor file: " + path.string());
  }
  return std::shared_ptr<const MappedStorage>(
      new MappedStorage(address, size, descriptor));
}

Tensor map_tensor_slice(std::shared_ptr<const MappedStorage> storage,
                        ir::DType dtype, std::vector<std::uint64_t> dims,
                        std::uint64_t file_offset,
                        std::uint64_t byte_count) {
  if (!storage || file_offset > storage->size() ||
      byte_count > storage->size() - file_offset ||
      file_offset > std::numeric_limits<std::size_t>::max() ||
      byte_count > std::numeric_limits<std::size_t>::max())
    fail("mapped tensor slice is outside the file");
  Tensor tensor{dtype, std::move(dims), {}};
  tensor.mapping = std::move(storage);
  tensor.mapping_offset = static_cast<std::size_t>(file_offset);
  tensor.mapping_bytes = static_cast<std::size_t>(byte_count);
  tensor.validate();
  return tensor;
}

Tensor map_tensor_slice(const std::filesystem::path &path, ir::DType dtype,
                        std::vector<std::uint64_t> dims,
                        std::uint64_t file_offset,
                        std::uint64_t byte_count) {
  return map_tensor_slice(map_readonly_file(path), dtype, std::move(dims),
                          file_offset, byte_count);
}

std::vector<Tensor>
map_tensor_slices(const std::filesystem::path &path,
                  const std::vector<TensorSlice> &slices) {
  if (slices.empty())
    return {};
  auto storage = map_readonly_file(path);
  const auto size = static_cast<std::uint64_t>(storage->size());
  for (const auto &slice : slices) {
    if (slice.file_offset > size ||
        slice.byte_count > size - slice.file_offset ||
        slice.file_offset > std::numeric_limits<std::size_t>::max() ||
        slice.byte_count > std::numeric_limits<std::size_t>::max())
      fail("mapped tensor slice is outside the file");
  }
  std::vector<Tensor> tensors;
  tensors.reserve(slices.size());
  for (const auto &slice : slices) {
    tensors.push_back(map_tensor_slice(storage, slice.dtype, slice.dims,
                                       slice.file_offset, slice.byte_count));
  }
  return tensors;
}

void write_tensor(const Tensor &tensor, const std::filesystem::path &path) {
  tensor.validate();
  std::vector<std::uint8_t> file;
  file.insert(file.end(), kMagic.begin(), kMagic.end());
  append_u32(file, kVersion);
  append_u32(file, static_cast<std::uint32_t>(tensor.dtype));
  append_u32(file, static_cast<std::uint32_t>(tensor.dims.size()));
  for (const auto dim : tensor.dims)
    append_u64(file, dim);
  append_u64(file, tensor.byte_size());
  file.insert(file.end(), tensor.data(), tensor.data() + tensor.byte_size());
  const auto digest = sha256(file);
  file.insert(file.end(), digest.begin(), digest.end());

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    fail("cannot create tensor file: " + path.string());
  output.write(reinterpret_cast<const char *>(file.data()),
               static_cast<std::streamsize>(file.size()));
  if (!output)
    fail("cannot write tensor file: " + path.string());
}

Tensor zeros(const ir::TensorDesc &desc) {
  Tensor tensor{desc.dtype, desc.dims, {}};
  tensor.bytes.resize(static_cast<std::size_t>(desc.byte_count()), 0U);
  return tensor;
}

Tensor convert_float_tensor(const Tensor &source, ir::DType destination) {
  source.validate();
  if (!is_float_dtype(source.dtype) || !is_float_dtype(destination))
    fail("convert_float_tensor requires floating source and destination");
  if (source.dtype == destination)
    fail("convert_float_tensor requires distinct source and destination dtypes");
  Tensor output{destination, source.dims, {}};
  output.bytes.resize(static_cast<std::size_t>(
      source.element_count() * ir::dtype_size(destination)));
  for (std::uint64_t index = 0U; index < source.element_count(); ++index)
    store_float(output, index, load_float(source, index));
  output.validate();
  return output;
}

} // namespace dif::runtime
