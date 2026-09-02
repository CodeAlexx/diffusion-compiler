#include "dif/weights/safetensors.hpp"

#include "dif/support/error.hpp"
#include "dif/support/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dif::weights {
namespace {

constexpr std::uint64_t kMaxJsonBytes = 128ULL * 1024ULL * 1024ULL;

std::string read_text(const std::filesystem::path &path,
                      std::uint64_t maximum = kMaxJsonBytes) {
  const auto size = std::filesystem::file_size(path);
  if (size > maximum)
    fail("JSON file exceeds safety limit: " + path.string());
  std::string output(static_cast<std::size_t>(size), '\0');
  std::ifstream input(path, std::ios::binary);
  if (!input)
    fail("cannot open JSON file: " + path.string());
  input.read(output.data(), static_cast<std::streamsize>(output.size()));
  if (!input)
    fail("cannot read JSON file: " + path.string());
  return output;
}

std::uint64_t unsigned_number(const json::Value &value, const char *label) {
  const auto number = value.number();
  // The native JSON representation is IEEE-754 double. Reject integers above
  // its exact range instead of silently rounding an attacker-controlled shape
  // or byte offset during conversion to uint64_t.
  constexpr double maximum_exact_integer = 9007199254740991.0;
  if (number < 0.0 || number > maximum_exact_integer ||
      std::floor(number) != number)
    fail(std::string("SafeTensors ") + label + " is not an unsigned integer");
  return static_cast<std::uint64_t>(number);
}

std::optional<ir::DType> dtype(std::string_view name) {
  if (name == "F32")
    return ir::DType::F32;
  if (name == "BF16")
    return ir::DType::BF16;
  if (name == "F16")
    return ir::DType::F16;
  if (name == "I8")
    return ir::DType::I8;
  if (name == "I32")
    return ir::DType::I32;
  if (name == "BOOL")
    return ir::DType::Bool;
  if (name == "F8_E4M3")
    return ir::DType::FP8E4M3;
  if (name == "F8_E8M0")
    return ir::DType::FP8E8M0;
  return std::nullopt;
}

std::size_t safetensors_dtype_size(std::string_view name) {
  if (const auto parsed = dtype(name))
    return ir::dtype_size(*parsed);
  if (name == "U8")
    return 1U;
  if (name == "I64")
    return 8U;
  fail("unsupported SafeTensors dtype: " + std::string(name));
}

std::uint64_t little_u64(const unsigned char *bytes) {
  std::uint64_t output = 0;
  for (unsigned shift = 0; shift < 64U; shift += 8U)
    output |= static_cast<std::uint64_t>(bytes[shift / 8U]) << shift;
  return output;
}

std::string safetensors_dtype_name(ir::DType value) {
  switch (value) {
  case ir::DType::F32:
    return "F32";
  case ir::DType::BF16:
    return "BF16";
  case ir::DType::F16:
    return "F16";
  case ir::DType::I8:
    return "I8";
  case ir::DType::I32:
    return "I32";
  case ir::DType::Bool:
    return "BOOL";
  case ir::DType::FP8E4M3:
    return "F8_E4M3";
  case ir::DType::FP8E8M0:
    return "F8_E8M0";
  }
  fail("cannot write unknown SafeTensors dtype");
}

std::string json_string(std::string_view value) {
  std::ostringstream output;
  output << '"';
  for (const auto character : value) {
    const auto byte = static_cast<unsigned char>(character);
    if (character == '"' || character == '\\')
      output << '\\' << character;
    else if (byte < 0x20U)
      output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
             << static_cast<unsigned>(byte) << std::dec;
    else
      output << character;
  }
  output << '"';
  return output.str();
}

} // namespace

struct SafeTensorWriter::Impl {
  explicit Impl(const std::filesystem::path &value_path,
                std::vector<SafeTensorWriteSpec> value_specs)
      : path(value_path), specs(std::move(value_specs)),
        output(path, std::ios::binary | std::ios::trunc) {
    if (specs.empty())
      fail("cannot write an empty SafeTensors file");
    if (!output)
      fail("cannot create SafeTensors file: " + path.string());
    std::uint64_t relative_offset = 0U;
    std::ostringstream header;
    header << '{';
    std::map<std::string, bool, std::less<>> names;
    for (std::size_t index = 0; index < specs.size(); ++index) {
      const auto &spec = specs[index];
      if (spec.name.empty() || !names.emplace(spec.name, true).second ||
          spec.dims.empty() || spec.dims.size() > ir::kMaxRank)
        fail("invalid SafeTensors write specification");
      std::uint64_t elements = 1U;
      for (const auto dimension : spec.dims) {
        if (dimension == 0U ||
            elements > std::numeric_limits<std::uint64_t>::max() / dimension)
          fail("SafeTensors write shape overflow");
        elements *= dimension;
      }
      const auto width = ir::dtype_size(spec.dtype);
      if (elements > std::numeric_limits<std::uint64_t>::max() / width)
        fail("SafeTensors write byte count overflow");
      const auto bytes = elements * width;
      if (relative_offset > std::numeric_limits<std::uint64_t>::max() - bytes)
        fail("SafeTensors write offset overflow");
      byte_counts.push_back(bytes);
      if (index != 0U)
        header << ',';
      header << json_string(spec.name) << ":{\"dtype\":"
             << json_string(safetensors_dtype_name(spec.dtype))
             << ",\"shape\":[";
      for (std::size_t axis = 0; axis < spec.dims.size(); ++axis)
        header << (axis == 0U ? "" : ",") << spec.dims[axis];
      header << "],\"data_offsets\":[" << relative_offset << ','
             << relative_offset + bytes << "]}";
      relative_offset += bytes;
    }
    header << '}';
    header_text = header.str();
    while (header_text.size() % 8U != 0U)
      header_text.push_back(' ');
    const auto header_size = static_cast<std::uint64_t>(header_text.size());
    for (unsigned shift = 0; shift < 64U; shift += 8U)
      output.put(static_cast<char>(header_size >> shift));
    output.write(header_text.data(),
                 static_cast<std::streamsize>(header_text.size()));
    if (!output)
      fail("cannot write SafeTensors header");
  }

  std::filesystem::path path;
  std::vector<SafeTensorWriteSpec> specs;
  std::vector<std::uint64_t> byte_counts;
  std::string header_text;
  std::ofstream output;
  std::size_t next{};
  bool finished{};
};

SafeTensorWriter::SafeTensorWriter(
    const std::filesystem::path &path,
    std::vector<SafeTensorWriteSpec> tensors)
    : impl_(std::make_unique<Impl>(path, std::move(tensors))) {}

SafeTensorWriter::~SafeTensorWriter() = default;
SafeTensorWriter::SafeTensorWriter(SafeTensorWriter &&) noexcept = default;
SafeTensorWriter &SafeTensorWriter::operator=(SafeTensorWriter &&) noexcept =
    default;

void SafeTensorWriter::append(std::string_view name,
                              std::span<const std::uint8_t> bytes) {
  if (!impl_ || impl_->finished || impl_->next >= impl_->specs.size())
    fail("SafeTensors writer append is out of sequence");
  if (impl_->specs[impl_->next].name != name ||
      impl_->byte_counts[impl_->next] != bytes.size())
    fail("SafeTensors writer tensor name or byte count mismatch");
  impl_->output.write(reinterpret_cast<const char *>(bytes.data()),
                      static_cast<std::streamsize>(bytes.size()));
  if (!impl_->output)
    fail("cannot append SafeTensors tensor: " + std::string(name));
  ++impl_->next;
}

SafeTensorFile SafeTensorWriter::finish() {
  if (!impl_ || impl_->finished || impl_->next != impl_->specs.size())
    fail("SafeTensors writer cannot finish before every tensor is appended");
  impl_->output.flush();
  if (!impl_->output)
    fail("cannot flush SafeTensors file");
  impl_->output.close();
  impl_->finished = true;
  return read_safetensors(impl_->path);
}

const SafeTensorEntry *SafeTensorFile::find(std::string_view name) const {
  const auto found = tensors.find(name);
  return found == tensors.end() ? nullptr : &found->second;
}

const SafeTensorMetadataEntry *
SafeTensorFile::find_metadata(std::string_view name) const {
  const auto found = metadata_tensors.find(name);
  return found == metadata_tensors.end() ? nullptr : &found->second;
}

SafeTensorFile read_safetensors(const std::filesystem::path &path) {
  SafeTensorFile output;
  output.path = path;
  output.file_size = std::filesystem::file_size(path);
  if (output.file_size < 10U)
    fail("SafeTensors file is too small: " + path.string());
  std::ifstream input(path, std::ios::binary);
  if (!input)
    fail("cannot open SafeTensors file: " + path.string());
  unsigned char length_bytes[8]{};
  input.read(reinterpret_cast<char *>(length_bytes), sizeof(length_bytes));
  if (!input)
    fail("cannot read SafeTensors header length");
  const auto header_bytes = little_u64(length_bytes);
  if (header_bytes == 0U || header_bytes > kMaxJsonBytes ||
      header_bytes > output.file_size - 8U)
    fail("invalid SafeTensors header length");
  std::string header(static_cast<std::size_t>(header_bytes), '\0');
  input.read(header.data(), static_cast<std::streamsize>(header.size()));
  if (!input)
    fail("cannot read SafeTensors header");
  output.data_offset = 8U + header_bytes;
  const auto root = json::parse(header);
  if (!root.is_object())
    fail("SafeTensors header root must be an object");

  std::vector<std::pair<std::uint64_t, std::uint64_t>> all_ranges;
  for (const auto &[name, value] : root.object()) {
    if (name == "__metadata__")
      continue;
    const auto *dtype_value = value.find("dtype");
    const auto *shape_value = value.find("shape");
    const auto *offset_value = value.find("data_offsets");
    if (!dtype_value || !shape_value || !offset_value || !value.is_object() ||
        !shape_value->is_array() || !offset_value->is_array() ||
        offset_value->array().size() != 2U)
      fail("invalid SafeTensors tensor record: " + name);
    SafeTensorEntry entry;
    entry.name = name;
    const auto dtype_name = dtype_value->string();
    const auto parsed_dtype = dtype(dtype_name);
    for (const auto &dimension : shape_value->array())
      entry.dims.push_back(unsigned_number(dimension, "shape"));
    // SafeTensors permits rank-zero scalar buffers (for example PyTorch's
    // num_batches_tracked). Keep them in the file inventory so an otherwise
    // valid creator checkpoint remains mappable; runtime Tensor values still
    // require rank >= 1 and fail closed if a caller tries to bind a scalar.
    if (entry.dims.size() > ir::kMaxRank)
      fail("invalid SafeTensors tensor rank: " + name);
    const auto relative_begin =
        unsigned_number(offset_value->array()[0], "data offset");
    const auto relative_end =
        unsigned_number(offset_value->array()[1], "data offset");
    if (relative_end < relative_begin ||
        relative_end > output.file_size - output.data_offset)
      fail("SafeTensors tensor offset is out of bounds: " + name);
    entry.file_offset = output.data_offset + relative_begin;
    entry.byte_count = relative_end - relative_begin;
    all_ranges.emplace_back(entry.file_offset, entry.byte_count);
    std::uint64_t elements = 1U;
    for (const auto dimension : entry.dims) {
      if (dimension == 0U ||
          elements > std::numeric_limits<std::uint64_t>::max() / dimension)
        fail("SafeTensors tensor shape overflow: " + name);
      elements *= dimension;
    }
    const auto width = safetensors_dtype_size(dtype_name);
    if (elements > std::numeric_limits<std::uint64_t>::max() / width ||
        elements * width != entry.byte_count)
      fail("SafeTensors tensor byte count mismatch: " + name);
    if (parsed_dtype) {
      entry.dtype = *parsed_dtype;
      output.tensors.emplace(name, std::move(entry));
    } else if (name.starts_with("__meta__.") || entry.dims.empty()) {
      // Creator checkpoints commonly carry unsupported scalar bookkeeping
      // buffers (not executable weights), such as an I64 BatchNorm counter.
      // Preserve their metadata/range while keeping them unbindable.
      output.metadata_tensors.emplace(
          name, SafeTensorMetadataEntry{name, dtype_name, entry.dims,
                                        entry.file_offset, entry.byte_count});
    } else {
      fail("unsupported non-metadata SafeTensors dtype " + dtype_name +
           " for tensor " + name);
    }
  }
  if (output.tensors.empty())
    fail("SafeTensors file has no tensors");

  std::sort(all_ranges.begin(), all_ranges.end(), [](const auto &a,
                                                     const auto &b) {
    return a.first < b.first;
  });
  std::uint64_t expected = output.data_offset;
  for (const auto &[offset, bytes] : all_ranges) {
    if (offset != expected)
      fail("SafeTensors data contains a hole or overlap");
    expected += bytes;
  }
  if (expected != output.file_size)
    fail("SafeTensors tensor data does not cover the file");
  output.mapping = runtime::map_readonly_file(path);
  return output;
}

SafeTensorIndex read_safetensors_index(const std::filesystem::path &path) {
  SafeTensorIndex output;
  output.path = path;
  const auto root = json::parse(read_text(path));
  const auto *weight_map = root.find("weight_map");
  if (!weight_map || !weight_map->is_object())
    fail("SafeTensors index lacks an object weight_map");
  for (const auto &[name, shard] : weight_map->object())
    output.weight_map.emplace(name, path.parent_path() / shard.string());
  if (output.weight_map.empty())
    fail("SafeTensors index weight_map is empty");
  return output;
}

runtime::Tensor map_safetensor(const SafeTensorFile &file,
                               std::string_view name) {
  const auto *entry = file.find(name);
  if (!entry)
    fail("SafeTensors tensor is missing: " + std::string(name));
  if (!file.mapping)
    fail("SafeTensors file has no shared read-only mapping");
  return runtime::map_tensor_slice(file.mapping, entry->dtype, entry->dims,
                                   entry->file_offset, entry->byte_count);
}

std::vector<std::uint8_t>
read_safetensor_metadata(const SafeTensorFile &file, std::string_view name) {
  const auto *entry = file.find_metadata(name);
  if (!entry)
    fail("SafeTensors metadata tensor is missing: " + std::string(name));
  if (entry->byte_count > std::numeric_limits<std::size_t>::max())
    fail("SafeTensors metadata tensor exceeds host size range");
  std::vector<std::uint8_t> bytes(
      static_cast<std::size_t>(entry->byte_count));
  std::ifstream input(file.path, std::ios::binary);
  if (!input)
    fail("cannot open SafeTensors metadata source: " + file.path.string());
  input.seekg(static_cast<std::streamoff>(entry->file_offset));
  if (!input)
    fail("cannot seek SafeTensors metadata source");
  input.read(reinterpret_cast<char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input)
    fail("cannot read SafeTensors metadata tensor: " + std::string(name));
  return bytes;
}

} // namespace dif::weights
