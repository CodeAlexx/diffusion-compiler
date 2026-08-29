#include "dif/opt/transform.hpp"

#include "dif/support/error.hpp"

#include <array>
#include <charconv>
#include <limits>
#include <string>

namespace dif::opt {
namespace {

struct KindName {
  TransformKind kind;
  std::string_view name;
  TransformClass classification;
};

constexpr std::array<KindName, 16> kKindNames = {{
    {TransformKind::FoldConstantSubgraph, "fold_constant_subgraph",
     TransformClass::Structural},
    {TransformKind::EliminateDeadOperations, "eliminate_dead_operations",
     TransformClass::Structural},
    {TransformKind::CommonSubexpression, "common_subexpression",
     TransformClass::Structural},
    // Folding a bias into a Linear seeds the accumulator with it instead of
    // adding it afterwards. That is a real change of accumulation order, so the
    // transform is classified numeric and has to clear the gate.
    {TransformKind::FuseLinearBias, "fuse_linear_bias",
     TransformClass::Numeric},
    {TransformKind::FuseQkvProjection, "fuse_qkv_projection",
     TransformClass::Structural},
    {TransformKind::SplitQkvProjection, "split_qkv_projection",
     TransformClass::Structural},
    {TransformKind::ElideCastRoundTrip, "elide_cast_round_trip",
     TransformClass::Structural},
    {TransformKind::RematerializeProducer, "rematerialize_producer",
     TransformClass::Structural},
    {TransformKind::SetBlockSize, "set_block_size", TransformClass::Schedule},
    {TransformKind::SetTileShape, "set_tile_shape", TransformClass::Schedule},
    {TransformKind::SetLinearImplementation, "set_linear_implementation",
     TransformClass::Numeric},
    {TransformKind::SetAttentionImplementation, "set_attention_implementation",
     TransformClass::Numeric},
    {TransformKind::SetOperationPrecision, "set_operation_precision",
     TransformClass::Numeric},
    {TransformKind::QuantizeConstantWeights, "quantize_constant_weights",
     TransformClass::Numeric},
    {TransformKind::SetConstantResidency, "set_constant_residency",
     TransformClass::Memory},
    {TransformKind::SetPrefetchDistance, "set_prefetch_distance",
     TransformClass::Memory},
}};

const KindName *lookup(TransformKind kind) {
  for (const auto &entry : kKindNames) {
    if (entry.kind == kind)
      return &entry;
  }
  return nullptr;
}

template <typename T>
void append_list(std::string &out, std::string_view label,
                 const std::vector<T> &values) {
  out += ' ';
  out += label;
  out += '=';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0U)
      out += ',';
    out += std::to_string(values[index]);
  }
}

template <typename T>
std::vector<T> parse_list(std::string_view text, const char *label) {
  std::vector<T> values;
  while (!text.empty()) {
    const auto comma = text.find(',');
    const auto item = text.substr(0, comma);
    if (item.empty())
      fail(std::string("transform ") + label + " list has an empty element");
    std::uint64_t value = 0;
    const auto *begin = item.data();
    const auto *end = item.data() + item.size();
    const auto parsed = std::from_chars(begin, end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end)
      fail(std::string("transform ") + label + " list has a non-numeric element");
    if (value > static_cast<std::uint64_t>(std::numeric_limits<T>::max()))
      fail(std::string("transform ") + label + " list element is out of range");
    values.push_back(static_cast<T>(value));
    if (comma == std::string_view::npos)
      break;
    text.remove_prefix(comma + 1U);
  }
  return values;
}

std::string_view field(std::string_view text, std::string_view label,
                       bool &found) {
  const auto prefix = std::string(label) + "=";
  if (text.substr(0, prefix.size()) != prefix) {
    found = false;
    return {};
  }
  found = true;
  return text.substr(prefix.size());
}

} // namespace

std::string_view transform_kind_name(TransformKind kind) {
  const auto *entry = lookup(kind);
  if (!entry)
    fail("unknown transform kind");
  return entry->name;
}

bool transform_kind_from_name(std::string_view name, TransformKind &kind) {
  for (const auto &entry : kKindNames) {
    if (entry.name == name) {
      kind = entry.kind;
      return true;
    }
  }
  return false;
}

TransformClass transform_class(TransformKind kind) {
  const auto *entry = lookup(kind);
  if (!entry)
    fail("unknown transform kind");
  return entry->classification;
}

std::string_view transform_class_name(TransformClass value) {
  switch (value) {
  case TransformClass::Structural:
    return "structural";
  case TransformClass::Schedule:
    return "schedule";
  case TransformClass::Numeric:
    return "numeric";
  case TransformClass::Memory:
    return "memory";
  }
  fail("unknown transform class");
}

bool changes_numerics(TransformKind kind) {
  return transform_class(kind) == TransformClass::Numeric;
}

std::string encode_transform(const Transform &transform) {
  std::string out(transform_kind_name(transform.kind));
  append_list(out, "ops", transform.operations);
  append_list(out, "tensors", transform.tensors);
  append_list(out, "params", transform.parameters);
  return out;
}

Transform decode_transform(std::string_view text) {
  Transform transform;
  const auto first = text.find(' ');
  if (first == std::string_view::npos ||
      !transform_kind_from_name(text.substr(0, first), transform.kind))
    fail("transform text does not name a known transform kind");
  text.remove_prefix(first + 1U);
  const auto second = text.find(' ');
  if (second == std::string_view::npos)
    fail("transform text is missing its tensor and parameter fields");
  bool found = false;
  transform.operations =
      parse_list<std::uint32_t>(field(text.substr(0, second), "ops", found),
                                "operation");
  if (!found)
    fail("transform text is missing its operation field");
  text.remove_prefix(second + 1U);
  const auto third = text.find(' ');
  if (third == std::string_view::npos)
    fail("transform text is missing its parameter field");
  transform.tensors =
      parse_list<std::uint32_t>(field(text.substr(0, third), "tensors", found),
                                "tensor");
  if (!found)
    fail("transform text is missing its tensor field");
  text.remove_prefix(third + 1U);
  transform.parameters =
      parse_list<std::uint64_t>(field(text, "params", found), "parameter");
  if (!found)
    fail("transform text is missing its parameter field");
  return transform;
}

std::string encode_transform_sequence(const std::vector<Transform> &transforms) {
  std::string out;
  for (std::size_t index = 0; index < transforms.size(); ++index) {
    if (index != 0U)
      out += " ; ";
    out += encode_transform(transforms[index]);
  }
  return out;
}

bool operator==(const Transform &left, const Transform &right) {
  return left.kind == right.kind && left.operations == right.operations &&
         left.tensors == right.tensors && left.parameters == right.parameters;
}

} // namespace dif::opt
