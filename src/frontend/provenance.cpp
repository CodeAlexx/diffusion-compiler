#include "dif/frontend/provenance.hpp"

#include "dif/support/error.hpp"
#include "dif/support/json.hpp"
#include "dif/telemetry/document.hpp"

#include <cmath>
#include <fstream>
#include <sstream>

namespace dif::frontend {

const ProvenanceRecord *ProvenanceTable::find(std::uint32_t operation_id) const {
  for (const auto &record : records)
    if (record.operation_id == operation_id)
      return &record;
  return nullptr;
}

const std::string *ProvenanceTable::weight_name(std::uint32_t tensor_id) const {
  for (const auto &[id, name] : weight_names)
    if (id == tensor_id)
      return &name;
  return nullptr;
}

std::string serialize_provenance(const ProvenanceTable &table) {
  telemetry::Object out;
  out.set("kind", kProvenanceKind);
  out.set("version", kProvenanceVersion);
  out.set("frontend", table.frontend);
  out.set("creator", table.creator);
  out.set("creator_revision", table.creator_revision);
  telemetry::Array records;
  for (const auto &record : table.records) {
    telemetry::Object entry;
    entry.set("operation", record.operation_id);
    entry.set("module", record.creator_module);
    entry.set("block", record.block);
    entry.set("tag", record.semantic_tag);
    records.push_back(std::move(entry));
  }
  out.set("operations", std::move(records));
  telemetry::Array weights;
  for (const auto &[id, name] : table.weight_names) {
    telemetry::Object entry;
    entry.set("tensor", id);
    entry.set("name", name);
    weights.push_back(std::move(entry));
  }
  out.set("weights", std::move(weights));
  return telemetry::serialize(telemetry::Value(std::move(out)));
}

namespace {

const json::Value &required(const json::Value &object, const char *key) {
  const auto *value = object.find(key);
  if (!value)
    fail(std::string("provenance table is missing field '") + key + "'");
  return *value;
}

std::uint32_t integer_u32(const json::Value &value, const char *label) {
  const auto number = value.number();
  if (!(number >= 0.0) || number != std::floor(number) || number > 4294967295.0)
    fail(std::string("provenance ") + label + " must be a U32 integer");
  return static_cast<std::uint32_t>(number);
}

} // namespace

ProvenanceTable parse_provenance(std::string_view text) {
  const auto document = json::parse(text);
  if (!document.is_object())
    fail("provenance table is not a JSON object");
  if (required(document, "kind").string() != kProvenanceKind)
    fail("file is not a diffusion-compiler provenance table");
  if (required(document, "version").number() != kProvenanceVersion)
    fail("unsupported provenance table version");
  ProvenanceTable table;
  table.frontend = required(document, "frontend").string();
  table.creator = required(document, "creator").string();
  table.creator_revision = required(document, "creator_revision").string();
  const auto &records = required(document, "operations");
  if (!records.is_array())
    fail("provenance operations must be an array");
  for (const auto &entry : records.array()) {
    ProvenanceRecord record;
    record.operation_id = integer_u32(required(entry, "operation"), "operation");
    record.creator_module = required(entry, "module").string();
    const auto block = required(entry, "block").number();
    if (block != std::floor(block))
      fail("provenance block must be an integer");
    record.block = static_cast<std::int64_t>(block);
    record.semantic_tag = required(entry, "tag").string();
    table.records.push_back(std::move(record));
  }
  const auto &weights = required(document, "weights");
  if (!weights.is_array())
    fail("provenance weights must be an array");
  for (const auto &entry : weights.array())
    table.weight_names.emplace_back(
        integer_u32(required(entry, "tensor"), "tensor"),
        required(entry, "name").string());
  return table;
}

void write_provenance(const ProvenanceTable &table,
                      const std::filesystem::path &path) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream)
    fail("cannot open provenance table for writing: " + path.string());
  const auto text = serialize_provenance(table);
  stream.write(text.data(), static_cast<std::streamsize>(text.size()));
  if (!stream)
    fail("cannot write provenance table: " + path.string());
}

ProvenanceTable read_provenance(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    fail("cannot open provenance table: " + path.string());
  std::stringstream buffer;
  buffer << stream.rdbuf();
  return parse_provenance(buffer.str());
}

std::filesystem::path provenance_sidecar_path(
    const std::filesystem::path &program_path) {
  return std::filesystem::path(program_path.string() + ".provenance.json");
}

} // namespace dif::frontend
