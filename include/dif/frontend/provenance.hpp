#pragma once

// Frontend-recorded provenance: which creator module, block, and semantic
// section each DiffIR operation was lowered from, and which creator
// checkpoint name each constant tensor carries. DiffIR itself stays
// hardware- and provenance-neutral; the table is a sidecar the frontend
// writes while it builds the program, never a reconstruction from tensor
// names or opcode patterns. `difinspect --source` joins it with bundles,
// plans, and runtime traces.

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dif::frontend {

inline constexpr const char *kProvenanceKind =
    "diffusion-compiler-provenance";
inline constexpr int kProvenanceVersion = 1;

struct ProvenanceRecord {
  std::uint32_t operation_id{};
  // Creator module the lowering followed (checkpoint namespace or creator
  // function), e.g. "attn" or "block.forward".
  std::string creator_module;
  // Block/layer index, or -1 when the operation is outside any block.
  std::int64_t block{-1};
  // Section within the module, e.g. "attention.qkv", "mlp.down",
  // "timestep.tower". Frontend vocabulary, recorded at build time.
  std::string semantic_tag;
};

struct ProvenanceTable {
  std::string frontend;
  std::string creator;
  std::string creator_revision;
  std::vector<ProvenanceRecord> records;
  // Constant tensor id -> creator checkpoint tensor name.
  std::vector<std::pair<std::uint32_t, std::string>> weight_names;

  const ProvenanceRecord *find(std::uint32_t operation_id) const;
  const std::string *weight_name(std::uint32_t tensor_id) const;
};

std::string serialize_provenance(const ProvenanceTable &table);
ProvenanceTable parse_provenance(std::string_view text);
void write_provenance(const ProvenanceTable &table,
                      const std::filesystem::path &path);
ProvenanceTable read_provenance(const std::filesystem::path &path);

// Conventional sidecar path for a program file: FILE.difir.provenance.json.
std::filesystem::path provenance_sidecar_path(
    const std::filesystem::path &program_path);

} // namespace dif::frontend
