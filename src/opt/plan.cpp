#include "dif/opt/plan.hpp"

#include "dif/ir/codec.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <span>
#include <vector>

namespace dif::opt {
namespace {

constexpr std::array<std::uint8_t, 8> kMagic = {
    'D', 'I', 'F', 'P', 'L', 'A', 'N', '1'};
constexpr std::uint32_t kVersion = 2U;
constexpr std::size_t kDigestBytes = 32U;
constexpr std::uint32_t kMaximumPasses = 1U << 16U;
constexpr std::uint32_t kMaximumPassName = 4096U;
constexpr std::uint32_t kMaximumTransformations = 1U << 20U;
constexpr std::uint64_t kMaximumFileBytes = 256ULL * 1024ULL * 1024ULL;

class Writer {
public:
  void u32(std::uint32_t value) {
    for (unsigned shift = 0U; shift < 32U; shift += 8U)
      bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  }
  void u64(std::uint64_t value) {
    for (unsigned shift = 0U; shift < 64U; shift += 8U)
      bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  }
  void raw(std::span<const std::uint8_t> value) {
    bytes.insert(bytes.end(), value.begin(), value.end());
  }
  void string(const std::string &value) {
    if (value.size() > kMaximumPassName)
      fail("optimization plan pass name is too large");
    u32(static_cast<std::uint32_t>(value.size()));
    bytes.insert(bytes.end(), value.begin(), value.end());
  }

  std::vector<std::uint8_t> bytes;
};

class Reader {
public:
  explicit Reader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}

  std::uint32_t u32() {
    require(4U);
    std::uint32_t value = 0U;
    for (unsigned shift = 0U; shift < 32U; shift += 8U)
      value |= static_cast<std::uint32_t>(bytes_[offset_++]) << shift;
    return value;
  }
  std::uint64_t u64() {
    require(8U);
    std::uint64_t value = 0U;
    for (unsigned shift = 0U; shift < 64U; shift += 8U)
      value |= static_cast<std::uint64_t>(bytes_[offset_++]) << shift;
    return value;
  }
  std::span<const std::uint8_t> raw(std::size_t size) {
    require(size);
    const auto value = bytes_.subspan(offset_, size);
    offset_ += size;
    return value;
  }
  std::string string() {
    const auto size = u32();
    if (size > kMaximumPassName)
      fail("optimization plan pass name is too large");
    const auto value = raw(size);
    return {reinterpret_cast<const char *>(value.data()), value.size()};
  }
  bool done() const { return offset_ == bytes_.size(); }

private:
  void require(std::size_t size) const {
    if (size > bytes_.size() - offset_)
      fail("truncated optimization plan");
  }

  std::span<const std::uint8_t> bytes_;
  std::size_t offset_{};
};

std::vector<std::uint8_t> read_all(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    fail("cannot open optimization plan: " + path.string());
  const auto end = input.tellg();
  if (end < 0 || static_cast<std::uint64_t>(end) > kMaximumFileBytes)
    fail("optimization plan size is invalid");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
  input.seekg(0);
  input.read(reinterpret_cast<char *>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  if (!input)
    fail("cannot read optimization plan: " + path.string());
  return bytes;
}

Sha256Digest digest_from(Reader &reader) {
  Sha256Digest digest{};
  const auto bytes = reader.raw(digest.size());
  std::copy(bytes.begin(), bytes.end(), digest.begin());
  return digest;
}

} // namespace

Plan make_plan(const ir::Program &base, const Candidate &candidate) {
  const auto base_fingerprint = ir::fingerprint(base);
  const auto replayed = apply_recipe(base, candidate.recipe);
  const auto replayed_fingerprint = ir::fingerprint(replayed);
  if (replayed_fingerprint != candidate.program_fingerprint ||
      replayed_fingerprint != ir::fingerprint(candidate.program))
    fail("optimization candidate recipe does not reproduce its DiffIR");
  const auto replayed_candidate = make_candidate(replayed, candidate.recipe);
  if (replayed_candidate.policy.stream_prefetch_distance !=
          candidate.policy.stream_prefetch_distance ||
      replayed_candidate.candidate_fingerprint !=
          candidate.candidate_fingerprint)
    fail("optimization candidate recipe does not reproduce its policy");
  return {base_fingerprint, replayed_fingerprint,
          candidate.candidate_fingerprint, candidate.policy,
          candidate.recipe};
}

void write_plan(const Plan &plan, const std::filesystem::path &path) {
  // This validates names, kinds, values, and duplicate targets before bytes are
  // persisted even when the caller constructed Plan directly.
  (void)plan.recipe.canonical_text();
  const auto derived_policy = execution_policy(plan.recipe);
  if (derived_policy.stream_prefetch_distance !=
      plan.policy.stream_prefetch_distance)
    fail("optimization plan policy does not match its recipe");
  if (plan.recipe.passes.size() > kMaximumPasses ||
      plan.recipe.transformations.size() > kMaximumTransformations)
    fail("optimization plan has too many entries");

  Writer writer;
  writer.raw(kMagic);
  writer.u32(kVersion);
  writer.raw(plan.base_program_fingerprint);
  writer.raw(plan.candidate_program_fingerprint);
  writer.raw(plan.candidate_fingerprint);
  writer.u64(plan.policy.stream_prefetch_distance);
  writer.u32(static_cast<std::uint32_t>(plan.recipe.passes.size()));
  for (const auto &pass : plan.recipe.passes)
    writer.string(pass);
  writer.u32(static_cast<std::uint32_t>(
      plan.recipe.transformations.size()));
  for (const auto &transformation : plan.recipe.transformations) {
    writer.u32(static_cast<std::uint32_t>(transformation.kind));
    writer.u32(transformation.target_id);
    writer.u32(static_cast<std::uint32_t>(transformation.attribute_key));
    writer.u64(transformation.value);
  }
  const auto digest = sha256(writer.bytes);
  writer.raw(digest);
  if (writer.bytes.size() > kMaximumFileBytes)
    fail("optimization plan exceeds the file-size limit");

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    fail("cannot create optimization plan: " + path.string());
  output.write(reinterpret_cast<const char *>(writer.bytes.data()),
               static_cast<std::streamsize>(writer.bytes.size()));
  if (!output)
    fail("cannot write optimization plan: " + path.string());
}

Plan read_plan(const std::filesystem::path &path) {
  const auto file = read_all(path);
  if (file.size() < kMagic.size() + sizeof(std::uint32_t) +
                        2U * kDigestBytes + 2U * sizeof(std::uint32_t) +
                        kDigestBytes)
    fail("optimization plan is too small");
  const auto bytes = std::span<const std::uint8_t>(file);
  const auto payload = bytes.first(bytes.size() - kDigestBytes);
  const auto expected_digest = sha256(payload);
  if (!std::equal(expected_digest.begin(), expected_digest.end(),
                  bytes.end() - static_cast<std::ptrdiff_t>(kDigestBytes)))
    fail("optimization plan SHA-256 mismatch");

  Reader reader(payload);
  const auto magic = reader.raw(kMagic.size());
  if (!std::equal(kMagic.begin(), kMagic.end(), magic.begin()))
    fail("invalid optimization plan magic");
  const auto version = reader.u32();
  if (version != 1U && version != kVersion)
    fail("unsupported optimization plan version");
  Plan plan;
  plan.base_program_fingerprint = digest_from(reader);
  plan.candidate_program_fingerprint = digest_from(reader);
  if (version >= 2U) {
    plan.candidate_fingerprint = digest_from(reader);
    plan.policy.stream_prefetch_distance = reader.u64();
    if (plan.policy.stream_prefetch_distance > 1U)
      fail("optimization plan has unsupported prefetch distance");
  }
  const auto pass_count = reader.u32();
  if (pass_count > kMaximumPasses)
    fail("optimization plan pass count is unreasonable");
  plan.recipe.passes.reserve(pass_count);
  for (std::uint32_t index = 0U; index < pass_count; ++index)
    plan.recipe.passes.push_back(reader.string());
  const auto transformation_count = reader.u32();
  if (transformation_count > kMaximumTransformations)
    fail("optimization plan transformation count is unreasonable");
  plan.recipe.transformations.reserve(transformation_count);
  for (std::uint32_t index = 0U; index < transformation_count; ++index) {
    Transformation transformation;
    transformation.kind = static_cast<TransformationKind>(reader.u32());
    transformation.target_id = reader.u32();
    transformation.attribute_key = static_cast<ir::AttrKey>(reader.u32());
    transformation.value = reader.u64();
    plan.recipe.transformations.push_back(transformation);
  }
  if (!reader.done())
    fail("trailing bytes in optimization plan");
  (void)plan.recipe.canonical_text();
  const auto derived_policy = execution_policy(plan.recipe);
  if (version == 1U) {
    plan.policy = derived_policy;
    plan.candidate_fingerprint = plan.candidate_program_fingerprint;
  } else if (derived_policy.stream_prefetch_distance !=
             plan.policy.stream_prefetch_distance) {
    fail("optimization plan policy does not match its recipe");
  }
  return plan;
}

ir::Program replay_plan(const ir::Program &base, const Plan &plan) {
  return replay_candidate(base, plan).program;
}

Candidate replay_candidate(const ir::Program &base, const Plan &plan) {
  if (ir::fingerprint(base) != plan.base_program_fingerprint)
    fail("optimization plan base DiffIR fingerprint mismatch");
  auto candidate = make_candidate(apply_recipe(base, plan.recipe), plan.recipe);
  if (candidate.program_fingerprint != plan.candidate_program_fingerprint)
    fail("optimization plan candidate DiffIR fingerprint mismatch");
  if (candidate.policy.stream_prefetch_distance !=
          plan.policy.stream_prefetch_distance ||
      candidate.candidate_fingerprint != plan.candidate_fingerprint)
    fail("optimization plan candidate policy fingerprint mismatch");
  return candidate;
}

} // namespace dif::opt
