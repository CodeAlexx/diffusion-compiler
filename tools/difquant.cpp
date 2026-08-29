#include "dif/compiler/int4.hpp"
#include "dif/compiler/memory_plan.hpp"
#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"
#include "dif/weights/bundle.hpp"
#include "dif/weights/safetensors.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

enum class PayloadKind {
  Copy,
  PackedInt4,
  Scales,
  ColumnScales,
  OutlierIndices,
  OutlierResiduals
};

struct Payload {
  std::uint32_t tensor_id{};
  std::uint32_t source_tensor_id{};
  std::string name;
  dif::ir::DType dtype{};
  std::vector<std::uint64_t> dims;
  PayloadKind kind{};
};

class PartialFileCleanup {
public:
  explicit PartialFileCleanup(std::vector<std::filesystem::path> paths)
      : paths_(std::move(paths)) {}
  ~PartialFileCleanup() {
    if (!armed_)
      return;
    for (const auto &path : paths_) {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
  }
  void disarm() { armed_ = false; }

private:
  std::vector<std::filesystem::path> paths_;
  bool armed_{true};
};

std::uint64_t number(const char *text, const char *label) {
  char *end = nullptr;
  const auto value = std::strtoull(text, &end, 10);
  if (!text[0] || !end || *end != '\0')
    dif::fail(std::string("invalid ") + label + ": " + text);
  return value;
}

float real_number(const char *text, const char *label) {
  char *end = nullptr;
  const auto value = std::strtof(text, &end);
  if (!text[0] || !end || *end != '\0' || !std::isfinite(value))
    dif::fail(std::string("invalid ") + label + ": " + text);
  return value;
}

std::filesystem::path partial_path(const std::filesystem::path &path) {
  return path.string() + ".partial";
}

void require_new_path(const std::filesystem::path &path) {
  if (std::filesystem::exists(path) || std::filesystem::exists(partial_path(path)))
    dif::fail("refusing to overwrite output or partial file: " + path.string());
  const auto parent = path.parent_path();
  if (!parent.empty() && !std::filesystem::is_directory(parent))
    dif::fail("output parent directory does not exist: " + parent.string());
}

dif::Sha256Digest quantization_fingerprint(
    const dif::weights::WeightBundle &source, const dif::ir::Program &rewritten,
    std::uint32_t bit_width, std::uint64_t group_size,
    dif::compiler::Int4Correction correction, bool activation_aware,
    float activation_exponent,
    const std::map<std::uint32_t, float> &tensor_exponents,
    bool direct_linear) {
  std::vector<std::uint8_t> identity;
  identity.insert(identity.end(), source.index_fingerprint.begin(),
                  source.index_fingerprint.end());
  identity.insert(identity.end(), source.program_fingerprint.begin(),
                  source.program_fingerprint.end());
  const auto rewritten_fingerprint = dif::ir::fingerprint(rewritten);
  identity.insert(identity.end(), rewritten_fingerprint.begin(),
                  rewritten_fingerprint.end());
  auto configuration = std::string("dif-lowbit-v1:range-scale:group=") +
                             std::to_string(group_size) + ":bits=" +
                             std::to_string(bit_width) + ":correction=" +
                             (correction ==
                                      dif::compiler::Int4Correction::OneOutlier
                                  ? "one-outlier"
                                  : "none") +
                             (activation_aware
                                  ? ":activation-rms-companding-awls-v1-alpha=" +
                                        std::to_string(activation_exponent)
                                  : ":activation=none");
  configuration += direct_linear ? ":direct-linear=1" : ":direct-linear=0";
  for (const auto &[tensor_id, exponent] : tensor_exponents)
    configuration += ":tensor-alpha-" + std::to_string(tensor_id) + "=" +
                     std::to_string(exponent);
  identity.insert(identity.end(), configuration.begin(), configuration.end());
  return dif::sha256(identity);
}

std::size_t enable_direct_int5_linears(dif::ir::Program &program) {
  std::map<std::uint32_t, const dif::ir::Operation *> producers;
  for (const auto &operation : program.operations)
    for (const auto output : operation.outputs)
      producers.emplace(output, &operation);
  std::size_t changed = 0U;
  for (auto &operation : program.operations) {
    if (operation.opcode != dif::ir::Opcode::Linear ||
        operation.inputs.size() < 2U)
      continue;
    auto producer = producers.find(operation.inputs[1]);
    if (producer == producers.end())
      continue;
    const dif::ir::Operation *dequant = producer->second;
    if (dequant->opcode == dif::ir::Opcode::H3DeinterleaveQkvWeight &&
        dequant->inputs.size() == 1U) {
      producer = producers.find(dequant->inputs[0]);
      dequant = producer == producers.end() ? nullptr : producer->second;
    }
    if (!dequant || dequant->opcode != dif::ir::Opcode::DequantizeInt5)
      continue;
    const auto implementation = std::find_if(
        operation.attributes.begin(), operation.attributes.end(),
        [](const dif::ir::Attribute &attribute) {
          return attribute.key == dif::ir::AttrKey::Implementation;
        });
    if (implementation != operation.attributes.end()) {
      *implementation = dif::ir::Attribute::u64(
          dif::ir::AttrKey::Implementation, 3U);
    } else {
      operation.attributes.push_back(dif::ir::Attribute::u64(
          dif::ir::AttrKey::Implementation, 3U));
    }
    ++changed;
  }
  dif::ir::verify(program);
  return changed;
}

void usage() {
  std::cerr
      << "usage: difquant plan IN.difir BITS GROUP\n"
         "       difquant int4 IN.difir IN.difbind OUT.difir "
         "OUT.safetensors OUT.difbind GROUP [--skip-source-digest]\n"
         "       difquant int4-outlier IN.difir IN.difbind OUT.difir "
         "OUT.safetensors OUT.difbind GROUP [--skip-source-digest]\n"
         "       difquant int5 IN.difir IN.difbind OUT.difir "
         "OUT.safetensors OUT.difbind GROUP [--skip-source-digest]\n"
         "       difquant int5-awq IN.difir IN.difbind OUT.difir "
         "OUT.safetensors OUT.difbind GROUP CALIBRATION_DIR ADALN_INPUT "
         "[--alpha N] [--tensor-alpha ID=N] [--direct-linear] "
         "[--skip-source-digest]\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    const std::string command = argc > 1 ? argv[1] : "";
    if (command == "plan" && argc == 5) {
      const auto program = dif::ir::read_file(argv[2]);
      const auto bit_width = static_cast<std::uint32_t>(
          number(argv[3], "low-bit width"));
      const auto group_size = number(argv[4], "low-bit group size");
      const auto rewrite = dif::compiler::rewrite_lowbit_weights(
          program, bit_width, group_size);
      const auto plan = dif::compiler::plan_memory(rewrite.program);
      std::uint64_t constant_bytes = 0U;
      for (const auto &tensor : rewrite.program.tensors) {
        if (tensor.has_role(dif::ir::TensorRole::Constant)) {
          if (constant_bytes >
              std::numeric_limits<std::uint64_t>::max() - tensor.byte_count())
            dif::fail("low-bit constant byte count overflow");
          constant_bytes += tensor.byte_count();
        }
      }
      std::cout << "LOWBIT_PLAN PASS bits=" << bit_width
                << " group=" << group_size
                << " tensors=" << rewrite.entries.size()
                << " constant_bytes=" << constant_bytes
                << " planned_bytes=" << plan.total_bytes
                << " naive_bytes=" << plan.naive_bytes
                << " fingerprint="
                << dif::hex_digest(dif::ir::fingerprint(rewrite.program))
                << "\n";
      return 0;
    }
    const bool activation_aware = command == "int5-awq";
    if (command != "int4" && command != "int4-outlier" &&
        command != "int5" && !activation_aware) {
      usage();
      return 2;
    }
    const auto required_arguments = activation_aware ? 10 : 8;
    if (argc < required_arguments) {
      usage();
      return 2;
    }
    bool skip_source_digest = false;
    float activation_exponent = 0.5F;
    bool saw_activation_exponent = false;
    bool direct_linear = false;
    std::map<std::uint32_t, float> tensor_exponents;
    for (int index = required_arguments; index < argc; ++index) {
      const std::string option = argv[index];
      if (option == "--skip-source-digest" && !skip_source_digest) {
        skip_source_digest = true;
      } else if (activation_aware && option == "--direct-linear" &&
                 !direct_linear) {
        direct_linear = true;
      } else if (activation_aware && option == "--alpha" &&
                 !saw_activation_exponent && index + 1 < argc) {
        activation_exponent =
            real_number(argv[++index], "activation exponent");
        saw_activation_exponent = true;
      } else if (activation_aware && option == "--tensor-alpha" &&
                 index + 1 < argc) {
        const std::string assignment = argv[++index];
        const auto separator = assignment.find('=');
        if (separator == std::string::npos || separator == 0U ||
            separator + 1U == assignment.size())
          dif::fail("tensor alpha must have the form ID=N");
        const auto tensor_id_value = number(
            assignment.substr(0U, separator).c_str(), "tensor alpha id");
        if (tensor_id_value == 0U ||
            tensor_id_value > std::numeric_limits<std::uint32_t>::max())
          dif::fail("tensor alpha id is outside the DiffIR id range");
        const auto exponent = real_number(
            assignment.substr(separator + 1U).c_str(), "tensor exponent");
        if (exponent < 0.0F || exponent > 1.0F)
          dif::fail("tensor exponent must be in [0,1]");
        if (!tensor_exponents
                 .emplace(static_cast<std::uint32_t>(tensor_id_value), exponent)
                 .second)
          dif::fail("duplicate tensor alpha id");
      } else {
        usage();
        return 2;
      }
    }
    if (activation_exponent < 0.0F || activation_exponent > 1.0F)
      dif::fail("activation exponent must be in [0,1]");
    const std::filesystem::path source_program_path = argv[2];
    const std::filesystem::path source_bundle_path = argv[3];
    const auto output_program_path =
        std::filesystem::absolute(argv[4]).lexically_normal();
    const auto output_shard_path =
        std::filesystem::absolute(argv[5]).lexically_normal();
    const auto output_bundle_path =
        std::filesystem::absolute(argv[6]).lexically_normal();
    const auto group_size = number(argv[7], "INT4 group size");
    const auto correction =
        command == "int4-outlier"
            ? dif::compiler::Int4Correction::OneOutlier
            : dif::compiler::Int4Correction::None;
    const std::uint32_t bit_width =
        (command == "int5" || activation_aware) ? 5U : 4U;
    const std::filesystem::path calibration_directory =
        activation_aware ? std::filesystem::path(argv[8])
                         : std::filesystem::path{};
    const std::filesystem::path adaln_input =
        activation_aware ? std::filesystem::path(argv[9])
                         : std::filesystem::path{};
    if (activation_aware &&
        (!std::filesystem::is_directory(calibration_directory) ||
         !std::filesystem::is_regular_file(adaln_input)))
      dif::fail("activation-aware quantization calibration paths are invalid");
    require_new_path(output_program_path);
    require_new_path(output_shard_path);
    require_new_path(output_bundle_path);

    const auto source_program = dif::ir::read_file(source_program_path);
    const auto source_bundle =
        dif::weights::read_weight_bundle(source_bundle_path);
    auto rewrite = dif::compiler::rewrite_lowbit_weights(
        source_program, bit_width, group_size, correction);
    if (activation_aware) {
      std::vector<std::uint32_t> calibrated_ids;
      calibrated_ids.reserve(rewrite.entries.size());
      for (const auto &entry : rewrite.entries)
        calibrated_ids.push_back(entry.source_tensor_id);
      rewrite = dif::compiler::rewrite_lowbit_weights(
          source_program, bit_width, group_size, correction, calibrated_ids);
    }
    std::size_t direct_linear_count = 0U;
    if (direct_linear) {
      direct_linear_count = enable_direct_int5_linears(rewrite.program);
      if (direct_linear_count == 0U)
        dif::fail("direct Linear candidate found no eligible INT5 chain");
    }
    if (rewrite.entries.empty())
      dif::fail("program has no eligible rank-2 weight constants");
    if (activation_aware) {
      std::set<std::uint32_t> target_ids;
      for (const auto &entry : rewrite.entries)
        target_ids.insert(entry.source_tensor_id);
      for (const auto &[tensor_id, exponent] : tensor_exponents) {
        (void)exponent;
        if (!target_ids.contains(tensor_id))
          dif::fail("tensor alpha does not name a quantized weight: " +
                    std::to_string(tensor_id));
      }
    }
    const auto source_tensors = dif::weights::load_weight_bundle(
        source_bundle, source_program, !skip_source_digest);

    std::map<std::uint32_t, dif::compiler::Int4RewriteEntry> targets;
    for (const auto &entry : rewrite.entries)
      targets.emplace(entry.source_tensor_id, entry);
    std::set<std::string> names;
    std::set<std::uint32_t> found_targets;
    std::vector<Payload> payloads;
    std::vector<dif::weights::SafeTensorWriteSpec> specs;
    for (const auto &binding : source_bundle.bindings) {
      const auto target = targets.find(binding.tensor_id);
      if (target == targets.end()) {
        if (!names.insert(binding.tensor_name).second)
          dif::fail("output SafeTensors name collision");
        payloads.push_back({binding.tensor_id, binding.tensor_id,
                            binding.tensor_name, binding.dtype, binding.dims,
                            PayloadKind::Copy});
      } else {
        found_targets.insert(binding.tensor_id);
        const auto encoding = bit_width == 4U ? ".dif_int4" : ".dif_int5";
        const auto packed_name = binding.tensor_name + encoding + "_packed";
        const auto scales_name = binding.tensor_name + encoding + "_scales";
        const auto outlier_indices_name =
            binding.tensor_name + ".dif_int4_outlier_indices";
        const auto outlier_residuals_name =
            binding.tensor_name + ".dif_int4_outlier_residuals";
        if (!names.insert(packed_name).second ||
            !names.insert(scales_name).second ||
            (correction == dif::compiler::Int4Correction::OneOutlier &&
             (!names.insert(outlier_indices_name).second ||
              !names.insert(outlier_residuals_name).second)))
          dif::fail("output SafeTensors name collision");
        const auto *packed = rewrite.program.tensor(target->second.packed_tensor_id);
        const auto *scales = rewrite.program.tensor(target->second.scales_tensor_id);
        if (!packed || !scales)
          dif::fail("INT4 rewrite references missing payload tensors");
        payloads.push_back({packed->id, binding.tensor_id, packed_name,
                            packed->dtype, packed->dims,
                            PayloadKind::PackedInt4});
        payloads.push_back({scales->id, binding.tensor_id, scales_name,
                            scales->dtype, scales->dims,
                            PayloadKind::Scales});
        if (target->second.column_scales_tensor_id != 0U) {
          const auto *columns = rewrite.program.tensor(
              target->second.column_scales_tensor_id);
          if (!columns)
            dif::fail("low-bit rewrite lost activation column scales");
          const auto columns_name =
              binding.tensor_name + encoding + "_column_scales";
          if (!names.insert(columns_name).second)
            dif::fail("output SafeTensors name collision");
          payloads.push_back({columns->id, binding.tensor_id, columns_name,
                              columns->dtype, columns->dims,
                              PayloadKind::ColumnScales});
        }
        if (correction == dif::compiler::Int4Correction::OneOutlier) {
          const auto *indices = rewrite.program.tensor(
              target->second.outlier_indices_tensor_id);
          const auto *residuals = rewrite.program.tensor(
              target->second.outlier_residuals_tensor_id);
          if (!indices || !residuals)
            dif::fail("INT4 rewrite lost outlier correction tensors");
          payloads.push_back({indices->id, binding.tensor_id,
                              outlier_indices_name, indices->dtype,
                              indices->dims, PayloadKind::OutlierIndices});
          payloads.push_back({residuals->id, binding.tensor_id,
                              outlier_residuals_name, residuals->dtype,
                              residuals->dims, PayloadKind::OutlierResiduals});
        }
      }
    }
    if (found_targets.size() != targets.size())
      dif::fail("weight bundle does not bind every INT4 rewrite target");
    for (const auto &payload : payloads)
      specs.push_back({payload.name, payload.dtype, payload.dims});

    const auto temporary_program = partial_path(output_program_path);
    const auto temporary_shard = partial_path(output_shard_path);
    const auto temporary_bundle = partial_path(output_bundle_path);
    PartialFileCleanup partial_cleanup(
        {temporary_program, temporary_shard, temporary_bundle});
    dif::ir::write_file(rewrite.program, temporary_program);
    dif::weights::SafeTensorWriter writer(temporary_shard, std::move(specs));
    std::map<std::uint32_t, dif::compiler::Int4QuantizedTensor> quantized;
    double squared_error = 0.0;
    double squared_reference = 0.0;
    float maximum_absolute_error = 0.0F;
    std::vector<std::pair<std::uint32_t, std::string>> binding_names;
    binding_names.reserve(payloads.size());
    for (const auto &payload : payloads) {
      const auto source = source_tensors.find(payload.source_tensor_id);
      if (source == source_tensors.end())
        dif::fail("source bundle did not map tensor " +
                  std::to_string(payload.source_tensor_id));
      if (payload.kind == PayloadKind::Copy) {
        writer.append(payload.name,
                      {source->second.data(), source->second.byte_size()});
        source->second.discard_mapped_pages();
      } else {
        auto existing = quantized.find(payload.source_tensor_id);
        if (existing == quantized.end()) {
          dif::runtime::Tensor calibration;
          const dif::runtime::Tensor *calibration_pointer = nullptr;
          if (activation_aware) {
            auto tensor_id_text = std::to_string(payload.source_tensor_id);
            if (tensor_id_text.size() < 4U)
              tensor_id_text.insert(0U, 4U - tensor_id_text.size(), '0');
            auto calibration_path =
                calibration_directory /
                ("tensor_" + tensor_id_text +
                 "_input.diftensor");
            if (!std::filesystem::is_regular_file(calibration_path)) {
              if (source->second.dims[1] != 2688U)
                dif::fail("activation calibration is missing for tensor " +
                          std::to_string(payload.source_tensor_id));
              calibration_path = adaln_input;
            }
            calibration = dif::runtime::read_tensor(calibration_path);
            calibration_pointer = &calibration;
          }
          const auto tensor_exponent =
              tensor_exponents.contains(payload.source_tensor_id)
                  ? tensor_exponents.at(payload.source_tensor_id)
                  : activation_exponent;
          auto result = dif::compiler::quantize_lowbit_weight(
              source->second, bit_width, group_size, correction,
              calibration_pointer, tensor_exponent);
          const auto relative_l2 =
              result.squared_reference == 0.0
                  ? 0.0
                  : std::sqrt(result.squared_error /
                              result.squared_reference);
          std::cout << "QUANT tensor=" << payload.source_tensor_id
                    << " rows=" << source->second.dims[0]
                    << " columns=" << source->second.dims[1]
                    << " alpha=" << tensor_exponent
                    << " rel_l2=" << relative_l2
                    << " max_abs=" << result.maximum_absolute_error << "\n"
                    << std::flush;
          squared_error += result.squared_error;
          squared_reference += result.squared_reference;
          maximum_absolute_error =
              std::max(maximum_absolute_error,
                       result.maximum_absolute_error);
          source->second.discard_mapped_pages();
          existing = quantized
                         .emplace(payload.source_tensor_id, std::move(result))
                         .first;
        }
        const dif::runtime::Tensor *tensor = nullptr;
        if (payload.kind == PayloadKind::PackedInt4)
          tensor = &existing->second.packed;
        else if (payload.kind == PayloadKind::Scales)
          tensor = &existing->second.scales;
        else if (payload.kind == PayloadKind::ColumnScales)
          tensor = &existing->second.column_scales;
        else if (payload.kind == PayloadKind::OutlierIndices)
          tensor = &existing->second.outlier_indices;
        else
          tensor = &existing->second.outlier_residuals;
        writer.append(payload.name, {tensor->data(), tensor->byte_size()});
        if ((correction == dif::compiler::Int4Correction::None &&
             payload.kind == (activation_aware ? PayloadKind::ColumnScales
                                               : PayloadKind::Scales)) ||
            (correction == dif::compiler::Int4Correction::OneOutlier &&
             payload.kind == PayloadKind::OutlierResiduals))
          quantized.erase(existing);
      }
      binding_names.emplace_back(payload.tensor_id, payload.name);
    }
    const auto metadata = writer.finish();
    const auto shard_size = std::filesystem::file_size(temporary_shard);
    const auto shard_digest = dif::sha256_file(temporary_shard);

    dif::weights::WeightBundle output_bundle;
    output_bundle.program_fingerprint = dif::ir::fingerprint(rewrite.program);
    output_bundle.index_fingerprint =
        quantization_fingerprint(source_bundle, rewrite.program, bit_width,
                                 group_size, correction, activation_aware,
                                 activation_exponent, tensor_exponents,
                                 direct_linear);
    output_bundle.shards.push_back(
        {output_shard_path, shard_size, shard_digest});
    for (const auto &[tensor_id, name] : binding_names) {
      const auto *entry = metadata.find(name);
      if (!entry)
        dif::fail("new SafeTensors metadata lost a payload");
      output_bundle.bindings.push_back({tensor_id, 0U, name, entry->dtype,
                                        entry->dims, entry->file_offset,
                                        entry->byte_count});
    }
    dif::weights::write_weight_bundle(output_bundle, temporary_bundle);
    std::filesystem::rename(temporary_program, output_program_path);
    std::filesystem::rename(temporary_shard, output_shard_path);
    std::filesystem::rename(temporary_bundle, output_bundle_path);
    partial_cleanup.disarm();
    dif::weights::verify_weight_bundle(output_bundle, rewrite.program, false);

    const auto relative_l2 =
        squared_reference == 0.0
            ? 0.0
            : std::sqrt(squared_error / squared_reference);
    std::cout << "LOWBIT_PACKAGE PASS program=" << output_program_path
              << " shard=" << output_shard_path
              << " bundle=" << output_bundle_path
              << " tensors=" << rewrite.entries.size()
              << " bits=" << bit_width << " group=" << group_size
              << " correction="
              << (correction == dif::compiler::Int4Correction::OneOutlier
                      ? "one_outlier"
                      : "none")
              << " activation="
              << (activation_aware ? "rms_companding" : "none")
              << " alpha="
              << (activation_aware ? std::to_string(activation_exponent)
                                   : "none")
              << " tensor_alpha_overrides=" << tensor_exponents.size()
              << " direct_linear=" << direct_linear_count
              << " rel_l2=" << relative_l2
              << " max_abs=" << maximum_absolute_error
              << " shard_bytes=" << shard_size
              << " shard_sha256=" << dif::hex_digest(shard_digest)
              << " program_fingerprint="
              << dif::hex_digest(output_bundle.program_fingerprint)
              << " index_fingerprint="
              << dif::hex_digest(output_bundle.index_fingerprint) << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difquant: " << error.what() << "\n";
    return 1;
  }
}
