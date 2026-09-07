#include "dif/ir/codec.hpp"
#include "dif/ir/ir.hpp"
#include "dif/frontend/h3_vae.hpp"
#include "dif/frontend/h3_video_encoder.hpp"
#include "dif/frontend/krea2.hpp"
#include "dif/frontend/krea2_vae.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"
#include "dif/weights/bundle.hpp"
#include "dif/weights/safetensors.hpp"
#include "dif/telemetry/schema.hpp"

#include <algorithm>
#include <set>
#include <array>
#include <filesystem>
#include <iostream>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using ShardIndexMap = std::map<std::string, std::uint32_t>;
using ShardMetadataMap =
    std::unordered_map<std::string, dif::weights::SafeTensorFile>;

dif::weights::WeightBundle materialize_mixed_bf16_bundle(
    const std::filesystem::path &source_path,
    const dif::ir::Program &program,
    const std::vector<std::uint32_t> &tensor_ids,
    const std::vector<std::string> &tensor_names,
    const std::filesystem::path &derived_path) {
  if (std::filesystem::exists(derived_path))
    dif::fail("refusing to overwrite derived BF16 shard");
  if (tensor_ids.size() != tensor_names.size() || tensor_ids.empty())
    dif::fail("prepared BF16 bundle requires matched tensor ids and names");
  const auto source = dif::weights::read_safetensors(source_path);
  std::vector<dif::weights::SafeTensorWriteSpec> specs;
  for (std::size_t index = 0U; index < tensor_ids.size(); ++index) {
    const auto *description = program.tensor(tensor_ids[index]);
    const auto *entry = source.find(tensor_names[index]);
    if (!description || !entry)
      dif::fail("checkpoint binding is missing " + tensor_names[index]);
    if (entry->dims != description->dims)
      dif::fail("checkpoint shape disagrees with DiffIR: " +
                tensor_names[index]);
    if (entry->dtype == description->dtype)
      continue;
    if (entry->dtype != dif::ir::DType::F32 ||
        description->dtype != dif::ir::DType::BF16)
      dif::fail("prepared bundle only supports F32 to BF16: " +
                tensor_names[index]);
    specs.push_back(
        {tensor_names[index], description->dtype, description->dims});
  }
  if (specs.empty())
    dif::fail("Krea 2 checkpoint has no F32 tensors to materialize");

  dif::weights::SafeTensorWriter writer(derived_path, std::move(specs));
  std::uint64_t converted_tensors = 0U;
  std::uint64_t converted_bytes = 0U;
  for (std::size_t index = 0U; index < tensor_ids.size(); ++index) {
    const auto *description = program.tensor(tensor_ids[index]);
    const auto *entry = source.find(tensor_names[index]);
    if (entry->dtype == description->dtype)
      continue;
    auto mapped =
        dif::weights::map_safetensor(source, tensor_names[index]);
    auto converted =
        dif::runtime::convert_float_tensor(mapped, dif::ir::DType::BF16);
    writer.append(tensor_names[index],
                  {converted.data(), converted.byte_size()});
    converted_bytes += converted.byte_size();
    ++converted_tensors;
    mapped.discard_mapped_pages();
  }
  const auto derived = writer.finish();

  dif::weights::WeightBundle bundle;
  bundle.program_fingerprint = dif::ir::fingerprint(program);
  const auto source_digest = dif::sha256_file(source_path);
  bundle.index_fingerprint = source_digest;
  bundle.shards.push_back(
      {std::filesystem::absolute(source_path).lexically_normal(),
       source.file_size, source_digest});
  bundle.shards.push_back(
      {std::filesystem::absolute(derived_path).lexically_normal(),
       derived.file_size, dif::sha256_file(derived_path)});
  for (std::size_t index = 0U; index < tensor_ids.size(); ++index) {
    const auto *description = program.tensor(tensor_ids[index]);
    const auto *source_entry = source.find(tensor_names[index]);
    const auto derived_binding = source_entry->dtype != description->dtype;
    const auto *entry = derived_binding
                            ? derived.find(tensor_names[index])
                            : source_entry;
    if (!entry || entry->dtype != description->dtype ||
        entry->dims != description->dims)
      dif::fail("prepared bundle lost " + tensor_names[index]);
    bundle.bindings.push_back(
        {tensor_ids[index], derived_binding ? 1U : 0U,
         tensor_names[index], entry->dtype, entry->dims,
         entry->file_offset, entry->byte_count});
  }
  dif::weights::verify_weight_bundle(bundle, program, false);
  std::cout << "BF16_MATERIALIZED tensors=" << converted_tensors
            << " bytes=" << converted_bytes << "\n";
  return bundle;
}

dif::runtime::Tensor krea2_vae_weight(
    const dif::weights::SafeTensorFile &source,
    const dif::frontend::Krea2VaeWeightBinding &binding,
    const dif::ir::TensorDesc &destination) {
  auto tensor = dif::weights::map_safetensor(source, binding.source_name);
  if (tensor.dtype == dif::ir::DType::F32 &&
      destination.dtype == dif::ir::DType::BF16)
    tensor = dif::runtime::convert_float_tensor(tensor, destination.dtype);
  switch (binding.transform) {
  case dif::frontend::Krea2VaeWeightTransform::Direct:
    break;
  case dif::frontend::Krea2VaeWeightTransform::FlattenSingletonDimensions:
    if (tensor.element_count() != destination.element_count())
      dif::fail("Qwen Image VAE flattened weight size mismatch: " +
                binding.source_name);
    tensor.dims = destination.dims;
    break;
  case dif::frontend::Krea2VaeWeightTransform::Conv3dLastTemporalSlice: {
    if (tensor.dtype != destination.dtype || tensor.dims.size() != 5U ||
        destination.dims.size() != 4U ||
        tensor.dims[0] != destination.dims[0] ||
        tensor.dims[1] != destination.dims[1] ||
        tensor.dims[3] != destination.dims[2] ||
        tensor.dims[4] != destination.dims[3])
      dif::fail("Qwen Image VAE Conv3d slice mismatch: " +
                binding.source_name);
    dif::runtime::Tensor sliced{destination.dtype, destination.dims, {}};
    sliced.bytes.resize(static_cast<std::size_t>(destination.byte_count()));
    const auto outer = tensor.dims[0] * tensor.dims[1];
    const auto temporal = tensor.dims[2];
    const auto plane = tensor.dims[3] * tensor.dims[4];
    const auto plane_bytes = static_cast<std::size_t>(
        plane * dif::ir::dtype_size(tensor.dtype));
    for (std::uint64_t index = 0U; index < outer; ++index) {
      const auto source_offset =
          (index * temporal + temporal - 1U) * plane_bytes;
      std::memcpy(sliced.mutable_data() + index * plane_bytes,
                  tensor.data() + source_offset, plane_bytes);
    }
    sliced.validate();
    tensor = std::move(sliced);
    break;
  }
  }
  if (tensor.dtype != destination.dtype || tensor.dims != destination.dims)
    dif::fail("Qwen Image VAE prepared weight mismatch: " +
              binding.source_name);
  tensor.validate();
  return tensor;
}

dif::weights::WeightBundle materialize_krea2_vae_bundle(
    const std::filesystem::path &source_path,
    const dif::frontend::Krea2VaeBuild &build,
    const std::filesystem::path &derived_path) {
  if (std::filesystem::exists(derived_path))
    dif::fail("refusing to overwrite Qwen Image VAE prepared shard");
  const auto source = dif::weights::read_safetensors(source_path);
  std::vector<dif::weights::SafeTensorWriteSpec> specs;
  specs.reserve(build.weights.size());
  for (const auto &binding : build.weights) {
    const auto *description = build.program.tensor(binding.tensor);
    if (!description)
      dif::fail("Qwen Image VAE binding lost its DiffIR tensor");
    specs.push_back(
        {binding.source_name, description->dtype, description->dims});
  }
  dif::weights::SafeTensorWriter writer(derived_path, std::move(specs));
  for (const auto &binding : build.weights) {
    const auto *description = build.program.tensor(binding.tensor);
    auto tensor = krea2_vae_weight(source, binding, *description);
    writer.append(binding.source_name, {tensor.data(), tensor.byte_size()});
  }
  const auto derived = writer.finish();
  dif::weights::WeightBundle bundle;
  bundle.program_fingerprint = dif::ir::fingerprint(build.program);
  bundle.index_fingerprint = dif::sha256_file(source_path);
  const auto absolute =
      std::filesystem::absolute(derived_path).lexically_normal();
  bundle.shards.push_back(
      {absolute, derived.file_size, dif::sha256_file(absolute)});
  for (const auto &binding : build.weights) {
    const auto *description = build.program.tensor(binding.tensor);
    const auto *entry = derived.find(binding.source_name);
    if (!description || !entry || entry->dtype != description->dtype ||
        entry->dims != description->dims)
      dif::fail("Qwen Image VAE prepared shard lost " +
                binding.source_name);
    bundle.bindings.push_back(
        {binding.tensor, 0U, binding.source_name, entry->dtype, entry->dims,
         entry->file_offset, entry->byte_count});
  }
  dif::weights::verify_weight_bundle(bundle, build.program, false);
  return bundle;
}

void append_checkpoint_binding(
    const dif::weights::SafeTensorIndex &index,
    const dif::ir::Program &program, dif::weights::WeightBundle &bundle,
    ShardIndexMap &shard_indices, ShardMetadataMap &metadata,
    std::uint32_t tensor_id, const std::string &name) {
  const auto location = index.weight_map.find(name);
  if (location == index.weight_map.end())
    dif::fail("H3 index is missing " + name);
  const auto absolute =
      std::filesystem::absolute(location->second).lexically_normal();
  const auto shard_key = absolute.string();
  auto shard = shard_indices.find(shard_key);
  if (shard == shard_indices.end()) {
    const auto shard_id = static_cast<std::uint32_t>(bundle.shards.size());
    shard = shard_indices.emplace(shard_key, shard_id).first;
    bundle.shards.push_back(
        {absolute, std::filesystem::file_size(absolute), {}});
    metadata.emplace(shard_key, dif::weights::read_safetensors(absolute));
  }
  const auto &file = metadata.at(shard_key);
  const auto *entry = file.find(name);
  if (!entry)
    dif::fail("H3 shard is missing indexed tensor " + name);
  const auto *description = program.tensor(tensor_id);
  if (!description ||
      !description->has_role(dif::ir::TensorRole::Constant) ||
      description->dtype != entry->dtype || description->dims != entry->dims ||
      description->byte_count() != entry->byte_count)
    dif::fail("H3 checkpoint tensor disagrees with DiffIR id " +
              std::to_string(tensor_id) + ": " + name);
  bundle.bindings.push_back(
      {tensor_id, shard->second, name, entry->dtype, entry->dims,
       entry->file_offset, entry->byte_count});
}

void seal_or_hash_shards(dif::weights::WeightBundle &bundle,
                         const dif::weights::WeightBundle *sealed_shards,
                         bool hash_shards = true) {
  for (std::size_t shard = 0; shard < bundle.shards.size(); ++shard) {
    if (sealed_shards) {
      const auto found = std::find_if(
          sealed_shards->shards.begin(), sealed_shards->shards.end(),
          [&](const auto &sealed) {
            return sealed.path == bundle.shards[shard].path &&
                   sealed.file_size == bundle.shards[shard].file_size;
          });
      if (found == sealed_shards->shards.end())
        dif::fail("sealed bundle lacks matching shard " +
                  bundle.shards[shard].path.string());
      bundle.shards[shard].digest = found->digest;
      continue;
    }
    if (hash_shards) {
      std::cout << "HASH shard=" << (shard + 1U) << "/"
                << bundle.shards.size() << " path=" << bundle.shards[shard].path
                << "\n";
      bundle.shards[shard].digest =
          dif::sha256_file(bundle.shards[shard].path);
    }
  }
}

dif::weights::WeightBundle materialize_h3_video_vae_bundle(
    const std::filesystem::path &source_path,
    const dif::frontend::H3VideoVaeBuild &build,
    const std::filesystem::path &output_shard) {
  if (std::filesystem::exists(output_shard))
    dif::fail("refusing to overwrite H3 video VAE derived shard");
  const auto source = dif::weights::read_safetensors(source_path);
  std::vector<dif::weights::SafeTensorWriteSpec> specs;
  specs.reserve(build.bindings.size());
  for (const auto &binding : build.bindings) {
    const auto *description = build.program.tensor(binding.tensor_id);
    if (!description ||
        !description->has_role(dif::ir::TensorRole::Constant))
      dif::fail("H3 video VAE binding does not target a constant");
    specs.push_back({binding.name, description->dtype, description->dims});
  }
  dif::weights::SafeTensorWriter writer(output_shard, std::move(specs));
  for (std::size_t index = 0; index < build.bindings.size(); ++index) {
    const auto &binding = build.bindings[index];
    const auto *description = build.program.tensor(binding.tensor_id);
    if (binding.source_name.empty()) {
      const auto found = build.generated_constants.find(binding.tensor_id);
      if (found == build.generated_constants.end())
        dif::fail("H3 video VAE generated binding has no payload");
      writer.append(binding.name,
                    {found->second.data(), found->second.byte_size()});
    } else {
      const auto *entry = source.find(binding.source_name);
      if (!entry)
        dif::fail("H3 video VAE source shard is missing " +
                  binding.source_name);
      if (entry->dtype != dif::ir::DType::F32)
        dif::fail("H3 video VAE source tensor is not F32: " +
                  binding.source_name);
      auto tensor = dif::weights::map_safetensor(source, binding.source_name);
      if (tensor.element_count() != description->element_count())
        dif::fail("H3 video VAE source tensor element count disagrees: " +
                  binding.source_name);
      if (description->dtype == dif::ir::DType::F32) {
        writer.append(binding.name, {tensor.data(), tensor.byte_size()});
      } else if (description->dtype == dif::ir::DType::F16) {
        std::vector<std::uint8_t> converted(
            static_cast<std::size_t>(description->byte_count()));
        for (std::uint64_t element = 0; element < tensor.element_count();
             ++element) {
          float value = 0.0F;
          std::memcpy(&value, tensor.data() + element * sizeof(value),
                      sizeof(value));
          const auto half = dif::runtime::float_to_f16(value);
          std::memcpy(converted.data() + element * sizeof(half), &half,
                      sizeof(half));
        }
        writer.append(binding.name, converted);
      } else {
        dif::fail("H3 video VAE derived checkpoint target must be F32 or F16");
      }
      tensor.discard_mapped_pages();
    }
    std::cout << "MATERIALIZE tensor=" << (index + 1U) << "/"
              << build.bindings.size() << " name=" << binding.name
              << " dtype=" << dif::ir::dtype_name(description->dtype)
              << " bytes=" << description->byte_count() << "\n";
  }
  const auto derived = writer.finish();
  dif::weights::WeightBundle bundle;
  bundle.program_fingerprint = dif::ir::fingerprint(build.program);
  std::cout << "HASH source=" << source_path << "\n";
  bundle.index_fingerprint = dif::sha256_file(source_path);
  const auto absolute =
      std::filesystem::absolute(output_shard).lexically_normal();
  std::cout << "HASH derived=" << absolute << "\n";
  bundle.shards.push_back(
      {absolute, derived.file_size, dif::sha256_file(absolute)});
  for (const auto &binding : build.bindings) {
    const auto *entry = derived.find(binding.name);
    if (!entry)
      dif::fail("H3 video VAE derived shard lost " + binding.name);
    bundle.bindings.push_back(
        {binding.tensor_id, 0U, binding.name, entry->dtype, entry->dims,
         entry->file_offset, entry->byte_count});
  }
  dif::weights::verify_weight_bundle(bundle, build.program, false);
  return bundle;
}

dif::weights::WeightBundle make_h3_video_encoder_bundle(
    const std::filesystem::path &source_path,
    const dif::frontend::H3VideoEncoderBuild &build) {
  const auto source = dif::weights::read_safetensors(source_path);
  dif::weights::WeightBundle bundle;
  bundle.program_fingerprint = dif::ir::fingerprint(build.program);
  const auto digest = dif::sha256_file(source_path);
  bundle.index_fingerprint = digest;
  bundle.shards.push_back(
      {std::filesystem::absolute(source_path).lexically_normal(),
       source.file_size, digest});
  for (const auto &binding : build.weights) {
    const auto *description = build.program.tensor(binding.tensor);
    const auto *entry = source.find(binding.source_name);
    if (!description || !description->has_role(dif::ir::TensorRole::Constant) ||
        !entry || entry->dtype != description->dtype ||
        entry->dims != description->dims ||
        entry->byte_count != description->byte_count())
      dif::fail("H3 video encoder checkpoint disagrees with DiffIR: " +
                binding.source_name);
    bundle.bindings.push_back(
        {binding.tensor, 0U, binding.source_name, entry->dtype, entry->dims,
         entry->file_offset, entry->byte_count});
  }
  dif::weights::verify_weight_bundle(bundle, build.program, false);
  return bundle;
}

dif::weights::WeightBundle seal_h3_video_vae_derived_bundle(
    const std::filesystem::path &derived_path,
    const std::filesystem::path &source_path,
    const std::filesystem::path &geometry_path,
    const dif::frontend::H3VideoVaeBuild &build) {
  if (std::filesystem::exists(geometry_path))
    dif::fail("refusing to overwrite H3 video VAE geometry shard");
  const auto derived = dif::weights::read_safetensors(derived_path);
  const auto source = dif::weights::read_safetensors(source_path);
  auto derived_matches = [&](const dif::frontend::H3VideoVaeBinding &binding) {
    const auto *description = build.program.tensor(binding.tensor_id);
    const auto *entry = derived.find(binding.name);
    return description && entry && entry->dtype == description->dtype &&
           entry->dims == description->dims &&
           entry->byte_count == description->byte_count();
  };
  std::vector<dif::weights::SafeTensorWriteSpec> geometry_specs;
  for (const auto &binding : build.bindings) {
    const auto *description = build.program.tensor(binding.tensor_id);
    if (!description)
      dif::fail("H3 video VAE binding lost its descriptor");
    if (binding.source_name.empty() || !derived_matches(binding))
      geometry_specs.push_back(
          {binding.name, description->dtype, description->dims});
  }
  dif::weights::SafeTensorWriter geometry_writer(geometry_path,
                                                  std::move(geometry_specs));
  for (const auto &binding : build.bindings) {
    if (!binding.source_name.empty() && derived_matches(binding))
      continue;
    const auto *description = build.program.tensor(binding.tensor_id);
    if (!description)
      dif::fail("H3 video VAE binding lost its descriptor");
    if (binding.source_name.empty()) {
      const auto found = build.generated_constants.find(binding.tensor_id);
      if (found == build.generated_constants.end())
        dif::fail("H3 video VAE generated binding has no payload");
      geometry_writer.append(
          binding.name, {found->second.data(), found->second.byte_size()});
      continue;
    }
    const auto *source_entry = source.find(binding.source_name);
    if (!source_entry || source_entry->dtype != dif::ir::DType::F32)
      dif::fail("H3 video VAE F32 source is missing " + binding.source_name);
    auto tensor = dif::weights::map_safetensor(source, binding.source_name);
    if (tensor.element_count() != description->element_count())
      dif::fail("H3 video VAE source tensor element count disagrees: " +
                binding.source_name);
    if (description->dtype == dif::ir::DType::F32) {
      geometry_writer.append(binding.name, {tensor.data(), tensor.byte_size()});
    } else if (description->dtype == dif::ir::DType::F16) {
      std::vector<std::uint8_t> converted(
          static_cast<std::size_t>(description->byte_count()));
      for (std::uint64_t element = 0; element < tensor.element_count();
           ++element) {
        float value = 0.0F;
        std::memcpy(&value, tensor.data() + element * sizeof(value),
                    sizeof(value));
        const auto half = dif::runtime::float_to_f16(value);
        std::memcpy(converted.data() + element * sizeof(half), &half,
                    sizeof(half));
      }
      geometry_writer.append(binding.name, converted);
    } else {
      dif::fail("H3 video VAE derived checkpoint target must be F32 or F16");
    }
    tensor.discard_mapped_pages();
  }
  const auto geometry = geometry_writer.finish();

  dif::weights::WeightBundle bundle;
  bundle.program_fingerprint = dif::ir::fingerprint(build.program);
  std::cout << "HASH source=" << source_path << "\n";
  bundle.index_fingerprint = dif::sha256_file(source_path);
  const auto absolute =
      std::filesystem::absolute(derived_path).lexically_normal();
  std::cout << "HASH derived=" << absolute << "\n";
  bundle.shards.push_back(
      {absolute, derived.file_size, dif::sha256_file(absolute)});
  const auto geometry_absolute =
      std::filesystem::absolute(geometry_path).lexically_normal();
  bundle.shards.push_back({geometry_absolute, geometry.file_size,
                           dif::sha256_file(geometry_absolute)});
  for (const auto &binding : build.bindings) {
    const auto *description = build.program.tensor(binding.tensor_id);
    const auto use_geometry =
        binding.source_name.empty() || !derived_matches(binding);
    const auto &shard = use_geometry ? geometry : derived;
    const auto *entry = shard.find(binding.name);
    if (!description || !entry || entry->dtype != description->dtype ||
        entry->dims != description->dims ||
        entry->byte_count != description->byte_count())
      dif::fail("H3 video VAE derived tensor disagrees with DiffIR: " +
                binding.name);
    bundle.bindings.push_back(
        {binding.tensor_id, use_geometry ? 1U : 0U, binding.name,
         entry->dtype, entry->dims, entry->file_offset, entry->byte_count});
  }
  dif::weights::verify_weight_bundle(bundle, build.program, false);
  return bundle;
}

dif::weights::WeightBundle reuse_h3_video_vae_bundle(
    const dif::weights::WeightBundle &sealed,
    const dif::frontend::H3VideoVaeBuild &build,
    const std::filesystem::path &geometry_shard) {
  if (std::filesystem::exists(geometry_shard))
    dif::fail("refusing to overwrite H3 video VAE geometry shard");
  std::vector<dif::weights::SafeTensorWriteSpec> specs;
  for (const auto &binding : build.bindings) {
    if (!binding.source_name.empty())
      continue;
    const auto *description = build.program.tensor(binding.tensor_id);
    if (!description)
      dif::fail("H3 video VAE generated binding lost its descriptor");
    specs.push_back({binding.name, description->dtype, description->dims});
  }
  dif::weights::SafeTensorWriter writer(geometry_shard, std::move(specs));
  for (const auto &binding : build.bindings) {
    if (!binding.source_name.empty())
      continue;
    const auto found = build.generated_constants.find(binding.tensor_id);
    if (found == build.generated_constants.end())
      dif::fail("H3 video VAE generated binding has no payload");
    writer.append(binding.name,
                  {found->second.data(), found->second.byte_size()});
  }
  const auto geometry = writer.finish();

  dif::weights::WeightBundle bundle;
  bundle.program_fingerprint = dif::ir::fingerprint(build.program);
  bundle.index_fingerprint = sealed.index_fingerprint;
  bundle.shards = sealed.shards;
  const auto absolute =
      std::filesystem::absolute(geometry_shard).lexically_normal();
  bundle.shards.push_back(
      {absolute, geometry.file_size, dif::sha256_file(absolute)});
  const auto geometry_index =
      static_cast<std::uint32_t>(bundle.shards.size() - 1U);

  for (const auto &binding : build.bindings) {
    const auto *description = build.program.tensor(binding.tensor_id);
    if (!description)
      dif::fail("H3 video VAE reuse binding lost its descriptor");
    if (binding.source_name.empty()) {
      const auto *entry = geometry.find(binding.name);
      if (!entry)
        dif::fail("H3 video VAE geometry shard lost " + binding.name);
      bundle.bindings.push_back(
          {binding.tensor_id, geometry_index, binding.name, entry->dtype,
           entry->dims, entry->file_offset, entry->byte_count});
      continue;
    }
    const auto reused = std::find_if(
        sealed.bindings.begin(), sealed.bindings.end(), [&](const auto &value) {
          return value.tensor_name == binding.name &&
                 value.dtype == description->dtype &&
                 value.dims == description->dims &&
                 value.byte_count == description->byte_count();
        });
    if (reused == sealed.bindings.end())
      dif::fail("sealed H3 video VAE bundle lacks reusable weight " +
                binding.name);
    bundle.bindings.push_back(
        {binding.tensor_id, reused->shard_index, reused->tensor_name,
         reused->dtype, reused->dims, reused->file_offset,
         reused->byte_count});
  }
  dif::weights::verify_weight_bundle(bundle, build.program, false);
  return bundle;
}

dif::weights::WeightBundle
make_h3_bundle(const std::filesystem::path &index_path,
               const dif::ir::Program &program,
               const dif::weights::WeightBundle *sealed_shards = nullptr) {
  constexpr std::array<const char *, 10> kSuffixes = {
      "adaln_proj.linear.weight", "adaln_proj.linear.bias",
      "attn.qkv_proj.weight",     "attn.q_norm.weight",
      "attn.k_norm.weight",       "attn.out_proj.weight",
      "mlp.fc1.weight",           "mlp.fc2.weight",
      "norm1.weight",             "norm2.weight",
  };
  std::uint32_t layer_tensor_stride = 0U;
  std::uint32_t layer_operation_count = 0U;
  for (const auto [tensor_stride, operation_count] :
       {std::pair{34U, 17U}, std::pair{32U, 15U}}) {
    if (!program.operations.empty() &&
        program.operations.size() % operation_count == 0U) {
      const auto candidate_layers =
          program.operations.size() / operation_count;
      if (program.tensors.size() == 5U + candidate_layers * tensor_stride) {
        layer_tensor_stride = tensor_stride;
        layer_operation_count = operation_count;
        break;
      }
    }
  }
  if (layer_tensor_stride == 0U)
    dif::fail("H3 transformer tensor stride does not match the bundle ABI");
  const auto layers = program.operations.size() / layer_operation_count;

  const auto index = dif::weights::read_safetensors_index(index_path);
  dif::weights::WeightBundle bundle;
  bundle.program_fingerprint = dif::ir::fingerprint(program);
  bundle.index_fingerprint = dif::sha256_file(index_path);
  std::map<std::string, std::uint32_t> shard_indices;
  std::unordered_map<std::string, dif::weights::SafeTensorFile> metadata;

  for (std::size_t layer = 0; layer < layers; ++layer) {
    for (std::size_t weight = 0; weight < kSuffixes.size(); ++weight) {
      const auto name = "blocks." + std::to_string(layer) + "." +
                        kSuffixes[weight];
      const auto tensor_id = static_cast<std::uint32_t>(
          6U + layer * layer_tensor_stride + weight);
      append_checkpoint_binding(index, program, bundle, shard_indices,
                                metadata, tensor_id, name);
    }
  }
  seal_or_hash_shards(bundle, sealed_shards);
  return bundle;
}

dif::weights::WeightBundle make_h3_token_refiner_bundle(
    const std::filesystem::path &index_path,
    const dif::ir::Program &program,
    const dif::weights::WeightBundle *sealed_shards = nullptr) {
  using dif::ir::Opcode;
  constexpr std::array<const char *, 8> kSuffixes = {
      "attn.qkv_proj.weight", "attn.q_norm.weight",
      "attn.k_norm.weight",   "attn.out_proj.weight",
      "mlp.fc1.weight",       "mlp.fc2.weight",
      "norm1.weight",         "norm2.weight",
  };
  constexpr std::array<Opcode, 15> kLayerOpcodes = {
      Opcode::RmsNorm,
      Opcode::H3DeinterleaveQkvWeight,
      Opcode::Linear,
      Opcode::Linear,
      Opcode::Linear,
      Opcode::RmsNorm,
      Opcode::RmsNorm,
      Opcode::Attention,
      Opcode::Linear,
      Opcode::Add,
      Opcode::RmsNorm,
      Opcode::Linear,
      Opcode::SwiGlu,
      Opcode::Linear,
      Opcode::Add,
  };
  if (program.operations.size() <= 1U ||
      (program.operations.size() - 1U) % kLayerOpcodes.size() != 0U)
    dif::fail("H3 token-refiner operation count does not match the bundle ABI");
  const auto layers =
      (program.operations.size() - 1U) / kLayerOpcodes.size();
  if (program.tensors.size() != 3U + layers * 25U ||
      program.operations.back().opcode != Opcode::RmsNorm)
    dif::fail("H3 token-refiner tensor layout does not match the bundle ABI");
  for (std::size_t layer = 0; layer < layers; ++layer) {
    for (std::size_t operation = 0; operation < kLayerOpcodes.size();
         ++operation) {
      if (program.operations[layer * kLayerOpcodes.size() + operation].opcode !=
          kLayerOpcodes[operation])
        dif::fail("H3 token-refiner operation layout does not match the "
                  "bundle ABI");
    }
  }

  const auto index = dif::weights::read_safetensors_index(index_path);
  dif::weights::WeightBundle bundle;
  bundle.program_fingerprint = dif::ir::fingerprint(program);
  bundle.index_fingerprint = dif::sha256_file(index_path);
  ShardIndexMap shard_indices;
  ShardMetadataMap metadata;
  for (std::size_t layer = 0; layer < layers; ++layer) {
    for (std::size_t weight = 0; weight < kSuffixes.size(); ++weight) {
      const auto name = "token_refiner.blocks." + std::to_string(layer) +
                        "." + kSuffixes[weight];
      const auto tensor_id =
          static_cast<std::uint32_t>(2U + layer * 25U + weight);
      append_checkpoint_binding(index, program, bundle, shard_indices,
                                metadata, tensor_id, name);
    }
  }
  append_checkpoint_binding(
      index, program, bundle, shard_indices, metadata,
      static_cast<std::uint32_t>(2U + layers * 25U),
      "token_refiner.final_norm.weight");
  seal_or_hash_shards(bundle, sealed_shards);
  return bundle;
}

dif::weights::WeightBundle make_h3_denoiser_bundle(
    const std::filesystem::path &index_path,
    const dif::ir::Program &program,
    const dif::weights::WeightBundle *sealed_shards = nullptr,
    bool hash_shards = true) {
  using dif::ir::Opcode;
  constexpr std::array<Opcode, 5> kPrefix = {
      Opcode::Linear, Opcode::Cast, Opcode::Linear, Opcode::Cast,
      Opcode::Linear,
  };
  constexpr std::array<Opcode, 15> kRefiner = {
      Opcode::RmsNorm,
      Opcode::H3DeinterleaveQkvWeight,
      Opcode::Linear,
      Opcode::Linear,
      Opcode::Linear,
      Opcode::RmsNorm,
      Opcode::RmsNorm,
      Opcode::Attention,
      Opcode::Linear,
      Opcode::Add,
      Opcode::RmsNorm,
      Opcode::Linear,
      Opcode::SwiGlu,
      Opcode::Linear,
      Opcode::Add,
  };
  constexpr std::array<Opcode, 17> kBlock = {
      Opcode::Linear,
      Opcode::H3AdaLNSelect,
      Opcode::RmsNormModulate,
      Opcode::H3DeinterleaveQkvWeight,
      Opcode::Linear,
      Opcode::Linear,
      Opcode::Linear,
      Opcode::QkNormPartialRope,
      Opcode::QkNormPartialRope,
      Opcode::Attention,
      Opcode::Linear,
      Opcode::ResidualGate,
      Opcode::RmsNormModulate,
      Opcode::Linear,
      Opcode::SwiGlu,
      Opcode::Linear,
      Opcode::ResidualGate,
  };
  constexpr std::array<Opcode, 8> kTail = {
      Opcode::Linear, Opcode::SelectRowChunks, Opcode::RmsNormModulate,
      Opcode::Cast,   Opcode::Linear,          Opcode::GatherRows,
      Opcode::Linear, Opcode::GatherRows,
  };
  const auto matches = [&](std::size_t start, const auto &pattern) {
    if (start + pattern.size() > program.operations.size())
      return false;
    for (std::size_t index = 0; index < pattern.size(); ++index) {
      if (program.operations[start + index].opcode != pattern[index])
        return false;
    }
    return true;
  };
  if (!matches(0U, kPrefix))
    dif::fail("H3 denoiser prefix does not match the bundle ABI");
  std::size_t cursor = kPrefix.size();
  std::size_t refiner_layers = 0U;
  while (matches(cursor, kRefiner)) {
    ++refiner_layers;
    cursor += kRefiner.size();
  }
  if (refiner_layers == 0U || cursor >= program.operations.size() ||
      program.operations[cursor++].opcode != Opcode::RmsNorm)
    dif::fail("H3 denoiser refiner does not match the bundle ABI");
  constexpr std::array<Opcode, 4> kPacking = {
      Opcode::Fill, Opcode::IndexedUpdateRows, Opcode::IndexedUpdateRows,
      Opcode::IndexedUpdateRows,
  };
  constexpr std::array<Opcode, 7> kTime = {
      Opcode::SinusoidalTimestep, Opcode::Linear, Opcode::SiLU,
      Opcode::Linear, Opcode::SiLU, Opcode::Cast,
      Opcode::RotaryPosition,
  };
  if (!matches(cursor, kPacking))
    dif::fail("H3 denoiser packing does not match the bundle ABI");
  cursor += kPacking.size();
  if (!matches(cursor, kTime))
    dif::fail("H3 denoiser timestep MLP does not match the bundle ABI");
  cursor += kTime.size();
  if (program.operations.size() < cursor + kTail.size() ||
      (program.operations.size() - cursor - kTail.size()) % kBlock.size() !=
          0U)
    dif::fail("H3 denoiser block count does not match the bundle ABI");
  const auto layers =
      (program.operations.size() - cursor - kTail.size()) / kBlock.size();
  if (layers == 0U)
    dif::fail("H3 denoiser must contain at least one transformer block");
  for (std::size_t layer = 0; layer < layers; ++layer) {
    if (!matches(cursor, kBlock))
      dif::fail("H3 denoiser block layout does not match the bundle ABI");
    cursor += kBlock.size();
  }
  if (!matches(cursor, kTail) ||
      program.tensors.size() !=
          58U + refiner_layers * 25U + layers * 34U)
    dif::fail("H3 denoiser tail/tensor layout does not match the bundle ABI");

  const auto index = dif::weights::read_safetensors_index(index_path);
  dif::weights::WeightBundle bundle;
  bundle.program_fingerprint = dif::ir::fingerprint(program);
  bundle.index_fingerprint = dif::sha256_file(index_path);
  ShardIndexMap shard_indices;
  ShardMetadataMap metadata;
  append_checkpoint_binding(index, program, bundle, shard_indices, metadata,
                            13U, "rope.inv_freq");
  constexpr std::array<const char *, 10> kInitialNames = {
      "video_patch_proj.weight",
      "video_patch_proj.bias",
      "audio_patch_proj.weight",
      "audio_patch_proj.bias",
      "condition_proj.weight",
      "condition_proj.bias",
      "time_embedder.proj_in.weight",
      "time_embedder.proj_in.bias",
      "time_embedder.proj_out.weight",
      "time_embedder.proj_out.bias",
  };
  for (std::size_t weight = 0; weight < kInitialNames.size(); ++weight)
    append_checkpoint_binding(index, program, bundle, shard_indices, metadata,
                              static_cast<std::uint32_t>(14U + weight),
                              kInitialNames[weight]);

  constexpr std::array<const char *, 8> kRefinerSuffixes = {
      "attn.qkv_proj.weight", "attn.q_norm.weight",
      "attn.k_norm.weight",   "attn.out_proj.weight",
      "mlp.fc1.weight",       "mlp.fc2.weight",
      "norm1.weight",         "norm2.weight",
  };
  for (std::size_t layer = 0; layer < refiner_layers; ++layer) {
    for (std::size_t weight = 0; weight < kRefinerSuffixes.size(); ++weight) {
      const auto name = "token_refiner.blocks." + std::to_string(layer) +
                        "." + kRefinerSuffixes[weight];
      append_checkpoint_binding(
          index, program, bundle, shard_indices, metadata,
          static_cast<std::uint32_t>(29U + layer * 25U + weight), name);
    }
  }
  append_checkpoint_binding(
      index, program, bundle, shard_indices, metadata,
      static_cast<std::uint32_t>(29U + refiner_layers * 25U),
      "token_refiner.final_norm.weight");

  constexpr std::array<const char *, 10> kBlockSuffixes = {
      "adaln_proj.linear.weight", "adaln_proj.linear.bias",
      "attn.qkv_proj.weight",     "attn.q_norm.weight",
      "attn.k_norm.weight",       "attn.out_proj.weight",
      "mlp.fc1.weight",           "mlp.fc2.weight",
      "norm1.weight",             "norm2.weight",
  };
  const auto block_base = 43U + refiner_layers * 25U;
  for (std::size_t layer = 0; layer < layers; ++layer) {
    for (std::size_t weight = 0; weight < kBlockSuffixes.size(); ++weight) {
      const auto name = "blocks." + std::to_string(layer) + "." +
                        kBlockSuffixes[weight];
      append_checkpoint_binding(
          index, program, bundle, shard_indices, metadata,
          static_cast<std::uint32_t>(block_base + layer * 34U + weight), name);
    }
  }
  const auto final_base = block_base + layers * 34U;
  constexpr std::array<const char *, 3> kFinalNames = {
      "final_layer.adaln_proj.linear.weight",
      "final_layer.adaln_proj.linear.bias",
      "final_layer.norm.weight",
  };
  for (std::size_t weight = 0; weight < kFinalNames.size(); ++weight)
    append_checkpoint_binding(
        index, program, bundle, shard_indices, metadata,
        static_cast<std::uint32_t>(final_base + weight), kFinalNames[weight]);
  constexpr std::array<const char *, 4> kHeadNames = {
      "final_layer.video_out.weight", "final_layer.video_out.bias",
      "final_layer.audio_out.weight", "final_layer.audio_out.bias",
  };
  for (std::size_t weight = 0; weight < kHeadNames.size(); ++weight)
    append_checkpoint_binding(
        index, program, bundle, shard_indices, metadata,
        static_cast<std::uint32_t>(final_base + 8U + weight),
        kHeadNames[weight]);
  seal_or_hash_shards(bundle, sealed_shards, hash_shards);
  return bundle;
}

void usage() {
  std::cerr << "usage: difweights stats FILE.safetensors|FILE.index.json [--top N] [--json]\n"
               "       difweights inspect-shard FILE.safetensors\n"
               "       difweights inspect-index FILE.index.json\n"
               "       difweights inspect-bundle FILE.difbind\n"
               "       difweights make-h3-bundle INDEX PROGRAM.difir OUT.difbind\n"
               "       difweights rebind-h3-bundle SEALED.difbind INDEX PROGRAM.difir OUT.difbind\n"
               "       difweights make-h3-token-refiner-bundle INDEX PROGRAM.difir OUT.difbind\n"
               "       difweights rebind-h3-token-refiner-bundle SEALED.difbind INDEX PROGRAM.difir OUT.difbind\n"
               "       difweights check-h3-denoiser-bindings INDEX PROGRAM.difir\n"
               "       difweights make-h3-denoiser-bundle INDEX PROGRAM.difir OUT.difbind\n"
               "       difweights rebind-h3-denoiser-bundle SEALED.difbind INDEX PROGRAM.difir OUT.difbind\n"
               "       difweights make-krea2-bf16-bundle SOURCE.safetensors PROGRAM.difir DERIVED.safetensors OUT.difbind\n"
               "       difweights make-krea2-text-bf16-bundle SOURCE.safetensors PROGRAM.difir DERIVED.safetensors OUT.difbind\n"
               "       difweights make-krea2-vae-bf16-bundle SOURCE.safetensors PROGRAM.difir DERIVED.safetensors OUT.difbind\n"
               "       difweights make-h3-video-vae-bundle SOURCE.safetensors PROGRAM.difir OUT.safetensors OUT.difbind LATENT_T LATENT_H LATENT_W LAYERS resident|streamed generated|cudnn\n"
               "       difweights make-h3-video-encoder-bundle SOURCE.safetensors PROGRAM.difir OUT.difbind FRAMES HEIGHT WIDTH resident|streamed\n"
               "       difweights seal-h3-video-vae-bundle DERIVED.safetensors SOURCE.safetensors PROGRAM.difir OUT.difbind LATENT_T LATENT_H LATENT_W LAYERS resident|streamed generated|cudnn\n"
               "       difweights reuse-h3-video-vae-bundle SEALED.difbind PROGRAM.difir GEOMETRY.safetensors OUT.difbind LATENT_T LATENT_H LATENT_W LAYERS resident|streamed generated|cudnn\n"
               "       difweights rebind-program SEALED.difbind PROGRAM.difir OUT.difbind\n"
               "       difweights subset-bundle SOURCE.difbind PROGRAM.difir OUT.difbind\n"
               "       difweights verify-bundle FILE.difbind PROGRAM.difir\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 3) {
      usage();
      return 2;
    }
    const std::string command = argv[1];
    if (command == "stats" && argc >= 3) {
      // Checkpoint storage statistics: counts, dtypes, bytes, shapes, and
      // repeated storage patterns. Deliberately no model semantics: tensor
      // names are reported as examples, never interpreted.
      const std::filesystem::path source = argv[2];
      bool json = false;
      std::size_t top = 20U;
      for (int argument = 3; argument < argc; ++argument) {
        const std::string option = argv[argument];
        if (option == "--json")
          json = true;
        else if (option == "--top" && argument + 1 < argc)
          top = static_cast<std::size_t>(std::stoul(argv[++argument]));
        else
          dif::fail("unknown difweights stats option: " + option);
      }
      struct Entry {
        std::string name;
        dif::ir::DType dtype{};
        std::vector<std::uint64_t> dims;
        std::uint64_t bytes{};
        std::string shard;
      };
      std::vector<Entry> entries;
      std::vector<std::pair<std::string, std::uint64_t>> shard_sizes;
      const auto collect = [&](const std::filesystem::path &shard) {
        const auto file = dif::weights::read_safetensors(shard);
        shard_sizes.emplace_back(shard.string(), file.file_size);
        for (const auto &[name, tensor] : file.tensors)
          entries.push_back({name, tensor.dtype, tensor.dims, tensor.byte_count,
                             shard.string()});
      };
      if (source.extension() == ".json") {
        const auto index = dif::weights::read_safetensors_index(source);
        std::set<std::filesystem::path> shards;
        for (const auto &[name, shard] : index.weight_map)
          shards.insert(shard.is_absolute() ? shard
                                            : source.parent_path() / shard);
        for (const auto &shard : shards)
          collect(shard);
      } else {
        collect(source);
      }
      std::uint64_t total_bytes = 0U;
      std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> by_dtype;
      std::map<std::size_t, std::uint64_t> by_rank;
      struct Pattern {
        std::uint64_t count{};
        std::uint64_t bytes{};
        std::vector<std::string> examples;
      };
      std::map<std::string, Pattern> patterns;
      for (const auto &entry : entries) {
        total_bytes += entry.bytes;
        auto &dtype = by_dtype[std::string(dif::ir::dtype_name(entry.dtype))];
        ++dtype.first;
        dtype.second += entry.bytes;
        ++by_rank[entry.dims.size()];
        std::string key = std::string(dif::ir::dtype_name(entry.dtype)) + ":";
        for (std::size_t index = 0; index < entry.dims.size(); ++index)
          key += (index ? "x" : "") + std::to_string(entry.dims[index]);
        auto &pattern = patterns[key];
        ++pattern.count;
        pattern.bytes += entry.bytes;
        if (pattern.examples.size() < 3U)
          pattern.examples.push_back(entry.name);
      }
      std::vector<std::pair<std::string, Pattern>> ordered_patterns(
          patterns.begin(), patterns.end());
      std::sort(ordered_patterns.begin(), ordered_patterns.end(),
                [](const auto &left, const auto &right) {
                  if (left.second.count != right.second.count)
                    return left.second.count > right.second.count;
                  if (left.second.bytes != right.second.bytes)
                    return left.second.bytes > right.second.bytes;
                  return left.first < right.first;
                });
      std::vector<const Entry *> largest;
      for (const auto &entry : entries)
        largest.push_back(&entry);
      std::sort(largest.begin(), largest.end(),
                [](const Entry *left, const Entry *right) {
                  if (left->bytes != right->bytes)
                    return left->bytes > right->bytes;
                  return left->name < right->name;
                });
      if (largest.size() > top)
        largest.resize(top);
      auto document = dif::telemetry::make_document("weights-report");
      document.set("source", std::filesystem::absolute(source).string());
      document.set("note", "storage statistics only; no model semantics are "
                           "inferred from tensor names");
      dif::telemetry::Object totals;
      totals.set("tensors", entries.size());
      totals.set("bytes", total_bytes);
      totals.set("shards", shard_sizes.size());
      totals.set("distinct_shape_patterns", patterns.size());
      document.set("totals", std::move(totals));
      dif::telemetry::Array shards;
      for (const auto &[path, bytes] : shard_sizes) {
        dif::telemetry::Object shard;
        shard.set("path", path);
        shard.set("bytes", bytes);
        shards.push_back(std::move(shard));
      }
      document.set("shards", std::move(shards));
      dif::telemetry::Object dtypes;
      for (const auto &[name, counts] : by_dtype) {
        dif::telemetry::Object entry;
        entry.set("count", counts.first);
        entry.set("bytes", counts.second);
        dtypes.set(name, std::move(entry));
      }
      document.set("by_dtype", std::move(dtypes));
      dif::telemetry::Object ranks;
      for (const auto &[rank, count] : by_rank)
        ranks.set(std::to_string(rank), count);
      document.set("by_rank", std::move(ranks));
      dif::telemetry::Array pattern_entries;
      std::size_t emitted = 0U;
      for (const auto &[key, pattern] : ordered_patterns) {
        if (emitted++ >= std::max<std::size_t>(top, 100U))
          break;
        dif::telemetry::Object entry;
        entry.set("pattern", key);
        entry.set("count", pattern.count);
        entry.set("bytes", pattern.bytes);
        entry.set("repeated", pattern.count > 1U);
        dif::telemetry::Array examples;
        for (const auto &example : pattern.examples)
          examples.push_back(example);
        entry.set("example_names", std::move(examples));
        pattern_entries.push_back(std::move(entry));
      }
      document.set("shape_patterns", std::move(pattern_entries));
      dif::telemetry::Array largest_entries;
      for (const auto *entry : largest) {
        dif::telemetry::Object item;
        item.set("name", entry->name);
        item.set("dtype", dif::ir::dtype_name(entry->dtype));
        dif::telemetry::Array dims;
        for (const auto dim : entry->dims)
          dims.push_back(dim);
        item.set("dims", std::move(dims));
        item.set("bytes", entry->bytes);
        item.set("shard", entry->shard);
        largest_entries.push_back(std::move(item));
      }
      document.set("largest", std::move(largest_entries));
      if (json) {
        std::cout << dif::telemetry::serialize(dif::telemetry::Value(document));
        return 0;
      }
      std::cout << "WEIGHTS source=" << source.string()
                << " tensors=" << entries.size() << " bytes=" << total_bytes
                << " shards=" << shard_sizes.size()
                << " shape_patterns=" << patterns.size() << "\n";
      for (const auto &[name, counts] : by_dtype)
        std::cout << "dtype " << name << " count=" << counts.first
                  << " bytes=" << counts.second << "\n";
      for (const auto &[key, pattern] : ordered_patterns) {
        if (pattern.count < 2U)
          break;
        std::cout << "pattern " << key << " count=" << pattern.count
                  << " bytes=" << pattern.bytes << " e.g. "
                  << pattern.examples.front() << "\n";
      }
      for (const auto *entry : largest)
        std::cout << "largest " << entry->name << " bytes=" << entry->bytes
                  << "\n";
      return 0;
    }
    if (command == "inspect-shard" && argc == 3) {
      const auto file = dif::weights::read_safetensors(argv[2]);
      std::cout << "SAFETENSORS path=" << file.path << " bytes=" << file.file_size
                << " data_offset=" << file.data_offset
                << " tensors=" << file.tensors.size() << "\n";
      for (const auto &[name, tensor] : file.tensors) {
        std::cout << "tensor name=" << name
                  << " dtype=" << dif::ir::dtype_name(tensor.dtype) << " shape=";
        for (std::size_t i = 0; i < tensor.dims.size(); ++i)
          std::cout << (i ? "x" : "") << tensor.dims[i];
        std::cout << " offset=" << tensor.file_offset
                  << " bytes=" << tensor.byte_count << "\n";
      }
      return 0;
    }
    if (command == "inspect-index" && argc == 3) {
      const auto index = dif::weights::read_safetensors_index(argv[2]);
      std::cout << "SAFETENSORS_INDEX path=" << index.path
                << " tensors=" << index.weight_map.size() << "\n";
      for (const auto &[name, shard] : index.weight_map)
        std::cout << "weight name=" << name << " shard=" << shard << "\n";
      return 0;
    }
    if (command == "make-h3-bundle" && argc == 5) {
      const auto program = dif::ir::read_file(argv[3]);
      const auto bundle = make_h3_bundle(argv[2], program);
      dif::weights::write_weight_bundle(bundle, argv[4]);
      std::cout << "BUNDLE path=" << argv[4]
                << " program=" << dif::hex_digest(bundle.program_fingerprint)
                << " index=" << dif::hex_digest(bundle.index_fingerprint)
                << " shards=" << bundle.shards.size()
                << " bindings=" << bundle.bindings.size() << "\n";
      return 0;
    }
    if (command == "rebind-h3-bundle" && argc == 6) {
      const auto sealed = dif::weights::read_weight_bundle(argv[2]);
      if (sealed.index_fingerprint != dif::sha256_file(argv[3]))
        dif::fail("sealed bundle targets a different checkpoint index");
      const auto program = dif::ir::read_file(argv[4]);
      const auto bundle = make_h3_bundle(argv[3], program, &sealed);
      dif::weights::write_weight_bundle(bundle, argv[5]);
      std::cout << "REBIND path=" << argv[5]
                << " program=" << dif::hex_digest(bundle.program_fingerprint)
                << " shards=" << bundle.shards.size()
                << " bindings=" << bundle.bindings.size() << "\n";
      return 0;
    }
    if (command == "make-h3-token-refiner-bundle" && argc == 5) {
      const auto program = dif::ir::read_file(argv[3]);
      const auto bundle = make_h3_token_refiner_bundle(argv[2], program);
      dif::weights::write_weight_bundle(bundle, argv[4]);
      std::cout << "BUNDLE path=" << argv[4]
                << " program=" << dif::hex_digest(bundle.program_fingerprint)
                << " index=" << dif::hex_digest(bundle.index_fingerprint)
                << " shards=" << bundle.shards.size()
                << " bindings=" << bundle.bindings.size() << "\n";
      return 0;
    }
    if (command == "rebind-h3-token-refiner-bundle" && argc == 6) {
      const auto sealed = dif::weights::read_weight_bundle(argv[2]);
      if (sealed.index_fingerprint != dif::sha256_file(argv[3]))
        dif::fail("sealed bundle targets a different checkpoint index");
      const auto program = dif::ir::read_file(argv[4]);
      const auto bundle =
          make_h3_token_refiner_bundle(argv[3], program, &sealed);
      dif::weights::write_weight_bundle(bundle, argv[5]);
      std::cout << "REBIND path=" << argv[5]
                << " program=" << dif::hex_digest(bundle.program_fingerprint)
                << " shards=" << bundle.shards.size()
                << " bindings=" << bundle.bindings.size() << "\n";
      return 0;
    }
    if (command == "check-h3-denoiser-bindings" && argc == 4) {
      const auto program = dif::ir::read_file(argv[3]);
      const auto bundle =
          make_h3_denoiser_bundle(argv[2], program, nullptr, false);
      std::cout << "CHECK_H3_DENOISER_BINDINGS PASS program="
                << dif::hex_digest(bundle.program_fingerprint)
                << " shards=" << bundle.shards.size()
                << " bindings=" << bundle.bindings.size() << "\n";
      return 0;
    }
    if (command == "make-h3-denoiser-bundle" && argc == 5) {
      const auto program = dif::ir::read_file(argv[3]);
      const auto bundle = make_h3_denoiser_bundle(argv[2], program);
      dif::weights::write_weight_bundle(bundle, argv[4]);
      std::cout << "BUNDLE path=" << argv[4]
                << " program=" << dif::hex_digest(bundle.program_fingerprint)
                << " index=" << dif::hex_digest(bundle.index_fingerprint)
                << " shards=" << bundle.shards.size()
                << " bindings=" << bundle.bindings.size() << "\n";
      return 0;
    }
    if (command == "rebind-h3-denoiser-bundle" && argc == 6) {
      const auto sealed = dif::weights::read_weight_bundle(argv[2]);
      if (sealed.index_fingerprint != dif::sha256_file(argv[3]))
        dif::fail("sealed bundle targets a different checkpoint index");
      const auto program = dif::ir::read_file(argv[4]);
      const auto bundle =
          make_h3_denoiser_bundle(argv[3], program, &sealed);
      dif::weights::write_weight_bundle(bundle, argv[5]);
      std::cout << "REBIND path=" << argv[5]
                << " program="
                << dif::hex_digest(bundle.program_fingerprint)
                << " shards=" << bundle.shards.size()
                << " bindings=" << bundle.bindings.size() << "\n";
      return 0;
    }
    if (command == "make-krea2-bf16-bundle" && argc == 6) {
      if (std::filesystem::exists(argv[5]))
        dif::fail("refusing to overwrite Krea 2 BF16 weight bundle");
      dif::frontend::Krea2Config config;
      config.streamed_constants = true;
      const auto build = dif::frontend::make_krea2_denoiser(config, false);
      const auto program = dif::ir::read_file(argv[3]);
      if (dif::ir::fingerprint(program) !=
          dif::ir::fingerprint(build.program))
        dif::fail("Krea 2 program does not match the production denoiser");
      const auto bundle = materialize_mixed_bf16_bundle(
          argv[2], build.program, build.checkpoint_tensors,
          build.checkpoint_names, argv[4]);
      dif::weights::write_weight_bundle(bundle, argv[5]);
      std::cout << "KREA2_BF16_BUNDLE path=" << argv[5]
                << " shard=" << argv[4]
                << " program="
                << dif::hex_digest(bundle.program_fingerprint)
                << " checkpoint="
                << dif::hex_digest(bundle.index_fingerprint)
                << " bindings=" << bundle.bindings.size() << "\n";
      return 0;
    }
    if (command == "make-krea2-text-bf16-bundle" && argc == 6) {
      if (std::filesystem::exists(argv[5]))
        dif::fail("refusing to overwrite Krea 2 text BF16 weight bundle");
      const auto build = dif::frontend::make_krea2_text_fusion(true);
      const auto program = dif::ir::read_file(argv[3]);
      if (dif::ir::fingerprint(program) !=
          dif::ir::fingerprint(build.program))
        dif::fail("Krea 2 text program does not match production TextFusion");
      const auto bundle = materialize_mixed_bf16_bundle(
          argv[2], build.program, build.checkpoint_tensors,
          build.checkpoint_names, argv[4]);
      dif::weights::write_weight_bundle(bundle, argv[5]);
      std::cout << "KREA2_TEXT_BF16_BUNDLE path=" << argv[5]
                << " shard=" << argv[4]
                << " program="
                << dif::hex_digest(bundle.program_fingerprint)
                << " checkpoint="
                << dif::hex_digest(bundle.index_fingerprint)
                << " bindings=" << bundle.bindings.size() << "\n";
      return 0;
    }
    if (command == "make-krea2-vae-bf16-bundle" && argc == 6) {
      if (std::filesystem::exists(argv[5]))
        dif::fail("refusing to overwrite Qwen Image VAE BF16 bundle");
      dif::frontend::Krea2VaeConfig config;
      config.capture_boundaries = false;
      const auto build = dif::frontend::make_krea2_qwen_image_vae(config);
      const auto program = dif::ir::read_file(argv[3]);
      if (dif::ir::fingerprint(program) !=
          dif::ir::fingerprint(build.program))
        dif::fail("Qwen Image VAE program is not the 32x32 production tile");
      const auto bundle = materialize_krea2_vae_bundle(
          argv[2], build, argv[4]);
      dif::weights::write_weight_bundle(bundle, argv[5]);
      std::cout << "KREA2_VAE_BF16_BUNDLE path=" << argv[5]
                << " shard=" << argv[4]
                << " program="
                << dif::hex_digest(bundle.program_fingerprint)
                << " checkpoint="
                << dif::hex_digest(bundle.index_fingerprint)
                << " bindings=" << bundle.bindings.size() << "\n";
      return 0;
    }
    if (command == "make-h3-video-vae-bundle" && argc == 12) {
      if (std::filesystem::exists(argv[5]))
        dif::fail("refusing to overwrite H3 video VAE weight bundle");
      const std::string residency = argv[10];
      if (residency != "resident" && residency != "streamed")
        dif::fail("H3 video VAE residency must be resident or streamed");
      const std::string attention = argv[11];
      if (attention != "generated" && attention != "cudnn")
        dif::fail("H3 video VAE attention must be generated or cudnn");
      dif::frontend::H3VideoVaeConfig config;
      config.latent_frames = std::stoull(argv[6]);
      config.latent_height = std::stoull(argv[7]);
      config.latent_width = std::stoull(argv[8]);
      config.layers = std::stoull(argv[9]);
      config.streamed_constants = residency == "streamed";
      config.attention_implementation = attention == "cudnn" ? 2U : 1U;
      const auto build = dif::frontend::make_h3_video_vae_decoder(config);
      const auto program = dif::ir::read_file(argv[3]);
      if (dif::ir::fingerprint(program) !=
          dif::ir::fingerprint(build.program))
        dif::fail("H3 video VAE program does not match the requested geometry");
      const auto bundle =
          materialize_h3_video_vae_bundle(argv[2], build, argv[4]);
      dif::weights::write_weight_bundle(bundle, argv[5]);
      std::cout << "H3_VIDEO_VAE_BUNDLE path=" << argv[5]
                << " shard=" << argv[4]
                << " program="
                << dif::hex_digest(bundle.program_fingerprint)
                << " source=" << dif::hex_digest(bundle.index_fingerprint)
                << " bindings=" << bundle.bindings.size() << "\n";
      return 0;
    }
    if (command == "make-h3-video-encoder-bundle" && argc == 9) {
      if (std::filesystem::exists(argv[4]))
        dif::fail("refusing to overwrite H3 video encoder bundle");
      const std::string residency = argv[8];
      if (residency != "resident" && residency != "streamed")
        dif::fail("H3 video encoder residency must be resident or streamed");
      dif::frontend::H3VideoEncoderConfig config;
      config.frames = std::stoull(argv[5]);
      config.height = std::stoull(argv[6]);
      config.width = std::stoull(argv[7]);
      config.streamed_constants = residency == "streamed";
      config.capture_boundaries = false;
      const auto build = dif::frontend::make_h3_video_encoder(config);
      const auto program = dif::ir::read_file(argv[3]);
      if (dif::ir::fingerprint(program) !=
          dif::ir::fingerprint(build.program))
        dif::fail("H3 video encoder program does not match requested geometry");
      const auto bundle = make_h3_video_encoder_bundle(argv[2], build);
      dif::weights::write_weight_bundle(bundle, argv[4]);
      std::cout << "H3_VIDEO_ENCODER_BUNDLE path=" << argv[4]
                << " program="
                << dif::hex_digest(bundle.program_fingerprint)
                << " source=" << dif::hex_digest(bundle.index_fingerprint)
                << " bindings=" << bundle.bindings.size() << "\n";
      return 0;
    }
    if (command == "seal-h3-video-vae-bundle" && argc == 12) {
      if (std::filesystem::exists(argv[5]))
        dif::fail("refusing to overwrite H3 video VAE weight bundle");
      const std::string residency = argv[10];
      if (residency != "resident" && residency != "streamed")
        dif::fail("H3 video VAE residency must be resident or streamed");
      const std::string attention = argv[11];
      if (attention != "generated" && attention != "cudnn")
        dif::fail("H3 video VAE attention must be generated or cudnn");
      dif::frontend::H3VideoVaeConfig config;
      config.latent_frames = std::stoull(argv[6]);
      config.latent_height = std::stoull(argv[7]);
      config.latent_width = std::stoull(argv[8]);
      config.layers = std::stoull(argv[9]);
      config.streamed_constants = residency == "streamed";
      config.attention_implementation = attention == "cudnn" ? 2U : 1U;
      const auto build = dif::frontend::make_h3_video_vae_decoder(config);
      const auto program = dif::ir::read_file(argv[4]);
      if (dif::ir::fingerprint(program) !=
          dif::ir::fingerprint(build.program))
        dif::fail("H3 video VAE program does not match the requested geometry");
      const auto geometry_path =
          std::filesystem::path(std::string(argv[5]) + ".geometry.safetensors");
      const auto bundle = seal_h3_video_vae_derived_bundle(
          argv[2], argv[3], geometry_path, build);
      dif::weights::write_weight_bundle(bundle, argv[5]);
      std::cout << "H3_VIDEO_VAE_SEALED_BUNDLE path=" << argv[5]
                << " shard=" << argv[2]
                << " geometry_shard=" << geometry_path
                << " program="
                << dif::hex_digest(bundle.program_fingerprint)
                << " source=" << dif::hex_digest(bundle.index_fingerprint)
                << " bindings=" << bundle.bindings.size() << "\n";
      return 0;
    }
    if (command == "reuse-h3-video-vae-bundle" && argc == 12) {
      if (std::filesystem::exists(argv[5]))
        dif::fail("refusing to overwrite H3 video VAE reused bundle");
      const std::string residency = argv[10];
      if (residency != "resident" && residency != "streamed")
        dif::fail("H3 video VAE residency must be resident or streamed");
      const std::string attention = argv[11];
      if (attention != "generated" && attention != "cudnn")
        dif::fail("H3 video VAE attention must be generated or cudnn");
      dif::frontend::H3VideoVaeConfig config;
      config.latent_frames = std::stoull(argv[6]);
      config.latent_height = std::stoull(argv[7]);
      config.latent_width = std::stoull(argv[8]);
      config.layers = std::stoull(argv[9]);
      config.streamed_constants = residency == "streamed";
      config.attention_implementation = attention == "cudnn" ? 2U : 1U;
      const auto build = dif::frontend::make_h3_video_vae_decoder(config);
      const auto program = dif::ir::read_file(argv[3]);
      if (dif::ir::fingerprint(program) !=
          dif::ir::fingerprint(build.program))
        dif::fail("H3 video VAE program does not match the requested geometry");
      const auto sealed = dif::weights::read_weight_bundle(argv[2]);
      const auto bundle =
          reuse_h3_video_vae_bundle(sealed, build, argv[4]);
      dif::weights::write_weight_bundle(bundle, argv[5]);
      std::cout << "H3_VIDEO_VAE_REUSE_BUNDLE path=" << argv[5]
                << " geometry_shard=" << argv[4]
                << " program="
                << dif::hex_digest(bundle.program_fingerprint)
                << " source=" << dif::hex_digest(bundle.index_fingerprint)
                << " shards=" << bundle.shards.size()
                << " bindings=" << bundle.bindings.size() << "\n";
      return 0;
    }
    if (command == "rebind-program" && argc == 5) {
      if (std::filesystem::exists(argv[4]))
        dif::fail("refusing to overwrite rebound bundle");
      auto bundle = dif::weights::read_weight_bundle(argv[2]);
      const auto program = dif::ir::read_file(argv[3]);
      bundle.program_fingerprint = dif::ir::fingerprint(program);
      dif::weights::verify_weight_bundle(bundle, program, false);
      dif::weights::write_weight_bundle(bundle, argv[4]);
      std::cout << "REBIND_PROGRAM path=" << argv[4]
                << " program=" << dif::hex_digest(bundle.program_fingerprint)
                << " payload_provenance="
                << dif::hex_digest(bundle.index_fingerprint)
                << " shards=" << bundle.shards.size()
                << " bindings=" << bundle.bindings.size() << "\n";
      return 0;
    }
    if (command == "inspect-bundle" && argc == 3) {
      const auto bundle = dif::weights::read_weight_bundle(argv[2]);
      std::cout << "BUNDLE path=" << argv[2]
                << " program=" << dif::hex_digest(bundle.program_fingerprint)
                << " index=" << dif::hex_digest(bundle.index_fingerprint)
                << " shards=" << bundle.shards.size()
                << " bindings=" << bundle.bindings.size() << "\n";
      for (std::size_t i = 0; i < bundle.shards.size(); ++i)
        std::cout << "shard id=" << i << " path=" << bundle.shards[i].path
                  << " bytes=" << bundle.shards[i].file_size
                  << " sha256=" << dif::hex_digest(bundle.shards[i].digest)
                  << "\n";
      for (const auto &binding : bundle.bindings)
        std::cout << "binding tensor=" << binding.tensor_id
                  << " shard=" << binding.shard_index
                  << " name=" << binding.tensor_name << "\n";
      return 0;
    }
    if (command == "subset-bundle" && argc == 5) {
      if (std::filesystem::exists(argv[4]))
        dif::fail("refusing to overwrite subset weight bundle");
      const auto source = dif::weights::read_weight_bundle(argv[2]);
      const auto program = dif::ir::read_file(argv[3]);
      const auto subset = dif::weights::subset_weight_bundle(source, program);
      dif::weights::write_weight_bundle(subset, argv[4]);
      std::cout << "SUBSET_BUNDLE path=" << argv[4]
                << " program="
                << dif::hex_digest(subset.program_fingerprint)
                << " index=" << dif::hex_digest(subset.index_fingerprint)
                << " shards=" << subset.shards.size()
                << " bindings=" << subset.bindings.size() << "\n";
      return 0;
    }
    if (command == "verify-bundle" && argc == 4) {
      const auto bundle = dif::weights::read_weight_bundle(argv[2]);
      const auto program = dif::ir::read_file(argv[3]);
      dif::weights::verify_weight_bundle(bundle, program, true);
      std::cout << "VERIFY_BUNDLE PASS path=" << argv[2]
                << " shards=" << bundle.shards.size()
                << " bindings=" << bundle.bindings.size() << "\n";
      return 0;
    }
    usage();
    return 2;
  } catch (const std::exception &error) {
    std::cerr << "difweights: " << error.what() << "\n";
    return 1;
  }
}
