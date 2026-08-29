#include "dif/ir/codec.hpp"
#include "dif/ir/ir.hpp"
#include "dif/frontend/h3_vae.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"
#include "dif/weights/bundle.hpp"
#include "dif/weights/safetensors.hpp"

#include <algorithm>
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
  std::cerr << "usage: difweights inspect-shard FILE.safetensors\n"
               "       difweights inspect-index FILE.index.json\n"
               "       difweights inspect-bundle FILE.difbind\n"
               "       difweights make-h3-bundle INDEX PROGRAM.difir OUT.difbind\n"
               "       difweights rebind-h3-bundle SEALED.difbind INDEX PROGRAM.difir OUT.difbind\n"
               "       difweights make-h3-token-refiner-bundle INDEX PROGRAM.difir OUT.difbind\n"
               "       difweights rebind-h3-token-refiner-bundle SEALED.difbind INDEX PROGRAM.difir OUT.difbind\n"
               "       difweights check-h3-denoiser-bindings INDEX PROGRAM.difir\n"
               "       difweights make-h3-denoiser-bundle INDEX PROGRAM.difir OUT.difbind\n"
               "       difweights rebind-h3-denoiser-bundle SEALED.difbind INDEX PROGRAM.difir OUT.difbind\n"
               "       difweights make-h3-video-vae-bundle SOURCE.safetensors PROGRAM.difir OUT.safetensors OUT.difbind LATENT_T LATENT_H LATENT_W LAYERS resident|streamed generated|cudnn\n"
               "       difweights reuse-h3-video-vae-bundle SEALED.difbind PROGRAM.difir GEOMETRY.safetensors OUT.difbind LATENT_T LATENT_H LATENT_W LAYERS resident|streamed generated|cudnn\n"
               "       difweights rebind-program SEALED.difbind PROGRAM.difir OUT.difbind\n"
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
