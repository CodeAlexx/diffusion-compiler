// Exercise the actual tool with small native-builder dimensions. No neural
// execution, checkpoint download, or substitute geometry-rebinding algorithm.
#define main difimport_tool_main
#include "../tools/difimport.cpp"
#undef main

#include <algorithm>
#include <cstdlib>

namespace {
void require(bool condition, const char *message) {
  if (!condition) dif::fail(message);
}
template<class F> void rejects(F &&operation, const char *message) {
  bool failed = false;
  try { operation(); } catch (const std::exception &) { failed = true; }
  require(failed, message);
}

dif::weights::WeightBundle fixture(const fs::path &directory,
    const dif::frontend::AudioBigVganBuild &build) {
  std::vector<dif::weights::SafeTensorWriteSpec> specs;
  for (const auto &binding : build.bindings) {
    const auto *desc = build.program.tensor(binding.tensor_id);
    specs.push_back({binding.name, desc->dtype, desc->dims});
  }
  dif::weights::SafeTensorWriter writer(directory / "source.safetensors", std::move(specs));
  for (const auto &binding : build.bindings) {
    if (binding.source_name.empty()) {
      const auto &tensor = build.generated_constants.at(binding.tensor_id);
      writer.append(binding.name, {tensor.data(), tensor.byte_size()});
    } else {
      auto tensor = dif::runtime::zeros(*build.program.tensor(binding.tensor_id));
      for (std::size_t i = 0; i < tensor.element_count(); ++i)
        tensor.f32()[i] = float((i + binding.tensor_id) % 31) / 32.0F;
      writer.append(binding.name, {tensor.data(), tensor.byte_size()});
    }
  }
  const auto shard = writer.finish();
  dif::weights::WeightBundle bundle;
  bundle.program_fingerprint = dif::ir::fingerprint(build.program);
  bundle.index_fingerprint = dif::sha256_file(directory / "source.safetensors");
  bundle.shards.push_back({directory / "source.safetensors", shard.file_size,
                          bundle.index_fingerprint});
  for (const auto &binding : build.bindings) {
    const auto *entry = shard.find(binding.name);
    bundle.bindings.push_back({binding.tensor_id, 0U, binding.name, entry->dtype,
        entry->dims, entry->file_offset, entry->byte_count});
  }
  dif::weights::write_weight_bundle(bundle, directory / "source.difbind");
  return bundle;
}
}

int main() {
  try {
    char pattern[] = "/tmp/dif-h3-audio-rebind-tests.XXXXXX";
    const auto *created = ::mkdtemp(pattern);
    if (!created) dif::fail("cannot create audio rebind fixture directory");
    const fs::path directory(created);
    dif::frontend::AudioBigVganConfig config;
    config.latent_dim = 16;
    config.decoder_dim = 8;
    const auto old_build = dif::frontend::build_audio_bigvgan_program(2, 3, 1, config);
    const auto new_build = dif::frontend::build_audio_bigvgan_program(1, 7, 1, config);
    dif::ir::write_file(old_build.program, directory / "old.difir");
    dif::ir::write_file(new_build.program, directory / "new.difir");
    const auto source = fixture(directory, old_build);
    const auto source_digest = dif::sha256_file(directory / "source.safetensors");
    rebind_audio_bundle(directory / "source.difbind", directory / "old.difir",
        directory / "new.difir", directory / "generated.safetensors",
        directory / "new.difbind", 1, 7, 1, true, config);
    const auto rebound = dif::weights::read_weight_bundle(directory / "new.difbind");
    const auto loaded = dif::weights::load_weight_bundle(rebound, new_build.program, true);
    const auto old_loaded = dif::weights::load_weight_bundle(source, old_build.program, true);
    require(rebound.index_fingerprint == source.index_fingerprint, "source index provenance changed");
    require(rebound.shards.size() == 2U && rebound.shards[0].path == source.shards[0].path &&
        rebound.shards[0].digest == source.shards[0].digest &&
        rebound.shards[0].file_size == source.shards[0].file_size,
        "learned shard receipt was not reused unchanged");
    std::size_t generated_count = 0U;
    for (const auto &binding : new_build.bindings) {
      const auto &actual = loaded.at(binding.tensor_id);
      if (binding.source_name.empty()) {
        ++generated_count;
        const auto &expected = new_build.generated_constants.at(binding.tensor_id);
        require(actual.dims == expected.dims && actual.byte_size() == expected.byte_size() &&
            std::memcmp(actual.data(), expected.data(), expected.byte_size()) == 0,
            "generated constants differ from native builder payload");
      } else {
        const auto &expected = old_loaded.at(binding.tensor_id);
        require(actual.dims == expected.dims &&
            std::memcmp(actual.data(), expected.data(), expected.byte_size()) == 0,
            "learned tensor changed during geometry rebind");
        const auto old_binding = std::find_if(source.bindings.begin(), source.bindings.end(),
            [&](const auto &value) { return value.tensor_id == binding.tensor_id; });
        const auto new_binding = std::find_if(rebound.bindings.begin(), rebound.bindings.end(),
            [&](const auto &value) { return value.tensor_id == binding.tensor_id; });
        require(old_binding->file_offset == new_binding->file_offset &&
            old_binding->byte_count == new_binding->byte_count &&
            old_binding->tensor_name == new_binding->tensor_name,
            "learned tensor receipt moved or changed");
      }
    }
    const auto generated = dif::weights::read_safetensors(directory / "generated.safetensors");
    require(generated_count == 3U && generated.tensors.size() == generated_count,
            "new shard contains learned tensors or misses generated constants");
    require(dif::sha256_file(directory / "source.safetensors") == source_digest,
            "source shard was modified");

    // Repeated geometry rebinding drops obsolete generated-only shards.
    rebind_audio_bundle(directory / "new.difbind", directory / "new.difir",
        directory / "old.difir", directory / "return-generated.safetensors",
        directory / "return.difbind", 2, 3, 1, false, config);
    const auto returned = dif::weights::read_weight_bundle(directory / "return.difbind");
    require(returned.shards.size() == 2U, "rebind accumulated unused generated shards");

    const auto reject_rebind = [&](const fs::path &bundle, const fs::path &old_program,
                                   const fs::path &new_program, std::uint64_t frames = 7) {
      rejects([&] { rebind_audio_bundle(bundle, old_program, new_program,
          directory / "rejected.safetensors", directory / "rejected.difbind",
          1, frames, 1, false, config); }, "bad audio source/geometry was silently accepted");
      require(!fs::exists(directory / "rejected.safetensors") &&
          !fs::exists(directory / "rejected.difbind"), "rejection wrote output before validation");
    };
    reject_rebind(directory / "source.difbind", directory / "new.difir", directory / "new.difir");
    reject_rebind(directory / "source.difbind", directory / "old.difir", directory / "new.difir", 8);
    auto altered = old_build.program;
    altered.operations.back().id += 1;
    dif::ir::write_file(altered, directory / "altered.difir");
    auto altered_bundle = source;
    altered_bundle.program_fingerprint = dif::ir::fingerprint(altered);
    dif::weights::write_weight_bundle(altered_bundle, directory / "altered.difbind");
    reject_rebind(directory / "altered.difbind", directory / "altered.difir", directory / "new.difir");
    auto missing = source;
    missing.bindings.pop_back();
    dif::weights::write_weight_bundle(missing, directory / "missing.difbind");
    reject_rebind(directory / "missing.difbind", directory / "old.difir", directory / "new.difir");
    auto wrong_name = source;
    wrong_name.bindings[0].tensor_name = "not_the_native_constant";
    dif::weights::write_weight_bundle(wrong_name, directory / "wrong-name.difbind");
    reject_rebind(directory / "wrong-name.difbind", directory / "old.difir", directory / "new.difir");
    rejects([&] { rebind_audio_bundle(directory / "source.difbind", directory / "old.difir",
        directory / "new.difir", directory / "generated.safetensors",
        directory / "never.difbind", 1, 7, 1, false, config); }, "existing output overwrite allowed");

    // Full official schema proves all seven length-dependent vectors are the
    // only shapes that differ; payload fixture stays deliberately small.
    const auto official_old = dif::frontend::build_audio_bigvgan_program(2, 188, 8);
    const auto official_new = dif::frontend::build_audio_bigvgan_program(2, 263, 8);
    std::size_t changed_generated = 0;
    for (std::size_t i = 0; i < official_old.bindings.size(); ++i) {
      const auto &old_binding = official_old.bindings[i];
      const auto &new_binding = official_new.bindings.at(i);
      require(old_binding.tensor_id == new_binding.tensor_id && old_binding.name == new_binding.name,
              "official geometry changed constant identity");
      const auto &old_dims = official_old.program.tensor(old_binding.tensor_id)->dims;
      const auto &new_dims = official_new.program.tensor(new_binding.tensor_id)->dims;
      if (old_dims != new_dims) {
        require(old_binding.source_name.empty(), "official geometry changes learned weights");
        ++changed_generated;
      }
    }
    require(changed_generated == 7U, "official decoder does not change exactly seven generated shapes");
    std::cout << "H3_AUDIO_REBIND_TEST PASS generated_payloads=byte_exact learned_receipts=unchanged "
                 "official_changed_shapes=7 fixture=" << directory.string() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "H3_AUDIO_REBIND_TEST FAIL " << error.what() << '\n';
    return 1;
  }
}
