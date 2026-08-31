// difaudiodecode — native BigVGAN audio decode (chunk 7 of
// docs/BIGVGAN_DECODE_PLAN.md). Replaces the Mojo fresh-process bridge:
// loads the recorded [2T, 32] audio state rows, unpacks them to the
// [2, 32, T] latent (rearrange.mojo semantics), runs the self-contained
// decoder program (denormalization is IN-PROGRAM), and writes the stereo
// 32 kHz 16-bit PCM WAV with wav.mojo's exact quantizer.

#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/support/wav.hpp"
#include "dif/weights/bundle.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

namespace {

void usage() {
  std::cerr << "usage: difaudiodecode --backend cpu|cuda --program FILE.difir "
               "--weight-bundle FILE.difbind --input ROWS.diftensor "
               "--output-wav FILE.wav [--output-waveform FILE.diftensor] "
               "[--sample-rate N] [--verify-shards] [--cache-dir DIR] "
               "[--min-free-mib N]\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::string backend = "cpu";
    std::filesystem::path program_path, bundle_path, input_path, wav_path,
        waveform_path;
    std::uint32_t sample_rate = 32000U;
    bool verify_shards = false;
    std::filesystem::path cache_directory;
    // The decoder program's resident weights are ~235 MB; the historical
    // free-VRAM guard is off by default here and overridable per the plan's
    // documented tool signature.
    std::uint64_t minimum_free_bytes = 0U;
    for (int index = 1; index < argc; ++index) {
      const std::string option = argv[index];
      auto value = [&](const char *label) -> std::string {
        if (++index >= argc)
          dif::fail(std::string("missing value for ") + label);
        return argv[index];
      };
      if (option == "--backend")
        backend = value("--backend");
      else if (option == "--program")
        program_path = value("--program");
      else if (option == "--weight-bundle")
        bundle_path = value("--weight-bundle");
      else if (option == "--input")
        input_path = value("--input");
      else if (option == "--output-wav")
        wav_path = value("--output-wav");
      else if (option == "--output-waveform")
        waveform_path = value("--output-waveform");
      else if (option == "--sample-rate")
        sample_rate = static_cast<std::uint32_t>(std::stoul(value("--sample-rate")));
      else if (option == "--verify-shards")
        verify_shards = true;
      else if (option == "--cache-dir")
        cache_directory = value("--cache-dir");
      else if (option == "--min-free-mib")
        minimum_free_bytes =
            std::stoull(value("--min-free-mib")) * 1024ULL * 1024ULL;
      else {
        usage();
        return 2;
      }
    }
    if (program_path.empty() || bundle_path.empty() || input_path.empty() ||
        wav_path.empty()) {
      usage();
      return 2;
    }

    const auto program = dif::ir::read_file(program_path);
    dif::ir::verify(program);

    // Rows -> [2, C, T] latent (rearrange.mojo minimax_h3_unpack_audio).
    const auto rows = dif::runtime::read_tensor(input_path);
    if (rows.dtype != dif::ir::DType::F32 || rows.dims.size() != 2U ||
        rows.dims[0] % 2U != 0U)
      dif::fail("audio rows must be F32 [2T, C]");
    const auto frames = rows.dims[0] / 2U;
    const auto channels = rows.dims[1];
    dif::runtime::Tensor latent{dif::ir::DType::F32, {2U, channels, frames},
                                {}};
    latent.bytes.resize(static_cast<std::size_t>(latent.element_count()) *
                        sizeof(float));
    {
      const auto *source = reinterpret_cast<const float *>(rows.data());
      auto *destination = reinterpret_cast<float *>(latent.bytes.data());
      for (std::uint64_t stereo = 0; stereo < 2U; ++stereo)
        for (std::uint64_t t = 0; t < frames; ++t)
          for (std::uint64_t c = 0; c < channels; ++c)
            destination[(stereo * channels + c) * frames + t] =
                source[(stereo * frames + t) * channels + c];
    }

    // Find the program's single input and output tensors.
    std::uint32_t input_id = 0U, output_id = 0U;
    for (const auto &tensor : program.tensors) {
      if (tensor.has_role(dif::ir::TensorRole::Input) &&
          !tensor.has_role(dif::ir::TensorRole::Constant)) {
        if (input_id)
          dif::fail("decoder program has more than one input");
        input_id = tensor.id;
      }
      if (tensor.has_role(dif::ir::TensorRole::Output)) {
        if (output_id)
          dif::fail("decoder program has more than one output");
        output_id = tensor.id;
      }
    }
    if (!input_id || !output_id)
      dif::fail("decoder program is missing its input or output");
    const auto *input_description = program.tensor(input_id);
    if (input_description->dims != latent.dims)
      dif::fail("decoder program input geometry does not match the rows");

    const auto bundle = dif::weights::read_weight_bundle(bundle_path);
    auto bindings =
        dif::weights::load_weight_bundle(bundle, program, verify_shards);
    bindings.emplace(input_id, std::move(latent));

    dif::runtime::RunOptions options;
    options.warmups = 0U;
    options.iterations = 1U;
    options.minimum_free_bytes = minimum_free_bytes;
    options.cache_directory = cache_directory;
    std::unique_ptr<dif::runtime::Executor> executor;
    if (backend == "cpu")
      executor = dif::runtime::make_cpu_executor();
    else if (backend == "cuda")
      executor = dif::runtime::make_cuda_executor();
    else
      dif::fail("unknown backend: " + backend);
    const auto result = executor->run(program, bindings, options);
    const auto &waveform = result.outputs.at(output_id);
    if (waveform.dtype != dif::ir::DType::F32 || waveform.dims.size() != 3U ||
        waveform.dims[1] != 1U)
      dif::fail("decoder output must be F32 [B, 1, samples]");
    const auto out_channels = static_cast<std::uint32_t>(waveform.dims[0]);
    const auto samples = waveform.dims[2];
    const auto *values = reinterpret_cast<const float *>(waveform.data());
    std::uint64_t nonfinite = 0U;
    for (std::uint64_t i = 0; i < waveform.element_count(); ++i)
      nonfinite += !std::isfinite(values[i]);
    if (waveform_path.empty() == false)
      dif::runtime::write_tensor(waveform, waveform_path);
    dif::support::write_wav_pcm16(
        wav_path, {values, static_cast<std::size_t>(waveform.element_count())},
        out_channels, samples, sample_rate);
    std::cout << "AUDIO_DECODE PASS backend=" << result.backend_name
              << " frames=" << frames << " samples_per_channel=" << samples
              << " channels=" << out_channels << " nonfinite=" << nonfinite
              << " wav=" << wav_path.string() << "\n";
    return nonfinite == 0U ? 0 : 3;
  } catch (const std::exception &error) {
    std::cerr << "difaudiodecode: " << error.what() << "\n";
    return 1;
  }
}
