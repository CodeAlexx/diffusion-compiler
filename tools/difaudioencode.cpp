// Native H3 reference AudioVAE encoder. Model operations live in the shared
// frontend/runtime; host glue follows minimax_h3_ref2va.mojo's
// _h3_ref2va_reference_audio_rows and minimax_h3_ref_encode.mojo's
// minimax_h3_audio_condition_rows (posterior mode, F32 normalization).

#include "dif/frontend/h3_audio_encoder.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/support/json.hpp"
#include "dif/weights/safetensors.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <unistd.h>

namespace {
namespace fs = std::filesystem;

struct Options {
  fs::path config, checkpoint, input, output_mean, output_rows, cache;
  std::string backend{"cuda"};
  std::uint64_t channels{};
  std::uint64_t minimum_free_mib{256U};
  std::optional<std::uint64_t> max_samples;
};

void usage() {
  std::cout
      << "usage: difaudioencode --config FILE.json --checkpoint FILE.safetensors "
         "--input-f32 RAW --channels 1|2 --output-mean FILE.diftensor "
         "[--output-rows FILE.diftensor] [--backend cuda|cpu] "
         "[--cache-dir DIR] [--min-free-mib N] [--max-samples N]\n"
         "RAW: interleaved little-endian F32 at the JSON sampling_rate; "
         "decode/resample media with ffmpeg first.\n"
         "The waveform is truncated to the JSON max_samples or the smaller "
         "--max-samples. Mono is repeated into stereo.\n"
         "Outputs: unnormalized posterior mean [2,C,T]; optional normalized "
         "F32 rows [2*T,C], left channel then right. Existing files are refused.\n";
}

std::uint64_t integer(std::string_view value, const std::string &label) {
  std::uint64_t result{};
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  if (value.empty() || parsed.ec != std::errc{} ||
      parsed.ptr != value.data() + value.size())
    dif::fail(label + " must be an unsigned integer");
  return result;
}

Options parse(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    auto value = [&]() -> std::string {
      if (++index >= argc)
        dif::fail("missing value for " + option);
      return argv[index];
    };
    if (option == "--config") options.config = value();
    else if (option == "--checkpoint") options.checkpoint = value();
    else if (option == "--input-f32") options.input = value();
    else if (option == "--channels") options.channels = integer(value(), option);
    else if (option == "--output-mean") options.output_mean = value();
    else if (option == "--output-rows") options.output_rows = value();
    else if (option == "--backend") options.backend = value();
    else if (option == "--cache-dir") options.cache = value();
    else if (option == "--min-free-mib") options.minimum_free_mib = integer(value(), option);
    else if (option == "--max-samples") options.max_samples = integer(value(), option);
    else dif::fail("unknown option: " + option);
  }
  if (options.config.empty() || options.checkpoint.empty() || options.input.empty() ||
      options.output_mean.empty())
    dif::fail("--config, --checkpoint, --input-f32 and --output-mean are required");
  if (options.channels != 1U && options.channels != 2U)
    dif::fail("--channels must be 1 or 2");
  if (options.backend != "cuda" && options.backend != "cpu")
    dif::fail("--backend must be cuda or cpu");
  if (options.max_samples && *options.max_samples == 0U)
    dif::fail("--max-samples must be positive");
  if (options.minimum_free_mib > std::numeric_limits<std::uint64_t>::max() / (1024U * 1024U))
    dif::fail("--min-free-mib is too large");
  return options;
}

dif::json::Value read_config(const fs::path &path) {
  if (!fs::is_regular_file(path))
    dif::fail("config must be a regular file: " + path.string());
  const auto size = fs::file_size(path);
  if (size == 0U || size > 1024U * 1024U)
    dif::fail("audio encoder JSON must contain between 1 byte and 1 MiB");
  std::string contents(static_cast<std::size_t>(size), '\0');
  std::ifstream stream(path, std::ios::binary);
  if (!stream.read(contents.data(), static_cast<std::streamsize>(contents.size())) ||
      stream.peek() != std::char_traits<char>::eof())
    dif::fail("failed to read complete config: " + path.string());
  const auto document = dif::json::parse(contents);
  if (!document.is_object()) dif::fail("audio encoder config must be a JSON object");
  return document;
}

std::uint64_t config_integer(const dif::json::Value &document, const char *key) {
  const auto *entry = document.find(key);
  if (!entry) dif::fail(std::string("audio encoder config requires ") + key);
  const double value = entry->number();
  // JSON numbers are doubles; restrict integers to their exact range.
  if (!std::isfinite(value) || value < 1.0 || value > 9007199254740991.0 ||
      std::floor(value) != value)
    dif::fail(std::string("audio encoder config requires a positive integer ") + key);
  return static_cast<std::uint64_t>(value);
}

std::vector<float> statistics(const dif::json::Value &document, const char *key,
                              std::uint64_t channels, bool positive) {
  const auto *entry = document.find(key);
  if (!entry || !entry->is_array() || entry->array().size() != channels)
    dif::fail(std::string(key) + " requires one value per latent channel");
  std::vector<float> result;
  result.reserve(entry->array().size());
  for (const auto &item : entry->array()) {
    const float value = static_cast<float>(item.number());
    if (!std::isfinite(value) || (positive && value <= 0.0F))
      dif::fail(std::string(key) + " contains an invalid F32 value");
    result.push_back(value);
  }
  return result;
}

void check_output(const fs::path &path) {
  if (fs::symlink_status(path).type() != fs::file_type::not_found)
    dif::fail("refusing to overwrite " + path.string());
  const auto parent = path.has_parent_path() ? path.parent_path() : fs::path(".");
  if (!fs::is_directory(parent))
    dif::fail("output parent directory does not exist: " + parent.string());
}

// Publish through an exclusive hard link, so even a file created after the
// preflight check cannot be overwritten by the shared tensor writer.
void write_new_tensor(const dif::runtime::Tensor &tensor, const fs::path &path) {
  std::string pattern = path.string() + ".tmp.XXXXXX";
  std::vector<char> name(pattern.begin(), pattern.end());
  name.push_back('\0');
  const int descriptor = ::mkstemp(name.data());
  if (descriptor < 0) dif::fail("cannot create tensor temporary file for " + path.string());
  ::close(descriptor);
  const fs::path temporary(name.data());
  try {
    dif::runtime::write_tensor(tensor, temporary);
    fs::create_hard_link(temporary, path);
  } catch (...) {
    std::error_code ignored;
    fs::remove(temporary, ignored);
    throw;
  }
  std::error_code ignored;
  fs::remove(temporary, ignored);
}

struct Waveform {
  dif::runtime::Tensor tensor;
  std::uint64_t source_samples{}, kept_samples{};
};

Waveform read_waveform(const Options &options, std::uint64_t max_samples) {
  if constexpr (std::endian::native != std::endian::little)
    dif::fail("raw little-endian F32 audio requires a little-endian host");
  if (!fs::is_regular_file(options.input))
    dif::fail("--input-f32 must be a regular file");
  const auto size = fs::file_size(options.input);
  const auto frame_bytes = options.channels * sizeof(float);
  if (size == 0U || size % frame_bytes != 0U)
    dif::fail("--input-f32 must contain complete, nonempty interleaved F32 frames");
  const auto source_samples = size / frame_bytes;
  const auto kept = std::min(source_samples, max_samples);
  if (kept > std::numeric_limits<std::size_t>::max() / (2U * sizeof(float)) ||
      kept > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()) / frame_bytes)
    dif::fail("audio input sample bound is too large");
  std::vector<float> interleaved(static_cast<std::size_t>(kept * options.channels));
  std::ifstream stream(options.input, std::ios::binary);
  if (!stream.read(reinterpret_cast<char *>(interleaved.data()),
                   static_cast<std::streamsize>(kept * frame_bytes)))
    dif::fail("failed to read bounded waveform: " + options.input.string());
  for (const auto sample : interleaved)
    if (!std::isfinite(sample)) dif::fail("--input-f32 contains a nonfinite retained sample");

  // Source prepare_reference_waveform repeats mono into stereo. Each channel
  // is an independent batch item of the mono encoder, [2,1,S].
  auto waveform = dif::runtime::zeros({0U, dif::ir::DType::F32, 0U, {2U, 1U, kept}});
  auto destination = waveform.f32();
  for (std::uint64_t channel = 0; channel < 2U; ++channel)
    for (std::uint64_t sample = 0; sample < kept; ++sample)
      destination[channel * kept + sample] =
          interleaved[sample * options.channels + (options.channels == 1U ? 0U : channel)];
  return {std::move(waveform), source_samples, kept};
}

dif::runtime::Tensor condition_rows(const dif::runtime::Tensor &mean,
                                     const std::vector<float> &latents_mean,
                                     const std::vector<float> &latents_std) {
  const auto channels = mean.dims[1], frames = mean.dims[2];
  auto rows = dif::runtime::zeros({0U, dif::ir::DType::F32, 0U, {2U * frames, channels}});
  const auto source = mean.f32();
  auto destination = rows.f32();
  for (std::uint64_t batch = 0U; batch < 2U; ++batch)
    for (std::uint64_t frame = 0U; frame < frames; ++frame)
      for (std::uint64_t channel = 0U; channel < channels; ++channel) {
        const float centered = source[(batch * channels + channel) * frames + frame] -
                               latents_mean[channel];
        const float value = centered / latents_std[channel];
        if (!std::isfinite(value)) dif::fail("audio normalization produced a nonfinite row");
        destination[(batch * frames + frame) * channels + channel] = value;
      }
  return rows;
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc == 1 || (argc == 2 && std::string_view(argv[1]) == "--help")) {
      usage();
      return argc == 1 ? 2 : 0;
    }
    const auto options = parse(argc, argv);
    check_output(options.output_mean);
    if (!options.output_rows.empty()) {
      check_output(options.output_rows);
      if (fs::weakly_canonical(options.output_mean) == fs::weakly_canonical(options.output_rows))
        dif::fail("--output-mean and --output-rows must be different files");
    }
    const auto document = read_config(options.config);
    const auto config = dif::frontend::h3_audio_encoder_config(document);
    const auto sampling_rate = config_integer(document, "sampling_rate");
    auto max_samples = config_integer(document, "max_samples");
    if (options.max_samples) max_samples = std::min(max_samples, *options.max_samples);
    const auto latents_mean = statistics(document, "latents_mean", config.latent_channels, false);
    const auto latents_std = statistics(document, "latents_std", config.latent_channels, true);
    auto waveform = read_waveform(options, max_samples);

    auto build = dif::frontend::build_h3_audio_encoder_program(2U, waveform.kept_samples, config);
    dif::ir::verify(build.program);
    const auto checkpoint = dif::weights::read_safetensors(options.checkpoint);
    auto bindings = dif::frontend::bind_h3_audio_encoder_weights(build, checkpoint);
    for (auto &[id, constant] : build.generated_constants)
      bindings.insert_or_assign(id, std::move(constant));
    bindings.emplace(build.waveform_input_id, std::move(waveform.tensor));

    dif::runtime::RunOptions run;
    run.warmups = 0U;
    run.iterations = 1U;
    run.minimum_free_bytes = options.minimum_free_mib * 1024U * 1024U;
    run.cache_directory = options.cache;
    run.requested_outputs = {build.mean_output_id};
    auto executor = options.backend == "cuda" ? dif::runtime::make_cuda_executor()
                                              : dif::runtime::make_cpu_executor();
    auto prepared = executor->prepare(build.program, bindings, run);
    const auto result = prepared->run(bindings, run);
    const auto &mean = result.outputs.at(build.mean_output_id);
    mean.validate();
    const auto *expected = build.program.tensor(build.mean_output_id);
    if (!expected || mean.dtype != dif::ir::DType::F32 || mean.dims != expected->dims ||
        mean.dims.size() != 3U || mean.dims[0] != 2U ||
        mean.dims[1] != config.latent_channels || mean.dims[2] == 0U)
      dif::fail("audio encoder mean must be F32 [2,C,T] with the configured geometry");
    for (const auto value : mean.f32())
      if (!std::isfinite(value)) dif::fail("audio encoder produced a nonfinite posterior mean");
    std::optional<dif::runtime::Tensor> rows;
    if (!options.output_rows.empty()) rows = condition_rows(mean, latents_mean, latents_std);
    write_new_tensor(mean, options.output_mean);
    if (rows) write_new_tensor(*rows, options.output_rows);
    std::cout << "AUDIO_ENCODE backend=" << result.backend_name
              << " sampling_rate=" << sampling_rate
              << " input_channels=" << options.channels << " output_channels=2"
              << " source_samples=" << waveform.source_samples
              << " kept_samples=" << waveform.kept_samples
              << " latent_frames=" << mean.dims[2] << " nonfinite=0"
              << " mean=" << options.output_mean.string();
    if (rows) std::cout << " rows=" << options.output_rows.string();
    std::cout << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difaudioencode: " << error.what() << '\n';
    return 1;
  }
}
