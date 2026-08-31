// difimport — native port of tools/import_serenity_h3_inputs.py.
//
// Byte-preserving container conversion of recorded Serenity H3
// conditioner/initial-state SafeTensors into DiffTensor v1 files plus a
// hashing manifest. It does not tokenize, encode, sample noise, cast,
// quantize, or otherwise synthesize model inputs — payload bytes are copied
// verbatim, exactly like the Python tool it replaces.
//
// The manifest is emitted in the same shape as Python's
// json.dumps(document, indent=2, sort_keys=True) so the two tools can be
// diffed byte-for-byte (modulo the differing --output-dir/--manifest paths
// each run was pointed at).

#include "dif/frontend/h3_audio_vae.hpp"
#include "dif/weights/bundle.hpp"
#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"
#include "dif/weights/safetensors.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Arguments {
  fs::path conditioning;
  fs::path initial_state;
  fs::path output_dir;
  fs::path manifest;
  fs::path prompt;
  fs::path tokenizer;
  fs::path tokenizer_config;
  fs::path encoder_index;
  fs::path checkpoint_index;
  fs::path source_encoder;
  long long width = -1;
  long long height = -1;
  long long frames = -1;
  long long fps = -1;
  long long steps = -1;
  long long seed = -1;
};

[[noreturn]] void usage_error(const std::string &message) {
  std::cerr << "difimport: " << message << "\n"
            << "usage: difimport --conditioning FILE.safetensors "
               "--initial-state FILE.safetensors --output-dir DIR "
               "--manifest FILE.json --prompt FILE --tokenizer FILE "
               "--tokenizer-config FILE --encoder-index FILE "
               "--checkpoint-index FILE --source-encoder FILE "
               "--width N --height N --frames N --fps N --steps N --seed N\n";
  std::exit(2);
}

long long parse_integer(const std::string &text, const char *name) {
  std::size_t consumed = 0;
  long long value = 0;
  try {
    value = std::stoll(text, &consumed, 10);
  } catch (const std::exception &) {
    usage_error(std::string("invalid integer for ") + name + ": " + text);
  }
  if (consumed != text.size())
    usage_error(std::string("invalid integer for ") + name + ": " + text);
  return value;
}

std::string dtype_label(dif::ir::DType dtype) {
  switch (dtype) {
  case dif::ir::DType::F32:
    return "F32";
  case dif::ir::DType::BF16:
    return "BF16";
  case dif::ir::DType::F16:
    return "F16";
  case dif::ir::DType::I8:
    return "I8";
  case dif::ir::DType::I32:
    return "I32";
  }
  dif::fail("unsupported tensor dtype");
}

std::string resolved(const fs::path &path) {
  // Mirrors Python Path.resolve(): absolute, symlink-resolved, normalized.
  return fs::weakly_canonical(fs::absolute(path)).string();
}

std::string file_sha256_hex(const fs::path &path) {
  return dif::hex_digest(dif::sha256_file(path));
}

std::string json_escaped(const std::string &text) {
  std::string out;
  out.reserve(text.size() + 2);
  out.push_back('"');
  for (const char c : text) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (static_cast<unsigned char>(c) < 0x20) {
        char buffer[8];
        std::snprintf(buffer, sizeof buffer, "\\u%04x",
                      static_cast<unsigned>(static_cast<unsigned char>(c)));
        out += buffer;
      } else {
        out.push_back(c);
      }
    }
  }
  out.push_back('"');
  return out;
}

struct WrittenTensor {
  std::string path;
  std::string dtype;
  std::vector<std::uint64_t> shape;
  std::uint64_t payload_bytes{};
  std::string payload_sha256;
  std::string file_sha256;
};

WrittenTensor write_diftensor(const fs::path &path,
                              const dif::runtime::Tensor &source,
                              bool squeeze_batch) {
  std::vector<std::uint64_t> dims = source.dims;
  if (squeeze_batch) {
    if (dims.size() < 2U || dims.front() != 1U)
      dif::fail("cannot squeeze non-unit batch shape");
    dims.erase(dims.begin());
  }
  dif::runtime::Tensor output{source.dtype, dims, {}};
  output.bytes.assign(source.data(), source.data() + source.byte_size());
  dif::runtime::write_tensor(output, path);

  WrittenTensor written;
  written.path = resolved(path);
  written.dtype = dtype_label(source.dtype);
  written.shape = dims;
  written.payload_bytes = source.byte_size();
  written.payload_sha256 = dif::hex_digest(
      dif::sha256({source.data(), static_cast<std::size_t>(source.byte_size())}));
  written.file_sha256 = file_sha256_hex(path);
  return written;
}

void emit_tensor_entry(std::ostringstream &out, const char *name,
                       const WrittenTensor &tensor, bool last) {
  out << "    " << json_escaped(name) << ": {\n";
  out << "      \"dtype\": " << json_escaped(tensor.dtype) << ",\n";
  out << "      \"file_sha256\": " << json_escaped(tensor.file_sha256)
      << ",\n";
  out << "      \"path\": " << json_escaped(tensor.path) << ",\n";
  out << "      \"payload_bytes\": " << tensor.payload_bytes << ",\n";
  out << "      \"payload_sha256\": " << json_escaped(tensor.payload_sha256)
      << ",\n";
  out << "      \"shape\": [\n";
  for (std::size_t i = 0; i < tensor.shape.size(); ++i)
    out << "        " << tensor.shape[i]
        << (i + 1U < tensor.shape.size() ? ",\n" : "\n");
  out << "      ]\n";
  out << "    }" << (last ? "\n" : ",\n");
}

} // namespace


// ── BigVGAN audio decoder weight-norm fold (chunk 3 of the audio decode
// plan). Every decoder conv except dec_in_proj ships as weight_g/weight_v;
// the effective weight is g * v / ||v|| with the norm over every dimension
// but dim 0 (for ConvTranspose ups.* that dim is the INPUT channel — the
// layout trap documented in the plan). The sum of squares accumulates in
// FLOAT64 (sequential F32 measured 2.5-3.4e-6 relative off on the widest
// folds; audio_decoder.mojo:56-88), then scale = g / float(sqrt) and the
// per-element multiply stay F32 — the exact arithmetic the accepted Serenity
// decode ran. The alias-free UPSAMPLE filters additionally absorb their
// x2 ratio (exact in F32: exponent bump); downsample filters are pure
// low-pass with no ratio term and pass through unchanged. Encoder-side
// tensors are dropped: this feeds the decoder-only program.
void fold_audio_weight_norm(const fs::path &source_path,
                            const fs::path &output_path) {
  const auto source = dif::weights::read_safetensors(source_path);
  std::uint64_t input_census = 0, folded = 0, scaled = 0, passthrough = 0;

  struct OutputTensor {
    std::string name;
    dif::ir::DType dtype{};
    std::vector<std::uint64_t> dims;
    std::vector<std::uint8_t> bytes;
  };
  std::vector<OutputTensor> outputs;

  const auto decoder_side = [](const std::string &name) {
    return name.rfind("dec_in_proj.", 0U) == 0U ||
           name.rfind("decoder.", 0U) == 0U;
  };

  for (const auto &[name, entry] : source.tensors) {
    if (!decoder_side(name))
      continue;
    ++input_census;
    if (entry.dtype != dif::ir::DType::F32)
      dif::fail("audio decoder tensor is not F32: " + name);
    if (name.size() > 9U &&
        name.compare(name.size() - 9U, 9U, ".weight_g") == 0)
      continue;  // consumed with its sibling weight_v
    const auto mapped = dif::weights::map_safetensor(source, name);
    OutputTensor output;
    output.dtype = mapped.dtype;
    output.dims = mapped.dims;
    if (name.size() > 9U &&
        name.compare(name.size() - 9U, 9U, ".weight_v") == 0) {
      const auto base = name.substr(0U, name.size() - 9U);
      const auto *g_entry = source.find(base + ".weight_g");
      if (!g_entry)
        dif::fail("weight_v without weight_g: " + name);
      const auto g = dif::weights::map_safetensor(source, base + ".weight_g");
      const auto channels = mapped.dims.at(0);
      if (g.element_count() != channels)
        dif::fail("weight_g does not match weight_v dim 0: " + name);
      const auto per_channel = mapped.element_count() / channels;
      const auto *v_values = reinterpret_cast<const float *>(mapped.data());
      const auto *g_values = reinterpret_cast<const float *>(g.data());
      output.name = base + ".weight";
      output.bytes.resize(mapped.byte_size());
      auto *w_values = reinterpret_cast<float *>(output.bytes.data());
      for (std::uint64_t channel = 0; channel < channels; ++channel) {
        double sum_squares = 0.0;
        const auto *slice = v_values + channel * per_channel;
        for (std::uint64_t i = 0; i < per_channel; ++i) {
          const auto value = static_cast<double>(slice[i]);
          sum_squares += value * value;
        }
        const float scale =
            g_values[channel] / static_cast<float>(std::sqrt(sum_squares));
        for (std::uint64_t i = 0; i < per_channel; ++i)
          w_values[channel * per_channel + i] = slice[i] * scale;
      }
      ++folded;
    } else {
      output.name = name;
      output.bytes.assign(mapped.data(), mapped.data() + mapped.byte_size());
      const auto ends_with = [&](std::string_view suffix) {
        return name.size() > suffix.size() &&
               name.compare(name.size() - suffix.size(), suffix.size(),
                            suffix) == 0;
      };
      const auto is_upsample_filter = ends_with(".upsample.filter");
      const auto is_downsample_filter =
          ends_with(".downsample.lowpass.filter");
      if (is_upsample_filter || is_downsample_filter) {
        // The checkpoint ships ONE [1,1,12] Kaiser filter per resampler,
        // shared across channels (torch expands at runtime). DiffIR has no
        // broadcast, so the importer materializes the depthwise weight
        // [C,1,K]; C comes from the sibling SnakeBeta act.alpha vector of
        // the same activation module. The UPSAMPLE filter also absorbs its
        // exact x2 ratio (transposed-conv x ratio in the reference).
        const auto module_end = name.rfind(
            is_upsample_filter ? ".upsample.filter"
                               : ".downsample.lowpass.filter");
        const auto alpha_name = name.substr(0U, module_end) + ".act.alpha";
        const auto *alpha_entry = source.find(alpha_name);
        if (!alpha_entry || alpha_entry->dims.size() != 1U)
          dif::fail("resampler filter has no sibling act.alpha: " + name);
        const auto channels = alpha_entry->dims[0];
        const auto kernel = mapped.dims.back();
        if (mapped.element_count() != kernel)
          dif::fail("resampler filter is not a shared [1,1,K] tap: " + name);
        const auto *taps = reinterpret_cast<const float *>(mapped.data());
        output.dims = {channels, 1U, kernel};
        output.bytes.resize(channels * kernel * 4U);
        auto *values = reinterpret_cast<float *>(output.bytes.data());
        const float scale = is_upsample_filter ? 2.0F : 1.0F;
        for (std::uint64_t channel = 0; channel < channels; ++channel)
          for (std::uint64_t k = 0; k < kernel; ++k)
            values[channel * kernel + k] = taps[k] * scale;
        ++scaled;
      } else {
        ++passthrough;
      }
    }
    outputs.push_back(std::move(output));
  }
  if (outputs.empty())
    dif::fail("no decoder-side tensors found in " + source_path.string());

  std::vector<dif::weights::SafeTensorWriteSpec> specs;
  specs.reserve(outputs.size());
  for (const auto &output : outputs)
    specs.push_back({output.name, output.dtype, output.dims});
  dif::weights::SafeTensorWriter writer(output_path, std::move(specs));
  for (const auto &output : outputs)
    writer.append(output.name, {output.bytes.data(), output.bytes.size()});
  const auto written = writer.finish();
  std::cout << "AUDIO_FOLD PASS input_census=" << input_census
            << " output_census=" << written.tensors.size()
            << " folded=" << folded << " filters_expanded=" << scaled
            << " passthrough=" << passthrough << " file_sha256="
            << dif::hex_digest(dif::sha256_file(output_path)) << "\n";
}


// ── BigVGAN decoder program + bundle emission (chunk 4/5 of the audio
// decode plan). make-audio-program writes the DiffIR program for (B, T) at
// the given stage truncation and prints the input/output tensor ids the
// stage-parity gate binds. make-audio-bundle materializes ONE derived shard
// (folded checkpoint tensors by name + the builder's generated denorm /
// block-average constants) and seals a .difbind against the program
// fingerprint — the exact materialize pattern difweights uses for the H3
// video VAE. unpack-audio-rows converts the recorded [2T, C] channel-major
// state rows into the [2, C, T] latent tensor
// (serenitymojo rearrange.mojo minimax_h3_unpack_audio).
void make_audio_program(const fs::path &output_path, std::uint64_t batch,
                        std::uint64_t frames, std::uint64_t stages) {
  const auto build =
      dif::frontend::build_audio_bigvgan_program(batch, frames, stages);
  dif::ir::verify(build.program);
  dif::ir::write_file(build.program, output_path);
  std::cout << "AUDIO_PROGRAM path=" << output_path.string()
            << " fingerprint="
            << dif::hex_digest(dif::ir::fingerprint(build.program))
            << " input_id=" << build.latent_input_id
            << " output_id=" << build.waveform_output_id
            << " ops=" << build.program.operations.size()
            << " conv1d=" << build.conv1d_operations
            << " snake_beta=" << build.snake_beta_operations << "\n";
}

void make_audio_bundle(const fs::path &folded_path,
                       const fs::path &program_path,
                       const fs::path &derived_path,
                       const fs::path &bundle_path, std::uint64_t batch,
                       std::uint64_t frames, std::uint64_t stages) {
  if (fs::exists(derived_path) || fs::exists(bundle_path))
    dif::fail("refusing to overwrite audio derived shard or bundle");
  const auto build =
      dif::frontend::build_audio_bigvgan_program(batch, frames, stages);
  const auto program = dif::ir::read_file(program_path);
  if (dif::ir::fingerprint(program) != dif::ir::fingerprint(build.program))
    dif::fail("audio program does not match the requested geometry");
  const auto folded = dif::weights::read_safetensors(folded_path);

  std::vector<dif::weights::SafeTensorWriteSpec> specs;
  specs.reserve(build.bindings.size());
  for (const auto &binding : build.bindings) {
    const auto *description = build.program.tensor(binding.tensor_id);
    if (!description ||
        !description->has_role(dif::ir::TensorRole::Constant))
      dif::fail("audio binding does not target a constant");
    specs.push_back({binding.name, description->dtype, description->dims});
  }
  dif::weights::SafeTensorWriter writer(derived_path, std::move(specs));
  for (const auto &binding : build.bindings) {
    const auto *description = build.program.tensor(binding.tensor_id);
    if (binding.source_name.empty()) {
      const auto found = build.generated_constants.find(binding.tensor_id);
      if (found == build.generated_constants.end())
        dif::fail("audio generated binding has no payload");
      writer.append(binding.name,
                    {found->second.data(), found->second.byte_size()});
      continue;
    }
    const auto *entry = folded.find(binding.source_name);
    if (!entry)
      dif::fail("folded audio shard is missing " + binding.source_name);
    auto tensor = dif::weights::map_safetensor(folded, binding.source_name);
    if (tensor.dtype != description->dtype ||
        tensor.dims != description->dims)
      dif::fail("folded audio tensor geometry disagrees: " +
                binding.source_name);
    writer.append(binding.name, {tensor.data(), tensor.byte_size()});
    tensor.discard_mapped_pages();
  }
  const auto derived = writer.finish();

  dif::weights::WeightBundle bundle;
  bundle.program_fingerprint = dif::ir::fingerprint(build.program);
  bundle.index_fingerprint = dif::sha256_file(folded_path);
  const auto absolute = fs::absolute(derived_path).lexically_normal();
  bundle.shards.push_back(
      {absolute, derived.file_size, dif::sha256_file(absolute)});
  for (const auto &binding : build.bindings) {
    const auto *entry = derived.find(binding.name);
    if (!entry)
      dif::fail("audio derived shard lost " + binding.name);
    bundle.bindings.push_back(
        {binding.tensor_id, 0U, binding.name, entry->dtype, entry->dims,
         entry->file_offset, entry->byte_count});
  }
  dif::weights::verify_weight_bundle(bundle, build.program, false);
  dif::weights::write_weight_bundle(bundle, bundle_path);
  std::cout << "AUDIO_BUNDLE path=" << bundle_path.string()
            << " derived=" << derived_path.string() << " program="
            << dif::hex_digest(bundle.program_fingerprint)
            << " bindings=" << bundle.bindings.size() << "\n";
}

void unpack_audio_rows(const fs::path &rows_path, const fs::path &out_path) {
  const auto rows = dif::runtime::read_tensor(rows_path);
  if (rows.dtype != dif::ir::DType::F32 || rows.dims.size() != 2U ||
      rows.dims[0] % 2U != 0U)
    dif::fail("audio rows must be F32 [2T, C]");
  const auto frames = rows.dims[0] / 2U;
  const auto channels = rows.dims[1];
  dif::runtime::Tensor latent{dif::ir::DType::F32,
                              {2U, channels, frames}, {}};
  latent.bytes.resize(static_cast<std::size_t>(latent.element_count()) *
                      sizeof(float));
  const auto *source = reinterpret_cast<const float *>(rows.data());
  auto *destination = reinterpret_cast<float *>(latent.bytes.data());
  for (std::uint64_t stereo = 0; stereo < 2U; ++stereo)
    for (std::uint64_t t = 0; t < frames; ++t)
      for (std::uint64_t c = 0; c < channels; ++c)
        destination[(stereo * channels + c) * frames + t] =
            source[(stereo * frames + t) * channels + c];
  dif::runtime::write_tensor(latent, out_path);
  std::cout << "AUDIO_UNPACK rows=" << rows.dims[0] << " channels="
            << channels << " frames=" << frames << "\n";
}

int main(int argc, char **argv) {
  try {
    if (argc >= 2 && std::string(argv[1]) == "fold-audio-weight-norm") {
      if (argc != 4)
        usage_error("fold-audio-weight-norm expects SRC.safetensors "
                    "OUT.safetensors");
      fold_audio_weight_norm(argv[2], argv[3]);
      return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "make-audio-program") {
      if (argc != 6)
        usage_error("make-audio-program expects OUT.difir B T STAGES");
      make_audio_program(argv[2], std::stoull(argv[3]), std::stoull(argv[4]),
                         std::stoull(argv[5]));
      return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "make-audio-bundle") {
      if (argc != 9)
        usage_error("make-audio-bundle expects FOLDED.safetensors "
                    "PROGRAM.difir OUT_DERIVED.safetensors OUT.difbind B T "
                    "STAGES");
      make_audio_bundle(argv[2], argv[3], argv[4], argv[5],
                        std::stoull(argv[6]), std::stoull(argv[7]),
                        std::stoull(argv[8]));
      return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "unpack-audio-rows") {
      if (argc != 4)
        usage_error("unpack-audio-rows expects ROWS.diftensor OUT.diftensor");
      unpack_audio_rows(argv[2], argv[3]);
      return 0;
    }
    Arguments arguments;
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      const auto value = [&](const char *name) -> std::string {
        if (i + 1 >= argc)
          usage_error(std::string("missing value for ") + name);
        return argv[++i];
      };
      if (option == "--conditioning")
        arguments.conditioning = value("--conditioning");
      else if (option == "--initial-state")
        arguments.initial_state = value("--initial-state");
      else if (option == "--output-dir")
        arguments.output_dir = value("--output-dir");
      else if (option == "--manifest")
        arguments.manifest = value("--manifest");
      else if (option == "--prompt")
        arguments.prompt = value("--prompt");
      else if (option == "--tokenizer")
        arguments.tokenizer = value("--tokenizer");
      else if (option == "--tokenizer-config")
        arguments.tokenizer_config = value("--tokenizer-config");
      else if (option == "--encoder-index")
        arguments.encoder_index = value("--encoder-index");
      else if (option == "--checkpoint-index")
        arguments.checkpoint_index = value("--checkpoint-index");
      else if (option == "--source-encoder")
        arguments.source_encoder = value("--source-encoder");
      else if (option == "--width")
        arguments.width = parse_integer(value("--width"), "--width");
      else if (option == "--height")
        arguments.height = parse_integer(value("--height"), "--height");
      else if (option == "--frames")
        arguments.frames = parse_integer(value("--frames"), "--frames");
      else if (option == "--fps")
        arguments.fps = parse_integer(value("--fps"), "--fps");
      else if (option == "--steps")
        arguments.steps = parse_integer(value("--steps"), "--steps");
      else if (option == "--seed")
        arguments.seed = parse_integer(value("--seed"), "--seed");
      else
        usage_error("unknown option " + option);
    }
    for (const auto *required :
         {&arguments.conditioning, &arguments.initial_state,
          &arguments.output_dir, &arguments.manifest, &arguments.prompt,
          &arguments.tokenizer, &arguments.tokenizer_config,
          &arguments.encoder_index, &arguments.checkpoint_index,
          &arguments.source_encoder})
      if (required->empty())
        usage_error("a required path argument is missing");
    for (const auto required :
         {arguments.width, arguments.height, arguments.frames, arguments.fps,
          arguments.steps, arguments.seed})
      if (required < 0)
        usage_error("a required integer argument is missing");

    if (arguments.frames < 5 || arguments.frames % 17 != 5)
      usage_error("H3 frames must be positive and satisfy frames % 17 == 5");
    if (arguments.width % 32 || arguments.height % 32)
      usage_error("H3 width and height must be divisible by 32");

    const auto conditioning =
        dif::weights::read_safetensors(arguments.conditioning);
    const auto initial = dif::weights::read_safetensors(arguments.initial_state);
    if (!conditioning.find("text_conditioning"))
      dif::fail("conditioning SafeTensors is missing text_conditioning");
    if (!initial.find("video_state_rows") || !initial.find("audio_state_rows"))
      dif::fail("initial-state SafeTensors is missing state rows");
    const auto text =
        dif::weights::map_safetensor(conditioning, "text_conditioning");
    const auto video = dif::weights::map_safetensor(initial, "video_state_rows");
    const auto audio = dif::weights::map_safetensor(initial, "audio_state_rows");

    std::vector<std::uint64_t> text_shape = text.dims;
    const bool squeeze_text = text_shape.size() == 3U && text_shape.front() == 1U;
    if (squeeze_text)
      text_shape.erase(text_shape.begin());
    const auto text_tokens = text_shape.at(0);
    const auto latent_t =
        static_cast<std::uint64_t>((arguments.frames - 5) / 17) * 5U + 2U;
    const auto latent_h = static_cast<std::uint64_t>(arguments.height) / 16U;
    const auto latent_w = static_cast<std::uint64_t>(arguments.width) / 16U;
    const auto video_tokens = latent_t * (latent_h / 2U) * (latent_w / 2U);
    const auto audio_latents = static_cast<std::uint64_t>(
        (arguments.frames * 40 + arguments.fps / 2) / arguments.fps);

    const std::vector<std::uint64_t> expected_text{text_tokens, 5120U};
    const std::vector<std::uint64_t> expected_video{video_tokens, 96U};
    const std::vector<std::uint64_t> expected_audio{2U * audio_latents, 32U};
    if (text_shape != expected_text || video.dims != expected_video ||
        audio.dims != expected_audio)
      dif::fail("Serenity input geometry mismatch");
    if (text.dtype != dif::ir::DType::BF16 ||
        video.dtype != dif::ir::DType::F32 ||
        audio.dtype != dif::ir::DType::F32)
      dif::fail("Serenity input dtype mismatch: text=" + dtype_label(text.dtype) +
                " video=" + dtype_label(video.dtype) +
                " audio=" + dtype_label(audio.dtype));

    if (fs::exists(arguments.output_dir))
      dif::fail("output directory already exists: " +
                arguments.output_dir.string());
    fs::create_directories(arguments.output_dir);

    const auto text_written = write_diftensor(
        arguments.output_dir / "text_conditioning.diftensor", text,
        squeeze_text);
    const auto video_written = write_diftensor(
        arguments.output_dir / "video_state_rows.diftensor", video, false);
    const auto audio_written = write_diftensor(
        arguments.output_dir / "audio_state_rows.diftensor", audio, false);

    const auto sequence_tokens =
        video_tokens + 2U * audio_latents + text_tokens;

    std::ostringstream manifest;
    manifest << "{\n";
    manifest << "  \"checkpoint\": {\n";
    manifest << "    \"index\": "
             << json_escaped(resolved(arguments.checkpoint_index)) << ",\n";
    manifest << "    \"index_sha256\": "
             << json_escaped(file_sha256_hex(arguments.checkpoint_index))
             << "\n  },\n";
    manifest << "  \"conditioning\": {\n";
    manifest << "    \"encoder_index\": "
             << json_escaped(resolved(arguments.encoder_index)) << ",\n";
    manifest << "    \"encoder_index_sha256\": "
             << json_escaped(file_sha256_hex(arguments.encoder_index)) << ",\n";
    manifest << "    \"hidden_state_layer\": 50,\n";
    manifest << "    \"source_encoder_path\": "
             << json_escaped(resolved(arguments.source_encoder)) << ",\n";
    manifest << "    \"source_file\": "
             << json_escaped(resolved(arguments.conditioning)) << ",\n";
    manifest << "    \"source_file_sha256\": "
             << json_escaped(file_sha256_hex(arguments.conditioning)) << ",\n";
    manifest << "    \"storage\": \"bf16\"\n  },\n";
    manifest << "  \"geometry\": {\n";
    manifest << "    \"audio_latents\": " << audio_latents << ",\n";
    manifest << "    \"audio_tokens\": " << 2U * audio_latents << ",\n";
    manifest << "    \"fps\": " << arguments.fps << ",\n";
    manifest << "    \"frames\": " << arguments.frames << ",\n";
    manifest << "    \"height\": " << arguments.height << ",\n";
    manifest << "    \"latent_h\": " << latent_h << ",\n";
    manifest << "    \"latent_t\": " << latent_t << ",\n";
    manifest << "    \"latent_w\": " << latent_w << ",\n";
    manifest << "    \"sequence_tokens\": " << sequence_tokens << ",\n";
    manifest << "    \"text_tokens\": " << text_tokens << ",\n";
    manifest << "    \"video_tokens\": " << video_tokens << ",\n";
    manifest << "    \"width\": " << arguments.width << "\n  },\n";
    manifest
        << "  \"negative_prompt_policy\": \"none_single_conditional_forward\",\n";
    manifest << "  \"outputs\": {\n";
    emit_tensor_entry(manifest, "audio_state_rows", audio_written, false);
    emit_tensor_entry(manifest, "text_conditioning", text_written, false);
    emit_tensor_entry(manifest, "video_state_rows", video_written, true);
    manifest << "  },\n";
    manifest << "  \"prompt\": {\n";
    manifest << "    \"bytes\": " << fs::file_size(arguments.prompt) << ",\n";
    manifest << "    \"path\": " << json_escaped(resolved(arguments.prompt))
             << ",\n";
    manifest << "    \"sha256\": "
             << json_escaped(file_sha256_hex(arguments.prompt)) << "\n  },\n";
    manifest << "  \"schedule\": {\n";
    manifest << "    \"audio_shift\": 3.0,\n";
    manifest << "    \"model_evaluations\": " << arguments.steps - 1 << ",\n";
    manifest << "    \"points\": " << arguments.steps << ",\n";
    manifest << "    \"sampler\": \"released_h3_data_ward_euler\",\n";
    manifest << "    \"video_shift\": 12.0\n  },\n";
    manifest << "  \"schema\": \"dif.h3.serenity_inputs.v1\",\n";
    manifest << "  \"start\": {\n";
    manifest << "    \"audio_seed\": " << arguments.seed + 1 << ",\n";
    manifest << "    \"source_file\": "
             << json_escaped(resolved(arguments.initial_state)) << ",\n";
    manifest << "    \"source_file_sha256\": "
             << json_escaped(file_sha256_hex(arguments.initial_state)) << ",\n";
    manifest << "    \"video_seed\": " << arguments.seed << "\n  },\n";
    manifest << "  \"task\": \"t2va\",\n";
    manifest << "  \"tokenizer\": {\n";
    manifest << "    \"add_special_tokens\": false,\n";
    manifest << "    \"chat_template\": false,\n";
    manifest << "    \"config_path\": "
             << json_escaped(resolved(arguments.tokenizer_config)) << ",\n";
    manifest << "    \"config_sha256\": "
             << json_escaped(file_sha256_hex(arguments.tokenizer_config))
             << ",\n";
    manifest << "    \"path\": " << json_escaped(resolved(arguments.tokenizer))
             << ",\n";
    manifest << "    \"sha256\": "
             << json_escaped(file_sha256_hex(arguments.tokenizer)) << ",\n";
    manifest << "    \"text_tokens\": " << text_tokens << "\n  }\n";
    manifest << "}\n";

    if (arguments.manifest.has_parent_path())
      fs::create_directories(arguments.manifest.parent_path());
    std::ofstream manifest_file(arguments.manifest,
                                std::ios::binary | std::ios::trunc);
    if (!manifest_file)
      dif::fail("cannot create manifest: " + arguments.manifest.string());
    const auto manifest_text = manifest.str();
    manifest_file.write(manifest_text.data(),
                        static_cast<std::streamsize>(manifest_text.size()));
    if (!manifest_file)
      dif::fail("cannot write manifest: " + arguments.manifest.string());
    manifest_file.close();

    std::cout << "H3_SERENITY_INPUT_IMPORT PASS text=" << text_tokens
              << " video=" << video_tokens << " audio=" << 2U * audio_latents
              << " sequence=" << sequence_tokens << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difimport: " << error.what() << "\n";
    return 1;
  }
}
