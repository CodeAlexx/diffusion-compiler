// SDXL base 1.0 prompt-to-PNG through the compiler.
//
// One process: the native CLIP tokenizer, the two text towers, the UNet with
// classifier-free guidance batched two rows at a time, and the VAE decoder,
// all as DiffIR programs over shared opcodes. The schedule is the reference
// sampler's discrete DDPM one; the noise is the reference's CPU generator.
//
// usage: difsdxlsample --checkpoint FILE --tokenizer-dir DIR --prompt TEXT
//            --output image.png [--negative TEXT] [--seed N] [--steps N]
//            [--cfg F] [--width N] [--height N] [--report FILE]
//            [--cache-dir DIR] [--unet-dtype f16|bf16] [--vae-dtype bf16|f32]

#include "dif/frontend/sdxl_clip.hpp"
#include "dif/frontend/sdxl_unet.hpp"
#include "dif/frontend/sdxl_vae.hpp"
#include "dif/ir/ir.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/sampling/discrete_schedule.hpp"
#include "dif/support/error.hpp"
#include "dif/support/png.hpp"
#include "dif/support/torch_cpu_rng.hpp"
#include "dif/text/clip_bpe_tokenizer.hpp"
#include "dif/weights/safetensors.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using dif::ir::DType;
using dif::runtime::Tensor;

double ms_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct Arguments {
  fs::path checkpoint;
  fs::path tokenizer_directory;
  fs::path output;
  fs::path report;
  fs::path cache_directory;
  std::string prompt;
  std::string negative;
  std::uint64_t seed{20260901};
  std::uint32_t steps{25};
  double cfg{5.0};
  std::uint64_t width{1024};
  std::uint64_t height{1024};
  DType unet_dtype{DType::F16};
  DType vae_dtype{DType::BF16};
  // The checkpoint stores the text towers in F16 and the reference casts
  // them to F32 per operation. Running them in F16 skips a multi-gigabyte
  // host conversion and matches the F32 oracle to cosine 0.999996.
  DType clip_dtype{DType::F16};
  bool report_steps{};
  // Generate this many images from one preparation, the way a long-lived
  // process would: the load and the plan building are paid once.
  std::uint32_t images{1};
};

DType parse_dtype(const std::string &name) {
  if (name == "f16")
    return DType::F16;
  if (name == "bf16")
    return DType::BF16;
  if (name == "f32")
    return DType::F32;
  dif::fail("unknown dtype " + name);
}

std::string read_file(const fs::path &path) {
  std::ifstream file(path);
  if (!file)
    dif::fail("cannot read " + path.string());
  std::string text((std::istreambuf_iterator<char>(file)),
                   std::istreambuf_iterator<char>());
  while (!text.empty() && (text.back() == '\n' || text.back() == '\r'))
    text.pop_back();
  return text;
}

Arguments parse(int argc, char **argv) {
  Arguments arguments;
  for (int index = 1; index < argc; ++index) {
    const std::string flag = argv[index];
    auto value = [&]() -> std::string {
      if (index + 1 >= argc)
        dif::fail("missing value for " + flag);
      return argv[++index];
    };
    if (flag == "--checkpoint")
      arguments.checkpoint = value();
    else if (flag == "--tokenizer-dir")
      arguments.tokenizer_directory = value();
    else if (flag == "--output")
      arguments.output = value();
    else if (flag == "--report")
      arguments.report = value();
    else if (flag == "--cache-dir")
      arguments.cache_directory = value();
    else if (flag == "--prompt")
      arguments.prompt = value();
    else if (flag == "--prompt-file")
      arguments.prompt = read_file(value());
    else if (flag == "--negative")
      arguments.negative = value();
    else if (flag == "--seed")
      arguments.seed = std::stoull(value());
    else if (flag == "--steps")
      arguments.steps = static_cast<std::uint32_t>(std::stoul(value()));
    else if (flag == "--cfg")
      arguments.cfg = std::stod(value());
    else if (flag == "--width")
      arguments.width = std::stoull(value());
    else if (flag == "--height")
      arguments.height = std::stoull(value());
    else if (flag == "--unet-dtype")
      arguments.unet_dtype = parse_dtype(value());
    else if (flag == "--vae-dtype")
      arguments.vae_dtype = parse_dtype(value());
    else if (flag == "--clip-dtype")
      arguments.clip_dtype = parse_dtype(value());
    else if (flag == "--report-steps")
      arguments.report_steps = true;
    else if (flag == "--images")
      arguments.images = static_cast<std::uint32_t>(std::stoul(value()));
    else
      dif::fail("unknown argument " + flag);
  }
  if (arguments.checkpoint.empty() || arguments.output.empty() ||
      arguments.tokenizer_directory.empty())
    dif::fail("--checkpoint, --tokenizer-dir and --output are required");
  if (arguments.width % 8U != 0U || arguments.height % 8U != 0U ||
      arguments.width == 0U || arguments.height == 0U)
    dif::fail("width and height must be positive multiples of eight");
  if (arguments.steps == 0U)
    dif::fail("--steps must be positive");
  return arguments;
}

Tensor make_tensor(DType dtype, std::vector<std::uint64_t> dims) {
  Tensor tensor{dtype, std::move(dims), {}};
  tensor.bytes.resize(static_cast<std::size_t>(tensor.element_count()) *
                      dif::ir::dtype_size(dtype));
  tensor.validate();
  return tensor;
}

Tensor i32_tensor(std::vector<std::uint64_t> dims,
                  const std::vector<std::int32_t> &values) {
  Tensor tensor{DType::I32, std::move(dims), {}};
  tensor.bytes.resize(values.size() * sizeof(std::int32_t));
  std::memcpy(tensor.bytes.data(), values.data(), tensor.bytes.size());
  tensor.validate();
  return tensor;
}

// Rows [index * rows, (index + 1) * rows) of a fused [3 * rows, ...] tensor.
Tensor fused_rows(const Tensor &source, std::uint64_t index,
                  std::uint64_t rows) {
  if (source.dims.empty() || source.dims[0] != 3U * rows)
    dif::fail("fused q/k/v tensor does not hold three row blocks");
  auto dims = source.dims;
  dims[0] = rows;
  const auto row_bytes =
      source.byte_size() / static_cast<std::size_t>(source.dims[0]);
  Tensor result{source.dtype, dims, {}};
  result.bytes.assign(source.data() + index * rows * row_bytes,
                      source.data() + (index + 1U) * rows * row_bytes);
  result.validate();
  return result;
}

Tensor transposed(const Tensor &source) {
  if (source.dims.size() != 2U)
    dif::fail("transpose expects a rank-2 tensor");
  const auto rows = source.dims[0];
  const auto columns = source.dims[1];
  Tensor result{source.dtype, {columns, rows}, {}};
  result.bytes.resize(source.byte_size());
  result.validate();
  for (std::uint64_t row = 0U; row < rows; ++row)
    for (std::uint64_t column = 0U; column < columns; ++column)
      dif::runtime::store_float(
          result, column * rows + row,
          dif::runtime::load_float(source, row * columns + column));
  return result;
}

Tensor as_dtype(Tensor tensor, DType dtype) {
  if (tensor.dtype == dtype)
    return tensor;
  return dif::runtime::convert_float_tensor(tensor, dtype);
}

void check_binding(const dif::ir::Program &program, std::uint32_t id,
                   const Tensor &tensor, const std::string &name) {
  const auto *description = program.tensor(id);
  if (!description || tensor.dtype != description->dtype ||
      tensor.dims != description->dims)
    dif::fail("checkpoint tensor disagrees with the program: " + name);
}

// One prepared text tower plus everything needed to run it per chunk.
struct Tower {
  dif::frontend::ClipTextTowerBuild build;
  // One binding map for the life of the tower: the weights are gigabytes,
  // so the token ids are overwritten in place rather than copied per chunk.
  dif::runtime::TensorMap bindings;
  std::unique_ptr<dif::runtime::PreparedExecution> prepared;
  std::uint64_t hidden_size{};
};

Tower prepare_tower(const dif::weights::SafeTensorFile &checkpoint,
                    dif::frontend::ClipTextTowerConfig config,
                    dif::runtime::Executor &backend,
                    const dif::runtime::RunOptions &options) {
  Tower tower;
  tower.hidden_size = config.hidden_size;
  tower.build = dif::frontend::make_clip_text_tower(config);
  const auto hidden = config.hidden_size;
  for (const auto &weight : tower.build.weights) {
    auto source = dif::weights::map_safetensor(checkpoint, weight.source_name);
    using Transform = dif::frontend::ClipWeightTransform;
    switch (weight.transform) {
    case Transform::Direct:
      break;
    case Transform::FusedRowsQ:
      source = fused_rows(source, 0U, hidden);
      break;
    case Transform::FusedRowsK:
      source = fused_rows(source, 1U, hidden);
      break;
    case Transform::FusedRowsV:
      source = fused_rows(source, 2U, hidden);
      break;
    case Transform::Transpose:
      source = transposed(source);
      break;
    }
    auto tensor = as_dtype(std::move(source), config.dtype);
    check_binding(tower.build.program, weight.tensor, tensor,
                  weight.source_name);
    tower.bindings.emplace(weight.tensor, std::move(tensor));
  }
  tower.bindings.emplace(tower.build.token_ids_input,
                         make_tensor(DType::I32, {config.positions}));
  if (config.pooled_output)
    tower.bindings.emplace(tower.build.pooled_row_input,
                           make_tensor(DType::I32, {1U}));
  tower.prepared =
      backend.prepare(tower.build.program, tower.bindings, options);
  return tower;
}

// The encoded prompt: the concatenated per-chunk hidden states and, for the
// G tower, the pooled vector of the first chunk.
struct Encoding {
  std::vector<float> hidden;  // [chunks * positions, hidden_size]
  std::vector<float> pooled;  // [hidden_size], G only
  std::uint64_t rows{};
};

Encoding encode_tower(Tower &tower, const dif::text::ClipPromptTokens &tokens,
                      const std::vector<std::int32_t> &empty_chunk,
                      const dif::runtime::RunOptions &options,
                      std::uint64_t positions) {
  Encoding encoding;
  const auto chunks = tokens.ids.size() / positions;
  encoding.rows = chunks * positions;
  encoding.hidden.resize(encoding.rows * tower.hidden_size);

  auto run_chunk = [&](const std::int32_t *ids, std::uint64_t pooled_row) {
    tower.bindings.insert_or_assign(
        tower.build.token_ids_input,
        i32_tensor({positions},
                   std::vector<std::int32_t>(ids, ids + positions)));
    if (tower.build.pooled_output != 0U)
      tower.bindings.insert_or_assign(
          tower.build.pooled_row_input,
          i32_tensor({1U}, {static_cast<std::int32_t>(pooled_row)}));
    return tower.prepared->run(tower.bindings, options);
  };

  // A weighted prompt blends every row against the empty-prompt encoding,
  // exactly as the reference's encode_token_weights does.
  std::vector<float> empty_hidden;
  if (tokens.weighted()) {
    const auto result = run_chunk(empty_chunk.data(), 0U);
    const auto &value = result.outputs.at(tower.build.hidden_output);
    empty_hidden.resize(static_cast<std::size_t>(value.element_count()));
    for (std::size_t index = 0; index < empty_hidden.size(); ++index)
      empty_hidden[index] = dif::runtime::load_float(value, index);
  }

  for (std::uint64_t chunk = 0U; chunk < chunks; ++chunk) {
    const auto *ids = tokens.ids.data() + chunk * positions;
    // The pooled vector comes from the first chunk's end-of-text row.
    const auto pooled_row = tokens.valid_tokens - 1U;
    const auto result = run_chunk(ids, pooled_row);
    const auto &value = result.outputs.at(tower.build.hidden_output);
    for (std::uint64_t row = 0U; row < positions; ++row) {
      const auto weight = tokens.weights[chunk * positions + row];
      for (std::uint64_t column = 0U; column < tower.hidden_size; ++column) {
        const auto at = row * tower.hidden_size + column;
        float element = dif::runtime::load_float(value, at);
        if (!empty_hidden.empty() && weight != 1.0F)
          element = (element - empty_hidden[at]) * weight + empty_hidden[at];
        encoding.hidden[(chunk * positions + row) * tower.hidden_size + column] =
            element;
      }
    }
    if (chunk == 0U && tower.build.pooled_output != 0U) {
      const auto &pooled = result.outputs.at(tower.build.pooled_output);
      encoding.pooled.resize(static_cast<std::size_t>(pooled.element_count()));
      for (std::size_t index = 0; index < encoding.pooled.size(); ++index)
        encoding.pooled[index] = dif::runtime::load_float(pooled, index);
    }
  }
  return encoding;
}

// The reference's Timestep(256) embedder: cos then sin, max period 10000.
std::vector<float> size_embedding(double value, std::uint64_t dim) {
  const auto half = dim / 2U;
  std::vector<float> out(dim);
  for (std::uint64_t index = 0U; index < half; ++index) {
    const auto frequency =
        std::exp(-std::log(10000.0) * static_cast<double>(index) /
                 static_cast<double>(half));
    const auto argument = value * frequency;
    out[index] = static_cast<float>(std::cos(argument));
    out[half + index] = static_cast<float>(std::sin(argument));
  }
  return out;
}

std::vector<std::uint8_t> rgb8(const Tensor &pixels, std::uint64_t width,
                               std::uint64_t height) {
  std::vector<std::uint8_t> out(static_cast<std::size_t>(width * height * 3U));
  for (std::uint64_t channel = 0U; channel < 3U; ++channel)
    for (std::uint64_t y = 0U; y < height; ++y)
      for (std::uint64_t x = 0U; x < width; ++x) {
        const auto source = (channel * height + y) * width + x;
        // The reference's process_output ((x + 1) / 2 clamped) then its
        // 255 * value rounding down to a byte.
        const float value = dif::runtime::load_float(pixels, source);
        const float unit = std::clamp((value + 1.0F) * 0.5F, 0.0F, 1.0F);
        const float scaled = std::clamp(255.0F * unit, 0.0F, 255.0F);
        out[static_cast<std::size_t>((y * width + x) * 3U + channel)] =
            static_cast<std::uint8_t>(scaled);
      }
  return out;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto arguments = parse(argc, argv);
    const auto total_start = Clock::now();
    const auto latent_height = arguments.height / 8U;
    const auto latent_width = arguments.width / 8U;
    constexpr std::uint64_t kPositions = 77;
    constexpr double kLatentScale = 0.13025;

    dif::runtime::RunOptions options;
    options.warmups = 0U;
    options.iterations = 1U;
    options.minimum_free_bytes = 256ULL * 1024ULL * 1024ULL;
    options.cache_directory = arguments.cache_directory;

    auto started = Clock::now();
    const auto tokenizer = dif::text::ClipBpeTokenizer::load(
        arguments.tokenizer_directory / "vocab.json",
        arguments.tokenizer_directory / "merges.txt");
    const auto positive = dif::text::sdxl_prompt_tokens(tokenizer,
                                                        arguments.prompt);
    const auto negative = dif::text::sdxl_prompt_tokens(tokenizer,
                                                        arguments.negative);
    const auto tokenize_ms = ms_since(started);
    const auto positive_chunks = positive.l.ids.size() / kPositions;
    const auto negative_chunks = negative.l.ids.size() / kPositions;
    // The reference batches conditionings of different lengths by repeating
    // the shorter one up to the least common multiple, which leaves
    // attention's result unchanged (duplicated keys share the softmax mass).
    // It refuses a repeat factor above four; so does this.
    const auto chunks = std::lcm(positive_chunks, negative_chunks);
    if (chunks / std::min(positive_chunks, negative_chunks) > 4U)
      dif::fail("the prompt and the negative prompt differ by more than a "
                "four-fold repeat; shorten one of them");

    const auto checkpoint = dif::weights::read_safetensors(arguments.checkpoint);
    auto backend = dif::runtime::make_cuda_executor();

    // --- text -------------------------------------------------------------
    started = Clock::now();
    auto l_config = dif::frontend::sdxl_clip_l_config();
    auto g_config = dif::frontend::sdxl_clip_g_config();
    for (auto *config : {&l_config, &g_config}) {
      config->dtype = arguments.clip_dtype;
      config->capture_boundaries = false;
      if (config->dtype != DType::F32)
        config->attention_implementation = 2U;
    }
    auto l_tower = prepare_tower(checkpoint, l_config, *backend, options);
    auto g_tower = prepare_tower(checkpoint, g_config, *backend, options);
    const auto text_prepare_ms = ms_since(started);

    started = Clock::now();
    const auto l_empty = dif::text::clip_empty_chunk(tokenizer,
                                                     tokenizer.eos_id());
    const auto g_empty =
        dif::text::clip_empty_chunk(tokenizer, dif::text::kSdxlClipGPadToken);
    const auto l_positive =
        encode_tower(l_tower, positive.l, l_empty, options, kPositions);
    const auto g_positive =
        encode_tower(g_tower, positive.g, g_empty, options, kPositions);
    const auto l_negative =
        encode_tower(l_tower, negative.l, l_empty, options, kPositions);
    const auto g_negative =
        encode_tower(g_tower, negative.g, g_empty, options, kPositions);
    const auto text_ms = ms_since(started);

    const auto context_tokens = chunks * kPositions;
    const std::uint64_t context_width = 2048;

    // --- the denoiser -----------------------------------------------------
    started = Clock::now();
    dif::frontend::SdxlUnetConfig unet_config;
    unet_config.batch = 2U;
    unet_config.latent_height = latent_height;
    unet_config.latent_width = latent_width;
    unet_config.context_tokens = context_tokens;
    unet_config.dtype = arguments.unet_dtype;
    unet_config.capture_boundaries = false;
    auto unet = dif::frontend::make_sdxl_unet(unet_config);
    dif::runtime::TensorMap unet_bindings;
    for (const auto &weight : unet.weights) {
      auto tensor = as_dtype(
          dif::weights::map_safetensor(checkpoint, weight.source_name),
          unet_config.dtype);
      check_binding(unet.program, weight.tensor, tensor, weight.source_name);
      unet_bindings.emplace(weight.tensor, std::move(tensor));
    }
    // Row 0 is the prompt, row 1 the negative prompt.
    auto context = make_tensor(unet_config.dtype,
                               {2U, context_tokens, context_width});
    for (std::uint64_t row = 0U; row < context_tokens; ++row)
      for (std::uint64_t column = 0U; column < context_width; ++column) {
        const auto value = [&](const Encoding &l, const Encoding &g) {
          // Repeat the shorter encoding over the batched row count.
          const auto l_row = row % l.rows;
          const auto g_row = row % g.rows;
          return column < l_tower.hidden_size
                     ? l.hidden[l_row * l_tower.hidden_size + column]
                     : g.hidden[g_row * g_tower.hidden_size + column -
                                l_tower.hidden_size];
        };
        dif::runtime::store_float(context, row * context_width + column,
                                  value(l_positive, g_positive));
        dif::runtime::store_float(
            context,
            (context_tokens + row) * context_width + column,
            value(l_negative, g_negative));
      }
    // encode_adm: the pooled text vector followed by six size embeddings
    // (height, width, crop top, crop left, target height, target width).
    const std::uint64_t adm_width = 2816;
    auto vector = make_tensor(unet_config.dtype, {2U, adm_width});
    const std::vector<double> sizes{
        static_cast<double>(arguments.height), static_cast<double>(arguments.width),
        0.0, 0.0, static_cast<double>(arguments.height),
        static_cast<double>(arguments.width)};
    std::vector<float> size_part;
    for (const auto size : sizes) {
      const auto embedded = size_embedding(size, 256U);
      size_part.insert(size_part.end(), embedded.begin(), embedded.end());
    }
    for (std::uint64_t row = 0U; row < 2U; ++row) {
      const auto &pooled = row == 0U ? g_positive.pooled : g_negative.pooled;
      for (std::uint64_t column = 0U; column < adm_width; ++column)
        dif::runtime::store_float(
            vector, row * adm_width + column,
            column < pooled.size() ? pooled[column]
                                   : size_part[column - pooled.size()]);
    }
    unet_bindings.emplace(unet.context_input, std::move(context));
    unet_bindings.emplace(unet.vector_input, std::move(vector));
    unet_bindings.emplace(unet.timestep_input, make_tensor(DType::F32, {2U}));
    unet_bindings.emplace(
        unet.latent_input,
        make_tensor(unet_config.dtype, {2U, 4U, latent_height, latent_width}));
    auto unet_prepared = backend->prepare(unet.program, unet_bindings, options);
    const auto unet_prepare_ms = ms_since(started);

    // --- the decoder ------------------------------------------------------
    // Built and prepared before sampling so a repeat pays neither again.
    started = Clock::now();
    dif::frontend::SdxlVaeConfig vae_config;
    vae_config.latent_height = latent_height;
    vae_config.latent_width = latent_width;
    vae_config.dtype = arguments.vae_dtype;
    vae_config.capture_boundaries = false;
    auto vae = dif::frontend::make_sdxl_vae_decoder(vae_config);
    dif::runtime::TensorMap vae_bindings;
    for (const auto &weight : vae.weights) {
      auto tensor = as_dtype(
          dif::weights::map_safetensor(checkpoint, weight.source_name),
          vae_config.dtype);
      check_binding(vae.program, weight.tensor, tensor, weight.source_name);
      vae_bindings.emplace(weight.tensor, std::move(tensor));
    }
    vae_bindings.emplace(
        vae.latent_input,
        make_tensor(vae_config.dtype, {1U, 4U, latent_height, latent_width}));
    auto vae_prepared = backend->prepare(vae.program, vae_bindings, options);
    const auto vae_prepare_ms = ms_since(started);

    // --- sampling ---------------------------------------------------------
    const auto ready_ms = ms_since(total_start);
    const auto table = dif::sampling::DiscreteSigmaTable::linear();
    const auto sigmas = dif::sampling::normal_schedule(table, arguments.steps);
    const auto timesteps = dif::sampling::unet_timesteps(table, sigmas);
    const bool denoise_fully = dif::sampling::max_denoise(table, sigmas);
    const auto elements =
        static_cast<std::size_t>(4U * latent_height * latent_width);
    const auto noise_scale =
        dif::sampling::initial_noise_scale(sigmas.front(), denoise_fully);
    double sample_ms = 0.0;
    double decode_ms = 0.0;
    std::vector<double> step_ms;
    std::vector<double> image_ms;
    // Each pass is one prompt to one PNG with everything already loaded,
    // which is the shape a long-lived process runs in.
    for (std::uint32_t image_index = 0U; image_index < arguments.images;
         ++image_index) {
    const auto image_start = Clock::now();
    started = Clock::now();
    auto latent =
        dif::torch_cpu_normal(elements, arguments.seed + image_index);
    for (auto &value : latent)
      value *= noise_scale;
    step_ms.clear();
    for (std::uint32_t step = 0U; step < arguments.steps; ++step) {
      const auto sigma = sigmas[step];
      const auto next_sigma = sigmas[step + 1U];
      const auto step_start = Clock::now();
      const auto scale = dif::sampling::eps_input_scale(sigma);
      auto scaled = make_tensor(unet_config.dtype,
                                {2U, 4U, latent_height, latent_width});
      for (std::uint64_t row = 0U; row < 2U; ++row)
        for (std::size_t index = 0; index < elements; ++index)
          dif::runtime::store_float(scaled, row * elements + index,
                                    latent[index] * scale);
      auto timestep = make_tensor(DType::F32, {2U});
      dif::runtime::store_float(timestep, 0U, timesteps[step]);
      dif::runtime::store_float(timestep, 1U, timesteps[step]);
      unet_bindings.insert_or_assign(unet.latent_input, std::move(scaled));
      unet_bindings.insert_or_assign(unet.timestep_input, std::move(timestep));
      const auto result = unet_prepared->run(unet_bindings, options);
      const auto &epsilon = result.outputs.at(unet.output);
      // The reference guides the DENOISED values, then takes the Karras
      // derivative; in epsilon terms that is the same blend, but keep the
      // reference's order so the arithmetic matches.
      for (std::size_t index = 0; index < elements; ++index) {
        const auto conditioned =
            latent[index] -
            dif::runtime::load_float(epsilon, index) * sigma;
        const auto unconditioned =
            latent[index] -
            dif::runtime::load_float(epsilon, elements + index) * sigma;
        const auto guided =
            unconditioned +
            (conditioned - unconditioned) * static_cast<float>(arguments.cfg);
        const auto derivative = (latent[index] - guided) / sigma;
        latent[index] += derivative * (next_sigma - sigma);
      }
      step_ms.push_back(ms_since(step_start));
      if (arguments.report_steps)
        std::cerr << "SDXL_STEP " << step << " sigma=" << sigma
                  << " timestep=" << timesteps[step]
                  << " ms=" << step_ms.back() << "\n";
    }
    sample_ms = ms_since(started);

    // --- decode -----------------------------------------------------------
    started = Clock::now();
    auto decode_input = make_tensor(vae_config.dtype,
                                    {1U, 4U, latent_height, latent_width});
    for (std::size_t index = 0; index < elements; ++index)
      dif::runtime::store_float(
          decode_input, index,
          static_cast<float>(latent[index] / kLatentScale));
    vae_bindings.insert_or_assign(vae.latent_input, std::move(decode_input));
    auto vae_result = vae_prepared->run(vae_bindings, options);
    const auto &pixels = vae_result.outputs.at(vae.raw_output);
    decode_ms = ms_since(started);

    const auto image = rgb8(pixels, arguments.width, arguments.height);
    auto output_path = arguments.output;
    if (arguments.images > 1U) {
      auto stem = output_path.stem().string();
      output_path.replace_filename(stem + "-" +
                                   std::to_string(image_index) +
                                   output_path.extension().string());
    }
    dif::write_png_rgb8(output_path,
                        static_cast<std::uint32_t>(arguments.width),
                        static_cast<std::uint32_t>(arguments.height), image);
    image_ms.push_back(ms_since(image_start));
    std::cout << "SDXL_IMAGE index=" << image_index
              << " seed=" << arguments.seed + image_index
              << " sample_ms=" << sample_ms << " decode_ms=" << decode_ms
              << " prompt_to_png_ms=" << image_ms.back() << " -> "
              << output_path << "\n";
    }
    const auto total_ms = ms_since(total_start);

    std::cout << "SDXL_SAMPLE steps=" << arguments.steps
              << " cfg=" << arguments.cfg << " seed=" << arguments.seed
              << " size=" << arguments.width << "x" << arguments.height
              << " chunks=" << chunks
              << " tokenize_ms=" << tokenize_ms
              << " text_prepare_ms=" << text_prepare_ms
              << " text_ms=" << text_ms
              << " unet_prepare_ms=" << unet_prepare_ms
              << " vae_prepare_ms=" << vae_prepare_ms
              << " ready_ms=" << ready_ms
              << " images=" << arguments.images
              << " best_prompt_to_png_ms="
              << *std::min_element(image_ms.begin(), image_ms.end())
              << " total_ms=" << total_ms << " -> " << arguments.output << "\n";
    if (!arguments.report.empty()) {
      std::ofstream report(arguments.report);
      if (!report)
        dif::fail("cannot write " + arguments.report.string());
      report << "{\n  \"runtime\": \"diffusion compiler\",\n"
             << "  \"steps\": " << arguments.steps << ",\n"
             << "  \"cfg\": " << arguments.cfg << ",\n"
             << "  \"seed\": " << arguments.seed << ",\n"
             << "  \"width\": " << arguments.width << ",\n"
             << "  \"height\": " << arguments.height << ",\n"
             << "  \"encoder_chunks\": " << chunks << ",\n"
             << "  \"timing_ms\": {\n"
             << "    \"tokenize\": " << tokenize_ms << ",\n"
             << "    \"text_prepare\": " << text_prepare_ms << ",\n"
             << "    \"text\": " << text_ms << ",\n"
             << "    \"unet_prepare\": " << unet_prepare_ms << ",\n"
             << "    \"sample\": " << sample_ms << ",\n"
             << "    \"decode\": " << decode_ms << ",\n"
             << "    \"total\": " << total_ms << "\n  },\n"
             << "  \"step_ms\": [";
      for (std::size_t index = 0; index < step_ms.size(); ++index)
        report << (index ? ", " : "") << step_ms[index];
      report << "]\n}\n";
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difsdxlsample: " << error.what() << "\n";
    return 1;
  }
}
