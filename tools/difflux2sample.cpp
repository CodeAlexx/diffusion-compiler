// difflux2sample -- native FLUX.2 [klein] Base 9B prompt-to-PNG path.
//
// The accepted execution path is C++/DiffIR only.  Creator/PyTorch artifacts
// may be supplied as optional parity fixtures, but neither Python nor a
// framework runtime is loaded by this process.

#include "dif/frontend/flux2.hpp"
#include "dif/frontend/squareq_w4.hpp"
#include "dif/frontend/flux2_prompt.hpp"
#include "dif/frontend/flux2_vae.hpp"
#include "dif/compiler/layout_plan.hpp"
#include "dif/compiler/residency_plan.hpp"
#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/sampling/rectified_flow.hpp"
#include "dif/support/error.hpp"
#include "dif/support/png.hpp"
#include "dif/support/sha256.hpp"
#include "dif/text/qwen_bpe_tokenizer.hpp"
#include "dif/weights/safetensors.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr std::string_view kCreatorCommit =
    "50fe5162777813d869182b139e83b10743caef15";
constexpr std::string_view kModelRevision =
    "32773329fbe7e81a90ef971740e8ba4b0364ecf3";

struct Arguments {
  fs::path model_directory;
  fs::path transformer_checkpoint;
  fs::path vae_checkpoint;
  fs::path positive_conditioning;
  fs::path negative_conditioning;
  fs::path initial_latent;
  std::string initial_latent_tensor{"initial_image_tokens"};
  fs::path output_png;
  fs::path report;
  fs::path state_output;
  fs::path cache_directory;
  std::string prompt;
  std::uint64_t seed{20260901U};
  std::uint32_t steps{50U};
  // --flux2-model klein9b|dev. dev = FLUX.2 [dev] geometry (8 + 48 blocks,
  // hidden 6144, Mistral context 15360, guidance embedding, no CFG batch);
  // its conditioning must come from --positive-conditioning until the Mistral
  // conditioner frontend lands.
  std::string flux2_model{"klein9b"};
  std::uint32_t start_step{};
  std::uint32_t stop_after{};
  std::uint32_t capture_every{};
  std::uint32_t width{1024U};
  std::uint32_t height{1024U};
  float guidance{4.0F};
  std::uint64_t transformer_attention_implementation{2U};
  std::uint32_t cudnn_attention_heuristic{};
  // Measured RTX 5080 BF16 plan. Qwen retains its separate safe 1/2 ring.
  std::uint32_t streamed_prefetch_depth{3U};
  std::uint32_t streamed_staging_buffers{4U};
  std::uint32_t streamed_stage_threads{3U};
  std::uint64_t resident_plan_mib{14000U};
  dif::compiler::StreamedResidencyOrder resident_order{
      dif::compiler::StreamedResidencyOrder::FirstConsumer};
  std::vector<std::uint32_t> tune_linear_operations;
  std::vector<dif::runtime::LinearAlgorithmChoice> linear_algorithm_choices;
  std::uint32_t linear_tuning_warmups{3U};
  std::uint32_t linear_tuning_iterations{10U};
  std::uint32_t linear_tuning_sessions{3U};
  bool expand_linear_algorithms{};
  bool persist_linear_heuristics{};
  bool lazy_resident_upload{};
  bool streamed_keep_mapped_pages{true};
  bool profile_pipeline{};
  bool int8_weight_only_all_linears{};
  // --squareq-w4-slab DIR: SquareQ v3 W4 slab (index + squareq-plan.json)
  // replacing every planned transformer Linear weight (dequant-first).
  fs::path squareq_w4_slab;
  bool int8_weight_only_row_scaled_all_linears{};
  std::uint32_t int8_weight_only_group_size{64U};
  std::vector<std::string> int8_weight_only_exclude_names;
  std::vector<std::string> int8_weight_only_group32_names;
  std::uint32_t w8a8_single_linear1_blocks{};
  std::vector<std::uint32_t> w8a8_single_linear1_block_ids;
  std::uint32_t w8a8_single_mlp_blocks{};
  std::vector<std::uint32_t> w8a8_single_mlp_block_ids;
  std::uint32_t w8a8_single_qk_blocks{};
  std::vector<std::uint32_t> w8a8_single_qk_block_ids;
  std::uint32_t w8a8_single_linear2_blocks{};
  std::vector<std::uint32_t> w8a8_single_linear2_block_ids;
  std::vector<std::uint32_t> w8a8_double_image_mlp_blocks;
  std::vector<std::uint32_t> w8a8_double_mlp_blocks;
  std::vector<std::uint32_t> w8a8_double_image_blocks;
  std::vector<std::uint32_t> w8a8_double_text_blocks;
  std::vector<std::uint32_t> w8a8_double_blocks;
  std::uint32_t fp8_single_linear1_blocks{};
  std::vector<std::uint32_t> fp8_single_linear1_block_ids;
  std::uint32_t fp8_single_mlp_blocks{};
  std::vector<std::uint32_t> fp8_single_mlp_block_ids;
  std::uint32_t fp8_single_linear2_blocks{};
  std::vector<std::uint32_t> fp8_single_linear2_block_ids;
  std::vector<std::uint32_t> fp8_double_image_mlp_blocks;
  std::vector<std::uint32_t> fp8_double_image_blocks;
  std::vector<std::uint32_t> fp8_double_text_blocks;
  std::vector<std::uint32_t> fp8_double_blocks;
  bool fp8_row_scaled{};
  dif::ir::Int8RowQuantization w8a8_quantization{
      dif::ir::Int8RowQuantization::Direct};
  bool w8a8_single_mlp_h256_convrot{};
  bool w8a8_weight_equalization{};
  bool w8a8_mse_weight_scale{};
  bool w8a8_activation_residual2{};
  bool w8a8_activation_residual2_single_linear1{};
  bool w8a8_activation_residual2_single_linear2{};
  bool w8a8_activation_residual2_double{};
  double w8a8_activation_clip_ratio{1.0};
  double w8a8_activation_clip_ratio_single_linear1{
      std::numeric_limits<double>::quiet_NaN()};
  double w8a8_activation_clip_ratio_single_linear2{
      std::numeric_limits<double>::quiet_NaN()};
  double w8a8_activation_clip_ratio_double{
      std::numeric_limits<double>::quiet_NaN()};
  std::uint32_t w8a8_activation_clip_switch_step{};
  double w8a8_activation_clip_after_ratio{
      std::numeric_limits<double>::quiet_NaN()};
};

[[noreturn]] void usage_error(const std::string &message) {
  std::cerr
      << "difflux2sample: " << message << "\n"
      << "usage: difflux2sample --model-dir DIR --vae-checkpoint ae.safetensors "
         "--prompt TEXT --output image.png --report report.json "
         "[--state-output state.safetensors] "
         "[--transformer-checkpoint model.safetensors] "
         "[--positive-conditioning positive.safetensors "
         "--negative-conditioning empty.safetensors] "
         "[--initial-latent initial.safetensors] "
         "[--initial-latent-tensor NAME] [--seed N] [--steps N] "
         "[--start-step N] [--stop-after N] [--capture-every N] "
         "[--width N --height N] [--guidance F] "
         "[--transformer-attention cudnn|flash] "
         "[--cudnn-attention-heuristic a|b|fallback|autotune] "
         "[--streamed-prefetch-depth N --streamed-staging-buffers N "
         "--streamed-stage-threads N] [--streamed-release-pages] "
         "[--resident-plan-mib N] [--resident-order largest|first] "
         "[--lazy-resident-upload] "
         "[--tune-linear-operation ID ...] [--expand-linear-algorithms] "
         "[--select-linear-algorithm OP_ID:HEURISTIC_RANK ...] "
         "[--persist-linear-heuristics] [--linear-tuning-warmups N "
         "--linear-tuning-iterations N --linear-tuning-sessions N] "
         "[--w8a8-single-linear1-blocks N | "
         "--w8a8-single-linear1-block ID ...] "
         "[--w8a8-convrot | --w8a8-f32-convrot | "
         "--w8a8-f32-signed-convrot | "
         "--w8a8-f32-signed-convrot-4096 | "
         "--w8a8-signed-convrot | "
         "--w8a8-signed-convrot-4096] "
         "[--w8a8-weight-equalization] "
         "[--w8a8-mse-weight-scale] "
         "[--w8a8-activation-residual2] "
         "[--w8a8-activation-residual2-single-linear1] "
         "[--w8a8-activation-residual2-single-linear2] "
         "[--w8a8-activation-residual2-double] "
         "[--w8a8-activation-clip-ratio F] "
         "[--w8a8-activation-clip-single-linear1 F] "
         "[--w8a8-activation-clip-single-linear2 F] "
         "[--w8a8-activation-clip-double F] "
         "[--w8a8-activation-clip-switch-step N "
         "--w8a8-activation-clip-after-ratio F] "
         "[--w8a8-single-mlp-blocks N | "
         "--w8a8-single-mlp-block ID ...] "
         "[--w8a8-single-mlp-h256-convrot] "
         "[--w8a8-single-qk-blocks N | "
         "--w8a8-single-qk-block ID ...] "
         "[--w8a8-single-linear2-blocks N | "
         "--w8a8-single-linear2-block ID ...] "
         "[--w8a8-double-image-mlp-block ID ...] "
         "[--w8a8-double-mlp-block ID ...] "
         "[--w8a8-double-image-block ID ...] "
         "[--w8a8-double-text-block ID ...] "
         "[--w8a8-double-block ID ...] "
         "[--fp8-single-linear1-blocks N | "
         "--fp8-single-linear1-block ID ...] "
         "[--fp8-single-mlp-blocks N | "
         "--fp8-single-mlp-block ID ...] "
         "[--fp8-single-linear2-blocks N | "
         "--fp8-single-linear2-block ID ...] "
         "[--fp8-double-image-mlp-block ID ...] "
         "[--fp8-double-image-block ID ...] "
         "[--fp8-double-text-block ID ...] "
         "[--fp8-double-block ID ...] "
         "[--fp8-row-scaled] "
         "[--int8-weight-only-all-linears | "
         "--int8-weight-only-row-scaled-all-linears] "
         "[--int8-weight-only-group-size 16|32|64] "
         "[--int8-weight-only-exclude CHECKPOINT_NAME ...] "
         "[--int8-weight-only-group32 CHECKPOINT_NAME ...] "
         "[--cache-dir DIR] [--profile-pipeline]\n";
  std::exit(2);
}

Arguments parse(int argc, char **argv) {
  Arguments result;
  bool w8a8_quantization_option{};
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    const auto value = [&]() -> std::string {
      if (++index >= argc)
        usage_error(option + " requires a value");
      return argv[index];
    };
    if (option == "--model-dir")
      result.model_directory = value();
    else if (option == "--transformer-checkpoint")
      result.transformer_checkpoint = value();
    else if (option == "--vae-checkpoint")
      result.vae_checkpoint = value();
    else if (option == "--positive-conditioning")
      result.positive_conditioning = value();
    else if (option == "--negative-conditioning")
      result.negative_conditioning = value();
    else if (option == "--initial-latent")
      result.initial_latent = value();
    else if (option == "--initial-latent-tensor")
      result.initial_latent_tensor = value();
    else if (option == "--output")
      result.output_png = value();
    else if (option == "--report")
      result.report = value();
    else if (option == "--state-output")
      result.state_output = value();
    else if (option == "--cache-dir")
      result.cache_directory = value();
    else if (option == "--prompt")
      result.prompt = value();
    else if (option == "--seed")
      result.seed = std::stoull(value());
    else if (option == "--flux2-model")
      result.flux2_model = value();
    else if (option == "--steps")
      result.steps = static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--start-step")
      result.start_step = static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--stop-after")
      result.stop_after = static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--capture-every")
      result.capture_every =
          static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--width")
      result.width = static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--height")
      result.height = static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--guidance")
      result.guidance = std::stof(value());
    else if (option == "--streamed-prefetch-depth")
      result.streamed_prefetch_depth =
          static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--streamed-staging-buffers")
      result.streamed_staging_buffers =
          static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--streamed-stage-threads")
      result.streamed_stage_threads =
          static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--resident-plan-mib")
      result.resident_plan_mib = std::stoull(value());
    else if (option == "--resident-order") {
      const auto order = value();
      if (order == "largest")
        result.resident_order =
            dif::compiler::StreamedResidencyOrder::LargestFirst;
      else if (order == "first")
        result.resident_order =
            dif::compiler::StreamedResidencyOrder::FirstConsumer;
      else
        usage_error("invalid resident order " + order);
    }
    else if (option == "--lazy-resident-upload")
      result.lazy_resident_upload = true;
    else if (option == "--tune-linear-operation")
      result.tune_linear_operations.push_back(
          static_cast<std::uint32_t>(std::stoul(value())));
    else if (option == "--select-linear-algorithm") {
      const auto selection = value();
      const auto split = selection.find(':');
      if (split == std::string::npos || split == 0U ||
          split + 1U >= selection.size())
        usage_error("Linear algorithm choice must be OP_ID:HEURISTIC_RANK");
      result.linear_algorithm_choices.push_back(
          {static_cast<std::uint32_t>(
               std::stoul(selection.substr(0U, split))),
           static_cast<std::uint32_t>(
               std::stoul(selection.substr(split + 1U)))});
    }
    else if (option == "--linear-tuning-warmups")
      result.linear_tuning_warmups =
          static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--linear-tuning-iterations")
      result.linear_tuning_iterations =
          static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--linear-tuning-sessions")
      result.linear_tuning_sessions =
          static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--expand-linear-algorithms")
      result.expand_linear_algorithms = true;
    else if (option == "--persist-linear-heuristics")
      result.persist_linear_heuristics = true;
    else if (option == "--streamed-release-pages")
      result.streamed_keep_mapped_pages = false;
    else if (option == "--transformer-attention") {
      const auto backend = value();
      if (backend == "cudnn")
        result.transformer_attention_implementation = 2U;
      else if (backend == "flash")
        result.transformer_attention_implementation = 4U;
      else
        usage_error("invalid transformer attention backend " + backend);
    }
    else if (option == "--cudnn-attention-heuristic") {
      const auto heuristic = value();
      if (heuristic == "a")
        result.cudnn_attention_heuristic = 0U;
      else if (heuristic == "b")
        result.cudnn_attention_heuristic = 1U;
      else if (heuristic == "fallback")
        result.cudnn_attention_heuristic = 2U;
      else if (heuristic == "autotune")
        result.cudnn_attention_heuristic = 3U;
      else
        usage_error("invalid cuDNN attention heuristic " + heuristic);
    }
    else if (option == "--profile-pipeline")
      result.profile_pipeline = true;
    else if (option == "--int8-weight-only-all-linears")
      result.int8_weight_only_all_linears = true;
    else if (option == "--squareq-w4-slab")
      result.squareq_w4_slab = value();
    else if (option == "--int8-weight-only-row-scaled-all-linears")
      result.int8_weight_only_row_scaled_all_linears = true;
    else if (option == "--int8-weight-only-group-size")
      result.int8_weight_only_group_size =
          static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--int8-weight-only-exclude")
      result.int8_weight_only_exclude_names.push_back(value());
    else if (option == "--int8-weight-only-group32")
      result.int8_weight_only_group32_names.push_back(value());
    else if (option == "--w8a8-single-linear1-blocks")
      result.w8a8_single_linear1_blocks =
          static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--w8a8-single-linear1-block")
      result.w8a8_single_linear1_block_ids.push_back(
          static_cast<std::uint32_t>(std::stoul(value())));
    else if (option == "--w8a8-single-mlp-blocks")
      result.w8a8_single_mlp_blocks =
          static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--w8a8-single-mlp-block")
      result.w8a8_single_mlp_block_ids.push_back(
          static_cast<std::uint32_t>(std::stoul(value())));
    else if (option == "--w8a8-single-mlp-h256-convrot")
      result.w8a8_single_mlp_h256_convrot = true;
    else if (option == "--w8a8-single-qk-blocks")
      result.w8a8_single_qk_blocks =
          static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--w8a8-single-qk-block")
      result.w8a8_single_qk_block_ids.push_back(
          static_cast<std::uint32_t>(std::stoul(value())));
    else if (option == "--w8a8-single-linear2-blocks")
      result.w8a8_single_linear2_blocks =
          static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--w8a8-single-linear2-block")
      result.w8a8_single_linear2_block_ids.push_back(
          static_cast<std::uint32_t>(std::stoul(value())));
    else if (option == "--w8a8-double-image-mlp-block")
      result.w8a8_double_image_mlp_blocks.push_back(
          static_cast<std::uint32_t>(std::stoul(value())));
    else if (option == "--w8a8-double-mlp-block")
      result.w8a8_double_mlp_blocks.push_back(
          static_cast<std::uint32_t>(std::stoul(value())));
    else if (option == "--w8a8-double-image-block")
      result.w8a8_double_image_blocks.push_back(
          static_cast<std::uint32_t>(std::stoul(value())));
    else if (option == "--w8a8-double-text-block")
      result.w8a8_double_text_blocks.push_back(
          static_cast<std::uint32_t>(std::stoul(value())));
    else if (option == "--w8a8-double-block")
      result.w8a8_double_blocks.push_back(
          static_cast<std::uint32_t>(std::stoul(value())));
    else if (option == "--fp8-single-linear1-blocks")
      result.fp8_single_linear1_blocks =
          static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--fp8-single-linear1-block")
      result.fp8_single_linear1_block_ids.push_back(
          static_cast<std::uint32_t>(std::stoul(value())));
    else if (option == "--fp8-single-mlp-blocks")
      result.fp8_single_mlp_blocks =
          static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--fp8-single-mlp-block")
      result.fp8_single_mlp_block_ids.push_back(
          static_cast<std::uint32_t>(std::stoul(value())));
    else if (option == "--fp8-single-linear2-blocks")
      result.fp8_single_linear2_blocks =
          static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--fp8-single-linear2-block")
      result.fp8_single_linear2_block_ids.push_back(
          static_cast<std::uint32_t>(std::stoul(value())));
    else if (option == "--fp8-double-image-mlp-block")
      result.fp8_double_image_mlp_blocks.push_back(
          static_cast<std::uint32_t>(std::stoul(value())));
    else if (option == "--fp8-double-image-block")
      result.fp8_double_image_blocks.push_back(
          static_cast<std::uint32_t>(std::stoul(value())));
    else if (option == "--fp8-double-text-block")
      result.fp8_double_text_blocks.push_back(
          static_cast<std::uint32_t>(std::stoul(value())));
    else if (option == "--fp8-double-block")
      result.fp8_double_blocks.push_back(
          static_cast<std::uint32_t>(std::stoul(value())));
    else if (option == "--fp8-row-scaled")
      result.fp8_row_scaled = true;
    else if (option == "--w8a8-convrot") {
      if (w8a8_quantization_option)
        usage_error("W8A8 rotation options are mutually exclusive");
      result.w8a8_quantization = dif::ir::Int8RowQuantization::H256ConvRot;
      w8a8_quantization_option = true;
    }
    else if (option == "--w8a8-f32-convrot") {
      if (w8a8_quantization_option)
        usage_error("W8A8 rotation options are mutually exclusive");
      result.w8a8_quantization =
          dif::ir::Int8RowQuantization::H256F32ConvRot;
      w8a8_quantization_option = true;
    }
    else if (option == "--w8a8-f32-signed-convrot") {
      if (w8a8_quantization_option)
        usage_error("W8A8 rotation options are mutually exclusive");
      result.w8a8_quantization =
          dif::ir::Int8RowQuantization::H256F32SignedConvRot;
      w8a8_quantization_option = true;
    }
    else if (option == "--w8a8-f32-signed-convrot-4096") {
      if (w8a8_quantization_option)
        usage_error("W8A8 rotation options are mutually exclusive");
      result.w8a8_quantization =
          dif::ir::Int8RowQuantization::H4096F32SignedConvRot;
      w8a8_quantization_option = true;
    }
    else if (option == "--w8a8-signed-convrot") {
      if (w8a8_quantization_option)
        usage_error("W8A8 rotation options are mutually exclusive");
      result.w8a8_quantization =
          dif::ir::Int8RowQuantization::H256SignedConvRot;
      w8a8_quantization_option = true;
    }
    else if (option == "--w8a8-signed-convrot-4096") {
      if (w8a8_quantization_option)
        usage_error("W8A8 rotation options are mutually exclusive");
      result.w8a8_quantization =
          dif::ir::Int8RowQuantization::H4096SignedConvRot;
      w8a8_quantization_option = true;
    }
    else if (option == "--w8a8-weight-equalization")
      result.w8a8_weight_equalization = true;
    else if (option == "--w8a8-mse-weight-scale")
      result.w8a8_mse_weight_scale = true;
    else if (option == "--w8a8-activation-residual2")
      result.w8a8_activation_residual2 = true;
    else if (option == "--w8a8-activation-residual2-single-linear1")
      result.w8a8_activation_residual2_single_linear1 = true;
    else if (option == "--w8a8-activation-residual2-single-linear2")
      result.w8a8_activation_residual2_single_linear2 = true;
    else if (option == "--w8a8-activation-residual2-double")
      result.w8a8_activation_residual2_double = true;
    else if (option == "--w8a8-activation-clip-ratio")
      result.w8a8_activation_clip_ratio = std::stod(value());
    else if (option == "--w8a8-activation-clip-single-linear1")
      result.w8a8_activation_clip_ratio_single_linear1 = std::stod(value());
    else if (option == "--w8a8-activation-clip-single-linear2")
      result.w8a8_activation_clip_ratio_single_linear2 = std::stod(value());
    else if (option == "--w8a8-activation-clip-double")
      result.w8a8_activation_clip_ratio_double = std::stod(value());
    else if (option == "--w8a8-activation-clip-switch-step")
      result.w8a8_activation_clip_switch_step =
          static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--w8a8-activation-clip-after-ratio")
      result.w8a8_activation_clip_after_ratio = std::stod(value());
    else
      usage_error("unknown option " + option);
  }
  if (result.model_directory.empty() || result.vae_checkpoint.empty() ||
      result.prompt.empty() || result.output_png.empty() ||
      result.report.empty() || result.steps == 0U ||
      result.start_step >= result.steps ||
      (result.stop_after != 0U &&
       result.stop_after > result.steps - result.start_step) ||
      result.width == 0U || result.height == 0U || result.width % 16U != 0U ||
      result.height % 16U != 0U || !std::isfinite(result.guidance) ||
      result.guidance < 0.0F ||
      !(result.w8a8_activation_clip_ratio > 0.0) ||
      result.w8a8_activation_clip_ratio > 1.0 ||
      (std::isfinite(result.w8a8_activation_clip_ratio_single_linear1) &&
       (!(result.w8a8_activation_clip_ratio_single_linear1 > 0.0) ||
        result.w8a8_activation_clip_ratio_single_linear1 > 1.0)) ||
      (std::isfinite(result.w8a8_activation_clip_ratio_single_linear2) &&
       (!(result.w8a8_activation_clip_ratio_single_linear2 > 0.0) ||
        result.w8a8_activation_clip_ratio_single_linear2 > 1.0)) ||
      (std::isfinite(result.w8a8_activation_clip_ratio_double) &&
       (!(result.w8a8_activation_clip_ratio_double > 0.0) ||
        result.w8a8_activation_clip_ratio_double > 1.0)) ||
      ((result.w8a8_activation_clip_switch_step == 0U) !=
       !std::isfinite(result.w8a8_activation_clip_after_ratio)) ||
      (std::isfinite(result.w8a8_activation_clip_after_ratio) &&
       (!(result.w8a8_activation_clip_after_ratio > 0.0) ||
        result.w8a8_activation_clip_after_ratio > 1.0)) ||
      result.w8a8_activation_clip_switch_step >= result.steps ||
      (result.w8a8_activation_clip_switch_step != 0U &&
       (std::isfinite(result.w8a8_activation_clip_ratio_single_linear1) ||
        std::isfinite(result.w8a8_activation_clip_ratio_single_linear2) ||
        std::isfinite(result.w8a8_activation_clip_ratio_double))) ||
      result.streamed_prefetch_depth == 0U ||
      result.streamed_stage_threads == 0U ||
      result.streamed_staging_buffers < result.streamed_prefetch_depth + 1U ||
      result.linear_tuning_iterations == 0U ||
      result.linear_tuning_sessions < 2U ||
      result.positive_conditioning.empty() !=
          result.negative_conditioning.empty() ||
      (result.int8_weight_only_group_size != 16U &&
       result.int8_weight_only_group_size != 32U &&
       result.int8_weight_only_group_size != 64U) ||
      result.w8a8_single_linear1_blocks > 24U ||
      result.w8a8_single_mlp_blocks > 24U ||
      result.w8a8_single_qk_blocks > 24U ||
      result.w8a8_single_linear2_blocks > 24U ||
      result.fp8_single_linear1_blocks > 24U ||
      result.fp8_single_mlp_blocks > 24U ||
      result.fp8_single_linear2_blocks > 24U ||
      (result.w8a8_single_linear1_blocks != 0U &&
       !result.w8a8_single_linear1_block_ids.empty()) ||
      (result.w8a8_single_mlp_blocks != 0U &&
       !result.w8a8_single_mlp_block_ids.empty()) ||
      (result.w8a8_single_qk_blocks != 0U &&
       !result.w8a8_single_qk_block_ids.empty()) ||
      (result.w8a8_single_linear2_blocks != 0U &&
       !result.w8a8_single_linear2_block_ids.empty()) ||
      (result.fp8_single_linear1_blocks != 0U &&
       !result.fp8_single_linear1_block_ids.empty()) ||
      (result.fp8_single_mlp_blocks != 0U &&
       !result.fp8_single_mlp_block_ids.empty()) ||
      (result.fp8_single_linear2_blocks != 0U &&
       !result.fp8_single_linear2_block_ids.empty()) ||
      ((result.fp8_single_linear1_blocks != 0U ||
        !result.fp8_single_linear1_block_ids.empty()) &&
       (result.fp8_single_mlp_blocks != 0U ||
        !result.fp8_single_mlp_block_ids.empty())) ||
      (result.fp8_row_scaled &&
       (result.fp8_single_mlp_blocks != 0U ||
        !result.fp8_single_mlp_block_ids.empty())) ||
      std::any_of(result.w8a8_single_linear1_block_ids.begin(),
                  result.w8a8_single_linear1_block_ids.end(),
                  [](std::uint32_t block) { return block >= 24U; }) ||
      std::any_of(result.w8a8_single_mlp_block_ids.begin(),
                  result.w8a8_single_mlp_block_ids.end(),
                  [](std::uint32_t block) { return block >= 24U; }) ||
      std::any_of(result.w8a8_single_qk_block_ids.begin(),
                  result.w8a8_single_qk_block_ids.end(),
                  [](std::uint32_t block) { return block >= 24U; }) ||
      std::any_of(result.w8a8_single_linear2_block_ids.begin(),
                  result.w8a8_single_linear2_block_ids.end(),
                  [](std::uint32_t block) { return block >= 24U; }) ||
      std::any_of(result.w8a8_double_image_mlp_blocks.begin(),
                  result.w8a8_double_image_mlp_blocks.end(),
                  [](std::uint32_t block) { return block >= 8U; }) ||
      std::any_of(result.w8a8_double_mlp_blocks.begin(),
                  result.w8a8_double_mlp_blocks.end(),
                  [](std::uint32_t block) { return block >= 8U; }) ||
      std::any_of(result.w8a8_double_image_blocks.begin(),
                  result.w8a8_double_image_blocks.end(),
                  [](std::uint32_t block) { return block >= 8U; }) ||
      std::any_of(result.w8a8_double_text_blocks.begin(),
                  result.w8a8_double_text_blocks.end(),
                  [](std::uint32_t block) { return block >= 8U; }) ||
      std::any_of(result.w8a8_double_blocks.begin(),
                  result.w8a8_double_blocks.end(),
                  [](std::uint32_t block) { return block >= 8U; }) ||
      std::any_of(result.fp8_single_linear1_block_ids.begin(),
                  result.fp8_single_linear1_block_ids.end(),
                  [](std::uint32_t block) { return block >= 24U; }) ||
      std::any_of(result.fp8_single_mlp_block_ids.begin(),
                  result.fp8_single_mlp_block_ids.end(),
                  [](std::uint32_t block) { return block >= 24U; }) ||
      std::any_of(result.fp8_single_linear2_block_ids.begin(),
                  result.fp8_single_linear2_block_ids.end(),
                  [](std::uint32_t block) { return block >= 24U; }) ||
      std::any_of(result.fp8_double_image_mlp_blocks.begin(),
                  result.fp8_double_image_mlp_blocks.end(),
                  [](std::uint32_t block) { return block >= 8U; }) ||
      std::any_of(result.fp8_double_image_blocks.begin(),
                  result.fp8_double_image_blocks.end(),
                  [](std::uint32_t block) { return block >= 8U; }) ||
      std::any_of(result.fp8_double_text_blocks.begin(),
                  result.fp8_double_text_blocks.end(),
                  [](std::uint32_t block) { return block >= 8U; }) ||
      std::any_of(result.fp8_double_blocks.begin(),
                  result.fp8_double_blocks.end(),
                  [](std::uint32_t block) { return block >= 8U; }) ||
      (result.w8a8_quantization != dif::ir::Int8RowQuantization::Direct &&
       result.w8a8_single_linear1_blocks == 0U &&
       result.w8a8_single_linear1_block_ids.empty() &&
       result.w8a8_single_mlp_blocks == 0U &&
       result.w8a8_single_mlp_block_ids.empty() &&
       result.w8a8_single_qk_blocks == 0U &&
       result.w8a8_single_qk_block_ids.empty() &&
       result.w8a8_single_linear2_blocks == 0U &&
       result.w8a8_single_linear2_block_ids.empty() &&
       result.w8a8_double_image_mlp_blocks.empty() &&
       result.w8a8_double_mlp_blocks.empty() &&
       result.w8a8_double_image_blocks.empty() &&
       result.w8a8_double_text_blocks.empty() &&
       result.w8a8_double_blocks.empty()))
    usage_error("missing or invalid required arguments");
  if (result.persist_linear_heuristics && result.cache_directory.empty())
    usage_error("persisted linear heuristics require --cache-dir");
  if (result.int8_weight_only_all_linears &&
      result.int8_weight_only_row_scaled_all_linears)
    usage_error("INT8 weight-only modes are mutually exclusive");
  if (!result.int8_weight_only_group32_names.empty() &&
      (!result.int8_weight_only_all_linears ||
       result.int8_weight_only_group_size != 64U))
    usage_error("mixed group-32 INT8 weights require group-64 "
                "--int8-weight-only-all-linears");
  if (result.w8a8_mse_weight_scale &&
      result.w8a8_quantization == dif::ir::Int8RowQuantization::Direct)
    usage_error("MSE weight scaling requires a ConvRot W8A8 mode");
  if (result.w8a8_single_mlp_h256_convrot &&
      result.w8a8_single_mlp_blocks == 0U &&
      result.w8a8_single_mlp_block_ids.empty())
    usage_error("single-MLP H256 ConvRot override requires a single-MLP "
                "W8A8 selection");
  if ((result.int8_weight_only_all_linears ||
       result.int8_weight_only_row_scaled_all_linears ||
       result.w8a8_single_linear1_blocks != 0U ||
       !result.w8a8_single_linear1_block_ids.empty() ||
       result.w8a8_single_mlp_blocks != 0U ||
       !result.w8a8_single_mlp_block_ids.empty() ||
       result.w8a8_single_qk_blocks != 0U ||
       !result.w8a8_single_qk_block_ids.empty() ||
       result.w8a8_single_linear2_blocks != 0U ||
       !result.w8a8_single_linear2_block_ids.empty() ||
       !result.w8a8_double_image_mlp_blocks.empty() ||
       !result.w8a8_double_mlp_blocks.empty() ||
       !result.w8a8_double_image_blocks.empty() ||
       !result.w8a8_double_text_blocks.empty() ||
       !result.w8a8_double_blocks.empty() ||
       result.fp8_single_linear1_blocks != 0U ||
       !result.fp8_single_linear1_block_ids.empty() ||
       result.fp8_single_mlp_blocks != 0U ||
       !result.fp8_single_mlp_block_ids.empty() ||
       result.fp8_single_linear2_blocks != 0U ||
       !result.fp8_single_linear2_block_ids.empty() ||
       !result.fp8_double_image_mlp_blocks.empty() ||
       !result.fp8_double_image_blocks.empty() ||
       !result.fp8_double_text_blocks.empty() ||
       !result.fp8_double_blocks.empty()) &&
      result.cache_directory.empty())
    usage_error("FLUX.2 low-precision checkpoint packing requires --cache-dir");
  auto unique_w8a8_blocks = result.w8a8_single_linear1_block_ids;
  std::sort(unique_w8a8_blocks.begin(), unique_w8a8_blocks.end());
  if (std::adjacent_find(unique_w8a8_blocks.begin(),
                         unique_w8a8_blocks.end()) !=
      unique_w8a8_blocks.end())
    usage_error("FLUX.2 W8A8 block ids must be unique");
  auto unique_mlp_blocks = result.w8a8_single_mlp_block_ids;
  std::sort(unique_mlp_blocks.begin(), unique_mlp_blocks.end());
  if (std::adjacent_find(unique_mlp_blocks.begin(),
                         unique_mlp_blocks.end()) !=
      unique_mlp_blocks.end())
    usage_error("FLUX.2 single-MLP W8A8 block ids must be unique");
  auto unique_qk_blocks = result.w8a8_single_qk_block_ids;
  std::sort(unique_qk_blocks.begin(), unique_qk_blocks.end());
  if (std::adjacent_find(unique_qk_blocks.begin(), unique_qk_blocks.end()) !=
      unique_qk_blocks.end())
    usage_error("FLUX.2 single-QK-protected W8A8 block ids must be unique");
  auto unique_linear2_blocks = result.w8a8_single_linear2_block_ids;
  std::sort(unique_linear2_blocks.begin(), unique_linear2_blocks.end());
  if (std::adjacent_find(unique_linear2_blocks.begin(),
                         unique_linear2_blocks.end()) !=
      unique_linear2_blocks.end())
    usage_error("FLUX.2 single-linear2 W8A8 block ids must be unique");
  auto unique_double_blocks = result.w8a8_double_image_mlp_blocks;
  std::sort(unique_double_blocks.begin(), unique_double_blocks.end());
  if (std::adjacent_find(unique_double_blocks.begin(),
                         unique_double_blocks.end()) !=
      unique_double_blocks.end())
    usage_error("FLUX.2 double-stream W8A8 block ids must be unique");
  auto unique_double_mlp_blocks = result.w8a8_double_mlp_blocks;
  std::sort(unique_double_mlp_blocks.begin(), unique_double_mlp_blocks.end());
  if (std::adjacent_find(unique_double_mlp_blocks.begin(),
                         unique_double_mlp_blocks.end()) !=
      unique_double_mlp_blocks.end())
    usage_error("FLUX.2 complete double-MLP W8A8 block ids must be unique");
  auto unique_w8a8_full_double_blocks = result.w8a8_double_image_blocks;
  std::sort(unique_w8a8_full_double_blocks.begin(),
            unique_w8a8_full_double_blocks.end());
  if (std::adjacent_find(unique_w8a8_full_double_blocks.begin(),
                         unique_w8a8_full_double_blocks.end()) !=
      unique_w8a8_full_double_blocks.end())
    usage_error("FLUX.2 full double-image W8A8 block ids must be unique");
  auto unique_w8a8_full_double_text_blocks = result.w8a8_double_text_blocks;
  std::sort(unique_w8a8_full_double_text_blocks.begin(),
            unique_w8a8_full_double_text_blocks.end());
  if (std::adjacent_find(unique_w8a8_full_double_text_blocks.begin(),
                         unique_w8a8_full_double_text_blocks.end()) !=
      unique_w8a8_full_double_text_blocks.end())
    usage_error("FLUX.2 full double-text W8A8 block ids must be unique");
  auto unique_w8a8_complete_double_blocks = result.w8a8_double_blocks;
  std::sort(unique_w8a8_complete_double_blocks.begin(),
            unique_w8a8_complete_double_blocks.end());
  if (std::adjacent_find(unique_w8a8_complete_double_blocks.begin(),
                         unique_w8a8_complete_double_blocks.end()) !=
      unique_w8a8_complete_double_blocks.end())
    usage_error("FLUX.2 complete double-stream W8A8 block ids must be unique");
  auto unique_fp8_blocks = result.fp8_single_linear1_block_ids;
  std::sort(unique_fp8_blocks.begin(), unique_fp8_blocks.end());
  if (std::adjacent_find(unique_fp8_blocks.begin(), unique_fp8_blocks.end()) !=
      unique_fp8_blocks.end())
    usage_error("FLUX.2 FP8 linear1 block ids must be unique");
  auto unique_fp8_mlp_blocks = result.fp8_single_mlp_block_ids;
  std::sort(unique_fp8_mlp_blocks.begin(), unique_fp8_mlp_blocks.end());
  if (std::adjacent_find(unique_fp8_mlp_blocks.begin(),
                         unique_fp8_mlp_blocks.end()) !=
      unique_fp8_mlp_blocks.end())
    usage_error("FLUX.2 FP8 single-MLP block ids must be unique");
  auto unique_fp8_linear2_blocks = result.fp8_single_linear2_block_ids;
  std::sort(unique_fp8_linear2_blocks.begin(),
            unique_fp8_linear2_blocks.end());
  if (std::adjacent_find(unique_fp8_linear2_blocks.begin(),
                         unique_fp8_linear2_blocks.end()) !=
      unique_fp8_linear2_blocks.end())
    usage_error("FLUX.2 FP8 linear2 block ids must be unique");
  auto unique_fp8_double_blocks = result.fp8_double_image_mlp_blocks;
  std::sort(unique_fp8_double_blocks.begin(), unique_fp8_double_blocks.end());
  if (std::adjacent_find(unique_fp8_double_blocks.begin(),
                         unique_fp8_double_blocks.end()) !=
      unique_fp8_double_blocks.end())
    usage_error("FLUX.2 double-stream FP8 block ids must be unique");
  auto unique_fp8_full_double_blocks = result.fp8_double_image_blocks;
  std::sort(unique_fp8_full_double_blocks.begin(),
            unique_fp8_full_double_blocks.end());
  if (std::adjacent_find(unique_fp8_full_double_blocks.begin(),
                         unique_fp8_full_double_blocks.end()) !=
      unique_fp8_full_double_blocks.end())
    usage_error("FLUX.2 full double-image FP8 block ids must be unique");
  auto unique_fp8_full_double_text_blocks = result.fp8_double_text_blocks;
  std::sort(unique_fp8_full_double_text_blocks.begin(),
            unique_fp8_full_double_text_blocks.end());
  if (std::adjacent_find(unique_fp8_full_double_text_blocks.begin(),
                         unique_fp8_full_double_text_blocks.end()) !=
      unique_fp8_full_double_text_blocks.end())
    usage_error("FLUX.2 full double-text FP8 block ids must be unique");
  auto unique_fp8_complete_double_blocks = result.fp8_double_blocks;
  std::sort(unique_fp8_complete_double_blocks.begin(),
            unique_fp8_complete_double_blocks.end());
  if (std::adjacent_find(unique_fp8_complete_double_blocks.begin(),
                         unique_fp8_complete_double_blocks.end()) !=
      unique_fp8_complete_double_blocks.end())
    usage_error("FLUX.2 complete double-stream FP8 block ids must be unique");
  if (result.resident_plan_mib >
      std::numeric_limits<std::uint64_t>::max() / (1024ULL * 1024ULL))
    usage_error("resident plan MiB overflows bytes");
  if (result.flux2_model != "klein9b" && result.flux2_model != "dev")
    usage_error("--flux2-model must be klein9b or dev");
  if (result.transformer_checkpoint.empty())
    result.transformer_checkpoint =
        result.model_directory / "flux-2-klein-base-9b.safetensors";
  if (fs::exists(result.output_png) || fs::exists(result.report) ||
      (!result.state_output.empty() && fs::exists(result.state_output)))
    dif::fail("refusing to overwrite an output artifact");
  return result;
}

double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

dif::runtime::Tensor float_tensor(dif::ir::DType dtype,
                                  std::vector<std::uint64_t> dims,
                                  std::span<const float> values) {
  dif::runtime::Tensor result{dtype, std::move(dims), {}};
  if (result.element_count() != values.size())
    dif::fail("float tensor value count does not match its dimensions");
  result.bytes.resize(static_cast<std::size_t>(
      result.element_count() * dif::ir::dtype_size(dtype)));
  for (std::uint64_t index = 0U; index < result.element_count(); ++index)
    dif::runtime::store_float(result, index,
                              values[static_cast<std::size_t>(index)]);
  result.validate();
  return result;
}

dif::runtime::Tensor scalar(dif::ir::DType dtype, float value) {
  return float_tensor(dtype, {1U}, std::span<const float>(&value, 1U));
}

dif::runtime::Tensor repeated_scalar(dif::ir::DType dtype, float value,
                                     std::uint64_t count) {
  const std::vector<float> values(static_cast<std::size_t>(count), value);
  return float_tensor(dtype, {count}, values);
}

dif::runtime::Tensor i32_tensor(std::span<const std::int32_t> values) {
  dif::runtime::Tensor result{
      dif::ir::DType::I32,
      {static_cast<std::uint64_t>(values.size())},
      {}};
  result.bytes.resize(values.size_bytes());
  std::memcpy(result.mutable_data(), values.data(), values.size_bytes());
  result.validate();
  return result;
}

dif::runtime::Tensor bool_tensor(std::span<const std::uint8_t> values) {
  dif::runtime::Tensor result{
      dif::ir::DType::Bool,
      {1U, static_cast<std::uint64_t>(values.size())},
      std::vector<std::uint8_t>(values.begin(), values.end())};
  result.validate();
  return result;
}

std::string payload_hash(const dif::runtime::Tensor &tensor) {
  return dif::hex_digest(dif::sha256(std::span<const std::uint8_t>(
      tensor.data(), static_cast<std::size_t>(tensor.byte_size()))));
}

dif::runtime::Tensor batch_pair(const dif::runtime::Tensor &first,
                                const dif::runtime::Tensor &second) {
  if (first.dtype != second.dtype || first.dims != second.dims)
    dif::fail("cannot batch tensors with different dtype or shape");
  dif::runtime::Tensor result{first.dtype, first.dims, {}};
  result.dims.insert(result.dims.begin(), 2U);
  result.bytes.resize(static_cast<std::size_t>(first.byte_size() * 2U));
  std::memcpy(result.mutable_data(), first.data(), first.byte_size());
  std::memcpy(result.mutable_data() + first.byte_size(), second.data(),
              second.byte_size());
  result.validate();
  return result;
}

dif::runtime::Tensor batch_row(const dif::runtime::Tensor &batch,
                               std::uint64_t index) {
  if (batch.dims.empty() || index >= batch.dims.front())
    dif::fail("batched tensor row index is out of range");
  dif::runtime::Tensor result{batch.dtype, batch.dims, {}};
  result.dims.erase(result.dims.begin());
  result.bytes.resize(static_cast<std::size_t>(
      result.element_count() * dif::ir::dtype_size(result.dtype)));
  std::memcpy(result.mutable_data(),
              batch.data() + index * result.byte_size(), result.byte_size());
  result.validate();
  return result;
}

struct ConditioningResult {
  dif::runtime::Tensor positive;
  dif::runtime::Tensor negative;
  double tokenizer_ms{};
  double preparation_ms{};
  double positive_ms{};
  double negative_ms{};
  std::uint64_t resident_bytes{};
  std::string backend;
  std::string device;
  std::string fingerprint;
  std::size_t prompt_tokens{};
  std::size_t empty_tokens{};
  bool precomputed{};
};

dif::runtime::TensorMap bind_indexed_conditioner_weights(
    const fs::path &model_directory,
    const dif::frontend::Qwen3VlConditionerBuild &build) {
  const auto index_path = model_directory / "text_encoder" /
                          "model.safetensors.index.json";
  const auto index = dif::weights::read_safetensors_index(index_path);
  std::map<fs::path, dif::weights::SafeTensorFile> shards;
  dif::runtime::TensorMap result = build.generated_constants;
  for (const auto &binding : build.bindings) {
    const auto indexed = index.weight_map.find(binding.name);
    if (indexed == index.weight_map.end())
      dif::fail("text checkpoint index has no tensor " + binding.name);
    const auto path = fs::absolute(indexed->second).lexically_normal();
    auto shard = shards.find(path);
    if (shard == shards.end())
      shard = shards.emplace(path, dif::weights::read_safetensors(path)).first;
    auto tensor = dif::weights::map_safetensor(shard->second, binding.name);
    const auto *description = build.program.tensor(binding.tensor_id);
    if (!description || tensor.dtype != description->dtype ||
        tensor.dims != description->dims)
      dif::fail("text checkpoint disagrees with FLUX.2 conditioner: " +
                binding.name);
    result.emplace(binding.tensor_id, std::move(tensor));
  }
  return result;
}

void bind_prompt_inputs(
    dif::runtime::TensorMap &bindings,
    const dif::frontend::Qwen3VlConditionerBuild &build,
    const dif::frontend::Flux2PromptInputs &inputs) {
  bindings.insert_or_assign(build.token_ids_input_id,
                            i32_tensor(inputs.input_ids));
  bindings.insert_or_assign(build.attention_mask_input_id,
                            bool_tensor(inputs.attention_mask));
  std::vector<float> positions(inputs.input_ids.size());
  for (std::size_t index = 0U; index < positions.size(); ++index)
    positions[index] = static_cast<float>(index);
  bindings.insert_or_assign(
      build.position_ids_input_id,
      float_tensor(dif::ir::DType::F32,
                   {static_cast<std::uint64_t>(positions.size()), 1U},
                   positions));
}

ConditioningResult condition(const Arguments &arguments) {
  ConditioningResult result;
  if (!arguments.positive_conditioning.empty()) {
    const auto positive_file =
        dif::weights::read_safetensors(arguments.positive_conditioning);
    result.positive =
        dif::weights::map_safetensor(positive_file, "conditioning");
    if (arguments.negative_conditioning.empty()) {
      if (arguments.flux2_model != "dev")
        dif::fail("--negative-conditioning is required with "
                  "--positive-conditioning for the CFG (klein) path");
      result.negative = result.positive; // unused by the dev path
    } else {
      const auto negative_file =
          dif::weights::read_safetensors(arguments.negative_conditioning);
      result.negative =
          dif::weights::map_safetensor(negative_file, "conditioning");
    }
    const auto canonicalize = [](dif::runtime::Tensor &tensor) {
      if (tensor.dims.size() == 3U && tensor.dims.front() == 1U)
        tensor.dims.erase(tensor.dims.begin());
      tensor.validate();
    };
    canonicalize(result.positive);
    canonicalize(result.negative);
    result.precomputed = true;
    return result;
  }

  auto started = Clock::now();
  const bool dev = arguments.flux2_model == "dev";
  const auto tokenizer = dif::text::QwenBpeTokenizer::load(
      arguments.model_directory / "tokenizer" / "tokenizer.json",
      arguments.model_directory / "tokenizer" / "tokenizer_config.json");
  const auto positive_inputs =
      dev ? dif::frontend::make_flux2_mistral_prompt_inputs(tokenizer,
                                                            arguments.prompt)
          : dif::frontend::make_flux2_qwen_prompt_inputs(tokenizer,
                                                         arguments.prompt);
  const auto negative_inputs =
      dev ? positive_inputs // [dev] is guidance-distilled: no negative pass
          : dif::frontend::make_flux2_qwen_prompt_inputs(tokenizer, "");
  result.prompt_tokens = positive_inputs.valid_tokens;
  result.empty_tokens = negative_inputs.valid_tokens;
  result.tokenizer_ms = elapsed_ms(started);

  const auto config = dev ? dif::frontend::make_flux2_dev_conditioner_config()
                          : dif::frontend::make_flux2_klein_9b_conditioner_config();
  const auto build =
      dif::frontend::build_qwen3vl_conditioner_program(512U, config);
  auto bindings =
      bind_indexed_conditioner_weights(arguments.model_directory, build);
  bind_prompt_inputs(bindings, build, positive_inputs);
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 256ULL * 1024ULL * 1024ULL;
  options.cache_directory = arguments.cache_directory;
  options.profile_pipeline = arguments.profile_pipeline;
  options.cudnn_attention_heuristic =
      arguments.cudnn_attention_heuristic;
  // Qwen3-8B has a 1.244 GiB maximum streamed projection. Keep its safe
  // historical two-slot ring; the deeper measured policy below applies to
  // the FLUX transformer whose maximum staged projection is much smaller.
  options.streamed_prefetch_depth = 1U;
  options.streamed_staging_buffers = 2U;
  options.streamed_stage_threads = arguments.streamed_stage_threads;
  options.streamed_release_mapped_pages_per_copy =
      !arguments.streamed_keep_mapped_pages;
  options.streamed_keep_mapped_pages_between_runs =
      arguments.streamed_keep_mapped_pages;
  auto backend = dif::runtime::make_cuda_executor();
  started = Clock::now();
  auto prepared = backend->prepare(build.program, bindings, options);
  result.preparation_ms = elapsed_ms(started);

  started = Clock::now();
  auto positive = prepared->run(bindings, options);
  result.positive_ms = elapsed_ms(started);
  result.positive = positive.outputs.at(build.conditioning_output_id);
  result.resident_bytes = positive.resident_bytes;
  result.backend = positive.backend_name;
  result.device = positive.device_name;
  result.fingerprint =
      dif::hex_digest(dif::ir::fingerprint(build.program));

  if (dev) {
    result.negative = result.positive; // unused by the dev path
    return result;
  }
  bind_prompt_inputs(bindings, build, negative_inputs);
  started = Clock::now();
  auto negative = prepared->run(bindings, options);
  result.negative_ms = elapsed_ms(started);
  result.negative = negative.outputs.at(build.conditioning_output_id);
  return result;
}

std::uint64_t splitmix64(std::uint64_t &state) {
  state += 0x9e3779b97f4a7c15ULL;
  auto value = state;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

double uniform_open(std::uint64_t &state) {
  const auto word = splitmix64(state) >> 11U;
  return (static_cast<double>(word) + 0.5) *
         (1.0 / 9007199254740992.0);
}

dif::runtime::Tensor native_initial_latent(std::uint64_t seed,
                                           std::uint64_t height,
                                           std::uint64_t width) {
  constexpr std::uint64_t channels = 128U;
  const auto count = channels * height * width;
  std::vector<std::uint16_t> nchw(static_cast<std::size_t>(count));
  std::uint64_t state = seed;
  for (std::uint64_t index = 0U; index < count; index += 2U) {
    const auto radius = std::sqrt(-2.0 * std::log(uniform_open(state)));
    const auto angle = 6.283185307179586476925286766559 *
                       uniform_open(state);
    nchw[static_cast<std::size_t>(index)] =
        dif::runtime::float_to_bf16(
            static_cast<float>(radius * std::cos(angle)));
    if (index + 1U < count)
      nchw[static_cast<std::size_t>(index + 1U)] =
          dif::runtime::float_to_bf16(
              static_cast<float>(radius * std::sin(angle)));
  }

  dif::runtime::Tensor result{
      dif::ir::DType::BF16, {height * width, channels}, {}};
  result.bytes.resize(static_cast<std::size_t>(result.element_count() *
                                                sizeof(std::uint16_t)));
  auto *tokens = reinterpret_cast<std::uint16_t *>(result.mutable_data());
  for (std::uint64_t y = 0U; y < height; ++y)
    for (std::uint64_t x = 0U; x < width; ++x)
      for (std::uint64_t channel = 0U; channel < channels; ++channel) {
        const auto source = (channel * height + y) * width + x;
        const auto target = (y * width + x) * channels + channel;
        tokens[target] = nchw[static_cast<std::size_t>(source)];
      }
  result.validate();
  return result;
}

dif::runtime::Tensor initial_latent(const Arguments &arguments,
                                    std::uint64_t latent_height,
                                    std::uint64_t latent_width) {
  if (arguments.initial_latent.empty())
    return native_initial_latent(arguments.seed, latent_height, latent_width);
  const auto file = dif::weights::read_safetensors(arguments.initial_latent);
  auto result =
      dif::weights::map_safetensor(file, arguments.initial_latent_tensor);
  if (result.dtype != dif::ir::DType::BF16 ||
      result.dims != std::vector<std::uint64_t>{latent_height * latent_width,
                                                128U})
    dif::fail("initial latent must contain BF16 initial_image_tokens [H*W,128]");
  return result;
}

dif::runtime::Tensor position_ids(std::uint64_t image_tokens,
                                  std::uint64_t text_tokens,
                                  std::uint64_t latent_width,
                                  std::uint64_t batch_size = 1U) {
  const auto total = image_tokens + text_tokens;
  std::vector<float> values(
      static_cast<std::size_t>(batch_size * total * 4U), 0.0F);
  for (std::uint64_t batch = 0U; batch < batch_size; ++batch) {
    const auto base = batch * total * 4U;
    for (std::uint64_t token = 0U; token < text_tokens; ++token)
      values[static_cast<std::size_t>(base + token * 4U + 3U)] =
          static_cast<float>(token);
    for (std::uint64_t token = 0U; token < image_tokens; ++token) {
      const auto row = text_tokens + token;
      values[static_cast<std::size_t>(base + row * 4U + 1U)] =
          static_cast<float>(token / latent_width);
      values[static_cast<std::size_t>(base + row * 4U + 2U)] =
          static_cast<float>(token % latent_width);
    }
  }
  return float_tensor(dif::ir::DType::F32,
                      {batch_size, total, 4U}, values);
}

struct DenoiseResult {
  dif::runtime::Tensor latent;
  std::optional<dif::runtime::Tensor> first_step_latent;
  std::optional<dif::runtime::Tensor> middle_step_latent;
  std::vector<std::pair<std::uint32_t, dif::runtime::Tensor>>
      periodic_step_latents;
  std::vector<double> cfg_transformer_ms;
  std::vector<double> scheduler_ms;
  double preparation_ms{};
  double residency_plan_ms{};
  std::uint64_t resident_bytes{};
  std::uint64_t resident_weight_bytes{};
  std::uint64_t streamed_weight_bytes{};
  std::uint64_t residency_required_bytes{};
  std::size_t resident_weight_tensors{};
  std::string backend;
  std::string device;
  std::string fingerprint;
  std::vector<dif::runtime::LinearTuningResult> linear_tuning_results;
  std::vector<dif::runtime::LinearAlgorithmChoice>
      selected_linear_algorithms;
  std::uint32_t w8a8_linear_count{};
  std::uint64_t w8a8_weight_bytes{};
  dif::ir::Int8RowQuantization w8a8_quantization{
      dif::ir::Int8RowQuantization::Direct};
  dif::ir::Int8RowQuantization w8a8_single_mlp_quantization{
      dif::ir::Int8RowQuantization::Direct};
  std::vector<std::uint32_t> w8a8_blocks;
  std::vector<std::uint32_t> w8a8_single_mlp_blocks;
  std::vector<std::uint32_t> w8a8_single_qk_blocks;
  std::vector<std::uint32_t> w8a8_single_linear2_blocks;
  std::vector<std::uint32_t> w8a8_double_image_mlp_blocks;
  std::vector<std::uint32_t> w8a8_double_mlp_blocks;
  std::vector<std::uint32_t> w8a8_double_image_blocks;
  std::vector<std::uint32_t> w8a8_double_text_blocks;
  std::vector<std::uint32_t> w8a8_double_blocks;
  std::uint32_t fp8_linear_count{};
  std::uint64_t fp8_weight_bytes{};
  std::vector<std::uint32_t> fp8_single_linear1_blocks;
  std::vector<std::uint32_t> fp8_single_mlp_blocks;
  std::vector<std::uint32_t> fp8_single_linear2_blocks;
  std::vector<std::uint32_t> fp8_double_image_mlp_blocks;
  std::vector<std::uint32_t> fp8_double_image_blocks;
  std::vector<std::uint32_t> fp8_double_text_blocks;
  std::vector<std::uint32_t> fp8_double_blocks;
  bool fp8_row_scaled{};
  std::uint32_t int8_weight_only_linear_count{};
  dif::frontend::SquareQW4RewriteResult squareq_w4;
  std::uint32_t int8_weight_only_group_size{64U};
  std::uint64_t int8_weight_only_bytes{};
  bool int8_weight_only_row_scaled{};
  std::vector<std::string> int8_weight_only_group32_names;
};

struct W8A8RewriteResult {
  std::uint32_t linear_count{};
  std::uint64_t quantized_weight_bytes{};
};

using Fp8RewriteResult = W8A8RewriteResult;

void apply_convrot(std::vector<float> &values, std::size_t rotation_group) {
  if ((rotation_group != 256U && rotation_group != 4096U) ||
      values.size() % rotation_group != 0U)
    dif::fail("ConvRot input width disagrees with its rotation group");
  for (const std::size_t stride : {1U, 4U, 16U, 64U, 256U, 1024U}) {
    if (stride >= rotation_group)
      break;
    for (std::size_t group = 0U; group < values.size();
         group += rotation_group) {
      for (std::size_t lane = 0U; lane < rotation_group / 4U; ++lane) {
        const auto index = group + (lane % stride) +
                           (lane / stride) * (4U * stride);
        const auto x0 = values[index];
        const auto x1 = values[index + stride];
        const auto x2 = values[index + 2U * stride];
        const auto x3 = values[index + 3U * stride];
        values[index] = 0.5F * (x0 + x1 + x2 - x3);
        values[index + stride] = 0.5F * (x0 + x1 - x2 + x3);
        values[index + 2U * stride] = 0.5F * (x0 - x1 + x2 + x3);
        values[index + 3U * stride] = 0.5F * (-x0 + x1 + x2 + x3);
      }
    }
  }
}

std::uint64_t convrot_group(dif::ir::Int8RowQuantization quantization,
                            std::uint64_t) {
  if (quantization == dif::ir::Int8RowQuantization::H4096SignedConvRot ||
      quantization == dif::ir::Int8RowQuantization::H4096F32SignedConvRot)
    return 4096U;
  return 256U;
}

float signed_convrot_sign(std::uint64_t column) {
  auto hash = static_cast<std::uint32_t>(column) + 0x9e3779b9U;
  hash = (hash ^ (hash >> 16U)) * 0x7feb352dU;
  hash = (hash ^ (hash >> 15U)) * 0x846ca68bU;
  hash ^= hash >> 16U;
  return (hash & 1U) != 0U ? -1.0F : 1.0F;
}

std::string_view quantization_name(
    dif::ir::Int8RowQuantization quantization) {
  if (quantization == dif::ir::Int8RowQuantization::H256ConvRot)
    return "h256-convrot";
  if (quantization == dif::ir::Int8RowQuantization::H256SignedConvRot)
    return "h256-signed-coherent-convrot";
  if (quantization == dif::ir::Int8RowQuantization::H4096SignedConvRot)
    return "h4096-signed-coherent-convrot";
  if (quantization == dif::ir::Int8RowQuantization::H256F32ConvRot)
    return "h256-f32-convrot";
  if (quantization == dif::ir::Int8RowQuantization::H256F32SignedConvRot)
    return "h256-f32-signed-convrot";
  if (quantization == dif::ir::Int8RowQuantization::H4096F32SignedConvRot)
    return "h4096-f32-signed-convrot";
  return "direct";
}

// Offline frontend packing policy. This deliberately remains outside DiffIR:
// the runtime sees the same generic I8 weight, row scale, activation
// quantization, and scaled-linear semantics regardless of how the scale was
// selected. The search matches the BF16 rounding used by the native ConvRot
// execution path and trades a small amount of clipping for lower row MSE.
float choose_mse_int8_weight_scale(std::span<const float> values) {
  float maximum = 0.0F;
  for (const auto value : values)
    maximum = std::max(maximum, std::fabs(value));
  if (maximum == 0.0F)
    return 1.0e-30F;
  constexpr std::array<float, 7U> clip_factors{
      1.0F, 0.96F, 0.92F, 0.88F, 0.84F, 0.80F, 0.76F};
  auto best_scale = std::numeric_limits<float>::infinity();
  auto best_error = std::numeric_limits<double>::infinity();
  for (const auto factor : clip_factors) {
    const auto scale = dif::runtime::bf16_to_float(
        dif::runtime::float_to_bf16(
            std::max(maximum * factor / 127.0F, 1.0e-30F)));
    double error = 0.0;
    // Deterministic quarter-row coverage keeps offline packing bounded while
    // sampling every transform lane across each H256 group.
    for (std::size_t index = 0U; index < values.size(); index += 4U) {
      const auto value = dif::runtime::bf16_to_float(
          dif::runtime::float_to_bf16(values[index]));
      const auto divided = dif::runtime::bf16_to_float(
          dif::runtime::float_to_bf16(value / scale));
      const auto encoded = std::clamp(
          static_cast<int>(std::nearbyint(divided)), -128, 127);
      const auto delta =
          static_cast<double>(value) - static_cast<double>(encoded) * scale;
      error += delta * delta;
    }
    if (error < best_error) {
      best_error = error;
      best_scale = scale;
    }
  }
  return best_scale;
}

W8A8RewriteResult rewrite_all_linear_weights_int8(
    dif::ir::Program &program, dif::runtime::TensorMap &bindings,
    const dif::frontend::Flux2KleinTransformerBuild &transformer,
    const fs::path &cache_directory, std::uint64_t group_size,
    std::span<const std::string> excluded_names = {},
    std::span<const std::string> included_names = {}) {
  W8A8RewriteResult result;
  auto next_tensor = std::uint32_t{0U};
  auto next_operation = std::uint32_t{0U};
  for (const auto &tensor : program.tensors)
    next_tensor = std::max(next_tensor, tensor.id);
  for (const auto &operation : program.operations)
    next_operation = std::max(next_operation, operation.id);

  std::vector<std::pair<std::string, std::uint32_t>> selected;
  std::unordered_set<std::uint32_t> seen;
  std::unordered_set<std::string_view> excluded(excluded_names.begin(),
                                                excluded_names.end());
  std::unordered_set<std::string_view> included(included_names.begin(),
                                                included_names.end());
  std::unordered_set<std::string_view> resolved_exclusions;
  std::unordered_set<std::string_view> resolved_inclusions;
  for (std::size_t index = 0U;
       index < transformer.checkpoint_tensors.size(); ++index) {
    const auto &name = transformer.checkpoint_names.at(index);
    const auto weight_id = transformer.checkpoint_tensors.at(index);
    if (!seen.insert(weight_id).second)
      continue;
    const auto *description = program.tensor(weight_id);
    if (!description || description->dtype != dif::ir::DType::BF16 ||
        description->dims.size() != 2U ||
        description->dims[1] % group_size != 0U)
      continue;
    const auto linear = std::find_if(
        program.operations.begin(), program.operations.end(),
        [&](const dif::ir::Operation &operation) {
          return operation.opcode == dif::ir::Opcode::Linear &&
                 operation.inputs.size() == 2U &&
                 operation.inputs[1] == weight_id;
        });
    if (linear == program.operations.end())
      continue;
    if (!included.empty() && !included.contains(name))
      continue;
    if (!included.empty())
      resolved_inclusions.insert(name);
    if (excluded.contains(name)) {
      resolved_exclusions.insert(name);
      continue;
    }
    selected.emplace_back(name, weight_id);
  }
  if (resolved_exclusions.size() != excluded.size())
    dif::fail("FLUX.2 weight-only INT8 exclusion did not resolve an eligible "
              "remaining Linear weight");
  if (resolved_inclusions.size() != included.size())
    dif::fail("FLUX.2 weight-only INT8 inclusion did not resolve an eligible "
              "remaining Linear weight");
  if (selected.empty())
    dif::fail("FLUX.2 weight-only INT8 found no eligible Linear weights");

  fs::create_directories(cache_directory);
  for (const auto &[name, weight_id] : selected) {
    const auto weight_desc = *program.tensor(weight_id);
    const auto binding = bindings.find(weight_id);
    if (binding == bindings.end())
      dif::fail("FLUX.2 weight-only INT8 weight is not bound");
    const auto rows = weight_desc.dims[0];
    const auto columns = weight_desc.dims[1];
    const auto groups = columns / group_size;
    const std::vector<std::uint64_t> scale_dims{rows, groups};
    const auto name_hash = dif::hex_digest(dif::sha256(
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t *>(name.data()),
            name.size())));
    const auto cache_path =
        cache_directory /
        ("flux2-klein-base-9b-" + std::string(kModelRevision) +
         "-weight-" + name_hash.substr(0U, 16U) +
         "-int8-group" + std::to_string(group_size) +
         "-bf16-dequant.safetensors");
    if (!fs::exists(cache_path)) {
      dif::runtime::Tensor quantized{dif::ir::DType::I8, weight_desc.dims,
                                     {}};
      quantized.bytes.resize(
          static_cast<std::size_t>(weight_desc.element_count()));
      dif::runtime::Tensor scales{dif::ir::DType::F32, scale_dims, {}};
      scales.bytes.resize(
          static_cast<std::size_t>(scales.element_count() * sizeof(float)));
      auto *encoded = reinterpret_cast<std::int8_t *>(
          quantized.mutable_data());
      const auto encode_rows = [&](std::uint64_t begin, std::uint64_t end) {
        for (auto row = begin; row < end; ++row) {
          const auto base = row * columns;
          for (std::uint64_t group = 0U; group < groups; ++group) {
            const auto group_begin = group * group_size;
            float maximum = 0.0F;
            for (std::uint64_t lane = 0U; lane < group_size; ++lane)
              maximum = std::max(
                  maximum,
                  std::fabs(dif::runtime::load_float(
                      binding->second, base + group_begin + lane)));
            const auto scale = std::max(maximum / 127.0F, 1.0e-30F);
            dif::runtime::store_float(scales, row * groups + group, scale);
            for (std::uint64_t lane = 0U; lane < group_size; ++lane) {
              const auto offset = base + group_begin + lane;
              const auto rounded = static_cast<int>(std::nearbyint(
                  dif::runtime::load_float(binding->second, offset) / scale));
              encoded[offset] = static_cast<std::int8_t>(
                  std::clamp(rounded, -127, 127));
            }
          }
        }
      };
      const auto workers_count = std::min<std::uint64_t>(
          rows, std::max(1U, std::thread::hardware_concurrency()));
      std::vector<std::thread> workers;
      workers.reserve(static_cast<std::size_t>(workers_count));
      for (std::uint64_t worker = 0U; worker < workers_count; ++worker)
        workers.emplace_back(encode_rows, rows * worker / workers_count,
                             rows * (worker + 1U) / workers_count);
      for (auto &worker : workers)
        worker.join();
      auto building_path = cache_path;
      building_path += ".building";
      if (fs::exists(building_path))
        dif::fail("refusing to overwrite an incomplete FLUX.2 groupwise "
                  "INT8 cache: " +
                  building_path.string());
      dif::weights::SafeTensorWriter writer(
          building_path,
          {{"weight", quantized.dtype, quantized.dims},
           {"scale", scales.dtype, scales.dims}});
      writer.append("weight", std::span<const std::uint8_t>(
                                  quantized.data(), quantized.byte_size()));
      writer.append("scale", std::span<const std::uint8_t>(
                                 scales.data(), scales.byte_size()));
      (void)writer.finish();
      fs::rename(building_path, cache_path);
      std::cerr << "FLUX2_INT8_WEIGHT_CACHE_BUILD name=" << name
                << " path=" << cache_path << '\n';
    }
    const auto cache = dif::weights::read_safetensors(cache_path);
    auto quantized = dif::weights::map_safetensor(cache, "weight");
    auto scales = dif::weights::map_safetensor(cache, "scale");
    if (quantized.dtype != dif::ir::DType::I8 ||
        quantized.dims != weight_desc.dims ||
        scales.dtype != dif::ir::DType::F32 || scales.dims != scale_dims)
      dif::fail("FLUX.2 groupwise INT8 cache shape/dtype disagreement");

    const auto linear = std::find_if(
        program.operations.begin(), program.operations.end(),
        [&](const dif::ir::Operation &operation) {
          return operation.opcode == dif::ir::Opcode::Linear &&
                 operation.inputs.size() == 2U &&
                 operation.inputs[1] == weight_id;
        });
    if (linear == program.operations.end())
      dif::fail("FLUX.2 weight-only INT8 lost its Linear consumer");
    const auto linear_index = static_cast<std::size_t>(
        std::distance(program.operations.begin(), linear));
    const auto quantized_id = ++next_tensor;
    const auto scales_id = ++next_tensor;
    const auto dequantized_id = ++next_tensor;
    program.tensors.push_back({quantized_id, dif::ir::DType::I8,
                               weight_desc.roles, weight_desc.dims});
    program.tensors.push_back({scales_id, dif::ir::DType::F32,
                               weight_desc.roles, scale_dims});
    program.tensors.push_back({dequantized_id, dif::ir::DType::BF16,
                               dif::ir::TensorRole::Internal,
                               weight_desc.dims});
    bindings.emplace(quantized_id, std::move(quantized));
    bindings.emplace(scales_id, std::move(scales));
    bindings.erase(weight_id);
    auto rewritten_linear = *linear;
    rewritten_linear.inputs[1] = dequantized_id;
    const dif::ir::Operation dequantize{
        ++next_operation, dif::ir::Opcode::DequantizeInt8Blocks,
        {quantized_id, scales_id}, {dequantized_id},
        {dif::ir::Attribute::u64(dif::ir::AttrKey::BlockSize, group_size)}};
    program.operations.at(linear_index) = dequantize;
    program.operations.insert(
        program.operations.begin() +
            static_cast<std::ptrdiff_t>(linear_index + 1U),
        std::move(rewritten_linear));
    program.tensors.erase(
        std::remove_if(program.tensors.begin(), program.tensors.end(),
                       [&](const dif::ir::TensorDesc &description) {
                         return description.id == weight_id;
                       }),
        program.tensors.end());
    ++result.linear_count;
    result.quantized_weight_bytes +=
        weight_desc.element_count() + rows * groups * sizeof(float);
  }
  dif::ir::verify(program);
  return result;
}

W8A8RewriteResult rewrite_all_linear_weights_int8_row_scaled(
    dif::ir::Program &program, dif::runtime::TensorMap &bindings,
    const dif::frontend::Flux2KleinTransformerBuild &transformer,
    const fs::path &cache_directory) {
  W8A8RewriteResult result;
  auto next_tensor = std::uint32_t{0U};
  for (const auto &tensor : program.tensors)
    next_tensor = std::max(next_tensor, tensor.id);

  std::vector<std::pair<std::string, std::uint32_t>> selected;
  std::unordered_set<std::uint32_t> seen;
  for (std::size_t index = 0U;
       index < transformer.checkpoint_tensors.size(); ++index) {
    const auto weight_id = transformer.checkpoint_tensors.at(index);
    if (!seen.insert(weight_id).second)
      continue;
    const auto *description = program.tensor(weight_id);
    if (!description || description->dtype != dif::ir::DType::BF16 ||
        description->dims.size() != 2U)
      continue;
    const auto linear = std::find_if(
        program.operations.begin(), program.operations.end(),
        [&](const dif::ir::Operation &operation) {
          return operation.opcode == dif::ir::Opcode::Linear &&
                 operation.inputs.size() == 2U &&
                 operation.inputs[1] == weight_id;
        });
    if (linear != program.operations.end())
      selected.emplace_back(transformer.checkpoint_names.at(index), weight_id);
  }
  if (selected.empty())
    dif::fail("FLUX.2 row-scaled weight-only INT8 found no eligible Linears");

  fs::create_directories(cache_directory);
  for (const auto &[name, weight_id] : selected) {
    const auto weight_desc = *program.tensor(weight_id);
    const auto binding = bindings.find(weight_id);
    if (binding == bindings.end())
      dif::fail("FLUX.2 row-scaled INT8 weight is not bound");
    const auto rows = weight_desc.dims[0];
    const auto columns = weight_desc.dims[1];
    const auto name_hash = dif::hex_digest(dif::sha256(
        std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t *>(name.data()),
            name.size())));
    const auto cache_path =
        cache_directory /
        ("flux2-klein-base-9b-" + std::string(kModelRevision) +
         "-weight-" + name_hash.substr(0U, 16U) +
         "-int8-row-scaled.safetensors");
    if (!fs::exists(cache_path)) {
      dif::runtime::Tensor quantized{dif::ir::DType::I8, weight_desc.dims, {}};
      quantized.bytes.resize(
          static_cast<std::size_t>(weight_desc.element_count()));
      dif::runtime::Tensor scales{dif::ir::DType::F32, {rows}, {}};
      scales.bytes.resize(static_cast<std::size_t>(rows * sizeof(float)));
      auto *encoded =
          reinterpret_cast<std::int8_t *>(quantized.mutable_data());
      const auto encode_rows = [&](std::uint64_t begin, std::uint64_t end) {
        for (auto row = begin; row < end; ++row) {
          const auto base = row * columns;
          float maximum = 0.0F;
          for (std::uint64_t column = 0U; column < columns; ++column)
            maximum = std::max(
                maximum,
                std::fabs(dif::runtime::load_float(binding->second,
                                                   base + column)));
          const auto scale = std::max(maximum / 127.0F, 1.0e-30F);
          dif::runtime::store_float(scales, row, scale);
          for (std::uint64_t column = 0U; column < columns; ++column) {
            const auto rounded = static_cast<int>(std::nearbyint(
                dif::runtime::load_float(binding->second, base + column) /
                scale));
            encoded[base + column] = static_cast<std::int8_t>(
                std::clamp(rounded, -127, 127));
          }
        }
      };
      const auto worker_count = std::min<std::uint64_t>(
          rows, std::max(1U, std::thread::hardware_concurrency()));
      std::vector<std::thread> workers;
      workers.reserve(static_cast<std::size_t>(worker_count));
      for (std::uint64_t worker = 0U; worker < worker_count; ++worker)
        workers.emplace_back(encode_rows, rows * worker / worker_count,
                             rows * (worker + 1U) / worker_count);
      for (auto &worker : workers)
        worker.join();
      auto building_path = cache_path;
      building_path += ".building";
      if (fs::exists(building_path))
        dif::fail("refusing to overwrite an incomplete FLUX.2 row-scaled "
                  "INT8 cache: " + building_path.string());
      dif::weights::SafeTensorWriter writer(
          building_path,
          {{"weight", quantized.dtype, quantized.dims},
           {"scale", scales.dtype, scales.dims}});
      writer.append("weight", std::span<const std::uint8_t>(
                                  quantized.data(), quantized.byte_size()));
      writer.append("scale", std::span<const std::uint8_t>(
                                 scales.data(), scales.byte_size()));
      (void)writer.finish();
      fs::rename(building_path, cache_path);
      std::cerr << "FLUX2_INT8_ROW_WEIGHT_CACHE_BUILD name=" << name
                << " path=" << cache_path << '\n';
    }
    const auto cache = dif::weights::read_safetensors(cache_path);
    auto quantized = dif::weights::map_safetensor(cache, "weight");
    auto scales = dif::weights::map_safetensor(cache, "scale");
    if (quantized.dtype != dif::ir::DType::I8 ||
        quantized.dims != weight_desc.dims ||
        scales.dtype != dif::ir::DType::F32 ||
        scales.dims != std::vector<std::uint64_t>{rows})
      dif::fail("FLUX.2 row-scaled INT8 cache shape/dtype disagreement");

    const auto linear = std::find_if(
        program.operations.begin(), program.operations.end(),
        [&](const dif::ir::Operation &operation) {
          return operation.opcode == dif::ir::Opcode::Linear &&
                 operation.inputs.size() == 2U &&
                 operation.inputs[1] == weight_id;
        });
    if (linear == program.operations.end())
      dif::fail("FLUX.2 row-scaled INT8 lost its Linear consumer");
    const auto quantized_id = ++next_tensor;
    const auto scales_id = ++next_tensor;
    program.tensors.push_back({quantized_id, dif::ir::DType::I8,
                               weight_desc.roles, weight_desc.dims});
    program.tensors.push_back({scales_id, dif::ir::DType::F32,
                               weight_desc.roles, {rows}});
    bindings.emplace(quantized_id, std::move(quantized));
    bindings.emplace(scales_id, std::move(scales));
    bindings.erase(weight_id);
    linear->opcode = dif::ir::Opcode::LinearInt8WeightScaled;
    linear->inputs = {linear->inputs[0], quantized_id, scales_id};
    program.tensors.erase(
        std::remove_if(program.tensors.begin(), program.tensors.end(),
                       [&](const dif::ir::TensorDesc &description) {
                         return description.id == weight_id;
                       }),
        program.tensors.end());
    ++result.linear_count;
    result.quantized_weight_bytes +=
        weight_desc.element_count() + rows * sizeof(float);
  }
  dif::ir::verify(program);
  return result;
}

W8A8RewriteResult rewrite_named_linear_w8a8(
    dif::ir::Program &program, dif::runtime::TensorMap &bindings,
    const dif::frontend::Flux2KleinTransformerBuild &transformer,
    std::span<const std::uint32_t> block_ids,
    dif::ir::Int8RowQuantization quantization,
    std::string_view checkpoint_prefix, std::string_view checkpoint_suffix,
    std::string_view cache_label, const fs::path &cache_directory,
    bool weight_equalization = false, bool mse_weight_scale = false,
    bool activation_residual2 = false) {
  W8A8RewriteResult result;
  if (block_ids.empty())
    return result;
  const auto convrot = quantization != dif::ir::Int8RowQuantization::Direct;
  const auto f32_convrot =
      quantization == dif::ir::Int8RowQuantization::H256F32ConvRot ||
      quantization == dif::ir::Int8RowQuantization::H256F32SignedConvRot ||
      quantization == dif::ir::Int8RowQuantization::H4096F32SignedConvRot;
  const auto signed_convrot =
      quantization == dif::ir::Int8RowQuantization::H256SignedConvRot ||
      quantization == dif::ir::Int8RowQuantization::H256F32SignedConvRot ||
      quantization == dif::ir::Int8RowQuantization::H4096SignedConvRot ||
      quantization == dif::ir::Int8RowQuantization::H4096F32SignedConvRot;
  auto next_tensor = std::uint32_t{0U};
  auto next_operation = std::uint32_t{0U};
  for (const auto &tensor : program.tensors)
    next_tensor = std::max(next_tensor, tensor.id);
  for (const auto &operation : program.operations)
    next_operation = std::max(next_operation, operation.id);

  std::vector<std::pair<std::uint32_t, std::uint32_t>> selected_weights;
  std::unordered_set<std::uint32_t> seen_weights;
  const std::unordered_set<std::uint32_t> requested_blocks(
      block_ids.begin(), block_ids.end());
  for (std::size_t index = 0U;
       index < transformer.checkpoint_names.size(); ++index) {
    const auto &name = transformer.checkpoint_names.at(index);
    if (!name.starts_with(checkpoint_prefix) ||
        !name.ends_with(checkpoint_suffix))
      continue;
    const auto block_end = name.find('.', checkpoint_prefix.size());
    if (block_end == std::string::npos)
      dif::fail("malformed FLUX.2 W8A8 checkpoint name");
    const auto block = static_cast<std::uint32_t>(std::stoul(name.substr(
        checkpoint_prefix.size(), block_end - checkpoint_prefix.size())));
    if (!requested_blocks.contains(block))
      continue;
    const auto weight = transformer.checkpoint_tensors.at(index);
    if (seen_weights.insert(weight).second)
      selected_weights.emplace_back(block, weight);
  }
  if (selected_weights.size() != block_ids.size())
    dif::fail("FLUX.2 W8A8 selection did not resolve every requested block");

  fs::create_directories(cache_directory);
  for (const auto [block, weight_id] : selected_weights) {
    const auto *weight_description = program.tensor(weight_id);
    const auto binding = bindings.find(weight_id);
    if (!weight_description || binding == bindings.end() ||
        weight_description->dtype != dif::ir::DType::BF16 ||
        weight_description->dims.size() != 2U)
      dif::fail("FLUX.2 W8A8 weight is not a bound rank-2 BF16 tensor");
    const auto weight_desc = *weight_description;
    const auto rotation_group =
        convrot_group(quantization, weight_desc.dims.at(1));
    const auto linear = std::find_if(
        program.operations.begin(), program.operations.end(),
        [&](const dif::ir::Operation &operation) {
          return operation.opcode == dif::ir::Opcode::Linear &&
                 operation.inputs.size() == 2U &&
                 operation.inputs.at(1) == weight_id;
        });
    if (linear == program.operations.end())
      dif::fail("FLUX.2 W8A8 weight has no exclusive unbiased Linear");
    const auto linear_id = linear->id;
    const auto linear_index = static_cast<std::size_t>(
        std::distance(program.operations.begin(), linear));
    const auto *input_description = program.tensor(linear->inputs.at(0));
    const auto *output_description = program.tensor(linear->outputs.at(0));
    if (!input_description || !output_description ||
        input_description->dtype != dif::ir::DType::BF16 ||
        output_description->dtype != dif::ir::DType::BF16)
      dif::fail("FLUX.2 W8A8 Linear input is not BF16");
    const auto input_desc = *input_description;
    const auto output_desc = *output_description;
    const auto linear_input_id = linear->inputs.at(0);
    const dif::ir::Operation *fused_concat = nullptr;
    const dif::ir::Operation *fused_reshape = nullptr;
    if (!weight_equalization) {
      const auto reshape = std::find_if(
          program.operations.begin(), program.operations.end(),
          [&](const dif::ir::Operation &operation) {
            return operation.opcode == dif::ir::Opcode::Reshape &&
                   operation.outputs ==
                       std::vector<std::uint32_t>{linear_input_id};
          });
      auto concat_output_id = linear_input_id;
      if (reshape != program.operations.end()) {
        const auto reshape_consumers = static_cast<std::size_t>(std::count_if(
            program.operations.begin(), program.operations.end(),
            [&](const dif::ir::Operation &operation) {
              return std::find(operation.inputs.begin(), operation.inputs.end(),
                               linear_input_id) != operation.inputs.end();
            }));
        if (reshape_consumers == 1U) {
          fused_reshape = &*reshape;
          concat_output_id = reshape->inputs.front();
        }
      }
      const auto producer = std::find_if(
          program.operations.begin(), program.operations.end(),
          [&](const dif::ir::Operation &operation) {
            return operation.opcode == dif::ir::Opcode::Concat &&
                   operation.outputs ==
                       std::vector<std::uint32_t>{concat_output_id};
          });
      const auto consumer_count = static_cast<std::size_t>(std::count_if(
          program.operations.begin(), program.operations.end(),
          [&](const dif::ir::Operation &operation) {
            return std::find(operation.inputs.begin(), operation.inputs.end(),
                             concat_output_id) != operation.inputs.end();
          }));
      const auto *concat_output = program.tensor(concat_output_id);
      if (producer != program.operations.end() && concat_output &&
          consumer_count == 1U &&
          producer->u64(dif::ir::AttrKey::Axis,
                        std::numeric_limits<std::uint64_t>::max()) +
                  1U ==
              concat_output->dims.size() &&
          std::all_of(producer->inputs.begin(), producer->inputs.end(),
                      [&](const std::uint32_t input_id) {
                        const auto *part = program.tensor(input_id);
                        return part && !part->dims.empty() &&
                               part->element_count() / part->dims.back() ==
                                   input_desc.element_count() /
                                       input_desc.dims.back() &&
                               part->dims.back() % rotation_group == 0U;
                      }))
        fused_concat = &*producer;
    }
    const auto fused_concat_id =
        fused_concat ? fused_concat->id : std::uint32_t{0U};
    const auto fused_reshape_id =
        fused_reshape ? fused_reshape->id : std::uint32_t{0U};
    const auto fused_concat_output_id =
        fused_concat ? fused_concat->outputs.front() : std::uint32_t{0U};
    if (convrot && weight_desc.dims.at(1) % rotation_group != 0U)
      dif::fail("FLUX.2 ConvRot weight width disagrees with its group");

    auto cache_path =
        cache_directory /
        ("flux2-klein-base-9b-" + std::string(kModelRevision) +
         "-" + std::string(cache_label) + "-" + std::to_string(block) +
         (cache_label == "single-linear2-block"
              ? (quantization ==
                         dif::ir::Int8RowQuantization::H4096SignedConvRot
                     ? "-linear2-h4096-signed-coherent-convrot-w8a8.safetensors"
                     : quantization ==
                               dif::ir::Int8RowQuantization::H4096F32SignedConvRot
                     ? "-linear2-h4096-f32-signed-convrot-w8a8.safetensors"
                     : f32_convrot
                     ? (signed_convrot
                            ? "-linear2-h256-f32-signed-convrot-w8a8.safetensors"
                            : "-linear2-h256-f32-convrot-w8a8.safetensors")
                     : signed_convrot
                     ? "-linear2-h256-signed-coherent-convrot-w8a8.safetensors"
                     : (convrot ? "-linear2-h256-convrot-w8a8.safetensors"
                                : "-linear2-w8a8.safetensors"))
              : (quantization ==
                         dif::ir::Int8RowQuantization::H4096SignedConvRot
                     ? "-linear1-h4096-signed-coherent-convrot-w8a8.safetensors"
                     : quantization ==
                               dif::ir::Int8RowQuantization::H4096F32SignedConvRot
                     ? "-linear1-h4096-f32-signed-convrot-w8a8.safetensors"
                     : f32_convrot
                     ? (signed_convrot
                            ? "-linear1-h256-f32-signed-convrot-w8a8.safetensors"
                            : "-linear1-h256-f32-convrot-w8a8.safetensors")
                     : signed_convrot
                     ? "-linear1-h256-signed-coherent-convrot-w8a8.safetensors"
                     : (convrot ? "-linear1-h256-convrot-w8a8.safetensors"
                                : "-linear1-w8a8.safetensors"))));
    if (weight_equalization)
      cache_path = cache_path.parent_path() /
                   (cache_path.stem().string() +
                    "-weight-equalized.safetensors");
    if (mse_weight_scale)
      cache_path = cache_path.parent_path() /
                   (cache_path.stem().string() +
                    "-mse-weight-scale.safetensors");
    if (!fs::exists(cache_path)) {
      dif::runtime::Tensor quantized_weight{
          dif::ir::DType::I8, weight_desc.dims, {}};
      quantized_weight.bytes.resize(
          static_cast<std::size_t>(weight_desc.element_count()));
      dif::runtime::Tensor weight_scales{
          dif::ir::DType::F32, {weight_desc.dims.at(0)}, {}};
      weight_scales.bytes.resize(static_cast<std::size_t>(
          weight_desc.dims.at(0) * sizeof(float)));
      dif::runtime::Tensor input_equalization{
          dif::ir::DType::BF16, {weight_desc.dims.at(1)}, {}};
      if (weight_equalization) {
        input_equalization.bytes.resize(static_cast<std::size_t>(
            weight_desc.dims.at(1) * sizeof(std::uint16_t)));
        std::vector<float> column_maxima(
            static_cast<std::size_t>(weight_desc.dims.at(1)), 0.0F);
        for (std::uint64_t row = 0U; row < weight_desc.dims.at(0); ++row)
          for (std::uint64_t column = 0U;
               column < weight_desc.dims.at(1); ++column)
            column_maxima[static_cast<std::size_t>(column)] = std::max(
                column_maxima[static_cast<std::size_t>(column)],
                std::fabs(dif::runtime::load_float(
                    binding->second,
                    row * weight_desc.dims.at(1) + column)));
        double log_sum = 0.0;
        for (const auto maximum : column_maxima)
          log_sum += std::log(std::max(maximum, 1.0e-30F));
        const auto center = static_cast<float>(
            std::exp(log_sum / static_cast<double>(column_maxima.size())));
        for (std::uint64_t column = 0U;
             column < weight_desc.dims.at(1); ++column) {
          const auto scale = std::clamp(
              std::sqrt(column_maxima[static_cast<std::size_t>(column)] /
                        std::max(center, 1.0e-30F)),
              0.5F, 2.0F);
          dif::runtime::store_float(input_equalization, column, scale);
        }
      }
      auto *quantized = reinterpret_cast<std::int8_t *>(
          quantized_weight.mutable_data());
      const auto columns = weight_desc.dims.at(1);
      const auto encode_rows = [&](std::uint64_t begin, std::uint64_t end) {
        std::vector<float> transformed(static_cast<std::size_t>(columns));
        for (auto row = begin; row < end; ++row) {
          const auto base = row * columns;
          float maximum = 0.0F;
          if (convrot) {
            for (std::uint64_t column = 0U; column < columns; ++column)
              transformed[static_cast<std::size_t>(column)] =
                  dif::runtime::load_float(binding->second, base + column) /
                  (weight_equalization
                       ? dif::runtime::load_float(input_equalization, column)
                       : 1.0F) *
                  (signed_convrot
                       ? signed_convrot_sign(column)
                       : 1.0F);
            apply_convrot(transformed,
                          static_cast<std::size_t>(rotation_group));
            for (const auto value : transformed)
              maximum = std::max(maximum, std::fabs(value));
          } else {
            for (std::uint64_t column = 0U; column < columns; ++column)
              maximum = std::max(
                  maximum,
                  std::fabs(dif::runtime::load_float(binding->second,
                                                     base + column)));
          }
          const auto scale =
              mse_weight_scale && convrot
                  ? choose_mse_int8_weight_scale(transformed)
                  : std::max(maximum / 127.0F, 1.0e-30F);
          const auto scale_bf16 = dif::runtime::bf16_to_float(
              dif::runtime::float_to_bf16(scale));
          dif::runtime::store_float(
              weight_scales, row,
              !f32_convrot &&
                      ((mse_weight_scale && convrot) || signed_convrot)
                  ? scale_bf16
                  : scale);
          for (std::uint64_t column = 0U; column < columns; ++column) {
            int rounded{};
            if (f32_convrot) {
              rounded = static_cast<int>(std::nearbyint(
                  transformed[static_cast<std::size_t>(column)] / scale));
            } else if (convrot) {
              const auto value_bf16 = dif::runtime::bf16_to_float(
                  dif::runtime::float_to_bf16(
                      transformed[static_cast<std::size_t>(column)]));
              const auto divided_bf16 = dif::runtime::bf16_to_float(
                  dif::runtime::float_to_bf16(value_bf16 / scale_bf16));
              rounded = static_cast<int>(std::nearbyint(divided_bf16));
            } else {
              rounded = static_cast<int>(std::nearbyint(
                  (dif::runtime::load_float(binding->second, base + column) /
                   (weight_equalization
                        ? dif::runtime::load_float(input_equalization, column)
                        : 1.0F)) /
                  scale));
            }
            quantized[base + column] = static_cast<std::int8_t>(std::clamp(
                rounded, convrot && !f32_convrot ? -128 : -127, 127));
          }
        }
      };
      const auto hardware_threads =
          std::max(1U, std::thread::hardware_concurrency());
      const auto worker_count = std::min<std::uint64_t>(
          weight_desc.dims.at(0), hardware_threads);
      std::vector<std::thread> workers;
      workers.reserve(static_cast<std::size_t>(worker_count));
      for (std::uint64_t worker = 0U; worker < worker_count; ++worker) {
        const auto begin = weight_desc.dims.at(0) * worker / worker_count;
        const auto end = weight_desc.dims.at(0) * (worker + 1U) / worker_count;
        workers.emplace_back(encode_rows, begin, end);
      }
      for (auto &worker : workers)
        worker.join();
      auto building_path = cache_path;
      building_path += ".building";
      if (fs::exists(building_path))
        dif::fail("refusing to overwrite an incomplete FLUX.2 W8A8 cache: " +
                  building_path.string());
      std::vector<dif::weights::SafeTensorWriteSpec> specs{
          {"weight", quantized_weight.dtype, quantized_weight.dims},
          {"scale", weight_scales.dtype, weight_scales.dims}};
      if (weight_equalization)
        specs.push_back({"input_scale", input_equalization.dtype,
                         input_equalization.dims});
      dif::weights::SafeTensorWriter writer(building_path, specs);
      writer.append("weight", std::span<const std::uint8_t>(
                                  quantized_weight.data(),
                                  quantized_weight.byte_size()));
      writer.append("scale", std::span<const std::uint8_t>(
                                 weight_scales.data(),
                                 weight_scales.byte_size()));
      if (weight_equalization)
        writer.append("input_scale", std::span<const std::uint8_t>(
                                         input_equalization.data(),
                                         input_equalization.byte_size()));
      (void)writer.finish();
      fs::rename(building_path, cache_path);
      std::cerr << "FLUX2_W8A8_CACHE_BUILD block=" << block
                << " path=" << cache_path << '\n';
    }
    const auto cache = dif::weights::read_safetensors(cache_path);
    auto quantized_weight = dif::weights::map_safetensor(cache, "weight");
    auto weight_scales = dif::weights::map_safetensor(cache, "scale");
    std::optional<dif::runtime::Tensor> input_equalization;
    if (weight_equalization)
      input_equalization.emplace(
          dif::weights::map_safetensor(cache, "input_scale"));
    if (quantized_weight.dtype != dif::ir::DType::I8 ||
        quantized_weight.dims != weight_desc.dims ||
        weight_scales.dtype != dif::ir::DType::F32 ||
        weight_scales.dims !=
            std::vector<std::uint64_t>{weight_desc.dims.at(0)} ||
        (weight_equalization &&
         (input_equalization->dtype != dif::ir::DType::BF16 ||
          input_equalization->dims !=
              std::vector<std::uint64_t>{weight_desc.dims.at(1)})))
      dif::fail("FLUX.2 W8A8 cache shape/dtype disagreement at block " +
                std::to_string(block));

    const auto quantized_weight_id = ++next_tensor;
    const auto weight_scales_id = ++next_tensor;
    const auto quantized_input_id = ++next_tensor;
    const auto input_scales_id = ++next_tensor;
    const auto equalization_id =
        weight_equalization ? ++next_tensor : std::uint32_t{0U};
    const auto equalized_input_id =
        weight_equalization ? ++next_tensor : std::uint32_t{0U};
    const auto residual_quantized_input_id =
        activation_residual2 ? ++next_tensor : std::uint32_t{0U};
    const auto residual_input_scales_id =
        activation_residual2 ? ++next_tensor : std::uint32_t{0U};
    const auto primary_output_id =
        activation_residual2 ? ++next_tensor : std::uint32_t{0U};
    const auto residual_output_id =
        activation_residual2 ? ++next_tensor : std::uint32_t{0U};
    program.tensors.push_back(
        {quantized_weight_id, dif::ir::DType::I8,
         weight_desc.roles, weight_desc.dims});
    program.tensors.push_back(
        {weight_scales_id, dif::ir::DType::F32,
         weight_desc.roles, {weight_desc.dims.at(0)}});
    program.tensors.push_back(
         {quantized_input_id, dif::ir::DType::I8,
         dif::ir::TensorRole::Internal, input_desc.dims});
    program.tensors.push_back(
        {input_scales_id, dif::ir::DType::F32,
         dif::ir::TensorRole::Internal,
         {input_desc.element_count() / input_desc.dims.back()}});
    if (weight_equalization) {
      program.tensors.push_back(
          {equalization_id, dif::ir::DType::BF16,
           dif::ir::TensorRole::Constant, {input_desc.dims.back()}});
      program.tensors.push_back(
          {equalized_input_id, dif::ir::DType::BF16,
           dif::ir::TensorRole::Internal, input_desc.dims});
      bindings.emplace(equalization_id, std::move(*input_equalization));
    }
    if (activation_residual2) {
      program.tensors.push_back(
          {residual_quantized_input_id, dif::ir::DType::I8,
           dif::ir::TensorRole::Internal, input_desc.dims});
      program.tensors.push_back(
          {residual_input_scales_id, dif::ir::DType::F32,
           dif::ir::TensorRole::Internal,
           {input_desc.element_count() / input_desc.dims.back()}});
      program.tensors.push_back(
          {primary_output_id, dif::ir::DType::BF16,
           dif::ir::TensorRole::Internal, output_desc.dims});
      program.tensors.push_back(
          {residual_output_id, dif::ir::DType::BF16,
           dif::ir::TensorRole::Internal, output_desc.dims});
    }
    bindings.emplace(quantized_weight_id, std::move(quantized_weight));
    bindings.emplace(weight_scales_id, std::move(weight_scales));
    bindings.erase(weight_id);

    auto scaled_linear = *linear;
    scaled_linear.opcode = dif::ir::Opcode::LinearInt8Scaled;
    scaled_linear.inputs = {quantized_input_id, quantized_weight_id,
                            input_scales_id, weight_scales_id};
    if (activation_residual2)
      scaled_linear.outputs = {primary_output_id};
    scaled_linear.attributes.clear();
    const dif::ir::Operation equalize{
        ++next_operation, dif::ir::Opcode::AffineLastDim,
        {linear->inputs.at(0), equalization_id}, {equalized_input_id}, {}};
    dif::ir::Operation quantize{
        ++next_operation, dif::ir::Opcode::QuantizeInt8Rows,
        weight_equalization
            ? std::vector<std::uint32_t>{equalized_input_id}
            : fused_concat ? fused_concat->inputs
                           : fused_reshape ? fused_reshape->inputs
                           : std::vector<std::uint32_t>{linear_input_id},
        {quantized_input_id, input_scales_id},
        {dif::ir::Attribute::u64(dif::ir::AttrKey::BlockSize, 256U),
         dif::ir::Attribute::u64(
             dif::ir::AttrKey::Implementation,
             static_cast<std::uint64_t>(quantization))}};
    if (activation_residual2) {
      quantize.outputs.push_back(residual_quantized_input_id);
      quantize.outputs.push_back(residual_input_scales_id);
    }
    std::vector<dif::ir::Operation> replacements;
    replacements.reserve(activation_residual2 ? 5U : 3U);
    if (weight_equalization)
      replacements.push_back(equalize);
    replacements.push_back(std::move(quantize));
    replacements.push_back(std::move(scaled_linear));
    if (activation_residual2) {
      auto residual_linear = *linear;
      residual_linear.id = ++next_operation;
      residual_linear.opcode = dif::ir::Opcode::LinearInt8Scaled;
      residual_linear.inputs = {residual_quantized_input_id,
                                quantized_weight_id,
                                residual_input_scales_id,
                                weight_scales_id};
      residual_linear.outputs = {residual_output_id};
      residual_linear.attributes.clear();
      replacements.push_back(std::move(residual_linear));
      replacements.push_back(
          {++next_operation, dif::ir::Opcode::Add,
           {primary_output_id, residual_output_id}, linear->outputs, {}});
    }
    program.operations.erase(program.operations.begin() +
                             static_cast<std::ptrdiff_t>(linear_index));
    program.operations.insert(
        program.operations.begin() + static_cast<std::ptrdiff_t>(linear_index),
        std::make_move_iterator(replacements.begin()),
        std::make_move_iterator(replacements.end()));
    if (fused_concat || fused_reshape)
      program.operations.erase(
          std::remove_if(program.operations.begin(), program.operations.end(),
                         [&](const dif::ir::Operation &operation) {
                           return (fused_concat &&
                                   operation.id == fused_concat_id) ||
                                  operation.id == fused_reshape_id;
                         }),
          program.operations.end());
    program.tensors.erase(
        std::remove_if(program.tensors.begin(), program.tensors.end(),
                       [&](const dif::ir::TensorDesc &description) {
                         return description.id == weight_id ||
                                (fused_reshape &&
                                 description.id == linear_input_id) ||
                                (fused_concat &&
                                 description.id == fused_concat_output_id);
                       }),
        program.tensors.end());
    ++result.linear_count;
    result.quantized_weight_bytes +=
        weight_desc.element_count() +
        weight_desc.dims.at(0) * sizeof(float);
  }
  dif::ir::verify(program);
  return result;
}

Fp8RewriteResult rewrite_named_linear_fp8_rows(
    dif::ir::Program &program, dif::runtime::TensorMap &bindings,
    const dif::frontend::Flux2KleinTransformerBuild &transformer,
    std::span<const std::uint32_t> block_ids,
    std::string_view checkpoint_prefix, std::string_view checkpoint_suffix,
    std::string_view cache_label, const fs::path &cache_directory) {
  Fp8RewriteResult result;
  if (block_ids.empty())
    return result;
  auto next_tensor = std::uint32_t{0U};
  auto next_operation = std::uint32_t{0U};
  for (const auto &tensor : program.tensors)
    next_tensor = std::max(next_tensor, tensor.id);
  for (const auto &operation : program.operations)
    next_operation = std::max(next_operation, operation.id);

  std::vector<std::pair<std::uint32_t, std::uint32_t>> selected_weights;
  std::unordered_set<std::uint32_t> seen_weights;
  const std::unordered_set<std::uint32_t> requested_blocks(
      block_ids.begin(), block_ids.end());
  for (std::size_t index = 0U;
       index < transformer.checkpoint_names.size(); ++index) {
    const auto &name = transformer.checkpoint_names.at(index);
    if (!name.starts_with(checkpoint_prefix) ||
        !name.ends_with(checkpoint_suffix))
      continue;
    const auto block_end = name.find('.', checkpoint_prefix.size());
    if (block_end == std::string::npos)
      dif::fail("malformed FLUX.2 row-scaled FP8 checkpoint name");
    const auto block = static_cast<std::uint32_t>(std::stoul(name.substr(
        checkpoint_prefix.size(), block_end - checkpoint_prefix.size())));
    if (!requested_blocks.contains(block))
      continue;
    const auto weight = transformer.checkpoint_tensors.at(index);
    if (seen_weights.insert(weight).second)
      selected_weights.emplace_back(block, weight);
  }
  if (selected_weights.size() != block_ids.size())
    dif::fail("FLUX.2 row-scaled FP8 selection did not resolve every block");

  fs::create_directories(cache_directory);
  for (const auto [block, weight_id] : selected_weights) {
    const auto *weight_description = program.tensor(weight_id);
    const auto binding = bindings.find(weight_id);
    if (!weight_description || binding == bindings.end() ||
        weight_description->dtype != dif::ir::DType::BF16 ||
        weight_description->dims.size() != 2U)
      dif::fail("FLUX.2 row-scaled FP8 weight is not rank-2 BF16");
    const auto weight_desc = *weight_description;
    const auto linear = std::find_if(
        program.operations.begin(), program.operations.end(),
        [&](const dif::ir::Operation &operation) {
          return operation.opcode == dif::ir::Opcode::Linear &&
                 operation.inputs.size() == 2U &&
                 operation.inputs.at(1) == weight_id;
        });
    if (linear == program.operations.end())
      dif::fail("FLUX.2 row-scaled FP8 weight has no unbiased Linear");
    const auto linear_index = static_cast<std::size_t>(
        std::distance(program.operations.begin(), linear));
    const auto *input_description = program.tensor(linear->inputs.at(0));
    if (!input_description || input_description->dtype != dif::ir::DType::BF16)
      dif::fail("FLUX.2 row-scaled FP8 Linear input is not BF16");
    const auto input_desc = *input_description;
    const auto rows = weight_desc.dims.at(0);
    const auto columns = weight_desc.dims.at(1);
    const auto cache_path =
        cache_directory /
        ("flux2-klein-base-9b-" + std::string(kModelRevision) + "-" +
         std::string(cache_label) + "-" + std::to_string(block) +
         "-fp8-e4m3-row-scaled.safetensors");
    if (!fs::exists(cache_path)) {
      dif::runtime::Tensor quantized_weight{
          dif::ir::DType::FP8E4M3, weight_desc.dims, {}};
      quantized_weight.bytes.resize(
          static_cast<std::size_t>(weight_desc.element_count()));
      dif::runtime::Tensor weight_scales{
          dif::ir::DType::F32, {rows}, {}};
      weight_scales.bytes.resize(static_cast<std::size_t>(rows * sizeof(float)));
      const auto encode_rows = [&](std::uint64_t begin, std::uint64_t end) {
        for (auto row = begin; row < end; ++row) {
          const auto base = row * columns;
          float maximum = 0.0F;
          for (std::uint64_t column = 0U; column < columns; ++column)
            maximum = std::max(
                maximum, std::fabs(dif::runtime::load_float(
                             binding->second, base + column)));
          const auto scale = std::max(maximum / 448.0F, 1.0e-30F);
          dif::runtime::store_float(weight_scales, row, scale);
          for (std::uint64_t column = 0U; column < columns; ++column)
            quantized_weight.mutable_data()[base + column] =
                dif::runtime::float_to_fp8_e4m3(
                    dif::runtime::load_float(binding->second,
                                             base + column) /
                    scale);
        }
      };
      const auto worker_count = std::min<std::uint64_t>(
          rows, std::max(1U, std::thread::hardware_concurrency()));
      std::vector<std::thread> workers;
      workers.reserve(static_cast<std::size_t>(worker_count));
      for (std::uint64_t worker = 0U; worker < worker_count; ++worker)
        workers.emplace_back(encode_rows, rows * worker / worker_count,
                             rows * (worker + 1U) / worker_count);
      for (auto &worker : workers)
        worker.join();
      auto building_path = cache_path;
      building_path += ".building";
      if (fs::exists(building_path))
        dif::fail("refusing to overwrite an incomplete row-scaled FP8 cache: " +
                  building_path.string());
      dif::weights::SafeTensorWriter writer(
          building_path,
          {{"weight", quantized_weight.dtype, quantized_weight.dims},
           {"scale", weight_scales.dtype, weight_scales.dims}});
      writer.append("weight", std::span<const std::uint8_t>(
                                  quantized_weight.data(),
                                  quantized_weight.byte_size()));
      writer.append("scale", std::span<const std::uint8_t>(
                                 weight_scales.data(),
                                 weight_scales.byte_size()));
      (void)writer.finish();
      fs::rename(building_path, cache_path);
      std::cerr << "FLUX2_FP8_ROW_CACHE_BUILD block=" << block
                << " path=" << cache_path << '\n';
    }
    const auto cache = dif::weights::read_safetensors(cache_path);
    auto quantized_weight = dif::weights::map_safetensor(cache, "weight");
    auto weight_scales = dif::weights::map_safetensor(cache, "scale");
    if (quantized_weight.dtype != dif::ir::DType::FP8E4M3 ||
        quantized_weight.dims != weight_desc.dims ||
        weight_scales.dtype != dif::ir::DType::F32 ||
        weight_scales.dims != std::vector<std::uint64_t>{rows})
      dif::fail("FLUX.2 row-scaled FP8 cache shape/dtype disagreement");

    const auto quantized_weight_id = ++next_tensor;
    const auto weight_scales_id = ++next_tensor;
    const auto quantized_input_id = ++next_tensor;
    const auto input_scales_id = ++next_tensor;
    program.tensors.push_back(
        {quantized_weight_id, dif::ir::DType::FP8E4M3, weight_desc.roles,
         weight_desc.dims});
    program.tensors.push_back(
        {weight_scales_id, dif::ir::DType::F32, weight_desc.roles, {rows}});
    program.tensors.push_back(
        {quantized_input_id, dif::ir::DType::FP8E4M3,
         dif::ir::TensorRole::Internal, input_desc.dims});
    program.tensors.push_back(
        {input_scales_id, dif::ir::DType::F32,
         dif::ir::TensorRole::Internal,
         {input_desc.element_count() / input_desc.dims.back()}});
    bindings.emplace(quantized_weight_id, std::move(quantized_weight));
    bindings.emplace(weight_scales_id, std::move(weight_scales));
    bindings.erase(weight_id);

    auto scaled_linear = *linear;
    scaled_linear.opcode = dif::ir::Opcode::LinearFp8Scaled;
    scaled_linear.inputs = {quantized_input_id, quantized_weight_id,
                            input_scales_id, weight_scales_id};
    scaled_linear.attributes.clear();
    const dif::ir::Operation quantize{
        ++next_operation, dif::ir::Opcode::QuantizeFp8Rows,
        {linear->inputs.at(0)}, {quantized_input_id, input_scales_id}, {}};
    program.operations.at(linear_index) = quantize;
    program.operations.insert(program.operations.begin() +
                                  static_cast<std::ptrdiff_t>(linear_index + 1U),
                              std::move(scaled_linear));
    program.tensors.erase(
        std::remove_if(program.tensors.begin(), program.tensors.end(),
                       [&](const dif::ir::TensorDesc &description) {
                         return description.id == weight_id;
                       }),
        program.tensors.end());
    ++result.linear_count;
    result.quantized_weight_bytes +=
        weight_desc.element_count() + rows * sizeof(float);
  }
  dif::ir::verify(program);
  return result;
}

Fp8RewriteResult rewrite_named_linear_fp8(
    dif::ir::Program &program, dif::runtime::TensorMap &bindings,
    const dif::frontend::Flux2KleinTransformerBuild &transformer,
    std::span<const std::uint32_t> block_ids,
    std::string_view checkpoint_prefix, std::string_view checkpoint_suffix,
    std::string_view cache_label, const fs::path &cache_directory,
    bool row_scaled) {
  if (row_scaled)
    return rewrite_named_linear_fp8_rows(
        program, bindings, transformer, block_ids, checkpoint_prefix,
        checkpoint_suffix, cache_label, cache_directory);
  Fp8RewriteResult result;
  if (block_ids.empty())
    return result;
  auto next_tensor = std::uint32_t{0U};
  auto next_operation = std::uint32_t{0U};
  for (const auto &tensor : program.tensors)
    next_tensor = std::max(next_tensor, tensor.id);
  for (const auto &operation : program.operations)
    next_operation = std::max(next_operation, operation.id);

  std::vector<std::pair<std::uint32_t, std::uint32_t>> selected_weights;
  std::unordered_set<std::uint32_t> seen_weights;
  const std::unordered_set<std::uint32_t> requested_blocks(
      block_ids.begin(), block_ids.end());
  for (std::size_t index = 0U;
       index < transformer.checkpoint_names.size(); ++index) {
    const auto &name = transformer.checkpoint_names.at(index);
    if (!name.starts_with(checkpoint_prefix) ||
        !name.ends_with(checkpoint_suffix))
      continue;
    const auto block_end = name.find('.', checkpoint_prefix.size());
    if (block_end == std::string::npos)
      dif::fail("malformed FLUX.2 FP8 checkpoint name");
    const auto block = static_cast<std::uint32_t>(std::stoul(name.substr(
        checkpoint_prefix.size(), block_end - checkpoint_prefix.size())));
    if (!requested_blocks.contains(block))
      continue;
    const auto weight = transformer.checkpoint_tensors.at(index);
    if (seen_weights.insert(weight).second)
      selected_weights.emplace_back(block, weight);
  }
  if (selected_weights.size() != block_ids.size())
    dif::fail("FLUX.2 FP8 selection did not resolve every requested block");

  fs::create_directories(cache_directory);
  for (const auto [block, weight_id] : selected_weights) {
    const auto *weight_description = program.tensor(weight_id);
    const auto binding = bindings.find(weight_id);
    if (!weight_description || binding == bindings.end() ||
        weight_description->dtype != dif::ir::DType::BF16 ||
        weight_description->dims.size() != 2U)
      dif::fail("FLUX.2 FP8 weight is not a bound rank-2 BF16 tensor");
    const auto weight_desc = *weight_description;
    const auto linear = std::find_if(
        program.operations.begin(), program.operations.end(),
        [&](const dif::ir::Operation &operation) {
          return operation.opcode == dif::ir::Opcode::Linear &&
                 operation.inputs.size() == 2U &&
                 operation.inputs.at(1) == weight_id;
        });
    if (linear == program.operations.end())
      dif::fail("FLUX.2 FP8 weight has no exclusive unbiased Linear");
    const auto linear_index = static_cast<std::size_t>(
        std::distance(program.operations.begin(), linear));
    const auto *input_description = program.tensor(linear->inputs.at(0));
    if (!input_description || input_description->dtype != dif::ir::DType::BF16)
      dif::fail("FLUX.2 FP8 Linear input is not BF16");
    const auto input_desc = *input_description;
    const auto columns = weight_desc.dims.at(1);
    const auto scale_blocks = (columns + 31U) / 32U;
    const auto padded_scale_blocks = ((scale_blocks + 3U) / 4U) * 4U;
    const auto padded_weight_rows =
        ((weight_desc.dims.at(0) + 127U) / 128U) * 128U;
    const std::vector<std::uint64_t> weight_scale_dims{
        padded_weight_rows, padded_scale_blocks};
    const auto input_rows =
        input_desc.element_count() / input_desc.dims.back();
    const auto padded_input_rows = ((input_rows + 127U) / 128U) * 128U;
    const std::vector<std::uint64_t> input_scale_dims{
        padded_input_rows, padded_scale_blocks};
    const auto scale_offset = [padded_scale_blocks](std::uint64_t outer,
                                                    std::uint64_t block) {
      const auto tile_outer = outer / 128U;
      const auto tile_inner = (block / 4U) * 4U;
      const auto within = (outer % 32U) * 16U +
                          ((outer % 128U) / 32U) * 4U + block % 4U;
      return (tile_inner + tile_outer * padded_scale_blocks) * 128U +
             within;
    };

    const auto cache_path =
        cache_directory /
        ("flux2-klein-base-9b-" + std::string(kModelRevision) + "-" +
         std::string(cache_label) + "-" + std::to_string(block) +
         "-mxfp8-e4m3-ue8m0-block32.safetensors");
    if (!fs::exists(cache_path)) {
      dif::runtime::Tensor quantized_weight{
          dif::ir::DType::FP8E4M3, weight_desc.dims, {}};
      quantized_weight.bytes.resize(
          static_cast<std::size_t>(weight_desc.element_count()));
      dif::runtime::Tensor weight_scales{
          dif::ir::DType::FP8E8M0, weight_scale_dims, {}};
      weight_scales.bytes.resize(
          static_cast<std::size_t>(weight_scales.element_count()), 0U);
      const auto encode_rows = [&](std::uint64_t begin, std::uint64_t end) {
        for (auto row = begin; row < end; ++row) {
          const auto base = row * columns;
          for (std::uint64_t scale_block = 0U;
               scale_block < scale_blocks; ++scale_block) {
            const auto begin_column = scale_block * 32U;
            const auto end_column =
                std::min(columns, begin_column + 32U);
            float maximum = 0.0F;
            for (auto column = begin_column; column < end_column; ++column)
              maximum = std::max(
                  maximum,
                  std::fabs(dif::runtime::load_float(binding->second,
                                                      base + column)));
            const auto encoded_scale =
                dif::runtime::float_to_fp8_e8m0_round_up(maximum / 448.0F);
            weight_scales.mutable_data()[scale_offset(row, scale_block)] =
                encoded_scale;
            const auto scale =
                dif::runtime::fp8_e8m0_to_float(encoded_scale);
            for (auto column = begin_column; column < end_column; ++column)
              quantized_weight.mutable_data()[base + column] =
                  dif::runtime::float_to_fp8_e4m3(
                      dif::runtime::load_float(binding->second,
                                               base + column) /
                      scale);
          }
        }
      };
      const auto hardware_threads =
          std::max(1U, std::thread::hardware_concurrency());
      const auto worker_count = std::min<std::uint64_t>(
          weight_desc.dims.at(0), hardware_threads);
      std::vector<std::thread> workers;
      workers.reserve(static_cast<std::size_t>(worker_count));
      for (std::uint64_t worker = 0U; worker < worker_count; ++worker) {
        const auto begin = weight_desc.dims.at(0) * worker / worker_count;
        const auto end =
            weight_desc.dims.at(0) * (worker + 1U) / worker_count;
        workers.emplace_back(encode_rows, begin, end);
      }
      for (auto &worker : workers)
        worker.join();
      auto building_path = cache_path;
      building_path += ".building";
      if (fs::exists(building_path))
        dif::fail("refusing to overwrite an incomplete FLUX.2 FP8 cache: " +
                  building_path.string());
      dif::weights::SafeTensorWriter writer(
          building_path,
          {{"weight", quantized_weight.dtype, quantized_weight.dims},
           {"scale", weight_scales.dtype, weight_scales.dims}});
      writer.append("weight", std::span<const std::uint8_t>(
                                  quantized_weight.data(),
                                  quantized_weight.byte_size()));
      writer.append("scale", std::span<const std::uint8_t>(
                                 weight_scales.data(),
                                 weight_scales.byte_size()));
      (void)writer.finish();
      fs::rename(building_path, cache_path);
      std::cerr << "FLUX2_FP8_CACHE_BUILD block=" << block
                << " path=" << cache_path << '\n';
    }
    const auto cache = dif::weights::read_safetensors(cache_path);
    auto quantized_weight = dif::weights::map_safetensor(cache, "weight");
    auto weight_scales = dif::weights::map_safetensor(cache, "scale");
    if (quantized_weight.dtype != dif::ir::DType::FP8E4M3 ||
        quantized_weight.dims != weight_desc.dims ||
        weight_scales.dtype != dif::ir::DType::FP8E8M0 ||
        weight_scales.dims != weight_scale_dims)
      dif::fail("FLUX.2 FP8 cache shape/dtype disagreement at block " +
                std::to_string(block));

    const auto quantized_weight_id = ++next_tensor;
    const auto weight_scales_id = ++next_tensor;
    const auto quantized_input_id = ++next_tensor;
    const auto input_scales_id = ++next_tensor;
    program.tensors.push_back(
        {quantized_weight_id, dif::ir::DType::FP8E4M3, weight_desc.roles,
         weight_desc.dims});
    program.tensors.push_back(
        {weight_scales_id, dif::ir::DType::FP8E8M0, weight_desc.roles,
         weight_scale_dims});
    program.tensors.push_back(
        {quantized_input_id, dif::ir::DType::FP8E4M3,
         dif::ir::TensorRole::Internal, input_desc.dims});
    program.tensors.push_back(
        {input_scales_id, dif::ir::DType::FP8E8M0,
         dif::ir::TensorRole::Internal, input_scale_dims});
    bindings.emplace(quantized_weight_id, std::move(quantized_weight));
    bindings.emplace(weight_scales_id, std::move(weight_scales));
    bindings.erase(weight_id);

    auto scaled_linear = *linear;
    scaled_linear.opcode = dif::ir::Opcode::LinearFp8BlockScaled;
    scaled_linear.inputs = {quantized_input_id, quantized_weight_id,
                            input_scales_id, weight_scales_id};
    scaled_linear.attributes.clear();
    const dif::ir::Operation quantize{
        ++next_operation, dif::ir::Opcode::QuantizeFp8Blocks32,
        {linear->inputs.at(0)}, {quantized_input_id, input_scales_id},
        {dif::ir::Attribute::u64(dif::ir::AttrKey::BlockSize, 256U)}};
    program.operations.at(linear_index) = quantize;
    program.operations.insert(program.operations.begin() +
                                  static_cast<std::ptrdiff_t>(linear_index + 1U),
                              std::move(scaled_linear));
    program.tensors.erase(
        std::remove_if(program.tensors.begin(), program.tensors.end(),
                       [&](const dif::ir::TensorDesc &description) {
                         return description.id == weight_id;
                       }),
        program.tensors.end());
    ++result.linear_count;
    result.quantized_weight_bytes +=
        weight_desc.element_count() +
        padded_weight_rows * padded_scale_blocks;
  }
  dif::ir::verify(program);
  return result;
}

// Preserve the attention-sensitive Q/K/V rows of the packed single-stream
// projection in BF16 while lowering only its MLP rows to the generic Blackwell
// block-scaled FP8 path.
Fp8RewriteResult rewrite_single_mlp_fp8(
    dif::ir::Program &program, dif::runtime::TensorMap &bindings,
    const dif::frontend::Flux2KleinTransformerBuild &transformer,
    std::span<const std::uint32_t> block_ids,
    const fs::path &cache_directory) {
  Fp8RewriteResult result;
  if (block_ids.empty())
    return result;
  constexpr std::uint64_t hidden = 4096U;
  constexpr std::uint64_t qkv_rows = 3U * hidden;
  constexpr std::uint64_t mlp_rows = 6U * hidden;
  constexpr std::uint64_t packed_rows = qkv_rows + mlp_rows;
  constexpr std::uint64_t scale_blocks = hidden / 32U;
  constexpr std::uint64_t padded_scale_blocks = scale_blocks;
  constexpr std::uint64_t padded_weight_rows = mlp_rows;

  auto next_tensor = std::uint32_t{0U};
  auto next_operation = std::uint32_t{0U};
  for (const auto &tensor : program.tensors)
    next_tensor = std::max(next_tensor, tensor.id);
  for (const auto &operation : program.operations)
    next_operation = std::max(next_operation, operation.id);

  std::vector<std::pair<std::uint32_t, std::uint32_t>> selected_weights;
  const std::unordered_set<std::uint32_t> requested_blocks(
      block_ids.begin(), block_ids.end());
  for (std::size_t index = 0U;
       index < transformer.checkpoint_names.size(); ++index) {
    const auto &name = transformer.checkpoint_names.at(index);
    constexpr std::string_view prefix = "single_blocks.";
    constexpr std::string_view suffix = ".linear1.weight";
    if (!name.starts_with(prefix) || !name.ends_with(suffix))
      continue;
    const auto block_end = name.find('.', prefix.size());
    if (block_end == std::string::npos)
      dif::fail("malformed FLUX.2 single-MLP FP8 checkpoint name");
    const auto block = static_cast<std::uint32_t>(std::stoul(
        name.substr(prefix.size(), block_end - prefix.size())));
    if (requested_blocks.contains(block))
      selected_weights.emplace_back(block,
                                    transformer.checkpoint_tensors.at(index));
  }
  if (selected_weights.size() != block_ids.size())
    dif::fail("FLUX.2 single-MLP FP8 selection did not resolve every block");

  const auto scale_offset = [](std::uint64_t outer, std::uint64_t block,
                               std::uint64_t padded_blocks) {
    const auto tile_outer = outer / 128U;
    const auto tile_inner = (block / 4U) * 4U;
    const auto within = (outer % 32U) * 16U +
                        ((outer % 128U) / 32U) * 4U + block % 4U;
    return (tile_inner + tile_outer * padded_blocks) * 128U + within;
  };

  fs::create_directories(cache_directory);
  for (const auto [block, weight_id] : selected_weights) {
    const auto *weight_description = program.tensor(weight_id);
    const auto binding = bindings.find(weight_id);
    if (!weight_description || binding == bindings.end() ||
        weight_description->dtype != dif::ir::DType::BF16 ||
        weight_description->dims !=
            std::vector<std::uint64_t>{packed_rows, hidden})
      dif::fail("FLUX.2 single packed projection is not BF16 [36864,4096]");
    const auto weight_desc = *weight_description;
    const auto linear = std::find_if(
        program.operations.begin(), program.operations.end(),
        [&](const dif::ir::Operation &operation) {
          return operation.opcode == dif::ir::Opcode::Linear &&
                 operation.inputs.size() == 2U &&
                 operation.inputs.at(1) == weight_id;
        });
    if (linear == program.operations.end())
      dif::fail("FLUX.2 single packed projection has no unbiased Linear");
    const auto linear_template = *linear;
    const auto linear_id = linear->id;
    const auto input_id = linear->inputs.at(0);
    const auto output_id = linear->outputs.at(0);
    const auto *input_description = program.tensor(input_id);
    const auto *output_description = program.tensor(output_id);
    if (!input_description || !output_description ||
        input_description->dtype != dif::ir::DType::BF16 ||
        output_description->dtype != dif::ir::DType::BF16 ||
        input_description->dims.empty() || output_description->dims.empty() ||
        input_description->dims.back() != hidden ||
        output_description->dims.back() != packed_rows)
      dif::fail("FLUX.2 single packed FP8 projection shape disagreement");
    const auto input_desc = *input_description;
    const auto output_desc = *output_description;
    if ((output_desc.roles & dif::ir::TensorRole::Output) != 0U)
      dif::fail("single-MLP FP8 cannot remove a captured packed projection");
    const auto packed_reshape = std::find_if(
        program.operations.begin(), program.operations.end(),
        [&](const dif::ir::Operation &operation) {
          if (operation.opcode != dif::ir::Opcode::Reshape ||
              operation.inputs != std::vector<std::uint32_t>{output_id} ||
              operation.outputs.size() != 1U)
            return false;
          const auto *description = program.tensor(operation.outputs.front());
          return description && !description->dims.empty() &&
                 description->dims.back() == packed_rows;
        });
    if (packed_reshape == program.operations.end())
      dif::fail("single-MLP FP8 requires the packed projection reshape");
    const auto packed_reshape_id = packed_reshape->id;
    const auto packed_reshape_output_id = packed_reshape->outputs.front();
    const auto find_projection_slice =
        [&](std::uint64_t start, std::uint64_t width) {
          return std::find_if(
              program.operations.begin(), program.operations.end(),
              [&](const dif::ir::Operation &operation) {
                if (operation.opcode != dif::ir::Opcode::Slice ||
                    operation.inputs !=
                        std::vector<std::uint32_t>{packed_reshape_output_id} ||
                    operation.outputs.size() != 1U ||
                    operation.u64(dif::ir::AttrKey::Start,
                                  std::numeric_limits<std::uint64_t>::max()) !=
                        start)
                  return false;
                const auto *description = program.tensor(operation.outputs.front());
                return description && !description->dims.empty() &&
                       description->dims.back() == width;
              });
        };
    const auto qkv_slice = find_projection_slice(0U, qkv_rows);
    const auto mlp_slice = find_projection_slice(qkv_rows, mlp_rows);
    if (qkv_slice == program.operations.end() ||
        mlp_slice == program.operations.end())
      dif::fail("single-MLP FP8 requires exclusive packed projection slices");
    const auto qkv_slice_id = qkv_slice->id;
    const auto mlp_slice_id = mlp_slice->id;
    const auto qkv_output_id = qkv_slice->outputs.front();
    const auto mlp_output_id = mlp_slice->outputs.front();

    const std::vector<std::uint64_t> mlp_weight_dims{mlp_rows, hidden};
    const std::vector<std::uint64_t> weight_scale_dims{
        padded_weight_rows, padded_scale_blocks};
    const auto cache_path =
        cache_directory /
        ("flux2-klein-base-9b-" + std::string(kModelRevision) +
         "-single-mlp-block-" + std::to_string(block) +
         "-mxfp8-e4m3-ue8m0-block32.safetensors");
    if (!fs::exists(cache_path)) {
      dif::runtime::Tensor quantized_weight{
          dif::ir::DType::FP8E4M3, mlp_weight_dims, {}};
      quantized_weight.bytes.resize(
          static_cast<std::size_t>(mlp_rows * hidden));
      dif::runtime::Tensor weight_scales{
          dif::ir::DType::FP8E8M0, weight_scale_dims, {}};
      weight_scales.bytes.resize(
          static_cast<std::size_t>(weight_scales.element_count()), 0U);
      const auto encode_rows = [&](std::uint64_t begin, std::uint64_t end) {
        for (auto row = begin; row < end; ++row) {
          const auto source_base = (qkv_rows + row) * hidden;
          const auto target_base = row * hidden;
          for (std::uint64_t scale_block = 0U;
               scale_block < scale_blocks; ++scale_block) {
            const auto begin_column = scale_block * 32U;
            const auto end_column = begin_column + 32U;
            float maximum = 0.0F;
            for (auto column = begin_column; column < end_column; ++column)
              maximum = std::max(
                  maximum,
                  std::fabs(dif::runtime::load_float(
                      binding->second, source_base + column)));
            const auto encoded_scale =
                dif::runtime::float_to_fp8_e8m0_round_up(maximum / 448.0F);
            weight_scales.mutable_data()[scale_offset(
                row, scale_block, padded_scale_blocks)] = encoded_scale;
            const auto scale =
                dif::runtime::fp8_e8m0_to_float(encoded_scale);
            for (auto column = begin_column; column < end_column; ++column)
              quantized_weight.mutable_data()[target_base + column] =
                  dif::runtime::float_to_fp8_e4m3(
                      dif::runtime::load_float(binding->second,
                                               source_base + column) /
                      scale);
          }
        }
      };
      const auto worker_count = std::min<std::uint64_t>(
          mlp_rows, std::max(1U, std::thread::hardware_concurrency()));
      std::vector<std::thread> workers;
      workers.reserve(static_cast<std::size_t>(worker_count));
      for (std::uint64_t worker = 0U; worker < worker_count; ++worker)
        workers.emplace_back(encode_rows, mlp_rows * worker / worker_count,
                             mlp_rows * (worker + 1U) / worker_count);
      for (auto &worker : workers)
        worker.join();
      auto building_path = cache_path;
      building_path += ".building";
      if (fs::exists(building_path))
        dif::fail("refusing to overwrite an incomplete FLUX.2 single-MLP FP8 cache: " +
                  building_path.string());
      dif::weights::SafeTensorWriter writer(
          building_path,
          {{"weight", quantized_weight.dtype, quantized_weight.dims},
           {"scale", weight_scales.dtype, weight_scales.dims}});
      writer.append("weight", std::span<const std::uint8_t>(
                                  quantized_weight.data(),
                                  quantized_weight.byte_size()));
      writer.append("scale", std::span<const std::uint8_t>(
                                 weight_scales.data(),
                                 weight_scales.byte_size()));
      (void)writer.finish();
      fs::rename(building_path, cache_path);
      std::cerr << "FLUX2_SINGLE_MLP_FP8_CACHE_BUILD block=" << block
                << " path=" << cache_path << '\n';
    }
    const auto cache = dif::weights::read_safetensors(cache_path);
    auto quantized_weight = dif::weights::map_safetensor(cache, "weight");
    auto weight_scales = dif::weights::map_safetensor(cache, "scale");
    if (quantized_weight.dtype != dif::ir::DType::FP8E4M3 ||
        quantized_weight.dims != mlp_weight_dims ||
        weight_scales.dtype != dif::ir::DType::FP8E8M0 ||
        weight_scales.dims != weight_scale_dims)
      dif::fail("FLUX.2 single-MLP FP8 cache shape/dtype disagreement");

    auto qkv_weight = binding->second;
    qkv_weight.dims = {qkv_rows, hidden};
    const auto qkv_bytes = static_cast<std::size_t>(
        qkv_rows * hidden * dif::ir::dtype_size(dif::ir::DType::BF16));
    if (qkv_weight.is_mapped())
      qkv_weight.mapping_bytes = qkv_bytes;
    else
      qkv_weight.bytes.resize(qkv_bytes);
    qkv_weight.validate();

    const auto input_rows = input_desc.element_count() / hidden;
    const auto padded_input_rows = ((input_rows + 127U) / 128U) * 128U;
    const std::vector<std::uint64_t> input_scale_dims{
        padded_input_rows, padded_scale_blocks};
    const auto qkv_weight_id = ++next_tensor;
    const auto quantized_weight_id = ++next_tensor;
    const auto weight_scales_id = ++next_tensor;
    const auto quantized_input_id = ++next_tensor;
    const auto input_scales_id = ++next_tensor;
    const auto qkv_flat_output_id = ++next_tensor;
    const auto mlp_flat_output_id = ++next_tensor;
    auto qkv_flat_output_dims = input_desc.dims;
    auto mlp_flat_output_dims = input_desc.dims;
    qkv_flat_output_dims.back() = qkv_rows;
    mlp_flat_output_dims.back() = mlp_rows;
    program.tensors.push_back({qkv_weight_id, dif::ir::DType::BF16,
                               weight_desc.roles, {qkv_rows, hidden}});
    program.tensors.push_back({quantized_weight_id, dif::ir::DType::FP8E4M3,
                               weight_desc.roles, mlp_weight_dims});
    program.tensors.push_back({weight_scales_id, dif::ir::DType::FP8E8M0,
                               weight_desc.roles, weight_scale_dims});
    program.tensors.push_back({quantized_input_id, dif::ir::DType::FP8E4M3,
                               dif::ir::TensorRole::Internal,
                               input_desc.dims});
    program.tensors.push_back({input_scales_id, dif::ir::DType::FP8E8M0,
                               dif::ir::TensorRole::Internal,
                               input_scale_dims});
    program.tensors.push_back({qkv_flat_output_id, dif::ir::DType::BF16,
                               dif::ir::TensorRole::Internal,
                               qkv_flat_output_dims});
    program.tensors.push_back({mlp_flat_output_id, dif::ir::DType::BF16,
                               dif::ir::TensorRole::Internal,
                               mlp_flat_output_dims});
    bindings.emplace(qkv_weight_id, std::move(qkv_weight));
    bindings.emplace(quantized_weight_id, std::move(quantized_weight));
    bindings.emplace(weight_scales_id, std::move(weight_scales));
    bindings.erase(weight_id);

    auto qkv_linear = linear_template;
    qkv_linear.inputs = {input_id, qkv_weight_id};
    qkv_linear.outputs = {qkv_flat_output_id};
    const dif::ir::Operation quantize{
        ++next_operation, dif::ir::Opcode::QuantizeFp8Blocks32,
        {input_id}, {quantized_input_id, input_scales_id},
        {dif::ir::Attribute::u64(dif::ir::AttrKey::BlockSize, 256U)}};
    const dif::ir::Operation scaled_linear{
        ++next_operation, dif::ir::Opcode::LinearFp8BlockScaled,
        {quantized_input_id, quantized_weight_id, input_scales_id,
         weight_scales_id},
        {mlp_flat_output_id}, {}};
    const dif::ir::Operation qkv_reshape{
        ++next_operation, dif::ir::Opcode::Reshape,
        {qkv_flat_output_id}, {qkv_output_id}, {}};
    const dif::ir::Operation mlp_reshape{
        ++next_operation, dif::ir::Opcode::Reshape,
        {mlp_flat_output_id}, {mlp_output_id}, {}};
    std::vector<dif::ir::Operation> rewritten_operations;
    rewritten_operations.reserve(program.operations.size() + 2U);
    for (const auto &operation : program.operations) {
      if (operation.id == linear_id) {
        rewritten_operations.push_back(qkv_linear);
        rewritten_operations.push_back(quantize);
        rewritten_operations.push_back(scaled_linear);
        rewritten_operations.push_back(qkv_reshape);
        rewritten_operations.push_back(mlp_reshape);
      } else if (operation.id != qkv_slice_id &&
                 operation.id != mlp_slice_id &&
                 operation.id != packed_reshape_id) {
        rewritten_operations.push_back(operation);
      }
    }
    program.operations = std::move(rewritten_operations);
    program.tensors.erase(
        std::remove_if(program.tensors.begin(), program.tensors.end(),
                       [&](const dif::ir::TensorDesc &description) {
                         return description.id == weight_id ||
                                description.id == output_id ||
                                description.id == packed_reshape_output_id;
                       }),
        program.tensors.end());
    ++result.linear_count;
    result.quantized_weight_bytes +=
        mlp_rows * hidden + padded_weight_rows * padded_scale_blocks;
  }
  dif::ir::verify(program);
  return result;
}

// Keep either Q/K/V or only Q/K rows of a FLUX.2 single-stream packed
// projection in BF16 and apply the shared W8A8 primitive to the remaining
// packed tail. This is frontend graph surgery over generic Linear,
// QuantizeInt8Rows, LinearInt8Scaled, and Concat semantics; the runtime remains
// model agnostic.
W8A8RewriteResult rewrite_single_packed_tail_w8a8(
    dif::ir::Program &program, dif::runtime::TensorMap &bindings,
    const dif::frontend::Flux2KleinTransformerBuild &transformer,
    std::span<const std::uint32_t> block_ids,
    dif::ir::Int8RowQuantization quantization,
    const fs::path &cache_directory, bool weight_equalization = false,
    bool mse_weight_scale = false, bool activation_residual2 = false,
    bool protect_qk_only = false) {
  W8A8RewriteResult result;
  if (block_ids.empty())
    return result;
  if (activation_residual2)
    dif::fail("activation residual2 is not implemented for split "
              "single-stream W8A8 rewrites");
  const auto convrot = quantization != dif::ir::Int8RowQuantization::Direct;
  const auto f32_convrot =
      quantization == dif::ir::Int8RowQuantization::H256F32ConvRot ||
      quantization == dif::ir::Int8RowQuantization::H256F32SignedConvRot ||
      quantization == dif::ir::Int8RowQuantization::H4096F32SignedConvRot;
  const auto signed_convrot =
      quantization == dif::ir::Int8RowQuantization::H256SignedConvRot ||
      quantization == dif::ir::Int8RowQuantization::H256F32SignedConvRot ||
      quantization == dif::ir::Int8RowQuantization::H4096SignedConvRot ||
      quantization == dif::ir::Int8RowQuantization::H4096F32SignedConvRot;
  constexpr std::uint64_t hidden = 4096U;
  constexpr std::uint64_t qkv_rows = 3U * hidden;
  constexpr std::uint64_t mlp_rows = 6U * hidden;
  constexpr std::uint64_t packed_rows = qkv_rows + mlp_rows;
  const auto protected_rows = (protect_qk_only ? 2U : 3U) * hidden;
  const auto quantized_rows = packed_rows - protected_rows;
  const auto rotation_group = convrot_group(quantization, hidden);

  auto next_tensor = std::uint32_t{0U};
  auto next_operation = std::uint32_t{0U};
  for (const auto &tensor : program.tensors)
    next_tensor = std::max(next_tensor, tensor.id);
  for (const auto &operation : program.operations)
    next_operation = std::max(next_operation, operation.id);

  std::vector<std::pair<std::uint32_t, std::uint32_t>> selected_weights;
  const std::unordered_set<std::uint32_t> requested_blocks(
      block_ids.begin(), block_ids.end());
  for (std::size_t index = 0U;
       index < transformer.checkpoint_names.size(); ++index) {
    const auto &name = transformer.checkpoint_names.at(index);
    constexpr std::string_view prefix = "single_blocks.";
    constexpr std::string_view suffix = ".linear1.weight";
    if (!name.starts_with(prefix) || !name.ends_with(suffix))
      continue;
    const auto block_end = name.find('.', prefix.size());
    if (block_end == std::string::npos)
      dif::fail("malformed FLUX.2 split single-stream W8A8 checkpoint name");
    const auto block = static_cast<std::uint32_t>(std::stoul(
        name.substr(prefix.size(), block_end - prefix.size())));
    if (requested_blocks.contains(block))
      selected_weights.emplace_back(block,
                                    transformer.checkpoint_tensors.at(index));
  }
  if (selected_weights.size() != block_ids.size())
    dif::fail("FLUX.2 split single-stream W8A8 selection did not resolve "
              "every block");

  fs::create_directories(cache_directory);
  for (const auto [block, weight_id] : selected_weights) {
    const auto *weight_description = program.tensor(weight_id);
    const auto binding = bindings.find(weight_id);
    if (!weight_description || binding == bindings.end() ||
        weight_description->dtype != dif::ir::DType::BF16 ||
        weight_description->dims !=
            std::vector<std::uint64_t>{packed_rows, hidden})
      dif::fail("FLUX.2 single packed projection is not BF16 [36864,4096]");
    const auto weight_desc = *weight_description;
    const auto linear = std::find_if(
        program.operations.begin(), program.operations.end(),
        [&](const dif::ir::Operation &operation) {
          return operation.opcode == dif::ir::Opcode::Linear &&
                 operation.inputs.size() == 2U &&
                 operation.inputs.at(1) == weight_id;
        });
    if (linear == program.operations.end())
      dif::fail("FLUX.2 single packed projection has no unbiased Linear");
    const auto linear_template = *linear;
    const auto linear_id = linear->id;
    const auto input_id = linear->inputs.at(0);
    const auto output_id = linear->outputs.at(0);
    const auto *input_description = program.tensor(input_id);
    const auto *output_description = program.tensor(output_id);
    if (!input_description || !output_description ||
        input_description->dtype != dif::ir::DType::BF16 ||
        output_description->dtype != dif::ir::DType::BF16 ||
        input_description->dims.empty() || output_description->dims.empty() ||
        input_description->dims.back() != hidden ||
        output_description->dims.back() != packed_rows)
      dif::fail("FLUX.2 single packed projection shape disagreement");
    const auto input_desc = *input_description;
    const auto output_desc = *output_description;
    if ((output_desc.roles & dif::ir::TensorRole::Output) != 0U)
      dif::fail("split single-stream W8A8 cannot remove a captured packed "
                "projection");
    auto cache_path =
        cache_directory /
        ("flux2-klein-base-9b-" + std::string(kModelRevision) +
         (protect_qk_only ? "-single-v-mlp-block-" : "-single-mlp-block-") +
         std::to_string(block) +
         (quantization ==
                        dif::ir::Int8RowQuantization::H4096SignedConvRot
              ? "-h4096-signed-coherent-convrot-w8a8.safetensors"
              : quantization ==
                        dif::ir::Int8RowQuantization::H4096F32SignedConvRot
              ? "-h4096-f32-signed-convrot-w8a8.safetensors"
              : f32_convrot
              ? (signed_convrot
                     ? "-h256-f32-signed-convrot-w8a8.safetensors"
                     : "-h256-f32-convrot-w8a8.safetensors")
              : signed_convrot
              ? "-h256-signed-coherent-convrot-w8a8.safetensors"
              : (convrot ? "-h256-convrot-w8a8.safetensors"
                         : "-w8a8.safetensors")));
    if (weight_equalization)
      cache_path = cache_path.parent_path() /
                   (cache_path.stem().string() +
                    "-weight-equalized.safetensors");
    if (mse_weight_scale)
      cache_path = cache_path.parent_path() /
                   (cache_path.stem().string() +
                    "-mse-weight-scale.safetensors");
    const std::vector<std::uint64_t> quantized_weight_dims{quantized_rows,
                                                           hidden};
    if (!fs::exists(cache_path)) {
      dif::runtime::Tensor quantized_weight{
          dif::ir::DType::I8, quantized_weight_dims, {}};
      quantized_weight.bytes.resize(
          static_cast<std::size_t>(quantized_rows * hidden));
      dif::runtime::Tensor weight_scales{
          dif::ir::DType::F32, {quantized_rows}, {}};
      weight_scales.bytes.resize(
          static_cast<std::size_t>(quantized_rows * sizeof(float)));
      dif::runtime::Tensor input_equalization{
          dif::ir::DType::BF16, {hidden}, {}};
      if (weight_equalization) {
        input_equalization.bytes.resize(
            static_cast<std::size_t>(hidden * sizeof(std::uint16_t)));
        std::vector<float> column_maxima(static_cast<std::size_t>(hidden),
                                         0.0F);
        for (std::uint64_t row = 0U; row < quantized_rows; ++row)
          for (std::uint64_t column = 0U; column < hidden; ++column)
            column_maxima[static_cast<std::size_t>(column)] = std::max(
                column_maxima[static_cast<std::size_t>(column)],
                std::fabs(dif::runtime::load_float(
                    binding->second,
                    (protected_rows + row) * hidden + column)));
        double log_sum = 0.0;
        for (const auto maximum : column_maxima)
          log_sum += std::log(std::max(maximum, 1.0e-30F));
        const auto center = static_cast<float>(
            std::exp(log_sum / static_cast<double>(column_maxima.size())));
        for (std::uint64_t column = 0U; column < hidden; ++column)
          dif::runtime::store_float(
              input_equalization, column,
              std::clamp(
                  std::sqrt(column_maxima[static_cast<std::size_t>(column)] /
                            std::max(center, 1.0e-30F)),
                  0.5F, 2.0F));
      }
      auto *quantized = reinterpret_cast<std::int8_t *>(
          quantized_weight.mutable_data());
      const auto encode_rows = [&](std::uint64_t begin, std::uint64_t end) {
        std::vector<float> transformed(static_cast<std::size_t>(hidden));
        for (auto row = begin; row < end; ++row) {
          const auto source_base = (protected_rows + row) * hidden;
          const auto target_base = row * hidden;
          float maximum = 0.0F;
          if (convrot) {
            for (std::uint64_t column = 0U; column < hidden; ++column)
              transformed[static_cast<std::size_t>(column)] =
                  dif::runtime::load_float(binding->second,
                                           source_base + column) /
                  (weight_equalization
                       ? dif::runtime::load_float(input_equalization, column)
                       : 1.0F) *
                  (signed_convrot
                       ? signed_convrot_sign(column)
                       : 1.0F);
            apply_convrot(transformed,
                          static_cast<std::size_t>(rotation_group));
            for (const auto value : transformed)
              maximum = std::max(maximum, std::fabs(value));
          } else {
            for (std::uint64_t column = 0U; column < hidden; ++column)
              maximum = std::max(
                  maximum,
                  std::fabs(dif::runtime::load_float(
                      binding->second, source_base + column)));
          }
          const auto scale =
              mse_weight_scale && convrot
                  ? choose_mse_int8_weight_scale(transformed)
                  : std::max(maximum / 127.0F, 1.0e-30F);
          const auto scale_bf16 = dif::runtime::bf16_to_float(
              dif::runtime::float_to_bf16(scale));
          dif::runtime::store_float(
              weight_scales, row,
              !f32_convrot &&
                      ((mse_weight_scale && convrot) || signed_convrot)
                  ? scale_bf16
                  : scale);
          for (std::uint64_t column = 0U; column < hidden; ++column) {
            int rounded{};
            if (f32_convrot) {
              rounded = static_cast<int>(std::nearbyint(
                  transformed[static_cast<std::size_t>(column)] / scale));
            } else if (convrot) {
              const auto value_bf16 = dif::runtime::bf16_to_float(
                  dif::runtime::float_to_bf16(
                      transformed[static_cast<std::size_t>(column)]));
              const auto divided_bf16 = dif::runtime::bf16_to_float(
                  dif::runtime::float_to_bf16(value_bf16 / scale_bf16));
              rounded = static_cast<int>(std::nearbyint(divided_bf16));
            } else {
              rounded = static_cast<int>(std::nearbyint(
                  (dif::runtime::load_float(binding->second,
                                            source_base + column) /
                   (weight_equalization
                        ? dif::runtime::load_float(input_equalization, column)
                        : 1.0F)) /
                  scale));
            }
            quantized[target_base + column] =
                static_cast<std::int8_t>(std::clamp(
                    rounded, convrot && !f32_convrot ? -128 : -127, 127));
          }
        }
      };
      const auto hardware_threads =
          std::max(1U, std::thread::hardware_concurrency());
      const auto worker_count =
          std::min<std::uint64_t>(quantized_rows, hardware_threads);
      std::vector<std::thread> workers;
      workers.reserve(static_cast<std::size_t>(worker_count));
      for (std::uint64_t worker = 0U; worker < worker_count; ++worker) {
        const auto begin = quantized_rows * worker / worker_count;
        const auto end = quantized_rows * (worker + 1U) / worker_count;
        workers.emplace_back(encode_rows, begin, end);
      }
      for (auto &worker : workers)
        worker.join();
      auto building_path = cache_path;
      building_path += ".building";
      if (fs::exists(building_path))
        dif::fail("refusing to overwrite an incomplete FLUX.2 split "
                  "single-stream cache: " + building_path.string());
      std::vector<dif::weights::SafeTensorWriteSpec> specs{
          {"weight", quantized_weight.dtype, quantized_weight.dims},
          {"scale", weight_scales.dtype, weight_scales.dims}};
      if (weight_equalization)
        specs.push_back({"input_scale", input_equalization.dtype,
                         input_equalization.dims});
      dif::weights::SafeTensorWriter writer(building_path, specs);
      writer.append("weight", std::span<const std::uint8_t>(
                                  quantized_weight.data(),
                                  quantized_weight.byte_size()));
      writer.append("scale", std::span<const std::uint8_t>(
                                 weight_scales.data(),
                                 weight_scales.byte_size()));
      if (weight_equalization)
        writer.append("input_scale", std::span<const std::uint8_t>(
                                         input_equalization.data(),
                                         input_equalization.byte_size()));
      (void)writer.finish();
      fs::rename(building_path, cache_path);
      std::cerr << "FLUX2_SINGLE_MLP_W8A8_CACHE_BUILD block=" << block
                << " path=" << cache_path << '\n';
    }

    const auto cache = dif::weights::read_safetensors(cache_path);
    auto quantized_weight = dif::weights::map_safetensor(cache, "weight");
    auto weight_scales = dif::weights::map_safetensor(cache, "scale");
    std::optional<dif::runtime::Tensor> input_equalization;
    if (weight_equalization)
      input_equalization.emplace(
          dif::weights::map_safetensor(cache, "input_scale"));
    if (quantized_weight.dtype != dif::ir::DType::I8 ||
        quantized_weight.dims != quantized_weight_dims ||
        weight_scales.dtype != dif::ir::DType::F32 ||
        weight_scales.dims != std::vector<std::uint64_t>{quantized_rows} ||
        (weight_equalization &&
         (input_equalization->dtype != dif::ir::DType::BF16 ||
          input_equalization->dims != std::vector<std::uint64_t>{hidden})))
      dif::fail("FLUX.2 split single-stream W8A8 cache shape/dtype "
                "disagreement");

    auto protected_weight = binding->second;
    protected_weight.dims = {protected_rows, hidden};
    const auto protected_bytes = static_cast<std::size_t>(
        protected_rows * hidden * dif::ir::dtype_size(dif::ir::DType::BF16));
    if (protected_weight.is_mapped()) {
      protected_weight.mapping_bytes = protected_bytes;
    } else {
      protected_weight.bytes.resize(protected_bytes);
    }
    protected_weight.validate();

    const auto protected_weight_id = ++next_tensor;
    const auto quantized_weight_id = ++next_tensor;
    const auto weight_scales_id = ++next_tensor;
    const auto quantized_input_id = ++next_tensor;
    const auto input_scales_id = ++next_tensor;
    const auto equalization_id =
        weight_equalization ? ++next_tensor : std::uint32_t{0U};
    const auto equalized_input_id =
        weight_equalization ? ++next_tensor : std::uint32_t{0U};
    const auto protected_flat_output_id = ++next_tensor;
    const auto quantized_flat_output_id = ++next_tensor;
    auto protected_flat_output_dims = input_desc.dims;
    auto quantized_flat_output_dims = input_desc.dims;
    protected_flat_output_dims.back() = protected_rows;
    quantized_flat_output_dims.back() = quantized_rows;
    program.tensors.push_back(
        {protected_weight_id, dif::ir::DType::BF16,
         weight_desc.roles, {protected_rows, hidden}});
    program.tensors.push_back(
        {quantized_weight_id, dif::ir::DType::I8,
         weight_desc.roles, quantized_weight_dims});
    program.tensors.push_back(
        {weight_scales_id, dif::ir::DType::F32,
         weight_desc.roles, {quantized_rows}});
    program.tensors.push_back(
        {quantized_input_id, dif::ir::DType::I8,
         dif::ir::TensorRole::Internal, input_desc.dims});
    program.tensors.push_back(
        {input_scales_id, dif::ir::DType::F32,
         dif::ir::TensorRole::Internal,
         {input_desc.element_count() / hidden}});
    program.tensors.push_back(
        {protected_flat_output_id, dif::ir::DType::BF16,
         dif::ir::TensorRole::Internal, protected_flat_output_dims});
    program.tensors.push_back(
        {quantized_flat_output_id, dif::ir::DType::BF16,
         dif::ir::TensorRole::Internal, quantized_flat_output_dims});
    if (weight_equalization) {
      program.tensors.push_back(
          {equalization_id, dif::ir::DType::BF16,
           dif::ir::TensorRole::Constant, {hidden}});
      program.tensors.push_back(
          {equalized_input_id, dif::ir::DType::BF16,
           dif::ir::TensorRole::Internal, input_desc.dims});
      bindings.emplace(equalization_id, std::move(*input_equalization));
    }
    bindings.emplace(protected_weight_id, std::move(protected_weight));
    bindings.emplace(quantized_weight_id, std::move(quantized_weight));
    bindings.emplace(weight_scales_id, std::move(weight_scales));
    bindings.erase(weight_id);

    auto protected_linear = linear_template;
    protected_linear.inputs = {input_id, protected_weight_id};
    protected_linear.outputs = {protected_flat_output_id};
    const dif::ir::Operation equalize{
        ++next_operation, dif::ir::Opcode::AffineLastDim,
        {input_id, equalization_id}, {equalized_input_id}, {}};
    dif::ir::Operation quantize{
        ++next_operation, dif::ir::Opcode::QuantizeInt8Rows,
        {weight_equalization ? equalized_input_id : input_id},
        {quantized_input_id, input_scales_id},
        {dif::ir::Attribute::u64(dif::ir::AttrKey::BlockSize, 256U),
         dif::ir::Attribute::u64(
             dif::ir::AttrKey::Implementation,
             static_cast<std::uint64_t>(quantization))}};
    const dif::ir::Operation scaled_linear{
        ++next_operation, dif::ir::Opcode::LinearInt8Scaled,
        {quantized_input_id, quantized_weight_id, input_scales_id,
         weight_scales_id},
        {quantized_flat_output_id}, {}};
    const dif::ir::Operation join_projection{
        ++next_operation, dif::ir::Opcode::Concat,
        {protected_flat_output_id, quantized_flat_output_id}, {output_id},
        {dif::ir::Attribute::u64(dif::ir::AttrKey::Axis,
                                 input_desc.dims.size() - 1U)}};
    std::vector<dif::ir::Operation> rewritten_operations;
    rewritten_operations.reserve(program.operations.size());
    for (const auto &operation : program.operations) {
      if (operation.id == linear_id) {
        rewritten_operations.push_back(protected_linear);
        if (weight_equalization)
          rewritten_operations.push_back(equalize);
        rewritten_operations.push_back(quantize);
        rewritten_operations.push_back(scaled_linear);
        rewritten_operations.push_back(join_projection);
      } else {
        rewritten_operations.push_back(operation);
      }
    }
    program.operations = std::move(rewritten_operations);
    program.tensors.erase(
        std::remove_if(program.tensors.begin(), program.tensors.end(),
                       [&](const dif::ir::TensorDesc &description) {
                         return description.id == weight_id;
                       }),
        program.tensors.end());
    ++result.linear_count;
    result.quantized_weight_bytes +=
        quantized_rows * hidden + quantized_rows * sizeof(float);
  }
  dif::ir::verify(program);
  return result;
}

DenoiseResult denoise(const Arguments &arguments,
                       const ConditioningResult &conditioning,
                       dif::runtime::Tensor latent,
                       std::uint64_t latent_width) {
  DenoiseResult result;
  const auto image_tokens = latent.dims.at(0);
  const bool dev = arguments.flux2_model == "dev";
  dif::frontend::Flux2KleinTransformerConfig config;
  if (dev)
    config.geometry = dif::frontend::flux2_dev_geometry();
  // [dev] is guidance-distilled: one conditional pass with the guidance
  // embedded, no CFG batch. Klein runs the CFG pair as before.
  config.batch_size = dev ? 1U : 2U;
  config.image_tokens = image_tokens;
  config.text_tokens = dev ? conditioning.positive.dims.at(0) : 512U;
  config.streamed_constants = true;
  config.attention_implementation =
      arguments.transformer_attention_implementation;
  auto transformer =
      dif::frontend::make_flux2_klein_9b_transformer(config);
  const auto checkpoint =
      dif::weights::read_safetensors(arguments.transformer_checkpoint);
  dif::runtime::TensorMap bindings = transformer.generated_constants;
  bindings.emplace(transformer.latent_input,
                   dev ? latent : batch_pair(latent, latent));
  bindings.emplace(transformer.conditioning_input,
                   dev ? conditioning.positive
                       : batch_pair(conditioning.negative,
                                    conditioning.positive));
  bindings.emplace(transformer.timestep_input,
                   repeated_scalar(dif::ir::DType::BF16, 1.0F,
                                   config.batch_size));
  if (transformer.guidance_input != 0U)
    bindings.emplace(transformer.guidance_input,
                     scalar(dif::ir::DType::BF16, arguments.guidance));
  bindings.emplace(transformer.position_ids_input,
                   position_ids(image_tokens, 512U, latent_width, 2U));
  for (std::size_t index = 0U;
       index < transformer.checkpoint_tensors.size(); ++index) {
    auto tensor = dif::weights::map_safetensor(
        checkpoint, transformer.checkpoint_names.at(index));
    const auto id = transformer.checkpoint_tensors.at(index);
    const auto *description = transformer.program.tensor(id);
    if (!description || tensor.dtype != description->dtype ||
        tensor.dims != description->dims)
      dif::fail("transformer checkpoint disagreement at " +
                transformer.checkpoint_names.at(index));
    bindings.emplace(id, std::move(tensor));
  }
  if (!arguments.squareq_w4_slab.empty()) {
    result.squareq_w4 = dif::frontend::rewrite_linear_weights_squareq_w4(
        transformer.program, bindings, transformer.checkpoint_tensors,
        transformer.checkpoint_names, arguments.squareq_w4_slab);
    std::cerr << "FLUX2_SQUAREQ_W4 format=" << result.squareq_w4.format
              << " rank=" << result.squareq_w4.rank
              << " linears=" << result.squareq_w4.linear_count
              << " slab_bytes=" << result.squareq_w4.quantized_bytes
              << " bf16_bytes_replaced=" << result.squareq_w4.bf16_bytes_replaced
              << " plan_cos_w_min=" << result.squareq_w4.plan_cos_w_min << '\n';
  }
  auto w8a8_blocks = arguments.w8a8_single_linear1_block_ids;
  if (w8a8_blocks.empty())
    for (std::uint32_t block = 0U;
         block < arguments.w8a8_single_linear1_blocks; ++block)
      w8a8_blocks.push_back(block);
  std::sort(w8a8_blocks.begin(), w8a8_blocks.end());
  auto single_mlp_w8a8_blocks = arguments.w8a8_single_mlp_block_ids;
  if (single_mlp_w8a8_blocks.empty())
    for (std::uint32_t block = 0U;
         block < arguments.w8a8_single_mlp_blocks; ++block)
      single_mlp_w8a8_blocks.push_back(block);
  std::sort(single_mlp_w8a8_blocks.begin(), single_mlp_w8a8_blocks.end());
  auto single_qk_w8a8_blocks = arguments.w8a8_single_qk_block_ids;
  if (single_qk_w8a8_blocks.empty())
    for (std::uint32_t block = 0U;
         block < arguments.w8a8_single_qk_blocks; ++block)
      single_qk_w8a8_blocks.push_back(block);
  std::sort(single_qk_w8a8_blocks.begin(), single_qk_w8a8_blocks.end());
  auto single_linear2_w8a8_blocks =
      arguments.w8a8_single_linear2_block_ids;
  if (single_linear2_w8a8_blocks.empty())
    for (std::uint32_t block = 0U;
         block < arguments.w8a8_single_linear2_blocks; ++block)
      single_linear2_w8a8_blocks.push_back(block);
  std::sort(single_linear2_w8a8_blocks.begin(),
            single_linear2_w8a8_blocks.end());
  auto double_w8a8_blocks = arguments.w8a8_double_image_mlp_blocks;
  std::sort(double_w8a8_blocks.begin(), double_w8a8_blocks.end());
  auto double_mlp_w8a8_blocks = arguments.w8a8_double_mlp_blocks;
  std::sort(double_mlp_w8a8_blocks.begin(),
            double_mlp_w8a8_blocks.end());
  auto full_double_w8a8_blocks = arguments.w8a8_double_image_blocks;
  std::sort(full_double_w8a8_blocks.begin(), full_double_w8a8_blocks.end());
  auto full_double_text_w8a8_blocks = arguments.w8a8_double_text_blocks;
  std::sort(full_double_text_w8a8_blocks.begin(),
            full_double_text_w8a8_blocks.end());
  auto complete_double_w8a8_blocks = arguments.w8a8_double_blocks;
  std::sort(complete_double_w8a8_blocks.begin(),
            complete_double_w8a8_blocks.end());
  auto single_linear1_fp8_blocks = arguments.fp8_single_linear1_block_ids;
  if (single_linear1_fp8_blocks.empty())
    for (std::uint32_t block = 0U;
         block < arguments.fp8_single_linear1_blocks; ++block)
      single_linear1_fp8_blocks.push_back(block);
  std::sort(single_linear1_fp8_blocks.begin(),
            single_linear1_fp8_blocks.end());
  auto single_mlp_fp8_blocks = arguments.fp8_single_mlp_block_ids;
  if (single_mlp_fp8_blocks.empty())
    for (std::uint32_t block = 0U;
         block < arguments.fp8_single_mlp_blocks; ++block)
      single_mlp_fp8_blocks.push_back(block);
  std::sort(single_mlp_fp8_blocks.begin(), single_mlp_fp8_blocks.end());
  auto single_linear2_fp8_blocks = arguments.fp8_single_linear2_block_ids;
  if (single_linear2_fp8_blocks.empty())
    for (std::uint32_t block = 0U;
         block < arguments.fp8_single_linear2_blocks; ++block)
      single_linear2_fp8_blocks.push_back(block);
  std::sort(single_linear2_fp8_blocks.begin(),
            single_linear2_fp8_blocks.end());
  auto double_fp8_blocks = arguments.fp8_double_image_mlp_blocks;
  std::sort(double_fp8_blocks.begin(), double_fp8_blocks.end());
  auto full_double_fp8_blocks = arguments.fp8_double_image_blocks;
  std::sort(full_double_fp8_blocks.begin(), full_double_fp8_blocks.end());
  auto full_double_text_fp8_blocks = arguments.fp8_double_text_blocks;
  std::sort(full_double_text_fp8_blocks.begin(),
            full_double_text_fp8_blocks.end());
  auto complete_double_fp8_blocks = arguments.fp8_double_blocks;
  std::sort(complete_double_fp8_blocks.begin(),
            complete_double_fp8_blocks.end());
  const auto intersects = [](std::span<const std::uint32_t> left,
                             std::span<const std::uint32_t> right) {
    return std::any_of(left.begin(), left.end(), [&](std::uint32_t value) {
      return std::binary_search(right.begin(), right.end(), value);
    });
  };
  if (intersects(w8a8_blocks, single_linear1_fp8_blocks) ||
      intersects(w8a8_blocks, single_mlp_w8a8_blocks) ||
      intersects(w8a8_blocks, single_qk_w8a8_blocks) ||
      intersects(single_mlp_w8a8_blocks, single_qk_w8a8_blocks) ||
      intersects(single_mlp_w8a8_blocks, single_linear1_fp8_blocks) ||
      intersects(single_qk_w8a8_blocks, single_linear1_fp8_blocks) ||
      intersects(w8a8_blocks, single_mlp_fp8_blocks) ||
      intersects(single_mlp_w8a8_blocks, single_mlp_fp8_blocks) ||
      intersects(single_qk_w8a8_blocks, single_mlp_fp8_blocks) ||
      intersects(single_linear2_w8a8_blocks, single_linear2_fp8_blocks) ||
      intersects(double_mlp_w8a8_blocks, double_w8a8_blocks) ||
      intersects(double_mlp_w8a8_blocks, full_double_w8a8_blocks) ||
      intersects(double_mlp_w8a8_blocks, full_double_text_w8a8_blocks) ||
      intersects(double_mlp_w8a8_blocks, complete_double_w8a8_blocks) ||
      intersects(double_mlp_w8a8_blocks, double_fp8_blocks) ||
      intersects(double_mlp_w8a8_blocks, full_double_fp8_blocks) ||
      intersects(double_mlp_w8a8_blocks, full_double_text_fp8_blocks) ||
      intersects(double_mlp_w8a8_blocks, complete_double_fp8_blocks) ||
      intersects(double_w8a8_blocks, double_fp8_blocks) ||
      intersects(double_w8a8_blocks, full_double_fp8_blocks) ||
      intersects(double_w8a8_blocks, full_double_w8a8_blocks) ||
      intersects(double_w8a8_blocks, complete_double_w8a8_blocks) ||
      intersects(double_fp8_blocks, full_double_fp8_blocks) ||
      intersects(double_w8a8_blocks, complete_double_fp8_blocks) ||
      intersects(double_fp8_blocks, complete_double_fp8_blocks) ||
      intersects(full_double_fp8_blocks, complete_double_fp8_blocks) ||
      intersects(full_double_text_fp8_blocks, complete_double_fp8_blocks) ||
      intersects(full_double_w8a8_blocks, complete_double_w8a8_blocks) ||
      intersects(full_double_text_w8a8_blocks,
                 complete_double_w8a8_blocks) ||
      intersects(full_double_w8a8_blocks, double_fp8_blocks) ||
      intersects(full_double_w8a8_blocks, full_double_fp8_blocks) ||
      intersects(full_double_w8a8_blocks, complete_double_fp8_blocks) ||
      intersects(full_double_text_w8a8_blocks,
                 full_double_text_fp8_blocks) ||
      intersects(full_double_text_w8a8_blocks,
                 complete_double_fp8_blocks) ||
      intersects(complete_double_w8a8_blocks, double_fp8_blocks) ||
      intersects(complete_double_w8a8_blocks, full_double_fp8_blocks) ||
      intersects(complete_double_w8a8_blocks,
                 full_double_text_fp8_blocks) ||
      intersects(complete_double_w8a8_blocks,
                 complete_double_fp8_blocks))
    dif::fail("FLUX.2 W8A8 and FP8 candidates overlap the same Linear");
  const auto single_w8a8 = rewrite_named_linear_w8a8(
      transformer.program, bindings, transformer, w8a8_blocks,
      arguments.w8a8_quantization, "single_blocks.", ".linear1.weight",
      "single-block", arguments.cache_directory, false,
      arguments.w8a8_mse_weight_scale,
      arguments.w8a8_activation_residual2 ||
          arguments.w8a8_activation_residual2_single_linear1);
  const auto single_mlp_w8a8 = rewrite_single_packed_tail_w8a8(
      transformer.program, bindings, transformer, single_mlp_w8a8_blocks,
      arguments.w8a8_single_mlp_h256_convrot
          ? dif::ir::Int8RowQuantization::H256ConvRot
          : arguments.w8a8_quantization,
      arguments.cache_directory,
      arguments.w8a8_weight_equalization, arguments.w8a8_mse_weight_scale,
      arguments.w8a8_activation_residual2);
  const auto single_qk_w8a8 = rewrite_single_packed_tail_w8a8(
      transformer.program, bindings, transformer, single_qk_w8a8_blocks,
      arguments.w8a8_quantization, arguments.cache_directory,
      arguments.w8a8_weight_equalization, arguments.w8a8_mse_weight_scale,
      arguments.w8a8_activation_residual2, true);
  const auto single_linear2_w8a8 = rewrite_named_linear_w8a8(
      transformer.program, bindings, transformer, single_linear2_w8a8_blocks,
      arguments.w8a8_quantization, "single_blocks.", ".linear2.weight",
      "single-linear2-block", arguments.cache_directory,
      arguments.w8a8_weight_equalization, arguments.w8a8_mse_weight_scale,
      arguments.w8a8_activation_residual2 ||
          arguments.w8a8_activation_residual2_single_linear2);
  const auto double_w8a8 = rewrite_named_linear_w8a8(
      transformer.program, bindings, transformer, double_w8a8_blocks,
      arguments.w8a8_quantization, "double_blocks.", ".img_mlp.0.weight",
      "double-image-mlp-block", arguments.cache_directory, false,
      arguments.w8a8_mse_weight_scale,
      arguments.w8a8_activation_residual2 ||
      arguments.w8a8_activation_residual2_double);
  const auto double_mlp_image_0_w8a8 = rewrite_named_linear_w8a8(
      transformer.program, bindings, transformer, double_mlp_w8a8_blocks,
      arguments.w8a8_quantization, "double_blocks.", ".img_mlp.0.weight",
      "double-image-mlp-block", arguments.cache_directory,
      arguments.w8a8_weight_equalization, arguments.w8a8_mse_weight_scale,
      arguments.w8a8_activation_residual2 ||
      arguments.w8a8_activation_residual2_double);
  const auto double_mlp_image_2_w8a8 = rewrite_named_linear_w8a8(
      transformer.program, bindings, transformer, double_mlp_w8a8_blocks,
      arguments.w8a8_quantization, "double_blocks.", ".img_mlp.2.weight",
      "double-image-mlp2-block", arguments.cache_directory,
      arguments.w8a8_weight_equalization, arguments.w8a8_mse_weight_scale,
      arguments.w8a8_activation_residual2 ||
      arguments.w8a8_activation_residual2_double);
  const auto double_mlp_text_0_w8a8 = rewrite_named_linear_w8a8(
      transformer.program, bindings, transformer, double_mlp_w8a8_blocks,
      arguments.w8a8_quantization, "double_blocks.", ".txt_mlp.0.weight",
      "double-text-mlp-block", arguments.cache_directory,
      arguments.w8a8_weight_equalization, arguments.w8a8_mse_weight_scale,
      arguments.w8a8_activation_residual2 ||
      arguments.w8a8_activation_residual2_double);
  const auto double_mlp_text_2_w8a8 = rewrite_named_linear_w8a8(
      transformer.program, bindings, transformer, double_mlp_w8a8_blocks,
      arguments.w8a8_quantization, "double_blocks.", ".txt_mlp.2.weight",
      "double-text-mlp2-block", arguments.cache_directory,
      arguments.w8a8_weight_equalization, arguments.w8a8_mse_weight_scale,
      arguments.w8a8_activation_residual2 ||
      arguments.w8a8_activation_residual2_double);
  const auto double_image_qkv_w8a8 = rewrite_named_linear_w8a8(
      transformer.program, bindings, transformer, full_double_w8a8_blocks,
      arguments.w8a8_quantization, "double_blocks.", ".img_attn.qkv.weight",
      "double-image-qkv-block", arguments.cache_directory, false,
      arguments.w8a8_mse_weight_scale,
      arguments.w8a8_activation_residual2 ||
      arguments.w8a8_activation_residual2_double);
  const auto double_image_projection_w8a8 = rewrite_named_linear_w8a8(
      transformer.program, bindings, transformer, full_double_w8a8_blocks,
      arguments.w8a8_quantization, "double_blocks.", ".img_attn.proj.weight",
      "double-image-proj-block", arguments.cache_directory, false,
      arguments.w8a8_mse_weight_scale,
      arguments.w8a8_activation_residual2 ||
      arguments.w8a8_activation_residual2_double);
  const auto double_image_mlp0_w8a8 = rewrite_named_linear_w8a8(
      transformer.program, bindings, transformer, full_double_w8a8_blocks,
      arguments.w8a8_quantization, "double_blocks.", ".img_mlp.0.weight",
      "double-image-mlp-block", arguments.cache_directory, false,
      arguments.w8a8_mse_weight_scale,
      arguments.w8a8_activation_residual2 ||
      arguments.w8a8_activation_residual2_double);
  const auto double_image_mlp2_w8a8 = rewrite_named_linear_w8a8(
      transformer.program, bindings, transformer, full_double_w8a8_blocks,
      arguments.w8a8_quantization, "double_blocks.", ".img_mlp.2.weight",
      "double-image-mlp2-block", arguments.cache_directory, false,
      arguments.w8a8_mse_weight_scale,
      arguments.w8a8_activation_residual2 ||
      arguments.w8a8_activation_residual2_double);
  const auto double_text_qkv_w8a8 = rewrite_named_linear_w8a8(
      transformer.program, bindings, transformer,
      full_double_text_w8a8_blocks, arguments.w8a8_quantization,
      "double_blocks.", ".txt_attn.qkv.weight", "double-text-qkv-block",
      arguments.cache_directory, false, arguments.w8a8_mse_weight_scale,
      arguments.w8a8_activation_residual2 ||
      arguments.w8a8_activation_residual2_double);
  const auto double_text_projection_w8a8 = rewrite_named_linear_w8a8(
      transformer.program, bindings, transformer,
      full_double_text_w8a8_blocks, arguments.w8a8_quantization,
      "double_blocks.", ".txt_attn.proj.weight", "double-text-proj-block",
      arguments.cache_directory, false, arguments.w8a8_mse_weight_scale,
      arguments.w8a8_activation_residual2 ||
      arguments.w8a8_activation_residual2_double);
  const auto double_text_mlp0_w8a8 = rewrite_named_linear_w8a8(
      transformer.program, bindings, transformer,
      full_double_text_w8a8_blocks, arguments.w8a8_quantization,
      "double_blocks.", ".txt_mlp.0.weight", "double-text-mlp-block",
      arguments.cache_directory, false, arguments.w8a8_mse_weight_scale,
      arguments.w8a8_activation_residual2 ||
      arguments.w8a8_activation_residual2_double);
  const auto double_text_mlp2_w8a8 = rewrite_named_linear_w8a8(
      transformer.program, bindings, transformer,
      full_double_text_w8a8_blocks, arguments.w8a8_quantization,
      "double_blocks.", ".txt_mlp.2.weight", "double-text-mlp2-block",
      arguments.cache_directory, false, arguments.w8a8_mse_weight_scale,
      arguments.w8a8_activation_residual2 ||
      arguments.w8a8_activation_residual2_double);
  const auto complete_double_image_qkv_w8a8 = rewrite_named_linear_w8a8(
      transformer.program, bindings, transformer,
      complete_double_w8a8_blocks, arguments.w8a8_quantization,
      "double_blocks.", ".img_attn.qkv.weight", "double-image-qkv-block",
      arguments.cache_directory, false, arguments.w8a8_mse_weight_scale,
      arguments.w8a8_activation_residual2 ||
      arguments.w8a8_activation_residual2_double);
  const auto complete_double_image_projection_w8a8 =
      rewrite_named_linear_w8a8(
          transformer.program, bindings, transformer,
          complete_double_w8a8_blocks, arguments.w8a8_quantization,
          "double_blocks.", ".img_attn.proj.weight",
          "double-image-proj-block", arguments.cache_directory, false,
          arguments.w8a8_mse_weight_scale,
      arguments.w8a8_activation_residual2 ||
      arguments.w8a8_activation_residual2_double);
  const auto complete_double_image_mlp0_w8a8 = rewrite_named_linear_w8a8(
      transformer.program, bindings, transformer,
      complete_double_w8a8_blocks, arguments.w8a8_quantization,
      "double_blocks.", ".img_mlp.0.weight", "double-image-mlp-block",
      arguments.cache_directory, false, arguments.w8a8_mse_weight_scale,
      arguments.w8a8_activation_residual2 ||
      arguments.w8a8_activation_residual2_double);
  const auto complete_double_image_mlp2_w8a8 = rewrite_named_linear_w8a8(
      transformer.program, bindings, transformer,
      complete_double_w8a8_blocks, arguments.w8a8_quantization,
      "double_blocks.", ".img_mlp.2.weight", "double-image-mlp2-block",
      arguments.cache_directory, false, arguments.w8a8_mse_weight_scale,
      arguments.w8a8_activation_residual2 ||
      arguments.w8a8_activation_residual2_double);
  const auto complete_double_text_qkv_w8a8 = rewrite_named_linear_w8a8(
      transformer.program, bindings, transformer,
      complete_double_w8a8_blocks, arguments.w8a8_quantization,
      "double_blocks.", ".txt_attn.qkv.weight", "double-text-qkv-block",
      arguments.cache_directory, false, arguments.w8a8_mse_weight_scale,
      arguments.w8a8_activation_residual2 ||
      arguments.w8a8_activation_residual2_double);
  const auto complete_double_text_projection_w8a8 =
      rewrite_named_linear_w8a8(
          transformer.program, bindings, transformer,
          complete_double_w8a8_blocks, arguments.w8a8_quantization,
          "double_blocks.", ".txt_attn.proj.weight", "double-text-proj-block",
          arguments.cache_directory, false, arguments.w8a8_mse_weight_scale,
      arguments.w8a8_activation_residual2 ||
      arguments.w8a8_activation_residual2_double);
  const auto complete_double_text_mlp0_w8a8 = rewrite_named_linear_w8a8(
      transformer.program, bindings, transformer,
      complete_double_w8a8_blocks, arguments.w8a8_quantization,
      "double_blocks.", ".txt_mlp.0.weight", "double-text-mlp-block",
      arguments.cache_directory, false, arguments.w8a8_mse_weight_scale,
      arguments.w8a8_activation_residual2 ||
      arguments.w8a8_activation_residual2_double);
  const auto complete_double_text_mlp2_w8a8 = rewrite_named_linear_w8a8(
      transformer.program, bindings, transformer,
      complete_double_w8a8_blocks, arguments.w8a8_quantization,
      "double_blocks.", ".txt_mlp.2.weight", "double-text-mlp2-block",
      arguments.cache_directory, false, arguments.w8a8_mse_weight_scale,
      arguments.w8a8_activation_residual2 ||
      arguments.w8a8_activation_residual2_double);
  const auto single_linear1_fp8 = rewrite_named_linear_fp8(
      transformer.program, bindings, transformer, single_linear1_fp8_blocks,
      "single_blocks.", ".linear1.weight", "single-linear1-block",
      arguments.cache_directory, arguments.fp8_row_scaled);
  const auto single_mlp_fp8 = rewrite_single_mlp_fp8(
      transformer.program, bindings, transformer, single_mlp_fp8_blocks,
      arguments.cache_directory);
  const auto single_linear2_fp8 = rewrite_named_linear_fp8(
      transformer.program, bindings, transformer, single_linear2_fp8_blocks,
      "single_blocks.", ".linear2.weight", "single-linear2-block",
      arguments.cache_directory, arguments.fp8_row_scaled);
  const auto double_fp8 = rewrite_named_linear_fp8(
      transformer.program, bindings, transformer, double_fp8_blocks,
      "double_blocks.", ".img_mlp.0.weight", "double-image-mlp-block",
      arguments.cache_directory, arguments.fp8_row_scaled);
  const auto double_image_qkv_fp8 = rewrite_named_linear_fp8(
      transformer.program, bindings, transformer, full_double_fp8_blocks,
      "double_blocks.", ".img_attn.qkv.weight", "double-image-qkv-block",
      arguments.cache_directory, arguments.fp8_row_scaled);
  const auto double_image_projection_fp8 = rewrite_named_linear_fp8(
      transformer.program, bindings, transformer, full_double_fp8_blocks,
      "double_blocks.", ".img_attn.proj.weight", "double-image-proj-block",
      arguments.cache_directory, arguments.fp8_row_scaled);
  const auto double_image_mlp0_fp8 = rewrite_named_linear_fp8(
      transformer.program, bindings, transformer, full_double_fp8_blocks,
      "double_blocks.", ".img_mlp.0.weight", "double-image-mlp-block",
      arguments.cache_directory, arguments.fp8_row_scaled);
  const auto double_image_mlp2_fp8 = rewrite_named_linear_fp8(
      transformer.program, bindings, transformer, full_double_fp8_blocks,
      "double_blocks.", ".img_mlp.2.weight", "double-image-mlp2-block",
      arguments.cache_directory, arguments.fp8_row_scaled);
  const auto double_text_qkv_fp8 = rewrite_named_linear_fp8(
      transformer.program, bindings, transformer, full_double_text_fp8_blocks,
      "double_blocks.", ".txt_attn.qkv.weight", "double-text-qkv-block",
      arguments.cache_directory, arguments.fp8_row_scaled);
  const auto double_text_projection_fp8 = rewrite_named_linear_fp8(
      transformer.program, bindings, transformer, full_double_text_fp8_blocks,
      "double_blocks.", ".txt_attn.proj.weight", "double-text-proj-block",
      arguments.cache_directory, arguments.fp8_row_scaled);
  const auto double_text_mlp0_fp8 = rewrite_named_linear_fp8(
      transformer.program, bindings, transformer, full_double_text_fp8_blocks,
      "double_blocks.", ".txt_mlp.0.weight", "double-text-mlp-block",
      arguments.cache_directory, arguments.fp8_row_scaled);
  const auto double_text_mlp2_fp8 = rewrite_named_linear_fp8(
      transformer.program, bindings, transformer, full_double_text_fp8_blocks,
      "double_blocks.", ".txt_mlp.2.weight", "double-text-mlp2-block",
      arguments.cache_directory, arguments.fp8_row_scaled);
  const auto complete_double_image_qkv_fp8 = rewrite_named_linear_fp8(
      transformer.program, bindings, transformer, complete_double_fp8_blocks,
      "double_blocks.", ".img_attn.qkv.weight", "double-image-qkv-block",
      arguments.cache_directory, arguments.fp8_row_scaled);
  const auto complete_double_image_projection_fp8 = rewrite_named_linear_fp8(
      transformer.program, bindings, transformer, complete_double_fp8_blocks,
      "double_blocks.", ".img_attn.proj.weight", "double-image-proj-block",
      arguments.cache_directory, arguments.fp8_row_scaled);
  const auto complete_double_image_mlp0_fp8 = rewrite_named_linear_fp8(
      transformer.program, bindings, transformer, complete_double_fp8_blocks,
      "double_blocks.", ".img_mlp.0.weight", "double-image-mlp-block",
      arguments.cache_directory, arguments.fp8_row_scaled);
  const auto complete_double_image_mlp2_fp8 = rewrite_named_linear_fp8(
      transformer.program, bindings, transformer, complete_double_fp8_blocks,
      "double_blocks.", ".img_mlp.2.weight", "double-image-mlp2-block",
      arguments.cache_directory, arguments.fp8_row_scaled);
  const auto complete_double_text_qkv_fp8 = rewrite_named_linear_fp8(
      transformer.program, bindings, transformer, complete_double_fp8_blocks,
      "double_blocks.", ".txt_attn.qkv.weight", "double-text-qkv-block",
      arguments.cache_directory, arguments.fp8_row_scaled);
  const auto complete_double_text_projection_fp8 = rewrite_named_linear_fp8(
      transformer.program, bindings, transformer, complete_double_fp8_blocks,
      "double_blocks.", ".txt_attn.proj.weight", "double-text-proj-block",
      arguments.cache_directory, arguments.fp8_row_scaled);
  const auto complete_double_text_mlp0_fp8 = rewrite_named_linear_fp8(
      transformer.program, bindings, transformer, complete_double_fp8_blocks,
      "double_blocks.", ".txt_mlp.0.weight", "double-text-mlp-block",
      arguments.cache_directory, arguments.fp8_row_scaled);
  const auto complete_double_text_mlp2_fp8 = rewrite_named_linear_fp8(
      transformer.program, bindings, transformer, complete_double_fp8_blocks,
      "double_blocks.", ".txt_mlp.2.weight", "double-text-mlp2-block",
      arguments.cache_directory, arguments.fp8_row_scaled);
  result.w8a8_linear_count = single_w8a8.linear_count +
      single_mlp_w8a8.linear_count +
      single_qk_w8a8.linear_count +
      single_linear2_w8a8.linear_count + double_w8a8.linear_count +
      double_mlp_image_0_w8a8.linear_count +
      double_mlp_image_2_w8a8.linear_count +
      double_mlp_text_0_w8a8.linear_count +
      double_mlp_text_2_w8a8.linear_count +
      double_image_qkv_w8a8.linear_count +
      double_image_projection_w8a8.linear_count +
      double_image_mlp0_w8a8.linear_count +
      double_image_mlp2_w8a8.linear_count +
      double_text_qkv_w8a8.linear_count +
      double_text_projection_w8a8.linear_count +
      double_text_mlp0_w8a8.linear_count +
      double_text_mlp2_w8a8.linear_count +
      complete_double_image_qkv_w8a8.linear_count +
      complete_double_image_projection_w8a8.linear_count +
      complete_double_image_mlp0_w8a8.linear_count +
      complete_double_image_mlp2_w8a8.linear_count +
      complete_double_text_qkv_w8a8.linear_count +
      complete_double_text_projection_w8a8.linear_count +
      complete_double_text_mlp0_w8a8.linear_count +
      complete_double_text_mlp2_w8a8.linear_count;
  result.w8a8_weight_bytes = single_w8a8.quantized_weight_bytes +
                              single_mlp_w8a8.quantized_weight_bytes +
                              single_qk_w8a8.quantized_weight_bytes +
                              single_linear2_w8a8.quantized_weight_bytes +
                              double_w8a8.quantized_weight_bytes +
                              double_mlp_image_0_w8a8.quantized_weight_bytes +
                              double_mlp_image_2_w8a8.quantized_weight_bytes +
                              double_mlp_text_0_w8a8.quantized_weight_bytes +
                              double_mlp_text_2_w8a8.quantized_weight_bytes +
                              double_image_qkv_w8a8.quantized_weight_bytes +
                              double_image_projection_w8a8.quantized_weight_bytes +
                              double_image_mlp0_w8a8.quantized_weight_bytes +
                              double_image_mlp2_w8a8.quantized_weight_bytes +
                              double_text_qkv_w8a8.quantized_weight_bytes +
                              double_text_projection_w8a8.quantized_weight_bytes +
                              double_text_mlp0_w8a8.quantized_weight_bytes +
                              double_text_mlp2_w8a8.quantized_weight_bytes +
                              complete_double_image_qkv_w8a8.quantized_weight_bytes +
                              complete_double_image_projection_w8a8.quantized_weight_bytes +
                              complete_double_image_mlp0_w8a8.quantized_weight_bytes +
                              complete_double_image_mlp2_w8a8.quantized_weight_bytes +
                              complete_double_text_qkv_w8a8.quantized_weight_bytes +
                              complete_double_text_projection_w8a8.quantized_weight_bytes +
                              complete_double_text_mlp0_w8a8.quantized_weight_bytes +
                              complete_double_text_mlp2_w8a8.quantized_weight_bytes;
  result.w8a8_quantization = arguments.w8a8_quantization;
  result.w8a8_single_mlp_quantization =
      arguments.w8a8_single_mlp_h256_convrot
          ? dif::ir::Int8RowQuantization::H256ConvRot
          : arguments.w8a8_quantization;
  result.w8a8_blocks = std::move(w8a8_blocks);
  result.w8a8_single_mlp_blocks = std::move(single_mlp_w8a8_blocks);
  result.w8a8_single_qk_blocks = std::move(single_qk_w8a8_blocks);
  result.w8a8_single_linear2_blocks =
      std::move(single_linear2_w8a8_blocks);
  result.w8a8_double_image_mlp_blocks = std::move(double_w8a8_blocks);
  result.w8a8_double_mlp_blocks = std::move(double_mlp_w8a8_blocks);
  result.w8a8_double_image_blocks = std::move(full_double_w8a8_blocks);
  result.w8a8_double_text_blocks =
      std::move(full_double_text_w8a8_blocks);
  result.w8a8_double_blocks = std::move(complete_double_w8a8_blocks);
  if (result.w8a8_linear_count != 0U)
    std::cerr << "FLUX2_W8A8_CANDIDATE linear_count="
              << result.w8a8_linear_count
              << " quantized_weight_bytes=" << result.w8a8_weight_bytes
              << " row_quantization="
              << quantization_name(arguments.w8a8_quantization)
              << " single_mlp_row_quantization="
              << quantization_name(result.w8a8_single_mlp_quantization)
              << " activation_residual2="
              << (arguments.w8a8_activation_residual2 ? "true" : "false")
              << '\n';
  result.fp8_linear_count = single_linear1_fp8.linear_count +
                            single_mlp_fp8.linear_count +
                            single_linear2_fp8.linear_count +
                            double_fp8.linear_count +
                            double_image_qkv_fp8.linear_count +
                            double_image_projection_fp8.linear_count +
                            double_image_mlp0_fp8.linear_count +
                            double_image_mlp2_fp8.linear_count +
                            double_text_qkv_fp8.linear_count +
                            double_text_projection_fp8.linear_count +
                            double_text_mlp0_fp8.linear_count +
                            double_text_mlp2_fp8.linear_count +
                            complete_double_image_qkv_fp8.linear_count +
                            complete_double_image_projection_fp8.linear_count +
                            complete_double_image_mlp0_fp8.linear_count +
                            complete_double_image_mlp2_fp8.linear_count +
                            complete_double_text_qkv_fp8.linear_count +
                            complete_double_text_projection_fp8.linear_count +
                            complete_double_text_mlp0_fp8.linear_count +
                            complete_double_text_mlp2_fp8.linear_count;
  result.fp8_weight_bytes = single_linear1_fp8.quantized_weight_bytes +
                            single_mlp_fp8.quantized_weight_bytes +
                            single_linear2_fp8.quantized_weight_bytes +
                            double_fp8.quantized_weight_bytes +
                            double_image_qkv_fp8.quantized_weight_bytes +
                            double_image_projection_fp8.quantized_weight_bytes +
                            double_image_mlp0_fp8.quantized_weight_bytes +
                            double_image_mlp2_fp8.quantized_weight_bytes +
                            double_text_qkv_fp8.quantized_weight_bytes +
                            double_text_projection_fp8.quantized_weight_bytes +
                            double_text_mlp0_fp8.quantized_weight_bytes +
                            double_text_mlp2_fp8.quantized_weight_bytes +
                            complete_double_image_qkv_fp8.quantized_weight_bytes +
                            complete_double_image_projection_fp8.quantized_weight_bytes +
                            complete_double_image_mlp0_fp8.quantized_weight_bytes +
                            complete_double_image_mlp2_fp8.quantized_weight_bytes +
                            complete_double_text_qkv_fp8.quantized_weight_bytes +
                            complete_double_text_projection_fp8.quantized_weight_bytes +
                            complete_double_text_mlp0_fp8.quantized_weight_bytes +
                            complete_double_text_mlp2_fp8.quantized_weight_bytes;
  result.fp8_single_linear1_blocks = std::move(single_linear1_fp8_blocks);
  result.fp8_single_mlp_blocks = std::move(single_mlp_fp8_blocks);
  result.fp8_single_linear2_blocks = std::move(single_linear2_fp8_blocks);
  result.fp8_double_image_mlp_blocks = std::move(double_fp8_blocks);
  result.fp8_double_image_blocks = std::move(full_double_fp8_blocks);
  result.fp8_double_text_blocks = std::move(full_double_text_fp8_blocks);
  result.fp8_double_blocks = std::move(complete_double_fp8_blocks);
  result.fp8_row_scaled = arguments.fp8_row_scaled;
  if (result.fp8_linear_count != 0U)
    std::cerr << "FLUX2_FP8_CANDIDATE linear_count="
              << result.fp8_linear_count
              << " quantized_weight_bytes=" << result.fp8_weight_bytes
              << " format="
              << (arguments.fp8_row_scaled
                      ? "fp8-e4m3-f32-row-scaled"
                      : "mxfp8-e4m3-ue8m0-block32")
              << '\n';
  const auto weight_only_group32 =
      arguments.int8_weight_only_group32_names.empty()
          ? W8A8RewriteResult{}
          : rewrite_all_linear_weights_int8(
                transformer.program, bindings, transformer,
                arguments.cache_directory, 32U, {},
                arguments.int8_weight_only_group32_names);
  const auto weight_only =
      arguments.int8_weight_only_row_scaled_all_linears
          ? rewrite_all_linear_weights_int8_row_scaled(
                transformer.program, bindings, transformer,
                arguments.cache_directory)
          : arguments.int8_weight_only_all_linears
                ? rewrite_all_linear_weights_int8(
                      transformer.program, bindings, transformer,
                      arguments.cache_directory,
                      arguments.int8_weight_only_group_size,
                      arguments.int8_weight_only_exclude_names)
                : W8A8RewriteResult{};
  result.int8_weight_only_linear_count = weight_only.linear_count;
  result.int8_weight_only_linear_count += weight_only_group32.linear_count;
  result.int8_weight_only_group_size =
      arguments.int8_weight_only_group_size;
  result.int8_weight_only_bytes = weight_only.quantized_weight_bytes;
  result.int8_weight_only_bytes += weight_only_group32.quantized_weight_bytes;
  result.int8_weight_only_row_scaled =
      arguments.int8_weight_only_row_scaled_all_linears;
  result.int8_weight_only_group32_names =
      arguments.int8_weight_only_group32_names;
  if (result.int8_weight_only_linear_count != 0U)
    std::cerr << "FLUX2_INT8_WEIGHT_ONLY_CANDIDATE linear_count="
              << result.int8_weight_only_linear_count
              << " quantized_weight_bytes="
              << result.int8_weight_only_bytes
              << " mode="
              << (arguments.int8_weight_only_row_scaled_all_linears
                      ? "fused-row-scaled"
                      : "groupwise-bf16-dequant")
              << " group_size="
              << (arguments.int8_weight_only_row_scaled_all_linears
                      ? 0U
                      : arguments.int8_weight_only_group_size)
              << " group32_override_count="
              << result.int8_weight_only_group32_names.size()
              << " compute=bf16\n";

  std::optional<std::uint32_t> dynamic_clip_input;
  if (result.w8a8_linear_count != 0U) {
    const auto family_ratio = [&](const dif::ir::Operation &quantize) {
      const auto scaled = std::find_if(
          transformer.program.operations.begin(),
          transformer.program.operations.end(),
          [&](const dif::ir::Operation &operation) {
            return operation.opcode == dif::ir::Opcode::LinearInt8Scaled &&
                   !operation.inputs.empty() && !quantize.outputs.empty() &&
                   operation.inputs.front() == quantize.outputs.front();
          });
      if (scaled == transformer.program.operations.end() ||
          scaled->inputs.size() < 2U)
        dif::fail("FLUX.2 W8A8 quantizer lost its scaled Linear consumer");
      const auto *weight = transformer.program.tensor(scaled->inputs[1]);
      if (!weight || weight->dims.size() != 2U)
        dif::fail("FLUX.2 W8A8 scaled Linear lost its weight shape");
      double override = arguments.w8a8_activation_clip_ratio_double;
      if (weight->dims == std::vector<std::uint64_t>{36864U, 4096U})
        override = arguments.w8a8_activation_clip_ratio_single_linear1;
      else if (weight->dims == std::vector<std::uint64_t>{4096U, 16384U})
        override = arguments.w8a8_activation_clip_ratio_single_linear2;
      return std::isfinite(override)
                 ? override
                 : arguments.w8a8_activation_clip_ratio;
    };
    for (auto &operation : transformer.program.operations) {
      if (operation.opcode != dif::ir::Opcode::QuantizeInt8Rows)
        continue;
      operation.attributes.erase(
          std::remove_if(operation.attributes.begin(),
                         operation.attributes.end(), [](const auto &attribute) {
                           return attribute.key == dif::ir::AttrKey::Scale;
                         }),
          operation.attributes.end());
      const auto ratio = family_ratio(operation);
      if (ratio != 1.0)
        operation.attributes.push_back(dif::ir::Attribute::f64(
            dif::ir::AttrKey::Scale, ratio));
    }
    if (arguments.w8a8_activation_clip_switch_step != 0U) {
      auto next_tensor = std::uint32_t{0U};
      for (const auto &tensor : transformer.program.tensors)
        next_tensor = std::max(next_tensor, tensor.id);
      dynamic_clip_input = ++next_tensor;
      transformer.program.tensors.push_back(
          {*dynamic_clip_input, dif::ir::DType::F32,
           dif::ir::TensorRole::Input, {1U}});
      for (auto &operation : transformer.program.operations)
        if (operation.opcode == dif::ir::Opcode::QuantizeInt8Rows)
          operation.inputs.push_back(*dynamic_clip_input);
      bindings.emplace(*dynamic_clip_input,
                       scalar(dif::ir::DType::F32,
                              static_cast<float>(
                                  arguments.w8a8_activation_clip_ratio)));
    }
    dif::ir::verify(transformer.program);
    std::cerr << "FLUX2_W8A8_ACTIVATION_CLIP default="
              << arguments.w8a8_activation_clip_ratio
              << " single_linear1="
              << (std::isfinite(
                      arguments.w8a8_activation_clip_ratio_single_linear1)
                      ? arguments.w8a8_activation_clip_ratio_single_linear1
                      : arguments.w8a8_activation_clip_ratio)
              << " single_linear2="
              << (std::isfinite(
                      arguments.w8a8_activation_clip_ratio_single_linear2)
                      ? arguments.w8a8_activation_clip_ratio_single_linear2
                      : arguments.w8a8_activation_clip_ratio)
              << " double="
              << (std::isfinite(arguments.w8a8_activation_clip_ratio_double)
                      ? arguments.w8a8_activation_clip_ratio_double
                      : arguments.w8a8_activation_clip_ratio)
              << " switch_step="
              << arguments.w8a8_activation_clip_switch_step
              << " after="
              << (std::isfinite(arguments.w8a8_activation_clip_after_ratio)
                      ? arguments.w8a8_activation_clip_after_ratio
                      : arguments.w8a8_activation_clip_ratio)
              << '\n';
  }

  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 256ULL * 1024ULL * 1024ULL;
  options.cache_directory = arguments.cache_directory;
  options.profile_pipeline = arguments.profile_pipeline;
  options.cudnn_attention_heuristic =
      arguments.cudnn_attention_heuristic;
  options.streamed_prefetch_depth = arguments.streamed_prefetch_depth;
  options.streamed_staging_buffers = arguments.streamed_staging_buffers;
  options.streamed_stage_threads = arguments.streamed_stage_threads;
  options.streamed_release_mapped_pages_per_copy =
      !arguments.streamed_keep_mapped_pages;
  options.streamed_keep_mapped_pages_between_runs =
      arguments.streamed_keep_mapped_pages;
  options.tune_linear_operations = arguments.tune_linear_operations;
  options.linear_algorithm_choices = arguments.linear_algorithm_choices;
  options.linear_tuning_warmups = arguments.linear_tuning_warmups;
  options.linear_tuning_iterations = arguments.linear_tuning_iterations;
  options.linear_tuning_sessions = arguments.linear_tuning_sessions;
  options.expand_linear_algorithms = arguments.expand_linear_algorithms;
  options.persist_linear_heuristics = arguments.persist_linear_heuristics;
  const auto reshape_aliases =
      dif::compiler::plan_reshape_aliases(transformer.program);
  options.alias_reshape_operations = reshape_aliases.operation_ids;
  if (arguments.resident_plan_mib != 0U) {
    const auto plan_started = Clock::now();
    const auto residency = dif::compiler::plan_streamed_residency(
        transformer.program,
        arguments.resident_plan_mib * 1024ULL * 1024ULL,
        64ULL * 1024ULL * 1024ULL, arguments.streamed_prefetch_depth, {},
        arguments.resident_order);
    result.residency_plan_ms = elapsed_ms(plan_started);
    result.resident_weight_bytes = residency.resident_constant_bytes;
    result.streamed_weight_bytes = residency.streamed_constant_bytes;
    result.residency_required_bytes = residency.required_bytes;
    result.resident_weight_tensors = residency.resident_tensor_ids.size();
    options.resident_streamed_constants = residency.resident_tensor_ids;
    options.pipelined_resident_upload = !arguments.lazy_resident_upload;
    options.lazy_resident_upload = arguments.lazy_resident_upload;
    std::cerr << "FLUX2_RESIDENCY_PLAN resident_tensors="
              << residency.resident_tensor_ids.size()
              << " resident_bytes=" << residency.resident_constant_bytes
              << " streamed_bytes=" << residency.streamed_constant_bytes
              << " required_bytes=" << residency.required_bytes
              << " budget_bytes=" << residency.maximum_total_bytes << '\n';
  }
  auto backend = dif::runtime::make_cuda_executor();
  auto started = Clock::now();
  auto prepared = backend->prepare(transformer.program, bindings, options);
  result.preparation_ms = elapsed_ms(started);

  const auto cfg =
      dif::frontend::make_flux2_klein_base_cfg_euler_step(latent.dims);
  dif::runtime::TensorMap cfg_bindings;
  cfg_bindings.emplace(cfg.sample_input, latent);
  cfg_bindings.emplace(cfg.conditional_velocity_input, latent);
  cfg_bindings.emplace(cfg.unconditional_velocity_input, latent);
  // dev: both velocity inputs receive the single prediction and the CFG
  // scale is 1, so the Euler update sees exactly that prediction.
  cfg_bindings.emplace(cfg.guidance_input,
                       scalar(dif::ir::DType::BF16,
                              dev ? 1.0F : arguments.guidance));
  cfg_bindings.emplace(cfg.current_timestep_input,
                       scalar(dif::ir::DType::F32, 1.0F));
  cfg_bindings.emplace(cfg.next_timestep_input,
                       scalar(dif::ir::DType::F32, 0.0F));
  cfg_bindings.emplace(cfg.negative_one_constant,
                       scalar(dif::ir::DType::BF16, -1.0F));
  dif::runtime::RunOptions cfg_options;
  cfg_options.warmups = 0U;
  cfg_options.iterations = 1U;
  cfg_options.minimum_free_bytes = 0U;
  cfg_options.cache_directory = arguments.cache_directory;
  auto cfg_prepared =
      backend->prepare(cfg.program, cfg_bindings, cfg_options);

  const auto schedule =
      dif::sampling::make_flux2_klein_schedule(arguments.steps, image_tokens);
  const auto executed =
      arguments.stop_after == 0U
          ? arguments.steps - arguments.start_step
          : arguments.stop_after;
  result.cfg_transformer_ms.reserve(executed);
  result.scheduler_ms.reserve(executed);
  for (std::uint32_t step = 0U; step < executed; ++step) {
    const auto absolute_step = arguments.start_step + step;
    const auto current = schedule.at(absolute_step);
    const auto next = schedule.at(absolute_step + 1U);
    bindings.insert_or_assign(transformer.latent_input,
                              dev ? latent : batch_pair(latent, latent));
    bindings.insert_or_assign(transformer.timestep_input,
                              repeated_scalar(dif::ir::DType::BF16, current,
                                              config.batch_size));
    if (dynamic_clip_input)
      bindings.insert_or_assign(
          *dynamic_clip_input,
          scalar(dif::ir::DType::F32,
                 static_cast<float>(
                     absolute_step <
                             arguments.w8a8_activation_clip_switch_step
                         ? arguments.w8a8_activation_clip_ratio
                         : arguments.w8a8_activation_clip_after_ratio)));
    started = Clock::now();
    auto prediction = prepared->run(bindings, options);
    result.cfg_transformer_ms.push_back(elapsed_ms(started));
    if (arguments.profile_pipeline) {
      const auto &profile = prediction.pipeline_profile;
      const auto &telemetry = prediction.run_telemetry;
      std::cerr << "FLUX2_PIPELINE_PROFILE step=" << (step + 1U)
                << " streamed_weight_bytes=" << profile.streamed_weight_bytes
                << " host_stage_ms="
                << profile.streamed_host_stage_milliseconds
                << " host_wait_ms="
                << profile.streamed_host_wait_milliseconds
                << " h2d_ms=" << profile.streamed_h2d_milliseconds
                << " operation_kernel_ms="
                << profile.operation_kernel_milliseconds
                << " attention_kernel_ms="
                << profile.attention_kernel_milliseconds
                << " non_kernel_device_ms="
                << profile.non_kernel_device_timeline_milliseconds
                << " resident_prefault_ms="
                << profile.resident_host_prefault_milliseconds
                << " resident_minor_faults="
                << profile.resident_minor_page_faults
                << " resident_major_faults="
                << profile.resident_major_page_faults
                << " resident_h2d_ms="
                << profile.resident_h2d_milliseconds
                << " resident_upload_ms="
                << profile.resident_upload_milliseconds
                << " kernel_launches=" << telemetry.kernel_launches
                << " cublaslt_matmuls=" << telemetry.cublaslt_matmuls
                << " h2d_copies=" << telemetry.h2d_copies << '\n';
      auto timings = prediction.operation_timings;
      std::sort(timings.begin(), timings.end(),
                [](const auto &left, const auto &right) {
                  return left.mean_milliseconds > right.mean_milliseconds;
                });
      const auto limit = timings.size();
      for (std::size_t index = 0U; index < limit; ++index) {
        const auto &timing = timings[index];
        std::cerr << "FLUX2_OPERATION_PROFILE rank=" << (index + 1U)
                  << " operation=" << timing.operation_id
                  << " opcode=" << dif::ir::opcode_name(timing.opcode)
                  << " mean_ms=" << timing.mean_milliseconds << '\n';
      }
      for (const auto &timing : timings)
        if (timing.opcode == dif::ir::Opcode::QuantizeInt8Rows ||
            timing.opcode == dif::ir::Opcode::LinearInt8Scaled)
          std::cerr << "FLUX2_W8A8_OPERATION_PROFILE operation="
                    << timing.operation_id
                    << " opcode=" << dif::ir::opcode_name(timing.opcode)
                    << " mean_ms=" << timing.mean_milliseconds << '\n';
      for (const auto &timing : timings) {
        if (timing.opcode != dif::ir::Opcode::Slice)
          continue;
        const auto found = std::find_if(
            transformer.program.operations.begin(),
            transformer.program.operations.end(), [&](const auto &operation) {
              return operation.id == timing.operation_id;
            });
        const auto *input = transformer.program.tensor(found->inputs.front());
        const auto *output = transformer.program.tensor(found->outputs.front());
        std::cerr << "FLUX2_SLICE_PROFILE operation=" << timing.operation_id
                  << " mean_ms=" << timing.mean_milliseconds
                  << " bytes=" << output->byte_count()
                  << " axis=" << found->u64(dif::ir::AttrKey::Axis, 0U)
                  << " start=" << found->u64(dif::ir::AttrKey::Start, 0U)
                  << " input_dims=";
        for (const auto dimension : input->dims)
          std::cerr << dimension << 'x';
        std::cerr << " output_dims=";
        for (const auto dimension : output->dims)
          std::cerr << dimension << 'x';
        std::cerr << '\n';
      }
    }
    if (step == 0U) {
      result.resident_bytes = prediction.resident_bytes;
      result.backend = prediction.backend_name;
      result.device = prediction.device_name;
      result.fingerprint =
          dif::hex_digest(dif::ir::fingerprint(transformer.program));
      result.linear_tuning_results = prediction.linear_tuning_results;
      result.selected_linear_algorithms =
          prediction.selected_linear_algorithms;
    }
    const auto velocity_batch =
        prediction.outputs.at(transformer.prediction_output);
    const auto unconditional =
        dev ? velocity_batch : batch_row(velocity_batch, 0U);
    const auto conditional =
        dev ? velocity_batch : batch_row(velocity_batch, 1U);

    cfg_bindings.insert_or_assign(cfg.sample_input, latent);
    cfg_bindings.insert_or_assign(
        cfg.conditional_velocity_input, conditional);
    cfg_bindings.insert_or_assign(
        cfg.unconditional_velocity_input, unconditional);
    cfg_bindings.insert_or_assign(cfg.current_timestep_input,
                                  scalar(dif::ir::DType::F32, current));
    cfg_bindings.insert_or_assign(cfg.next_timestep_input,
                                  scalar(dif::ir::DType::F32, next));
    started = Clock::now();
    auto update = cfg_prepared->run(cfg_bindings, cfg_options);
    result.scheduler_ms.push_back(elapsed_ms(started));
    latent = update.outputs.at(cfg.sample_output);
    if (absolute_step == 0U)
      result.first_step_latent = latent;
    if (absolute_step + 1U == (arguments.steps + 1U) / 2U)
      result.middle_step_latent = latent;
    if (arguments.capture_every != 0U &&
        (absolute_step + 1U) % arguments.capture_every == 0U)
      result.periodic_step_latents.emplace_back(absolute_step + 1U, latent);
    std::cout << "FLUX2_NATIVE_STEP step=" << absolute_step + 1U << '/'
              << arguments.steps
              << " cfg_batch_ms=" << result.cfg_transformer_ms.back()
              << " scheduler_ms=" << result.scheduler_ms.back() << "\n"
              << std::flush;
  }
  result.latent = std::move(latent);
  return result;
}

struct VaeResult {
  dif::runtime::Tensor pixels;
  double preparation_ms{};
  double execution_ms{};
  std::uint64_t resident_bytes{};
  std::string backend;
  std::string fingerprint;
};

VaeResult decode(const Arguments &arguments,
                 const dif::runtime::Tensor &latent,
                 std::uint64_t latent_height,
                 std::uint64_t latent_width) {
  dif::frontend::Flux2VaeConfig config;
  config.latent_height = latent_height;
  config.latent_width = latent_width;
  config.capture_boundaries = false;
  const auto vae = dif::frontend::make_flux2_vae_decoder(config);
  const auto checkpoint =
      dif::weights::read_safetensors(arguments.vae_checkpoint);
  dif::runtime::TensorMap bindings;
  bindings.emplace(vae.latent_tokens_input, latent);
  for (const auto &binding : vae.weights) {
    auto tensor = dif::weights::map_safetensor(checkpoint, binding.name);
    if (binding.transform ==
        dif::frontend::Flux2VaeWeightTransform::
            BatchNormStandardDeviation) {
      dif::runtime::Tensor transformed{tensor.dtype, tensor.dims, {}};
      transformed.bytes.resize(static_cast<std::size_t>(tensor.byte_size()));
      for (std::uint64_t element = 0U; element < tensor.element_count();
           ++element)
        dif::runtime::store_float(
            transformed, element,
            std::sqrt(dif::runtime::load_float(tensor, element) + 1.0e-4F));
      transformed.validate();
      tensor = std::move(transformed);
    }
    const auto *description = vae.program.tensor(binding.tensor_id);
    if (!description || tensor.dtype != description->dtype ||
        tensor.dims != description->dims)
      dif::fail("VAE checkpoint disagreement at " + binding.name);
    bindings.emplace(binding.tensor_id, std::move(tensor));
  }
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 256ULL * 1024ULL * 1024ULL;
  options.cache_directory = arguments.cache_directory;
  options.profile_pipeline = arguments.profile_pipeline;
  auto backend = dif::runtime::make_cuda_executor();
  auto started = Clock::now();
  auto prepared = backend->prepare(vae.program, bindings, options);
  VaeResult result;
  result.preparation_ms = elapsed_ms(started);
  started = Clock::now();
  auto decoded = prepared->run(bindings, options);
  result.execution_ms = elapsed_ms(started);
  result.pixels = decoded.outputs.at(vae.clamped_output);
  result.resident_bytes = decoded.resident_bytes;
  result.backend = decoded.backend_name;
  result.fingerprint = dif::hex_digest(dif::ir::fingerprint(vae.program));
  return result;
}

std::vector<std::uint8_t> rgb8(const dif::runtime::Tensor &pixels,
                               std::uint32_t width,
                               std::uint32_t height) {
  if (pixels.dtype != dif::ir::DType::F32 ||
      pixels.dims != std::vector<std::uint64_t>{1U, 3U, height, width})
    dif::fail("FLUX.2 VAE output is not F32 [1,3,H,W]");
  std::vector<std::uint8_t> output(
      static_cast<std::size_t>(width) * height * 3U);
  for (std::uint64_t channel = 0U; channel < 3U; ++channel)
    for (std::uint64_t y = 0U; y < height; ++y)
      for (std::uint64_t x = 0U; x < width; ++x) {
        const auto source = (channel * height + y) * width + x;
        volatile float shifted =
            dif::runtime::load_float(pixels, source) + 1.0F;
        volatile float encoded = 127.5F * shifted;
        const auto target = (y * width + x) * 3U + channel;
        const auto bounded =
            std::clamp(static_cast<float>(encoded), 0.0F, 255.0F);
        output[static_cast<std::size_t>(target)] =
            static_cast<std::uint8_t>(bounded);
      }
  return output;
}

double sum(const std::vector<double> &values) {
  double result = 0.0;
  for (const auto value : values)
    result += value;
  return result;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto process_started = Clock::now();
    const auto arguments = parse(argc, argv);
    const auto latent_height = arguments.height / 16U;
    const auto latent_width = arguments.width / 16U;
    const auto image_tokens =
        static_cast<std::uint64_t>(latent_height) * latent_width;

    const auto conditioning_started = Clock::now();
    const auto conditioning = condition(arguments);
    const auto conditioning_wall_ms = elapsed_ms(conditioning_started);
    auto latent = initial_latent(arguments, latent_height, latent_width);
    const auto initial = latent;
    const auto initial_hash = payload_hash(initial);

    const auto denoise_started = Clock::now();
    const auto denoised = denoise(arguments, conditioning, std::move(latent),
                                  latent_width);
    const auto denoise_wall_ms = elapsed_ms(denoise_started);
    const auto final_latent_hash = payload_hash(denoised.latent);

    const auto vae_started = Clock::now();
    const auto vae = decode(arguments, denoised.latent, latent_height,
                            latent_width);
    const auto vae_wall_ms = elapsed_ms(vae_started);
    const auto pixels_hash = payload_hash(vae.pixels);

    const auto png_started = Clock::now();
    fs::create_directories(arguments.output_png.parent_path());
    dif::write_png_rgb8(arguments.output_png, arguments.width,
                        arguments.height,
                        rgb8(vae.pixels, arguments.width, arguments.height));
    const auto png_ms = elapsed_ms(png_started);
    const auto total_ms = elapsed_ms(process_started);
    const auto png_hash =
        dif::hex_digest(dif::sha256_file(arguments.output_png));
    const auto executed =
        arguments.stop_after == 0U
            ? arguments.steps - arguments.start_step
            : arguments.stop_after;

    if (!arguments.state_output.empty()) {
      const auto schedule = dif::sampling::make_flux2_klein_schedule(
          arguments.steps, image_tokens);
      const auto schedule_tensor = float_tensor(
          dif::ir::DType::F32,
          {static_cast<std::uint64_t>(schedule.size())}, schedule);
      fs::create_directories(arguments.state_output.parent_path());
      std::vector<dif::weights::SafeTensorWriteSpec> state_specs{
          {"initial_image_tokens", initial.dtype, initial.dims},
           {"positive_conditioning", conditioning.positive.dtype,
            conditioning.positive.dims},
           {"negative_conditioning", conditioning.negative.dtype,
            conditioning.negative.dims},
           {"timesteps", schedule_tensor.dtype, schedule_tensor.dims},
           {"final_image_tokens", denoised.latent.dtype,
            denoised.latent.dims},
           {"clamped_output", vae.pixels.dtype, vae.pixels.dims}};
      if (denoised.first_step_latent)
        state_specs.push_back({"first_step_image_tokens",
                               denoised.first_step_latent->dtype,
                               denoised.first_step_latent->dims});
      if (denoised.middle_step_latent)
        state_specs.push_back({"middle_step_image_tokens",
                               denoised.middle_step_latent->dtype,
                               denoised.middle_step_latent->dims});
      for (const auto &[step, tensor] : denoised.periodic_step_latents)
        state_specs.push_back({"step_" + std::to_string(step) +
                                   "_image_tokens",
                               tensor.dtype, tensor.dims});
      dif::weights::SafeTensorWriter writer(arguments.state_output,
                                             state_specs);
      const auto append = [&](std::string_view name,
                              const dif::runtime::Tensor &tensor) {
        writer.append(std::string(name), std::span<const std::uint8_t>(
                                             tensor.data(), tensor.byte_size()));
      };
      append("initial_image_tokens", initial);
      append("positive_conditioning", conditioning.positive);
      append("negative_conditioning", conditioning.negative);
      append("timesteps", schedule_tensor);
      append("final_image_tokens", denoised.latent);
      append("clamped_output", vae.pixels);
      if (denoised.first_step_latent)
        append("first_step_image_tokens", *denoised.first_step_latent);
      if (denoised.middle_step_latent)
        append("middle_step_image_tokens", *denoised.middle_step_latent);
      for (const auto &[step, tensor] : denoised.periodic_step_latents)
        append("step_" + std::to_string(step) + "_image_tokens", tensor);
      (void)writer.finish();
    }

    fs::create_directories(arguments.report.parent_path());
    std::ofstream report(arguments.report, std::ios::trunc);
    report << std::setprecision(17)
           << "{\n  \"creator_commit\": " << std::quoted(kCreatorCommit)
           << ",\n  \"model_revision\": " << std::quoted(kModelRevision)
           << ",\n  \"model_family\": \"FLUX.2-klein-base-9B\",\n"
           << "  \"prompt\": " << std::quoted(arguments.prompt)
           << ",\n  \"seed\": " << arguments.seed
           << ",\n  \"noise_generator\": "
           << std::quoted(arguments.initial_latent.empty()
                              ? "native-splitmix64-box-muller-bf16-v1"
                              : "external-parity-fixture")
           << ",\n  \"width\": " << arguments.width
           << ",\n  \"height\": " << arguments.height
           << ",\n  \"image_tokens\": " << image_tokens
           << ",\n  \"steps\": " << arguments.steps
           << ",\n  \"start_step\": " << arguments.start_step
           << ",\n  \"executed_steps\": " << executed
           << ",\n  \"flux2_model\": " << std::quoted(arguments.flux2_model)
           << ",\n  \"guidance\": " << arguments.guidance
           << ",\n  \"scheduler\": \"creator generalized-time Euler\",\n"
           << "  \"denoiser_mode\": \"creator batch-two CFG\",\n"
           << "  \"transformer_attention\": "
           << std::quoted(arguments.transformer_attention_implementation == 4U
                              ? "native-flash-attention"
                              : "cudnn-sdpa")
           << ",\n"
           << "  \"cudnn_attention_heuristic\": "
           << arguments.cudnn_attention_heuristic << ",\n"
           << "  \"streamed_weight_plan\": {\n"
           << "    \"keep_mapped_pages_between_runs\": "
           << (arguments.streamed_keep_mapped_pages ? "true" : "false")
           << ",\n    \"conditioner_prefetch_depth\": 1"
           << ",\n    \"conditioner_staging_buffers\": 2"
           << ",\n    \"transformer_prefetch_depth\": "
           << arguments.streamed_prefetch_depth
           << ",\n    \"transformer_staging_buffers\": "
           << arguments.streamed_staging_buffers
           << ",\n    \"stage_threads\": "
           << arguments.streamed_stage_threads << "\n  },\n"
           << "  \"residency_plan\": {\n"
           << "    \"budget_mib\": " << arguments.resident_plan_mib
           << ",\n    \"order\": "
           << std::quoted(
                  arguments.resident_order ==
                          dif::compiler::StreamedResidencyOrder::LargestFirst
                      ? "largest-first"
                      : "first-consumer")
           << ",\n    \"upload\": "
           << std::quoted(arguments.lazy_resident_upload ? "lazy-first-use"
                                                        : "pipelined-prepare")
           << ",\n    \"resident_tensors\": "
           << denoised.resident_weight_tensors
           << ",\n    \"resident_weight_bytes\": "
           << denoised.resident_weight_bytes
           << ",\n    \"streamed_weight_bytes_per_step\": "
           << denoised.streamed_weight_bytes
           << ",\n    \"required_bytes\": "
           << denoised.residency_required_bytes << "\n  },\n"
           << "  \"squareq_w4\": {\n"
           << "    \"format\": " << std::quoted(denoised.squareq_w4.format)
           << ",\n    \"rank\": " << denoised.squareq_w4.rank
           << ",\n    \"linear_count\": " << denoised.squareq_w4.linear_count
           << ",\n    \"slab_bytes\": " << denoised.squareq_w4.quantized_bytes
           << ",\n    \"bf16_bytes_replaced\": " << denoised.squareq_w4.bf16_bytes_replaced
           << ",\n    \"plan_cos_w_min\": " << denoised.squareq_w4.plan_cos_w_min
           << "\n  },\n"
           << "  \"int8_weight_only_candidate\": {\n"
           << "    \"linear_count\": "
           << denoised.int8_weight_only_linear_count
           << ",\n    \"mode\": "
           << std::quoted(denoised.int8_weight_only_row_scaled
                              ? "fused-row-scaled"
                              : "groupwise-bf16-dequant")
           << ",\n    \"group_size\": "
           << (denoised.int8_weight_only_row_scaled
                   ? 0U
                   : denoised.int8_weight_only_group_size)
           << ",\n    \"compute_dtype\": \"bf16\""
           << ",\n    \"quantized_weight_bytes\": "
           << denoised.int8_weight_only_bytes
           << ",\n    \"bf16_excluded_weights\": [";
    for (std::size_t index = 0U;
         index < arguments.int8_weight_only_exclude_names.size(); ++index) {
      if (index != 0U)
        report << ',';
      report << std::quoted(arguments.int8_weight_only_exclude_names[index]);
    }
    report << "]\n  },\n"
           << "  \"int8_weight_only_group32_weights\": [";
    for (std::size_t index = 0U;
         index < denoised.int8_weight_only_group32_names.size(); ++index) {
      if (index != 0U)
        report << ',';
      report << std::quoted(
          denoised.int8_weight_only_group32_names[index]);
    }
    report << "],\n"
           << "  \"w8a8_candidate\": {\n"
           << "    \"linear_count\": " << denoised.w8a8_linear_count
           << ",\n"
           << "    \"single_linear1_blocks\": "
           << denoised.w8a8_blocks.size()
           << ",\n    \"row_quantization\": "
           << std::quoted(quantization_name(denoised.w8a8_quantization))
           << ",\n    \"single_mlp_row_quantization\": "
           << std::quoted(
                  quantization_name(denoised.w8a8_single_mlp_quantization))
           << ",\n    \"activation_clip_ratio\": "
           << arguments.w8a8_activation_clip_ratio
           << ",\n    \"activation_clip_single_linear1\": "
           << (std::isfinite(
                   arguments.w8a8_activation_clip_ratio_single_linear1)
                   ? arguments.w8a8_activation_clip_ratio_single_linear1
                   : arguments.w8a8_activation_clip_ratio)
           << ",\n    \"activation_clip_single_linear2\": "
           << (std::isfinite(
                   arguments.w8a8_activation_clip_ratio_single_linear2)
                   ? arguments.w8a8_activation_clip_ratio_single_linear2
                   : arguments.w8a8_activation_clip_ratio)
           << ",\n    \"activation_clip_double\": "
           << (std::isfinite(arguments.w8a8_activation_clip_ratio_double)
                   ? arguments.w8a8_activation_clip_ratio_double
                   : arguments.w8a8_activation_clip_ratio)
           << ",\n    \"activation_clip_switch_step\": "
           << arguments.w8a8_activation_clip_switch_step
           << ",\n    \"activation_clip_after_ratio\": "
           << (std::isfinite(arguments.w8a8_activation_clip_after_ratio)
                   ? arguments.w8a8_activation_clip_after_ratio
                   : arguments.w8a8_activation_clip_ratio)
           << ",\n    \"weight_equalization\": "
           << (arguments.w8a8_weight_equalization ? "true" : "false")
           << ",\n    \"mse_weight_scale\": "
           << (arguments.w8a8_mse_weight_scale ? "true" : "false")
           << ",\n    \"activation_residual2\": "
           << (arguments.w8a8_activation_residual2 ? "true" : "false")
           << ",\n    \"activation_residual2_single_linear1\": "
           << (arguments.w8a8_activation_residual2_single_linear1 ? "true"
                                                                  : "false")
           << ",\n    \"activation_residual2_single_linear2\": "
           << (arguments.w8a8_activation_residual2_single_linear2 ? "true"
                                                                  : "false")
           << ",\n    \"activation_residual2_double\": "
           << (arguments.w8a8_activation_residual2_double ? "true" : "false")
           << ",\n    \"blocks\": [";
    for (std::size_t index = 0U; index < denoised.w8a8_blocks.size(); ++index) {
      if (index != 0U)
        report << ',';
      report << denoised.w8a8_blocks[index];
    }
    report << "]"
           << ",\n    \"single_mlp_only_blocks\": [";
    for (std::size_t index = 0U;
         index < denoised.w8a8_single_mlp_blocks.size(); ++index) {
      if (index != 0U)
        report << ',';
      report << denoised.w8a8_single_mlp_blocks[index];
    }
    report << "]"
           << ",\n    \"single_qk_protected_blocks\": [";
    for (std::size_t index = 0U;
         index < denoised.w8a8_single_qk_blocks.size(); ++index) {
      if (index != 0U)
        report << ',';
      report << denoised.w8a8_single_qk_blocks[index];
    }
    report << "]"
           << ",\n    \"single_linear2_blocks\": [";
    for (std::size_t index = 0U;
         index < denoised.w8a8_single_linear2_blocks.size(); ++index) {
      if (index != 0U)
        report << ',';
      report << denoised.w8a8_single_linear2_blocks[index];
    }
    report << "]"
           << ",\n    \"double_image_mlp_blocks\": [";
    for (std::size_t index = 0U;
         index < denoised.w8a8_double_image_mlp_blocks.size(); ++index) {
      if (index != 0U)
        report << ',';
      report << denoised.w8a8_double_image_mlp_blocks[index];
    }
    report << "]"
           << ",\n    \"double_mlp_blocks\": [";
    for (std::size_t index = 0U;
         index < denoised.w8a8_double_mlp_blocks.size(); ++index) {
      if (index != 0U)
        report << ',';
      report << denoised.w8a8_double_mlp_blocks[index];
    }
    report << "]"
           << ",\n    \"double_image_blocks\": [";
    for (std::size_t index = 0U;
         index < denoised.w8a8_double_image_blocks.size(); ++index) {
      if (index != 0U)
        report << ',';
      report << denoised.w8a8_double_image_blocks[index];
    }
    report << "]"
           << ",\n    \"double_text_blocks\": [";
    for (std::size_t index = 0U;
         index < denoised.w8a8_double_text_blocks.size(); ++index) {
      if (index != 0U)
        report << ',';
      report << denoised.w8a8_double_text_blocks[index];
    }
    report << "]"
           << ",\n    \"double_complete_blocks\": [";
    for (std::size_t index = 0U;
         index < denoised.w8a8_double_blocks.size(); ++index) {
      if (index != 0U)
        report << ',';
      report << denoised.w8a8_double_blocks[index];
    }
    report << "]"
           << ",\n    \"quantized_weight_bytes\": "
           << denoised.w8a8_weight_bytes << "\n  },\n"
           << "  \"fp8_candidate\": {\n"
           << "    \"linear_count\": " << denoised.fp8_linear_count
           << ",\n    \"format\": "
           << std::quoted(denoised.fp8_row_scaled
                              ? "fp8-e4m3-f32-row-scaled"
                              : "mxfp8-e4m3-ue8m0-block32")
           << ",\n    \"single_linear1_blocks\": [";
    for (std::size_t index = 0U;
         index < denoised.fp8_single_linear1_blocks.size(); ++index) {
      if (index != 0U)
        report << ',';
      report << denoised.fp8_single_linear1_blocks[index];
    }
    report << "]"
           << ",\n    \"single_mlp_only_blocks\": [";
    for (std::size_t index = 0U;
         index < denoised.fp8_single_mlp_blocks.size(); ++index) {
      if (index != 0U)
        report << ',';
      report << denoised.fp8_single_mlp_blocks[index];
    }
    report << "]"
           << ",\n    \"single_linear2_blocks\": [";
    for (std::size_t index = 0U;
         index < denoised.fp8_single_linear2_blocks.size(); ++index) {
      if (index != 0U)
        report << ',';
      report << denoised.fp8_single_linear2_blocks[index];
    }
    report << "]"
           << ",\n    \"double_image_mlp_blocks\": [";
    for (std::size_t index = 0U;
         index < denoised.fp8_double_image_mlp_blocks.size(); ++index) {
      if (index != 0U)
        report << ',';
      report << denoised.fp8_double_image_mlp_blocks[index];
    }
    report << "]"
           << ",\n    \"double_image_full_blocks\": [";
    for (std::size_t index = 0U;
         index < denoised.fp8_double_image_blocks.size(); ++index) {
      if (index != 0U)
        report << ',';
      report << denoised.fp8_double_image_blocks[index];
    }
    report << "]"
           << ",\n    \"double_text_full_blocks\": [";
    for (std::size_t index = 0U;
         index < denoised.fp8_double_text_blocks.size(); ++index) {
      if (index != 0U)
        report << ',';
      report << denoised.fp8_double_text_blocks[index];
    }
    report << "]"
           << ",\n    \"double_complete_blocks\": [";
    for (std::size_t index = 0U;
         index < denoised.fp8_double_blocks.size(); ++index) {
      if (index != 0U)
        report << ',';
      report << denoised.fp8_double_blocks[index];
    }
    report << "]"
           << ",\n    \"quantized_weight_bytes\": "
           << denoised.fp8_weight_bytes << "\n  },\n"
           << "  \"linear_plan\": {\n"
           << "    \"expand_algorithms\": "
           << (arguments.expand_linear_algorithms ? "true" : "false")
           << ",\n    \"persist_heuristics\": "
           << (arguments.persist_linear_heuristics ? "true" : "false")
           << ",\n    \"tuned_operations\": [";
    for (std::size_t index = 0U;
         index < arguments.tune_linear_operations.size(); ++index) {
      if (index != 0U)
        report << ',';
      report << arguments.tune_linear_operations[index];
    }
    report << "],\n    \"selected_algorithms\": [";
    for (std::size_t index = 0U;
         index < denoised.selected_linear_algorithms.size(); ++index) {
      if (index != 0U)
        report << ',';
      const auto &choice = denoised.selected_linear_algorithms[index];
      report << "{\"operation_id\":" << choice.operation_id
             << ",\"heuristic_rank\":" << choice.heuristic_rank << '}';
    }
    report << "],\n    \"tuning_results\": [";
    for (std::size_t index = 0U;
         index < denoised.linear_tuning_results.size(); ++index) {
      if (index != 0U)
        report << ',';
      const auto &tuning = denoised.linear_tuning_results[index];
      report << "{\"operation_id\":" << tuning.operation_id
             << ",\"selected_heuristic_rank\":"
             << tuning.selected_heuristic_index
             << ",\"selected_algorithm_id\":"
             << tuning.selected_algorithm_id
             << ",\"default_mean_ms\":"
             << tuning.default_mean_milliseconds
             << ",\"selected_mean_ms\":"
             << tuning.selected_mean_milliseconds
             << ",\"tuning_ms\":" << tuning.tuning_milliseconds
             << ",\"changed_from_default\":"
             << (tuning.changed_from_default ? "true" : "false")
             << ",\"decision\":" << std::quoted(tuning.decision) << '}';
    }
    report << "]\n  },\n"
           << "  \"transformer_checkpoint\": "
           << std::quoted(arguments.transformer_checkpoint.string())
           << ",\n  \"vae_checkpoint\": "
           << std::quoted(arguments.vae_checkpoint.string())
           << ",\n  \"conditioning_mode\": "
           << std::quoted(conditioning.precomputed
                              ? "precomputed-native-parity"
                              : "native-qwen3-8b-bf16")
           << ",\n  \"prompt_tokens\": " << conditioning.prompt_tokens
           << ",\n  \"empty_tokens\": " << conditioning.empty_tokens
           << ",\n  \"device\": " << std::quoted(denoised.device)
           << ",\n  \"conditioner_backend\": "
           << std::quoted(conditioning.backend)
           << ",\n  \"transformer_backend\": "
           << std::quoted(denoised.backend)
           << ",\n  \"vae_backend\": " << std::quoted(vae.backend)
           << ",\n  \"conditioner_diffir\": "
           << std::quoted(conditioning.fingerprint)
           << ",\n  \"positive_conditioning_sha256\": "
           << std::quoted(payload_hash(conditioning.positive))
           << ",\n  \"negative_conditioning_sha256\": "
           << std::quoted(payload_hash(conditioning.negative))
           << ",\n  \"transformer_diffir\": "
           << std::quoted(denoised.fingerprint)
           << ",\n  \"vae_diffir\": " << std::quoted(vae.fingerprint)
           << ",\n  \"initial_latent_sha256\": "
           << std::quoted(initial_hash)
           << ",\n  \"final_latent_sha256\": "
           << std::quoted(final_latent_hash)
           << ",\n  \"pixels_sha256\": " << std::quoted(pixels_hash)
           << ",\n  \"png_sha256\": " << std::quoted(png_hash)
           << ",\n  \"state_output\": "
           << std::quoted(arguments.state_output.string())
           << ",\n  \"peak_prepared_resident_bytes\": "
           << std::max({conditioning.resident_bytes, denoised.resident_bytes,
                        vae.resident_bytes})
           << ",\n  \"timing_ms\": {"
           << "\n    \"prompt_to_png\": " << total_ms
           << ",\n    \"conditioning_wall\": " << conditioning_wall_ms
           << ",\n    \"tokenizer\": " << conditioning.tokenizer_ms
           << ",\n    \"conditioner_prepare\": "
           << conditioning.preparation_ms
           << ",\n    \"conditioner_prompt\": " << conditioning.positive_ms
           << ",\n    \"conditioner_empty\": " << conditioning.negative_ms
           << ",\n    \"denoiser_prepare\": " << denoised.preparation_ms
           << ",\n    \"residency_plan\": " << denoised.residency_plan_ms
           << ",\n    \"denoise_wall\": " << denoise_wall_ms
           << ",\n    \"cfg_transformer_total\": "
           << sum(denoised.cfg_transformer_ms)
           << ",\n    \"scheduler_total\": " << sum(denoised.scheduler_ms)
           << ",\n    \"vae_prepare\": " << vae.preparation_ms
           << ",\n    \"vae_execute\": " << vae.execution_ms
           << ",\n    \"vae_wall\": " << vae_wall_ms
           << ",\n    \"png\": " << png_ms
           << "\n  },\n  \"output_png\": "
           << std::quoted(arguments.output_png.string()) << "\n}\n";
    if (!report)
      dif::fail("failed to write FLUX.2 prompt-to-PNG report");

    std::cout << "FLUX2_PROMPT_TO_PNG_PASS output=" << arguments.output_png
              << " report=" << arguments.report << " total_ms=" << total_ms
              << " png_sha256=" << png_hash << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difflux2sample: " << error.what() << "\n";
    return 1;
  }
}
