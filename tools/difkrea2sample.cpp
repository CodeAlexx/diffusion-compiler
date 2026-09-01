#include "dif/compiler/residency_plan.hpp"
#include "dif/compiler/layout_plan.hpp"
#include "dif/compiler/memory_plan.hpp"
#include "dif/frontend/krea2.hpp"
#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/opt/rewrite.hpp"
#include "dif/opt/transform.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"
#include "dif/weights/bundle.hpp"
#include "dif/weights/safetensors.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

struct Arguments {
  std::filesystem::path checkpoint;
  std::filesystem::path bundle;
  std::filesystem::path positive_conditioning;
  std::filesystem::path positive_tokenizer;
  std::filesystem::path negative_conditioning;
  std::filesystem::path negative_tokenizer;
  std::filesystem::path initial_fixture;
  std::filesystem::path reference;
  std::filesystem::path output;
  std::filesystem::path report;
  std::filesystem::path diffir;
  std::filesystem::path cache_directory;
  std::vector<std::uint32_t> tune_linear_operations;
  std::vector<dif::runtime::LinearAlgorithmChoice> linear_algorithm_choices;
  std::vector<dif::runtime::CutlassLinearChoice> cutlass_linear_choices;
  std::vector<std::uint32_t> fuse_parallel_swiglu_operations;
  std::vector<std::vector<std::uint32_t>> fuse_parallel_linear_groups;
  std::vector<std::vector<std::uint32_t>> parallel_linear_groups;
  std::uint64_t resident_plan_mib{};
  dif::compiler::StreamedResidencyOrder resident_order{
      dif::compiler::StreamedResidencyOrder::FirstConsumer};
  bool alias_reshapes{};
  bool expand_linear_algorithms{};
  bool persist_linear_heuristics{};
  bool pipelined_resident_upload{};
  bool lazy_resident_upload{};
  bool profile_pipeline{};
  bool whole_plan{};
  bool lower_fused_swiglu{};
  std::uint32_t streamed_stage_threads{1U};
  std::uint32_t streamed_prefetch_depth{1U};
  std::uint32_t streamed_staging_buffers{2U};
  std::uint32_t linear_tuning_warmups{3U};
  std::uint32_t linear_tuning_iterations{10U};
  std::uint32_t linear_tuning_sessions{3U};
  std::uint32_t steps{52U};
  std::uint32_t stop_after{};
  std::uint32_t text_token_cap{512U};
  float guidance{3.5F};
  std::optional<double> fixed_mu;
  std::uint64_t seed{20260831U};
};

std::vector<std::uint32_t> parse_operation_group(const std::string &text) {
  std::vector<std::uint32_t> result;
  auto begin = std::size_t{0U};
  while (begin < text.size()) {
    const auto comma = text.find(',', begin);
    const auto item = text.substr(
        begin, comma == std::string::npos ? std::string::npos : comma - begin);
    if (item.empty())
      dif::fail("parallel Linear operation group contains an empty id");
    std::size_t consumed = 0U;
    const auto id = std::stoul(item, &consumed);
    if (consumed != item.size())
      dif::fail("parallel Linear operation group contains a nonnumeric id");
    result.push_back(static_cast<std::uint32_t>(id));
    if (comma == std::string::npos)
      break;
    begin = comma + 1U;
  }
  if (result.size() < 2U)
    dif::fail("parallel Linear fusion requires at least two operation ids");
  return result;
}

Arguments parse(int argc, char **argv) {
  Arguments result;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    const auto value = [&]() -> std::string {
      if (++index >= argc)
        dif::fail(option + " requires a value");
      return argv[index];
    };
    if (option == "--checkpoint") result.checkpoint = value();
    else if (option == "--bundle") result.bundle = value();
    else if (option == "--positive-conditioning")
      result.positive_conditioning = value();
    else if (option == "--positive-tokenizer")
      result.positive_tokenizer = value();
    else if (option == "--negative-conditioning")
      result.negative_conditioning = value();
    else if (option == "--negative-tokenizer")
      result.negative_tokenizer = value();
    else if (option == "--initial-fixture") result.initial_fixture = value();
    else if (option == "--reference") result.reference = value();
    else if (option == "--output") result.output = value();
    else if (option == "--report") result.report = value();
    else if (option == "--diffir") result.diffir = value();
    else if (option == "--cache-dir") result.cache_directory = value();
    else if (option == "--tune-linear")
      result.tune_linear_operations.push_back(
          static_cast<std::uint32_t>(std::stoul(value())));
    else if (option == "--select-linear-algorithm") {
      const auto selection = value();
      const auto split = selection.find(':');
      if (split == std::string::npos || split == 0U ||
          split + 1U >= selection.size())
        dif::fail("Linear algorithm choice must be OP_ID:HEURISTIC_RANK");
      result.linear_algorithm_choices.push_back(
          {static_cast<std::uint32_t>(
               std::stoul(selection.substr(0U, split))),
           static_cast<std::uint32_t>(
               std::stoul(selection.substr(split + 1U)))});
    }
    else if (option == "--cutlass-linear") {
      const auto selection = value();
      const auto split = selection.find(':');
      if (split == std::string::npos || split == 0U ||
          split + 1U >= selection.size())
        dif::fail("CUTLASS Linear choice must be OP_ID:SCHEDULE_ID");
      result.cutlass_linear_choices.push_back(
          {static_cast<std::uint32_t>(
               std::stoul(selection.substr(0U, split))),
           static_cast<std::uint32_t>(
               std::stoul(selection.substr(split + 1U)))});
    }
    else if (option == "--expand-linear-algorithms")
      result.expand_linear_algorithms = true;
    else if (option == "--persist-linear-heuristics")
      result.persist_linear_heuristics = true;
    else if (option == "--pipelined-resident-upload")
      result.pipelined_resident_upload = true;
    else if (option == "--lazy-resident-upload")
      result.lazy_resident_upload = true;
    else if (option == "--profile-pipeline")
      result.profile_pipeline = true;
    else if (option == "--whole-plan")
      result.whole_plan = true;
    else if (option == "--fuse-parallel-swiglu")
      result.fuse_parallel_swiglu_operations.push_back(
          static_cast<std::uint32_t>(std::stoul(value())));
    else if (option == "--fuse-parallel-linears")
      result.fuse_parallel_linear_groups.push_back(
          parse_operation_group(value()));
    else if (option == "--parallel-linears")
      result.parallel_linear_groups.push_back(parse_operation_group(value()));
    else if (option == "--lower-fused-swiglu")
      result.lower_fused_swiglu = true;
    else if (option == "--linear-tune-warmups")
      result.linear_tuning_warmups =
          static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--linear-tune-iterations")
      result.linear_tuning_iterations =
          static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--linear-tune-sessions")
      result.linear_tuning_sessions =
          static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--streamed-stage-threads")
      result.streamed_stage_threads =
          static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--streamed-prefetch-depth")
      result.streamed_prefetch_depth =
          static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--streamed-staging-buffers")
      result.streamed_staging_buffers =
          static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--resident-plan-mib")
      result.resident_plan_mib = std::stoull(value());
    else if (option == "--resident-order") {
      const auto order = value();
      if (order == "first")
        result.resident_order =
            dif::compiler::StreamedResidencyOrder::FirstConsumer;
      else if (order == "largest")
        result.resident_order =
            dif::compiler::StreamedResidencyOrder::LargestFirst;
      else
        dif::fail("resident order accepts first or largest");
    }
    else if (option == "--alias-reshapes")
      result.alias_reshapes = true;
    else if (option == "--steps")
      result.steps = static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--stop-after")
      result.stop_after = static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--text-token-cap")
      result.text_token_cap =
          static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--guidance") result.guidance = std::stof(value());
    else if (option == "--mu") result.fixed_mu = std::stod(value());
    else if (option == "--seed") result.seed = std::stoull(value());
    else dif::fail("invalid difkrea2sample argument: " + option);
  }
  const auto cfg = result.guidance > 0.0F;
  if ((result.checkpoint.empty() == result.bundle.empty()) ||
      result.positive_conditioning.empty() ||
      result.positive_tokenizer.empty() ||
      (cfg && (result.negative_conditioning.empty() ||
               result.negative_tokenizer.empty())) || result.initial_fixture.empty() ||
      result.reference.empty() || result.output.empty() ||
      result.report.empty() || result.diffir.empty() || result.steps == 0U ||
      result.guidance < 0.0F || !std::isfinite(result.guidance) ||
      (result.fixed_mu.has_value() && !std::isfinite(*result.fixed_mu)) ||
      result.linear_tuning_iterations == 0U ||
      result.linear_tuning_sessions < 2U ||
      result.streamed_stage_threads == 0U ||
      result.streamed_prefetch_depth == 0U ||
      result.streamed_staging_buffers <= result.streamed_prefetch_depth ||
      result.text_token_cap == 0U || result.text_token_cap > 512U ||
      result.text_token_cap % 256U != 0U ||
      (result.pipelined_resident_upload && result.lazy_resident_upload) ||
      (result.whole_plan &&
       (cfg || !result.fuse_parallel_swiglu_operations.empty() ||
        !result.fuse_parallel_linear_groups.empty() ||
        !result.parallel_linear_groups.empty() ||
        !result.tune_linear_operations.empty() ||
        !result.linear_algorithm_choices.empty() ||
        !result.cutlass_linear_choices.empty())) ||
      (result.stop_after != 0U && result.stop_after > result.steps))
    dif::fail("difkrea2sample requires exactly one of checkpoint or bundle, "
              "positive conditioning and "
              "tokenizer, initial fixture, reference, output, report, diffir, "
              "positive steps, nonnegative guidance, and negative inputs only "
              "when CFG is enabled");
  return result;
}

dif::runtime::Tensor float_tensor(dif::ir::DType dtype,
                                  std::vector<std::uint64_t> dims,
                                  std::span<const float> values) {
  dif::runtime::Tensor result{dtype, std::move(dims), {}};
  result.bytes.resize(values.size() * dif::ir::dtype_size(dtype));
  result.validate();
  for (std::size_t index = 0U; index < values.size(); ++index)
    dif::runtime::store_float(result, index, values[index]);
  return result;
}

dif::runtime::Tensor scalar(dif::ir::DType dtype, float value) {
  return float_tensor(dtype, {1U}, std::span<const float>(&value, 1U));
}

std::string residency_plan_hash(
    const dif::compiler::StreamedResidencyPlan &plan,
    const dif::compiler::ReshapeAliasPlan *aliases,
    std::uint32_t stage_threads, std::uint32_t prefetch_depth,
    std::uint32_t staging_buffers,
    dif::compiler::StreamedResidencyOrder order,
    bool pipelined_resident_upload, bool lazy_resident_upload) {
  std::string identity =
      "streamed-residency-v2|max=" +
      std::to_string(plan.maximum_total_bytes) + "|fixed=" +
      std::to_string(plan.fixed_runtime_bytes) + "|stage_threads=" +
      std::to_string(stage_threads) + "|prefetch_depth=" +
      std::to_string(prefetch_depth) + "|staging_buffers=" +
      std::to_string(staging_buffers) + "|order=" +
      std::to_string(static_cast<std::uint32_t>(order)) +
      "|pipelined_resident_upload=" +
      std::to_string(pipelined_resident_upload ? 1U : 0U) +
      "|lazy_resident_upload=" +
      std::to_string(lazy_resident_upload ? 1U : 0U) + "|ids=";
  for (const auto id : plan.resident_tensor_ids)
    identity += std::to_string(id) + ",";
  identity += "|reshape_ops=";
  if (aliases) {
    for (const auto id : aliases->operation_ids)
      identity += std::to_string(id) + ",";
  }
  return dif::hex_digest(dif::sha256(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(identity.data()),
      identity.size())));
}

dif::runtime::Tensor i32_tensor(std::vector<std::int32_t> values) {
  dif::runtime::Tensor result{dif::ir::DType::I32,
                              {static_cast<std::uint64_t>(values.size())}, {}};
  result.bytes.resize(values.size() * sizeof(std::int32_t));
  std::memcpy(result.mutable_data(), values.data(), result.byte_size());
  result.validate();
  return result;
}

dif::runtime::Tensor compact_text(const dif::runtime::Tensor &conditioning,
                                  const dif::runtime::Tensor &tokenizer_mask,
                                  std::uint64_t text_tokens) {
  if (conditioning.dtype != dif::ir::DType::BF16 ||
      conditioning.dims != std::vector<std::uint64_t>{1U, 512U, 6144U} ||
      tokenizer_mask.dtype != dif::ir::DType::Bool ||
      tokenizer_mask.dims != std::vector<std::uint64_t>{1U, 546U} ||
      text_tokens == 0U || text_tokens > 512U)
    dif::fail("Krea text compaction requires BF16 [1,512,6144] conditioning "
              "and BOOL [1,546] mask");
  dif::runtime::Tensor result{
      conditioning.dtype, {1U, text_tokens, 6144U}, {}};
  result.bytes.resize(static_cast<std::size_t>(result.element_count() * 2U),
                      0U);
  if (text_tokens == 512U) {
    std::memcpy(result.mutable_data(), conditioning.data(), result.byte_size());
    result.validate();
    return result;
  }
  constexpr std::size_t row_bytes = 6144U * 2U;
  std::uint64_t destination = 0U;
  for (std::uint64_t source = 0U; source < 512U; ++source) {
    if (tokenizer_mask.data()[34U + source] == 0U)
      continue;
    if (destination == text_tokens)
      dif::fail("Krea text-token cap would discard a valid prompt token");
    std::memcpy(result.mutable_data() + destination * row_bytes,
                conditioning.data() + source * row_bytes, row_bytes);
    ++destination;
  }
  result.validate();
  return result;
}

dif::runtime::Tensor combined_mask(const dif::runtime::Tensor &tokenizer_mask,
                                   std::uint64_t text_tokens) {
  if (tokenizer_mask.dtype != dif::ir::DType::Bool ||
      tokenizer_mask.dims != std::vector<std::uint64_t>{1U, 546U})
    dif::fail("Krea tokenizer attention mask must be BOOL [1,546]");
  const auto sequence = text_tokens + 4096U;
  dif::runtime::Tensor result{dif::ir::DType::Bool, {1U, sequence}, {}};
  result.bytes.resize(static_cast<std::size_t>(sequence), 1U);
  if (text_tokens == 512U) {
    std::memcpy(result.mutable_data(), tokenizer_mask.data() + 34U, 512U);
    result.validate();
    return result;
  }
  std::fill(result.bytes.begin(),
            result.bytes.begin() + static_cast<std::ptrdiff_t>(text_tokens),
            0U);
  std::uint64_t valid = 0U;
  for (std::uint64_t source = 0U; source < 512U; ++source)
    valid += tokenizer_mask.data()[34U + source] != 0U ? 1U : 0U;
  if (valid > text_tokens)
    dif::fail("Krea text-token cap would discard a valid prompt token");
  std::fill(result.bytes.begin(),
            result.bytes.begin() + static_cast<std::ptrdiff_t>(valid), 1U);
  result.validate();
  return result;
}

dif::runtime::Tensor positions(std::uint64_t text_tokens) {
  const auto sequence = text_tokens + 4096U;
  std::vector<float> values(sequence * 3U, 0.0F);
  for (std::uint64_t y = 0U; y < 64U; ++y) {
    for (std::uint64_t x = 0U; x < 64U; ++x) {
      const auto token = text_tokens + y * 64U + x;
      values[token * 3U + 1U] = static_cast<float>(y);
      values[token * 3U + 2U] = static_cast<float>(x);
    }
  }
  return float_tensor(dif::ir::DType::F32, {1U, sequence, 3U}, values);
}

void bind_rotary(const dif::frontend::Krea2DenoiserBuild &build,
                 dif::runtime::TensorMap &bindings) {
  std::vector<std::int32_t> pair_axes;
  std::vector<std::int32_t> pair_indices;
  for (std::int32_t axis = 0; axis < 3; ++axis) {
    const std::int32_t dimension = axis == 0 ? 32 : 48;
    for (std::int32_t pair = 0; pair < dimension / 2; ++pair) {
      pair_axes.push_back(axis);
      pair_indices.push_back(pair);
    }
  }
  bindings.emplace(build.rotary_pair_axes, i32_tensor(std::move(pair_axes)));
  bindings.emplace(build.rotary_pair_indices,
                   i32_tensor(std::move(pair_indices)));
  bindings.emplace(build.rotary_axis_dims, i32_tensor({32, 48, 48}));
}

struct Metrics {
  double cosine{}, relative_l2{}, max_absolute{}, norm_ratio{};
  std::uint64_t nonfinite{}, bit_mismatches{}, elements{};
};

Metrics measure(const dif::runtime::Tensor &reference,
                const dif::runtime::Tensor &actual) {
  if (reference.dtype != actual.dtype || reference.dims != actual.dims)
    dif::fail("Krea sampler trajectory comparison shape/dtype mismatch");
  long double dot = 0.0L;
  long double reference_squared = 0.0L;
  long double actual_squared = 0.0L;
  long double error_squared = 0.0L;
  Metrics result;
  result.elements = reference.element_count();
  const auto width = dif::ir::dtype_size(reference.dtype);
  for (std::uint64_t index = 0U; index < result.elements; ++index) {
    result.bit_mismatches +=
        std::memcmp(reference.data() + index * width,
                    actual.data() + index * width, width) != 0;
    const auto expected =
        static_cast<double>(dif::runtime::load_float(reference, index));
    const auto observed =
        static_cast<double>(dif::runtime::load_float(actual, index));
    if (!std::isfinite(expected) || !std::isfinite(observed)) {
      ++result.nonfinite;
      continue;
    }
    const auto error = observed - expected;
    dot += static_cast<long double>(expected) * observed;
    reference_squared += static_cast<long double>(expected) * expected;
    actual_squared += static_cast<long double>(observed) * observed;
    error_squared += static_cast<long double>(error) * error;
    result.max_absolute = std::max(result.max_absolute, std::abs(error));
  }
  const auto denominator = std::sqrt(reference_squared * actual_squared);
  result.cosine = denominator == 0.0L
                      ? 1.0
                      : static_cast<double>(dot / denominator);
  result.relative_l2 =
      reference_squared == 0.0L
          ? 0.0
          : static_cast<double>(std::sqrt(error_squared / reference_squared));
  result.norm_ratio =
      reference_squared == 0.0L
          ? 1.0
          : static_cast<double>(std::sqrt(actual_squared / reference_squared));
  return result;
}

void emit_metrics(std::ostream &out, const Metrics &value) {
  out << "{\"cosine\":" << value.cosine
      << ",\"relative_l2\":" << value.relative_l2
      << ",\"max_absolute\":" << value.max_absolute
      << ",\"norm_ratio\":" << value.norm_ratio
      << ",\"nonfinite\":" << value.nonfinite
      << ",\"bit_mismatches\":" << value.bit_mismatches
      << ",\"elements\":" << value.elements << "}";
}

double milliseconds_since(const std::chrono::steady_clock::time_point &start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto process_start = std::chrono::steady_clock::now();
    const auto arguments = parse(argc, argv);
    const auto executed_steps =
        arguments.stop_after == 0U ? arguments.steps : arguments.stop_after;
    auto phase_start = std::chrono::steady_clock::now();
    dif::frontend::Krea2Config config;
    config.streamed_constants = true;
    config.text_tokens = arguments.text_token_cap;
    dif::frontend::Krea2ScheduleConfig schedule_config;
    schedule_config.steps = arguments.steps;
    schedule_config.fixed_mu = arguments.fixed_mu;
    const auto schedule = dif::frontend::make_krea2_schedule(config, schedule_config);
    auto denoiser = dif::frontend::make_krea2_denoiser(config, false);
    const auto frontend_build_ms = milliseconds_since(phase_start);

    phase_start = std::chrono::steady_clock::now();
    std::optional<dif::weights::SafeTensorFile> checkpoint;
    std::optional<dif::weights::WeightBundle> weight_bundle;
    if (!arguments.checkpoint.empty())
      checkpoint.emplace(
          dif::weights::read_safetensors(arguments.checkpoint));
    else
      weight_bundle.emplace(dif::weights::read_weight_bundle(arguments.bundle));
    const auto positive_file =
        dif::weights::read_safetensors(arguments.positive_conditioning);
    const auto positive_tokenizer_file =
        dif::weights::read_safetensors(arguments.positive_tokenizer);
    const auto cfg_enabled = arguments.guidance > 0.0F;
    std::optional<dif::weights::SafeTensorFile> negative_file;
    std::optional<dif::weights::SafeTensorFile> negative_tokenizer_file;
    if (cfg_enabled) {
      negative_file.emplace(
          dif::weights::read_safetensors(arguments.negative_conditioning));
      negative_tokenizer_file.emplace(
          dif::weights::read_safetensors(arguments.negative_tokenizer));
    }
    const auto initial_file =
        dif::weights::read_safetensors(arguments.initial_fixture);
    const auto reference_file = dif::weights::read_safetensors(arguments.reference);
    const auto input_open_ms = milliseconds_since(phase_start);

    phase_start = std::chrono::steady_clock::now();
    const auto positive_tokenizer_mask = dif::weights::map_safetensor(
        positive_tokenizer_file, "attention_mask");
    auto positive = compact_text(
        dif::weights::map_safetensor(positive_file, "conditioning_output"),
        positive_tokenizer_mask, config.text_tokens);
    std::optional<dif::runtime::Tensor> negative;
    if (cfg_enabled)
      negative.emplace(compact_text(
          dif::weights::map_safetensor(*negative_file, "conditioning_output"),
          dif::weights::map_safetensor(*negative_tokenizer_file,
                                       "attention_mask"),
          config.text_tokens));
    auto positive_mask =
        combined_mask(positive_tokenizer_mask, config.text_tokens);
    std::optional<dif::runtime::Tensor> negative_mask;
    if (cfg_enabled)
      negative_mask.emplace(combined_mask(dif::weights::map_safetensor(
          *negative_tokenizer_file, "attention_mask"), config.text_tokens));
    auto image =
        dif::weights::map_safetensor(initial_file, "initial_image_tokens");
    const auto reference_schedule =
        dif::weights::map_safetensor(reference_file, "timesteps");
    const auto native_schedule = float_tensor(
        dif::ir::DType::F32,
        {static_cast<std::uint64_t>(schedule.timesteps.size())},
        schedule.timesteps);
    if (reference_schedule.dtype != native_schedule.dtype ||
        reference_schedule.dims != native_schedule.dims ||
        reference_schedule.byte_size() != native_schedule.byte_size() ||
        std::memcmp(reference_schedule.data(), native_schedule.data(),
                    native_schedule.byte_size()) != 0)
      dif::fail("native Krea schedule is not bit-exact to the creator fixture");
    const auto input_prepare_ms = milliseconds_since(phase_start);

    phase_start = std::chrono::steady_clock::now();
    dif::runtime::TensorMap denoiser_bindings;
    denoiser_bindings.emplace(denoiser.image_tokens_input, image);
    denoiser_bindings.emplace(denoiser.context_input, positive);
    denoiser_bindings.emplace(denoiser.timestep_input,
                              scalar(dif::ir::DType::BF16, 1.0F));
    denoiser_bindings.emplace(denoiser.positions_input,
                              positions(config.text_tokens));
    denoiser_bindings.emplace(denoiser.validity_mask_input, positive_mask);
    bind_rotary(denoiser, denoiser_bindings);

    std::uint64_t converted_parameters = 0U;
    std::uint64_t converted_bytes = 0U;
    if (weight_bundle) {
      auto prepared_weights = dif::weights::load_weight_bundle(
          *weight_bundle, denoiser.program, false);
      for (const auto tensor_id : denoiser.checkpoint_tensors) {
        auto found = prepared_weights.find(tensor_id);
        if (found == prepared_weights.end())
          dif::fail("Krea prepared bundle is missing tensor " +
                    std::to_string(tensor_id));
        denoiser_bindings.emplace(tensor_id, std::move(found->second));
      }
    } else {
      for (std::size_t index = 0U; index < denoiser.checkpoint_tensors.size();
           ++index) {
        auto tensor = dif::weights::map_safetensor(
            *checkpoint, denoiser.checkpoint_names[index]);
        if (tensor.dtype == dif::ir::DType::F32) {
          tensor = dif::runtime::convert_float_tensor(
              tensor, dif::ir::DType::BF16);
          ++converted_parameters;
          converted_bytes += tensor.byte_size();
        }
        const auto *description =
            denoiser.program.tensor(denoiser.checkpoint_tensors[index]);
        if (!description || description->dtype != tensor.dtype ||
            description->dims != tensor.dims)
          dif::fail("Krea sampler checkpoint mismatch: " +
                    denoiser.checkpoint_names[index]);
        denoiser_bindings.emplace(denoiser.checkpoint_tensors[index],
                                  std::move(tensor));
      }
    }
    std::optional<dif::frontend::Krea2TurboExecutionBuild> whole_execution;
    std::optional<dif::compiler::StreamedResidencyPlan>
        whole_base_residency;
    if (arguments.whole_plan) {
      if (arguments.resident_plan_mib == 0U)
        dif::fail("whole-plan execution requires an explicit resident budget");
      const auto base_aliases =
          dif::compiler::plan_reshape_aliases(denoiser.program);
      constexpr std::uint64_t whole_plan_state_reserve =
          256ULL * 1024ULL * 1024ULL;
      const auto whole_budget =
          arguments.resident_plan_mib * 1024ULL * 1024ULL;
      if (whole_budget <= whole_plan_state_reserve)
        dif::fail("whole-plan resident budget cannot cover state reserve");
      whole_base_residency.emplace(
          dif::compiler::plan_streamed_residency(
          denoiser.program,
          whole_budget - whole_plan_state_reserve,
          64ULL * 1024ULL * 1024ULL,
          arguments.streamed_prefetch_depth,
          base_aliases.output_to_root_input, arguments.resident_order));
      whole_execution.emplace(dif::frontend::make_krea2_turbo_execution(
          config, executed_steps, true,
          whole_base_residency->resident_tensor_ids));
      for (const auto &[alias, source] :
           whole_execution->constant_sources) {
        const auto found = denoiser_bindings.find(source);
        if (found == denoiser_bindings.end())
          dif::fail("whole-plan streamed alias lost its checkpoint source");
        denoiser_bindings.emplace(alias, found->second);
      }
      denoiser.program = whole_execution->program;
      for (std::uint32_t step = 0U; step < executed_steps; ++step) {
        denoiser_bindings.emplace(
            whole_execution->model_timestep_inputs.at(step),
            scalar(dif::ir::DType::BF16, schedule.timesteps.at(step)));
        denoiser_bindings.emplace(
            whole_execution->current_timestep_inputs.at(step),
            scalar(dif::ir::DType::F32, schedule.timesteps.at(step)));
        denoiser_bindings.emplace(
            whole_execution->next_timestep_inputs.at(step),
            scalar(dif::ir::DType::F32, schedule.timesteps.at(step + 1U)));
      }
    }
    for (const auto operation_id :
         arguments.fuse_parallel_swiglu_operations) {
      dif::opt::RewriteContext context{std::move(denoiser.program),
                                       std::move(denoiser_bindings), 1U};
      dif::opt::Transform transform;
      transform.kind = dif::opt::TransformKind::FuseParallelLinearSwiGlu;
      transform.operations = {operation_id};
      dif::opt::apply(transform, context);
      denoiser.program = std::move(context.program);
      denoiser_bindings = std::move(context.bindings);
    }
    for (const auto &operation_ids : arguments.fuse_parallel_linear_groups) {
      dif::opt::RewriteContext context{std::move(denoiser.program),
                                       std::move(denoiser_bindings), 1U};
      dif::opt::Transform transform;
      transform.kind = dif::opt::TransformKind::FuseParallelLinears;
      transform.operations = operation_ids;
      dif::opt::apply(transform, context);
      denoiser.program = std::move(context.program);
      denoiser_bindings = std::move(context.bindings);
    }
    dif::ir::verify(denoiser.program);
    dif::ir::write_file(denoiser.program, arguments.diffir);
    const auto checkpoint_bind_ms = milliseconds_since(phase_start);

    phase_start = std::chrono::steady_clock::now();
    dif::runtime::RunOptions denoiser_options;
    denoiser_options.warmups = 0U;
    denoiser_options.iterations = 1U;
    denoiser_options.minimum_free_bytes = 512ULL * 1024ULL * 1024ULL;
    denoiser_options.cache_directory = arguments.cache_directory;
    denoiser_options.streamed_release_mapped_pages_per_copy = false;
    denoiser_options.tune_linear_operations =
        arguments.tune_linear_operations;
    denoiser_options.linear_algorithm_choices =
        arguments.linear_algorithm_choices;
    denoiser_options.cutlass_linear_operations =
        arguments.cutlass_linear_choices;
    denoiser_options.expand_linear_algorithms =
        arguments.expand_linear_algorithms;
    denoiser_options.persist_linear_heuristics =
        arguments.persist_linear_heuristics;
    denoiser_options.parallel_linear_groups =
        arguments.parallel_linear_groups;
    denoiser_options.linear_tuning_warmups =
        arguments.linear_tuning_warmups;
    denoiser_options.linear_tuning_iterations =
        arguments.linear_tuning_iterations;
    denoiser_options.linear_tuning_sessions =
        arguments.linear_tuning_sessions;
    denoiser_options.streamed_stage_threads =
        arguments.streamed_stage_threads;
    denoiser_options.streamed_prefetch_depth =
        arguments.streamed_prefetch_depth;
    denoiser_options.streamed_staging_buffers =
        arguments.streamed_staging_buffers;
    denoiser_options.pipelined_resident_upload =
        arguments.pipelined_resident_upload;
    denoiser_options.lazy_resident_upload =
        arguments.lazy_resident_upload;
    denoiser_options.profile_pipeline = arguments.profile_pipeline;
    if (arguments.lower_fused_swiglu) {
      for (const auto swiglu_id :
           arguments.fuse_parallel_swiglu_operations) {
        const auto swiglu = std::find_if(
            denoiser.program.operations.begin(), denoiser.program.operations.end(),
            [&](const dif::ir::Operation &operation) {
              return operation.id == swiglu_id &&
                     operation.opcode == dif::ir::Opcode::SwiGlu;
            });
        if (swiglu == denoiser.program.operations.end())
          dif::fail("transformed SwiGlu operation is missing");
        const auto producer = std::find_if(
            denoiser.program.operations.begin(), denoiser.program.operations.end(),
            [&](const dif::ir::Operation &operation) {
              return std::find(operation.outputs.begin(), operation.outputs.end(),
                               swiglu->inputs.front()) != operation.outputs.end();
            });
        if (producer == denoiser.program.operations.end() ||
            producer->opcode != dif::ir::Opcode::Linear)
          dif::fail("transformed SwiGlu input has no packed Linear producer");
        denoiser_options.fuse_linear_swiglu_operations.push_back(producer->id);
      }
    }
    std::optional<dif::compiler::ReshapeAliasPlan> reshape_alias_plan;
    if (arguments.alias_reshapes) {
      reshape_alias_plan.emplace(
          dif::compiler::plan_reshape_aliases(denoiser.program));
      denoiser_options.alias_reshape_operations =
          reshape_alias_plan->operation_ids;
    }
    std::optional<dif::compiler::StreamedResidencyPlan> residency_plan;
    std::string residency_hash;
    if (arguments.resident_plan_mib != 0U) {
      if (arguments.resident_plan_mib >
          std::numeric_limits<std::uint64_t>::max() / (1024ULL * 1024ULL))
        dif::fail("Krea resident-plan MiB value overflows bytes");
      constexpr std::uint64_t linear_workspace_bytes =
          64ULL * 1024ULL * 1024ULL;
      auto maximum_parallel_width = std::size_t{1U};
      for (const auto &group : arguments.parallel_linear_groups)
        maximum_parallel_width =
            std::max(maximum_parallel_width, group.size());
      const auto fixed_runtime_bytes =
          linear_workspace_bytes * maximum_parallel_width;
      if (arguments.whole_plan) {
        if (!whole_base_residency)
          dif::fail("whole-plan residency selection is missing");
        residency_plan.emplace();
        residency_plan->maximum_total_bytes =
            arguments.resident_plan_mib * 1024ULL * 1024ULL;
        residency_plan->fixed_runtime_bytes = fixed_runtime_bytes;
        residency_plan->resident_tensor_ids =
            whole_base_residency->resident_tensor_ids;
        residency_plan->resident_constant_bytes =
            whole_base_residency->resident_constant_bytes;
        const std::unordered_set<std::uint32_t> resident(
            residency_plan->resident_tensor_ids.begin(),
            residency_plan->resident_tensor_ids.end());
        const auto memory = dif::compiler::plan_memory(
            denoiser.program, 256U,
            denoiser_options.streamed_prefetch_depth, {}, resident,
            reshape_alias_plan
                ? reshape_alias_plan->output_to_root_input
                : std::unordered_map<std::uint32_t, std::uint32_t>{},
            false);
        residency_plan->memory_plan_bytes = memory.total_bytes;
        for (const auto &tensor : denoiser.program.tensors) {
          if (!tensor.has_role(dif::ir::TensorRole::Constant) ||
              !tensor.has_role(dif::ir::TensorRole::Streamed) ||
              resident.contains(tensor.id))
            continue;
          residency_plan->streamed_constant_bytes += tensor.byte_count();
        }
        residency_plan->required_bytes =
            memory.total_bytes + fixed_runtime_bytes +
            residency_plan->resident_constant_bytes;
        if (residency_plan->required_bytes >
            residency_plan->maximum_total_bytes)
          dif::fail("whole-plan execution exceeds its fixed memory budget");
      } else {
        residency_plan.emplace(dif::compiler::plan_streamed_residency(
            denoiser.program,
            arguments.resident_plan_mib * 1024ULL * 1024ULL,
            fixed_runtime_bytes,
            denoiser_options.streamed_prefetch_depth,
            reshape_alias_plan
                ? reshape_alias_plan->output_to_root_input
                : std::unordered_map<std::uint32_t, std::uint32_t>{},
            arguments.resident_order));
      }
      denoiser_options.resident_streamed_constants =
          residency_plan->resident_tensor_ids;
      residency_hash = residency_plan_hash(
          *residency_plan,
          reshape_alias_plan ? &*reshape_alias_plan : nullptr,
          denoiser_options.streamed_stage_threads,
          denoiser_options.streamed_prefetch_depth,
          denoiser_options.streamed_staging_buffers,
          arguments.resident_order,
          arguments.pipelined_resident_upload,
          arguments.lazy_resident_upload);
      std::cout << "KREA2_RESIDENCY_PLAN hash=" << residency_hash
                << " resident_tensors="
                << residency_plan->resident_tensor_ids.size()
                << " resident_bytes="
                << residency_plan->resident_constant_bytes
                << " streamed_bytes="
                << residency_plan->streamed_constant_bytes
                << " required_bytes=" << residency_plan->required_bytes
                << " maximum_bytes="
                << residency_plan->maximum_total_bytes << "\n"
                << std::flush;
    }
    const auto execution_policy_ms = milliseconds_since(phase_start);
    phase_start = std::chrono::steady_clock::now();
    auto backend = dif::runtime::make_cuda_executor();
    const auto executor_create_ms = milliseconds_since(phase_start);
    phase_start = std::chrono::steady_clock::now();
    auto prepared =
        backend->prepare(denoiser.program, denoiser_bindings, denoiser_options);
    const auto denoiser_prepare_call_ms = milliseconds_since(phase_start);

    phase_start = std::chrono::steady_clock::now();
    const auto cfg = dif::frontend::make_krea2_cfg_euler_step(image.dims);
    const auto euler = dif::frontend::make_krea2_euler_step(image.dims);
    dif::runtime::TensorMap scheduler_bindings;
    if (cfg_enabled) {
      scheduler_bindings.emplace(cfg.sample_input, image);
      scheduler_bindings.emplace(cfg.conditional_velocity_input, image);
      scheduler_bindings.emplace(cfg.unconditional_velocity_input, image);
      scheduler_bindings.emplace(
          cfg.guidance_input,
          scalar(dif::ir::DType::BF16, arguments.guidance));
      scheduler_bindings.emplace(cfg.current_timestep_input,
                                 scalar(dif::ir::DType::F32, 1.0F));
      scheduler_bindings.emplace(
          cfg.next_timestep_input,
          scalar(dif::ir::DType::F32, schedule.timesteps[1U]));
      scheduler_bindings.emplace(cfg.negative_one_constant,
                                 scalar(dif::ir::DType::BF16, -1.0F));
    } else {
      scheduler_bindings.emplace(euler.sample_input, image);
      scheduler_bindings.emplace(euler.velocity_input, image);
      scheduler_bindings.emplace(euler.current_timestep_input,
                                 scalar(dif::ir::DType::F32, 1.0F));
      scheduler_bindings.emplace(
          euler.next_timestep_input,
          scalar(dif::ir::DType::F32, schedule.timesteps[1U]));
    }
    dif::runtime::RunOptions scheduler_options;
    scheduler_options.warmups = 0U;
    scheduler_options.iterations = 1U;
    scheduler_options.minimum_free_bytes = 0U;
    std::unique_ptr<dif::runtime::PreparedExecution> scheduler_prepared;
    if (!arguments.whole_plan)
      scheduler_prepared = backend->prepare(
          cfg_enabled ? cfg.program : euler.program, scheduler_bindings,
          scheduler_options);
    const auto scheduler_prepare_ms = milliseconds_since(phase_start);

    std::vector<std::pair<std::string, dif::runtime::Tensor>> captures;
    captures.emplace_back("initial_image_tokens", image);
    captures.emplace_back("timesteps", native_schedule);
    std::vector<double> conditional_ms;
    std::vector<double> unconditional_ms;
    std::vector<double> scheduler_ms;
    std::vector<dif::runtime::LinearTuningResult> linear_tuning_results;
    std::vector<dif::runtime::LinearAlgorithmChoice>
        selected_linear_algorithms;
    std::vector<dif::runtime::PipelineProfile> denoiser_pipeline_profiles;
    std::vector<dif::runtime::LaunchTelemetry> denoiser_launch_telemetry;
    std::vector<std::vector<dif::runtime::OperationTiming>>
        denoiser_operation_timings;
    conditional_ms.reserve(executed_steps);
    unconditional_ms.reserve(executed_steps);
    scheduler_ms.reserve(executed_steps);
    denoiser_pipeline_profiles.reserve(executed_steps);
    denoiser_launch_telemetry.reserve(executed_steps);
    denoiser_operation_timings.reserve(executed_steps);
    const auto wall_start = std::chrono::steady_clock::now();

    if (arguments.whole_plan) {
      const auto start = std::chrono::steady_clock::now();
      auto complete = prepared->run(denoiser_bindings, denoiser_options);
      conditional_ms.push_back(milliseconds_since(start));
      unconditional_ms.push_back(0.0);
      scheduler_ms.push_back(0.0);
      if (arguments.profile_pipeline) {
        denoiser_pipeline_profiles.push_back(complete.pipeline_profile);
        denoiser_launch_telemetry.push_back(complete.run_telemetry);
        denoiser_operation_timings.push_back(complete.operation_timings);
      }
      linear_tuning_results = complete.linear_tuning_results;
      selected_linear_algorithms = complete.selected_linear_algorithms;
      for (std::uint32_t step = 0U; step < executed_steps; ++step) {
        const auto completed = step + 1U;
        const auto &velocity = complete.outputs.at(
            whole_execution->velocity_outputs.at(step));
        const auto &step_image = complete.outputs.at(
            whole_execution->image_outputs.at(step));
        captures.emplace_back(
            "step_" + std::to_string(completed) + "_velocity", velocity);
        captures.emplace_back(
            "step_" + std::to_string(completed) + "_image_tokens",
            step_image);
        if (completed == 1U) {
          captures.emplace_back("first_conditional_velocity", velocity);
          captures.emplace_back("first_velocity", velocity);
        }
        if (completed == arguments.steps / 2U)
          captures.emplace_back("midpoint_image_tokens", step_image);
      }
      image = complete.outputs.at(whole_execution->final_image_output);
      std::cout << "KREA2_NATIVE_WHOLE_PLAN evaluations=" << executed_steps
                << " wall_ms=" << conditional_ms.back() << "\n"
                << std::flush;
    } else for (std::uint32_t step = 0U; step < executed_steps; ++step) {
      const auto current = schedule.timesteps[step];
      const auto next = schedule.timesteps[step + 1U];
      denoiser_bindings.insert_or_assign(denoiser.image_tokens_input, image);
      denoiser_bindings.insert_or_assign(denoiser.context_input, positive);
      denoiser_bindings.insert_or_assign(denoiser.timestep_input,
                                         scalar(dif::ir::DType::BF16, current));
      denoiser_bindings.insert_or_assign(denoiser.validity_mask_input,
                                         positive_mask);
      auto start = std::chrono::steady_clock::now();
      auto conditional = prepared->run(denoiser_bindings, denoiser_options);
      conditional_ms.push_back(milliseconds_since(start));
      if (arguments.profile_pipeline) {
        denoiser_pipeline_profiles.push_back(conditional.pipeline_profile);
        denoiser_launch_telemetry.push_back(conditional.run_telemetry);
        denoiser_operation_timings.push_back(conditional.operation_timings);
      }
      if (step == 0U) {
        linear_tuning_results = conditional.linear_tuning_results;
        selected_linear_algorithms = conditional.selected_linear_algorithms;
      }

      const auto &conditional_velocity =
          conditional.outputs.at(denoiser.velocity_output);
      std::optional<dif::runtime::RunResult> unconditional;
      if (cfg_enabled) {
        denoiser_bindings.insert_or_assign(denoiser.context_input, *negative);
        denoiser_bindings.insert_or_assign(denoiser.validity_mask_input,
                                           *negative_mask);
        start = std::chrono::steady_clock::now();
        unconditional.emplace(
            prepared->run(denoiser_bindings, denoiser_options));
        unconditional_ms.push_back(milliseconds_since(start));
        const auto &unconditional_velocity =
            unconditional->outputs.at(denoiser.velocity_output);
        scheduler_bindings.insert_or_assign(cfg.sample_input, image);
        scheduler_bindings.insert_or_assign(cfg.conditional_velocity_input,
                                            conditional_velocity);
        scheduler_bindings.insert_or_assign(cfg.unconditional_velocity_input,
                                            unconditional_velocity);
        scheduler_bindings.insert_or_assign(
            cfg.current_timestep_input,
            scalar(dif::ir::DType::F32, current));
        scheduler_bindings.insert_or_assign(
            cfg.next_timestep_input, scalar(dif::ir::DType::F32, next));
      } else {
        unconditional_ms.push_back(0.0);
        scheduler_bindings.insert_or_assign(euler.sample_input, image);
        scheduler_bindings.insert_or_assign(euler.velocity_input,
                                            conditional_velocity);
        scheduler_bindings.insert_or_assign(
            euler.current_timestep_input,
            scalar(dif::ir::DType::F32, current));
        scheduler_bindings.insert_or_assign(
            euler.next_timestep_input, scalar(dif::ir::DType::F32, next));
      }
      start = std::chrono::steady_clock::now();
      auto update =
          scheduler_prepared->run(scheduler_bindings, scheduler_options);
      scheduler_ms.push_back(milliseconds_since(start));
      image = update.outputs.at(cfg_enabled ? cfg.sample_output
                                            : euler.sample_output);

      const auto completed = step + 1U;
      captures.emplace_back(
          "step_" + std::to_string(completed) + "_velocity",
          cfg_enabled ? update.outputs.at(cfg.velocity_output)
                      : conditional_velocity);
      captures.emplace_back(
          "step_" + std::to_string(completed) + "_image_tokens", image);
      if (completed == 1U) {
        captures.emplace_back("first_conditional_velocity", conditional_velocity);
        if (cfg_enabled) {
          captures.emplace_back(
              "first_unconditional_velocity",
              unconditional->outputs.at(denoiser.velocity_output));
          captures.emplace_back("first_guided_velocity",
                                update.outputs.at(cfg.velocity_output));
        } else {
          captures.emplace_back("first_velocity", conditional_velocity);
        }
      }
      if (completed == arguments.steps / 2U)
        captures.emplace_back("midpoint_image_tokens", image);
      std::cout << "KREA2_NATIVE_STEP step=" << completed << "/"
                << executed_steps << " cond_ms=" << conditional_ms.back()
                << (cfg_enabled ? " uncond_ms=" : " cfg=off uncond_ms=")
                << unconditional_ms.back() << " scheduler_ms="
                << scheduler_ms.back() << "\n"
                << std::flush;
    }
    if (executed_steps == arguments.steps)
      captures.emplace_back("final_image_tokens", image);
    const auto wall_ms = milliseconds_since(wall_start);

    std::vector<dif::weights::SafeTensorWriteSpec> specs;
    specs.reserve(captures.size());
    for (const auto &[name, tensor] : captures)
      specs.push_back({name, tensor.dtype, tensor.dims});
    dif::weights::SafeTensorWriter writer(arguments.output, std::move(specs));
    for (const auto &[name, tensor] : captures)
      writer.append(name, std::span<const std::uint8_t>(tensor.data(),
                                                       tensor.byte_size()));
    (void)writer.finish();

    std::ofstream report(arguments.report, std::ios::trunc);
    if (!report)
      dif::fail("cannot create Krea sampler report");
    report << std::setprecision(17)
           << "{\n  \"source_commit\": "
           << std::quoted("db3984fbc6e13b34c0064990fc2d95ac64d00058")
           << ",\n  \"diffir_fingerprint\": "
           << std::quoted(dif::hex_digest(dif::ir::fingerprint(denoiser.program)))
           << ",\n  \"checkpoint\": " << std::quoted(arguments.checkpoint.string())
           << ",\n  \"weight_bundle\": " << std::quoted(arguments.bundle.string())
           << ",\n  \"seed\": " << arguments.seed
           << ",\n  \"steps\": " << arguments.steps
           << ",\n  \"executed_steps\": " << executed_steps
           << ",\n  \"whole_plan\": "
           << (arguments.whole_plan ? "true" : "false")
           << ",\n  \"guidance\": " << arguments.guidance
           << ",\n  \"mu\": " << schedule.mu
           << ",\n  \"dtype\": \"BF16\",\n  \"geometry\": \"1024x1024\","
           << "\n  \"text_token_cap\": " << config.text_tokens << ","
           << "\n  \"f32_to_bf16_parameters\": " << converted_parameters
           << ",\n  \"converted_bf16_bytes\": " << converted_bytes
           << ",\n  \"compiler_transforms\": {\"parallel_linear_swiglu\":[";
    for (std::size_t index = 0U;
         index < arguments.fuse_parallel_swiglu_operations.size(); ++index) {
      if (index != 0U)
        report << ',';
      report << arguments.fuse_parallel_swiglu_operations[index];
    }
    report << "],\"lower_fused_swiglu\":"
           << (arguments.lower_fused_swiglu ? "true" : "false")
           << ",\"parallel_linear_groups\":[";
    for (std::size_t group = 0U;
         group < arguments.fuse_parallel_linear_groups.size(); ++group) {
      if (group != 0U)
        report << ',';
      report << '[';
      for (std::size_t index = 0U;
           index < arguments.fuse_parallel_linear_groups[group].size();
           ++index) {
        if (index != 0U)
          report << ',';
        report << arguments.fuse_parallel_linear_groups[group][index];
      }
      report << ']';
    }
    report << "],\"parallel_execution_groups\":[";
    for (std::size_t group = 0U;
         group < arguments.parallel_linear_groups.size(); ++group) {
      if (group != 0U)
        report << ',';
      report << '[';
      for (std::size_t index = 0U;
           index < arguments.parallel_linear_groups[group].size(); ++index) {
        if (index != 0U)
          report << ',';
        report << arguments.parallel_linear_groups[group][index];
      }
      report << ']';
    }
    report << "]}"
           << ",\n  \"host_phases_ms\": {\"frontend_build\":"
           << frontend_build_ms << ",\"input_open\":" << input_open_ms
           << ",\"input_prepare\":" << input_prepare_ms
           << ",\"checkpoint_bind\":" << checkpoint_bind_ms
           << ",\"execution_policy\":" << execution_policy_ms
           << ",\"executor_create\":" << executor_create_ms
           << ",\"denoiser_prepare_call\":" << denoiser_prepare_call_ms
           << ",\"scheduler_prepare\":" << scheduler_prepare_ms
           << ",\"pre_denoise_total\":"
           << std::chrono::duration<double, std::milli>(
                  wall_start - process_start).count()
           << "}"
           << ",\n  \"denoiser_preparation_ms\": "
           << prepared->preparation_milliseconds()
           << ",\n  \"denoiser_resident_bytes\": " << prepared->resident_bytes()
           << ",\n  \"reshape_alias_plan\": ";
    if (reshape_alias_plan) {
      report << "{\"operations\":"
             << reshape_alias_plan->operation_ids.size()
             << ",\"eliminated_materialization_bytes\":"
             << reshape_alias_plan->eliminated_materialization_bytes
             << ",\"operation_ids\":[";
      for (std::size_t index = 0U;
           index < reshape_alias_plan->operation_ids.size(); ++index) {
        if (index != 0U)
          report << ',';
        report << reshape_alias_plan->operation_ids[index];
      }
      report << "]}";
    } else {
      report << "null";
    }
    report
           << ",\n  \"resident_plan\": ";
    if (residency_plan) {
      report << "{\"identity_sha256\":" << std::quoted(residency_hash)
             << ",\"selection_order\":"
             << std::quoted(
                    arguments.resident_order ==
                            dif::compiler::StreamedResidencyOrder::LargestFirst
                        ? "largest_first"
                        : "first_consumer")
             << ",\"maximum_total_bytes\":"
             << residency_plan->maximum_total_bytes
             << ",\"fixed_runtime_bytes\":"
             << residency_plan->fixed_runtime_bytes
             << ",\"memory_plan_bytes\":"
             << residency_plan->memory_plan_bytes
             << ",\"resident_constant_bytes\":"
             << residency_plan->resident_constant_bytes
             << ",\"streamed_constant_bytes\":"
             << residency_plan->streamed_constant_bytes
             << ",\"required_bytes\":" << residency_plan->required_bytes
             << ",\"resident_tensor_ids\":[";
      for (std::size_t index = 0U;
           index < residency_plan->resident_tensor_ids.size(); ++index) {
        if (index != 0U)
          report << ',';
        report << residency_plan->resident_tensor_ids[index];
      }
      report << "]}";
    } else {
      report << "null";
    }
    report
           << ",\n  \"streamed_transport\": {\"stage_threads\":"
           << denoiser_options.streamed_stage_threads
           << ",\"prefetch_depth\":"
           << denoiser_options.streamed_prefetch_depth
           << ",\"staging_buffers\":"
           << denoiser_options.streamed_staging_buffers
           << ",\"pipelined_resident_upload\":"
           << (denoiser_options.pipelined_resident_upload ? "true" : "false")
           << ",\"lazy_resident_upload\":"
           << (denoiser_options.lazy_resident_upload ? "true" : "false")
           << "},\n  \"linear_plan\": {\"expanded_algorithms\":"
           << (arguments.expand_linear_algorithms ? "true" : "false")
           << ",\"persist_heuristics\":"
           << (arguments.persist_linear_heuristics ? "true" : "false")
           << ",\"requested_tuning_operations\":[";
    for (std::size_t index = 0U;
         index < arguments.tune_linear_operations.size(); ++index) {
      if (index != 0U)
        report << ',';
      report << arguments.tune_linear_operations[index];
    }
    report << "],\"requested_algorithm_choices\":[";
    for (std::size_t index = 0U;
         index < arguments.linear_algorithm_choices.size(); ++index) {
      if (index != 0U)
        report << ',';
      report << "{\"operation_id\":"
             << arguments.linear_algorithm_choices[index].operation_id
             << ",\"heuristic_rank\":"
             << arguments.linear_algorithm_choices[index].heuristic_rank
             << '}';
    }
    report << "],\"selected_algorithm_choices\":[";
    for (std::size_t index = 0U;
         index < selected_linear_algorithms.size(); ++index) {
      if (index != 0U)
        report << ',';
      report << "{\"operation_id\":"
             << selected_linear_algorithms[index].operation_id
             << ",\"heuristic_rank\":"
             << selected_linear_algorithms[index].heuristic_rank << '}';
    }
    report << "],\"tuning_results\":[";
    for (std::size_t index = 0U; index < linear_tuning_results.size();
         ++index) {
      if (index != 0U)
        report << ',';
      const auto &tuning = linear_tuning_results[index];
      report << "{\"operation_id\":" << tuning.operation_id
             << ",\"selected_heuristic_rank\":"
             << tuning.selected_heuristic_index
             << ",\"selected_algorithm_id\":"
             << tuning.selected_algorithm_id
             << ",\"default_mean_ms\":"
             << tuning.default_mean_milliseconds
             << ",\"selected_mean_ms\":"
             << tuning.selected_mean_milliseconds
             << ",\"observed_noise_ms\":"
             << tuning.observed_noise_milliseconds
             << ",\"tuning_ms\":" << tuning.tuning_milliseconds
             << ",\"changed_from_default\":"
             << (tuning.changed_from_default ? "true" : "false")
             << ",\"decision\":" << std::quoted(tuning.decision)
             << ",\"candidates\":[";
      for (std::size_t candidate_index = 0U;
           candidate_index < tuning.candidates.size(); ++candidate_index) {
        if (candidate_index != 0U)
          report << ',';
        const auto &candidate = tuning.candidates[candidate_index];
        report << "{\"heuristic_rank\":" << candidate.heuristic_index
               << ",\"algorithm_id\":" << candidate.algorithm_id
               << ",\"tile_id\":" << candidate.tile_id
               << ",\"stages_id\":" << candidate.stages_id
               << ",\"split_k\":" << candidate.split_k
               << ",\"reduction_scheme\":"
               << candidate.reduction_scheme
               << ",\"cta_swizzle\":" << candidate.cta_swizzle
               << ",\"custom_option\":" << candidate.custom_option
               << ",\"waves_count\":" << candidate.waves_count
               << ",\"workspace_bytes\":" << candidate.workspace_bytes
               << ",\"mean_ms\":" << candidate.mean_milliseconds
               << ",\"session_min_ms\":"
               << candidate.minimum_session_milliseconds
               << ",\"session_max_ms\":"
               << candidate.maximum_session_milliseconds << '}';
      }
      report << "]}";
    }
    report << "]},\n  \"wall_ms\": " << wall_ms
           << ",\n  \"profile_pipeline\": {\"enabled\":"
           << (arguments.profile_pipeline ? "true" : "false")
           << ",\"steps\":[";
    for (std::size_t step = 0U; step < denoiser_pipeline_profiles.size();
         ++step) {
      if (step != 0U)
        report << ',';
      const auto &profile = denoiser_pipeline_profiles[step];
      const auto &telemetry = denoiser_launch_telemetry[step];
      report << "{\"step\":" << step + 1U
             << ",\"resident_weight_bytes\":"
             << profile.resident_weight_bytes
             << ",\"resident_host_prefault_ms\":"
             << profile.resident_host_prefault_milliseconds
             << ",\"resident_minor_page_faults\":"
             << profile.resident_minor_page_faults
             << ",\"resident_major_page_faults\":"
             << profile.resident_major_page_faults
             << ",\"resident_h2d_ms\":"
             << profile.resident_h2d_milliseconds
             << ",\"resident_upload_ms\":"
             << profile.resident_upload_milliseconds
             << ",\"streamed_weight_bytes\":"
             << profile.streamed_weight_bytes
             << ",\"streamed_host_stage_ms\":"
             << profile.streamed_host_stage_milliseconds
             << ",\"streamed_host_wait_ms\":"
             << profile.streamed_host_wait_milliseconds
             << ",\"streamed_h2d_ms\":"
             << profile.streamed_h2d_milliseconds
             << ",\"operation_kernel_ms\":"
             << profile.operation_kernel_milliseconds
             << ",\"attention_kernel_ms\":"
             << profile.attention_kernel_milliseconds
             << ",\"non_kernel_device_timeline_ms\":"
             << profile.non_kernel_device_timeline_milliseconds
             << ",\"telemetry\":{\"kernel_launches\":"
             << telemetry.kernel_launches
             << ",\"cublaslt_matmuls\":" << telemetry.cublaslt_matmuls
             << ",\"cublas_gemms\":" << telemetry.cublas_gemms
             << ",\"cudnn_attention_dispatches\":"
             << telemetry.cudnn_attention_dispatches
             << ",\"cutlass_launches\":" << telemetry.cutlass_launches
             << ",\"h2d_copies\":" << telemetry.h2d_copies
             << ",\"h2d_bytes\":" << telemetry.h2d_bytes
             << ",\"d2h_copies\":" << telemetry.d2h_copies
             << ",\"d2h_bytes\":" << telemetry.d2h_bytes
             << ",\"event_records\":" << telemetry.event_records
             << ",\"stream_wait_events\":"
             << telemetry.stream_wait_events
             << ",\"host_event_synchronizes\":"
             << telemetry.host_event_synchronizes
             << ",\"host_stream_synchronizes\":"
             << telemetry.host_stream_synchronizes << "},\"operations\":[";
      const auto &timings = denoiser_operation_timings[step];
      for (std::size_t index = 0U; index < timings.size(); ++index) {
        if (index != 0U)
          report << ',';
        const auto &timing = timings[index];
        report << "{\"operation_id\":" << timing.operation_id
               << ",\"opcode\":"
               << std::quoted(dif::ir::opcode_name(timing.opcode))
               << ",\"mean_ms\":" << timing.mean_milliseconds
               << ",\"min_ms\":" << timing.minimum_milliseconds
               << ",\"max_ms\":" << timing.maximum_milliseconds << '}';
      }
      report << "]}";
    }
    report << "]}"
           << ",\n  \"steps_detail\": [\n";
    for (std::size_t index = 0U; index < conditional_ms.size(); ++index) {
      report << "    {\"step\":" << index + 1U
             << ",\"conditional_ms\":" << conditional_ms[index]
             << ",\"unconditional_ms\":" << unconditional_ms[index]
             << ",\"scheduler_ms\":" << scheduler_ms[index] << "}"
             << (index + 1U == conditional_ms.size() ? "\n" : ",\n");
    }
    report << "  ],\n  \"trajectory\": {\n";
    bool first_metric = true;
    bool passed = true;
    for (const auto &[name, tensor] : captures) {
      if (name == "timesteps")
        continue;
      const auto reference =
          dif::weights::map_safetensor(reference_file, name);
      const auto metric = measure(reference, tensor);
      if (!first_metric) report << ",\n";
      first_metric = false;
      report << "    " << std::quoted(name) << ": ";
      emit_metrics(report, metric);
      if (name != "initial_image_tokens" &&
          (metric.cosine < 0.999 || metric.relative_l2 > 0.02 ||
           metric.nonfinite != 0U))
        passed = false;
    }
    report << "\n  },\n  \"parity_pass\": " << (passed ? "true" : "false")
           << ",\n  \"output\": " << std::quoted(arguments.output.string())
           << ",\n  \"output_sha256\": "
           << std::quoted(dif::hex_digest(dif::sha256_file(arguments.output)))
           << "\n}\n";
    if (!passed)
      dif::fail("Krea sampler trajectory missed the fixed parity bars");
    std::cout << "KREA2_SAMPLER_PASS steps=" << executed_steps
              << " fingerprint="
              << dif::hex_digest(dif::ir::fingerprint(denoiser.program))
              << " output_sha256="
              << dif::hex_digest(dif::sha256_file(arguments.output)) << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difkrea2sample: " << error.what() << "\n";
    return 1;
  }
}
