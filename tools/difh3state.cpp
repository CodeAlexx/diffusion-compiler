#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Options {
  std::vector<std::filesystem::path> conditions;
  std::vector<std::filesystem::path> condition_noise;
  std::filesystem::path target_noise;
  std::filesystem::path output;
  float condition_timestep{0.999F};
};

void usage() {
  std::cerr
      << "usage: difh3state --condition CLEAN.diftensor [--condition CLEAN2]"
         " --condition-noise NOISE.diftensor [--condition-noise NOISE2]"
         " --target-noise NOISE.diftensor"
         " --condition-timestep F32 --output STATE.diftensor\n";
}

Options parse(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    auto value = [&](const char *name) -> std::string {
      if (index + 1 >= argc)
        dif::fail(std::string("missing value for ") + name);
      return argv[++index];
    };
    if (option == "--condition")
      options.conditions.emplace_back(value("--condition"));
    else if (option == "--condition-noise")
      options.condition_noise.emplace_back(value("--condition-noise"));
    else if (option == "--target-noise")
      options.target_noise = value("--target-noise");
    else if (option == "--condition-timestep") {
      const auto text = value("--condition-timestep");
      char *end = nullptr;
      options.condition_timestep = std::strtof(text.c_str(), &end);
      if (!end || *end != '\0' ||
          !std::isfinite(options.condition_timestep))
        dif::fail("invalid condition timestep");
    } else if (option == "--output")
      options.output = value("--output");
    else {
      usage();
      dif::fail("invalid difh3state command line");
    }
  }
  if (options.conditions.empty() || options.condition_noise.empty() ||
      options.target_noise.empty() || options.output.empty()) {
    usage();
    dif::fail("difh3state is missing a required argument");
  }
  if (options.condition_noise.size() != options.conditions.size())
    dif::fail("each H3 visual condition requires its own restarted noise tensor");
  if (options.condition_timestep < 0.0F ||
      options.condition_timestep > 1.0F)
    dif::fail("condition timestep must be in [0,1]");
  if (std::filesystem::exists(options.output))
    dif::fail("refusing to overwrite " + options.output.string());
  return options;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto options = parse(argc, argv);
    std::vector<dif::runtime::Tensor> clean;
    std::uint64_t condition_rows = 0U;
    std::uint64_t columns = 0U;
    for (const auto &path : options.conditions) {
      auto tensor = dif::runtime::read_tensor(path);
      if (tensor.dtype != dif::ir::DType::F32 || tensor.dims.size() != 2U ||
          tensor.dims[0] == 0U || tensor.dims[1] == 0U)
        dif::fail("clean condition rows must be F32 [N,C]");
      if (columns == 0U)
        columns = tensor.dims[1];
      if (tensor.dims[1] != columns)
        dif::fail("clean condition tensors have different column counts");
      condition_rows += tensor.dims[0];
      clean.push_back(std::move(tensor));
    }
    std::vector<dif::runtime::Tensor> condition_noise;
    condition_noise.reserve(options.condition_noise.size());
    for (std::size_t index = 0U; index < options.condition_noise.size();
         ++index) {
      auto tensor = dif::runtime::read_tensor(options.condition_noise[index]);
      if (tensor.dtype != dif::ir::DType::F32 ||
          tensor.dims != clean[index].dims)
        dif::fail("condition noise shape must equal its clean condition rows");
      condition_noise.push_back(std::move(tensor));
    }
    const auto target_noise = dif::runtime::read_tensor(options.target_noise);
    if (target_noise.dtype != dif::ir::DType::F32 ||
        target_noise.dims.size() != 2U || target_noise.dims[1] != columns)
      dif::fail("target noise must be F32 [N,C] matching condition columns");
    dif::runtime::Tensor state{
        dif::ir::DType::F32,
        {condition_rows + target_noise.dims[0], columns}, {}};
    state.bytes.resize(static_cast<std::size_t>(state.element_count() *
                                                sizeof(float)));
    auto output = state.f32();
    const auto timestep = options.condition_timestep;
    std::uint64_t row_offset = 0U;
    for (std::size_t condition = 0U; condition < clean.size(); ++condition) {
      const auto &tensor = clean[condition];
      const auto values = tensor.f32();
      const auto noise = condition_noise[condition].f32();
      for (std::uint64_t index = 0U; index < tensor.element_count(); ++index) {
        const auto target = row_offset * columns + index;
        output[static_cast<std::size_t>(target)] =
            timestep * values[static_cast<std::size_t>(index)] +
            (1.0F - timestep) * noise[static_cast<std::size_t>(index)];
      }
      row_offset += tensor.dims[0];
    }
    std::copy(target_noise.f32().begin(), target_noise.f32().end(),
              output.begin() + static_cast<std::ptrdiff_t>(condition_rows *
                                                           columns));
    dif::runtime::write_tensor(state, options.output);
    const auto hash = dif::sha256(
        {state.data(), static_cast<std::size_t>(state.byte_size())});
    std::cout << "H3_INITIAL_STATE PASS condition_rows=" << condition_rows
              << " target_rows=" << target_noise.dims[0]
              << " columns=" << columns
              << " condition_timestep=" << timestep
              << " payload_sha256=" << dif::hex_digest(hash) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difh3state: " << error.what() << '\n';
    return 1;
  }
}
