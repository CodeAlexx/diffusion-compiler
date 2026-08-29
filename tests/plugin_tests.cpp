#include "dif/backend/plugin.hpp"
#include "dif/ir/ir.hpp"
#include "dif/runtime/tensor.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

dif::runtime::Tensor tensor(const std::vector<float> &values) {
  dif::runtime::Tensor result{dif::ir::DType::F32,
                              {1U, static_cast<std::uint64_t>(values.size())}, {}};
  result.bytes.resize(values.size() * sizeof(float));
  std::memcpy(result.bytes.data(), values.data(), result.bytes.size());
  return result;
}

} // namespace

bool run_plugin(const char *path, const char *backend_name,
                const char *device_name, std::uint64_t resident_bytes) {
  using namespace dif::ir;
  Program program;
  program.tensors = {
      {1, DType::F32, TensorRole::Input, {1, 4}},
      {2, DType::F32, TensorRole::Constant, {1, 4}},
      {3, DType::F32, TensorRole::Output, {1, 4}},
  };
  program.operations = {{1, Opcode::Add, {1, 2}, {3}, {}}};
  dif::runtime::TensorMap inputs;
  inputs.emplace(1, tensor({1.0F, 2.0F, 3.0F, 4.0F}));
  inputs.emplace(2, tensor({0.5F, 1.0F, 1.5F, 2.0F}));
  auto executor = dif::backend::make_plugin_executor(path);
  dif::runtime::RunOptions options;
  options.warmups = 0;
  options.iterations = 1;
  const auto result = executor->run(program, inputs, options);
  const auto output = result.outputs.at(3).f32();
  const std::vector<float> expected = {1.5F, 3.0F, 4.5F, 6.0F};
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (std::abs(output[i] - expected[i]) > 1.0e-7F) {
      std::cerr << "plugin output mismatch\n";
      return false;
    }
  }
  if (result.backend_name != backend_name || result.device_name != device_name ||
      result.resident_bytes != resident_bytes ||
      result.preparation_milliseconds <= 0.0) {
    std::cerr << "plugin telemetry mismatch\n";
    return false;
  }
  return true;
}

int main() {
  if (!run_plugin(DIF_MOCK_BACKEND_V1_PATH, "mock-v1", "mock-device", 0U) ||
      !run_plugin(DIF_MOCK_BACKEND_V2_PATH, "mock-v2", "mock-v2-device",
                  4U * sizeof(float)))
    return 1;
  std::cout << "PASS: backend v1 fallback and v2 resident-constant ABI\n";
  return 0;
}
