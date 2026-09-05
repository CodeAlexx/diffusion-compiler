// Runtime-level regression gate for Attention Implementation 5: finite BF16
// values outside FP16's range, overflowing FP16 tile sums, sequence tails,
// batch/head strides, and nonuniform attention against the CPU executor.
#include "dif/ir/ir.hpp"
#include "dif/runtime/device_probe.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/runtime/tensor.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

dif::ir::Program program(std::uint64_t sequence, bool batched) {
  using namespace dif::ir;
  const std::vector<std::uint64_t> shape =
      batched ? std::vector<std::uint64_t>{2U, sequence, 3U, 128U}
              : std::vector<std::uint64_t>{sequence, 2U, 128U};
  Program result;
  result.tensors = {{1U, DType::BF16, TensorRole::Input, shape},
                    {2U, DType::BF16, TensorRole::Input, shape},
                    {3U, DType::BF16, TensorRole::Input, shape},
                    {4U, DType::BF16, TensorRole::Output, shape}};
  result.operations = {{1U, Opcode::Attention, {1U, 2U, 3U}, {4U},
                        {Attribute::u64(AttrKey::Implementation, 5U)}}};
  return result;
}

dif::runtime::RunOptions options() {
  dif::runtime::RunOptions result;
  result.warmups = 0U;
  result.iterations = 1U;
  result.minimum_free_bytes = 0U;
  return result;
}

void gate(std::uint64_t sequence, bool batched) {
  const auto p = program(sequence, batched);
  const auto batches = batched ? 2U : 1U;
  const auto heads = batched ? 3U : 2U;
  dif::runtime::TensorMap inputs;
  for (std::uint32_t id = 1U; id <= 3U; ++id)
    inputs.emplace(id, dif::runtime::zeros(*p.tensor(id)));
  auto executor = dif::runtime::make_cuda_executor();
  auto prepared = executor->prepare(p, inputs, options());
  // With zero Q/K, each output is the mean V over the sequence. Give each
  // batch, head, and channel a distinct constant to catch stride errors too.
  for (const auto amplitude : {1.0F, 4096.0F, 131072.0F, 0x1p-30F}) {
    for (std::uint32_t batch = 0U; batch < batches; ++batch)
      for (std::uint64_t row = 0U; row < sequence; ++row)
        for (std::uint32_t head = 0U; head < heads; ++head)
          for (std::uint32_t channel = 0U; channel < 128U; ++channel) {
            const auto index = ((batch * sequence + row) * heads + head) *
                                   128U + channel;
            const auto sign = channel % 2U == 0U ? 1.0F : -1.0F;
            dif::runtime::store_float(inputs.at(3U), index,
                                      sign * amplitude * (1U + batch + head));
          }
    const auto result = prepared->run(inputs, options());
    const auto &output = result.outputs.at(4U);
    for (std::uint64_t index = 0U; index < output.element_count(); ++index) {
      const auto value = dif::runtime::load_float(output, index);
      if (!std::isfinite(value) ||
          value != dif::runtime::load_float(inputs.at(3U), index))
        throw std::runtime_error("uniform attention mismatch at S=" +
                                 std::to_string(sequence) + " amplitude=" +
                                 std::to_string(amplitude));
    }
  }
  // Nonuniform inputs exercise softmax and changing running maxima across
  // tiles. Compare full outputs, not just finiteness or constant fixtures.
  for (std::uint32_t id = 1U; id <= 3U; ++id) {
    auto &tensor = inputs.at(id);
    std::uint64_t state = id * 6364136223846793005ULL;
    for (std::uint64_t index = 0U; index < tensor.element_count(); ++index) {
      state = state * 6364136223846793005ULL + 1442695040888963407ULL;
      const auto value = static_cast<float>((state >> 32U) & 0xffffU) /
                             32768.0F - 1.0F;
      dif::runtime::store_float(tensor, index, value);
    }
  }
  // The CPU executor evaluates semantic attention regardless of the physical
  // implementation attribute. Keep 5 so batched graphs remain legal in IR.
  const auto reference = dif::runtime::make_cpu_executor()->run(
      p, inputs, options());
  const auto result = prepared->run(inputs, options());
  const auto &expected = reference.outputs.at(4U);
  const auto &actual = result.outputs.at(4U);
  double error = 0.0, norm = 0.0;
  for (std::uint64_t index = 0U; index < actual.element_count(); ++index) {
    const double value = dif::runtime::load_float(actual, index);
    const double target = dif::runtime::load_float(expected, index);
    if (!std::isfinite(value))
      throw std::runtime_error("nonfinite random attention output");
    error += (value - target) * (value - target);
    norm += target * target;
  }
  const auto relative_l2 = std::sqrt(error / norm);
  if (!(relative_l2 < 0.005))
    throw std::runtime_error("random attention relative L2 exceeds 0.005");
  if (prepared->run(inputs, options()).outputs.at(4U).bytes != actual.bytes)
    throw std::runtime_error("exact attention is not repeatable");
  std::cout << "EXACT_STREAM S=" << sequence << " B=" << batches
            << " H=" << heads << " relative_l2=" << relative_l2 << '\n';
}

} // namespace

int main() {
#if !DIF_HAS_EXACT_STREAM_ATTENTION
  std::cout << "exact stream attention not built; gates skipped\n";
  return 0;
#else
  if (!dif::runtime::cuda_available()) {
    std::cout << "CUDA unavailable; exact stream attention gates skipped\n";
    return 0;
  }
  try {
    const auto target = dif::runtime::probe_target(dif::runtime::ProbeBackend::Cuda);
    if (target.compute_major < 8U) {
      std::cout << "exact stream attention requires sm_80+; gates skipped\n";
      return 0;
    }
    for (const auto sequence : {1U, 31U, 32U, 33U, 65U, 129U})
      gate(sequence, false);
    gate(65U, true);
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "exact stream attention tests passed\n";
  return 0;
#endif
}
