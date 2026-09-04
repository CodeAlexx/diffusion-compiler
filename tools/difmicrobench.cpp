// One opcode, one shape, measured in isolation.
//
// This exists because a step-level profile can tell you that everything is
// slow without telling you why. When every operation in a 9,400-launch step
// runs two orders of magnitude off memory bandwidth regardless of what it
// does, the question is whether the kernels are slow on their own or slow
// only in that context -- and the only way to know is to run one by itself.
//
//   difmicrobench --rows 768 --columns 6144 [--iterations 50]

#include "dif/ir/ir.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using dif::ir::DType;
using dif::ir::Opcode;
using dif::ir::Program;
using dif::ir::TensorRole;

dif::runtime::Tensor filled(DType dtype, std::vector<std::uint64_t> dims) {
  dif::runtime::Tensor tensor;
  tensor.dtype = dtype;
  tensor.dims = std::move(dims);
  tensor.bytes.assign(static_cast<std::size_t>(tensor.element_count()) *
                          dif::ir::dtype_size(dtype),
                      static_cast<unsigned char>(0x3C));
  return tensor;
}

struct Case {
  std::string name;
  Program program;
  // Bytes the kernel must read and write if it is perfectly efficient.
  std::uint64_t traffic{};
};

// The same kernel, N times in one program, each reading what the last wrote.
// If a kernel costs the same here as it does alone, a long program is not
// what makes it slow.
Case chained(std::uint64_t rows, std::uint64_t columns, std::uint64_t length) {
  const std::vector<std::uint64_t> shape{rows, columns};
  Program program;
  program.tensors.push_back({1U, DType::BF16, TensorRole::Input, shape});
  for (std::uint64_t index = 0U; index < length; ++index) {
    const auto id = static_cast<std::uint32_t>(index + 2U);
    program.tensors.push_back(
        {id, DType::BF16,
         index + 1U == length ? TensorRole::Output : TensorRole::Internal,
         shape});
    program.operations.push_back({static_cast<std::uint32_t>(index + 1U),
                                  Opcode::Add,
                                  {static_cast<std::uint32_t>(index + 1U),
                                   static_cast<std::uint32_t>(index + 1U)},
                                  {id},
                                  {}});
  }
  dif::ir::verify(program);
  return {"add x" + std::to_string(length), std::move(program),
          rows * columns * 6U * length};
}

std::vector<Case> cases(std::uint64_t rows, std::uint64_t columns) {
  const std::vector<std::uint64_t> shape{rows, columns};
  const auto elements = rows * columns;
  std::vector<Case> built;
  {
    Program program;
    program.tensors = {{1U, DType::BF16, TensorRole::Input, shape},
                       {2U, DType::F32, TensorRole::Output, shape}};
    program.operations = {{1U, Opcode::Cast, {1U}, {2U}, {}}};
    built.push_back({"cast bf16->f32", std::move(program), elements * 6U});
  }
  {
    Program program;
    program.tensors = {{1U, DType::BF16, TensorRole::Input, shape},
                       {2U, DType::BF16, TensorRole::Input, shape},
                       {3U, DType::BF16, TensorRole::Output, shape}};
    program.operations = {{1U, Opcode::Multiply, {1U, 2U}, {3U}, {}}};
    built.push_back({"multiply bf16", std::move(program), elements * 6U});
  }
  {
    Program program;
    program.tensors = {{1U, DType::BF16, TensorRole::Input, shape},
                       {2U, DType::BF16, TensorRole::Input, shape},
                       {3U, DType::BF16, TensorRole::Output, shape}};
    program.operations = {{1U, Opcode::Add, {1U, 2U}, {3U}, {}}};
    built.push_back({"add bf16", std::move(program), elements * 6U});
  }
  {
    // Fill reads nothing, so its traffic is the write alone.
    Program program;
    program.tensors = {{1U, DType::BF16, TensorRole::Output, shape}};
    program.operations = {
        {1U, Opcode::Fill, {}, {1U},
         {dif::ir::Attribute::f64(dif::ir::AttrKey::Value, 1.0)}}};
    built.push_back({"fill bf16", std::move(program), elements * 2U});
  }
  for (auto &value : built)
    dif::ir::verify(value.program);
  return built;
}

} // namespace

int main(int argc, char **argv) {
  std::uint64_t rows = 768U;
  std::uint64_t columns = 6144U;
  std::uint64_t iterations = 50U;
  std::uint64_t chain = 0U;
  for (int index = 1; index < argc; ++index) {
    const std::string flag = argv[index];
    const auto value = [&]() -> std::uint64_t {
      if (index + 1 >= argc) {
        std::cerr << flag << " needs a value\n";
        std::exit(2);
      }
      return std::stoull(argv[++index]);
    };
    if (flag == "--rows")
      rows = value();
    else if (flag == "--columns")
      columns = value();
    else if (flag == "--iterations")
      iterations = value();
    else if (flag == "--chain")
      chain = value();
    else {
      std::cerr << "unknown argument " << flag << "\n";
      return 2;
    }
  }

  try {
    auto executor = dif::runtime::make_cuda_executor();
    if (!executor) {
      std::cerr << "no CUDA device\n";
      return 1;
    }
    std::cout << "MICROBENCH rows=" << rows << " columns=" << columns
              << " iterations=" << iterations << "\n";
    auto built = cases(rows, columns);
    if (chain != 0U) {
      built.clear();
      built.push_back(chained(rows, columns, chain));
    }
    for (auto &value : built) {
      dif::runtime::TensorMap inputs;
      for (const auto &tensor : value.program.tensors)
        if (tensor.has_role(TensorRole::Input))
          inputs.emplace(tensor.id, filled(tensor.dtype, tensor.dims));
      std::cout << "  " << value.name << " ..." << std::flush;
      dif::runtime::RunOptions options;
      options.warmups = 5U;
      options.iterations = static_cast<std::uint32_t>(iterations);
      const auto prepared = executor->prepare(value.program, inputs, options);
      const auto result = prepared->run(inputs, options);
      const double milliseconds = result.mean_milliseconds;
      const double bandwidth =
          static_cast<double>(value.traffic) / (milliseconds * 1.0e-3) / 1.0e9;
      std::cout << "\r  " << std::left << std::setw(18) << value.name
                << " mean_ms=" << std::setw(10) << milliseconds
                << " traffic_mb="
                << static_cast<double>(value.traffic) / 1.0e6
                << " effective_gb_s=" << bandwidth << "\n";
    }
  } catch (const std::exception &error) {
    std::cerr << "difmicrobench: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
