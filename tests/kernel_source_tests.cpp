// Byte-identity gate for generated CUDA source.
//
// For a corpus of small programs (one per kernel emitter and attribute
// branch), the CUDA text produced by dif::compiler::emit_cuda must match the
// committed snapshot under perf/regress/fixtures/kernel-sources/ byte for
// byte. The snapshots were taken before kernel bodies moved out of the C++
// emitters into src/compiler/kernels/*.cu templates, so the gate proves the
// move changed nothing the GPU sees (the runtime keys its PTX cache by a hash
// of this text).
//
//   dif_kernel_source_tests FIXTURE_DIR            compare
//   dif_kernel_source_tests FIXTURE_DIR --update   rewrite the snapshots
#include "dif/compiler/compiler.hpp"
#include "dif/ir/ir.hpp"
#include "dif/support/error.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace dif::ir;

struct Case {
  std::string name;
  std::function<Program()> build;
};

Program elementwise(Opcode opcode, DType dtype,
                    std::vector<Attribute> attributes = {}) {
  Program program;
  program.tensors = {{1, dtype, TensorRole::Input, {4, 8}},
                     {2, dtype, TensorRole::Output, {4, 8}}};
  program.operations = {{1, opcode, {1}, {2}, std::move(attributes)}};
  return program;
}

Program dequantize_int4(bool outliers) {
  Program program;
  program.tensors = {{1, DType::I8, TensorRole::Constant, {4, 64}},
                     {2, DType::BF16, TensorRole::Constant, {4, 2}},
                     {3, DType::BF16, TensorRole::Output, {4, 128}}};
  std::vector<std::uint32_t> inputs{1, 2};
  if (outliers) {
    program.tensors.push_back({4, DType::I8, TensorRole::Constant, {4, 2}});
    program.tensors.push_back({5, DType::BF16, TensorRole::Constant, {4, 2}});
    inputs = {1, 2, 4, 5};
  }
  program.operations = {{1, Opcode::DequantizeInt4, inputs, {3},
                         {Attribute::u64(AttrKey::GroupSize, 64U)}}};
  return program;
}

Program dequantize_int5(bool column_scales) {
  Program program;
  program.tensors = {{1, DType::I8, TensorRole::Constant, {4, 80}},
                     {2, DType::BF16, TensorRole::Constant, {4, 2}},
                     {3, DType::BF16, TensorRole::Output, {4, 128}}};
  std::vector<std::uint32_t> inputs{1, 2};
  if (column_scales) {
    program.tensors.push_back({4, DType::BF16, TensorRole::Constant, {128}});
    inputs = {1, 2, 4};
  }
  program.operations = {{1, Opcode::DequantizeInt5, inputs, {3},
                         {Attribute::u64(AttrKey::GroupSize, 64U)}}};
  return program;
}

Program rms_norm(std::uint64_t implementation, std::uint64_t columns) {
  Program program;
  program.tensors = {{1, DType::BF16, TensorRole::Input, {4, columns}},
                     {2, DType::BF16, TensorRole::Constant, {columns}},
                     {3, DType::BF16, TensorRole::Output, {4, columns}}};
  std::vector<Attribute> attributes{Attribute::f64(AttrKey::Epsilon, 1.0e-6)};
  if (implementation != 1U) {
    attributes.push_back(
        Attribute::u64(AttrKey::Implementation, implementation));
    attributes.push_back(Attribute::u64(AttrKey::BlockSize, 128U));
  }
  program.operations = {{1, Opcode::RmsNorm, {1, 2}, {3}, attributes}};
  return program;
}

Program quantize_int8_rows(Int8RowQuantization implementation,
                           std::uint64_t width, bool residual2,
                           bool dynamic_clip) {
  Program program;
  program.tensors = {{1, DType::BF16, TensorRole::Input, {2, width}},
                     {2, DType::I8, TensorRole::Output, {2, width}},
                     {3, DType::F32, TensorRole::Output, {2}}};
  std::vector<std::uint32_t> inputs{1};
  std::vector<std::uint32_t> outputs{2, 3};
  if (dynamic_clip) {
    program.tensors.push_back({4, DType::F32, TensorRole::Input, {1}});
    inputs.push_back(4);
  }
  if (residual2) {
    program.tensors.push_back({5, DType::I8, TensorRole::Output, {2, width}});
    program.tensors.push_back({6, DType::F32, TensorRole::Output, {2}});
    outputs = {2, 3, 5, 6};
  }
  program.operations = {
      {1, Opcode::QuantizeInt8Rows, inputs, outputs,
       {Attribute::u64(AttrKey::BlockSize, 256U),
        Attribute::u64(AttrKey::Implementation,
                       static_cast<std::uint64_t>(implementation))}}};
  return program;
}

std::vector<Case> corpus() {
  using Q = Int8RowQuantization;
  return {
      {"silu_f32", [] { return elementwise(Opcode::SiLU, DType::F32); }},
      {"silu_bf16", [] { return elementwise(Opcode::SiLU, DType::BF16); }},
      {"sigmoid_bf16", [] { return elementwise(Opcode::Sigmoid, DType::BF16); }},
      {"gelu_tanh",
       [] {
         return elementwise(Opcode::Gelu, DType::BF16,
                            {Attribute::u64(AttrKey::Approximation,
                                            static_cast<std::uint64_t>(
                                                GeluApproximation::Tanh))});
       }},
      {"gelu_exacterf",
       [] {
         return elementwise(Opcode::Gelu, DType::BF16,
                            {Attribute::u64(AttrKey::Approximation,
                                            static_cast<std::uint64_t>(
                                                GeluApproximation::ExactErf))});
       }},
      {"dequantize_int4_g64", [] { return dequantize_int4(false); }},
      {"dequantize_int4_outliers", [] { return dequantize_int4(true); }},
      {"dequantize_int5_g64", [] { return dequantize_int5(false); }},
      {"dequantize_int5_column_scales", [] { return dequantize_int5(true); }},
      {"rms_norm_default_128", [] { return rms_norm(1U, 128U); }},
      {"rms_norm_default_768", [] { return rms_norm(1U, 768U); }},
      {"rms_norm_implementation2_128", [] { return rms_norm(2U, 128U); }},
      {"quantize_int8_rows_direct",
       [] { return quantize_int8_rows(Q::Direct, 512U, false, false); }},
      {"quantize_int8_rows_direct_dynamic_clip",
       [] { return quantize_int8_rows(Q::Direct, 512U, false, true); }},
      {"quantize_int8_rows_direct_residual2",
       [] { return quantize_int8_rows(Q::Direct, 512U, true, false); }},
      {"quantize_int8_rows_h256_convrot",
       [] { return quantize_int8_rows(Q::H256ConvRot, 512U, false, false); }},
      {"quantize_int8_rows_h256_signed_convrot",
       [] { return quantize_int8_rows(Q::H256SignedConvRot, 512U, false, false); }},
      {"quantize_int8_rows_h4096_signed_convrot",
       [] { return quantize_int8_rows(Q::H4096SignedConvRot, 4096U, false, false); }},
      {"quantize_int8_rows_h256_f32_convrot",
       [] { return quantize_int8_rows(Q::H256F32ConvRot, 512U, false, false); }},
      {"quantize_int8_rows_h256_f32_signed_convrot",
       [] { return quantize_int8_rows(Q::H256F32SignedConvRot, 512U, false, false); }},
      {"quantize_int8_rows_h4096_f32_signed_convrot",
       [] { return quantize_int8_rows(Q::H4096F32SignedConvRot, 4096U, false, false); }},
      {"quantize_int8_rows_h4096_f32_signed_convrot_residual2",
       [] { return quantize_int8_rows(Q::H4096F32SignedConvRot, 4096U, true, false); }},
      {"quantize_int8_rows_h256_f32_sylvester_convrot",
       [] { return quantize_int8_rows(Q::H256F32SylvesterConvRot, 512U, false, false); }},
  };
}

std::string read_file(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

void report_first_difference(const std::string &expected,
                             const std::string &actual) {
  std::size_t line = 1, column = 1, index = 0;
  const auto limit = std::min(expected.size(), actual.size());
  while (index < limit && expected[index] == actual[index]) {
    if (expected[index] == '\n') {
      ++line;
      column = 1;
    } else {
      ++column;
    }
    ++index;
  }
  std::cerr << "  first difference at line " << line << " column " << column
            << " (expected " << expected.size() << " bytes, got "
            << actual.size() << ")\n  expected: "
            << expected.substr(index, 80) << "\n  actual:   "
            << actual.substr(index, 80) << '\n';
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: dif_kernel_source_tests FIXTURE_DIR [--update]\n";
    return 2;
  }
  const std::filesystem::path directory = argv[1];
  const bool update = argc > 2 && std::string(argv[2]) == "--update";
  if (update)
    std::filesystem::create_directories(directory);
  int failures = 0;
  std::size_t checked = 0;
  for (const auto &entry : corpus()) {
    std::string source;
    try {
      source = dif::compiler::emit_cuda(entry.build()).source;
    } catch (const dif::Error &error) {
      std::cerr << "FAIL: " << entry.name << ": emit_cuda threw: "
                << error.what() << '\n';
      ++failures;
      continue;
    }
    const auto path = directory / (entry.name + ".cu");
    if (update) {
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      out << source;
      std::cout << "wrote " << path.string() << " (" << source.size()
                << " bytes)\n";
      continue;
    }
    if (!std::filesystem::exists(path)) {
      std::cerr << "FAIL: " << entry.name << ": no snapshot at " << path
                << " (run with --update to create it)\n";
      ++failures;
      continue;
    }
    const auto expected = read_file(path);
    if (expected != source) {
      std::cerr << "FAIL: " << entry.name
                << ": generated CUDA source differs from the snapshot\n";
      report_first_difference(expected, source);
      ++failures;
      continue;
    }
    ++checked;
  }
  if (update) {
    std::cout << "updated " << corpus().size() << " kernel source snapshots\n";
    return 0;
  }
  if (failures != 0) {
    std::cerr << failures << " kernel source snapshot(s) failed\n";
    return 1;
  }
  std::cout << "PASS: " << checked
            << " generated CUDA sources are byte-identical to their snapshots\n";
  return 0;
}
