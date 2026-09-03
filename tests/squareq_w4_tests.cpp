// Gate for the SquareQ W4 rewrite (dif/frontend/squareq_w4.hpp) against the
// byte-level SquareQ v3 oracle fixture (perf/regress/fixtures/squareq-w4-tiny,
// emitted by tools/export_squareq_w4_fixture.py): the rewritten program must
// reproduce y = x W_hat^T on the CPU executor and (when available) the CUDA
// executor, the two executors must agree, the CUDA run must be repeat-bit-
// exact, and every disagreement with the slab must fail closed.

#include "dif/frontend/squareq_w4.hpp"
#include "dif/ir/ir.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/weights/safetensors.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;
void expect(bool condition, const std::string &label) {
  if (condition)
    return;
  ++failures;
  std::cerr << "FAIL: " << label << "\n";
}

float bf16_value(std::uint16_t bits) {
  const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16U;
  float value = 0.0f;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

std::vector<double> as_doubles(const dif::runtime::Tensor &tensor) {
  std::vector<double> out(tensor.element_count());
  if (tensor.dtype == dif::ir::DType::BF16) {
    for (std::size_t i = 0; i < out.size(); ++i) {
      std::uint16_t bits = 0;
      std::memcpy(&bits, tensor.data() + i * 2U, 2U);
      out[i] = bf16_value(bits);
    }
  } else if (tensor.dtype == dif::ir::DType::F32) {
    for (std::size_t i = 0; i < out.size(); ++i) {
      float value = 0.0f;
      std::memcpy(&value, tensor.data() + i * 4U, 4U);
      out[i] = value;
    }
  } else {
    dif::fail("unexpected fixture dtype");
  }
  return out;
}

struct Comparison {
  double cosine{};
  double max_abs{};
  double max_ref{};
  bool finite{true};
};

Comparison compare(const std::vector<double> &a, const std::vector<double> &b) {
  Comparison c;
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(b[i]))
      c.finite = false;
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
    c.max_abs = std::max(c.max_abs, std::fabs(a[i] - b[i]));
    c.max_ref = std::max(c.max_ref, std::fabs(a[i]));
  }
  c.cosine = (na > 0.0 && nb > 0.0) ? dot / std::sqrt(na * nb) : 0.0;
  return c;
}

// Linear(x [rows,in], W [out,in]) -> y [rows,out]; W is the checkpoint weight.
struct Fixture {
  dif::ir::Program program;
  dif::runtime::TensorMap bindings;
  std::vector<std::uint32_t> ids{2U};
  std::vector<std::string> names{"blocks.0.linear.weight"};
};

Fixture make_program(const dif::runtime::Tensor &x, std::uint64_t out_features,
                     std::uint64_t in_features) {
  using namespace dif::ir;
  Fixture f;
  f.program.tensors = {
      {1U, DType::BF16, TensorRole::Input, {x.dims[0], in_features}},
      {2U, DType::BF16, TensorRole::Input, {out_features, in_features}},
      {3U, DType::BF16, TensorRole::Output, {x.dims[0], out_features}},
  };
  f.program.operations = {{1U, Opcode::Linear, {1U, 2U}, {3U}, {}}};
  verify(f.program);
  f.bindings.emplace(1U, x);
  // The BF16 weight the slab replaces: zeros, so any use of it would show.
  f.bindings.emplace(2U, dif::runtime::Tensor{DType::BF16, {out_features, in_features},
                                              std::vector<std::uint8_t>(out_features * in_features * 2U, 0U)});
  return f;
}

bool refused(const std::filesystem::path &slab, const dif::runtime::Tensor &x,
             std::uint64_t out_features, std::uint64_t in_features, const char *needle) {
  auto f = make_program(x, out_features, in_features);
  try {
    (void)dif::frontend::rewrite_linear_weights_squareq_w4(f.program, f.bindings, f.ids,
                                                           f.names, slab);
  } catch (const dif::Error &error) {
    return std::string(error.what()).find(needle) != std::string::npos;
  }
  return false;
}

void write_text(const std::filesystem::path &path, const std::string &text) {
  std::ofstream out(path, std::ios::binary);
  out << text;
}

std::string read_text(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: dif_squareq_w4_tests FIXTURE_DIR\n";
    return 2;
  }
  const std::filesystem::path fixture = argv[1];
  const auto oracle = dif::weights::read_safetensors(fixture / "oracle.safetensors");
  const auto w_hat = dif::weights::map_safetensor(oracle, "w_hat");
  const auto x = dif::weights::map_safetensor(oracle, "x");
  const auto y_ref = dif::weights::map_safetensor(oracle, "y_ref");
  const auto out_features = w_hat.dims[0];
  const auto in_features = w_hat.dims[1];

  // 1. Rewrite + receipt + program shape.
  auto f = make_program(x, out_features, in_features);
  const auto receipt = dif::frontend::rewrite_linear_weights_squareq_w4(
      f.program, f.bindings, f.ids, f.names, fixture);
  expect(receipt.linear_count == 1U && receipt.format == "squareq_w4_v1" && receipt.rank == 32U,
         "receipt names one squareq_w4_v1 rank-32 Linear");
  expect(receipt.bf16_bytes_replaced == out_features * in_features * 2U,
         "receipt counts the replaced BF16 bytes");
  expect(receipt.quantized_bytes < receipt.bf16_bytes_replaced / 2U,
         "slab bytes are under half of the BF16 bytes");
  expect(!f.bindings.contains(2U), "the BF16 weight binding is dropped");
  expect(f.program.operations.size() == 7U, "six reconstruction ops precede the Linear");

  // 2. CPU executor reproduces the oracle.
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  const auto cpu = dif::runtime::make_cpu_executor()->run(f.program, f.bindings, options);
  const auto ref = as_doubles(y_ref);
  const auto cpu_y = as_doubles(cpu.outputs.at(3U));
  const auto cpu_cmp = compare(ref, cpu_y);
  std::cout << "SQUAREQ_W4 cpu cosine=" << cpu_cmp.cosine << " max_abs=" << cpu_cmp.max_abs
            << " max_ref=" << cpu_cmp.max_ref << "\n";
  expect(cpu_cmp.finite && cpu_cmp.cosine >= 0.9999 && cpu_cmp.max_abs <= 0.02 * cpu_cmp.max_ref,
         "CPU output matches y = x W_hat^T");

  // 3. Identity input exposes W_hat^T itself.
  {
    std::vector<std::uint8_t> identity(in_features * in_features * 2U, 0U);
    const std::uint16_t one = 0x3F80U;
    for (std::uint64_t i = 0; i < in_features; ++i)
      std::memcpy(identity.data() + (i * in_features + i) * 2U, &one, 2U);
    dif::runtime::Tensor eye{dif::ir::DType::BF16, {in_features, in_features}, std::move(identity)};
    auto g = make_program(eye, out_features, in_features);
    (void)dif::frontend::rewrite_linear_weights_squareq_w4(g.program, g.bindings, g.ids, g.names, fixture);
    const auto run = dif::runtime::make_cpu_executor()->run(g.program, g.bindings, options);
    const auto got = as_doubles(run.outputs.at(3U)); // [in, out] = W_hat^T
    const auto want = as_doubles(w_hat);              // [out, in]
    std::vector<double> want_t(got.size());
    for (std::uint64_t o = 0; o < out_features; ++o)
      for (std::uint64_t k = 0; k < in_features; ++k)
        want_t[k * out_features + o] = want[o * in_features + k];
    const auto c = compare(want_t, got);
    std::cout << "SQUAREQ_W4 reconstructed-weight cosine=" << c.cosine << " max_abs=" << c.max_abs
              << " max_ref=" << c.max_ref << "\n";
    expect(c.cosine >= 0.99999 && c.max_abs <= 0.01 * c.max_ref,
           "reconstructed weight matches the SquareQ oracle W_hat");
  }

  // 4. CUDA executor agrees and is repeat-bit-exact.
  if (dif::runtime::cuda_available()) {
    auto h = make_program(x, out_features, in_features);
    (void)dif::frontend::rewrite_linear_weights_squareq_w4(h.program, h.bindings, h.ids, h.names, fixture);
    const auto once = dif::runtime::make_cuda_executor()->run(h.program, h.bindings, options);
    const auto twice = dif::runtime::make_cuda_executor()->run(h.program, h.bindings, options);
    const auto gpu_y = as_doubles(once.outputs.at(3U));
    const auto gpu_cmp = compare(ref, gpu_y);
    const auto cross = compare(cpu_y, gpu_y);
    std::cout << "SQUAREQ_W4 cuda cosine=" << gpu_cmp.cosine << " max_abs=" << gpu_cmp.max_abs
              << " cpu_vs_cuda_max_abs=" << cross.max_abs << "\n";
    expect(gpu_cmp.finite && gpu_cmp.cosine >= 0.9999 && gpu_cmp.max_abs <= 0.02 * gpu_cmp.max_ref,
           "CUDA output matches y = x W_hat^T");
    expect(cross.max_abs <= 0.02 * cpu_cmp.max_ref, "CPU and CUDA agree");
    expect(once.outputs.at(3U).bytes == twice.outputs.at(3U).bytes, "CUDA run is repeat-bit-exact");
  } else {
    std::cout << "CUDA unavailable; CUDA half of the gate skipped\n";
  }

  // 4b. INT8 compute mode: rotated low-rank folded into the residual,
  //     device row-quantized weight, H256-rotated INT8 activations. Looser
  //     bar (INT8 activation + weight quantization), CPU vs CUDA agree,
  //     CUDA repeat-bit-exact.
  {
    auto m = make_program(x, out_features, in_features);
    const auto r8 = dif::frontend::rewrite_linear_weights_squareq_w4(
        m.program, m.bindings, m.ids, m.names, fixture,
        dif::frontend::SquareQW4Mode::Int8Compute);
    expect(r8.mode == dif::frontend::SquareQW4Mode::Int8Compute && r8.linear_count == 1U,
           "int8 mode rewrites the Linear");
    const auto has_int8 = std::any_of(
        m.program.operations.begin(), m.program.operations.end(),
        [](const dif::ir::Operation &op) {
          return op.opcode == dif::ir::Opcode::LinearInt8Scaled;
        });
    expect(has_int8, "int8 mode emits a scaled INT8 Linear");
    const auto cpu8 = dif::runtime::make_cpu_executor()->run(m.program, m.bindings, options);
    const auto cpu8_y = as_doubles(cpu8.outputs.at(3U));
    const auto c8 = compare(ref, cpu8_y);
    std::cout << "SQUAREQ_W4 int8-compute cpu cosine=" << c8.cosine << " max_abs=" << c8.max_abs
              << " max_ref=" << c8.max_ref << "\n";
    expect(c8.finite && c8.cosine >= 0.999 && c8.max_abs <= 0.06 * c8.max_ref,
           "int8 compute output tracks y = x W_hat^T within INT8 tolerance");
    if (dif::runtime::cuda_available()) {
      auto n = make_program(x, out_features, in_features);
      (void)dif::frontend::rewrite_linear_weights_squareq_w4(
          n.program, n.bindings, n.ids, n.names, fixture,
          dif::frontend::SquareQW4Mode::Int8Compute);
      const auto once = dif::runtime::make_cuda_executor()->run(n.program, n.bindings, options);
      const auto twice = dif::runtime::make_cuda_executor()->run(n.program, n.bindings, options);
      const auto g8 = compare(ref, as_doubles(once.outputs.at(3U)));
      const auto cross = compare(cpu8_y, as_doubles(once.outputs.at(3U)));
      std::cout << "SQUAREQ_W4 int8-compute cuda cosine=" << g8.cosine << " max_abs=" << g8.max_abs
                << " cpu_vs_cuda cosine=" << cross.cosine << " max_abs=" << cross.max_abs << "\n";
      expect(g8.finite && g8.cosine >= 0.999 && g8.max_abs <= 0.06 * g8.max_ref,
             "int8 compute CUDA output tracks y = x W_hat^T within INT8 tolerance");
      expect(cross.cosine >= 0.9999, "int8 compute CPU and CUDA agree");
      expect(once.outputs.at(3U).bytes == twice.outputs.at(3U).bytes,
             "int8 compute CUDA run is repeat-bit-exact");
    }
  }

  // 5. Fail closed: wrong format tag, wrong geometry, missing slab tensor,
  //    input width not a multiple of 256, key not consumed by a Linear.
  const auto scratch = std::filesystem::temp_directory_path() / "dif-squareq-w4-gate";
  std::filesystem::remove_all(scratch);
  std::filesystem::create_directories(scratch);
  for (const auto &entry : std::filesystem::directory_iterator(fixture))
    std::filesystem::copy(entry.path(), scratch / entry.path().filename());
  const auto plan_text = read_text(fixture / "squareq-plan.json");
  write_text(scratch / "squareq-plan.json",
             std::string(plan_text).replace(plan_text.find("squareq_w4_v1"), 13U, "squareq_w4_v9"));
  expect(refused(scratch, x, out_features, in_features, "format is not"), "wrong format tag is refused");
  write_text(scratch / "squareq-plan.json", plan_text);
  expect(refused(scratch, x, out_features + 64U, in_features, "geometry disagrees"),
         "plan/program geometry disagreement is refused");
  {
    auto wrong_width = plan_text;
    const auto pos = wrong_width.find("\"in\": " + std::to_string(in_features));
    wrong_width.replace(pos, std::string("\"in\": " + std::to_string(in_features)).size(),
                        "\"in\": " + std::to_string(in_features + 64U));
    write_text(scratch / "squareq-plan.json", wrong_width);
    expect(refused(scratch, x, out_features, in_features + 64U, "not divisible by 256"),
           "input width not divisible by 256 is refused");
    write_text(scratch / "squareq-plan.json", plan_text);
  }
  {
    auto index_text = read_text(fixture / "model.safetensors.index.json");
    index_text.replace(index_text.find("lora_up"), 7U, "lora_xx");
    write_text(scratch / "model.safetensors.index.json", index_text);
    expect(refused(scratch, x, out_features, in_features, "index lacks"),
           "missing slab tensor is refused");
  }
  std::filesystem::remove_all(scratch);

  if (failures != 0) {
    std::cerr << failures << " SquareQ W4 gate failure(s)\n";
    return 1;
  }
  std::cout << "SquareQ W4 gate passed\n";
  return 0;
}
