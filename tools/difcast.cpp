#include "dif/runtime/tensor.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/support/error.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  try {
    if (argc < 4) {
      std::cerr << "usage: difcast INPUT.diftensor OUTPUT.diftensor f32|bf16|f16 [DIM ...]\n";
      return 2;
    }
    const auto input = dif::runtime::read_tensor(argv[1]);
    const std::string target = argv[3];
    const auto target_dtype = target == "f32"    ? dif::ir::DType::F32
                              : target == "bf16" ? dif::ir::DType::BF16
                              : target == "f16"  ? dif::ir::DType::F16
                                                  : static_cast<dif::ir::DType>(0);
    if (!dif::runtime::is_float_dtype(target_dtype))
      dif::fail("target dtype must be f32, bf16, or f16");
    if (!dif::runtime::is_float_dtype(input.dtype))
      dif::fail("difcast input must be f32, bf16, or f16");
    std::vector<std::uint64_t> output_dims = input.dims;
    if (argc > 4) {
      output_dims.clear();
      for (int argument = 4; argument < argc; ++argument) {
        std::size_t consumed = 0;
        const auto dim = std::stoull(argv[argument], &consumed, 10);
        if (consumed != std::strlen(argv[argument]) || dim == 0U)
          dif::fail("reshape dimensions must be positive integers");
        output_dims.push_back(dim);
      }
    }

    dif::runtime::Tensor output{target_dtype, std::move(output_dims), {}};
    if (output.element_count() != input.element_count())
      dif::fail("reshape element count does not match input");
    output.bytes.resize(static_cast<std::size_t>(
        output.element_count() * dif::ir::dtype_size(target_dtype)));
    if (input.dtype == target_dtype) {
      output.bytes.assign(input.data(), input.data() + input.byte_size());
    } else {
      for (std::uint64_t i = 0; i < input.element_count(); ++i)
        dif::runtime::store_float(output, i, dif::runtime::load_float(input, i));
    }
    dif::runtime::write_tensor(output, argv[2]);
    std::cout << "CAST input=" << argv[1] << " output=" << argv[2]
              << " elements=" << output.element_count() << " dtype=" << target
              << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difcast: " << error.what() << "\n";
    return 1;
  }
}
