#include "dif/runtime/scalar.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/weights/safetensors.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string>

namespace {

double number(const char *text, const char *label) {
  char *end = nullptr;
  const auto value = std::strtod(text, &end);
  if (!end || *end != '\0' || !std::isfinite(value))
    dif::fail(std::string("invalid ") + label);
  return value;
}

void usage() {
  std::cerr << "usage: difcompare REFERENCE ACTUAL"
               " [--min-cos N] [--max-rel-l2 N]"
               " [--min-norm-ratio N] [--max-norm-ratio N]"
               " [--max-abs N] [--flatten]\n"
               "REFERENCE/ACTUAL may be FILE.diftensor or "
               "FILE.safetensors::TENSOR_NAME\n";
}

dif::runtime::Tensor load_tensor_spec(std::string_view spec) {
  const auto separator = spec.find("::");
  if (separator == std::string_view::npos)
    return dif::runtime::read_tensor(std::filesystem::path(spec));
  if (separator == 0U || separator + 2U == spec.size())
    dif::fail("invalid SafeTensors tensor specification");
  const auto path = std::filesystem::path(spec.substr(0U, separator));
  const auto name = spec.substr(separator + 2U);
  return dif::weights::map_safetensor(dif::weights::read_safetensors(path),
                                      name);
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 3) {
      usage();
      return 2;
    }
    double minimum_cosine = 0.999;
    double maximum_relative_l2 = 0.02;
    double minimum_norm_ratio = 0.98;
    double maximum_norm_ratio = 1.02;
    double maximum_absolute_bar = std::numeric_limits<double>::infinity();
    bool flatten = false;
    for (int argument = 3; argument < argc; ++argument) {
      const std::string option = argv[argument];
      if (option == "--min-cos" && argument + 1 < argc)
        minimum_cosine = number(argv[++argument], "minimum cosine");
      else if (option == "--max-rel-l2" && argument + 1 < argc)
        maximum_relative_l2 =
            number(argv[++argument], "maximum relative L2");
      else if (option == "--min-norm-ratio" && argument + 1 < argc)
        minimum_norm_ratio =
            number(argv[++argument], "minimum norm ratio");
      else if (option == "--max-norm-ratio" && argument + 1 < argc)
        maximum_norm_ratio =
            number(argv[++argument], "maximum norm ratio");
      else if (option == "--max-abs" && argument + 1 < argc)
        maximum_absolute_bar =
            number(argv[++argument], "maximum absolute error");
      else if (option == "--flatten")
        flatten = true;
      else {
        usage();
        return 2;
      }
    }
    if (minimum_cosine < -1.0 || minimum_cosine > 1.0 ||
        maximum_relative_l2 < 0.0 || minimum_norm_ratio < 0.0 ||
        minimum_norm_ratio > maximum_norm_ratio ||
        maximum_absolute_bar < 0.0)
      dif::fail("invalid comparison admission bars");

    const auto reference = load_tensor_spec(argv[1]);
    const auto actual = load_tensor_spec(argv[2]);
    if (!dif::runtime::is_float_dtype(reference.dtype) ||
        !dif::runtime::is_float_dtype(actual.dtype))
      dif::fail("difcompare requires floating-point tensors");
    if (reference.dims != actual.dims && !flatten)
      dif::fail("reference and actual tensor shapes differ");
    if (reference.element_count() != actual.element_count())
      dif::fail("reference and actual tensor element counts differ");
    if (reference.element_count() == 0U)
      dif::fail("difcompare requires at least one tensor element");

    long double dot = 0.0L;
    long double reference_squared = 0.0L;
    long double actual_squared = 0.0L;
    long double error_squared = 0.0L;
    long double absolute_sum = 0.0L;
    double maximum_absolute = 0.0;
    std::uint64_t nonfinite = 0U;
    std::optional<std::uint64_t> exact_mismatches;
    if (reference.dtype == actual.dtype) {
      exact_mismatches = 0U;
      const auto element_bytes =
          reference.byte_size() / reference.element_count();
      for (std::uint64_t index = 0; index < reference.element_count(); ++index) {
        if (std::memcmp(reference.data() + index * element_bytes,
                        actual.data() + index * element_bytes,
                        static_cast<std::size_t>(element_bytes)) != 0)
          ++*exact_mismatches;
      }
    }
    for (std::uint64_t index = 0; index < reference.element_count(); ++index) {
      const auto expected =
          static_cast<double>(dif::runtime::load_float(reference, index));
      const auto observed =
          static_cast<double>(dif::runtime::load_float(actual, index));
      if (!std::isfinite(expected) || !std::isfinite(observed)) {
        ++nonfinite;
        continue;
      }
      const auto error = observed - expected;
      dot += static_cast<long double>(expected) * observed;
      reference_squared += static_cast<long double>(expected) * expected;
      actual_squared += static_cast<long double>(observed) * observed;
      error_squared += static_cast<long double>(error) * error;
      const auto absolute = std::abs(error);
      absolute_sum += absolute;
      maximum_absolute = std::max(maximum_absolute, absolute);
    }
    const auto denominator = std::sqrt(reference_squared * actual_squared);
    const auto cosine = denominator == 0.0L
                            ? (reference_squared == actual_squared ? 1.0 : 0.0)
                            : static_cast<double>(dot / denominator);
    const auto relative_l2 =
        reference_squared == 0.0L
            ? (error_squared == 0.0L
                   ? 0.0
                   : std::numeric_limits<double>::infinity())
            : static_cast<double>(std::sqrt(error_squared / reference_squared));
    const auto norm_ratio =
        reference_squared == 0.0L
            ? (actual_squared == 0.0L
                   ? 1.0
                   : std::numeric_limits<double>::infinity())
            : static_cast<double>(std::sqrt(actual_squared / reference_squared));
    const auto mean_absolute =
        static_cast<double>(absolute_sum /
                            static_cast<long double>(reference.element_count()));
    const auto accepted =
        nonfinite == 0U && cosine >= minimum_cosine &&
        relative_l2 <= maximum_relative_l2 &&
        norm_ratio >= minimum_norm_ratio &&
        norm_ratio <= maximum_norm_ratio &&
        maximum_absolute <= maximum_absolute_bar;
    std::cout << std::setprecision(17)
              << "GATE " << (accepted ? "PASS" : "FAIL")
              << " reference=" << argv[1] << " actual=" << argv[2]
              << " elements=" << reference.element_count()
              << " max_abs=" << maximum_absolute
              << " mean_abs=" << mean_absolute
              << " rel_l2=" << relative_l2 << " cosine=" << cosine
              << " norm_ratio=" << norm_ratio
              << " nonfinite=" << nonfinite << " exact_mismatches=";
    if (exact_mismatches)
      std::cout << *exact_mismatches;
    else
      std::cout << "n/a";
    std::cout << " bars={min_cos:"
              << minimum_cosine << ",max_rel_l2:" << maximum_relative_l2
              << ",min_norm_ratio:" << minimum_norm_ratio
              << ",max_norm_ratio:" << maximum_norm_ratio
              << ",max_abs:" << maximum_absolute_bar << "}\n";
    return accepted ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "difcompare: " << error.what() << "\n";
    return 1;
  }
}
