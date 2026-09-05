#include "dif/runtime/tensor.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {

struct TemporaryFile {
  int descriptor{-1};
  std::filesystem::path path;
  TemporaryFile() {
    auto pattern = (std::filesystem::temp_directory_path() /
                    "dif-tensor-io-XXXXXX").string();
    descriptor = mkstemp(pattern.data());
    if (descriptor < 0)
      throw std::runtime_error("cannot create residency fixture");
    path = pattern;
  }
  ~TemporaryFile() {
    if (descriptor >= 0)
      close(descriptor);
    std::error_code error;
    std::filesystem::remove(path, error);
  }
};

} // namespace

int main() {
  try {
    const auto page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0)
      throw std::runtime_error("cannot determine page size");
    const auto page = static_cast<std::size_t>(page_size);
    TemporaryFile file;
    for (const auto size : {std::size_t{1U}, page - 1U, page, page + 1U,
                           2U * page - 1U}) {
      if (ftruncate(file.descriptor, static_cast<off_t>(size)) != 0)
        throw std::runtime_error("cannot size residency fixture");
      const auto mapping = dif::runtime::map_readonly_file(file.path);
      // Touch every page, including the last partial page, before asking
      // mincore about it. The newly extended file contains only zero bytes.
      unsigned sum = 0U;
      for (std::size_t offset = 0U; offset < size; offset += page)
        sum += mapping->data()[offset];
      if (sum != 0U || mapping->resident_fraction(0U, size) != 1.0 ||
          mapping->resident_fraction(size - 1U, 1U) != 1.0)
        throw std::runtime_error("incorrect residency for file size " +
                                 std::to_string(size));
      if (mapping->resident_fraction(0U, 0U) != 0.0 ||
          mapping->resident_fraction(size, 1U) != 0.0)
        throw std::runtime_error("invalid residency range was not rejected");
    }
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
  std::cout << "tensor IO tests passed\n";
  return 0;
}
