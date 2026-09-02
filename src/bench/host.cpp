#include "dif/bench/host.hpp"

#include "dif/support/error.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace dif::bench {

FileResidency measure_residency(const std::filesystem::path &path) {
  FileResidency result;
  result.path = path;
  struct stat info {};
  if (::stat(path.c_str(), &info) != 0 || !S_ISREG(info.st_mode))
    return result;
  result.exists = true;
  result.bytes = static_cast<std::uint64_t>(info.st_size);
  if (result.bytes == 0U)
    return result;
  const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor < 0)
    return result;
  void *address = ::mmap(nullptr, result.bytes, PROT_READ, MAP_SHARED,
                         descriptor, 0);
  ::close(descriptor);
  if (address == MAP_FAILED)
    return result;
  const auto page = static_cast<std::uint64_t>(::sysconf(_SC_PAGESIZE));
  const auto pages = (result.bytes + page - 1U) / page;
  std::vector<unsigned char> vector(pages);
  if (::mincore(address, result.bytes, vector.data()) == 0) {
    std::uint64_t resident_pages = 0U;
    for (const auto flag : vector)
      resident_pages += (flag & 1U) != 0U ? 1U : 0U;
    result.resident_bytes = std::min(result.bytes, resident_pages * page);
  }
  ::munmap(address, result.bytes);
  return result;
}

void drop_file_cache(const std::filesystem::path &path) {
  const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor < 0)
    return;
  ::fdatasync(descriptor);
  ::posix_fadvise(descriptor, 0, 0, POSIX_FADV_DONTNEED);
  ::close(descriptor);
}

ResidencySummary summarize_residency(const std::vector<FileResidency> &files) {
  ResidencySummary summary;
  bool any = false;
  for (const auto &file : files) {
    if (!file.exists)
      continue;
    any = true;
    summary.total_bytes += file.bytes;
    summary.resident_bytes += file.resident_bytes;
  }
  if (!any || summary.total_bytes == 0U) {
    summary.condition = "unknown";
    return summary;
  }
  summary.resident_fraction = static_cast<double>(summary.resident_bytes) /
                              static_cast<double>(summary.total_bytes);
  if (summary.resident_fraction < 0.05)
    summary.condition = "cold";
  else if (summary.resident_fraction > 0.95)
    summary.condition = "warm";
  else
    summary.condition = "mixed";
  return summary;
}

std::string hostname() {
  char buffer[256];
  if (::gethostname(buffer, sizeof(buffer)) != 0)
    return "unknown";
  buffer[sizeof(buffer) - 1U] = '\0';
  return buffer;
}

} // namespace dif::bench
