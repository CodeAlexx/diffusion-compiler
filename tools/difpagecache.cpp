// difpagecache — return the page cache held by large streamed model files.
//
// An H3 evaluation streams the whole denoiser trunk from disk on every step
// (about 69 GB of h2d traffic on the 3090 Ti box). Those pages stay resident in
// the page cache afterwards, so a few consecutive renders leave the host with
// almost no free memory, and the runtime guard then cancels the NEXT job on
// memory-pressure stalls even though the child itself peaked under 5 GB — the
// pressure was the cache, not the job.
//
// posix_fadvise(POSIX_FADV_DONTNEED) drops those clean pages without root and
// without touching the files. It is advisory: dirty pages are not discarded,
// and anything still mapped by a live process stays. Re-reading is a cold read
// again, which is exactly what a streamed runtime does anyway.
//
// This replaces a Python script that did the same walk. Shelling out to
// python3 from the runner had no place in this stack, and the indirection cost
// a real bug: the interpreter resolved a relative script path against the job's
// output directory instead of the repository.

#include "dif/support/error.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace {

constexpr std::uint64_t mib = 1024ULL * 1024ULL;

// Weight containers only. Logs, manifests and tensors small enough to be free
// to reread are left alone.
bool streamed_weight_file(const std::filesystem::path &path) {
  static const char *suffixes[] = {".safetensors", ".difbind", ".diftensor",
                                   ".gguf", ".bin", ".pt"};
  const auto extension = path.extension().string();
  for (const auto *candidate : suffixes)
    if (extension == candidate)
      return true;
  return false;
}

struct Released {
  std::uint64_t files{0};
  std::uint64_t bytes{0};
};

void release_one(const std::filesystem::path &path, std::uint64_t minimum_bytes,
                 Released &released) {
  std::error_code code;
  const auto size = std::filesystem::file_size(path, code);
  if (code || size < minimum_bytes)
    return;
  const int descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor < 0)
    return;
  // Advisory and best effort: a failure here costs nothing but the hint.
  if (::posix_fadvise(descriptor, 0, 0, POSIX_FADV_DONTNEED) == 0) {
    released.files += 1;
    released.bytes += size;
  }
  ::close(descriptor);
}

void release_tree(const std::filesystem::path &root,
                  std::uint64_t minimum_bytes, Released &released) {
  std::error_code code;
  if (std::filesystem::is_regular_file(root, code)) {
    if (streamed_weight_file(root))
      release_one(root, minimum_bytes, released);
    return;
  }
  if (!std::filesystem::is_directory(root, code))
    return;
  std::filesystem::recursive_directory_iterator walk(
      root, std::filesystem::directory_options::skip_permission_denied, code);
  if (code)
    return;
  for (const auto &entry : walk) {
    std::error_code entry_code;
    if (!entry.is_regular_file(entry_code) || entry_code)
      continue;
    if (streamed_weight_file(entry.path()))
      release_one(entry.path(), minimum_bytes, released);
  }
}

} // namespace

int main(int argc, char **argv) {
  std::uint64_t minimum_mib = 64;
  std::vector<std::filesystem::path> roots;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--min-mib" && index + 1 < argc) {
      char *end = nullptr;
      const auto value = std::strtoull(argv[++index], &end, 10);
      if (end && *end == '\0')
        minimum_mib = value;
      continue;
    }
    if (argument == "--help" || argument == "-h") {
      std::cout << "usage: difpagecache [--min-mib N] PATH [PATH ...]\n"
                   "Drops the clean page cache behind large streamed weight "
                   "files. Advisory; never fails the caller.\n";
      return 0;
    }
    if (!argument.empty())
      roots.emplace_back(argument);
  }

  Released released;
  for (const auto &root : roots)
    release_tree(root, minimum_mib * mib, released);

  if (released.files != 0)
    std::cerr << "[page-cache] released " << released.files
              << " streamed file(s), "
              << static_cast<double>(released.bytes) / 1e9 << " GB advised\n";
  // A page-cache hint is never worth failing a finished render over.
  return 0;
}
