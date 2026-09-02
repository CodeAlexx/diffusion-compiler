#pragma once

// Facts about the saved output that end a benchmark boundary. Native
// parsing covers PNG headers and MP4 box structure; ffprobe, when present,
// is embedded as an additional diagnostic rather than relied on.

#include "dif/telemetry/document.hpp"

#include <filesystem>
#include <string>

namespace dif::bench {

struct ArtifactFacts {
  bool exists{};
  std::uint64_t bytes{};
  std::string sha256;
  std::string format; // "png", "mp4", "unknown"
  // PNG
  std::uint32_t width{};
  std::uint32_t height{};
  std::uint32_t bit_depth{};
  std::uint32_t color_type{};
  // MP4
  std::string major_brand;
  std::uint64_t timescale{};
  std::uint64_t duration_units{};
  double duration_seconds{};
  std::uint32_t track_count{};
};

ArtifactFacts inspect_artifact(const std::filesystem::path &path);

// True when the facts satisfy the recipe's output kind: a decodable PNG
// header for image, an MP4 with a movie header for video.
bool artifact_matches_kind(const ArtifactFacts &facts,
                           const std::string &output_kind);

telemetry::Object artifact_section(const std::filesystem::path &path,
                                   const ArtifactFacts &facts,
                                   bool run_ffprobe);

} // namespace dif::bench
