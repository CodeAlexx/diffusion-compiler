#include "dif/bench/artifact.hpp"

#include "dif/bench/process.hpp"
#include "dif/support/json.hpp"
#include "dif/support/sha256.hpp"

#include <cstring>
#include <fstream>
#include <vector>

namespace dif::bench {
namespace {

std::uint32_t read_u32(const unsigned char *bytes) {
  return (static_cast<std::uint32_t>(bytes[0]) << 24U) |
         (static_cast<std::uint32_t>(bytes[1]) << 16U) |
         (static_cast<std::uint32_t>(bytes[2]) << 8U) |
         static_cast<std::uint32_t>(bytes[3]);
}

std::uint64_t read_u64(const unsigned char *bytes) {
  return (static_cast<std::uint64_t>(read_u32(bytes)) << 32U) |
         read_u32(bytes + 4);
}

void parse_png(std::ifstream &stream, ArtifactFacts &facts) {
  unsigned char header[33];
  stream.seekg(0);
  stream.read(reinterpret_cast<char *>(header), sizeof(header));
  if (stream.gcount() != static_cast<std::streamsize>(sizeof(header)))
    return;
  if (std::memcmp(header + 12, "IHDR", 4) != 0)
    return;
  facts.format = "png";
  facts.width = read_u32(header + 16);
  facts.height = read_u32(header + 20);
  facts.bit_depth = header[24];
  facts.color_type = header[25];
}

void parse_mp4_moov(std::ifstream &stream, std::uint64_t begin,
                    std::uint64_t end, ArtifactFacts &facts) {
  auto offset = begin;
  while (offset + 8U <= end) {
    unsigned char header[16];
    stream.seekg(static_cast<std::streamoff>(offset));
    stream.read(reinterpret_cast<char *>(header), 8);
    if (stream.gcount() != 8)
      return;
    std::uint64_t size = read_u32(header);
    std::uint64_t header_bytes = 8U;
    if (size == 1U) {
      stream.read(reinterpret_cast<char *>(header + 8), 8);
      if (stream.gcount() != 8)
        return;
      size = read_u64(header + 8);
      header_bytes = 16U;
    } else if (size == 0U) {
      size = end - offset;
    }
    if (size < header_bytes)
      return;
    const std::string type(reinterpret_cast<const char *>(header + 4), 4);
    if (type == "mvhd") {
      unsigned char body[32];
      stream.read(reinterpret_cast<char *>(body), sizeof(body));
      if (stream.gcount() == static_cast<std::streamsize>(sizeof(body))) {
        const auto version = body[0];
        if (version == 1U) {
          facts.timescale = read_u32(body + 20);
          facts.duration_units = read_u64(body + 24);
        } else {
          facts.timescale = read_u32(body + 12);
          facts.duration_units = read_u32(body + 16);
        }
        if (facts.timescale != 0U)
          facts.duration_seconds =
              static_cast<double>(facts.duration_units) /
              static_cast<double>(facts.timescale);
      }
    } else if (type == "trak") {
      ++facts.track_count;
    }
    offset += size;
  }
}

void parse_mp4(std::ifstream &stream, std::uint64_t file_size,
               ArtifactFacts &facts) {
  std::uint64_t offset = 0U;
  bool saw_ftyp = false;
  bool saw_moov = false;
  while (offset + 8U <= file_size) {
    unsigned char header[16];
    stream.seekg(static_cast<std::streamoff>(offset));
    stream.read(reinterpret_cast<char *>(header), 8);
    if (stream.gcount() != 8)
      break;
    std::uint64_t size = read_u32(header);
    std::uint64_t header_bytes = 8U;
    if (size == 1U) {
      stream.read(reinterpret_cast<char *>(header + 8), 8);
      if (stream.gcount() != 8)
        break;
      size = read_u64(header + 8);
      header_bytes = 16U;
    } else if (size == 0U) {
      size = file_size - offset;
    }
    if (size < header_bytes || offset + size > file_size)
      break;
    const std::string type(reinterpret_cast<const char *>(header + 4), 4);
    if (type == "ftyp") {
      unsigned char brand[4];
      stream.read(reinterpret_cast<char *>(brand), 4);
      if (stream.gcount() == 4) {
        facts.major_brand.assign(reinterpret_cast<const char *>(brand), 4);
        saw_ftyp = true;
      }
    } else if (type == "moov") {
      saw_moov = true;
      parse_mp4_moov(stream, offset + header_bytes, offset + size, facts);
    }
    offset += size;
  }
  if (saw_ftyp && saw_moov)
    facts.format = "mp4";
}

} // namespace

ArtifactFacts inspect_artifact(const std::filesystem::path &path) {
  ArtifactFacts facts;
  facts.format = "unknown";
  std::error_code error;
  if (!std::filesystem::is_regular_file(path, error))
    return facts;
  facts.exists = true;
  facts.bytes = std::filesystem::file_size(path, error);
  facts.sha256 = hex_digest(sha256_file(path));
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    return facts;
  unsigned char magic[12] = {};
  stream.read(reinterpret_cast<char *>(magic), sizeof(magic));
  if (stream.gcount() < 8)
    return facts;
  static const unsigned char png_signature[8] = {0x89, 'P',  'N',  'G',
                                                 0x0D, 0x0A, 0x1A, 0x0A};
  stream.clear();
  if (std::memcmp(magic, png_signature, 8) == 0)
    parse_png(stream, facts);
  else if (std::memcmp(magic + 4, "ftyp", 4) == 0)
    parse_mp4(stream, facts.bytes, facts);
  return facts;
}

bool artifact_matches_kind(const ArtifactFacts &facts,
                           const std::string &output_kind) {
  if (!facts.exists || facts.bytes == 0U)
    return false;
  if (output_kind == "image")
    return facts.format == "png" && facts.width != 0U && facts.height != 0U;
  if (output_kind == "video")
    return facts.format == "mp4" && facts.track_count != 0U &&
           facts.duration_seconds > 0.0;
  return false;
}

telemetry::Object artifact_section(const std::filesystem::path &path,
                                   const ArtifactFacts &facts,
                                   bool run_ffprobe) {
  telemetry::Object out;
  out.set("path", path.string());
  out.set("exists", facts.exists);
  out.set("bytes", facts.bytes);
  out.set("sha256", facts.sha256);
  out.set("format", facts.format);
  if (facts.format == "png") {
    telemetry::Object image;
    image.set("width", facts.width);
    image.set("height", facts.height);
    image.set("bit_depth", facts.bit_depth);
    image.set("color_type", facts.color_type);
    out.set("image", std::move(image));
  } else if (facts.format == "mp4") {
    telemetry::Object video;
    video.set("major_brand", facts.major_brand);
    video.set("timescale", facts.timescale);
    video.set("duration_units", facts.duration_units);
    video.set("duration_seconds", facts.duration_seconds);
    video.set("track_count", facts.track_count);
    out.set("video", std::move(video));
  }
  if (run_ffprobe && facts.exists) {
    const auto probe = run_capture(
        {"ffprobe", "-v", "error", "-show_entries",
         "format=duration,size:stream=index,codec_type,codec_name,width,"
         "height,r_frame_rate,nb_frames,sample_rate,channels",
         "-of", "json", path.string()});
    if (probe.exit_status == 0) {
      try {
        out.set("ffprobe", telemetry::from_parsed(json::parse(probe.output)));
      } catch (const std::exception &) {
        out.set("ffprobe", nullptr);
      }
    } else {
      out.set("ffprobe", nullptr);
    }
  }
  return out;
}

} // namespace dif::bench
