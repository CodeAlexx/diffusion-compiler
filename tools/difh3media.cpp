#include "dif/frontend/h3_media.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

struct Options {
  std::filesystem::path video;
  std::filesystem::path audio_wav;
  std::filesystem::path output_directory;
  std::string ffmpeg{"ffmpeg"};
  std::string encoder{"libx264"};
  std::uint64_t input_fps{};
  std::uint64_t output_fps{};
};

std::uint64_t number(const std::string &text, const char *label) {
  char *end = nullptr;
  const auto value = std::strtoull(text.c_str(), &end, 10);
  if (text.empty() || !end || *end != '\0' || value == 0U)
    dif::fail(std::string("invalid ") + label + ": " + text);
  return value;
}

void usage() {
  std::cerr << "usage: difh3media --video DECODED.diftensor --audio-wav audio.wav --output-dir DIR --input-fps N [--output-fps N] [--ffmpeg FILE] [--encoder h264_nvenc|libx264]\n";
}

Options parse(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    auto value = [&](const char *label) -> std::string {
      if (++index >= argc)
        dif::fail(std::string("missing value for ") + label);
      return argv[index];
    };
    if (option == "--video")
      options.video = value("--video");
    else if (option == "--audio-wav")
      options.audio_wav = value("--audio-wav");
    else if (option == "--output-dir")
      options.output_directory = value("--output-dir");
    else if (option == "--input-fps")
      options.input_fps = number(value("--input-fps"), "input fps");
    else if (option == "--output-fps")
      options.output_fps = number(value("--output-fps"), "output fps");
    else if (option == "--ffmpeg")
      options.ffmpeg = value("--ffmpeg");
    else if (option == "--encoder")
      options.encoder = value("--encoder");
    else {
      usage();
      dif::fail("unknown difh3media option: " + option);
    }
  }
  if (options.video.empty() || options.audio_wav.empty() ||
      options.output_directory.empty() || options.input_fps == 0U) {
    usage();
    dif::fail("difh3media is missing a required argument");
  }
  if (options.output_fps == 0U)
    options.output_fps = options.input_fps;
  if (options.encoder != "h264_nvenc" && options.encoder != "libx264")
    dif::fail("video encoder must be h264_nvenc or libx264");
  if (!std::filesystem::is_regular_file(options.audio_wav) ||
      std::filesystem::file_size(options.audio_wav) <= 44U)
    dif::fail("audio WAV is missing or lacks sample data");
  return options;
}

void write_bytes(const std::filesystem::path &path,
                 const std::vector<std::uint8_t> &bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    dif::fail("cannot create RGB24 stream: " + path.string());
  output.write(reinterpret_cast<const char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!output)
    dif::fail("cannot write RGB24 stream: " + path.string());
}

void run(const std::vector<std::string> &arguments) {
  std::vector<char *> argv;
  argv.reserve(arguments.size() + 1U);
  for (const auto &argument : arguments)
    argv.push_back(const_cast<char *>(argument.c_str()));
  argv.push_back(nullptr);
  const auto child = fork();
  if (child < 0)
    dif::fail("cannot fork ffmpeg process");
  if (child == 0) {
    execvp(argv[0], argv.data());
    _exit(127);
  }
  int status = 0;
  if (waitpid(child, &status, 0) != child || !WIFEXITED(status) ||
      WEXITSTATUS(status) != 0)
    dif::fail("ffmpeg H3 A/V mux failed");
}

std::string json_escape(const std::string &value) {
  constexpr char hex[] = "0123456789abcdef";
  std::string escaped;
  escaped.reserve(value.size());
  for (const auto character : value) {
    const auto byte = static_cast<unsigned char>(character);
    if (character == '"' || character == '\\') {
      escaped.push_back('\\');
      escaped.push_back(character);
    } else if (character == '\b') {
      escaped += "\\b";
    } else if (character == '\f') {
      escaped += "\\f";
    } else if (character == '\n') {
      escaped += "\\n";
    } else if (character == '\r') {
      escaped += "\\r";
    } else if (character == '\t') {
      escaped += "\\t";
    } else if (byte < 0x20U) {
      escaped += "\\u00";
      escaped.push_back(hex[byte >> 4U]);
      escaped.push_back(hex[byte & 0x0fU]);
    } else {
      escaped.push_back(character);
    }
  }
  return escaped;
}

void write_result(const std::filesystem::path &path,
                  const std::filesystem::path &artifact,
                  const dif::frontend::H3Rgb24Video &video,
                  const Options &options, double mux_milliseconds) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    dif::fail("cannot create H3 result JSON");
  output << "{\n"
         << "  \"schema\":\"serenity.minimax_h3.result.v1\",\n"
         << "  \"state\":\"done\",\n"
         << "  \"artifact_path\":\"" << json_escape(artifact.string())
         << "\",\n"
         << "  \"width\":" << video.width << ",\n"
         << "  \"height\":" << video.height << ",\n"
         << "  \"frames\":" << video.frames << ",\n"
         << "  \"fps\":" << options.output_fps << ",\n"
         << "  \"audio\":true,\n"
         << "  \"video_encoder\":\"" << json_escape(options.encoder)
         << "\",\n"
         << "  \"mux_ms\":" << mux_milliseconds << "\n"
         << "}\n";
  if (!output)
    dif::fail("cannot write H3 result JSON");
}

} // namespace

int main(int argc, char **argv) {
  try {
    const auto options = parse(argc, argv);
    std::filesystem::create_directories(options.output_directory);
    const auto rgb_path = options.output_directory / "frames.rgb";
    const auto media_path = options.output_directory / "video.mp4";
    const auto result_path = options.output_directory / "result.json";
    for (const auto &path : {rgb_path, media_path, result_path}) {
      if (std::filesystem::exists(path))
        dif::fail("refusing to overwrite output: " + path.string());
    }
    const auto video = dif::frontend::make_h3_rgb24_video(
        dif::runtime::read_tensor(options.video));
    write_bytes(rgb_path, video.bytes);
    const auto start = std::chrono::steady_clock::now();
    std::vector<std::string> arguments = {
        options.ffmpeg, "-v", "error", "-y", "-f", "rawvideo",
        "-pixel_format", "rgb24", "-video_size",
        std::to_string(video.width) + "x" + std::to_string(video.height),
        "-framerate", std::to_string(options.input_fps), "-i",
        rgb_path.string(), "-i", options.audio_wav.string()};
    if (options.output_fps != options.input_fps) {
      arguments.push_back("-vf");
      arguments.push_back("fps=" + std::to_string(options.output_fps));
    }
    arguments.insert(arguments.end(),
                     {"-frames:v", std::to_string(video.frames), "-c:v",
                      options.encoder});
    if (options.encoder == "h264_nvenc")
      arguments.insert(arguments.end(), {"-preset", "p7", "-tune", "hq",
                                         "-rc", "vbr", "-cq", "18",
                                         "-b:v", "0"});
    else
      arguments.insert(arguments.end(), {"-preset", "slow", "-crf", "18"});
    const auto media_duration_seconds =
        std::to_string(static_cast<double>(video.frames) /
                       static_cast<double>(options.output_fps));
    arguments.insert(arguments.end(),
                     {"-pix_fmt", "yuv420p", "-af",
                      "atrim=duration=" + media_duration_seconds +
                          ",apad=whole_dur=" + media_duration_seconds,
                      "-c:a", "aac",
                      "-movflags", "+faststart",
                      media_path.string()});
    run(arguments);
    const auto mux_milliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start)
            .count();
    if (!std::filesystem::is_regular_file(media_path) ||
        std::filesystem::file_size(media_path) == 0U)
      dif::fail("ffmpeg returned success without a media artifact");
    write_result(result_path, media_path, video, options, mux_milliseconds);
    std::cout << "H3_MEDIA PASS artifact=" << media_path
              << " frames=" << video.frames << " geometry=" << video.width
              << 'x' << video.height << " input_fps=" << options.input_fps
              << " output_fps=" << options.output_fps
              << " audio_wav=" << options.audio_wav
              << " encoder=" << options.encoder
              << " mux_ms=" << mux_milliseconds << " range=["
              << video.minimum << ',' << video.maximum << "]\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difh3media: " << error.what() << '\n';
    return 1;
  }
}
