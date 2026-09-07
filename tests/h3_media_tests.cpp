#include "dif/frontend/h3_media.hpp"
#include "dif/support/wav.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

#include <sys/wait.h>
#include <unistd.h>

namespace {
int failures{};
void expect(bool condition, const char *label) {
  if (!condition) { ++failures; std::cerr << "FAIL " << label << '\n'; }
}
template<class F> void rejects(F function, const char *label) {
  bool rejected = false;
  try { function(); } catch (const std::exception &) { rejected = true; }
  expect(rejected, label);
}
std::vector<std::uint8_t> read(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot read evidence " + path.string());
  return {std::istreambuf_iterator<char>(file), {}};
}
int run(const std::vector<std::string> &arguments) {
  std::vector<char *> args;
  for (const auto &a : arguments) args.push_back(const_cast<char *>(a.c_str()));
  args.push_back(nullptr);
  const auto child = fork();
  if (child < 0) throw std::runtime_error("cannot fork media mechanics gate");
  if (child == 0) { execvp(args[0], args.data()); _exit(127); }
  int status{};
  if (waitpid(child, &status, 0) != child || !WIFEXITED(status)) return -1;
  return WEXITSTATUS(status);
}
} // namespace
int main(int argc, char **argv) {
  using namespace dif;
  try {
    const auto directory = std::filesystem::temp_directory_path() /
        ("dif-h3-media-tests-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(directory);
    auto decoded = runtime::zeros({0, ir::DType::F32, 0, {1,3,56,32,32}});
    for (std::size_t c = 0; c < 3; ++c)
      for (std::size_t f = 0; f < 56; ++f)
        for (std::size_t y = 0; y < 32; ++y)
          for (std::size_t x = 0; x < 32; ++x)
            decoded.f32()[((c * 56 + f) * 32 + y) * 32 + x] =
                c == 0 ? static_cast<float>(f) / 55 :
                c == 1 ? static_cast<float>(y) / 31 : static_cast<float>(x) / 31;
    const auto full = frontend::make_h3_rgb24_video(decoded);
    const auto selected = frontend::make_h3_rgb24_video(decoded, 22, 24);
    expect(full.frames == 56 && selected.frames == 24 && selected.width == 32 && selected.height == 32,
           "native media window changes time only, no resize");
    expect(std::equal(selected.bytes.begin(), selected.bytes.end(), full.bytes.begin() + 22 * 32 * 32 * 3),
           "delivered RGB24 window exactly equals original frame byte slice");
    expect(frontend::make_h3_rgb24_video(decoded, 22).frames == 34, "omitted cap keeps every remaining frame");
    rejects([&] { frontend::make_h3_rgb24_video(decoded, 56); }, "empty delivery rejected");
    rejects([&] { frontend::make_h3_rgb24_video(decoded, 22, 35); }, "undersized decoded video rejected");
    const auto trim = frontend::make_h3_motion_trim(22, 24);
    expect(trim.trim_start_audio_samples == 29333 && trim.output_audio_samples == 32000,
           "22-frame overlap maps to rounded29333 samples and exact1-second cap");
    expect(frontend::make_h3_motion_trim(0, 24).trim_start_audio_samples == 0,
           "cap-only delivery has no overlap");
    constexpr std::uint64_t samples = 75000;
    std::vector<float> waveform(samples * 2);
    for (std::size_t i = 0; i < samples; ++i) {
      waveform[i] = static_cast<float>(static_cast<int>(i % 4096) - 2048) / 32768.0F;
      waveform[samples + i] = -waveform[i];
    }
    const auto audio = directory / "source.wav";
    support::write_wav_pcm16(audio, waveform, 2, samples, 32000);
    const auto trimmed = directory / "trimmed.wav";
    frontend::write_h3_trimmed_audio_wav(audio, trimmed, trim);
    const auto source_bytes = read(audio), trimmed_bytes = read(trimmed);
    expect(trimmed_bytes.size() == 44 + 32000 * 4 &&
               std::equal(trimmed_bytes.begin() + 44, trimmed_bytes.end(),
                          source_bytes.begin() + 44 + trim.trim_start_audio_samples * 4),
           "trimmed stereo PCM16 samples byte-exact, no float requantization or timestamp seek");
    rejects([&] { frontend::write_h3_trimmed_audio_wav(audio, trimmed, trim); }, "existing WAV protected");
    rejects([&] { frontend::write_h3_trimmed_audio_wav(audio, directory / "too-long.wav",
        frontend::make_h3_motion_trim(39, 56)); }, "undersized decoded audio rejected instead of padded");
    const auto wrong_rate = directory / "wrong-rate.wav";
    support::write_wav_pcm16(wrong_rate, waveform, 2, samples, 44100);
    rejects([&] { frontend::write_h3_trimmed_audio_wav(wrong_rate, directory / "resampled.wav", trim); },
            "non-native audio rate rejected instead of resampled");
    const auto tensor_path = directory / "decoded.diftensor";
    runtime::write_tensor(decoded, tensor_path);
    if (argc == 2) {
      const auto output = directory / "delivery";
      expect(run({argv[1], "--video", tensor_path.string(), "--audio-wav", audio.string(),
                  "--output-dir", output.string(), "--input-fps", "24", "--trim-start-frames", "22",
                  "--output-frames", "24", "--encoder", "libx264"}) == 0,
             "real CPU ffmpeg H3 delivery tool succeeds");
      expect(read(output / "frames.rgb") == selected.bytes && read(output / "audio_trimmed.wav") == trimmed_bytes,
             "real CLI feeds exact selected frame/sample bytes to ffmpeg");
      const auto video_probe = directory / "video-probe.csv";
      expect(run({"ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                  "stream=width,height,nb_read_frames", "-of", "csv=p=0", "-o", video_probe.string(),
                  (output / "video.mp4").string()}) == 0, "real MP4 video stream decodes for frame count");
      const auto text = read(video_probe);
      expect(std::string(text.begin(), text.end()).find("32,32,24") != std::string::npos,
             "saved MP4 contains exactly24 frames at unchanged32x32 geometry");
      const auto audio_probe = directory / "audio-probe.csv";
      expect(run({"ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels",
                  "-of", "csv=p=0", "-o", audio_probe.string(), (output / "video.mp4").string()}) == 0,
             "saved MP4 has an audio stream");
      const auto audio_text = read(audio_probe);
      expect(std::string(audio_text.begin(), audio_text.end()).find("32000,2") != std::string::npos,
             "saved MP4 retains32kHz stereo audio");
      const auto no_trim = directory / "no-trim";
      expect(run({argv[1], "--video", tensor_path.string(), "--audio-wav", audio.string(),
                  "--output-dir", no_trim.string(), "--input-fps", "24", "--encoder", "libx264"}) == 0,
             "original no-trim CLI remains operational");
      expect(read(no_trim / "frames.rgb") == full.bytes &&
                 !std::filesystem::exists(no_trim / "audio_trimmed.wav"),
             "no-trim path keeps complete original RGB and original audio handoff");
      const auto cap_only = directory / "cap-only";
      expect(run({argv[1], "--video", tensor_path.string(), "--audio-wav", audio.string(),
                  "--output-dir", cap_only.string(), "--input-fps", "24", "--trim-start-frames", "0",
                  "--output-frames", "24", "--encoder", "libx264"}) == 0,
             "cap-only CLI does not invent continuation overlap");
      const auto capped_audio = read(cap_only / "audio_trimmed.wav");
      expect(capped_audio.size() == 44 + 32000 * 4 &&
                 std::equal(capped_audio.begin() + 44, capped_audio.end(), source_bytes.begin() + 44),
             "cap-only stereo audio starts at original sample zero");
      expect(run({argv[1], "--video", tensor_path.string(), "--audio-wav", audio.string(),
                  "--output-dir", (directory / "invalid-fps").string(), "--input-fps", "30",
                  "--trim-start-frames", "22", "--output-frames", "24"}) != 0,
             "CLI refuses nonnative continuation FPS");
    }
    if (failures) return 1;
    std::cout << "H3_MEDIA_CPU PASS RGB24=frame-byte-exact PCM16=sample-byte-exact overlap=22 output=24"
              << " ffmpeg=" << (argc == 2 ? "real-CPU-MP4-pass" : "not-run")
              << " evidence=" << directory << " model_decoded_quality=not-run\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL " << error.what() << '\n'; return 1;
  }
}
