#include "dif/frontend/h3_motion.hpp"
#include "dif/support/error.hpp"
#include "dif/weights/safetensors.hpp"

#include <charconv>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <string_view>

namespace {

void usage() {
  std::cerr
      << "usage: difh3motion save (--input-handoff FILE | --video ROWS --audio ROWS)"
         " --latent-t N --audio-latents N --width N --height N --output FILE"
         " [--endpoint-frames N --condition-video-rows N --condition-audio-rows N]\n"
      << "       difh3motion extract --context FILE --context-frames 5|22|39"
         " --width N --height N --output-dir DIR [--output-frames N]\n"
      << "       difh3motion prepare --context FILE --context-frames 5|22|39"
         " --width N --height N --condition-noise ROWS --video ROWS --audio ROWS"
         " --output-dir DIR [--output-frames N]\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      usage();
      return 2;
    }
    const std::string mode = argv[1];
    const std::map<std::string, std::string> allowed = {
        {"--input-handoff", "save"}, {"--video", "save prepare"},
        {"--audio", "save prepare"}, {"--latent-t", "save"},
        {"--audio-latents", "save"}, {"--width", "save extract prepare"},
        {"--height", "save extract prepare"}, {"--output", "save"},
        {"--endpoint-frames", "save"}, {"--condition-video-rows", "save"},
        {"--condition-audio-rows", "save"}, {"--context", "extract prepare"},
        {"--context-frames", "extract prepare"}, {"--condition-noise", "prepare"},
        {"--output-dir", "extract prepare"}, {"--output-frames", "extract prepare"}};
    if (mode != "save" && mode != "extract" && mode != "prepare")
      dif::fail("motion command must be save, extract, or prepare");
    std::map<std::string, std::string> options;
    for (int i = 2; i < argc; ++i) {
      const std::string key = argv[i];
      const auto found = allowed.find(key);
      if (found == allowed.end() || found->second.find(mode) == std::string::npos ||
          i + 1 >= argc || options.contains(key))
        dif::fail("unknown, duplicate, or incomplete motion option: " + key);
      options.emplace(key, argv[++i]);
    }
    auto required = [&](const std::string &key) -> const std::string & {
      const auto found = options.find(key);
      if (found == options.end() || found->second.empty())
        dif::fail("missing motion option: " + key);
      return found->second;
    };
    auto number = [&](const std::string &key, std::uint64_t fallback = 0U) {
      if (!options.contains(key))
        return fallback;
      const auto &text = required(key);
      std::uint64_t result{};
      const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
      if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        dif::fail("invalid unsigned integer for " + key);
      return result;
    };
    const auto width = number("--width");
    const auto height = number("--height");
    if (mode == "save") {
      dif::runtime::Tensor video, audio;
      if (options.contains("--input-handoff")) {
        if (options.contains("--video") || options.contains("--audio"))
          dif::fail("choose a handoff or separate video/audio tensors");
        const auto file = dif::weights::read_safetensors(required("--input-handoff"));
        video = dif::weights::map_safetensor(file, "video_state_rows");
        audio = dif::weights::map_safetensor(file, "audio_state_rows");
      } else {
        video = dif::runtime::map_tensor(required("--video"));
        audio = dif::runtime::map_tensor(required("--audio"));
      }
      const auto frames = dif::frontend::save_h3_motion_context_tail(
          required("--output"), video, audio, number("--latent-t"),
          number("--audio-latents"), width, height,
          number("--endpoint-frames", dif::frontend::h3_motion_context_pixel_frames(
                                         number("--latent-t"))),
          number("--condition-video-rows"), number("--condition-audio-rows"));
      std::cout << "H3_MOTION_SAVED context_frames=" << frames
                << " path=" << required("--output") << '\n';
      return 0;
    }
    const auto context = dif::frontend::load_h3_motion_context(
        required("--context"), width, height, number("--context-frames", 22U));
    const std::filesystem::path output = required("--output-dir");
    if (std::filesystem::exists(output))
      dif::fail("refusing to overwrite motion output directory: " + output.string());
    dif::frontend::H3MotionState state;
    if (mode == "prepare")
      state = dif::frontend::prepare_h3_motion_context_state(
          context, dif::runtime::map_tensor(required("--condition-noise")),
          dif::runtime::map_tensor(required("--video")),
          dif::runtime::map_tensor(required("--audio")));
    const auto output_frames = number("--output-frames");
    const auto trim = output_frames
                          ? dif::frontend::make_h3_motion_trim(context.context_frames,
                                                               output_frames)
                          : dif::frontend::H3MotionTrim{};
    std::filesystem::create_directories(output);
    dif::runtime::write_tensor(context.video_rows, output / "video_context_rows.diftensor");
    dif::runtime::write_tensor(context.audio_rows, output / "audio_context_rows.diftensor");
    if (mode == "prepare") {
      dif::runtime::write_tensor(state.video_rows, output / "video_state_rows.diftensor");
      dif::runtime::write_tensor(state.audio_rows, output / "audio_state_rows.diftensor");
    }
    std::ofstream meta(output / "motion_context.json");
    if (!meta)
      dif::fail("cannot open motion metadata output");
    meta << std::setprecision(17)
         << "{\"schema\":1,\"fps\":24,\"audio_sample_rate\":32000"
         << ",\"context_frames\":" << context.context_frames
         << ",\"condition_video_rows\":" << context.video_rows.dims[0]
         << ",\"condition_audio_rows\":" << context.audio_rows.dims[0]
         << ",\"source_video_latent_frames\":" << context.source_video_latent_frames
         << ",\"source_audio_latents\":" << context.source_audio_latents
         << ",\"source_audio_overhang\":" << context.source_audio_overhang;
    if (output_frames)
      meta << ",\"trim_start_frames\":" << trim.trim_start_frames
           << ",\"trim_start_audio_samples\":" << trim.trim_start_audio_samples
           << ",\"output_frames\":" << trim.output_frames
           << ",\"output_audio_samples\":" << trim.output_audio_samples
           << ",\"continuation_endpoint_frames\":" << trim.continuation_endpoint_frames;
    meta << "}\n";
    meta.close();
    if (!meta)
      dif::fail("failed writing motion metadata");
    std::cout << "H3_MOTION_PREPARED mode=" << mode
              << " context_frames=" << context.context_frames
              << " video_rows=" << context.video_rows.dims[0]
              << " audio_rows=" << context.audio_rows.dims[0]
              << " source_audio_overhang=" << context.source_audio_overhang << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difh3motion: " << error.what() << '\n';
    return 1;
  }
}
