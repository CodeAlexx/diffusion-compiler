#include "dif/frontend/h3_conditioning.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::uint64_t number(const char *text, const char *label) {
  char *end = nullptr;
  const auto value = std::strtoull(text, &end, 10);
  if (!text[0] || !end || *end != '\0')
    dif::fail(std::string("invalid ") + label + ": " + text);
  return value;
}

float real(const char *text, const char *label) {
  char *end = nullptr;
  const auto value = std::strtof(text, &end);
  if (!text[0] || !end || *end != '\0' || !std::isfinite(value))
    dif::fail(std::string("invalid ") + label + ": " + text);
  return value;
}

template <typename T>
dif::runtime::Tensor tensor(dif::ir::DType dtype,
                            std::vector<std::uint64_t> dims,
                            const std::vector<T> &values) {
  dif::runtime::Tensor result{dtype, std::move(dims), {}};
  result.bytes.resize(values.size() * sizeof(T));
  std::memcpy(result.bytes.data(), values.data(), result.bytes.size());
  result.validate();
  return result;
}

void write_i32(const std::filesystem::path &directory, const char *name,
               const std::vector<std::int32_t> &values) {
  dif::runtime::write_tensor(
      tensor(dif::ir::DType::I32,
             {static_cast<std::uint64_t>(values.size())}, values),
      directory / name);
}

void usage() {
  std::cerr
      << "usage: difh3layout t2va OUT_DIR TEXT_TAGS.diftensor FRAMES HEIGHT "
         "WIDTH AUDIO_LATENTS PATCH_T PATCH_H PATCH_W VIDEO_T AUDIO_T "
         "CONDITION_VIDEO_T CONDITION_AUDIO_T [first|last ...]\n"
      << "       difh3layout ref2va OUT_DIR TEXT_TAGS.diftensor FRAMES "
         "HEIGHT WIDTH AUDIO_LATENTS PATCH_T PATCH_H PATCH_W VIDEO_T AUDIO_T "
         "CONDITION_VIDEO_T CONDITION_AUDIO_T KIND:T:H:W:A [...]\n";
}

dif::frontend::H3ReferenceGeometry
reference_geometry(const std::string &specification) {
  std::array<std::string, 5> fields;
  std::size_t start = 0U;
  for (std::size_t field = 0U; field < fields.size(); ++field) {
    const auto separator = specification.find(':', start);
    if ((field + 1U < fields.size() && separator == std::string::npos) ||
        (field + 1U == fields.size() && separator != std::string::npos))
      dif::fail("reference geometry must be KIND:T:H:W:A");
    fields[field] = specification.substr(
        start, separator == std::string::npos ? separator : separator - start);
    if (separator == std::string::npos)
      break;
    start = separator + 1U;
  }
  dif::frontend::H3ReferenceKind kind;
  if (fields[0] == "image")
    kind = dif::frontend::H3ReferenceKind::Image;
  else if (fields[0] == "video")
    kind = dif::frontend::H3ReferenceKind::Video;
  else if (fields[0] == "audio")
    kind = dif::frontend::H3ReferenceKind::Audio;
  else
    dif::fail("reference kind must be image, video, or audio");
  return {kind, number(fields[1].c_str(), "reference latent frames"),
          number(fields[2].c_str(), "reference latent height"),
          number(fields[3].c_str(), "reference latent width"),
          number(fields[4].c_str(), "reference audio latents")};
}

} // namespace

int main(int argc, char **argv) {
  try {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (argc < 15 || (mode != "t2va" && mode != "ref2va")) {
      usage();
      return 2;
    }
    const auto tags_tensor = dif::runtime::read_tensor(argv[3]);
    if (tags_tensor.dtype != dif::ir::DType::I32 ||
        tags_tensor.dims.size() != 1U)
      dif::fail("TEXT_TAGS must be an I32 rank-1 tensor");
    const auto *tags_data =
        reinterpret_cast<const std::int32_t *>(tags_tensor.data());
    const auto tags = std::span<const std::int32_t>(
        tags_data, static_cast<std::size_t>(tags_tensor.element_count()));
    std::vector<dif::frontend::H3KeyframeAnchor> anchors;
    std::vector<dif::frontend::H3ReferenceGeometry> references;
    for (int index = 15; index < argc; ++index) {
      const std::string name = argv[index];
      if (mode == "ref2va")
        references.push_back(reference_geometry(name));
      else if (name == "first")
        anchors.push_back(dif::frontend::H3KeyframeAnchor::First);
      else if (name == "last")
        anchors.push_back(dif::frontend::H3KeyframeAnchor::Last);
      else
        dif::fail("keyframe anchor must be first or last");
    }
    const auto layout =
        mode == "ref2va"
            ? dif::frontend::make_h3_ref2va_layout(
                  tags, references, number(argv[4], "frames"),
                  number(argv[5], "height"), number(argv[6], "width"),
                  number(argv[7], "audio latents"),
                  number(argv[8], "patch t"), number(argv[9], "patch h"),
                  number(argv[10], "patch w"))
            : dif::frontend::make_h3_t2va_layout(
                  tags, number(argv[4], "frames"),
                  number(argv[5], "height"), number(argv[6], "width"),
                  number(argv[7], "audio latents"),
                  number(argv[8], "patch t"), number(argv[9], "patch h"),
                  number(argv[10], "patch w"), anchors);
    const auto plan = dif::frontend::make_h3_row_timestep_plan(
        layout, real(argv[11], "video timestep"),
        real(argv[12], "audio timestep"),
        real(argv[13], "condition video timestep"),
        real(argv[14], "condition audio timestep"));

    const std::filesystem::path output = argv[2];
    std::filesystem::create_directories(output);
    dif::runtime::write_tensor(
        tensor(dif::ir::DType::F32, {layout.sequence_length, 3U},
               layout.position_ids),
        output / "position_ids.diftensor");
    write_i32(output, "token_tags.diftensor", layout.token_tags);
    write_i32(output, "text_indices.diftensor", layout.text_indices);
    write_i32(output, "video_indices.diftensor", layout.video_indices);
    write_i32(output, "audio_indices.diftensor", layout.audio_indices);
    write_i32(output, "text_map.diftensor", layout.text_map);
    write_i32(output, "video_map.diftensor", layout.video_map);
    write_i32(output, "audio_map.diftensor", layout.audio_map);
    write_i32(output, "timestep_indices.diftensor", plan.timestep_indices);
    write_i32(output, "adaln_indices.diftensor", plan.adaln_indices);
    dif::runtime::write_tensor(
        tensor(dif::ir::DType::F32,
               {static_cast<std::uint64_t>(plan.timesteps.size())},
               plan.timesteps),
        output / "timesteps.diftensor");
    std::cout << "H3_LAYOUT PASS task=" << mode << " path=" << output
              << " sequence=" << layout.sequence_length
              << " text_rows=" << layout.text_indices.size()
              << " video_rows=" << layout.video_indices.size()
              << " audio_rows=" << layout.audio_indices.size()
              << " condition_video_rows="
              << layout.num_condition_video_rows
              << " condition_audio_rows="
              << layout.num_condition_audio_rows
              << " timestep_tables=" << plan.timesteps.size()
              << " padding_rows=0 attention_document=single\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difh3layout: " << error.what() << "\n";
    return 1;
  }
}
