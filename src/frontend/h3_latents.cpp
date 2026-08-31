#include "dif/frontend/h3_latents.hpp"

#include "dif/support/error.hpp"
#include "dif/weights/safetensors.hpp"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <span>
#include <vector>

namespace dif::frontend {
namespace {

constexpr std::uint64_t kVideoChannels = 24U;
constexpr std::uint64_t kAudioChannels = 2U;
constexpr std::uint64_t kAudioLatentChannels = 32U;

constexpr std::array<float, kAudioLatentChannels> kAudioLatentMean = {
    -0.020211687488382354F, 0.3876466479950502F,
    -0.04398279799186767F, -0.28591514936373F,
    0.08179686214561671F, -0.35782641352446604F,
    0.040623809960919084F, -0.01552534501956604F,
    -0.223362481667332F, 0.1821006842509091F,
    0.2941778783780663F, -0.07901167601970885F,
    -0.056815072777201F, -0.3699028221860095F,
    -0.31616315591624855F, 0.5905951377425391F,
    -0.052139568068853864F, 0.013673160263486295F,
    -0.03691647864630577F, 0.09732660653298163F,
    -0.3394662328788498F, -0.30685677538541667F,
    -0.24504598907458763F, -0.034698524462007344F,
    0.02868032184767538F, -0.21217779266454084F,
    -0.1678263169941987F, 0.3221287889040614F,
    -0.1223055851554907F, 0.4356604928128464F,
    -0.0502599202236253F, 0.3979258376211797F};

constexpr std::array<float, kAudioLatentChannels> kAudioLatentStd = {
    1.6895524230479284F, 2.76263727217653F,
    1.7945344281264435F, 1.6801681847309828F,
    1.6390226546605453F, 2.7788298348882177F,
    1.7659090095747236F, 1.6199757612137327F,
    2.6336525640336896F, 1.8539356672817833F,
    2.5056497896915633F, 1.811019237886178F,
    1.9579657790720237F, 1.6685498243529284F,
    1.4922469314453364F, 3.298670198067373F,
    1.9491804496832168F, 1.8720003270431442F,
    1.8334080103291832F, 1.6488070416529093F,
    1.6176957696319716F, 1.9131449234774398F,
    1.5695245398428617F, 1.6943659940415912F,
    1.8318420762504692F, 1.5540637421583379F,
    1.9344930328968526F, 1.599198216109855F,
    1.718045989838149F, 1.6307219190837705F,
    1.8661226051202384F, 1.5613768203168363F};

std::uint64_t checked_product(std::initializer_list<std::uint64_t> values) {
  std::uint64_t product = 1U;
  for (const auto value : values) {
    if (value == 0U ||
        product > std::numeric_limits<std::uint64_t>::max() / value)
      fail("H3 latent geometry overflows");
    product *= value;
  }
  return product;
}

} // namespace

runtime::Tensor unpack_h3_video_rows(const runtime::Tensor &video_rows,
                                     std::uint64_t condition_rows,
                                     std::uint64_t latent_frames,
                                     std::uint64_t latent_height,
                                     std::uint64_t latent_width,
                                     std::uint64_t patch_height,
                                     std::uint64_t patch_width) {
  if (patch_height == 0U || patch_width == 0U)
    fail("H3 video patch geometry must be positive");
  const auto rows_per_frame = checked_product(
      {latent_height / patch_height, latent_width / patch_width});
  const auto target_rows = checked_product({latent_frames, rows_per_frame});
  const auto patch_values =
      checked_product({kVideoChannels, patch_height, patch_width});
  if (latent_height % patch_height != 0U ||
      latent_width % patch_width != 0U ||
      video_rows.dtype != ir::DType::F32 || video_rows.dims.size() != 2U ||
      video_rows.dims[0] != condition_rows + target_rows ||
      video_rows.dims[1] != patch_values)
    fail("H3 video rows do not match the requested latent geometry");

  runtime::Tensor latent{
      ir::DType::F32,
      {1U, kVideoChannels, latent_frames, latent_height, latent_width}, {}};
  latent.bytes.resize(static_cast<std::size_t>(latent.element_count() *
                                                sizeof(float)));
  const auto rows = video_rows.f32();
  auto output = latent.f32();
  const auto patches_h = latent_height / patch_height;
  const auto patches_w = latent_width / patch_width;
  for (std::uint64_t frame = 0U; frame < latent_frames; ++frame) {
    for (std::uint64_t patch_y = 0U; patch_y < patches_h; ++patch_y) {
      for (std::uint64_t patch_x = 0U; patch_x < patches_w; ++patch_x) {
        const auto row = condition_rows +
                         (frame * patches_h + patch_y) * patches_w + patch_x;
        for (std::uint64_t channel = 0U; channel < kVideoChannels; ++channel) {
          for (std::uint64_t y = 0U; y < patch_height; ++y) {
            for (std::uint64_t x = 0U; x < patch_width; ++x) {
              const auto column =
                  (channel * patch_height + y) * patch_width + x;
              const auto target =
                  ((channel * latent_frames + frame) * latent_height +
                   patch_y * patch_height + y) *
                      latent_width +
                  patch_x * patch_width + x;
              output[static_cast<std::size_t>(target)] =
                  rows[static_cast<std::size_t>(row * patch_values + column)];
            }
          }
        }
      }
    }
  }
  return latent;
}

runtime::Tensor unpack_h3_audio_rows(const runtime::Tensor &audio_rows,
                                     std::uint64_t condition_rows,
                                     std::uint64_t num_audio_latents,
                                     bool denormalize) {
  const auto generated_rows =
      checked_product({kAudioChannels, num_audio_latents});
  if (audio_rows.dtype != ir::DType::F32 || audio_rows.dims.size() != 2U ||
      audio_rows.dims[0] != condition_rows + generated_rows ||
      audio_rows.dims[1] != kAudioLatentChannels)
    fail("H3 audio rows do not match the requested latent geometry");
  runtime::Tensor latent{ir::DType::F32,
                         {kAudioChannels, kAudioLatentChannels,
                          num_audio_latents}, {}};
  latent.bytes.resize(static_cast<std::size_t>(latent.element_count() *
                                                sizeof(float)));
  const auto rows = audio_rows.f32();
  auto output = latent.f32();
  for (std::uint64_t channel = 0U; channel < kAudioChannels; ++channel) {
    for (std::uint64_t time = 0U; time < num_audio_latents; ++time) {
      const auto row = condition_rows + channel * num_audio_latents + time;
      for (std::uint64_t latent_channel = 0U;
           latent_channel < kAudioLatentChannels; ++latent_channel) {
        auto value = rows[static_cast<std::size_t>(
            row * kAudioLatentChannels + latent_channel)];
        if (denormalize)
          value = value * kAudioLatentStd[latent_channel] +
                  kAudioLatentMean[latent_channel];
        const auto target =
            (channel * kAudioLatentChannels + latent_channel) *
                num_audio_latents +
            time;
        output[static_cast<std::size_t>(target)] = value;
      }
    }
  }
  return latent;
}

void write_h3_latent_handoff(const std::filesystem::path &path,
                             const runtime::Tensor &video_rows,
                             const runtime::Tensor &audio_rows) {
  video_rows.validate();
  audio_rows.validate();
  if (video_rows.dtype != ir::DType::F32 || audio_rows.dtype != ir::DType::F32 ||
      video_rows.dims.size() != 2U || audio_rows.dims.size() != 2U)
    fail("H3 latent handoff requires rank-two F32 row tensors");
  weights::SafeTensorWriter writer(
      path, {{"video_state_rows", video_rows.dtype, video_rows.dims},
             {"audio_state_rows", audio_rows.dtype, audio_rows.dims}});
  writer.append("video_state_rows",
                std::span<const std::uint8_t>(video_rows.data(),
                                              video_rows.byte_size()));
  writer.append("audio_state_rows",
                std::span<const std::uint8_t>(audio_rows.data(),
                                              audio_rows.byte_size()));
  (void)writer.finish();
}

} // namespace dif::frontend
