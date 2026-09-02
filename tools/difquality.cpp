// difquality: generic image/video/audio gate assistant. Scalar metrics can
// support admission or prove a broken output; they never replace the
// required perceptual review. Verdicts are PASS, FAIL, or MANUAL REVIEW
// REQUIRED, and PASS needs a recorded human review.

#include "dif/bench/artifact.hpp"
#include "dif/bench/process.hpp"
#include "dif/quality/metrics.hpp"
#include "dif/support/error.hpp"
#include "dif/support/png.hpp"
#include "dif/telemetry/schema.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

void usage() {
  std::cerr
      << "usage: difquality CANDIDATE [--reference FILE] [--kind image|video|audio]\n"
         "                  [--min-psnr DB] [--min-ssim F] [--min-snr DB]\n"
         "                  [--frames N] [--no-ffmpeg]\n"
         "                  [--reviewer NAME --review accept|reject [--review-note TEXT]]\n"
         "                  [--json] [--report FILE]\n"
         "\n"
         "Verdict rules: FAIL when the artifact is missing, undecodable, of the wrong\n"
         "kind, geometrically mismatched with the reference, constant/silent, below a\n"
         "numeric bar against the reference, or rejected by a recorded review. PASS\n"
         "only when numeric admission holds (or no reference was given and sanity\n"
         "holds) AND a human review is recorded as accept. Otherwise MANUAL REVIEW\n"
         "REQUIRED. Defaults: PSNR 30 dB, SSIM 0.90, SNR 20 dB, 8 sampled frames.\n";
}

double number(const std::string &text, const char *label) {
  char *end = nullptr;
  const auto value = std::strtod(text.c_str(), &end);
  if (!end || *end != '\0' || !std::isfinite(value))
    dif::fail(std::string("invalid ") + label);
  return value;
}

std::string kind_from_facts(const dif::bench::ArtifactFacts &facts) {
  if (facts.format == "png")
    return "image";
  if (facts.format == "mp4")
    return "video";
  return "";
}

bool tool_available(const std::string &tool) {
  const char *path = std::getenv("PATH");
  if (!path)
    return false;
  std::stringstream entries(path);
  std::string entry;
  while (std::getline(entries, entry, ':'))
    if (!entry.empty() && std::filesystem::exists(std::filesystem::path(entry) / tool))
      return true;
  return false;
}

dif::telemetry::Value finite_or_null(double value) {
  if (!std::isfinite(value))
    return dif::telemetry::Value(nullptr);
  return dif::telemetry::Value(value);
}

dif::telemetry::Object image_sanity_section(const dif::quality::ImageSanity &sanity) {
  dif::telemetry::Object out;
  out.set("decodable", sanity.decodable);
  out.set("width", sanity.width);
  out.set("height", sanity.height);
  out.set("constant_fraction", sanity.constant_fraction);
  out.set("mean_luma", sanity.mean_luma);
  out.set("luma_stddev", sanity.luma_stddev);
  out.set("problem", sanity.problem);
  return out;
}

dif::telemetry::Object image_comparison_section(
    const dif::quality::ImageComparison &comparison) {
  dif::telemetry::Object out;
  out.set("comparable", comparison.comparable);
  out.set("problem", comparison.problem);
  out.set("psnr_db", finite_or_null(comparison.psnr_db));
  out.set("psnr_infinite", std::isinf(comparison.psnr_db));
  out.set("ssim_8x8", comparison.ssim_8x8);
  out.set("mean_absolute_difference", comparison.mean_absolute_difference);
  out.set("max_absolute_difference", comparison.max_absolute_difference);
  out.set("identical_pixel_fraction", comparison.identical_pixel_fraction);
  out.set("bit_identical", comparison.bit_identical);
  return out;
}

dif::telemetry::Object audio_sanity_section(const dif::quality::AudioSanity &sanity) {
  dif::telemetry::Object out;
  out.set("decodable", sanity.decodable);
  out.set("sample_rate", sanity.sample_rate);
  out.set("channels", sanity.channels);
  out.set("samples_per_channel", sanity.samples_per_channel);
  out.set("duration_seconds", sanity.duration_seconds);
  out.set("rms", sanity.rms);
  out.set("peak", sanity.peak);
  out.set("silence_fraction", sanity.silence_fraction);
  out.set("clipping_fraction", sanity.clipping_fraction);
  out.set("problem", sanity.problem);
  return out;
}

dif::telemetry::Object audio_comparison_section(
    const dif::quality::AudioComparison &comparison) {
  dif::telemetry::Object out;
  out.set("comparable", comparison.comparable);
  out.set("problem", comparison.problem);
  out.set("snr_db", finite_or_null(comparison.snr_db));
  out.set("snr_infinite", std::isinf(comparison.snr_db) && comparison.snr_db > 0.0);
  out.set("relative_l2", finite_or_null(comparison.relative_l2));
  out.set("max_absolute_difference", comparison.max_absolute_difference);
  out.set("compared_samples", comparison.compared_samples);
  out.set("bit_identical", comparison.bit_identical);
  return out;
}

struct Bars {
  double min_psnr{30.0};
  double min_ssim{0.90};
  double min_snr{20.0};
};

struct Judgement {
  // "pass", "fail", or "not-computed" (no reference).
  std::string numeric_admission{"not-computed"};
  std::vector<std::string> failures;
  std::vector<std::string> notes;
};

void judge_image(Judgement &judgement, const Bars &bars,
                 const dif::quality::ImageSanity &sanity,
                 const dif::quality::ImageComparison *comparison) {
  if (!sanity.decodable) {
    judgement.failures.push_back("candidate image is not decodable");
    return;
  }
  if (!sanity.problem.empty())
    judgement.failures.push_back("candidate sanity: " + sanity.problem);
  if (!comparison)
    return;
  if (!comparison->comparable) {
    judgement.failures.push_back("reference comparison: " + comparison->problem);
    return;
  }
  bool ok = true;
  if (comparison->psnr_db < bars.min_psnr) {
    judgement.failures.push_back("PSNR below bar");
    ok = false;
  }
  if (comparison->ssim_8x8 < bars.min_ssim) {
    judgement.failures.push_back("SSIM below bar");
    ok = false;
  }
  judgement.numeric_admission = ok ? "pass" : "fail";
}

void judge_audio(Judgement &judgement, const Bars &bars,
                 const dif::quality::AudioSanity &sanity,
                 const dif::quality::AudioComparison *comparison) {
  if (!sanity.decodable) {
    judgement.failures.push_back("candidate audio is not decodable");
    return;
  }
  if (!sanity.problem.empty())
    judgement.failures.push_back("candidate sanity: " + sanity.problem);
  if (!comparison)
    return;
  if (!comparison->comparable) {
    judgement.failures.push_back("reference comparison: " + comparison->problem);
    return;
  }
  if (!comparison->problem.empty())
    judgement.notes.push_back(comparison->problem);
  const bool ok = comparison->snr_db >= bars.min_snr;
  if (!ok)
    judgement.failures.push_back("SNR below bar");
  if (judgement.numeric_admission != "fail")
    judgement.numeric_admission = ok ? "pass" : "fail";
}

struct TempDir {
  std::filesystem::path path;
  TempDir() {
    std::string pattern = (std::filesystem::temp_directory_path() / "difquality-XXXXXX").string();
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    if (!::mkdtemp(buffer.data()))
      dif::fail("cannot create a temporary directory");
    path = buffer.data();
  }
  ~TempDir() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

std::vector<std::filesystem::path>
sample_frames(const std::filesystem::path &video, double duration,
              std::uint32_t frames, const std::filesystem::path &directory,
              const std::string &prefix) {
  std::vector<std::filesystem::path> paths;
  for (std::uint32_t index = 0; index < frames; ++index) {
    const auto at = duration <= 0.0
                        ? 0.0
                        : duration * (static_cast<double>(index) + 0.5) /
                              static_cast<double>(frames);
    const auto out = directory / (prefix + std::to_string(index) + ".png");
    std::ostringstream timestamp;
    timestamp << std::fixed << std::setprecision(4) << at;
    const auto capture = dif::bench::run_capture(
        {"ffmpeg", "-v", "error", "-y", "-ss", timestamp.str(), "-i",
         video.string(), "-frames:v", "1", out.string()});
    if (capture.exit_status == 0 && std::filesystem::exists(out))
      paths.push_back(out);
  }
  return paths;
}

std::optional<std::filesystem::path>
extract_audio(const std::filesystem::path &video,
              const std::filesystem::path &directory, const std::string &name) {
  const auto out = directory / (name + ".wav");
  const auto capture = dif::bench::run_capture(
      {"ffmpeg", "-v", "error", "-y", "-i", video.string(), "-vn", "-acodec",
       "pcm_s16le", out.string()});
  if (capture.exit_status == 0 && std::filesystem::exists(out))
    return out;
  return std::nullopt;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      usage();
      return 2;
    }
    std::filesystem::path candidate = argv[1];
    std::filesystem::path reference;
    std::string kind;
    Bars bars;
    std::uint32_t frames = 8U;
    bool use_ffmpeg = true;
    std::string reviewer;
    std::string review;
    std::string review_note;
    bool json = false;
    std::filesystem::path report;
    for (int index = 2; index < argc; ++index) {
      const std::string option = argv[index];
      const auto value = [&]() -> std::string {
        if (index + 1 >= argc)
          dif::fail("missing value after " + option);
        return argv[++index];
      };
      if (option == "--reference")
        reference = value();
      else if (option == "--kind")
        kind = value();
      else if (option == "--min-psnr")
        bars.min_psnr = number(value(), "minimum PSNR");
      else if (option == "--min-ssim")
        bars.min_ssim = number(value(), "minimum SSIM");
      else if (option == "--min-snr")
        bars.min_snr = number(value(), "minimum SNR");
      else if (option == "--frames")
        frames = static_cast<std::uint32_t>(std::stoul(value()));
      else if (option == "--no-ffmpeg")
        use_ffmpeg = false;
      else if (option == "--reviewer")
        reviewer = value();
      else if (option == "--review") {
        review = value();
        if (review != "accept" && review != "reject")
          dif::fail("--review accepts accept or reject");
      } else if (option == "--review-note")
        review_note = value();
      else if (option == "--json")
        json = true;
      else if (option == "--report")
        report = value();
      else if (option == "--help" || option == "-h") {
        usage();
        return 0;
      } else
        dif::fail("unknown difquality option: " + option);
    }
    if (!review.empty() && reviewer.empty())
      dif::fail("a recorded review needs --reviewer");

    auto document = dif::telemetry::make_document("quality-report");
    const auto facts = dif::bench::inspect_artifact(candidate);
    if (kind.empty()) {
      kind = kind_from_facts(facts);
      if (kind.empty() && candidate.extension() == ".wav")
        kind = "audio";
    }
    document.set("kind_under_test", kind.empty() ? "unknown" : kind);
    document.set("candidate", dif::bench::artifact_section(candidate, facts, false));
    if (!reference.empty())
      document.set("reference",
                   dif::bench::artifact_section(
                       reference, dif::bench::inspect_artifact(reference), false));
    else
      document.set("reference", nullptr);
    dif::telemetry::Object bars_section;
    bars_section.set("min_psnr_db", bars.min_psnr);
    bars_section.set("min_ssim_8x8", bars.min_ssim);
    bars_section.set("min_snr_db", bars.min_snr);
    document.set("bars", std::move(bars_section));

    Judgement judgement;
    dif::telemetry::Object metrics;
    if (!facts.exists) {
      judgement.failures.push_back("candidate artifact does not exist");
    } else if (kind == "image") {
      if (facts.format != "png") {
        judgement.failures.push_back("candidate is not a PNG");
      } else {
        const auto image = dif::read_png_rgb8(candidate);
        const auto sanity = dif::quality::image_sanity(image);
        metrics.set("sanity", image_sanity_section(sanity));
        std::optional<dif::quality::ImageComparison> comparison;
        if (!reference.empty()) {
          const auto expected = dif::read_png_rgb8(reference);
          comparison = dif::quality::compare_images(expected, image);
          metrics.set("comparison", image_comparison_section(*comparison));
        }
        judge_image(judgement, bars, sanity, comparison ? &*comparison : nullptr);
      }
    } else if (kind == "audio") {
      const auto waveform = dif::quality::read_wav_pcm16(candidate);
      const auto sanity = dif::quality::audio_sanity(waveform);
      metrics.set("sanity", audio_sanity_section(sanity));
      std::optional<dif::quality::AudioComparison> comparison;
      if (!reference.empty()) {
        const auto expected = dif::quality::read_wav_pcm16(reference);
        comparison = dif::quality::compare_audio(expected, waveform);
        metrics.set("comparison", audio_comparison_section(*comparison));
      }
      judge_audio(judgement, bars, sanity, comparison ? &*comparison : nullptr);
    } else if (kind == "video") {
      if (facts.format != "mp4") {
        judgement.failures.push_back("candidate is not an MP4 with a movie header");
      } else if (facts.track_count == 0U || facts.duration_seconds <= 0.0) {
        judgement.failures.push_back("candidate MP4 has no tracks or zero duration");
      } else {
        dif::telemetry::Object structure;
        structure.set("duration_seconds", facts.duration_seconds);
        structure.set("track_count", facts.track_count);
        metrics.set("structure", std::move(structure));
        std::optional<dif::bench::ArtifactFacts> reference_facts;
        if (!reference.empty()) {
          reference_facts = dif::bench::inspect_artifact(reference);
          if (reference_facts->format != "mp4")
            judgement.failures.push_back("reference is not an MP4");
          else if (std::fabs(reference_facts->duration_seconds - facts.duration_seconds) >
                   0.5)
            judgement.failures.push_back("duration differs from the reference by more "
                                         "than 0.5 s");
        }
        const bool ffmpeg = use_ffmpeg && tool_available("ffmpeg");
        metrics.set("ffmpeg_sampling", ffmpeg);
        if (!ffmpeg) {
          judgement.notes.push_back(
              "ffmpeg unavailable or disabled: no frame or audio metrics; "
              "structural facts only");
        } else {
          TempDir scratch;
          const auto candidate_frames = sample_frames(
              candidate, facts.duration_seconds, frames, scratch.path, "candidate-");
          dif::telemetry::Array frame_entries;
          double min_psnr = std::numeric_limits<double>::infinity();
          double min_ssim = 1.0;
          bool any_constant = false;
          std::vector<std::filesystem::path> reference_frames;
          if (reference_facts && reference_facts->format == "mp4")
            reference_frames = sample_frames(reference, reference_facts->duration_seconds,
                                             frames, scratch.path, "reference-");
          for (std::size_t index = 0; index < candidate_frames.size(); ++index) {
            const auto image = dif::read_png_rgb8(candidate_frames[index]);
            const auto sanity = dif::quality::image_sanity(image);
            any_constant = any_constant || !sanity.problem.empty();
            dif::telemetry::Object entry;
            entry.set("index", index);
            entry.set("sanity", image_sanity_section(sanity));
            if (index < reference_frames.size()) {
              const auto expected = dif::read_png_rgb8(reference_frames[index]);
              const auto comparison = dif::quality::compare_images(expected, image);
              entry.set("comparison", image_comparison_section(comparison));
              if (comparison.comparable) {
                min_psnr = std::min(min_psnr, comparison.psnr_db);
                min_ssim = std::min(min_ssim, comparison.ssim_8x8);
              } else {
                judgement.failures.push_back("frame " + std::to_string(index) + ": " +
                                             comparison.problem);
              }
            }
            frame_entries.push_back(std::move(entry));
          }
          metrics.set("sampled_frames", std::move(frame_entries));
          metrics.set("sampled_frame_count", candidate_frames.size());
          if (candidate_frames.empty())
            judgement.failures.push_back("no frame could be decoded from the candidate");
          if (any_constant)
            judgement.failures.push_back("a sampled frame is constant");
          if (!reference_frames.empty() && !candidate_frames.empty()) {
            dif::telemetry::Object worst;
            worst.set("min_psnr_db", finite_or_null(min_psnr));
            worst.set("min_ssim_8x8", min_ssim);
            metrics.set("worst_frame", std::move(worst));
            bool ok = true;
            if (min_psnr < bars.min_psnr) {
              judgement.failures.push_back("worst sampled frame PSNR below bar");
              ok = false;
            }
            if (min_ssim < bars.min_ssim) {
              judgement.failures.push_back("worst sampled frame SSIM below bar");
              ok = false;
            }
            judgement.numeric_admission = ok ? "pass" : "fail";
          }
          if (const auto audio = extract_audio(candidate, scratch.path, "candidate")) {
            const auto waveform = dif::quality::read_wav_pcm16(*audio);
            const auto sanity = dif::quality::audio_sanity(waveform);
            dif::telemetry::Object audio_section;
            audio_section.set("sanity", audio_sanity_section(sanity));
            if (!sanity.problem.empty())
              judgement.failures.push_back("audio track sanity: " + sanity.problem);
            if (reference_facts && reference_facts->format == "mp4") {
              if (const auto expected_audio =
                      extract_audio(reference, scratch.path, "reference")) {
                const auto expected = dif::quality::read_wav_pcm16(*expected_audio);
                const auto comparison = dif::quality::compare_audio(expected, waveform);
                audio_section.set("comparison", audio_comparison_section(comparison));
                if (comparison.comparable && comparison.snr_db < bars.min_snr) {
                  judgement.failures.push_back("audio SNR below bar");
                  judgement.numeric_admission = "fail";
                } else if (!comparison.comparable) {
                  judgement.failures.push_back("audio comparison: " + comparison.problem);
                }
              }
            }
            metrics.set("audio", std::move(audio_section));
          } else {
            metrics.set("audio", nullptr);
          }
        }
      }
    } else {
      judgement.failures.push_back("unrecognized artifact kind; pass --kind");
    }
    document.set("metrics", std::move(metrics));

    dif::telemetry::Object perceptual;
    perceptual.set("recorded", !review.empty());
    perceptual.set("reviewer", reviewer);
    perceptual.set("verdict", review);
    perceptual.set("note", review_note);
    document.set("perceptual_review", std::move(perceptual));

    std::string verdict;
    std::string statement;
    if (!judgement.failures.empty() || review == "reject") {
      verdict = "FAIL";
      statement = review == "reject" ? "rejected by the recorded perceptual review"
                                     : "a numeric or structural check failed";
    } else if (review == "accept") {
      verdict = "PASS";
      statement = judgement.numeric_admission == "pass"
                      ? "numeric admission holds and a perceptual review accepted "
                        "the artifact"
                      : "sanity holds and a perceptual review accepted the "
                        "artifact; no reference metrics were computed";
    } else {
      verdict = "MANUAL REVIEW REQUIRED";
      statement = judgement.numeric_admission == "pass"
                      ? "numeric admission holds; a perceptual review must be "
                        "recorded before PASS"
                      : "no numeric admission was computed; a perceptual review "
                        "must be recorded before PASS";
    }
    dif::telemetry::Object result;
    result.set("numeric_admission", judgement.numeric_admission);
    dif::telemetry::Array failures;
    for (const auto &failure : judgement.failures)
      failures.push_back(failure);
    result.set("failures", std::move(failures));
    dif::telemetry::Array notes;
    for (const auto &note : judgement.notes)
      notes.push_back(note);
    result.set("notes", std::move(notes));
    result.set("verdict", verdict);
    result.set("statement", statement);
    document.set("result", std::move(result));

    const auto text = dif::telemetry::serialize(dif::telemetry::Value(document));
    if (!report.empty()) {
      std::ofstream stream(report, std::ios::binary | std::ios::trunc);
      stream.write(text.data(), static_cast<std::streamsize>(text.size()));
    }
    if (json) {
      std::cout << text;
    } else {
      std::cout << "candidate " << candidate.string() << " kind=" << kind << "\n";
      for (const auto &failure : judgement.failures)
        std::cout << "  failure: " << failure << "\n";
      for (const auto &note : judgement.notes)
        std::cout << "  note: " << note << "\n";
      std::cout << "numeric_admission=" << judgement.numeric_admission
                << " review=" << (review.empty() ? "none" : review) << "\n"
                << "DIFQUALITY " << verdict << ": " << statement << "\n";
    }
    return verdict == "FAIL" ? 1 : 0;
  } catch (const std::exception &error) {
    std::cerr << "difquality: " << error.what() << "\n";
    return 1;
  }
}
