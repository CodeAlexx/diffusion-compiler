#include "dif/frontend/flux2_prompt.hpp"
#include "dif/support/error.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const std::string &message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << "\n";
  }
}

template <std::size_t N>
void verify_case(const dif::text::QwenBpeTokenizer &tokenizer,
                 std::string_view name, std::string_view prompt,
                 std::string_view rendered,
                 const std::array<std::int32_t, N> &valid_ids) {
  const auto inputs =
      dif::frontend::make_flux2_qwen_prompt_inputs(tokenizer, prompt);
  expect(inputs.rendered_chat == rendered,
         std::string(name) + " rendered chat is creator-exact");
  expect(inputs.input_ids.size() == 512U,
         std::string(name) + " input length is 512");
  expect(inputs.attention_mask.size() == 512U,
         std::string(name) + " mask length is 512");
  expect(inputs.valid_tokens == valid_ids.size(),
         std::string(name) + " valid token count");
  for (std::size_t index = 0U; index < valid_ids.size(); ++index) {
    expect(inputs.input_ids[index] == valid_ids[index],
           std::string(name) + " token " + std::to_string(index));
    expect(inputs.attention_mask[index] == 1U,
           std::string(name) + " valid mask " + std::to_string(index));
  }
  for (std::size_t index = valid_ids.size(); index < 512U; ++index) {
    expect(inputs.input_ids[index] == 151643,
           std::string(name) + " right padding " + std::to_string(index));
    expect(inputs.attention_mask[index] == 0U,
           std::string(name) + " padding mask " + std::to_string(index));
  }
}

} // namespace

int main() {
  try {
    // The pinned Qwen3 tokenizer: DIF_FLUX2_QWEN_TOKENIZER, else the intake
    // cache, else the tokenizer/ directory of the HF FLUX.2 Klein snapshot
    // (identical tokenizer.json; the intake cache lived on a drive that died).
    const char *root_env = std::getenv("DIF_FLUX2_QWEN_TOKENIZER");
    std::vector<std::filesystem::path> candidates;
    if (root_env != nullptr)
      candidates.emplace_back(root_env);
    candidates.emplace_back(
        "/mnt/disk1/diffusion-compiler-cache/flux2-intake/qwen3-8b-fp8");
    if (const char *home = std::getenv("HOME"); home != nullptr)
      candidates.emplace_back(
          std::filesystem::path(home) /
          ".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-base-9B/"
          "snapshots/32773329fbe7e81a90ef971740e8ba4b0364ecf3/tokenizer");
    std::filesystem::path root;
    for (const auto &candidate : candidates)
      if (std::filesystem::exists(candidate / "tokenizer.json") &&
          std::filesystem::exists(candidate / "tokenizer_config.json")) {
        root = candidate;
        break;
      }
    if (root.empty()) {
      std::cout << "SKIP: pinned FLUX.2 Qwen tokenizer not found under any of "
                << candidates.size() << " candidate paths\n";
      return 77;
    }
    const auto tokenizer_json = root / "tokenizer.json";
    const auto tokenizer_config = root / "tokenizer_config.json";

    const auto tokenizer = dif::text::QwenBpeTokenizer::load(
        tokenizer_json, tokenizer_config);
    constexpr std::string_view empty_rendered =
        "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n";
    constexpr std::array<std::int32_t, 12> empty_ids{
        151644, 872, 198, 151645, 198, 151644,
        77091,  198, 151667, 271, 151668, 271};
    verify_case(tokenizer, "empty CFG", "", empty_rendered, empty_ids);

    constexpr std::string_view prompt =
        "A cat holding a sign that says hello world";
    constexpr std::string_view literal_rendered =
        "<|im_start|>user\nA cat holding a sign that says hello world"
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    constexpr std::array<std::int32_t, 21> literal_ids{
        151644, 872,    198,    32,     8251,   9963, 264,
        1841,   429,    2727,   23811,  1879,   151645,
        198,    151644, 77091,  198,    151667, 271,
        151668, 271};
    verify_case(tokenizer, "literal prompt", prompt, literal_rendered,
                literal_ids);

    if (failures != 0) {
      std::cerr << failures << " failure(s)\n";
      return 1;
    }
    std::cout << "FLUX.2 creator prompt parity passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FLUX.2 prompt test: " << error.what() << "\n";
    return 1;
  }
}
