#pragma once

#include "dif/text/qwen_bpe_tokenizer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace dif::frontend {

// Exact creator Qwen3 prompt surface used by FLUX.2 [klein]. The creator
// supplies one user message, requests an assistant generation prompt, and
// disables thinking. Chat policy belongs to this frontend, not the generic
// tokenizer or runtime.
std::string render_flux2_qwen_user_chat(std::string_view prompt);

struct Flux2PromptInputs {
  std::string rendered_chat;
  std::vector<std::int32_t> input_ids;
  std::vector<std::uint8_t> attention_mask;
  std::size_t valid_tokens{};
};

// Reproduce creator right truncation and right max-length padding. The pinned
// Qwen3 tokenizer uses <|endoftext|> id 151643 as its padding token.
Flux2PromptInputs make_flux2_qwen_prompt_inputs(
    const text::QwenBpeTokenizer &tokenizer, std::string_view prompt,
    std::size_t max_length = 512U, std::int32_t pad_token_id = 151643);

} // namespace dif::frontend
