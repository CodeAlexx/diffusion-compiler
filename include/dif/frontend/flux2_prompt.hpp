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

// FLUX.2 [dev]: the Mistral Small 3.1 chat template with the pipeline's
// system message and the user prompt, add_generation_prompt=False, i.e.
//   <s>[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST]{prompt}[/INST]
// (the BOS text stands in for the tokenizer's TemplateProcessing, exactly as
// apply_chat_template(tokenize=True) does).
inline constexpr std::string_view kFlux2DevSystemMessage =
    "You are an AI that reasons about image descriptions. You give "
    "structured responses focusing on object relationships, object "
    "attribution and actions without speculation.";
std::string render_flux2_mistral_chat(std::string_view prompt,
                                      std::string_view system_message =
                                          kFlux2DevSystemMessage);

struct Flux2PromptInputs {
  std::string rendered_chat;
  std::vector<std::int32_t> input_ids;
  std::vector<std::uint8_t> attention_mask;
  std::size_t valid_tokens{};
};

// Reproduce creator right truncation and right max-length padding. The pinned
// Qwen3 tokenizer uses <|endoftext|> id 151643 as its padding token.
// Mistral side: LlamaTokenizerFast pads on the LEFT (pad id 11 `<pad>`),
// so the valid tokens occupy the last `valid_tokens` positions; positions are
// still 0..max_length-1 over the padded sequence (transformers cache_position).
Flux2PromptInputs make_flux2_mistral_prompt_inputs(
    const text::QwenBpeTokenizer &tokenizer, std::string_view prompt,
    std::size_t max_length = 512U, std::int32_t pad_token_id = 11);

Flux2PromptInputs make_flux2_qwen_prompt_inputs(
    const text::QwenBpeTokenizer &tokenizer, std::string_view prompt,
    std::size_t max_length = 512U, std::int32_t pad_token_id = 151643);

} // namespace dif::frontend
