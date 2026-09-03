#include "dif/frontend/flux2_prompt.hpp"

#include "dif/support/error.hpp"

#include <algorithm>

namespace dif::frontend {

std::string render_flux2_qwen_user_chat(std::string_view prompt) {
  constexpr std::string_view prefix = "<|im_start|>user\n";
  constexpr std::string_view suffix =
      "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
  std::string rendered;
  rendered.reserve(prefix.size() + prompt.size() + suffix.size());
  rendered.append(prefix);
  rendered.append(prompt);
  rendered.append(suffix);
  return rendered;
}

std::string render_flux2_mistral_chat(std::string_view prompt,
                                      std::string_view system_message) {
  std::string rendered;
  rendered.reserve(prompt.size() + system_message.size() + 64U);
  rendered.append("<s>[SYSTEM_PROMPT]");
  rendered.append(system_message);
  rendered.append("[/SYSTEM_PROMPT][INST]");
  rendered.append(prompt);
  rendered.append("[/INST]");
  return rendered;
}

Flux2PromptInputs make_flux2_mistral_prompt_inputs(
    const text::QwenBpeTokenizer &tokenizer, std::string_view prompt,
    std::size_t max_length, std::int32_t pad_token_id) {
  if (max_length == 0U)
    fail("FLUX.2 Mistral prompt length must be nonzero");
  if (pad_token_id < 0 || pad_token_id >= tokenizer.id_space())
    fail("FLUX.2 Mistral padding token is outside the tokenizer id space");
  Flux2PromptInputs result;
  result.rendered_chat = render_flux2_mistral_chat(prompt);
  auto ids = tokenizer.encode(result.rendered_chat);
  if (ids.size() > max_length)
    ids.resize(max_length);
  result.valid_tokens = ids.size();
  const auto padding = max_length - ids.size();
  result.input_ids.assign(max_length, pad_token_id);
  std::copy(ids.begin(), ids.end(),
            result.input_ids.begin() + static_cast<std::ptrdiff_t>(padding));
  result.attention_mask.assign(max_length, 0U);
  std::fill(result.attention_mask.begin() + static_cast<std::ptrdiff_t>(padding),
            result.attention_mask.end(), 1U);
  return result;
}

Flux2PromptInputs make_flux2_qwen_prompt_inputs(
    const text::QwenBpeTokenizer &tokenizer, std::string_view prompt,
    std::size_t max_length, std::int32_t pad_token_id) {
  if (max_length == 0U)
    fail("FLUX.2 Qwen prompt length must be nonzero");
  if (pad_token_id < 0 || pad_token_id >= tokenizer.id_space())
    fail("FLUX.2 Qwen padding token is outside the tokenizer id space");

  Flux2PromptInputs result;
  result.rendered_chat = render_flux2_qwen_user_chat(prompt);
  result.input_ids = tokenizer.encode(result.rendered_chat);
  if (result.input_ids.size() > max_length)
    result.input_ids.resize(max_length);
  result.valid_tokens = result.input_ids.size();
  result.attention_mask.assign(max_length, 0U);
  std::fill(result.attention_mask.begin(),
            result.attention_mask.begin() + result.valid_tokens, 1U);
  result.input_ids.resize(max_length, pad_token_id);
  return result;
}

} // namespace dif::frontend
