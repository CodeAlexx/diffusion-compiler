#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace dif::text {

struct AddedToken {
  std::string content;
  std::int32_t id{};
  bool special{};
};

// Native Qwen-family byte-level BPE tokenizer.
//
// Loads a checkpoint's `processor/tokenizer.json` (HF `tokenizers` format)
// plus, when given, `tokenizer_config.json`, whose `additional_special_tokens`
// can declare tokens that exist in NEITHER the vocab NOR `added_tokens` —
// MiniMax-H3 ships seven such tokens (`<d>`, `</d>`, `<|cutoff|>`,
// `<|lyrics_start|>`, `<|lyrics_end|>`, `<|caption_start|>`,
// `<|caption_end|>`). transformers appends them at load time (measured:
// ids 151669..151675, in config order), and H3 prompts use `<d>` dialogue
// markup, so skipping the config changes the ids and therefore the video.
//
// The loader is FAIL-CLOSED: it verifies the file describes exactly the
// pipeline this implementation reproduces — NFC normalizer, the Qwen2 Split
// regex (Isolated) + ByteLevel(add_prefix_space=false, use_regex=false)
// pre-tokenizer, and a plain BPE model (no dropout, no unk, no byte
// fallback, no subword prefixes) — and rejects anything else.
//
// encode() reproduces `transformers` `Qwen2TokenizerFast` with
// add_special_tokens=False: added tokens are split out of the raw text
// first (leftmost scan, longest match at a position; `normalized=false`),
// then each remaining segment goes through NFC -> regex pre-tokenization ->
// GPT-2 byte-level remap -> lowest-rank-first BPE -> vocab lookup.
class QwenBpeTokenizer {
public:
  // `tokenizer_config_json` may be empty to skip the additional-special-token
  // merge (NOT correct for MiniMax-H3; provided for other Qwen checkpoints).
  static QwenBpeTokenizer load(const std::filesystem::path &tokenizer_json,
                               const std::filesystem::path &tokenizer_config_json);

  std::vector<std::int32_t> encode(std::string_view utf8_text) const;

  std::size_t base_vocab_size() const { return vocab_.size(); }
  const std::vector<AddedToken> &added_tokens() const { return added_; }
  // Total token-id space: max id + 1 (vocab plus added tokens).
  std::int32_t id_space() const;

private:
  QwenBpeTokenizer() = default;

  void encode_segment(std::string_view utf8_segment,
                      std::vector<std::int32_t> &ids) const;
  std::vector<std::string> bpe(std::vector<std::string> word) const;

  std::unordered_map<std::string, std::int32_t> vocab_;
  std::unordered_map<std::string, std::uint32_t> merge_ranks_;
  std::vector<AddedToken> added_;
  std::string byte_map_[256];
};

} // namespace dif::text
