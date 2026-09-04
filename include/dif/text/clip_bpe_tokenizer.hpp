#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace dif::text {

// Native CLIP byte-level BPE tokenizer: text -> [BOS, ids..., EOS].
//
// Serves both SDXL text towers: CLIP-L (openai/clip-vit-large-patch14) and
// OpenCLIP-G (laion CLIP-ViT-bigG-14) share the same 49408-entry vocab and
// merge list, so ONE loaded tokenizer feeds both.
//
// Loads the standard pair shipped with the creator's tokenizer directory:
//   vocab.json   token string -> id ("cat</w>" -> 2368, "<|startoftext|>"
//                -> 49406, "<|endoftext|>" -> 49407)
//   merges.txt   "#version" comment line, then one "left right" pair per
//                line; the merge rank is the line order.
// The loader is FAIL-CLOSED: malformed lines, missing BOS/EOS entries, or a
// vocab that lacks any byte-level symbol (which would make encode() miss)
// are rejected with dif::fail.
//
// The algorithm is a port of the proven Mojo clip_tokenizer.mojo, which
// reproduces the creator's tokenizer (HuggingFace CLIPTokenizer):
//   1. whitespace_clean + lowercase: runs of whitespace collapse to one
//      space, leading/trailing whitespace is dropped.
//   2. The CLIP word-split regex, emulated in alternation order:
//        's|'t|'re|'ve|'m|'ll|'d | [\p{L}]+ | [\p{N}] | [^\s\p{L}\p{N}]+
//      (letter runs, SINGLE digits, punctuation runs; whitespace separates).
//   3. GPT-2 bytes_to_unicode: every UTF-8 byte -> one printable codepoint.
//   4. BPE per pre-token with "</w>" appended to the last symbol,
//      lowest-rank-first, merging every occurrence of the pair per round.
//   5. Vocab lookup; BOS/EOS wrap.
//
// Approximations inherited from the Mojo implementation (kept identical so
// the two implementations agree; exercised by tests/clip_tokenizer_tests):
//   * \p{L}, \p{N}, \s are codepoint-range approximations: exact for ASCII,
//     Latin, Greek, Cyrillic, CJK and the other scripts listed in the .cpp.
//   * Lowercasing covers ASCII and the Latin-1 supplement only; the
//     reference folds all of Unicode (Greek/Cyrillic capitals diverge).
//   * No NFC normalization: exact for NFC-stable (precomposed) input.
//   * A literal "<|startoftext|>" / "<|endoftext|>" inside the prompt is
//     byte-encoded like any other text instead of matched as one token.
class ClipBpeTokenizer {
public:
  static ClipBpeTokenizer load(const std::filesystem::path &vocab_json,
                               const std::filesystem::path &merges_txt);

  // [BOS, ids..., EOS]; no padding (see clip_prompt_tokens for the 77-slot
  // batching contract).
  std::vector<std::uint32_t> encode(std::string_view text) const;

  std::uint32_t bos_id() const { return bos_id_; }
  std::uint32_t eos_id() const { return eos_id_; }
  std::size_t vocab_size() const { return vocab_.size(); }

private:
  ClipBpeTokenizer() = default;

  std::vector<std::string> bpe(std::vector<std::string> word) const;

  std::unordered_map<std::string, std::uint32_t> vocab_;
  std::unordered_map<std::string, std::uint32_t> merge_ranks_;
  std::string byte_map_[256];
  std::uint32_t bos_id_{};
  std::uint32_t eos_id_{};
};

// Stages of the tokenizer exposed for checkpoint-free tests.
//
// GPT-2 bytes_to_unicode: the codepoint the byte-level alphabet uses for
// `byte` (printable Latin-1 bytes map to themselves, the rest to U+0100+).
char32_t clip_byte_to_unicode(unsigned char byte);
// whitespace_clean + lowercase + the CLIP word-split regex; the pre-tokens
// are returned as UTF-8 strings (before the byte-level remap).
std::vector<std::string> clip_pretokenize(std::string_view text);

// The reference sampler's prompt contract for one CLIP tower, for plain
// text (no attention-weight syntax, no textual-inversion embeddings):
//
//   ids            max_length * chunks slots. Each chunk is
//                  [BOS] + body tokens + [EOS] + `pad_token` fill. The
//                  whole prompt is ONE token group; a group of >= 8 tokens
//                  is split across chunks (`max_length - 2` body tokens per
//                  chunk, the tail into a new chunk).
//   valid_tokens   position of the first EOS in the FIRST chunk + 1, i.e.
//                  1 + body tokens in that chunk + 1 — the reference's
//                  num_tokens, which selects the pooled output.
struct ClipPromptTokens {
  std::vector<std::int32_t> ids;
  std::uint64_t valid_tokens{};
};

ClipPromptTokens clip_prompt_tokens(const ClipBpeTokenizer &tokenizer,
                                    std::string_view text,
                                    std::uint32_t pad_token,
                                    std::uint64_t max_length = 77);

// SDXL pair: CLIP-L pads with the EOS id (the reference's pad_with_end),
// CLIP-G pads with 0. Same tokenizer files serve both towers.
struct SdxlPromptTokens {
  ClipPromptTokens l;
  ClipPromptTokens g;
};

inline constexpr std::uint32_t kSdxlClipGPadToken = 0;

SdxlPromptTokens sdxl_prompt_tokens(const ClipBpeTokenizer &tokenizer,
                                    std::string_view text);

} // namespace dif::text
