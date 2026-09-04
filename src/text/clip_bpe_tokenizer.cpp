#include "dif/text/clip_bpe_tokenizer.hpp"

#include "dif/support/error.hpp"
#include "dif/support/json.hpp"
#include "dif/text/unicode.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dif::text {

namespace {

constexpr std::string_view kBosToken = "<|startoftext|>";
constexpr std::string_view kEosToken = "<|endoftext|>";
constexpr std::string_view kEndOfWord = "</w>";

// ---- Unicode category approximations -----------------------------------
// Ported verbatim from the Mojo tokenizer so both implementations split
// identically. They stand in for the reference regex's \p{N}, \p{L} and \s;
// the Mojo header documents them as exact for ASCII, Latin, Greek,
// Cyrillic, CJK and the other scripts listed below.

// \p{N}: ASCII digits. The CLIP regex matches a SINGLE numeral.
bool clip_is_digit(char32_t cp) { return cp >= U'0' && cp <= U'9'; }

// \p{L} covering the scripts the encoder realistically sees.
bool clip_is_letter(char32_t cp) {
  if (cp < 0x80U)
    return (cp >= U'A' && cp <= U'Z') || (cp >= U'a' && cp <= U'z');
  if (cp >= 0x00C0U && cp <= 0x00FFU)
    return cp != 0x00D7U && cp != 0x00F7U; // Latin-1 letters minus x and /
  if (cp == 0x00AAU || cp == 0x00B5U || cp == 0x00BAU)
    return true;
  struct Range {
    char32_t lo;
    char32_t hi;
  };
  static constexpr Range kRanges[] = {
      {0x0100U, 0x02AFU},   // Latin Extended-A/B, IPA
      {0x0370U, 0x03FFU},   // Greek and Coptic
      {0x0400U, 0x04FFU},   // Cyrillic
      {0x0530U, 0x058FU},   // Armenian
      {0x0590U, 0x05FFU},   // Hebrew
      {0x0600U, 0x06FFU},   // Arabic
      {0x0900U, 0x097FU},   // Devanagari
      {0x0E00U, 0x0E7FU},   // Thai
      {0x1100U, 0x11FFU},   // Hangul Jamo
      {0x3040U, 0x30FFU},   // Hiragana + Katakana
      {0x3400U, 0x4DBFU},   // CJK Extension A
      {0x4E00U, 0x9FFFU},   // CJK Unified Ideographs
      {0xAC00U, 0xD7A3U},   // Hangul Syllables
      {0xF900U, 0xFAFFU},   // CJK Compatibility Ideographs
      {0x20000U, 0x2FA1FU}, // CJK Extensions B..F
  };
  for (const Range &range : kRanges)
    if (cp >= range.lo && cp <= range.hi)
      return true;
  return false;
}

// \s: the Unicode White_Space set.
bool clip_is_whitespace(char32_t cp) {
  switch (cp) {
  case 0x0009U:
  case 0x000AU:
  case 0x000BU:
  case 0x000CU:
  case 0x000DU:
  case 0x0020U:
  case 0x0085U:
  case 0x00A0U:
  case 0x1680U:
  case 0x2028U:
  case 0x2029U:
  case 0x202FU:
  case 0x205FU:
  case 0x3000U:
    return true;
  default:
    return cp >= 0x2000U && cp <= 0x200AU;
  }
}

// ASCII A-Z plus the Latin-1 supplement capitals (A-grave..Thorn, excluding
// the multiplication sign). The reference lowercases all of Unicode; this
// is the documented divergence for Greek/Cyrillic capitals.
char32_t clip_lower(char32_t cp) {
  if (cp >= U'A' && cp <= U'Z')
    return cp + 32U;
  if (cp >= 0x00C0U && cp <= 0x00DEU && cp != 0x00D7U)
    return cp + 32U;
  return cp;
}

std::string read_file(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    dif::fail("cannot open " + path.string());
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return std::move(buffer).str();
}

std::uint32_t to_token_id(double value, const std::string &token) {
  if (!(value >= 0.0) || value > 4294967295.0 || value != std::floor(value))
    dif::fail("vocab.json: id of '" + token + "' is not a non-negative u32");
  return static_cast<std::uint32_t>(value);
}

// whitespace_clean(text).lower(): whitespace runs collapse to one space,
// leading/trailing whitespace is dropped, letters are lowercased.
std::vector<char32_t> normalize(const std::vector<char32_t> &cps) {
  std::vector<char32_t> out;
  out.reserve(cps.size());
  bool pending_space = false;
  for (const char32_t cp : cps) {
    if (clip_is_whitespace(cp)) {
      pending_space = true;
      continue;
    }
    if (pending_space && !out.empty())
      out.push_back(U' ');
    pending_space = false;
    out.push_back(clip_lower(cp));
  }
  return out;
}

// The CLIP word-split regex, alternation order preserved:
//   's|'t|'re|'ve|'m|'ll|'d | [\p{L}]+ | [\p{N}] | [^\s\p{L}\p{N}]+
// Returns [begin, end) codepoint ranges over the normalized text.
std::vector<std::pair<std::size_t, std::size_t>>
split(const std::vector<char32_t> &s) {
  std::vector<std::pair<std::size_t, std::size_t>> out;
  const std::size_t n = s.size();
  std::size_t i = 0;
  while (i < n) {
    const char32_t cp = s[i];
    // 1) contractions. The text is already lowercased, so the reference's
    //    case-insensitive flag needs no counterpart here.
    if (cp == U'\'') {
      const char32_t c1 = i + 1 < n ? s[i + 1] : U'\0';
      const char32_t c2 = i + 2 < n ? s[i + 2] : U'\0';
      std::size_t len = 0;
      if (c1 == U's' || c1 == U't' || c1 == U'm' || c1 == U'd')
        len = 2;
      else if ((c1 == U'r' && c2 == U'e') || (c1 == U'v' && c2 == U'e') ||
               (c1 == U'l' && c2 == U'l'))
        len = 3;
      if (len != 0) {
        out.emplace_back(i, i + len);
        i += len;
        continue;
      }
      // A lone apostrophe falls through to the punctuation rule.
    }
    // 2) letter run: [\p{L}]+
    if (clip_is_letter(cp)) {
      std::size_t j = i + 1;
      while (j < n && clip_is_letter(s[j]))
        ++j;
      out.emplace_back(i, j);
      i = j;
      continue;
    }
    // 3) single digit: [\p{N}]
    if (clip_is_digit(cp)) {
      out.emplace_back(i, i + 1);
      ++i;
      continue;
    }
    // 4) whitespace is captured by no alternative: it separates.
    if (clip_is_whitespace(cp)) {
      ++i;
      continue;
    }
    // 5) punctuation / other run: [^\s\p{L}\p{N}]+. The reference matches
    //    this greedily, so an apostrophe inside a punctuation run is
    //    swallowed here ("x!'s" -> "x", "!'", "s"); the contraction rule
    //    above only fires when the apostrophe starts a match.
    std::size_t j = i;
    while (j < n && !clip_is_whitespace(s[j]) && !clip_is_letter(s[j]) &&
           !clip_is_digit(s[j]))
      ++j;
    out.emplace_back(i, j);
    i = j;
  }
  return out;
}

// GPT-2 bytes_to_unicode: printable Latin-1 bytes map to themselves, the
// rest to U+0100 + running index, in ascending byte order.
const std::array<char32_t, 256> &byte_to_unicode_table() {
  static const std::array<char32_t, 256> table = [] {
    std::array<char32_t, 256> out{};
    char32_t next = 0;
    for (unsigned b = 0; b < 256U; ++b) {
      const bool direct = (b >= 33U && b <= 126U) ||
                          (b >= 161U && b <= 172U) || (b >= 174U && b <= 255U);
      out[b] = direct ? static_cast<char32_t>(b) : 256U + next++;
    }
    return out;
  }();
  return table;
}

} // namespace

char32_t clip_byte_to_unicode(unsigned char byte) {
  return byte_to_unicode_table()[byte];
}

std::vector<std::string> clip_pretokenize(std::string_view text) {
  const std::vector<char32_t> norm = normalize(unicode::decode_utf8(text));
  std::vector<std::string> out;
  for (const auto &[begin, end] : split(norm)) {
    std::string piece;
    for (std::size_t k = begin; k < end; ++k)
      unicode::append_utf8(piece, norm[k]);
    out.push_back(std::move(piece));
  }
  return out;
}

ClipBpeTokenizer
ClipBpeTokenizer::load(const std::filesystem::path &vocab_json,
                       const std::filesystem::path &merges_txt) {
  ClipBpeTokenizer tok;

  const dif::json::Value root = dif::json::parse(read_file(vocab_json));
  if (!root.is_object())
    dif::fail("vocab.json: top level must be an object of token -> id");
  const auto &vocab = root.object();
  tok.vocab_.reserve(vocab.size());
  for (const auto &[token, id] : vocab)
    tok.vocab_.emplace(token, to_token_id(id.number(), token));

  const auto bos = tok.vocab_.find(std::string(kBosToken));
  const auto eos = tok.vocab_.find(std::string(kEosToken));
  if (bos == tok.vocab_.end() || eos == tok.vocab_.end())
    dif::fail("vocab.json: missing <|startoftext|> / <|endoftext|>");
  tok.bos_id_ = bos->second;
  tok.eos_id_ = eos->second;

  // Every byte-level symbol and its "</w>" form must be in the vocab: that
  // is what guarantees encode() can never miss a lookup.
  for (unsigned b = 0; b < 256U; ++b) {
    unicode::append_utf8(tok.byte_map_[b], clip_byte_to_unicode(
                                               static_cast<unsigned char>(b)));
    if (tok.vocab_.count(tok.byte_map_[b]) == 0U ||
        tok.vocab_.count(tok.byte_map_[b] + std::string(kEndOfWord)) == 0U)
      dif::fail("vocab.json: byte-level symbol for byte " + std::to_string(b) +
                " is missing; not a CLIP byte-level BPE vocab");
  }

  // merges.txt: "#version" header, then one "left right" pair per line whose
  // rank is its position. The key is the line itself (single ASCII space);
  // byte-level symbols never contain a raw space, so the key is unambiguous.
  std::istringstream lines(read_file(merges_txt));
  std::string line;
  if (!std::getline(lines, line) || line.rfind("#version", 0) != 0)
    dif::fail("merges.txt: expected a '#version' header line");
  std::uint32_t rank = 0;
  while (std::getline(lines, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    const std::size_t space = line.find(' ');
    if (space == std::string::npos || space == 0U ||
        space + 1U == line.size() ||
        line.find(' ', space + 1U) != std::string::npos)
      dif::fail("merges.txt: malformed merge at rank " + std::to_string(rank));
    if (!tok.merge_ranks_.emplace(line, rank).second)
      dif::fail("merges.txt: duplicate merge at rank " + std::to_string(rank));
    ++rank;
  }
  if (rank == 0U)
    dif::fail("merges.txt: no merges");

  return tok;
}

std::vector<std::uint32_t>
ClipBpeTokenizer::encode(std::string_view text) const {
  std::vector<std::uint32_t> ids;
  ids.push_back(bos_id_);
  const std::vector<char32_t> norm = normalize(unicode::decode_utf8(text));
  std::string utf8;
  for (const auto &[begin, end] : split(norm)) {
    // Byte-level expand: each codepoint -> UTF-8 bytes -> one symbol each.
    std::vector<std::string> word;
    for (std::size_t k = begin; k < end; ++k) {
      utf8.clear();
      unicode::append_utf8(utf8, norm[k]);
      for (const char byte : utf8)
        word.push_back(byte_map_[static_cast<unsigned char>(byte)]);
    }
    for (const std::string &symbol : bpe(std::move(word))) {
      const auto it = vocab_.find(symbol);
      if (it == vocab_.end())
        dif::fail("clip tokenizer: merged symbol missing from vocab "
                  "(byte-level invariant violated)");
      ids.push_back(it->second);
    }
  }
  ids.push_back(eos_id_);
  return ids;
}

// The reference bpe(): the last symbol carries "</w>"; each round merges
// EVERY non-overlapping occurrence of the lowest-ranked adjacent pair,
// left to right, until no adjacent pair has a rank.
std::vector<std::string>
ClipBpeTokenizer::bpe(std::vector<std::string> word) const {
  if (word.empty())
    return word;
  word.back().append(kEndOfWord);
  if (word.size() < 2U)
    return word;
  std::string key;
  while (true) {
    std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
    std::size_t best_index = word.size();
    for (std::size_t i = 0; i + 1U < word.size(); ++i) {
      key.assign(word[i]);
      key.push_back(' ');
      key.append(word[i + 1U]);
      const auto it = merge_ranks_.find(key);
      if (it != merge_ranks_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_index = i;
      }
    }
    if (best_index == word.size())
      break;
    const std::string first = word[best_index];
    const std::string second = word[best_index + 1U];
    const std::string merged = first + second;
    std::vector<std::string> next;
    next.reserve(word.size());
    for (std::size_t i = 0; i < word.size();) {
      if (i + 1U < word.size() && word[i] == first && word[i + 1U] == second) {
        next.push_back(merged);
        i += 2U;
      } else {
        next.push_back(std::move(word[i]));
        ++i;
      }
    }
    word = std::move(next);
    if (word.size() < 2U)
      break;
  }
  return word;
}

ClipPromptTokens clip_prompt_tokens(const ClipBpeTokenizer &tokenizer,
                                    std::string_view text,
                                    std::uint32_t pad_token,
                                    std::uint64_t max_length) {
  if (max_length < 3U)
    dif::fail("clip_prompt_tokens: max_length must hold BOS, a token and EOS");
  // The reference's word-length threshold: a token group this long is
  // broken across chunks instead of being moved whole to the next one.
  constexpr std::size_t kMaxWordLength = 8;

  const std::vector<std::uint32_t> raw = tokenizer.encode(text);
  const auto bos = static_cast<std::int32_t>(tokenizer.bos_id());
  const auto eos = static_cast<std::int32_t>(tokenizer.eos_id());
  const auto pad = static_cast<std::int32_t>(pad_token);
  // Plain text is ONE token group: the reference tokenizes the whole prompt
  // in one call and strips that call's BOS/EOS before batching.
  std::vector<std::int32_t> group;
  group.reserve(raw.size() - 2U);
  for (std::size_t i = 1; i + 1U < raw.size(); ++i)
    group.push_back(static_cast<std::int32_t>(raw[i]));
  const bool is_large = group.size() >= kMaxWordLength;
  // Slots available before the chunk's EOS.
  const std::size_t capacity = static_cast<std::size_t>(max_length) - 1U;

  ClipPromptTokens out;
  std::vector<std::int32_t> batch{bos};
  const auto flush = [&] {
    out.ids.insert(out.ids.end(), batch.begin(), batch.end());
    batch.assign(1U, bos);
  };
  std::size_t consumed = 0;
  while (consumed < group.size()) {
    const std::size_t left = group.size() - consumed;
    if (left + batch.size() > capacity) {
      const std::size_t remaining = capacity - batch.size();
      if (is_large) {
        // Break the group: fill this chunk, the tail starts the next one.
        batch.insert(batch.end(), group.begin() + consumed,
                     group.begin() + consumed + remaining);
        batch.push_back(eos);
        consumed += remaining;
      } else {
        // A short group moves whole to the next chunk. Unreachable with a
        // single group (fewer than kMaxWordLength tokens always fit) but
        // kept so this loop stays the reference's.
        batch.push_back(eos);
        batch.insert(batch.end(), remaining, pad);
      }
      flush();
    } else {
      batch.insert(batch.end(), group.begin() + consumed, group.end());
      consumed = group.size();
    }
  }
  batch.push_back(eos);
  batch.insert(batch.end(), static_cast<std::size_t>(max_length) - batch.size(),
               pad);
  flush();

  // num_tokens of the first chunk: the reference's attention mask is 1 up to
  // and including the first EOS, 0 after it (also when the pad IS the EOS).
  out.valid_tokens = 0;
  for (std::size_t i = 0; i < static_cast<std::size_t>(max_length); ++i) {
    ++out.valid_tokens;
    if (out.ids[i] == eos)
      break;
  }
  return out;
}

SdxlPromptTokens sdxl_prompt_tokens(const ClipBpeTokenizer &tokenizer,
                                    std::string_view text) {
  SdxlPromptTokens out;
  out.l = clip_prompt_tokens(tokenizer, text, tokenizer.eos_id());
  out.g = clip_prompt_tokens(tokenizer, text, kSdxlClipGPadToken);
  return out;
}

} // namespace dif::text
