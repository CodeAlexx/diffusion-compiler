#include "dif/text/qwen_bpe_tokenizer.hpp"

#include "dif/support/error.hpp"
#include "dif/support/json.hpp"
#include "dif/text/unicode.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace dif::text {

namespace {

// The exact Split pattern of the Qwen2 fast tokenizer. The loader compares
// the checkpoint's pattern against this string byte-for-byte and refuses
// anything else: the scanner below implements THIS pattern, not a general
// regex engine.
constexpr std::string_view kQwenSplitPattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

// The Mistral Tekken (Mistral Small 3.x) Split pattern: identical to the Qwen2
// pattern except numbers are taken in runs of up to three digits.
constexpr std::string_view kTekkenSplitPattern =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

std::string read_file(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    dif::fail("cannot open " + path.string());
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return std::move(buffer).str();
}

const dif::json::Value &require_key(const dif::json::Value &value,
                                    std::string_view key, const char *context) {
  const dif::json::Value *found = value.find(key);
  if (found == nullptr)
    dif::fail(std::string(context) + ": missing key '" + std::string(key) +
              "'");
  return *found;
}

bool is_null(const dif::json::Value &value) {
  return std::holds_alternative<std::nullptr_t>(value.storage);
}

// True when the key is absent, null, or exactly false.
bool flag_is_false(const dif::json::Value &object, std::string_view key) {
  const dif::json::Value *value = object.find(key);
  if (value == nullptr || is_null(*value))
    return true;
  return std::holds_alternative<bool>(value->storage) &&
         !std::get<bool>(value->storage);
}

bool null_or_empty_string(const dif::json::Value &object,
                          std::string_view key) {
  const dif::json::Value *value = object.find(key);
  if (value == nullptr || is_null(*value))
    return true;
  return std::holds_alternative<std::string>(value->storage) &&
         std::get<std::string>(value->storage).empty();
}

std::int32_t to_token_id(double value, const char *context) {
  if (!(value >= 0.0) || value > 2147483647.0 || value != std::floor(value))
    dif::fail(std::string(context) + ": token id is not a non-negative i32");
  return static_cast<std::int32_t>(value);
}

struct PipelineDialect {
  bool nfc{true};
  std::size_t digit_run{1U};
};

PipelineDialect verify_pipeline(const dif::json::Value &root) {
  for (const char *key : {"truncation", "padding"}) {
    const dif::json::Value *value = root.find(key);
    if (value != nullptr && !is_null(*value))
      dif::fail(std::string("tokenizer.json: unsupported non-null '") + key +
                "'");
  }

  const dif::json::Value &normalizer =
      require_key(root, "normalizer", "tokenizer.json");
  PipelineDialect dialect;
  if (is_null(normalizer))
    dialect.nfc = false; // Mistral Tekken: no normalizer
  else if (!normalizer.is_object() ||
           require_key(normalizer, "type", "normalizer").string() != "NFC")
    dif::fail("tokenizer.json: normalizer must be null or exactly {type: NFC}");

  const dif::json::Value &pre =
      require_key(root, "pre_tokenizer", "tokenizer.json");
  if (require_key(pre, "type", "pre_tokenizer").string() != "Sequence")
    dif::fail("tokenizer.json: pre_tokenizer.type must be Sequence");
  const auto &parts = require_key(pre, "pretokenizers", "pre_tokenizer").array();
  if (parts.size() != 2U)
    dif::fail("tokenizer.json: expected exactly 2 pretokenizers");

  const dif::json::Value &split = parts[0];
  if (require_key(split, "type", "pretokenizers[0]").string() != "Split")
    dif::fail("tokenizer.json: pretokenizers[0].type must be Split");
  if (require_key(split, "behavior", "Split").string() != "Isolated")
    dif::fail("tokenizer.json: Split.behavior must be Isolated");
  if (!flag_is_false(split, "invert"))
    dif::fail("tokenizer.json: Split.invert must be false");
  const dif::json::Value &pattern = require_key(split, "pattern", "Split");
  const dif::json::Value *regex = pattern.find("Regex");
  if (regex == nullptr)
    dif::fail("tokenizer.json: Split.pattern must be a Regex");
  if (regex->string() == kQwenSplitPattern)
    dialect.digit_run = 1U;
  else if (regex->string() == kTekkenSplitPattern)
    dialect.digit_run = 3U;
  else
    dif::fail("tokenizer.json: Split.pattern.Regex is neither the Qwen2 nor "
              "the Mistral Tekken pattern this tokenizer implements");

  const dif::json::Value &byte_level = parts[1];
  if (require_key(byte_level, "type", "pretokenizers[1]").string() !=
      "ByteLevel")
    dif::fail("tokenizer.json: pretokenizers[1].type must be ByteLevel");
  if (!flag_is_false(byte_level, "add_prefix_space"))
    dif::fail("tokenizer.json: ByteLevel.add_prefix_space must be false");
  if (!flag_is_false(byte_level, "use_regex"))
    dif::fail("tokenizer.json: ByteLevel.use_regex must be false");

  const dif::json::Value *post = root.find("post_processor");
  if (post != nullptr && !is_null(*post)) {
    const auto &type = require_key(*post, "type", "post_processor").string();
    // TemplateProcessing (Mistral: prepend <s>) is accepted because encode()
    // reproduces add_special_tokens=False; the prompt renderer writes the
    // BOS text explicitly, exactly as apply_chat_template does.
    if (type != "ByteLevel" && type != "TemplateProcessing")
      dif::fail("tokenizer.json: post_processor must be null, ByteLevel or "
                "TemplateProcessing (anything else could add ids)");
  }
  return dialect;
}

char32_t lower_ascii(char32_t cp) {
  return (cp >= U'A' && cp <= U'Z') ? cp + 32U : cp;
}

bool is_cr_lf(char32_t cp) { return cp == U'\r' || cp == U'\n'; }

// Hand implementation of kQwenSplitPattern over one NFC-normalized segment,
// reproducing Oniguruma's leftmost-alternation backtracking semantics.
// Returns [begin, end) codepoint ranges.
//
// Class facts measured against the HF fast pre-tokenizer (see the parity
// fixtures): \s is the Unicode White_Space set (U+00A0/U+3000/U+0085 ARE
// \s; U+001C..1F, U+200B, U+00AD, U+180E are NOT); (?i:) folds ASCII only
// (U+017F LATIN SMALL LETTER LONG S does not match 's).
std::vector<std::pair<std::size_t, std::size_t>>
pretokenize(const std::vector<char32_t> &s, std::size_t digit_run) {
  namespace uni = dif::text::unicode;
  std::vector<std::pair<std::size_t, std::size_t>> out;
  const std::size_t n = s.size();
  std::size_t i = 0;
  while (i < n) {
    // Alt 1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (s[i] == U'\'' && i + 1 < n) {
      const char32_t c1 = lower_ascii(s[i + 1]);
      if (c1 == U's' || c1 == U't' || c1 == U'm' || c1 == U'd') {
        out.emplace_back(i, i + 2);
        i += 2;
        continue;
      }
      if (i + 2 < n) {
        const char32_t c2 = lower_ascii(s[i + 2]);
        if ((c1 == U'r' && c2 == U'e') || (c1 == U'v' && c2 == U'e') ||
            (c1 == U'l' && c2 == U'l')) {
          out.emplace_back(i, i + 3);
          i += 3;
          continue;
        }
      }
    }
    // Alt 2: [^\r\n\p{L}\p{N}]?\p{L}+  (greedy ?: prefix form tried first)
    {
      const bool prefix_ok = !is_cr_lf(s[i]) && !uni::is_letter(s[i]) &&
                             !uni::is_number(s[i]);
      if (prefix_ok && i + 1 < n && uni::is_letter(s[i + 1])) {
        std::size_t j = i + 2;
        while (j < n && uni::is_letter(s[j]))
          ++j;
        out.emplace_back(i, j);
        i = j;
        continue;
      }
      if (uni::is_letter(s[i])) {
        std::size_t j = i + 1;
        while (j < n && uni::is_letter(s[j]))
          ++j;
        out.emplace_back(i, j);
        i = j;
        continue;
      }
    }
    // Alt 3: \p{N} (Qwen2) or \p{N}{1,3} (Tekken)
    if (uni::is_number(s[i])) {
      std::size_t j = i + 1;
      while (j < n && j - i < digit_run && uni::is_number(s[j]))
        ++j;
      out.emplace_back(i, j);
      i = j;
      continue;
    }
    // Alt 4: ` ?[^\s\p{L}\p{N}]+[\r\n]*`
    {
      std::size_t j = i;
      if (s[j] == U' ')
        ++j;
      std::size_t k = j;
      while (k < n && !uni::is_whitespace(s[k]) && !uni::is_letter(s[k]) &&
             !uni::is_number(s[k]))
        ++k;
      if (k > j) {
        while (k < n && is_cr_lf(s[k]))
          ++k;
        out.emplace_back(i, k);
        i = k;
        continue;
      }
      // Backtracking the optional space cannot help: without it the
      // character class must match at i, and s[i] == ' ' is \s.
    }
    // Alts 5-7 all require \s at i.
    if (uni::is_whitespace(s[i])) {
      std::size_t e = i;
      while (e < n && uni::is_whitespace(s[e]))
        ++e;
      // Alt 5: \s*[\r\n]+ — greedy \s* backtracks until [\r\n]+ can match,
      // so the first success puts [\r\n]+ at the LAST \r/\n of the run.
      std::size_t last_cr_lf = n; // sentinel: none
      for (std::size_t m = e; m > i;) {
        --m;
        if (is_cr_lf(s[m])) {
          last_cr_lf = m;
          break;
        }
      }
      if (last_cr_lf != n) {
        out.emplace_back(i, last_cr_lf + 1);
        i = last_cr_lf + 1;
        continue;
      }
      // Alt 6: \s+(?!\S)
      if (e == n) {
        out.emplace_back(i, e);
        i = e;
        continue;
      }
      if (e - i >= 2) { // leave the final \s to attach to what follows
        out.emplace_back(i, e - 1);
        i = e - 1;
        continue;
      }
      // Alt 7: \s+
      out.emplace_back(i, e);
      i = e;
      continue;
    }
    dif::fail("pretokenize: unreachable — no alternative matched");
  }
  return out;
}

} // namespace

QwenBpeTokenizer
QwenBpeTokenizer::load(const std::filesystem::path &tokenizer_json,
                       const std::filesystem::path &tokenizer_config_json) {
  QwenBpeTokenizer tok;

  const std::string text = read_file(tokenizer_json);
  const dif::json::Value root = dif::json::parse(text);
  const auto dialect = verify_pipeline(root);
  tok.nfc_ = dialect.nfc;
  tok.digit_run_ = dialect.digit_run;

  const dif::json::Value &model = require_key(root, "model", "tokenizer.json");
  if (require_key(model, "type", "model").string() != "BPE")
    dif::fail("tokenizer.json: model.type must be BPE");
  for (const char *key : {"byte_fallback", "fuse_unk"})
    if (!flag_is_false(model, key))
      dif::fail(std::string("tokenizer.json: unsupported model flag '") + key +
                "'");
  tok.ignore_merges_ = !flag_is_false(model, "ignore_merges");
  if (const dif::json::Value *dropout = model.find("dropout");
      dropout != nullptr && !is_null(*dropout))
    dif::fail("tokenizer.json: BPE dropout is unsupported");
  if (const dif::json::Value *unk = model.find("unk_token");
      unk != nullptr && !is_null(*unk))
    dif::fail("tokenizer.json: unk_token is unsupported (byte-level BPE "
              "needs none)");
  if (!null_or_empty_string(model, "continuing_subword_prefix") ||
      !null_or_empty_string(model, "end_of_word_suffix"))
    dif::fail("tokenizer.json: subword prefixes/suffixes are unsupported");

  const auto &vocab = require_key(model, "vocab", "model").object();
  tok.vocab_.reserve(vocab.size());
  for (const auto &[token, id_value] : vocab)
    tok.vocab_.emplace(token, to_token_id(id_value.number(), "vocab"));

  const auto &merges = require_key(model, "merges", "model").array();
  tok.merge_ranks_.reserve(merges.size());
  for (std::size_t rank = 0; rank < merges.size(); ++rank) {
    const dif::json::Value &entry = merges[rank];
    std::string key;
    if (std::holds_alternative<std::string>(entry.storage)) {
      key = std::get<std::string>(entry.storage);
      const std::size_t space = key.find(' ');
      if (space == std::string::npos || space == 0U ||
          space + 1U == key.size() || key.find(' ', space + 1U) != std::string::npos)
        dif::fail("tokenizer.json: malformed merge entry at rank " +
                  std::to_string(rank));
    } else if (entry.is_array() && entry.array().size() == 2U) {
      key = entry.array()[0].string();
      key.push_back(' ');
      key += entry.array()[1].string();
    } else {
      dif::fail("tokenizer.json: merge entry is neither \"A B\" nor [A, B]");
    }
    tok.merge_ranks_.emplace(std::move(key),
                             static_cast<std::uint32_t>(rank));
  }

  if (const dif::json::Value *added = root.find("added_tokens");
      added != nullptr && added->is_array()) {
    for (const dif::json::Value &entry : added->array()) {
      AddedToken token;
      token.content = require_key(entry, "content", "added_tokens").string();
      token.id =
          to_token_id(require_key(entry, "id", "added_tokens").number(),
                      "added_tokens");
      token.special = !flag_is_false(entry, "special");
      for (const char *key : {"lstrip", "rstrip", "single_word", "normalized"})
        if (!flag_is_false(entry, key))
          dif::fail(std::string("added_tokens: unsupported flag '") + key +
                    "' on '" + token.content + "'");
      tok.added_.push_back(std::move(token));
    }
  }

  // GPT-2 byte-level map: printable/latin bytes map to themselves, the rest
  // to U+0100 + running index, in ascending byte order.
  {
    std::uint32_t next = 0;
    for (unsigned b = 0; b < 256U; ++b) {
      const bool direct = (b >= 33U && b <= 126U) || (b >= 161U && b <= 172U) ||
                          (b >= 174U && b <= 255U);
      const char32_t cp = direct ? static_cast<char32_t>(b)
                                 : static_cast<char32_t>(256U + next++);
      unicode::append_utf8(tok.byte_map_[b], cp);
    }
  }

  if (!tokenizer_config_json.empty()) {
    const std::string config_text = read_file(tokenizer_config_json);
    const dif::json::Value config = dif::json::parse(config_text);
    const dif::json::Value *extra = config.find("additional_special_tokens");
    if (extra != nullptr && extra->is_array()) {
      // transformers assignment rule (measured on the H3 processor):
      // known content keeps its id; new content takes the next id after the
      // current maximum, in config order.
      std::int32_t next_id = tok.id_space();
      for (const dif::json::Value &entry : extra->array()) {
        std::string content;
        if (std::holds_alternative<std::string>(entry.storage))
          content = std::get<std::string>(entry.storage);
        else
          content = require_key(entry, "content",
                                "additional_special_tokens").string();
        const bool known =
            tok.vocab_.count(content) != 0U ||
            std::any_of(tok.added_.begin(), tok.added_.end(),
                        [&](const AddedToken &t) { return t.content == content; });
        if (known)
          continue;
        tok.added_.push_back(AddedToken{std::move(content), next_id++, true});
      }
    }
  }

  return tok;
}

std::int32_t QwenBpeTokenizer::id_space() const {
  std::int32_t max_id = -1;
  for (const auto &[token, id] : vocab_)
    max_id = std::max(max_id, id);
  for (const AddedToken &token : added_)
    max_id = std::max(max_id, token.id);
  return max_id + 1;
}

std::vector<std::int32_t>
QwenBpeTokenizer::encode(std::string_view utf8_text) const {
  std::vector<std::int32_t> ids;
  // 1. Added-token extraction on the RAW text (all added tokens are
  //    normalized=false): leftmost scan; longest content wins at a position.
  std::size_t segment_start = 0;
  std::size_t i = 0;
  while (i < utf8_text.size()) {
    const AddedToken *best = nullptr;
    if ((static_cast<unsigned char>(utf8_text[i]) & 0xC0U) != 0x80U) {
      for (const AddedToken &token : added_) {
        const std::size_t len = token.content.size();
        if (len == 0U || i + len > utf8_text.size())
          continue;
        if (best != nullptr && len <= best->content.size())
          continue;
        if (utf8_text.compare(i, len, token.content) == 0)
          best = &token;
      }
    }
    if (best != nullptr) {
      encode_segment(utf8_text.substr(segment_start, i - segment_start), ids);
      ids.push_back(best->id);
      i += best->content.size();
      segment_start = i;
      continue;
    }
    ++i;
  }
  encode_segment(utf8_text.substr(segment_start), ids);
  return ids;
}

void QwenBpeTokenizer::encode_segment(std::string_view utf8_segment,
                                      std::vector<std::int32_t> &ids) const {
  if (utf8_segment.empty())
    return;
  const std::vector<char32_t> raw = unicode::decode_utf8(utf8_segment);
  const std::vector<char32_t> norm = nfc_ ? unicode::nfc(raw) : raw;
  std::string utf8;
  for (const auto &[begin, end] : pretokenize(norm, digit_run_)) {
    std::vector<std::string> word;
    for (std::size_t k = begin; k < end; ++k) {
      utf8.clear();
      unicode::append_utf8(utf8, norm[k]);
      for (const char byte : utf8)
        word.push_back(byte_map_[static_cast<unsigned char>(byte)]);
    }
    for (std::string &symbol : bpe(std::move(word))) {
      const auto it = vocab_.find(symbol);
      if (it == vocab_.end())
        dif::fail("tokenizer: merged symbol missing from vocab (byte-level "
                  "invariant violated)");
      ids.push_back(it->second);
    }
  }
}

std::vector<std::string>
QwenBpeTokenizer::bpe(std::vector<std::string> word) const {
  if (word.size() < 2U)
    return word;
  std::string key;
  if (ignore_merges_) {
    for (const auto &symbol : word)
      key.append(symbol);
    if (vocab_.count(key) != 0U)
      return {key};
    key.clear();
  }
  while (true) {
    std::uint32_t best_rank = std::numeric_limits<std::uint32_t>::max();
    std::size_t best_index = word.size();
    for (std::size_t i = 0; i + 1U < word.size(); ++i) {
      key.assign(word[i]);
      key.push_back(' '); // byte-level symbols never contain 0x20
      key.append(word[i + 1U]);
      const auto it = merge_ranks_.find(key);
      if (it != merge_ranks_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_index = i;
      }
    }
    if (best_index == word.size())
      break;
    word[best_index].append(word[best_index + 1U]);
    word.erase(word.begin() + static_cast<std::ptrdiff_t>(best_index) + 1);
  }
  return word;
}

} // namespace dif::text
