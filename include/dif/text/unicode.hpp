#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace dif::text::unicode {

// Strict UTF-8 decoder: rejects overlong encodings, surrogates, values above
// U+10FFFF, and truncated sequences (fail-closed via dif::fail).
std::vector<char32_t> decode_utf8(std::string_view text);

// Appends the UTF-8 encoding of `cp` to `out`.
void append_utf8(std::string &out, char32_t cp);

// Unicode general-category classes as Oniguruma's UTF-8 \p{L} / \p{N}
// resolve them (tables generated from Unicode 15.0).
bool is_letter(char32_t cp);
bool is_number(char32_t cp);

// Oniguruma's UTF-8 \s: the Unicode White_Space set. MEASURED against the
// HF fast pre-tokenizer: U+00A0 behaves as \s, U+001C..U+001F do not.
bool is_whitespace(char32_t cp);

// Canonical composition (NFC) per UAX #15: full canonical decomposition
// (Hangul algorithmic), canonical ordering, then primary composition.
std::vector<char32_t> nfc(const std::vector<char32_t> &input);

} // namespace dif::text::unicode
