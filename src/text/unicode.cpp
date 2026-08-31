#include "dif/text/unicode.hpp"

#include "dif/support/error.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace dif::text::unicode {

namespace {

#include "unicode_tables.inc"

bool in_ranges(const tables::Range *begin, const tables::Range *end,
               char32_t cp) {
  const auto *it = std::upper_bound(
      begin, end, cp,
      [](char32_t value, const tables::Range &range) { return value < range.lo; });
  if (it == begin)
    return false;
  --it;
  return cp >= it->lo && cp <= it->hi;
}

std::uint8_t combining_class(char32_t cp) {
  const auto *begin = std::begin(tables::kCcc);
  const auto *end = std::end(tables::kCcc);
  const auto *it = std::lower_bound(
      begin, end, cp,
      [](const tables::Ccc &entry, char32_t value) { return entry.cp < value; });
  if (it != end && it->cp == cp)
    return it->ccc;
  return 0;
}

const tables::Decomp *find_decomposition(char32_t cp) {
  const auto *begin = std::begin(tables::kDecomp);
  const auto *end = std::end(tables::kDecomp);
  const auto *it = std::lower_bound(
      begin, end, cp, [](const tables::Decomp &entry, char32_t value) {
        return entry.cp < value;
      });
  if (it != end && it->cp == cp)
    return it;
  return nullptr;
}

// Hangul syllable constants (UAX #15 section 3.12).
constexpr char32_t kSBase = 0xAC00;
constexpr char32_t kLBase = 0x1100;
constexpr char32_t kVBase = 0x1161;
constexpr char32_t kTBase = 0x11A7;
constexpr std::uint32_t kLCount = 19;
constexpr std::uint32_t kVCount = 21;
constexpr std::uint32_t kTCount = 28;
constexpr std::uint32_t kNCount = kVCount * kTCount;
constexpr std::uint32_t kSCount = kLCount * kNCount;

char32_t compose_pair(char32_t a, char32_t b) {
  // Algorithmic Hangul composition.
  if (a >= kLBase && a < kLBase + kLCount && b >= kVBase &&
      b < kVBase + kVCount)
    return kSBase + ((a - kLBase) * kVCount + (b - kVBase)) * kTCount;
  if (a >= kSBase && a < kSBase + kSCount && (a - kSBase) % kTCount == 0 &&
      b > kTBase && b < kTBase + kTCount)
    return a + (b - kTBase);
  if (a >= (char32_t{1} << 21) || b >= (char32_t{1} << 21))
    return 0;
  const std::uint64_t key = (std::uint64_t{a} << 21) | std::uint64_t{b};
  const auto *begin = std::begin(tables::kComp);
  const auto *end = std::end(tables::kComp);
  const auto *it = std::lower_bound(
      begin, end, key, [](const tables::Comp &entry, std::uint64_t value) {
        return entry.key < value;
      });
  if (it != end && it->key == key)
    return it->out;
  return 0;
}

} // namespace

std::vector<char32_t> decode_utf8(std::string_view text) {
  std::vector<char32_t> out;
  out.reserve(text.size());
  std::size_t i = 0;
  const auto fail_at = [&](const char *what) {
    dif::fail(std::string("invalid UTF-8 (") + what + ") at byte offset " +
              std::to_string(i));
  };
  while (i < text.size()) {
    const auto b0 = static_cast<unsigned char>(text[i]);
    char32_t cp = 0;
    std::size_t extra = 0;
    if (b0 < 0x80) {
      cp = b0;
    } else if ((b0 & 0xE0) == 0xC0) {
      cp = b0 & 0x1FU;
      extra = 1;
    } else if ((b0 & 0xF0) == 0xE0) {
      cp = b0 & 0x0FU;
      extra = 2;
    } else if ((b0 & 0xF8) == 0xF0) {
      cp = b0 & 0x07U;
      extra = 3;
    } else {
      fail_at("bad lead byte");
    }
    if (i + extra >= text.size())
      fail_at("truncated sequence");
    for (std::size_t k = 1; k <= extra; ++k) {
      const auto bk = static_cast<unsigned char>(text[i + k]);
      if ((bk & 0xC0) != 0x80)
        fail_at("bad continuation byte");
      cp = (cp << 6) | (bk & 0x3FU);
    }
    if ((extra == 1 && cp < 0x80) || (extra == 2 && cp < 0x800) ||
        (extra == 3 && cp < 0x10000))
      fail_at("overlong encoding");
    if (cp >= 0xD800 && cp <= 0xDFFF)
      fail_at("surrogate codepoint");
    if (cp > 0x10FFFF)
      fail_at("codepoint above U+10FFFF");
    out.push_back(cp);
    i += extra + 1;
  }
  return out;
}

void append_utf8(std::string &out, char32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

bool is_letter(char32_t cp) {
  return in_ranges(std::begin(tables::kLetterRanges),
                   std::end(tables::kLetterRanges), cp);
}

bool is_number(char32_t cp) {
  return in_ranges(std::begin(tables::kNumberRanges),
                   std::end(tables::kNumberRanges), cp);
}

bool is_whitespace(char32_t cp) {
  const auto *begin = std::begin(tables::kWhitespace);
  const auto *end = std::end(tables::kWhitespace);
  return std::binary_search(begin, end, cp);
}

std::vector<char32_t> nfc(const std::vector<char32_t> &input) {
  // 1. Full canonical decomposition (tables hold the recursive NFD form;
  //    Hangul syllables decompose algorithmically).
  std::vector<char32_t> d;
  d.reserve(input.size());
  for (const char32_t cp : input) {
    if (cp >= kSBase && cp < kSBase + kSCount) {
      const std::uint32_t s = cp - kSBase;
      d.push_back(kLBase + s / kNCount);
      d.push_back(kVBase + (s % kNCount) / kTCount);
      if (s % kTCount != 0)
        d.push_back(kTBase + s % kTCount);
      continue;
    }
    if (const tables::Decomp *entry = find_decomposition(cp)) {
      for (std::uint32_t k = 0; k < entry->len; ++k)
        d.push_back(tables::kDecompPool[entry->off + k]);
      continue;
    }
    d.push_back(cp);
  }

  // 2. Canonical ordering: stable insertion sort of nonzero-ccc runs.
  for (std::size_t i = 1; i < d.size(); ++i) {
    const std::uint8_t cc = combining_class(d[i]);
    if (cc == 0)
      continue;
    std::size_t j = i;
    while (j > 0 && combining_class(d[j - 1]) > cc) {
      std::swap(d[j - 1], d[j]);
      --j;
    }
  }

  // 3. Primary composition.
  std::vector<char32_t> out;
  out.reserve(d.size());
  std::ptrdiff_t starter = -1;
  for (const char32_t c : d) {
    const std::uint8_t cc = combining_class(c);
    if (starter >= 0) {
      const bool follows_starter =
          out.size() == static_cast<std::size_t>(starter) + 1;
      const std::uint8_t prev_cc =
          follows_starter ? std::uint8_t{0} : combining_class(out.back());
      if (follows_starter || prev_cc < cc) {
        if (const char32_t p =
                compose_pair(out[static_cast<std::size_t>(starter)], c)) {
          out[static_cast<std::size_t>(starter)] = p;
          continue;
        }
      }
    }
    out.push_back(c);
    if (cc == 0)
      starter = static_cast<std::ptrdiff_t>(out.size()) - 1;
  }
  return out;
}

} // namespace dif::text::unicode
