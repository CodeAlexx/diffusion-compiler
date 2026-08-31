#!/usr/bin/env python3
"""Dev-time oracle exporter for tests/tokenizer_parity_fixtures.inc.

Oracle: `transformers` AutoTokenizer on the MiniMax-H3 processor directory
(Qwen2TokenizerFast; tokenizer.json + tokenizer_config.json
additional_special_tokens), encode(..., add_special_tokens=False). This is
the exact pipeline that produced the recorded 439-token conditioning, so it
is the semantic target — NOT bare tokenizers.Tokenizer.from_file, which
misses the seven config-only special tokens (<d>, </d>, ...) and tokenizes
the golden prompt to 440.

Python is a development tool; the generated .inc is committed and the C++
test needs no Python at run time.
"""
import hashlib
import sys
import unicodedata

PROCESSOR = "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/processor"
GOLDEN_PROMPT = ("/home/alex/diffusion-compiler/artifacts/"
                 "h3-quality-natural-language-2026-08-30/prompt.txt")
OUT = sys.argv[1] if len(sys.argv) > 1 else "tests/tokenizer_parity_fixtures.inc"

from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(PROCESSOR)
assert tok.is_fast

raw = open(GOLDEN_PROMPT, "rb").read().decode("utf-8")
stripped = raw
while stripped.endswith("\n"):
    stripped = stripped[:-1]

CASES = [
    ("empty", ""),
    ("single_letter", "a"),
    ("single_space", " "),
    ("lone_newline", "\n"),
    ("newlines_3", "\n\n\n"),
    ("lone_cr", "\r"),
    ("crlf", "\r\n"),
    ("mixed_cr_lf", "a\r\nb\rc\nd"),
    ("hello_world", "hello world"),
    ("two_space_word", "  hello"),
    ("tab_sep", "tab\tsep"),
    ("ws_newline_mix", "a  \n\n  b"),
    ("crlf_soup", " \r\n \r\n"),
    ("punct_run", "...!?"),
    ("space_punct_newline", " ,,,\n\nx"),
    ("contractions", "it's it'S IT'S don't've"),
    ("contraction_edges", "'t start 'll 'r '' '''"),
    ("contraction_variants", "'s 've 'LL 'Re 'd 'M"),
    ("long_s_contraction", "x 'ſt rest"),
    ("digits", "12345 67890"),
    ("number_punct", "3.14159 2026-08-31"),
    ("arabic_digits", "١٢٣ ٤"),
    ("number_categories", "Ⅻ ① x² \U0001d7d8\U0001d7e1"),
    ("cjk", "你好吗？这是一个测试。"),
    ("japanese", "こんにちは世界"),
    ("hangul_precomposed", "안녕하세요"),
    ("hangul_decomposed", unicodedata.normalize("NFD", "안녕")),
    ("cafe_mixed", "café café"),
    ("singletons", "Å Å Ω Ω"),
    ("combining_soup", "ę́ q̣̇ q̣̇ x̵́"),
    ("emoji", "\U0001f600 \U0001f3fd \U0001f44d\U0001f3fd "
              "\U0001f468‍\U0001f469‍\U0001f467‍\U0001f466 "
              "\U0001f1fa\U0001f1f8"),
    ("url", "https://example.com/path?q=1&r=2#frag"),
    ("path_flags", "/home/alex/x_y.txt --flag=value"),
    ("nbsp", "a b    c  d"),
    ("wide_spaces", "x　　y    z  w"),
    ("nel_double", "nm"),
    ("more_ws", "p  q      r"),
    ("en_quad_run", "u    v"),
    ("not_ws_zwsp_shy", "w​​x y­­z ᠎᠎"),
    ("info_separators", "c\x1c\x1d\x1e\x1fd e\x1c\x1cf"),
    ("vt_ff", "e\x0b\x0cf g\x0b\x0bh"),
    ("dialogue_markup", "<d>[English] hi</d>"),
    ("dialogue_spaced", "a <d> b </d> c"),
    ("qwen_specials", "<|endoftext|><|im_start|>system<|im_end|>"),
    ("nonspecial_added", "<think>x</think><tool_call>y</tool_call>"),
    ("h3_config_tokens", "<|cutoff|><|lyrics_start|>la<|lyrics_end|>"
                         "<|caption_start|>c<|caption_end|>"),
    ("special_lookalikes", "<d><d></d> <dd> < d> <|cutoff| |cutoff|>"),
    ("vision_tokens", "<|vision_start|><|image_pad|><|video_pad|>"
                      "<|vision_end|><|quad_start|><|quad_end|>"),
    ("ref_box_tokens", "<|object_ref_start|>obj<|object_ref_end|>"
                       "<|box_start|>1<|box_end|>"),
    ("fim_tokens", "<|fim_prefix|>a<|fim_middle|>b<|fim_suffix|>c<|fim_pad|>"
                   "<|repo_name|>r<|file_sep|>s<tool_response>t</tool_response>"),
    ("long_word", "a" * 300),
    ("byte_alphabet_literal", "Ġ ĠĠ Âł ġ"),
    ("sharp_s", "ß ẞ Straße STRASSE"),
    ("turkish", "İstanbul ırmak"),
    ("ligatures", "ﬃ ﬀ office"),
    ("indic", "मराठी हिन्द"
              "ी বাংলা"),
    ("devanagari_conjunct", "क्षत्रिय"),
    ("rtl", "مرحبا بالعا"
            "لم שלום"),
    ("arabic_marks", "ًٌٍ"),
    ("zalgo", "z̸̢̛a̶̡͝l̷̨̛g"
              "̸̛̥o̶̧͠"),
    ("smp_letters", "\U0001d573\U0001d58a\U0001d591\U0001d591\U0001d594 "
                    "\U0001d4d7\U0001d4ee\U0001d4f5\U0001d4f5\U0001d4f8"),
    ("gothic_fullwidth", "\U00010348 Ｈｅｌｌｏ　Ｗ"),
    ("golden_prompt_stripped", stripped),
    ("golden_prompt_raw", raw),
]

NFC_INPUTS = [
    "café",
    "Å",
    unicodedata.normalize("NFD", "안녕하세요"),
    "q̣̇",
    "q̣̇",
    "ę́",
    "ḍ̇",
    "ά",
    "İ",
    "x̵́y",
    "각",
    "가",
    "ﬃ",  # compatibility ligature: NFC must keep it
    "ą́b",
]

def c_escape(data: bytes) -> str:
    parts = []
    for b in data:
        if b in (0x22, 0x5C):
            parts.append("\\" + chr(b))
        elif 0x20 <= b <= 0x7E:
            parts.append(chr(b))
        else:
            parts.append("\\%03o" % b)
    chunks = []
    line = []
    width = 0
    for piece in parts:
        if width + len(piece) > 88 and line:
            chunks.append("".join(line))
            line = []
            width = 0
        line.append(piece)
        width += len(piece)
    chunks.append("".join(line))
    return "\n      ".join('"%s"' % chunk for chunk in chunks)

def sha_ids(ids):
    return hashlib.sha256((",".join(map(str, ids))).encode()).hexdigest()

with open(OUT, "w") as f:
    w = f.write
    w("// GENERATED by tools/export_tokenizer_parity_fixtures.py — do not edit.\n")
    w("// Oracle: transformers AutoTokenizer (Qwen2TokenizerFast) on the\n")
    w("// MiniMax-H3 processor dir, add_special_tokens=False.\n")
    w("// clang-format off\n")
    w("namespace fixtures {\n\n")
    w("struct ParityCase {\n  const char *name;\n  const char *utf8;\n"
      "  const std::int32_t *ids;\n  std::size_t id_count;\n};\n\n")
    w("struct NfcCase {\n  const char *input;\n  const char *expected;\n};\n\n")

    id_arrays = []
    for idx, (name, text) in enumerate(CASES):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        id_arrays.append(ids)
        if ids:
            w("inline constexpr std::int32_t kIds_%d[] = {\n" % idx)
            for i in range(0, len(ids), 14):
                w("    " + ",".join(map(str, ids[i:i+14])) + ",\n")
            w("};\n")
    w("\ninline constexpr ParityCase kParityCases[] = {\n")
    for idx, (name, text) in enumerate(CASES):
        ids = id_arrays[idx]
        ptr = ("kIds_%d" % idx) if ids else "nullptr"
        w('    {"%s",\n      %s,\n      %s, %d},\n'
          % (name, c_escape(text.encode("utf-8")), ptr, len(ids)))
    w("};\n\n")

    w("inline constexpr NfcCase kNfcCases[] = {\n")
    for s in NFC_INPUTS:
        out = unicodedata.normalize("NFC", s)
        w("    {%s,\n     %s},\n"
          % (c_escape(s.encode("utf-8")), c_escape(out.encode("utf-8"))))
    w("};\n\n")

    ids_raw = tok(raw, add_special_tokens=False)["input_ids"]
    ids_stripped = tok(stripped, add_special_tokens=False)["input_ids"]
    w("// Golden prompt (artifact prompt.txt) oracle results.\n")
    w("inline constexpr std::size_t kGoldenStrippedCount = %d;\n" % len(ids_stripped))
    w('inline constexpr const char *kGoldenStrippedSha = "%s";\n' % sha_ids(ids_stripped))
    w("inline constexpr std::int32_t kGoldenStrippedLastId = %d;\n" % ids_stripped[-1])
    w("inline constexpr std::size_t kGoldenRawCount = %d;\n" % len(ids_raw))
    w('inline constexpr const char *kGoldenRawSha = "%s";\n' % sha_ids(ids_raw))
    w("inline constexpr std::int32_t kGoldenRawLastId = %d;\n" % ids_raw[-1])
    w("inline constexpr std::int32_t kExpectedIdSpace = %d;\n" % len(tok))
    w("inline constexpr std::size_t kExpectedAddedTokens = %d;\n"
      % len(tok.added_tokens_decoder))
    w("\n} // namespace fixtures\n// clang-format on\n")

print("cases:%d nfc:%d golden(stripped/raw):%d/%d id_space:%d added:%d -> %s"
      % (len(CASES), len(NFC_INPUTS), len(ids_stripped), len(ids_raw),
         len(tok), len(tok.added_tokens_decoder), OUT))
