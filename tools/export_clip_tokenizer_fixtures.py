#!/usr/bin/env python3
"""Dev-time oracle exporter for tests/clip_tokenizer_fixtures.inc.

Oracle: the reference sampler's SDXL tokenizer pair (comfy.sdxl_clip
.SDXLTokenizer: CLIP-L padded with EOS, CLIP-G padded with 0) driven through
tokenize_with_weights() on plain prompts, plus the raw HF CLIPTokenizer ids
(BOS ... EOS, no padding) from comfy.sd1_clip.SDTokenizer().tokenizer.

Run with the reference checkout's own interpreter so the tokenizer stack is
the one that produced the frozen SDXL images, e.g.

  <reference>/venv/bin/python tools/export_clip_tokenizer_fixtures.py \
      --reference-source <reference> --out tests/clip_tokenizer_fixtures.inc

Python is a development tool; the generated .inc is committed and the C++
test needs no Python at run time.
"""
import argparse
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--reference-source", required=True,
                help="checkout of the reference sampler (its `comfy` package "
                     "and sd1_tokenizer directory are imported from here)")
ap.add_argument("--out", default="tests/clip_tokenizer_fixtures.inc")
args = ap.parse_args()

sys.path.insert(0, args.reference_source)
from comfy import sd1_clip, sdxl_clip  # noqa: E402

commit = subprocess.check_output(
    ["git", "-C", args.reference_source, "rev-parse", "HEAD"],
    text=True).strip()

raw_tok = sd1_clip.SDTokenizer()          # CLIP-L contract (pad = EOS)
sdxl_tok = sdxl_clip.SDXLTokenizer()      # {"l": ..., "g": ...}
hf = raw_tok.tokenizer

MAX_LENGTH = raw_tok.max_length
assert MAX_LENGTH == sdxl_tok.clip_g.max_length == 77
BOS, EOS = raw_tok.start_token, raw_tok.end_token
assert (BOS, EOS) == (49406, 49407)
assert raw_tok.pad_token == EOS and sdxl_tok.clip_g.pad_token == 0
assert sdxl_tok.clip_l.pad_token == EOS

LONG_PROMPT = (
    "a highly detailed cinematic photograph of an old lighthouse keeper "
    "standing on the rocky shore at dawn, weathered face, thick wool sweater, "
    "holding a brass lantern, seagulls circling overhead, dramatic storm "
    "clouds breaking apart to reveal golden sunlight, crashing waves, wet "
    "stones glistening, shallow depth of field, 85mm lens, kodak portra 400 "
    "film grain, volumetric light, ultra realistic, masterpiece, best quality, "
    "intricate textures, moody atmosphere, wide shot, rule of thirds "
    "composition, soft rim lighting, cold blue and warm orange color palette")

CASES = [
    ("frozen_sdxl_prompt", "A cat holding a sign that says hello world"),
    ("empty", ""),
    ("whitespace_only", "   \t\n  "),
    ("punctuation_runs",
     "Hello, world!!! What?? ... -- ;;; ~*~ #1 $5 100% [ok] {x} <y> @me & "
     "a|b / \\ \"quoted\" 'single'"),
    ("numbers", "33 4k 1024x1024 3.14159 2026-09-03 10,000"),
    ("contractions", "don't it's we're I've I'm they'll you'd it'S DON'T"),
    ("apostrophe_edges", "x!'s y's 'tis o'clock rock'n'roll '' 're 'll"),
    ("mixed_case", "A CAT Holding A SIGN That SAYS Hello World"),
    ("latin1_case", "ÀÉÎÕÜ Ç É à é î õ ü ç ß ÿ Ñoño"),
    ("long_prompt_chunked", LONG_PROMPT),
    ("unicode_accents", "café naïve résumé façade"),
    ("cjk", "東京タワー"),
    ("emoji", "\U0001f431 a cat \U0001f600\U0001f525 with a sign "
              "\U0001f44d\U0001f3fd"),
    ("whitespace_soup",
     "  leading and   multiple    spaces\ttabs\nnewlines\r\nmixed  "),
    ("body_75_exact_fill", "a " * 75),
    ("body_76_two_chunks", "a " * 76),
    ("body_150_two_full", "a " * 150),
    ("body_151_three_chunks", "a " * 151),
    ("sd_style_prompt",
     "photo of a woman, 8k, uhd, dslr, soft lighting, high quality, film "
     "grain, Fujifilm XT3"),
    ("booru_style_prompt",
     "1girl, solo, long_hair, looking_at_viewer, smile, blue_eyes"),
    # Non-Latin-1 capitals: the native tokenizer lowercases ASCII + Latin-1
    # only, so this prompt is the documented case-folding divergence.
    ("greek_cyrillic_caps", "Ωmega ΣΙΓΜΑ Привет МИР"),
]


def num_tokens(chunk, pad_token, end_token):
    """The reference's process_tokens attention-mask sum for one chunk."""
    mask = []
    eos = False
    left_pad = False
    for index, token in enumerate(chunk):
        if index == 0 and token == pad_token:
            left_pad = True
        if eos or (left_pad and token == pad_token):
            mask.append(0)
        else:
            mask.append(1)
            left_pad = False
        if not eos and token == end_token and not left_pad:
            eos = True
    return sum(mask)


def ids_of(chunks):
    out = []
    for chunk in chunks:
        ids = []
        for token, weight in chunk:
            assert isinstance(token, int), "plain prompts must not yield embeds"
            assert weight == 1.0, "plain prompts must carry weight 1.0"
            ids.append(token)
        assert len(ids) == MAX_LENGTH, len(ids)
        out.append(ids)
    return out


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


def write_ids(w, name, ids):
    if not ids:
        return "nullptr"
    w("inline constexpr std::int32_t %s[] = {\n" % name)
    for i in range(0, len(ids), 14):
        w("    " + ",".join(map(str, ids[i:i + 14])) + ",\n")
    w("};\n")
    return name


records = []
for name, text in CASES:
    raw = hf(text)["input_ids"]
    assert raw[0] == BOS and raw[-1] == EOS, (name, raw)
    out = sdxl_tok.tokenize_with_weights(text)
    l_chunks = ids_of(out["l"])
    g_chunks = ids_of(out["g"])
    body = len(raw) - 2
    expected_chunks = max(1, -(-body // (MAX_LENGTH - 2)))
    assert len(l_chunks) == len(g_chunks) == expected_chunks, (name, body)
    records.append((name, text, raw, l_chunks, g_chunks,
                    num_tokens(l_chunks[0], EOS, EOS),
                    num_tokens(g_chunks[0], 0, EOS)))

long_index = [n for n, _ in CASES].index("long_prompt_chunked")
long_body = len(records[long_index][2]) - 2
assert long_body >= 90, long_body

with open(args.out, "w") as f:
    w = f.write
    w("// GENERATED by tools/export_clip_tokenizer_fixtures.py — do not "
      "edit.\n")
    w("// Oracle: the reference sampler's SDXL tokenizer pair "
      "(tokenize_with_weights\n")
    w("// on plain prompts; CLIP-L pads with EOS, CLIP-G pads with 0) and "
      "the raw\n")
    w("// HF CLIPTokenizer ids (BOS ... EOS) from its sd1_tokenizer "
      "directory.\n")
    w("// Reference source commit: %s\n" % commit)
    w("// clang-format off\n")
    w("namespace clip_fixtures {\n\n")
    w("struct ClipCase {\n"
      "  const char *name;\n"
      "  const char *utf8;\n"
      "  const std::int32_t *raw;   // [BOS, body..., EOS]\n"
      "  std::size_t raw_count;\n"
      "  const std::int32_t *l;     // CLIP-L chunks, flattened (77 each)\n"
      "  std::size_t l_count;\n"
      "  const std::int32_t *g;     // CLIP-G chunks, flattened (77 each)\n"
      "  std::size_t g_count;\n"
      "  std::size_t l_num_tokens;  // reference num_tokens of the 1st chunk\n"
      "  std::size_t g_num_tokens;\n"
      "};\n\n")
    w("inline constexpr std::int32_t kBosId = %d;\n" % BOS)
    w("inline constexpr std::int32_t kEosId = %d;\n" % EOS)
    w("inline constexpr std::int32_t kClipLPadId = %d;\n" % raw_tok.pad_token)
    w("inline constexpr std::int32_t kClipGPadId = %d;\n"
      % sdxl_tok.clip_g.pad_token)
    w("inline constexpr std::size_t kMaxLength = %d;\n\n" % MAX_LENGTH)

    names = []
    for idx, record in enumerate(records):
        raw, l_chunks, g_chunks = record[2], record[3], record[4]
        raw_name = write_ids(w, "kRaw_%d" % idx, raw)
        l_name = write_ids(w, "kL_%d" % idx, sum(l_chunks, []))
        g_name = write_ids(w, "kG_%d" % idx, sum(g_chunks, []))
        names.append((raw_name, l_name, g_name))
    w("\ninline constexpr ClipCase kClipCases[] = {\n")
    for idx, record in enumerate(records):
        name, text, raw, l_chunks, g_chunks, l_n, g_n = record
        raw_name, l_name, g_name = names[idx]
        w('    {"%s",\n      %s,\n      %s, %d,\n      %s, %d,\n      %s, %d,\n'
          '      %d, %d},\n'
          % (name, c_escape(text.encode("utf-8")), raw_name, len(raw),
             l_name, len(l_chunks) * MAX_LENGTH, g_name,
             len(g_chunks) * MAX_LENGTH, l_n, g_n))
    w("};\n\n} // namespace clip_fixtures\n// clang-format on\n")

print("cases:%d long_prompt_body:%d commit:%s -> %s"
      % (len(records), long_body, commit, args.out))
for name, text, raw, l_chunks, g_chunks, l_n, g_n in records:
    print("  %-24s body=%3d chunks=%d num_tokens(l/g)=%d/%d"
          % (name, len(raw) - 2, len(l_chunks), l_n, g_n))
