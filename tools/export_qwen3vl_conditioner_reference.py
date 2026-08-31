#!/usr/bin/env python3
"""Qwen3-VL conditioner PyTorch reference oracle (dev-only; torch is never a
runtime dependency of the compiler).

Produces the fixtures the native DiffIR conditioner port is gated against,
per docs/QWEN3VL_CONDITIONER_PLAN.md §5:

  * gate 2  - embedding output              embed_tokens(ids)      [S, 5120] BF16
  * gate 3  - per-op fixtures at ENCODER shapes:
                QkNormPartialRope, RotaryDim = 128 = head_dim, H=64 and H=8
                GQA causal SDPA, H=64 / KvH=8 / D=128
  * gate 4  - depth ladder: RAW residual stream after k layers, BEFORE
              model.norm, for k in --depths (default 1,2,23,49,50)
  * gate 5  - depth-50 payload SHA-256, directly comparable to the recorded
              Serenity payload 9b1609bd... (raw BF16 bytes, no header)

METHOD (inherited from the proven Serenity oracle,
serenitymojo/models/text_encoder/parity/minimax_h3_conditioner_real_weight_oracle.py):
instantiate transformers' OWN Qwen3-VL modules — never a transcription — load
REAL checkpoint bytes, run on GPU in the checkpoint's native BF16 (never
CPU/F32: a wrong-dtype "reference" diverges from a GPU-BF16 port by more than
the port's own error), and read the raw per-depth hidden state by looping the
decoder layers by hand rather than through `output_hidden_states` bookkeeping.

DELIBERATE DEVIATION from that script, and the reason for it: it builds the
whole `Qwen3VLTextModel(text_config)`. This checkpoint declares
num_hidden_layers = 64, i.e. ~62 GiB of BF16 decoder weights, which does not
fit in this box's 24 GiB of VRAM (nor, with the 1.6 GiB embedding table, in
its 62 GiB of RAM under the mem-safe 24 GiB cap). Reaching depth 50 therefore
requires STREAMING: this script instantiates exactly ONE
`Qwen3VLTextDecoderLayer` — the same class, with the real config — and
overwrites its 11 parameters from the checkpoint shards once per depth, so
resident weights stay at ~0.98 GiB regardless of depth. Every op executed is
still transformers' own module code; only the residency policy differs.

Shard handles are opened lazily, closed as soon as the last tensor they own
has been consumed, and their page cache is dropped with POSIX_FADV_DONTNEED
(mirroring Serenity's MADV_DONTNEED policy) so a ~49 GiB streaming scan does
not accumulate host residency. Run under scripts/mem_safe_runtime.sh.

Usage:
  export_qwen3vl_conditioner_reference.py --ids-json IDS --out-dir DIR
      [--text-encoder DIR] [--depths 1,2,23,49,50] [--attn sdpa|eager]
      [--no-op-fixtures] [--compare-to PREV_OUT_DIR]

`--compare-to` re-reads a previous run's fixtures and reports the six
run-to-run metrics per tensor; that spread IS the BF16 noise floor the port is
judged against, so it is measured here rather than assumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextConfig,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextRotaryEmbedding,
    apply_rotary_pos_emb,
)

DEFAULT_TEXT_ENCODER = (
    "/home/alex/.serenity/models/checkpoints/MiniMax-H3/FL2VA/text_encoder"
)
LAYER_PREFIX = "model.language_model."
DEVICE = "cuda"
DTYPE = torch.bfloat16

# The 11 parameters a Qwen3-VL text decoder layer owns. `attention_bias` is
# false and the MLP is bias-less, so this list is exhaustive - asserted below.
LAYER_PARAMS = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)

# .diftensor container (docs/RUNTIME_MODULES.md); same encoding as
# tools/export_dit_backward_fixtures.py and tools/import_serenity_h3_inputs.py.
DIFTENSOR_DTYPE_CODES = {torch.float32: 1, torch.bfloat16: 2}


# ---------------------------------------------------------------------------
# fixture I/O
# ---------------------------------------------------------------------------


def raw_bytes(tensor: torch.Tensor) -> bytes:
    """Little-endian payload bytes, the same convention the recorded
    `payload_sha256` values in artifacts/*/input-manifest.json use."""
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy().tobytes(order="C")
    return value.numpy().tobytes(order="C")


def write_diftensor(path: Path, tensor: torch.Tensor) -> dict:
    value = tensor.detach().cpu().contiguous()
    if value.dtype not in DIFTENSOR_DTYPE_CODES:
        raise TypeError(f".diftensor export needs f32/bf16, got {value.dtype}")
    raw = raw_bytes(value)
    shape = tuple(int(d) for d in value.shape)
    body = bytearray(b"DIFTNS01")
    body += struct.pack("<III", 1, DIFTENSOR_DTYPE_CODES[value.dtype], len(shape))
    for dim in shape:
        body += struct.pack("<Q", dim)
    body += struct.pack("<Q", len(raw))
    body += raw
    body += hashlib.sha256(body).digest()
    path.write_bytes(body)
    return {
        "path": str(path),
        "payload_bytes": len(raw),
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
        "file_sha256": hashlib.sha256(body).hexdigest(),
    }


def describe(name: str, tensor: torch.Tensor) -> dict:
    raw = raw_bytes(tensor)
    return {
        "name": name,
        "shape": [int(d) for d in tensor.shape],
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "payload_bytes": len(raw),
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
    }


def emit(out_dir: Path, stem: str, tensors: dict[str, torch.Tensor],
         diftensor: bool = True) -> dict:
    """Write one .safetensors bundle plus (optionally) one .diftensor per
    float tensor, and return the manifest record for it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    st_path = out_dir / f"{stem}.safetensors"
    save_file({k: v.detach().cpu().contiguous() for k, v in tensors.items()}, str(st_path))
    record = {
        "safetensors": str(st_path),
        "safetensors_sha256": hashlib.sha256(st_path.read_bytes()).hexdigest(),
        "tensors": [],
    }
    for name, value in tensors.items():
        entry = describe(name, value)
        if diftensor and value.dtype in DIFTENSOR_DTYPE_CODES:
            entry["diftensor"] = write_diftensor(
                out_dir / f"{stem}.{name}.diftensor", value)
        record["tensors"].append(entry)
    return record


# ---------------------------------------------------------------------------
# streaming shard reader
# ---------------------------------------------------------------------------


class ShardReader:
    """Reads named tensors from the sharded checkpoint, holding at most the
    shards still needed and dropping consumed page cache."""

    def __init__(self, directory: Path, weight_map: dict[str, str]) -> None:
        self.directory = directory
        self.weight_map = weight_map
        self._handles: dict[str, object] = {}
        self.opened: list[str] = []

    def get(self, name: str) -> torch.Tensor:
        fname = self.weight_map[name]
        if fname not in self._handles:
            self._handles[fname] = safe_open(
                str(self.directory / fname), framework="pt", device="cpu")
            self.opened.append(fname)
        return self._handles[fname].get_tensor(name)

    def retain_only(self, files: set[str]) -> None:
        for fname in [f for f in self._handles if f not in files]:
            del self._handles[fname]
            self._drop_cache(fname)

    def close(self) -> None:
        self.retain_only(set())

    def _drop_cache(self, fname: str) -> None:
        try:
            fd = os.open(str(self.directory / fname), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------


def load_text_config(text_encoder: Path, attn: str) -> Qwen3VLTextConfig:
    with open(text_encoder / "config.json") as handle:
        cfg = json.load(handle)["text_config"]
    config = Qwen3VLTextConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        rope_theta=cfg["rope_theta"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_scaling=cfg["rope_scaling"],
        attention_bias=cfg["attention_bias"],
        attention_dropout=cfg["attention_dropout"],
        hidden_act=cfg["hidden_act"],
        max_position_embeddings=cfg["max_position_embeddings"],
    )
    # Qwen3VLTextModel(config) resolves this to "sdpa" during PreTrainedModel
    # init; we bypass that machinery, so set it explicitly to keep the same
    # attention kernel the reference model would have used.
    config._attn_implementation = attn
    return config


def build_layer(config: Qwen3VLTextConfig) -> Qwen3VLTextDecoderLayer:
    """One real decoder layer, allocated uninitialised (every parameter is
    overwritten from the checkpoint before any forward runs)."""
    with torch.device("meta"):
        layer = Qwen3VLTextDecoderLayer(config, 0)
    layer = layer.to_empty(device=DEVICE).to(DTYPE).eval()
    present = {name for name, _ in layer.named_parameters()}
    missing = present.symmetric_difference(set(LAYER_PARAMS))
    if missing:
        raise RuntimeError(
            f"decoder-layer parameter set changed; unhandled: {sorted(missing)}")
    return layer


def load_layer(layer: Qwen3VLTextDecoderLayer, reader: ShardReader, index: int) -> None:
    prefix = f"{LAYER_PREFIX}layers.{index}."
    params = dict(layer.named_parameters())
    with torch.no_grad():
        for suffix in LAYER_PARAMS:
            source = reader.get(prefix + suffix)
            target = params[suffix]
            if tuple(source.shape) != tuple(target.shape):
                raise RuntimeError(
                    f"shape mismatch for {prefix + suffix}: checkpoint "
                    f"{tuple(source.shape)} vs module {tuple(target.shape)}")
            target.copy_(source.to(DTYPE))
            del source


def layer_shard_files(weight_map: dict[str, str], index: int) -> set[str]:
    prefix = f"{LAYER_PREFIX}layers.{index}."
    return {weight_map[prefix + suffix] for suffix in LAYER_PARAMS}


# ---------------------------------------------------------------------------
# per-op fixtures (gate 3)
# ---------------------------------------------------------------------------


def op_fixtures(layer: Qwen3VLTextDecoderLayer, hidden: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                cos_f32: torch.Tensor, sin_f32: torch.Tensor,
                attn_mask: torch.Tensor | None,
                config: Qwen3VLTextConfig) -> dict[str, dict]:
    """Torch references for the two nontrivial encoder ops, driven by REAL
    layer-0 activations and REAL layer-0 weights (not synthetic noise), so the
    fixtures carry the magnitude distribution the port actually sees.

    Every op below is transformers' own code: `Qwen3VLTextRMSNorm.forward` for
    the per-head q/k norm and `apply_rotary_pos_emb` (rotate-half) for RoPE.
    """
    attn = layer.self_attn
    heads = config.num_attention_heads
    kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    seq = hidden.shape[1]
    out: dict[str, dict] = {}

    with torch.no_grad():
        normed = layer.input_layernorm(hidden)
        q_lin = attn.q_proj(normed).view(1, seq, heads, head_dim)
        k_lin = attn.k_proj(normed).view(1, seq, kv_heads, head_dim)
        v_lin = attn.v_proj(normed).view(1, seq, kv_heads, head_dim)

        # QkNormPartialRope: per-head RMSNorm over head_dim, then rotate-half
        # RoPE over the FULL 128 dims (RotaryDim == head_dim).
        q_normed = attn.q_norm(q_lin).transpose(1, 2)   # [1,H,S,D]
        k_normed = attn.k_norm(k_lin).transpose(1, 2)   # [1,KvH,S,D]
        q_rope, k_rope = apply_rotary_pos_emb(q_normed, k_normed, cos, sin)

        def rows(value: torch.Tensor) -> torch.Tensor:  # [1,H,S,D] -> [S,H,D]
            return value[0].transpose(0, 1).contiguous()

        # cos/sin are dumped twice on purpose: transformers computes them in
        # F32 and casts to the activation dtype before use, so `cos`/`sin` are
        # the BF16 tables that actually produced `expected_output`, while
        # `cos_f32`/`sin_f32` are the pre-cast values a port whose
        # RotaryPosition op emits F32 should compare its table against.
        out["op_qknorm_rope_q_h64"] = {
            "input": q_lin[0].contiguous(),          # [S,64,128]
            "weight": attn.q_norm.weight.detach().clone(),   # [128]
            "cos": cos[0].contiguous(),              # [S,128] BF16, as used
            "sin": sin[0].contiguous(),              # [S,128] BF16, as used
            "cos_f32": cos_f32.contiguous(),         # [S,128] F32, pre-cast
            "sin_f32": sin_f32.contiguous(),
            "expected_output": rows(q_rope),         # [S,64,128]
        }
        out["op_qknorm_rope_k_h8"] = {
            "input": k_lin[0].contiguous(),          # [S,8,128]
            "weight": attn.k_norm.weight.detach().clone(),
            "cos": cos[0].contiguous(),
            "sin": sin[0].contiguous(),
            "cos_f32": cos_f32.contiguous(),
            "sin_f32": sin_f32.contiguous(),
            "expected_output": rows(k_rope),         # [S,8,128]
        }

        # GQA causal SDPA at H=64 / KvH=8 / D=128, BF16.
        v_heads = v_lin.transpose(1, 2)              # [1,KvH,S,D]
        scale = 1.0 / math.sqrt(head_dim)
        gqa = F.scaled_dot_product_attention(
            q_rope, k_rope, v_heads, attn_mask=None, dropout_p=0.0,
            scale=scale, is_causal=True, enable_gqa=True)
        groups = heads // kv_heads
        repeated = F.scaled_dot_product_attention(
            q_rope,
            k_rope.repeat_interleave(groups, dim=1),
            v_heads.repeat_interleave(groups, dim=1),
            attn_mask=None, dropout_p=0.0, scale=scale, is_causal=True)

        # THIRD variant, and the one the depth ladder actually executed:
        # transformers passes `position_ids` into `create_causal_mask`, which
        # triggers packed-sequence detection and therefore sets
        # allow_is_causal_skip=False (masking_utils.py:735, 817). The encoder
        # thus runs SDPA with an EXPLICIT additive [1,1,S,S] mask and
        # is_causal=False, and because that mask is not None,
        # `use_gqa_in_sdpa()` is false so transformers materialises K/V with
        # `repeat_kv` rather than enable_gqa (integrations/sdpa_attention.py:30).
        masked = None
        if attn_mask is not None:
            masked = F.scaled_dot_product_attention(
                q_rope,
                k_rope.repeat_interleave(groups, dim=1),
                v_heads.repeat_interleave(groups, dim=1),
                attn_mask=attn_mask, dropout_p=0.0, scale=scale, is_causal=False)

        out["op_gqa_sdpa_h64_kv8"] = {
            "q": rows(q_rope),                       # [S,64,128]
            "k": rows(k_rope),                       # [S,8,128]
            "v": rows(v_heads),                      # [S,8,128]
            # PRIMARY reference, and the DiffIR semantic under test:
            # Attention + Causal attr + KvHeads=8, i.e.
            # F.scaled_dot_product_attention(is_causal=True, enable_gqa=True).
            "expected_output": rows(gqa),            # [S,64,128]
            # CROSS-CHECK A: same math, K/V materially repeated to 64 heads.
            # Any delta is pure kernel choice, not semantics.
            "expected_output_repeat_interleave": rows(repeated),
        }
        if masked is not None:
            # CROSS-CHECK B: the exact call the encoder layer made. A delta
            # here is the cost of expressing causality as the Causal attr
            # instead of an additive mask.
            out["op_gqa_sdpa_h64_kv8"]["expected_output_additive_mask"] = rows(masked)
    return out


# ---------------------------------------------------------------------------
# comparison (run-to-run spread / port gating)
# ---------------------------------------------------------------------------


def _ulp_key(value: torch.Tensor) -> torch.Tensor:
    """Map BF16 bit patterns onto a monotonic integer line so that adjacent
    representable values differ by 1. Raw bit subtraction is WRONG for
    sign-magnitude floats: +0x0001 and -0x0001 are neighbours around zero but
    their raw patterns differ by 0x8000."""
    bits = value.view(torch.uint16).flatten().to(torch.int64)
    negative = bits >= 0x8000
    return torch.where(negative, 0x8000 - (bits & 0x7FFF), bits + 0x8000)


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    """The six reported metrics: cosine, rel-L2, max-abs, mean-abs,
    bit-mismatch count, and max ULP distance.

    Accumulation is float64 on purpose. In F32 the cosine of two BF16 tensors
    with ~2.2 M elements accumulates enough error to read above 1.0 even when
    the tensors are bit-identical, which would silently flatter a port.
    """
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"error": f"shape/dtype mismatch {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"}
    af, bf = a.double().flatten(), b.double().flatten()
    diff = (af - bf).abs()
    denom = af.norm().item()
    identical = bool(torch.equal(a, b))
    if a.dtype == torch.bfloat16:
        bits_a, bits_b = a.view(torch.uint16).flatten(), b.view(torch.uint16).flatten()
        mismatch = int((bits_a != bits_b).sum().item())
        max_ulp = int((_ulp_key(a) - _ulp_key(b)).abs().max().item())
    else:
        mismatch = int((af != bf).sum().item())
        max_ulp = -1  # only defined for the BF16 storage contract
    norms = af.norm().item() * bf.norm().item()
    return {
        "elements": int(af.numel()),
        "cosine": float(torch.dot(af, bf).item() / norms) if norms > 0 else 1.0,
        "rel_l2": float((af - bf).norm().item() / denom) if denom > 0 else 0.0,
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "bit_mismatch": mismatch,
        "bit_mismatch_frac": mismatch / af.numel(),
        "max_ulp": max_ulp,
        "bit_identical": identical,
    }


def compare_runs(current: Path, previous: Path) -> dict:
    report: dict[str, dict] = {}
    for path in sorted(current.glob("*.safetensors")):
        other = previous / path.name
        if not other.exists():
            continue
        here, there = load_file(str(path)), load_file(str(other))
        for name, value in here.items():
            if name not in there or value.dtype not in (torch.bfloat16, torch.float32):
                continue
            report[f"{path.stem}:{name}"] = metrics(value, there[name])
    return report


def print_comparison(report: dict, current: Path, previous: Path) -> None:
    print(f"\n  {current} vs {previous}")
    print(f"    {'tensor':44s} {'cosine':>14s} {'rel_l2':>11s} {'max_abs':>10s} "
          f"{'mean_abs':>10s} {'bit_mism':>10s} {'frac':>8s} {'ulp':>7s}")
    for key, value in report.items():
        if "error" in value:
            print(f"    {key:44s} {value['error']}")
            continue
        print(f"    {key:44s} {value['cosine']:14.11f} {value['rel_l2']:11.3e} "
              f"{value['max_abs']:10.3e} {value['mean_abs']:10.3e} "
              f"{value['bit_mismatch']:10d} {value['bit_mismatch_frac']:8.2%} "
              f"{value['max_ulp']:7d}")


# ---------------------------------------------------------------------------


def read_ids(path: Path) -> list[int]:
    text = path.read_text()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed = parsed["ids"]
        return [int(v) for v in parsed]
    except json.JSONDecodeError:
        return [int(token) for token in text.split()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-encoder", type=Path, default=Path(DEFAULT_TEXT_ENCODER))
    parser.add_argument("--ids-json", type=Path, default=None,
                        help="token ids from tools/diftokenize --ids-out "
                             "(required unless --compare-only)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--depths", default="1,2,23,49,50")
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    parser.add_argument("--no-op-fixtures", action="store_true")
    parser.add_argument("--compare-to", type=Path, default=None)
    parser.add_argument("--compare-only", action="store_true",
                        help="compare --out-dir against --compare-to using the "
                             "fixtures already on disk; runs no model and needs "
                             "no GPU. Also the way to gate a port: point "
                             "--out-dir at the port's fixtures.")
    args = parser.parse_args()

    if args.compare_only:
        if args.compare_to is None:
            raise SystemExit("--compare-only requires --compare-to")
        report = compare_runs(args.out_dir, args.compare_to)
        print_comparison(report, args.out_dir, args.compare_to)
        target = args.out_dir / "comparison.json"
        target.write_text(json.dumps(
            {"current": str(args.out_dir), "previous": str(args.compare_to),
             "comparison": report}, indent=2) + "\n")
        print(f"\n  wrote {target}")
        return

    if args.ids_json is None:
        raise SystemExit("--ids-json is required (see tools/diftokenize --ids-out)")

    depths = sorted({int(d) for d in args.depths.split(",") if d.strip()})
    max_depth = max(depths)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    config = load_text_config(args.text_encoder, args.attn)
    if max_depth > config.num_hidden_layers:
        raise SystemExit(
            f"--depths asks for {max_depth} layers, checkpoint declares "
            f"{config.num_hidden_layers}")

    index_path = args.text_encoder / "model.safetensors.index.json"
    with open(index_path) as handle:
        weight_map = json.load(handle)["weight_map"]

    ids = read_ids(args.ids_json)
    seq = len(ids)
    print(f"Qwen3-VL conditioner oracle | transformers Qwen3VLTextDecoderLayer, "
          f"streamed one layer at a time")
    print(f"  checkpoint      {args.text_encoder}")
    print(f"  index sha256    {hashlib.sha256(index_path.read_bytes()).hexdigest()}")
    print(f"  tensors indexed {len(weight_map)}")
    print(f"  ids             {seq} from {args.ids_json}")
    print(f"  hidden {config.hidden_size} heads {config.num_attention_heads}/"
          f"{config.num_key_value_heads} head_dim {config.head_dim} "
          f"eps {config.rms_norm_eps} theta {config.rope_theta}")
    print(f"  attn impl       {config._attn_implementation}   dtype {DTYPE}")
    print(f"  depths          {depths}")

    reader = ShardReader(args.text_encoder, weight_map)
    manifest: dict = {
        "generator": "tools/export_qwen3vl_conditioner_reference.py",
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        "dtype": "bfloat16",
        "attn_implementation": config._attn_implementation,
        "text_encoder": str(args.text_encoder),
        "encoder_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "ids_source": str(args.ids_json),
        "ids_sha256": hashlib.sha256(
            b"".join(struct.pack("<i", i) for i in ids)).hexdigest(),
        "sequence_length": seq,
        "depths": depths,
        "fixtures": {},
    }

    ids_t = torch.tensor([ids], device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        # ---- embedding (gate 2) ------------------------------------------
        embed_name = LAYER_PREFIX + "embed_tokens.weight"
        table = reader.get(embed_name)
        print(f"  embed table     {tuple(table.shape)} {table.dtype}")
        hidden = table[ids_t[0].cpu()].to(DEVICE, DTYPE).unsqueeze(0)
        del table
        reader.retain_only(layer_shard_files(weight_map, 0))

        manifest["fixtures"]["embedding"] = emit(out_dir, "embedding", {
            "ids": torch.tensor(ids, dtype=torch.int64),
            "hidden_states_0": hidden[0],
        })
        print(f"  depth  0 (embed) |h|={hidden.float().norm().item():.6f}")

        # ---- shared position machinery -----------------------------------
        rotary = Qwen3VLTextRotaryEmbedding(config=config).to(DEVICE)
        cache_position = torch.arange(0, seq, device=DEVICE)
        position_ids = cache_position.view(1, 1, -1).expand(3, 1, -1)
        text_position_ids = position_ids[0]
        attn_mask = create_causal_mask(
            config=config,
            input_embeds=hidden,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=text_position_ids,
        )
        cos, sin = rotary(hidden, position_ids)
        print(f"  causal mask     {'None (sdpa is_causal path)' if attn_mask is None else tuple(attn_mask.shape)}")
        print(f"  cos/sin         {tuple(cos.shape)} {cos.dtype}")

        # MRoPE collapse check (plan §1, open hypothesis H1): for an all-text
        # prompt the three MRoPE axes carry identical positions, so
        # `apply_interleaved_mrope` must be a no-op and the 3-axis tables must
        # equal ordinary 1-D RoPE. Built here from scratch in F32 -
        # inv_freq[i] = theta^(-2i/128), emb = cat(freqs, freqs) - and
        # compared against transformers' own output. Measured, not assumed.
        inv_freq = rotary.inv_freq.float()                       # [64]
        freqs_1d = torch.outer(cache_position.float(), inv_freq)  # [S,64]
        emb_1d = torch.cat((freqs_1d, freqs_1d), dim=-1)          # [S,128]
        cos_f32 = emb_1d.cos() * rotary.attention_scaling
        sin_f32 = emb_1d.sin() * rotary.attention_scaling
        mrope_cos = (cos[0].float() - cos_f32.to(DTYPE).float()).abs().max().item()
        mrope_sin = (sin[0].float() - sin_f32.to(DTYPE).float()).abs().max().item()
        manifest["mrope_collapse_max_abs"] = {"cos": mrope_cos, "sin": mrope_sin}
        manifest["rope_attention_scaling"] = float(rotary.attention_scaling)
        print(f"  MRoPE collapse  max|3axis - 1d| cos={mrope_cos} sin={mrope_sin}"
              f"  (attention_scaling={rotary.attention_scaling})")

        # ---- depth ladder (gate 4) ---------------------------------------
        layer = build_layer(config)
        started = time.time()
        for index in range(max_depth):
            load_layer(layer, reader, index)
            if index == 0 and not args.no_op_fixtures:
                fixtures = op_fixtures(layer, hidden, cos, sin, cos_f32, sin_f32,
                                       attn_mask, config)
                for stem, tensors in fixtures.items():
                    manifest["fixtures"][stem] = emit(out_dir, stem, tensors)
                    print(f"  op fixture      {stem}")
            hidden = layer(
                hidden,
                attention_mask=attn_mask,
                position_ids=text_position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=(cos, sin),
            )
            depth = index + 1
            if depth in depths:
                torch.cuda.synchronize()
                record = emit(out_dir, f"hidden_states_{depth}",
                              {f"hidden_states_{depth}": hidden[0]})
                manifest["fixtures"][f"hidden_states_{depth}"] = record
                sha = record["tensors"][0]["payload_sha256"]
                print(f"  depth {depth:2d}        |h|={hidden.float().norm().item():.6f}"
                      f"  payload_sha256={sha}")
            if index + 1 < max_depth:
                reader.retain_only(layer_shard_files(weight_map, index + 1))
        reader.close()
        elapsed = time.time() - started

    manifest["shards_read"] = reader.opened
    manifest["stream_seconds"] = elapsed
    manifest["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
    print(f"  streamed {max_depth} layers in {elapsed:.1f}s over "
          f"{len(reader.opened)} shard opens; peak VRAM "
          f"{manifest['peak_vram_bytes'] / 2**30:.2f} GiB")

    if args.compare_to is not None:
        manifest["compare_to"] = str(args.compare_to)
        manifest["comparison"] = compare_runs(out_dir, args.compare_to)
        print_comparison(manifest["comparison"], out_dir, args.compare_to)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print("\n  MANIFEST")
    for stem, record in manifest["fixtures"].items():
        for entry in record["tensors"]:
            print(f"    {stem}/{entry['name']:38s} {str(entry['shape']):18s} "
                  f"{entry['dtype']:9s} {entry['payload_sha256']}")
    print(f"\n  wrote {manifest_path}")


if __name__ == "__main__":
    main()
