#!/usr/bin/env python3
"""Development-only creator oracle for one FLUX.2 [klein] double block.

This script is not part of the native product path. It imports the pinned
Black Forest Labs implementation, loads exactly one block, and writes a
first-divergence fixture for the shared DiffIR/native C++ implementation.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import load_file, save_file


HIDDEN = 4096
HEADS = 32
HEAD_DIM = 128
SOURCE_COMMIT = "50fe5162777813d869182b139e83b10743caef15"
MODEL_REVISION = "32773329fbe7e81a90ef971740e8ba4b0364ecf3"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--creator-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--image-tokens", type=int, default=8)
    parser.add_argument("--text-tokens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def flat(value: torch.Tensor) -> torch.Tensor:
    return value.detach().squeeze(0).contiguous().cpu()


def token_head(value: torch.Tensor) -> torch.Tensor:
    return value.detach().permute(0, 2, 1, 3).contiguous().cpu()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if not (0 <= args.block < 8):
        raise SystemExit("--block must be in [0, 7]")
    if args.image_tokens <= 0 or args.text_tokens <= 0:
        raise SystemExit("token counts must be positive")

    source = args.creator_source.resolve()
    sys.path.insert(0, str(source / "src"))
    from flux2.model import DoubleStreamBlock, EmbedND, apply_rope

    device = torch.device(args.device)
    prefix = f"double_blocks.{args.block}."
    with torch.device("meta"):
        block = DoubleStreamBlock(HIDDEN, HEADS, 3.0)
    with safe_open(args.checkpoint, framework="pt", device="cpu") as reader:
        state = {
            name.removeprefix(prefix): reader.get_tensor(name).to(device)
            for name in reader.keys()
            if name.startswith(prefix)
        }
    incompatible = block.load_state_dict(state, strict=True, assign=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")
    block.eval()

    if args.inputs:
        supplied = load_file(args.inputs)
        image = supplied["image_projected"].unsqueeze(0).to(device)
        text = supplied["text_projected"].unsqueeze(0).to(device)
        image_modulation = supplied["image_modulation"].to(device)
        text_modulation = supplied["text_modulation"].to(device)
        positions = supplied["position_ids"].to(device)
        text_positions = positions[:, : args.text_tokens]
        image_positions = positions[:, args.text_tokens :]
    else:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        image = (
            torch.randn(
                (1, args.image_tokens, HIDDEN),
                generator=generator,
                dtype=torch.float32,
            )
            * 0.25
        ).to(device=device, dtype=torch.bfloat16)
        text = (
            torch.randn(
                (1, args.text_tokens, HIDDEN),
                generator=generator,
                dtype=torch.float32,
            )
            * 0.25
        ).to(device=device, dtype=torch.bfloat16)
        image_modulation = (
            torch.randn((6, HIDDEN), generator=generator, dtype=torch.float32)
            * 0.05
        ).to(device=device, dtype=torch.bfloat16)
        text_modulation = (
            torch.randn((6, HIDDEN), generator=generator, dtype=torch.float32)
            * 0.05
        ).to(device=device, dtype=torch.bfloat16)

        text_positions = torch.zeros(
            (1, args.text_tokens, 4), device=device, dtype=torch.float32
        )
        text_positions[0, :, 3] = torch.arange(
            args.text_tokens, device=device, dtype=torch.float32
        )
        image_positions = torch.zeros(
            (1, args.image_tokens, 4), device=device, dtype=torch.float32
        )
        side = math.isqrt(args.image_tokens)
        height, width = (
            (side, side)
            if side * side == args.image_tokens
            else (1, args.image_tokens)
        )
        del height
        image_index = torch.arange(args.image_tokens, device=device)
        image_positions[0, :, 1] = (image_index // width).to(torch.float32)
        image_positions[0, :, 2] = (image_index % width).to(torch.float32)
        positions = torch.cat((text_positions, image_positions), dim=1)

    def mods(packed: torch.Tensor):
        rows = tuple(row.reshape(1, 1, HIDDEN) for row in packed)
        return rows[:3], rows[3:]

    image_mod1, image_mod2 = mods(image_modulation)
    text_mod1, text_mod2 = mods(text_modulation)

    embedder = EmbedND(HEAD_DIM, 2000, [32, 32, 32, 32]).to(device)
    pe = embedder(image_positions)
    pe_ctx = embedder(text_positions)

    captures: dict[str, torch.Tensor] = {
        "image_input": flat(image),
        "text_input": flat(text),
        "position_ids": positions.detach().contiguous().cpu(),
        "image_modulation": image_modulation.detach().contiguous().cpu(),
        "text_modulation": text_modulation.detach().contiguous().cpu(),
    }

    def prepare_stream(
        stream: str,
        label: str,
        value: torch.Tensor,
        norm: torch.nn.Module,
        attention: torch.nn.Module,
        modulation,
    ):
        shift, scale, _gate = modulation
        normalized = norm(value)
        modulated = (1 + scale) * normalized + shift
        packed = attention.qkv(modulated)
        q, k, v = rearrange(
            packed, "b l (n h d) -> n b h l d", n=3, h=HEADS
        )
        def rms_before_scale(x: torch.Tensor) -> torch.Tensor:
            x_dtype = x.dtype
            x_float = x.float()
            rrms = torch.rsqrt(
                torch.mean(x_float**2, dim=-1, keepdim=True) + 1e-6
            )
            return (x_float * rrms).to(dtype=x_dtype)

        q_rms = rms_before_scale(q)
        k_rms = rms_before_scale(k)
        q_normalized, k_normalized = attention.norm(q, k, v)
        captures[f"{label}_attention_norm"] = flat(normalized)
        captures[f"{label}_attention_modulated"] = flat(modulated)
        captures[f"{stream}_qkv"] = flat(packed)
        captures[f"{stream}_q"] = token_head(q).squeeze(0)
        captures[f"{stream}_k"] = token_head(k).squeeze(0)
        captures[f"{stream}_v"] = token_head(v).squeeze(0)
        captures[f"{stream}_q_rms"] = token_head(q_rms).squeeze(0)
        captures[f"{stream}_k_rms"] = token_head(k_rms).squeeze(0)
        captures[f"{stream}_q_norm"] = token_head(q_normalized).squeeze(0)
        captures[f"{stream}_k_norm"] = token_head(k_normalized).squeeze(0)
        return q_normalized, k_normalized, v

    with torch.no_grad():
        image_q, image_k, image_v = prepare_stream(
            "img", "image", image, block.img_norm1, block.img_attn, image_mod1
        )
        text_q, text_k, text_v = prepare_stream(
            "txt", "text", text, block.txt_norm1, block.txt_attn, text_mod1
        )
        query = torch.cat((text_q, image_q), dim=2)
        key = torch.cat((text_k, image_k), dim=2)
        value = torch.cat((text_v, image_v), dim=2)
        pe_full = torch.cat((pe_ctx, pe), dim=2)
        rotated_query, rotated_key = apply_rope(query, key, pe_full)
        attention_heads = F.scaled_dot_product_attention(
            rotated_query, rotated_key, value, is_causal=False
        )
        attention_flat = rearrange(attention_heads, "b h n d -> b n (h d)")
        text_attention = attention_flat[:, : args.text_tokens]
        image_attention = attention_flat[:, args.text_tokens :]

        captures["query"] = token_head(query)
        captures["key"] = token_head(key)
        captures["value"] = token_head(value)
        captures["rotated_query"] = token_head(rotated_query)
        captures["rotated_key"] = token_head(rotated_key)
        captures["attention"] = token_head(attention_heads)

        def finish_stream(
            stream: str,
            residual: torch.Tensor,
            attention_rows: torch.Tensor,
            attention_module: torch.nn.Module,
            norm2: torch.nn.Module,
            mlp: torch.nn.Sequential,
            mod1,
            mod2,
        ) -> torch.Tensor:
            _shift1, _scale1, gate1 = mod1
            shift2, scale2, gate2 = mod2
            projected = attention_module.proj(attention_rows)
            attention_residual = residual + gate1 * projected
            mlp_normalized = norm2(attention_residual)
            mlp_modulated = (1 + scale2) * mlp_normalized + shift2
            packed = mlp[0](mlp_modulated)
            gate, mlp_value = packed.chunk(2, dim=-1)
            activation = F.silu(gate) * mlp_value
            down = mlp[2](activation)
            output = attention_residual + gate2 * down
            captures[f"{stream}_attention_projected"] = flat(projected)
            captures[f"{stream}_attention_residual"] = flat(attention_residual)
            captures[f"{stream}_mlp_norm"] = flat(mlp_normalized)
            captures[f"{stream}_mlp_modulated"] = flat(mlp_modulated)
            captures[f"{stream}_mlp_packed"] = flat(packed)
            captures[f"{stream}_mlp_gate"] = flat(gate)
            captures[f"{stream}_mlp_value"] = flat(mlp_value)
            captures[f"{stream}_mlp_silu"] = flat(F.silu(gate))
            captures[f"{stream}_mlp_activation"] = flat(activation)
            captures[f"{stream}_mlp_down"] = flat(down)
            captures[f"{stream}_output"] = flat(output)
            return output

        image_output = finish_stream(
            "img",
            image,
            image_attention,
            block.img_attn,
            block.img_norm2,
            block.img_mlp,
            image_mod1,
            image_mod2,
        )
        text_output = finish_stream(
            "txt",
            text,
            text_attention,
            block.txt_attn,
            block.txt_norm2,
            block.txt_mlp,
            text_mod1,
            text_mod2,
        )
        captures["image_output"] = flat(image_output)
        captures["text_output"] = flat(text_output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: value.contiguous() for name, value in captures.items()},
        args.output,
        metadata={
            "oracle": "black-forest-labs/flux2 DoubleStreamBlock",
            "source_commit": SOURCE_COMMIT,
            "model_revision": MODEL_REVISION,
            "checkpoint": str(args.checkpoint.resolve()),
            "block": str(args.block),
            "image_tokens": str(args.image_tokens),
            "text_tokens": str(args.text_tokens),
            "seed": str(args.seed),
            "torch": torch.__version__,
            "attention": "scaled_dot_product_attention",
        },
    )
    print(
        f"FLUX2_DOUBLE_BLOCK_ORACLE output={args.output} "
        f"block={args.block} image_tokens={args.image_tokens} "
        f"text_tokens={args.text_tokens} captures={len(captures)}"
    )


if __name__ == "__main__":
    main()
