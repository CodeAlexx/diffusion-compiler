#!/usr/bin/env python3
"""Development-only creator oracle for one FLUX.2 [klein] single block."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import save_file


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


def positions(image_tokens: int, text_tokens: int, device: torch.device):
    text = torch.zeros((1, text_tokens, 4), device=device, dtype=torch.float32)
    text[0, :, 3] = torch.arange(text_tokens, device=device, dtype=torch.float32)
    image = torch.zeros((1, image_tokens, 4), device=device, dtype=torch.float32)
    side = math.isqrt(image_tokens)
    height, width = (side, side) if side * side == image_tokens else (1, image_tokens)
    del height
    index = torch.arange(image_tokens, device=device)
    image[0, :, 1] = (index // width).to(torch.float32)
    image[0, :, 2] = (index % width).to(torch.float32)
    return torch.cat((text, image), dim=1)


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if not (0 <= args.block < 24):
        raise SystemExit("--block must be in [0, 23]")
    if args.image_tokens <= 0 or args.text_tokens <= 0:
        raise SystemExit("token counts must be positive")
    source = args.creator_source.resolve()
    sys.path.insert(0, str(source / "src"))
    from flux2.model import EmbedND, SingleStreamBlock, apply_rope

    device = torch.device(args.device)
    prefix = f"single_blocks.{args.block}."
    with torch.device("meta"):
        block = SingleStreamBlock(HIDDEN, HEADS, 3.0)
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

    tokens = args.text_tokens + args.image_tokens
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    sequence = (
        torch.randn((1, tokens, HIDDEN), generator=generator, dtype=torch.float32)
        * 0.25
    ).to(device=device, dtype=torch.bfloat16)
    modulation = (
        torch.randn((3, HIDDEN), generator=generator, dtype=torch.float32) * 0.05
    ).to(device=device, dtype=torch.bfloat16)
    shift, scale, gate = tuple(row.reshape(1, 1, HIDDEN) for row in modulation)
    position_ids = positions(args.image_tokens, args.text_tokens, device)
    pe = EmbedND(HEAD_DIM, 2000, [32, 32, 32, 32]).to(device)(position_ids)

    captures: dict[str, torch.Tensor] = {
        "sequence_input": flat(sequence),
        "position_ids": position_ids.detach().contiguous().cpu(),
        "modulation": modulation.detach().contiguous().cpu(),
    }
    with torch.no_grad():
        normalized = block.pre_norm(sequence)
        modulated = (1 + scale) * normalized + shift
        first = block.linear1(modulated)
        packed_qkv, packed_feedforward = torch.split(
            first, [3 * HIDDEN, 2 * block.mlp_hidden_dim], dim=-1
        )
        query, key, value = rearrange(
            packed_qkv, "b l (n h d) -> n b h l d", n=3, h=HEADS
        )
        query_normalized, key_normalized = block.norm(query, key, value)
        rotated_query, rotated_key = apply_rope(
            query_normalized, key_normalized, pe
        )
        attention_heads = F.scaled_dot_product_attention(
            rotated_query, rotated_key, value, is_causal=False
        )
        attention = rearrange(attention_heads, "b h n d -> b n (h d)")
        mlp_activation = block.mlp_act(packed_feedforward)
        linear2_input = torch.cat((attention, mlp_activation), dim=2)
        branch = block.linear2(linear2_input)
        output = sequence + gate * branch

        captures.update(
            {
                "pre_norm": flat(normalized),
                "modulated": flat(modulated),
                "linear1": flat(first),
                "query": token_head(query).squeeze(0),
                "key": token_head(key).squeeze(0),
                "value": token_head(value).squeeze(0),
                "query_normalized": token_head(query_normalized).squeeze(0),
                "key_normalized": token_head(key_normalized).squeeze(0),
                "rotated_query": token_head(rotated_query),
                "rotated_key": token_head(rotated_key),
                "attention": token_head(attention_heads),
                "mlp_activation": flat(mlp_activation),
                "linear2_input": flat(linear2_input),
                "linear2": flat(branch),
                "output": flat(output),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: value.contiguous() for name, value in captures.items()},
        args.output,
        metadata={
            "oracle": "black-forest-labs/flux2 SingleStreamBlock",
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
        f"FLUX2_SINGLE_BLOCK_ORACLE output={args.output} block={args.block} "
        f"tokens={tokens} captures={len(captures)}"
    )


if __name__ == "__main__":
    main()
