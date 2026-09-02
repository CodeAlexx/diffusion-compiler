#!/usr/bin/env python3
"""Development-only, layer-streamed FLUX.2 [klein] transformer oracle.

The pinned creator modules are the semantic authority. Weights are loaded one
layer at a time so the 16.91 GiB BF16 transformer can be evaluated on the
16 GiB RTX 5080. This script is never part of the native product path.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file


HIDDEN = 4096
HEADS = 32
SOURCE_COMMIT = "50fe5162777813d869182b139e83b10743caef15"
MODEL_REVISION = "32773329fbe7e81a90ef971740e8ba4b0364ecf3"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--creator-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-tokens", type=int, default=8)
    parser.add_argument("--text-tokens", type=int, default=5)
    parser.add_argument("--double-depth", type=int, default=8)
    parser.add_argument("--single-depth", type=int, default=24)
    parser.add_argument("--timestep", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def unbatch(value: torch.Tensor) -> torch.Tensor:
    return value.detach().squeeze(0).contiguous().cpu()


def make_positions(image_tokens: int, text_tokens: int, device: torch.device):
    text = torch.zeros((1, text_tokens, 4), device=device, dtype=torch.float32)
    text[0, :, 3] = torch.arange(text_tokens, device=device, dtype=torch.float32)
    image = torch.zeros((1, image_tokens, 4), device=device, dtype=torch.float32)
    side = math.isqrt(image_tokens)
    width = side if side * side == image_tokens else image_tokens
    index = torch.arange(image_tokens, device=device)
    image[0, :, 1] = (index // width).to(torch.float32)
    image[0, :, 2] = (index % width).to(torch.float32)
    return image, text


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if not (0 <= args.double_depth <= 8 and 0 <= args.single_depth <= 24):
        raise SystemExit("depth is outside the creator architecture")
    if args.double_depth < 8 and args.single_depth != 0:
        raise SystemExit("single blocks require all eight double blocks")
    if args.image_tokens <= 0 or args.text_tokens <= 0:
        raise SystemExit("token counts must be positive")

    source = args.creator_source.resolve()
    sys.path.insert(0, str(source / "src"))
    from flux2.model import (
        DoubleStreamBlock,
        EmbedND,
        SingleStreamBlock,
        timestep_embedding,
    )

    device = torch.device(args.device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latent = (
        torch.randn(
            (1, args.image_tokens, 128), generator=generator, dtype=torch.float32
        )
        * 0.25
    ).to(device=device, dtype=torch.bfloat16)
    conditioning = (
        torch.randn(
            (1, args.text_tokens, 12288),
            generator=generator,
            dtype=torch.float32,
        )
        * 0.25
    ).to(device=device, dtype=torch.bfloat16)
    timestep = torch.full(
        (1,), args.timestep, device=device, dtype=torch.bfloat16
    )
    image_positions, text_positions = make_positions(
        args.image_tokens, args.text_tokens, device
    )
    position_ids = torch.cat((text_positions, image_positions), dim=1)
    captures: dict[str, torch.Tensor] = {
        "latent_input": unbatch(latent),
        "conditioning_input": unbatch(conditioning),
        "timestep_input": timestep.detach().contiguous().cpu(),
        "position_ids": position_ids.detach().contiguous().cpu(),
    }

    with safe_open(args.checkpoint, framework="pt", device="cpu") as reader:
        def linear(value: torch.Tensor, name: str) -> torch.Tensor:
            weight = reader.get_tensor(name).to(device)
            result = F.linear(value, weight)
            del weight
            return result

        with torch.no_grad():
            image = linear(latent, "img_in.weight")
            text = linear(conditioning, "txt_in.weight")
            captures["image_projected"] = unbatch(image)
            captures["text_projected"] = unbatch(text)

            embedded = timestep_embedding(timestep, 256)
            captures["timestep_embedding"] = embedded.detach().contiguous().cpu()
            vector = linear(F.silu(linear(embedded, "time_in.in_layer.weight")),
                            "time_in.out_layer.weight")
            captures["time_vector"] = vector.detach().contiguous().cpu()
            modulation_input = F.silu(vector)
            image_modulation = linear(
                modulation_input, "double_stream_modulation_img.lin.weight"
            ).reshape(6, HIDDEN)
            text_modulation = linear(
                modulation_input, "double_stream_modulation_txt.lin.weight"
            ).reshape(6, HIDDEN)
            single_modulation = linear(
                modulation_input, "single_stream_modulation.lin.weight"
            ).reshape(3, HIDDEN)
            captures["image_modulation"] = image_modulation.detach().contiguous().cpu()
            captures["text_modulation"] = text_modulation.detach().contiguous().cpu()
            captures["single_modulation"] = single_modulation.detach().contiguous().cpu()

            def double_mods(packed: torch.Tensor):
                rows = tuple(row.reshape(1, 1, HIDDEN) for row in packed)
                return rows[:3], rows[3:]

            image_mods = double_mods(image_modulation)
            text_mods = double_mods(text_modulation)
            pe_image = EmbedND(128, 2000, [32, 32, 32, 32]).to(device)(
                image_positions
            )
            pe_text = EmbedND(128, 2000, [32, 32, 32, 32]).to(device)(
                text_positions
            )

            for depth in range(args.double_depth):
                prefix = f"double_blocks.{depth}."
                with torch.device("meta"):
                    block = DoubleStreamBlock(HIDDEN, HEADS, 3.0)
                state = {
                    name.removeprefix(prefix): reader.get_tensor(name).to(device)
                    for name in reader.keys()
                    if name.startswith(prefix)
                }
                block.load_state_dict(state, strict=True, assign=True)
                block.eval()
                del state
                image, text, _cache = block.forward_kv_extract(
                    image,
                    text,
                    pe_image,
                    pe_text,
                    image_mods,
                    text_mods,
                    num_ref_tokens=0,
                )
                captures[f"double_{depth + 1}_image"] = unbatch(image)
                captures[f"double_{depth + 1}_text"] = unbatch(text)
                del block, _cache
                torch.cuda.empty_cache()

            if args.double_depth < 8:
                prediction = image
            else:
                sequence = torch.cat((text, image), dim=1)
                pe_full = torch.cat((pe_text, pe_image), dim=2)
                single_rows = tuple(
                    row.reshape(1, 1, HIDDEN) for row in single_modulation
                )
                for depth in range(args.single_depth):
                    prefix = f"single_blocks.{depth}."
                    with torch.device("meta"):
                        block = SingleStreamBlock(HIDDEN, HEADS, 3.0)
                    state = {
                        name.removeprefix(prefix): reader.get_tensor(name).to(device)
                        for name in reader.keys()
                        if name.startswith(prefix)
                    }
                    block.load_state_dict(state, strict=True, assign=True)
                    block.eval()
                    del state
                    sequence, _cache = block.forward_kv_extract(
                        sequence,
                        pe_full,
                        single_rows,
                        args.text_tokens,
                        num_ref_tokens=0,
                    )
                    captures[f"single_{depth + 1}"] = unbatch(sequence)
                    del block, _cache
                    torch.cuda.empty_cache()

                if args.single_depth < 24:
                    prediction = sequence
                else:
                    image_hidden = sequence[:, args.text_tokens :]
                    normalized = F.layer_norm(
                        image_hidden, (HIDDEN,), weight=None, bias=None, eps=1e-6
                    )
                    final_modulation = linear(
                        modulation_input, "final_layer.adaLN_modulation.1.weight"
                    )
                    shift, scale = final_modulation.chunk(2, dim=-1)
                    final_modulated = (1 + scale[:, None, :]) * normalized + shift[
                        :, None, :
                    ]
                    captures["final_modulated"] = unbatch(final_modulated)
                    prediction = linear(final_modulated, "final_layer.linear.weight")

            captures["prediction"] = unbatch(prediction)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: value.contiguous() for name, value in captures.items()},
        args.output,
        metadata={
            "oracle": "black-forest-labs/flux2 layer-streamed transformer",
            "source_commit": SOURCE_COMMIT,
            "model_revision": MODEL_REVISION,
            "checkpoint": str(args.checkpoint.resolve()),
            "double_depth": str(args.double_depth),
            "single_depth": str(args.single_depth),
            "image_tokens": str(args.image_tokens),
            "text_tokens": str(args.text_tokens),
            "timestep": str(args.timestep),
            "seed": str(args.seed),
            "torch": torch.__version__,
        },
    )
    print(
        f"FLUX2_TRANSFORMER_ORACLE output={args.output} "
        f"depth={args.double_depth}+{args.single_depth} "
        f"tokens={args.text_tokens + args.image_tokens} captures={len(captures)}"
    )


if __name__ == "__main__":
    main()
