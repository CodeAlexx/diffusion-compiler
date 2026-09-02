#!/usr/bin/env python3
"""Development-only official FLUX.2 VAE decoder oracle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from einops import rearrange
from safetensors.torch import load_file, save_file


SOURCE_COMMIT = "50fe5162777813d869182b139e83b10743caef15"
VAE_REVISION = "26afe3a78bb242c0a8bb181dcc8937bb16e5c66c"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--creator-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latent-height", type=int, default=2)
    parser.add_argument("--latent-width", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().contiguous().cpu()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if args.latent_height <= 0 or args.latent_width <= 0:
        raise SystemExit("latent geometry must be positive")
    sys.path.insert(0, str(args.creator_source.resolve() / "src"))
    from flux2.autoencoder import AutoEncoder, AutoEncoderParams, swish

    device = torch.device(args.device)
    with torch.device("meta"):
        autoencoder = AutoEncoder(AutoEncoderParams())
    state = {
        name: value.to(device) for name, value in load_file(args.checkpoint).items()
    }
    autoencoder.load_state_dict(state, strict=True, assign=True)
    del state
    autoencoder.eval()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    tokens = (
        torch.randn(
            (args.latent_height * args.latent_width, 128),
            generator=generator,
            dtype=torch.float32,
        )
        * 0.25
    ).to(device=device, dtype=torch.bfloat16)
    z = rearrange(
        tokens,
        "(h w) c -> 1 c h w",
        h=args.latent_height,
        w=args.latent_width,
    )
    captures: dict[str, torch.Tensor] = {"latent_tokens_input": cpu(tokens)}

    with torch.no_grad():
        scale = torch.sqrt(
            autoencoder.bn.running_var.view(1, -1, 1, 1) + autoencoder.bn_eps
        )
        mean = autoencoder.bn.running_mean.view(1, -1, 1, 1)
        value = z * scale + mean
        captures["inv_normalize"] = cpu(value)
        value = rearrange(value, "b (c pi pj) i j -> b c (i pi) (j pj)", pi=2, pj=2)
        captures["pixel_shuffle"] = cpu(value)

        decoder = autoencoder.decoder
        value = decoder.post_quant_conv(value)
        captures["post_quant_conv"] = cpu(value)
        value = decoder.conv_in(value)
        captures["decoder_conv_in"] = cpu(value)
        value = decoder.mid.block_1(value)
        captures["mid_block_1"] = cpu(value)
        value = decoder.mid.attn_1(value)
        captures["mid_attention"] = cpu(value)
        value = decoder.mid.block_2(value)
        captures["mid_block_2"] = cpu(value)

        for level in reversed(range(decoder.num_resolutions)):
            for block_index in range(decoder.num_res_blocks + 1):
                value = decoder.up[level].block[block_index](value)
            captures[f"up_{level}_blocks"] = cpu(value)
            if level != 0:
                value = decoder.up[level].upsample(value)
                captures[f"up_{level}_upsample"] = cpu(value)

        value = decoder.norm_out(value)
        captures["decoder_norm_out"] = cpu(value)
        value = swish(value)
        value = decoder.conv_out(value)
        captures["raw_output"] = cpu(value)
        captures["clamped_output"] = cpu(value.clamp(-1, 1))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        captures,
        args.output,
        metadata={
            "oracle": "black-forest-labs/flux2 AutoEncoder.decode",
            "source_commit": SOURCE_COMMIT,
            "vae_revision": VAE_REVISION,
            "checkpoint": str(args.checkpoint.resolve()),
            "latent_height": str(args.latent_height),
            "latent_width": str(args.latent_width),
            "seed": str(args.seed),
            "torch": torch.__version__,
        },
    )
    print(
        f"FLUX2_VAE_ORACLE output={args.output} "
        f"latent={args.latent_height}x{args.latent_width} "
        f"pixels={args.latent_height * 16}x{args.latent_width * 16} "
        f"captures={len(captures)}"
    )


if __name__ == "__main__":
    main()
