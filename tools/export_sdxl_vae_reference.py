#!/usr/bin/env python3
"""Creator oracle for the SDXL VAE decoder: the reference sampler's own KL
decoder (bf16, as it runs on this GPU class) on a seeded latent, with every
stage boundary the compiler frontend captures."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sdxl_reference_common import (
    VAE_PREFIX, Captures, add_reference_source, cpu, load_prefixed, refuse_overwrite, save)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latent-height", type=int, default=16)
    parser.add_argument("--latent-width", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--dtype", choices=["bf16", "f32"], default="bf16")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    refuse_overwrite(args.output)
    commit = add_reference_source(args.reference_source)
    import comfy.sd  # noqa: E402  (creator module)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    vae = comfy.sd.VAE(sd=load_prefixed(args.checkpoint, VAE_PREFIX), dtype=dtype)
    model = vae.first_stage_model.to(device="cuda", dtype=dtype).eval()
    decoder = model.decoder

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    # Unscaled latent (the sampler's process_latent_out already divided by
    # 0.13025), so magnitudes are those of a real decode input.
    latent = torch.randn(
        (1, 4, args.latent_height, args.latent_width), generator=generator, dtype=torch.float32
    ) * 5.0
    captures = Captures()
    captures.output(model.post_quant_conv, "post_quant_conv")
    captures.output(decoder.conv_in, "conv_in")
    captures.output(decoder.mid.block_1, "mid_block_1")
    captures.output(decoder.mid.attn_1, "mid_attn_1")
    captures.output(decoder.mid.block_2, "mid_block_2")
    for level in range(4):
        captures.output(decoder.up[level].block[2], f"up_{level}_blocks")
        if level != 0:
            captures.output(decoder.up[level].upsample, f"up_{level}_upsample")
    captures.output(decoder.norm_out, "norm_out")

    with torch.no_grad():
        z = latent.to(device="cuda", dtype=dtype)
        raw = model.decode(z)
    captures.release()
    values = {"latent_input": cpu(z), "raw_output": cpu(raw)}
    values.update(captures.values)
    save(args.output, values, {
        "creator_commit": commit, "checkpoint": args.checkpoint.name, "dtype": args.dtype,
        "seed": str(args.seed), "latent_scale": "5.0",
    })


if __name__ == "__main__":
    main()
