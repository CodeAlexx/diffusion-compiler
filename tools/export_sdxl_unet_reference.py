#!/usr/bin/env python3
"""Creator oracle for the SDXL UNet: the reference sampler's own UNetModel
(F16 weights, as it ran the speed baseline) on seeded inputs, with every
input/middle/output block boundary and the embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sdxl_reference_common import (
    Captures, add_reference_source, cpu, refuse_overwrite, save)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--latent-height", type=int, default=16)
    parser.add_argument("--latent-width", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--timesteps", type=float, nargs="+", default=[999.0, 500.0])
    parser.add_argument("--dtype", choices=["f16", "bf16"], default="f16")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    refuse_overwrite(args.output)
    if len(args.timesteps) != args.batch:
        raise SystemExit("one timestep per batch row")
    commit = add_reference_source(args.reference_source)
    import comfy.sd  # noqa: E402
    import comfy.model_management  # noqa: E402

    dtype = torch.float16 if args.dtype == "f16" else torch.bfloat16
    model, _clip, _vae, _vision = comfy.sd.load_checkpoint_guess_config(
        str(args.checkpoint), output_vae=False, output_clip=False, output_model=True)
    comfy.model_management.load_model_gpu(model)
    unet = model.model.diffusion_model.to(device="cuda", dtype=dtype).eval()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    shape = (args.batch, 4, args.latent_height, args.latent_width)
    # The UNet input is the sampler's scaled latent x / sqrt(sigma^2 + 1):
    # unit scale, so plain normal noise is representative.
    x = torch.randn(shape, generator=generator, dtype=torch.float32)
    timesteps = torch.tensor(args.timesteps, dtype=torch.float32)
    context = torch.randn((args.batch, 77, 2048), generator=generator, dtype=torch.float32)
    vector = torch.randn((args.batch, 2816), generator=generator, dtype=torch.float32)

    captures = Captures()
    captures.input(unet.time_embed, "time_embedding")
    captures.output(unet.time_embed, "time_embed_out")
    captures.output(unet.label_emb, "label_emb_out")
    for index, block in enumerate(unet.input_blocks):
        captures.output(block, f"input_block_{index}")
    captures.output(unet.middle_block, "middle_block")
    for index, block in enumerate(unet.output_blocks):
        captures.output(block, f"output_block_{index}")
    with torch.no_grad():
        xc = x.to(device="cuda", dtype=dtype)
        ctx = context.to(device="cuda", dtype=dtype)
        y = vector.to(device="cuda", dtype=dtype)
        out = unet(xc, timesteps.to("cuda"), context=ctx, y=y)
    captures.release()
    values = {
        "latent_input": cpu(xc), "timesteps": timesteps, "context": cpu(ctx),
        "vector": cpu(y), "output": cpu(out),
    }
    values.update(captures.values)
    values["emb"] = values.pop("time_embed_out") + values.pop("label_emb_out")
    save(args.output, values, {
        "creator_commit": commit, "checkpoint": args.checkpoint.name, "dtype": args.dtype,
        "seed": str(args.seed), "timesteps": " ".join(str(t) for t in args.timesteps),
    })


if __name__ == "__main__":
    main()
