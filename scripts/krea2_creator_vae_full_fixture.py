#!/usr/bin/env python3
"""Official tiled Qwen-Image VAE full-image oracle (development only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from einops import rearrange
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file


SOURCE_COMMIT = "db3984fbc6e13b34c0064990fc2d95ac64d00058"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--sampler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--png", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.png.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite an existing full VAE artifact")
    if not torch.cuda.is_available():
        raise RuntimeError("Krea 2 creator full VAE fixture requires CUDA")

    from diffusers import AutoencoderKLQwenImage

    torch.backends.cuda.matmul.allow_tf32 = False
    with safe_open(args.sampler, framework="pt", device="cpu") as source:
        tokens = source.get_tensor("final_image_tokens")
    if tuple(tokens.shape) != (1, 4096, 64):
        raise ValueError(f"unexpected final token shape {tuple(tokens.shape)}")
    latent = rearrange(
        tokens,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        ph=2,
        pw=2,
        h=64,
        w=64,
    ).contiguous()

    started = time.perf_counter()
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.vae,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    vae.enable_tiling()
    latent_cuda = latent.to(device="cuda", dtype=torch.bfloat16).unsqueeze(2)
    latent_std = torch.tensor(
        vae.config.latents_std, device="cuda", dtype=torch.bfloat16
    ).view(1, -1, 1, 1, 1)
    latent_mean = torch.tensor(
        vae.config.latents_mean, device="cuda", dtype=torch.bfloat16
    ).view(1, -1, 1, 1, 1)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    decode_started = time.perf_counter()
    with torch.inference_mode():
        denormalized = (latent_cuda * latent_std) + latent_mean
        raw = vae.decode(denormalized).sample
        decoded = raw.clamp(-1.0, 1.0)
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_started
    raw_cpu = raw.squeeze(2).cpu().contiguous()
    decoded_cpu = decoded.squeeze(2).cpu().contiguous()
    save_file(
        {"raw_output": raw_cpu, "clamped_output": decoded_cpu},
        args.output,
        metadata={
            "creator_commit": SOURCE_COMMIT,
            "dtype": "BF16",
            "tiling": "latent32_stride24_sample256_stride192_blend64",
            "sampler_sha256": sha256(args.sampler),
        },
    )
    rgb = rearrange(
        (decoded * 0.5 + 0.5) * 255.0,
        "b c t h w -> b t h w c",
    ).cpu().byte().numpy()
    Image.fromarray(rgb[0, 0]).save(args.png)

    checkpoint = args.vae / "diffusion_pytorch_model.safetensors"
    report = {
        "creator_commit": SOURCE_COMMIT,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "sampler": str(args.sampler.resolve()),
        "sampler_sha256": sha256(args.sampler),
        "dtype": "BF16",
        "geometry": "1024x1024",
        "tiling": {
            "latent_tile": 32,
            "latent_stride": 24,
            "sample_tile": 256,
            "sample_stride": 192,
            "blend": 64,
            "tiles": 36,
        },
        "decode_seconds": decode_seconds,
        "wall_seconds": time.perf_counter() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "output": str(args.output.resolve()),
        "png": str(args.png.resolve()),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["output_sha256"] = sha256(args.output)
    report["png_sha256"] = sha256(args.png)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
