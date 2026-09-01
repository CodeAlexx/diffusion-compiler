#!/usr/bin/env python3
"""Official Krea 2 Qwen-Image VAE tile oracle (development only).

This runs the released Diffusers decoder in creator BF16 mode on the first
official 32x32 latent tile and records the same boundaries exposed by the
native DiffIR frontend. PyTorch remains an oracle only; production decode is
implemented by the shared C++ runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import save_file


SOURCE_COMMIT = "db3984fbc6e13b34c0064990fc2d95ac64d00058"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--sampler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--tile-y", type=int, default=0)
    parser.add_argument("--tile-x", type=int, default=0)
    parser.add_argument("--profile-attention", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite an existing VAE artifact")
    if args.tile_y < 0 or args.tile_x < 0:
        raise ValueError("tile coordinates must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("Krea 2 creator VAE fixture requires CUDA")

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
    tile = latent[
        :,
        :,
        args.tile_y : args.tile_y + 32,
        args.tile_x : args.tile_x + 32,
    ].contiguous()
    if tuple(tile.shape) != (1, 16, 32, 32):
        raise ValueError("requested tile does not have the official 32x32 geometry")

    started = time.perf_counter()
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.vae,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)

    captures: dict[str, torch.Tensor] = {
        "latent_input": tile,
        "latent_std": torch.tensor(vae.latents_std, dtype=torch.bfloat16),
        "latent_mean": torch.tensor(vae.latents_mean, dtype=torch.bfloat16),
    }
    hooks = []

    def capture(name: str):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            if value.ndim == 5 and value.shape[2] == 1:
                value = value.squeeze(2)
            captures[name] = value.detach().cpu().contiguous()

        return hook

    hooks.append(vae.post_quant_conv.register_forward_hook(capture("post_quant_conv")))
    hooks.append(vae.decoder.conv_in.register_forward_hook(capture("decoder_conv_in")))
    first_residual = vae.decoder.mid_block.resnets[0]
    hooks.append(first_residual.norm1.register_forward_hook(capture("mid_resnet_0_norm1")))
    activation_index = [0]

    def capture_first_residual_activation(_module, _inputs, output):
        activation_index[0] += 1
        captures[
            f"mid_resnet_0_activation{activation_index[0]}"
        ] = output.detach().squeeze(2).cpu().contiguous()

    hooks.append(
        first_residual.nonlinearity.register_forward_hook(
            capture_first_residual_activation
        )
    )
    hooks.append(first_residual.conv1.register_forward_hook(capture("mid_resnet_0_conv1")))
    hooks.append(first_residual.norm2.register_forward_hook(capture("mid_resnet_0_norm2")))
    hooks.append(first_residual.conv2.register_forward_hook(capture("mid_resnet_0_conv2")))
    hooks.append(
        vae.decoder.mid_block.resnets[0].register_forward_hook(
            capture("mid_resnet_0")
        )
    )
    hooks.append(
        vae.decoder.mid_block.attentions[0].register_forward_hook(
            capture("mid_attention")
        )
    )
    attention = vae.decoder.mid_block.attentions[0]
    hooks.append(attention.norm.register_forward_hook(capture("mid_attention_norm")))
    hooks.append(attention.to_qkv.register_forward_hook(capture("mid_attention_qkv")))

    def capture_attention_input(_module, inputs):
        captures["mid_attention_sdpa"] = inputs[0].detach().cpu().contiguous()

    hooks.append(attention.proj.register_forward_pre_hook(capture_attention_input))
    hooks.append(attention.proj.register_forward_hook(capture("mid_attention_proj")))
    hooks.append(
        vae.decoder.mid_block.resnets[1].register_forward_hook(
            capture("mid_resnet_1")
        )
    )
    for index, block in enumerate(vae.decoder.up_blocks):
        hooks.append(
            block.resnets[-1].register_forward_hook(
                capture(f"up_block_{index}_residual")
            )
        )
        if block.upsamplers is not None:
            hooks.append(
                block.upsamplers[0].register_forward_hook(
                    capture(f"up_block_{index}_upsample")
                )
            )
    hooks.append(
        vae.decoder.norm_out.register_forward_hook(capture("decoder_norm_out"))
    )
    hooks.append(vae.decoder.conv_out.register_forward_hook(capture("raw_output")))

    tile_cuda = tile.to(device="cuda", dtype=torch.bfloat16).unsqueeze(2)
    latent_std = torch.tensor(
        vae.latents_std, device="cuda", dtype=torch.bfloat16
    ).view(1, -1, 1, 1, 1)
    latent_mean = torch.tensor(
        vae.latents_mean, device="cuda", dtype=torch.bfloat16
    ).view(1, -1, 1, 1, 1)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    decode_started = time.perf_counter()
    profiler = None
    if args.profile_attention:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        )
        profiler.__enter__()
    try:
        with torch.inference_mode():
            denormalized = (tile_cuda * latent_std) + latent_mean
            decoded = vae.decode(denormalized).sample
    finally:
        if profiler is not None:
            profiler.__exit__(None, None, None)
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_started
    captures["denormalized_latent"] = denormalized.squeeze(2).cpu().contiguous()
    captures["clamped_output"] = decoded.squeeze(2).cpu().contiguous()
    for handle in hooks:
        handle.remove()

    expected = {
        "post_quant_conv",
        "decoder_conv_in",
        "mid_resnet_0_norm1",
        "mid_resnet_0_activation1",
        "mid_resnet_0_conv1",
        "mid_resnet_0_norm2",
        "mid_resnet_0_activation2",
        "mid_resnet_0_conv2",
        "mid_resnet_0",
        "mid_attention_norm",
        "mid_attention_qkv",
        "mid_attention_sdpa",
        "mid_attention_proj",
        "mid_attention",
        "mid_resnet_1",
        "up_block_0_residual",
        "up_block_0_upsample",
        "up_block_1_residual",
        "up_block_1_upsample",
        "up_block_2_residual",
        "up_block_2_upsample",
        "up_block_3_residual",
        "decoder_norm_out",
        "raw_output",
        "clamped_output",
    }
    missing = sorted(expected - captures.keys())
    if missing:
        raise RuntimeError(f"missing VAE boundary captures: {missing}")

    save_file(
        {name: value.contiguous() for name, value in captures.items()},
        args.output,
        metadata={
            "creator_commit": SOURCE_COMMIT,
            "dtype": "BF16",
            "tile": f"{args.tile_y},{args.tile_x},32,32",
            "sampler_sha256": sha256(args.sampler),
        },
    )
    checkpoint = args.vae / "diffusion_pytorch_model.safetensors"
    report = {
        "creator_commit": SOURCE_COMMIT,
        "vae": str(args.vae.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "sampler": str(args.sampler.resolve()),
        "sampler_sha256": sha256(args.sampler),
        "dtype": "BF16",
        "tile": [args.tile_y, args.tile_x, 32, 32],
        "decode_seconds": decode_seconds,
        "wall_seconds": time.perf_counter() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "attention_profile_events": (
            sorted(
                event.key
                for event in profiler.key_averages()
                if "attention" in event.key.lower()
                or "scaled_dot_product" in event.key.lower()
                or "bmm" in event.key.lower()
            )
            if profiler is not None
            else []
        ),
        "output": str(args.output.resolve()),
        "tensors": {name: list(value.shape) for name, value in captures.items()},
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["output_sha256"] = sha256(args.output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
