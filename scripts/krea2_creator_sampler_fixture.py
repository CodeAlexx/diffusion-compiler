#!/usr/bin/env python3
"""Official Krea 2 Raw/Turbo sampler trajectory oracle (development only).

The released 26.28 GB checkpoint does not fit wholly on a 24 GB GPU. This
fixture keeps the small top/final modules resident and streams one official
SingleStreamBlock at a time. It executes the creator equations, CFG ordering,
BF16 eager boundaries, schedule, CUDA Philox starting noise, and Euler updates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import save_file


SOURCE_COMMIT = "db3984fbc6e13b34c0064990fc2d95ac64d00058"
FEATURES = 6144
TIMESTEP_DIM = 256
HEADS = 48
KV_HEADS = 12
TEXT_TOKENS = 512
IMAGE_TOKENS = 4096
PATCH_FEATURES = 64
PREFIX_TOKENS = 34
LAYERS = 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--creator", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--positive-conditioning", type=Path, required=True)
    parser.add_argument("--positive-tokenizer", type=Path, required=True)
    parser.add_argument("--negative-conditioning", type=Path)
    parser.add_argument("--negative-tokenizer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=52)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument(
        "--mu",
        type=float,
        default=None,
        help="fixed timestep shift; official Turbo uses 1.15",
    )
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--stop-after",
        type=int,
        default=0,
        help="development smoke only; zero executes the complete schedule",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_state(checkpoint, prefix: str) -> dict[str, torch.Tensor]:
    names = sorted(name for name in checkpoint.keys() if name.startswith(prefix))
    return {name[len(prefix) :]: checkpoint.get_tensor(name) for name in names}


def load_meta(module, checkpoint, prefix: str):
    state = module_state(checkpoint, prefix)
    missing, unexpected = module.load_state_dict(state, strict=True, assign=True)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch for {prefix}: missing={missing} unexpected={unexpected}"
        )
    del state
    return module.to(device="cuda", dtype=torch.bfloat16).eval().requires_grad_(False)


def load_conditioning(conditioning_path: Path, tokenizer_path: Path):
    with safe_open(conditioning_path, framework="pt", device="cpu") as source:
        context = source.get_tensor("conditioning_output")
    with safe_open(tokenizer_path, framework="pt", device="cpu") as source:
        mask = source.get_tensor("attention_mask")[:, PREFIX_TOKENS:]
    if tuple(context.shape) != (1, TEXT_TOKENS, FEATURES):
        raise ValueError(f"unexpected conditioning shape {tuple(context.shape)}")
    if tuple(mask.shape) != (1, TEXT_TOKENS):
        raise ValueError(f"unexpected text mask shape {tuple(mask.shape)}")
    return context.contiguous(), mask.bool().contiguous()


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite an existing sampler artifact")
    if args.steps <= 0 or args.guidance < 0:
        raise ValueError("sampler requires positive steps and nonnegative guidance")
    cfg = args.guidance > 0
    if cfg and (args.negative_conditioning is None or args.negative_tokenizer is None):
        raise ValueError("positive guidance requires negative conditioning and tokenizer")
    execute_steps = args.steps if args.stop_after == 0 else args.stop_after
    if execute_steps <= 0 or execute_steps > args.steps:
        raise ValueError("--stop-after must be zero or in [1,steps]")
    if not torch.cuda.is_available():
        raise RuntimeError("Krea 2 creator sampler fixture requires CUDA")

    sys.path.insert(0, str(args.creator))
    from mmdit import (  # pylint: disable=import-error,import-outside-toplevel
        LastLayer,
        PositionalEncoding,
        SingleStreamBlock,
        _mask,
        temb,
    )
    from sampling import timesteps  # pylint: disable=import-error,import-outside-toplevel

    torch.backends.cuda.matmul.allow_tf32 = False
    positive_cpu, positive_mask_cpu = load_conditioning(
        args.positive_conditioning, args.positive_tokenizer
    )
    if cfg:
        negative_cpu, negative_mask_cpu = load_conditioning(
            args.negative_conditioning, args.negative_tokenizer
        )

    started = time.perf_counter()
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    noise = torch.randn(
        (1, 16, 128, 128),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    image = rearrange(
        noise,
        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
        ph=2,
        pw=2,
    ).contiguous()
    initial = image.cpu()

    positions = torch.zeros((1, TEXT_TOKENS + IMAGE_TOKENS, 3), device="cuda")
    image_y = torch.arange(64, device="cuda", dtype=torch.float32)[:, None]
    image_x = torch.arange(64, device="cuda", dtype=torch.float32)[None, :]
    positions[0, TEXT_TOKENS:, 1] = image_y.expand(64, 64).reshape(-1)
    positions[0, TEXT_TOKENS:, 2] = image_x.expand(64, 64).reshape(-1)
    image_mask = torch.ones((1, IMAGE_TOKENS), device="cuda", dtype=torch.bool)
    positive_mask = torch.cat((positive_mask_cpu.cuda(), image_mask), dim=1)
    if cfg:
        negative_mask = torch.cat((negative_mask_cpu.cuda(), image_mask), dim=1)
    positive_attention = _mask(positive_mask)
    if cfg:
        negative_attention = _mask(negative_mask)
    positive = positive_cpu.cuda()
    if cfg:
        negative = negative_cpu.cuda()
    with torch.device("meta"):
        positional = PositionalEncoding(FEATURES, [32, 48, 48], theta=1000.0)
    frequencies = positional(positions)

    x1 = (256 // 16) ** 2
    x2 = (1280 // 16) ** 2
    schedule = timesteps(
        IMAGE_TOKENS, args.steps, x1, x2, y1=0.5, y2=1.15, mu=args.mu
    )
    if args.mu is None:
        slope = (1.15 - 0.5) / (x2 - x1)
        used_mu = slope * IMAGE_TOKENS + (0.5 - slope * x1)
    else:
        used_mu = args.mu
    midpoint = args.steps // 2
    captures: dict[str, torch.Tensor] = {
        "initial_image_tokens": initial,
        "timesteps": torch.tensor(schedule, dtype=torch.float32),
    }
    evaluation_seconds: list[dict[str, float]] = []
    torch.cuda.reset_peak_memory_stats()

    with safe_open(args.checkpoint, framework="pt", device="cpu") as checkpoint:
        with torch.device("meta"):
            first = torch.nn.Linear(PATCH_FEATURES, FEATURES, bias=True)
            tmlp = torch.nn.Sequential(
                torch.nn.Linear(TIMESTEP_DIM, FEATURES),
                torch.nn.GELU(approximate="tanh"),
                torch.nn.Linear(FEATURES, FEATURES),
            )
            tproj = torch.nn.Sequential(
                torch.nn.GELU(approximate="tanh"),
                torch.nn.Linear(FEATURES, 6 * FEATURES),
            )
            last = LastLayer(FEATURES, patch=2, channels=16)
        first = load_meta(first, checkpoint, "first.")
        tmlp = load_meta(tmlp, checkpoint, "tmlp.")
        tproj = load_meta(tproj, checkpoint, "tproj.")
        last = load_meta(last, checkpoint, "last.")

        def evaluate(context, attention, timestep_value: float) -> torch.Tensor:
            timestep = torch.full(
                (1,), timestep_value, dtype=image.dtype, device=image.device
            )
            with torch.inference_mode():
                projected = first(image)
                time_vector = tmlp(
                    temb(
                        timestep,
                        TIMESTEP_DIM,
                        device=image.device,
                        dtype=image.dtype,
                    )
                )
                modulation = tproj(time_vector)
                sequence = torch.cat((context, projected), dim=1)
            del projected
            for layer in range(LAYERS):
                with torch.device("meta"):
                    block = SingleStreamBlock(
                        features=FEATURES,
                        heads=HEADS,
                        multiplier=4,
                        bias=False,
                        kvheads=KV_HEADS,
                    )
                block = load_meta(block, checkpoint, f"blocks.{layer}.")
                with torch.inference_mode():
                    sequence = block(sequence, modulation, frequencies, attention)
                del block
                torch.cuda.empty_cache()
            with torch.inference_mode():
                velocity = last(sequence, time_vector)[
                    :, TEXT_TOKENS : TEXT_TOKENS + IMAGE_TOKENS, :
                ]
            return velocity

        for step, (current, following) in enumerate(
            zip(schedule[:-1], schedule[1:]), start=1
        ):
            if step > execute_steps:
                break
            torch.cuda.synchronize()
            conditional_start = time.perf_counter()
            conditional = evaluate(positive, positive_attention, current)
            torch.cuda.synchronize()
            conditional_seconds = time.perf_counter() - conditional_start
            with torch.inference_mode():
                if cfg:
                    unconditional_start = time.perf_counter()
                    unconditional = evaluate(negative, negative_attention, current)
                    torch.cuda.synchronize()
                    unconditional_seconds = time.perf_counter() - unconditional_start
                    velocity = conditional + args.guidance * (
                        conditional - unconditional
                    )
                else:
                    unconditional_seconds = 0.0
                    velocity = conditional
                image = image + (following - current) * velocity
            torch.cuda.synchronize()
            captures[f"step_{step}_velocity"] = velocity.cpu()
            captures[f"step_{step}_image_tokens"] = image.cpu()
            if step == 1:
                captures["first_conditional_velocity"] = conditional.cpu()
                if cfg:
                    captures["first_unconditional_velocity"] = unconditional.cpu()
                    captures["first_guided_velocity"] = velocity.cpu()
                else:
                    captures["first_velocity"] = velocity.cpu()
            if step == midpoint:
                captures["midpoint_image_tokens"] = image.cpu()
            evaluation_seconds.append(
                {
                    "step": step,
                    "conditional": conditional_seconds,
                    "unconditional": unconditional_seconds,
                }
            )
            suffix = (
                f" uncond_s={unconditional_seconds:.3f}" if cfg else " cfg=off"
            )
            print(
                f"KREA2_CREATOR_STEP step={step}/{execute_steps} "
                f"cond_s={conditional_seconds:.3f}{suffix}",
                flush=True,
            )
            del conditional, velocity
            if cfg:
                del unconditional

    captures["final_image_tokens"] = image.cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: value.contiguous() for name, value in captures.items()},
        args.output,
        metadata={
            "creator_commit": SOURCE_COMMIT,
            "seed": str(args.seed),
            "steps": str(args.steps),
            "executed_steps": str(execute_steps),
            "guidance": repr(args.guidance),
            "mu": repr(used_mu),
            "dtype": "BF16",
            "geometry": "1024x1024_B1_text512_image4096_patch2",
        },
    )
    report = {
        "creator_commit": SOURCE_COMMIT,
        "checkpoint": str(args.checkpoint.resolve()),
        "positive_conditioning": str(args.positive_conditioning.resolve()),
        "negative_conditioning": (
            str(args.negative_conditioning.resolve()) if cfg else None
        ),
        "seed": args.seed,
        "steps": args.steps,
        "executed_steps": execute_steps,
        "guidance": args.guidance,
        "mu": used_mu,
        "dtype": "BF16",
        "geometry": "1024x1024",
        "evaluation_seconds": evaluation_seconds,
        "denoiser_seconds": sum(
            item["conditional"] + item["unconditional"]
            for item in evaluation_seconds
        ),
        "wall_seconds": time.perf_counter() - started,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "output": str(args.output.resolve()),
        "tensors": {name: list(value.shape) for name, value in captures.items()},
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["output_sha256"] = sha256(args.output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
