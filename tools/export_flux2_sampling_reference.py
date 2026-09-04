#!/usr/bin/env python3
"""Development-only, matched full FLUX.2 [klein] Base 9B creator oracle.

The native replay state supplies the exact initial latent, positive/negative
conditioning, and schedule.  This script uses the pinned creator modules for
the batch-two CFG denoise loop and official VAE.  Transformer weights are
loaded one layer at a time so the BF16 source reference can run on a 16 GiB
RTX 5080.  It is an oracle only and is never part of the native product path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file


# Creator geometry per model (flux2/model.py Klein9BParams / Flux2Params).
# klein9b: batch-two CFG sampler (guidance applied outside the model).
# dev: guidance-distilled, batch one, guidance fed through guidance_in.
GEOMETRY = {
    "klein9b": dict(hidden=4096, heads=32, double=8, single=24, context=12288,
                    guidance_embed=False, cfg=True,
                    revision="32773329fbe7e81a90ef971740e8ba4b0364ecf3"),
    "dev": dict(hidden=6144, heads=48, double=8, single=48, context=15360,
                guidance_embed=True, cfg=False,
                revision="26afe3a78bb242c0a8bb181dcc8937bb16e5c66c"),
}
HIDDEN = 4096
HEADS = 32
SOURCE_COMMIT = "50fe5162777813d869182b139e83b10743caef15"
MODEL_REVISION = "32773329fbe7e81a90ef971740e8ba4b0364ecf3"
VAE_REVISION = "26afe3a78bb242c0a8bb181dcc8937bb16e5c66c"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--creator-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--native-state", type=Path, required=True)
    parser.add_argument("--output-state", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--capture-every", type=int, default=0)
    parser.add_argument("--model", choices=sorted(GEOMETRY), default="klein9b")
    return parser.parse_args()


def cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().contiguous().cpu()


def payload_hash(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def positions(batch: int, image_tokens: int, text_tokens: int,
              device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    side = math.isqrt(image_tokens)
    if side * side != image_tokens:
        raise SystemExit("the matched image-token sequence must be square")
    text = torch.zeros((batch, text_tokens, 4), device=device,
                       dtype=torch.float32)
    text[:, :, 3] = torch.arange(text_tokens, device=device,
                                 dtype=torch.float32)
    image = torch.zeros((batch, image_tokens, 4), device=device,
                        dtype=torch.float32)
    index = torch.arange(image_tokens, device=device)
    image[:, :, 1] = (index // side).to(torch.float32)
    image[:, :, 2] = (index % side).to(torch.float32)
    return image, text


def main() -> None:
    args = arguments()
    for output in (args.output_state, args.output_png, args.report):
        if output.exists():
            raise SystemExit(f"refusing to overwrite {output}")
    if not math.isfinite(args.guidance) or args.guidance < 0:
        raise SystemExit("guidance must be finite and nonnegative")
    if args.capture_every < 0:
        raise SystemExit("capture-every must be nonnegative")

    geometry = GEOMETRY[args.model]
    global HIDDEN, HEADS, MODEL_REVISION
    HIDDEN, HEADS = geometry["hidden"], geometry["heads"]
    MODEL_REVISION = geometry["revision"]
    source = args.creator_source.resolve()
    sys.path.insert(0, str(source / "src"))
    from flux2.autoencoder import AutoEncoder, AutoEncoderParams
    from flux2.model import (
        DoubleStreamBlock,
        EmbedND,
        SingleStreamBlock,
        timestep_embedding,
    )

    device = torch.device(args.device)
    replay = load_file(args.native_state, device="cpu")
    required = {
        "initial_image_tokens",
        "positive_conditioning",
        "negative_conditioning",
        "timesteps",
    }
    if not required.issubset(replay):
        raise SystemExit(f"native state is missing {sorted(required - replay.keys())}")

    initial = replay["initial_image_tokens"]
    positive = replay["positive_conditioning"]
    negative = replay["negative_conditioning"]
    schedule = replay["timesteps"].float().tolist()
    if initial.ndim != 2 or initial.shape[1] != 128:
        raise SystemExit("initial_image_tokens must be [L,128]")
    if positive.shape != negative.shape or positive.shape != (512, geometry["context"]):
        raise SystemExit(f"conditioning tensors must both be [512,{geometry['context']}]")
    if len(schedule) < 2 or schedule[0] != 1.0 or schedule[-1] != 0.0:
        raise SystemExit("native schedule must include exact 1.0 and 0.0 endpoints")

    image_tokens = initial.shape[0]
    latent_side = math.isqrt(image_tokens)
    if latent_side * latent_side != image_tokens:
        raise SystemExit("image token count must be a square")
    batch = 2 if geometry["cfg"] else 1
    latent = initial.unsqueeze(0).to(device)
    if geometry["cfg"]:
        latent = torch.cat((latent, latent), dim=0)
        conditioning = torch.stack((negative, positive), dim=0).to(device)
    else:
        conditioning = positive.unsqueeze(0).to(device)
    image_positions, text_positions = positions(
        batch, image_tokens, positive.shape[0], device
    )

    captures: dict[str, torch.Tensor] = {
        "initial_image_tokens": initial.contiguous(),
        "timesteps": replay["timesteps"].contiguous(),
    }
    timings: list[float] = []
    keys_by_prefix: dict[str, list[str]] = {}

    with safe_open(args.checkpoint, framework="pt", device="cpu") as reader:
        checkpoint_keys = list(reader.keys())

        def names(prefix: str) -> list[str]:
            if prefix not in keys_by_prefix:
                keys_by_prefix[prefix] = [
                    name for name in checkpoint_keys if name.startswith(prefix)
                ]
            return keys_by_prefix[prefix]

        def linear(value: torch.Tensor, name: str) -> torch.Tensor:
            weight = reader.get_tensor(name).to(device)
            result = F.linear(value, weight)
            del weight
            return result

        def forward(value: torch.Tensor, context: torch.Tensor,
                    timestep_value: float) -> torch.Tensor:
            timestep = torch.full(
                (batch,), timestep_value, device=device, dtype=torch.bfloat16
            )
            embedded = timestep_embedding(timestep, 256)
            vector = linear(
                F.silu(linear(embedded, "time_in.in_layer.weight")),
                "time_in.out_layer.weight",
            )
            if geometry["guidance_embed"]:
                guidance_t = torch.full(
                    (batch,), args.guidance, device=device, dtype=torch.bfloat16
                )
                vector = vector + linear(
                    F.silu(linear(timestep_embedding(guidance_t, 256),
                                  "guidance_in.in_layer.weight")),
                    "guidance_in.out_layer.weight",
                )
            modulation_input = F.silu(vector)

            def modulation(name: str, chunks: int):
                packed = linear(modulation_input, name)[:, None, :]
                rows = packed.chunk(chunks, dim=-1)
                return rows[:3], rows[3:] if chunks == 6 else None

            image_modulation = modulation(
                "double_stream_modulation_img.lin.weight", 6
            )
            text_modulation = modulation(
                "double_stream_modulation_txt.lin.weight", 6
            )
            single_modulation, _ = modulation(
                "single_stream_modulation.lin.weight", 3
            )
            image = linear(value, "img_in.weight")
            text = linear(context, "txt_in.weight")
            pe_image = EmbedND(128, 2000, [32, 32, 32, 32]).to(device)(
                image_positions
            )
            pe_text = EmbedND(128, 2000, [32, 32, 32, 32]).to(device)(
                text_positions
            )

            for depth in range(geometry["double"]):
                prefix = f"double_blocks.{depth}."
                with torch.device("meta"):
                    block = DoubleStreamBlock(HIDDEN, HEADS, 3.0)
                state = {
                    name.removeprefix(prefix): reader.get_tensor(name).to(device)
                    for name in names(prefix)
                }
                block.load_state_dict(state, strict=True, assign=True)
                block.eval()
                del state
                image, text, cache = block.forward_kv_extract(
                    image,
                    text,
                    pe_image,
                    pe_text,
                    image_modulation,
                    text_modulation,
                    num_ref_tokens=0,
                )
                del block, cache

            sequence = torch.cat((text, image), dim=1)
            pe_full = torch.cat((pe_text, pe_image), dim=2)
            for depth in range(geometry["single"]):
                prefix = f"single_blocks.{depth}."
                with torch.device("meta"):
                    block = SingleStreamBlock(HIDDEN, HEADS, 3.0)
                state = {
                    name.removeprefix(prefix): reader.get_tensor(name).to(device)
                    for name in names(prefix)
                }
                block.load_state_dict(state, strict=True, assign=True)
                block.eval()
                del state
                sequence, cache = block.forward_kv_extract(
                    sequence,
                    pe_full,
                    single_modulation,
                    positive.shape[0],
                    num_ref_tokens=0,
                )
                del block, cache

            hidden = sequence[:, positive.shape[0]:]
            normalized = F.layer_norm(
                hidden, (HIDDEN,), weight=None, bias=None, eps=1e-6
            )
            final_modulation = linear(
                modulation_input, "final_layer.adaLN_modulation.1.weight"
            )
            shift, scale = final_modulation.chunk(2, dim=-1)
            modulated = (1 + scale[:, None, :]) * normalized + shift[:, None, :]
            return linear(modulated, "final_layer.linear.weight")

        with torch.no_grad():
            steps = len(schedule) - 1
            for step, (current, following) in enumerate(
                zip(schedule[:-1], schedule[1:])
            ):
                started = time.perf_counter()
                prediction = forward(latent, conditioning, current)
                if geometry["cfg"]:
                    unconditional, conditional = prediction.chunk(2)
                    guided = unconditional + args.guidance * (
                        conditional - unconditional
                    )
                    guided = torch.cat((guided, guided), dim=0)
                else:
                    guided = prediction  # guidance already embedded
                latent = latent + (following - current) * guided
                elapsed = (time.perf_counter() - started) * 1000.0
                timings.append(elapsed)
                if step == 0:
                    captures["first_step_image_tokens"] = cpu(latent[0])
                if step + 1 == (steps + 1) // 2:
                    captures["middle_step_image_tokens"] = cpu(latent[0])
                if args.capture_every and (step + 1) % args.capture_every == 0:
                    captures[f"step_{step + 1}_image_tokens"] = cpu(latent[0])
                print(
                    f"FLUX2_CREATOR_STEP step={step + 1}/{steps} ms={elapsed:.3f}",
                    flush=True,
                )

    final_tokens = cpu(latent[0])
    captures["final_image_tokens"] = final_tokens
    del latent, conditioning, image_positions, text_positions
    torch.cuda.empty_cache()

    with torch.device("meta"):
        autoencoder = AutoEncoder(AutoEncoderParams())
    vae_state = {
        name: value.to(device) for name, value in load_file(args.vae_checkpoint).items()
    }
    autoencoder.load_state_dict(vae_state, strict=True, assign=True)
    del vae_state
    autoencoder.eval()
    vae_started = time.perf_counter()
    with torch.no_grad():
        z = rearrange(
            final_tokens.to(device),
            "(h w) c -> 1 c h w",
            h=latent_side,
            w=latent_side,
        )
        pixels = autoencoder.decode(z).clamp(-1, 1)
    vae_ms = (time.perf_counter() - vae_started) * 1000.0
    captures["clamped_output"] = cpu(pixels)

    rgb = (
        ((pixels[0].permute(1, 2, 0).float() + 1.0) * 127.5)
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(args.output_png)

    args.output_state.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: value.contiguous() for name, value in captures.items()},
        args.output_state,
        metadata={
            "oracle": f"black-forest-labs/flux2 {args.model} matched sampler",
            "model": args.model,
            "source_commit": SOURCE_COMMIT,
            "model_revision": MODEL_REVISION,
            "vae_revision": VAE_REVISION,
            "checkpoint": str(args.checkpoint.resolve()),
            "vae_checkpoint": str(args.vae_checkpoint.resolve()),
            "native_state": str(args.native_state.resolve()),
            "guidance": str(args.guidance),
            "steps": str(len(schedule) - 1),
            "capture_every": str(args.capture_every),
            "torch": torch.__version__,
        },
    )
    report = {
        "oracle": f"black-forest-labs/flux2 {args.model} matched sampler",
        "model": args.model,
        "source_commit": SOURCE_COMMIT,
        "model_revision": MODEL_REVISION,
        "vae_revision": VAE_REVISION,
        "native_state": str(args.native_state.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "vae_checkpoint": str(args.vae_checkpoint.resolve()),
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "guidance": args.guidance,
        "steps": len(schedule) - 1,
        "capture_every": args.capture_every,
        "image_tokens": image_tokens,
        "initial_latent_sha256": payload_hash(initial),
        "final_latent_sha256": payload_hash(final_tokens),
        "pixels_sha256": payload_hash(captures["clamped_output"]),
        "png_sha256": hashlib.sha256(args.output_png.read_bytes()).hexdigest(),
        "timing_ms": {
            "denoise_total": sum(timings),
            "denoise_steps": timings,
            "vae": vae_ms,
        },
        "output_state": str(args.output_state),
        "output_png": str(args.output_png),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"FLUX2_CREATOR_ORACLE_PASS output={args.output_state} "
        f"png={args.output_png} denoise_ms={sum(timings):.3f} vae_ms={vae_ms:.3f}"
    )


if __name__ == "__main__":
    main()
