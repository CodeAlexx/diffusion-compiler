#!/usr/bin/env python3
"""Creator-oracle fixture for one complete real-dimension Krea 2 evaluation.

The 26.28 GB Raw/Turbo checkpoint cannot reside on a 24 GB GPU. This development
oracle therefore mirrors the official module equations while loading one
block at a time. PyTorch is never part of the accepted native production path.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
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
SEQUENCE = TEXT_TOKENS + IMAGE_TOKENS
SEED = 20260831
CAPTURE_LAYERS = set(range(28))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--creator", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text-fixture", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-fixture",
        type=Path,
        help="tokenizer inputs paired with a native conditioning fixture",
    )
    parser.add_argument(
        "--image-fixture",
        type=Path,
        help="sampler fixture supplying initial_image_tokens",
    )
    parser.add_argument(
        "--image-key",
        default="initial_image_tokens",
        help="tensor name to read from --image-fixture",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--timestep", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.creator))
    from mmdit import (  # pylint: disable=import-error,import-outside-toplevel
        LastLayer,
        PositionalEncoding,
        SingleStreamBlock,
        _mask,
        temb,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Krea 2 creator fixture requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()

    with safe_open(args.text_fixture, framework="pt", device="cpu") as source:
        context_cpu = source.get_tensor("conditioning_output")
        if "validity_mask" in source.keys():
            text_mask_cpu = source.get_tensor("validity_mask")
        elif args.tokenizer_fixture is None:
            raise ValueError(
                "text fixture has no validity_mask; provide --tokenizer-fixture"
            )
    if "text_mask_cpu" not in locals():
        with safe_open(args.tokenizer_fixture, framework="pt", device="cpu") as source:
            text_mask_cpu = source.get_tensor("attention_mask")[:, 34:]
    if tuple(context_cpu.shape) != (1, TEXT_TOKENS, FEATURES):
        raise ValueError(f"unexpected text context shape {tuple(context_cpu.shape)}")
    if tuple(text_mask_cpu.shape) != (1, TEXT_TOKENS):
        raise ValueError(f"unexpected text mask shape {tuple(text_mask_cpu.shape)}")

    if args.image_fixture is None:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        image_tokens_cpu = torch.randn(
            (1, IMAGE_TOKENS, PATCH_FEATURES),
            generator=generator,
            dtype=torch.float32,
        ).to(torch.bfloat16)
    else:
        with safe_open(args.image_fixture, framework="pt", device="cpu") as source:
            image_tokens_cpu = source.get_tensor(args.image_key)
    if tuple(image_tokens_cpu.shape) != (1, IMAGE_TOKENS, PATCH_FEATURES):
        raise ValueError(f"unexpected image-token shape {tuple(image_tokens_cpu.shape)}")
    timestep_cpu = torch.tensor([args.timestep], dtype=torch.bfloat16)
    positions_cpu = torch.zeros((1, SEQUENCE, 3), dtype=torch.float32)
    image_y = torch.arange(64, dtype=torch.float32)[:, None]
    image_x = torch.arange(64, dtype=torch.float32)[None, :]
    positions_cpu[0, TEXT_TOKENS:, 1] = image_y.expand(64, 64).reshape(-1)
    positions_cpu[0, TEXT_TOKENS:, 2] = image_x.expand(64, 64).reshape(-1)
    validity_cpu = torch.cat(
        (text_mask_cpu, torch.ones((1, IMAGE_TOKENS), dtype=torch.bool)), dim=1
    )

    image_tokens = image_tokens_cpu.cuda()
    context = context_cpu.cuda()
    timestep = timestep_cpu.cuda()
    positions = positions_cpu.cuda()
    validity = validity_cpu.cuda()
    expanded_mask = _mask(validity)
    with torch.device("meta"):
        posemb = PositionalEncoding(FEATURES, [32, 48, 48], theta=1000.0)
    frequencies = posemb(positions)

    captures: dict[str, torch.Tensor] = {
        "image_tokens_input": image_tokens_cpu,
        "context_input": context_cpu,
        "timestep_input": timestep_cpu,
        "positions": positions_cpu,
        "validity_mask": validity_cpu,
    }
    block_seconds: list[float] = []
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
        first = load_meta(first, checkpoint, "first.")
        tmlp = load_meta(tmlp, checkpoint, "tmlp.")
        tproj = load_meta(tproj, checkpoint, "tproj.")
        torch.cuda.synchronize()
        top_started = time.perf_counter()
        with torch.inference_mode():
            projected_image = first(image_tokens)
            timestep_embedding = temb(
                timestep,
                TIMESTEP_DIM,
                device=image_tokens.device,
                dtype=image_tokens.dtype,
            )
            timestep_first_linear = tmlp[0](timestep_embedding)
            timestep_first_activation = tmlp[1](timestep_first_linear)
            time_vector = tmlp[2](timestep_first_activation)
            timestep_projection_activation = tproj[0](time_vector)
            modulation = tproj[1](timestep_projection_activation)
            sequence = torch.cat((context, projected_image), dim=1)
        torch.cuda.synchronize()
        top_seconds = time.perf_counter() - top_started
        captures["projected_image"] = projected_image.cpu()
        captures["timestep_embedding"] = timestep_embedding.cpu()
        captures["timestep_first_linear"] = timestep_first_linear.cpu()
        captures["timestep_first_activation"] = timestep_first_activation.cpu()
        captures["timestep_output"] = time_vector.cpu()
        captures["timestep_projection_activation"] = (
            timestep_projection_activation.cpu()
        )
        captures["modulation_output"] = modulation.cpu()
        del first, tmlp, tproj, projected_image
        torch.cuda.empty_cache()

        for layer in range(28):
            with torch.device("meta"):
                block = SingleStreamBlock(
                    features=FEATURES,
                    heads=HEADS,
                    multiplier=4,
                    bias=False,
                    kvheads=KV_HEADS,
                )
            block = load_meta(block, checkpoint, f"blocks.{layer}.")
            torch.cuda.synchronize()
            block_started = time.perf_counter()
            with torch.inference_mode():
                sequence = block(sequence, modulation, frequencies, expanded_mask)
            torch.cuda.synchronize()
            block_seconds.append(time.perf_counter() - block_started)
            if layer in CAPTURE_LAYERS:
                captures[f"block_{layer + 1}"] = sequence.cpu()
            del block
            torch.cuda.empty_cache()

        with torch.device("meta"):
            last = LastLayer(FEATURES, patch=2, channels=16)
        last = load_meta(last, checkpoint, "last.")
        torch.cuda.synchronize()
        last_started = time.perf_counter()
        with torch.inference_mode():
            final = last(sequence, time_vector)
            velocity = final[:, TEXT_TOKENS : TEXT_TOKENS + IMAGE_TOKENS, :]
        torch.cuda.synchronize()
        last_seconds = time.perf_counter() - last_started
        captures["velocity_output"] = velocity.cpu()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        captures,
        args.output,
        metadata={
            "creator_commit": SOURCE_COMMIT,
            "checkpoint": str(args.checkpoint),
            "text_fixture": str(args.text_fixture),
            "image_fixture": str(args.image_fixture) if args.image_fixture else "",
            "image_key": args.image_key,
            "seed": str(args.seed),
            "timestep": repr(args.timestep),
            "geometry": "B1_text512_image4096_D6144_patch64",
            "dtype": "BF16",
        },
    )
    report = {
        "creator_commit": SOURCE_COMMIT,
        "checkpoint": str(args.checkpoint),
        "text_fixture": str(args.text_fixture),
        "image_fixture": str(args.image_fixture) if args.image_fixture else None,
        "image_key": args.image_key,
        "seed": args.seed,
        "timestep": args.timestep,
        "dtype": "BF16",
        "top_seconds": top_seconds,
        "block_seconds": block_seconds,
        "block_total_seconds": sum(block_seconds),
        "last_seconds": last_seconds,
        "wall_seconds": time.perf_counter() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "fixture": str(args.output),
        "tensors": {name: list(tensor.shape) for name, tensor in captures.items()},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
