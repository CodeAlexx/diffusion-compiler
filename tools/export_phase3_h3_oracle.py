#!/usr/bin/env python3
"""Capture a product-geometry H3 block oracle from the pinned source runtime.

This wrapper deliberately imports the accepted external Diffusers exporter and
does not import Diffusion Compiler.  It only replaces that exporter's small
fixed-input fixture with the exact geometry from a named Serenity product
profile.  All projections, token refinement, AdaLN, rotary construction,
released transformer blocks, and output tensors still execute in the pinned
source runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


DEFAULT_EXPORTER = Path(
    "/home/alex/diffusion-fixtures/oracles/export_minimax_h3_denoiser.py"
)
DEFAULT_PROFILES = Path(
    "/home/alex/mojodiffusion/serenitymojo/configs/"
    "minimax_h3_request_profiles.json"
)
DEFAULT_PROFILE = "minimax-h3-832x480x73-24fps"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_profile(path: Path, profile_id: str) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    for profile in document["profiles"]:
        if profile["id"] == profile_id:
            return profile
    raise RuntimeError(f"product profile not found: {profile_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--source-exporter", type=Path, default=DEFAULT_EXPORTER)
    parser.add_argument("--index", type=Path, default=None)
    parser.add_argument("--dump-block-stages", type=int, default=0)
    args = parser.parse_args()
    if args.layers < 1 or args.layers > 50:
        parser.error("layers must be in [1, 50]")
    if args.dump_block_stages < 0 or args.dump_block_stages > args.layers:
        parser.error("dump-block-stages must be zero or in [1, layers]")

    profile_path = args.profiles.resolve()
    exporter_path = args.source_exporter.resolve()
    profile = load_profile(profile_path, args.profile)
    width = int(profile["width"])
    height = int(profile["height"])
    frames = int(profile["frames"])
    fps = int(profile["fps"])
    text_tokens = int(profile["text_tokens"])
    expected_sequence = int(profile["sequence_tokens"])
    if width % 16 != 0 or height % 16 != 0:
        raise RuntimeError("product canvas must be divisible by H3's VAE ratio 16")
    if frames < 5 or (frames - 5) % 17 != 0:
        raise RuntimeError("product frame count must follow H3's 17n+5 contract")

    # Released H3 geometry: 17n+5 pixel frames become 5n+2 latent frames;
    # the video VAE is 16x spatial and the transformer patch is 1x2x2.
    latent_frames = ((frames - 5) // 17) * 5 + 2
    latent_height = height // 16
    latent_width = width // 16
    # The stereo audio VAE emits 40 latent rows per second per channel.
    audio_frames = (frames * 40 + fps - 1) // fps

    sys.path.insert(0, str(exporter_path.parent))
    import export_minimax_h3_denoiser as source  # pylint: disable=import-error

    text_tags = torch.ones(text_tokens, dtype=torch.int32)
    layout = source.build_packed_sequence(
        text_tags,
        latent_frames,
        latent_height,
        latent_width,
        audio_frames,
        (1, 2, 2),
        (),
    )
    timestep, timestep_indices = source.build_row_timesteps(
        layout, 0.25, 0.5, 0.999, 1.0
    )
    video_tokens = int(layout.video_indices.numel())
    audio_tokens = int(layout.audio_indices.numel())
    sequence = int(layout.sequence_length)
    if sequence != expected_sequence:
        raise RuntimeError(
            f"source packing produced S={sequence}, profile records "
            f"S={expected_sequence}"
        )
    if sequence != video_tokens + audio_tokens + text_tokens:
        raise RuntimeError("source packing modality counts do not sum")
    if int(timestep.numel()) != 2:
        raise RuntimeError("T2VA product fixture must have two timestep tables")

    def inverse_map(indices: torch.Tensor) -> torch.Tensor:
        result = torch.full((sequence,), -1, dtype=torch.int32)
        result[indices] = torch.arange(indices.numel(), dtype=torch.int32)
        return result

    def product_inputs(_conditioning_layout: bool = False):
        video = torch.sin(
            torch.arange(video_tokens * source.VIDEO_DIM, dtype=torch.float32)
            / 19.0
        ).reshape(video_tokens, source.VIDEO_DIM)
        audio = torch.cos(
            torch.arange(audio_tokens * source.AUDIO_DIM, dtype=torch.float32)
            / 11.0
        ).reshape(audio_tokens, source.AUDIO_DIM)
        text = 0.25 * torch.sin(
            torch.arange(text_tokens * source.TEXT_DIM, dtype=torch.float32)
            / 29.0
        ).reshape(text_tokens, source.TEXT_DIM)
        tags = layout.token_tags.to(torch.int32)
        timestep_ids = timestep_indices.to(torch.int32)
        return {
            "video": video,
            "audio": audio,
            "text": text.to(torch.bfloat16),
            "timestep": timestep,
            "text_map": inverse_map(layout.text_indices),
            "video_map": inverse_map(layout.video_indices),
            "audio_map": inverse_map(layout.audio_indices),
            "token_tags": tags,
            "timestep_indices": timestep_ids,
            "adaln_indices": timestep_ids * 3 + tags,
            "position_ids": layout.position_ids.to(torch.float32),
            "video_indices": layout.video_indices.to(torch.int32),
            "audio_indices": layout.audio_indices.to(torch.int32),
            "text_indices": layout.text_indices.to(torch.int32),
        }

    source.VIDEO_TOKENS = video_tokens
    source.AUDIO_TOKENS = audio_tokens
    source.TEXT_TOKENS = text_tokens
    source.TIMESTEP_TABLES = 2
    source.SEQUENCE = sequence
    source.fixed_inputs = product_inputs

    export_args = argparse.Namespace(
        output=args.output,
        index=args.index if args.index is not None else source.DEFAULT_INDEX,
        layers=args.layers,
        conditioning_layout=False,
        dump_block_stages=args.dump_block_stages,
    )
    source.export(export_args)

    manifest_path = args.output.resolve() / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["phase3_product_profile"] = {
        "id": args.profile,
        "profiles_path": str(profile_path),
        "profiles_sha256": sha256(profile_path),
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "latent_frames": latent_frames,
        "latent_height": latent_height,
        "latent_width": latent_width,
        "audio_latents_per_channel": audio_frames,
        "video_tokens": video_tokens,
        "audio_tokens": audio_tokens,
        "text_tokens": text_tokens,
        "sequence_tokens": sequence,
    }
    manifest["phase3_capture_instrumentation"] = {
        "wrapper": str(Path(__file__).resolve()),
        "wrapper_sha256": sha256(Path(__file__).resolve()),
        "accepted_source_exporter": str(exporter_path),
        "accepted_source_exporter_sha256": sha256(exporter_path),
        "input_content": (
            "deterministic analytic raw video/audio/text tensors; exact source "
            "product packing and all released frontend/block weights"
        ),
        "oracle_derived_from_diffir": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "PHASE3_PRODUCT_ORACLE PASS "
        f"profile={args.profile} sequence={sequence} video={video_tokens} "
        f"audio={audio_tokens} text={text_tokens} layers={args.layers}",
        flush=True,
    )


if __name__ == "__main__":
    main()
