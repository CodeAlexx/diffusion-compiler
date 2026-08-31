#!/usr/bin/env python3
"""Import real Serenity H3 conditioner/noise evidence into DiffTensor v1.

This is a byte-preserving container conversion. It does not tokenize, encode,
sample noise, cast, quantize, or otherwise synthesize model inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path


DIFF_DTYPES = {"F32": 1, "BF16": 2, "F16": 3, "I8": 4, "I32": 5}
DTYPE_WIDTHS = {"F32": 4, "BF16": 2, "F16": 2, "I8": 1, "I32": 4}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SafeTensor:
    dtype: str
    shape: tuple[int, ...]
    raw: bytes


def read_safetensors(path: Path) -> dict[str, SafeTensor]:
    payload = path.read_bytes()
    if len(payload) < 8:
        raise ValueError(f"truncated SafeTensors file: {path}")
    (header_size,) = struct.unpack_from("<Q", payload)
    data_start = 8 + header_size
    if data_start > len(payload):
        raise ValueError(f"invalid SafeTensors header size: {path}")
    header = json.loads(payload[8:data_start].decode("utf-8"))
    tensors: dict[str, SafeTensor] = {}
    for name, record in header.items():
        if name == "__metadata__":
            continue
        dtype = str(record["dtype"])
        if dtype not in DIFF_DTYPES:
            raise ValueError(f"unsupported tensor dtype {dtype}: {name}")
        shape = tuple(int(dim) for dim in record["shape"])
        begin, end = (int(value) for value in record["data_offsets"])
        raw = payload[data_start + begin : data_start + end]
        elements = 1
        for dim in shape:
            if dim <= 0:
                raise ValueError(f"non-positive tensor dimension: {name}")
            elements *= dim
        expected = elements * DTYPE_WIDTHS[dtype]
        if len(raw) != expected:
            raise ValueError(
                f"SafeTensors byte mismatch for {name}: {len(raw)} != {expected}"
            )
        tensors[name] = SafeTensor(dtype=dtype, shape=shape, raw=raw)
    return tensors


def write_diftensor(path: Path, tensor: SafeTensor, *, squeeze_batch: bool) -> dict:
    shape = tensor.shape
    if squeeze_batch:
        if len(shape) < 2 or shape[0] != 1:
            raise ValueError(f"cannot squeeze non-unit batch shape {shape}")
        shape = shape[1:]
    body = bytearray(b"DIFTNS01")
    body += struct.pack("<III", 1, DIFF_DTYPES[tensor.dtype], len(shape))
    for dim in shape:
        body += struct.pack("<Q", dim)
    body += struct.pack("<Q", len(tensor.raw))
    body += tensor.raw
    body += hashlib.sha256(body).digest()
    path.write_bytes(body)
    return {
        "path": str(path.resolve()),
        "dtype": tensor.dtype,
        "shape": list(shape),
        "payload_bytes": len(tensor.raw),
        "payload_sha256": hashlib.sha256(tensor.raw).hexdigest(),
        "file_sha256": hashlib.sha256(body).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditioning", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-config", type=Path, required=True)
    parser.add_argument("--encoder-index", type=Path, required=True)
    parser.add_argument("--checkpoint-index", type=Path, required=True)
    parser.add_argument("--source-encoder", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    if args.frames < 5 or args.frames % 17 != 5:
        parser.error("H3 frames must be positive and satisfy frames % 17 == 5")
    if args.width % 32 or args.height % 32:
        parser.error("H3 width and height must be divisible by 32")

    conditioning = read_safetensors(args.conditioning)
    initial = read_safetensors(args.initial_state)
    text = conditioning["text_conditioning"]
    video = initial["video_state_rows"]
    audio = initial["audio_state_rows"]
    text_shape = text.shape[1:] if len(text.shape) == 3 and text.shape[0] == 1 else text.shape
    text_tokens = text_shape[0]
    latent_t = ((args.frames - 5) // 17) * 5 + 2
    latent_h = args.height // 16
    latent_w = args.width // 16
    video_tokens = latent_t * (latent_h // 2) * (latent_w // 2)
    audio_latents = (args.frames * 40 + args.fps // 2) // args.fps
    expected = {
        "text": (text_tokens, 5120),
        "video": (video_tokens, 96),
        "audio": (2 * audio_latents, 32),
    }
    actual = {
        "text": text_shape,
        "video": video.shape,
        "audio": audio.shape,
    }
    if actual != expected:
        raise ValueError(f"Serenity input geometry mismatch: {actual} != {expected}")
    if text.dtype != "BF16" or video.dtype != "F32" or audio.dtype != "F32":
        raise ValueError(
            f"Serenity input dtype mismatch: text={text.dtype} video={video.dtype} audio={audio.dtype}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {
        "text_conditioning": write_diftensor(
            args.output_dir / "text_conditioning.diftensor",
            text,
            squeeze_batch=len(text.shape) == 3,
        ),
        "video_state_rows": write_diftensor(
            args.output_dir / "video_state_rows.diftensor", video, squeeze_batch=False
        ),
        "audio_state_rows": write_diftensor(
            args.output_dir / "audio_state_rows.diftensor", audio, squeeze_batch=False
        ),
    }
    document = {
        "schema": "dif.h3.serenity_inputs.v1",
        "task": "t2va",
        "prompt": {
            "path": str(args.prompt.resolve()),
            "sha256": sha256_file(args.prompt),
            "bytes": args.prompt.stat().st_size,
        },
        "tokenizer": {
            "path": str(args.tokenizer.resolve()),
            "sha256": sha256_file(args.tokenizer),
            "config_path": str(args.tokenizer_config.resolve()),
            "config_sha256": sha256_file(args.tokenizer_config),
            "add_special_tokens": False,
            "chat_template": False,
            "text_tokens": text_tokens,
        },
        "conditioning": {
            "source_file": str(args.conditioning.resolve()),
            "source_file_sha256": sha256_file(args.conditioning),
            "source_encoder_path": str(args.source_encoder.resolve()),
            "encoder_index": str(args.encoder_index.resolve()),
            "encoder_index_sha256": sha256_file(args.encoder_index),
            "hidden_state_layer": 50,
            "storage": "bf16",
        },
        "checkpoint": {
            "index": str(args.checkpoint_index.resolve()),
            "index_sha256": sha256_file(args.checkpoint_index),
        },
        "start": {
            "source_file": str(args.initial_state.resolve()),
            "source_file_sha256": sha256_file(args.initial_state),
            "video_seed": args.seed,
            "audio_seed": args.seed + 1,
        },
        "geometry": {
            "width": args.width,
            "height": args.height,
            "frames": args.frames,
            "fps": args.fps,
            "latent_t": latent_t,
            "latent_h": latent_h,
            "latent_w": latent_w,
            "audio_latents": audio_latents,
            "video_tokens": video_tokens,
            "audio_tokens": 2 * audio_latents,
            "text_tokens": text_tokens,
            "sequence_tokens": video_tokens + 2 * audio_latents + text_tokens,
        },
        "schedule": {
            "points": args.steps,
            "model_evaluations": args.steps - 1,
            "video_shift": 12.0,
            "audio_shift": 3.0,
            "sampler": "released_h3_data_ward_euler",
        },
        "negative_prompt_policy": "none_single_conditional_forward",
        "outputs": outputs,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(
        "H3_SERENITY_INPUT_IMPORT PASS "
        f"text={text_tokens} video={video_tokens} audio={2 * audio_latents} "
        f"sequence={document['geometry']['sequence_tokens']}"
    )


if __name__ == "__main__":
    main()
