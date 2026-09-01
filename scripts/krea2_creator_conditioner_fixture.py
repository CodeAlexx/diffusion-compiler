#!/usr/bin/env python3
"""Official Krea 2 Qwen3-VL-4B conditioner oracle (development only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen2TokenizerFast, Qwen3VLForConditionalGeneration


PREFIX = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, "
    "size, texture, quantity, text, spatial relationships of the objects and "
    "background:<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
SELECTED = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
PREFIX_TOKENS = 34
SUFFIX_START = 5
OUTPUT_TOKENS = 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    prompt = (
        args.prompt
        if args.prompt is not None
        else args.prompt_file.read_text(encoding="utf-8").strip()
    )
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite an existing conditioner artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, max_length=OUTPUT_TOKENS, local_files_only=True
    )
    suffix_tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.model, max_length=OUTPUT_TOKENS, local_files_only=True
    )
    prefix_inputs = tokenizer(
        [PREFIX + prompt],
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        padding="max_length",
        max_length=OUTPUT_TOKENS + PREFIX_TOKENS - SUFFIX_START,
        return_tensors="pt",
    )
    suffix_inputs = suffix_tokenizer(text=[SUFFIX], return_tensors="pt")
    input_ids = torch.cat(
        [prefix_inputs["input_ids"], suffix_inputs["input_ids"]], dim=1
    )
    mask = torch.cat(
        [
            prefix_inputs["attention_mask"].bool(),
            suffix_inputs["attention_mask"].bool(),
        ],
        dim=1,
    )
    if input_ids.shape != (1, 546) or mask[:, PREFIX_TOKENS:].shape != (1, 512):
        raise RuntimeError(f"unexpected official token geometry: {input_ids.shape}")

    # Text-only Qwen3-VL position IDs are validity cumsum; padding positions
    # are 1 and the five-token assistant suffix continues after the prompt.
    position_ids = mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(~mask, 1)

    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval().requires_grad_(False).to("cuda")
    load_seconds = time.perf_counter() - load_start
    ids_cuda = input_ids.to("cuda")
    mask_cuda = mask.to("cuda")
    torch.cuda.synchronize()
    run_start = time.perf_counter()
    with torch.no_grad():
        result = model(
            input_ids=ids_cuda,
            attention_mask=mask_cuda,
            output_hidden_states=True,
            use_cache=False,
        )
    torch.cuda.synchronize()
    run_seconds = time.perf_counter() - run_start

    tensors: dict[str, torch.Tensor] = {
        "input_ids": input_ids.to(torch.int32).contiguous(),
        "attention_mask": mask.contiguous(),
        "position_ids": position_ids.to(torch.float32).view(546, 1).contiguous(),
    }
    for output_index, hidden_index in enumerate(SELECTED):
        tensors[f"tap_{output_index:02d}_hidden_{hidden_index:02d}"] = (
            result.hidden_states[hidden_index][:, PREFIX_TOKENS:]
            .squeeze(0)
            .to(torch.bfloat16)
            .cpu()
            .contiguous()
        )
    save_file(tensors, args.output)
    creator_commit = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parent.parent), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    report = {
        "creator_source_commit": "db3984fbc6e13b34c0064990fc2d95ac64d00058",
        "fixture_generator_commit": creator_commit,
        "model": str(args.model.resolve()),
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "selected_hidden_states": list(SELECTED),
        "input_shape": list(input_ids.shape),
        "output_shape": [1, 512, 12, 2560],
        "valid_input_tokens": int(mask.sum()),
        "valid_output_tokens": int(mask[:, PREFIX_TOKENS:].sum()),
        "load_seconds": load_seconds,
        "run_seconds": run_seconds,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "fixture": str(args.output.resolve()),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    report["fixture_sha256"] = sha256(args.output)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
