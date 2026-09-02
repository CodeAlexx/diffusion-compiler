#!/usr/bin/env python3
"""Development-only creator oracle for the native Qwen3-VL vision gate.

This is never imported or launched by the production runtime. It loads only
the official text encoder's ``model.visual.*`` tensors and exports the creator
BF16 boundaries used by ``difh3vision`` parity tests.
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoProcessor
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel


def load_visual_state(text_encoder: Path) -> dict[str, torch.Tensor]:
    index = json.loads((text_encoder / "model.safetensors.index.json").read_text())
    names = index["weight_map"]
    shards = sorted({value for key, value in names.items() if key.startswith("model.visual.")})
    state: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(text_encoder / shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith("model.visual."):
                    state[key.removeprefix("model.visual.")] = handle.get_tensor(key)
    return state


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--processor", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    if not torch.cuda.is_available():
        raise SystemExit("creator vision oracle requires CUDA")

    full_config = json.loads((args.text_encoder / "config.json").read_text())
    config = Qwen3VLVisionConfig(**full_config["vision_config"])
    config._attn_implementation = "eager"
    state = load_visual_state(args.text_encoder)
    model = Qwen3VLVisionModel(config).to(torch.bfloat16)
    missing, unexpected = model.load_state_dict(
        state, strict=False
    )
    if missing or unexpected:
        raise SystemExit(f"vision state mismatch: missing={missing} unexpected={unexpected}")
    model = model.cuda().eval()
    captured: dict[str, torch.Tensor] = {}

    def capture_block(name: str):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[name] = value.detach().clone()

        return hook

    handles = [
        model.blocks[index].register_forward_hook(capture_block(f"block_{index:02d}"))
        for index in (0, 8, 16, 24, config.depth - 1)
    ]

    def capture_after_patch(_module, inputs):
        captured["after_patch"] = inputs[0].detach().clone()

    handles.append(model.blocks[0].register_forward_pre_hook(capture_after_patch))
    processor = AutoProcessor.from_pretrained(args.processor)
    image = Image.open(args.image).convert("RGB")
    prepared = processor.image_processor(images=[image], return_tensors="pt")
    pixels = prepared["pixel_values"].to("cuda", torch.bfloat16)
    grid = prepared["image_grid_thw"].to("cuda")
    embeds, deepstack = model(pixels, grid_thw=grid)
    for handle in handles:
        handle.remove()
    embeds_cpu = embeds.cpu().contiguous()
    deepstack_cpu = [value.cpu().contiguous() for value in deepstack]
    captured_cpu = {
        name: value.cpu().contiguous() for name, value in captured.items()
    }
    del model, embeds, deepstack, captured
    gc.collect()
    torch.cuda.empty_cache()

    model_f32 = Qwen3VLVisionModel(config).to(torch.float32)
    missing, unexpected = model_f32.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"f32 vision state mismatch: missing={missing} unexpected={unexpected}"
        )
    model_f32 = model_f32.cuda().eval()
    embeds_f32, deepstack_f32 = model_f32(
        prepared["pixel_values"].to("cuda", torch.float32), grid_thw=grid
    )

    def deficit(bf16: torch.Tensor, f32: torch.Tensor) -> float:
        return 1.0 - torch.nn.functional.cosine_similarity(
            bf16.double().flatten().cuda(), f32.double().flatten(), dim=0
        ).item()

    tensors = {
        "pixel_patches": pixels.cpu().contiguous(),
        "grid_thw": grid.to(torch.int32).cpu().contiguous(),
        "vision_embeds": embeds_cpu,
        "noise.vision_embeds": torch.tensor(
            [deficit(embeds_cpu, embeds_f32)], dtype=torch.float32
        ),
    }
    for index, value in enumerate(deepstack_cpu):
        tensors[f"deepstack_{index}"] = value
        tensors[f"noise.deepstack_{index}"] = torch.tensor(
            [deficit(value, deepstack_f32[index])], dtype=torch.float32
        )
    tensors.update(captured_cpu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.output)
    print(
        f"H3_VISION_ORACLE output={args.output} grid={grid[0].tolist()} "
        f"patches={tuple(pixels.shape)} embeds={tuple(embeds_cpu.shape)} "
        f"dtype={embeds_cpu.dtype}"
    )


if __name__ == "__main__":
    main()
