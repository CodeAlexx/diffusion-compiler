"""Shared helpers for the SDXL creator-oracle exporters.

The creator is the reference sampler checkout the speed baseline was measured
with (ComfyUI). Its own modules build the models, so every oracle carries the
creator's numerics, not a re-implementation.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

UNET_PREFIX = "model.diffusion_model."
VAE_PREFIX = "first_stage_model."
CLIP_L_PREFIX = "conditioner.embedders.0.transformer."
CLIP_G_PREFIX = "conditioner.embedders.1.model."


def add_reference_source(path: Path) -> str:
    path = path.resolve()
    if not (path / "comfy").is_dir():
        raise SystemExit(f"{path} is not the reference sampler checkout")
    sys.path.insert(0, str(path))
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def refuse_overwrite(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")


def cpu(value: torch.Tensor) -> torch.Tensor:
    return value.detach().contiguous().cpu()


def load_prefixed(checkpoint: Path, prefix: str) -> dict[str, torch.Tensor]:
    state = load_file(str(checkpoint))
    return {
        name[len(prefix):]: value for name, value in state.items() if name.startswith(prefix)
    }


class Captures:
    """Forward hooks that record module outputs (or inputs) by name."""

    def __init__(self) -> None:
        self.values: dict[str, torch.Tensor] = {}
        self.handles = []

    def output(self, module: torch.nn.Module, name: str) -> None:
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            self.values[name] = cpu(value)

        self.handles.append(module.register_forward_hook(hook))

    def input(self, module: torch.nn.Module, name: str, index: int = 0) -> None:
        def hook(_module, inputs):
            self.values[name] = cpu(inputs[index])

        self.handles.append(module.register_forward_pre_hook(hook))

    def release(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def save(path: Path, captures: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in captures.items()}, str(path), metadata=metadata)
    print(f"wrote {path} ({len(captures)} tensors)")
    for name, value in captures.items():
        print(f"  {name}: {tuple(value.shape)} {value.dtype}")
