#!/usr/bin/env python3
"""Deterministic PyTorch fixtures for the DiffIR DiT backward opcodes.

For every case this writes, per dtype (f32, bf16):
  - the operation inputs,
  - a fixed upstream gradient grad_output,
  - the expected input gradients from torch.autograd.grad on a forward that
    mirrors the DiffIR semantics exactly (rms_norm formula, modulate
    ordering, SwiGLU GateFirst, LayerNorm affine, qk-norm partial halfsplit
    rotation with both table conventions, softmax attention with the
    1/sqrt(D) default scale and hard-skip causal masking).

Dtype contract (mirrors the C++ kernels: BF16 storage, F32 opmath, one
round at the store): the BF16 fixtures quantize the inputs to BF16, run the
reference forward/backward in F32 on those values, and round the expected
gradients to BF16.  Torch's own BF16 autograd rounds at every primitive
instead and is NOT the reference here.

Layout: OUT_DIR/<case>/<dtype>/{<input>.diftensor, grad-output.diftensor,
expected-grad-<input>.diftensor}, plus OUT_DIR/manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import torch

DTYPE_CODES = {torch.float32: 1, torch.bfloat16: 2}


def write_diftensor(path: Path, tensor: torch.Tensor) -> dict[str, object]:
    value = tensor.detach().cpu().contiguous()
    if value.dtype not in DTYPE_CODES:
        raise TypeError(f"fixture export requires float32/bfloat16, got {value.dtype}")
    if value.dtype == torch.bfloat16:
        raw = value.view(torch.uint16).numpy().tobytes(order="C")
        dtype_name = "bf16"
    else:
        raw = value.numpy().tobytes(order="C")
        dtype_name = "f32"
    shape = tuple(int(dimension) for dimension in value.shape)
    payload = bytearray(b"DIFTNS01")
    payload += struct.pack("<III", 1, DTYPE_CODES[value.dtype], len(shape))
    for dimension in shape:
        payload += struct.pack("<Q", dimension)
    payload += struct.pack("<Q", len(raw))
    payload += raw
    payload += hashlib.sha256(payload).digest()
    path.write_bytes(payload)
    return {
        "path": str(path),
        "shape": list(shape),
        "dtype": dtype_name,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * inv * weight


def qk_norm_rope_ref(x, weight, cos, sin, rotary: int, eps: float):
    # x [S,H,D]; cos/sin [S,T] with T in {rotary, rotary//2}; mirrors the CPU
    # reference executor's qk_norm_rope loop.
    S, H, D = x.shape
    T = cos.shape[1]
    half = rotary // 2
    inv = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    n = x * inv * weight  # [S,H,D]
    out = n.clone()
    c1 = cos[:, :half].unsqueeze(1)  # [S,1,half]
    s1 = sin[:, :half].unsqueeze(1)
    if T == rotary:
        c2 = cos[:, half:rotary].unsqueeze(1)
        s2 = sin[:, half:rotary].unsqueeze(1)
    else:
        c2 = c1
        s2 = s1
    a = n[:, :, :half]
    b = n[:, :, half:rotary]
    out = torch.cat(
        [a * c1 - b * s1, b * c2 + a * s2, n[:, :, rotary:]], dim=-1)
    return out


def attention_ref(q, k, v, causal: bool):
    S, H, D = q.shape
    scale = 1.0 / math.sqrt(D)
    scores = torch.einsum("qhd,khd->hqk", q, k) * scale
    if causal:
        mask = torch.ones(S, S, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probabilities, v)


def build_cases(generator: torch.Generator):
    eps = 1.0e-5

    def rand(*shape, amplitude=1.0, offset=0.0):
        return (torch.rand(*shape, generator=generator, dtype=torch.float32)
                * 2.0 - 1.0) * amplitude + offset

    cases = {}

    R, C = 8, 16
    cases["rms_norm"] = {
        "inputs": {"x": rand(R, C), "weight": rand(C, amplitude=0.5, offset=1.0)},
        "grad_shape": (R, C),
        "forward": lambda t: rms_norm_ref(t["x"], t["weight"], eps),
        "wrt": ["x", "weight"],
    }
    cases["rms_norm_modulate_weighted"] = {
        "inputs": {
            "x": rand(R, C),
            "weight": rand(C, amplitude=0.5, offset=1.0),
            "scale": rand(R, C, amplitude=0.3),
            "shift": rand(R, C, amplitude=0.3),
        },
        "grad_shape": (R, C),
        "forward": lambda t: rms_norm_ref(t["x"], t["weight"], eps)
        * (1.0 + t["scale"]) + t["shift"],
        "wrt": ["x", "weight", "scale", "shift"],
    }
    cases["rms_norm_modulate_plain"] = {
        "inputs": {
            "x": rand(R, C),
            "scale": rand(R, C, amplitude=0.3),
            "shift": rand(R, C, amplitude=0.3),
        },
        "grad_shape": (R, C),
        "forward": lambda t: t["x"]
        * torch.rsqrt(t["x"].pow(2).mean(dim=-1, keepdim=True) + eps)
        * (1.0 + t["scale"]) + t["shift"],
        "wrt": ["x", "scale", "shift"],
    }
    W = 8
    for gate_first in (True, False):
        name = "swiglu_gatefirst" if gate_first else "swiglu_valuefirst"
        def swiglu_forward(t, gate_first=gate_first):
            x = t["x"]
            width = x.shape[-1] // 2
            if gate_first:
                gate, value = x[:, :width], x[:, width:]
            else:
                value, gate = x[:, :width], x[:, width:]
            return value * torch.nn.functional.silu(gate)
        cases[name] = {
            "inputs": {"x": rand(R, 2 * W)},
            "grad_shape": (R, W),
            "forward": swiglu_forward,
            "wrt": ["x"],
        }
    cases["residual_gate"] = {
        "inputs": {
            "residual": rand(R, C),
            "branch": rand(R, C),
            "gate": rand(R, C, amplitude=0.5),
        },
        "grad_shape": (R, C),
        "forward": lambda t: t["residual"] + t["gate"] * t["branch"],
        "wrt": ["residual", "branch", "gate"],
    }
    cases["layer_norm"] = {
        "inputs": {
            "x": rand(R, C),
            "weight": rand(C, amplitude=0.5, offset=1.0),
            "bias": rand(C, amplitude=0.3),
        },
        "grad_shape": (R, C),
        "forward": lambda t: torch.nn.functional.layer_norm(
            t["x"], (t["x"].shape[-1],), t["weight"], t["bias"], eps),
        "wrt": ["x", "weight", "bias"],
    }
    S, H, D, ROT = 6, 2, 8, 6
    for full_table in (True, False):
        name = ("qk_norm_rope_fulltable" if full_table
                else "qk_norm_rope_halftable")
        T = ROT if full_table else ROT // 2
        cases[name] = {
            "inputs": {
                "x": rand(S, H, D),
                "weight": rand(D, amplitude=0.5, offset=1.0),
                "cos": rand(S, T),
                "sin": rand(S, T),
            },
            "grad_shape": (S, H, D),
            "forward": lambda t, rot=ROT: qk_norm_rope_ref(
                t["x"], t["weight"], t["cos"], t["sin"], rot, eps),
            "wrt": ["x", "weight"],
            "attrs": {"rotary_dim": ROT},
        }
    AS, AH, AD = 7, 2, 6
    for causal in (False, True):
        name = "attention_causal" if causal else "attention_full"
        cases[name] = {
            "inputs": {
                "q": rand(AS, AH, AD),
                "k": rand(AS, AH, AD),
                "v": rand(AS, AH, AD),
            },
            "grad_shape": (AS, AH, AD),
            "forward": lambda t, causal=causal: attention_ref(
                t["q"], t["k"], t["v"], causal),
            "wrt": ["q", "k", "v"],
            "attrs": {"causal": causal},
        }
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20260831)
    arguments = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    generator = torch.Generator().manual_seed(arguments.seed)
    cases = build_cases(generator)
    grad_generator = torch.Generator().manual_seed(arguments.seed + 1)

    manifest = {"seed": arguments.seed, "torch": torch.__version__, "cases": {}}
    for name, case in cases.items():
        grad_output = (torch.rand(*case["grad_shape"],
                                  generator=grad_generator,
                                  dtype=torch.float32) * 2.0 - 1.0)
        for dtype_name in ("f32", "bf16"):
            directory = arguments.output / name / dtype_name
            directory.mkdir(parents=True, exist_ok=True)
            records = {}
            if dtype_name == "f32":
                inputs = {key: value.clone() for key, value in
                          case["inputs"].items()}
                upstream = grad_output.clone()
                store = torch.float32
            else:
                # BF16 contract: BF16-valued inputs, F32 reference math,
                # expected gradients rounded once to BF16.
                inputs = {key: value.to(torch.bfloat16).to(torch.float32)
                          for key, value in case["inputs"].items()}
                upstream = grad_output.to(torch.bfloat16).to(torch.float32)
                store = torch.bfloat16
            leaves = {key: value.clone().requires_grad_(key in case["wrt"])
                      for key, value in inputs.items()}
            prediction = case["forward"](leaves)
            gradients = torch.autograd.grad(
                outputs=prediction, inputs=[leaves[key] for key in case["wrt"]],
                grad_outputs=upstream)
            for key, value in inputs.items():
                records[key] = write_diftensor(
                    directory / f"{key}.diftensor", value.to(store))
            records["grad-output"] = write_diftensor(
                directory / "grad-output.diftensor", upstream.to(store))
            for key, gradient in zip(case["wrt"], gradients):
                records[f"expected-grad-{key}"] = write_diftensor(
                    directory / f"expected-grad-{key}.diftensor",
                    gradient.to(store))
            manifest["cases"].setdefault(name, {})[dtype_name] = {
                "wrt": case["wrt"],
                "attrs": case.get("attrs", {}),
                "tensors": records,
            }
    (arguments.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"wrote {len(cases)} cases x2 dtypes to {arguments.output}")


if __name__ == "__main__":
    main()
