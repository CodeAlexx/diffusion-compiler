#!/usr/bin/env python3
"""Deterministic PyTorch mixed-precision (BF16/F32) oracle for DiffIR training.

Mirrors /home/alex/diffusion-fixtures/oracles/export_mlp_training.py but with
the flame-style mixed-precision semantics the BF16 DiffIR training graph
implements:

- parameters and activations stored in BF16 (every op rounds to BF16 at its
  store, computing in F32 internally);
- the prediction crosses an explicit cast into an F32 MSE loss;
- gradients carry the dtype of their forward tensors (BF16);
- AdamW keeps both moments in F32 ALWAYS and updates with the exact receipt
  of the C++ AdamWUpdate kernel: F32 math, decoupled weight decay applied to
  the parameter BEFORE subtracting the moment-based update, decay never
  folded into the gradient ahead of the moment updates, and a single
  round-to-nearest-even BF16 store of the updated parameter.

torch.optim.AdamW is NOT used because it would keep BF16 moments for BF16
parameters; the manual loop below is the reference for F32 moments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import time
from pathlib import Path

import torch
import torch.nn.functional as functional

ROWS = 32
INPUT_WIDTH = 8
HIDDEN_WIDTH = 16
OUTPUT_WIDTH = 4
PARAMETER_IDS = (3, 4, 5, 6)

DTYPE_CODES = {torch.float32: 1, torch.bfloat16: 2}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_diftensor(path: Path, tensor: torch.Tensor) -> dict[str, object]:
    value = tensor.detach().cpu().contiguous()
    if value.dtype not in DTYPE_CODES:
        raise TypeError(f"oracle export requires float32/bfloat16, got {value.dtype}")
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
        "path": path.name,
        "shape": list(shape),
        "dtype": dtype_name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--epsilon", type=float, default=1.0e-8)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.steps < 1:
        raise RuntimeError("--steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the recorded source gate")

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")

    features_f32 = torch.linspace(
        -1.0, 1.0, ROWS * INPUT_WIDTH, dtype=torch.float32, device=device
    ).reshape(ROWS, INPUT_WIDTH)
    features = features_f32.to(torch.bfloat16)
    # Targets are computed from the BF16-quantized feature values so both
    # sides of the parity comparison see the identical F32 target tensor.
    dequantized = features.to(torch.float32)
    target = torch.stack(
        (
            0.70 * dequantized[:, 0] - 0.20 * dequantized[:, 1] + 0.10,
            torch.sin(1.30 * dequantized[:, 2]) + 0.15 * dequantized[:, 3],
            dequantized[:, 4] * dequantized[:, 5] - 0.25,
            -0.40 * dequantized[:, 6] + 0.60 * dequantized[:, 7] + 0.05,
        ),
        dim=1,
    ).contiguous()

    initial_f32 = [
        torch.linspace(
            -0.20, 0.20, HIDDEN_WIDTH * INPUT_WIDTH, dtype=torch.float32,
            device=device,
        ).reshape(HIDDEN_WIDTH, INPUT_WIDTH),
        torch.linspace(-0.05, 0.05, HIDDEN_WIDTH, dtype=torch.float32,
                       device=device),
        torch.linspace(
            -0.15, 0.15, OUTPUT_WIDTH * HIDDEN_WIDTH, dtype=torch.float32,
            device=device,
        ).reshape(OUTPUT_WIDTH, HIDDEN_WIDTH),
        torch.linspace(0.03, -0.03, OUTPUT_WIDTH, dtype=torch.float32,
                       device=device),
    ]
    parameters = [
        torch.nn.Parameter(value.to(torch.bfloat16)) for value in initial_f32
    ]
    initial = [parameter.detach().clone() for parameter in parameters]
    first_moments = [
        torch.zeros_like(parameter, dtype=torch.float32)
        for parameter in parameters
    ]
    second_moments = [
        torch.zeros_like(parameter, dtype=torch.float32)
        for parameter in parameters
    ]

    records: dict[str, object] = {
        "features": write_diftensor(output / "features.diftensor", features),
        "targets": write_diftensor(output / "targets.diftensor", target),
    }
    for name, value in zip(("w1", "b1", "w2", "b2"), initial, strict=True):
        records[f"initial_{name}"] = write_diftensor(
            output / f"initial-{name}.diftensor", value
        )

    learning_rate = args.learning_rate
    beta1 = args.beta1
    beta2 = args.beta2
    epsilon = args.epsilon
    weight_decay = args.weight_decay

    losses: list[torch.Tensor] = []
    first_gradients: list[torch.Tensor] = []
    step1_parameters: list[torch.Tensor] = []
    final_gradients: list[torch.Tensor] = []
    final_prediction: torch.Tensor | None = None
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(args.steps):
        for parameter in parameters:
            parameter.grad = None
        hidden = functional.linear(features, parameters[0])
        hidden = hidden + parameters[1]
        hidden = functional.silu(hidden)
        prediction = functional.linear(hidden, parameters[2])
        prediction = prediction + parameters[3]
        loss = torch.mean((prediction.to(torch.float32) - target) ** 2)
        loss.backward()
        losses.append(loss.detach().clone())
        if step == 0:
            first_gradients = [
                parameter.grad.detach().clone() for parameter in parameters
            ]
        if step + 1 == args.steps:
            final_prediction = prediction.detach().clone()
            final_gradients = [
                parameter.grad.detach().clone() for parameter in parameters
            ]
        with torch.no_grad():
            step_number = step + 1
            bias1 = 1.0 - beta1**step_number
            bias2_sqrt = math.sqrt(1.0 - beta2**step_number)
            for index, parameter in enumerate(parameters):
                grad = parameter.grad.to(torch.float32)
                m = beta1 * first_moments[index] + (1.0 - beta1) * grad
                v = beta2 * second_moments[index] + (1.0 - beta2) * grad * grad
                first_moments[index].copy_(m)
                second_moments[index].copy_(v)
                decayed = parameter.to(torch.float32) * (
                    1.0 - learning_rate * weight_decay
                )
                denominator = torch.sqrt(v) / bias2_sqrt + epsilon
                updated = decayed - (learning_rate / bias1) * m / denominator
                parameter.copy_(updated.to(torch.bfloat16))
        if step == 0:
            step1_parameters = [
                parameter.detach().clone() for parameter in parameters
            ]
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if final_prediction is None:
        raise AssertionError("training loop did not execute")

    loss_tensor = torch.stack(losses)
    records["losses"] = write_diftensor(output / "losses.diftensor", loss_tensor)
    records["final_prediction"] = write_diftensor(
        output / "final-prediction.diftensor", final_prediction
    )
    for index, parameter_id in enumerate(PARAMETER_IDS):
        records[f"gradient_step_1_{parameter_id}"] = write_diftensor(
            output / f"gradient-step-1-{parameter_id}.diftensor",
            first_gradients[index],
        )
        records[f"parameter_step_1_{parameter_id}"] = write_diftensor(
            output / f"parameter-step-1-{parameter_id}.diftensor",
            step1_parameters[index],
        )
        records[f"gradient_final_{parameter_id}"] = write_diftensor(
            output / f"gradient-final-{parameter_id}.diftensor",
            final_gradients[index],
        )
        records[f"state_{parameter_id}"] = write_diftensor(
            output / f"state-{parameter_id}.diftensor", parameters[index]
        )
        records[f"state_first_{parameter_id}"] = write_diftensor(
            output / f"state-first-{parameter_id}.diftensor",
            first_moments[index],
        )
        records[f"state_second_{parameter_id}"] = write_diftensor(
            output / f"state-second-{parameter_id}.diftensor",
            second_moments[index],
        )

    manifest = {
        "oracle": (
            "PyTorch public functional linear/SiLU and autograd in BF16 with "
            "a manual F32-moment AdamW matching the DiffIR AdamWUpdate receipt"
        ),
        "oracle_script": str(Path(__file__).resolve()),
        "oracle_script_sha256": sha256(Path(__file__).resolve()),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "geometry": {
            "rows": ROWS,
            "input_width": INPUT_WIDTH,
            "hidden_width": HIDDEN_WIDTH,
            "output_width": OUTPUT_WIDTH,
        },
        "precision": {
            "parameters": "bf16",
            "activations": "bf16",
            "loss": "f32",
            "gradients": "bf16",
            "adamw_moments": "f32",
        },
        "optimizer": {
            "name": "manual decoupled AdamW (F32 moments, BF16 parameters)",
            "learning_rate": learning_rate,
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
            "weight_decay": weight_decay,
        },
        "steps": args.steps,
        "initial_loss": float(losses[0].item()),
        "final_loss": float(losses[-1].item()),
        "loop_wall_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "tensors": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"output": str(output), "steps": args.steps,
                      "initial_loss": manifest["initial_loss"],
                      "final_loss": manifest["final_loss"]}, indent=2))


if __name__ == "__main__":
    main()
