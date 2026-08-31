#!/usr/bin/env python3
"""PyTorch oracle for the compiler's F32 activation-path LoRA trainer.

Mirrors /home/alex/diffusion-fixtures/oracles/export_rectified_flow_training.py
for the LoRA vertical: one rectified-flow microbatch whose three Linears are
LoRA-augmented (frozen base, trainable A/B, delta scaled by alpha/rank),
trained with adapter-only AdamW.  Writes the diftrain fixture (data, frozen
constants, initial adapters) plus expected losses, predictions, gradients,
and optimizer state for the parity gate.

Canonical tensor ids of `diftrain make-lora OUT 16 8 4 16 4 8.0`:
  data 1..6, frozen constants 7..11 (constant-<id>.diftensor),
  adapters 12..17 (initial-adapter-<id>.diftensor):
  12=A latent [R,L], 13=B latent [H,R], 14=A time [R,T], 15=B time [H,R],
  16=A out [R,H], 17=B out [L,R].

Flame LoRA contract: A [rank,in] within the Kaiming-uniform bound
1/sqrt(in) (deterministic values here so both sides seed from this fixture),
B [out,rank] zeros, delta = (x @ A^T @ B^T) * (alpha/rank), the dense delta
never materialized.  Step-1 ordering law asserted below: dL/dA == 0 exactly
while B == 0, dL/dB nonzero.
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

ROWS = 16
LATENT_WIDTH = 8
TIMESTEP_WIDTH = 4
HIDDEN_WIDTH = 16
RANK = 4
ALPHA = 8.0
CONSTANT_IDS = (7, 8, 9, 10, 11)
ADAPTER_IDS = (12, 13, 14, 15, 16, 17)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_diftensor(path: Path, tensor: torch.Tensor) -> dict[str, object]:
    value = tensor.detach().cpu().float().contiguous()
    raw = value.numpy().tobytes(order="C")
    payload = bytearray(b"DIFTNS01")
    payload += struct.pack("<III", 1, 1, value.ndim)
    for dim in value.shape:
        payload += struct.pack("<Q", dim)
    payload += struct.pack("<Q", len(raw))
    payload += raw
    payload += hashlib.sha256(payload).digest()
    path.write_bytes(payload)
    return {
        "path": path.name,
        "shape": list(value.shape),
        "dtype": "f32",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def data_batch(device: torch.device) -> dict[str, torch.Tensor]:
    flat = torch.arange(
        ROWS * LATENT_WIDTH, dtype=torch.float32, device=device
    ).reshape(ROWS, LATENT_WIDTH)
    clean = torch.sin((flat + 5.0) / 11.0)
    noise = torch.cos((flat * 1.7 + 3.0) / 17.0)
    timestep = torch.linspace(
        0.08, 0.92, ROWS, dtype=torch.float32, device=device
    ).reshape(ROWS, 1)
    clean_scale = timestep.expand(ROWS, LATENT_WIDTH).contiguous()
    noise_scale = (1.0 - timestep).expand(ROWS, LATENT_WIDTH).contiguous()
    time_features = torch.cat(
        (
            timestep,
            timestep * timestep,
            torch.sin(math.pi * timestep),
            torch.cos(math.pi * timestep),
        ),
        dim=1,
    ).contiguous()
    return {
        "clean": clean.contiguous(),
        "noise": noise.contiguous(),
        "clean-scale": clean_scale,
        "noise-scale": noise_scale,
        "time-features": time_features,
        "target-velocity": (clean - noise).contiguous(),
    }


def linspace_matrix(
    first: float, last: float, rows: int, cols: int, device: torch.device
) -> torch.Tensor:
    return torch.linspace(
        first, last, rows * cols, dtype=torch.float32, device=device
    ).reshape(rows, cols)


def kaiming_bounded(
    in_features: int, rows: int, phase: float, device: torch.device
) -> torch.Tensor:
    """Deterministic values strictly inside the Kaiming-uniform bound
    1/sqrt(in_features) (torch nn.Linear default init bound)."""
    bound = 1.0 / math.sqrt(in_features)
    flat = torch.arange(
        rows * in_features, dtype=torch.float32, device=device
    )
    return (bound * 0.97 * torch.sin(flat * 1.7 + phase)).reshape(
        rows, in_features
    )


def lora_linear(
    x: torch.Tensor,
    base: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
) -> torch.Tensor:
    scale = ALPHA / RANK
    return functional.linear(x, base) + functional.linear(
        functional.linear(x, lora_a), lora_b
    ) * scale


def prediction(
    batch: dict[str, torch.Tensor],
    constants: list[torch.Tensor],
    adapters: list[torch.Tensor],
) -> torch.Tensor:
    noised = (
        batch["clean"] * batch["clean-scale"]
        + batch["noise"] * batch["noise-scale"]
    )
    hidden = lora_linear(noised, constants[0], adapters[0], adapters[1])
    hidden = hidden + constants[1]
    hidden = hidden + lora_linear(
        batch["time-features"], constants[2], adapters[2], adapters[3]
    )
    hidden = functional.silu(hidden)
    return (
        lora_linear(hidden, constants[3], adapters[4], adapters[5])
        + constants[4]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5.0e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--epsilon", type=float, default=1.0e-8)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(
            f"refusing to overwrite non-empty directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    if args.steps <= 0:
        raise RuntimeError("--steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the training source oracle")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False

    batch = data_batch(device)
    constants = [
        linspace_matrix(-0.18, 0.18, HIDDEN_WIDTH, LATENT_WIDTH, device),
        torch.linspace(
            -0.04, 0.04, HIDDEN_WIDTH, dtype=torch.float32, device=device
        ),
        linspace_matrix(0.12, -0.12, HIDDEN_WIDTH, TIMESTEP_WIDTH, device),
        linspace_matrix(-0.14, 0.14, LATENT_WIDTH, HIDDEN_WIDTH, device),
        torch.linspace(
            0.025, -0.025, LATENT_WIDTH, dtype=torch.float32, device=device
        ),
    ]
    adapters = [
        torch.nn.Parameter(
            kaiming_bounded(LATENT_WIDTH, RANK, 0.3, device)
        ),
        torch.nn.Parameter(
            torch.zeros(
                HIDDEN_WIDTH, RANK, dtype=torch.float32, device=device
            )
        ),
        torch.nn.Parameter(
            kaiming_bounded(TIMESTEP_WIDTH, RANK, 1.1, device)
        ),
        torch.nn.Parameter(
            torch.zeros(
                HIDDEN_WIDTH, RANK, dtype=torch.float32, device=device
            )
        ),
        torch.nn.Parameter(
            kaiming_bounded(HIDDEN_WIDTH, RANK, 2.4, device)
        ),
        torch.nn.Parameter(
            torch.zeros(
                LATENT_WIDTH, RANK, dtype=torch.float32, device=device
            )
        ),
    ]
    initial_adapters = [value.detach().clone() for value in adapters]
    optimizer = torch.optim.AdamW(
        adapters,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.epsilon,
        weight_decay=args.weight_decay,
        foreach=False,
        fused=False,
        capturable=False,
    )

    records: dict[str, object] = {}
    for name, value in batch.items():
        records[name.replace("-", "_")] = write_diftensor(
            output / f"{name}.diftensor", value
        )
    for constant_id, value in zip(CONSTANT_IDS, constants, strict=True):
        records[f"constant_{constant_id}"] = write_diftensor(
            output / f"constant-{constant_id}.diftensor", value
        )
    for adapter_id, value in zip(ADAPTER_IDS, initial_adapters, strict=True):
        records[f"initial_adapter_{adapter_id}"] = write_diftensor(
            output / f"initial-adapter-{adapter_id}.diftensor", value
        )

    losses: list[torch.Tensor] = []
    first_gradients: list[torch.Tensor] = []
    final_gradients: list[torch.Tensor] = []
    final_prediction: torch.Tensor | None = None
    step_one_a_grad_zero = False
    step_one_b_grad_nonzero = False
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        value = prediction(batch, constants, adapters)
        loss = torch.mean((value - batch["target-velocity"]) ** 2)
        loss.backward()
        losses.append(loss.detach().clone())
        if step == 0:
            first_gradients = [
                parameter.grad.detach().clone() for parameter in adapters
            ]
            step_one_a_grad_zero = all(
                torch.all(first_gradients[index] == 0.0).item()
                for index in (0, 2, 4)
            )
            step_one_b_grad_nonzero = all(
                torch.any(first_gradients[index] != 0.0).item()
                for index in (1, 3, 5)
            )
        if step + 1 == args.steps:
            final_prediction = value.detach().clone()
            final_gradients = [
                parameter.grad.detach().clone() for parameter in adapters
            ]
        optimizer.step()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    if not step_one_a_grad_zero:
        raise RuntimeError(
            "flame ordering law violated: dL/dA must be exactly zero at "
            "step 1 while B == 0"
        )
    if not step_one_b_grad_nonzero:
        raise RuntimeError("step-1 dL/dB is unexpectedly zero")

    records["losses"] = write_diftensor(
        output / "losses.diftensor", torch.stack(losses)
    )
    assert final_prediction is not None
    records["prediction"] = write_diftensor(
        output / "prediction.diftensor", final_prediction
    )
    for index, adapter_id in enumerate(ADAPTER_IDS):
        records[f"gradient_step_1_{adapter_id}"] = write_diftensor(
            output / f"gradient-step-1-{adapter_id}.diftensor",
            first_gradients[index],
        )
        records[f"gradient_final_{adapter_id}"] = write_diftensor(
            output / f"gradient-final-{adapter_id}.diftensor",
            final_gradients[index],
        )
        records[f"state_{adapter_id}"] = write_diftensor(
            output / f"state-{adapter_id}.diftensor", adapters[index]
        )
        state = optimizer.state[adapters[index]]
        records[f"moment1_{adapter_id}"] = write_diftensor(
            output / f"moment1-{adapter_id}.diftensor", state["exp_avg"]
        )
        records[f"moment2_{adapter_id}"] = write_diftensor(
            output / f"moment2-{adapter_id}.diftensor", state["exp_avg_sq"]
        )

    script = Path(__file__).resolve()
    manifest = {
        "oracle": (
            "PyTorch F32 activation-path LoRA rectified-flow objective, "
            "autograd, adapter-only AdamW"
        ),
        "oracle_script": str(script),
        "oracle_script_sha256": sha256(script),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "geometry": {
            "rows": ROWS,
            "latent_width": LATENT_WIDTH,
            "timestep_width": TIMESTEP_WIDTH,
            "hidden_width": HIDDEN_WIDTH,
            "rank": RANK,
            "alpha": ALPHA,
            "scale": ALPHA / RANK,
        },
        "optimizer": {
            "kind": "AdamW",
            "learning_rate": args.learning_rate,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "epsilon": args.epsilon,
            "weight_decay": args.weight_decay,
            "foreach": False,
            "fused": False,
            "capturable": False,
        },
        "steps": args.steps,
        "initial_loss": losses[0].item(),
        "final_loss": losses[-1].item(),
        "step_one_a_grad_exactly_zero": step_one_a_grad_zero,
        "step_one_b_grad_nonzero": step_one_b_grad_nonzero,
        "loop_wall_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "tensors": records,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"LORA_ORACLE PASS output={output} steps={args.steps} "
        f"initial_loss={losses[0].item():.10f} "
        f"final_loss={losses[-1].item():.10f} "
        f"a_grad_zero_at_step_1={step_one_a_grad_zero} "
        f"manifest_sha256={sha256(manifest_path)}"
    )


if __name__ == "__main__":
    main()
