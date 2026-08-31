#!/usr/bin/env python3
"""Deterministic PyTorch reference for the DiffIR DiT-block training gate.

Mirrors dif::frontend::make_dit_block_training exactly: per block
RmsNormModulate -> q/k/v Linear(+bias) -> per-head RMSNorm + partial
halfsplit RoPE on q,k -> softmax attention (scale 1/sqrt(D), optional
hard-skip causal) -> out Linear -> ResidualGate -> RmsNormModulate ->
fc1 -> SwiGlu (gate first) -> fc2 -> ResidualGate, stacked, MseLoss against
a fixed target, and a manual AdamW loop with the exact AdamWUpdate receipt
(F32 math, biased moments from the raw gradient, bias correction, DECOUPLED
weight decay applied to the parameter, never folded into the gradient).

Writes, into OUT_DIR:
  config.json
  x/cos/sin/target .diftensor
  block<b>-{scale1,shift1,gate1,scale2,shift2,gate2}.diftensor
  param-<i>.diftensor                (initial parameters, canonical order)
  ref-losses.diftensor               [steps] F32
  ref-grad1-<i>.diftensor            step-1 gradients
  ref-param-<i>.diftensor            final parameters
  ref-grad-<i>.diftensor             final-step gradients
  ref-moment1-<i>.diftensor / ref-moment2-<i>.diftensor (final moments)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import torch

PARAM_NAMES = [
    "norm1_w", "q_w", "q_b", "k_w", "k_b", "v_w", "v_b", "q_norm_w",
    "k_norm_w", "out_w", "out_b", "norm2_w", "fc1_w", "fc1_b", "fc2_w",
    "fc2_b",
]


def write_diftensor(path: Path, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    if value.dtype != torch.float32:
        raise TypeError(f"expected float32, got {value.dtype}")
    raw = value.numpy().tobytes(order="C")
    shape = tuple(int(dimension) for dimension in value.shape)
    payload = bytearray(b"DIFTNS01")
    payload += struct.pack("<III", 1, 1, len(shape))
    for dimension in shape:
        payload += struct.pack("<Q", dimension)
    payload += struct.pack("<Q", len(raw))
    payload += raw
    payload += hashlib.sha256(payload).digest()
    path.write_bytes(payload)


def rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


def qk_norm_rope(x, weight, cos, sin, rotary: int, eps: float):
    half = rotary // 2
    n = rms_norm(x, eps) * weight
    c1 = cos[:, :half].unsqueeze(1)
    s1 = sin[:, :half].unsqueeze(1)
    if cos.shape[1] == rotary:
        c2 = cos[:, half:rotary].unsqueeze(1)
        s2 = sin[:, half:rotary].unsqueeze(1)
    else:
        c2, s2 = c1, s1
    a = n[:, :, :half]
    b = n[:, :, half:rotary]
    return torch.cat([a * c1 - b * s1, b * c2 + a * s2, n[:, :, rotary:]],
                     dim=-1)


def attention(q, k, v, causal: bool):
    S, H, D = q.shape
    scale = 1.0 / math.sqrt(D)
    scores = torch.einsum("qhd,khd->hqk", q, k) * scale
    if causal:
        mask = torch.ones(S, S, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probabilities, v)


def block_forward(x, params, modulation, cos, sin, config):
    eps = config["epsilon_norm"]
    S = config["sequence"]
    H, D = config["heads"], config["head_dim"]
    (norm1_w, q_w, q_b, k_w, k_b, v_w, v_b, qn_w, kn_w, out_w, out_b,
     norm2_w, fc1_w, fc1_b, fc2_w, fc2_b) = params
    scale1, shift1, gate1, scale2, shift2, gate2 = modulation
    m1 = rms_norm(x, eps) * norm1_w * (1.0 + scale1) + shift1
    q = torch.nn.functional.linear(m1, q_w, q_b).view(S, H, D)
    k = torch.nn.functional.linear(m1, k_w, k_b).view(S, H, D)
    v = torch.nn.functional.linear(m1, v_w, v_b).view(S, H, D)
    qr = qk_norm_rope(q, qn_w, cos, sin, config["rotary_dim"], eps)
    kr = qk_norm_rope(k, kn_w, cos, sin, config["rotary_dim"], eps)
    att = attention(qr, kr, v, config["causal"]).reshape(S, H * D)
    projected = torch.nn.functional.linear(att, out_w, out_b)
    x1 = x + gate1 * projected
    m2 = rms_norm(x1, eps) * norm2_w * (1.0 + scale2) + shift2
    h = torch.nn.functional.linear(m2, fc1_w, fc1_b)
    M = config["mlp_width"]
    gate_half, value_half = h[:, :M], h[:, M:]
    sw = value_half * torch.nn.functional.silu(gate_half)
    f = torch.nn.functional.linear(sw, fc2_w, fc2_b)
    return x1 + gate2 * f


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--sequence", type=int, default=16)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--mlp-width", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--rotary-dim", type=int, default=8)
    parser.add_argument("--half-table", action="store_true")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5.0e-3)
    parser.add_argument("--seed", type=int, default=20260901)
    arguments = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    out = arguments.output
    out.mkdir(parents=True, exist_ok=True)

    config = {
        "sequence": arguments.sequence,
        "heads": arguments.heads,
        "head_dim": arguments.head_dim,
        "mlp_width": arguments.mlp_width,
        "blocks": arguments.blocks,
        "rotary_dim": arguments.rotary_dim,
        "full_rope_table": not arguments.half_table,
        "causal": arguments.causal,
        "learning_rate": arguments.lr,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon_adam": 1.0e-8,
        "weight_decay": 1.0e-2,
        "epsilon_norm": 1.0e-5,
        "steps": arguments.steps,
        "seed": arguments.seed,
        "torch": torch.__version__,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    generator = torch.Generator().manual_seed(arguments.seed)

    def rand(*shape, amplitude=1.0, offset=0.0):
        return (torch.rand(*shape, generator=generator, dtype=torch.float32)
                * 2.0 - 1.0) * amplitude + offset

    S = arguments.sequence
    H, D, M = arguments.heads, arguments.head_dim, arguments.mlp_width
    HD = H * D
    T = arguments.rotary_dim if not arguments.half_table \
        else arguments.rotary_dim // 2

    x = rand(S, HD)
    positions = torch.arange(S, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(T, dtype=torch.float32) / max(T, 1))
    angles = positions * frequencies
    cos, sin = torch.cos(angles), torch.sin(angles)
    target = rand(S, HD)
    write_diftensor(out / "x.diftensor", x)
    write_diftensor(out / "cos.diftensor", cos)
    write_diftensor(out / "sin.diftensor", sin)
    write_diftensor(out / "target.diftensor", target)

    modulations = []
    parameters = []
    for block in range(arguments.blocks):
        modulation = [rand(S, HD, amplitude=0.2) for _ in range(6)]
        for tensor, name in zip(modulation, ("scale1", "shift1", "gate1",
                                             "scale2", "shift2", "gate2")):
            write_diftensor(out / f"block{block}-{name}.diftensor", tensor)
        modulations.append(modulation)
        shapes = {
            "norm1_w": (HD,), "q_w": (HD, HD), "q_b": (HD,),
            "k_w": (HD, HD), "k_b": (HD,), "v_w": (HD, HD), "v_b": (HD,),
            "q_norm_w": (D,), "k_norm_w": (D,), "out_w": (HD, HD),
            "out_b": (HD,), "norm2_w": (HD,), "fc1_w": (2 * M, HD),
            "fc1_b": (2 * M,), "fc2_w": (HD, M), "fc2_b": (HD,),
        }
        for name in PARAM_NAMES:
            shape = shapes[name]
            if name.endswith("_w") and len(shape) == 2:
                bound = 1.0 / math.sqrt(shape[1])
                value = rand(*shape, amplitude=bound)
            elif name in ("norm1_w", "norm2_w", "q_norm_w", "k_norm_w"):
                value = rand(*shape, amplitude=0.2, offset=1.0)
            else:
                value = rand(*shape, amplitude=0.05)
            parameters.append(value)
    for index, value in enumerate(parameters):
        write_diftensor(out / f"param-{index}.diftensor", value)

    leaves = [p.clone().requires_grad_(True) for p in parameters]
    first_moments = [torch.zeros_like(p) for p in parameters]
    second_moments = [torch.zeros_like(p) for p in parameters]
    beta1, beta2 = config["beta1"], config["beta2"]
    lr, eps_adam = config["learning_rate"], config["epsilon_adam"]
    weight_decay = config["weight_decay"]
    losses = []
    per_block = len(PARAM_NAMES)
    step1_gradients = None
    final_gradients = None
    for step in range(1, arguments.steps + 1):
        activation = x
        for block in range(arguments.blocks):
            params = leaves[block * per_block:(block + 1) * per_block]
            activation = block_forward(activation, params,
                                       modulations[block], cos, sin, config)
        loss = torch.nn.functional.mse_loss(activation, target)
        gradients = torch.autograd.grad(loss, leaves)
        losses.append(float(loss))
        if step == 1:
            step1_gradients = [g.detach().clone() for g in gradients]
        final_gradients = [g.detach().clone() for g in gradients]
        with torch.no_grad():
            for index, (parameter, gradient) in enumerate(
                    zip(leaves, gradients)):
                m = beta1 * first_moments[index] + (1.0 - beta1) * gradient
                v = beta2 * second_moments[index] + \
                    (1.0 - beta2) * gradient * gradient
                first_moments[index] = m
                second_moments[index] = v
                bias1 = 1.0 - beta1 ** step
                bias2_sqrt = math.sqrt(1.0 - beta2 ** step)
                decayed = parameter * (1.0 - lr * weight_decay)
                denominator = torch.sqrt(v) / bias2_sqrt + eps_adam
                parameter.copy_(decayed - (lr / bias1) * m / denominator)

    write_diftensor(out / "ref-losses.diftensor",
                    torch.tensor(losses, dtype=torch.float32))
    for index in range(len(leaves)):
        write_diftensor(out / f"ref-grad1-{index}.diftensor",
                        step1_gradients[index])
        write_diftensor(out / f"ref-grad-{index}.diftensor",
                        final_gradients[index])
        write_diftensor(out / f"ref-param-{index}.diftensor", leaves[index])
        write_diftensor(out / f"ref-moment1-{index}.diftensor",
                        first_moments[index])
        write_diftensor(out / f"ref-moment2-{index}.diftensor",
                        second_moments[index])
    print(f"loss {losses[0]:.10f} -> {losses[-1]:.10f} over "
          f"{arguments.steps} steps; {len(leaves)} parameters")


if __name__ == "__main__":
    main()
