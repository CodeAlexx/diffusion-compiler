#!/usr/bin/env python3
"""Deterministic PyTorch reference for the BF16 DiT-block LoRA training gate.

Mirrors dif::frontend::make_dit_lora_training exactly, with the repository's
BF16 dtype contract: BF16-valued
tensors, F32 reference math, ONE round-to-nearest-even at each stored-tensor
boundary — i.e. a q() fake-quant after every DiffIR operation output.
Torch's own BF16 autograd (per-primitive rounding at torch-op granularity)
is NOT the reference; the q() boundaries below place the rounding points
exactly where the DiffIR kernels store, in forward AND (via autograd through
the cast pairs) in backward.

LoRA per site (q,k,v,out,fc1,fc2): adapters A [rank,in] / B [out,rank]
stored F32 leaves, entering through a q() boundary (the graph's Cast);
delta = (x @ A16^T @ B16^T) * bf16(alpha/rank), each stored tensor rounded;
base weights/biases frozen BF16 constants.  Loss is F32 MSE of the BF16
prediction; AdamW is the exact F32 receipt over the F32 adapters with F32
moments (decoupled decay, never folded into the gradient).

Writes into OUT_DIR: config.json; x/cos/sin/target (BF16);
block<b>-{scale1,shift1,gate1,scale2,shift2,gate2} (BF16);
frozen-<i> (BF16, canonical 16-per-block order); adapter-<i> (F32, per-site
A,B order); ref-losses (F32 [steps]); ref-prediction (BF16);
ref-grad1-<i>, ref-grad-<i>, ref-param-<i>, ref-moment1-<i>,
ref-moment2-<i> (all F32, adapter order).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import torch

FROZEN_NAMES = [
    "norm1_w", "q_w", "q_b", "k_w", "k_b", "v_w", "v_b", "q_norm_w",
    "k_norm_w", "out_w", "out_b", "norm2_w", "fc1_w", "fc1_b", "fc2_w",
    "fc2_b",
]
SITE_NAMES = ["q", "k", "v", "out", "fc1", "fc2"]

DTYPE_CODES = {torch.float32: 1, torch.bfloat16: 2}


def write_diftensor(path: Path, tensor: torch.Tensor) -> None:
    value = tensor.detach().cpu().contiguous()
    if value.dtype not in DTYPE_CODES:
        raise TypeError(f"expected float32/bfloat16, got {value.dtype}")
    if value.dtype == torch.bfloat16:
        raw = value.view(torch.uint16).numpy().tobytes(order="C")
    else:
        raw = value.numpy().tobytes(order="C")
    shape = tuple(int(dimension) for dimension in value.shape)
    payload = bytearray(b"DIFTNS01")
    payload += struct.pack("<III", 1, DTYPE_CODES[value.dtype], len(shape))
    for dimension in shape:
        payload += struct.pack("<Q", dimension)
    payload += struct.pack("<Q", len(raw))
    payload += raw
    payload += hashlib.sha256(payload).digest()
    path.write_bytes(payload)


def q(tensor: torch.Tensor) -> torch.Tensor:
    """One stored-tensor boundary: round to BF16, compute onward in F32."""
    return tensor.to(torch.bfloat16).to(torch.float32)


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


def attention(qq, kk, vv, causal: bool):
    S, H, D = qq.shape
    scale = 1.0 / math.sqrt(D)
    scores = torch.einsum("qhd,khd->hqk", qq, kk) * scale
    if causal:
        mask = torch.ones(S, S, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("hqk,khd->qhd", probabilities, vv)


def lora_linear(x, base_w, base_b, lora_a, lora_b, scale16, out_shape,
                bf16: bool):
    """One LoRA-augmented Linear with a boundary at every stored tensor."""
    r = q if bf16 else (lambda t: t)
    S = x.shape[0]
    x2 = x.reshape(S, -1)
    base = r(torch.nn.functional.linear(x2, base_w, base_b)).view(*out_shape)
    a16 = r(lora_a)
    b16 = r(lora_b)
    low = r(torch.nn.functional.linear(x2, a16))
    delta = r(torch.nn.functional.linear(low, b16)).view(*out_shape)
    scaled = r(delta * scale16)
    return r(base + scaled)


def block_forward(x, frozen, sites, modulation, cos, sin, scale16, config):
    bf16 = config["lora_bf16"]
    r = q if bf16 else (lambda t: t)
    eps = config["epsilon_norm"]
    S = config["sequence"]
    H, D = config["heads"], config["head_dim"]
    (norm1_w, q_w, q_b, k_w, k_b, v_w, v_b, qn_w, kn_w, out_w, out_b,
     norm2_w, fc1_w, fc1_b, fc2_w, fc2_b) = frozen
    scale1, shift1, gate1, scale2, shift2, gate2 = modulation
    m1 = r(rms_norm(x, eps) * norm1_w * (1.0 + scale1) + shift1)
    qs = lora_linear(m1, q_w, q_b, sites[0][0], sites[0][1], scale16,
                     (S, H, D), bf16)
    ks = lora_linear(m1, k_w, k_b, sites[1][0], sites[1][1], scale16,
                     (S, H, D), bf16)
    vs = lora_linear(m1, v_w, v_b, sites[2][0], sites[2][1], scale16,
                     (S, H, D), bf16)
    qr = r(qk_norm_rope(qs, qn_w, cos, sin, config["rotary_dim"], eps))
    kr = r(qk_norm_rope(ks, kn_w, cos, sin, config["rotary_dim"], eps))
    att = r(attention(qr, kr, vs, config["causal"]))
    projected = lora_linear(att, out_w, out_b, sites[3][0], sites[3][1],
                            scale16, (S, H * D), bf16)
    x1 = r(x + gate1 * projected)
    m2 = r(rms_norm(x1, eps) * norm2_w * (1.0 + scale2) + shift2)
    h = lora_linear(m2, fc1_w, fc1_b, sites[4][0], sites[4][1], scale16,
                    (S, 2 * config["mlp_width"]), bf16)
    M = config["mlp_width"]
    gate_half, value_half = h[:, :M], h[:, M:]
    sw = r(value_half * torch.nn.functional.silu(gate_half))
    f = lora_linear(sw, fc2_w, fc2_b, sites[5][0], sites[5][1], scale16,
                    (S, H * D), bf16)
    return r(x1 + gate2 * f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--sequence", type=int, default=16)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=8)
    parser.add_argument("--mlp-width", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--rotary-dim", type=int, default=8)
    parser.add_argument("--half-table", action="store_true")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--f32", action="store_true",
                        help="build the F32 ablation variant instead")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5.0e-3)
    parser.add_argument("--seed", type=int, default=20260902)
    arguments = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    out = arguments.output
    out.mkdir(parents=True, exist_ok=True)
    bf16 = not arguments.f32

    config = {
        "sequence": arguments.sequence,
        "heads": arguments.heads,
        "head_dim": arguments.head_dim,
        "mlp_width": arguments.mlp_width,
        "blocks": arguments.blocks,
        "rotary_dim": arguments.rotary_dim,
        "full_rope_table": not arguments.half_table,
        "causal": arguments.causal,
        "lora_rank": arguments.rank,
        "lora_alpha": arguments.alpha,
        "lora_bf16": bf16,
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
    R = arguments.rank
    T = arguments.rotary_dim if not arguments.half_table \
        else arguments.rotary_dim // 2
    store_dtype = torch.bfloat16 if bf16 else torch.float32

    def stored(value):
        """Quantize a fixture tensor to the storage dtype; return the
        dequantized F32 view used by the reference math."""
        held = value.to(store_dtype)
        return held, held.to(torch.float32)

    x_held, x = stored(rand(S, HD))
    positions = torch.arange(S, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(T, dtype=torch.float32) / max(T, 1))
    angles = positions * frequencies
    cos_held, cos = stored(torch.cos(angles))
    sin_held, sin = stored(torch.sin(angles))
    target_held, target = stored(rand(S, HD))
    write_diftensor(out / "x.diftensor", x_held)
    write_diftensor(out / "cos.diftensor", cos_held)
    write_diftensor(out / "sin.diftensor", sin_held)
    write_diftensor(out / "target.diftensor", target_held)

    shapes = {
        "norm1_w": (HD,), "q_w": (HD, HD), "q_b": (HD,), "k_w": (HD, HD),
        "k_b": (HD,), "v_w": (HD, HD), "v_b": (HD,), "q_norm_w": (D,),
        "k_norm_w": (D,), "out_w": (HD, HD), "out_b": (HD,),
        "norm2_w": (HD,), "fc1_w": (2 * M, HD), "fc1_b": (2 * M,),
        "fc2_w": (HD, M), "fc2_b": (HD,),
    }
    site_in = {"q": HD, "k": HD, "v": HD, "out": HD, "fc1": HD, "fc2": M}
    site_out = {"q": HD, "k": HD, "v": HD, "out": HD, "fc1": 2 * M,
                "fc2": HD}

    modulations = []
    frozen_blocks = []
    adapters = []  # F32 leaves, (A,B) per site, block-major
    frozen_index = 0
    for block in range(arguments.blocks):
        modulation = []
        for name in ("scale1", "shift1", "gate1", "scale2", "shift2",
                     "gate2"):
            held, deq = stored(rand(S, HD, amplitude=0.2))
            write_diftensor(out / f"block{block}-{name}.diftensor", held)
            modulation.append(deq)
        modulations.append(modulation)
        frozen = []
        for name in FROZEN_NAMES:
            shape = shapes[name]
            if name.endswith("_w") and len(shape) == 2:
                bound = 1.0 / math.sqrt(shape[1])
                value = rand(*shape, amplitude=bound)
            elif name in ("norm1_w", "norm2_w", "q_norm_w", "k_norm_w"):
                value = rand(*shape, amplitude=0.2, offset=1.0)
            else:
                value = rand(*shape, amplitude=0.05)
            held, deq = stored(value)
            write_diftensor(out / f"frozen-{frozen_index}.diftensor", held)
            frozen_index += 1
            frozen.append(deq)
        frozen_blocks.append(frozen)
        for name in SITE_NAMES:
            bound = 1.0 / math.sqrt(site_in[name])
            lora_a = rand(R, site_in[name], amplitude=0.97 * bound)
            lora_b = torch.zeros(site_out[name], R, dtype=torch.float32)
            adapters.append(lora_a)
            adapters.append(lora_b)
    for index, value in enumerate(adapters):
        write_diftensor(out / f"adapter-{index}.diftensor", value)

    leaves = [value.clone().requires_grad_(True) for value in adapters]
    first_moments = [torch.zeros_like(value) for value in adapters]
    second_moments = [torch.zeros_like(value) for value in adapters]
    scale_value = arguments.alpha / R
    scale16 = (torch.tensor(scale_value).to(store_dtype)
               .to(torch.float32).item())
    beta1, beta2 = config["beta1"], config["beta2"]
    lr, eps_adam = config["learning_rate"], config["epsilon_adam"]
    weight_decay = config["weight_decay"]
    losses = []
    step1_gradients = None
    final_gradients = None
    final_prediction = None
    for step in range(1, arguments.steps + 1):
        activation = x
        for block in range(arguments.blocks):
            sites = [
                (leaves[(block * 6 + site) * 2],
                 leaves[(block * 6 + site) * 2 + 1])
                for site in range(6)
            ]
            activation = block_forward(activation, frozen_blocks[block],
                                       sites, modulations[block], cos, sin,
                                       scale16, config)
        loss = torch.mean((activation - target) ** 2)
        gradients = torch.autograd.grad(loss, leaves)
        losses.append(float(loss))
        if step == 1:
            step1_gradients = [g.detach().clone() for g in gradients]
            a_zero = all(torch.all(step1_gradients[i] == 0.0).item()
                         for i in range(0, len(leaves), 2))
            b_nonzero = all(torch.any(step1_gradients[i] != 0.0).item()
                            for i in range(1, len(leaves), 2))
            if not a_zero:
                raise RuntimeError(
                    "flame ordering law violated: dL/dA must be exactly "
                    "zero at step 1 while B == 0")
            if not b_nonzero:
                raise RuntimeError("step-1 dL/dB is unexpectedly zero")
        final_gradients = [g.detach().clone() for g in gradients]
        if step == arguments.steps:
            final_prediction = activation.detach().to(store_dtype)
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
    write_diftensor(out / "ref-prediction.diftensor", final_prediction)
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
          f"{arguments.steps} steps; {len(leaves)} adapter parameters; "
          f"dtype={'bf16' if bf16 else 'f32'}; "
          f"step1 A-grads exactly zero: True")


if __name__ == "__main__":
    main()
