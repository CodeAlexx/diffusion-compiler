#!/usr/bin/env python3
"""Creator-faithful Krea 2 text-fusion oracle (development only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def subset(path: Path, prefix: str) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            if name.startswith(prefix):
                result[name[len(prefix) :]] = handle.get_tensor(name)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--creator", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--conditioner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--capture-first-block", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite an existing text-fusion artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.creator))
    from mmdit import RMSNorm, TextFusionTransformer, attention

    text_fusion = TextFusionTransformer(
        num_txt_layers=12,
        txt_dim=2560,
        heads=20,
        multiplier=4,
        bias=False,
        kvheads=20,
    )
    text_mlp = torch.nn.Sequential(
        RMSNorm(2560),
        torch.nn.Linear(2560, 6144),
        torch.nn.GELU(approximate="tanh"),
        torch.nn.Linear(6144, 6144),
    )
    text_fusion.load_state_dict(subset(args.checkpoint, "txtfusion."), strict=True)
    text_mlp.load_state_dict(subset(args.checkpoint, "txtmlp."), strict=True)
    text_fusion = text_fusion.to("cuda", dtype=torch.bfloat16).eval().requires_grad_(False)
    text_mlp = text_mlp.to("cuda", dtype=torch.bfloat16).eval().requires_grad_(False)

    source = load_file(args.conditioner)
    taps = [source[f"tap_{i:02d}_hidden_{h:02d}"] for i, h in enumerate((2,5,8,11,14,17,20,23,26,29,32,35))]
    context = torch.stack(taps, dim=1).unsqueeze(0).to("cuda")
    mask = source["attention_mask"][:, 34:].to("cuda")
    expanded_mask = mask.unsqueeze(1).unsqueeze(2) * mask.unsqueeze(1).unsqueeze(3)

    outputs: dict[str, torch.Tensor] = {"context_input": context.cpu(), "validity_mask": mask.cpu()}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        x = context.reshape(512, 12, 2560)
        block = text_fusion.layerwise_blocks[0]
        prenorm = block.prenorm(x.contiguous())
        q = block.attn.wq(prenorm)
        k = block.attn.wk(prenorm)
        v = block.attn.wv(prenorm)
        q_heads = rearrange(q, "B L (H D) -> B H L D", H=block.attn.heads)
        k_heads = rearrange(k, "B L (H D) -> B H L D", H=block.attn.kvheads)
        v_heads = rearrange(v, "B L (H D) -> B H L D", H=block.attn.kvheads)
        qnorm, knorm, _ = block.attn.qknorm(q_heads, k_heads, v_heads)
        attended = attention(qnorm, knorm, v_heads, mask=None, gqa=False)
        gate_logits = block.attn.gate(prenorm)
        gate = torch.sigmoid(gate_logits)
        gated = attended * gate
        projected_attention = block.attn.wo(gated)
        attention_residual = x + projected_attention
        postnorm = block.postnorm(attention_residual)
        mlp_gate = block.mlp.gate(postnorm)
        mlp_up = block.mlp.up(postnorm)
        mlp_gate_activated = torch.nn.functional.silu(mlp_gate)
        mlp_activation = mlp_gate_activated * mlp_up
        mlp_output = block.mlp.down(mlp_activation)
        x = attention_residual + mlp_output
        if args.capture_first_block:
            outputs.update({
            "first_prenorm": prenorm.cpu(),
            "first_q": q.reshape(512 * 12, 2560).cpu(),
            "first_k": k.reshape(512 * 12, 2560).cpu(),
            "first_v": v.reshape(512 * 12, 2560).cpu(),
            "first_qnorm": qnorm.permute(0, 2, 1, 3).contiguous().cpu(),
            "first_knorm": knorm.permute(0, 2, 1, 3).contiguous().cpu(),
            "first_attention": attended.reshape(512, 12, 20, 128).cpu(),
            "first_gate_logits": gate_logits.reshape(512 * 12, 2560).cpu(),
            "first_gate": gate.reshape(512 * 12, 2560).cpu(),
            "first_gated_attention": gated.reshape(512 * 12, 2560).cpu(),
            "first_attention_projection": projected_attention.reshape(512 * 12, 2560).cpu(),
            "first_attention_residual": attention_residual.cpu(),
            "first_postnorm": postnorm.cpu(),
            "first_mlp_gate": mlp_gate.reshape(512 * 12, -1).cpu(),
            "first_mlp_up": mlp_up.reshape(512 * 12, -1).cpu(),
            "first_mlp_gate_activated": mlp_gate_activated.reshape(512 * 12, -1).cpu(),
            "first_mlp_activation": mlp_activation.reshape(512 * 12, -1).cpu(),
            "first_mlp_output": mlp_output.reshape(512 * 12, 2560).cpu(),
            })
        outputs["layerwise_0"] = x.reshape(1, 512, 12, 2560).cpu()
        x = text_fusion.layerwise_blocks[1](x.contiguous(), mask=None)
        outputs["layerwise_1"] = x.reshape(1, 512, 12, 2560).cpu()
        x = rearrange(x, "(b l) n d -> b l d n", b=1, l=512)
        x = text_fusion.projector(x).squeeze(-1)
        outputs["projected"] = x.cpu()
        for index, block in enumerate(text_fusion.refiner_blocks):
            x = block(x, mask=expanded_mask)
            outputs[f"refiner_{index}"] = x.cpu()
        x = text_mlp(x)
        outputs["conditioning_output"] = x.cpu()
    torch.cuda.synchronize()
    run_seconds = time.perf_counter() - start
    save_file({name: tensor.contiguous() for name, tensor in outputs.items()}, args.output)
    report = {
        "creator_commit": "db3984fbc6e13b34c0064990fc2d95ac64d00058",
        "checkpoint": str(args.checkpoint.resolve()),
        "conditioner_fixture": str(args.conditioner.resolve()),
        "output": str(args.output.resolve()),
        "run_seconds": run_seconds,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "output_shape": list(outputs["conditioning_output"].shape),
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
