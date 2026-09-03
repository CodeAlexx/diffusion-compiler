#!/usr/bin/env python3
"""Emit the SquareQ W4 gate fixture (perf/regress/fixtures/squareq-w4-tiny).

Offline fixture generation only. Quantizes one random Linear weight with the
SquareQ v3 codec (format squareq_w4_v1: Hadamard-256 rotated residual, int4
group-64, rank-R low-rank) exactly as scripts/squareq_build_slab.py does, and
writes a one-shard slab (index + plan) plus the byte-level oracle:
  oracle.safetensors: w_hat [out,in] bf16 (core.reconstruct_weight), x [M,in]
  bf16, y_ref [M,out] f32 = x @ w_hat^T in float64.
Usage: export_squareq_w4_fixture.py --squareq-core DIR --out DIR
  (DIR = the directory holding the `squareq` package with core.py)
"""
import argparse, json, os, sys
import torch
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--squareq-core", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--out-features", type=int, default=256)
ap.add_argument("--in-features", type=int, default=512)
ap.add_argument("--rows", type=int, default=64)
ap.add_argument("--rank", type=int, default=32)
a = ap.parse_args()
sys.path.insert(0, a.squareq_core)
from squareq import core  # noqa: E402

torch.manual_seed(7)
key = "blocks.0.linear.weight"
base = key[: -len(".weight")]
w = (torch.randn(a.out_features, a.in_features) * 0.05).to(torch.bfloat16)
tensors, stats = core.quantize_layer(w, rank=a.rank, seed=core.__dict__.get("layer_seed", lambda k: 0)(key) if False else 0)
w_hat = core.reconstruct_weight(tensors["qweight"], tensors["wscales"], tensors["lora_down"], tensors["lora_up"])
x = (torch.randn(a.rows, a.in_features) * 0.5).to(torch.bfloat16)
y_ref = (x.double() @ w_hat.double().t()).float()
os.makedirs(a.out, exist_ok=True)
shard = "model-00001-of-00001.safetensors"
slab = {base + ".qweight": tensors["qweight"], base + ".wscales": tensors["wscales"],
        base + ".lora_down": tensors["lora_down"], base + ".lora_up": tensors["lora_up"]}
save_file(slab, os.path.join(a.out, shard))
json.dump({"metadata": {"format": core.FORMAT_TAG}, "weight_map": {k: shard for k in slab}},
          open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)
plan = {"plan_version": 1, "format": core.FORMAT_TAG, "model": "fixture", "source": "random",
        "hblock": core.HBLOCK, "group": core.GROUP,
        "layers": {key: {"out": a.out_features, "in": a.in_features, "rank": a.rank,
                         "format": core.FORMAT_TAG, "cos_w": stats["cos_w"], "rel_l2": stats["rel_l2"],
                         "bytes_q": stats["bytes_q"], "bytes_bf16": stats["bytes_bf16"]}},
        "passthrough": [], "totals": {"quantized_layers": 1, "cos_w_min": stats["cos_w"], "cos_w_mean": stats["cos_w"]}}
json.dump(plan, open(os.path.join(a.out, "squareq-plan.json"), "w"), indent=1)
save_file({"w_hat": w_hat.to(torch.bfloat16).contiguous(), "w": w, "x": x, "y_ref": y_ref.contiguous()},
          os.path.join(a.out, "oracle.safetensors"))
print(f"fixture written to {a.out}: cos_w={stats['cos_w']:.6f} rel_l2={stats['rel_l2']:.5f} bytes_q={stats['bytes_q']}")
