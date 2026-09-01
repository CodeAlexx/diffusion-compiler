#!/usr/bin/env python3
"""Deterministic PyTorch fixtures for the DiffIR audio opcodes (Conv1d 47,
SnakeBeta 48).

Per case this writes the operation inputs and the torch-computed expected
output:
  OUT_DIR/<case>/{input,weight[,bias][,alpha][,beta],expected}.diftensor
plus OUT_DIR/manifest.json.

Case list covers the plan's inventory: plain / dilated 3,5 / grouped /
depthwise / strided / transposed (stride 2 and 5, grouped, depthwise),
zero and replicate padding, the asymmetric 5/6 downsampler pad, transposed
output trim, K in {1,3,4,7,9,11,12}, bias and bias-less, batch 1 and 2.
All fixtures are F32 (the BigVGAN decode path is all-F32; the dtype rules
admit BF16/F16 but gate 1 pins the F32 contract). Non-degenerate random
data (the H=30-silent-zero lesson: never constant fills).

Replicate-padded cases mirror the DiffIR semantics exactly: F.pad(mode=
"replicate") in sample space, then the (transposed) convolution with
padding=0, then output trim for the transposed form. The alias-free
upsampler's x-ratio scale is NOT applied here — in the real pipeline it is
folded into the filter constant at import; the opcode is the pure
convolution.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

torch.manual_seed(0)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

DTYPE_CODES = {torch.float32: 1}


def write_diftensor(path: Path, tensor: torch.Tensor) -> dict[str, object]:
    value = tensor.detach().cpu().contiguous()
    if value.dtype not in DTYPE_CODES:
        raise TypeError(f"audio fixtures are F32-only, got {value.dtype}")
    raw = value.numpy().tobytes(order="C")
    shape = tuple(int(d) for d in value.shape)
    payload = bytearray(b"DIFTNS01")
    payload += struct.pack("<III", 1, DTYPE_CODES[value.dtype], len(shape))
    for dimension in shape:
        payload += struct.pack("<Q", dimension)
    payload += struct.pack("<Q", len(raw))
    payload += raw
    payload += hashlib.sha256(payload).digest()
    path.write_bytes(payload)
    return {"path": str(path), "shape": list(shape),
            "sha256": hashlib.sha256(payload).hexdigest()}


def rand(*shape, amplitude=1.0, offset=0.0, generator=None):
    return (torch.rand(*shape, generator=generator, dtype=torch.float32)
            * 2.0 - 1.0) * amplitude + offset


def conv_case(name, *, b, c_in, c_out, k, length, stride=1, dilation=1,
              groups=1, pad_left=0, pad_right=0, pad_mode="zero",
              transposed=False, trim_left=0, trim_right=0, bias=True,
              generator=None):
    tensors = {"input": rand(b, c_in, length, generator=generator)}
    if transposed:
        tensors["weight"] = rand(c_in, c_out // groups, k,
                                 amplitude=0.7, generator=generator)
    else:
        tensors["weight"] = rand(c_out, c_in // groups, k,
                                 amplitude=0.7, generator=generator)
    if bias:
        tensors["bias"] = rand(c_out, amplitude=0.4, generator=generator)

    x = tensors["input"]
    if pad_mode == "replicate":
        if pad_left or pad_right:
            x = F.pad(x, (pad_left, pad_right), mode="replicate")
        torch_pad = 0
    else:
        torch_pad = 0
        if pad_left or pad_right:
            x = F.pad(x, (pad_left, pad_right))
    bias_tensor = tensors.get("bias")
    if transposed:
        full = F.conv_transpose1d(x, tensors["weight"], bias_tensor,
                                  stride=stride, padding=torch_pad,
                                  groups=groups)
        end = full.shape[-1] - trim_right
        expected = full[..., trim_left:end]
    else:
        expected = F.conv1d(x, tensors["weight"], bias_tensor, stride=stride,
                            padding=torch_pad, dilation=dilation,
                            groups=groups)
    tensors["expected"] = expected
    attrs = {"stride": stride, "dilation": dilation, "groups": groups,
             "pad_left": pad_left, "pad_right": pad_right,
             "pad_mode": 1 if pad_mode == "replicate" else 0,
             "transposed": transposed, "trim_left": trim_left,
             "trim_right": trim_right, "bias": bias}
    return name, "conv1d", attrs, tensors


def snake_case_(name, *, b, c, length, generator=None):
    tensors = {
        "input": rand(b, c, length, amplitude=2.0, generator=generator),
        "alpha": rand(c, amplitude=0.8, generator=generator),
        "beta": rand(c, amplitude=0.8, generator=generator),
    }
    alpha = torch.exp(tensors["alpha"]).unsqueeze(0).unsqueeze(-1)
    beta = torch.exp(tensors["beta"]).unsqueeze(0).unsqueeze(-1)
    tensors["expected"] = tensors["input"] + (beta + 1e-9).reciprocal() * \
        torch.sin(alpha * tensors["input"]).pow(2)
    return name, "snake_beta", {"epsilon": 1e-9}, tensors


def build_cases():
    g = torch.Generator().manual_seed(20260831)
    yield conv_case("conv_k1_pointwise", b=2, c_in=8, c_out=12, k=1,
                    length=16, generator=g)
    yield conv_case("conv_k3_dilated3", b=2, c_in=6, c_out=6, k=3, length=32,
                    dilation=3, pad_left=3, pad_right=3, generator=g)
    yield conv_case("conv_k7_plain", b=1, c_in=4, c_out=10, k=7, length=21,
                    pad_left=3, pad_right=3, generator=g)
    yield conv_case("conv_k11_dilated5", b=2, c_in=6, c_out=6, k=11,
                    length=64, dilation=5, pad_left=25, pad_right=25,
                    generator=g)
    yield conv_case("conv_k9_grouped3", b=2, c_in=9, c_out=6, k=9, length=40,
                    groups=3, pad_left=4, pad_right=4, generator=g)
    yield conv_case("conv_k3_stride2_nopad", b=2, c_in=5, c_out=7, k=3,
                    length=17, stride=2, bias=False, generator=g)
    yield conv_case("conv_k12_depthwise_replicate_asym", b=2, c_in=5,
                    c_out=5, k=12, length=48, stride=2, groups=5, pad_left=5,
                    pad_right=6, pad_mode="replicate", bias=False,
                    generator=g)
    yield conv_case("conv_k4_transposed_stride2", b=2, c_in=6, c_out=4, k=4,
                    length=15, stride=2, transposed=True, trim_left=1,
                    trim_right=1, generator=g)
    yield conv_case("conv_k9_transposed_stride5", b=1, c_in=8, c_out=4, k=9,
                    length=7, stride=5, transposed=True, trim_left=2,
                    trim_right=2, generator=g)
    yield conv_case("conv_k12_transposed_depthwise_replicate", b=2, c_in=5,
                    c_out=5, k=12, length=20, stride=2, groups=5, pad_left=5,
                    pad_right=5, pad_mode="replicate", transposed=True,
                    trim_left=15, trim_right=15, bias=False, generator=g)
    yield conv_case("conv_k7_transposed_grouped", b=1, c_in=6, c_out=4, k=7,
                    length=9, stride=3, groups=2, transposed=True,
                    trim_left=2, trim_right=2, generator=g)
    yield snake_case_("snake_beta_c7", b=2, c=7, length=33, generator=g)
    yield snake_case_("snake_beta_c1", b=1, c=1, length=19, generator=g)
    yield snake_case_("snake_beta_c64", b=2, c=64, length=40, generator=g)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: export_audio_opcode_fixtures.py OUT_DIR")
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "dif.audio_opcode_fixtures.v1",
                "torch": torch.__version__, "cases": {}}
    for name, op, attrs, tensors in build_cases():
        case_dir = out_dir / name
        case_dir.mkdir(parents=True, exist_ok=True)
        entry = {"op": op, "attrs": attrs, "tensors": {}}
        for tensor_name, tensor in tensors.items():
            entry["tensors"][tensor_name] = write_diftensor(
                case_dir / f"{tensor_name}.diftensor", tensor)
        manifest["cases"][name] = entry
        print(f"AUDIO_FIXTURE {name} op={op} "
              f"expected={list(tensors['expected'].shape)}")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"AUDIO_FIXTURES PASS cases={len(manifest['cases'])}")


if __name__ == "__main__":
    main()
