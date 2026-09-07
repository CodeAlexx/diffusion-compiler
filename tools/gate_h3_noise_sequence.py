#!/usr/bin/env python3
"""CPU oracle for the creator's Ref2VA request RNG order; never a runtime path.

Source: pinned diffusers minimax_h3 encoders.py (one condition draw per
reference), before_denoise.py (target video, then directly packed audio rows).
Also uses non-block-multiple prior draws to catch incorrectly summed skips.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

import numpy as np
import torch


def read_tensor(path):
    data = path.read_bytes()
    rank = struct.unpack_from("<I", data, 16)[0]
    dims = struct.unpack_from(f"<{rank}Q", data, 20)
    return np.frombuffer(data, dtype="<f4", offset=28 + rank * 8,
                         count=int(np.prod(dims))).reshape(dims)


def pack_video(t):
    b, c, frames, h, w = t.shape
    return t.reshape(b, c, frames, h//2, 2, w//2, 2).permute(0, 2, 3, 5, 1, 4, 6).reshape(-1, c*4)


p = argparse.ArgumentParser()
p.add_argument("--binary", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
results = []
for seed in [42, 9001, 4294967295]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    skips = []
    draws = [("condition0", (1, 24, 1, 8, 12)),
             ("condition1", (1, 24, 1, 12, 8)),
             ("target", (1, 24, 7, 4, 6)), ("audio", (74, 32))]
    for label, shape in draws:
        expected = torch.randn(shape, generator=generator, dtype=torch.float32)
        count = expected.numel()
        if len(shape) == 5:
            args = ["--layout", "h3-video", "--latent-frames", str(shape[2]),
                    "--latent-height", str(shape[3]), "--latent-width", str(shape[4])]
            expected = pack_video(expected)
        else:
            args = ["--layout", "flat", "--rows", str(shape[0]), "--cols", str(shape[1])]
        output = a.output / f"{seed}-{label}.diftensor"
        subprocess.run([str(a.binary), "--rng", "torch-cpu", "--seed", str(seed),
                        *skips, *args, "--output", str(output)], check=True, capture_output=True)
        actual = read_tensor(output)
        expected = expected.numpy()
        assert actual.shape == expected.shape
        mismatch = np.count_nonzero(actual.view(np.uint32) != expected.view(np.uint32))
        results.append({"seed":seed,"draw":label,"mismatches":int(mismatch),
                        "sha256":hashlib.sha256(actual.tobytes()).hexdigest()})
        assert mismatch == 0, results[-1]
        skips.extend(["--skip-normal-draw", str(count)])
    # Counts cannot be lumped into one skip when individual draws have tails.
    g = torch.Generator().manual_seed(seed)
    torch.randn(17, generator=g); torch.randn(31, generator=g)
    expected = torch.randn((2, 32), generator=g).numpy()
    output = a.output / f"{seed}-tails.diftensor"
    subprocess.run([str(a.binary), "--rng", "torch-cpu", "--seed", str(seed),
                    "--skip-normal-draw", "17", "--skip-normal-draw", "31",
                    "--rows", "2", "--cols", "32", "--output", str(output)], check=True)
    assert np.array_equal(read_tensor(output).view(np.uint32), expected.view(np.uint32))
    results.append({"seed":seed,"draw":"non16_tails","mismatches":0})
print(json.dumps({"passed":len(results),"torch":torch.__version__,"cases":results}, indent=2))
