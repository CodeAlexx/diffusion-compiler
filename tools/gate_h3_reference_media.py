#!/usr/bin/env python3
"""CPU oracle only; Python/Pillow are never called by the native runner."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--binary", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
rng = np.random.default_rng(9001)
results = []
for n, (w, h, short) in enumerate([(32,32,32), (97,65,64), (65,97,64),
        (160,80,64), (129,129,64), (7,11,64), (53,37,2048), (128,32,64)]):
    image = Image.fromarray(rng.integers(0, 256, (h,w,3), dtype=np.uint8))
    src, dst = a.output / f"{n}-source.png", a.output / f"{n}-native.png"
    image.save(src)
    report = subprocess.check_output([str(a.binary), "prepare-image", "--image", str(src),
        "--short-edge", str(short), "--output", str(dst)], text=True)
    scale = short / min(w,h)
    expected_size = (round(w*scale/32)*32, round(h*scale/32)*32)
    expected = np.asarray(image.resize(expected_size, Image.Resampling.LANCZOS))
    actual = np.asarray(Image.open(dst).convert("RGB"))
    assert actual.shape == expected.shape
    mismatch = int(np.count_nonzero(actual != expected))
    results.append({"source":[w,h], "output":expected_size, "mismatches":mismatch,
        "native_report":json.loads(report)})
    assert mismatch == 0, results[-1]
print(json.dumps({"passed":len(results), "cases":results}, indent=2))
