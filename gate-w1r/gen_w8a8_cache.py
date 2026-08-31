#!/usr/bin/env python3
"""Synthetic H3 W8A8 MLP cache for the W1-R gate program (12 blocks,
hidden=3072, ffn=8192). Valid shapes/dtypes with seeded content; byte-identity
gates only need the SAME cache through both binaries."""
import json, struct, sys
import numpy as np

HIDDEN, FFN, BLOCKS = 3072, 8192, 12
tensors = {}
order = []
for n in range(BLOCKS):
    rng = np.random.default_rng(1000 + n)
    w1 = rng.integers(-64, 65, size=(HIDDEN, HIDDEN), dtype=np.int8)
    s1 = (1e-3 * (0.5 + rng.random(HIDDEN))).astype(np.float32)
    w2 = rng.integers(-64, 65, size=(2 * FFN, HIDDEN), dtype=np.int8)
    s2 = (1e-3 * (0.5 + rng.random(2 * FFN))).astype(np.float32)
    w3 = rng.integers(-64, 65, size=(HIDDEN, FFN), dtype=np.int8)
    s3 = (1e-3 * (0.5 + rng.random(HIDDEN))).astype(np.float32)
    for suffix, arr, dt in ((".weight.1", w1, "I8"), (".scale.1", s1, "F32"),
                            (".weight.2", w2, "I8"), (".scale.2", s2, "F32"),
                            (".weight.3", w3, "I8"), (".scale.3", s3, "F32")):
        name = f"block.{n}{suffix}"
        tensors[name] = (dt, list(arr.shape), arr.tobytes())
        order.append(name)

header, offset = {}, 0
for name in order:
    dt, shape, blob = tensors[name]
    header[name] = {"dtype": dt, "shape": shape,
                    "data_offsets": [offset, offset + len(blob)]}
    offset += len(blob)
header_bytes = json.dumps(header).encode()
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<Q", len(header_bytes)))
    f.write(header_bytes)
    for name in order:
        f.write(tensors[name][2])
print(f"wrote {sys.argv[1]} blocks={BLOCKS} bytes={8+len(header_bytes)+offset}")
