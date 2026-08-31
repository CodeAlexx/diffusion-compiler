#!/usr/bin/env python3
"""Deterministic .diftensor generator for the W1-R streamed gate program.

Parses difinspect output, emits every Input/Constant tensor with seeded
content, and writes difrun argument fragments. Dev tool only; not shipped.
"""
import hashlib, os, re, struct, sys
import numpy as np

DTYPE = {"f32": 1, "bf16": 2, "f16": 3, "i8": 4, "i32": 5}

def bf16(f32arr):
    return (f32arr.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)

def write_diftensor(path, dtype_code, dims, payload_bytes):
    header = bytearray(b"DIFTNS01")
    header += struct.pack("<I", 1)              # version
    header += struct.pack("<I", dtype_code)     # dtype
    header += struct.pack("<I", len(dims))      # rank
    for d in dims:
        header += struct.pack("<Q", d)
    header += struct.pack("<Q", len(payload_bytes))
    blob = bytes(header) + payload_bytes
    digest = hashlib.sha256(blob).digest()
    with open(path, "wb") as f:
        f.write(blob); f.write(digest)

inspect_path, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
inputs, outputs, const_bytes = [], [], 0
for line in open(inspect_path):
    m = re.match(r"tensor id=(\d+) dtype=(\w+) roles=(\d+) shape=([0-9x]+) bytes=(\d+)", line)
    if not m:
        continue
    tid, dtype, roles, shape, nbytes = int(m[1]), m[2], int(m[3]), m[4], int(m[5])
    dims = [int(x) for x in shape.split("x")]
    n = int(np.prod(dims))
    if roles & 2:
        outputs.append(tid)
    if not (roles & 1 or roles & 4):
        continue
    assert dtype == "bf16", f"unexpected dtype {dtype}"
    rng = np.random.default_rng((tid * 2654435761) & 0xFFFFFFFF)
    if len(dims) == 1:
        values = 1.0 + 0.05 * rng.standard_normal(n)      # norm/gate vectors
    elif roles & 1:
        values = 0.05 * rng.standard_normal(n)            # dynamic inputs
    else:
        values = 0.02 * rng.standard_normal(n)            # weights
    path = os.path.join(out_dir, f"t{tid}.diftensor")
    write_diftensor(path, DTYPE[dtype], dims, bf16(values).tobytes())
    if roles & 4:
        const_bytes += nbytes
    inputs.append((tid, path))

with open(os.path.join(out_dir, "difrun_inputs.txt"), "w") as f:
    for tid, path in inputs:
        f.write(f"--input {tid}={path}\n")
with open(os.path.join(out_dir, "difrun_outputs.txt"), "w") as f:
    for tid in outputs:
        f.write(f"--output {tid}={out_dir}/out_{tid}.diftensor\n")
print(f"generated={len(inputs)} outputs={outputs} constant_bytes={const_bytes} ({const_bytes/2**30:.3f} GiB)")
