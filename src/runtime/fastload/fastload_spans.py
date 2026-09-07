#!/usr/bin/env python3
"""fastload_spans.py -- span table for fastload_bench from a safetensors header.

Reads only the header (never the data), sorts tensors by file offset, packs
each into a 256-byte aligned arena slot in file order, and writes:
  <out>.bin   uint64 nspans, then nspans x {file_off, nbytes, dst_off, convert, 0}
  <out>.json  the same rows with names/dtypes/shapes, for humans and the gate
Conversion: BF16 -> 0 (raw), F16 -> 1, F32 -> 2. Other dtypes are rejected.
"""
import json, struct, sys

CONVERT = {"BF16": 0, "F16": 1, "F32": 2}

def main():
    if len(sys.argv) != 3:
        print("usage: fastload_spans.py <checkpoint.safetensors> <out-prefix>")
        return 2
    path, out = sys.argv[1], sys.argv[2]
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    base = 8 + n
    rows = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dt = info["dtype"]
        if dt not in CONVERT:
            print(f"unsupported dtype {dt} for {name}")
            return 3
        a, b = info["data_offsets"]
        rows.append((base + a, b - a, name, dt, info["shape"]))
    rows.sort()
    dst = 0
    spans = []
    for off, nbytes, name, dt, shape in rows:
        conv = CONVERT[dt]
        dev = nbytes // 2 if conv == 2 else nbytes
        spans.append({"file_off": off, "nbytes": nbytes, "dst_off": dst, "convert": conv,
                      "dev_bytes": dev, "name": name, "dtype": dt, "shape": shape})
        dst = (dst + dev + 255) & ~255
    with open(out + ".bin", "wb") as f:
        f.write(struct.pack("<Q", len(spans)))
        for s in spans:
            f.write(struct.pack("<QQQII", s["file_off"], s["nbytes"], s["dst_off"], s["convert"], 0))
    with open(out + ".json", "w") as f:
        json.dump({"path": path, "arena_bytes": dst, "spans": spans}, f)
    print(f"spans={len(spans)} file_bytes={sum(s['nbytes'] for s in spans)} arena_bytes={dst}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
