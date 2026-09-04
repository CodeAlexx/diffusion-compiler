#!/usr/bin/env python3
"""Compare a compiler SDXL stage output against its creator oracle.

usage: gate_sdxl_stage.py EXPECTED.safetensors ACTUAL.safetensors
           [--min-cos 0.999] [--max-rel-l2 0.05] [--skip name ...] [--measure]

Every tensor present in both files is compared in float64: cosine, relative
L2, max abs. Bars are frozen per stage by the caller; --measure prints without
failing. Exit 0 = all pass, 1 = a bar failed, 2 = a boundary is missing.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
from safetensors.torch import load_file as load_torch


def load_file(path: str) -> dict[str, np.ndarray]:
    # torch loading so bf16/f16 fixtures decode; compare in float64.
    return {name: value.float().numpy() for name, value in load_torch(path).items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("expected")
    parser.add_argument("actual")
    parser.add_argument("--min-cos", type=float, default=0.999)
    parser.add_argument("--max-rel-l2", type=float, default=0.05)
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--tag", default="SDXL_STAGE")
    args = parser.parse_args()
    expected = load_file(args.expected)
    actual = load_file(args.actual)
    names = [n for n in expected if n in actual and n not in args.skip]
    if args.only:
        names = [n for n in names if n in args.only]
    missing = [n for n in actual if n not in expected and n not in args.skip]
    if not names:
        print(f"{args.tag}: no shared boundaries")
        sys.exit(2)
    failed = 0
    for name in names:
        e = expected[name].astype(np.float64).ravel()
        a = actual[name].astype(np.float64).ravel()
        if e.shape != a.shape:
            print(f"{args.tag} {name}: shape {expected[name].shape} vs {actual[name].shape} FAIL")
            failed += 1
            continue
        denom = np.linalg.norm(e) * np.linalg.norm(a)
        cos = float(np.dot(e, a) / denom) if denom > 0 else (1.0 if np.allclose(e, a) else 0.0)
        rel = float(np.linalg.norm(e - a) / max(np.linalg.norm(e), 1e-30))
        mad = float(np.max(np.abs(e - a))) if e.size else 0.0
        ok = cos >= args.min_cos and rel <= args.max_rel_l2
        status = "ok" if ok or args.measure else "FAIL"
        if not ok and not args.measure:
            failed += 1
        print(f"{args.tag} {name}: n={e.size} cosine={cos:.7f} rel_l2={rel:.3e} max_abs={mad:.3e} "
              f"ref_absmax={float(np.max(np.abs(e))) if e.size else 0:.3e} {status}")
    if missing:
        print(f"{args.tag}: actual-only tensors ignored: {sorted(missing)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
