#!/usr/bin/env python3
"""Gate the compiler's FLUX.2 [dev] Mistral conditioning against the oracle fixture.

The fixture (perf/regress/fixtures/flux2-dev-conditioning/reference.safetensors)
holds, for the frozen prompt, the transformers oracle's valid-token rows of the
[512, 15360] conditioning (hidden states 10/20/30 concatenated), plus
input_ids / attention_mask / position_ids for all 512 positions. It was cut from
tools/export_flux2_dev_conditioning_reference.py with --fixture-out.

The candidate is difflux2sample --flux2-model dev --conditioning-only
--state-output OUT (positive_conditioning [512, 15360] or [1, 512, 15360]).

Bars: valid rows cosine >= 0.9999 and relative L2 <= 0.01 overall and per hidden
state; every padded row must be zero to 1e-3 absolute (the oracle's are ~1e-13,
and the DiT reads all 512 rows). Exit 0 on PASS, 1 on FAIL, 2 on a malformed input.
"""
import argparse, sys
import torch
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument("--fixture", required=True)
ap.add_argument("--candidate", required=True)
ap.add_argument("--min-cosine", type=float, default=0.9999)
ap.add_argument("--max-rel-l2", type=float, default=0.01)
ap.add_argument("--max-pad-abs", type=float, default=1e-3)
a = ap.parse_args()

fixture = load_file(a.fixture)
candidate = load_file(a.candidate)
mask = fixture["attention_mask"].bool()
valid_ref = fixture["conditioning_valid_rows"].float()
seq, width = mask.numel(), valid_ref.shape[1]
if "positive_conditioning" not in candidate:
    print("FAIL: candidate has no positive_conditioning"); sys.exit(2)
cand = candidate["positive_conditioning"].float().reshape(-1, width)
if cand.shape[0] != seq:
    print(f"FAIL: candidate rows {cand.shape[0]} != fixture positions {seq}"); sys.exit(2)
valid_cand = cand[mask]
if valid_cand.shape != valid_ref.shape:
    print(f"FAIL: valid-row shape {tuple(valid_cand.shape)} != {tuple(valid_ref.shape)}"); sys.exit(2)

ok = True
def check(ref, got, label):
    global ok
    r = ref.reshape(-1).double(); g = got.reshape(-1).double()
    cos = float(torch.dot(r, g) / (r.norm() * g.norm() + 1e-30))
    rel = float((r - g).norm() / (r.norm() + 1e-30))
    passed = cos >= a.min_cosine and rel <= a.max_rel_l2
    ok = ok and passed
    print(f"FLUX2_DEV_CONDITIONING {label}: cosine={cos:.6f} rel_l2={rel:.5f} "
          f"max_abs={float((r-g).abs().max()):.4f} {'ok' if passed else 'FAIL'}")

check(valid_ref, valid_cand, f"valid_rows[{int(mask.sum())}]")
states = width // 5120
for k in range(states):
    check(valid_ref[:, k*5120:(k+1)*5120], valid_cand[:, k*5120:(k+1)*5120], f"hidden_state_{k}")
# The oracle's padded rows (fully-masked left padding) are a constant of order
# 1e-13 in every channel and the compiler reproduces that value; the bar is
# absolute so BF16-level residue never trips it while a real pad-row error does.
pad_absmax = float(cand[~mask].abs().max()) if int((~mask).sum()) else 0.0
pad_ok = pad_absmax <= a.max_pad_abs
ok = ok and pad_ok
print(f"FLUX2_DEV_CONDITIONING pad_rows[{int((~mask).sum())}]: max_abs={pad_absmax:.3e} {'ok' if pad_ok else 'FAIL'}")
print("PASS: FLUX.2 dev conditioning matches the oracle fixture" if ok else "FAIL: FLUX.2 dev conditioning gate")
sys.exit(0 if ok else 1)
