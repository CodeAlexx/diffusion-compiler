#!/usr/bin/env python3
"""Summarize one H3 streaming arm: cold vs hot split, staging attribution."""
import json, re, subprocess, sys, statistics as st
from pathlib import Path

d = Path(sys.argv[1]); log = (d / "run.log").read_text(errors="ignore")
ev = [float(x) / 1000.0 for x in re.findall(r"denoiser_ms=([0-9.]+)", log)]
out = {"arm": d.name, "evals": len(ev)}
if ev:
    hot = ev[1:] or ev
    out |= {
        "cold_first_s": round(ev[0], 2),
        "hot_mean_s": round(sum(hot) / len(hot), 2),
        "hot_median_s": round(st.median(hot), 2),
        "hot_min_s": round(min(hot), 2),
        "hot_max_s": round(max(hot), 2),
        "hot_stdev_s": round(st.pstdev(hot), 2) if len(hot) > 1 else 0.0,
        "per_eval_s": [round(x, 1) for x in ev],
    }
prof = re.search(r"PIPELINE_PROFILE (.*)", log)
if prof:
    for kv in prof.group(1).split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            try: out["prof_" + k] = float(v)
            except ValueError: out["prof_" + k] = v
tel = re.findall(r"CUDA_LAUNCH_TELEMETRY phase=run (.*)", log)
if tel:
    for kv in tel[-1].split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            try: out["tel_" + k] = float(v)
            except ValueError: pass
for pat, key in [(r"Major \(requiring I/O\) page faults: (\d+)", "major_faults"),
                 (r"Minor \(reclaiming a frame\) page faults: (\d+)", "minor_faults"),
                 (r"File system inputs: (\d+)", "fs_inputs"),
                 (r"Maximum resident set size \(kbytes\): (\d+)", "max_rss_kib"),
                 (r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([0-9:.]+)", "wall")]:
    m = re.search(pat, log)
    if m: out[key] = m.group(1)
for name in ("video-latent.diftensor", "audio-rows.diftensor"):
    f = d / name
    if f.exists():
        out["sha_" + name.split("-")[0]] = subprocess.run(
            ["sha256sum", str(f)], capture_output=True, text=True).stdout.split()[0][:16]
print(json.dumps(out, indent=1))
(d / "summary.json").write_text(json.dumps(out, indent=1))
