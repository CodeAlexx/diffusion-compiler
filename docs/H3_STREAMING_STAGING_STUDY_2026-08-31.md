# H3 streaming/staging study — 2026-08-31

## Verdict

**NO ADMITTED STREAMING SPEEDUP.** The page-cache hypothesis for the slow H3
run is CONFIRMED by attribution, but none of the Wave-1 staging knobs earned
admission: a controlled A/B/A shows the apparent keep-mapped-pages win was
page-cache state, not the flag. Nothing is promoted; defaults are unchanged.

The study was stopped after the keep-mapped-pages arm by direction; threaded
staging and the deeper prefetch ring were NOT measured at H3 scale and remain
untested here.

## Workload (fixed across every arm)

The accepted golden H3 inputs, unchanged: `h3-832x480x175-t439-exact-cudnn`
program/bundle, the accepted conditioning/state tensors, released sigmas,
`--denoise-only --profile-pipeline --max-evaluations 6`, exact BF16, exact
cuDNN attention, modulation cache, no W8A8/CK/groupwise. Only staging flags
varied. Harness: `perf/run_arm.sh`, `perf/rotation.sh`, `perf/summarize.py`;
raw per-arm results under `perf/results/`.

## The measured cause (Experiment 1)

Baseline A0, 6 evaluations, per-evaluation seconds:

    104.5, 32.4, 69.4, 21.4, 21.6, 21.7

Identical work each evaluation, yet a 4.9x spread. Compute cannot vary that
way. Attribution over the run (241.2 GB of streamed weights):

| Component | Time |
|---|---:|
| host staging (mmap -> pinned, includes fault/storage wait) | 188.9 s |
| host wait | 73.8 s |
| **actual H2D copy-engine** | **9.5 s** |
| non-kernel device timeline | 145.2 s |

Host staging costs **20x the real transfer**; the copy engine is nearly idle.
OS counters agree: 31,636 major (I/O) faults and 38,269,184 filesystem input
blocks (~19.6 GB read) in a run whose H2D total is 40.2 GB. The slow
full-native run earlier that day followed 67 GB of conditioner streaming plus
a 49 GB oracle pass, which evicted the denoiser's checkpoint pages.

**Warm steady-state native denoise is ~21.4-23.2 s/evaluation** — at or
better than the accepted run's 22.4 s/evaluation. Cold and hot must not be
blended: the same binary measures 104.5 s cold and 21.4 s warm.

## A/B/A (Experiment 2, keep-mapped-pages only)

| Arm | Cold | Hot mean | Hot median | Major faults | FS input blocks | Wall |
|---|---:|---:|---:|---:|---:|---:|
| A0 baseline | 104.5 s | 33.3 s | 21.7 s | 31,636 | 38,269,184 | 4:37.06 |
| B keep-pages | 21.0 s | 21.8 s | 21.7 s | 27 | 14,032 | 2:23.00 |
| A1 baseline | 21.6 s | 23.0 s | 23.2 s | 2 | — | 2:21.92 |

**A1 is the control that settles it.** A1 runs the unmodified default and is
as fast as B (2:21.92 vs 2:23.00) because B had already warmed the cache. The
A0 -> B improvement is therefore attributable to page-cache state, not to the
flag. B's hot mean is ~5% under A1's, which is inside single-sample drift and
is NOT claimed as a win.

Note also that the hot MEDIAN is identical in A0 and B (21.7 s): keep-pages
never moved the warm floor. It removes stalls, and once the kernel has the
pages cached there are no stalls left to remove.

Correctness: every arm produced **byte-identical** outputs — video latent
`e97367f68f4ef8af…`, audio rows `f990f4e028a3da84…`. Zero nonfinite. A
staging-only change must be bit-identical, and it was.

Memory: B's cgroup-charged peak was 1.17 GB with zero memory events
(low/high/max/oom all 0). `time -v` reports 38.6 GB max RSS for B, but that
is resident *mapped file* pages — reclaimable page cache, not anonymous
allocation; 54 GB stayed available system-wide. The distinction matters
before anyone reads that RSS number as a leak.

## Launch and time attribution (for the next bottleneck)

Per evaluation: 483 kernel launches, 319 cuBLASLt matmuls, 52 cuDNN
attention calls, 445 H2D copies (6.7 GB), 2 D2H, 4,333 event records, 854
stream waits, 427 host event syncs, 0 device allocations (the arena holds).

With staging warm, the remaining time is GPU work plus the non-kernel device
timeline. Launch fusion was deliberately NOT touched in this phase.

## Status of the knobs

| Knob | Status |
|---|---|
| keep-mapped-pages | **NOT ADMITTED** — effect indistinguishable from cache warming under A/B/A |
| threaded staging | **UNTESTED at H3 scale** (study stopped by direction) |
| deeper prefetch ring | **UNTESTED at H3 scale** (study stopped by direction) |

Defaults unchanged. The knobs remain available and default-off.

## Practical guidance

The actionable finding is operational, not a code change: **H3 denoise timing
is only meaningful once the checkpoint pages are cached.** Report cold and
hot separately, and treat any run that follows a large streaming job (the
conditioner, an oracle pass) as cold. A single blended "s/eval" number across
a cache transition is not a measurement.

## Reproduction

    /home/alex/dc-perf/perf/run_arm.sh LABEL OUTDIR 6 [flags...]
    # baseline:   (no flags)
    # keep-pages: --streamed-keep-pages
    python3 /home/alex/dc-perf/perf/summarize.py OUTDIR/LABEL

Branch `runtime-streaming-perf` (worktree /home/alex/dc-perf), cut from
`flame-runtime-integration`. The only source change is exposing the existing
RunOptions staging knobs as `difh3infer` flags; no runtime behavior changed,
and all defaults are identical to before.
