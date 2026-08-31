# cuBLASLt bias-epilogue absorption + heuristic persistence — 2026-08-31 (W3)

## 1. Bias-epilogue absorption (`difrun --absorb-linear-bias OP_ID`)

Absorbs an unbiased Linear's exclusive, immediately-following graph-level
`BiasAdd` into the cuBLASLt bias epilogue: one library launch, no
materialized intermediate (excluded from the memory plan through the same
`skipped_operations`/`excluded_tensors` path every fused plan uses).  The
absorbed Linear builds and launches the existing biased-Linear plan form —
input and weight plus the BiasAdd's bias, writing the BiasAdd's output.

Candidate identity: ids are explicit (the `--fuse-linear-swiglu` precedent);
malformed patterns fail closed with a named reason.  DiffIR semantics and
fingerprints are untouched — this is a backend prepared-execution choice
that difopt must carry in candidate identity.

### Numerics classification (measured, see §3)

The biased plan form uses different cuBLASLt layouts than the unbiased form
(the PyTorch-addmm column-major reinterpretation vs native row-major), so
the GEMM itself may select a different algorithm, and the bias is added to
the F32 accumulator before the single output rounding, where the separate
BiasAdd kernel rounds the GEMM output to storage first and rounds again
after the add.  Byte-identity against the unfused pair is therefore NOT a
design guarantee; it is measured per dtype below and the candidate is
accepted or rejected through gate discipline, exactly like every other
approximate-class backend route.

### Write-early alias safety (the arena audit's hazard, enforced)

The absorbed launch runs at the Linear's position but writes the BiasAdd's
output one position early.  The matcher admits a site only when the BiasAdd
immediately follows its Linear and each of input/weight/bias is (a) in a
dedicated slot (Input/Output/resident-Constant), (b) semantically live
through the BiasAdd's position, or (c) impossible for the planner to hand
to the early-written output — no streamed constants in the program AND
either the output is dedicated or the read tensor's aligned slot is smaller
than the output needs.  Anything else fails closed.

### Where sites exist (census)

- mlp training programs (`difc make-mlp-training`, F32 and BF16): 2 sites.
- rectified-flow training (`difc make-rectified-flow-training`): 2 sites
  per microbatch.
- LoRA training (`diftrain make-lora`): 0 sites — its BiasAdd follows the
  LoRA combine Add, not a Linear.
- DiT-block training: 0 sites — bias is already inside the 3-input Linear
  (the epilogue win already exists there by construction).
- gate-w1r / H3 stack: 0 sites — no graph-level BiasAdd.

### Activation epilogues: declined, with evidence

cuBLASLt 12.4 offers RELU/GELU(+BIAS/AUX) epilogues only (cublasLt.h
enum: DEFAULT, RELU*, BIAS, GELU* — no SiLU/SwiGLU).  DiffIR has no ReLU
or GELU opcode (ir.hpp opcodes 1–46; GELU is queued IR-generality work),
and the DiT/H3 activation is SiLU/SwiGLU, which cuBLASLt cannot express.
There is no matchable pattern in any current program, so activation
absorption would be dead code; it extends naturally from this matcher when
a GELU opcode lands.

## 2. Heuristic persistence (`difrun --persist-linear-heuristics`)

PTX-cache-style keyed store under `cache_directory` (same default directory
as the PTX cache).  Key = SHA-256 of the full problem identity: m, n, k,
storage dtype, compute type, bias form, preference workspace bytes,
`cublasLtGetVersion()`, and device arch.  Value = the 7 algorithm-config
knobs (id, tile, stages, split-k, reduction scheme, CTA swizzle, custom
option).  Distinct namespaces: `linear-algo-passive-<sha>` written at plan
build (heuristic front), `linear-algo-tuned-<sha>` written after a
measured tuning selection; restore prefers tuned.

Restores rebuild the algo via `cublasLtMatmulAlgoInit` + the 6 config
attributes and validate through `cublasLtMatmulAlgoCheck` against the
exact operation descriptor and layouts, bounded by the preference
workspace; any failure counts as `rejected` and falls open to fresh
heuristics.  Cache writes fail open (a filesystem problem never fails a
prepare).  Restore is skipped for ids under `--tune-linear`,
`--select-linear-algorithm`, or `--expand-linear-algorithms` (those need
the full candidate list).  Default OFF; enabling is an execution-policy
choice for candidate identity.  Counters are reported in
`RunResult.linear_heuristic_cache` and printed by difrun as
`LINEAR_HEURISTIC_CACHE`.

## 3. Measurements

Measured 2026-08-31 on the integration build (RTX 3090 Ti, SM86), all of
it enforced in-tree by `dif_epilogue_tests` (ctest suite 11/11) so these
are regression gates, not one-shot runs.

### Absorption

| Program / dtype | Sites | Launch delta | cuBLASLt dispatches | Output identity |
|---|---:|---|---:|---|
| MLP training, F32 | 2 | kernel launches drop by exactly the site count | unchanged | **byte-identical** to the unfused pair (asserted) |
| MLP training, BF16 | 2 | drop by site count | unchanged | prediction max_abs **0.125** vs the frozen 0.25 bar — NOT byte-identical, as designed (§1) |
| rectified-flow training | 2/microbatch | drop by site count | unchanged | byte-identical (F32) |
| LoRA training, DiT-block, gate-w1r / H3 | 0 | n/a — honest zero census (§1) | n/a | n/a |

The BF16 non-identity is the documented consequence of adding the bias into
the F32 accumulator before a single rounding, versus the unfused pair's two
roundings; it is why absorption ships as an explicit per-op candidate knob
(`--absorb-linear-bias OP_ID`, default off) rather than a silent default.

### Persistence

| Check | Measured |
|---|---|
| Flag off | `restored=0`, `saved_passive=0` — provably inert |
| First persisted prepare | `saved_passive + restored ==` every Linear plan in the program |
| Second persisted prepare | `restored ==` every Linear plan, `rejected=0` |
| Output identity across off / first / second | **byte-identical** |
| Prepare time | first 1.36234 ms, second 1.15619 ms (restore path; single small program — a real saving claim needs a large-program measurement, recorded as open) |

### Regression status

Full ctest 11/11 green (10 pre-existing suites plus the new
`dif_epilogue_tests`); the pre-change baseline was 10/10.

## 4. Composition boundaries

- Executor-side H3 lowerings (`--fuse-linear-swiglu`, `--h3-w8a8-*`,
  `--h3-groupwise-*`, `--h3-modulation-*`) claim Linear ops in the same
  prepare; do not aim `--absorb-linear-bias` at an op those routes claim
  (today they are disjoint program families).
- `--tune-linear` and `--select-linear-algorithm` compose with absorption
  (the tuned launch uses the absorbed biased form) and disable cache
  restore for their ids.
