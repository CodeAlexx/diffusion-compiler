# Optimization engine

This document describes the backend-neutral optimization and search layer that
sits between a verified DiffIR program and backend lowering, and records the
measurements taken while building it.

## Where the layer sits

```text
source/model frontend
        |
      DiffIR
        |
  verify / fingerprint
        |
  OPTIMIZATION ENGINE  ......... dif::opt
    /       |        \
 graph    memory    numeric
rewrites  policy    policy
    \       |        /
     candidate DiffIR
           |
        verify
           |
    backend lowering
           |
       benchmark
           |
 numerical / memory gate
      /          \
   REJECT       ACCEPT
                   |
            tuning database
                   |
               best plan
```

Nothing below `dif::opt` changed. The IR, the verifier, `ir::fingerprint`, the
backend ABI, `compiler::plan_memory`, `tune::Database`, the CUDA runtime, and
the H3 frontends are used exactly as they were.

| Component | Header | Role |
|---|---|---|
| Transform vocabulary | `dif/opt/transform.hpp` | The explicit, serializable unit of optimization |
| Source-fidelity guards | `dif/opt/semantics.hpp` | Which operations may be reinterpreted, and how |
| Legality and application | `dif/opt/rewrite.hpp` | `discover` enumerates legal transforms, `apply` performs one |
| Acceptance oracle | `dif/opt/gate.hpp` | **Trusted.** Measures a candidate and applies fixed bars |
| Search | `dif/opt/search.hpp` | Beam search, measurement, winner selection, journal |
| Plan | `dif/opt/plan.hpp` | Serialize and replay the winning transform sequence |
| Experiment bindings | `dif/opt/bindings.hpp` | Deterministic constant/input fixture for A/B runs |
| Command line | `tools/difopt.cpp` | `difopt` |

## Transforms are values, not code paths

A candidate program is fully described by `(base program, base bindings,
ordered transform sequence)`. `Transform` is a plain value with a kind, an
operation scope, a tensor scope, and kind-specific parameters. It encodes to one
canonical line of text and to JSON, and `apply` is a pure function of the
transform and the context it is given. Consequences:

- Optimization decisions live in the IR layer. `emit_cuda` was not touched.
- An empty operation scope means "every legal site, in program order", so one
  transform value describes the same rewrite on a one-block region and on a
  fifty-block denoiser.
- A named scope must still be legal when replayed. A plan that no longer applies
  reports drift instead of silently doing less.

A `RewriteContext` carries the program, its bound constants and inputs, and the
streaming prefetch distance. Constants travel with the program because constant
folding and quantization both rewrite values, not only graph shape.

### Vocabulary

| Kind | Class | What it does | Legality |
|---|---|---|---|
| `fold_constant_subgraph` | structural | Evaluates an operation whose inputs are all constants and replaces its results with new constants, deleting the operation and any constant it made redundant | Opcode is bit-exact data movement (or arithmetic folding was explicitly requested); every input is a bound constant; every output is internal |
| `eliminate_dead_operations` | structural | Removes operations whose results nothing consumes, to a fixed point | Operation is pure and no output carries an interface role |
| `common_subexpression` | structural | Redirects a duplicated operation's consumers at the original | Same opcode, inputs, attributes and result types; outputs internal |
| `fuse_linear_bias` | **numeric** | Folds a `bias_add` epilogue into its producing `Linear` | Intermediate is internal and singly consumed; the bias is available where the `Linear` runs; neither operation pins its accumulation |
| `fuse_qkv_projection` | structural | Replaces a weight-side QKV split plus three `Linear` projections with one packed `Linear` plus an activation-side split | The three projections share an activation and attributes; the split's results are internal and singly consumed; the activation is available at the earliest replaced operation |
| `split_qkv_projection` | structural | The inverse rewrite | Packed activation is internal and singly consumed by the split; producer is a two-input `Linear` |
| `elide_cast_round_trip` | structural | Removes a widen-then-narrow cast pair | The widening leg is exact (`bf16`/`f16` to `f32`) and the narrowing leg returns to the source dtype |
| `rematerialize_producer` | structural | Recomputes a value immediately before its last consumer to shorten its live range | Producer is pure with one internal result and at least two consumers; its own inputs are still live at the last consumer |
| `set_block_size` | schedule | Sets the launch block size on named operations | Power of two inside each operation's verifier range |
| `set_tile_shape` | schedule | Sets `TileM`/`TileN`/`TileK` on `Linear` operations | Nonzero extents |
| `set_linear_implementation` | numeric | Selects strict (1) or TF32 (2) `Linear` math | TF32 requires `f32` storage; the operation must not pin its semantics |
| `set_attention_implementation` | numeric | Selects the generated (1) or cuDNN (2) attention candidate | cuDNN requires `bf16`/`f16` |
| `set_operation_precision` | numeric | Re-expresses one operation at another floating-point precision by bracketing it with casts | Operation is dtype-uniform and does not pin its semantics; dtype-dependent implementation selections are reset rather than carried across |
| `quantize_constant_weights` | numeric | Rewrites constant `Linear` and packed-QKV weights into INT4/INT5 payloads plus a dequantization operation, using the existing `compiler::rewrite_lowbit_weights` | Every selected weight's K dimension is compatible with the group size and bit width |
| `set_constant_residency` | memory | Marks constants resident or streamed | Named tensors are constants |
| `set_prefetch_distance` | memory | Changes the streaming prefetch distance handed to the memory planner | Distance actually changes |

### Preserving explicit rounding and accumulation

`dif::opt::pinned_numeric_semantics` marks an operation whose attributes record a
decision taken to reproduce a reference implementation: an explicit
`AccumulatorDType`, the SwiGLU `GateFirst` ordering, the timestep embedding's
`FlipSinToCos` / `DownscaleFreqShift` / `MaxPeriod` conventions, the direct
packed INT5 `Linear`, and the schedule, optimizer-state, and dequantization
opcodes. Numeric-class transforms refuse those operations. Structural transforms
copy every attribute through verbatim.

Two consequences of that rule showed up while building this:

- **Constant folding is restricted to bit-exact data movement.** Folding an
  arithmetic operation would bake the reference backend's rounding into the
  program's constants, which is a different program on any other backend.
  `DiscoveryOptions::arithmetic_constant_folding` exists but is off, and folding
  a `Cast` is admitted only when the conversion widens.
- **Bias epilogue fusion is a numeric transform, not a structural one.** The CPU
  reference seeds a `Linear`'s accumulator with the bias, whereas `bias_add`
  applies it after the reduction. The test suite originally asserted bit
  equality for this rewrite and caught the difference; the transform was
  reclassified rather than the assertion weakened.

## The acceptance oracle is trusted infrastructure

`gate.hpp` / `gate.cpp` contain the whole admission decision. An `AcceptanceGate`
captures its bars at construction, exposes no mutator, and both of its methods
are `const`. Search takes it by `const&`. No transform, candidate, or search
state is visible to it. `optimize` also refuses to run when the base program
cannot clear its own bars on the measurement backend, because every comparison
after that would be meaningless.

Acceptance order, applied per candidate in exactly this sequence:

1. The candidate DiffIR verifies (`ir::verify`) — otherwise `rejected_verify`.
2. The program prepares and executes — otherwise `rejected_execution`.
3. No output value is non-finite — otherwise `rejected_nonfinite`.
4. Max absolute error, cosine similarity, norm ratio and relative L2 are inside
   the bars — otherwise `rejected_numerical`.
5. Planned working set is inside the memory budget — otherwise
   `rejected_memory`.
6. Only then is performance compared, and only among accepted candidates.

Reference outputs come from the portable typed CPU executor running the **base**
program, so candidates are judged against DiffIR semantics rather than against
whatever the measurement backend happened to produce. Outputs are read through
the typed scalar accessors, so a `bf16` or `f16` result is measured at its
declared precision.

## Search

`optimize` runs a deterministic beam search:

- Depth 0 is the baseline. One untimed run of the base program precedes it so
  first-touch page faults are not charged to the baseline and credited to every
  candidate after it.
- At each depth, `discover` enumerates legal transforms for each beam member,
  each is applied to a copy, and the result is measured.
- Candidates are deduplicated by **candidate fingerprint**: SHA-256 over the
  encoded program, the prefetch distance, and the SHA-256 of every bound
  constant. Two transform sequences that reach the same executable program are
  measured once.
- Only accepted candidates become parents. Contexts outside the beam are
  released, because each one holds a full copy of the model's constants.
- `max_candidates` bounds the whole search.

### Objectives and the noise floor

`Objective::Latency` minimizes measured latency with planned memory as a hard
constraint. `Objective::PlannedMemory` minimizes the planned working set,
admitting only candidates whose latency stays within a tolerance of the
baseline.

Both rules are widened to a measured noise bound before any winner is chosen.
The bound is the larger of the spread across the baseline's own timed iterations
and the drift between a baseline measured before the candidates and one measured
after them. A latency objective's improvement margin and a memory objective's
latency tolerance are both raised to it, and the effective values are reported.
The search therefore cannot claim a latency win inside its own measurement
error. With a single iteration per candidate the within-run term vanishes and
the bound is an underestimate, so latency work wants at least a few iterations.

## Reproducibility

An `OptimizationPlan` records the base program fingerprint, the base candidate
fingerprint (program plus constant values), the winning transform sequence, and
the resulting candidate fingerprints. `replay` refuses a base that does not match,
refuses a transform that is no longer legal, and refuses a rebuild that does not
reproduce the recorded candidate fingerprint. `optimize` returns its optimized
context by replaying its own plan, so the program it hands back is exactly what a
clean rebuild produces.

Every candidate is recorded twice.

As a `tune::Measurement` in the existing tuning database: base program
fingerprint (`program_hash`), candidate fingerprint (`candidate_hash`), backend,
device, timings, planned memory, the numerical metrics, the verdict including
its rejection reason (`status`), and the canonical transform sequence (`plan`).
The record format grew two fields for this — `planned_memory_bytes` and `plan` —
so the database went to version 3. Version 1 and 2 files still load, exactly as
version 1 already did when version 2 added the norm ratio, and are rewritten at
version 3 on the next record. `diftune` writes the same table and simply leaves
the two new fields empty.

The `plan` field is the candidate's reproducible identity, not a label:
`decode_transform_sequence` inverts the canonical text form, so a persisted row
can be decoded and replayed against the base program, and must rebuild the
`candidate_hash` recorded beside it. `test_measurements_persist_reproducible_provenance`
holds every row of a search to exactly that, rejected rows included — a refused
candidate keeps full provenance so a later run can see what was already tried
and why it was refused.

And as a full entry in a JSON journal, which additionally carries the depth and
parent, the naive/resident/streamed byte counts, the relative L2, the acceptance
bars, the noise bound, and the effective decision rules.

## Measured results

All numbers below were produced on a four-core x86-64 Linux container with **no
CUDA device**, so the measurement backend is the portable typed CPU reference.
Planned memory is `compiler::plan_memory(...).total_bytes` at 256-byte
alignment: a deterministic, backend-neutral quantity. Latency is
minimum-of-iterations wall time on a shared cloud VM and is reported with its
measured noise bound.

Constants and inputs are the deterministic fixture from
`dif::opt::synthesize_bindings`. **They are an experiment fixture, not a model.
Nothing here is a statement about output quality.**

### Phase-1 proving gate

Each required piece of evidence, and the check that produces it. Every row is
executed by `ctest`; the searches are also reproducible from the command line.

| Required evidence | Where it is demonstrated | Result |
|---|---|---|
| A candidate fails a hard gate | `test_phase_one_h3_optimization_search` | 11 `rejected_numerical`, 11 `rejected_memory` of 48 measured |
| Multiple valid candidates are timed | same | 26 accepted, each timed |
| The fastest accepted candidate is selected automatically | `test_latency_objective_selects_the_fastest_accepted_candidate` | winner is the fastest of every accepted candidate; a candidate 21.8x faster was refused |
| Transformation provenance is persisted | `test_measurements_persist_reproducible_provenance` | 12/12 rows carry a plan that rebuilds their own candidate hash |
| The winning plan serializes | `test_plan_serialization_round_trip` | JSON round-trips |
| Clean-process replay gives the same candidate fingerprint | `difopt --replay` in a fresh build tree | byte-identical `.difir` |
| Replay produces the same accepted numerical result | `test_phase_one_h3_optimization_search`, latency test | outputs bit-identical |
| A memory limit independently rejects a valid candidate | `test_search_enforces_the_memory_constraint` | numerically exact candidates refused for footprint alone |
| Thresholds are unreachable from the transformation APIs | structural | `transform.hpp`, `plan.hpp` and `rewrite.hpp` do not include `gate.hpp` or name `AcceptanceBars`; no non-const `AcceptanceGate&` exists in the tree |
| No manually selected winner | all searches | the winner is chosen by `objective_value` alone; no test names an expected winning transform |

### Phase 1: one real H3 denoiser block

`make_h3_denoiser` with `layers = 1`, `refiner_layers = 1`, hidden 256, 8 heads,
head dim 32, FFN 512, rotary 24, sequence 5, two timestep tables — 57 operations,
117 tensors. Objective: planned memory. Bars: max abs 1e-4, cosine 0.999999,
norm ratio [0.9999, 1.0001], relative L2 1e-3, memory budget = the baseline's own
planned working set. Memory-policy discovery is disabled for this search (see
*Threats to validity*), so the winner has to be an actual program rewrite.

These are the numbers from the exact `difopt` invocation printed under
*Running it* below.

| | value |
|---|---|
| Transforms discovered | 55 |
| Candidates measured | 48 |
| Accepted | 26 |
| Rejected by the numerical gate | 11 |
| Rejected by the memory constraint | 11 |
| Baseline planned working set | 4 078 848 B |
| Winner planned working set | 3 456 768 B |
| **Planned memory reduction** | **15.25 %** |
| Winner max absolute error | 0.0 (bit-exact) |
| Winner plan | `fuse_qkv_projection ; rematerialize_producer ops=5,15` |
| Winner candidate fingerprint | `1f7efe27663da374c51071b4ecd40403d89fdc520ad39380e8be2b1a983d3415` |

Representative rejections, all discovered and measured automatically:

| Candidate | Status | max abs | relative L2 | planned bytes |
|---|---|---|---|---|
| `quantize_constant_weights params=4,16,0` | rejected_numerical | 1.61e-1 | 4.63e-2 | 3 546 368 |
| `quantize_constant_weights params=5,16,0` | rejected_numerical | 1.11e-1 | 3.32e-2 | 3 755 008 |
| `set_operation_precision ops=33 params=3` (f16) | rejected_numerical | 7.82e-3 | 2.26e-3 | 4 537 600 |
| `set_operation_precision ops=48 params=3` (f16) | rejected_numerical | 4.32e-3 | 9.51e-4 | 4 340 992 |
| `set_operation_precision ops=33 params=1` (f32) | rejected_memory | 0.0 | 0.0 | 5 127 424 |
| `set_operation_precision ops=17 params=3` (f16) | rejected_memory | 0.0 | 0.0 | 4 472 064 |

The INT4 and INT5 candidates are the cheapest in planned memory of anything the
search produced and are still rejected, because the numerical gate runs before
performance is looked at. The `f32` precision promotion is numerically perfect
and is still rejected, because the memory constraint runs before performance too.

The winning plan serializes to JSON, replays from the base program to the same
candidate fingerprint, and produces bit-identical outputs. Replay was checked
three ways: in-process inside `dif_opt_tests`, through `difopt --replay` in a
separate process, and through a `difopt` built in a **fresh build tree from a
clean configure**, which rebuilt the optimized `.difir` byte for byte from the
recorded plan alone.

**No latency improvement is claimed for this rewrite on this backend.** Two runs
of the identical command gave winner-versus-baseline latency ratios of 0.91 and
1.16 (baseline 106.1 ms / winner 96.4 ms, and baseline 89.9 ms / winner
104.6 ms), with the search reporting in-run noise bounds of 1.01 and 1.19. The
run-to-run spread on this shared VM is therefore larger than the in-run bound and
far larger than any difference between candidates. Within a single run, several
candidates that are semantic no-ops on the CPU backend (`set_block_size`,
`set_linear_implementation` on a graph the CPU executor does not branch on)
measured between 84.7 ms and 102.4 ms. Planned memory carries none of this error:
it is identical across every run.

The value of the drift control here is not that it rescued a speedup but that it
prevented one from being claimed, and that it correctly loosened the memory
objective's latency constraint when the machine was noisy: in the noisier run the
tolerance widened from 1.05 to 1.19, which is what allowed the genuinely
smallest-memory candidate to win instead of a same-memory candidate that happened
to time faster.

### The latency objective, on a deterministic clock

The Phase-1 search above minimizes planned memory, because wall-clock latency on
this container is not resolvable at the magnitudes the search compares between
candidates. That is a property of the machine, not of the rule, and it would be
dishonest to leave the latency rule itself unproven — "fastest accepted candidate
wins" is the claim the whole acceptance order exists to constrain.

`test_latency_objective_selects_the_fastest_accepted_candidate` proves it against
a deterministic clock instead of a noisy one. A `ScriptedLatencyExecutor`
delegates every *value* to the portable CPU reference — verification, execution,
numerics and planned memory are measured exactly as they are everywhere else —
and replaces only the timer with a pure function of the candidate program. That
function is deliberately adversarial: it rewards precisely what the numerical
gate exists to catch, scoring `f16` and INT4/INT5 candidates fastest. The fastest
program the search can measure is therefore always one that must be rejected.

Measured, on the same one-block H3 fixture (48 candidates, 50 transforms
discovered):

| | latency | verdict |
|---|---|---|
| Baseline | 100.000 ms | — |
| Fastest candidate measured | **0.654 ms** | **`rejected_numerical`** |
| Winner | 14.286 ms | accepted |

The winner is `set_linear_implementation ops=1,3,27,29,54,56 params=2`, and it is
the fastest of every accepted candidate. A candidate **21.8x faster than the
winner** was measured, ranked, and refused. The test asserts each link
separately: the fastest candidate overall is not accepted, its verdict is
specifically the numerical gate rather than memory or an execution failure, it is
strictly faster than the winner, the winner is accepted and cleared the
improvement margin, and no accepted candidate is faster than the winner. The
winning plan then replays to its recorded fingerprint and to identical outputs,
like any other plan.

The deterministic clock has no spread and no drift, so the search reports a noise
bound of exactly `1.0` and leaves the improvement margin at its requested `0.02`.
That is checked too: the noise machinery must not inflate the margin on a quiet
machine any more than it may be bypassed on a noisy one.

### Expansion to the full fifty-block denoiser

`make_h3_denoiser` with `layers = 50`: 890 operations, 1 783 tensors, baseline
planned working set 11 840 512 B. The mechanism carries over unchanged —
discovery, application, verification, execution, gating and plan replay all run
on the large program — but the two searches answer differently, and the
difference is the interesting result:

| Search | Discovery | Winner | Planned bytes | Ratio |
|---|---|---|---|---|
| Program rewrites | structural + schedule + numeric | none above margin | 11 840 512 | 1.000 |
| Memory policy | memory only | `set_constant_residency params=1` | 280 064 | **0.024** |

At fifty blocks the planned working set is dominated by resident per-block
constants. A rewrite that removes activation-shaped intermediates moves a small
fraction of it, so the search reports no improvement above its margin rather than
manufacturing one. The lever that does move a constant-dominated working set is
the residency policy, and the search finds it: streaming the constants cuts the
planned working set by 42x with bit-exact results.

### Threats to validity

- **The CPU reference does not model streaming cost.** Marking constants streamed
  is free there, so a memory objective would take it for nothing. That is why the
  Phase-1 search disables memory-policy discovery: the winner has to be a program
  rewrite. On a device that charges host-to-device bandwidth, the latency
  constraint is what pushes back, and resolving that trade is exactly what the
  search plus the tuning database exist for. The fifty-block residency result is
  reported as a planned-footprint result and nothing more.
- **Latency on this machine is not resolvable at these magnitudes.** Every
  latency figure here is quoted with the search's measured noise bound. Where the
  bound exceeds the difference, no claim is made.
- **The scripted clock proves the selection rule, not hardware performance.** The
  latency numbers in *The latency objective* are generated by a deterministic
  function, so they establish that the search ranks, gates and selects correctly.
  They are not a measurement of any device and imply nothing about how fast any
  of those candidates would actually run. A real latency result needs a real
  backend, which is what `difopt --backend cuda` is for.
- **Planned memory is the planner's model**, not an allocator measurement.
- **`TileM`/`TileN`/`TileK` are not consumed by any current backend**, so
  `set_tile_shape` is implemented but left out of default discovery; enabling it
  would spend measurements on behaviourally identical programs.
- **The search is a greedy beam**, not an exhaustive exploration.

## H3 opcodes: frontend semantics or general primitives?

The engine is model-neutral. Three files mention an H3 opcode at all, and each
mention is a table entry or a peephole pattern, not engine logic:

- `semantics.cpp` lists `H3DeinterleaveQkv`, `H3DeinterleaveQkvWeight` and
  `H3AdaLNSelect` as bit-exact data movement, alongside `GatherRows`,
  `SelectRowChunks`, `IndexedUpdateRows`, `Fill` and `Cast`.
- `rewrite.cpp` matches the two deinterleave opcodes in the QKV fusion and split
  peepholes.
- `bindings.cpp` knows the index range `H3AdaLNSelect` admits, as it does for the
  other index-taking opcodes.

Building the QKV rewrites made the shape of the problem clear:

- **`H3DeinterleaveQkv` / `H3DeinterleaveQkvWeight` should become one general
  primitive.** Their content is a strided de-interleave of a packed axis laid out
  as `[outer][groups][inner]` into `groups` separate `[outer][inner]` values.
  Nothing about that is H3. The fusion and split rules are entirely
  layout-generic; only the opcode names are model-specific. A general
  `DeinterleaveGroups` with `Groups`, `Outer`, `Inner` and an axis attribute
  would subsume both the weight-side (rank-2, axis 0) and activation-side
  (rank-2 to rank-3, axis 1) forms, and would immediately make
  `fuse_qkv_projection` / `split_qkv_projection` apply to every packed-QKV
  transformer. This is the recommended next IR change.
- **`H3AdaLNSelect` should become frontend sugar**, not a new primitive. It is a
  broadcast row gather followed by a six-way chunk select, and both `GatherRows`
  and `SelectRowChunks` already exist. What blocks the decomposition today is not
  an H3 concept but a missing general one: there is no reshape or view operation,
  and the AdaLN table is `[T, 18H]` while the select indexes rows of a virtual
  `[3T, 6H]`. The general primitive worth adding is `ReshapeView`; once it
  exists, `H3AdaLNSelect` is a lowering choice rather than a semantic one, and
  the optimizer can fold and CSE its pieces.
- **No new model-specific opcode was added for this work**, and none is
  recommended. The two rewrites above are the only model-shaped knowledge in the
  layer, and both dissolve into general primitives under the changes described.

## Running it

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure     # includes dif_opt_tests
```

Reproduce the Phase-1 search, writing a plan, a journal, the optimized program
and a tuning-database record:

```sh
build/difopt --h3-denoiser --layers 1 --hidden 256 --heads 8 --head-dim 32 \
    --ffn 512 --rotary 24 --audio-input-dim 16 --synthetic-bindings 13 \
    --objective memory --warmups 1 --iterations 2 --depth 3 --beam 2 \
    --max-candidates 48 --no-memory --blocks 64,512 --quant-groups 16 \
    --memory-budget-mib 4 \
    --plan plan.json --journal journal.json --out optimized.difir --db tune.db
```

Rebuild the winner from the recorded plan in a separate process and check that it
reproduces:

```sh
build/difopt --h3-denoiser --layers 1 --hidden 256 --heads 8 --head-dim 32 \
    --ffn 512 --rotary 24 --audio-input-dim 16 --synthetic-bindings 13 \
    --replay plan.json --out replayed.difir
cmp optimized.difir replayed.difir
```

`difopt` also accepts `--program FILE` with `--bind ID=FILE` for a real weight
set, and `--backend cuda` where a device is present. Run it with no arguments for
the complete option list.
