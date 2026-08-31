# Flame->C++ runtime port: continuation handoff

Living document (final revision lands when the last Wave-3 lanes merge).
Audience: any future engineer/agent (Codex/OpenAI included) continuing this
work without reverse-engineering the swarm. Date: 2026-08-31.

## Where everything lives

- Integration branch: `flame-runtime-integration` (cut from main@305a321).
  NOT yet pushed (final step; see "Before pushing" below).
- The plan + status of record: docs/FLAME_CPP_RUNTIME_PORT.md (port matrix,
  wave results, milestones M1/M2 achieved, removal ledger).
- flame-core extraction (the porting bible): docs/FLAME_PORT_SOURCE_NOTES.md.
- Module map: docs/RUNTIME_MODULES.md. Experiments:
  docs/FLAME_PORT_EXPERIMENT_LEDGER.md. Build plans:
  docs/BIGVGAN_DECODE_PLAN.md, docs/QWEN3VL_CONDITIONER_PLAN.md.
- Gate records (each with bars + provenance + repro):
  MIXED_PRECISION_TRAINING_GATE, LORA_TRAINING_GATE, DIT_BACKWARD_GATE,
  M2_DIT_LORA_TRAINING_GATE, ELEMENTWISE_FUSION (all *_2026-08-31.md).
- Worktrees (agent lanes; remove after their branches merge):
  /home/alex/dc-* . Do NOT touch /home/alex/diffusion-compiler-phase2
  (phase2-real-h3 evidence) or the protected repos
  /home/alex/diffusion-judge, /home/alex/diffusion-fixtures.

## How to reproduce the two milestones

- M1 (inference preserved): rerun
  artifacts/h3-quality-natural-language-2026-08-30/compiler/run_full_exact_bf16.sh
  with build/difh3infer instead of build-main-phase3; the three latent
  outputs must be byte-identical to
  .../compiler/full_exact_bf16/{video-latent,audio-rows,audio-latent}.diftensor
  (SHAs 5a6b8e15... / 47495ca9... / e1b87749...). Wrap in
  scripts/mem_safe_runtime.sh; serialize GPU work with
  `flock /tmp/dc-gpu.lock -c '...'`.
- M2 (native training): `flock /tmp/dc-gpu.lock -c 'bash
  tools/run_dit_lora_gate.sh WORKDIR 2 100'` (and `4 100`).

## Standing rules (non-negotiable, learned the hard way)

1. Nothing is PROVEN because it compiles: reference semantics -> impl ->
   deterministic test -> GPU parity -> real dims -> measurement -> PROVEN.
2. Bars are set AFTER measurement, frozen with provenance, never lowered.
3. Byte-identity is the default gate for runtime/transport changes; numeric
   changes ship as fingerprinted candidate properties (difopt discipline) —
   execution-policy knobs enter candidate identity, never silent defaults.
4. Composition amplifies BF16 error: gate at depth (measured F32 2.4e-4 /
   BF16 ~1000x at 4 blocks); single-block parity is never depth parity.
5. GPU work is serialized machine-wide via flock /tmp/dc-gpu.lock; big runs
   wrapped in scripts/mem_safe_runtime.sh (host-OOM precedent).
6. artifacts/ and every rejected-experiment record are read-only evidence.

## Open work, in priority order (details in port doc §7)

1. **Qwen3-VL conditioner build — the ONLY remaining Mojo dependency in the
   generation path, and the last XL item.** Complete plan with op-by-op
   mapping, streaming strategy and gate design in
   QWEN3VL_CONDITIONER_PLAN.md. Its one IR dependency (GQA) is DONE and
   merged, so the build is unblocked: execute that plan's chunks. The
   native tokenizer already produces its input contract.
3. Matched-conditions H3-scale measurement of the Wave-1 staging wins
   (keep-pages/threads) — expected to grow on the 61.7 GiB >RAM case.
4. cuDNN SDPA backward (decomposed AttentionBackward + fixtures are the
   ready-made parity reference; honor flame's d_o-cast/saved-O and
   128-alignment traps — FLAME_PORT_SOURCE_NOTES §2).
5. IR generality (Transpose/Concat/GELU/DeinterleaveGroups/ReshapeView,
   batched+masked Attention), grad clip, EMA (LinearBlend wiring), CUDA
   graphs, device-resident optimizer loop, gradient checkpointing,
   planner lifetime-remap for the flagged fused-plan write-early hazard.
6. Block-1 BF16 divergence vs Serenity (parity thread, tooling ready:
   first_eval_trace + --max-evaluations/--first-eval-input-dir/
   --h3-modulation-total-layers; next experiments recorded in
   H3_NATURAL_LANGUAGE_QUALITY_GATE and the port doc §6).

## Build configuration (important)

cuDNN is vendored at `/home/alex/dif-vendor/cudnn` (NVIDIA's own
redistributable, copied out of the pip wheel tree so that no runtime
library resolves from a Python site-packages path). Configure with:

    cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release \
          -DDIF_CUDNN_ROOT=/home/alex/dif-vendor/cudnn

The directory is outside the repo and is not committed (1 GB of vendor
binaries); re-create it by copying `nvidia/cudnn/{lib,include}` from any
NVIDIA cuDNN 9 distribution.

## Pre-push checklist (state at handoff)

1. cuDNN vendored off the Python path — DONE; M1 byte-identity re-proved on
   the rebuilt binary (see port doc §M1).
2. Public-remote hygiene — checked: the branch adds no artifacts, logs, or
   weight files, and a secrets/PII scan of the full diff is clean.
3. Final full ctest (11 suites) and the final SHA are recorded in the port
   doc and below.
