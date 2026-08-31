# Flame->C++ runtime port: accepted / rejected experiment ledger

Living document; updated at each integration. Date: 2026-08-31. Every entry
names its evidence. Rejected experiments are preserved evidence — never clean.

## Accepted (merged into flame-runtime-integration, each gated)

| Experiment | Evidence | Record |
|---|---|---|
| Launch/memcpy/host-stall telemetry | byte-identical vs pristine 5eb5f13 binary; silent without --profile-pipeline | w1-runtime report |
| Streamed keep-mapped-pages (opt-in) | -34%/iter, host-stage -48%, sha identical | w1-runtime report |
| Staging ring + prefetch depth + threaded copy (opt-in) | combo -43%/iter (458.6->265.4 ms), sha identical x5 | w1-runtime report |
| Pinned I/O (opt-in) | identity-proven; perf hypothesis at gate scale | w1-runtime report |
| W8A8 determinism fix (fenced resident promotion) | cross-mode convergence: 12 runs x 6 configs -> one SHA == resident result; production graphs unaffected | merge 1029b50 |
| Device arena (single allocation) | prepare allocs 13-16 -> 1; byte-identity across configs; live on production H3 (M1: device_mem_allocs=1) | merge 79270b5 |
| BF16/F32-accumulate training ops + Cast backward + AdamW dtype split | recorded F32 gates bit-identical; BF16 state BIT-IDENTICAL to torch 100-step CPU+CUDA | MIXED_PRECISION_TRAINING_GATE_2026-08-31.md |
| F32 LoRA vertical (.alpha-guarded export) | 70/70 torch comparisons; byte-identical resume | LORA_TRAINING_GATE_2026-08-31.md |
| DiT backward opcodes 39-46 | FD gradchecks; 104/104 torch fixtures; composed 2/4-block gates 322/322+642/642 | DIT_BACKWARD_GATE_2026-08-31.md |
| M2: BF16 LoRA on DiT blocks | 248/248+488/488; base bits byte-unchanged; resume byte-identical; three-way-symmetric spread proves intrinsic BF16 separation | M2_DIT_LORA_TRAINING_GATE_2026-08-31.md |
| Elementwise region fuser (fingerprinted, default-off) | byte-identical incl. FMA-hazard defeat (__fmul_rn); 4.9x real chain; default proven inert | ELEMENTWISE_FUSION_2026-08-31.md |
| Native tokenizer | golden 439 ids exact; 0/3564 oracle mismatches | w3-conditioner merge db56584 |
| Native modcache builder | whole-file SHA == recorded Serenity cache (0/184,267,776); runtime-accepted in bounded run | difmodcache commit body + integrator repro |
| Native seeded noise | seeds 4242/4243 byte-identical payloads | difh3noise commit body + integrator repro |
| Native importer (difimport) | byte-identical to Python tool on recorded evidence | difimport commit body |
| BigVGAN opcodes 47/48 + CPU reference + fold importer + builder | fold BIT-EXACT vs f64 torch ref 779/779; stage parity on real rows >2.5x margin | w3-audio merge f4f3971 |
| M1: golden H3 rerun on consolidated runtime | ALL THREE latents byte-identical to accepted artifact | scratch m1_rerun/run.log + port doc |
| GQA (KvHeads) on the Attention family | 184/184 torch comparisons incl. conditioner 64/8; cuDNN grouped descriptors; fingerprint-stable when absent | GQA_ATTENTION_GATE_2026-08-31.md |
| cuBLASLt bias-epilogue absorption (default-off knob) | F32 byte-identical; launches drop by site count; BF16 0.125 under frozen 0.25 bar | EPILOGUE_PERSISTENCE_2026-08-31.md |
| cuBLASLt heuristic persistence (default-off knob) | inert when off; 2nd prepare restores all plans, rejected=0; byte-identical outputs | EPILOGUE_PERSISTENCE_2026-08-31.md |
| Native BigVGAN audio decode | header byte-identical, max int16 delta 1, 0.079% differing, SNR 86.87 dB, 0.63 s vs 3.27 s bar | H3_AUDIO_DECODE_GATE_2026-08-31.md |
| cuDNN vendored out of the Python tree | binary link set contains no site-packages path; M1 re-proved on it | port doc §5 |

## Rejected / declined (with cause; keep the evidence)

| Experiment | Why rejected | Record |
|---|---|---|
| Enabling W8A8 tail-copy-stream by default | deterministic now, but no measured perf win at gate scale | w1-runtime W2 report |
| asm-barrier and mul-by-1.0 as cheaper fused-rounding barriers | ptxas contracts/eliminates them — measured byte-identity FAILURES | ELEMENTWISE_FUSION doc |
| RmsNorm-headed fused regions (this tier) | anchor-geometry mismatch costs O(cols)/thread; needs executor geometry override | ELEMENTWISE_FUSION doc |
| Dying-input region fusion without contiguity guarantees | slot-alias hazard vs best-fit planner | ELEMENTWISE_FUSION doc |
| Fusing frozen-base dW emission via opt-pass wiring | cheaper fixed at differentiate() emission site | w1-lora report |
| Tight end-state BF16 direction parity at depth | measured impossible under reduction-order freedom (three-way symmetric spread) | M2 gate doc |
| torch.optim.AdamW as BF16 oracle | allocates BF16 moments for BF16 params — wrong semantics; manual receipt used | MIXED_PRECISION gate doc |
| bare tokenizer.json as the H3 tokenizer | misses 7 config-only special tokens -> 440 not 439 tokens | w3-conditioner report |
| Device-side zero-insert+unflipped-conv transposed-conv shortcut (BigVGAN) | only valid for symmetric filters; real transposed conv implemented | BIGVGAN_DECODE_PLAN.md |
| Cold-vs-warm H3 perf comparison for M1 | unmatched conditions (132 vs 22.4 s/eval) — recorded as NOT comparable | port doc M1 note |
| cuBLASLt activation (RELU/GELU) epilogues | cuBLASLt offers no SiLU/SwiGLU; DiffIR has no GELU opcode; no matchable pattern exists — would be dead code | EPILOGUE_PERSISTENCE_2026-08-31.md |
| Widening the F32 absolute bar globally for GQA | rejected in favour of a shape-scoped bar with measured provenance (CUDA arm of the same case straddles the small-shape bar) | GQA_ATTENTION_GATE_2026-08-31.md |
| Whole-file byte identity as the audio bar | different executor/reduction order cannot reproduce another implementation's F32 summation order; int16-level + staged torch parity used instead | H3_AUDIO_DECODE_GATE_2026-08-31.md |

## Pre-existing recorded rejections (inherited, preserved)

INT4 all variants; plain INT5 full-stack (cos 0.7726); calibrated INT5
full-stack; direct packed INT5 (speed) — see H3_LOWBIT_RESIDENCY_GATE and
the h3-layer1/stack int4/int5 artifact families. Failed placement-search
logs (empty/header-only) retained under optimization-full-h3-denoiser.
