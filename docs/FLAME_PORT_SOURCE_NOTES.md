# flame-core -> C++ runtime: portable-semantics source notes

Companion to FLAME_CPP_RUNTIME_PORT.md. This is the verified extraction of
flame-core's proven, portable semantics (kernels, backward equations, memory
policies, optimizer behavior, dtype contract, dispatch policies) with source
anchors, produced 2026-08-31. File:line refs are relative to
/home/alex/EriDiffusion/flame-core/ unless absolute. Port SEMANTICS, not Rust
architecture.

## 0. Ranked port-value table

| # | Item | Source | What it proves | Effort | C++ target area |
|---|------|--------|----------------|--------|-----------------|
| 1 | cuDNN v9 SDPA fwd-train/bwd shim + graph cache keyed on (shape, per-tensor strides, scale) + Stats(LSE) contract | src/cuda/cudnn_sdpa.cpp, cudnn_sdpa_bwd.cpp | 12.1x vs in-tree WMMA fwd; 30-50x vs decomposed bwd; parity cos>=0.999996 | M | cuDNN-SDPA executor (port dispatch gates + Stats layout + bail conditions, sec 7) |
| 2 | SDPA dispatch policy (head_dim gate -> cuDNN; causal FA2; Q-tiled materialized fallback; >=2G-elem stream route; NO silent fallthrough) | src/sdpa.rs:1110-1500, 1880-2139 | runs every DiT (d=64..256) at 1024^2 on 24 GB without OOM (commit 7b4e281) | M | attention dispatcher |
| 3 | Backward-equation set for DiT ops incl. dtype contract (matmul/linear via trans-flag GEMM, RMSNorm/LN, RoPE-with-layout-tag, GateResidual, SwiGLU, QKV-split) | src/autograd.rs arms (sec 2) | months of measured bug history baked into exact forms | M | autodiff op library |
| 4 | RoPE layout tag (Interleaved / Halfsplit / HalfsplitPytorch) carried on the op, never shape-sniffed; bwd = fwd(-sin) same layout | src/autograd.rs:150-190, 4967-5010 | HiDream-O1 Q/K LoRA-B grad collapse (cos ~0.01) root-caused to shape-sniffing | S | RoPE primitive + VJP |
| 5 | Sync contract: no host stalls in-step; inline kernel-arg metadata (<=4 KB) OR cached per-device workspace | SPEED_CONTRACT clause 1; cuda/narrow_strided.cu (NarrowMeta, commit b552f61); cuda/sdpa_stream_bf16.cu:197-345 | klein 9B: 268 host syncs/step -> 0 | S | every launch wrapper; memory planner |
| 6 | AdamW decoupled-WD receipt + fused single-pass kernels + multi-tensor 5-region packed buffer + stochastic-round BF16 store | src/adam.rs:1-51, 1145-1315 | LoRA-A "unlearning" runaway root-caused to L2-into-grad; bit-exact multi-vs-per-tensor parity gate | M | optimizer |
| 7 | cuBLASLt fused linear3d: BF16 in/out, F32 accum, BIAS epilogue; _native [Cout,Cin] TRANSA=T; _pytorch_parity bit-exact variant (6 documented cuBLASLt knob diffs) | src/cuda/fused_linear3d.cu:24,135,369 | byte-identical to at::cuda::blas::gemm_and_bias, <=1.4% overhead | S-M | cuBLASLt executor epilogue policy |
| 8 | Trans-flag GEMM backward: grads via cuBLASLt trans flags, zero materialized transposes; absorb rank-3 transpose views into flags | src/autograd.rs:3743-3835 (MatMul), :5354-5454 (Linear); src/ops/gemm_bf16.rs:11-85 | removed 2 full BF16 memcpys per matmul-bwd | S | GEMM VJP |
| 9 | Vectorized RMSNorm fwd/bwd + separate cross-row grad-weight reduction kernel | norm.rs kernels (:1368, :1522, :1644) | 13.5-16.1x fwd, 9.5-14.8x bwd, ~500x fewer atomicAdds (12.6M->25k) | M | RMSNorm/QK-norm kernel pair |
| 10 | LayerNorm bwd vec pair (dx kernel + tiled dgamma/dbeta cross-row reduction) | cuda/src/flame_norm_bf16.cu:40, :189 | 1.8-3.4x on production shapes | M | LayerNorm kernel pair |
| 11 | Ring allocator: two-cursor bidirectional slab ring (fwd grows end, bwd shrinks start, ceil16/floor16, lazy slab malloc, error-not-wrap on cursor meet) | docs/RING_ALLOC_DESIGN.md; src/ring_alloc/mod.rs | structural fix for bucketed-pool step-2 corruption under fwd/bwd+offload | M | memory planner (validate plans against this invariant) |
| 12 | BlockOffloader policy: pinned host, dedicated transfer stream(s), event-chained (never host sync), double-buffered prefetch, weights-resident-never-reuploaded | src/offload/mod.rs; src/activation_offload.rs:1-80; SPEED_CONTRACT clause 5 | A2 lesson: weights re-uploaded per linear = 73.7->42.65 GB H2D/step, -31% step time when made resident | M | weight/activation streaming |
| 13 | Activation offload: LIFO slot pool; push = done-event on default stream -> transfer stream waits -> DtoH; pull = HtoD same transfer stream -> ready-event -> default stream waits; keep-alive clone until pull; FP8-E4M3 compression on transfer stream | src/activation_offload.rs:1-80; src/cuda/fp8_quant.cu | zero host syncs; halves PCIe bytes | M | activation spill |
| 14 | SDPA-bwd numerical traps: d_o F32->BF16 cast safe ONLY with correct saved-O identity (direct id, not shape-find); 128-token seq alignment; offset-view misalignment crash; padded-recompute softmax contamination | src/autograd.rs:1081-1230, 6458-6665 | grad_norm=inf and CUDA_ERROR_MISALIGNED_ADDRESS incidents each root-caused | S | attention VJP guards |
| 15 | LoRA contract: F32 storage / Kaiming-A / zero-B, cast-to-BF16 compute, scale=alpha/rank in delta, export MUST write .alpha | EriDiffusion-v2/crates/eridiffusion-core/src/lora.rs; tests/lora_alpha_export.rs | missing .alpha -> loader scale=1.0 -> ~16x over-application at inference | S | adapter layer |
| 16 | Fused DiT pointwise family: modulate (1+scale)*x+shift, RMSNorm+modulate, gate-residual, SwiGLU (with PyTorch-parity BF16 round point), GeGLU, patchify | src/cuda/fused_{modulate,norm_modulate,residual_gate,rms_norm}.cu; bf16_ops.rs; ops/fused_swiglu.rs | swiglu 5.1x; (w+1) pre-add trick 5000x on MagiHuman | S each | NVRTC kernel set |
| 17 | Adam8bit: bnb-0.49.2-parity blockwise dynamic-LUT (256-elem blocks, signed/unsigned qmaps, absmax, single fused dequant+step+requant kernel) | src/adam8bit_kernel.rs:1-130 | bit-exact-equivalent to bitsandbytes AdamW8bit | M | 8-bit optimizer state |
| 18 | Torch-compat RNG: bit-exact torch.randn/rand/bernoulli/randint (Philox4x32-10, exact grid policy) | src/rng/torch_compat.rs | fixture-tested bit parity -> enables bit-exact parity harnesses | M | test/parity infra |
| 19 | Grouped-MM + fused gated scatter-add (MoE) | src/cuda/grouped_mm.cu; fused_gated_scatter_add.cu | 845 GB/s scatter | M | MoE (if in scope) |
| 20 | Multi-tensor L2-norm (2-stage) + inplace-scale foreach primitives | src/ops/multi_tensor.rs | 2N launches -> 3; grad-clip path | S | optimizer utilities |
| 21 | Stride-hazard discipline: every backward kernel reading saved-tensor bytes consumes a CONTIGUIFIED view; backward grads must be dense | FLAME_AUTOGRAD_INTERNALS "Bug #4"; autograd.rs:5563-5575 | cos~0.05 direction-wrong grads with correct magnitude, twice | S (rule) | autodiff engine invariant |
| 22 | Leading-axis slice fast path: axis-0 slice of contiguous = one cudaMemcpyAsync | cuda/bf16_slice_index.cu:190 | 3.4-4x on 440-3206 calls/step | S | view/copy lowering |
| 23 | Bilinear upsample: F32-accumulate 4 taps, round on store | cuda/upsample_bilinear.cu | "don't accumulate 4 taps in a 7-bit mantissa" | S | VAE path |

## 1. Kernels

Two pipelines: NVRTC string kernels in .rs files (maps 1:1 to the NVRTC executor) and build-time .cu (cuBLASLt/cuDNN shims, WMMA, conv).

- RMSNorm trio (norm.rs): rms_norm_forward_bf16_vec: block/row, 256 threads, bf16x4 8-byte loads, warp-shuffle + smem inter-warp reduce, requires norm_size%4==0; rms_norm_backward_bf16_vec writes dx only; rms_norm_grad_weight_bf16_vec is a separate cross-row tile reduction (COLS 64 x ROWS 512, one atomicAdd per (col,row-tile)). The SPLIT dx-kernel / grad-weight-kernel design is the transferable idea; inline atomicAdd of dgamma was the 12.6M-atomics bug. Forward saves inv_rms so bwd skips the reduction.
- LayerNorm pair (cuda/src/flame_norm_bf16.cu): same split. Bwd recomputes mean/var inline; the compiler can instead materialize mean/rstd from forward (the F32-precision escape at autograd.rs:5230-5270 relies on saved F32 mean/rstd).
- AdaLN/modulate family: fused_modulate.cu ((1+scale)*x+shift), fused_norm_modulate.cu (RMSNorm->modulate one kernel), fused_residual_gate.cu (out = x + gate*attn), kernels/adaln_layernorm_bf16.cu (NHWC affine LN). For (weight+1) RMSNorm, pre-add 1.0 at load time (documented 5000x pathological win).
- RoPE: bf16_ops.rs:343 interleaved (out[2i]=x[2i]cos-x[2i+1]sin; FLUX/Klein/Chroma/Wan), :376 halfsplit (HF rotate_half; Qwen/LLaMA), rope_halfsplit_bf16_pytorch (preserves PyTorch BF16 rounding points), rope_fused_bf16_f32pe (F32 tables). Contract: x [B,H,N,D] BF16, cos/sin [1,1,N,D/2]. Trap 1 partial rotation: kernels rotate the FULL last dim; prefix-rotation models must narrow->rotate->cat. Trap 2: BF16 cos/sin tables are a latent precision floor - make F32 PE tables the main entry in C++.
- SwiGLU: swiglu_fused_bf16_vec2 (bf16x2, F32 sigmoid) with deliberate BF16 round on silu(gate) between sigmoid and multiply to match PyTorch eager bit-for-bit. Backward kernels/swiglu_backward.cu emits d_gate+d_up in one kernel.
- GELU: tanh-approx default; gelu_exact (erf) exists because bare nn.GELU() set a 0.02%/block parity ceiling on Cosmos. Keep both; never flip a model post-training.
- Softmax last-dim (bf16_elementwise.rs:472): 2-pass online softmax, warp-shuffle; 1.5x PT. Non-last-dim 5-step pipeline is a wart - don't copy.
- Elementwise: vectorized flat path gated on same-shape AND both-contiguous (bf16_elementwise.rs:721 - flat kernel on permuted views reads storage-linear garbage). ATen-shaped answer for generic broadcast: TensorIterator port (src/cuda/tensor_iterator.cuh, OffsetCalculator <= rank 6).
- Reductions: sum_bf16_to_f32_scalar_kernel - grid-stride, per-thread F32 accumulator, smem tree, atomicAdd block partial into F32 scalar, fused *scale so mean never syncs to host. Known offender: sum_dim_keepdim_bf16 one-thread-per-output (134x slower than PT); fix = block-per-output-row + warp shuffle. Legacy F32 sum_kernel silently dropped elements past 262144 (grid cap without grid-stride = silent wrong answers).
- Permute family (cuda/permute0213.cu): vectorized 0213 (bf16x4, no divmod, 1.5-1.8x PT), tiled 021 (32x33 smem anti-bank-conflict), rank-2 [1,0] entry (3.4x), [0,1,3,2] collapsed to rank-3. Port the dispatch table.
- GEMM: cuda/gemm_bf16_fp32acc.cu strided-batched BF16/F32-acc with per-shape algo cache keyed (m,n,k,ld*,stride*,batch,opA,opB) + persistent workspace. fused_linear3d.cu per row 7. All CUBLAS_COMPUTE_32F + CUBLASLT_EPILOGUE_BIAS.
- Casts/quant: FP8 E4M3 quant/dequant (clamp +-448, RTN), fused dequant+transpose, MXFP4->BF16 (HF-parity LUT), stochastic-round F32->BF16 (keep hi-16, increment w.p. lo16/2^16).
- Measured baseline (3090 Ti vs PT 2.8): 14/17 hot ops within 1.5x of PT, 10 faster; overall 474/484 ops faster, median 0.6x. Catastrophic outliers: sdpa_stream_bf16 causal-d64 (do not port math) and sum_dim_keepdim.

## 2. Backward equations + traps (src/autograd.rs, ~66 Op variants at :192-620)

Port the ARMS, not the tape machinery. Grad policy: accumulate internally F32; each BF16 arm casts incoming grad F32->BF16 (kernels re-accumulate F32 internally).

- MatMul (:3743): d_lhs = g @ rhs^T, d_rhs = lhs^T @ g via cuBLASLt trans flags - zero materialized transposes. FLAME_BWD_F32 escape exists because cumulative BF16 GEMM-boundary rounding was PROVEN to rotate Klein's composed 32-block backward (per-block bwd cos 0.51-0.68 while fwd cos 0.99). F32 training autodiff is immune; a BF16-grad mode must know this failure is compositional and invisible at single-block scale.
- Linear (:5354): d_x = g @ W (no trans), d_W = g^T @ x (lands [out,in] directly), d_b = sum over all-but-last dims; leading dims flattened; one fused GEMM per grad.
- BatchMatMul (:5455): trans-flag pattern; rank-3 pure-transpose VIEWS absorbed into flags (stride pattern check flips the flag instead of copying, ops/gemm_bf16.rs:44).
- SDPA backward (:6458-6665): three-tier. (a) cuDNN bwd bails on: kill-switch env, causal (unless opted), any mask, missing saved O/Stats, rank!=4, GQA, d not in {64,96,128}, Nq/Nkv not 128-aligned (64-but-not-128 "succeeds" then MISALIGNED_ADDRESS), non-BF16, any of 6 base pointers not 16-byte aligned. d_o F32->BF16 cast is safe ONLY because saved-O is found by direct TensorId (the old shape-find heuristic picked Q as O and destroyed the dO.O^T identity -> grad_norm=inf). Stats contract: contiguous FP32 [B*H,N_q] == [B,H,N_q,1] strides [H*Nq,Nq,1,1]. (b) HD128 shim. (c) decomposed recompute (:1715): dV = P^T dO; dP = dO V^T; dS = (dP - rowsum(dP.P)).P; scale; dQ = dS K; dK = dS^T Q; mask/causal additive -1e9. Padding trap: if fwd zero-padded for cuDNN, recompute on padded K gives wrong softmax - slice to real lengths, recompute, zero-pad grads back.
- RMSNorm (:5303): fwd saves input + inv_rms; bwd = split vec kernels; weight grad only when required.
- LayerNorm (:5136): saved [input, w?, b?, mean, rstd]. F32 escape recomputes dx = rstd*(g - mean(g) - xhat*mean(g*xhat)) from saved F32 mean/rstd - required for the NON-AFFINE LN of modulate_pre (BF16 cancellation amplifies rounding ~sigma-x). For F32 autodiff: always keep mean/rstd from forward and use this form.
- RoPePrecomputed (:4967): bwd = same fused kernel with (cos, -sin) SELECTED BY THE layout TAG recorded at forward. Never shape-sniff.
- GateResidual (:4924): d_res = g; d_x = g*gate[:,None,:]; d_gate = sum_N (g*x).
- FusedSwiGLU (:4679): one kernel -> d_gate = dsilu(gate)*up*g, d_up = silu(gate)*g; split variant returns one packed [gate|up] grad.
- QkvSplitPermute (:4825): each of Q/K/V's grad scatters its slice back into one [B,N,3HD] grad via dedicated per-part kernel.
- Narrow/Slice (:6013): zeros of input shape + scatter-assign per axis; dtype-agnostic byte copy (no BF16->F32 detour); inline NarrowMeta struct by value = the no-sync pattern; tests/narrow_sync_microbench.rs is the regression-gate template.
- Cat (:5984): slice grad per input at running offset. Split (:6079) is KNOWN-WEAK (doesn't track which output) - do it properly in C++.
- Permute (:5563): inverse-permute then .contiguous() - strided grads handed downstream misread (proven sign-flip). Rule: backward grads are always dense.
- Broadcast (:5526): sum over left-padded + size-1 axes descending, reshape to src (broadcast_to used to silently detach grads - dead-LoRA incident).
- Unary activations: fused *_backward_{bf16,f32}, common ABI (grad_out, input, grad_in, n, stream); GELU bwd = tanh-approx derivative matching fwd.
- Checkpointing: block-granular saves only block I/O (~2 tensors/block vs ~80).
- Engine invariants: frozen-param grads filtered by needed-set BEFORE accumulation (-5 GB); every saved tensor a kernel reads is contiguified at fetch; CUDA-graph capture/replay of backward exists behind env flag (warmup->capture->replay, invalidate on tape-length change).

## 3. Memory/alloc policies

- cuda_alloc_pool.rs lesson: a bucketed free list has no fwd/bwd direction notion and structurally corrupts under bidirectional lifetimes + offload replay (step-2 crash class). The compile-time planner sidesteps this; treat the ring invariant as the correctness spec.
- ring_alloc (RING_ALLOC_DESIGN.md): one logical byte space over N fixed slabs; allocation_end grows forward (ceil-16), allocation_start retreats backward (floor-16); slab-boundary jump wastes tail; lazy cudaMalloc on first touch; NO per-allocation free - cursors reset at step boundary; cursor-meet ERRORS instead of silently wrapping; single alloc may not span slabs; direction-typed handles.
- static_slab_v2.rs: bump allocator, one big slab per (device,dtype), reset per step, external-range registry protects mid-slab view pointers.
- Workspace pattern (SPEED_CONTRACT clause 1): small fixed metadata -> by-value kernel-arg struct (<=4KB); variable/large -> per-device cached buffer grown monotonically (SdpaStreamWorkspace fused 11 sub-buffers into one 256-B-aligned alloc + cached cublasHandle; was 11 malloc + 11 free + cublasCreate per call).
- All H2D weight traffic pinned; cudaHostRegister to adopt existing memory.

## 4. Offload

- BlockOffloader: all block weights pinned at init; TWO GPU slots - compute on block N overlaps H2D of N+1 on dedicated transfer stream; two events per slot; FP8-pinned mode keeps raw FP8 in host RAM, GPU dequant in prepare (28->14 GB). Strategy layer: TwoSlot (default), Knapsack (value-based resident set), Adaptive (VRAM headroom); one-time PCIe profiling persisted JSON; auto-select 2x max_block < 0.3x free VRAM -> TwoSlot else Adaptive.
- Activation: LIFO slots matching backward's reverse consumption; push = done-event on default stream -> transfer stream waits -> DtoH; pull = HtoD same transfer stream -> ready-event -> default stream waits; keep-alive clone until pull; FP8-E4M3 compression on transfer stream; grow-on-demand pinned cache superseded fixed slots for checkpointing.
- A1/A2 lessons (measured): frozen weights RESIDENT instead of re-uploaded per linear = 73.7->42.65 GB H2D/step, -31% step; skip d_W for frozen base weights on the LoRA path = -12 GB D2H. Encode both first-class: weight residency table + requires_grad-pruned wgrad.
- Anti-overbuild lesson: coordinator/ring/fraction offload phases 3-5 were DELETED as redundant once measured.
- Clause-5 checklist: pinned, async on transfer stream, event-chained, double-buffered, no re-upload, batched named D2H.

## 5. Optimizers

- AdamW receipt (src/adam.rs:1-51): m = b1*m+(1-b1)g; v = b2*v+(1-b2)g^2; p -= lr*mhat/(sqrt(vhat)+eps); THEN p -= lr*wd*p (decoupled). NEVER fold wd into grad pre-moments: with adaptive normalization that collapses small-signal params (fresh LoRA-A with zero B) to lr*sign(p) shrinkage - measured LoRA-A L2 50->0.85 @ step 400 ("unlearning" runaway).
- Kernel matrix: per-tensor {BF16p/BF16g, BF16p/F32g, F32p/F32g, F32p/BF16g}, all F32 m/v, single pass in-place. Multi-tensor: one launch for all params, 5-region packed u64 buffer [params|grads|ms|vs|sizes], one block/tensor + grid-stride, meta buffer cached; bit-exact parity gate vs per-tensor (tests/adam_multi_tensor_parity.rs). Stochastic rounding at BF16 param store: splitmix64 keyed (seed=step, idx). Bias corrections host-computed per step, passed as scalars.
- Adam8bit (src/adam8bit_kernel.rs): bnb 0.49.2 blockwise parity - u8 m/v codes + per-256-block F32 absmax x2 + two shared 256-entry dynamic LUTs; one fused dequant+step+requant kernel; requant by LINEAR-SCAN argmin over LUT; tail blocks contribute 0 to absmax; all-zero block absmax floors 1e-12.

## 6. LoRA

- Live contract = EriDiffusion-v2/crates/eridiffusion-core/src/lora.rs (flame-core src/lora.rs is legacy - do not port): A [rank,in] Kaiming-uniform (bound=1/sqrt(in)), B [out,rank] zeros, both STORED F32, forward delta = (x @ A^T @ B^T)*(alpha/rank), params cast to BF16 per call through autograd-aware cast, low-rank projection stays EXPLICIT (never materialize dense delta; the fused variant OOMed). Call: out = base_linear(x) + adapter.forward_delta(x). Backward = the two matmul VJPs.
- Export contract (burned lesson): serialize .lora_A.weight, .lora_B.weight AND .alpha (scalar). Missing .alpha -> loader scale=1.0 -> ~16x over-application at inference (2026-05-27 all-models runaway). alpha = export metadata, NOT trainable. Regression test: tests/lora_alpha_export.rs.

## 7. SDPA dispatch policy (src/sdpa.rs live; attention/sdpa.rs legacy)

1. Autograd-recording + requires_grad -> training path automatically (forgetting this silently broke every attention-LoRA gradient chain once).
2. Training fwd: zero-pad Q/K/V seq to 128 when eligible (unmasked, BF16, d in {64,96,128}); cuDNN train-forward emits O + Stats(LSE); record ONE fused attention node saving Q,K,V,O-by-id,Stats + real lengths; slice output back.
3. Inference fwd: d in {64,96,128} unmasked -> cuDNN (errors SURFACE, no silent fallthrough); causal same-length -> in-tree FA2 WMMA; GQA via HD128 shim; everything else (d=256, masks, odd dims) -> materialized fallback with Q-tiling; > 2e9 score elements -> chunked streaming route (replace the math, keep the routing threshold).
4. Materialized fallback (:1880-2139, most portable piece): per Q-tile - Q@K^T via cuBLASLt into BF16 logits -> upcast F32 -> scale -> additive -1e9 mask/causal -> F32 softmax -> downcast BF16 -> P@V. Tile size: BH*Qtile*K <= budget (default 256 MiB F32 elems; masked paths /8). Bit-identical to single-shot (row-independent softmax).
5. Mask semantics: binary keep-mask [B|1,H|1,Q,K] (>=0.5 attend) applied additive -inf; additive-bias entry F32 end-to-end; sink-column softmax variant. mask+causal combined = unsupported everywhere.
6. The streaming kernel silently produces garbage at d>128 (should hard-error) - do not reproduce.

## 8. Dtype contract

Storage BF16 end-to-end; F32 only as opmath inside kernels and optimizer master state - never a storage round-trip a caller can accidentally take. Burn list: Tensor::sum BF16->F32->reduce->F32->BF16 with no fast alternative (TENETS sec 2); F32-grad-storage cost ~19k cast launches/step (BF16-grad mode should be BF16-in/out with F32 internal accumulation); fallback SDPA must upcast QK^T to F32 for scale/mask/softmax (PT math kernel does); bias grads and every cross-element reduction need F32 accumulators; COMPOSITION AMPLIFIES BF16 ERROR - single-block backward parity >=0.999 does NOT imply full-depth parity (Klein 32-block self-consistency ratio 0.67) - parity-gate at full depth. cuBLASLt: CUBLAS_COMPUTE_32F with BF16 I/O everywhere.

## 9. Do not port

Autograd generations (autograd_v2/, autograd_v3.rs, autograd_v4/, autograd_simple/engine/ops/ops_complete/debug); attention legacies (attention/sdpa.rs module, sdpa_legacy.rs, sage_attention.rs, streaming_attn_bf16.cu, old flash stub, sdpa_stream_bf16.cu math - keep only its cached-workspace pattern); tensor legacies (cuda_tensor*.rs, CudaKernels F32 surface, devtensor.rs); memory_pool.rs and the default-off free-list caching mode; dead narrow_strided*.cu duplicates (contain the pre-fix sync anti-pattern); flame-core/src/lora.rs; Rust-workaround patterns (CudaSliceMirror transmute, dual saved-ref paths, global-mutex-held-backward, Op::Split's accumulate-everything backward, broken conv1d k=1 fast path, non-last-dim softmax pipeline). No Flash Attention 2/3 port exists anywhere - do not plan around one.

Cross-cutting TENETS: fix cost in the primitive, never per-model; the single spelling of a primitive must be fast at the storage dtype; measurement gates every perf claim; each primitive fix leaves a microbench regression gate behind (narrow_sync_microbench.rs is the template).
