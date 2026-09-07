# fastload_x86.S -- the CPU byte loops of the checkpoint loader, AVX2 + F16C.
#
# These are the only places the loader touches bytes on the CPU: dtype
# conversion for checkpoints that are not stored in BF16, and a streaming
# copy that keeps a multi-gigabyte pass from evicting the caches. BF16
# checkpoints never enter these loops; their bytes go disk -> pinned -> GPU
# by DMA.
#
# Target: x86-64 with AVX2 and F16C (Zen 3 here; no AVX-512 on this part).
# System V ABI. All three are C-callable:
#
#   void serenity_f32_to_bf16_rne(const float *src, uint16_t *dst, uint64_t n);
#   void serenity_f16_to_bf16_rne(const uint16_t *src, uint16_t *dst, uint64_t n);
#   void serenity_nt_memcpy(void *dst, const void *src, uint64_t nbytes);
#
# Rounding is round-to-nearest-even on the 16 dropped mantissa bits, the
# same rule torch uses for float -> bfloat16, and every NaN becomes the
# canonical quiet NaN 0x7FC0 (sign dropped), also torch's choice. Infinities,
# zeros and denormals round like any other value. The reference these are
# gated against is the scalar C loop in fastload_test.cpp.

        .text

# ---------------------------------------------------------------------------
# Constants (32 bytes each so a single vbroadcast-free vmovdqa serves a ymm).
        .section .rodata
        .p2align 5
c_one:      .long 1,1,1,1,1,1,1,1
c_bias:     .long 0x7FFF,0x7FFF,0x7FFF,0x7FFF,0x7FFF,0x7FFF,0x7FFF,0x7FFF
c_absmask:  .long 0x7FFFFFFF,0x7FFFFFFF,0x7FFFFFFF,0x7FFFFFFF,0x7FFFFFFF,0x7FFFFFFF,0x7FFFFFFF,0x7FFFFFFF
c_inf:      .long 0x7F800000,0x7F800000,0x7F800000,0x7F800000,0x7F800000,0x7F800000,0x7F800000,0x7F800000
c_qnan:     .long 0x7FC0,0x7FC0,0x7FC0,0x7FC0,0x7FC0,0x7FC0,0x7FC0,0x7FC0
# vpshufb control: keep bytes 0,1 / 4,5 / 8,9 / 12,13 of each 128-bit lane
# in the lane's low 8 bytes (the high 8 bytes are don't-care, zeroed here).
c_pack16:   .byte 0,1,4,5,8,9,12,13, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80
            .byte 0,1,4,5,8,9,12,13, 0x80,0x80,0x80,0x80,0x80,0x80,0x80,0x80

        .text

# ---------------------------------------------------------------------------
# Shared tail: ymm0 holds 8 x f32 (as bit patterns). Produces 8 x bf16 in xmm0
# (low 128 bits, packed). Clobbers ymm1-ymm4. Constants are loaded by the
# caller into ymm8-ymm12 once per call:
#   ymm8 = c_one, ymm9 = c_bias, ymm10 = c_absmask, ymm11 = c_inf,
#   ymm12 = c_qnan (as 32-bit lanes), ymm13 = c_pack16
        .macro ROUND8_TO_BF16
        vpsrld   $16, %ymm0, %ymm1        # lsb of the kept mantissa
        vpand    %ymm8, %ymm1, %ymm1
        vpaddd   %ymm9, %ymm1, %ymm1      # 0x7FFF + lsb
        vpaddd   %ymm1, %ymm0, %ymm2      # x + bias
        vpsrld   $16, %ymm2, %ymm2        # rounded bf16 in the low 16 bits
        vpand    %ymm10, %ymm0, %ymm3     # |x| bits
        vpcmpgtd %ymm11, %ymm3, %ymm3     # |x| > +inf  <=> NaN  (all-ones lanes)
        vblendvps %ymm3, %ymm12, %ymm2, %ymm2   # NaN lanes -> 0x7FC0
        vpshufb  %ymm13, %ymm2, %ymm2     # pack u32 -> u16 within each lane
        vpermq   $0x08, %ymm2, %ymm2      # lanes' low qwords -> xmm2[0:128]
        vmovdqa  %xmm2, %xmm0
        .endm

        .macro LOAD_CONSTS
        vmovdqa  c_one(%rip), %ymm8
        vmovdqa  c_bias(%rip), %ymm9
        vmovdqa  c_absmask(%rip), %ymm10
        vmovdqa  c_inf(%rip), %ymm11
        vmovdqa  c_qnan(%rip), %ymm12
        vmovdqa  c_pack16(%rip), %ymm13
        .endm

# ---------------------------------------------------------------------------
# void serenity_f32_to_bf16_rne(const float *src, uint16_t *dst, uint64_t n)
#   rdi = src, rsi = dst, rdx = n (element count)
        .globl  serenity_f32_to_bf16_rne
        .type   serenity_f32_to_bf16_rne, @function
        .p2align 4
serenity_f32_to_bf16_rne:
        test    %rdx, %rdx
        jz      .Lf32_done
        LOAD_CONSTS
        mov     %rdx, %rcx
        shr     $3, %rcx                  # full groups of 8
        jz      .Lf32_tail
.Lf32_loop:
        vmovups (%rdi), %ymm0
        ROUND8_TO_BF16
        vmovdqu %xmm0, (%rsi)
        add     $32, %rdi
        add     $16, %rsi
        dec     %rcx
        jnz     .Lf32_loop
.Lf32_tail:
        and     $7, %rdx
        jz      .Lf32_done
        # Scalar tail, same arithmetic, one element at a time.
.Lf32_tail_loop:
        mov     (%rdi), %eax
        mov     %eax, %r8d
        and     $0x7FFFFFFF, %r8d
        cmp     $0x7F800000, %r8d
        ja      .Lf32_tail_nan
        mov     %eax, %r9d
        shr     $16, %r9d
        and     $1, %r9d
        add     $0x7FFF, %r9d
        add     %r9d, %eax
        shr     $16, %eax
        jmp     .Lf32_tail_store
.Lf32_tail_nan:
        mov     $0x7FC0, %eax
.Lf32_tail_store:
        mov     %ax, (%rsi)
        add     $4, %rdi
        add     $2, %rsi
        dec     %rdx
        jnz     .Lf32_tail_loop
.Lf32_done:
        vzeroupper
        ret
        .size   serenity_f32_to_bf16_rne, .-serenity_f32_to_bf16_rne

# ---------------------------------------------------------------------------
# void serenity_f16_to_bf16_rne(const uint16_t *src, uint16_t *dst, uint64_t n)
#   rdi = src (IEEE half), rsi = dst, rdx = n
# f16 -> f32 is exact (vcvtph2ps), then the same RNE to bf16. A half NaN
# becomes 0x7FC0 like every other NaN.
        .globl  serenity_f16_to_bf16_rne
        .type   serenity_f16_to_bf16_rne, @function
        .p2align 4
serenity_f16_to_bf16_rne:
        test    %rdx, %rdx
        jz      .Lf16_done
        LOAD_CONSTS
        mov     %rdx, %rcx
        shr     $3, %rcx
        jz      .Lf16_tail
.Lf16_loop:
        vcvtph2ps (%rdi), %ymm0           # 8 halves -> 8 floats
        ROUND8_TO_BF16
        vmovdqu %xmm0, (%rsi)
        add     $16, %rdi
        add     $16, %rsi
        dec     %rcx
        jnz     .Lf16_loop
.Lf16_tail:
        and     $7, %rdx
        jz      .Lf16_done
        # Scalar tail through the vector unit one element at a time: convert a
        # single half in xmm0 lane 0, then round exactly as the vector path.
.Lf16_tail_loop:
        movzwl  (%rdi), %eax
        vmovd   %eax, %xmm0
        vcvtph2ps %xmm0, %xmm0
        vmovd   %xmm0, %eax
        mov     %eax, %r8d
        and     $0x7FFFFFFF, %r8d
        cmp     $0x7F800000, %r8d
        ja      .Lf16_tail_nan
        mov     %eax, %r9d
        shr     $16, %r9d
        and     $1, %r9d
        add     $0x7FFF, %r9d
        add     %r9d, %eax
        shr     $16, %eax
        jmp     .Lf16_tail_store
.Lf16_tail_nan:
        mov     $0x7FC0, %eax
.Lf16_tail_store:
        mov     %ax, (%rsi)
        add     $2, %rdi
        add     $2, %rsi
        dec     %rdx
        jnz     .Lf16_tail_loop
.Lf16_done:
        vzeroupper
        ret
        .size   serenity_f16_to_bf16_rne, .-serenity_f16_to_bf16_rne

# ---------------------------------------------------------------------------
# void serenity_nt_memcpy(void *dst, const void *src, uint64_t nbytes)
#   rdi = dst, rsi = src, rdx = nbytes
# Streaming copy: 128 bytes per iteration with non-temporal stores, so a
# multi-gigabyte pass does not wash the caches the kernels behind it need.
# Requires a 32-byte aligned dst for the vector body; an unaligned head is
# copied with rep movsb until dst is aligned, and the tail likewise. Source
# alignment is not required (vmovdqu loads).
        .globl  serenity_nt_memcpy
        .type   serenity_nt_memcpy, @function
        .p2align 4
serenity_nt_memcpy:
        cmp     $256, %rdx
        jb      .Lnt_small
        # Head: bytes until dst is 32-byte aligned.
        mov     %rdi, %rcx
        neg     %rcx
        and     $31, %rcx                 # (32 - dst%32) % 32
        jz      .Lnt_aligned
        sub     %rcx, %rdx
        rep movsb                          # rdi, rsi advance; rcx -> 0
.Lnt_aligned:
        mov     %rdx, %rcx
        shr     $7, %rcx                  # 128-byte blocks
        jz      .Lnt_tail
.Lnt_loop:
        vmovdqu   (%rsi), %ymm0
        vmovdqu 32(%rsi), %ymm1
        vmovdqu 64(%rsi), %ymm2
        vmovdqu 96(%rsi), %ymm3
        vmovntdq %ymm0,   (%rdi)
        vmovntdq %ymm1, 32(%rdi)
        vmovntdq %ymm2, 64(%rdi)
        vmovntdq %ymm3, 96(%rdi)
        add     $128, %rsi
        add     $128, %rdi
        dec     %rcx
        jnz     .Lnt_loop
        sfence
.Lnt_tail:
        and     $127, %rdx
        mov     %rdx, %rcx
        rep movsb
        vzeroupper
        ret
.Lnt_small:
        mov     %rdx, %rcx
        rep movsb
        ret
        .size   serenity_nt_memcpy, .-serenity_nt_memcpy

        .section .note.GNU-stack,"",@progbits
