# fastload_ring.S -- the checkpoint loader. GNU as, x86-64 System V.
#
#   int serenity_fastload_run(const struct serenity_fastload_request *req);
#
# The request carries two descriptors the caller already holds open on the
# file, one plain and one O_DIRECT (either may be -1; mode 0 then has only
# the other to choose from). The loader never opens or closes anything.
#
# Destinations: arena + dst_off. With arena == 0 (and arena_bytes == 0) each
# dst_off is an absolute device address, so a caller that already owns one
# device buffer per tensor can pass those buffers straight through. A span
# with host == 1 is written to HOST memory instead (arena + dst_off is then a
# host address), by the CPU from the pinned slot -- for weights the caller
# must transform on the host before they can go to the device.
#
# Pipeline: only page-aligned ranges covering requested spans are read, cut
# into at most 16 MiB chunks. Gaps between those ranges are never read.
# Up to 8 chunks are in flight as io_uring reads into 8 pinned slots.
# When a chunk's completion arrives, every span piece inside it is issued as
# a cudaMemcpyAsync straight from the slot to its arena offset (after an
# in-slot BF16 conversion when the span asks for one), and an event is
# recorded on the stream. A slot is reused only after that event has
# completed, so the GPU always drains a slot before the drive refills it.
# Nothing waits on anything it does not have to: reads, DMA and conversion
# overlap, and the only full stop is one cudaStreamSynchronize at the end.
#
# Reads are O_DIRECT (nothing enters the page cache, nothing is charged to
# the cgroup) unless the file is already resident, in which case plain reads
# copy out of the cache at memory speed. Mode 0 decides by mincore.
#
# The outside world: raw syscalls (lseek, mmap, munmap, mincore,
# clock_gettime, io_uring_setup, io_uring_enter, close of the ring) and six
# CUDA driver entry points through the PLT. Everything else is here.
#
# Gate: fastload_bench.cpp loads a real checkpoint through this and compares
# every device byte against a pread of the file.

        .set FL_C,      0x1000000        # 16 MiB per chunk
        .set FL_R,      8                # pinned slots, 128 MiB
        .set FL_SQ,     16               # io_uring entries
        .set PAGE,      4096

        .set SYS_close, 3
        .set SYS_sched_yield, 24
        .set SYS_lseek, 8
        .set SYS_mmap,  9
        .set SYS_munmap, 11
        .set SYS_mincore, 27
        .set SYS_clock_gettime, 228
        .set SYS_io_uring_setup, 425
        .set SYS_io_uring_enter, 426
        .set SEEK_END,  2
        .set PROT_RW,   3
        .set MAP_SHARED, 1
        .set MAP_PRIVATE_ANON, 0x22
        .set MAP_SHARED_POPULATE, 0x8001
        .set CLOCK_MONOTONIC, 1
        .set EINTR,     4
        .set IORING_OFF_SQ_RING, 0
        .set IORING_OFF_CQ_RING, 0x8000000
        .set IORING_OFF_SQES,    0x10000000
        .set IORING_ENTER_GETEVENTS, 1
        .set IORING_OP_READ, 22
        .set IOSQE_ASYNC, 0x10
        .set CU_EVENT_DISABLE_TIMING, 2

# struct io_uring_params
        .set P_SQ_ENTRIES, 0
        .set P_CQ_ENTRIES, 4
        .set P_SQ_TAIL,   44
        .set P_SQ_MASK,   48
        .set P_SQ_ARRAY,  64
        .set P_CQ_HEAD,   80
        .set P_CQ_TAIL,   84
        .set P_CQ_MASK,   88
        .set P_CQ_CQES,   100
        .set P_SIZE,      120

# struct serenity_fastload_span (32 bytes)
        .set SP_OFF,   0
        .set SP_N,     8
        .set SP_DST,   16
        .set SP_CONV,  24
        .set SP_HOST,  28              # u32: 0 device (DMA), 1 host (CPU copy)
        .set SP_SIZE,  32

# struct serenity_fastload_stats (64 bytes)
        .set STAT_BYTES,   0
        .set STAT_CHUNKS,  8
        .set STAT_MODE,    16
        .set STAT_ERR,     24
        .set STAT_NS,      32
        .set STAT_RESID,   40
        .set STAT_PAGES,   48

# struct serenity_fastload_request
        .set RQ_FD,        0             # int64 plain descriptor or -1
        .set RQ_FD_DIRECT, 8             # int64 O_DIRECT descriptor or -1
        .set RQ_SPANS,     16
        .set RQ_NSPANS,    24
        .set RQ_ARENA,     32
        .set RQ_ARENAB,    40
        .set RQ_STREAM,    48
        .set RQ_MODE,      56            # int64
        .set RQ_STATS,     64
        .set RQ_ERR,       72
        .set RQ_ERRCAP,    80

# loader state (lives in the run frame; helpers take a pointer in %rdi)
        .set ST_PATH,     0
        .set ST_SPANS,    8
        .set ST_NSPANS,   16
        .set ST_ARENA,    24
        .set ST_ARENAB,   32
        .set ST_STREAM,   40
        .set ST_MODE,     48
        .set ST_STATS,    56
        .set ST_ERR,      64
        .set ST_ERRCAP,   72
        .set ST_FD,       80
        .set ST_FSIZE,    88
        .set ST_LO,       96
        .set ST_HI,       104
        .set ST_NCHUNKS,  112
        .set ST_RINGFD,   120
        .set ST_SQRING,   128
        .set ST_SQRINGSZ, 136
        .set ST_CQRING,   144
        .set ST_CQRINGSZ, 152
        .set ST_SQES,     160
        .set ST_SQESSZ,   168
        .set ST_SQTAILP,  176
        .set ST_SQMASK,   184
        .set ST_SQARRAY,  192
        .set ST_SQTAIL,   200
        .set ST_CQHEADP,  208
        .set ST_CQTAILP,  216
        .set ST_CQMASK,   224
        .set ST_CQES,     232
        .set ST_CQHEAD,   240
        .set ST_FIRST,    248
        .set ST_FIRSTSZ,  256
        .set ST_NEXT,     264
        .set ST_INFLIGHT, 272
        .set ST_PENDING,  280
        .set ST_DONE,     288
        .set ST_BYTES,    296
        .set ST_NEEDCONV, 304
        .set ST_SLOTBUSY, 312          # 8 bytes, one per slot
        .set ST_T0,       320
        .set ST_DIRECT,   328
        .set ST_FD_PLAIN, 336
        .set ST_FD_DIRECT, 344
        .set ST_PARAMS,   352          # io_uring_params, 120 bytes
        .set ST_TS,       472          # timespec, 16 bytes
        .set ST_H2DCOPIES,488
        .set ST_NEEDDEVICE,496         # any span targets CUDA rather than host
        .set ST_SIZE,     512
        .set FRAME,       ST_SIZE + 8  # keeps %rsp 16-aligned after 6 pushes

        .bss
        .p2align 3
fl_ring:    .quad 0                    # pinned data slots, FL_R * FL_C
fl_conv:    .quad 0                    # pinned conversion slots, same size
fl_events:  .zero 8*FL_R
fl_lock:    .quad 0                    # process-wide: one load at a time

        .section .rodata
e_open:   .asciz "fastload: no usable file descriptor"
e_setup:  .asciz "fastload: io_uring_setup failed"
e_mmap:   .asciz "fastload: mmap failed"
e_read:   .asciz "fastload: read failed"
e_short:  .asciz "fastload: short read inside a span"
e_cuda:   .asciz "fastload: CUDA call failed"
e_spans:  .asciz "fastload: bad span table (unsorted, overlapping, misaligned, or bad convert)"
e_pinned: .asciz "fastload: pinned allocation failed"
e_arena:  .asciz "fastload: span outside the arena"
        .section .data.rel.ro,"aw"
        .p2align 3
e_table:  .quad e_open, e_setup, e_mmap, e_read, e_short, e_cuda, e_spans, e_pinned, e_arena

        .text

# ---------------------------------------------------------------------------
# uint64_t fl_now(void)  -- CLOCK_MONOTONIC in nanoseconds
        .p2align 4
fl_now:
        sub     $24, %rsp
        mov     $SYS_clock_gettime, %eax
        mov     $CLOCK_MONOTONIC, %edi
        mov     %rsp, %rsi
        syscall
        mov     (%rsp), %rax
        imul    $1000000000, %rax, %rax
        add     8(%rsp), %rax
        add     $24, %rsp
        ret

# ---------------------------------------------------------------------------
# void fl_fail(state *rdi, int64 code (rsi)) -- record the error text/code
        .p2align 4
fl_fail:
        mov     ST_STATS(%rdi), %rax
        test    %rax, %rax
        jz      1f
        mov     %rsi, STAT_ERR(%rax)
1:      mov     ST_ERR(%rdi), %r8            # destination
        test    %r8, %r8
        jz      9f
        mov     ST_ERRCAP(%rdi), %rcx
        test    %rcx, %rcx
        jz      9f
        mov     %rsi, %rax
        neg     %rax                          # code 1..9
        lea     e_table(%rip), %r9
        mov     -8(%r9,%rax,8), %r9           # message
        dec     %rcx                          # room for NUL
2:      test    %rcx, %rcx
        jz      3f
        movzbl  (%r9), %eax
        test    %al, %al
        jz      3f
        mov     %al, (%r8)
        inc     %r8
        inc     %r9
        dec     %rcx
        jmp     2b
3:      movb    $0, (%r8)
9:      ret

# ---------------------------------------------------------------------------
# int fl_submit(state *rdi, uint64 k (rsi))
#   0 submitted, 1 slot busy (caller must wait for completions), <0 error
        .p2align 4
fl_submit:
        push    %rbp
        mov     %rsp, %rbp
        push    %rbx
        push    %r12
        push    %r13
        push    %r14
        push    %r15
        sub     $8, %rsp
        mov     %rdi, %rbx                    # state
        mov     %rsi, %r12                    # k
        mov     %r12, %r13
        and     $(FL_R-1), %r13               # slot
        cmpb    $0, ST_SLOTBUSY(%rbx,%r13)
        jne     .Lsub_busy
        cmpq    $0, ST_NEEDDEVICE(%rbx)
        je      .Lsub_fresh
        cmp     $FL_R, %r12
        jb      .Lsub_fresh
        # The slot held chunk k-R: wait for the GPU to have drained it.
        lea     fl_events(%rip), %rax
        mov     (%rax,%r13,8), %rdi
        vzeroupper
        call    cuEventSynchronize@PLT
        test    %eax, %eax
        jnz     .Lsub_cuda
.Lsub_fresh:
        # Sparse chunk table: {file offset, read length, first span, pad}.
        mov     %r12, %rax
        shl     $5, %rax
        add     ST_FIRST(%rbx), %rax
        mov     (%rax), %r14
        mov     8(%rax), %r15
        # sqe
        mov     ST_SQTAIL(%rbx), %rax
        and     ST_SQMASK(%rbx), %rax         # idx
        mov     %rax, %rcx
        shl     $6, %rcx
        add     ST_SQES(%rbx), %rcx           # sqe = sqes + idx*64
        # zero the 64 bytes
        vpxor   %xmm0, %xmm0, %xmm0
        vmovdqu %xmm0, (%rcx)
        vmovdqu %xmm0, 16(%rcx)
        vmovdqu %xmm0, 32(%rcx)
        vmovdqu %xmm0, 48(%rcx)
        movb    $IORING_OP_READ, (%rcx)
        # Buffered reads of a resident file would otherwise run inline in
        # io_uring_enter on this one thread; IOSQE_ASYNC hands them to the
        # kernel's workers so the page-cache copies proceed in parallel.
        cmpq    $2, ST_DIRECT(%rbx)
        jne     2f
        movb    $IOSQE_ASYNC, 1(%rcx)
2:      mov     ST_FD(%rbx), %edx
        mov     %edx, 4(%rcx)                 # fd
        mov     %r14, 8(%rcx)                 # off
        mov     %r13, %rdx
        shl     $24, %rdx
        add     fl_ring(%rip), %rdx
        mov     %rdx, 16(%rcx)                # addr = slot buffer
        mov     %r15d, 24(%rcx)               # len
        mov     %r12, 32(%rcx)                # user_data = k
        # sq array + tail
        mov     ST_SQARRAY(%rbx), %rdx
        mov     %eax, (%rdx,%rax,4)
        mov     ST_SQTAIL(%rbx), %rax
        inc     %rax
        mov     %rax, ST_SQTAIL(%rbx)
        mov     ST_SQTAILP(%rbx), %rdx
        mov     %eax, (%rdx)                  # publish tail (x86 stores are ordered)
        movb    $1, ST_SLOTBUSY(%rbx,%r13)
        incq    ST_PENDING(%rbx)
        incq    ST_INFLIGHT(%rbx)
        incq    ST_NEXT(%rbx)
        xor     %eax, %eax
        jmp     .Lsub_out
.Lsub_busy:
        mov     $1, %eax
        jmp     .Lsub_out
.Lsub_cuda:
        mov     $-6, %rax
.Lsub_out:
        add     $8, %rsp
        pop     %r15
        pop     %r14
        pop     %r13
        pop     %r12
        pop     %rbx
        pop     %rbp
        ret

# ---------------------------------------------------------------------------
# int fl_process(state *rdi, uint64 k (rsi), int64 res (rdx))
#   Issue the DMA for every span piece inside chunk k, record the slot event.
        .p2align 4
fl_process:
        push    %rbp
        mov     %rsp, %rbp
        push    %rbx
        push    %r12
        push    %r13
        push    %r14
        push    %r15
        sub     $88, %rsp                     # locals reach -128(%rbp)
        .set L_CHUNK, -56                     # chunk_start
        .set L_VALID, -64                     # chunk_start + res
        .set L_SLOTBASE, -72                  # fl_ring + slot*C
        .set L_CONVBASE, -80                  # fl_conv + slot*C
        .set L_SLOT, -88
        .set L_END, -96                       # chunk_start + C
        mov     %rdi, %rbx                    # state
        mov     %rsi, %r12                    # k
        test    %rdx, %rdx
        js      .Lpr_readerr
        mov     %r12, %rax
        and     $(FL_R-1), %rax
        mov     %rax, L_SLOT(%rbp)
        mov     %rax, %rcx
        shl     $24, %rcx
        mov     fl_ring(%rip), %r8
        add     %rcx, %r8
        mov     %r8, L_SLOTBASE(%rbp)
        mov     fl_conv(%rip), %r8
        add     %rcx, %r8
        mov     %r8, L_CONVBASE(%rbp)
        mov     %r12, %rax
        shl     $5, %rax
        add     ST_FIRST(%rbx), %rax
        mov     8(%rax), %r9                  # requested chunk length
        mov     16(%rax), %r13                # first intersecting span
        mov     (%rax), %rax                  # actual sparse file offset
        mov     %rax, L_CHUNK(%rbp)
        lea     (%rax,%rdx), %rcx
        mov     %rcx, L_VALID(%rbp)
        add     %r9, %rax
        mov     %rax, L_END(%rbp)
        add     %rdx, ST_BYTES(%rbx)
.Lpr_loop:
        cmp     ST_NSPANS(%rbx), %r13
        jae     .Lpr_record
        mov     %r13, %rax
        shl     $5, %rax
        add     ST_SPANS(%rbx), %rax
        mov     %rax, %r14                    # sp
        mov     SP_OFF(%r14), %r15            # ov_lo = max(off, chunk)
        cmp     L_END(%rbp), %r15
        jae     .Lpr_record                   # span starts after this chunk
        cmp     L_CHUNK(%rbp), %r15
        jae     1f
        mov     L_CHUNK(%rbp), %r15
1:      mov     SP_OFF(%r14), %r10
        add     SP_N(%r14), %r10              # span end
        mov     L_END(%rbp), %r11
        cmp     %r11, %r10
        cmovb   %r10, %r11                    # ov_hi = min(end, chunk end)
        cmp     %r15, %r11
        jbe     .Lpr_next                     # empty piece
        cmp     L_VALID(%rbp), %r11
        ja      .Lpr_short                    # the drive did not deliver it
        # piece [r15, r11): src, rel, n
        mov     %r11, %rcx
        sub     %r15, %rcx                    # n (file bytes)
        mov     %r15, %r9
        sub     L_CHUNK(%rbp), %r9            # in-slot offset
        mov     L_SLOTBASE(%rbp), %rsi
        add     %r9, %rsi                     # src
        mov     %r15, %r10
        sub     SP_OFF(%r14), %r10            # rel (file bytes into the tensor)
        cmpl    $0, SP_HOST(%r14)
        jne     .Lpr_host
        mov     SP_CONV(%r14), %eax
        test    %eax, %eax
        jz      .Lpr_raw
        cmp     $1, %eax
        je      .Lpr_f16
        # f32 -> bf16: convert n/4 elements into conv slot at rel_in_chunk/2
        mov     L_CONVBASE(%rbp), %rdi
        shr     $1, %r9
        add     %r9, %rdi                     # cdst
        mov     %rcx, %rdx
        shr     $2, %rdx                      # elements
        mov     %rdi, -104(%rbp)              # keep cdst
        mov     %rcx, -112(%rbp)              # keep n
        mov     %r10, -120(%rbp)              # keep rel
        mov     %rsi, %rax
        mov     %rax, %rdi                    # src
        mov     -104(%rbp), %rsi              # dst
        call    serenity_f32_to_bf16_rne
        mov     ST_ARENA(%rbx), %rdi
        add     SP_DST(%r14), %rdi
        mov     -120(%rbp), %r10
        shr     $1, %r10
        add     %r10, %rdi                    # arena + dst_off + rel/2
        mov     -104(%rbp), %rsi              # cdst
        mov     -112(%rbp), %rdx
        shr     $1, %rdx                      # n/2 device bytes
        jmp     .Lpr_copy
.Lpr_f16:
        mov     L_CONVBASE(%rbp), %rdi
        add     %r9, %rdi                     # cdst (same offsets)
        mov     %rcx, %rdx
        shr     $1, %rdx                      # elements
        mov     %rdi, -104(%rbp)
        mov     %rcx, -112(%rbp)
        mov     %r10, -120(%rbp)
        mov     %rsi, %rdi                    # src
        mov     -104(%rbp), %rsi              # dst
        call    serenity_f16_to_bf16_rne
        mov     ST_ARENA(%rbx), %rdi
        add     SP_DST(%r14), %rdi
        add     -120(%rbp), %rdi              # arena + dst_off + rel
        mov     -104(%rbp), %rsi
        mov     -112(%rbp), %rdx              # n device bytes
        jmp     .Lpr_copy
.Lpr_raw:
        mov     ST_ARENA(%rbx), %rdi
        add     SP_DST(%r14), %rdi
        add     %r10, %rdi                    # arena + dst_off + rel
        mov     %rcx, %rdx                    # n
.Lpr_copy:
        incq    ST_H2DCOPIES(%rbx)
        mov     ST_STREAM(%rbx), %rcx
        vzeroupper
        call    cuMemcpyHtoDAsync_v2@PLT
        test    %eax, %eax
        jnz     .Lpr_cuda
.Lpr_next:
        inc     %r13
        jmp     .Lpr_loop
        # Host destination: the CPU writes the piece straight from the pinned
        # slot into the caller's buffer -- streaming copy for raw bytes, the
        # conversion kernels for F16/F32 -> BF16. No DMA, no conversion slot.
        #   rsi = src, rcx = n (file bytes), r10 = rel (file bytes into tensor)
.Lpr_host:
        mov     SP_CONV(%r14), %eax
        mov     ST_ARENA(%rbx), %rdi
        add     SP_DST(%r14), %rdi
        test    %eax, %eax
        jnz     1f
        add     %r10, %rdi                    # dst + rel
        mov     %rsi, %rax
        mov     %rax, %rsi                    # (src already in rsi)
        mov     %rcx, %rdx                    # n
        call    serenity_nt_memcpy
        jmp     .Lpr_next
1:      cmp     $1, %eax
        jne     2f
        add     %r10, %rdi                    # dst + rel (same size)
        mov     %rcx, %rdx
        shr     $1, %rdx                      # elements
        xchg    %rdi, %rsi                    # src -> rdi, dst -> rsi
        call    serenity_f16_to_bf16_rne
        jmp     .Lpr_next
2:      shr     $1, %r10
        add     %r10, %rdi                    # dst + rel/2
        mov     %rcx, %rdx
        shr     $2, %rdx                      # elements
        xchg    %rdi, %rsi
        call    serenity_f32_to_bf16_rne
        jmp     .Lpr_next
.Lpr_record:
        cmpq    $0, ST_NEEDDEVICE(%rbx)
        je      .Lpr_release
        mov     L_SLOT(%rbp), %rax
        lea     fl_events(%rip), %rcx
        mov     (%rcx,%rax,8), %rdi
        mov     ST_STREAM(%rbx), %rsi
        vzeroupper
        call    cuEventRecord@PLT
        test    %eax, %eax
        jnz     .Lpr_cuda
.Lpr_release:
        mov     L_SLOT(%rbp), %rax
        movb    $0, ST_SLOTBUSY(%rbx,%rax)
        decq    ST_INFLIGHT(%rbx)
        incq    ST_DONE(%rbx)
        xor     %eax, %eax
        jmp     .Lpr_out
.Lpr_readerr:
        mov     $-4, %rax
        jmp     .Lpr_out
.Lpr_short:
        mov     $-5, %rax
        jmp     .Lpr_out
.Lpr_cuda:
        mov     $-6, %rax
.Lpr_out:
        add     $88, %rsp
        pop     %r15
        pop     %r14
        pop     %r13
        pop     %r12
        pop     %rbx
        pop     %rbp
        ret

# ---------------------------------------------------------------------------
# int serenity_fastload_run(const serenity_fastload_request *req)
        .globl  serenity_fastload_run
        .type   serenity_fastload_run, @function
        .p2align 4
serenity_fastload_run:
        push    %rbp
        mov     %rsp, %rbp
        push    %rbx
        push    %r12
        push    %r13
        push    %r14
        push    %r15
        sub     $FRAME, %rsp
        mov     %rsp, %rbx                    # state
        mov     %rdi, %r12                    # request
        # zero the state
        mov     %rbx, %rdi
        xor     %eax, %eax
        mov     $(ST_SIZE/8), %ecx
        rep stosq
        # arguments
        mov     RQ_FD(%r12), %rax
        mov     %rax, ST_FD_PLAIN(%rbx)
        mov     RQ_FD_DIRECT(%r12), %rax
        mov     %rax, ST_FD_DIRECT(%rbx)
        mov     RQ_SPANS(%r12), %rax
        mov     %rax, ST_SPANS(%rbx)
        mov     RQ_NSPANS(%r12), %rax
        mov     %rax, ST_NSPANS(%rbx)
        mov     RQ_ARENA(%r12), %rax
        mov     %rax, ST_ARENA(%rbx)
        mov     RQ_ARENAB(%r12), %rax
        mov     %rax, ST_ARENAB(%rbx)
        mov     RQ_STREAM(%r12), %rax
        mov     %rax, ST_STREAM(%rbx)
        mov     RQ_MODE(%r12), %rax
        mov     %rax, ST_MODE(%rbx)
        mov     RQ_STATS(%r12), %rax
        mov     %rax, ST_STATS(%rbx)
        mov     RQ_ERR(%r12), %rax
        mov     %rax, ST_ERR(%rbx)
        mov     RQ_ERRCAP(%r12), %rax
        mov     %rax, ST_ERRCAP(%rbx)
        movq    $-1, ST_FD(%rbx)
        # Serialize: the pinned ring and its events are process-wide. Callers
        # on other threads wait here (yielding), so concurrent prepares are
        # safe and simply queue their I/O behind each other.
.Lrun_lock:
        xor     %eax, %eax
        mov     $1, %ecx
        lock cmpxchg %rcx, fl_lock(%rip)
        jz      .Lrun_locked
        mov     $SYS_sched_yield, %eax
        syscall
        jmp     .Lrun_lock
.Lrun_locked:
        movq    $-1, ST_RINGFD(%rbx)
        # stats zero + err NUL
        mov     ST_STATS(%rbx), %rdi
        test    %rdi, %rdi
        jz      1f
        vpxor   %xmm0, %xmm0, %xmm0
        vmovdqu %ymm0, (%rdi)
        vmovdqu %ymm0, 32(%rdi)
1:      mov     ST_ERR(%rbx), %rdi
        test    %rdi, %rdi
        jz      2f
        cmpq    $0, ST_ERRCAP(%rbx)
        je      2f
        movb    $0, (%rdi)
2:      call    fl_now
        mov     %rax, ST_T0(%rbx)

        # ---- validate spans, find [lo, hi), note conversions
        mov     ST_NSPANS(%rbx), %rcx
        test    %rcx, %rcx
        jz      .Lrun_espans
        mov     ST_SPANS(%rbx), %r12
        xor     %r13, %r13                    # i
        xor     %r14, %r14                    # previous end
.Lval_loop:
        cmp     ST_NSPANS(%rbx), %r13
        jae     .Lval_done
        mov     %r13, %rax
        shl     $5, %rax
        lea     (%r12,%rax), %r15             # sp
        mov     SP_OFF(%r15), %rax
        cmp     %r14, %rax
        jb      .Lrun_espans                  # unsorted / overlapping
        mov     SP_N(%r15), %rdx
        test    %rdx, %rdx
        jz      .Lrun_espans
        mov     SP_CONV(%r15), %ecx
        cmp     $2, %ecx
        ja      .Lrun_espans
        mov     SP_HOST(%r15), %r9d
        cmp     $1, %r9d
        ja      .Lrun_espans
        test    %r9d, %r9d
        jnz     2f
        movq    $1, ST_NEEDDEVICE(%rbx)
2:
        mov     %rdx, %r8                     # device bytes
        cmp     $2, %ecx
        jne     3f
        shr     $1, %r8
        test    $3, %rax
        jnz     .Lrun_espans
        test    $3, %rdx
        jnz     .Lrun_espans
3:      cmp     $1, %ecx
        jne     4f
        test    $1, %rax
        jnz     .Lrun_espans
        test    $1, %rdx
        jnz     .Lrun_espans
4:      test    %ecx, %ecx
        jz      5f
        movq    $1, ST_NEEDCONV(%rbx)
5:      cmpq    $0, ST_ARENAB(%rbx)
        je      55f                           # arena 0: dst_off is an absolute device address
        mov     SP_DST(%r15), %r9
        add     %r8, %r9
        jc      .Lrun_earena
        cmp     ST_ARENAB(%rbx), %r9
        ja      .Lrun_earena
55:
        mov     %rax, %r9
        and     $-PAGE, %r9                   # aligned span start
        add     %rdx, %rax
        jc      .Lrun_espans
        mov     %rax, %r14                    # unaligned span end
        add     $(PAGE-1), %rax
        jc      .Lrun_espans
        and     $-PAGE, %rax
        sub     %r9, %rax
        add     $(FL_C-1), %rax
        jc      .Lrun_espans
        shr     $24, %rax
        add     %rax, ST_NCHUNKS(%rbx)        # upper bound before coalescing
        jc      .Lrun_espans
        inc     %r13
        jmp     .Lval_loop
.Lval_done:
        mov     SP_OFF(%r12), %rax
        and     $-PAGE, %rax
        mov     %rax, ST_LO(%rbx)
        lea     (PAGE-1)(%r14), %rax
        and     $-PAGE, %rax
        mov     %rax, ST_HI(%rbx)

        # ---- choose the descriptor: mode 1 direct, 2 plain, 0 by residency
        mov     ST_MODE(%rbx), %rax
        cmp     $1, %rax
        je      .Luse_direct
        cmp     $2, %rax
        je      .Luse_plain
        # mode 0: needs the plain descriptor to probe; without one, go direct
        cmpq    $0, ST_FD_PLAIN(%rbx)
        jl      .Luse_direct
        mov     ST_FD_PLAIN(%rbx), %rax
        mov     %rax, ST_FD(%rbx)
        call    .Lfsize
        mov     %rbx, %rdi
        call    fl_probe
        test    %rax, %rax                    # 1 = resident enough
        jnz     .Luse_plain
.Luse_direct:
        cmpq    $0, ST_FD_DIRECT(%rbx)
        jl      .Luse_plain_forced            # no O_DIRECT descriptor: plain it is
        mov     ST_FD_DIRECT(%rbx), %rax
        mov     %rax, ST_FD(%rbx)
        movq    $1, ST_DIRECT(%rbx)
        jmp     .Lsized
.Luse_plain:
.Luse_plain_forced:
        cmpq    $0, ST_FD_PLAIN(%rbx)
        jl      .Lrun_eopen
        mov     ST_FD_PLAIN(%rbx), %rax
        mov     %rax, ST_FD(%rbx)
        movq    $2, ST_DIRECT(%rbx)
.Lsized:
        call    .Lfsize
        # Validate the final requested byte, before allocating/issuing I/O.
        cmp     ST_FSIZE(%rbx), %r14
        ja      .Lrun_eshort

        # ---- pinned slots + device events (once per process)
        # Host-only calls need pinned read slots but submit no CUDA work, so
        # they must not synchronize the caller's stream.
        cmpq    $0, fl_ring(%rip)
        jne     71f
        lea     fl_ring(%rip), %rdi
        mov     $(FL_R*FL_C), %esi
        xor     %edx, %edx
        vzeroupper
        call    cuMemHostAlloc@PLT
        test    %eax, %eax
        jnz     .Lrun_epinned
71:     cmpq    $0, ST_NEEDDEVICE(%rbx)
        je      7f
        cmpq    $0, fl_events(%rip)
        jne     7f
        xor     %r13, %r13
8:      lea     fl_events(%rip), %rdi
        lea     (%rdi,%r13,8), %rdi
        mov     $CU_EVENT_DISABLE_TIMING, %esi
        call    cuEventCreate@PLT
        test    %eax, %eax
        jnz     .Lrun_ecuda
        inc     %r13
        cmp     $FL_R, %r13
        jb      8b
7:      cmpq    $0, ST_NEEDCONV(%rbx)
        je      9f
        cmpq    $0, fl_conv(%rip)
        jne     9f
        lea     fl_conv(%rip), %rdi
        mov     $(FL_R*FL_C), %esi
        xor     %edx, %edx
        call    cuMemHostAlloc@PLT
        test    %eax, %eax
        jnz     .Lrun_epinned
9:
        # ---- io_uring
        lea     ST_PARAMS(%rbx), %rdi
        xor     %eax, %eax
        mov     $(P_SIZE/8), %ecx
        rep stosq
        mov     $SYS_io_uring_setup, %eax
        mov     $FL_SQ, %edi
        lea     ST_PARAMS(%rbx), %rsi
        syscall
        test    %rax, %rax
        js      .Lrun_esetup
        mov     %rax, ST_RINGFD(%rbx)
        # sq ring
        mov     ST_PARAMS+P_SQ_ENTRIES(%rbx), %eax
        shl     $2, %rax
        add     ST_PARAMS+P_SQ_ARRAY(%rbx), %eax
        mov     %rax, ST_SQRINGSZ(%rbx)
        mov     %rax, %rsi
        mov     $IORING_OFF_SQ_RING, %r9d
        call    .Lringmap
        mov     %rax, ST_SQRING(%rbx)
        # cq ring
        mov     ST_PARAMS+P_CQ_ENTRIES(%rbx), %eax
        shl     $4, %rax
        add     ST_PARAMS+P_CQ_CQES(%rbx), %eax
        mov     %rax, ST_CQRINGSZ(%rbx)
        mov     %rax, %rsi
        mov     $IORING_OFF_CQ_RING, %r9d
        call    .Lringmap
        mov     %rax, ST_CQRING(%rbx)
        # sqes
        mov     ST_PARAMS+P_SQ_ENTRIES(%rbx), %eax
        shl     $6, %rax
        mov     %rax, ST_SQESSZ(%rbx)
        mov     %rax, %rsi
        mov     $IORING_OFF_SQES, %r9d
        call    .Lringmap
        mov     %rax, ST_SQES(%rbx)
        # derived pointers
        mov     ST_SQRING(%rbx), %rax
        mov     ST_PARAMS+P_SQ_TAIL(%rbx), %ecx
        lea     (%rax,%rcx), %rdx
        mov     %rdx, ST_SQTAILP(%rbx)
        mov     ST_PARAMS+P_SQ_MASK(%rbx), %ecx
        mov     (%rax,%rcx), %ecx
        mov     %rcx, ST_SQMASK(%rbx)
        mov     ST_PARAMS+P_SQ_ARRAY(%rbx), %ecx
        lea     (%rax,%rcx), %rdx
        mov     %rdx, ST_SQARRAY(%rbx)
        mov     ST_SQTAILP(%rbx), %rdx
        mov     (%rdx), %ecx
        mov     %rcx, ST_SQTAIL(%rbx)
        mov     ST_CQRING(%rbx), %rax
        mov     ST_PARAMS+P_CQ_HEAD(%rbx), %ecx
        lea     (%rax,%rcx), %rdx
        mov     %rdx, ST_CQHEADP(%rbx)
        mov     (%rdx), %ecx
        mov     %rcx, ST_CQHEAD(%rbx)
        mov     ST_PARAMS+P_CQ_TAIL(%rbx), %ecx
        lea     (%rax,%rcx), %rdx
        mov     %rdx, ST_CQTAILP(%rbx)
        mov     ST_PARAMS+P_CQ_MASK(%rbx), %ecx
        mov     (%rax,%rcx), %ecx
        mov     %rcx, ST_CQMASK(%rbx)
        mov     ST_PARAMS+P_CQ_CQES(%rbx), %ecx
        lea     (%rax,%rcx), %rdx
        mov     %rdx, ST_CQES(%rbx)

        # ---- sparse chunk table. The per-span ceiling computed at validation
        # bounds the chunk count even when adjacent page ranges coalesce.
        mov     ST_NCHUNKS(%rbx), %rax
        mov     %rax, %rcx
        shr     $58, %rcx
        jnz     .Lrun_espans
        shl     $5, %rax
        add     $(PAGE-1), %rax
        and     $-PAGE, %rax
        mov     %rax, ST_FIRSTSZ(%rbx)
        mov     $SYS_mmap, %eax
        xor     %edi, %edi
        mov     ST_FIRSTSZ(%rbx), %rsi
        mov     $PROT_RW, %edx
        mov     $MAP_PRIVATE_ANON, %r10d
        mov     $-1, %r8d
        xor     %r9d, %r9d
        syscall
        cmp     $-4096, %rax
        ja      .Lrun_emmap
        mov     %rax, ST_FIRST(%rbx)
        xor     %r13, %r13                    # k
        xor     %r14, %r14                    # i
        mov     ST_SPANS(%rbx), %r12
.Lfs_loop:
        cmp     ST_NSPANS(%rbx), %r14
        jae     .Lfs_done
        mov     %r14, %rcx
        shl     $5, %rcx
        mov     SP_OFF(%r12,%rcx), %r8
        and     $-PAGE, %r8                   # next range start
        # The previous chunk may end inside this same span.
        cmp     ST_LO(%rbx), %r8
        cmovb   ST_LO(%rbx), %r8
        mov     SP_OFF(%r12,%rcx), %r9
        add     SP_N(%r12,%rcx), %r9
        add     $(PAGE-1), %r9
        and     $-PAGE, %r9                   # range end
        lea     FL_C(%r8), %r10               # chunk size limit
        cmp     %r10, %r9
        cmova   %r10, %r9
        mov     %r14, %r15                    # first span for this chunk
.Lfs_merge:
        mov     %r14, %rcx
        shl     $5, %rcx
        mov     SP_OFF(%r12,%rcx), %rdx
        add     SP_N(%r12,%rcx), %rdx
        cmp     %r9, %rdx
        ja      .Lfs_emit                    # this span continues next chunk
        inc     %r14
        cmp     ST_NSPANS(%rbx), %r14
        jae     .Lfs_emit
        mov     %r14, %rcx
        shl     $5, %rcx
        mov     SP_OFF(%r12,%rcx), %rdx
        and     $-PAGE, %rdx
        cmp     %r9, %rdx
        ja      .Lfs_emit                    # skip the gap before next span
        cmp     %r10, %rdx
        jae     .Lfs_emit                    # next span starts after capacity
        mov     SP_OFF(%r12,%rcx), %rdx
        add     SP_N(%r12,%rcx), %rdx
        add     $(PAGE-1), %rdx
        and     $-PAGE, %rdx
        cmp     %r10, %rdx
        cmova   %r10, %rdx
        cmp     %r9, %rdx
        cmova   %rdx, %r9
        jmp     .Lfs_merge
.Lfs_emit:
        mov     %r13, %rcx
        shl     $5, %rcx
        add     ST_FIRST(%rbx), %rcx
        mov     %r8, (%rcx)
        mov     %r9, %rdx
        sub     %r8, %rdx
        mov     %rdx, 8(%rcx)
        mov     %r15, 16(%rcx)
        mov     %r9, ST_LO(%rbx)              # continuation cursor
        inc     %r13
        jmp     .Lfs_loop
.Lfs_done:
        mov     %r13, ST_NCHUNKS(%rbx)

        # ---- the pipeline
.Lmain_loop:
        # submit while there is room and the next slot is free
12:     mov     ST_NEXT(%rbx), %rsi
        cmp     ST_NCHUNKS(%rbx), %rsi
        jae     13f
        cmpq    $FL_R, ST_INFLIGHT(%rbx)
        jae     13f
        mov     %rbx, %rdi
        call    fl_submit
        test    %rax, %rax
        js      .Lrun_err
        jz      12b                           # submitted, try another
13:     cmpq    $0, ST_INFLIGHT(%rbx)
        je      .Lmain_done
        # submit pending + wait for at least one completion
14:     mov     $SYS_io_uring_enter, %eax
        mov     ST_RINGFD(%rbx), %edi
        mov     ST_PENDING(%rbx), %esi
        mov     $1, %edx
        mov     $IORING_ENTER_GETEVENTS, %r10d
        xor     %r8d, %r8d
        xor     %r9d, %r9d
        syscall
        cmp     $-EINTR, %rax
        je      14b
        test    %rax, %rax
        js      .Lrun_eread
        movq    $0, ST_PENDING(%rbx)
        # drain completions
15:     mov     ST_CQTAILP(%rbx), %rax
        mov     (%rax), %eax
        cmp     ST_CQHEAD(%rbx), %rax
        je      .Lmain_loop
        mov     ST_CQHEAD(%rbx), %rax
        and     ST_CQMASK(%rbx), %rax
        shl     $4, %rax
        add     ST_CQES(%rbx), %rax
        mov     (%rax), %rsi                  # user_data = k
        movslq  8(%rax), %rdx                 # res
        incq    ST_CQHEAD(%rbx)
        mov     ST_CQHEAD(%rbx), %rax
        mov     ST_CQHEADP(%rbx), %rcx
        mov     %eax, (%rcx)
        mov     %rbx, %rdi
        call    fl_process
        test    %rax, %rax
        js      .Lrun_err
        jmp     15b
.Lmain_done:
        cmpq    $0, ST_NEEDDEVICE(%rbx)
        je      .Lmain_synced
        mov     ST_STREAM(%rbx), %rdi
        vzeroupper
        call    cuStreamSynchronize@PLT
        test    %eax, %eax
        jnz     .Lrun_ecuda
.Lmain_synced:
        # stats
        mov     ST_STATS(%rbx), %rdi
        test    %rdi, %rdi
        jz      16f
        mov     ST_BYTES(%rbx), %rax
        mov     %rax, STAT_BYTES(%rdi)
        mov     ST_DONE(%rbx), %rax
        mov     %rax, STAT_CHUNKS(%rdi)
        mov     ST_DIRECT(%rbx), %rax
        mov     %rax, STAT_MODE(%rdi)
        mov     ST_H2DCOPIES(%rbx), %rax
        mov     %rax, 56(%rdi)
        call    fl_now
        sub     ST_T0(%rbx), %rax
        mov     ST_STATS(%rbx), %rdi
        mov     %rax, STAT_NS(%rdi)
16:     xor     %r12d, %r12d                  # return 0
        jmp     .Lrun_cleanup

        # ---- errors (code in r12), then cleanup
.Lrun_eopen:   mov $-1, %r12; jmp .Lrun_err2
.Lrun_esetup:  mov $-2, %r12; jmp .Lrun_err2
.Lrun_emmap:   mov $-3, %r12; jmp .Lrun_err2
.Lrun_eread:   mov $-4, %r12; jmp .Lrun_err2
.Lrun_eshort:  mov $-5, %r12; jmp .Lrun_err2
.Lrun_ecuda:   mov $-6, %r12; jmp .Lrun_err2
.Lrun_espans:  mov $-7, %r12; jmp .Lrun_err2
.Lrun_epinned: mov $-8, %r12; jmp .Lrun_err2
.Lrun_earena:  mov $-9, %r12; jmp .Lrun_err2
.Lrun_err:     mov %rax, %r12
.Lrun_err2:
        mov     %rbx, %rdi
        mov     %r12, %rsi
        call    fl_fail
        # let outstanding DMA finish before the slots are reused by anyone
        cmpq    $0, ST_NEEDDEVICE(%rbx)
        je      .Lrun_cleanup
        mov     ST_STREAM(%rbx), %rdi
        vzeroupper
        call    cuStreamSynchronize@PLT
.Lrun_cleanup:
        mov     ST_FIRST(%rbx), %rdi
        mov     ST_FIRSTSZ(%rbx), %rsi
        call    .Lunmap
        mov     ST_SQES(%rbx), %rdi
        mov     ST_SQESSZ(%rbx), %rsi
        call    .Lunmap
        mov     ST_CQRING(%rbx), %rdi
        mov     ST_CQRINGSZ(%rbx), %rsi
        call    .Lunmap
        mov     ST_SQRING(%rbx), %rdi
        mov     ST_SQRINGSZ(%rbx), %rsi
        call    .Lunmap
        mov     ST_RINGFD(%rbx), %rdi
        call    .Lclose
        movq    $0, fl_lock(%rip)             # release (x86 stores are ordered)
        mov     %r12, %rax
        add     $FRAME, %rsp
        pop     %r15
        pop     %r14
        pop     %r13
        pop     %r12
        pop     %rbx
        pop     %rbp
        vzeroupper
        ret

# local subroutines (state in %rbx; C ABI scratch registers only)
.Lfsize:
        mov     $SYS_lseek, %eax
        mov     ST_FD(%rbx), %edi
        xor     %esi, %esi
        mov     $SEEK_END, %edx
        syscall
        mov     %rax, ST_FSIZE(%rbx)
        ret
.Lringmap:                                    # rsi = size, r9d = ring offset -> rax
        mov     $SYS_mmap, %eax
        xor     %edi, %edi
        mov     $PROT_RW, %edx
        mov     $MAP_SHARED_POPULATE, %r10d
        mov     ST_RINGFD(%rbx), %r8d
        syscall
        cmp     $-4096, %rax
        ja      .Lringmap_fail
        ret
.Lringmap_fail:
        add     $8, %rsp                      # drop the return address
        jmp     .Lrun_emmap
.Lunmap:                                      # rdi = ptr, rsi = size
        test    %rdi, %rdi
        jz      1f
        mov     $SYS_munmap, %eax
        syscall
1:      ret
.Lclose:                                      # rdi = fd
        cmp     $0, %rdi
        jl      1f
        mov     $SYS_close, %eax
        syscall
1:      ret
        .size   serenity_fastload_run, .-serenity_fastload_run

# ---------------------------------------------------------------------------
# int fl_probe(state *rdi) -- residency of the requested pages; returns 1 when
# at least 95% are resident (buffered reads win), else 0 (go O_DIRECT).
# Fills stats.resident_pages / total_pages. Skip gaps, and count shared
# boundary pages only once. Probing the entire checkpoint for each block
# makes block prefetch O(blocks * checkpoint_size) instead of O(bytes_used).
        .p2align 4
fl_probe:
        push    %rbp
        mov     %rsp, %rbp
        push    %rbx
        push    %r12
        push    %r13
        push    %r14
        push    %r15
        sub     $56, %rsp
        mov     %rdi, %rbx
        mov     ST_FSIZE(%rbx), %r12          # size
        test    %r12, %r12
        jz      .Lpb_no
        lea     (PAGE-1)(%r12), %r13
        shr     $12, %r13                     # npages
        # map the file
        mov     $SYS_mmap, %eax
        xor     %edi, %edi
        mov     %r12, %rsi
        mov     $1, %edx                      # PROT_READ
        mov     $MAP_SHARED, %r10d
        mov     ST_FD(%rbx), %r8d
        xor     %r9d, %r9d
        syscall
        cmp     $-4096, %rax
        ja      .Lpb_no
        mov     %rax, %r14                    # map
        # vector
        lea     (PAGE-1)(%r13), %rsi
        and     $-PAGE, %rsi
        mov     %rsi, -48(%rbp)               # vec size
        mov     $SYS_mmap, %eax
        xor     %edi, %edi
        mov     $PROT_RW, %edx
        mov     $MAP_PRIVATE_ANON, %r10d
        mov     $-1, %r8d
        xor     %r9d, %r9d
        syscall
        cmp     $-4096, %rax
        ja      .Lpb_unmap_file
        mov     %rax, %r15                    # vec
        movq    $0, -56(%rbp)                # total selected pages
        movq    $0, -64(%rbp)                # resident selected pages
        movq    $0, -72(%rbp)                # previous aligned end
        movq    $0, -80(%rbp)                # span index
.Lpb_span:
        mov     -80(%rbp), %rax
        cmp     ST_NSPANS(%rbx), %rax
        jae     .Lpb_counted
        shl     $5, %rax
        add     ST_SPANS(%rbx), %rax
        mov     SP_OFF(%rax), %rdi
        mov     %rdi, %rsi
        add     SP_N(%rax), %rsi
        cmp     ST_FSIZE(%rbx), %rsi
        ja      .Lpb_unmap_both              # main path reports invalid span
        and     $-PAGE, %rdi
        add     $(PAGE-1), %rsi
        and     $-PAGE, %rsi
        cmp     -72(%rbp), %rdi
        cmovb   -72(%rbp), %rdi              # do not count a shared page twice
        cmp     %rdi, %rsi
        jbe     .Lpb_next
        mov     %rsi, -72(%rbp)
        sub     %rdi, %rsi                   # selected mapping extent
        mov     %rsi, %r13
        shr     $12, %r13
        add     %r13, -56(%rbp)
        add     %r14, %rdi
        mov     %r15, %rdx
        mov     $SYS_mincore, %eax
        syscall
        test    %rax, %rax
        jnz     .Lpb_unmap_both
        # count resident
        xor     %eax, %eax                    # count
        xor     %ecx, %ecx                    # i
1:      cmp     %r13, %rcx
        jae     2f
        movzbl  (%r15,%rcx), %edx
        and     $1, %edx
        add     %rdx, %rax
        inc     %rcx
        jmp     1b
2:      add     %rax, -64(%rbp)
.Lpb_next:
        incq    -80(%rbp)
        jmp     .Lpb_span
.Lpb_counted:
        mov     -64(%rbp), %rax
        mov     -56(%rbp), %r13
        test    %r13, %r13
        jz      .Lpb_unmap_both
        mov     ST_STATS(%rbx), %rdi
        test    %rdi, %rdi
        jz      3f
        mov     %rax, STAT_RESID(%rdi)
        mov     %r13, STAT_PAGES(%rdi)
3:      # resident*20 >= npages*19 ?
        imul    $20, %rax, %rax
        imul    $19, %r13, %rcx
        cmp     %rcx, %rax
        setae   %r12b
        movzbl  %r12b, %r12d
        # unmap both
        mov     $SYS_munmap, %eax
        mov     %r15, %rdi
        mov     -48(%rbp), %rsi
        syscall
        mov     $SYS_munmap, %eax
        mov     %r14, %rdi
        mov     ST_FSIZE(%rbx), %rsi
        syscall
        mov     %r12, %rax
        jmp     .Lpb_out
.Lpb_unmap_both:
        mov     $SYS_munmap, %eax
        mov     %r15, %rdi
        mov     -48(%rbp), %rsi
        syscall
.Lpb_unmap_file:
        mov     $SYS_munmap, %eax
        mov     %r14, %rdi
        mov     ST_FSIZE(%rbx), %rsi
        syscall
.Lpb_no:
        xor     %eax, %eax
.Lpb_out:
        add     $56, %rsp
        pop     %r15
        pop     %r14
        pop     %r13
        pop     %r12
        pop     %rbx
        pop     %rbp
        ret

        .section .note.GNU-stack,"",@progbits
