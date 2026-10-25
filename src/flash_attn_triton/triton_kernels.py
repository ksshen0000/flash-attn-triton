# This file must not import torch.
# Kernels only. No Python-side logic.
#
# FlashAttention-2 style kernels:
#   * forward: one program per query block
#   * backward dQ: one program per query block (also writes Delta)
#   * backward dK/dV: one program per key block, both gradients in one pass
#
# Inputs/outputs keep their original dtype.  Matrix multiplies run on tensor
# cores using the input dtype (fp16/bf16), while softmax state and accumulators
# stay in fp32.  For fp32 inputs Triton uses TF32 tensor cores by default.
#
# Each kernel splits its inner loop into an unmasked bulk stage and a masked
# stage that only covers the blocks touching the causal diagonal or the
# sequence boundary, so most iterations skip the `tl.where` masking.
import triton
import triton.language as tl


@triton.jit
def _fwd_inner(
    acc, l_i, m_i, q,
    k_base, v_base, stride_kn, stride_kd, stride_vn, stride_vd,
    offs_m, offs_n, offs_d,
    lo, hi, N, D,
    qk_scale,
    BLOCK_N: tl.constexpr,
    CAUSAL_MASK: tl.constexpr,
    BOUNDARY_N: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_cur = start_n + offs_n

        k_ptrs = k_base + offs_n_cur[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n_cur[:, None] * stride_vn + offs_d[None, :] * stride_vd
        if BOUNDARY_N:
            kv_mask = (offs_n_cur[:, None] < N) & (offs_d[None, :] < D)
            k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
            v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
        elif EVEN_D:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            d_mask = offs_d[None, :] < D
            k = tl.load(k_ptrs, mask=d_mask, other=0.0)
            v = tl.load(v_ptrs, mask=d_mask, other=0.0)

        # [BLOCK_M, BLOCK_N], fp32 accumulator (tensor cores for fp16/bf16).
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        if BOUNDARY_N:
            qk = tl.where(offs_n_cur[None, :] < N, qk, -float("inf"))
        if CAUSAL_MASK:
            qk = tl.where(offs_n_cur[None, :] <= offs_m[:, None], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new
    return acc, l_i, m_i


@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    B, H, M, N, D,
    qk_scale,                   # softmax_scale * log2(e)
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lseb, stride_lseh, stride_lsem,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = (
        Q_ptr + b * stride_qb + h * stride_qh
        + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    )
    if EVEN_M and EVEN_D:
        q = tl.load(q_ptrs)
    else:
        q = tl.load(
            q_ptrs,
            mask=(offs_m[:, None] < M) & (offs_d[None, :] < D),
            other=0.0,
        )

    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    if IS_CAUSAL:
        # Keys before the diagonal block are fully visible; keys past the
        # query block can never be attended to.  M == N, so the bulk stage
        # never crosses the sequence end.
        diag_lo = (start_m // BLOCK_N) * BLOCK_N
        hi = tl.minimum(N, start_m + BLOCK_M)
        acc, l_i, m_i = _fwd_inner(
            acc, l_i, m_i, q,
            k_base, v_base, stride_kn, stride_kd, stride_vn, stride_vd,
            offs_m, offs_n, offs_d, 0, diag_lo, N, D, qk_scale,
            BLOCK_N, False, False, EVEN_D,
        )
        acc, l_i, m_i = _fwd_inner(
            acc, l_i, m_i, q,
            k_base, v_base, stride_kn, stride_kd, stride_vn, stride_vd,
            offs_m, offs_n, offs_d, diag_lo, hi, N, D, qk_scale,
            BLOCK_N, True, not EVEN_N, EVEN_D,
        )
    else:
        n_full = (N // BLOCK_N) * BLOCK_N
        acc, l_i, m_i = _fwd_inner(
            acc, l_i, m_i, q,
            k_base, v_base, stride_kn, stride_kd, stride_vn, stride_vd,
            offs_m, offs_n, offs_d, 0, n_full, N, D, qk_scale,
            BLOCK_N, False, False, EVEN_D,
        )
        if not EVEN_N:
            acc, l_i, m_i = _fwd_inner(
                acc, l_i, m_i, q,
                k_base, v_base, stride_kn, stride_kd, stride_vn, stride_vd,
                offs_m, offs_n, offs_d, n_full, N, N, D, qk_scale,
                BLOCK_N, False, True, EVEN_D,
            )

    acc = acc / l_i[:, None]
    # LSE is expressed in natural log units, like torch's reference.
    lse = (tl.log2(l_i) + m_i) * 0.6931471805599453

    lse_ptrs = (
        LSE_ptr + b * stride_lseb + h * stride_lseh + offs_m * stride_lsem
    )
    if EVEN_M:
        tl.store(lse_ptrs, lse)
    else:
        tl.store(lse_ptrs, lse, mask=offs_m < M)

    o_ptrs = (
        O_ptr + b * stride_ob + h * stride_oh
        + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    )
    if EVEN_M and EVEN_D:
        tl.store(o_ptrs, acc)
    else:
        tl.store(
            o_ptrs,
            acc,
            mask=(offs_m[:, None] < M) & (offs_d[None, :] < D),
        )


@triton.jit
def _bwd_dq_inner(
    dq, q, do, lse, delta,
    k_base, v_base, stride_kn, stride_kd, stride_vn, stride_vd,
    offs_m, offs_n, offs_d,
    lo, hi, N, D,
    qk_scale,
    BLOCK_N: tl.constexpr,
    CAUSAL_MASK: tl.constexpr,
    BOUNDARY_N: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    # Keys past N load as zeros, so their contribution ds @ k vanishes and the
    # scores need no boundary mask here; only the loads must be guarded.
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_cur = start_n + offs_n

        k_ptrs = k_base + offs_n_cur[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n_cur[:, None] * stride_vn + offs_d[None, :] * stride_vd
        if BOUNDARY_N:
            kv_mask = (offs_n_cur[:, None] < N) & (offs_d[None, :] < D)
            k = tl.load(k_ptrs, mask=kv_mask, other=0.0)
            v = tl.load(v_ptrs, mask=kv_mask, other=0.0)
        elif EVEN_D:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            d_mask = offs_d[None, :] < D
            k = tl.load(k_ptrs, mask=d_mask, other=0.0)
            v = tl.load(v_ptrs, mask=d_mask, other=0.0)

        qk = tl.dot(q, tl.trans(k)) * qk_scale
        if CAUSAL_MASK:
            qk = tl.where(offs_n_cur[None, :] <= offs_m[:, None], qk, -float("inf"))

        p = tl.exp2(qk - lse[:, None])
        dp = tl.dot(do, tl.trans(v))
        # Gradient wrt the scaled scores; the softmax scale is applied to dq
        # once after the loop.
        ds = p * (dp - delta[:, None])

        dq = tl.dot(ds.to(q.dtype), k, dq)
    return dq


@triton.jit
def flash_attn_bwd_dQ_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, dQ_ptr, LSE_ptr, Delta_ptr,
    B, H, M, N, D,
    qk_scale,                   # softmax_scale * log2(e)
    scale,                      # softmax_scale
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    stride_lseb, stride_lseh, stride_lsem,
    stride_db, stride_dh, stride_dm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = (
        Q_ptr + b * stride_qb + h * stride_qh
        + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    )
    do_ptrs = (
        dO_ptr + b * stride_dob + h * stride_doh
        + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
    )
    o_ptrs = (
        O_ptr + b * stride_ob + h * stride_oh
        + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    )

    if EVEN_M and EVEN_D:
        q = tl.load(q_ptrs)
        do = tl.load(do_ptrs)
        o = tl.load(o_ptrs)
    else:
        md_mask = (offs_m[:, None] < M) & (offs_d[None, :] < D)
        q = tl.load(q_ptrs, mask=md_mask, other=0.0)
        do = tl.load(do_ptrs, mask=md_mask, other=0.0)
        o = tl.load(o_ptrs, mask=md_mask, other=0.0)

    lse_ptrs = (
        LSE_ptr + b * stride_lseb + h * stride_lseh + offs_m * stride_lsem
    )
    if EVEN_M:
        lse = tl.load(lse_ptrs) * 1.4426950408889634
    else:
        lse = tl.load(lse_ptrs, mask=offs_m < M, other=0.0) * 1.4426950408889634

    # Delta_i = sum_j O_ij * dO_ij, fused into the dQ kernel.
    delta = tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1)
    delta_ptrs = Delta_ptr + b * stride_db + h * stride_dh + offs_m * stride_dm
    if EVEN_M:
        tl.store(delta_ptrs, delta)
    else:
        tl.store(delta_ptrs, delta, mask=offs_m < M)

    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh

    dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    if IS_CAUSAL:
        diag_lo = (start_m // BLOCK_N) * BLOCK_N
        hi = tl.minimum(N, start_m + BLOCK_M)
        dq = _bwd_dq_inner(
            dq, q, do, lse, delta,
            k_base, v_base, stride_kn, stride_kd, stride_vn, stride_vd,
            offs_m, offs_n, offs_d, 0, diag_lo, N, D, qk_scale,
            BLOCK_N, False, False, EVEN_D,
        )
        dq = _bwd_dq_inner(
            dq, q, do, lse, delta,
            k_base, v_base, stride_kn, stride_kd, stride_vn, stride_vd,
            offs_m, offs_n, offs_d, diag_lo, hi, N, D, qk_scale,
            BLOCK_N, True, not EVEN_N, EVEN_D,
        )
    else:
        n_full = (N // BLOCK_N) * BLOCK_N
        dq = _bwd_dq_inner(
            dq, q, do, lse, delta,
            k_base, v_base, stride_kn, stride_kd, stride_vn, stride_vd,
            offs_m, offs_n, offs_d, 0, n_full, N, D, qk_scale,
            BLOCK_N, False, False, EVEN_D,
        )
        if not EVEN_N:
            dq = _bwd_dq_inner(
                dq, q, do, lse, delta,
                k_base, v_base, stride_kn, stride_kd, stride_vn, stride_vd,
                offs_m, offs_n, offs_d, n_full, N, N, D, qk_scale,
                BLOCK_N, False, True, EVEN_D,
            )
    dq = dq * scale

    dq_ptrs = (
        dQ_ptr + b * stride_dqb + h * stride_dqh
        + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd
    )
    if EVEN_M and EVEN_D:
        tl.store(dq_ptrs, dq)
    else:
        tl.store(
            dq_ptrs,
            dq,
            mask=(offs_m[:, None] < M) & (offs_d[None, :] < D),
        )


@triton.jit
def _bwd_dkdv_inner(
    dk, dv, k, v,
    q_base, do_base, lse_base, delta_base,
    stride_qm, stride_qd, stride_dom, stride_dod, stride_lsem, stride_dm,
    offs_n, offs_m, offs_d,
    lo, hi, M, D,
    qk_scale,
    BLOCK_M: tl.constexpr,
    CAUSAL_MASK: tl.constexpr,
    BOUNDARY_M: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    # Padded query rows load q = dO = 0, so they add nothing to dK/dV; padded
    # key columns are never stored.  Neither boundary needs a score mask.
    for start_m in range(lo, hi, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_cur = start_m + offs_m

        q_ptrs = q_base + offs_m_cur[:, None] * stride_qm + offs_d[None, :] * stride_qd
        do_ptrs = do_base + offs_m_cur[:, None] * stride_dom + offs_d[None, :] * stride_dod
        lse_ptrs = lse_base + offs_m_cur * stride_lsem
        delta_ptrs = delta_base + offs_m_cur * stride_dm
        if BOUNDARY_M:
            md_mask = (offs_m_cur[:, None] < M) & (offs_d[None, :] < D)
            q = tl.load(q_ptrs, mask=md_mask, other=0.0)
            do = tl.load(do_ptrs, mask=md_mask, other=0.0)
            lse = tl.load(lse_ptrs, mask=offs_m_cur < M, other=0.0) * 1.4426950408889634
            delta = tl.load(delta_ptrs, mask=offs_m_cur < M, other=0.0)
        else:
            if EVEN_D:
                q = tl.load(q_ptrs)
                do = tl.load(do_ptrs)
            else:
                d_mask = offs_d[None, :] < D
                q = tl.load(q_ptrs, mask=d_mask, other=0.0)
                do = tl.load(do_ptrs, mask=d_mask, other=0.0)
            lse = tl.load(lse_ptrs) * 1.4426950408889634
            delta = tl.load(delta_ptrs)

        qk = tl.dot(q, tl.trans(k)) * qk_scale
        if CAUSAL_MASK:
            qk = tl.where(offs_n[None, :] <= offs_m_cur[:, None], qk, -float("inf"))

        p = tl.exp2(qk - lse[:, None])
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])

        dv = tl.dot(tl.trans(p.to(do.dtype)), do, dv)
        dk = tl.dot(tl.trans(ds.to(q.dtype)), q, dk)
    return dk, dv


@triton.jit
def flash_attn_bwd_dKdV_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, dK_ptr, dV_ptr, LSE_ptr, Delta_ptr,
    B, H, M, N, D,
    qk_scale,                   # softmax_scale * log2(e)
    scale,                      # softmax_scale
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_lseb, stride_lseh, stride_lsem,
    stride_db, stride_dh, stride_dm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    k_ptrs = (
        K_ptr + b * stride_kb + h * stride_kh
        + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    )
    v_ptrs = (
        V_ptr + b * stride_vb + h * stride_vh
        + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    )
    if EVEN_N and EVEN_D:
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
    else:
        nd_mask = (offs_n[:, None] < N) & (offs_d[None, :] < D)
        k = tl.load(k_ptrs, mask=nd_mask, other=0.0)
        v = tl.load(v_ptrs, mask=nd_mask, other=0.0)

    q_base = Q_ptr + b * stride_qb + h * stride_qh
    do_base = dO_ptr + b * stride_dob + h * stride_doh
    lse_base = LSE_ptr + b * stride_lseb + h * stride_lseh
    delta_base = Delta_ptr + b * stride_db + h * stride_dh

    dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    if IS_CAUSAL:
        # Queries before this key block see none of it; query blocks that
        # overlap the diagonal need the mask; later ones see all of it.
        m_lo = (start_n // BLOCK_M) * BLOCK_M
        diag_hi = tl.minimum(
            M, tl.cdiv(start_n + BLOCK_N, BLOCK_M) * BLOCK_M
        )
        dk, dv = _bwd_dkdv_inner(
            dk, dv, k, v,
            q_base, do_base, lse_base, delta_base,
            stride_qm, stride_qd, stride_dom, stride_dod, stride_lsem, stride_dm,
            offs_n, offs_m, offs_d, m_lo, diag_hi, M, D, qk_scale,
            BLOCK_M, True, not EVEN_M, EVEN_D,
        )
        dk, dv = _bwd_dkdv_inner(
            dk, dv, k, v,
            q_base, do_base, lse_base, delta_base,
            stride_qm, stride_qd, stride_dom, stride_dod, stride_lsem, stride_dm,
            offs_n, offs_m, offs_d, diag_hi, M, M, D, qk_scale,
            BLOCK_M, False, not EVEN_M, EVEN_D,
        )
    else:
        m_full = (M // BLOCK_M) * BLOCK_M
        dk, dv = _bwd_dkdv_inner(
            dk, dv, k, v,
            q_base, do_base, lse_base, delta_base,
            stride_qm, stride_qd, stride_dom, stride_dod, stride_lsem, stride_dm,
            offs_n, offs_m, offs_d, 0, m_full, M, D, qk_scale,
            BLOCK_M, False, False, EVEN_D,
        )
        if not EVEN_M:
            dk, dv = _bwd_dkdv_inner(
                dk, dv, k, v,
                q_base, do_base, lse_base, delta_base,
                stride_qm, stride_qd, stride_dom, stride_dod, stride_lsem, stride_dm,
                offs_n, offs_m, offs_d, m_full, M, M, D, qk_scale,
                BLOCK_M, False, True, EVEN_D,
            )
    dk = dk * scale

    dk_ptrs = (
        dK_ptr + b * stride_dkb + h * stride_dkh
        + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
    )
    dv_ptrs = (
        dV_ptr + b * stride_dvb + h * stride_dvh
        + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
    )
    if EVEN_N and EVEN_D:
        tl.store(dk_ptrs, dk)
        tl.store(dv_ptrs, dv)
    else:
        nd_mask = (offs_n[:, None] < N) & (offs_d[None, :] < D)
        tl.store(dk_ptrs, dk, mask=nd_mask)
        tl.store(dv_ptrs, dv, mask=nd_mask)
