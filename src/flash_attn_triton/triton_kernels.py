# This file must not import torch.
# Kernels only. No Python-side logic.
import triton
import triton.language as tl

@triton.jit
def flash_attn_bwd_dV_kernel(
    Q_ptr, K_ptr, dO_ptr, dV_ptr,
    LSE_ptr,                    # [B,H,M]  = log(l) + m
    B, H, M, N, D, scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dvb, stride_dvh, stride_dvn, stride_dvd,
    stride_lseb, stride_lseh, stride_lsem,
    M_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n  = tl.program_id(0)   # V 行块（N 维）
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offset_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_d = tl.arange(0, BLOCK_D)
    offset_m = tl.arange(0, BLOCK_M)
    
    n_mask = offset_n < N
    d_mask = offset_d < D
    m_mask = offset_m < M
    
    dV = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    
    k_ptrs = K_ptr + \
        b * stride_kb + \
        h * stride_kh + \
        offset_n[:, None] * stride_kn + \
        offset_d[None, :] * stride_kd
        
    k = tl.load(
        k_ptrs,
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    
    for m_start in range(0, M_CTX, BLOCK_M):
        m = m_start + offset_m
        m_mask = m < M
        
        q_ptrs = Q_ptr + \
            b * stride_qb + \
            h * stride_qh + \
            m[:, None] * stride_qm + \
            offset_d[None, :] * stride_qd
        
        q = tl.load(
            q_ptrs,
            mask=m_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        
        do_ptrs = dO_ptr + \
            b * stride_dob + \
            h * stride_doh + \
            m[:, None] * stride_dom + \
            offset_d[None, :] * stride_dod
            
        dO = tl.load(
            do_ptrs,
            mask=m_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        
        lse_ptrs = LSE_ptr + \
            b * stride_lseb + \
            h * stride_lseh + \
            m * stride_lsem
            
        lse = tl.load(
            lse_ptrs,
            mask=m_mask,
            other=-float("inf"),
        ).to(tl.float32)
        
        scores = tl.dot(q.to(tl.float32), tl.trans(k).to(tl.float32)) * scale
        scores = tl.where(n_mask[None, :], scores, -float("inf"))
        p = tl.exp(scores - lse[:, None])
        # Cast p to dO.dtype before the dot (matching reference), then accumulate as float32
        dv_part = tl.dot(tl.trans(p.to(dO.dtype)), dO)
        dV += dv_part.to(tl.float32)
        
    dv_ptrs = dV_ptr + \
        b * stride_dvb + \
        h * stride_dvh + \
        offset_n[:, None] * stride_dvn + \
        offset_d[None, :] * stride_dvd
        
    tl.store(
        dv_ptrs,
        dV,
        mask=n_mask[:, None] & d_mask[None, :],
    )



@triton.jit
def flash_attn_bwd_dK_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, dK_ptr,
    LSE_ptr, Delta_ptr,
    B, H, M, N, D, scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dkb, stride_dkh, stride_dkn, stride_dkd,
    stride_lseb, stride_lseh, stride_lsem,
    stride_db, stride_dh, stride_dm,
    M_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n  = tl.program_id(0)   # N 方向
    pid_bh = tl.program_id(1)   # B*H

    b = pid_bh // H
    h = pid_bh % H

    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    off_d = tl.arange(0, BLOCK_D)

    n_mask = off_n < N
    d_mask = off_d < D
    
    K = tl.load(
        K_ptr + b*stride_kb+h*stride_kh+off_n[:,None]*stride_kn+off_d[None,:]*stride_kd,
        mask = n_mask[:,None] & d_mask[None, :],
        other= 0.0
    ).to(tl.float32)
    
    V = tl.load(
        V_ptr + b*stride_vb+h*stride_vh+off_n[:,None]*stride_vn+off_d[None,:]*stride_vd,
        mask = n_mask[:,None] & d_mask[None, :],
        other= 0.0  
    ).to(tl.float32)
    
    dK = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    
    offs_m = tl.arange(0, BLOCK_M)
    
    for m_start in range(0, M_CTX, BLOCK_M):
        m = m_start + offs_m
        m_mask = m < M
        
        # load Q
        Q = tl.load(
            Q_ptr + b*stride_qb + h*stride_qh + m[:, None]*stride_qm + off_d[None, :]*stride_qd,
            mask=m_mask[:, None] & d_mask[None, :],
            other=0.0
        ).to(tl.float32)
        
        
        # load dO
        dO = tl.load(
            dO_ptr + b*stride_dob + h*stride_doh + m[:, None]*stride_dom + off_d[None, :]*stride_dod,
            mask=m_mask[:, None] & d_mask[None, :],
            other=0.0
        ).to(tl.float32)
        
        lse = tl.load(
            LSE_ptr + b*stride_lseb + h*stride_lseh + m*stride_lsem,
            mask=m_mask,
            other=-float("inf")
        ).to(tl.float32)
        
        delta = tl.load(
            Delta_ptr + b*stride_db + h*stride_dh + m*stride_dm,
            mask=m_mask,
            other=0.0
        ).to(tl.float32)
        
        scores = tl.dot(Q, tl.trans(K.to(tl.float32))) * scale
        scores = tl.where(n_mask[None, :], scores, -float("inf"))
        P = tl.exp(scores - lse[:, None])
        dP = tl.dot(dO, tl.trans(V.to(tl.float32)))
        dS = P * (dP - delta[:, None])
        dK += tl.dot(tl.trans(dS), Q)
    tl.store(
        dK_ptr + b*stride_dkb + h*stride_dkh + off_n[:, None]*stride_dkn + off_d[None, :]*stride_dkd,
        dK,
        mask=n_mask[:, None] & d_mask[None, :],
    )
        
     
        



@triton.jit
def flash_attn_bwd_dQ_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, dQ_ptr,
    LSE_ptr, Delta_ptr,
    B, H, M, N, D, scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    stride_lseb, stride_lseh, stride_lsem,
    stride_db, stride_dh, stride_dm,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_D)

    m_mask = off_m < M
    d_mask = off_d < D

    Q = tl.load(
        Q_ptr + b*stride_qb + h*stride_qh
        + off_m[:, None]*stride_qm + off_d[None, :]*stride_qd,
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0
    ).to(tl.float32)

    dO = tl.load(
        dO_ptr + b*stride_dob + h*stride_doh
        + off_m[:, None]*stride_dom + off_d[None, :]*stride_dod,
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0
    ).to(tl.float32)

    lse = tl.load(
        LSE_ptr + b*stride_lseb + h*stride_lseh
        + off_m*stride_lsem,
        mask=m_mask,
        other=-float("inf")
    ).to(tl.float32)

    delta = tl.load(
        Delta_ptr + b*stride_db + h*stride_dh
        + off_m*stride_dm,
        mask=m_mask,
        other=0.0
    ).to(tl.float32)

    dQ = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    offs_n0 = tl.arange(0, BLOCK_N)
    for n_start in range(0, N_CTX, BLOCK_N):
        off_n = n_start + offs_n0
        n_mask = off_n < N

        K = tl.load(
            K_ptr + b*stride_kb + h*stride_kh
            + off_n[:, None]*stride_kn + off_d[None, :]*stride_kd,
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0
        ).to(tl.float32)

        V = tl.load(
            V_ptr + b*stride_vb + h*stride_vh
            + off_n[:, None]*stride_vn + off_d[None, :]*stride_vd,
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0
        ).to(tl.float32)

        scores = tl.dot(Q, tl.trans(K.to(tl.float32))) * scale
        scores = tl.where(n_mask[None, :], scores, -float("inf"))
        P = tl.exp(scores - lse[:, None])

        dP = tl.dot(dO, tl.trans(V.to(tl.float32)))
        dS = P * (dP - delta[:, None])

        dQ += tl.dot(dS, K)

    tl.store(
        dQ_ptr + b*stride_dqb + h*stride_dqh
        + off_m[:, None]*stride_dqm + off_d[None, :]*stride_dqd,
        dQ,
        mask=m_mask[:, None] & d_mask[None, :]
    )


import triton
import triton.language as tl

@triton.jit
def flash_attn_bwd_delta_kernel(
    O_ptr, dO_ptr, Delta_ptr,
    B, H, M, D,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_db, stride_dh, stride_dm,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m  = tl.program_id(0)   # block over M
    pid_bh = tl.program_id(1)   # over B*H

    b = pid_bh // H
    h = pid_bh % H

    m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d = tl.arange(0, BLOCK_D)

    m_mask = m < M
    d_mask = d < D

    o_ptrs = O_ptr + b*stride_ob + h*stride_oh + m[:, None]*stride_om + d[None, :]*stride_od
    do_ptrs = dO_ptr + b*stride_dob + h*stride_doh + m[:, None]*stride_dom + d[None, :]*stride_dod

    O  = tl.load(o_ptrs,  mask=m_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    dO = tl.load(do_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    delta = tl.sum(O * dO, axis=1)   # [BLOCK_M]

    delta_ptrs = Delta_ptr + b*stride_db + h*stride_dh + m*stride_dm
    tl.store(delta_ptrs, delta, mask=m_mask)

@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, LSE_ptr,
    B, H, M, N, D, scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_lseb, stride_lseh, stride_lsem,
    N_CTX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m  = tl.program_id(0)   # Q 行块
    pid_bh = tl.program_id(1)   # batch-head

    b = pid_bh // H
    h = pid_bh % H

    offset_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_n = tl.arange(0, BLOCK_N)
    offset_d = tl.arange(0, BLOCK_D)

    d_mask = offset_d < D
    m_mask = offset_m < M

    # ---- Q ----
    q_ptrs = (
        Q_ptr
        + b * stride_qb
        + h * stride_qh
        + offset_m[:, None] * stride_qm
        + offset_d[None, :] * stride_qd
    )
    q = tl.load(
        q_ptrs,
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    # ---- online softmax state ----
    m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for n_start in range(0, N_CTX, BLOCK_N):
        n = n_start + offset_n
        col_mask = n < N

        # ---- K ----
        k_ptrs = (
            K_ptr
            + b * stride_kb
            + h * stride_kh
            + n[:, None] * stride_kn
            + offset_d[None, :] * stride_kd
        )
        k = tl.load(
            k_ptrs,
            mask=col_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # ---- V ----
        v_ptrs = (
            V_ptr
            + b * stride_vb
            + h * stride_vh
            + n[:, None] * stride_vn
            + offset_d[None, :] * stride_vd
        )
        v = tl.load(
            v_ptrs,
            mask=col_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # ---- attention ----
        scores = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        scores = tl.where(col_mask[None, :], scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m, block_max)

        scores_exp = tl.exp(scores - m_new[:, None])

        l = l * tl.exp(m - m_new) + tl.sum(scores_exp, axis=1)
        acc = acc * tl.exp(m - m_new)[:, None] + tl.dot(scores_exp, v.to(tl.float32))

        m = m_new

    o = acc / l[:, None]

    lse = tl.log(l) + m

    lse_ptrs = (
        LSE_ptr
        + b * stride_lseb
        + h * stride_lseh
        + offset_m * stride_lsem
    )
    tl.store(lse_ptrs, lse, mask=m_mask)
    # ---- store O ----
    o_ptrs = (
        O_ptr
        + b * stride_ob
        + h * stride_oh
        + offset_m[:, None] * stride_om
        + offset_d[None, :] * stride_od
    )
    tl.store(
        o_ptrs,
        o,
        mask=m_mask[:, None] & d_mask[None, :],
    )
