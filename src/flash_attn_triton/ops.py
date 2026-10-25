import math
from collections import namedtuple

import torch
import triton

# Single source of truth for supported dtypes.
# Accumulation always runs in float32; only I/O and the matmul inputs use the
# caller's dtype, which lets fp16/bf16 use tensor cores.
_SUPPORTED_DTYPES: frozenset = frozenset({
    torch.float32,
    torch.float16,
    torch.bfloat16,
})

_LOG2E = 1.4426950408889634

from .triton_kernels import (
    flash_attn_fwd_kernel,
    flash_attn_bwd_dQ_kernel,
    flash_attn_bwd_dKdV_kernel,
)

# (BLOCK_M, BLOCK_N, num_warps, num_stages)
_Config = namedtuple("_Config", ["fwd", "bwd_dq", "bwd_dkdv"])


def _default_config(D: int, dtype: torch.dtype, is_causal: bool = False) -> _Config:
    """
    Tile/launch configuration tuned on an Ampere-class consumer GPU
    (RTX 3060 Laptop, 6 GB, sm_86).

    Tiles shrink as the head dimension grows so the fp32 accumulators
    (BLOCK_* x BLOCK_D) fit in registers/shared memory.  Half precision can
    afford larger tiles than fp32 because its operands are half the size.

    The dK/dV kernel uses a small query tile and a large key tile: it keeps
    K, V, dK, dV resident while streaming Q/dO, which minimizes register
    pressure and global traffic per program.
    """
    if dtype == torch.float32:
        if D <= 64:
            return _Config(
                fwd=(64, 32, 4, 2),
                bwd_dq=(32, 32, 4, 1),
                bwd_dkdv=(16, 32, 4, 1),
            )
        if D <= 128:
            if is_causal:
                return _Config(
                    fwd=(32, 32, 4, 1),
                    bwd_dq=(64, 32, 4, 1),
                    bwd_dkdv=(16, 64, 4, 2),
                )
            return _Config(
                fwd=(32, 32, 4, 1),
                bwd_dq=(64, 32, 4, 1),
                bwd_dkdv=(16, 64, 4, 1),
            )
        return _Config(
            fwd=(16, 16, 4, 1),
            bwd_dq=(16, 16, 4, 1),
            bwd_dkdv=(16, 16, 4, 1),
        )

    # fp16 / bf16
    if D <= 64:
        if is_causal:
            return _Config(
                fwd=(64, 32, 4, 3),
                bwd_dq=(64, 64, 4, 3),
                bwd_dkdv=(32, 32, 4, 3),
            )
        return _Config(
            fwd=(64, 64, 4, 2),
            bwd_dq=(64, 32, 4, 3),
            bwd_dkdv=(32, 32, 4, 2),
        )
    if D <= 128:
        if is_causal:
            return _Config(
                fwd=(32, 32, 4, 2),
                bwd_dq=(64, 32, 4, 3),
                bwd_dkdv=(32, 32, 4, 3),
            )
        return _Config(
            fwd=(128, 64, 8, 3),
            bwd_dq=(64, 32, 4, 3),
            bwd_dkdv=(16, 64, 4, 2),
        )
    return _Config(
        fwd=(32, 32, 4, 2),
        bwd_dq=(32, 32, 4, 2),
        bwd_dkdv=(16, 32, 4, 2),
    )


def _resolve_config(
    D: int,
    dtype: torch.dtype,
    is_causal: bool = False,
    BLOCK_M: int | None = None,
    BLOCK_N: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> _Config:
    cfg = _default_config(D, dtype, is_causal)

    if BLOCK_M is not None or BLOCK_N is not None or num_warps is not None \
            or num_stages is not None:
        def override(t):
            bm, bn, warps, stages = t
            return (
                BLOCK_M if BLOCK_M is not None else bm,
                BLOCK_N if BLOCK_N is not None else bn,
                num_warps if num_warps is not None else warps,
                num_stages if num_stages is not None else stages,
            )

        cfg = _Config(
            override(cfg.fwd),
            override(cfg.bwd_dq),
            override(cfg.bwd_dkdv),
        )
    return cfg


def _check_inputs(Q, K, V, is_causal: bool):
    assert Q.ndim == 4 and K.ndim == 4 and V.ndim == 4
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.dtype in _SUPPORTED_DTYPES, (
        f"Unsupported dtype {Q.dtype}; expected one of {_SUPPORTED_DTYPES}"
    )
    assert K.dtype == Q.dtype and V.dtype == Q.dtype, \
        "Q, K, V must share the same dtype"
    B, H, M, D = Q.shape
    N = K.shape[2]
    assert K.shape[0] == B and K.shape[1] == H and K.shape[3] == D
    assert V.shape[0] == B and V.shape[1] == H \
        and V.shape[2] == N and V.shape[3] == D
    if is_causal:
        assert M == N, \
            f"causal masking requires M == N (self-attention); got M={M}, N={N}"
    return B, H, M, N, D


def _launch_fwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    O: torch.Tensor,
    LSE: torch.Tensor,
    *,
    is_causal: bool,
    scale: float,
    cfg: tuple,
    B: int,
    H: int,
    M: int,
    N: int,
    D: int,
):
    BLOCK_M, BLOCK_N, num_warps, num_stages = cfg
    BLOCK_D = triton.next_power_of_2(D)
    flash_attn_fwd_kernel[
        (triton.cdiv(M, BLOCK_M), B * H)
    ](
        Q, K, V, O, LSE,
        B, H, M, N, D, scale * _LOG2E,
        *Q.stride(), *K.stride(), *V.stride(), *O.stride(), *LSE.stride(),
        N_CTX=N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        IS_CAUSAL=is_causal,
        EVEN_M=(M % BLOCK_M == 0),
        EVEN_N=(N % BLOCK_N == 0),
        EVEN_D=(D == BLOCK_D),
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _launch_bwd_dq(
    Q, K, V, O, dO, dQ, LSE, Delta, *,
    is_causal: bool, scale: float, cfg: tuple,
    B: int, H: int, M: int, N: int, D: int,
):
    BLOCK_M, BLOCK_N, num_warps, num_stages = cfg
    BLOCK_D = triton.next_power_of_2(D)
    flash_attn_bwd_dQ_kernel[
        (triton.cdiv(M, BLOCK_M), B * H)
    ](
        Q, K, V, O, dO, dQ, LSE, Delta,
        B, H, M, N, D, scale * _LOG2E, scale,
        *Q.stride(), *K.stride(), *V.stride(), *O.stride(), *dO.stride(),
        *dQ.stride(), *LSE.stride(), *Delta.stride(),
        N_CTX=N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        IS_CAUSAL=is_causal,
        EVEN_M=(M % BLOCK_M == 0),
        EVEN_N=(N % BLOCK_N == 0),
        EVEN_D=(D == BLOCK_D),
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _launch_bwd_dkdv(
    Q, K, V, dO, dK, dV, LSE, Delta, *,
    is_causal: bool, scale: float, cfg: tuple,
    B: int, H: int, M: int, N: int, D: int,
):
    BLOCK_M, BLOCK_N, num_warps, num_stages = cfg
    BLOCK_D = triton.next_power_of_2(D)
    flash_attn_bwd_dKdV_kernel[
        (triton.cdiv(N, BLOCK_N), B * H)
    ](
        Q, K, V, dO, dK, dV, LSE, Delta,
        B, H, M, N, D, scale * _LOG2E, scale,
        *Q.stride(), *K.stride(), *V.stride(), *dO.stride(),
        *dK.stride(), *dV.stride(), *LSE.stride(), *Delta.stride(),
        M_CTX=M,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        IS_CAUSAL=is_causal,
        EVEN_M=(M % BLOCK_M == 0),
        EVEN_N=(N % BLOCK_N == 0),
        EVEN_D=(D == BLOCK_D),
        num_warps=num_warps,
        num_stages=num_stages,
    )


def flash_attn_fwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    return_lse: bool = False,
    *,
    is_causal: bool = False,
    scale: float | None = None,
    BLOCK_M: int | None = None,
    BLOCK_N: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    FlashAttention forward pass.

    Args:
        Q: [B, H, M, D]
        K: [B, H, N, D]
        V: [B, H, N, D]
        return_lse: also return log-sum-exp [B, H, M]
        is_causal: lower-triangular mask (requires M == N)
        scale: softmax scale; defaults to 1/sqrt(D)
        BLOCK_M / BLOCK_N / num_warps / num_stages: override the tuned config
    """
    B, H, M, N, D = _check_inputs(Q, K, V, is_causal)
    Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    else:
        scale = float(scale)

    cfg = _resolve_config(
        D, Q.dtype, is_causal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )

    LSE = torch.empty((B, H, M), device=Q.device, dtype=torch.float32)
    O = torch.empty_like(Q)
    _launch_fwd(
        Q, K, V, O, LSE,
        is_causal=is_causal, scale=scale, cfg=cfg.fwd,
        B=B, H=H, M=M, N=N, D=D,
    )
    if return_lse:
        return O, LSE
    return O


def attention_ref(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """Reference (pure-PyTorch) attention used for correctness verification."""
    D = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
    if is_causal:
        M, N = Q.shape[-2], K.shape[-2]
        causal_mask = torch.triu(
            torch.ones(M, N, device=Q.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, V)


class _FlashAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, attn_mask, dropout_p, is_causal, scale, cfg):
        if attn_mask is not None:
            raise NotImplementedError("attn_mask not supported yet.")
        if dropout_p and dropout_p != 0.0:
            raise NotImplementedError("dropout not supported yet.")

        B, H, M, N, D = _check_inputs(q, k, v, is_causal)
        # Save contiguous views so the backward pass can use plain strides.
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        if scale is None:
            scale = 1.0 / math.sqrt(D)
        else:
            scale = float(scale)

        if cfg is None:
            cfg = _resolve_config(D, q.dtype, is_causal)

        O = torch.empty_like(q)
        LSE = torch.empty((B, H, M), device=q.device, dtype=torch.float32)
        _launch_fwd(
            q, k, v, O, LSE,
            is_causal=is_causal, scale=scale, cfg=cfg.fwd,
            B=B, H=H, M=M, N=N, D=D,
        )

        ctx.save_for_backward(q, k, v, O, LSE)
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.cfg = cfg
        return O

    @staticmethod
    def backward(ctx, dO):
        q, k, v, O, LSE = ctx.saved_tensors
        B, H, M, _ = q.shape
        is_causal = ctx.is_causal

        dO = dO.contiguous()
        dQ = torch.empty_like(q)
        dK = torch.empty_like(k)
        dV = torch.empty_like(v)
        Delta = torch.empty((B, H, M), device=q.device, dtype=torch.float32)

        # dQ pass also computes Delta = rowsum(O * dO) for the dK/dV pass.
        _launch_bwd_dq(
            q, k, v, O, dO, dQ, LSE, Delta,
            is_causal=is_causal, scale=ctx.scale, cfg=ctx.cfg.bwd_dq,
            B=B, H=H, M=M, N=k.shape[2], D=q.shape[3],
        )
        _launch_bwd_dkdv(
            q, k, v, dO, dK, dV, LSE, Delta,
            is_causal=is_causal, scale=ctx.scale, cfg=ctx.cfg.bwd_dkdv,
            B=B, H=H, M=M, N=k.shape[2], D=q.shape[3],
        )

        # q, k, v, attn_mask, dropout_p, is_causal, scale, cfg
        return dQ, dK, dV, None, None, None, None, None


class FlashAttnFunction(torch.autograd.Function):
    """Backward-compatible 4-input entry point (no attention-mask args)."""

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, scale=None):
        return _FlashAttnFunction.forward(
            ctx, Q, K, V, None, 0.0, is_causal, scale, None
        )

    @staticmethod
    def backward(ctx, dO):
        dQ, dK, dV, *_ = _FlashAttnFunction.backward(ctx, dO)
        return dQ, dK, dV, None, None


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    *,
    BLOCK_M: int | None = None,
    BLOCK_N: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> torch.Tensor:
    """
    Scaled Dot Product Attention using FlashAttention Triton kernels.

    API-compatible with torch.nn.functional.scaled_dot_product_attention.
    Supports float32 / float16 / bfloat16 and arbitrary head dims.
    BLOCK_D and tile sizes are derived automatically from D; pass BLOCK_M /
    BLOCK_N / num_warps / num_stages only for manual tuning.

    Args:
        q / k / v: [B, H, M, D] / [B, H, N, D] / [B, H, N, D]
        attn_mask:  reserved, not yet supported
        dropout_p:  reserved, not yet supported
        is_causal:  causal (lower-triangular) mask; requires M == N
        scale:      override for 1/sqrt(D)
        BLOCK_M / BLOCK_N / num_warps / num_stages: override auto config
    """
    cfg = _resolve_config(
        q.shape[-1], q.dtype, is_causal,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )
    return _FlashAttnFunction.apply(
        q, k, v, attn_mask, dropout_p, is_causal, scale, cfg
    )


def flash_attn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Convenience wrapper around flash_attn_fwd."""
    return flash_attn_fwd(Q, K, V, **kwargs)
