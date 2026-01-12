import math
import torch


import triton
import triton.language as tl

from .triton_kernels import (
    flash_attn_fwd_kernel,
    flash_attn_bwd_delta_kernel,
    flash_attn_bwd_dV_kernel,
    flash_attn_bwd_dK_kernel,
    flash_attn_bwd_dQ_kernel,
)
    

def flash_attn_fwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    return_lse=False,
    *,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
    BLOCK_D: int = 128,
    num_warps: int = 4,
    num_stages: int = 1,
):
    assert Q.ndim == 4 and K.ndim == 4 and V.ndim == 4
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    assert Q.dtype in (torch.float16, torch.bfloat16)
    
    B, H, M, D = Q.shape
    _, _, N, _ = K.shape
    LSE = torch.empty((B, H, M), device=Q.device, dtype=torch.float32)

    O = torch.empty_like(Q)

    stride_qb, stride_qh, stride_qm, stride_qd = Q.stride()
    stride_kb, stride_kh, stride_kn, stride_kd = K.stride()
    stride_vb, stride_vh, stride_vn, stride_vd = V.stride()
    stride_ob, stride_oh, stride_om, stride_od = O.stride()
    stride_lseb, stride_lseh, stride_lsem = LSE.stride()
    
    grid = (
        triton.cdiv(M, BLOCK_M),
        B * H,
    )
    scale = 1.0 / math.sqrt(D)

    assert D <= BLOCK_D
    flash_attn_fwd_kernel[grid](
        Q, K, V, O, LSE,
        B, H, M, N, D, scale,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        stride_lseb, stride_lseh, stride_lsem,
        N_CTX=N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if return_lse:
        return O, LSE
    return O

def launch_flash_attn_bwd_delta(
    O: torch.Tensor,
    dO: torch.Tensor,
    Delta: torch.Tensor,
    *,
    BLOCK_M=128,
    BLOCK_D=128,
    num_warps=4,
    num_stages=1,
):
    B, H, M, D = O.shape

    grid = (
        triton.cdiv(M, BLOCK_M),
        B * H,
    )

    flash_attn_bwd_delta_kernel[grid](
        O, dO, Delta,
        B, H, M, D,
        *O.stride(),
        *dO.stride(),
        *Delta.stride(),
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )

def launch_flash_attn_bwd_dV(
    Q: torch.Tensor,
    K: torch.Tensor,
    dO: torch.Tensor,
    dV: torch.Tensor,
    LSE: torch.Tensor,
    *,
    BLOCK_M=128,
    BLOCK_N=128,
    BLOCK_D=128,
    num_warps=4,
    num_stages=1,
):
    B, H, M, D = Q.shape
    _, _, N, _ = K.shape
    scale = 1.0 / math.sqrt(D)
    grid = (
        triton.cdiv(N, BLOCK_N),
        B * H,
    )

    flash_attn_bwd_dV_kernel[grid](
        Q, K, dO, dV,
        LSE,
        B, H, M, N, D, scale,
        *Q.stride(),
        *K.stride(),
        *dO.stride(),
        *dV.stride(),
        *LSE.stride(),
        M_CTX=M,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    
    
def launch_flash_attn_bwd_dK(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    dO: torch.Tensor,
    dK: torch.Tensor,
    LSE: torch.Tensor,
    Delta: torch.Tensor,
    *,
    BLOCK_M=128,
    BLOCK_N=128,
    BLOCK_D=128,
    num_warps=4,
    num_stages=1,
):
    B, H, M, D = Q.shape
    _, _, N, _ = K.shape
    scale = 1.0 / math.sqrt(D)
    grid = (
        triton.cdiv(N, BLOCK_N),
        B * H,
    )

    flash_attn_bwd_dK_kernel[grid](
        Q, K, V, dO, dK,
        LSE, Delta,
        B, H, M, N, D, scale,
        *Q.stride(),
        *K.stride(),
        *V.stride(),
        *dO.stride(),
        *dK.stride(),
        *LSE.stride(),
        *Delta.stride(),
        M_CTX=M,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )

def launch_flash_attn_bwd_dQ(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    dO: torch.Tensor,
    dQ: torch.Tensor,
    LSE: torch.Tensor,
    Delta: torch.Tensor,
    *,
    BLOCK_M=128,
    BLOCK_N=128,
    BLOCK_D=128,
    num_warps=4,
    num_stages=1,
):
    B, H, M, D = Q.shape
    _, _, N, _ = K.shape
    scale = 1.0 / math.sqrt(D)
    grid = (
        triton.cdiv(M, BLOCK_M),
        B * H,
    )

    flash_attn_bwd_dQ_kernel[grid](
        Q, K, V, dO, dQ,
        LSE, Delta,
        B, H, M, N, D, scale,
        *Q.stride(),
        *K.stride(),
        *V.stride(),
        *dO.stride(),
        *dQ.stride(),
        *LSE.stride(),
        *Delta.stride(),
        N_CTX=N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )



def attention_ref(Q, K, V):
    B, H, M, D = Q.shape
    _, _, N, _ = K.shape
    out = torch.empty_like(Q)
    for b in range(B):
        for h in range(H):
            scores = Q[b,h] @ K[b,h].T / math.sqrt(D)
            probs = torch.softmax(scores, dim=-1)
            out[b,h] = probs @ V[b,h]
    return out




class FlashAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V):
        O, LSE = flash_attn_fwd(Q, K, V, return_lse=True)
        ctx.save_for_backward(Q, K, V, O, LSE)
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, LSE = ctx.saved_tensors

        # 1) compute Delta
        Delta = torch.empty_like(LSE)
        launch_flash_attn_bwd_delta(O, dO, Delta)

        # 2) compute dQ, dK, dV
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)

        launch_flash_attn_bwd_dQ(Q, K, V, dO, dQ, LSE, Delta)
        launch_flash_attn_bwd_dK(Q, K, V, dO, dK, LSE, Delta)
        launch_flash_attn_bwd_dV(Q, K, dO, dV, LSE)

        return dQ, dK, dV


import math
import torch

def _make_flash_attn_fn(
    *,
    BLOCK_M: int,
    BLOCK_N: int,
    BLOCK_D: int,
    num_warps: int,
    num_stages: int,
):
    """
    Factory that returns a torch.autograd.Function subclass capturing
    kernel launch configs in a closure. Thread-safe (no global state).
    """

    class _FlashAttnFn(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attn_mask=None,
            dropout_p: float = 0.0,
            is_causal: bool = False,
            scale: float | None = None,
        ):
            # ---- API 对齐：先接住参数，但你实现不支持的先拒绝 ----
            if attn_mask is not None:
                raise NotImplementedError("attn_mask not supported yet.")
            if dropout_p and dropout_p != 0.0:
                raise NotImplementedError("dropout not supported yet.")
            if is_causal:
                raise NotImplementedError("causal attention not supported yet.")

            # ---- scale 对齐（可选）----
            # 你的 flash_attn_fwd 如果暂时不支持 scale override，可以先不传进去；
            # 先把接口接住即可。
            if scale is None:
                scale = 1.0 / math.sqrt(q.shape[-1])

            # forward：拿 O 和 LSE（你 wrapper 需要支持 return_lse=True）
            O, LSE = flash_attn_fwd(
                q, k, v,
                return_lse=True,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_D=BLOCK_D,
                num_warps=num_warps,
                num_stages=num_stages,
                # scale=scale,  # 如果你 wrapper 已支持 scale override，再打开
            )

            ctx.save_for_backward(q, k, v, O, LSE)
            # 如果后面要支持 scale override，可存 ctx.scale = scale
            return O

        @staticmethod
        def backward(ctx, dO: torch.Tensor):
            q, k, v, O, LSE = ctx.saved_tensors
            B, H, M, D = q.shape
            _, _, N, _ = k.shape

            dQ = torch.empty_like(q)
            dK = torch.empty_like(k)
            dV = torch.empty_like(v)

            # Delta 通常用 fp32
            Delta = torch.empty((B, H, M), device=q.device, dtype=torch.float32)

            # 1) Delta
            launch_flash_attn_bwd_delta(
                O, dO, Delta,
                BLOCK_M=BLOCK_M,
                BLOCK_D=BLOCK_D,
                num_warps=num_warps,
                num_stages=num_stages,
            )

            # 2) dV
            launch_flash_attn_bwd_dV(
                q, k, dO, dV, LSE,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_D=BLOCK_D,
                num_warps=num_warps,
                num_stages=num_stages,
            )

            # 3) dK
            launch_flash_attn_bwd_dK(
                q, k, v, dO, dK,
                LSE, Delta,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_D=BLOCK_D,
                num_warps=num_warps,
                num_stages=num_stages,
            )

            # 4) dQ
            launch_flash_attn_bwd_dQ(
                q, k, v, dO, dQ,
                LSE, Delta,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_D=BLOCK_D,
                num_warps=num_warps,
                num_stages=num_stages,
            )

            # forward 入参是 7 个：q,k,v,attn_mask,dropout_p,is_causal,scale
            return dQ, dK, dV, None, None, None, None

    return _FlashAttnFn

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    *,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
    BLOCK_D: int = 128,
    num_warps: int = 4,
    num_stages: int = 1,
):
    """
    PyTorch-like SDPA API (subset).
    Expected shapes:
      q: [B, H, M, D]
      k: [B, H, N, D]
      v: [B, H, N, D]
    """
    Fn = _make_flash_attn_fn(
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    # apply 只能位置参数
    return Fn.apply(q, k, v, attn_mask, dropout_p, is_causal, scale)

def flash_attn(Q, K, V, **kwargs):
    """
    Public API: FlashAttention forward.
    """
    return flash_attn_fwd(Q, K, V, **kwargs)