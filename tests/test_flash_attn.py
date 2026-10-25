import math
import torch
import pytest
from flash_attn_triton import scaled_dot_product_attention as sdpa
from flash_attn_triton import flash_attn_fwd
from flash_attn_triton.ops import (
    attention_ref,
    _SUPPORTED_DTYPES,
    FlashAttnFunction,
)

# Tolerance per dtype: computation is always float32; error comes from
# rounding at output.  float16 (10 mantissa bits) and bfloat16 (7 bits)
# have different absolute precision floors.
_FWD_ATOL = {
    torch.float32: 0.01,
    torch.float16: 0.01,
    torch.bfloat16: 0.05,
}


class TestFlashAttentionCorrectness:
    """Test forward and backward pass correctness."""

    def test_float32_forward(self):
        """Verify float32 forward pass."""
        B, H, M, N, D = 2, 4, 128, 128, 64
        device = "cuda"
        dtype = torch.float32

        torch.manual_seed(42)
        q = torch.randn(B, H, M, D, device=device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype)

        # Reference
        o_ref = attention_ref(q.float(), k.float(), v.float()).to(dtype)

        # Triton
        o_triton = sdpa(q, k, v)

        diff = (o_ref - o_triton).abs().max().item()
        assert diff < 0.01, f"Forward max diff too large: {diff}"


class TestFlashAttentionDtypes:
    """Forward and backward correctness for every supported dtype."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward(self, dtype):
        B, H, M, N, D = 2, 4, 128, 128, 64
        device = "cuda"
        torch.manual_seed(42)
        q = torch.randn(B, H, M, D, device=device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype)

        o_ref = attention_ref(q.float(), k.float(), v.float()).to(dtype)
        o_triton = sdpa(q, k, v)

        assert o_triton.dtype == dtype, "output dtype must match input"
        atol = _FWD_ATOL[dtype]
        diff = (o_ref.float() - o_triton.float()).abs().max().item()
        assert diff < atol, f"{dtype} forward max diff {diff:.4f} > atol {atol}"

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_backward(self, dtype):
        B, H, M, N, D = 2, 4, 128, 128, 64
        device = "cuda"
        torch.manual_seed(42)
        q = torch.randn(B, H, M, D, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)

        o = sdpa(q, k, v)
        o.float().pow(2).mean().backward()

        for name, t in (("q", q), ("k", k), ("v", v)):
            assert t.grad is not None, f"{name}.grad is None"
            assert t.grad.dtype == dtype, f"{name}.grad dtype mismatch"
            assert torch.isfinite(t.grad).all(), f"{name}.grad contains non-finite values"
            assert t.grad.abs().sum() > 0, f"{name}.grad is all zeros"


class TestFlashAttentionCausal:
    """Causal mask correctness: forward output, backward gradients, grad vs ref."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_causal_forward(self, dtype):
        B, H, M, N, D = 2, 4, 128, 128, 64
        device = "cuda"
        torch.manual_seed(42)
        q = torch.randn(B, H, M, D, device=device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype)

        o_ref = attention_ref(q.float(), k.float(), v.float(), is_causal=True).to(dtype)
        o_triton = sdpa(q, k, v, is_causal=True)

        assert o_triton.dtype == dtype, "output dtype must match input"
        atol = _FWD_ATOL[dtype]
        diff = (o_ref.float() - o_triton.float()).abs().max().item()
        assert diff < atol, f"causal {dtype} fwd max diff {diff:.4f} > {atol}"

    def test_causal_backward_finite(self):
        B, H, M, N, D = 2, 4, 128, 128, 64
        device, dtype = "cuda", torch.float32
        torch.manual_seed(42)
        q = torch.randn(B, H, M, D, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)

        sdpa(q, k, v, is_causal=True).pow(2).mean().backward()

        for name, t in (("q", q), ("k", k), ("v", v)):
            assert t.grad is not None, f"{name}.grad is None"
            assert torch.isfinite(t.grad).all(), f"{name}.grad has non-finite values"
            assert t.grad.abs().sum() > 0, f"{name}.grad is all zeros"

    def test_causal_backward_vs_ref(self):
        """Compare dQ/dK/dV against the PyTorch reference."""
        B, H, M, N, D = 2, 4, 64, 64, 64
        device, dtype = "cuda", torch.float32
        torch.manual_seed(42)
        base = torch.randn(B, H, M, D, device=device, dtype=dtype)

        def run(fn):
            q = base.clone().requires_grad_(True)
            k = base.clone().requires_grad_(True)
            v = base.clone().requires_grad_(True)
            fn(q, k, v).pow(2).mean().backward()
            return q.grad.clone(), k.grad.clone(), v.grad.clone()

        dq_ref, dk_ref, dv_ref = run(
            lambda q, k, v: attention_ref(q, k, v, is_causal=True)
        )
        dq_tri, dk_tri, dv_tri = run(
            lambda q, k, v: sdpa(q, k, v, is_causal=True)
        )

        atol = 0.01
        for name, ref, tri in [("dQ", dq_ref, dq_tri), ("dK", dk_ref, dk_tri), ("dV", dv_ref, dv_tri)]:
            diff = (ref - tri).abs().max().item()
            assert diff < atol, f"causal {name} max diff {diff:.4f} > {atol}"


class TestFlashAttentionHeadDim:
    """Verify arbitrary head dimensions (D != 64) work correctly."""

    @pytest.mark.parametrize("D", [32, 96, 128])
    def test_forward_head_dim(self, D):
        B, H, M, N = 2, 4, 64, 64
        device, dtype = "cuda", torch.float32
        torch.manual_seed(42)
        q = torch.randn(B, H, M, D, device=device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype)

        o_ref = attention_ref(q, k, v)
        o_triton = sdpa(q, k, v)

        diff = (o_ref - o_triton).abs().max().item()
        assert diff < 0.01, f"D={D} forward max diff {diff:.4f}"

    @pytest.mark.parametrize("D", [128])
    def test_backward_head_dim(self, D):
        B, H, M, N = 2, 4, 64, 64
        device, dtype = "cuda", torch.float32
        torch.manual_seed(42)
        q = torch.randn(B, H, M, D, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)

        sdpa(q, k, v).pow(2).mean().backward()

        for name, t in (("q", q), ("k", k), ("v", v)):
            assert t.grad is not None
            assert torch.isfinite(t.grad).all(), f"D={D} {name}.grad has non-finite values"
            assert t.grad.abs().sum() > 0

    def test_non_contiguous_input(self):
        """Transposed inputs (.contiguous() must be called internally)."""
        B, H, M, D = 2, 4, 64, 64
        device, dtype = "cuda", torch.float32
        torch.manual_seed(42)
        # [B, M, H, D] → permute → [B, H, M, D], non-contiguous
        q = torch.randn(B, M, H, D, device=device, dtype=dtype).permute(0, 2, 1, 3)
        k = torch.randn(B, M, H, D, device=device, dtype=dtype).permute(0, 2, 1, 3)
        v = torch.randn(B, M, H, D, device=device, dtype=dtype).permute(0, 2, 1, 3)

        assert not q.is_contiguous()
        o_ref = attention_ref(q.contiguous(), k.contiguous(), v.contiguous())
        o_triton = sdpa(q, k, v)

        diff = (o_ref - o_triton).abs().max().item()
        assert diff < 0.01, f"non-contiguous input max diff {diff:.4f}"


class TestFlashAttentionSmoke:
    """Basic smoke tests."""

    def test_sdpa_autograd_smoke(self):
        """Original smoke test - verify backward pass produces finite gradients."""
        B, H, M, N, D = 2, 4, 128, 128, 64
        device = "cuda"
        dtype = torch.float32

        q = torch.randn(B, H, M, D, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)

        o = sdpa(q, k, v)
        loss = o.float().pow(2).mean()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

        for grad in (q.grad, k.grad, v.grad):
            assert torch.isfinite(grad).all()
            assert grad.abs().sum() > 0


def _attention_ref_with_scale(q, k, v, is_causal, scale):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        M, N = q.shape[-2], k.shape[-2]
        mask = torch.triu(
            torch.ones(M, N, device=q.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def _autograd_grads(fn, q, k, v, do):
    q = q.detach().clone().requires_grad_(True)
    k = k.detach().clone().requires_grad_(True)
    v = v.detach().clone().requires_grad_(True)
    out = fn(q, k, v)
    return torch.autograd.grad(out, (q, k, v), do)


class TestFlashAttentionStrict:
    """Stricter numerical checks (scale, LSE, non-multiple lengths, tails)."""

    def test_lse_matches_logsumexp(self):
        B, H, M, D = 2, 4, 96, 64
        torch.manual_seed(7)
        q = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32)

        for causal in (False, True):
            _, lse = flash_attn_fwd(q, k, v, return_lse=True, is_causal=causal)
            scores = q @ k.transpose(-2, -1) / math.sqrt(D)
            if causal:
                mask = torch.triu(
                    torch.ones(M, M, device="cuda", dtype=torch.bool), diagonal=1
                )
                scores = scores.masked_fill(mask, float("-inf"))
            lse_ref = torch.logsumexp(scores, dim=-1)
            torch.testing.assert_close(
                lse, lse_ref, rtol=1e-3, atol=1e-2,
                msg=f"causal={causal}",
            )

    @pytest.mark.parametrize("is_causal", [False, True])
    def test_backward_matches_reference(self, is_causal):
        # 96 is not a multiple of the default fp32 tiles, so this also
        # exercises the masked tail blocks.
        B, H, M, D = 2, 4, 96, 64
        torch.manual_seed(11)
        q = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32)
        do = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32)

        ref = _autograd_grads(
            lambda a, b, c: attention_ref(a, b, c, is_causal=is_causal),
            q, k, v, do,
        )
        tri = _autograd_grads(
            lambda a, b, c: sdpa(a, b, c, is_causal=is_causal),
            q, k, v, do,
        )
        for name, r, t in zip(("dQ", "dK", "dV"), ref, tri):
            torch.testing.assert_close(
                t, r, rtol=2e-2, atol=2e-2,
                msg=f"causal={is_causal} {name}",
            )

    def test_scale_override(self):
        B, H, M, D, scale = 2, 4, 96, 64, 0.3
        torch.manual_seed(13)
        q = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32)
        do = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32)

        o = sdpa(q, k, v, scale=scale)
        o_ref = _attention_ref_with_scale(q, k, v, False, scale)
        torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)

        ref = _autograd_grads(
            lambda a, b, c: _attention_ref_with_scale(a, b, c, False, scale),
            q, k, v, do,
        )
        tri = _autograd_grads(lambda a, b, c: sdpa(a, b, c, scale=scale), q, k, v, do)
        for r, t in zip(ref, tri):
            torch.testing.assert_close(t, r, rtol=2e-2, atol=2e-2)

    @pytest.mark.parametrize("is_causal", [False, True])
    def test_non_multiple_sequence_lengths(self, is_causal):
        # M != N is only meaningful for non-causal attention.
        if is_causal:
            B, H, M, N, D = 2, 3, 80, 80, 48
        else:
            B, H, M, N, D = 2, 3, 80, 144, 48
        torch.manual_seed(17)
        q = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float32)
        do = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32)

        o = sdpa(q, k, v, is_causal=is_causal)
        o_ref = attention_ref(q, k, v, is_causal=is_causal)
        torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)

        ref = _autograd_grads(
            lambda a, b, c: attention_ref(a, b, c, is_causal=is_causal),
            q, k, v, do,
        )
        tri = _autograd_grads(
            lambda a, b, c: sdpa(a, b, c, is_causal=is_causal), q, k, v, do
        )
        for r, t in zip(ref, tri):
            torch.testing.assert_close(t, r, rtol=2e-2, atol=2e-2)

    def test_non_contiguous_backward(self):
        B, H, M, D = 2, 4, 96, 64
        torch.manual_seed(19)
        base = torch.randn(B, M, H, D, device="cuda", dtype=torch.float32)
        q = base.permute(0, 2, 1, 3).requires_grad_(True)
        k = base.permute(0, 2, 1, 3).requires_grad_(True)
        v = base.permute(0, 2, 1, 3).requires_grad_(True)
        do = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32)
        assert not q.is_contiguous()

        o = sdpa(q, k, v)
        o_ref = attention_ref(q.contiguous(), k.contiguous(), v.contiguous())
        torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)
        grads = torch.autograd.grad(o, (q, k, v), do)
        ref = _autograd_grads(
            lambda a, b, c: attention_ref(a, b, c), q, k, v, do
        )
        for r, t in zip(ref, grads):
            torch.testing.assert_close(t, r, rtol=2e-2, atol=2e-2)

    def test_legacy_function_wrapper(self):
        B, H, M, D = 1, 2, 96, 64
        torch.manual_seed(23)
        q = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32,
                        requires_grad=True)
        k = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32,
                        requires_grad=True)
        v = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32,
                        requires_grad=True)
        do = torch.randn(B, H, M, D, device="cuda", dtype=torch.float32)

        o = FlashAttnFunction.apply(q, k, v, True)
        o_ref = attention_ref(q, k, v, is_causal=True)
        torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)
        grads = torch.autograd.grad(o, (q, k, v), do)
        assert all(torch.isfinite(g).all() for g in grads)

    @pytest.mark.parametrize("is_causal", [False, True])
    def test_tiny_sequence(self, is_causal):
        B, H, M, D = 1, 1, 1, 16
        torch.manual_seed(29)
        q = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16,
                        requires_grad=True)
        k = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16,
                        requires_grad=True)
        v = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16,
                        requires_grad=True)
        do = torch.randn(B, H, M, D, device="cuda", dtype=torch.float16)

        o = sdpa(q, k, v, is_causal=is_causal)
        o_ref = attention_ref(q.float(), k.float(), v.float(),
                              is_causal=is_causal).to(torch.float16)
        torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)
        grads = torch.autograd.grad(o, (q, k, v), do)
        assert all(torch.isfinite(g.float()).all() for g in grads)

