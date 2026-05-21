import math
import torch
import pytest
from flash_attn_triton import scaled_dot_product_attention as sdpa
from flash_attn_triton.ops import attention_ref, _SUPPORTED_DTYPES

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
