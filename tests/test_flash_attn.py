import math
import torch
import pytest
from flash_attn_triton import scaled_dot_product_attention as sdpa
from flash_attn_triton.ops import attention_ref


class TestFlashAttentionCorrectness:
    """Test forward and backward pass correctness."""

    def test_float16_forward(self):
        """Verify float16 forward pass."""
        B, H, M, N, D = 2, 4, 128, 128, 64
        device = "cuda"
        dtype = torch.float16

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


class TestFlashAttentionSmoke:
    """Basic smoke tests."""

    def test_sdpa_autograd_smoke(self):
        """Original smoke test - verify backward pass produces finite gradients."""
        B, H, M, N, D = 2, 4, 128, 128, 64
        device = "cuda"
        dtype = torch.float16

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
