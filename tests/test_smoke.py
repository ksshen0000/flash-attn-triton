import torch
from flash_attn_triton import scaled_dot_product_attention as sdpa


def test_sdpa_autograd_smoke():
    B, H, M, N, D = 2, 4, 128, 128, 64
    device = "cuda"
    dtype = torch.float32

    q = torch.randn(B, H, M, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype, requires_grad=True)

    o = sdpa(q, k, v)
    loss = o.float().pow(2).mean()
    loss.backward()

    # ---- 核心断言 ----
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None

    # 梯度不应全 0 / NaN / Inf
    for grad in (q.grad, k.grad, v.grad):
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0