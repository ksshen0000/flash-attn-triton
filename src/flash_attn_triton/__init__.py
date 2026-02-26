"""
FlashAttention implemented in Triton.

This module provides a PyTorch-compatible implementation of FlashAttention
using Triton kernels for both forward and backward passes.

Main functions:
    - scaled_dot_product_attention: PyTorch SDPA API compatible function
    - flash_attn: Direct FlashAttention forward pass
    - flash_attn_fwd: Low-level forward pass with LSE output
    - attention_ref: Reference implementation for testing

Version:
    0.1.0
"""

from .ops import (
    scaled_dot_product_attention,
    flash_attn,
    flash_attn_fwd,
    attention_ref,
)

__version__ = "0.1.0"

__all__ = [
    "scaled_dot_product_attention",
    "flash_attn",
    "flash_attn_fwd",
    "attention_ref",
    "__version__",
]