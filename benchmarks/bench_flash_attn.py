"""
Performance benchmark for the Triton FlashAttention implementation.

Usage:
    python benchmarks/bench_flash_attn.py
    python benchmarks/bench_flash_attn.py --seqlen 2048 --causal
    python benchmarks/bench_flash_attn.py --dtype bf16 --head-dim 128 --no-torch

TFLOPS use the usual FlashAttention convention:
    forward  = 4 * B * H * M * N * D
    backward = 10 * B * H * M * N * D
Causal runs count only the useful lower triangle.
"""

import argparse
import sys

import torch
from triton.testing import do_bench

from flash_attn_triton import scaled_dot_product_attention as sdpa


DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def bench_suite(args):
    dtype = DTYPES[args.dtype]
    B, H, M, D = args.batch, args.heads, args.seqlen, args.head_dim
    N = M
    dev = "cuda"
    factor = 0.5 if args.causal else 1.0
    fwd_flops = factor * 4 * B * H * M * N * D
    bwd_flops = factor * 10 * B * H * M * N * D

    torch.manual_seed(0)
    q = torch.randn(B, H, M, D, device=dev, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, N, D, device=dev, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, N, D, device=dev, dtype=dtype, requires_grad=True)
    do = torch.randn(B, H, M, D, device=dev, dtype=dtype)

    with torch.no_grad():
        fwd_ms = do_bench(
            lambda: sdpa(q, k, v, is_causal=args.causal),
            warmup=args.warmup, rep=args.rep,
        )

    o = sdpa(q, k, v, is_causal=args.causal)
    torch.autograd.grad(o, (q, k, v), do, retain_graph=True)  # compile/warm up
    bwd_ms = do_bench(
        lambda: torch.autograd.grad(o, (q, k, v), do, retain_graph=True),
        warmup=args.warmup, rep=args.rep,
    )

    label = (
        f"B={B} H={H} M={M} N={N} D={D} {args.dtype} "
        f"causal={int(args.causal)}"
    )
    print(f"\n{label}")
    print(
        f"  triton  fwd {fwd_ms:8.3f} ms {fwd_flops / fwd_ms / 1e9:7.1f} TFLOPS"
        f" | bwd {bwd_ms:8.3f} ms {bwd_flops / bwd_ms / 1e9:7.1f} TFLOPS"
    )

    if not args.no_torch:
        with torch.no_grad():
            torch_fwd_ms = do_bench(
                lambda: torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=args.causal
                ),
                warmup=args.warmup, rep=args.rep,
            )
        torch_o = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=args.causal
        )
        torch.autograd.grad(torch_o, (q, k, v), do, retain_graph=True)
        torch_bwd_ms = do_bench(
            lambda: torch.autograd.grad(
                torch_o, (q, k, v), do, retain_graph=True
            ),
            warmup=args.warmup, rep=args.rep,
        )
        print(
            f"  torch   fwd {torch_fwd_ms:8.3f} ms"
            f" {fwd_flops / torch_fwd_ms / 1e9:7.1f} TFLOPS"
            f" | bwd {torch_bwd_ms:8.3f} ms"
            f" {bwd_flops / torch_bwd_ms / 1e9:7.1f} TFLOPS"
        )
        print(
            f"  speedup fwd {torch_fwd_ms / fwd_ms:5.2f}x"
            f" | bwd {torch_bwd_ms / bwd_ms:5.2f}x"
        )
    sys.stdout.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--seqlen", type=int, default=1024)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    p.add_argument("--causal", action="store_true")
    p.add_argument("--no-torch", action="store_true",
                   help="skip the torch SDPA reference")
    p.add_argument("--warmup", type=int, default=1000,
                   help="warmup time in ms (longer stabilizes laptop GPU clocks)")
    p.add_argument("--rep", type=int, default=1000,
                   help="measurement time in ms")
    args = p.parse_args()
    bench_suite(args)


if __name__ == "__main__":
    main()
