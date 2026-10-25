# FlashAttention in Triton

A FlashAttention-2 style implementation in Triton with a PyTorch-compatible
`scaled_dot_product_attention` API. Supports `float32`, `float16`, `bfloat16`,
arbitrary head dimensions, causal masking, custom softmax scale, and
forward + backward passes.

## Performance

Measured on an **NVIDIA RTX 3060 Laptop GPU (6 GB, sm_86)** with `B=2, H=8`
unless noted. `bwd` is backward-only (forward excluded). Baseline is the
original implementation in this repository before the optimization pass.
Both variants were benchmarked in the same process with a long warm-up so the
GPU clocks are stable.

| shape | baseline fwd | optimized fwd | speedup | baseline bwd | optimized bwd | speedup |
|---|---:|---:|---:|---:|---:|---:|
| D=64 fp16, 1024 | 0.426 ms | **0.186 ms** | 2.29x | 1.527 ms | **0.652 ms** | 2.34x |
| D=64 fp16, 1024, causal | 0.222 ms | **0.118 ms** | 1.87x | 0.943 ms | **0.383 ms** | 2.46x |
| D=64 fp16, 2048, causal | 0.753 ms | **0.388 ms** | 1.94x | 3.228 ms | **1.321 ms** | 2.44x |
| D=128 fp16, 1024 | 0.752 ms | **0.398 ms** | 1.89x | 3.035 ms | **1.272 ms** | 2.38x |
| D=128 fp16, 1024, causal | 0.455 ms | **0.283 ms** | 1.61x | 1.658 ms | **0.730 ms** | 2.27x |
| D=64 fp32, 1024 | 0.448 ms | **0.361 ms** | 1.24x | 2.077 ms | **1.454 ms** | 1.43x |
| D=64 fp32, 1024, causal | 0.263 ms | **0.227 ms** | 1.15x | 1.233 ms | **0.807 ms** | 1.53x |
| D=128 fp32, 512 | 0.286 ms | **0.282 ms** | 1.01x | 1.233 ms | **0.992 ms** | 1.24x |

For reference, on the same GPU the optimized kernels are roughly on par with
`torch.nn.functional.scaled_dot_product_attention` for fp16 D=64
(within ±10%) and 2–3x faster than torch's fp32 fallback path.

## What was optimized

- **FlashAttention-2 backward.** The original backward used three independent
  kernels (dQ, dK, dV), each re-reading Q/K/V/dO. dK and dV are now computed in
  a single pass, and `Delta = rowsum(O * dO)` is fused into the dQ kernel.
  Backward now launches 2 kernels instead of 4 and does one less full pass over
  the tensors.
- **Tensor cores for fp16/bf16.** Q/K/V are kept in their native dtype for the
  QK/PV matmuls instead of being upcast to fp32. Softmax state and accumulators
  remain fp32 for stability.
- **`exp2`-based softmax.** Scales are folded into a log2-domain multiplier,
  replacing `exp`/`log` with cheaper `exp2`/`log2`; the public LSE remains in
  natural-log units.
- **Tail-specialized kernels.** Compile-time `EVEN_M`/`EVEN_N`/`EVEN_D` flags
  remove load/store masks when the shape divides the tile, while non-multiple
  sequence lengths and head dims still use masked paths.
- **Causal loop bounds.** Forward/dQ skip KV blocks past the diagonal; dK/dV
  skip Q blocks before the diagonal.
- **Tuned tile/launch configs** for fp16/bf16 vs fp32, causal vs non-causal,
  and head-dim buckets, selected automatically at runtime.
- **Correctness fixes found during the pass:** backward dQ/dK were missing the
  softmax scale factor (off by `1/sqrt(D)`), and the public `scale=` argument
  was ignored. Both are fixed and covered by tests.

## Usage

```python
import torch
from flash_attn_triton import scaled_dot_product_attention as sdpa

q = torch.randn(2, 8, 1024, 64, device="cuda", dtype=torch.float16, requires_grad=True)
k = torch.randn(2, 8, 1024, 64, device="cuda", dtype=torch.float16, requires_grad=True)
v = torch.randn(2, 8, 1024, 64, device="cuda", dtype=torch.float16, requires_grad=True)

out = sdpa(q, k, v, is_causal=True)
out.sum().backward()
```

Low-level forward with LSE:

```python
from flash_attn_triton import flash_attn_fwd
out, lse = flash_attn_fwd(q, k, v, return_lse=True)
```

## Setup

The project is developed with the `cuda134` conda environment:

```bash
conda activate cuda134
pip install -e .
# dependencies (torch/triton) are declared in pyproject.toml
# dev extras:
pip install -e ".[dev]"
```

Tested with Python 3.14, PyTorch 2.14.1+cu130, Triton 3.8.0.

## Tests and benchmarks

```bash
pytest -q
python benchmarks/bench_flash_attn.py
python benchmarks/bench_flash_attn.py --dtype fp32 --causal
python benchmarks/bench_flash_attn.py --head-dim 128 --seqlen 2048
```

## Limitations

- `attn_mask` and `dropout_p` are not implemented (they raise
  `NotImplementedError`).
- Causal attention requires `M == N`.
- V must have the same head dimension as Q/K.
