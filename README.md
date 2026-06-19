# Fused LNCC

A **fully-fused CUDA** kernel for the 3D **Local (squared) Normalized Cross-Correlation** loss, the
similarity metric used in deformable image registration and as a perceptual/structural loss.

It is **~3.5x faster and ~3x lighter on memory** than the previous fastest differentiable 3D LNCC
(FFDP, ICLR'26 Oral — the fused-kernel framework built on the FireANTs registration library),
~6-10x faster than MONAI, and ~16-18x faster than naive PyTorch, while
producing identical gradients. Verified on V100, A100, A40, and Blackwell.

## Install

```bash
pip install -e . --no-build-isolation     # needs an NVIDIA GPU and a CUDA toolchain (nvcc, gcc)
```

The wheel ships SASS for sm_70..90 plus sm_120 and JIT-compiles from PTX on newer GPUs, so it runs out
of the box on Volta through Blackwell. (CUDA only; no Apple/Metal or AMD/ROCm backend.)

Tested with **PyTorch 2.3 to 2.12** and **CUDA 11.8 to 13.x** (it is a source build, so it compiles
against your own torch + CUDA). One caveat: CUDA 13 dropped Volta, so a V100 needs a CUDA 12.x or
older toolkit (the build selects arches automatically).

## Usage

```python
from fused_lncc import fused_lncc_loss

loss = fused_lncc_loss(pred, target, kernel_size=7)   # pred, target: (N,C,D,H,W) CUDA -> scalar in [0,1]
loss.backward()                                        # gradient flows through `pred`
```

It is a standard `torch.autograd.Function`, so it drops into any training loop and composes with other
loss terms:

```python
for x, target in loader:
    opt.zero_grad()
    pred = model(x)
    loss = F.l1_loss(pred, target) + 0.5 * fused_lncc_loss(pred, target, kernel_size=7)
    loss.backward()
    opt.step()
```

`fp32` or `bf16`, odd `kernel_size` in {3,5,7,9}. Returns `1 - mean(local correlation)` (lower is
better). Values match MONAI's rectangular LNCC and a PyTorch reference to ~1e-7.

Notes: the gradient flows through `pred` only (`target` is treated as a fixed reference). Inputs must
be `fp32` or `bf16`, not `fp16`, so under `torch.autocast(dtype=float16)` cast first, e.g.
`fused_lncc_loss(pred.float(), target.float(), 7)` (bf16 autocast works directly).

## Performance

Speed and peak memory for the forward + backward step as the volume grows, against the common
alternatives:

![scaling](assets/scaling.png)

Headline at `(2,16,128³)`, A40, fp32, k=7 (`time / peak-VRAM`, lower is better):

| shape (N,C,D,H,W) | **fused_lncc** | FFDP (ICLR'26) | MONAI | naive (PyTorch) |
|---|---|---|---|---|
| (2,16,128³) | **24.5 ms / 1.07 GB** | 85.7 / 3.22 | 162.5 / 7.21 | 432.5 / 3.49 |

**3.5x faster and 3x less memory than the SOTA**, and the gap holds across V100/A100/A40/Blackwell. At
high resolution the memory advantage becomes an OOM boundary: at 256³, fused_lncc runs in ~13 GB while
the baselines need 30-61 GB and run out of memory on a 24 GB card.

Full benchmarks, the four-GPU comparison, the memory/OOM envelope, and end-to-end registration are in
**[BENCHMARKS.md](BENCHMARKS.md)**.

## Why it's fast

LNCC needs, at every voxel, five local box-sums (`Σp, Σt, Σp², Σt², Σpt`) over a `k³` window, then a
per-window correlation. Baselines run five separate convolutions and materialize every intermediate;
even the prior fused kernel still routes the convolutions through cuDNN. We pull the **whole
computation into one shared-memory-tiled kernel**, in both directions:

- **Forward:** each block loads its tile once and computes all five statistics, the correlation, and
  the backward adjoints in a single pass, so the intermediates never touch global memory.
- **Backward:** a single analytic kernel, `dloss/dp = -(1/M)·(box(A) + 2p·box(B) + t·box(C))`, with no
  autograd tape, which is where the ~3x memory saving comes from.
- **fp32 accumulation** guards the variance's sum-of-squares cancellation; degenerate windows clamp to
  keep the loss finite and in `[0,1]`.

## GPU support

Verified on **V100 (sm_70), A100 (sm_80), A40 (sm_86), and Blackwell RTX PRO 6000 (sm_120)**. The same
wheel ran on each (no rebuild on V100/A100; PTX-JIT on Blackwell), with all tests and compute-sanitizer
clean on fp32 and bf16. The full per-arch matrix and caveats (including Turing's shared-memory limit)
are in **[BENCHMARKS.md](BENCHMARKS.md#gpu-support)**.

## Acknowledgments

- **[fused-ssim](https://github.com/rahul-goel/fused-ssim)** (Rahul Goel): this project is directly
  inspired by it. SSIM and LNCC are the same shape of computation (local statistics via a separable
  windowed convolution plus a per-window formula), and the shared-memory-tiling and fused-backward
  design here mirrors fused-ssim's, applied to the box-window LNCC.
- **[FFDP](https://arxiv.org/abs/2511.09173)** (Jena et al., ICLR'26 Oral): the prior fused 3D LNCC
  kernel and the analytic-backward idea; used here as the primary speed/memory baseline. FFDP is the
  fused-kernel framework built on the [FireANTs](https://github.com/rohitrango/fireants) registration
  library (Jena et al., *Nature Communications*).
- **[MONAI](https://github.com/Project-MONAI/MONAI)** `LocalNormalizedCrossCorrelationLoss`: the
  reference semantics we value-match against.

## License

MIT, see [LICENSE](LICENSE).
