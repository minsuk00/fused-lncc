"""Fused 3D Local (squared) Normalized Cross-Correlation loss.

A fully-fused CUDA forward (5 box-sums + ncc + backward adjoints in one shared-memory pass)
and an analytic backward (adjoint scatter via the same tiled box-sum). Drop-in similarity in
[0,1]; loss convention `1 - mean(ncc)`. fp32, 3D, sm_86 (A40).
"""
import os
import torch

try:                                  # prefer the prebuilt extension (pip install .)
    import fused_lncc_cuda as _ext     # JIT fallback below auto-detects the running GPU's arch
except ImportError:                   # fall back to JIT compile from source
    from torch.utils.cpp_extension import load as _load
    _ext = _load(
        name="fused_lncc_cuda",
        sources=[os.path.join(os.path.dirname(__file__), "csrc", "lncc3d.cu")],  # ships inside the package
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


# Wrap the two opaque CUDA-kernel calls as torch custom ops so the loss is traceable by
# torch.compile / TorchDynamo. Without this, `_ext.*` is a "function marked as skipped" and
# forces a graph break on every call. The wrappers are numerically identical to calling
# `_ext` directly (verified bit-exact on both loss and gradient).
@torch.library.custom_op("fused_lncc::forward", mutates_args=())
def _lncc_forward_op(pf: torch.Tensor, tf: torch.Tensor, kernel_size: int, smooth_dr: float) -> list[torch.Tensor]:
    return [t.contiguous() for t in _ext.lncc_forward(pf, tf, kernel_size, smooth_dr)]


@_lncc_forward_op.register_fake
def _(pf, tf, kernel_size, smooth_dr):
    # ncc, A, B, C are ALWAYS fp32 (kernel forces fp32 outputs even for bf16 input) — the fake
    # MUST match, else torch.compile propagates the input dtype downstream and the bf16 path NaNs.
    return [torch.empty_like(pf, dtype=torch.float32) for _ in range(4)]  # all (N*C, D, H, W)


@torch.library.custom_op("fused_lncc::backward", mutates_args=())
def _lncc_backward_op(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, pf: torch.Tensor,
                      tf: torch.Tensor, kernel_size: int, scale: float) -> torch.Tensor:
    return _ext.lncc_backward(A, B, C, pf, tf, kernel_size, scale).contiguous()


@_lncc_backward_op.register_fake
def _(A, B, C, pf, tf, kernel_size, scale):
    return torch.empty_like(pf)


class _FusedLNCC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, target, kernel_size, smooth_dr):
        N, C = pred.shape[:2]
        S = pred.shape[2:]
        pf = pred.reshape(N * C, *S).contiguous()
        tf = target.reshape(N * C, *S).contiguous()
        ncc, A, Bc, Cc = torch.ops.fused_lncc.forward(pf, tf, kernel_size, smooth_dr)  # ncc/A/B/C fp32
        ctx.save_for_backward(pf, tf, A, Bc, Cc)                             # pf,tf keep input dtype
        ctx.k = kernel_size
        ctx.M = ncc.numel()
        ctx.bc = (N, C) + tuple(S)
        ctx.dtype = pred.dtype
        return 1.0 - ncc.mean()

    @staticmethod
    def backward(ctx, grad_out):
        pf, tf, A, Bc, Cc = ctx.saved_tensors
        # dloss/dp = -grad_out/M * (box(A) + 2p*box(B) + t*box(Cc)) — fully fused in one kernel.
        # Call with scale=1 then apply -grad_out/M as a TENSOR multiply (no .item() host-sync)
        # so the backward stays traceable. The kernel is linear in `scale`, so this equals the old
        # scale=(-grad_out/M).item() path: bit-exact for fp32 (and for bf16 when M is a power of two,
        # where the scale is an exact power of two). For bf16 with non-pow2 M the box-sum is rounded
        # before scaling instead of after, differing at the ULP level (~5e-3 rel; grads still correct).
        grad_p = torch.ops.fused_lncc.backward(A, Bc, Cc, pf, tf, ctx.k, 1.0) * (-grad_out / ctx.M)
        return grad_p.reshape(*ctx.bc).to(ctx.dtype), None, None, None


def fused_lncc_loss(pred, target, kernel_size=7, smooth_dr=1e-5):
    """1 - mean(squared local NCC). Gradient flows through `pred` only.

    pred, target: (N, C, D, H, W) fp32 CUDA. kernel_size odd. Returns a scalar loss in [0,1]."""
    assert pred.dim() == 5 and pred.shape == target.shape, "expect matching (N,C,D,H,W)"
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    return _FusedLNCC.apply(pred.contiguous(), target.contiguous(), kernel_size, smooth_dr)
