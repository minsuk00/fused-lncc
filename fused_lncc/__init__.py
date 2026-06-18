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


class _FusedLNCC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pred, target, kernel_size, smooth_dr):
        N, C = pred.shape[:2]
        S = pred.shape[2:]
        pf = pred.reshape(N * C, *S).contiguous()
        tf = target.reshape(N * C, *S).contiguous()
        ncc, A, Bc, Cc = _ext.lncc_forward(pf, tf, kernel_size, smooth_dr)   # ncc/A/B/C are fp32
        ctx.save_for_backward(pf, tf, A, Bc, Cc)                             # pf,tf keep input dtype
        ctx.k = kernel_size
        ctx.M = ncc.numel()
        ctx.bc = (N, C) + tuple(S)
        ctx.dtype = pred.dtype
        return 1.0 - ncc.mean()

    @staticmethod
    def backward(ctx, grad_out):
        pf, tf, A, Bc, Cc = ctx.saved_tensors
        # dloss/dp = -grad_out/M * (box(A) + 2p*box(B) + t*box(Cc)) — fully fused in one kernel
        # (the 3 box-sums + the elementwise combine; identical math to the scatter version).
        scale = (-grad_out / ctx.M).item()   # grad_out is a scalar (loss is scalar)
        grad_p = _ext.lncc_backward(A, Bc, Cc, pf, tf, ctx.k, scale)   # returned in input dtype
        return grad_p.reshape(*ctx.bc).to(ctx.dtype), None, None, None


def fused_lncc_loss(pred, target, kernel_size=7, smooth_dr=1e-5):
    """1 - mean(squared local NCC). Gradient flows through `pred` only.

    pred, target: (N, C, D, H, W) fp32 CUDA. kernel_size odd. Returns a scalar loss in [0,1]."""
    assert pred.dim() == 5 and pred.shape == target.shape, "expect matching (N,C,D,H,W)"
    assert kernel_size % 2 == 1, "kernel_size must be odd"
    return _FusedLNCC.apply(pred.contiguous(), target.contiguous(), kernel_size, smooth_dr)
