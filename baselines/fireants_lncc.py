"""Minimal standalone wrapper around FireANTs' fused LNCC kernel (fireants_fused_ops).

Replicates the rectangular-box path of fireants/losses/fusedcc.py::FusedNCC3d, depending
ONLY on `fireants_fused_ops` (+ torch) — drops the full FireANTs framework, the Gaussian
variant, masking, the ants-gradient shortcut, and the >2^31 numel fallback (our shapes are
well under MAX_INT32). For evaluation/benchmarking only.

Semantics: returns the MONAI-convention loss = -mean(ncc) where ncc = cross^2/(var_p*var_t)
is the squared local Pearson correlation. So our `1 - mean(ncc)` == 1 + fused_out.
"""
import torch
import torch.nn.functional as F
import fireants_fused_ops as ffo

_RED = {"none": ffo.Reduction.NONE, "sum": ffo.Reduction.SUM, "mean": ffo.Reduction.MEAN}


class FusedNCC3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_img, target_img, kernel_size, nr, dr, reduction, use_separable):
        reduction = _RED[reduction.lower()]
        B, C, H, W, D = input_img.shape
        assert input_img.is_contiguous() and target_img.is_contiguous()
        interm = torch.zeros(B, 5 * C, H, W, D, device=input_img.device, dtype=input_img.dtype)
        ffo.create_intermediates(input_img, target_img, interm)
        kernel_vol = kernel_size ** 3
        if use_separable:
            f1 = torch.ones(5 * C, 1, kernel_size, 1, 1, device=input_img.device, dtype=input_img.dtype) / kernel_size
            f2 = torch.ones(5 * C, 1, 1, kernel_size, 1, device=input_img.device, dtype=input_img.dtype) / kernel_size
            f3 = torch.ones(5 * C, 1, 1, 1, kernel_size, device=input_img.device, dtype=input_img.dtype) / kernel_size
            interm = F.conv3d(interm, f1, padding="same", stride=1, groups=interm.shape[1])
            interm = F.conv3d(interm, f2, padding="same", stride=1, groups=interm.shape[1])
            interm = F.conv3d(interm, f3, padding="same", stride=1, groups=interm.shape[1])
        else:
            avg = torch.ones(5 * C, 1, kernel_size, kernel_size, kernel_size, device=input_img.device, dtype=input_img.dtype) / kernel_vol
            interm = F.conv3d(interm, avg, padding=(kernel_size - 1) // 2, stride=1, groups=interm.shape[1])
        out = ffo.cc3d_fwd_interm_v1(interm, int(kernel_vol), reduction, nr, dr)
        ctx.save_for_backward(interm, input_img, target_img, out)
        ctx.kernel_size, ctx.nr, ctx.dr, ctx.reduction, ctx.use_separable = kernel_size, nr, dr, reduction, use_separable
        return out

    @staticmethod
    def backward(ctx, grad_output):
        interm, input_img, target_img, out = ctx.saved_tensors
        k, nr, dr, reduction, use_separable = ctx.kernel_size, ctx.nr, ctx.dr, ctx.reduction, ctx.use_separable
        B, C, H, W, D = input_img.shape
        kernel_vol = k ** 3
        grad_input = torch.zeros(B, C, H, W, D, device=input_img.device, dtype=input_img.dtype) if input_img.requires_grad else None
        grad_target = torch.zeros(B, C, H, W, D, device=input_img.device, dtype=input_img.dtype) if target_img.requires_grad else None
        ffo.cc3d_bwd_modify_interm_v1(interm, input_img, target_img, grad_output, grad_input, grad_target, k, kernel_vol, nr, dr, reduction)
        pad = (k - 1) // 2
        # scatter the modified intermediates back via the same box conv (gradient-through-conv-is-conv)
        nch = 5 * C if grad_target is not None else 3 * C  # if only input needs grad, only first 3C channels matter
        if use_separable:
            f1 = torch.ones(nch, 1, k, 1, 1, device=input_img.device, dtype=input_img.dtype) / k
            f2 = torch.ones(nch, 1, 1, k, 1, device=input_img.device, dtype=input_img.dtype) / k
            f3 = torch.ones(nch, 1, 1, 1, k, device=input_img.device, dtype=input_img.dtype) / k
            interm[:, :nch] = F.conv3d(interm[:, :nch], f1, padding="same", stride=1, groups=nch)
            interm[:, :nch] = F.conv3d(interm[:, :nch], f2, padding="same", stride=1, groups=nch)
            interm[:, :nch] = F.conv3d(interm[:, :nch], f3, padding="same", stride=1, groups=nch)
        else:
            avg = torch.ones(nch, 1, k, k, k, device=input_img.device, dtype=input_img.dtype) / kernel_vol
            interm[:, :nch] = F.conv3d(interm[:, :nch], avg, padding=pad, stride=1, groups=nch)
        ffo.cc3d_bwd_compute_grads(interm, input_img, target_img, grad_input, grad_target)
        return grad_input, grad_target, None, None, None, None, None


def fused_lncc3d(pred, target, kernel_size=7, smooth_nr=0.0, smooth_dr=1e-5, reduction="mean", use_separable=True):
    """Fused LNCC loss (FireANTs kernel). Returns MONAI-convention -mean(ncc).
    `1 - mean(ncc)` (our _ncc_loss convention) == 1 + fused_lncc3d(...)."""
    pred = pred.contiguous()
    target = target.contiguous()
    return FusedNCC3d.apply(pred, target, kernel_size, smooth_nr, smooth_dr, reduction, use_separable)
