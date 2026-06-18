"""Pure-PyTorch LNCC reference implementations (dense and separable box-sum).

Squared local Pearson NCC, MONAI/`_ncc_loss`-compatible:
  cross=Spt-Sp*St/n ; var=Sxx-Sx^2/n ; ncc=cross^2/((var_p+dr)(var_t+dr)) clamped to [0,1]
Loss convention: 1 - mean(ncc). These are the autograd-differentiable oracles the fused
CUDA kernel is verified against.
"""
import torch
import torch.nn.functional as F


def _box_dense(x, k):
    C = x.shape[1]
    return F.conv3d(x, x.new_ones(C, 1, k, k, k), padding=k // 2, groups=C)


def _box_separable(x, k):
    C = x.shape[1]
    o = x
    for sh in [(C, 1, k, 1, 1), (C, 1, 1, k, 1), (C, 1, 1, 1, k)]:
        o = F.conv3d(o, x.new_ones(*sh), padding="same", groups=C)
    return o


def _lncc(p, t, k, dr, box):
    n = k ** 3
    Sp, St = box(p, k), box(t, k)
    cross = box(p * t, k) - Sp * St / n
    var_p = box(p * p, k) - Sp * Sp / n + dr
    var_t = box(t * t, k) - St * St / n + dr
    ncc = ((cross * cross) / (var_p * var_t)).clamp(max=1.0)
    return 1.0 - ncc.mean()


def lncc_dense(pred, target, kernel_size=7, smooth_dr=1e-5):
    """Dense box-sum (single k^3 grouped conv) LNCC loss."""
    return _lncc(pred, target, kernel_size, smooth_dr, _box_dense)


def lncc_separable(pred, target, kernel_size=7, smooth_dr=1e-5):
    """Separable box-sum (3x 1D grouped conv) LNCC loss."""
    return _lncc(pred, target, kernel_size, smooth_dr, _box_separable)
