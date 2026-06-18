"""Assert-based pytest suite for CI. (The script-style tests/*.py have richer diagnostics;
this file is the machine-checkable subset.) conftest.py puts the repo root + baselines on sys.path."""
import pytest
import torch
import torch.nn.functional as F
from fused_lncc import fused_lncc_loss
from torch_lncc import lncc_dense

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@cuda
@pytest.mark.parametrize("k", [3, 5, 7, 9])
def test_forward_matches_reference(k):
    torch.manual_seed(0)
    p = torch.randn(2, 4, 32, 32, 32, device="cuda"); t = torch.randn_like(p)
    assert abs(fused_lncc_loss(p, t, k).item() - lncc_dense(p, t, k).item()) < 1e-4


@cuda
@pytest.mark.parametrize("k", [3, 7])
def test_backward_matches_reference(k):
    torch.manual_seed(0)
    p = torch.randn(2, 4, 32, 32, 32, device="cuda", requires_grad=True); t = torch.randn_like(p)
    fused_lncc_loss(p, t, k).backward(); gf = p.grad.clone(); p.grad = None
    lncc_dense(p, t, k).backward(); gr = p.grad
    assert F.cosine_similarity(gf.flatten(), gr.flatten(), dim=0).item() > 0.9999


@cuda
def test_bf16_runs():
    p = torch.randn(1, 2, 16, 16, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    t = torch.randn_like(p)
    fused_lncc_loss(p, t, 7).backward()
    assert p.grad.dtype == torch.bfloat16 and torch.isfinite(p.grad).all()


@cuda
@pytest.mark.parametrize("k", [3, 5, 7, 9])
def test_fused_backward_matches_scatter(k):
    """The fused backward kernel must equal the box-sum scatter formula it replaced (identical math)."""
    import fused_lncc_cuda as ext
    torch.manual_seed(0)
    p = torch.randn(2, 3, 24, 20, 28, device="cuda"); t = torch.randn_like(p)
    pf = p.reshape(6, 24, 20, 28).contiguous(); tf = t.reshape(6, 24, 20, 28).contiguous()
    ncc, A, Bc, Cc = ext.lncc_forward(pf, tf, k, 1e-5)
    sc = -1.0 / ncc.numel()
    box = lambda x: ext.box3d_sep_forward(x.contiguous(), k)
    ref = sc * (box(A) + 2.0 * pf * box(Bc) + tf * box(Cc))     # old scatter path
    got = ext.lncc_backward(A, Bc, Cc, pf, tf, k, sc)           # fused kernel
    assert (ref - got).abs().max().item() < 1e-8


@cuda
def test_degenerate_in_range():
    p = torch.zeros(1, 1, 16, 16, 16, device="cuda"); t = torch.zeros_like(p)
    l = fused_lncc_loss(p, t, 7)
    assert torch.isfinite(l) and 0.0 <= l.item() <= 1.001


@cuda
def test_unsupported_dtype_raises():
    p = torch.randn(1, 1, 16, 16, 16, device="cuda", dtype=torch.float64); t = torch.randn_like(p)
    with pytest.raises(Exception):
        fused_lncc_loss(p, t, 7)
