import os
"""Verify the fused analytic backward vs the torch reference's autograd gradient."""
import sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fused_lncc import fused_lncc_loss

dev = "cuda"

def ref_loss(p, t, k=7, dr=1e-5):
    C = p.shape[1]; n = k ** 3; pad = k // 2
    box = lambda x: F.conv3d(x, x.new_ones(C, 1, k, k, k), padding=pad, groups=C)
    Sp, St = box(p), box(t)
    cross = box(p * t) - Sp * St / n
    var_p = box(p * p) - Sp * Sp / n + dr
    var_t = box(t * t) - St * St / n + dr
    ncc = ((cross * cross) / (var_p * var_t)).clamp(max=1.0)
    return 1.0 - ncc.mean()

print("=== backward: fused analytic vs torch-reference autograd ===")
torch.manual_seed(0)
for k in [3, 5, 7]:
    for shp in [(2, 4, 32, 32, 32), (1, 8, 24, 40, 32)]:
        p = torch.randn(*shp, device=dev, requires_grad=True)
        t = torch.randn(*shp, device=dev)
        fused_lncc_loss(p, t, k).backward(); gf = p.grad.clone(); p.grad = None
        ref_loss(p, t, k).backward(); gr = p.grad.clone(); p.grad = None
        cos = F.cosine_similarity(gf.flatten(), gr.flatten(), dim=0).item()
        rel = (gf - gr).abs().mean().item() / (gr.abs().mean().item() + 1e-12)
        ok = cos > 0.9999 and rel < 1e-3
        print(f"  k={k} {str(shp):20s} cos={cos:.6f} rel|gf-gr|={rel:.2e}  {'OK' if ok else 'FAIL'}")

# also confirm loss value forward matches through the autograd.Function
p = torch.rand(2, 4, 40, 40, 40, device=dev); t = torch.rand(2, 4, 40, 40, 40, device=dev)
print(f"loss fused={fused_lncc_loss(p,t,7).item():.6f}  ref={ref_loss(p,t,7).item():.6f}")
