import os
"""Robustness / no-crash battery + regression check after variance hardening."""
import sys, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baselines"))
from fused_lncc import fused_lncc_loss
from torch_lncc import lncc_dense

dev = "cuda"; ok = True
def chk(name, cond):
    global ok; ok &= bool(cond); print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

print("=== regression: fused == torch reference on well-conditioned (after hardening) ===")
torch.manual_seed(0)
for k in [3, 5, 7, 9]:
    p = torch.randn(2, 4, 28, 28, 28, device=dev, requires_grad=True); t = torch.randn(2, 4, 28, 28, 28, device=dev)
    lf = fused_lncc_loss(p, t, k); lf.backward(); gf = p.grad.clone(); p.grad = None
    lr = lncc_dense(p, t, k); lr.backward(); gr = p.grad.clone(); p.grad = None
    chk(f"k={k} loss & grad match ref", abs(lf.item()-lr.item()) < 1e-4 and
        F.cosine_similarity(gf.flatten(), gr.flatten(), dim=0).item() > 0.9999)

print("=== degenerate inputs: finite, in [0,1], no crash, grad finite ===")
def degen(name, p, t):
    p = p.clone().requires_grad_(True)
    l = fused_lncc_loss(p, t, 7); l.backward()
    inb = 0.0 <= l.item() <= 1.0001
    chk(f"{name}: loss={l.item():.4f} in[0,1]={inb} finite_grad={torch.isfinite(p.grad).all().item()}",
        torch.isfinite(l) and inb and torch.isfinite(p.grad).all())
degen("all-zero",        torch.zeros(2, 2, 32, 32, 32, device=dev), torch.zeros(2, 2, 32, 32, 32, device=dev))
degen("constant=5",      torch.full((2, 2, 32, 32, 32), 5.0, device=dev), torch.full((2, 2, 32, 32, 32), 5.0, device=dev))
degen("constant hi=1e4", torch.full((1, 1, 32, 32, 32), 1e4, device=dev), torch.full((1, 1, 32, 32, 32), 1e4, device=dev))
degen("near-const",      torch.full((1, 2, 24, 24, 24), 3.0, device=dev) + 1e-3*torch.randn(1, 2, 24, 24, 24, device=dev),
                         torch.full((1, 2, 24, 24, 24), 3.0, device=dev))

print("=== shape edge cases: no crash, match ref ===")
for shp in [(1, 1, 2, 2, 2), (1, 1, 2, 40, 40), (1, 3, 7, 9, 11), (3, 5, 13, 17, 19), (1, 1, 128, 1, 1)]:
    try:
        p = torch.randn(*shp, device=dev); t = torch.randn(*shp, device=dev)
        a = fused_lncc_loss(p, t, 7); b = lncc_dense(p, t, 7)
        chk(f"shape {shp}: |fused-ref|={abs(a.item()-b.item()):.2e}", abs(a.item()-b.item()) < 1e-3)
    except Exception as e:
        chk(f"shape {shp} CRASHED: {str(e)[:40]}", False)

print("=== unsupported dtype -> clean error, not crash (fp32/bf16 are supported) ===")
try:
    p = torch.randn(1, 1, 16, 16, 16, device=dev, dtype=torch.float64); t = torch.randn_like(p)
    fused_lncc_loss(p, t, 7); chk("float64 should have raised", False)
except Exception:
    chk("float64 raises cleanly (only fp32/bf16 supported)", True)

print("=== larger shape (no OOM/crash) ===")
try:
    p = torch.randn(2, 32, 128, 128, 128, device=dev, requires_grad=True); t = torch.randn_like(p)
    fused_lncc_loss(p, t, 7).backward()
    chk("(2,32,128^3) fwd+bwd ok, grad finite", torch.isfinite(p.grad).all().item())
    del p, t; torch.cuda.empty_cache()
except Exception as e:
    chk(f"large shape CRASHED: {str(e)[:40]}", False)

print("\n" + ("ALL PASS" if ok else "SOME FAIL"))
