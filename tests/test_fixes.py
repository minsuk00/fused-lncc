import os
import sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fused_lncc import fused_lncc_loss
dev="cuda"; ok=True
def chk(n,c):
    global ok; ok &= bool(c); print(f"  [{'PASS' if c else 'FAIL'}] {n}")

print("=== gridDim FIX: large N*C at D=128 (was launch-error >65535) ===")
try:
    p=torch.randn(2080,1,128,8,8,device=dev,requires_grad=True); t=torch.randn_like(p)  # N*C=2080>2047
    l=fused_lncc_loss(p,t,7); l.backward()
    chk(f"N*C=2080,D=128 runs (loss={l.item():.4f}, grad finite={torch.isfinite(p.grad).all().item()})",
        torch.isfinite(l) and torch.isfinite(p.grad).all())
    del p,t; torch.cuda.empty_cache()
except Exception as e:
    chk(f"large N*C CRASHED: {str(e)[:50]}", False)

print("=== NaN FIX: NaN/Inf input -> VISIBLE non-finite loss (not silently masked) ===")
t=torch.randn(1,1,16,16,16,device=dev)
for name,bad in [("NaN",float('nan')),("Inf",float('inf'))]:
    p=torch.randn(1,1,16,16,16,device=dev); p[0,0,8,8,8]=bad; p=p.requires_grad_(True)
    l=fused_lncc_loss(p,t,7)
    chk(f"{name} input -> loss non-finite (visible): loss={l.item()}", not torch.isfinite(l))

print("=== REGRESSION: clean input still correct & in [0,1] ===")
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baselines")); from torch_lncc import lncc_dense
for k in [3,5,7,9]:
    p=torch.randn(2,8,40,40,40,device=dev,requires_grad=True); t=torch.randn_like(p)
    lf=fused_lncc_loss(p,t,k); lf.backward(); gf=p.grad.clone(); p.grad=None
    lr=lncc_dense(p,t,k); lr.backward(); gr=p.grad.clone(); p.grad=None
    cos=F.cosine_similarity(gf.flatten(),gr.flatten(),dim=0).item()
    chk(f"k={k} loss&grad match ref (|dl|={abs(lf.item()-lr.item()):.1e}, cos={cos:.6f})",
        abs(lf.item()-lr.item())<1e-4 and cos>0.9999 and 0<=lf.item()<=1)
print("\n"+("ALL PASS" if ok else "SOME FAIL"))
