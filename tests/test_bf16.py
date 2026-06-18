import os
import sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "baselines"))
from fused_lncc import fused_lncc_loss
from torch_lncc import lncc_dense
dev="cuda"; ok=True
def chk(n,c):
    global ok; ok&=bool(c); print(f"  [{'PASS' if c else 'FAIL'}] {n}")
torch.manual_seed(0)
print("=== bf16 forward matches fp32-of-same-values (internal math is fp32) ===")
for k in [3,7]:
    xb=torch.randn(2,4,40,40,40,device=dev,dtype=torch.bfloat16); tb=torch.randn(2,4,40,40,40,device=dev,dtype=torch.bfloat16)
    lb=fused_lncc_loss(xb,tb,k); lf=fused_lncc_loss(xb.float(),tb.float(),k)
    chk(f"k={k} bf16 loss vs fp32(bf16-vals): |d|={abs(lb.item()-lf.item()):.2e}", abs(lb.item()-lf.item())<1e-3)
print("=== bf16 backward: grad is bf16, finite, ~matches fp32 grad ===")
xb=torch.randn(2,4,32,32,32,device=dev,dtype=torch.bfloat16,requires_grad=True); tb=torch.randn(2,4,32,32,32,device=dev,dtype=torch.bfloat16)
lb=fused_lncc_loss(xb,tb,7); lb.backward(); gb=xb.grad.clone()
xf=xb.detach().float().requires_grad_(True); lf=fused_lncc_loss(xf,tb.float(),7); lf.backward(); gf=xf.grad
rel=(gb.float()-gf).abs().mean().item()/(gf.abs().mean().item()+1e-12)
chk(f"grad dtype=bf16: {gb.dtype==torch.bfloat16}", gb.dtype==torch.bfloat16)
chk(f"grad finite & ~fp32 (rel={rel:.2e} < 2e-2)", torch.isfinite(gb).all().item() and rel<2e-2)
print("=== regression: fp32 still exact vs reference ===")
import torch.nn.functional as F
for k in [3,5,7,9]:
    p=torch.randn(2,8,32,32,32,device=dev,requires_grad=True); t=torch.randn_like(p)
    lf=fused_lncc_loss(p,t,k); lf.backward(); gfp=p.grad.clone(); p.grad=None
    lr=lncc_dense(p,t,k); lr.backward(); gr=p.grad.clone(); p.grad=None
    chk(f"k={k} fp32 exact (|dl|={abs(lf.item()-lr.item()):.1e}, cos={F.cosine_similarity(gfp.flatten(),gr.flatten(),dim=0).item():.6f})",
        abs(lf.item()-lr.item())<1e-4 and F.cosine_similarity(gfp.flatten(),gr.flatten(),dim=0).item()>0.9999)
print("\n"+("ALL PASS" if ok else "SOME FAIL"))
