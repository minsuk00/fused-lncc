import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fused_lncc import fused_lncc_loss
for dt in [torch.float32, torch.bfloat16]:
    for shp in [(3,5,13,17,11), (1,2,16,16,16)]:   # B>1 + odd W + both dtypes
        p = torch.randn(*shp, device="cuda", dtype=dt, requires_grad=True); t = torch.randn_like(p)
        fused_lncc_loss(p, t, 7).backward()
torch.cuda.synchronize(); print("sanitize-bf16 done")
