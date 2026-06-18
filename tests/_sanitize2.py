import os
import torch, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fused_lncc import fused_lncc_loss
# B=N*C>1 + odd/non-tile-divisible W exercise the new grid.x batch-folding + edges
for shp in [(3,5,13,17,11), (2,4,16,16,16), (1,1,2,2,2)]:
    p = torch.randn(*shp, device="cuda", requires_grad=True); t = torch.randn_like(p)
    fused_lncc_loss(p, t, 7).backward()
torch.cuda.synchronize(); print("sanitize2 done")
