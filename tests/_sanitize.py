import os
import torch, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fused_lncc import fused_lncc_loss
# small shapes that exercise borders + non-tile-divisible dims + backward
for shp in [(1,2,16,16,16), (1,1,13,17,11), (1,1,2,2,2)]:
    p = torch.randn(*shp, device="cuda", requires_grad=True); t = torch.randn_like(p)
    fused_lncc_loss(p, t, 7).backward()
torch.cuda.synchronize(); print("sanitize workload done")
