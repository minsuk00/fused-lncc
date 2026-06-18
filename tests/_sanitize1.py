import os
import torch, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fused_lncc import fused_lncc_loss
p = torch.randn(1,2,16,16,16, device="cuda", requires_grad=True); t = torch.randn_like(p)
fused_lncc_loss(p, t, 7).backward()
torch.cuda.synchronize(); print("racecheck workload done")
