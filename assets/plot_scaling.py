"""Scaling plot (fused-ssim style): vs number of voxels (millions), at fixed N,C.
Two panels for the forward+backward step: speed (CUDA-event median ms) and peak VRAM (GB).
Re-measures on whatever GPU it runs on."""
import os, sys, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..")); sys.path.insert(0, os.path.join(_here, "..", "baselines"))
from fused_lncc import fused_lncc_loss
from torch_lncc import lncc_dense, lncc_separable
from monai.losses import LocalNormalizedCrossCorrelationLoss as MLNCC
dev = "cuda"; K = 7; DR = 1e-5; N, C = 1, 8
_monai = MLNCC(spatial_dims=3, kernel_size=K, kernel_type="rectangular", smooth_nr=0, smooth_dr=DR, reduction="mean").to(dev)
FNS = {
    "fused_lncc (ours)":  ("#2ca02c", lambda p, t: fused_lncc_loss(p, t, K, DR)),
    "MONAI":              ("#ff7f0e", lambda p, t: _monai(p, t)),
    "naive (PyTorch)":    ("#7f7f7f", lambda p, t: lncc_dense(p, t, K, DR)),
}
try:  # optional SOTA baseline: pip install fireants_fused_ops
    from fireants_lncc import fused_lncc3d as fa
    FNS["FireANTs (ICLR'26)"] = ("#1f77b4", lambda p, t: 1.0 - fa(p, t, kernel_size=K, smooth_dr=DR))
except Exception:
    pass
try:  # separable box-sum + torch.compile (best pure-PyTorch approach)
    _sep = torch.compile(lncc_separable)
    FNS["sep+compile"] = ("#9467bd", lambda p, t: _sep(p, t, K, DR))
except Exception:
    pass
sizes = [32, 48, 64, 80, 96, 112, 128, 144]
mvox = [N * C * s**3 / 1e6 for s in sizes]

def t_train(fn, s, n=30):
    p = torch.randn(N, C, s, s, s, device=dev, requires_grad=True); t = torch.randn(N, C, s, s, s, device=dev)
    st = lambda i: (setattr(p, "grad", None), e0[i].record(), fn(p, t).backward(), e1[i].record())
    e0 = [torch.cuda.Event(enable_timing=True) for _ in range(n)]; e1 = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for _ in range(10): setattr(p, "grad", None); fn(p, t).backward()
    torch.cuda.synchronize()
    for i in range(n): st(i)
    torch.cuda.synchronize(); return sorted(e0[i].elapsed_time(e1[i]) for i in range(n))[n // 2]

def m_train(fn, s):  # total peak VRAM (GB) for one fwd+bwd from a clean slate
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    p = torch.randn(N, C, s, s, s, device=dev, requires_grad=True); t = torch.randn(N, C, s, s, s, device=dev)
    for _ in range(2):
        if p.grad is not None: p.grad = None
        fn(p, t).backward()
    torch.cuda.reset_peak_memory_stats(); p.grad = None; fn(p, t).backward()
    pk = torch.cuda.max_memory_allocated() / 1e9
    del p, t; torch.cuda.empty_cache(); return pk

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for name, (col, fn) in FNS.items():
    tr, mem = [], []
    for s in sizes:
        try: tr.append(t_train(fn, s)); mem.append(m_train(fn, s))
        except Exception: tr.append(float("nan")); mem.append(float("nan"))
    ax[0].plot(mvox, tr, "o-", color=col, label=name)
    ax[1].plot(mvox, mem, "o-", color=col, label=name)
    print(f"{name:20s} train@{sizes[-1]}³={tr[-1]:.1f}ms  peak={mem[-1]:.2f}GB")
ax[0].set_ylabel("time / training step (ms)"); ax[0].set_title("Speed (forward + backward)")
ax[1].set_ylabel("peak memory (GB)"); ax[1].set_title("Memory (forward + backward)")
for a in ax:
    a.set_xlabel("number of voxels (millions)"); a.grid(alpha=0.3); a.legend(fontsize=8)
fig.suptitle("Fused LNCC scaling (NVIDIA A100, fp32, k=7)", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(_here, "scaling.png"), dpi=130, bbox_inches="tight")
print("wrote assets/scaling.png")
