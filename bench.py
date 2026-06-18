"""Benchmark fused_lncc vs baselines: the FireANTs fused kernel, a separable box-sum
plus torch.compile, MONAI, and naive PyTorch. Forward+backward wall-clock and peak VRAM
across a few feature shapes. All as a loss (1 - mean(ncc)); gradient through `pred`."""
import os, sys, torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "baselines"))
from fused_lncc import fused_lncc_loss
from torch_lncc import lncc_dense
from monai.losses import LocalNormalizedCrossCorrelationLoss as MLNCC

dev = "cuda"; K = 7; DR = 1e-5
_monai = MLNCC(spatial_dims=3, kernel_size=K, kernel_type="rectangular", smooth_nr=0, smooth_dr=DR, reduction="mean").to(dev)

# Core contenders (self-contained): ours, MONAI (common library), naive PyTorch.
FNS = {
    "fused_lncc (ours)":   lambda p, t: fused_lncc_loss(p, t, K, DR),
    "MONAI":               lambda p, t: _monai(p, t),
    "naive (PyTorch)":     lambda p, t: lncc_dense(p, t, K, DR),
}
# Optional SOTA baseline: FireANTs fused kernel (pip install fireants_fused_ops).
try:
    from fireants_lncc import fused_lncc3d as _fa
    FNS["FireANTs fused"] = lambda p, t: 1.0 - _fa(p, t, kernel_size=K, smooth_dr=DR)
except Exception:
    print("(skip FireANTs baseline: fireants_fused_ops not importable)")
# sep+compile baseline: a separable box-sum LNCC + torch.compile (the best pure-PyTorch approach).
try:
    from torch_lncc import lncc_separable
    _sep = torch.compile(lncc_separable)
    FNS["sep+compile"] = lambda p, t: _sep(p, t, K, DR)
except Exception:
    print("(skip sep+compile baseline: torch.compile unavailable)")

def run(fn, p, t, n=40):
    """CUDA-event-timed median over n iters on PAIRED inputs; peak VRAM over a clean step."""
    def step():
        if p.grad is not None: p.grad = None
        fn(p, t).backward()
    for _ in range(12): step()                      # warmup: torch.compile + GPU DVFS clocks
    torch.cuda.synchronize(); p.grad = None; torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2 * n)]
    for i in range(n):
        if p.grad is not None: p.grad = None
        ev[2 * i].record(); fn(p, t).backward(); ev[2 * i + 1].record()
    torch.cuda.synchronize()
    times = sorted(ev[2 * i].elapsed_time(ev[2 * i + 1]) for i in range(n))
    pk = (torch.cuda.max_memory_allocated() - base) / 1e9
    return times[n // 2], pk                         # median ms, peak GB

print(f"fwd+bwd, fp32, k={K}, CUDA-event median over 40 iters, paired inputs\n")
for s in [(1, 256, 16, 16, 16), (2, 64, 64, 64, 64), (2, 16, 128, 128, 128)]:
    torch.manual_seed(hash(s) % (2**31))
    p = torch.randn(*s, device=dev, requires_grad=True)   # identical inputs for every contender
    t = torch.randn(*s, device=dev)
    print(f"shape {s}:")
    base_ms = None
    for name, fn in FNS.items():
        try:
            ms, pk = run(fn, p, t)
        except Exception as e:
            print(f"  {name:20s} ERR {str(e)[:45]}"); continue
        if name == "fused_lncc (ours)": base_ms = ms
        spd = f"({ms/base_ms:.2f}x slower)" if base_ms and name != "fused_lncc (ours)" else ""
        print(f"  {name:20s} {ms:8.2f} ms   peak +{pk:5.2f} GB   {spd}")
        p.grad = None
    del p, t; torch.cuda.empty_cache()
    print()
