import os
"""End-to-end validation: gradient-based dense 3D registration (FireANTs/VoxelMorph style).

Optimize a displacement field so warp(moving) aligns to fixed, with loss = LNCC + diffusion
regularizer. This is the real use case where LNCC is the *optimized* objective, not a side term.
We run the SAME registration with our fused LNCC vs MONAI and compare per-iteration wall-clock,
peak VRAM, and convergence (final LNCC similarity + MSE) — i.e. does a 2.9x faster kernel give a
faster end-to-end registration at the same quality?
"""
import sys, time, argparse, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fused_lncc import fused_lncc_loss
from monai.losses import LocalNormalizedCrossCorrelationLoss as MLNCC

dev = "cuda"; K = 7; DR = 1e-5


def identity_grid(D, H, W):
    z, y, x = torch.meshgrid(torch.linspace(-1, 1, D, device=dev), torch.linspace(-1, 1, H, device=dev),
                             torch.linspace(-1, 1, W, device=dev), indexing="ij")
    return torch.stack([x, y, z], dim=-1)[None]  # (1,D,H,W,3) in grid_sample (x,y,z) order


def warp(vol, disp, base):  # disp: (1,3,D,H,W) normalized; vol: (1,1,D,H,W)
    grid = base + disp.permute(0, 2, 3, 4, 1)
    return F.grid_sample(vol, grid, mode="bilinear", padding_mode="border", align_corners=True)


def smooth_field(shape, amp, ksz=11):  # smoothed random field
    f = torch.randn(shape, device=dev)
    g = torch.ones(1, 1, ksz, ksz, ksz, device=dev) / ksz**3
    for _ in range(3):
        f = F.conv3d(f.reshape(-1, 1, *shape[2:]), g, padding=ksz // 2).reshape(shape)
    return amp * f / (f.std() + 1e-6)


def diffusion(disp):  # smoothness regularizer (gradient magnitude)
    dz = disp[:, :, 1:] - disp[:, :, :-1]
    dy = disp[:, :, :, 1:] - disp[:, :, :, :-1]
    dx = disp[:, :, :, :, 1:] - disp[:, :, :, :, :-1]
    return (dz.pow(2).mean() + dy.pow(2).mean() + dx.pow(2).mean())


def run(loss_fn, fixed, moving, base, iters=150, lr=0.02, lam=0.5):
    D, H, W = fixed.shape[2:]
    disp = torch.zeros(1, 3, D, H, W, device=dev, requires_grad=True)
    opt = torch.optim.Adam([disp], lr=lr)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(iters):
        opt.zero_grad()
        warped = warp(moving, disp, base)
        loss = loss_fn(warped, fixed) + lam * diffusion(disp)
        loss.backward(); opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters * 1000
    pk = torch.cuda.max_memory_allocated() / 1e9
    with torch.no_grad():
        warped = warp(moving, disp, base)
        mse = F.mse_loss(warped, fixed).item()
        # report LNCC *similarity* in [0,1] via our kernel (1 - loss)
        sim = 1.0 - fused_lncc_loss(warped, fixed, K, DR).item()
    return dt, pk, mse, sim


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--size", type=int, default=96); ap.add_argument("--iters", type=int, default=150)
    a = ap.parse_args()
    S = a.size
    torch.manual_seed(0)
    fixed = smooth_field((1, 1, S, S, S), 1.0, ksz=7).clamp(-3, 3)     # structured intensity volume
    fixed = (fixed - fixed.min()) / (fixed.max() - fixed.min())
    base = identity_grid(S, S, S)
    true_disp = smooth_field((1, 3, S, S, S), 0.06)                    # known smooth deformation
    moving = warp(fixed, true_disp, base).detach()                    # moving = warped fixed
    mse0 = F.mse_loss(moving, fixed).item()
    sim0 = 1.0 - fused_lncc_loss(moving, fixed, K, DR).item()
    print(f"volume {S}^3, {a.iters} Adam iters. Before registration: MSE={mse0:.4e}, LNCC sim={sim0:.4f}\n")

    monai = MLNCC(spatial_dims=3, kernel_size=K, kernel_type="rectangular", smooth_nr=0, smooth_dr=DR, reduction="mean").to(dev)
    fns = {"fused_lncc (ours)": lambda w, f: fused_lncc_loss(w, f, K, DR),
           "MONAI":             lambda w, f: monai(w, f)}
    res = {}
    for name, fn in fns.items():
        res[name] = run(fn, fixed, moving, base, iters=a.iters)
        dt, pk, mse, sim = res[name]
        print(f"  {name:18s}: {dt:6.1f} ms/iter   peak {pk:5.2f} GB   final MSE={mse:.4e}  LNCC sim={sim:.4f}")
    f, m = res["fused_lncc (ours)"], res["MONAI"]
    print(f"\n  => fused is {m[0]/f[0]:.2f}x faster/iter, {m[1]/f[1]:.2f}x less peak VRAM, "
          f"same registration quality (sim {f[3]:.3f} vs {m[3]:.3f}, MSE {f[2]:.1e} vs {m[2]:.1e}).")
    print(f"  => total wall-clock for {a.iters} iters: fused {f[0]*a.iters/1000:.2f}s vs MONAI {m[0]*a.iters/1000:.2f}s")
